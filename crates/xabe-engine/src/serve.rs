//! Bringing up the server: turning resolved stages into a running process.
//!
//! This is where the two halves meet. `xabe-serve` owns HTTP and refuses to
//! know what a model is; `xabe-tts` owns the model and refuses to know what a
//! socket is. This module is the only place that holds both, and the join is a
//! channel: a synthesiser thread reads [`SynthesisJob`]s and writes WAV chunks
//! back, and nothing on either side of that channel learns anything about the
//! other.
//!
//! The synthesiser runs on its **own OS thread**, not on the async executor.
//! A forward pass is a blocking, GPU-bound 48 ms that would otherwise stall
//! every socket the runtime is polling - and there is exactly one thread
//! because the model is one utterance at a time by design, so a second would
//! only queue on the same device.

use crate::error::EngineError;
use crate::stage::{Device, Stage, Stages};
use crate::{Args, tts::Checkpoint};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use xabe_serve::{
    AppState, AsrBackend, CompletionJob, GatewayConfig, Inner, LlmBackend, SpeechSpan,
    SynthesisJob, TranscribeJob, TranslateJob, TranslatorBackend, TtsBackend, Upstream, VadJob,
};

/// The longest translation the engine will produce for one chunk.
///
/// `gateway.py` sends `n_predict: 256` and chunks a reply at clause
/// boundaries, so a chunk is a sentence at most. The limit is a runaway guard,
/// not a budget.
const TRANSLATE_MAX_NEW: usize = 256;

/// How many synthesis jobs may wait before the caller blocks.
///
/// Two. The queue is not where latency should be absorbed: a request that has
/// been waiting behind two utterances is one the user has already given up on.
const JOB_QUEUE: usize = 2;

/// Runs the server until the process is asked to stop.
pub fn serve(args: &Args, stages: &Stages, addr: &str) -> Result<(), EngineError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| EngineError::Io {
            what: "starting the runtime",
            path: addr.to_string(),
            source,
        })?;

    let state = build_state(args, stages)?;
    runtime.block_on(async move { xabe_serve::serve(addr, state).await })?;
    Ok(())
}

/// Assembles what this process can reach from what its flags said.
/// The system prompt this run will use, resolved before anything is loaded.
pub fn system_prompt(args: &Args) -> Result<String, EngineError> {
    let given = match (&args.system_prompt, &args.prompt_file) {
        (Some(_), Some(_)) => return Err(EngineError::BothPrompts),
        (Some(text), None) => Some(("--system-prompt", text.clone())),
        (None, Some(path)) => Some((
            "--prompt-file",
            std::fs::read_to_string(path).map_err(|source| EngineError::Io {
                what: "reading the prompt file",
                path: path.display().to_string(),
                source,
            })?,
        )),
        (None, None) => None,
    };

    let Some((flag, text)) = given else {
        return Ok(if args.direct_taigi {
            xabe_serve::direct_taigi_prompt(&args.person, &args.bot)
        } else {
            xabe_serve::mandarin_prompt(&args.person, &args.bot)
        });
    };

    let text = text.trim().to_string();
    if text.is_empty() {
        return Err(EngineError::EmptyPrompt { flag });
    }
    Ok(text)
}

fn build_state(args: &Args, stages: &Stages) -> Result<AppState, EngineError> {
    let mut config = GatewayConfig {
        person: args.person.clone(),
        bot: args.bot.clone(),
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        history_turns: args.history_turns,
        min_chunk: args.min_chunk,
        first_chunk: args.first_chunk,
        asr_lang: args.asr_lang.clone(),
        // Overlapping translation with synthesis only pays when they are not
        // competing for one card's SMs. Sharing a card it costs first audio,
        // so the comparison is made here, where the devices are resolved,
        // rather than guessed at in the serving layer. The `or_else` is the
        // same fallback a registered engine uses: with no local `--tts-model`
        // there is no stage to take the card from and `--tts-device` says it.
        // Anything not local to this process is not competing for its card.
        translate_ahead: usize::from(
            match (
                stages.translator.device(),
                stages
                    .tts
                    .device()
                    .or_else(|| args.tts_device.as_deref().and_then(Device::parse)),
            ) {
                (Some(translator), Some(tts)) => translator != tts,
                _ => true,
            },
        ),
        ..GatewayConfig::default()
    };

    config.system_prompt = system_prompt(args)?;
    tracing::info!(
        source = if args.system_prompt.is_some() {
            "--system-prompt"
        } else if args.prompt_file.is_some() {
            "--prompt-file"
        } else if args.direct_taigi {
            "built-in, taigi"
        } else {
            "built-in, mandarin"
        },
        chars = config.system_prompt.chars().count(),
        "system prompt",
    );

    let asr = match &stages.asr {
        Stage::Remote { url } => Some(AsrBackend::Remote(Upstream::new(url)?)),
        Stage::Local { path, device } => Some(AsrBackend::Local(spawn_transcriber(path, *device)?)),
        Stage::Off => None,
    };

    // --direct-taigi takes the translator out of the reply path even when one
    // is configured, which is what the measurement 3.8 s -> 1.6 s was of. The
    // flag wins over the stage, so a running translator can be bypassed
    // without restarting it.
    let translator = match (&stages.translator, args.direct_taigi) {
        (_, true) => None,
        (Stage::Remote { url }, _) => Some(TranslatorBackend::Remote(Upstream::new(url)?)),
        (Stage::Local { path, device }, _) => {
            Some(TranslatorBackend::Local(spawn_translator(path, *device)?))
        }
        (Stage::Off, _) => None,
    };

    let llm = match &stages.llm {
        Stage::Remote { url } => Some(LlmBackend::Remote(Upstream::new(url)?)),
        Stage::Local { path, device } => Some(LlmBackend::Local(spawn_chat(
            path,
            *device,
            config.stops(),
        )?)),
        Stage::Off => None,
    };

    let vad = match &stages.vad {
        Stage::Local { path, .. } => Some(spawn_detector(path)?),
        // Always local. The VAD is 15 tensors and a millisecond of CPU, so a
        // round trip would cost more than the work.
        Stage::Remote { .. } => {
            return Err(EngineError::LocalOnly {
                stage: crate::stage::Kind::Vad,
            });
        }
        Stage::Off => None,
    };

    let mut tts: BTreeMap<String, TtsBackend> = BTreeMap::new();
    match &stages.tts {
        Stage::Local { path, device } => {
            // `--tts-model` takes either checkpoint; which one it is, is a
            // property of the directory rather than of a second flag.
            let local = spawn_local(args, path, *device)?;
            tts.insert(
                xabe_serve::LOCAL_ENGINE.to_string(),
                TtsBackend::Local(local),
            );
        }
        Stage::Remote { url } => {
            tts.insert(
                xabe_serve::LOCAL_ENGINE.to_string(),
                TtsBackend::Remote(Upstream::new(url)?),
            );
        }
        Stage::Off => {}
    }
    for spec in &args.tts_engines {
        let Some((name, target)) = spec.split_once('=') else {
            return Err(EngineError::BadEngine(spec.clone()));
        };
        // A URL is another process; anything else is a directory in this one.
        // Sniffed by scheme rather than by trying to open it both ways, so a
        // typo in a path fails as a missing checkpoint rather than as a DNS
        // lookup of a directory name.
        let backend = if target.starts_with("http://") || target.starts_with("https://") {
            TtsBackend::Remote(Upstream::new(target)?)
        } else {
            let dir = std::path::PathBuf::from(target);
            // A local engine registered here shares `--tts-device` with
            // `--tts-model`. With no local TTS stage there is no card to
            // share, and `--tts-device` alone is what says which one.
            let device = stages
                .tts
                .device()
                .or_else(|| args.tts_device.as_deref().and_then(Device::parse))
                .unwrap_or(Device::Cpu);
            TtsBackend::Local(spawn_local(args, &dir, device)?)
        };
        tracing::info!(
            engine = name,
            local = matches!(backend, TtsBackend::Local(_)),
            target,
            "tts engine",
        );
        tts.insert(name.to_string(), backend);
    }
    if let Some(name) = &args.tts_default {
        config.tts_default = name.clone();
    }

    let mut tts_scripts = std::collections::HashMap::new();
    for spec in &args.tts_scripts {
        let Some((name, script)) = spec.split_once('=') else {
            return Err(EngineError::BadScript(spec.clone()));
        };
        if !tts.contains_key(name) {
            return Err(EngineError::UnknownEngine {
                name: name.to_string(),
                known: tts.keys().cloned().collect::<Vec<_>>().join(", "),
            });
        }
        tts_scripts.insert(name.to_string(), script.to_string());
    }

    // Refused here rather than discovered as a silent turn. A local engine that
    // reads romanisation gets it from the translator and from nowhere else, so
    // without one it is handed Han and says nothing. Remote engines are left
    // alone: what an upstream accepts is its own business.
    //
    // Only when the reply path is *here*, which is what an LLM means. The
    // script is read by `script_for`, reached only from the converse path, and
    // a process with no chat model never walks it: a synthesiser-only worker
    // is handed text over `/tts` and says what it is given, whoever
    // romanised it. Checking it anyway is what made a split worker declare
    // `mms=HAN` - a label that was false, inert, and load-bearing.
    if translator.is_none() && llm.is_some() {
        for (name, backend) in &tts {
            if !matches!(backend, TtsBackend::Local(_)) {
                continue;
            }
            let script = tts_scripts.get(name).unwrap_or(&args.translator_target);
            if !script.eq_ignore_ascii_case("HAN") {
                return Err(EngineError::ScriptNeedsTranslator {
                    engine: name.clone(),
                    script: script.clone(),
                });
            }
        }
    }

    Ok(AppState(Arc::new(Inner {
        config,
        asr,
        vad,
        llm,
        translator,
        translator_target: args.translator_target.clone(),
        tts_scripts,
        tts,
        page: xabe_serve::PAGE,
    })))
}

/// Starts the detector thread and returns the queue that feeds it.
///
/// Its own thread for the same reason as the synthesiser: the forward pass is
/// blocking, and the detector is stateful, so one worker also guarantees that
/// two clips can never interleave through the same LSTM.
fn spawn_detector(path: &std::path::Path) -> Result<mpsc::Sender<VadJob>, EngineError> {
    let mut vad = xabe_vad::open(path)?;
    let (tx, mut rx) = mpsc::channel::<VadJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-vad".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                // Reset first, not last. A clip that arrives after a panic or a
                // dropped reply would otherwise start with the previous clip's
                // memory of what was being said.
                vad.reset();
                let probs = vad.probabilities(&job.samples);
                let spans = xabe_vad::segments(&probs, xabe_vad::SegmentParams::default())
                    .into_iter()
                    .map(|s| SpeechSpan {
                        start: s.start.min(job.samples.len()),
                        end: s.end.min(job.samples.len()),
                    })
                    .collect();
                // The receiver going away means the turn was abandoned, which
                // is the prefetch-cancel path and entirely normal.
                let _ = job.reply.send(spans);
            }
            tracing::debug!("detector thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the detector thread",
            path: path.display().to_string(),
            source,
        })?;
    Ok(tx)
}

/// Starts the synthesiser thread and returns the queue that feeds it.
/// Starts the transcriber thread and returns its work queue.
///
/// One OS thread, started before the listener binds, exactly as the
/// synthesiser is: a forward pass this size must not run on a tokio executor
/// thread, where it would block every other socket for the duration.
///
/// The clip arrives as a WAV and must already be 16 kHz. That is not a new
/// requirement - the VAD in front of this has always assumed it, silently -
/// and the server tells the browser which rate to send in `/api/config`. It is
/// checked here rather than resampled because a resampler good enough for an
/// ASR is a real piece of work, and one that is not good enough is a
/// transcript that is quietly worse.
fn spawn_transcriber(
    path: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<TranscribeJob>, EngineError> {
    // `Kind::has_cpu` refuses `--asr-device cpu` at preflight, so by here the
    // device is a card.
    let Device::Cuda(ordinal) = device else {
        unreachable!("--asr-device cpu is refused when the stages resolve");
    };
    // Opened on the calling thread, so a bad checkpoint fails preflight rather
    // than at the first turn - by which time a user is waiting.
    let model = xabe_asr::AsrModel::open(path, ordinal)?;

    let (tx, mut rx) = mpsc::channel::<TranscribeJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-asr".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                let _ = job
                    .reply
                    .send(transcribe_one(&model, &job.wav, &job.language));
            }
            tracing::debug!("transcriber thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the transcriber thread",
            path: path.display().to_string(),
            source,
        })?;
    Ok(tx)
}

/// One transcription, with its failures turned into a message the caller can
/// put in front of a user.
fn transcribe_one(
    model: &xabe_asr::AsrModel,
    wav: &[u8],
    language: &str,
) -> Result<String, String> {
    let audio = xabe_audio::parse_wav(wav).map_err(|e| e.to_string())?;
    if audio.sample_rate != 16_000 {
        return Err(format!(
            "the clip is {} Hz; this stage wants 16000",
            audio.sample_rate
        ));
    }
    model
        .transcribe(&audio.samples, language)
        .map_err(|e| e.to_string())
}

/// Starts the translator thread and returns its work queue.
///
/// One OS thread, started before the listener binds, exactly as the ASR and
/// the synthesiser are. The load is about 27 GB of transfers and takes as long
/// as that sounds, which is the point of doing it at preflight: a service is
/// started once and answers for hours.
fn spawn_translator(
    path: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<TranslateJob>, EngineError> {
    // `Kind::has_cpu` refuses `--translator-device cpu` at preflight: the 13 B
    // is 27 GB of weights and 26 GFLOP a token.
    let Device::Cuda(ordinal) = device else {
        unreachable!("--translator-device cpu is refused when the stages resolve");
    };
    let model = xabe_translate::Translator::open(path, ordinal)?;

    let (tx, mut rx) = mpsc::channel::<TranslateJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-translate".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                let out = model
                    .translate(
                        &job.text,
                        &job.target,
                        TRANSLATE_MAX_NEW,
                        xabe_translate::Translator::REPEAT_PENALTY,
                    )
                    .map_err(|e| e.to_string());
                let _ = job.reply.send(out);
            }
            tracing::debug!("translator thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the translator thread",
            path: path.display().to_string(),
            source,
        })?;
    Ok(tx)
}

/// Starts the chat model's thread and returns its work queue.
///
/// One OS thread, started before the listener binds, exactly as the ASR, the
/// synthesiser and the translator are - a GPU step must not run on a tokio
/// executor thread. The load is about 16 GB of transfers and takes as long as
/// that sounds, which is the point of doing it at preflight: a service is
/// started once and answers for hours.
fn spawn_chat(
    path: &std::path::Path,
    device: Device,
    // Taken from the same `GatewayConfig` the remote path serialises into its
    // request body, so the two backends stop on the same strings. Building
    // them here from `person` and `bot` a second time is how they would drift.
    stops: Vec<String>,
) -> Result<mpsc::Sender<CompletionJob>, EngineError> {
    // `Kind::has_cpu` refuses `--llm-device cpu` at preflight: the 8 B is
    // 16 GFLOP a token against scalar kernels that manage under 2 GFLOP/s.
    let Device::Cuda(ordinal) = device else {
        unreachable!("--llm-device cpu is refused when the stages resolve");
    };
    let model = xabe_chat::ChatModel::open(path, ordinal)?;

    let (tx, mut rx) = mpsc::channel::<CompletionJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-chat".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                // The sampler is `gateway.py`'s by default, which is what the
                // remote path sends in its request body - so switching a
                // running pipeline between `--llm-url` and `--llm-model` does
                // not quietly change how the model is sampled.
                let sampling = xabe_chat::Sampling::default();
                let out = model.complete(&job.prompt, &sampling, &stops, &mut |piece| {
                    // `blocking_send` failing means the receiver is gone,
                    // which is the browser having cancelled the turn. Returning
                    // false stops generation at the next token rather than at
                    // the end of the sentence, so a barged-in reply stops
                    // costing GPU time immediately.
                    job.pieces.blocking_send(piece.to_string()).is_ok()
                });
                if let Err(e) = out {
                    tracing::warn!(error = %e, "the chat model refused a turn");
                }
            }
            tracing::debug!("chat thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the chat thread",
            path: path.display().to_string(),
            source,
        })?;
    Ok(tx)
}

fn spawn_synthesiser(
    args: &Args,
    path: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<SynthesisJob>, EngineError> {
    let ck = Checkpoint::locate(path, args.config.as_deref());
    let seed = args.seed;
    let overrides = (
        args.noise_scale,
        args.noise_scale_duration,
        args.speaking_rate,
    );

    // Opened here, on the calling thread, so a bad checkpoint fails preflight
    // rather than at the first turn - by which time a user is waiting.
    //
    // Which of the two VITS dialects this is, is a property of the directory
    // rather than of a flag, exactly as the choice between VITS, CosyVoice and
    // Tacotron2 is.
    let coqui = is_coqui(&ck.dir);
    if coqui {
        tracing::warn!("this engine speaks IPA phonemes; no stage in this pipeline produces them",);
    }
    let mut model = match (device, coqui) {
        (Device::Cuda(ordinal), true) => {
            Synth::Gpu(Box::new(xabe_tts::GpuModel::open_coqui(&ck.dir, ordinal)?))
        }
        (Device::Cuda(ordinal), false) => {
            Synth::Gpu(Box::new(xabe_tts::GpuModel::open(&ck.dir, ordinal)?))
        }
        (Device::Cpu, true) => Synth::Cpu(Box::new(xabe_tts::Synthesizer::open_coqui(&ck.dir)?)),
        (Device::Cpu, false) => Synth::Cpu(Box::new(xabe_tts::Synthesizer::open_files(
            &ck.dir.join("model.safetensors"),
            &ck.config,
            &ck.dir,
        )?)),
    };
    model.apply(overrides);
    let rate = model.sampling_rate();

    let (tx, mut rx) = mpsc::channel::<SynthesisJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-tts".into())
        .spawn(move || {
            // A blocking receive on a channel the async side writes to. The
            // thread exists only to keep a forward pass off the executor.
            while let Some(job) = rx.blocking_recv() {
                synthesise(&model, &job, rate, seed, coqui);
            }
            tracing::debug!("synthesiser thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the synthesiser thread",
            path: ck.dir.display().to_string(),
            source,
        })?;

    Ok(tx)
}

/// Whether a directory holds a CosyVoice3 checkpoint rather than a VITS one.
///
/// All three files, not one: a half-converted directory - `tools/convert_cosyvoice.py`
/// interrupted, say - would otherwise be opened as CosyVoice and fail deep
/// inside a weight schema instead of here, where the path is still in hand.
fn is_cosyvoice(dir: &std::path::Path) -> bool {
    ["llm.safetensors", "flow.safetensors", "hift.safetensors"]
        .iter()
        .all(|f| dir.join(f).is_file())
}

/// Whether a directory holds a Coqui VITS checkpoint rather than a 🤗 export.
///
/// Both files, for the reason the two checks above want all of theirs: a
/// directory with `best_model.pth` and no `config.json` has nothing to read the
/// geometry from, and saying so here beats failing inside a weight schema with
/// the path already out of scope.
///
/// **A Coqui VITS engine is fed IPA phonemes, not text.** Nothing in this
/// pipeline produces them, so serving one is only useful with a client that
/// sends phonemes; the one-shot `--text` path is where it earns its keep. See
/// `xabe_vits::CoquiTokenizer`.
pub fn is_coqui(dir: &std::path::Path) -> bool {
    dir.join("best_model.pth").is_file() && dir.join("config.json").is_file()
}

/// Whether a directory holds a converted Tacotron2 + WaveGlow pair.
///
/// All three files for the same reason `is_cosyvoice` wants all of its: a
/// directory with the weights and no `tacotron2.json` is a half-run of
/// `tools/convert_tacotron2.py`, and saying so here beats failing inside a
/// symbol table with the path already out of scope.
pub fn is_tacotron(dir: &std::path::Path) -> bool {
    xabe_taco::FILES.iter().all(|f| dir.join(f).is_file())
}

/// Opens whichever synthesiser a directory holds.
///
/// Which model it is, is a property of the directory rather than of a flag.
/// The three checks are disjoint - no checkpoint carries another's filenames -
/// so the order is readability, not precedence.
fn spawn_local(
    args: &Args,
    dir: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<SynthesisJob>, EngineError> {
    if is_cosyvoice(dir) {
        spawn_cosy(args, dir, device)
    } else if is_tacotron(dir) {
        spawn_taco(args, dir, device)
    } else {
        spawn_synthesiser(args, dir, device)
    }
}

/// Starts a Tacotron2 + WaveGlow synthesiser on its own thread.
///
/// Mirrors [`spawn_cosy`] and differs in two things, both properties of the
/// model:
///
/// - It is **CUDA only**, for the same reason CosyVoice is: WaveGlow is 87.9 M
///   parameters of dilated convolution run at the sample rate.
/// - It reads **romanisation**, like mms rather than like CosyVoice, so pair it
///   with `--tts-script <name>=POJ`. POJ is transliterated to the Tâi-lô the
///   checkpoint was trained on inside the crate, since that is a fact about
///   this checkpoint and not about the pipeline.
fn spawn_taco(
    args: &Args,
    dir: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<SynthesisJob>, EngineError> {
    let Device::Cuda(ordinal) = device else {
        return Err(EngineError::LocalOnly {
            stage: crate::stage::Kind::Tts,
        });
    };

    // Opened on the calling thread, so a geometry this crate cannot run is a
    // preflight failure rather than a first turn of silence.
    let taco = xabe_taco::Taco::open(dir, ordinal, args.taco_sigma, args.seed)?;
    let rate = taco.sample_rate() as u32;

    let (tx, mut rx) = mpsc::channel::<SynthesisJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-taco".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                speak_taco(&taco, &job, rate);
            }
            tracing::debug!("tacotron2 thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the tacotron2 thread",
            path: dir.display().to_string(),
            source,
        })?;

    Ok(tx)
}

/// Speaks one job with Tacotron2, sending each clause as it is produced.
///
/// Chunked on romanisation like the VITS path rather than on Han like the
/// CosyVoice one, and for a second reason besides getting first audio out
/// sooner: this decoder is autoregressive with a learned stop, and a long
/// input is where an attention that has lost its place runs to the step limit.
fn speak_taco(taco: &xabe_taco::Taco, job: &SynthesisJob, rate: u32) {
    let text = xabe_serve::clean(&job.text);
    if text.is_empty() {
        return;
    }

    for (i, chunk) in xabe_serve::split_poj(&text, 120).iter().enumerate() {
        match taco.synthesize(chunk) {
            Ok(audio) if !audio.is_empty() => {
                let wav = xabe_audio::wav_bytes(&audio, rate);
                let msg = xabe_serve::TtsChunk {
                    seq: i as u64 + 1,
                    wav: B64.encode(&wav),
                    taigi: String::new(),
                    roman: chunk.clone(),
                };
                if job.reply.blocking_send(msg).is_err() {
                    return;
                }
            }
            // Nothing in the clause was in the 71-symbol alphabet.
            Ok(_) => {}
            // One unspeakable clause should not silence the whole reply.
            Err(e) => tracing::warn!(%e, chunk = %chunk, "could not synthesise a clause"),
        }
    }
}

/// Starts a CosyVoice3 synthesiser on its own thread.
///
/// Mirrors [`spawn_synthesiser`] and differs in three things, each of which is
/// a property of the model rather than a choice:
///
/// - It is **CUDA only**. There is no scalar path for a 642 M-parameter decode
///   plus a 22-layer diffusion transformer, and pretending otherwise would give
///   a configuration that starts and then never answers.
/// - It reads **Han**, so the text is split on sentences rather than
///   romanised. Pair it with `--tts-script <name>=HAN`.
/// - It carries a speaker bundle and an instruct string, both pinned here
///   because they decide what the model *is* for this process.
fn spawn_cosy(
    args: &Args,
    dir: &std::path::Path,
    device: Device,
) -> Result<mpsc::Sender<SynthesisJob>, EngineError> {
    let Device::Cuda(ordinal) = device else {
        return Err(EngineError::LocalOnly {
            stage: crate::stage::Kind::Tts,
        });
    };
    let voice = args
        .cosy_voice
        .clone()
        .unwrap_or_else(|| dir.join("voices/taigi-ref.safetensors"));

    // Opened on the calling thread, so a missing bundle or an instruct without
    // `<|endofprompt|>` fails preflight rather than at the first turn.
    let cosy = xabe_cosy::Cosy::open(dir, &voice, &args.cosy_instruct, ordinal)?;
    let rate = cosy.sample_rate() as u32;

    let (tx, mut rx) = mpsc::channel::<SynthesisJob>(JOB_QUEUE);
    std::thread::Builder::new()
        .name("xabe-cosy".into())
        .spawn(move || {
            while let Some(job) = rx.blocking_recv() {
                speak_cosy(&cosy, &job, rate);
            }
            tracing::debug!("cosyvoice thread stopping");
        })
        .map_err(|source| EngineError::Io {
            what: "starting the cosyvoice thread",
            path: dir.display().to_string(),
            source,
        })?;

    Ok(tx)
}

/// Speaks one job with CosyVoice, sending each sentence as it is produced.
///
/// Chunked for the same reason the VITS path is: the first chunk is what the
/// listener waits on, and a long reply synthesised whole would hold every
/// sentence back until the last one was done. Sixty Han characters is about
/// the same duration as the 120 characters of romanisation mms is given.
fn speak_cosy(cosy: &xabe_cosy::Cosy, job: &SynthesisJob, rate: u32) {
    let text = xabe_serve::clean(&job.text);
    if text.is_empty() {
        return;
    }

    for (i, chunk) in xabe_serve::split_sentences(&text, 60).iter().enumerate() {
        match cosy.synthesize(chunk) {
            Ok(audio) if !audio.is_empty() => {
                let wav = xabe_audio::wav_bytes(&audio, rate);
                let msg = xabe_serve::TtsChunk {
                    seq: i as u64 + 1,
                    wav: B64.encode(&wav),
                    taigi: chunk.clone(),
                    roman: String::new(),
                };
                if job.reply.blocking_send(msg).is_err() {
                    return;
                }
            }
            Ok(_) => {}
            // One unspeakable clause should not silence the whole reply.
            Err(e) => tracing::warn!(%e, chunk = %chunk, "could not synthesise a clause"),
        }
    }
}

/// Speaks one job, sending each chunk as it is produced.
fn synthesise(model: &Synth, job: &SynthesisJob, rate: u32, seed: u64, phonemes: bool) {
    // The same cleaning and chunking the Python daemon did, because VITS
    // degrades on long inputs and chunking also gets first audio out sooner.
    let text = xabe_serve::clean(&job.text);
    if text.is_empty() {
        return;
    }
    let poj = xabe_serve::normalize_for_mms(&text);
    // 120 characters of romanisation, which is roughly the 60 Han characters
    // the daemon used for the same duration of speech.
    let chunks = xabe_serve::split_poj(&poj, 120);

    for (i, chunk) in chunks.iter().enumerate() {
        // The two VITS checkpoints read different scripts. `mms-tts-nan` was
        // trained on POJ and takes the chunk as it stands; the Coqui one was
        // trained on IPA, so the chunk is transliterated here - which is
        // spelling, not a pronunciation guess, because the translator upstream
        // has already decided how every word is read. See `xabe-taigi`.
        let spoken = if phonemes {
            let p = xabe_taigi::poj_to_ipa(chunk);
            if p.dropped > 0 {
                tracing::debug!(
                    dropped = p.dropped,
                    "runs that are not Tâi-lô were discarded"
                );
            }
            std::borrow::Cow::Owned(p.text)
        } else {
            std::borrow::Cow::Borrowed(chunk.as_str())
        };
        match model.speak(&spoken, seed) {
            Ok(audio) => {
                let wav = xabe_audio::wav_bytes(&audio, rate);
                let msg = xabe_serve::TtsChunk {
                    seq: i as u64 + 1,
                    wav: B64.encode(&wav),
                    taigi: String::new(),
                    roman: chunk.clone(),
                };
                if job.reply.blocking_send(msg).is_err() {
                    // The listener went away mid-utterance. Stopping now saves
                    // the rest of a reply nobody will hear.
                    return;
                }
            }
            // One unspeakable clause should not silence the whole reply: text
            // outside the 48-symbol vocabulary tokenises to nothing, and that
            // is a property of the clause, not of the turn.
            Err(e) => tracing::warn!(%e, chunk = %chunk, "could not synthesise a clause"),
        }
    }
}

/// Whichever device the synthesiser was opened on.
///
/// The GPU variant is boxed because it is several kilobytes of device handles
/// against the CPU variant's few hundred bytes, and every `Synth` would
/// otherwise be sized for the larger of the two.
enum Synth {
    /// The scalar reference path.
    Cpu(Box<xabe_tts::Synthesizer>),
    /// CUDA.
    Gpu(Box<xabe_tts::GpuModel>),
}

impl Synth {
    fn apply(&mut self, (noise, noise_dur, rate): (Option<f32>, Option<f32>, Option<f32>)) {
        let cfg = match self {
            Synth::Cpu(s) => s.config_mut(),
            Synth::Gpu(g) => g.config_mut(),
        };
        if let Some(v) = noise {
            cfg.noise_scale = v;
        }
        if let Some(v) = noise_dur {
            cfg.noise_scale_duration = v;
        }
        if let Some(v) = rate {
            cfg.speaking_rate = v;
        }
    }

    fn sampling_rate(&self) -> u32 {
        match self {
            Synth::Cpu(s) => s.config().sampling_rate,
            Synth::Gpu(g) => g.config().sampling_rate,
        }
    }

    fn speak(&self, text: &str, seed: u64) -> Result<Vec<f32>, xabe_tts::SynthesisError> {
        match self {
            Synth::Cpu(s) => s.synthesize(text, seed),
            Synth::Gpu(g) => g.synthesize(text, seed),
        }
    }
}
