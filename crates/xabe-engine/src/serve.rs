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
use xabe_serve::{AppState, GatewayConfig, Inner, SynthesisJob, TtsBackend, Upstream};

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
        ..GatewayConfig::default()
    };

    // Which prompt depends on who is expected to produce Taigi. With
    // --direct-taigi the chat model does it and the translator is out of the
    // reply path entirely; otherwise the model writes Mandarin and the
    // translator converts it.
    config.system_prompt = if args.direct_taigi {
        xabe_serve::direct_taigi_prompt(&args.person, &args.bot)
    } else {
        xabe_serve::mandarin_prompt(&args.person, &args.bot)
    };
    if let Some(path) = &args.prompt_file {
        config.system_prompt = std::fs::read_to_string(path)
            .map_err(|source| EngineError::Io {
                what: "reading the prompt file",
                path: path.display().to_string(),
                source,
            })?
            .trim()
            .to_string();
    }

    let asr = match &stages.asr {
        Stage::Remote { url } => Some(Upstream::new(url)?),
        Stage::Local { .. } => {
            return Err(EngineError::NotImplemented {
                stage: crate::stage::Kind::Asr,
                phase: "4",
            });
        }
        Stage::Off => None,
    };

    // --direct-taigi takes the translator out of the reply path even when one
    // is configured, which is what the measurement 3.8 s -> 1.6 s was of. The
    // flag wins over the stage, so a running translator can be bypassed
    // without restarting it.
    let translator = match (&stages.translator, args.direct_taigi) {
        (_, true) => None,
        (Stage::Remote { url }, _) => Some(Upstream::new(url)?),
        (Stage::Local { .. }, _) => {
            return Err(EngineError::NotImplemented {
                stage: crate::stage::Kind::Translator,
                phase: "5a",
            });
        }
        (Stage::Off, _) => None,
    };

    let llm = match &stages.llm {
        Some(url) => Some(Upstream::new(url)?),
        None => None,
    };

    let mut tts: BTreeMap<String, TtsBackend> = BTreeMap::new();
    match &stages.tts {
        Stage::Local { path, device } => {
            tts.insert(
                xabe_serve::LOCAL_ENGINE.to_string(),
                TtsBackend::Local(spawn_synthesiser(args, path, *device)?),
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
        let Some((name, url)) = spec.split_once('=') else {
            return Err(EngineError::BadEngine(spec.clone()));
        };
        tts.insert(name.to_string(), TtsBackend::Remote(Upstream::new(url)?));
    }
    if let Some(name) = &args.tts_default {
        config.tts_default = name.clone();
    }

    Ok(AppState(Arc::new(Inner {
        config,
        asr,
        llm,
        translator,
        translator_target: args.translator_target.clone(),
        tts,
        page: xabe_serve::PAGE,
    })))
}

/// Starts the synthesiser thread and returns the queue that feeds it.
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
    let mut model = match device {
        Device::Cuda(ordinal) => Synth::Gpu(Box::new(xabe_tts::GpuModel::open(&ck.dir, ordinal)?)),
        Device::Cpu => Synth::Cpu(Box::new(xabe_tts::Synthesizer::open_files(
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
                synthesise(&model, &job, rate, seed);
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

/// Speaks one job, sending each chunk as it is produced.
fn synthesise(model: &Synth, job: &SynthesisJob, rate: u32, seed: u64) {
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
        match model.speak(chunk, seed) {
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
