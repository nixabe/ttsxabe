//! The one-shot text-to-speech run.
//!
//! This is the whole of `xabe-tts`'s old `main.rs`, moved: the synthesiser is
//! no longer the whole program, so its CLI is no longer the program's CLI. The
//! library it drives has not changed.
//!
//! It refuses to know about serving or about any other stage. Given text and a
//! destination it produces a WAV, and everything about *which* text and *where
//! from* was settled in `action.rs` before this was called.

use crate::args::Args;
use crate::error::EngineError;
use crate::stage::Device;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use xabe_audio::write_wav;
use xabe_tts::{GpuModel, Synthesizer};

/// A checkpoint located on disk, however the flag spelled it.
pub struct Checkpoint {
    /// The directory holding the weights, config and vocabulary.
    pub dir: PathBuf,
    /// The config to read, which may have been overridden.
    pub config: PathBuf,
}

impl Checkpoint {
    /// Accepts either the model directory or the safetensors file inside it.
    ///
    /// Both spellings are in use: the consolidated tree names directories
    /// (`models/tts/mms-tts-nan`), while the old flag and every test pointed at
    /// `model.safetensors`. Rejecting one of them would break working commands
    /// to no purpose, since the directory is recoverable from the file.
    pub fn locate(path: &Path, config_override: Option<&Path>) -> Checkpoint {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        let config = config_override
            .map(Path::to_path_buf)
            .unwrap_or_else(|| dir.join("config.json"));
        Checkpoint { dir, config }
    }
}

/// Synthesises `text` and writes it to `out`.
pub fn speak(
    args: &Args,
    path: &Path,
    device: Device,
    text: &str,
    out: &Path,
) -> Result<(), EngineError> {
    let text = read_text(text)?;
    let ck = Checkpoint::locate(path, args.config.as_deref());

    let (rate, audio) = match device {
        Device::Cpu => {
            let mut synth =
                Synthesizer::open_files(&ck.dir.join("model.safetensors"), &ck.config, &ck.dir)?;
            apply_overrides(synth.config_mut(), args);
            let rate = synth.config().sampling_rate;
            (rate, synth.synthesize(&text, args.seed)?)
        }
        Device::Cuda(ordinal) => {
            let mut model = GpuModel::open(&ck.dir, ordinal)?;
            apply_overrides(model.config_mut(), args);
            let rate = model.config().sampling_rate;
            (rate, model.synthesize(&text, args.seed)?)
        }
    };

    tracing::info!(
        seconds = format!("{:.2}", audio.len() as f32 / rate as f32),
        samples = audio.len(),
        "synthesised",
    );
    write_out(out, &audio, rate)?;
    tracing::info!(out = %out.display(), "wrote");
    Ok(())
}

/// Applies the three sampling overrides the CLI exposes.
///
/// Only these three: they are temperatures and a rate, not geometry. Anything
/// else would contradict the checkpoint.
fn apply_overrides(cfg: &mut xabe_vits::VitsConfig, args: &Args) {
    if let Some(v) = args.noise_scale {
        cfg.noise_scale = v;
    }
    if let Some(v) = args.noise_scale_duration {
        cfg.noise_scale_duration = v;
    }
    if let Some(v) = args.speaking_rate {
        cfg.speaking_rate = v;
    }
}

/// Reads the text argument, or stdin when it is `-`.
fn read_text(arg: &str) -> Result<String, EngineError> {
    if arg != "-" {
        return Ok(arg.to_string());
    }
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|source| EngineError::Io {
            what: "reading",
            path: "stdin".into(),
            source,
        })?;
    Ok(s.trim_end_matches('\n').to_string())
}

/// Writes the WAV to a path, or stdout when it is `-`.
fn write_out(path: &Path, audio: &[f32], rate: u32) -> Result<(), EngineError> {
    let io = |source| EngineError::Io {
        what: "writing",
        path: path.display().to_string(),
        source,
    };
    if path.as_os_str() == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        write_wav(&mut lock, audio, rate).map_err(io)?;
        return lock.flush().map_err(io);
    }
    let mut f = std::fs::File::create(path).map_err(io)?;
    write_wav(&mut f, audio, rate).map_err(io)?;
    f.flush().map_err(io)
}

/// Synthesises through another process and writes the result.
///
/// The remote returns one self-contained WAV per clause, so they are decoded
/// and rewritten as a single file rather than concatenated: a WAV is a header
/// plus samples, and appending whole files produces something no player will
/// read past the first chunk.
pub fn speak_remote(url: &str, text: &str, out: &Path) -> Result<(), EngineError> {
    let text = read_text(text)?;
    let runtime = runtime()?;
    let chunks = runtime.block_on(async move {
        let up = xabe_serve::Upstream::new(url)?;
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let stream = tokio::spawn({
            let up = up.clone();
            let text = text.clone();
            async move { up.stream_tts(&text, tx).await }
        });
        let mut got = Vec::new();
        while let Some(chunk) = rx.recv().await {
            got.push(chunk);
        }
        match stream.await {
            Ok(Ok(())) => Ok(got),
            Ok(Err(e)) => Err(e),
            Err(e) => Err(xabe_serve::ServeError::Upstream {
                stage: "tts",
                message: e.to_string(),
            }),
        }
    })?;

    let mut samples = Vec::new();
    let mut rate = 16_000;
    for chunk in &chunks {
        let bytes = B64
            .decode(&chunk.wav)
            .map_err(|e| xabe_serve::ServeError::BadPcm(e.to_string()))?;
        let wav = xabe_audio::parse_wav(&bytes)?;
        rate = wav.sample_rate;
        samples.extend_from_slice(&wav.samples);
    }
    tracing::info!(
        seconds = format!("{:.2}", samples.len() as f32 / rate as f32),
        chunks = chunks.len(),
        "synthesised remotely",
    );
    write_out(out, &samples, rate)?;
    tracing::info!(out = %out.display(), "wrote");
    Ok(())
}

/// Transcribes a file through another process and prints the transcript.
pub fn transcribe_remote(args: &Args, url: &str, input: &Path) -> Result<(), EngineError> {
    let wav = if input.as_os_str() == "-" {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|source| EngineError::Io {
                what: "reading",
                path: "stdin".into(),
                source,
            })?;
        buf
    } else {
        std::fs::read(input).map_err(|source| EngineError::Io {
            what: "reading",
            path: input.display().to_string(),
            source,
        })?
    };
    // Parsed before sending so a file that is not audio fails here, naming the
    // problem, rather than as an opaque error from the other process.
    let parsed = xabe_audio::parse_wav(&wav)?;
    tracing::debug!(
        seconds = format!("{:.2}", parsed.seconds()),
        rate = parsed.sample_rate,
        "read audio",
    );

    let lang = args.asr_lang.clone();
    let url = url.to_string();
    let text = runtime()?.block_on(async move {
        xabe_serve::Upstream::new(&url)?
            .transcribe(wav, &lang)
            .await
    })?;
    // The transcript is the tool's output, so it goes to stdout at INFO - the
    // level that appears by default - not to a log the caller has to enable.
    tracing::info!("{text}");
    Ok(())
}

/// A runtime for the one-shot paths, which are otherwise synchronous.
fn runtime() -> Result<tokio::runtime::Runtime, EngineError> {
    // Current-thread: a one-shot run makes one request and has nothing to
    // overlap it with, so a thread pool would be startup cost for no work.
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|source| EngineError::Io {
            what: "starting the runtime",
            path: "one-shot".into(),
            source,
        })
}
