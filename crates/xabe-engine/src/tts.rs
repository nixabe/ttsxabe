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
