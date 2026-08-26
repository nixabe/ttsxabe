//! Synthesises Taiwanese Hokkien speech from Pe̍h-ōe-jī.
//!
//! ```sh
//! xabe-tts --model model.safetensors \
//!          --text "lí hó, kin-á-ji̍t thinn-khì chin hó." --out hello.wav
//! ```
//!
//! The doc comments on [`Args`] *are* the `--help` text; there is no second
//! copy to drift. See `docs/CLI.md` for the flag design and the conventions
//! behind it.

use clap::Parser;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use xabe_tts::{GpuModel, Synthesizer, write_wav};

/// Synthesise Taiwanese Hokkien speech from Pe̍h-ōe-jī text.
#[derive(Debug, Parser)]
#[command(name = "xabe-tts", version, about)]
struct Args {
    /// Safetensors checkpoint.
    #[arg(long, env = "XABE_TTS_MODEL")]
    model: PathBuf,

    /// Model config. Defaults to config.json beside the checkpoint.
    #[arg(long, env = "XABE_TTS_CONFIG")]
    config: Option<PathBuf>,

    /// POJ text to speak. Use - to read stdin.
    #[arg(long)]
    text: String,

    /// Output WAV. Use - to write stdout.
    #[arg(long)]
    out: PathBuf,

    /// Seed for the duration and prior draws.
    #[arg(long, env = "XABE_TTS_SEED", default_value_t = 0)]
    seed: u64,

    /// Prior temperature. Higher is more varied.
    #[arg(long)]
    noise_scale: Option<f32>,

    /// Duration temperature. Higher varies the rhythm more.
    #[arg(long)]
    noise_scale_duration: Option<f32>,

    /// Speaking rate. Higher is faster.
    #[arg(long)]
    speaking_rate: Option<f32>,

    /// Where to synthesise: cpu, or a CUDA device ordinal.
    ///
    /// The CPU path is the scalar reference and is roughly 45x slower than real
    /// time; it exists to be read and to be correct, not to be used.
    #[arg(long, env = "XABE_TTS_DEVICE", default_value = "0")]
    device: String,

    /// Log verbosity: info, debug or trace.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    // INFO/DEBUG/TRACE to stdout, WARN/ERROR to stderr, so `--out -` stays
    // pipeable even while the tool is talking.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&args.log_level)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stdout)
        .without_time()
        .with_target(false)
        .init();

    // A numbered preflight: each stage reports its own failure and returns,
    // rather than unwinding a Result chain that loses which stage broke.
    let config = args.config.clone().unwrap_or_else(|| {
        args.model
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("config.json")
    });
    let dir = args
        .model
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();

    // 1: the text, before anything expensive is loaded.
    let text = match read_text(&args.text) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("1/4 reading text: {e}");
            return ExitCode::FAILURE;
        }
    };

    // 2: the model, on whichever device was asked for.
    let ordinal = if args.device == "cpu" {
        None
    } else {
        match args.device.parse::<usize>() {
            Ok(o) => Some(o),
            Err(_) => {
                tracing::error!(
                    "2/4 --device must be `cpu` or a CUDA device ordinal, got {}",
                    args.device,
                );
                return ExitCode::FAILURE;
            }
        }
    };

    let (rate, audio) = match ordinal {
        None => {
            let mut synth = match Synthesizer::open_files(&args.model, &config, &dir) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("2/4 loading model: {e}");
                    return ExitCode::FAILURE;
                }
            };
            apply_overrides(synth.config_mut(), &args);
            let rate = synth.config().sampling_rate;
            match synth.synthesize(&text, args.seed) {
                Ok(a) => (rate, a),
                Err(e) => {
                    tracing::error!("3/4 synthesising: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Some(o) => {
            let mut model = match GpuModel::open(&dir, o) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("2/4 loading model on device {o}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            apply_overrides(model.config_mut(), &args);
            let rate = model.config().sampling_rate;
            match model.synthesize(&text, args.seed) {
                Ok(a) => (rate, a),
                Err(e) => {
                    tracing::error!("3/4 synthesising: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };
    tracing::info!(
        seconds = format!("{:.2}", audio.len() as f32 / rate as f32),
        samples = audio.len(),
        "synthesised",
    );

    // 4: the file.
    if let Err(e) = write_out(&args.out, &audio, rate) {
        tracing::error!("4/4 writing {}: {e}", args.out.display());
        return ExitCode::FAILURE;
    }
    tracing::info!(out = %args.out.display(), "wrote");
    ExitCode::SUCCESS
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
fn read_text(arg: &str) -> std::io::Result<String> {
    if arg != "-" {
        return Ok(arg.to_string());
    }
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s.trim_end_matches('\n').to_string())
}

/// Writes the WAV to a path, or stdout when it is `-`.
fn write_out(path: &std::path::Path, audio: &[f32], rate: u32) -> std::io::Result<()> {
    if path.as_os_str() == "-" {
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        write_wav(&mut lock, audio, rate)?;
        return lock.flush();
    }
    let mut f = std::fs::File::create(path)?;
    write_wav(&mut f, audio, rate)?;
    f.flush()
}
