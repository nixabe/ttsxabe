//! Times synthesis, on CPU or CUDA.
//!
//! Mirrors `tools/bench/pytorch_baseline.py` exactly - same text, same device,
//! warm-up discarded, median of the timed runs, model load excluded - so the
//! two numbers are comparable. `docs/BENCHMARKS.md` says how to measure and why
//! the alternation matters.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use xabe_tts::{GpuModel, Synthesizer};

/// Time synthesis and report the median.
#[derive(Debug, Parser)]
#[command(name = "xabe-tts-bench", version, about)]
struct Args {
    /// Safetensors checkpoint.
    #[arg(long, env = "XABE_TTS_MODEL")]
    model: PathBuf,

    /// POJ text to synthesise.
    #[arg(long, default_value = "lí hó, kin-á-ji̍t thinn-khì chin hó.")]
    text: String,

    /// cpu, or a CUDA device ordinal.
    #[arg(long, env = "XABE_TTS_DEVICE", default_value = "0")]
    device: String,

    /// Untimed runs before measuring.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Timed runs.
    #[arg(long, default_value_t = 20)]
    runs: usize,

    /// Report per-stage timings instead of the total. CUDA only.
    #[arg(long)]
    stages: bool,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_writer(std::io::stderr)
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();
    let dir = args
        .model
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();

    // The seed varies per run so that every iteration synthesises a different
    // duration draw. Timing one fixed utterance twenty times would measure a
    // single frame count rather than the model.
    if args.stages && args.device != "cpu" {
        let ordinal: usize = args.device.parse().unwrap_or(0);
        let m = match GpuModel::open(&dir, ordinal) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("loading model on device {ordinal}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut totals: Vec<(&'static str, f64)> = Vec::new();
        for i in 0..args.warmup + args.runs {
            let (_, stages) = match m.synthesize_timed(&args.text, i as u64) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("synthesising: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if i < args.warmup {
                continue;
            }
            if totals.is_empty() {
                totals = stages;
            } else {
                for (t, s) in totals.iter_mut().zip(&stages) {
                    t.1 += s.1;
                }
            }
        }
        let whole: f64 = totals.iter().map(|(_, v)| v).sum();
        for (name, total) in &totals {
            let mean = total / args.runs as f64;
            println!("{name:14} {mean:7.2} ms  {:5.1}%", 100.0 * total / whole);
        }
        println!("{:14} {:7.2} ms", "total", whole / args.runs as f64);
        return ExitCode::SUCCESS;
    }

    let (times, samples, rate) = if args.device == "cpu" {
        let s = match Synthesizer::open(&dir) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("loading model: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut times = Vec::new();
        let mut samples = 0;
        for i in 0..args.warmup + args.runs {
            let t0 = Instant::now();
            let a = match s.synthesize(&args.text, i as u64) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("synthesising: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if i >= args.warmup {
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                samples += a.len();
            }
        }
        (times, samples, s.config().sampling_rate)
    } else {
        let ordinal = match args.device.parse::<usize>() {
            Ok(o) => o,
            Err(_) => {
                eprintln!(
                    "--device must be `cpu` or a device ordinal, got {}",
                    args.device
                );
                return ExitCode::FAILURE;
            }
        };
        let m = match GpuModel::open(&dir, ordinal) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("loading model on device {ordinal}: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut times = Vec::new();
        let mut samples = 0;
        for i in 0..args.warmup + args.runs {
            let t0 = Instant::now();
            // `synthesize` ends in a device-to-host copy, which synchronises.
            // Without that this would time how fast work can be enqueued,
            // which on a model this small is most of what there is to measure.
            let a = match m.synthesize(&args.text, i as u64) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("synthesising: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if i >= args.warmup {
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                samples += a.len();
            }
        }
        (times, samples, m.config().sampling_rate)
    };

    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let mean_samples = samples as f64 / times.len() as f64;
    let seconds = mean_samples / f64::from(rate);

    println!("device        {}", args.device);
    println!("text          {:?}", args.text);
    println!("samples       {mean_samples:.0} mean ({seconds:.2} s at {rate} Hz)",);
    println!("runs          {} after {} warm-up", args.runs, args.warmup);
    println!("median        {median:.2} ms");
    println!(
        "min / max     {:.2} / {:.2} ms",
        sorted[0],
        sorted[sorted.len() - 1],
    );
    println!("realtime x    {:.1}", seconds * 1000.0 / median);
    ExitCode::SUCCESS
}
