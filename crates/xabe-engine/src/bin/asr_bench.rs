//! Times transcription against `whisper-server`, alternated in pairs.
//!
//! `docs/BENCHMARKS.md` says how to measure and why the alternation matters:
//! this card thermally drifts, and a difference measured in blocks is
//! indistinguishable from drift. So each round runs one of each, and the
//! medians are taken over the rounds.
//!
//! The comparison is only honest if both sides do the same job. Point
//! `--url` at a `whisper-server` started **without** `--vad`: the live one in
//! `run.sh` gates and time-compresses its input before transcribing, which is
//! a different and much smaller amount of work.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Time transcription and report the medians.
#[derive(Debug, Parser)]
#[command(name = "xabe-asr-bench", version, about)]
struct Args {
    /// Checkpoint directory.
    #[arg(long)]
    model: PathBuf,

    /// A 16 kHz mono WAV to transcribe.
    #[arg(long)]
    input: PathBuf,

    /// CUDA device ordinal. Check `nvidia-smi` first.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// A `whisper-server` to compare against, started without `--vad`.
    #[arg(long)]
    url: Option<String>,

    /// Whisper language code.
    #[arg(long, default_value = "zh")]
    language: String,

    /// Untimed runs before measuring.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Timed rounds. Each round runs both implementations once.
    #[arg(long, default_value_t = 20)]
    runs: usize,

    /// Report where this engine's time goes instead of comparing.
    #[arg(long)]
    stages: bool,
}

/// The median and the spread of a set of durations.
fn summarise(name: &str, mut d: Vec<Duration>, seconds: f64) {
    d.sort();
    let ms = |x: Duration| x.as_secs_f64() * 1000.0;
    let median = ms(d[d.len() / 2]);
    println!(
        "  {name:<22} {median:7.1} ms   [{:.1}, {:.1}]   {:.1}x realtime",
        ms(d[0]),
        ms(d[d.len() - 1]),
        seconds * 1000.0 / median,
    );
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("warn"))
        .with_writer(std::io::stderr)
        .without_time()
        .with_target(false)
        .init();

    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let wav = std::fs::read(&args.input)?;
    let audio = xabe_audio::parse_wav(&wav)?;
    let seconds = f64::from(audio.seconds());
    println!(
        "{}: {seconds:.2} s at {} Hz",
        args.input.display(),
        audio.sample_rate,
    );

    let model = xabe_asr::AsrModel::open(&args.model, args.device)?;

    if args.stages {
        return stages(&model, &audio.samples, args, seconds);
    }

    let upstream = args
        .url
        .as_deref()
        .map(xabe_serve::Upstream::new)
        .transpose()?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let ours = |first: bool| -> Result<Duration, Box<dyn std::error::Error>> {
        let t = Instant::now();
        let text = model.transcribe(&audio.samples, &args.language)?;
        let d = t.elapsed();
        if first {
            println!("  xabe-asr says       {text:?}");
        }
        Ok(d)
    };

    for i in 0..args.warmup {
        ours(i == 0)?;
        if let Some(u) = &upstream {
            rt.block_on(u.transcribe(wav.clone(), &args.language))?;
        }
    }

    let mut mine = Vec::with_capacity(args.runs);
    let mut theirs = Vec::with_capacity(args.runs);
    for i in 0..args.runs {
        mine.push(ours(false)?);
        if let Some(u) = &upstream {
            let t = Instant::now();
            let text = rt.block_on(u.transcribe(wav.clone(), &args.language))?;
            theirs.push(t.elapsed());
            if i == 0 {
                println!("  whisper-server says {text:?}");
            }
        }
    }

    println!();
    summarise("xabe-asr, CUDA, f32", mine.clone(), seconds);
    if !theirs.is_empty() {
        summarise("whisper-server, f16", theirs.clone(), seconds);
        // Medians of the same rounds, so drift cancels rather than being
        // averaged into one side.
        mine.sort();
        theirs.sort();
        let ratio = theirs[theirs.len() / 2].as_secs_f64() / mine[mine.len() / 2].as_secs_f64();
        println!("\n  xabe-asr is {ratio:.2}x whisper-server on this clip");
    }
    Ok(())
}

/// Where one transcription's time goes.
///
/// Each stage is timed with the device synchronised on both sides of it -
/// otherwise the number measures how long it took to *queue* the work, which
/// on a warm stream is a few microseconds and tells you nothing.
fn stages(
    model: &xabe_asr::AsrModel,
    samples: &[f32],
    args: &Args,
    seconds: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let gpu = model.gpu();
    let mut mel = Vec::new();
    let mut enc = Vec::new();
    let mut kv = Vec::new();
    let mut dec = Vec::new();
    let mut tokens = 0;

    for i in 0..args.runs + args.warmup {
        let t = Instant::now();
        let features = model.frontend().log_mel(samples);
        let t_mel = t.elapsed();

        let t = Instant::now();
        let encoded = model.encode(&features)?;
        gpu.synchronize()?;
        let t_enc = t.elapsed();

        let t = Instant::now();
        let mut cache = model.cache(&encoded)?;
        gpu.synchronize()?;
        let t_kv = t.elapsed();

        let t = Instant::now();
        let ids = model.generate(&features, &args.language, 64)?;
        gpu.synchronize()?;
        let t_dec = t.elapsed();

        // `generate` runs the encoder again, so the decode-only figure is what
        // it took minus what the encoder and the cache cost. Timing it any
        // other way would need a second entry point that exists for the
        // benchmark, which is how a benchmark stops measuring the product.
        let _ = &mut cache;
        if i >= args.warmup {
            mel.push(t_mel);
            enc.push(t_enc);
            kv.push(t_kv);
            dec.push(t_dec.saturating_sub(t_enc).saturating_sub(t_kv));
            tokens = ids.len();
        }
    }

    println!("\n  {tokens} tokens generated");
    summarise("mel frontend (CPU)", mel, seconds);
    summarise("encoder", enc, seconds);
    summarise("cross-attention KV", kv, seconds);
    summarise("decode loop", dec, seconds);
    Ok(())
}
