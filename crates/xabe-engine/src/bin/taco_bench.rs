//! Times Tacotron2 + WaveGlow and reports where the time goes.
//!
//! `docs/BENCHMARKS.md` says how to measure. Two things this does differently
//! from `asr_bench`, both forced by the model:
//!
//! - **The total and the breakdown come from different runs.** Timing a stage
//!   means synchronising after it, and a pipeline synchronised at every stage
//!   is not the pipeline that runs. So the total is measured with timing off
//!   and the breakdown with it on, and the breakdown's own total is printed
//!   next to the real one so the gap between them is visible rather than
//!   assumed away.
//! - **Medians over rounds, not a mean.** Synthesis is stochastic - the prenet
//!   keeps its dropout at inference - so the frame count moves between runs on
//!   the same text. A mean over a handful of runs is mostly measuring how many
//!   frames each happened to produce.
//!
//! Reported against the audio produced, because that is what "fast enough"
//! means for a synthesiser: 1.0x realtime is speech arriving as fast as it is
//! spoken, and the turn budget is what is left after the ASR and the LLM.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

/// Time synthesis and report the medians.
#[derive(Debug, Parser)]
#[command(name = "xabe-taco-bench", version, about)]
struct Args {
    /// A converted checkpoint directory.
    #[arg(long, default_value = "models/tts/tacotron2-nan")]
    model: PathBuf,

    /// CUDA device ordinal. Check `nvidia-smi` first.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// What to say. Repeatable; each is timed separately.
    #[arg(long)]
    text: Vec<String>,

    /// Untimed runs before measuring.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Timed rounds.
    #[arg(long, default_value_t = 7)]
    rounds: usize,

    /// Also print the per-stage breakdown.
    #[arg(long, default_value_t = true)]
    breakdown: bool,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let lines = if args.text.is_empty() {
        vec![
            "li2 ho2".to_string(),
            "gua2 si7 tai5-uan5-lang5".to_string(),
            "li2-ho2 , tsin1 hua1-hi2 jin7-sik4 li2 . gua2 si7 tai5-uan5-lang5 .".to_string(),
        ]
    } else {
        args.text.clone()
    };

    let taco = match xabe_taco::Taco::open(&args.model, args.device, None, 1) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("could not open {}: {e}", args.model.display());
            return ExitCode::FAILURE;
        }
    };
    let rate = taco.sample_rate() as f64;

    println!(
        "{:<52} {:>8} {:>9} {:>8} {:>9}",
        "text", "audio s", "median ms", "x rt", "steps"
    );
    for line in &lines {
        for _ in 0..args.warmup {
            if let Err(e) = taco.synthesize(line) {
                eprintln!("warmup failed: {e}");
                return ExitCode::FAILURE;
            }
        }

        let mut ms = Vec::new();
        let mut secs = Vec::new();
        for _ in 0..args.rounds {
            let t0 = Instant::now();
            let audio = match taco.synthesize(line) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("synthesis failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            secs.push(audio.len() as f64 / rate);
        }

        let (m, s) = (median(ms), median(secs));
        let short: String = line.chars().take(50).collect();
        println!(
            "{short:<52} {s:>8.2} {m:>9.1} {:>8.2} {:>9.0}",
            s * 1e3 / m,
            s * rate / 256.0
        );

        if args.breakdown {
            let (_, marks, steps) = match taco.synthesize_timed(line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("timed run failed: {e}");
                    return ExitCode::FAILURE;
                }
            };
            // Every mark is a disjoint leaf - the indentation groups them for
            // reading, it does not nest them - so the shares are of one total.
            let total: f64 = marks.iter().map(|(_, v)| v).sum();
            for (name, v) in &marks {
                println!("    {name:<44} {v:>8.2} ms  {:>5.1}%", 100.0 * v / total);
            }
            println!(
                "    {:<44} {total:>8.2} ms  ({steps} decoder steps, synchronised; \
                 the median above is the untimed run)",
                "timed total"
            );
        }
    }
    ExitCode::SUCCESS
}
