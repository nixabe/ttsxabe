//! Times `gemm` against `linear` at the shapes the ASR encoder actually runs.
//!
//! ```sh
//! cargo run --release -p xabe-cuda --bin bench-gemm
//! XABE_DEVICE=1 cargo run --release -p xabe-cuda --bin bench-gemm
//! ```
//!
//! Medians of repeated runs, warmed first. The point is not a leaderboard: it
//! is to know whether the tensor-core path is worth the operand rounding it
//! costs, at the shapes that matter, on this card.

use std::process::ExitCode;
use std::time::Instant;
use xabe_cuda::Gpu;

/// Runs discarded before timing, so the first NVRTC compile is not measured.
const WARMUP: usize = 3;

/// Timed runs. The median is reported; a mean would follow the tail.
const REPS: usize = 20;

/// The shapes, and what each one is in the model.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (1500, 1280, 1280, "encoder q/k/v/o projection"),
    (1500, 1280, 5120, "encoder mlp up"),
    (1500, 5120, 1280, "encoder mlp down"),
    (1, 1280, 1280, "decoder projection, one token"),
    (1, 1280, 51864, "decoder output head"),
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// A deterministic spread, so two runs measure the same arithmetic.
fn seq(n: usize, salt: u64) -> Vec<f32> {
    let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) * 0.2 - 0.1
        })
        .collect()
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .without_time()
        .with_target(false)
        .init();

    let ordinal: usize = std::env::var("XABE_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let gpu = match Gpu::open(ordinal) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("no usable device {ordinal}: {e}");
            return ExitCode::FAILURE;
        }
    };

    tracing::info!(
        "{:>6} {:>6} {:>6}  {:>10} {:>10} {:>8}  {:>7}",
        "m",
        "k",
        "n",
        "gemm",
        "linear",
        "speedup",
        "TFLOP/s"
    );
    for &(m, k, n, what) in SHAPES {
        let a = gpu.upload(&seq(m * k, 1)).expect("upload a");
        let w = gpu.upload(&seq(n * k, 2)).expect("upload w");

        let mut gemm = Vec::with_capacity(REPS);
        let mut linear = Vec::with_capacity(REPS);
        for i in 0..WARMUP + REPS {
            // `synchronize`, not `download`. Downloading the result to force
            // the queue to drain measures the PCIe copy as well as the kernel,
            // and for the wider shapes here the copy is *most* of it: 1500x5120
            // floats is 31 MB, about 5 ms at 6 GB/s, against a kernel that runs
            // in well under one. Every number in the first version of this
            // benchmark was the bus.
            let t = Instant::now();
            let _out = gpu.gemm(&a, &w, None, m, k, n).expect("gemm");
            gpu.synchronize().expect("sync");
            let dt = t.elapsed().as_secs_f64();
            if i >= WARMUP {
                gemm.push(dt);
            }

            let t = Instant::now();
            let _out = gpu.linear(&a, &w, None, m, k, n).expect("linear");
            gpu.synchronize().expect("sync");
            let dt = t.elapsed().as_secs_f64();
            if i >= WARMUP {
                linear.push(dt);
            }
        }

        let g = median(gemm);
        let l = median(linear);
        let flop = 2.0 * m as f64 * k as f64 * n as f64;
        tracing::info!(
            "{m:>6} {k:>6} {n:>6}  {:>9.2}ms {:>9.2}ms {:>7.2}x  {:>7.1}   {what}",
            g * 1e3,
            l * 1e3,
            l / g,
            flop / g / 1e12,
        );
    }
    ExitCode::SUCCESS
}
