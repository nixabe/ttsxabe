//! Times the single-row mat-vec against an f16 weight at the shapes the
//! Whisper decoder and the Llama stages run it at, and reports the bandwidth
//! the weight stream reaches.
//!
//! ```sh
//! XABE_DEVICE=0 cargo run --release -p xabe-cuda --bin bench-gemv
//! ```
//!
//! Medians of repeated launches, warmed first. The point is the GB/s column:
//! a decode step is the weight stream, and a mat-vec below the card's rate is
//! time a token pays for nothing.
use std::process::ExitCode;
use std::time::Instant;
use xabe_cuda::{Batch, Gpu, Operand};

const WARMUP: usize = 20;
const REPS: usize = 200;

/// `(k, n, what)`.
const SHAPES: &[(usize, usize, &str)] = &[
    (1280, 1280, "whisper decoder projection"),
    (1280, 5120, "whisper decoder fc1"),
    (5120, 1280, "whisper decoder fc2"),
    (4096, 4096, "an 8 B projection at f16"),
    (4096, 128256, "the 8 B head at f16"),
];

fn seq(n: usize, salt: u64) -> Vec<f32> {
    let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) * 2.0 - 1.0
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> ExitCode {
    let ordinal: usize = std::env::var("XABE_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let gpu = match Gpu::open(ordinal) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no CUDA device {ordinal}: {e}");
            return ExitCode::SUCCESS;
        }
    };
    println!(
        "{:32} {:>10} {:>9} {:>7}",
        "shape", "bytes", "median", "GB/s"
    );
    for &(k, n, what) in SHAPES {
        let a = gpu.upload(&seq(k, 1)).expect("upload a");
        let w = gpu.upload_f16(&seq(n * k, 2)).expect("upload w");
        let run = |g: &Gpu| {
            g.gemm_batched(
                Operand::F32(&a),
                Operand::F16(&w),
                None,
                Batch::single(n),
                1,
                k,
                n,
            )
            .expect("gemv")
        };
        for _ in 0..WARMUP {
            let _ = run(&gpu);
        }
        gpu.synchronize().expect("sync");
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            let _ = run(&gpu);
            gpu.synchronize().expect("sync");
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let us = median(times);
        let bytes = (n * k * 2) as f64;
        println!(
            "{:32} {:>10} {:>7.1} us {:>7.0}",
            format!("{what} {k}x{n}"),
            format!("{:.1} MB", bytes / 1e6),
            us,
            bytes / us / 1e3
        );
    }
    ExitCode::SUCCESS
}
