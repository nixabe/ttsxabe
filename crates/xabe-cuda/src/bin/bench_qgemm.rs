//! Times the packed-weight matmul at a prefill's shapes: the translator's and
//! the chat model's projections, at the row counts a prompt brings.
//!
//! ```sh
//! XABE_DEVICE=0 cargo run --release -p xabe-cuda --bin bench-qgemm
//! ```
//!
//! The weights are synthetic blocks of the file's own formats - the kernel's
//! speed does not depend on what the codes say - and each row is thirty-two
//! launches back to back and one synchronise, medians of nine. The GB/s
//! column is the weight bytes over the time, which is the rate a short
//! prompt is bound by.

use std::process::ExitCode;
use std::time::Instant;
use xabe_cuda::{Batch, Gpu, Operand, Quant};

const REPS: usize = 9;
const CALLS: usize = 32;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Deterministic bytes, with every block's f16 scales set small and finite.
fn blocks(q: Quant, count: usize) -> Vec<u8> {
    let ts = q.type_size();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut raw = vec![0u8; count * ts];
    for b in raw.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *b = (s >> 33) as u8;
    }
    // A small positive half, 0x1400 = 2^-10, wherever a block keeps one.
    let offs: &[usize] = match q {
        Quant::Q4K => &[0, 2],
        Quant::Q6K => &[208],
        _ => &[],
    };
    for i in 0..count {
        for &o in offs {
            raw[i * ts + o] = 0x00;
            raw[i * ts + o + 1] = 0x14;
        }
    }
    raw
}

fn seq(n: usize) -> Vec<f32> {
    let mut s: u64 = 12345;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) * 4.0 - 2.0
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
    let g = match Gpu::open(ordinal) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("no usable device {ordinal}: {e}");
            return ExitCode::FAILURE;
        }
    };
    // What the kernels cost in registers, spill and shared memory: a short
    // prompt's matmul is latency-bound, and its occupancy is the budget.
    for name in [
        "gemm_i8_q4k",
        "gemm_i8_q6k",
        "gemm_i8_q6k_narrow",
        "gemm_i8_q4k_skinny",
        "gemm_i8_q6k_skinny",
        "gemm_i8_q4k_narrow",
        "gemm_i8_stream_q4k_m3",
        "gemm_i8_stream_q6k_m3",
    ] {
        if let Some((regs, local, shared)) = g.kernel_resources(name) {
            tracing::info!("{name}: {regs} registers, {local} B spilled, {shared} B shared");
        }
    }
    // (what, k, n, format)
    let weights: &[(&str, usize, usize, Quant)] = &[
        ("13B gate+up", 5120, 27648, Quant::Q4K),
        ("13B down", 13824, 5120, Quant::Q6K),
        ("13B q", 5120, 5120, Quant::Q4K),
        ("8B gate+up", 4096, 28672, Quant::Q4K),
        ("8B at 216 tiles", 4096, 27648, Quant::Q4K),
    ];
    let rows: Vec<usize> = std::env::var("XABE_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 16, 24, 32, 48, 64, 128]);
    tracing::info!("{:<14} {:>5} {:>10} {:>8}", "weight", "rows", "us", "GB/s");
    for &(what, k, n, q) in weights {
        let raw = blocks(q, n * k / q.block_size());
        let w = g.upload_quant(q, &raw, k).expect("upload");
        let bytes = raw.len() as f64;
        for &m in &rows {
            let a = g.upload(&seq(m * k)).expect("upload a");
            let run = || {
                let _ = g
                    .gemm_batched(
                        Operand::F32(&a),
                        Operand::Q { data: &w, ty: q },
                        None,
                        Batch::single(m * n),
                        m,
                        k,
                        n,
                    )
                    .expect("gemm");
            };
            for _ in 0..3 {
                run();
            }
            g.synchronize().expect("sync");
            let mut t = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let s = Instant::now();
                for _ in 0..CALLS {
                    run();
                }
                g.synchronize().expect("sync");
                t.push(s.elapsed().as_secs_f64() / CALLS as f64);
            }
            let us = median(t) * 1e6;
            tracing::info!("{what:<14} {m:>5} {us:>10.1} {:>8.0}", bytes / us / 1e3);
        }
    }
    ExitCode::SUCCESS
}
