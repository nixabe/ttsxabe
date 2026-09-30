//! Times the prefill attention, `flash_attn`, at the shapes the engine runs it.
//!
//! ```sh
//! LLMTIE_DEVICE=0 cargo run --release -p llmtie-cuda --bin bench-flash
//! ```
//!
//! Each row is one layer's attention, a synchronise, medians of twenty after
//! three warm-up calls. The TFLOP/s column counts the two products a causal
//! row actually needs - `2 * 2 * hd` flops a (query, key) pair it attends to -
//! so a causal shape is credited half the square and the encoder the whole.

use llmtie_cuda::Gpu;
use std::process::ExitCode;
use std::time::Instant;

const WARMUP: usize = 3;
const REPS: usize = 20;

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
            ((s >> 40) as f32 / 8_388_608.0) * 2.0 - 1.0
        })
        .collect()
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .without_time()
        .with_target(false)
        .init();
    let ordinal: usize = std::env::var("LLMTIE_DEVICE")
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

    // (name, heads, kv_heads, hd, tq, causal, f16 cache)
    let shapes: &[(&str, usize, usize, usize, usize, bool, bool)] = &[
        ("chat 8B, 512", 32, 8, 128, 512, true, true),
        ("chat 8B, 2048", 32, 8, 128, 2048, true, true),
        ("chat 8B, 4096", 32, 8, 128, 4096, true, true),
        ("chat 8B, 8192", 32, 8, 128, 8192, true, true),
        ("translator 13B, 512", 40, 40, 128, 512, true, true),
        ("translator 13B, 3968", 40, 40, 128, 3968, true, true),
        ("whisper encoder, 1500", 20, 20, 64, 1500, false, false),
    ];
    for name in ["flash_attn_h", "flash_attn_64", "flash_attn_64_rows"] {
        if let Some((regs, local, shared)) = g.kernel_resources(name) {
            tracing::info!("{name}: {regs} registers, {local} B spilled, {shared} B shared");
        }
    }
    tracing::info!("{:<24} {:>10} {:>8}", "shape, one layer", "ms", "TFLOP/s");
    for &(name, heads, kv, hd, tq, causal, half) in shapes {
        // The capacity the engine would hold: even, and at least the prompt.
        let cap = tq.next_multiple_of(256);
        let q = g.upload(&seq(tq * heads * hd, 1)).unwrap();
        let kf = seq(kv * cap * hd, 2);
        let vf = seq(kv * hd * cap, 3);
        let scale = (hd as f32).powf(-0.5);
        let mut times = Vec::with_capacity(REPS);
        if half {
            let k = g.upload_f16(&kf).unwrap();
            let v = g.upload_f16(&vf).unwrap();
            for i in 0..WARMUP + REPS {
                let t = Instant::now();
                let _o = g
                    .flash_attn_f16(&q, &k, &v, tq, 0, heads, kv, hd, cap, scale, causal)
                    .unwrap();
                g.synchronize().unwrap();
                if i >= WARMUP {
                    times.push(t.elapsed().as_secs_f64());
                }
            }
        } else {
            let k = g.upload(&kf).unwrap();
            let v = g.upload(&vf).unwrap();
            for i in 0..WARMUP + REPS {
                let t = Instant::now();
                let _o = g
                    .flash_attn(&q, &k, &v, tq, 0, heads, kv, hd, cap, scale, causal)
                    .unwrap();
                g.synchronize().unwrap();
                if i >= WARMUP {
                    times.push(t.elapsed().as_secs_f64());
                }
            }
        }
        let t = median(times);
        let pairs = if causal {
            (tq * (tq + 1) / 2) as f64
        } else {
            (tq * tq) as f64
        };
        let flop = 4.0 * hd as f64 * pairs * heads as f64;
        tracing::info!("{name:<24} {:>9.3} {:>8.1}", t * 1e3, flop / t / 1e12);
    }
    ExitCode::SUCCESS
}
