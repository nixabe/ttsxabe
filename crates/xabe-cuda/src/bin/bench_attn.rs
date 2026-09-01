//! Times the single-query decode attention against the three-kernel chain it
//! replaces, at the shapes the engine decodes.
//!
//! ```sh
//! XABE_DEVICE=0 cargo run --release -p xabe-cuda --bin bench-attn
//! ```
//!
//! Each row is thirty-two layers' worth of one step, a synchronise on both
//! sides, medians of twenty. Both sides read the same caches.

use std::process::ExitCode;
use std::time::Instant;
use xabe_cuda::{Batch, DecodeScratch, Gpu, Operand};

const WARMUP: usize = 3;
const REPS: usize = 20;
const LAYERS: usize = 32;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

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
    let ordinal: usize = std::env::var("XABE_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let g = match Gpu::open(ordinal) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no device: {e}");
            return ExitCode::FAILURE;
        }
    };
    // (name, heads, kv_heads, hd, tk, cap)
    let shapes = [
        (
            "chat 8B, 128 ctx",
            32usize,
            8usize,
            128usize,
            128usize,
            256usize,
        ),
        ("chat 8B, 256 ctx", 32, 8, 128, 256, 256),
        ("chat 8B, 512 ctx", 32, 8, 128, 512, 512),
        ("chat 8B, 1024 ctx", 32, 8, 128, 1024, 1024),
        ("chat 8B, 2048 ctx", 32, 8, 128, 2048, 2048),
        ("translator 13B, 128 ctx", 40, 40, 128, 128, 256),
        ("translator 13B, 256 ctx", 40, 40, 128, 256, 256),
        ("translator 13B, 512 ctx", 40, 40, 128, 512, 512),
        ("translator 13B, 1024 ctx", 40, 40, 128, 1024, 1024),
        ("whisper self, 40 of 448", 20, 20, 64, 40, 448),
        ("whisper cross, 1500", 20, 20, 64, 1500, 1500),
    ];
    println!(
        "{:<28} {:>10} {:>10} {:>8}",
        "shape x32", "chain", "fused", "GB/s"
    );
    for (name, heads, kv, hd, tk, cap) in shapes {
        let group = heads / kv;
        let q = g.upload(&seq(heads * hd, 1)).unwrap();
        let k = g.upload_f16(&seq(kv * cap * hd, 2)).unwrap();
        let v = g.upload_f16(&seq(kv * hd * cap, 3)).unwrap();
        let scale = (hd as f32).powf(-0.5);
        let mut scratch = DecodeScratch::new();

        let chain = || {
            for _ in 0..LAYERS {
                let mut scores = g
                    .gemm_batched(
                        Operand::F32(&q),
                        Operand::F16(&k),
                        None,
                        Batch {
                            count: kv,
                            a: group * hd,
                            w: cap * hd,
                            out: group * tk,
                            w_row: 0,
                        },
                        group,
                        hd,
                        tk,
                    )
                    .unwrap();
                g.softmax_causal(&mut scores, heads, tk, 1, tk - 1, scale)
                    .unwrap();
                let _ctx = g
                    .gemm_batched(
                        Operand::F32(&scores),
                        Operand::F16(&v),
                        None,
                        Batch {
                            count: kv,
                            a: group * tk,
                            w: hd * cap,
                            out: group * hd,
                            w_row: cap,
                        },
                        group,
                        tk,
                        hd,
                    )
                    .unwrap();
            }
            g.synchronize().unwrap();
        };
        let mut fused = || {
            for _ in 0..LAYERS {
                let _ = g
                    .attn_decode_f16(
                        &q,
                        &k,
                        &v,
                        heads,
                        kv,
                        hd,
                        tk,
                        cap,
                        scale,
                        false,
                        &mut scratch,
                    )
                    .unwrap();
            }
            g.synchronize().unwrap();
        };
        let mut tc = Vec::new();
        let mut tf = Vec::new();
        for i in 0..WARMUP + REPS {
            let t = Instant::now();
            chain();
            let c = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            fused();
            let f = t.elapsed().as_secs_f64() * 1e3;
            if i >= WARMUP {
                tc.push(c);
                tf.push(f);
            }
        }
        let (mc, mf) = (median(tc), median(tf));
        let bytes = (LAYERS * 2 * kv * tk * hd * 2) as f64;
        println!(
            "{name:<28} {mc:>8.3} ms {mf:>8.3} ms {:>8.0}",
            bytes / (mf * 1e-3) / 1e9
        );
    }
    ExitCode::SUCCESS
}
