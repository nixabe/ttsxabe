//! One estimator evaluation on seeded inputs, written out for a build-to-build
//! comparison: `probe_est <device> <out.f32>` writes `dit_step0`, `[2, 80, n]`.
use std::path::PathBuf;
fn seq(n: usize, salt: u64) -> Vec<f32> {
    let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
fn main() {
    let mut args = std::env::args().skip(1);
    let dev: usize = args.next().expect("device").parse().expect("device");
    let out = args.next().expect("out path");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let flow = xabe_cosy::Flow::open(
        &root.join("models/tts/cosyvoice3-0.5b/flow.safetensors"),
        dev,
    )
    .expect("flow");
    let (m, n) = (80usize, 96usize);
    let x = seq(2 * m * n, 1);
    let mu = seq(2 * m * n, 2);
    let cond = seq(2 * m * n, 3);
    let spk = seq(2 * m, 4);
    let taps = flow
        .estimate_tapped(&x, &mu, &cond, &spk, 0.37, n)
        .expect("estimate");
    std::fs::create_dir_all(&out).expect("dir");
    for (name, d) in &taps {
        let bytes: Vec<u8> = d.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{out}/{name}.f32"), bytes).expect("write");
        println!("wrote {name} {} values", d.len());
    }
}
