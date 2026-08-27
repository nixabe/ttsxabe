//! Where the vocoder starts to disagree with the reference.
//!
//!     cargo run --release -p xabe-cosy --example probe_vocoder -- <device>
//!
//! Not a test - a test says pass or fail, and by the time one fails the useful
//! question is which stage. This prints one line per tap.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn npy(p: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let (hlen, at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let head = std::str::from_utf8(&b[at..at + hlen]).expect("ascii");
    let o = head.find('(').expect("shape");
    let c = head[o..].find(')').expect("shape") + o;
    let shape: Vec<usize> = head[o + 1..c]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("dim"))
        .collect();
    let v = b[at + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (shape, v)
}

fn report(name: &str, want: &[f32], got: &[f32]) {
    if want.len() != got.len() {
        println!("  {name:14} LENGTH {} against {}", got.len(), want.len());
        return;
    }
    let n = want.len() as f64;
    let (mw, mg) = (
        want.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        got.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut num, mut dw, mut dg) = (0.0, 0.0, 0.0);
    for (&a, &b) in want.iter().zip(got) {
        let (a, b) = (f64::from(a) - mw, f64::from(b) - mg);
        num += a * b;
        dw += a * a;
        dg += b * b;
    }
    let corr = num / (dw.sqrt() * dg.sqrt());
    let rms = |x: &[f32]| (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / n).sqrt();
    let worst = want
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  {name:14} corr {corr:9.6}  rms {:.5} -> {:.5} (x{:.4})  worst {worst:.5}",
        rms(want),
        rms(got),
        rms(got) / rms(want)
    );
}

fn main() {
    let dev: usize = std::env::args()
        .nth(1)
        .expect("usage: probe_vocoder <device>")
        .parse()
        .expect("a device number");
    let dir = root().join(".golden/cosyvoice");
    let v = xabe_cosy::Vocoder::open(
        &root().join("models/tts/cosyvoice3-0.5b/hift.safetensors"),
        dev,
    )
    .expect("open");

    let (mel_shape, mel) = npy(&dir.join("mel.npy"));
    let (src_shape, source) = npy(&dir.join("source.npy"));
    let frames = mel_shape[2];
    let source_len = *src_shape.last().expect("len");

    let taps = v
        .decode_tapped(
            &v.gpu().upload(&mel).expect("mel"),
            frames,
            &v.gpu().upload(&source).expect("source"),
            source_len,
        )
        .expect("decode");

    for (name, got) in &taps {
        let p = dir.join(format!("{name}.npy"));
        if !p.is_file() {
            println!("  {name:14} (no capture)");
            continue;
        }
        let (_, want) = npy(&p);
        report(name, &want, got);
    }
}
