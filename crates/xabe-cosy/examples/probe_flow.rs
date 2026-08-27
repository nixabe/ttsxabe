//! Where the DiT estimator starts to disagree with the reference.
//!
//!     cargo run --release -p xabe-cosy --example probe_flow -- <device>

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
    // See `tools/oracle/capture_cosyvoice.py`: a Fortran-order file has the
    // right shape in its header and its axes the other way round in its bytes,
    // so a reader that trusts the shape gets a permutation of the right values.
    assert!(
        head.contains("'fortran_order': False"),
        "{}: fortran order",
        p.display()
    );
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
    let n = want.len().min(got.len()) as f64;
    if want.len() != got.len() {
        println!("  {name:18} LENGTH {} against {}", got.len(), want.len());
    }
    let (w, g) = (&want[..n as usize], &got[..n as usize]);
    let (mw, mg) = (
        w.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        g.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut num, mut dw, mut dg) = (0.0, 0.0, 0.0);
    for (&a, &b) in w.iter().zip(g) {
        let (a, b) = (f64::from(a) - mw, f64::from(b) - mg);
        num += a * b;
        dw += a * a;
        dg += b * b;
    }
    let rms = |x: &[f32]| (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / n).sqrt();
    let worst = w
        .iter()
        .zip(g)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  {name:18} corr {:9.6}  rms {:.5} -> {:.5}  worst {worst:.5}",
        num / (dw.sqrt() * dg.sqrt()),
        rms(w),
        rms(g)
    );
}

fn main() {
    let dev: usize = std::env::args()
        .nth(1)
        .expect("usage: probe_flow <device>")
        .parse()
        .expect("a device number");
    let dir = root().join(".golden/cosyvoice");
    let flow = xabe_cosy::Flow::open(
        &root().join("models/tts/cosyvoice3-0.5b/flow.safetensors"),
        dev,
    )
    .expect("open");

    // The capture saves everything through `.float()`, so ids arrive as f32.
    let ids = |name: &str| -> Vec<u32> {
        npy(&dir.join(name))
            .1
            .iter()
            .map(|v| v.round() as u32)
            .collect()
    };
    let prompt_tokens = ids("flow_prompt_speech_token.npy");
    let tokens = ids("speech_token.npy");
    let (_, prompt_feat) = npy(&dir.join("prompt_speech_feat.npy"));
    let (_, embedding) = npy(&dir.join("flow_embedding.npy"));
    let (_, noise) = npy(&dir.join("cfm_noise.npy"));

    for (name, got) in flow
        .mel_tapped(&prompt_tokens, &tokens, &prompt_feat, &embedding, &noise)
        .expect("mel")
    {
        let path = dir.join(format!("{name}.npy"));
        if !path.is_file() {
            println!("  {name:18} no capture");
            continue;
        }
        let (_, want) = npy(&path);
        report(&name, &want, &got);
    }
}
