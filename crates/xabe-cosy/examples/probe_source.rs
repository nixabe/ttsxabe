//! Compares the engine's own dither against the capture's, through the vocoder.

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

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

fn main() {
    let dev: usize = std::env::args()
        .nth(1)
        .expect("usage: probe_source <device>")
        .parse()
        .expect("device");
    let dir = root().join(".golden/cosyvoice");
    let model = root().join("models/tts/cosyvoice3-0.5b");

    let v = xabe_cosy::Vocoder::open(&model.join("hift.safetensors"), dev).expect("vocoder");
    let f = xabe_st::StFile::open(model.join("hift.safetensors")).expect("hift");
    let f0p = xabe_cosy::F0Predictor::bind(&f, v.gpu()).expect("f0");
    let lw = f
        .tensor_shaped("m_source.l_linear.weight", &[1, xabe_cosy::HARMONICS])
        .expect("w");
    let lb = f.tensor_shaped("m_source.l_linear.bias", &[1]).expect("b")[0];

    let (mel_shape, mel) = npy(&dir.join("mel.npy"));
    let frames = mel_shape[2];
    let (_, want_f0) = npy(&dir.join("f0.npy"));
    let (_, want_src) = npy(&dir.join("source.npy"));
    let (_, want_wav) = npy(&dir.join("wav.npy"));
    let (_, cap_waves) = npy(&dir.join("sine_waves.npy"));
    let (_, cap_ini) = npy(&dir.join("sine_rand_ini.npy"));

    let gmel = v.gpu().upload(&mel).expect("mel");
    let f0 = f0p.predict(v.gpu(), &gmel, frames).expect("f0");
    println!("  f0            corr {:.6}", corr(&f0, &want_f0));

    let samples = frames * v.config().hop();
    for (name, dither) in [
        (
            "captured",
            xabe_cosy::Dither {
                rand_ini: cap_ini.to_vec(),
                sine_waves: cap_waves.to_vec(),
                len: samples,
            },
        ),
        ("seeded", xabe_cosy::Dither::seeded(samples, 1986)),
    ] {
        let src = xabe_cosy::excitation(&f0, &Default::default(), &dither, lw, lb).expect("src");
        let g = v.gpu().upload(&src).expect("up");
        let wav = v.decode(&gmel, frames, &g, samples).expect("decode");
        let e = |x: &[f32]| x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
        println!(
            "  {name:12}  source corr {:.6}  wav corr {:.6}  gain {:.4}",
            corr(&src, &want_src),
            corr(&wav, &want_wav),
            (e(&wav) / e(&want_wav)).sqrt()
        );
    }
}
