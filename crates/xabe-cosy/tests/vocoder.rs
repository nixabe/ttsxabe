//! The HiFT vocoder against CosyVoice3's own, on the captured mel.
//!
//! # Why the excitation is an input here and not a stage
//!
//! `decode` takes the mel *and* the excitation, and the capture provides both.
//! That is on purpose: the excitation's own path runs an F0 predictor in
//! float64 and a bank of sine oscillators seeded from buffers that are not in
//! the checkpoint, and testing all of it at once would mean a failure anywhere
//! looks like a failure everywhere.
//!
//! So the bulk of the weights - three upsampling stages, eighteen residual
//! blocks, the inverse-STFT head - is verified against a known-good excitation
//! first. `source.rs` is then a separate question with its own test.
//!
//! # A waveform is compared three ways
//!
//! Sample-for-sample equality is the wrong bar for a vocoder: it is a hundred
//! and thirty thousand samples through eighteen residual blocks and an inverse
//! transform, in float32, with reduction orders that differ from PyTorch's. So
//! three statistics, each catching what the others miss:
//!
//! - **Correlation**, which is what separates "rounds differently" from "is a
//!   different signal". A transposed weight or a wrong upsampling gives a
//!   waveform that is plausible, audible, and uncorrelated.
//! - **Relative energy error**, which catches a gain mistake that correlation
//!   is blind to - dropping the `/ num_kernels` average, for instance.
//! - **Worst absolute sample**, which catches a small number of very wrong
//!   samples that the other two would average away. An edge-handling mistake
//!   in the inverse transform looks exactly like that.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A minimal `.npy` reader; see `tests/speech_llm.rs` for why it is here.
fn npy_f32(p: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not a .npy", p.display());
    let (hlen, data_at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => panic!("{}: .npy version {v}", p.display()),
    };
    let head = std::str::from_utf8(&b[data_at..data_at + hlen]).expect("header is ascii");
    assert!(head.contains("'<f4'"), "{}: not float32", p.display());
    assert!(
        head.contains("'fortran_order': False"),
        "{}: fortran order",
        p.display()
    );
    let open = head.find('(').expect("a shape tuple");
    let close = head[open..].find(')').expect("a shape tuple") + open;
    let shape: Vec<usize> = head[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("a dimension"))
        .collect();
    let v: Vec<f32> = b[data_at + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(v.len(), shape.iter().product::<usize>(), "{}", p.display());
    (shape, v)
}

fn capture() -> Option<PathBuf> {
    let d = root().join(".golden/cosyvoice");
    d.join("wav.npy").is_file().then_some(d)
}

fn model_path() -> Option<PathBuf> {
    let p = root().join("models/tts/cosyvoice3-0.5b/hift.safetensors");
    p.is_file().then_some(p)
}

fn device() -> Option<usize> {
    std::env::var("XABE_COSY_DEVICE").ok()?.parse().ok()
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
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

#[test]
fn the_vocoder_reproduces_the_reference_waveform() {
    let (Some(dir), Some(model), Some(dev)) = (capture(), model_path(), device()) else {
        println!(
            "SKIP: needs models/tts/cosyvoice3-0.5b/hift.safetensors \
             (tools/convert_cosyvoice.py), .golden/cosyvoice \
             (tools/oracle/capture_cosyvoice.py) and XABE_COSY_DEVICE=<free card>"
        );
        return;
    };

    let v = xabe_cosy::Vocoder::open(&model, dev).expect("open the vocoder");
    let (mel_shape, mel) = npy_f32(&dir.join("mel.npy"));
    let (src_shape, source) = npy_f32(&dir.join("source.npy"));
    let (wav_shape, want) = npy_f32(&dir.join("wav.npy"));

    assert_eq!(mel_shape.len(), 3, "mel is [1, bands, frames]");
    let (bands, frames) = (mel_shape[1], mel_shape[2]);
    assert_eq!(bands, v.config().in_channels);
    let source_len = *src_shape.last().expect("source length");
    assert_eq!(wav_shape.last(), Some(&source_len), "source and wav agree");
    // The whole reason the frame arithmetic has to be exact: one sample per
    // mel frame per hop, with nothing left over.
    assert_eq!(
        source_len,
        frames * v.config().hop(),
        "the excitation should be one hop per frame"
    );

    let got = v
        .decode(
            &v.gpu().upload(&mel).expect("mel"),
            frames,
            &v.gpu().upload(&source).expect("source"),
            source_len,
        )
        .expect("decode");

    assert_eq!(got.len(), want.len(), "waveform length");

    let corr = correlation(&got, &want);
    let energy = |x: &[f32]| x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
    let (eg, ew) = (energy(&got), energy(&want));
    let gain = (eg / ew).sqrt();
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "  {} samples, correlation {corr:.6}, gain {gain:.6}, worst sample {worst:.5}",
        got.len()
    );

    // The bars are tight because the measurement is: 137,280 samples through
    // eighteen residual blocks and an inverse transform came out at
    // correlation 1.000000 and a worst sample of 1e-5. Leaving them loose
    // would let the trap this caught back in - the final leaky ReLU's slope,
    // which is 0.01 where every other one is 0.1. That mistake left every
    // stage exact, moved `conv_post` to 0.962, and gave the waveform ten times
    // its energy once the magnitudes were exponentiated.
    assert!(
        corr > 0.9999,
        "the waveform correlates {corr:.6} with the reference"
    );
    assert!(
        (gain - 1.0).abs() < 0.005,
        "the waveform's energy is {gain:.4} of the reference's"
    );
    assert!(worst < 1e-3, "worst sample differs by {worst:.5}");
    assert!(
        got.iter().all(|v| v.is_finite() && v.abs() <= 0.99 + 1e-6),
        "a sample escaped the audio limit"
    );
}
