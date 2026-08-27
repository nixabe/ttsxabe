//! The excitation path against CosyVoice3's own.
//!
//! Two stages with two very different characters, tested separately:
//!
//! - **The F0 predictor** is five convolutions and a linear head, and upstream
//!   runs it in **float64** on the grounds that "precision is crucial for
//!   causal inference". This runs float32. That is a deliberate difference, so
//!   the test measures how much it costs rather than assuming it away.
//! - **The oscillator bank** is arithmetic with no weights except a nine-wide
//!   linear, and its inputs include three buffers that are not in the
//!   checkpoint. Given the same buffers it should reproduce the reference
//!   closely; anything else is a misreading of the algorithm.

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

fn capture() -> Option<PathBuf> {
    let d = root().join(".golden/cosyvoice");
    d.join("source.npy").is_file().then_some(d)
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
fn the_excitation_path_reproduces_the_reference() {
    let (Some(dir), Some(dev)) = (capture(), device()) else {
        println!("SKIP: needs .golden/cosyvoice and XABE_COSY_DEVICE=<free card>");
        return;
    };
    let model = root().join("models/tts/cosyvoice3-0.5b/hift.safetensors");
    if !model.is_file() {
        println!("SKIP: run tools/convert_cosyvoice.py");
        return;
    }

    let f = xabe_st::StFile::open(&model).expect("open hift");
    let gpu = xabe_cuda::Gpu::open(dev).expect("device");
    let f0p = xabe_cosy::F0Predictor::bind(&f, &gpu).expect("bind the f0 predictor");

    let (mel_shape, mel) = npy(&dir.join("mel.npy"));
    let frames = mel_shape[2];
    let (_, want_f0) = npy(&dir.join("f0.npy"));

    // 1. F0, float32 here against float64 there.
    let got_f0 = f0p
        .predict(&gpu, &gpu.upload(&mel).expect("mel"), frames)
        .expect("predict");
    assert_eq!(got_f0.len(), want_f0.len());
    let corr = correlation(&got_f0, &want_f0);
    let worst = got_f0
        .iter()
        .zip(&want_f0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want_f0.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("  f0: correlation {corr:.6}, worst {worst:.4} Hz against a peak of {scale:.1} Hz");
    assert!(corr > 0.9999, "f0 correlates {corr:.6}");
    // A tenth of a hertz is inaudible; the point of the bound is that float32
    // costs a tenth of a hertz and not ten.
    assert!(worst < 0.5, "f0 differs by {worst} Hz");

    // 2. The oscillators, given the reference's own F0 so the two stages stay
    //    separable, and the reference's own dither because it is not derivable.
    let (_, rand_ini) = npy(&dir.join("sine_rand_ini.npy"));
    let (sw_shape, sine_waves) = npy(&dir.join("sine_waves.npy"));
    let (src_shape, want_src) = npy(&dir.join("source.npy"));
    assert_eq!(sw_shape[2], xabe_cosy::HARMONICS, "nine harmonics");

    let dither = xabe_cosy::Dither {
        rand_ini,
        sine_waves,
        len: sw_shape[1],
    };
    let lw = f
        .tensor_shaped("m_source.l_linear.weight", &[1, xabe_cosy::HARMONICS])
        .expect("l_linear.weight");
    let lb = f
        .tensor_shaped("m_source.l_linear.bias", &[1])
        .expect("bias")[0];

    let got =
        xabe_cosy::excitation(&want_f0, &Default::default(), &dither, lw, lb).expect("excitation");
    assert_eq!(got.len(), *src_shape.last().expect("len"));

    let corr = correlation(&got, &want_src);
    let worst = got
        .iter()
        .zip(&want_src)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  excitation: correlation {corr:.6}, worst {worst:.6}");
    assert!(corr > 0.9999, "the excitation correlates {corr:.6}");
    assert!(worst < 5e-3, "worst excitation sample differs by {worst}");
}

#[test]
fn the_whole_vocoder_chain_reproduces_the_reference_waveform() {
    // Stages two and three wired together: mel in, waveform out, with nothing
    // taken from the capture but the dither. Each half is verified on its own
    // elsewhere; this is the one that catches a mistake *between* them - a
    // wrong rate, a transposed excitation, an off-by-one in the frame count -
    // which is invisible to both.
    let (Some(dir), Some(dev)) = (capture(), device()) else {
        println!("SKIP: needs .golden/cosyvoice and XABE_COSY_DEVICE=<free card>");
        return;
    };
    let model = root().join("models/tts/cosyvoice3-0.5b/hift.safetensors");
    if !model.is_file() {
        println!("SKIP: run tools/convert_cosyvoice.py");
        return;
    }

    let v = xabe_cosy::Vocoder::open(&model, dev).expect("open the vocoder");
    let f = xabe_st::StFile::open(&model).expect("open hift");
    let f0p = xabe_cosy::F0Predictor::bind(&f, v.gpu()).expect("bind");

    let (mel_shape, mel) = npy(&dir.join("mel.npy"));
    let frames = mel_shape[2];
    let (_, rand_ini) = npy(&dir.join("sine_rand_ini.npy"));
    let (sw_shape, sine_waves) = npy(&dir.join("sine_waves.npy"));
    let (_, want) = npy(&dir.join("wav.npy"));

    let dither = xabe_cosy::Dither {
        rand_ini,
        sine_waves,
        len: sw_shape[1],
    };
    let lw = f
        .tensor_shaped("m_source.l_linear.weight", &[1, xabe_cosy::HARMONICS])
        .expect("l_linear.weight");
    let lb = f
        .tensor_shaped("m_source.l_linear.bias", &[1])
        .expect("bias")[0];

    let gmel = v.gpu().upload(&mel).expect("mel");
    let f0 = f0p.predict(v.gpu(), &gmel, frames).expect("f0");
    let source =
        xabe_cosy::excitation(&f0, &Default::default(), &dither, lw, lb).expect("excitation");
    let got = v
        .decode(
            &gmel,
            frames,
            &v.gpu().upload(&source).expect("source"),
            source.len(),
        )
        .expect("decode");

    assert_eq!(got.len(), want.len());
    let corr = correlation(&got, &want);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  chained: correlation {corr:.6}, worst sample {worst:.6}");
    assert!(corr > 0.9999, "the chained waveform correlates {corr:.6}");
    // Looser than the vocoder's own 1e-3, and the reason is worth writing
    // down: fed the reference's excitation the waveform came out at a worst
    // sample of 1e-5, and fed this engine's - which differs from the
    // reference's by 1.3e-4 - it comes out at 6.5e-3. The magnitude head
    // exponentiates, so a small difference in the excitation is a larger one
    // in the waveform, and that is arithmetic rather than a defect. At -44 dB
    // against the peak on one sample in 137,280 it is inaudible; at ten times
    // this it would not be, which is where the bound sits.
    assert!(worst < 1e-2, "worst chained sample differs by {worst}");
}

#[test]
fn a_dither_bundle_too_short_for_the_utterance_is_refused() {
    // Silently truncating would make the tail of a long sentence fade to
    // nothing, which is audible and very hard to trace back to a capture taken
    // on a shorter one.
    let d = xabe_cosy::Dither {
        rand_ini: vec![0.0; xabe_cosy::HARMONICS],
        sine_waves: vec![0.0; 10 * xabe_cosy::HARMONICS],
        len: 10,
    };
    assert!(d.check(10).is_ok());
    match d.check(11) {
        Err(xabe_cosy::CosyError::Speaker { what }) => assert!(what.contains("11"), "{what}"),
        other => panic!("wanted a refusal, got {other:?}"),
    }
}
