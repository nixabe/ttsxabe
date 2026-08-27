//! The acoustic half end to end: speech tokens in, a waveform out.
//!
//! # Why this is compared as an envelope and not sample for sample
//!
//! The excitation's phase is a **cumulative sum** of the predicted F0 over
//! every frame, so a relative difference of 1e-4 in F0 - which is what float32
//! through a flow, a mel and a five-layer predictor gives - has grown into a
//! fraction of a cycle by the end of a six-second utterance. The waveform is
//! then the same speech with its carrier slid along, which sounds identical
//! and correlates at **zero** sample for sample. Measured here: -0.001.
//!
//! `tests/vocoder.rs` can compare samples because it is handed the reference's
//! own excitation and never predicts an F0. This one predicts, so it compares
//! what survives a phase shift: the short-time energy envelope, the total
//! level, and the length - which is arithmetic and has no tolerance at all.
//!
//! The dither is a second, smaller reason. It comes from `torch.rand` buffers
//! that are not in the checkpoint and that upstream redraws on every
//! construction (see `src/source.rs`), so the engine draws its own; on the
//! reference mel that alone still leaves the waveform at correlation 0.996.
//!
//! # Why the tokens come from the capture and not from the language model
//!
//! `ras_sampling` draws with a PRNG, so two correct implementations produce
//! different speech tokens for the same sentence - measured at 21 of 143
//! positions of agreement between upstream's own sampled run and its greedy
//! argmax. The language model is therefore tested by forced log-probabilities
//! in `tests/speech_llm.rs`, and this test starts from the tokens it produced.

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
    assert!(head.contains("'fortran_order': False"), "fortran order");
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
fn the_pipeline_speaks_the_capture_s_tokens() {
    let dir = root().join(".golden/cosyvoice");
    let model = root().join("models/tts/cosyvoice3-0.5b");
    let voice = model.join("voices/taigi-ref.safetensors");
    let Some(dev) = std::env::var("XABE_COSY_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        println!("SKIP: set XABE_COSY_DEVICE=<free card>; see docs/TESTING.md");
        return;
    };
    if !dir.join("wav.npy").is_file() || !voice.is_file() {
        println!(
            "SKIP: needs .golden/cosyvoice (tools/oracle/capture_cosyvoice.py) and \
             voices/taigi-ref.safetensors (tools/make_cosyvoice_voice.py)"
        );
        return;
    }

    let instruct = "You are a helpful assistant. 請用閩南話表達。<|endofprompt|>";
    let cosy = xabe_cosy::Cosy::open(&model, &voice, instruct, dev).expect("open cosyvoice");

    let tokens: Vec<u32> = npy(&dir.join("speech_token.npy"))
        .1
        .iter()
        .map(|v| v.round() as u32)
        .collect();
    let (want_shape, want) = npy(&dir.join("wav.npy"));

    let got = cosy.vocode(&tokens).expect("vocode");
    assert_eq!(
        got.len(),
        *want_shape.last().expect("samples"),
        "the waveform is one hop per mel frame, so its length is arithmetic \
         and not a matter of tolerance"
    );

    let energy = |x: &[f32]| x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>();
    let gain = (energy(&got) / energy(&want)).sqrt();

    // One value per 10 ms, which is short enough to follow a syllable and long
    // enough not to care where inside it the carrier happens to be.
    let envelope = |x: &[f32]| -> Vec<f32> {
        x.chunks(240)
            .map(|c| (energy(c) / c.len() as f64).sqrt() as f32)
            .collect()
    };
    let (ge, we) = (envelope(&got), envelope(&want));
    let shape = correlation(&ge, &we);
    println!(
        "  {} samples, envelope correlation {shape:.4}, gain {gain:.4}, \
         samples {:.4}",
        got.len(),
        correlation(&got, &want)
    );

    // A wrong stage - a transposed mel, a condition at the wrong end of the
    // timeline, an excitation at the wrong rate - gives an envelope that is
    // audibly and measurably different speech, which is what this catches. A
    // phase shift leaves it alone.
    assert!(
        shape > 0.95,
        "the envelope correlates {shape:.4} with the reference"
    );
    assert!(
        (gain - 1.0).abs() < 0.1,
        "the waveform's energy is {gain:.3} of the reference's"
    );
    assert!(
        got.iter().all(|v| v.is_finite() && v.abs() <= 0.99 + 1e-6),
        "a sample escaped the audio limit"
    );
}
