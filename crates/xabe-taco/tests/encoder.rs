//! The encoder against the PyTorch reference, tensor for tensor.
//!
//! This is the only part of Tacotron2 that can be compared this way, and it is
//! the part where the risk sits. The prenet keeps its dropout at inference and
//! WaveGlow starts from noise, so the decoder and the vocoder are stochastic by
//! design; the encoder's dropout is conditioned on training mode and so is
//! absent, leaving embedding, three convolutions with their batch norms, and
//! one bidirectional LSTM as a deterministic function of the token ids.
//!
//! Three things this pins that nothing else would:
//!
//! - **Batch norm folding.** Eight of them are collapsed into the convolution
//!   before them at load. A wrong epsilon or a forgotten `beta` shifts every
//!   channel by a constant, which is inaudible as an error and fatal as one.
//! - **The LSTM gate order.** PyTorch lays the four gates out as input, forget,
//!   cell, output. Any other reading still produces a bounded, plausible
//!   sequence.
//! - **The direction concatenation.** A bidirectional LSTM's output is the
//!   forward state followed by the backward state *per timestep*. Reversing
//!   that, or reversing the backward pass's write order, gives a memory that is
//!   the right shape and the wrong content.
//!
//! Capture with `tools/oracle/capture_tacotron2.py`.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn the_encoder_matches_the_reference() {
    let dir = root().join("models/tts/tacotron2-nan");
    let cap = root().join(".golden/tacotron2/nan");
    if !dir.join("tacotron2.safetensors").is_file() {
        println!("SKIP: no models/tts/tacotron2-nan; run tools/convert_tacotron2.py");
        return;
    }
    if !cap.join("encoder.bin").is_file() {
        println!("SKIP: no .golden/tacotron2/nan; run tools/oracle/capture_tacotron2.py");
        return;
    }
    let Some(dev) = std::env::var("XABE_TACO_DEVICE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        println!("SKIP: set XABE_TACO_DEVICE=<free card>; see docs/TESTING.md");
        return;
    };

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(cap.join("encoder.json")).unwrap()).unwrap();
    let text = meta["text"].as_str().expect("text");
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let raw = std::fs::read(cap.join("encoder.bin")).unwrap();
    let want: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(
        want.len(),
        shape[0] * shape[1],
        "capture is not its own shape"
    );

    let taco = xabe_taco::Taco::open(&dir, dev, None, 1).expect("open");
    let (got, tokens) = taco.encoder(text).expect("encoder");

    // The token count first: if the tokeniser disagreed with the reference's
    // `text_to_sequence` there is no point comparing numbers, and the failure
    // should say so rather than show a wall of mismatched floats.
    assert_eq!(tokens, shape[0], "tokenised to a different length");
    assert_eq!(got.len(), want.len(), "memory is a different size");

    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = want
        .iter()
        .zip(&got)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb);

    println!("encoder: max-abs {worst:.3e}, cosine {cosine:.9}, {tokens} tokens");

    // Measured at 1.25e-6 on this capture, which is float32 agreeing with
    // itself across three 2560-term convolutions, a recurrence, and a `__expf`
    // sigmoid. The bound is an order of magnitude above that rather than the
    // 2e-4 the arithmetic would permit: a tolerance far looser than the
    // observed error is a test that passes through the bug it exists to catch.
    // The output is tanh-bounded to [-1, 1], so this is absolute on values of
    // order one.
    assert!(
        worst < 1e-5,
        "max-abs {worst:.3e} is too far from the reference"
    );
    assert!(
        cosine > 0.999_999,
        "cosine {cosine:.9} - this is a different tensor"
    );
}
