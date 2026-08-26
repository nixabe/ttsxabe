//! The forward pass against 🤗 `WhisperForConditionalGeneration`, per layer.
//!
//! The captures come from `tools/oracle/capture_asr.py`, which runs the
//! reference on CPU in float32 with one thread and writes every stage as raw
//! little-endian f32. The taps are per layer on purpose: "the encoder is
//! wrong" is not a fact anyone can act on, and "layer 7 is wrong" is.

use std::path::{Path, PathBuf};
use xabe_asr::AsrModel;

/// How many layers the capture tapped individually.
const TAPS: usize = 4;

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

/// Every captured clip directory.
fn captures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/asr");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    out.sort();
    out
}

/// Which device to use. See `docs/TESTING.md`; check `nvidia-smi` first.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Opens the model, skipping only when there genuinely is no device.
///
/// Skipping on *any* error is a trap this workspace has fallen into once: a
/// kernel that failed to compile was reported as an absent GPU and twelve
/// tests passed without running.
fn model(dir: &Path) -> Option<AsrModel> {
    match AsrModel::open(dir, ordinal()) {
        Ok(m) => Some(m),
        Err(xabe_asr::AsrError::Cuda(xabe_cuda::CudaError::NoDevice(why))) => {
            eprintln!("SKIP: no CUDA device ({why})");
            None
        }
        Err(e) => panic!("the checkpoint is present but unusable: {e}"),
    }
}

/// Reads a `.bin` of little-endian f32.
fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Largest absolute disagreement scaled by the reference's own scale, with the
/// index it happened at.
///
/// Scaled by the largest magnitude in the reference rather than element by
/// element: a residual stream has values at every order of magnitude, and a
/// per-element relative error on the small ones reports noise. See the trap
/// recorded in `docs/TESTING.md`.
fn worst(want: &[f32], got: &[f32]) -> (f32, usize, f32) {
    assert_eq!(want.len(), got.len(), "length mismatch");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let (mut e, mut at) = (0.0f32, 0);
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        let d = (a - b).abs();
        if d > e {
            e = d;
            at = i;
        }
    }
    (e / scale, at, scale)
}

/// The gate every stage is held to, as a fraction of the tensor's own scale.
///
/// The matmul rounds both operands to f16 before multiplying and accumulates
/// in f32; the reference does neither. One projection of a 1280-long
/// contraction costs about 6.5e-5 of full scale - measured, in `xabe-cuda`'s
/// `f16_operands_cost_six_parts_in_a_hundred_thousand` - and a 32-layer
/// residual stream compounds it. Measured worst across the corpus is printed
/// by every run of these tests; the gate is set a little above it.
///
/// Measured: 1e-4 at encoder layer 0 rising to 7.5e-3 at `encoder_out` 32
/// layers later, which is the f16 rounding compounding with depth at roughly
/// 1.2e-4 a layer. The decoder, reading that output, stays under 1e-3. The
/// gate is 1.5e-2 - twice the worst - and it is *not* the test that matters:
/// the argmax check below and the transcript test in `transcribe.rs` are.
/// This table exists so that a change in the shape of the drift is visible,
/// not so that a number can be defended.
const GATE: f32 = 1.5e-2;

/// Every stage's disagreement, reported together.
///
/// `docs/TESTING.md` asks for all the metrics rather than the first failure:
/// one stage over the gate is a number, and the whole table is a shape - a
/// single bad layer looks different from a drift that compounds, and only the
/// table tells them apart.
#[derive(Default)]
struct Report(Vec<(String, f32)>);

impl Report {
    fn add(&mut self, name: String, want: &[f32], got: &[f32]) {
        let (e, at, scale) = worst(want, got);
        println!("  {name:<34} {e:.3e}   at {at:>8}, scale {scale:.4}");
        self.0.push((name, e));
    }

    fn check(self) {
        let (name, e) = self
            .0
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("no NaN"))
            .expect("nothing was compared");
        println!("  worst: {name} at {e:.3e}, gate {GATE:.3e}");
        assert!(e < GATE, "{name}: {e:e} of full scale, gate {GATE:e}");
    }
}

#[test]
fn the_encoder_matches_the_oracle_layer_by_layer() {
    let Some(dir) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let Some(m) = model(&dir) else { return };
    let clips = captures();
    assert!(!clips.is_empty(), "no captures in .golden/asr");
    let mut report = Report::default();

    for clip in clips {
        let name = clip.file_name().unwrap().to_string_lossy().to_string();
        // The *captured* features, not ones computed here, so a frontend
        // disagreement cannot leak into an encoder measurement. The frontend
        // has its own test.
        let mel = read_f32(&clip.join("input_features.bin"));
        let (out, taps) = m.encode_tapped(&mel, TAPS).expect("encode");

        for (i, got) in taps.iter().enumerate() {
            let want = read_f32(&clip.join(format!("encoder_layer_{i}.bin")));
            report.add(format!("{name} encoder layer {i}"), &want, got);
        }
        let want = read_f32(&clip.join("encoder_out.bin"));
        let got = m.gpu().download(&out).expect("download");
        report.add(format!("{name} encoder out"), &want, &got);
    }
    report.check();
}

#[test]
fn the_decoder_matches_the_oracle_layer_by_layer() {
    let Some(dir) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let Some(m) = model(&dir) else { return };

    let mut report = Report::default();
    for clip in captures() {
        let name = clip.file_name().unwrap().to_string_lossy().to_string();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip.join("manifest.json")).unwrap())
                .expect("manifest");
        let ids: Vec<u32> = manifest["decoder_ids"]
            .as_array()
            .expect("decoder_ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();

        // The encoder's own output, not the capture's, because that is what
        // the decoder will see in production. Its disagreement is inside the
        // gate and this test measures what the two stages do together.
        let mel = read_f32(&clip.join("input_features.bin"));
        let enc = m.encode(&mel).expect("encode");
        let mut cache = m.cache(&enc).expect("cache");
        let (logits, taps) = m.decode_tapped(&ids, &mut cache, TAPS).expect("decode");

        for (i, got) in taps.iter().take(TAPS).enumerate() {
            let want = read_f32(&clip.join(format!("decoder_layer_{i}.bin")));
            report.add(format!("{name} decoder layer {i}"), &want, got);
        }
        let want = read_f32(&clip.join("decoder_out.bin"));
        report.add(
            format!("{name} decoder out"),
            &want,
            taps.last().expect("final norm tap"),
        );

        let want = read_f32(&clip.join("logits.bin"));
        let got = m.gpu().download(&logits).expect("download");
        report.add(format!("{name} logits"), &want, &got);

        // What actually matters about the logits is which token wins, so that
        // is asserted separately from how close the values are.
        let vocab = m.config().vocab_size;
        for t in 0..ids.len() {
            let arg = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
                    .expect("non-empty")
                    .0
            };
            assert_eq!(
                arg(&want[t * vocab..][..vocab]),
                arg(&got[t * vocab..][..vocab]),
                "{name}: a different token wins at position {t}",
            );
        }
    }
    report.check();
}
