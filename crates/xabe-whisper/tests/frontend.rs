//! The mel frontend against 🤗 `WhisperFeatureExtractor`.
//!
//! The captures come from `tools/oracle/capture_asr.py`, which runs the
//! reference on CPU in float32 with one thread and writes every stage as raw
//! little-endian f32. Nothing here is hand-transcribed; if a number in a
//! comment disagrees with a capture, the capture is right.

use std::path::{Path, PathBuf};
use xabe_audio::{MelConfig, mel_filters};
use xabe_whisper::{F_MAX, Frontend, WhisperConfig};

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    p.join("config.json").is_file().then_some(p)
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

/// Largest absolute disagreement, and where it was.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| ((x - y).abs(), i))
        .fold((0.0f32, 0), |acc, e| if e.0 > acc.0 { e } else { acc })
}

#[test]
fn the_filter_bank_is_computed_not_shipped() {
    let Some(dir) = captures().into_iter().next() else {
        panic!("no captures in .golden/asr - run tools/oracle/capture_asr.py");
    };
    let cfg = MelConfig::default();
    let want = read_f32(&dir.join("mel_filters.bin"));
    assert_eq!(
        want.len(),
        cfg.n_freq() * cfg.n_mels,
        "capture is not [201, 80]"
    );

    let got = mel_filters(&cfg, F_MAX);
    let (e, at) = worst(&want, &got);
    // Bit-identical, measured, not hoped for. Both sides evaluate the same
    // closed form in float64 and round once at the end, and there is no
    // reduction anywhere in it for an ordering to disagree about - so the
    // bound here is zero rather than a tolerance. If this ever needs a
    // tolerance, something changed that is worth finding out about.
    assert_eq!(e, 0.0, "filter {at}: {e:e}");
}

#[test]
fn log_mel_matches_the_reference_on_every_clip() {
    let Some(model) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let cfg = WhisperConfig::from_dir(&model).expect("config.json");
    let fe = Frontend::new(&cfg);

    let clips = captures();
    assert!(!clips.is_empty(), "no captures in .golden/asr");
    for dir in clips {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let samples = read_f32(&dir.join("samples.bin"));
        let want = read_f32(&dir.join("input_features.bin"));
        assert_eq!(want.len(), cfg.num_mel_bins * cfg.n_frames(), "{name}");

        let got = fe.log_mel(&samples);
        let (e, at) = worst(&want, &got);
        // Both sides take a 400-point transform of the same float32 samples
        // and neither is the definition of the other's rounding, so the bound
        // is a transform's worth of float32 error, not zero. Measured worst
        // case across the corpus is 2.4e-5, on features that live in about
        // [-1, 1] after the affine rescale - a couple of parts in a hundred
        // thousand of full scale. The gate is set at twice that.
        assert!(
            e < 5e-5,
            "{name}: mel[{}, {}] off by {e:e}",
            at / cfg.n_frames(),
            at % cfg.n_frames(),
        );
    }
}
