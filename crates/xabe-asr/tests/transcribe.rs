//! Greedy decoding against the reference's, token for token.
//!
//! This is the test that matters. The per-layer diffs in `oracle.rs` locate a
//! failure; this one says whether the engine transcribes what the reference
//! transcribes, which is the only claim a user cares about.

use std::path::{Path, PathBuf};
use xabe_asr::AsrModel;

/// The checkpoint, or `None` if it is not on this machine.
///
/// The same distinction [`model`] draws one step later, applied one step
/// earlier: **absent is a skip, present-but-broken is a failure.** A machine
/// that was never given the models - a fresh checkout, or CI, where
/// `.gitignore` keeps `models/` out of the repository entirely - has nothing
/// to test and says so. A machine where `models/asr/` exists but the shard
/// index inside it does not has a half-populated tree, which is a setup
/// mistake worth shouting about rather than passing over in silence.
///
/// These tests used to panic on both, which made them correct on a developer's
/// box and wrong everywhere else.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    if !p.is_dir() {
        return None;
    }
    assert!(
        p.join("model.safetensors.index.json").is_file(),
        "{} exists but has no model.safetensors.index.json; \
         a half-populated checkpoint is a setup error, not an absent one",
        p.display()
    );
    Some(p)
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

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn greedy_decoding_reproduces_the_reference_transcripts() {
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is not on this machine");
        return;
    };
    let Some(m) = model(&dir) else { return };
    let clips = captures();
    assert!(!clips.is_empty(), "no captures in .golden/asr");

    for clip in clips {
        let name = clip.file_name().unwrap().to_string_lossy().to_string();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip.join("manifest.json")).unwrap())
                .expect("manifest");
        let want_ids: Vec<u32> = manifest["generated_ids"]
            .as_array()
            .expect("generated_ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let want_text = manifest["transcript"].as_str().expect("transcript");

        // The captured features, so this measures decoding rather than the
        // frontend - which has its own test, in `xabe-whisper`.
        let mel = read_f32(&clip.join("input_features.bin"));
        let got_ids = m.generate(&mel, "zh", 64).expect("generate");
        let got_text = m.tokenizer().decode(&got_ids, true);

        println!("  {name:<12} {got_text:?}");
        assert_eq!(got_ids, want_ids, "{name}: {got_text:?} vs {want_text:?}");
        assert_eq!(got_text, want_text, "{name}");
    }
}

#[test]
fn the_whole_pipeline_from_samples_agrees_too() {
    // The same clips through this engine's own frontend rather than the
    // reference's features. Both stages have their own gate; this is the one
    // that says they compose.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is not on this machine");
        return;
    };
    let Some(m) = model(&dir) else { return };

    for clip in captures() {
        let name = clip.file_name().unwrap().to_string_lossy().to_string();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(clip.join("manifest.json")).unwrap())
                .expect("manifest");
        let want = manifest["transcript"].as_str().expect("transcript");
        let samples = read_f32(&clip.join("samples.bin"));
        let got = m.transcribe(&samples, "zh").expect("transcribe");
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn a_decode_past_the_learned_positions_is_refused_by_name() {
    // Whisper's decoder has 448 learned positions and no extrapolation. Past
    // them the position embedding would be read out of bounds; a model that
    // wrapped instead would produce fluent nonsense.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is not on this machine");
        return;
    };
    let Some(m) = model(&dir) else { return };
    let mel = vec![0.0f32; m.config().num_mel_bins * m.config().n_frames()];
    let encoded = m.encode(&mel).expect("encode");
    let mut cache = m.cache(&encoded).expect("cache");
    let too_many = vec![m.config().decoder_start_token_id; m.config().max_target_positions + 1];
    let e = m.decode(&too_many, &mut cache).expect_err("past the end");
    assert!(e.to_string().contains("449"), "{e}");
}
