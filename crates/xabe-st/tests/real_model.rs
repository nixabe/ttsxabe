//! Reads the actual mms-tts-nan checkpoint.
//!
//! Skips loudly when the file is absent: a skipped test is not a passing test,
//! so it says why. Point `XABE_TTS_MODEL` at a `model.safetensors`, or let it
//! find the HuggingFace cache copy.

use xabe_st::StFile;

/// Locates the checkpoint, or `None` if this machine does not have it.
fn find_model() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("XABE_TTS_MODEL") {
        let p = std::path::PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    // The consolidated model tree is the canonical home. The HuggingFace cache
    // is kept as a fallback so a checkout that never ran the move still tests.
    let local = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/mms-tts-nan/model.safetensors");
    if local.is_file() {
        return Some(local);
    }
    let home = std::env::var("HOME").ok()?;
    let root = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--facebook--mms-tts-nan/snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    let f = snap.join("model.safetensors");
    f.is_file().then_some(f)
}

#[test]
fn opens_the_real_checkpoint() {
    let Some(path) = find_model() else {
        eprintln!("SKIP: mms-tts-nan checkpoint not found; set XABE_TTS_MODEL");
        return;
    };

    let f = StFile::open(&path).expect("open the real checkpoint");
    assert_eq!(f.len(), 762, "checkpoint tensor count");

    // Spot-check the two ends of the network. The embedding is the entry point
    // and conv_post the exit; if either shape moved, the weight schema is stale.
    let emb = f
        .tensor_shaped("text_encoder.embed_tokens.weight", &[48, 192])
        .expect("embed_tokens");
    assert_eq!(emb.len(), 48 * 192);
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "embedding holds a non-finite value"
    );

    f.tensor_shaped("decoder.conv_post.weight", &[1, 32, 7])
        .expect("conv_post");

    let params: usize = f.tensors().map(|(_, i)| i.numel()).sum();
    assert_eq!(params, 36_286_512, "total parameter count");
}

#[test]
fn every_tensor_is_finite_and_addressable() {
    let Some(path) = find_model() else {
        eprintln!("SKIP: mms-tts-nan checkpoint not found; set XABE_TTS_MODEL");
        return;
    };
    let f = StFile::open(&path).expect("open");

    for (name, info) in f.tensors() {
        let data = f
            .tensor(name)
            .expect("tensor listed in the directory exists");
        assert_eq!(
            data.len(),
            info.numel(),
            "{name}: length disagrees with shape"
        );
        assert!(
            data.iter().all(|v| v.is_finite()),
            "{name}: holds a non-finite value"
        );
    }
}
