//! The schema against the checkpoint's own inventory.
//!
//! Binding 1,259 tensors by name proves the names are right. It does not
//! prove nothing was *missed* - a schema that binds 1,200 of them and ignores
//! the rest passes every shape check it makes. So the count and the parameter
//! total are computed from the geometry and compared against what the file
//! says it holds, in both directions. That is the test that caught the
//! weight-norm mistake in `xabe-vits`.

use std::path::{Path, PathBuf};
use xabe_st::{Dtype, StSet};
use xabe_whisper::{WhisperConfig, WhisperWeights};

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

#[test]
fn every_tensor_binds_and_none_is_left_over() {
    let Some(dir) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let st = StSet::open(&dir).expect("open the sharded checkpoint");
    let cfg = WhisperConfig::from_dir(&dir).expect("config.json");
    let w = WhisperWeights::load(&st, &cfg).expect("bind the checkpoint");

    assert_eq!(st.shards(), 2, "this checkpoint is two shards");
    // Both directions. The file holding what the schema expects rules out a
    // missing tensor; the schema expecting what the file holds rules out one
    // the schema never looked at.
    assert_eq!(
        w.tensor_count(),
        st.len(),
        "the schema binds {} tensors, the checkpoint holds {}",
        w.tensor_count(),
        st.len(),
    );
    assert_eq!(
        w.parameter_count(),
        st.total_elements(),
        "the schema binds {} parameters, the checkpoint holds {}",
        w.parameter_count(),
        st.total_elements(),
    );
}

#[test]
fn the_checkpoint_is_float32_throughout() {
    let Some(dir) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let st = StSet::open(&dir).expect("open the sharded checkpoint");
    // This is why the ASR comes before the translator: no dtype conversion is
    // on its critical path, so the whole port runs on the existing F32 loader.
    // The moment this stops being true, the borrowed slices in `WhisperWeights`
    // stop being borrowable and the failure should be here, by name.
    assert_eq!(st.dtypes(), vec![Dtype::F32], "expected an F32 checkpoint");
}

#[test]
fn the_geometry_is_the_one_this_implementation_is_written_for() {
    let Some(dir) = checkpoint() else {
        panic!("models/asr/breeze-asr-26 is missing");
    };
    let cfg = WhisperConfig::from_dir(&dir).expect("config.json");
    // large-v2, which is what Breeze-ASR-26 fine-tunes. Asserted rather than
    // assumed because every buffer size downstream is derived from it.
    assert_eq!(cfg.d_model, 1280);
    assert_eq!((cfg.encoder_layers, cfg.decoder_layers), (32, 32));
    assert_eq!(cfg.encoder_head_dim(), 64);
    assert_eq!(cfg.decoder_head_dim(), 64);
    assert_eq!(cfg.num_mel_bins, 80);
    assert_eq!(cfg.n_frames(), 3000);
    assert_eq!(cfg.n_samples(), 480_000, "thirty seconds at 16 kHz");
    // 51,865 rows against a `vocab.json` of 50,258 - the difference is 1,607
    // special tokens that are synthesised, not stored. See the tokenizer.
    assert_eq!(cfg.vocab_size, 51_865);
}

#[test]
fn a_ragged_head_count_is_refused_by_name() {
    let cfg: WhisperConfig = serde_json::from_str(
        r#"{"d_model":1280,"encoder_layers":1,"decoder_layers":1,
            "encoder_attention_heads":20,"decoder_attention_heads":7,
            "encoder_ffn_dim":5120,"decoder_ffn_dim":5120,"num_mel_bins":80,
            "vocab_size":51865,"max_source_positions":1500,
            "max_target_positions":448,"decoder_start_token_id":50258,
            "eos_token_id":50257,"pad_token_id":50257}"#,
    )
    .expect("parse");
    let e = cfg.check().expect_err("1280 does not divide by 7");
    assert!(
        e.to_string().contains("decoder") && e.to_string().contains("7 heads"),
        "{e}",
    );
}
