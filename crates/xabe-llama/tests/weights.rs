//! The schema against the checkpoint's own inventory.
//!
//! Binding 363 tensors by name proves the names are right; it does not prove
//! nothing was missed, since a schema that binds 350 and ignores the rest
//! passes every shape check it makes. So the count and the parameter total are
//! computed from the geometry and compared against what the file says it
//! holds, in both directions - the test that caught the weight-norm mistake in
//! `xabe-vits`, at forty times the scale.

use std::path::{Path, PathBuf};
use xabe_llama::{LlamaConfig, LlamaWeights};
use xabe_st::{Dtype, StSet};

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/translator/taigi-llama2-13b");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

#[test]
fn every_tensor_binds_and_none_is_left_over() {
    let Some(dir) = checkpoint() else {
        panic!("models/translator/taigi-llama2-13b is missing");
    };
    let st = StSet::open(&dir).expect("open the sharded checkpoint");
    let cfg = LlamaConfig::from_dir(&dir).expect("config.json");
    let w = LlamaWeights::load(&st, &cfg).expect("bind the checkpoint");

    assert_eq!(st.shards(), 6, "this checkpoint is six shards");
    assert_eq!(w.tensor_count(), 363);
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
    // Thirteen billion, which is the number on the tin - stated here so that a
    // future checkpoint quietly swapped for a 7 B one fails rather than loads.
    assert_eq!(w.parameter_count(), 13_261_870_080);
}

#[test]
fn the_checkpoint_is_brain_float_which_this_card_cannot_run() {
    let Some(dir) = checkpoint() else {
        panic!("models/translator/taigi-llama2-13b is missing");
    };
    let st = StSet::open(&dir).expect("open the sharded checkpoint");
    // This is the finding that shaped the whole phase. `sm_75` has no bf16, so
    // the weights cannot be used as stored; f32 would be 53 GB against a 48 GB
    // card, so widening is not an option either. f16 is the only width the
    // model fits in, which is why `StFile::tensor_f16` exists and why its
    // range check is not optional.
    assert_eq!(st.dtypes(), vec![Dtype::Bf16]);
    assert_eq!(
        st.total_elements() * 4,
        53_047_480_320,
        "f32 would be 53 GB"
    );
}

#[test]
fn narrowing_a_real_tensor_to_f16_stays_in_range() {
    let Some(dir) = checkpoint() else {
        panic!("models/translator/taigi-llama2-13b is missing");
    };
    let st = StSet::open(&dir).expect("open the sharded checkpoint");
    // The check has to run on real weights at least once, or it only proves
    // that a synthetic overflow is caught. A trained Llama's weights sit well
    // inside f16's range - but "well inside" is a belief until something reads
    // 26 million of them and says so.
    for name in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.39.mlp.down_proj.weight",
        "model.norm.weight",
    ] {
        let packed = st
            .tensor_f16(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let wide = st.tensor_f32(name).expect("widen");
        assert_eq!(packed.len(), wide.len(), "{name}");
        assert!(
            packed.iter().all(|&b| half::f16::from_bits(b).is_finite()),
            "{name} produced a non-finite value",
        );
    }
}

#[test]
fn the_geometry_is_the_one_this_schema_is_written_for() {
    let Some(dir) = checkpoint() else {
        panic!("models/translator/taigi-llama2-13b is missing");
    };
    let cfg = LlamaConfig::from_dir(&dir).expect("config.json");
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_attention_heads, 40);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.intermediate_size, 13_824);
    assert!(!cfg.tie_word_embeddings, "lm_head is a tensor of its own");
    // 56,024 rows against a tokenizer of 56,020. The embedding is padded and
    // the last four rows are unused; a loader that takes the tokenizer's size
    // for the vocabulary binds `lm_head` four rows short.
    assert_eq!(cfg.vocab_size, 56_024);
}

#[test]
fn grouped_query_attention_is_refused_rather_than_bound_wrongly() {
    // Llama-2 13B has as many key-value heads as query heads, so all four
    // projections are square. A checkpoint with fewer would bind here with the
    // wrong expected shape and be refused for the wrong reason - "k_proj is
    // [1280, 5120], expected [5120, 5120]" tells a reader nothing about what
    // is actually going on.
    let cfg: LlamaConfig = serde_json::from_str(
        r#"{"hidden_size":5120,"intermediate_size":13824,"num_hidden_layers":40,
            "num_attention_heads":40,"num_key_value_heads":8,"vocab_size":56024,
            "max_position_embeddings":4096,"rms_norm_eps":1e-5,"rope_theta":10000.0,
            "tie_word_embeddings":false,"bos_token_id":1,"eos_token_id":2}"#,
    )
    .expect("parse");
    let e = cfg.check().expect_err("8 kv heads against 40 query heads");
    assert!(e.to_string().contains("key-value heads"), "{e}");
}
