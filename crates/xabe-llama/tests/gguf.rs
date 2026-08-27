//! The Breeze2 8B GGUF, bound against its own metadata.
//!
//! The chat model was url-only by decision for most of this project's life, so
//! this is the first time its weights are read here at all. Everything below
//! is the loader half: names, shapes, dtypes and counts, with no arithmetic.

use std::path::PathBuf;
use xabe_gguf::GgufFile;
use xabe_llama::{LlamaConfig, LlamaWeights};

fn model() -> Option<PathBuf> {
    let p =
        PathBuf::from(std::env::var("XABE_LLM_GGUF").unwrap_or_else(|_| {
            "models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf".to_string()
        }));
    // Resolve against the workspace root, since tests run in the crate dir.
    let p = if p.is_absolute() {
        p
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(p)
    };
    if p.is_file() { Some(p) } else { None }
}

macro_rules! model_or_skip {
    () => {
        match model() {
            Some(p) => p,
            None => {
                println!("SKIP: the Breeze2 GGUF is not in models/llm; set XABE_LLM_GGUF");
                return;
            }
        }
    };
}

#[test]
fn the_geometry_is_read_from_the_metadata_rather_than_assumed() {
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");
    let cfg = LlamaConfig::from_gguf(&f).expect("read the geometry");

    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.intermediate_size, 14336);
    assert_eq!(cfg.num_hidden_layers, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8, "grouped-query, 4 to 1");
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.kv_dim(), 1024);
    assert_eq!(cfg.vocab_size, 128_256);
    assert_eq!(cfg.max_position_embeddings, 131_072);
    assert_eq!(cfg.rms_norm_eps, 1e-5);

    // Not 10000. Llama-3 stretched the rope base by fifty times to reach a
    // 128k context, and a loader that defaulted this would produce a model
    // that is fluent for a sentence and drifts after that.
    assert_eq!(cfg.rope_theta, 500_000.0);

    assert!(
        !cfg.tie_word_embeddings,
        "output.weight exists, so the embeddings are untied"
    );
    assert_eq!(cfg.bos_token_id, 128_000);
    assert_eq!(cfg.eos_token_id, 128_009);
}

#[test]
fn every_one_of_the_292_tensors_binds_and_none_is_left_over() {
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");
    let cfg = LlamaConfig::from_gguf(&f).expect("geometry");
    let w = LlamaWeights::from_gguf(&f, &cfg).expect("bind every tensor");

    // The whole point of the exercise: the schema accounts for the file
    // exactly. 32 blocks of 9, plus the embedding, the final norm, the output
    // projection and Llama-3's rope scaling.
    assert_eq!(f.len(), 292, "what the file says it holds");
    assert_eq!(w.tensor_count(), 292, "what the schema bound");
    assert_eq!(32 * 9 + 4, 292, "the arithmetic behind that number");

    let bound: std::collections::HashSet<&str> = w.tensors().map(|b| b.name.as_str()).collect();
    let mut missed: Vec<&str> = f
        .tensors()
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !bound.contains(n))
        .collect();
    missed.sort_unstable();
    assert!(missed.is_empty(), "unbound tensors: {missed:?}");

    assert_eq!(
        w.parameter_count(),
        8_030_261_312,
        "8.03 B, as the name says"
    );
    assert!(w.rope_freqs.is_some(), "Llama-3 carries rope_freqs.weight");
}

#[test]
fn the_key_and_value_projections_are_narrow_because_the_model_is_grouped_query() {
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");
    let cfg = LlamaConfig::from_gguf(&f).expect("geometry");
    let w = LlamaWeights::from_gguf(&f, &cfg).expect("bind");

    let a = &w.layers[0].attn;
    assert_eq!(a.q.shape, vec![4096, 4096]);
    assert_eq!(a.o.shape, vec![4096, 4096]);
    // The assertion that would have failed under the old square-only schema,
    // and the reason grouped-query had to become bindable rather than refused.
    assert_eq!(a.k.shape, vec![1024, 4096]);
    assert_eq!(a.v.shape, vec![1024, 4096]);
}

#[test]
fn the_stored_dims_are_the_reverse_of_the_bound_shapes() {
    // GGUF writes the fastest-varying dimension first. Binding against `dims`
    // instead of `shape` would agree for every square projection and silently
    // transpose the two that are not, which is the worst possible failure
    // shape: 30 of 32 layers correct.
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");

    let k = f.info("blk.0.attn_k.weight").expect("k exists");
    assert_eq!(k.dims, vec![4096, 1024], "as stored");
    assert_eq!(k.shape(), vec![1024, 4096], "as the geometry reads it");

    let down = f.info("blk.0.ffn_down.weight").expect("ffn_down exists");
    assert_eq!(down.dims, vec![14336, 4096]);
    assert_eq!(down.shape(), vec![4096, 14336]);
}

#[test]
fn the_weights_are_f16_and_the_norms_are_f32() {
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");
    let cfg = LlamaConfig::from_gguf(&f).expect("geometry");
    let w = LlamaWeights::from_gguf(&f, &cfg).expect("bind");

    use xabe_st::Dtype;
    assert_eq!(w.layers[0].attn.q.dtype, Dtype::F16);
    assert_eq!(w.embed_tokens.dtype, Dtype::F16);
    // Norms stay f32 in a GGUF: they are one vector per layer, so the space
    // saved is nothing and the precision matters more than anywhere else.
    assert_eq!(w.layers[0].attn_norm.dtype, Dtype::F32);
    assert_eq!(w.norm.dtype, Dtype::F32);

    let counts = f
        .tensors()
        .iter()
        .fold((0, 0), |(h, s), t| match t.ggml_type {
            xabe_gguf::GgmlType::F16 => (h + 1, s),
            _ => (h, s + 1),
        });
    assert_eq!(
        counts,
        (226, 66),
        "226 f16 weights, 66 f32 norms and scales"
    );
}

#[test]
fn a_tensor_reads_back_at_both_widths() {
    // Proves the mapping actually resolves to bytes, not just that the
    // directory parsed. The final norm is 4096 f32 and cheap to touch; the
    // embedding is 2 GB and deliberately not read here.
    let path = model_or_skip!();
    let f = GgufFile::open(&path).expect("open the GGUF");

    let wide = f.tensor_f32("output_norm.weight").expect("f32");
    assert_eq!(wide.len(), 4096);
    assert!(
        wide.iter().all(|v| v.is_finite()),
        "a norm full of NaN means the offset arithmetic is wrong"
    );

    let narrow = f.tensor_f16("output_norm.weight").expect("f16");
    assert_eq!(narrow.len(), 4096);
    assert_eq!(
        half::f16::from_bits(narrow[0]),
        half::f16::from_f32(wide[0]),
        "the two accessors must agree on the same element"
    );
}
