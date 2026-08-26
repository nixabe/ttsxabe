//! Sharded checkpoints, against the two real ones on this machine.
//!
//! Both models past the synthesiser ship split: the ASR across two files and
//! the translator across six. Neither can be read at all until the index is,
//! so this is the first thing phase 4 needs and the last place a mistake would
//! be noticed - a checkpoint that half-agrees with its own manifest loads and
//! is wrong somewhere specific.

use std::path::{Path, PathBuf};
use xabe_st::{Dtype, StSet};

fn models() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn asr() -> Option<PathBuf> {
    let p = models().join("asr/breeze-asr-26");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

fn translator() -> Option<PathBuf> {
    let p = models().join("translator/taigi-llama2-13b");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

#[test]
fn the_asr_checkpoint_reads_as_one_set_of_two_shards() {
    let Some(dir) = asr() else {
        eprintln!("SKIP: no ASR checkpoint under models/asr/breeze-asr-26");
        return;
    };
    let set = StSet::open(&dir).expect("open the ASR checkpoint");

    // Cross-checked against the file's own index, not against a number written
    // here: 989 tensors in the first shard and 270 in the second.
    assert_eq!(set.shards(), 2);
    assert_eq!(set.len(), 1259);
    assert_eq!(set.dtypes(), vec![Dtype::F32]);

    // 1.5 B parameters. The exact count is the checkpoint's business; what this
    // asserts is that reading it as one set does not lose or double anything.
    let total = set.total_elements();
    assert!(
        (1_500_000_000..1_600_000_000).contains(&total),
        "{total} parameters is not a 1.5 B model",
    );

    // A tensor from each shard, to prove placement is used rather than guessed.
    let first = set
        .tensor("model.encoder.conv1.weight")
        .expect("an encoder tensor");
    assert_eq!(
        first.len(),
        1280 * 80 * 3,
        "1280 channels out of 80 mel bins, kernel 3"
    );
    assert!(set.info("model.decoder.embed_tokens.weight").is_some());
}

#[test]
fn the_translator_reads_as_six_shards_of_bf16() {
    let Some(dir) = translator() else {
        eprintln!("SKIP: no translator checkpoint under models/translator");
        return;
    };
    let set = StSet::open(&dir).expect("open the translator");

    assert_eq!(set.shards(), 6);
    assert_eq!(set.len(), 363);

    // This card is Turing and has no bf16 arithmetic at all, which is the whole
    // reason xabe-st widens rather than borrowing.
    assert_eq!(set.dtypes(), vec![Dtype::Bf16]);

    let embed = set
        .info("model.embed_tokens.weight")
        .expect("the embedding table");
    assert_eq!(embed.shape, vec![56_024, 5_120], "vocab x hidden");

    // Borrowing bf16 as f32 must be refused rather than reinterpreted: two
    // bf16 values read as one f32 is a number, just not this one.
    let err = set.tensor("model.embed_tokens.weight").unwrap_err();
    assert!(err.to_string().contains("BF16"), "{err}");
}

#[test]
fn widening_a_sharded_tensor_gives_the_right_count_and_finite_values() {
    let Some(dir) = translator() else {
        eprintln!("SKIP: no translator checkpoint");
        return;
    };
    let set = StSet::open(&dir).expect("open the translator");

    // A small one, so the test does not widen 287 M elements to prove a point.
    let name = "model.layers.0.input_layernorm.weight";
    let info = set.info(name).expect("a layernorm");
    let got = set.tensor_f32(name).expect("widen");

    assert_eq!(got.len(), info.numel());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "widening produced non-finite values"
    );
    // An RMSNorm gain is positive and near one; a byte-order or shift mistake
    // shows up here as values that are enormous or zero.
    let mean = got.iter().sum::<f32>() / got.len() as f32;
    assert!(
        (0.01..10.0).contains(&mean),
        "mean gain {mean} is not plausible"
    );
}

#[test]
fn a_single_file_checkpoint_is_a_set_of_one() {
    // So callers never branch on whether a model happens to be sharded.
    let dir = models().join("tts/mms-tts-nan");
    if !dir.join("model.safetensors").is_file() {
        eprintln!("SKIP: no mms-tts-nan checkpoint");
        return;
    }
    let set = StSet::open(&dir).expect("open the TTS checkpoint");
    assert_eq!(set.shards(), 1);
    assert_eq!(set.len(), 762, "the VITS checkpoint's full inventory");
    assert!(set.tensor("text_encoder.embed_tokens.weight").is_ok());
}

#[test]
fn every_tensor_the_index_names_is_reachable_and_no_others_exist() {
    let Some(dir) = asr() else {
        eprintln!("SKIP: no ASR checkpoint");
        return;
    };
    let set = StSet::open(&dir).expect("open");

    // The index is parsed independently here, so this compares the reader
    // against the file rather than against itself.
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).expect("index");
    let index: serde_json::Value = serde_json::from_str(&text).expect("parse index");
    let map = index["weight_map"].as_object().expect("weight_map");

    assert_eq!(set.len(), map.len());
    for name in map.keys() {
        assert!(set.info(name).is_some(), "{name} is named but unreachable");
    }
    for (name, _) in set.tensors() {
        assert!(map.contains_key(name), "{name} is reachable but unindexed");
    }
}
