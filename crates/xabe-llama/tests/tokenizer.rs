//! The tokenizer against 🤗 `LlamaTokenizer`, id for id.
//!
//! The corpus in `tools/oracle/capture_llama_tokenizer.py` is chosen for the
//! ways a SentencePiece BPE goes wrong: the dummy prefix and what it does to
//! leading whitespace, Han that the extended vocabulary was trained for, POJ
//! with combining diacritics, characters that fall through to byte fallback,
//! and the special tokens that are matched before the text is.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use xabe_llama::Tokenizer;

/// One captured case.
#[derive(Debug, Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    pieces: Vec<String>,
    decoded: String,
    decoded_skip: String,
}

/// The whole capture.
#[derive(Debug, Deserialize)]
struct Capture {
    tokenizer_size: usize,
    added: BTreeMap<String, u32>,
    bos: u32,
    eos: u32,
    unk: u32,
    cases: Vec<Case>,
}

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/translator/taigi-llama2-13b");
    p.join("tokenizer.model").is_file().then_some(p)
}

/// The capture, or `None` if it has not been made.
fn capture() -> Option<Capture> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/translator/tokenizer.json");
    let text = std::fs::read_to_string(p).ok()?;
    Some(serde_json::from_str(&text).expect("parse the tokenizer capture"))
}

#[test]
fn encodes_every_case_exactly() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_llama_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    for c in &cap.cases {
        let got = tok.encode(&c.text);
        let spell = |ids: &[u32]| -> Vec<String> {
            ids.iter()
                .map(|&i| tok.piece(i).map_or("?".into(), |p| p.text.clone()))
                .collect()
        };
        assert_eq!(
            got,
            c.ids,
            "{:?}\n  ours:   {:?}\n  theirs: {:?}",
            c.text,
            spell(&got),
            c.pieces,
        );
    }
}

#[test]
fn decodes_every_case_exactly() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_llama_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    for c in &cap.cases {
        assert_eq!(tok.decode(&c.ids, false), c.decoded, "{:?}", c.text);
        assert_eq!(tok.decode(&c.ids, true), c.decoded_skip, "{:?}", c.text);
    }
}

#[test]
fn the_vocabulary_and_its_special_tokens_are_the_reference_ones() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_llama_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    assert_eq!(tok.len(), cap.tokenizer_size);
    assert_eq!(
        (tok.bos(), tok.eos(), tok.unk()),
        (cap.bos, cap.eos, cap.unk)
    );
    for (name, &id) in &cap.added {
        assert_eq!(tok.special(name), Some(id), "{name}");
        assert!(tok.is_special(id), "{name} is not marked special");
    }
}

#[test]
fn pad_is_special_by_declaration_rather_than_by_type() {
    // `<pad>` is a NORMAL piece in the SentencePiece model - nothing in
    // `tokenizer.model` says it is anything else - and the checkpoint promotes
    // it in `special_tokens_map.json`. That is the same trap `<|endoftext|>`
    // sets in the ASR's tokenizer, running in the other direction: there a
    // special token hides in the ordinary vocabulary, here an ordinary piece
    // is declared special elsewhere. Reading only one of the two files gets it
    // wrong either way.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/translator/taigi-llama2-13b is missing");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    let pad = tok.special("<pad>").expect("<pad>");
    assert_eq!(pad, 32_000);
    assert_eq!(
        tok.piece(pad).expect("piece").kind,
        xabe_llama::Kind::Normal,
        "the model file calls it normal",
    );
    assert!(tok.is_special(pad), "and the checkpoint calls it special");
}

#[test]
fn byte_fallback_makes_the_tokenizer_total() {
    // Every input has an encoding, because every byte has a piece. The check
    // that matters is the round trip, since byte fallback splits a character
    // across several ids and joining them wrongly gives U+FFFD.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/translator/taigi-llama2-13b is missing");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    for text in ["🎧", "ＡＢＣ", "\u{0}\u{1}", "𐌰𐌱", "chiaⁿ-goe̍h"] {
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "{text:?} encoded to nothing");
        assert_eq!(tok.decode(&ids, false), text, "{text:?}");
    }
}
