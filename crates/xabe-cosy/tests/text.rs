//! The Qwen2 tokenizer against the ids CosyVoice3's own frontend produced.
//!
//! Two strings, and they are the two that matter: the instruct string, which
//! is where `<|endofprompt|>` comes from and therefore the one special token
//! the speech LLM cannot run without, and the Han utterance, which is what a
//! Taigi reply actually looks like. Both come from the same capture the rest
//! of this crate is tested against, so a mismatch here is a mismatch in the
//! *engine's own input*, not in a corpus assembled separately.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The capture writes ids through `.float()`, so they arrive as float32.
fn npy_ids(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not a .npy", p.display());
    let (hlen, at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => panic!("{}: .npy version {v}", p.display()),
    };
    let head = std::str::from_utf8(&b[at..at + hlen]).expect("header is ascii");
    assert!(head.contains("'<f4'"), "{}: not float32", p.display());
    b[at + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c).round() as u32)
        .collect()
}

#[test]
fn the_capture_s_text_and_instruct_tokenize_identically() {
    let dir = root().join(".golden/cosyvoice");
    let vocab = root().join("models/tts/cosyvoice3-0.5b/CosyVoice-BlankEN");
    if !dir.join("text.npy").is_file() || !vocab.join("added_tokens.json").is_file() {
        println!(
            "SKIP: needs .golden/cosyvoice (tools/oracle/capture_cosyvoice.py) and \
             CosyVoice-BlankEN/added_tokens.json (tools/dump_cosyvoice_tokens.py)"
        );
        return;
    }

    let t = xabe_cosy::Tokenizer::from_dir(&vocab).expect("open the tokenizer");
    // 151,643 learned plus 281 added. The count is asserted because a
    // half-written `added_tokens.json` would otherwise shift every special id
    // and show up much later as a model that speaks nonsense.
    assert_eq!(t.len(), 151_924, "vocabulary size");
    assert_eq!(t.special("<|endofprompt|>"), Some(151_646));

    // The manifest records both strings verbatim; they are repeated here so
    // the test states its own input rather than reading it out of a capture
    // that could have been taken on a different sentence.
    let instruct = "You are a helpful assistant. 請用閩南話表達。<|endofprompt|>";
    let text = "台北今仔日好天，溫度差不多二十五度。";

    let got = t.encode(instruct);
    let want = npy_ids(&dir.join("prompt_text.npy"));
    assert_eq!(got, want, "the instruct string");
    assert_eq!(
        got.last(),
        Some(&151_646),
        "the instruct has to end on <|endofprompt|>"
    );

    let got = t.encode(text);
    let want = npy_ids(&dir.join("text.npy"));
    assert_eq!(got, want, "the utterance");

    // Round-tripping is not part of the model's path, but a decode that does
    // not return the input means the byte alphabet is wrong in a way the
    // encode direction can hide.
    assert_eq!(t.decode(&got, true), text);
}
