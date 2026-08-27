//! The tokenizer against 🤗 `WhisperTokenizer`, id for id.
//!
//! The corpus in `tools/oracle/capture_tokenizer.py` is chosen for the ways a
//! byte-level BPE goes wrong rather than for being representative text. Every
//! expectation here comes out of the capture; none is written down twice.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use xabe_whisper::Tokenizer;

/// One captured case.
#[derive(Debug, Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
    decoded_skip: String,
}

/// The whole capture.
#[derive(Debug, Deserialize)]
struct Capture {
    vocab_size: usize,
    specials: std::collections::BTreeMap<String, u32>,
    cases: Vec<Case>,
}

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    p.join("vocab.json").is_file().then_some(p)
}

/// The capture, or `None` if it has not been made.
fn capture() -> Option<Capture> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/asr/tokenizer.json");
    let text = std::fs::read_to_string(p).ok()?;
    Some(serde_json::from_str(&text).expect("parse the tokenizer capture"))
}

#[test]
fn encodes_every_case_exactly() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    for c in &cap.cases {
        let got = tok.encode(&c.text);
        assert_eq!(
            got,
            c.ids,
            "{:?}\n  ours: {:?}\n  theirs: {:?}",
            c.text,
            got.iter().map(|&i| tok.spelling(i)).collect::<Vec<_>>(),
            c.ids.iter().map(|&i| tok.spelling(i)).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn decodes_every_case_exactly() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    for c in &cap.cases {
        assert_eq!(tok.decode(&c.ids, false), c.decoded, "{:?}", c.text);
        assert_eq!(tok.decode(&c.ids, true), c.decoded_skip, "{:?}", c.text);
    }
}

#[test]
fn the_special_ids_are_the_ones_the_reference_uses() {
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    assert_eq!(tok.len(), cap.vocab_size);
    for (name, &want) in &cap.specials {
        let spelling = match name.as_str() {
            "timestamp_begin" => "<|0.00|>".to_string(),
            // Captured to record what the reference answers, not to be looked
            // up - see the test below.
            "nospeech_or_unk" => continue,
            other => format!("<|{other}|>"),
        };
        assert_eq!(tok.special(&spelling), Some(want), "{spelling}");
        assert!(tok.is_special(want), "{spelling} is not marked special");
    }
}

#[test]
fn the_no_speech_token_is_spelled_the_old_way_here() {
    // 50362 is `<|nocaptions|>` on this checkpoint, OpenAI's original name,
    // not the `<|nospeech|>` that later documentation uses. The id is the same
    // and the spelling is not, which is the worst of both: the reference
    // answers a lookup for `<|nospeech|>` with the *unknown* id, 50257, and
    // 50257 is also end-of-text. A decoder that reads `no_speech_prob` at the
    // wrong column, or stops on the wrong token, would look like a model that
    // simply gives up early.
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    assert_eq!(tok.special("<|nocaptions|>"), Some(50362));
    assert_eq!(tok.special("<|nospeech|>"), None, "not in this checkpoint");
    assert_eq!(
        cap.specials.get("nospeech_or_unk"),
        Some(&50257),
        "the reference answers with the unknown id rather than refusing",
    );
}

#[test]
fn the_arithmetic_whisper_cpp_uses_agrees_with_the_file() {
    // whisper.cpp has nowhere to store the 1,607 special tokens, so it derives
    // them: `num_languages = n_vocab - 51765 - 1`, and every id after
    // `<|startoftranscript|>` follows from that. This engine reads them
    // instead - but the arithmetic is checked once, here, against the file,
    // because an off-by-one in it is a transcript in the wrong language with
    // nothing at all to indicate why.
    let (Some(dir), Some(cap)) = (checkpoint(), capture()) else {
        println!("SKIP: run tools/oracle/capture_tokenizer.py first");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");

    let n_vocab = cap.vocab_size;
    let num_languages = n_vocab - 51_765 - 1;
    assert_eq!(num_languages, 99, "large-v2 carries 99 languages");

    let sot = tok.special("<|startoftranscript|>").expect("sot");
    // Languages run from sot + 1, in the reference's fixed order; English is
    // first and Chinese second.
    assert_eq!(tok.special("<|en|>"), Some(sot + 1));
    assert_eq!(tok.special("<|zh|>"), Some(sot + 2));
    // Then translate, transcribe, and the rest of the control tokens.
    assert_eq!(
        tok.special("<|translate|>"),
        Some(sot + 1 + num_languages as u32)
    );
    assert_eq!(
        tok.special("<|transcribe|>"),
        Some(sot + 2 + num_languages as u32)
    );
    // Timestamps begin one past `<|notimestamps|>` and step by 20 ms.
    let t0 = tok.special("<|0.00|>").expect("timestamp base");
    assert_eq!(t0, tok.special("<|notimestamps|>").expect("nots") + 1);
    assert_eq!(tok.special("<|30.00|>"), Some(t0 + 1500));
    assert_eq!(tok.len() as u32, t0 + 1501);
}

#[test]
fn timestamps_go_even_when_the_other_specials_stay() {
    // The reference's default, and a real asymmetry rather than an oversight:
    // `<|zh|>` is information about the transcript, `<|2.50|>` is punctuation
    // for a feature this engine does not run. That half is captured.
    //
    // The other half - `decode_with_timestamps` - is asserted here rather than
    // against a capture, because the reference gets it wrong. transformers
    // 5.15.1 computes `timestamp_begin = self.all_special_ids[-1] + 1`, and
    // `all_special_ids` on this checkpoint holds exactly one entry
    // (`<|endoftext|>`, 50257), so it renders `<|startoftranscript|>` as
    // `<|0.00|>` and every control token after it as a timestamp. Capturing
    // that would enshrine a bug; this engine uses the real boundary, the id of
    // `<|0.00|>`.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is missing");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    let ids = tok.encode("<|zh|><|0.00|>hi<|2.50|>");
    assert_eq!(tok.decode(&ids, false), "<|zh|>hi");
    assert_eq!(tok.decode(&ids, true), "hi");
    assert_eq!(
        tok.decode_with_timestamps(&ids, false),
        "<|zh|><|0.00|>hi<|2.50|>",
    );
}

#[test]
fn the_end_of_text_token_is_special_despite_living_in_the_vocabulary() {
    // It predates the multilingual tokens, so it is in `vocab.json` and not in
    // `added_tokens.json`. A loader that reads only the latter gets 1,607 of
    // 1,608 specials right and leaves the one the decoder stops on looking
    // like ordinary text.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is missing");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    let eot = tok.special("<|endoftext|>").expect("<|endoftext|>");
    assert_eq!(eot, 50257);
    assert!(tok.is_special(eot));
    let hello = tok.encode("hello");
    assert_eq!(
        tok.encode("hello<|endoftext|>"),
        [hello.clone(), vec![eot]].concat()
    );
    assert_eq!(tok.decode(&[hello, vec![eot]].concat(), true), "hello");
}

#[test]
fn a_han_character_survives_being_split_across_tokens() {
    // The failure this rules out: decoding token by token and joining the
    // strings. A single Han character is three bytes and BPE splits it
    // routinely, so per-token decoding yields U+FFFD where the reference
    // yields text - and only on some characters, which is worse.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/asr/breeze-asr-26 is missing");
        return;
    };
    let tok = Tokenizer::from_dir(&dir).expect("load the tokenizer");
    let text = "毋過真濟人食飽矣";
    let ids = tok.encode(text);
    assert_eq!(tok.decode(&ids, false), text);
    // And the split really happens - otherwise this passes for the wrong
    // reason on a checkpoint where every Han character is one token.
    assert!(
        ids.iter()
            .any(|&i| tok.decode(&[i], false).contains('\u{fffd}')),
        "no token here is a partial character, so this proves nothing",
    );
}

#[test]
fn the_pre_tokenizer_reproduces_the_lookahead() {
    use xabe_whisper::pre_tokenize;
    // A run of k spaces before a word splits as k-1 spaces plus a word that
    // owns the last one. This is the piece of the reference's pattern the
    // `regex` crate has no syntax for, so it is the piece most likely to be
    // quietly wrong.
    assert_eq!(pre_tokenize("a  b"), vec!["a", " ", " b"]);
    assert_eq!(pre_tokenize("a   b"), vec!["a", "  ", " b"]);
    assert_eq!(pre_tokenize("a b"), vec!["a", " b"]);
    // Trailing whitespace has nothing to give its last character to.
    assert_eq!(pre_tokenize("a  "), vec!["a", "  "]);
    assert_eq!(pre_tokenize("  "), vec!["  "]);
}
