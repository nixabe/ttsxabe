//! Differential test: our tokeniser against the reference's, case for case.
//!
//! Milestone 4 asks for an *exact* match, so the corpus is not a sentence but a
//! set of inputs chosen for where a reimplementation plausibly diverges -
//! casing, normalisation form, out-of-vocabulary characters, the literal
//! spelling of the unknown token, and the blank at the edges. It is captured by
//! `tools/oracle/tokenize_cases.py`; see `docs/ORACLE.md`.
//!
//! Skips when either the checkpoint or the capture is absent.

use std::path::PathBuf;
use xabe_vits::Tokenizer;

/// One captured case: the input, what the reference's normaliser made of it,
/// and the ids it produced.
#[derive(serde::Deserialize)]
struct Case {
    text: String,
    normalized: String,
    ids: Vec<i64>,
}

#[derive(serde::Deserialize)]
struct Cases {
    add_blank: bool,
    normalize: bool,
    vocab_size: usize,
    unk_token_id: i64,
    cases: Vec<Case>,
}

fn find_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("XABE_TTS_MODEL") {
        return PathBuf::from(p).parent().map(Into::into);
    }
    let home = std::env::var("HOME").ok()?;
    let root = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--facebook--mms-tts-nan/snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    snap.join("vocab.json").is_file().then_some(snap)
}

fn load() -> Option<(Tokenizer, Cases)> {
    let snap = find_snapshot()?;
    let path = match std::env::var("XABE_GOLDEN_TOKENIZER") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.golden/tokenizer/cases.json")
        }
    };
    let text = std::fs::read_to_string(path).ok()?;
    Some((
        Tokenizer::load(&snap).expect("load tokenizer"),
        serde_json::from_str(&text).expect("parse cases"),
    ))
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/tokenizer; see docs/ORACLE.md");
}

#[test]
fn every_captured_case_tokenizes_identically() {
    let Some((tok, c)) = load() else {
        skip();
        return;
    };

    let mut failures = Vec::new();
    for case in &c.cases {
        let got = tok.encode(&case.text);
        if got != case.ids {
            // Print the code points, not the characters: half of these cases
            // differ only in normalisation form, and `lí` and `lí` are
            // indistinguishable in a terminal.
            let cps: Vec<String> = case
                .text
                .chars()
                .map(|ch| format!("U+{:04X}", ch as u32))
                .collect();
            failures.push(format!(
                "  [{}] normalised by the reference to {:?}\n    oracle {:?}\n    ours   {:?}",
                cps.join(" "),
                case.normalized,
                case.ids,
                got,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} cases differ:\n{}",
        failures.len(),
        c.cases.len(),
        failures.join("\n"),
    );
}

#[test]
fn the_capture_agrees_with_the_tokenizer_config() {
    let Some((tok, c)) = load() else {
        skip();
        return;
    };
    assert!(
        c.add_blank && c.normalize,
        "the cases assume this model's config"
    );
    assert_eq!(tok.vocab_size(), c.vocab_size);
    // One past the end of the vocabulary, and of the embedding table. Nothing
    // may emit it.
    assert_eq!(c.unk_token_id, c.vocab_size as i64);
    assert!(
        c.cases
            .iter()
            .all(|case| case.ids.iter().all(|&i| i < c.unk_token_id)),
        "a captured case reached the unknown id, which the embedding lacks",
    );
}

#[test]
fn the_blank_frames_every_symbol() {
    let Some((tok, c)) = load() else {
        skip();
        return;
    };

    // The interspersion is the one structural claim about the output, and it is
    // easy to get right at the joins and wrong at the ends.
    for case in &c.cases {
        let ids = tok.encode(&case.text);
        if ids.is_empty() {
            continue;
        }
        assert_eq!(ids.len() % 2, 1, "{:?} produced an even length", case.text);
        assert!(
            ids.iter().step_by(2).all(|&i| i == 0),
            "{:?} does not have a blank at every even position",
            case.text,
        );
        assert!(
            ids.iter().skip(1).step_by(2).all(|&i| i != 0),
            "{:?} has a blank where a symbol should be",
            case.text,
        );
    }
}

#[test]
fn an_empty_input_produces_no_symbols_rather_than_a_lone_blank() {
    let Some((tok, _)) = load() else {
        skip();
        return;
    };

    // The natural reading of "intersperse a blank" gives `[0]` for empty input,
    // which would make the model synthesise a frame of nothing. The reference
    // gives an empty sequence, because slice-assigning an empty list leaves an
    // empty list.
    for text in ["", " ", "   ", "你好", "..."] {
        assert!(
            tok.encode(text).is_empty(),
            "{text:?} should tokenise to nothing",
        );
    }
}

#[test]
fn the_normalisation_form_of_the_input_changes_the_words() {
    let Some((tok, _)) = load() else {
        skip();
        return;
    };

    // The vocabulary has precomposed í but not combining acute, so NFD input
    // loses its tone marks silently. Tones are lexical in this language, so
    // this is the difference between two words - not a rounding error. Pinning
    // it here means a future "helpfully normalise the input" change has to
    // argue with a test.
    let nfc = tok.encode("l\u{ed}");
    let nfd = tok.encode("li\u{301}");
    assert_ne!(nfc, nfd, "NFC and NFD must not tokenise alike");
    assert_eq!(
        nfc.len(),
        nfd.len(),
        "the loss is in the value, not the length"
    );

    // U+030D and U+0358 have no precomposed form and are in the vocabulary, so
    // they must survive - the required input form is NFC, which leaves exactly
    // these two combining.
    assert_eq!(tok.encode("ji\u{30d}t").len(), 9);
    assert_eq!(tok.encode("o\u{358}").len(), 5);
}

#[test]
fn the_waveform_capture_tokenizes_to_the_ids_it_recorded() {
    let Some((tok, _)) = load() else {
        skip();
        return;
    };
    let Some(g) = xabe_golden::Golden::open_default() else {
        eprintln!("SKIP: no waveform capture");
        return;
    };

    // The two captures are taken by different scripts, so this is the only
    // thing that ties them together: the text in the waveform capture's
    // manifest, tokenised here, must be the `input_ids` that capture recorded.
    // Without it, milestone 5 onwards could be diffing against a waveform
    // synthesised from a different sentence than the one being fed in.
    let ids = tok.encode(&g.manifest().text);
    let recorded = g.i64s("input_ids").expect("read input_ids");
    assert_eq!(
        ids, recorded,
        "the oracle's waveform was synthesised from different ids than we produce",
    );
    assert_eq!(g.shape("input_ids").unwrap(), [1, ids.len()]);
}
