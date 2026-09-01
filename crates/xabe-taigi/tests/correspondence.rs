//! The IPA table against what goruut actually wrote, over the training corpus.
//!
//! `tests/spelling.rs` checks the table's structure by hand. This checks it
//! against 28,489 syllable tokens nobody wrote down, captured by
//! `tools/oracle/capture_tailo_ipa.py` from SuiSiann's own Han/Tâi-lô pairs
//! phonemised with goruut - the corpus and the phonemiser this checkpoint was
//! trained with.
//!
//! # Why this is a rate and not an equality
//!
//! goruut goes from Han and has to guess which reading a character takes; the
//! corpus's romanisation is what the speaker actually said. They disagree on
//! about a quarter of tokens - 我 as `ŋɔ` against `ɡua`, 人 as `dzin` against
//! `laŋ` - and on every one of those this crate is right and goruut is wrong.
//! An exact-equality test would therefore be a test that this crate reproduces
//! goruut's *mistakes*, which is not the property wanted.
//!
//! So the corpus half is a floor on agreement, and the sharp half is
//! structural: every initial and every tone letter this crate can produce must
//! be one goruut writes. A table that invents a phoneme is wrong however well
//! it scores.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use xabe_taigi::tailo_to_ipa;

/// The initials, longest first. Written out again rather than imported: this
/// file is checking the table, so sharing its data would make the check
/// circular.
const INITIALS: [&str; 17] = [
    "tsʰ", "ts", "pʰ", "tʰ", "kʰ", "dz", "p", "b", "m", "t", "n", "l", "s", "k", "\u{0261}", "ŋ",
    "h",
];

#[derive(serde::Deserialize)]
struct Capture {
    aligned: usize,
    tokens: usize,
    correspondence: BTreeMap<String, Vec<(String, usize)>>,
    inventory: Inventory,
}

#[derive(serde::Deserialize)]
struct Inventory {
    initials: BTreeSet<String>,
    tones: BTreeSet<String>,
    bodies: BTreeSet<String>,
}

fn capture() -> Option<Capture> {
    let dir = match std::env::var("XABE_TAIGI_GOLDEN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".golden/coqui-tailo"),
    };
    let path = dir.join("correspondence.json");
    let text = std::fs::read_to_string(&path).ok()?;
    Some(serde_json::from_str(&text).expect("correspondence.json is malformed"))
}

fn skip() {
    eprintln!("SKIP: need .golden/coqui-tailo; see docs/ORACLE.md");
}

/// Splits an IPA syllable into initial, rime and tone.
fn parts(syllable: &str) -> (String, String, String) {
    let tone: String = syllable
        .chars()
        .filter(|c| ('\u{02E5}'..='\u{02E9}').contains(c))
        .collect();
    let body: String = syllable
        .chars()
        .filter(|c| !('\u{02E5}'..='\u{02E9}').contains(c))
        .collect();
    for i in INITIALS {
        if let Some(rest) = body.strip_prefix(i)
            && !rest.is_empty()
        {
            return (i.to_string(), rest.to_string(), tone);
        }
    }
    (String::new(), body, tone)
}

#[test]
fn every_initial_and_tone_is_one_goruut_writes() {
    let Some(c) = capture() else {
        skip();
        return;
    };
    // The sharp check. An initial this crate emits that goruut never wrote is
    // a phoneme the model has never seen, and no agreement rate would show it -
    // it would just be quietly mispronounced.
    for tailo in c.correspondence.keys() {
        let got = tailo_to_ipa(tailo);
        if got.syllables == 0 {
            continue;
        }
        let (initial, _, tone) = parts(&got.text);
        assert!(
            initial.is_empty() || c.inventory.initials.contains(&initial),
            "{tailo} -> {} has initial {initial:?}, which goruut never writes",
            got.text,
        );
        assert!(
            c.inventory.tones.contains(&tone),
            "{tailo} -> {} has tone {tone:?}, which goruut never writes",
            got.text,
        );
    }
}

#[test]
fn nearly_every_syllable_body_is_one_goruut_writes() {
    let Some(c) = capture() else {
        skip();
        return;
    };
    let mut total = 0;
    let mut attested = 0;
    let mut missing = Vec::new();
    for tailo in c.correspondence.keys() {
        let got = tailo_to_ipa(tailo);
        if got.syllables == 0 {
            continue;
        }
        let (initial, rime, _) = parts(&got.text);
        let body = format!("{initial}{rime}");
        total += 1;
        if c.inventory.bodies.contains(&body) {
            attested += 1;
        } else {
            missing.push(format!("{tailo} -> {body}"));
        }
    }
    let rate = 100.0 * attested as f64 / total as f64;
    eprintln!("syllable bodies attested in goruut's dictionary: {attested}/{total} ({rate:.1}%)");
    // Not 100%, and it cannot be: the dictionary has 4,608 entries and the
    // language has more syllables than that has words. The residue is ordinary
    // Taiwanese - `ɡun`, `tʰue`, `dzik` - plus two loanwords in the corpus.
    // Measured at 97.9%. A drop here means a rime the table is inventing.
    assert!(
        rate >= 95.0,
        "only {rate:.1}% of bodies are attested: {missing:?}"
    );
}

#[test]
fn the_spelling_agrees_wherever_the_reading_does() {
    let Some(c) = capture() else {
        skip();
        return;
    };
    let (mut exact, mut anywhere, mut parsed) = (0usize, 0usize, 0usize);
    let (mut tokens, mut agreed) = (0usize, 0usize);

    for (tailo, readings) in &c.correspondence {
        let got = tailo_to_ipa(tailo);
        if got.syllables == 0 {
            continue;
        }
        parsed += 1;
        if readings.first().is_some_and(|(r, _)| *r == got.text) {
            exact += 1;
        }
        if readings.iter().any(|(r, _)| *r == got.text) {
            anywhere += 1;
        }
        for (reading, n) in readings {
            tokens += n;
            if *reading == got.text {
                agreed += n;
            }
        }
    }

    let n = c.correspondence.len() as f64;
    let pct = |v: usize| 100.0 * v as f64 / n;
    eprintln!(
        "{} sentences aligned, {} syllable tokens, {} distinct syllables, {parsed} parsed",
        c.aligned,
        c.tokens,
        c.correspondence.len(),
    );
    eprintln!(
        "matches goruut's commonest reading: {exact} ({:.1}%); attested among its readings: {anywhere} ({:.1}%)",
        pct(exact),
        pct(anywhere),
    );
    eprintln!(
        "token-weighted agreement: {:.1}%",
        100.0 * agreed as f64 / tokens as f64,
    );

    // Floors, not targets. The gap to 100% is goruut choosing a different
    // *reading* of a Han character - which this crate never has to do, because
    // the romanisation already did. Measured at 71.5 / 80.2 / 71.3 when this
    // was written; a real regression in the table moves these by tens of
    // points, not by fractions.
    assert!(
        pct(exact) >= 70.0,
        "exact agreement fell to {:.1}%",
        pct(exact)
    );
    assert!(
        pct(anywhere) >= 78.0,
        "attested agreement fell to {:.1}%",
        pct(anywhere)
    );
    assert!(
        parsed as f64 / n >= 0.99,
        "only {parsed} of {n} syllables parsed"
    );
}
