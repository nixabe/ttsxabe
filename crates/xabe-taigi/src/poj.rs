//! Pe̍h-ōe-jī to Tâi-lô: the same language in two orthographies.
//!
//! The difference is mechanical: `chh`/`ch` are `tsh`/`ts`, `oa`/`oe` are
//! `ua`/`ue`, `eng`/`ek` are `ing`/`ik`, `o͘` is `oo`, `ⁿ` is `nn`, and the tone
//! mark becomes a trailing digit.
//!
//! # Passing POJ through unconverted is not a degraded reading
//!
//! It is silence. `á` is not in any of the three alphabets downstream, so it
//! vanishes, and the tone vanishes with it - the tones *are* the words in this
//! language, so what is left is a different sentence, not an accented one.
//!
//! # This module moved here from `xabe-taco`
//!
//! Unchanged, and the comments below are the ones written when it was found to
//! be needed the first time. What changed is who needs it: two checkpoints now
//! do, and the Coqui one reaches it through [`crate::tailo_to_ipa`].
//!
//! Punctuation is **not** folded here, which is the one behavioural difference
//! from the version that lived in `xabe-taco`. Tacotron2 needs `。` turned into
//! `.` because its gate learned end-of-utterance from ASCII punctuation and a
//! line without one runs the decoder to its step limit; the Coqui VITS saw no
//! punctuation at all during training. Those are opposite requirements, so
//! neither belongs in a shared spelling table - `xabe-taco` folds on the way
//! out of here, and `xabe-taigi` leaves the marks alone.

/// A combining mark's tone number, or `None` if it is not a tone.
fn combining_tone(c: char) -> Option<u8> {
    match c {
        '\u{0301}' => Some(2), // acute
        '\u{0300}' => Some(3), // grave
        '\u{0302}' => Some(5), // circumflex
        '\u{030C}' => Some(6), // caron
        '\u{0304}' => Some(7), // macron
        '\u{030D}' => Some(8), // vertical line above
        '\u{030B}' => Some(9), // double acute
        _ => None,
    }
}

/// Splits a precomposed vowel into its base letter and tone.
///
/// Only the forms POJ actually uses. Tone 8 has no precomposed character in
/// Unicode at all - it is always a base plus U+030D - which is why the
/// combining path exists as well as this one.
fn precomposed(c: char) -> Option<(char, u8)> {
    let t = match c {
        'á' | 'é' | 'í' | 'ó' | 'ú' | 'ń' => 2,
        'à' | 'è' | 'ì' | 'ò' | 'ù' | 'ǹ' => 3,
        'â' | 'ê' | 'î' | 'ô' | 'û' => 5,
        'ǎ' | 'ě' | 'ǐ' | 'ǒ' | 'ǔ' => 6,
        'ā' | 'ē' | 'ī' | 'ō' | 'ū' | 'ḿ' => 7,
        _ => return None,
    };
    let base = match c {
        'á' | 'à' | 'â' | 'ǎ' | 'ā' => 'a',
        'é' | 'è' | 'ê' | 'ě' | 'ē' => 'e',
        'í' | 'ì' | 'î' | 'ǐ' | 'ī' => 'i',
        'ó' | 'ò' | 'ô' | 'ǒ' | 'ō' => 'o',
        'ú' | 'ù' | 'û' | 'ǔ' | 'ū' => 'u',
        'ḿ' => 'm',
        'ń' | 'ǹ' => 'n',
        _ => return None,
    };
    Some((base, t))
}

/// Whether a character can be part of a romanised syllable.
///
/// Digits included, so that a syllable which already carries its tone as a
/// number is one run and can be recognised as needing nothing done to it.
fn syllabic(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || c == '\u{207F}'
        || c == '\u{1D3A}'
        || combining_tone(c).is_some()
        || c == '\u{0358}'
        || precomposed(c).is_some()
}

/// POJ's digraphs, in the order they must be tried.
///
/// `chh` before `ch`, or `chh` becomes `tsh` spelled `tsh`... and `ch`+`h`
/// spells `tsh` too, but only by accident of both rules firing. Longest first
/// is the rule, and it is the reason this is a list and not a map.
const SPELLING: &[(&str, &str)] = &[
    ("chh", "tsh"),
    ("ch", "ts"),
    ("eng", "ing"),
    ("ek", "ik"),
    ("oa", "ua"),
    ("oe", "ue"),
];

/// Converts one syllable from POJ to Tâi-lô with a numeric tone.
///
/// A syllable that already carries a digit is returned as it came: it is
/// already numeric-tone romanisation, and running the spelling rules over it a
/// second time would turn `tshiat4` into something that is not a word.
fn syllable(raw: &str) -> String {
    if raw.chars().any(|c| c.is_ascii_digit()) {
        return raw.to_string();
    }
    let mut base = String::with_capacity(raw.len());
    let mut tone: Option<u8> = None;

    for c in raw.chars() {
        if let Some(t) = combining_tone(c) {
            tone = tone.or(Some(t));
        } else if c == '\u{0358}' {
            // The dot that turns `o` into `oo`. A mark on the preceding vowel
            // rather than a letter of its own, so it is applied backwards.
            if base.ends_with('o') || base.ends_with('O') {
                base.push('o');
            }
        } else if c == '\u{207F}' || c == '\u{1D3A}' {
            base.push_str("nn");
        } else if let Some((b, t)) = precomposed(c) {
            tone = tone.or(Some(t));
            base.push(b);
        } else {
            base.push(c);
        }
    }

    let lower = base.to_lowercase();
    let mut out = String::with_capacity(lower.len() + 1);
    let bytes: Vec<char> = lower.chars().collect();
    let mut i = 0;
    'outer: while i < bytes.len() {
        for (from, to) in SPELLING {
            let n = from.chars().count();
            if i + n <= bytes.len() && bytes[i..i + n].iter().collect::<String>() == *from {
                out.push_str(to);
                i += n;
                continue 'outer;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }

    // An unmarked syllable is tone 1, except that a stop-final one is tone 4.
    // Tone 8 is the marked stop-final, which is why this only applies when
    // nothing was marked.
    let tone = tone.unwrap_or_else(|| match out.chars().last() {
        Some('p' | 't' | 'k' | 'h') => 4,
        _ => 1,
    });
    out.push(char::from(b'0' + tone));
    out
}

/// Rewrites POJ as the Tâi-lô-with-tone-digits this checkpoint reads.
///
/// The decision is taken per syllable rather than per line. A line-wide test
/// was the first version and it was wrong on mixed input: one numeric syllable
/// anywhere made the whole line pass through, and every diacritic in it was
/// then dropped by the tokeniser without a word - which is the silence this
/// function exists to prevent, arrived at by way of trying to prevent it.
///
/// Everything that is not part of a syllable passes through verbatim,
/// punctuation included. `xabe-taco` folds full-width marks to ASCII on the way
/// out of here because its gate needs them; the Coqui VITS wants them left
/// alone. See the module header.
pub fn poj_to_tailo(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 4);
    let mut buf = String::new();
    for c in text.chars() {
        if syllabic(c) {
            buf.push(c);
        } else {
            if !buf.is_empty() {
                out.push_str(&syllable(&std::mem::take(&mut buf)));
            }
            out.push(c);
        }
    }
    if !buf.is_empty() {
        out.push_str(&syllable(&buf));
    }
    out
}
