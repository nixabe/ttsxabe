//! Tâi-lô to the IPA the Coqui SuiSiann checkpoint was trained on.
//!
//! # Where this table comes from
//!
//! Not from a phonology textbook. The checkpoint was trained on whatever
//! `pygoruut` emitted for `MinnanHokkien2`, so the target is *that* program's
//! conventions, and the table below was derived from its own data and checked
//! against it: every initial and every tone letter this module can produce is
//! attested in goruut's dictionary, and 97.9% of the syllable bodies are - the
//! rest being ordinary Taiwanese syllables that a 4,608-entry dictionary simply
//! has no word for. `docs/ORACLE.md` has how that was measured.
//!
//! # Why romanisation and not Han
//!
//! goruut's own path is Han to IPA, and this is not a port of it. It cannot be:
//! a Han character has several readings and choosing between them is a learned
//! model, not a table - `的` comes out `tik˨` there where the corpus says `ê`.
//! Romanisation has already made that choice, upstream, in the translator that
//! knows the sentence. So this converts the choice rather than making it, and
//! is a table for exactly that reason.
//!
//! It also means this disagrees with goruut on purpose. Over the 2,152
//! alignable sentences of SuiSiann, goruut picks a different *reading* from the
//! corpus's own romanisation for about a quarter of syllable tokens - 我 as
//! `ŋɔ` rather than `ɡua`, 人 as `dzin` rather than `laŋ`. Where the reading
//! agrees, the spelling agrees; where it does not, this module follows the
//! romanisation, which is what the audio actually says. Measured: 71.3% of
//! syllable tokens come out identical to goruut's reading, and the gap is
//! readings rather than spelling.
//!
//! # Three things are dropped, and all three are measured
//!
//! The model's symbol table is `IPAPhonemes`, and it does not contain
//! everything goruut writes. Feeding it what it never saw would be worse than
//! matching the reference's own losses, so this module reproduces them rather
//! than improving on them:
//!
//! - **Aspiration.** `ʰ` (U+02B0) is not in the vocabulary. 5,633 of them are
//!   discarded across the corpus, so `pʰ` and `p` reach the model identically.
//!   It is emitted anyway, because that is what the reference emits and what a
//!   retrained vocabulary would want.
//! - **Nasal vowels.** goruut writes them precomposed - `ã`, `ĩ` - and the
//!   vocabulary holds only the combining tilde, so every one is discarded.
//!   49% of the corpus's sentences lose at least one. Again emitted as goruut
//!   emits it: writing the decomposed form would survive tokenisation and hand
//!   the model a symbol it has essentially never seen.
//! - **Punctuation and spaces**, which this module drops outright rather than
//!   passing on. That *is* a deliberate divergence, and it is the only one.
//!   goruut passes marks through and the tokenizer then discards the full-width
//!   ones; what survives is 0.337% of the training characters, with `,` seen
//!   four times in the whole corpus and `.` nine. Those embeddings are noise.
//!   Clause boundaries reach the listener as separate synthesis calls anyway,
//!   so nothing is lost by not sending them.
//!
//! # Already-phonemised input passes through
//!
//! Text containing an IPA tone letter is returned unchanged. The two forms are
//! trivially distinguishable - U+02E5..U+02E9 never occur in romanisation - and
//! the alternative is worse than useless: a run of `li` inside `li˥˧` has no
//! digit, would take the default tone, and would come back `li˥˥˥˧`.

use crate::poj::poj_to_tailo;

/// What a conversion produced, and what it could not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phonemes {
    /// The IPA, with nothing between syllables. goruut is called with an empty
    /// separator, so the model has never seen a space.
    pub text: String,
    /// Syllables converted.
    pub syllables: usize,
    /// Runs that looked like a syllable and could not be parsed as one. These
    /// are dropped rather than passed through: a Latin letter *is* in this
    /// model's vocabulary, so passing one on would put a phoneme in the
    /// sequence rather than a gap.
    pub dropped: usize,
}

/// Chao tone letters, indexed by Tâi-lô tone number.
///
/// Seven, not nine: 6 has merged with 2 in every variety this checkpoint saw,
/// and 9 is the high-level tone of a few loans. Both are mapped to their
/// nearest so that a marked syllable is never silently lost, and neither
/// appears in goruut's data.
const TONES: [&str; 10] = [
    "",   // 0, unused
    "˥˥", // 1  high level
    "˥˧", // 2  high falling
    "˨˩", // 3  low falling
    "˨",  // 4  mid checked
    "˨˦", // 5  rising
    "˥˧", // 6  merged with 2
    "˧˧", // 7  mid level
    "˦",  // 8  high checked
    "˥˥", // 9  high level
];

/// Initials, longest first so `tsh` is not read as `ts` followed by `h`.
///
/// `j` is `dz` and not `ʑ`: goruut writes the affricate, and this checkpoint
/// was trained on a variety that has it. `g` is U+0261, not ASCII `g` - the
/// vocabulary contains only the former, and Coqui's own wrapper translates
/// goruut's output for exactly this reason.
const INITIALS: [(&str, &str); 17] = [
    ("tsh", "tsʰ"),
    ("ts", "ts"),
    ("ph", "pʰ"),
    ("th", "tʰ"),
    ("kh", "kʰ"),
    ("ng", "ŋ"),
    ("p", "p"),
    ("b", "b"),
    ("m", "m"),
    ("t", "t"),
    ("n", "n"),
    ("l", "l"),
    ("s", "s"),
    ("j", "dz"),
    ("k", "k"),
    ("g", "\u{0261}"),
    ("h", "h"),
];

/// Syllables that are a nasal on their own, with no vowel at all.
///
/// They are also what an initial can be followed by - `hng`, `tshng`, `thng`
/// are syllables - which is why the initial match below accepts one of these
/// as a rime and not only a vowel. Missing that reads `thng` as an unanalysable
/// run and drops it.
///
/// A bare `n` is deliberately not here. Tâi-lô has no syllabic `n`; POJ's `ń`
/// is the mark sitting on the first letter of `ng`, and `poj_to_tailo` has
/// already put it back. Accepting one would invent a phoneme goruut never
/// writes, out of input that is malformed anyway.
const SYLLABIC: [(&str, &str); 4] = [("ngh", "ŋʔ"), ("mh", "mʔ"), ("ng", "ŋ"), ("m", "m")];

/// Final consonants. `-h` is a glottal stop; the other six are themselves.
const CODAS: [(&str, &str); 7] = [
    ("ng", "ŋ"),
    ("p", "p"),
    ("t", "t"),
    ("k", "k"),
    ("h", "ʔ"),
    ("m", "m"),
    ("n", "n"),
];

/// The nasalised vowels, precomposed, as goruut writes them.
fn nasalise(v: char) -> char {
    match v {
        'a' => 'ã',
        'e' => 'ẽ',
        'i' => 'ĩ',
        'o' => 'õ',
        'u' => 'ũ',
        other => other,
    }
}

/// The Chao letters, which mark text as already phonemised.
fn is_tone_letter(c: char) -> bool {
    ('\u{02E5}'..='\u{02E9}').contains(&c)
}

/// Whether a character can be part of a romanised syllable here.
///
/// Only ASCII: [`poj_to_tailo`] has already turned every diacritic into a
/// trailing digit, so anything left with a mark on it is not romanisation this
/// crate produced and is not a syllable.
fn syllabic(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

/// Converts Tâi-lô with numeric tones into IPA.
///
/// Input that already carries IPA tone letters is returned unchanged - see the
/// module header.
pub fn tailo_to_ipa(text: &str) -> Phonemes {
    if text.chars().any(is_tone_letter) {
        return Phonemes {
            text: text.to_string(),
            syllables: 0,
            dropped: 0,
        };
    }

    let mut out = String::with_capacity(text.len() * 2);
    let mut syllables = 0;
    let mut dropped = 0;
    let mut buf = String::new();

    let mut flush = |buf: &mut String, out: &mut String| {
        if buf.is_empty() {
            return;
        }
        match syllable(&std::mem::take(buf)) {
            Some(ipa) => {
                out.push_str(&ipa);
                syllables += 1;
            }
            None => dropped += 1,
        }
    };

    for c in text.chars() {
        if syllabic(c) {
            buf.push(c);
        } else {
            // Everything else - space, hyphen, punctuation - is a separator and
            // not a sound. goruut is called with an empty separator and its
            // marks are discarded by the tokenizer, so dropping them here is
            // what the model was trained on.
            flush(&mut buf, &mut out);
        }
    }
    flush(&mut buf, &mut out);

    if dropped > 0 {
        tracing::debug!(dropped, "runs that are not Tâi-lô syllables were discarded");
    }
    Phonemes {
        text: out,
        syllables,
        dropped,
    }
}

/// POJ straight to IPA, which is the conversion the pipeline needs.
///
/// The pass-through check happens *before* the POJ step, not after. Going the
/// other way round is wrong in a way that is easy to miss: `poj_to_tailo` sees
/// the `li` inside `li˥˧`, finds no tone mark on it, and appends the default -
/// so the string reaching [`tailo_to_ipa`] is `li1˥˧`, which it then passes
/// through unchanged because it does contain a tone letter. The result is the
/// input with stray digits in it, and nothing anywhere reports a problem.
pub fn poj_to_ipa(text: &str) -> Phonemes {
    if text.chars().any(is_tone_letter) {
        return Phonemes {
            text: text.to_string(),
            syllables: 0,
            dropped: 0,
        };
    }
    tailo_to_ipa(&poj_to_tailo(text))
}

/// One syllable: `tshiat4` to `tsʰiat˨`.
///
/// `None` for a run that is not a syllable at all - a bare number, a Latin word
/// that survived translation, `russia`.
fn syllable(raw: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    let (body, tone) = split_tone(&lower)?;
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    Some(format!("{}{}", segments(body)?, TONES[usize::from(tone)]))
}

/// Splits a trailing tone digit off, supplying the unmarked tone if there is
/// none.
///
/// An unmarked syllable is tone 1, except that a stop-final one is tone 4 -
/// the same rule [`poj_to_tailo`] applies, repeated here because this function
/// is also reachable with hand-written Tâi-lô that never went through it.
fn split_tone(s: &str) -> Option<(&str, u8)> {
    match s.as_bytes().last() {
        Some(d) if d.is_ascii_digit() => {
            let tone = d - b'0';
            // 0 is not a tone, and a digit that large is a number rather than a
            // marked syllable.
            if tone == 0 || usize::from(tone) >= TONES.len() {
                return None;
            }
            Some((&s[..s.len() - 1], tone))
        }
        Some(_) => {
            let tone = match s.as_bytes()[s.len() - 1] {
                b'p' | b't' | b'k' | b'h' => 4,
                _ => 1,
            };
            Some((s, tone))
        }
        None => None,
    }
}

/// Initial, rime and coda, without the tone.
fn segments(body: &str) -> Option<String> {
    if let Some(whole) = syllabic_nasal(body) {
        return Some(whole.to_string());
    }

    let (initial, rest) = split_initial(body);
    if let Some(whole) = syllabic_nasal(rest) {
        return Some(format!("{initial}{whole}"));
    }

    // The nasal marker comes off before the coda, or `ainn` reads as `ain`
    // plus a stray `n` and the nasalisation is lost while the shape stays
    // plausible. That was a real bug, caught by the syllable inventory.
    let (rest, nasal, mut coda) = match rest.strip_suffix("nnh") {
        Some(r) => (r, true, "ʔ"),
        None => match rest.strip_suffix("nn") {
            Some(r) => (r, true, ""),
            None => (rest, false, ""),
        },
    };

    let mut vowels = rest;
    if !nasal {
        for (from, to) in CODAS {
            if let Some(r) = vowels.strip_suffix(from)
                && !r.is_empty()
            {
                coda = to;
                vowels = r;
                break;
            }
        }
    }
    if vowels.is_empty() {
        return None;
    }

    // `oo` is the open o and `o` the close one; the digraph has to go first or
    // it reads as two syllables' worth of vowel.
    let mut nucleus = String::with_capacity(vowels.len() * 2);
    let mut chars = vowels.chars().peekable();
    while let Some(c) = chars.next() {
        let v = if c == 'o' && chars.peek() == Some(&'o') {
            chars.next();
            'ɔ'
        } else {
            c
        };
        if !matches!(v, 'a' | 'e' | 'i' | 'o' | 'u' | 'ɔ') {
            return None;
        }
        nucleus.push(if nasal { nasalise(v) } else { v });
    }

    Some(format!("{initial}{nucleus}{coda}"))
}

/// A whole syllable that is just a nasal, or `None`.
fn syllabic_nasal(body: &str) -> Option<&'static str> {
    SYLLABIC
        .iter()
        .find(|(from, _)| *from == body)
        .map(|(_, to)| *to)
}

/// Peels the initial off, if there is one.
///
/// An initial only counts when something follows it that can be a rime: a
/// vowel, or one of the syllabic nasals. `m` alone is a syllable, not an
/// initial with nothing after it.
fn split_initial(body: &str) -> (&'static str, &str) {
    for (from, to) in INITIALS {
        if let Some(rest) = body.strip_prefix(from)
            && !rest.is_empty()
            && (rest.starts_with(['a', 'e', 'i', 'o', 'u']) || syllabic_nasal(rest).is_some())
        {
            return (to, rest);
        }
    }
    ("", body)
}
