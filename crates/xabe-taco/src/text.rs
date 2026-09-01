//! Getting text into the 71 symbols this model was trained on.
//!
//! The reference throughout is yfliao/taiwanese_tonal_tlpa_tacotron2
//! (BSD-3-Clause); see NOTICE.
//!
//! Two jobs, and the second is the one that decides whether it speaks at all.
//!
//! # The alphabet is small and silent about what it drops
//!
//! `text/symbols.py` in the reference builds a 71-entry table - pad, `-`,
//! `!,.:;? `, `A-Za-z`, `0-9` - and its `text_to_sequence` discards everything
//! else without a word. The same file *defines* 20,950 Han characters and an
//! ARPAbet set and exports neither, so Han fed to this model tokenises to
//! nothing and it synthesises near-silence instead of failing. [`Tokenizer::encode`] keeps
//! that behaviour, because it is the checkpoint's, and reports the count so a
//! caller can tell the difference between a short line and a dropped one.
//!
//! # The script has to be converted, not just cleaned
//!
//! The engine's translator emits POJ with diacritics; this model reads Tâi-lô
//! with the tone as a trailing ASCII digit. Those are the same language in two
//! orthographies, and the conversion lives in `xabe-taigi` - it did live here,
//! and moved when a second checkpoint needed it. What stays is the punctuation
//! fold below, which is this model's own requirement and the opposite of the
//! other's.
//!
//! Passing POJ through unconverted is not a degraded reading. `á` is not in the
//! table, so it vanishes, and the tone vanishes with it - which is the same
//! silence as feeding it Han, arrived at less obviously.

use rustc_hash::FxHashMap;

/// Maps the model's symbols to embedding rows.
pub struct Tokenizer {
    ids: FxHashMap<char, i64>,
}

impl Tokenizer {
    /// Builds the table from the config's symbol list.
    ///
    /// Multi-character symbols are skipped: the published table has none, and
    /// the reference's tokeniser walks the string one character at a time, so
    /// a longer entry could never be matched by it either.
    pub fn new(symbols: &[String]) -> Self {
        let mut ids = FxHashMap::default();
        for (i, s) in symbols.iter().enumerate() {
            let mut cs = s.chars();
            if let (Some(c), None) = (cs.next(), cs.next()) {
                ids.insert(c, i as i64);
            }
        }
        Self { ids }
    }

    /// `basic_cleaners` then the table: lowercase, collapse whitespace, drop
    /// anything unknown.
    ///
    /// Returns the ids and how many characters were dropped. The pad is dropped
    /// too even though it is in the table - the reference excludes it by
    /// identity, and a pad in the middle of a sequence is not a symbol the
    /// model ever saw.
    pub fn encode(&self, text: &str) -> (Vec<i64>, usize) {
        let mut out = Vec::new();
        let mut dropped = 0;
        let mut last_space = true;
        for c in text.chars().flat_map(|c| c.to_lowercase()) {
            if c.is_whitespace() {
                // Collapsing whitespace, and a leading run collapses to
                // nothing rather than to a space.
                if !last_space && let Some(&id) = self.ids.get(&' ') {
                    out.push(id);
                }
                last_space = true;
                continue;
            }
            last_space = false;
            match self.ids.get(&c) {
                Some(&id) if c != '_' => out.push(id),
                _ => dropped += 1,
            }
        }
        while out.last() == self.ids.get(&' ') && !out.is_empty() {
            out.pop();
        }
        (out, dropped)
    }
}

/// The model's punctuation, which is the ASCII half of its 71 symbols.
const PUNCT: [char; 6] = ['!', ',', '.', ':', ';', '?'];

/// The CJK punctuation this pipeline actually receives, and its ASCII twin.
///
/// The chat model writes Han and punctuates it full-width; the translator
/// carries that through into POJ; the reply is cut into clauses *at* those
/// marks, so nearly every line arriving here ends in one. None of them is in
/// the checkpoint's alphabet, so without this they are dropped in silence and
/// the line reaches the decoder with no punctuation at all - see
/// [`with_gate_cue`] for what that costs.
///
/// The reference never needed this: it reads romanised TLPA out of a file,
/// already punctuated in ASCII.
fn ascii_punct(c: char) -> Option<char> {
    Some(match c {
        '。' | '．' | '…' | '⋯' => '.',
        '，' | '、' => ',',
        '！' => '!',
        '？' => '?',
        '；' => ';',
        '：' => ':',
        _ => return None,
    })
}

/// Rewrites POJ as the Tâi-lô-with-tone-digits this checkpoint reads, and
/// folds full-width punctuation to ASCII.
///
/// The spelling half lives in `xabe-taigi` now, because the Coqui VITS needs
/// the same conversion and an edge from `xabe-tts` to this crate would be the
/// wrong shape. What stays here is the punctuation fold, which is this
/// checkpoint's alone: Tacotron2's gate learned end-of-utterance from ASCII
/// marks and a line without one runs the decoder to its step limit, where the
/// Coqui model saw no punctuation at all in training. Opposite requirements, so
/// neither belongs in the shared table.
///
/// Folding after the conversion rather than during it is exact: the spelling
/// rules never emit a full-width mark, so a second pass sees the same
/// characters the single pass would have.
pub fn poj_to_tlpa(text: &str) -> String {
    xabe_taigi::poj_to_tailo(text)
        .chars()
        .map(|c| ascii_punct(c).unwrap_or(c))
        .collect()
}

/// Appends a full stop when a line ends in no punctuation the model knows.
///
/// **This is what keeps the decoder from running away.** Tacotron2 stops when
/// its gate fires, and the gate learned end-of-utterance from the punctuation
/// that ended every line it was trained on. Give it a line with none and the
/// gate may simply never fire: the loop then runs to `max_decoder_steps`, and
/// 3000 frames at hop 256 is **34.8 seconds of held tone** - which is what a
/// listener hears as the voice getting stuck.
///
/// Measured on this checkpoint, medians of five, same sentence either way:
///
/// | line | median |
/// | --- | ---: |
/// | `tsiah8-pa2--bo5?` | 1.10 s |
/// | `tsiah8-pa2--bo5` | 34.83 s, every run |
/// | `li2 ho2, ... tsin1 ho2.` | 3.81 s |
/// | `li2 ho2, ... tsin1 ho2` | 34.83 s |
/// | `li2 ho2` | 0.62 s to 7.99 s, run to run |
///
/// That last row is why the fault reads as intermittent rather than as broken.
///
/// Any of the six marks will do - a trailing comma gates as well as a full
/// stop, measured - so this only appends where there is nothing at all, and
/// leaves a clause that already ends in `,` alone rather than repunctuating it
/// into a sentence it is not.
///
/// Runs on the output of [`poj_to_tlpa`] rather than its input, because that
/// is the pass which folds `。` into a mark this one recognises. The other
/// order sees `。` as unpunctuated, appends to it, and says `?.`.
///
/// Applied to synthesis and deliberately not to [`Taco::encoder`], which is
/// pinned against a captured oracle and has to transform text exactly as the
/// reference does, terminal punctuation and all.
///
/// [`Taco::encoder`]: crate::Taco::encoder
pub fn with_gate_cue(text: &str) -> std::borrow::Cow<'_, str> {
    match text.trim_end().chars().next_back() {
        Some(c) if PUNCT.contains(&c) => std::borrow::Cow::Borrowed(text),
        // Nothing speakable at all: leave it, so the caller still gets the
        // empty-sequence path and its warning rather than a lone full stop.
        None => std::borrow::Cow::Borrowed(text),
        Some(_) => std::borrow::Cow::Owned(format!("{}.", text.trim_end())),
    }
}
