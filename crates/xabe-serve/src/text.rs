//! Text handling that the pipeline's behaviour depends on.
//!
//! Everything here is a pure function over a string, which is deliberate: this
//! is the part of the gateway that was tuned against real speech over many
//! turns, and the reasons are not recoverable from the code alone. Keeping it
//! pure is what lets each rule be tested against the case that produced it.
//!
//! Three things live here:
//!
//! - [`sanitize_asr`], the last of three layers of hallucination defence.
//! - [`Chunker`], which decides when enough of a streaming reply has arrived to
//!   be worth synthesising.
//! - [`clean`], [`split_sentences`], [`split_poj`] and [`normalize_for_mms`],
//!   which prepare text for the synthesiser.
//!
//! It refuses to do any I/O and knows nothing about HTTP.

use regex::Regex;
use std::sync::LazyLock;

// --------------------------------------------------------- hallucination guard

/// Subtitle-style annotations: `(我會陪你一起走)`, `[音樂]`.
///
/// Breeze-ASR-26 emits these on noise because they are in its training data.
static ANNOTATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[（(\[【][^）)\]】]{0,30}[）)\]】]").expect("annotation regex"));

/// A transcript that survived cleaning but carries no words.
static PUNCT_ONLY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^[\s\W_…。，、！？!?.,'"“”‘’()（）\[\]【】-]*$"#).expect("punct regex")
});

/// The phrases Whisper reaches for when it has nothing to transcribe.
///
/// All of them are YouTube subtitle boilerplate. They arrive verbatim and
/// confidently, which is what makes them dangerous: the assistant then answers
/// the hallucination as though it were a turn.
static HALLUCINATION: LazyLock<Regex> = LazyLock::new(|| {
    // Built with `concat!` rather than one long literal. A raw string has no
    // line continuations - a trailing backslash in `r"..."` is a backslash -
    // and a non-raw one would need every `\s` doubled. Both spellings have
    // produced a pattern that compiled and then matched nothing.
    Regex::new(concat!(
        r"(?i)^(?:",
        r"謝謝(?:大家|觀看|收看|你的?觀看)?|感謝(?:大家|收看|觀看)|",
        r"請(?:不吝)?(?:點贊|點讚|訂閱|按讚|分享|關注).*|",
        r"字幕(?:由|提供|製作).*|下[集期]再見|我們下次再見|",
        r"以上是今天的.*|本影片.*|明鏡.*點點欄目.*|",
        // `Thanks? for watching` is what was ported, and it does not match
        // "Thank you for watching" - the commonest form of the phrase, and one
        // this ASR emits. Widened, with a test for the case it used to miss.
        r"Thank(?:s| you)? for watching.*|Subtitles? by.*|Please subscribe.*|",
        r"Thanks? for your watching.*|See you next time.*",
        r")[。.!！\s]*$",
    ))
    .expect("hallucination regex")
});

/// Collapses runs of whitespace to one space.
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("ws regex"));

/// Characters that carry no word content, for the length test.
static NON_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s\W_]").expect("nonword regex"));

/// Returns a cleaned transcript, or empty if it looks like a hallucination.
///
/// This is the third and last layer. Upstream of it: Silero VAD gating, which
/// removes almost all of it, and the decoder thresholds `-sns -nth 0.8 -et 2.2
/// -lpt -0.5`. What reaches here is what survived both — on pure digital
/// silence that was `我…`, on faint hiss `我現在在醫院`, on room noise
/// `(我會陪你一起走)`.
pub fn sanitize_asr(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let t = ANNOTATION.replace_all(text, "");
    let t = t.trim();
    let t = WHITESPACE.replace_all(t, " ");

    if PUNCT_ONLY.is_match(&t) || HALLUCINATION.is_match(&t) {
        return String::new();
    }
    // A single character after cleaning is noise far more often than a turn.
    //
    // The Python this was ported from dropped these *unconditionally*, which
    // also threw away 好 and 是 — legitimate one-character replies that a
    // conversational assistant should hear. The rule is kept, because on this
    // corpus it removes far more noise than speech, but the words that are
    // whole turns in their own right are now exempt.
    let words: String = NON_WORD.replace_all(&t, "").into_owned();
    if words.chars().count() < 2 && !is_whole_turn(&words) {
        return String::new();
    }
    t.into_owned()
}

/// One-character utterances that are real turns rather than noise.
///
/// Deliberately a short closed list rather than a rule. Any single character
/// *could* be speech; these are the ones common enough in reply position that
/// dropping them is noticeably wrong.
fn is_whole_turn(word: &str) -> bool {
    matches!(
        word,
        "好" | "是" | "對" | "嗯" | "有" | "無" | "行" | "欸" | "喔" | "會" | "不" | "袂"
    )
}

// -------------------------------------------------------------- reply chunking

/// Sentence boundaries, for every chunk after the first.
static SENT_END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[。！？!?\n]").expect("sent"));

/// Clause boundaries, for the first chunk only.
static CLAUSE_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[，、,；;。！？!?\n]").expect("clause"));

/// Decides when a streaming reply has produced enough to synthesise.
///
/// The first chunk breaks at a *clause* boundary and every later one at a
/// *sentence* boundary. That asymmetry is the whole point: a Taigi reply is
/// often one long sentence, so waiting for 。 means waiting for all of it
/// before any audio exists at all. Measured 4.1 s → 2.7 s to first audio.
#[derive(Debug)]
pub struct Chunker {
    pending: String,
    dispatched: bool,
    first_min: usize,
    later_min: usize,
}

impl Chunker {
    /// `first_min` and `later_min` are minimum *characters*, not bytes.
    pub fn new(first_min: usize, later_min: usize) -> Chunker {
        Chunker {
            pending: String::new(),
            dispatched: false,
            first_min,
            later_min,
        }
    }

    /// Feeds one streamed piece, returning a chunk when one is ready.
    pub fn push(&mut self, piece: &str) -> Option<String> {
        self.pending.push_str(piece);
        let boundary = if self.dispatched {
            &*SENT_END
        } else {
            &*CLAUSE_END
        };
        let min = if self.dispatched {
            self.later_min
        } else {
            self.first_min
        };
        let trimmed = self.pending.trim();
        if boundary.is_match(&self.pending) && trimmed.chars().count() >= min {
            let out = trimmed.to_string();
            self.pending.clear();
            self.dispatched = true;
            return Some(out);
        }
        None
    }

    /// Whatever is left when the stream ends.
    pub fn finish(&mut self) -> Option<String> {
        let trimmed = self.pending.trim();
        if trimmed.is_empty() {
            return None;
        }
        let out = trimmed.to_string();
        self.pending.clear();
        Some(out)
    }
}

// ------------------------------------------------------- preparing text to say

/// Markdown decoration that carries no speech.
static MARKDOWN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[*_`#]+").expect("markdown"));

/// Bracketed asides, which are written but not spoken.
static BRACKETED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\[(（【][^\])）】]*[\])）】]").expect("bracketed"));

/// Strips non-spoken decoration. Operates on characters, never bytes.
pub fn clean(text: &str) -> String {
    let t = BRACKETED.replace_all(text, "");
    let t = MARKDOWN.replace_all(&t, "");
    WHITESPACE.replace_all(&t, " ").trim().to_string()
}

/// Splits Han text into chunks short enough to synthesise well.
///
/// VITS degrades on long inputs, and chunking also gets first audio out sooner.
/// Sentences first, then commas, then a hard cut — the hard cut exists because
/// a sentence with no internal punctuation is possible and must still terminate.
pub fn split_sentences(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for part in split_after(text, &['。', '！', '？', '!', '?']) {
        let mut part = part.trim().to_string();
        while part.chars().count() > max_chars {
            let cut = last_index_of(&part, &['，', ','], max_chars);
            let cut = match cut {
                Some(i) => i + 1,
                None => max_chars,
            };
            let (head, tail) = split_at_char(&part, cut);
            out.push(head.trim().to_string());
            part = tail.trim().to_string();
        }
        if !part.is_empty() {
            out.push(part);
        }
    }
    out
}

/// Splits romanised text, which has ASCII punctuation rather than CJK.
pub fn split_poj(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for part in split_after(text, &['.', '!', '?']) {
        let mut part = part.trim().to_string();
        while part.chars().count() > max_chars {
            // A comma, else a word boundary, else a hard cut. Cutting POJ
            // mid-syllable produces a sound that is not a word, so the space
            // fallback matters more here than in the Han path.
            let cut = last_index_of(&part, &[','], max_chars)
                .map(|i| i + 1)
                .or_else(|| last_index_of(&part, &[' '], max_chars))
                .unwrap_or(max_chars);
            let (head, tail) = split_at_char(&part, cut);
            out.push(head.trim().to_string());
            part = tail.trim().to_string();
        }
        if !part.is_empty() {
            out.push(part);
        }
    }
    out
}

/// Normalises Pe̍h-ōe-jī for `facebook/mms-tts-nan`.
///
/// The model's 48-symbol vocabulary is **POJ, not Tâi-lô**: it contains `c`
/// (for ch/chh) and U+0358 COMBINING DOT ABOVE RIGHT (for o͘), neither of which
/// Tâi-lô uses, so converting POJ to Tâi-lô moves the text *away* from what the
/// model was trained on. The one POJ symbol the vocabulary lacks is the nasal
/// ⁿ (U+207F), which becomes `nn`. Everything else is left as written.
///
/// See `docs/MODEL.md` for the round-trip measurement that established this.
pub fn normalize_for_mms(s: &str) -> String {
    s.replace(['\u{207f}', '\u{1d3a}'], "nn")
}

/// Splits after any of `marks`, keeping the mark with the preceding part.
fn split_after(text: &str, marks: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if marks.contains(&c) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Character index of the last of `marks` before `limit` characters.
fn last_index_of(s: &str, marks: &[char], limit: usize) -> Option<usize> {
    s.chars()
        .take(limit)
        .enumerate()
        .filter(|(_, c)| marks.contains(c))
        .map(|(i, _)| i)
        .last()
}

/// Splits at a *character* index, not a byte index.
fn split_at_char(s: &str, at: usize) -> (String, String) {
    let byte = s.char_indices().nth(at).map(|(b, _)| b).unwrap_or(s.len());
    (s[..byte].to_string(), s[byte..].to_string())
}
