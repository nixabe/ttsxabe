//! Qwen2's byte-level BPE, which is what CosyVoice3 reads text with.
//!
//! # Why a third BPE in this workspace
//!
//! `xabe-whisper` has GPT-2's and `xabe-llama` has Llama-3's, and all three are
//! the same idea: split by a regex, read each piece as bytes, map every byte to
//! a printable code point, merge pairs in a learned order. What differs is the
//! regex and where the special tokens come from, and those are exactly the two
//! places a shared implementation would need a switch at every call. So this is
//! its own file, and the differences are stated rather than parameterised.
//!
//! Against Llama-3's, the pattern differs in one alternative: `\p{N}` rather
//! than `\p{N}{1,3}`, so **every digit is its own piece**. Against GPT-2's it
//! differs in four. The comment on [`PATTERN`] says what each one is for.
//!
//! # The special tokens are read, because they are not in the checkpoint
//!
//! `CosyVoice-BlankEN` ships the learned half and nothing else. The 281 special
//! tokens - `<|endofprompt|>`, the paralinguistic markers, three hundred-odd
//! pinyin finals - are a literal list in CosyVoice's *source*, handed to
//! `add_special_tokens` at construction, and their ids fall out of that list's
//! order. `tools/dump_cosyvoice_tokens.py` writes them down once; this reads
//! the file. Re-deriving them arithmetically is the mistake `xabe-whisper`'s
//! module already documents from the other side.
//!
//! # They are matched before the regex, and not all of them look special
//!
//! `<|endofprompt|>` would survive pre-tokenization as punctuation and merge
//! into something plausible. `[breath]`, `<strong>` and `[iǎng]` would too. So
//! the scan looks for a special at every position first, longest match wins,
//! and only the text between them goes through the regex. That is what 🤗 does
//! with `split_special_tokens=False`, which is this tokenizer's setting.

use crate::CosyError;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::LazyLock;

/// Qwen2's pre-tokenization pattern, minus the alternative `regex` cannot
/// express.
///
/// The reference is
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
/// Rust's `regex` has no lookaround, so `\s+(?!\S)` is dropped here and
/// reproduced in [`pre_tokenize`], where at least it is visible.
///
/// Alternative by alternative, since three of them are traps:
///
/// - `(?i:...)` - the contractions are **case-insensitive** here, so `'S`
///   tokenizes like `'s`. GPT-2's are not.
/// - `[^\r\n\p{L}\p{N}]?\p{L}+` - a letter run may take one leading character
///   that is neither a letter, a digit, nor a newline. That is how a space
///   joins the word after it.
/// - `\p{N}` - one digit at a time. Llama-3 takes up to three.
/// - `\s*[\r\n]+` - a run containing a newline is its own alternative, and it
///   is matched *before* `\s+`. This is why the trim in [`pre_tokenize`] must
///   not touch a piece with a newline in it.
static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(concat!(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
        r"|\p{N}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"|\s*[\r\n]+",
        r"|\s+",
    ))
    .expect("the qwen2 pattern is a literal")
});

/// Splits text the way the reference does, including the alternative the
/// pattern above had to give up.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let m = PATTERN
            .find(&text[pos..])
            .expect("every character is covered by some alternative");
        debug_assert_eq!(m.start(), 0, "the pattern skipped a character");
        let mut piece = &text[pos..pos + m.end()];

        // `\s+(?!\S)`: a whitespace run that something follows gives up its
        // last character, so the next piece can begin with a space - which is
        // why " hello" is one token and not two.
        //
        // Only for runs the *last* alternative matched. A run containing a
        // newline was matched by `\s*[\r\n]+`, which has no such lookahead,
        // and trimming it steals a character that belongs to the piece.
        if piece.chars().all(char::is_whitespace)
            && !piece.contains(['\n', '\r'])
            && piece.chars().count() > 1
            && pos + m.end() < text.len()
        {
            let last = piece.chars().next_back().expect("non-empty");
            piece = &piece[..piece.len() - last.len_utf8()];
        }

        out.push(piece);
        pos += piece.len();
    }
    out
}

/// GPT-2's byte-to-code-point alphabet, which Qwen2 inherits unchanged.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut extra = 0u32;
    for (b, slot) in out.iter_mut().enumerate() {
        let b = b as u32;
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        *slot = if printable {
            char::from_u32(b).expect("in range")
        } else {
            let c = char::from_u32(256 + extra).expect("in range");
            extra += 1;
            c
        };
    }
    out
}

/// Turns a byte-alphabet string back into the bytes it stands for.
fn decode_alphabet(s: &str, back: &FxHashMap<char, u8>) -> Option<Vec<u8>> {
    s.chars().map(|c| back.get(&c).copied()).collect()
}

/// The tokenizer: a vocabulary, a merge ranking, and the special tokens.
#[derive(Debug)]
pub struct Tokenizer {
    vocab: FxHashMap<Vec<u8>, u32>,
    pieces: Vec<Vec<u8>>,
    is_special: Vec<bool>,
    ranks: FxHashMap<(Vec<u8>, Vec<u8>), u32>,
    specials: FxHashMap<String, u32>,
    /// The longest special token in bytes, which bounds the match at each
    /// position - without it the scan is quadratic in the length of the text.
    longest_special: usize,
}

impl Tokenizer {
    /// Reads `vocab.json`, `merges.txt` and `added_tokens.json` from a
    /// `CosyVoice-BlankEN` directory.
    pub fn from_dir(dir: &Path) -> Result<Self, CosyError> {
        let alphabet = byte_alphabet();
        let back: FxHashMap<char, u8> = alphabet
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let read = |name: &str| -> Result<String, CosyError> {
            std::fs::read_to_string(dir.join(name)).map_err(|e| CosyError::Speaker {
                what: format!("{}: {e}", dir.join(name).display()),
            })
        };
        let bad = |name: &str, what: String| CosyError::Speaker {
            what: format!("{}: {what}", dir.join(name).display()),
        };

        let raw: FxHashMap<String, u32> = serde_json::from_str(&read("vocab.json")?)
            .map_err(|e| bad("vocab.json", e.to_string()))?;
        // Written by `tools/dump_cosyvoice_tokens.py`; see the module header
        // for why it cannot come from the checkpoint.
        let added: FxHashMap<String, u32> = serde_json::from_str(&read("added_tokens.json")?)
            .map_err(|e| bad("added_tokens.json", e.to_string()))?;

        let n = raw.len() + added.len();
        let mut pieces = vec![Vec::new(); n];
        let mut is_special = vec![false; n];
        let mut vocab = FxHashMap::default();
        vocab.reserve(raw.len());

        for (s, id) in raw {
            let bytes = decode_alphabet(&s, &back)
                .ok_or_else(|| bad("vocab.json", format!("{s:?} is not in the byte alphabet")))?;
            let slot = pieces
                .get_mut(id as usize)
                .ok_or_else(|| bad("vocab.json", format!("id {id} is past the vocabulary")))?;
            *slot = bytes.clone();
            vocab.insert(bytes, id);
        }
        for (s, id) in &added {
            let slot = pieces.get_mut(*id as usize).ok_or_else(|| {
                bad(
                    "added_tokens.json",
                    format!("id {id} is past the vocabulary"),
                )
            })?;
            *slot = s.as_bytes().to_vec();
            is_special[*id as usize] = true;
        }

        // `merges.txt` is ranked by line after a `#version:` header, so the
        // order of the file *is* the priority.
        let merges = read("merges.txt")?;
        let mut ranks = FxHashMap::default();
        for (rank, line) in merges.lines().skip(1).filter(|l| !l.is_empty()).enumerate() {
            let (a, b) = line
                .split_once(' ')
                .ok_or_else(|| bad("merges.txt", format!("line {rank} has no space")))?;
            let (a, b) = (
                decode_alphabet(a, &back)
                    .ok_or_else(|| bad("merges.txt", format!("{a:?} is not in the alphabet")))?,
                decode_alphabet(b, &back)
                    .ok_or_else(|| bad("merges.txt", format!("{b:?} is not in the alphabet")))?,
            );
            ranks.insert((a, b), rank as u32);
        }

        let longest_special = added.keys().map(String::len).max().unwrap_or(0);
        Ok(Self {
            vocab,
            pieces,
            is_special,
            ranks,
            specials: added,
            longest_special,
        })
    }

    /// How many ids the tokenizer knows, learned and special together.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether it knows nothing, which [`Tokenizer::from_dir`] would refuse.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The id of a special token, by its literal spelling.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials.get(name).copied()
    }

    /// Whether an id is one of the special tokens.
    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    /// Merges one pre-tokenized piece down to ids.
    ///
    /// The loop is the reference's: find the adjacent pair with the *lowest
    /// rank*, merge every occurrence of that pair, repeat. Merging the leftmost
    /// mergeable pair instead - the obvious reading - gives different,
    /// plausible, wrong tokens on perhaps one word in fifty.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        while let Some((at, _)) = parts
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                self.ranks
                    .get(&(w[0].clone(), w[1].clone()))
                    .map(|&r| (i, r))
            })
            .min_by_key(|&(_, r)| r)
        {
            let (a, b) = (parts[at].clone(), parts[at + 1].clone());
            let mut merged = Vec::with_capacity(parts.len() - 1);
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    let mut joined = a.clone();
                    joined.extend_from_slice(&b);
                    merged.push(joined);
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut parts[i]));
                    i += 1;
                }
            }
            parts = merged;
        }

        for p in parts {
            // A single byte is always in the vocabulary - that is what a
            // byte-level alphabet is for - so a miss means the merge table and
            // the vocabulary came from different checkpoints.
            let id = self
                .vocab
                .get(&p)
                .copied()
                .unwrap_or_else(|| panic!("merged piece {p:?} is not in the vocabulary"));
            out.push(id);
        }
    }

    /// Encodes text, recognising the special tokens where they appear.
    ///
    /// Nothing is prepended or appended. What surrounds the text - the instruct
    /// string, `<|endofprompt|>`, the speech markers - is a decision the caller
    /// makes, and `SpeechLlm::prompt` is where it is made.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut plain_from = 0;
        let mut at = 0;
        while at < text.len() {
            match self.match_special(text, at) {
                Some((len, id)) => {
                    if plain_from < at {
                        self.encode_plain(&text[plain_from..at], &mut out);
                    }
                    out.push(id);
                    at += len;
                    plain_from = at;
                }
                // Advance by a whole character: `at` has to stay on a
                // boundary or the next slice panics.
                None => {
                    at += text[at..]
                        .chars()
                        .next()
                        .expect("at is on a character boundary")
                        .len_utf8();
                }
            }
        }
        if plain_from < text.len() {
            self.encode_plain(&text[plain_from..], &mut out);
        }
        out
    }

    /// The longest special token spelled at `at`, and its length in bytes.
    fn match_special(&self, text: &str, at: usize) -> Option<(usize, u32)> {
        let end = (at + self.longest_special).min(text.len());
        // Longest first, so `[iǎng]` wins over any shorter marker that is a
        // prefix of it.
        for e in (at + 1..=end).rev() {
            if !text.is_char_boundary(e) {
                continue;
            }
            if let Some(&id) = self.specials.get(&text[at..e]) {
                return Some((e - at, id));
            }
        }
        None
    }

    /// Encodes a stretch with no special tokens in it.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pre_tokenize(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }

    /// Decodes ids back to text, dropping the special tokens.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            if let Some(p) = self.pieces.get(id as usize) {
                bytes.extend_from_slice(p);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
