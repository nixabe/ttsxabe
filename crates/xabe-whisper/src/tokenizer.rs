//! Byte-level BPE, matching 🤗 `WhisperTokenizer`.
//!
//! Whisper inherits GPT-2's tokenizer wholesale: text is pre-split by a fixed
//! regex, each piece is read as *bytes* rather than characters, each byte is
//! mapped into a printable code point, and the resulting string is merged pair
//! by pair in a learned order. Working in bytes is what lets it round-trip any
//! input at all, Han and emoji included, without an unknown token.
//!
//! # Why the special tokens are read and not synthesised
//!
//! whisper.cpp derives the 1,607 special ids arithmetically from `n_vocab`,
//! because its container has nowhere to put them: the file holds 50,258
//! entries against a vocabulary of 51,865, and the multilingual shift is
//! `n_vocab - 51765 - 1`. Get that expression wrong and every language and
//! timestamp id is off by one - a transcript in the wrong language, or
//! timestamps that are all 20 ms early, with nothing to indicate why.
//!
//! This checkpoint ships `added_tokens.json`, which states all 1,607 outright,
//! so they are read. The arithmetic is still checked, once, in the tests -
//! against the file rather than against itself.
//!
//! `<|endoftext|>` is *not* among them: it predates the multilingual tokens and
//! lives in `vocab.json` proper, declared special only by
//! `special_tokens_map.json`. Reading `added_tokens.json` alone gets 1,607 of
//! 1,608 specials and leaves the one the decoder stops on looking like
//! ordinary text.

use crate::WhisperError;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::LazyLock;

/// GPT-2's pre-tokenization pattern, minus the one piece the `regex` crate has
/// no syntax for.
///
/// The reference's final alternatives are `\s+(?!\S)|\s+`. Rust's `regex` has
/// no lookaround, and the negative lookahead is not decoration: it is what
/// makes a run of *k* spaces before a word split as *k-1* spaces plus a word
/// that owns the last one. That behaviour is reproduced explicitly in
/// [`pre_tokenize`] instead, where it is at least visible.
static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .expect("the GPT-2 pattern is a literal")
});

/// Splits text the way the reference does, and for the same reasons.
///
/// Every character is covered by some alternative, so a match always begins at
/// the position asked for and the loop makes progress.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let m = PATTERN
            .find(&text[pos..])
            .expect("the pattern matches at every position");
        debug_assert_eq!(m.start(), 0, "the pattern skipped a character");
        let mut piece = &text[pos..pos + m.end()];

        // `\s+(?!\S)`: a whitespace run that is followed by a non-space gives
        // up its last character, so the next piece can begin with a space.
        // That is why " hello" is one token and not two.
        if piece.chars().all(char::is_whitespace)
            && piece.chars().count() > 1
            && text[pos + m.end()..]
                .chars()
                .next()
                .is_some_and(|c| !c.is_whitespace())
        {
            let last = piece.chars().next_back().expect("non-empty");
            piece = &piece[..piece.len() - last.len_utf8()];
        }

        out.push(piece);
        pos += piece.len();
    }
    out
}

/// GPT-2's byte-to-code-point alphabet.
///
/// The printable ASCII range and two Latin-1 stretches map to themselves; the
/// 68 remaining bytes are pushed into the private-ish range starting at 256.
/// The point is only that every byte gets a distinct, printable code point, so
/// `vocab.json` and `merges.txt` can be plain text.
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
    /// Byte string to id, for the 50,258 learned tokens.
    vocab: FxHashMap<Vec<u8>, u32>,
    /// Id to byte string, learned and special alike; specials hold their
    /// literal `<|...|>` spelling so a decode that keeps them is a plain
    /// concatenation.
    pieces: Vec<Vec<u8>>,
    /// Which ids are special, indexed the same way as `pieces`.
    is_special: Vec<bool>,
    /// Merge rank of an adjacent pair of byte strings; lower merges first.
    ranks: FxHashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Spelling to id for the special tokens.
    specials: FxHashMap<String, u32>,
    /// Id of `<|0.00|>`, above which every id is a timestamp.
    timestamp_begin: u32,
    /// The byte alphabet, kept for encoding.
    alphabet: [char; 256],
}

impl Tokenizer {
    /// Reads `vocab.json`, `merges.txt` and `added_tokens.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, WhisperError> {
        let alphabet = byte_alphabet();
        let back: FxHashMap<char, u8> = alphabet
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let read = |name: &str| -> Result<String, WhisperError> {
            let path = dir.join(name);
            std::fs::read_to_string(&path).map_err(|source| WhisperError::Io { path, source })
        };
        let bad = |name: &str, what: String| WhisperError::Vocab {
            path: dir.join(name),
            what,
        };

        let raw: FxHashMap<String, u32> = serde_json::from_str(&read("vocab.json")?)
            .map_err(|e| bad("vocab.json", e.to_string()))?;
        let added: FxHashMap<String, u32> = serde_json::from_str(&read("added_tokens.json")?)
            .map_err(|e| bad("added_tokens.json", e.to_string()))?;
        let declared = special_names(&read("special_tokens_map.json")?)
            .map_err(|e| bad("special_tokens_map.json", e))?;

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
        let mut specials = added;
        for (s, id) in &specials {
            let slot = pieces.get_mut(*id as usize).ok_or_else(|| {
                bad(
                    "added_tokens.json",
                    format!("id {id} is past the vocabulary"),
                )
            })?;
            *slot = s.as_bytes().to_vec();
            is_special[*id as usize] = true;
        }
        // The ones `special_tokens_map.json` declares but `added_tokens.json`
        // does not: on this checkpoint that is `<|endoftext|>` alone, which
        // sits in `vocab.json` because it predates the multilingual tokens.
        for name in declared {
            let Some(&id) = vocab.get(name.as_bytes()) else {
                continue;
            };
            pieces[id as usize] = name.as_bytes().to_vec();
            is_special[id as usize] = true;
            specials.insert(name, id);
        }

        // `merges.txt` is ranked by line, after a `#version:` header. The rank
        // *is* the priority, so the order of the file is load-bearing and a
        // hash map keyed on the pair is the only structure that survives it.
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

        // Everything at or above `<|0.00|>` is a timestamp. Whisper's decoder
        // treats that block differently from every other special token, so the
        // boundary is worth a field rather than a comparison written out at
        // each of the three places that needs it.
        let timestamp_begin = specials.get("<|0.00|>").copied().ok_or_else(|| {
            bad(
                "added_tokens.json",
                "no <|0.00|>, so the timestamp block has no start".to_string(),
            )
        })?;

        Ok(Self {
            vocab,
            pieces,
            is_special,
            ranks,
            specials,
            timestamp_begin,
            alphabet,
        })
    }

    /// The id of `<|0.00|>`; every id from here up is a timestamp.
    pub fn timestamp_begin(&self) -> u32 {
        self.timestamp_begin
    }

    /// Whether an id names a timestamp rather than a token of text.
    pub fn is_timestamp(&self, id: u32) -> bool {
        id >= self.timestamp_begin
    }

    /// How many ids the tokenizer knows, learned and special together.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the tokenizer knows nothing, which [`Tokenizer::from_dir`]
    /// would have refused.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The id of a special token, by its `<|...|>` spelling.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials.get(name).copied()
    }

    /// Whether an id is one of the special tokens.
    pub fn is_special(&self, id: u32) -> bool {
        self.is_special.get(id as usize).copied().unwrap_or(false)
    }

    /// Merges one pre-tokenized piece down to its token ids.
    ///
    /// The loop is the reference's: find the adjacent pair with the lowest
    /// rank, merge every occurrence of *that* pair, repeat. Merging the
    /// leftmost mergeable pair instead - the obvious reading - produces
    /// different, plausible, wrong tokens on perhaps one word in fifty.
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
            // A single byte is always in the vocabulary - that is the whole
            // point of a byte-level alphabet - so a miss here means the merge
            // table and the vocabulary came from different checkpoints.
            let id = self
                .vocab
                .get(&p)
                .copied()
                .unwrap_or_else(|| panic!("merged piece {p:?} is not in the vocabulary"));
            out.push(id);
        }
    }

    /// Encodes text, recognising `<|...|>` special tokens where they appear.
    ///
    /// No special tokens are added. The prefix a transcript starts with is a
    /// decoding decision, not a tokenization one, and it is built explicitly
    /// where the decoding happens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Specials are matched before pre-tokenization, because the regex
            // would otherwise split `<|zh|>` into four pieces of punctuation.
            if let Some(at) = rest.find("<|") {
                let (plain, tail) = rest.split_at(at);
                if let Some((name, id)) = self.match_special(tail) {
                    self.encode_plain(plain, &mut out);
                    out.push(id);
                    rest = &tail[name..];
                    continue;
                }
                // A bare `<|` that begins no known token: consume through it
                // so the scan cannot stall, and let it tokenize as text.
                let cut = at + "<|".len();
                self.encode_plain(&rest[..cut], &mut out);
                rest = &rest[cut..];
            } else {
                self.encode_plain(rest, &mut out);
                break;
            }
        }
        out
    }

    /// Longest special token spelled at the start of `s`, and its byte length.
    fn match_special(&self, s: &str) -> Option<(usize, u32)> {
        let end = s.find("|>")? + "|>".len();
        self.specials.get(&s[..end]).map(|&id| (end, id))
    }

    /// Encodes text with no special tokens in it.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pre_tokenize(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }

    /// Decodes ids back to text, dropping timestamps.
    ///
    /// Timestamps go even when `skip_special` is false, which is the
    /// reference's default and looks inconsistent until you notice that
    /// `<|2.50|>` is not a thing anyone wants in a transcript. Use
    /// [`Tokenizer::decode_with_timestamps`] to keep them.
    ///
    /// The bytes are concatenated before they are read as UTF-8, which is the
    /// only order that works: a single Han character is three bytes and BPE
    /// routinely splits it across two tokens, so decoding token by token would
    /// produce replacement characters where the reference produces text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        self.decode_inner(ids, skip_special, false)
    }

    /// Decodes ids back to text, spelling timestamps out as `<|2.50|>`.
    pub fn decode_with_timestamps(&self, ids: &[u32], skip_special: bool) -> String {
        self.decode_inner(ids, skip_special, true)
    }

    /// The body of both, with the two decisions made explicit.
    fn decode_inner(&self, ids: &[u32], skip_special: bool, timestamps: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let drop = if self.is_timestamp(id) {
                !timestamps
            } else {
                skip_special && self.is_special(id)
            };
            if drop {
                continue;
            }
            if let Some(p) = self.pieces.get(id as usize) {
                bytes.extend_from_slice(p);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// A token's spelling in the byte alphabet, as `vocab.json` writes it.
    ///
    /// For inspecting a token stream by eye; nothing in the engine needs it.
    pub fn spelling(&self, id: u32) -> Option<String> {
        let p = self.pieces.get(id as usize)?;
        if self.is_special(id) {
            return Some(String::from_utf8_lossy(p).into_owned());
        }
        Some(p.iter().map(|&b| self.alphabet[b as usize]).collect())
    }
}

/// Every token `special_tokens_map.json` declares, by spelling.
///
/// The file mixes two spellings of the same thing: a bare string, or an object
/// with a `content` field and stripping flags. Both appear in checkpoints in
/// the wild, sometimes in the same file, so both are read.
fn special_names(text: &str) -> Result<Vec<String>, String> {
    let map: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut push = |v: &serde_json::Value| {
        if let Some(s) = v.as_str() {
            out.push(s.to_string());
        } else if let Some(s) = v.get("content").and_then(serde_json::Value::as_str) {
            out.push(s.to_string());
        }
    };
    for (key, value) in map.as_object().ok_or("not an object")? {
        if key == "additional_special_tokens" {
            for v in value
                .as_array()
                .ok_or("additional_special_tokens is not a list")?
            {
                push(v);
            }
        } else {
            push(value);
        }
    }
    Ok(out)
}
