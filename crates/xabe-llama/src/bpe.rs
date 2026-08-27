//! Llama-3's byte-level BPE, read out of a GGUF.
//!
//! # Why this is not `xabe-whisper`'s tokenizer
//!
//! It is the same *family* - byte-level BPE with a `Ġ`-escaped alphabet and a
//! ranked merge table - and it is not the same tokenizer. Three things differ,
//! and each of them changes the output:
//!
//! - **The pre-tokenizer pattern.** The GGUF says `tokenizer.ggml.pre` is
//!   `llama-bpe`, not `gpt2`. Llama-3 splits digit runs into groups of at most
//!   three, matches contractions case-insensitively, and lets a newline run
//!   attach to preceding punctuation. GPT-2 does none of that.
//! - **Where the vocabulary lives.** Whisper reads `vocab.json` and
//!   `merges.txt` from beside the checkpoint; this reads two metadata arrays
//!   from inside the file.
//! - **The special tokens.** Llama-3's are `<|begin_of_text|>` and friends,
//!   declared by `token_type` rather than by a separate file.
//!
//! What that leaves shared is the inner merge loop and the byte alphabet,
//! about eighty lines. Hoisting those into a fourth crate to be depended on by
//! two would be more machinery than the duplication costs, so they are written
//! twice and each is tested against its own reference.

use crate::LlamaError;
use rustc_hash::FxHashMap;
use std::sync::LazyLock;

/// Llama-3's pre-tokenizer, minus the one construct Rust's `regex` lacks.
///
/// The reference pattern ends `\s+(?!\S)|\s+`. There is no lookaround here, and
/// the negative lookahead is not decoration: it is what makes a run of *k*
/// spaces before a word split as *k-1* spaces plus a word owning the last one.
/// That is reproduced explicitly in [`pre_tokenize`], where it is visible.
///
/// `\p{N}{1,3}` is the Llama-3 change that bites hardest in practice: `1234`
/// becomes `123` + `4`, where GPT-2 would keep one run. A tokenizer that got
/// this wrong would agree with the reference on prose and disagree on every
/// number.
static PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(concat!(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
        r"|\p{N}{1,3}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"|\s*[\r\n]+",
        r"|\s+",
    ))
    .expect("the llama-bpe pattern is a literal")
});

/// Splits text the way the reference does.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let Some(m) = PATTERN.find(&text[pos..]) else {
            // Every character is covered by some alternative, so this cannot
            // happen; consuming one character anyway keeps the loop finite
            // rather than trusting that claim forever.
            let step = text[pos..].chars().next().map_or(1, char::len_utf8);
            out.push(&text[pos..pos + step]);
            pos += step;
            continue;
        };
        let mut piece = &text[pos..pos + m.end()];

        // `\s+(?!\S)`: a whitespace run followed by a non-space gives up its
        // last space to the next piece.
        //
        // Only the run that came from the *last* alternative does. A run
        // containing a newline was matched by `\s*[\r\n]+` further up the
        // pattern, which has no lookahead after it and keeps what it took -
        // so `"para\n\npara"` is one token for the blank line, not two for
        // the newlines. Testing for a newline in the piece is exactly the
        // condition that distinguishes the two alternatives, since the earlier
        // one always ends in one and the later one is only reached when it
        // failed.
        if piece.chars().all(char::is_whitespace)
            && !piece.contains(['\n', '\r'])
            && piece.chars().count() > 1
            && pos + m.end() < text.len()
        {
            let last = piece.chars().next_back().map_or(0, char::len_utf8);
            piece = &text[pos..pos + m.end() - last];
        }
        if piece.is_empty() {
            let step = text[pos..].chars().next().map_or(1, char::len_utf8);
            piece = &text[pos..pos + step];
        }
        pos += piece.len();
        out.push(piece);
    }
    out
}

/// The 256 printable stand-ins GPT-2 and Llama-3 both use.
///
/// Bytes that are not printable ASCII are mapped into a private run so the
/// vocabulary can be plain text. `Ġ` is a space, `Ċ` a newline.
fn byte_alphabet() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut next = 0u32;
    for (b, slot) in out.iter_mut().enumerate() {
        let b = b as u8;
        let printable =
            (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        *slot = if printable {
            b as char
        } else {
            let c = char::from_u32(256 + next).expect("inside the BMP");
            next += 1;
            c
        };
    }
    out
}

/// A byte-level BPE vocabulary and its merge ranks.
pub struct Bpe {
    /// Escaped spelling to id.
    ids: FxHashMap<String, u32>,
    /// Id to escaped spelling.
    pieces: Vec<String>,
    /// Merge rank of an adjacent pair; lower merges first.
    ranks: FxHashMap<(String, String), u32>,
    /// Byte to its stand-in character.
    alphabet: [char; 256],
    /// Spelling to id, for the tokens matched before the text is.
    specials: FxHashMap<String, u32>,
    bos: u32,
    eos: u32,
}

impl Bpe {
    /// Reads the vocabulary and merges out of a GGUF's metadata.
    pub fn from_gguf(f: &xabe_gguf::GgufFile) -> Result<Self, LlamaError> {
        let model = f
            .get_str("tokenizer.ggml.model")
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.model"))?;
        // `llama` is SentencePiece and a different algorithm entirely; it is
        // refused here rather than half-read into a byte-level table.
        if model != "gpt2" {
            return Err(LlamaError::Vocab {
                path: f.path().to_path_buf(),
                what: format!("tokenizer.ggml.model is `{model}`, not `gpt2`"),
            });
        }

        let tokens = f
            .get_strings("tokenizer.ggml.tokens")
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.tokens"))?;
        let merges = f
            .get_strings("tokenizer.ggml.merges")
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.merges"))?;
        let kinds = f
            .get_i32s("tokenizer.ggml.token_type")
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.token_type"))?;

        let mut ids = FxHashMap::default();
        ids.reserve(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }

        // Rank is line order, exactly as `merges.txt` would give it.
        let mut ranks = FxHashMap::default();
        ranks.reserve(merges.len());
        for (rank, line) in merges.iter().enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), rank as u32);
            }
        }

        // CONTROL (3) and USER_DEFINED (4) are the ones spelled `<|...|>` and
        // matched before the regex sees them. Taking the kind rather than the
        // spelling means a checkpoint that adds one is handled without a list
        // here going stale.
        let specials = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(kinds.get(*i), Some(3 | 4)))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        Ok(Self {
            ids,
            pieces: tokens.to_vec(),
            ranks,
            alphabet: byte_alphabet(),
            specials,
            bos: f.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(128_000),
            eos: f.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(128_009),
        })
    }

    /// How many pieces the vocabulary holds.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether it holds nothing.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Beginning of text.
    pub fn bos(&self) -> u32 {
        self.bos
    }

    /// End of turn.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// The id of a special token spelled exactly `name`.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials.get(name).copied()
    }

    /// Encodes text.
    ///
    /// `parse_special` decides whether `<|eot_id|>` in the input is *the*
    /// end-of-turn token or five ordinary ones. Both readings are wanted and
    /// neither is safe as a silent default: a chat template is assembled with
    /// it on, and user text is tokenized with it off, or a user who types
    /// `<|eot_id|>` ends the model's turn from inside the prompt. llama.cpp
    /// draws the same line and calls it `parse_special`.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if !parse_special {
            self.encode_plain(text, &mut out);
            return out;
        }
        let mut rest = text;
        while !rest.is_empty() {
            let Some(at) = rest.find("<|") else {
                self.encode_plain(rest, &mut out);
                break;
            };
            let (plain, tail) = rest.split_at(at);
            if let Some((len, id)) = self.match_special(tail) {
                self.encode_plain(plain, &mut out);
                out.push(id);
                rest = &tail[len..];
                continue;
            }
            // A bare `<|` beginning no known token: consume through it so the
            // scan cannot stall, and let it tokenize as ordinary text.
            let cut = at + "<|".len();
            self.encode_plain(&rest[..cut], &mut out);
            rest = &rest[cut..];
        }
        out
    }

    /// The longest special spelled at the start of `s`, and its byte length.
    fn match_special(&self, s: &str) -> Option<(usize, u32)> {
        let end = s.find("|>")? + "|>".len();
        self.specials.get(&s[..end]).map(|&id| (end, id))
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pre_tokenize(text) {
            self.bpe(piece.as_bytes(), out);
        }
    }

    /// Merges one pre-token to completion, lowest rank first.
    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        let mut parts: Vec<String> = bytes
            .iter()
            .map(|&b| self.alphabet[usize::from(b)].to_string())
            .collect();

        while parts.len() > 1 {
            // The lowest-ranked adjacent pair, and the leftmost of those - the
            // reference breaks ties by position, and a tie broken the other
            // way gives a different tokenization of the same text.
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                let key = (parts[i].clone(), parts[i + 1].clone());
                if let Some(&r) = self.ranks.get(&key)
                    && best.is_none_or(|(br, _)| r < br)
                {
                    best = Some((r, i));
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }

        for p in parts {
            match self.ids.get(&p) {
                Some(&id) => out.push(id),
                // Unreachable for a complete vocabulary: every single-byte
                // stand-in is a piece, so a fully un-merged run still resolves.
                None => tracing::warn!(piece = %p, "no id for a merged piece"),
            }
        }
    }

    /// Decodes ids back to text.
    ///
    /// Lossy, because a partial id sequence *is* partial text: one Han
    /// character is often two or three tokens, and the bytes of the first
    /// alone are not valid UTF-8. A streaming caller wants
    /// [`Bpe::decode_bytes`] instead, so it can hold an incomplete character
    /// back rather than emit a replacement character that the next token
    /// turns into something else.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)).into_owned()
    }

    /// The same, stopping at bytes.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Vec<u8> {
        let reverse: FxHashMap<char, u8> = self
            .alphabet
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let mut bytes = Vec::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if skip_special && self.specials.contains_key(p) {
                continue;
            }
            for c in p.chars() {
                match reverse.get(&c) {
                    Some(&b) => bytes.push(b),
                    // A piece outside the stand-in alphabet cannot be a
                    // byte-level token; emitting it as UTF-8 is the only
                    // reading left.
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        bytes
    }
}
