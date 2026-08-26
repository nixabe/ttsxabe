//! SentencePiece BPE, matching 🤗 `LlamaTokenizer`.
//!
//! `tokenizer.model` is a SentencePiece protobuf. Reading it needs about sixty
//! lines of wire-format decoding, which is why there is no `prost` or
//! `sentencepiece` in this workspace: the message shape is four fields and the
//! alternative is a dependency that pulls a code generator behind it.
//!
//! # How this differs from the ASR's tokenizer
//!
//! Both are BPE and they agree on nothing else. Whisper's works on *bytes*
//! through a printable alphabet and merges by a rank read from `merges.txt`;
//! this one works on *characters*, merges by a **score** stored with each
//! piece, escapes spaces as U+2581, and falls back to per-byte pieces only for
//! characters the vocabulary has never seen. Writing the second against
//! memories of the first is the way to get something that looks right on
//! English and mangles Han.

use crate::LlamaError;
use rustc_hash::FxHashMap;
use std::path::Path;

/// U+2581 LOWER ONE EIGHTH BLOCK, which SentencePiece uses for a space.
pub const UNDERLINE: char = '\u{2581}';

/// What a piece is for.
///
/// The numbers are SentencePiece's own, from `ModelProto.SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// An ordinary subword.
    Normal,
    /// The unknown token.
    Unknown,
    /// A control token: `<s>`, `</s>`.
    Control,
    /// Added by hand at training time.
    UserDefined,
    /// A hole in the vocabulary.
    Unused,
    /// One of the 256 `<0xNN>` fallbacks.
    Byte,
}

impl Kind {
    /// The wire value, or [`Kind::Normal`] for anything unrecognised.
    fn from_wire(v: u64) -> Self {
        match v {
            2 => Kind::Unknown,
            3 => Kind::Control,
            4 => Kind::UserDefined,
            5 => Kind::Unused,
            6 => Kind::Byte,
            _ => Kind::Normal,
        }
    }
}

/// One entry of the vocabulary.
#[derive(Debug, Clone)]
pub struct Piece {
    /// Its spelling, with spaces already escaped as U+2581.
    pub text: String,
    /// Merge priority: higher merges first. SentencePiece stores these as
    /// negative ranks, so `-1.0` is the first merge learned.
    pub score: f32,
    /// What it is for.
    pub kind: Kind,
}

/// A minimal protobuf reader: enough for `ModelProto`, and no more.
struct Wire<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Wire<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }

    /// A base-128 varint, little-endian, seven bits at a time.
    fn varint(&mut self) -> Option<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let b = *self.bytes.get(self.at)?;
            self.at += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
            if shift > 63 {
                return None;
            }
        }
    }

    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let out = self.bytes.get(self.at..self.at + n)?;
        self.at += n;
        Some(out)
    }

    /// Reads one field's key and its payload, skipping what it does not know.
    ///
    /// Returns `(field_number, payload)`, where a varint or fixed-width field
    /// is handed back as its raw bytes so the caller decides how to read it.
    fn field(&mut self) -> Option<(u64, Payload<'a>)> {
        let key = self.varint()?;
        let (number, wire) = (key >> 3, key & 7);
        let payload = match wire {
            0 => Payload::Varint(self.varint()?),
            1 => Payload::Fixed(self.bytes(8)?),
            2 => {
                let n = self.varint()? as usize;
                Payload::Delimited(self.bytes(n)?)
            }
            5 => Payload::Fixed(self.bytes(4)?),
            // Groups, which no modern encoder emits and this one will not meet.
            _ => return None,
        };
        Some((number, payload))
    }
}

/// One field's contents, still in wire form.
enum Payload<'a> {
    Varint(u64),
    Fixed(&'a [u8]),
    Delimited(&'a [u8]),
}

/// Parses `ModelProto.pieces`, ignoring every other field.
///
/// The trainer and normaliser specs are read and discarded on purpose: this
/// implementation is written for one normalisation - identity, escaped
/// whitespace, dummy prefix - and a checkpoint that wanted a different one
/// would need code, not a flag.
fn parse_pieces(bytes: &[u8]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    let mut w = Wire::new(bytes);
    while !w.done() {
        let (number, payload) = w.field()?;
        let Payload::Delimited(body) = payload else {
            continue;
        };
        if number != 1 {
            continue;
        }
        let (mut text, mut score, mut kind) = (None, 0.0f32, Kind::Normal);
        let mut inner = Wire::new(body);
        while !inner.done() {
            match inner.field()? {
                (1, Payload::Delimited(b)) => text = Some(std::str::from_utf8(b).ok()?.to_string()),
                (2, Payload::Fixed(b)) if b.len() == 4 => {
                    score = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                (3, Payload::Varint(v)) => kind = Kind::from_wire(v),
                _ => {}
            }
        }
        out.push(Piece {
            text: text?,
            score,
            kind,
        });
    }
    Some(out)
}

/// The tokenizer: a scored vocabulary, and the special tokens around it.
#[derive(Debug)]
pub struct Tokenizer {
    pieces: Vec<Piece>,
    by_text: FxHashMap<String, u32>,
    /// `<0x00>` through `<0xFF>`, by byte value.
    byte_ids: Vec<u32>,
    /// Spelling to id for the tokens that are matched before the text is.
    specials: FxHashMap<String, u32>,
    bos: u32,
    eos: u32,
    unk: u32,
}

impl Tokenizer {
    /// Reads `tokenizer.model` and the special-token declarations beside it.
    pub fn from_dir(dir: &Path) -> Result<Self, LlamaError> {
        let path = dir.join("tokenizer.model");
        let raw = std::fs::read(&path).map_err(|source| LlamaError::Io {
            path: path.clone(),
            source,
        })?;
        let pieces = parse_pieces(&raw).ok_or_else(|| LlamaError::Vocab {
            path: path.clone(),
            what: "not a SentencePiece ModelProto".to_string(),
        })?;
        if pieces.is_empty() {
            return Err(LlamaError::Vocab {
                path,
                what: "no pieces".to_string(),
            });
        }

        let mut by_text = FxHashMap::default();
        by_text.reserve(pieces.len());
        let mut byte_ids = vec![u32::MAX; 256];
        for (id, p) in pieces.iter().enumerate() {
            let id = id as u32;
            // First writer wins. A vocabulary with a duplicate spelling is
            // malformed, and taking the lower id matches what SentencePiece
            // does with one.
            by_text.entry(p.text.clone()).or_insert(id);
            if p.kind == Kind::Byte
                && let Some(b) = parse_byte_piece(&p.text)
            {
                byte_ids[usize::from(b)] = id;
            }
        }

        // Control and unknown pieces are special by their own declaration.
        // `<pad>` is not among them - it is a NORMAL piece that the checkpoint
        // promotes in `special_tokens_map.json` - which is exactly the trap
        // `<|endoftext|>` sets in the ASR's tokenizer, in the other direction.
        let mut specials: FxHashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.kind, Kind::Control | Kind::Unknown))
            .map(|(id, p)| (p.text.clone(), id as u32))
            .collect();
        for name in declared_specials(dir)? {
            if let Some(&id) = by_text.get(&name) {
                specials.insert(name, id);
            }
        }

        let id_of = |name: &str| by_text.get(name).copied();
        Ok(Self {
            bos: id_of("<s>").unwrap_or(1),
            eos: id_of("</s>").unwrap_or(2),
            unk: id_of("<unk>").unwrap_or(0),
            pieces,
            by_text,
            byte_ids,
            specials,
        })
    }

    /// How many pieces the vocabulary holds.
    ///
    /// Not the same as `config.json`'s `vocab_size`: the embedding is padded
    /// to a round number and the last rows are unused. 56,020 against 56,024
    /// here, and a loader that conflates them binds `lm_head` at the wrong
    /// shape.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the vocabulary is empty, which [`Tokenizer::from_dir`] refuses.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Beginning of sequence.
    pub fn bos(&self) -> u32 {
        self.bos
    }

    /// End of sequence.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// The unknown token.
    pub fn unk(&self) -> u32 {
        self.unk
    }

    /// One piece, by id.
    pub fn piece(&self, id: u32) -> Option<&Piece> {
        self.pieces.get(id as usize)
    }

    /// The id of a special token, by spelling.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.specials.get(name).copied()
    }

    /// Whether an id is one of the tokens that are matched before text is.
    pub fn is_special(&self, id: u32) -> bool {
        self.piece(id)
            .is_some_and(|p| self.specials.contains_key(&p.text))
    }

    /// Encodes text. No `<s>` is added; that is the caller's decision.
    ///
    /// # The dummy prefix, which is where this goes wrong
    ///
    /// SentencePiece escapes spaces as U+2581 and prepends one, so that a word
    /// at the start of a string tokenizes the same as one in the middle. The
    /// reference adds it only when the text does not *already* begin with a
    /// space - so `" "` is one underline and `"hello"` is `"▁hello"` - and
    /// only to the first segment, so the text after a special token starts
    /// bare. Getting that wrong shifts every token in the sentence, which
    /// looks like a merge-table problem and is not.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        let mut first = true;
        while !rest.is_empty() {
            match self.next_special(rest) {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], first && at > 0, &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, first, &mut out);
                    break;
                }
            }
            first = false;
        }
        out
    }

    /// The earliest special token in `s`, as `(offset, byte length, id)`.
    ///
    /// Longest match at the earliest position, so `</s>` is not read as `<` in
    /// a vocabulary that also has `<`.
    fn next_special(&self, s: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (name, &id) in &self.specials {
            if let Some(at) = s.find(name.as_str()) {
                let cand = (at, name.len(), id);
                if best.is_none_or(|b| {
                    (cand.0, std::cmp::Reverse(cand.1)) < (b.0, std::cmp::Reverse(b.1))
                }) {
                    best = Some(cand);
                }
            }
        }
        best
    }

    /// Encodes a run of text with no special tokens in it.
    fn encode_plain(&self, text: &str, prefix: bool, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut escaped = String::with_capacity(text.len() + 3);
        if prefix && !text.starts_with(' ') {
            escaped.push(UNDERLINE);
        }
        for c in text.chars() {
            escaped.push(if c == ' ' { UNDERLINE } else { c });
        }
        self.merge(&escaped, out);
    }

    /// The BPE merge loop, over characters.
    ///
    /// Repeatedly joins the adjacent pair whose concatenation scores highest,
    /// leftmost first on a tie. SentencePiece's scores are negative ranks, so
    /// "highest score" is "learned earliest" - the same rule as a rank-ordered
    /// merge table, spelled the other way round.
    fn merge(&self, escaped: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = escaped.chars().map(String::from).collect();
        loop {
            let mut best: Option<(usize, f32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let joined = format!("{}{}", parts[i], parts[i + 1]);
                let Some(&id) = self.by_text.get(&joined) else {
                    continue;
                };
                let score = self.pieces[id as usize].score;
                if best.is_none_or(|(_, b)| score > b) {
                    best = Some((i, score));
                }
            }
            let Some((at, _)) = best else { break };
            let joined = format!("{}{}", parts[at], parts[at + 1]);
            parts.splice(at..at + 2, [joined]);
        }

        for p in parts {
            match self.by_text.get(&p) {
                Some(&id) => out.push(id),
                // Byte fallback: a character the vocabulary has never seen
                // becomes its UTF-8 bytes, each of which always has a piece.
                // That is what makes this tokenizer total - there is no input
                // it cannot represent.
                None => out.extend(p.bytes().map(|b| self.byte_ids[usize::from(b)])),
            }
        }
    }

    /// Decodes ids back to text.
    ///
    /// The bytes are joined before they are read as UTF-8, because byte
    /// fallback splits a character across several ids by construction.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(p) = self.piece(id) else { continue };
            if skip_special && self.is_special(id) {
                continue;
            }
            match p.kind {
                Kind::Byte => match parse_byte_piece(&p.text) {
                    Some(b) => bytes.push(b),
                    None => bytes.extend_from_slice(p.text.as_bytes()),
                },
                _ => {
                    for c in p.text.chars() {
                        let c = if c == UNDERLINE { ' ' } else { c };
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        // One leading space, and only one: it is the dummy prefix coming back
        // off. A text that really began with two spaces keeps one of them,
        // which is why this strips rather than trims.
        text.strip_prefix(' ').map_or(text.clone(), str::to_string)
    }
}

/// The byte a `<0xNN>` piece stands for.
fn parse_byte_piece(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// Every token `special_tokens_map.json` declares, by spelling.
fn declared_specials(dir: &Path) -> Result<Vec<String>, LlamaError> {
    let path = dir.join("special_tokens_map.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        // Optional: the SentencePiece model already declares its control and
        // unknown pieces, and a checkpoint without this file is readable.
        return Ok(Vec::new());
    };
    let map: serde_json::Value =
        serde_json::from_str(&text).map_err(|source| LlamaError::Config { path, source })?;
    let mut out = Vec::new();
    let Some(obj) = map.as_object() else {
        return Ok(out);
    };
    for value in obj.values() {
        if let Some(s) = value.as_str() {
            out.push(s.to_string());
        } else if let Some(s) = value.get("content").and_then(serde_json::Value::as_str) {
            out.push(s.to_string());
        }
    }
    Ok(out)
}
