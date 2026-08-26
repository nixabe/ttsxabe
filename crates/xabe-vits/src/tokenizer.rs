//! Text to symbol ids, matching 🤗 `VitsTokenizer` exactly.
//!
//! The vocabulary is 48 single code points of Pe̍h-ōe-jī - see
//! [`MODEL.md`](../../../docs/MODEL.md) for why POJ and not Tâi-lô. There is no
//! subword model, no phonemiser and no romaniser on this path: the whole
//! tokeniser is *lower-case, drop what is not in the vocabulary, intersperse a
//! blank*. That simplicity is deceptive, because two of the three steps have
//! consequences that are invisible in the output.
//!
//! # Dropping is silent, and that is the whole difficulty
//!
//! Filtering removes anything outside the vocabulary without a warning, an
//! unknown token, or a change in length that anyone would notice. So the
//! difference between correct and subtly wrong input is not an error, it is a
//! slightly different sentence being spoken:
//!
//! - **Normalisation form matters.** The vocabulary holds precomposed `í`
//!   (U+00ED) but not combining acute (U+0301), so NFD input loses every tone
//!   mark: `lí` decomposed tokenises as `li`. The tones are the words in this
//!   language, so this is a mistranslation, not a blemish. Meanwhile U+030D
//!   (the entering tone) and U+0358 (the o-dot) have no precomposed form and
//!   *are* in the vocabulary, so the required input is NFC - which leaves those
//!   two combining, and the rest composed.
//! - **Punctuation is deleted, not honoured.** A comma is not a pause here. It
//!   simply ceases to exist, and the phrasing it would have implied is gone.
//! - **`<unk>` is unreachable.** It is an added token, so the normaliser
//!   preserves the literal five characters `<unk>` - and then the filter drops
//!   `<` and `>` and keeps `unk`, which tokenises as three ordinary letters. No
//!   input can produce the unknown id, which is just as well: it is 48, one
//!   past the end of the embedding table.

use crate::error::TokenizerError;
use rustc_hash::FxHashMap;
use std::path::Path;

/// The reference's tokeniser, loaded from a model directory.
#[derive(Debug)]
pub struct Tokenizer {
    /// Every vocabulary entry that is a single code point, which for this model
    /// is all 48 of them.
    chars: FxHashMap<char, i64>,
    /// Vocabulary entries longer than one code point, plus the added tokens,
    /// checked in the order the reference checks them.
    multi: Vec<(String, i64)>,
    /// Token id inserted between every pair of symbols, and at both ends. The
    /// reference defines it as "whatever token has id 0", which is a longer way
    /// of writing zero - but the vocabulary is checked for one at load time so
    /// that a checkpoint without it fails loudly rather than emitting a blank
    /// the embedding table does not have.
    blank: i64,
    /// Whether to insert [`Self::blank`] at all.
    add_blank: bool,
    /// Whether to lower-case and filter. False means the caller has already
    /// produced symbols, which no path in this project does.
    normalize: bool,
    /// The model's language tag, which selects one character substitution.
    language: String,
    /// Size of the vocabulary proper, excluding added tokens.
    vocab_size: usize,
}

impl Tokenizer {
    /// Loads `vocab.json` and `tokenizer_config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, TokenizerError> {
        let vocab: serde_json::Map<String, serde_json::Value> = read_json(&dir.join("vocab.json"))?;
        let config: serde_json::Map<String, serde_json::Value> =
            read_json(&dir.join("tokenizer_config.json"))?;

        let mut chars = FxHashMap::default();
        let mut multi = Vec::new();
        for (token, id) in &vocab {
            let id = id.as_i64().ok_or_else(|| TokenizerError::BadVocabEntry {
                token: token.clone(),
            })?;
            let mut cs = token.chars();
            match (cs.next(), cs.next()) {
                (Some(c), None) => {
                    chars.insert(c, id);
                }
                _ => multi.push((token.clone(), id)),
            }
        }

        // The unknown token is an *added* token: it is not in the vocabulary,
        // and its id is one past the end of it. It is included here only
        // because the reference includes it in the strings it tries to match
        // during normalisation - it can never actually be emitted.
        if let Some(unk) = config.get("unk_token").and_then(|v| v.as_str())
            && !vocab.contains_key(unk)
        {
            multi.push((unk.to_string(), vocab.len() as i64));
        }

        if !chars.values().any(|&id| id == 0) {
            return Err(TokenizerError::NoBlank);
        }
        let blank = 0;

        let flag = |name: &str, default: bool| {
            config
                .get(name)
                .and_then(|v| v.as_bool())
                .unwrap_or(default)
        };

        // Neither is supported, and neither is set for this model. Failing here
        // is better than silently tokenising unphonemised text, which would
        // produce fluent-sounding nonsense.
        if flag("phonemize", true) {
            return Err(TokenizerError::Unsupported { what: "phonemize" });
        }
        if flag("is_uroman", false) {
            return Err(TokenizerError::Unsupported { what: "is_uroman" });
        }

        let tok = Self {
            vocab_size: vocab.len(),
            chars,
            multi,
            blank,
            add_blank: flag("add_blank", true),
            normalize: flag("normalize", true),
            language: config
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        };
        tracing::debug!(
            vocab = tok.chars.len(),
            multi = tok.multi.len(),
            add_blank = tok.add_blank,
            language = %tok.language,
            "loaded tokenizer",
        );
        Ok(tok)
    }

    /// Size of the vocabulary proper. Added tokens are not counted, and the
    /// model's embedding table is exactly this wide.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Lower-cases the input, leaving anything that is already a vocabulary
    /// entry alone.
    ///
    /// The reference walks the string one position at a time and, at each,
    /// tries every vocabulary entry as a literal prefix before falling back to
    /// lower-casing a single character. For this model every vocabulary entry
    /// is one code point, so the prefix search only ever matters for the added
    /// `<unk>`; the loop is written the reference's way anyway, because a
    /// different MMS checkpoint may not be so simple.
    fn normalize_text(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut rest = input;
        while !rest.is_empty() {
            let c = rest.chars().next().expect("rest is non-empty");
            if self.chars.contains_key(&c) {
                out.push(c);
                rest = &rest[c.len_utf8()..];
                continue;
            }
            if let Some((token, _)) = self
                .multi
                .iter()
                .find(|(t, _)| rest.starts_with(t.as_str()))
            {
                out.push_str(token);
                rest = &rest[token.len()..];
                continue;
            }
            // `to_lowercase` can yield more than one character - 'İ' becomes
            // 'i' plus a combining dot - and the reference, using Python's
            // str.lower(), has the same behaviour.
            out.extend(c.to_lowercase());
            rest = &rest[c.len_utf8()..];
        }
        out
    }

    /// Encodes text into the symbol ids the model's embedding table expects.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        let normalized = if self.normalize {
            self.normalize_text(text)
        } else {
            text.to_string()
        };

        // The reference's only language-specific substitution.
        let normalized = if self.language == "ron" {
            normalized.replace('ț', "ţ")
        } else {
            normalized
        };

        let symbols: Vec<char> = if self.normalize {
            // Filter, then strip. Space is itself a vocabulary entry, so the
            // strip is what removes leading and trailing space rather than the
            // filter - and interior runs of space are left alone.
            let kept: String = normalized
                .chars()
                .filter(|c| self.chars.contains_key(c))
                .collect();
            kept.trim().chars().collect()
        } else {
            normalized.chars().collect()
        };

        let mut ids = Vec::with_capacity(symbols.len() * 2 + 1);
        for c in symbols {
            if self.add_blank {
                ids.push(self.blank);
            }
            ids.push(self.chars.get(&c).copied().unwrap_or(self.unk_id()));
        }
        // An empty input produces an empty sequence, not a lone blank: the
        // reference builds a list of length 2n+1 only when n > 0, because
        // `interspersed[1::2] = tokens` on an empty list leaves an empty list.
        if self.add_blank && !ids.is_empty() {
            ids.push(self.blank);
        }
        ids
    }

    /// The unknown id. Unreachable through [`Self::encode`] whenever
    /// normalisation is on, which for this model it is.
    fn unk_id(&self) -> i64 {
        self.multi.last().map_or(0, |(_, id)| *id)
    }
}

/// Reads a JSON object, naming the file when it is missing or malformed.
fn read_json(path: &Path) -> Result<serde_json::Map<String, serde_json::Value>, TokenizerError> {
    let text = std::fs::read_to_string(path).map_err(|source| TokenizerError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|source| TokenizerError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(TokenizerError::NotAnObject {
            path: path.to_path_buf(),
        }),
    }
}
