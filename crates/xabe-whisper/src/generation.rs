//! `generation_config.json`: how a transcript is decoded, as the checkpoint
//! ships it.

use crate::WhisperError;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// The decoding parameters this checkpoint was published with.
///
/// Only the fields greedy decoding reads. Beam search, the temperature ladder
/// and the timestamp machinery are deliberately absent - see the crate essay
/// for what this engine does not run and why.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    /// Tokens forced to negative infinity at every step.
    ///
    /// Eighty-eight of them here: every control token, so the model cannot
    /// emit `<|zh|>` in the middle of a sentence, plus a list of punctuation
    /// and formatting pieces OpenAI found it hallucinating.
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,

    /// Tokens forced to negative infinity at the *first* generated position
    /// only: a leading space, and an immediate end of transcript.
    #[serde(default)]
    pub begin_suppress_tokens: Vec<u32>,

    /// The longest sequence, prefix included.
    pub max_length: usize,

    /// The token every sequence starts with.
    pub decoder_start_token_id: u32,

    /// End of transcript.
    pub eos_token_id: u32,

    /// The token that says not to emit timestamps.
    pub no_timestamps_token_id: u32,

    /// `<|zh|>` and its ninety-eight siblings, by spelling.
    #[serde(default)]
    pub lang_to_id: BTreeMap<String, u32>,

    /// `transcribe` and `translate`.
    #[serde(default)]
    pub task_to_id: BTreeMap<String, u32>,
}

impl GenerationConfig {
    /// Reads `generation_config.json` from a checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self, WhisperError> {
        let path = dir.join("generation_config.json");
        let text = std::fs::read_to_string(&path).map_err(|source| WhisperError::Io {
            path: path.clone(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| WhisperError::Config { path, source })
    }

    /// The forced prefix for one language and task.
    ///
    /// `<|startoftranscript|>`, the language, the task, `<|notimestamps|>` -
    /// in that order, which is the order the position embedding was trained
    /// with. This engine always asks for no timestamps, because the live
    /// pipeline runs `-nt` and every downstream stage takes plain text.
    pub fn prefix(&self, language: &str, task: &str) -> Result<Vec<u32>, WhisperError> {
        let lang = format!("<|{language}|>");
        let want = |map: &BTreeMap<String, u32>, key: &str, what: &str| {
            map.get(key).copied().ok_or_else(|| WhisperError::Vocab {
                path: "generation_config.json".into(),
                what: format!("no {what} {key:?}"),
            })
        };
        Ok(vec![
            self.decoder_start_token_id,
            want(&self.lang_to_id, &lang, "language")?,
            want(&self.task_to_id, task, "task")?,
            self.no_timestamps_token_id,
        ])
    }
}
