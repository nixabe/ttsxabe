//! The geometry, read from the checkpoint rather than assumed.

use crate::LlamaError;
use serde::Deserialize;
use std::path::Path;

/// `config.json`, restricted to the fields that change what the model does.
///
/// The dropped fields are training knobs (`initializer_range`, the dropouts),
/// defaults this checkpoint does not vary (`attention_bias` is false,
/// `pretraining_tp` is 1), or cache policy the engine decides for itself.
/// Parsing them and ignoring them would be the same as not parsing them, with
/// a promise attached.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// What this checkpoint says it is.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Width of every residual stream.
    pub hidden_size: usize,
    /// Inner width of the feed-forward blocks.
    pub intermediate_size: usize,
    /// Transformer blocks.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key and value heads. Equal to the query heads on this checkpoint.
    pub num_key_value_heads: usize,
    /// Rows in the embedding.
    pub vocab_size: usize,
    /// Longest sequence RoPE was trained over.
    pub max_position_embeddings: usize,
    /// Epsilon inside the RMS normalisation.
    pub rms_norm_eps: f32,
    /// RoPE's base frequency.
    pub rope_theta: f32,
    /// Whether the output projection is the embedding.
    ///
    /// False here, which is why `lm_head.weight` is a tensor of its own and
    /// the parameter count carries the embedding twice.
    pub tie_word_embeddings: bool,
    /// Beginning of sequence.
    pub bos_token_id: u32,
    /// End of sequence.
    pub eos_token_id: u32,
}

impl LlamaConfig {
    /// Reads `config.json` from a checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self, LlamaError> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|source| LlamaError::Io {
            path: path.clone(),
            source,
        })?;
        let cfg: Self =
            serde_json::from_str(&text).map_err(|source| LlamaError::Config { path, source })?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Rejects a geometry no amount of correct arithmetic could survive.
    pub fn check(&self) -> Result<(), LlamaError> {
        const WANTED: &str = "LlamaForCausalLM";
        if let Some(arch) = self.architectures.first()
            && arch != WANTED
        {
            return Err(LlamaError::Architecture {
                found: arch.clone(),
                wanted: WANTED,
            });
        }
        if self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(LlamaError::RaggedHeads {
                hidden: self.hidden_size,
                heads: self.num_attention_heads,
            });
        }
        if self.num_key_value_heads != self.num_attention_heads {
            return Err(LlamaError::GroupedQuery {
                kv: self.num_key_value_heads,
                q: self.num_attention_heads,
            });
        }
        Ok(())
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
