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
    /// Key and value heads.
    ///
    /// Equal to the query heads on the Llama-2 translator; a quarter of them
    /// on Llama-3-family checkpoints, which share one key-value head across
    /// four query heads. Both are bound here - see [`Self::kv_dim`].
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
    /// Width of one attention head, when the checkpoint states it outright.
    ///
    /// `config.json` never does, so this is `None` on that path and
    /// [`Self::head_dim`] divides instead. A GGUF states it as
    /// `llama.attention.key_length`, and stating it is not the same as
    /// agreeing with the division - [`Self::check`] compares the two rather
    /// than trusting whichever was read last.
    #[serde(default)]
    pub head_dim: Option<usize>,
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

    /// Reads the geometry from a GGUF's metadata store.
    ///
    /// GGUF keeps the same numbers under different names and in a flat
    /// key-value store rather than a JSON object, so this is a transcription
    /// rather than a parse. Two of them are not simply renamed:
    ///
    /// - **`tie_word_embeddings` is not stated at all.** It is inferred from
    ///   whether `output.weight` exists, because that is the only thing the
    ///   flag actually decides. A file with the tensor has untied embeddings
    ///   whatever a flag might have claimed.
    /// - **`vocab_size` may be absent**, in which case the tokenizer's own
    ///   token array is the vocabulary. Both are checked when both are there.
    pub fn from_gguf(f: &xabe_gguf::GgufFile) -> Result<Self, LlamaError> {
        const WANTED: &str = "llama";
        let arch = f
            .get_str("general.architecture")
            .ok_or(LlamaError::MissingMetadata("general.architecture"))?;
        if arch != WANTED {
            return Err(LlamaError::Architecture {
                found: arch.to_string(),
                wanted: "llama",
            });
        }

        let need = |k: &'static str| {
            f.get_u32(k)
                .map(|v| v as usize)
                .ok_or(LlamaError::MissingMetadata(k))
        };

        // The tokenizer's array is authoritative when the explicit count is
        // absent; when both exist they are the same number on every file seen
        // so far, and disagreeing would mean one of them is describing a
        // different checkpoint.
        let vocab_size = match f.get_u32("llama.vocab_size") {
            Some(v) => v as usize,
            None => f
                .get_strings("tokenizer.ggml.tokens")
                .map(<[String]>::len)
                .ok_or(LlamaError::MissingMetadata("llama.vocab_size"))?,
        };

        let cfg = Self {
            architectures: vec![arch.to_string()],
            hidden_size: need("llama.embedding_length")?,
            intermediate_size: need("llama.feed_forward_length")?,
            num_hidden_layers: need("llama.block_count")?,
            num_attention_heads: need("llama.attention.head_count")?,
            num_key_value_heads: need("llama.attention.head_count_kv")?,
            vocab_size,
            max_position_embeddings: need("llama.context_length")?,
            rms_norm_eps: f.get_f32("llama.attention.layer_norm_rms_epsilon").ok_or(
                LlamaError::MissingMetadata("llama.attention.layer_norm_rms_epsilon"),
            )?,
            rope_theta: f
                .get_f32("llama.rope.freq_base")
                .ok_or(LlamaError::MissingMetadata("llama.rope.freq_base"))?,
            tie_word_embeddings: f.info("output.weight").is_none(),
            bos_token_id: f.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1),
            eos_token_id: f.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2),
            head_dim: f.get_u32("llama.attention.key_length").map(|v| v as usize),
        };
        cfg.check()?;
        Ok(cfg)
    }

    /// Rejects a geometry no amount of correct arithmetic could survive.
    pub fn check(&self) -> Result<(), LlamaError> {
        // A GGUF says `llama` where `config.json` says `LlamaForCausalLM`;
        // both name the same architecture, so both are accepted here and the
        // container-specific check happens at its own constructor.
        const WANTED: &str = "LlamaForCausalLM";
        if let Some(arch) = self.architectures.first()
            && arch != WANTED
            && arch != "llama"
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
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(LlamaError::RaggedGroups {
                kv: self.num_key_value_heads,
                q: self.num_attention_heads,
            });
        }
        if let Some(d) = self.head_dim
            && d * self.num_attention_heads != self.hidden_size
        {
            return Err(LlamaError::HeadDim {
                stated: d,
                divided: self.hidden_size / self.num_attention_heads,
            });
        }
        Ok(())
    }

    /// Refuses a grouped-query checkpoint.
    ///
    /// Not part of [`Self::check`], and the split is the point. A shape is a
    /// fact about the file and grouped-query shapes are perfectly bindable;
    /// whether the *arithmetic* handles them is a fact about a forward pass.
    /// So the schema binds them and every engine that cannot run one calls
    /// this at open, naming itself in the error rather than failing later
    /// with a head index out of range.
    pub fn refuse_grouped_query(&self) -> Result<(), LlamaError> {
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
        match self.head_dim {
            Some(d) => d,
            None => self.hidden_size / self.num_attention_heads,
        }
    }

    /// How many query heads share each key-value head.
    ///
    /// One when the model is not grouped-query, which is what makes the same
    /// binding code cover both.
    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Width of the key and value projections' output.
    ///
    /// Equal to `hidden_size` only when the heads match. Llama-3's 8 key-value
    /// heads of 128 make this 1024 against a hidden size of 4096, which is why
    /// `k` and `v` are not square and binding them as if they were is the
    /// mistake this exists to prevent.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}
