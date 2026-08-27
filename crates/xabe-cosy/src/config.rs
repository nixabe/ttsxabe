//! CosyVoice3's geometry, transcribed and then checked against the files.
//!
//! Every number here is read back out of the checkpoint at bind time rather
//! than trusted. That is the same discipline `xabe-vits` and `xabe-llama` keep,
//! and it matters more here than usual: this model has no `config.json` of its
//! own. Its shape lives in a `cosyvoice3.yaml` that is a *`hyperpyyaml`
//! program* - it constructs Python objects, so it cannot be parsed as data
//! without running it, and running it needs the CosyVoice package and its
//! pinned torch.
//!
//! So the constants are transcribed from that yaml and the tensors are what
//! confirm them. A transcription mistake becomes a shape error naming the
//! tensor, not a model that runs and sounds wrong.

use crate::CosyError;

/// The speech language model: a Qwen2 0.5 B backbone with a speech head.
///
/// Text in, speech tokens out, at 25 Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmConfig {
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Decoder blocks.
    pub num_hidden_layers: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key-value heads. Seven query heads share each one.
    pub num_key_value_heads: usize,
    /// Feed-forward width.
    pub intermediate_size: usize,
    /// Text vocabulary, which is Qwen2's own.
    pub vocab_size: usize,
    /// Speech tokens plus their markers.
    pub speech_vocab_size: usize,
    /// The FSQ codebook proper, without the markers.
    pub speech_token_size: usize,
    /// Longest position rope was trained over.
    pub max_position_embeddings: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        // Qwen2-0.5B's `config.json`, plus the two speech numbers from
        // `cosyvoice3.yaml`. `speech_vocab_size` is 6761 and not the 6564 a
        // reading of upstream CosyVoice2 would suggest: CosyVoice3 carries
        // 200 markers past the codebook, not 3.
        Self {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            vocab_size: 151_936,
            speech_vocab_size: 6761,
            speech_token_size: 6561,
            max_position_embeddings: 32_768,
        }
    }
}

impl LlmConfig {
    /// Qwen2's rope base, which is not Llama-2's and not Llama-3's either.
    pub const ROPE_THETA: f32 = 1_000_000.0;

    /// Qwen2's RMSNorm epsilon.
    pub const RMS_EPS: f32 = 1e-6;

    /// Start of sequence, as an index into the speech embedding.
    ///
    /// It is `speech_token_size` itself: the markers sit immediately past the
    /// codebook, so the first index that is not a real speech token is the
    /// first marker. Also the token that ends generation.
    pub const SOS: u32 = 6561;

    /// The marker that separates the text prompt from the speech to come.
    pub const TASK_ID: u32 = 6563;

    /// `<|endofprompt|>` in Qwen2's *text* vocabulary.
    ///
    /// CosyVoice3 asserts this appears somewhere in the prompt, because the
    /// instruct string is what carries it. A prompt without it is one the
    /// model was never trained to answer, and upstream refuses rather than
    /// producing something.
    pub const ENDOFPROMPT: u32 = 151_646;

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Width of the key and value projections together with their heads.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    /// How many query heads share one key-value head.
    pub fn group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Refuses a geometry whose divisions do not come out whole.
    pub fn check(&self) -> Result<(), CosyError> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(CosyError::Geometry {
                what: "hidden_size does not divide by the query heads",
                got: self.hidden_size,
                want: self.num_attention_heads,
            });
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(CosyError::Geometry {
                what: "the query heads do not divide by the key-value heads",
                got: self.num_attention_heads,
                want: self.num_key_value_heads,
            });
        }
        if self.speech_token_size >= self.speech_vocab_size {
            return Err(CosyError::Geometry {
                what: "the codebook is not smaller than the vocabulary it sits in",
                got: self.speech_token_size,
                want: self.speech_vocab_size,
            });
        }
        Ok(())
    }
}

/// `ras_sampling`, transcribed from `cosyvoice3.yaml`.
///
/// The knobs are here rather than in the sampler so that the geometry module
/// stays the one place a number from that yaml is written down.
#[derive(Debug, Clone, Copy)]
pub struct RasConfig {
    /// Nucleus mass.
    pub top_p: f32,
    /// How many candidates the nucleus may hold, regardless of mass.
    pub top_k: usize,
    /// How far back the repetition check looks.
    pub win_size: usize,
    /// The share of that window a token may occupy before it is rejected.
    pub tau_r: f32,
    /// The PRNG seed.
    pub seed: u64,
}

impl Default for RasConfig {
    fn default() -> Self {
        Self {
            top_p: 0.8,
            top_k: 25,
            win_size: 10,
            tau_r: 0.1,
            seed: 1986,
        }
    }
}
