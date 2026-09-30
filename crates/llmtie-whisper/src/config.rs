//! The geometry, read from the checkpoint rather than assumed.

use crate::WhisperError;
use serde::Deserialize;
use std::path::Path;

/// `config.json`, restricted to the fields that change what the model does.
///
/// The dropped fields are not oversights: dropout and layerdrop are training
/// knobs that are zero at inference, `apply_spec_augment` and the `mask_*`
/// family are augmentation, and `classifier_proj_size` belongs to an audio
/// classification head this engine does not build. Parsing them and ignoring
/// them would be the same as not parsing them, with an extra promise attached.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperConfig {
    /// Width of every residual stream.
    pub d_model: usize,
    /// Encoder blocks.
    pub encoder_layers: usize,
    /// Decoder blocks.
    pub decoder_layers: usize,
    /// Attention heads in the encoder.
    pub encoder_attention_heads: usize,
    /// Attention heads in the decoder, including cross-attention.
    pub decoder_attention_heads: usize,
    /// Inner width of the encoder's feed-forward blocks.
    pub encoder_ffn_dim: usize,
    /// Inner width of the decoder's feed-forward blocks.
    pub decoder_ffn_dim: usize,
    /// Mel bins the frontend must produce.
    pub num_mel_bins: usize,
    /// Rows in the embedding, which is also the width of the logits.
    pub vocab_size: usize,
    /// Encoder positions, after the stride-2 convolution halves the frames.
    pub max_source_positions: usize,
    /// Longest sequence the decoder's learned positions cover.
    pub max_target_positions: usize,
    /// The token every sequence starts with.
    pub decoder_start_token_id: u32,
    /// End of transcript.
    pub eos_token_id: u32,
    /// Padding, which this checkpoint shares with `eos_token_id`.
    pub pad_token_id: u32,
}

impl WhisperConfig {
    /// Reads `config.json` from a checkpoint directory.
    pub fn from_dir(dir: &Path) -> Result<Self, WhisperError> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|source| WhisperError::Io {
            path: path.clone(),
            source,
        })?;
        let cfg: Self =
            serde_json::from_str(&text).map_err(|source| WhisperError::Config { path, source })?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Rejects a geometry no amount of correct arithmetic could survive.
    ///
    /// Every one of these is checked because it is *silent* otherwise: a
    /// `d_model` that does not divide by the head count produces a ragged
    /// reshape that some frameworks round and this one would index past.
    pub fn check(&self) -> Result<(), WhisperError> {
        for (what, d, h) in [
            ("encoder", self.d_model, self.encoder_attention_heads),
            ("decoder", self.d_model, self.decoder_attention_heads),
        ] {
            if h == 0 || d % h != 0 {
                return Err(WhisperError::RaggedHeads {
                    what,
                    d_model: d,
                    heads: h,
                });
            }
        }
        Ok(())
    }

    /// Width of one attention head in the encoder.
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Width of one attention head in the decoder.
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Mel frames the encoder consumes: twice its positions, because `conv2`
    /// has stride 2.
    pub fn n_frames(&self) -> usize {
        self.max_source_positions * 2
    }

    /// Samples in one window, at the frontend's hop of 160.
    pub fn n_samples(&self) -> usize {
        self.n_frames() * 160
    }
}
