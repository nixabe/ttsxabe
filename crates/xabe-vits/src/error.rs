//! Errors raised while reading a VITS configuration or binding its weights.

use xabe_st::StError;

/// A configuration file could not be read, or describes a model this
/// implementation does not support.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The file could not be opened.
    #[error("cannot read config {path}: {source}")]
    Io {
        /// The config file.
        path: std::path::PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// The file is not the JSON this expects.
    #[error("malformed config JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// A field that must be positive is zero.
    #[error("{field} is zero")]
    Zero {
        /// The offending field.
        field: &'static str,
    },

    /// The hidden size does not divide evenly into attention heads, so the head
    /// dimension would be fractional.
    #[error("hidden_size {hidden} is not divisible by num_attention_heads {heads}")]
    HeadSplit {
        /// Model width.
        hidden: usize,
        /// Head count.
        heads: usize,
    },

    /// The decoder's upsample rates and kernel sizes disagree in length; each
    /// stage needs exactly one of each.
    #[error("{rates} upsample rates but {kernels} kernel sizes")]
    UpsampleMismatch {
        /// Number of rates.
        rates: usize,
        /// Number of kernel sizes.
        kernels: usize,
    },

    /// The resblock kernel sizes and dilation lists disagree in length.
    #[error("{kernels} resblock kernels but {dilations} dilation lists")]
    ResblockMismatch {
        /// Number of kernel sizes.
        kernels: usize,
        /// Number of dilation lists.
        dilations: usize,
    },

    /// The product of the upsample rates does not divide the initial channel
    /// count down to something a final 1-channel convolution can consume.
    #[error("upsample_initial_channel {channels} cannot halve {stages} times")]
    ChannelUnderflow {
        /// Initial channel count.
        channels: usize,
        /// Number of upsample stages.
        stages: usize,
    },

    /// This implementation only handles the stochastic duration predictor,
    /// which is what every MMS checkpoint uses.
    #[error("use_stochastic_duration_prediction=false is not supported")]
    DeterministicDuration,
}

/// A checkpoint could not be bound to the configured geometry.
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    /// A tensor was missing, misshapen, or the container rejected it. The
    /// wrapped [`StError`] names the tensor.
    #[error(transparent)]
    Container(#[from] StError),
}
