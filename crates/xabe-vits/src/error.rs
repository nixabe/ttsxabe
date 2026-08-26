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

/// A tokenizer could not be loaded, or is configured in a way this
/// implementation does not support.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// `vocab.json` or `tokenizer_config.json` could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: std::path::PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// One of the two JSON files is malformed.
    #[error("{path} is not valid JSON: {source}")]
    Parse {
        /// The file that could not be parsed.
        path: std::path::PathBuf,
        /// The underlying deserialisation error.
        #[source]
        source: serde_json::Error,
    },

    /// One of the two JSON files is valid JSON but not an object.
    #[error("{path} is valid JSON but not an object")]
    NotAnObject {
        /// The offending file.
        path: std::path::PathBuf,
    },

    /// A vocabulary entry's value is not an integer id.
    #[error("vocabulary entry {token} does not map to an integer id")]
    BadVocabEntry {
        /// The offending token.
        token: String,
    },

    /// No token has id 0. The blank inserted between symbols is defined as
    /// "whatever token id 0 is", so a vocabulary without one cannot be used.
    #[error("no token has id 0, so there is no blank to intersperse")]
    NoBlank,

    /// The tokenizer config asks for a preprocessing step this implementation
    /// does not have. Failing is deliberate: silently skipping phonemisation
    /// would produce fluent-sounding nonsense rather than an error.
    #[error("this tokenizer requires {what}, which is not implemented")]
    Unsupported {
        /// The unsupported setting's name.
        what: &'static str,
    },
}
