//! Errors raised while reading a VITS configuration or binding its weights.

use xabe_pt::PtError;
use xabe_st::StError;

/// A configuration file could not be read, or describes a model this
/// implementation does not support.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The feed-forward activation is not one this implementation has. Checked
    /// rather than assumed: a different activation changes every output while
    /// breaking no shape.
    #[error("hidden_act is {act}, but only relu is implemented")]
    UnsupportedActivation {
        /// The activation the config asked for.
        act: String,
    },

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

    /// A Coqui config describes a run that trained something other than VITS.
    #[error("this config trained {model}, not vits")]
    UnsupportedModel {
        /// The model the run named.
        model: String,
    },

    /// The decoder was built from `ResBlock2`, which has two convolutions per
    /// dilation where `ResBlock1` has four. Checked rather than assumed: the
    /// two share every channel count and differ only in how many tensors there
    /// are, so a schema written for one binds a prefix of the other.
    #[error("resblock_type_decoder is {kind}, but only type 1 is implemented")]
    UnsupportedResblock {
        /// The type the config named.
        kind: String,
    },

    /// The run conditioned the model on something this forward pass does not
    /// carry - a speaker embedding, a d-vector, a language embedding, or an
    /// encoder running at its own sample rate. Each adds a tensor to the
    /// arithmetic while breaking no shape.
    #[error("{field} is set, and that conditioning is not implemented")]
    UnsupportedConditioning {
        /// The field that turned it on.
        field: &'static str,
    },

    /// The symbol table built from `characters` is not the size the config
    /// declares, so the embedding would be indexed with ids from a different
    /// alphabet.
    #[error("num_chars is {declared} but the characters block builds {built} symbols")]
    VocabMismatch {
        /// What the config declared.
        declared: usize,
        /// What its own character block builds.
        built: usize,
    },

    /// The symbol table asks for deduplication without sorting, and the
    /// reference does that with a Python `set`, whose order is not defined.
    /// Reproducing it would be a guess at every id.
    #[error("is_unique without is_sorted leaves the symbol order undefined")]
    UnorderedVocab,
}

/// A checkpoint could not be bound to the configured geometry.
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    /// A tensor was missing, misshapen, or the container rejected it. The
    /// wrapped [`StError`] names the tensor.
    #[error(transparent)]
    Container(#[from] StError),

    /// The same, for a checkpoint that arrived as a torch `.pth`. The wrapped
    /// [`PtError`] names the tensor.
    #[error(transparent)]
    Torch(#[from] PtError),
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
    ///
    /// This is the 🤗 dialect's rule. A Coqui symbol table names its blank
    /// explicitly and puts it at 3, so it fails through [`Self::NoSuchBlank`]
    /// instead.
    #[error("no token has id 0, so there is no blank to intersperse")]
    NoBlank,

    /// A Coqui config names a blank token its own symbol table does not hold,
    /// so there is nothing to intersperse.
    #[error("the symbol table has no {token:?} to intersperse")]
    NoSuchBlank {
        /// The token the config named.
        token: String,
    },

    /// The symbol table could not be built from the config's `characters`
    /// block. Only a Coqui config reaches this.
    #[error("cannot build the symbol table: {source}")]
    Vocabulary {
        /// Why it could not be built.
        #[source]
        source: Box<ConfigError>,
    },

    /// The tokenizer config asks for a preprocessing step this implementation
    /// does not have. Failing is deliberate: silently skipping phonemisation
    /// would produce fluent-sounding nonsense rather than an error.
    #[error("this tokenizer requires {what}, which is not implemented")]
    Unsupported {
        /// The unsupported setting's name.
        what: &'static str,
    },
}
