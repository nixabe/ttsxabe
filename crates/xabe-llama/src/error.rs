//! What this crate refuses, and why each refusal exists.

use std::path::PathBuf;
use thiserror::Error;

/// A checkpoint this crate would not bind, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum LlamaError {
    /// A file in the checkpoint directory could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// The path that failed.
        path: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },

    /// `config.json` did not parse.
    #[error("parsing {path}: {source}")]
    Config {
        /// The path that failed.
        path: PathBuf,
        /// The underlying failure.
        source: serde_json::Error,
    },

    /// The architecture is one this schema is not written for.
    #[error("{found}, but this schema is written for {wanted}")]
    Architecture {
        /// What the checkpoint says it is.
        found: String,
        /// What this crate binds.
        wanted: &'static str,
    },

    /// `hidden_size` does not divide evenly by the head count.
    #[error("hidden_size {hidden} does not divide by {heads} heads")]
    RaggedHeads {
        /// The width that failed to divide.
        hidden: usize,
        /// The head count it failed to divide by.
        heads: usize,
    },

    /// Grouped-query attention, raised by an engine that cannot run it.
    ///
    /// The *schema* binds grouped-query checkpoints fine - `k` and `v` are
    /// simply narrower than `q`. This is what a forward pass without the head
    /// mapping calls at open, so the refusal names the real reason instead of
    /// surfacing later as a head index out of range.
    #[error("{kv} key-value heads against {q} query heads; this engine needs them to match")]
    GroupedQuery {
        /// Key-value heads.
        kv: usize,
        /// Query heads.
        q: usize,
    },

    /// Key-value heads that do not divide the query heads.
    ///
    /// Structurally impossible rather than merely unsupported: grouped-query
    /// attention shares one key-value head across a whole number of query
    /// heads, so a remainder means the geometry was misread.
    #[error("{q} query heads do not divide into {kv} key-value heads")]
    RaggedGroups {
        /// Key-value heads.
        kv: usize,
        /// Query heads.
        q: usize,
    },

    /// A stated head width that disagrees with the division.
    ///
    /// A GGUF states `key_length` outright while `config.json` leaves it to be
    /// derived. When both are available they must agree, because a checkpoint
    /// where they do not is one whose geometry has been misread somewhere.
    #[error("the file states a head width of {stated}, but the geometry divides to {divided}")]
    HeadDim {
        /// What the file said.
        stated: usize,
        /// What `hidden_size / num_attention_heads` gives.
        divided: usize,
    },

    /// The GGUF container could not be read.
    #[error(transparent)]
    Gguf(#[from] xabe_gguf::GgufError),

    /// A GGUF metadata key the schema needs is absent or the wrong type.
    #[error("the GGUF has no usable `{0}`")]
    MissingMetadata(&'static str),

    /// A tensor the schema requires is not in the checkpoint.
    #[error("no tensor named {0}")]
    MissingTensor(String),

    /// A tensor is present but not the shape the geometry implies.
    #[error("{name} is {found:?}, expected {want:?}")]
    Shape {
        /// The tensor that disagreed.
        name: String,
        /// The shape the checkpoint declares.
        found: Vec<usize>,
        /// The shape `config.json` implies.
        want: Vec<usize>,
    },

    /// The checkpoint could not be opened or a tensor could not be read.
    #[error(transparent)]
    St(#[from] xabe_st::StError),

    /// The tokenizer model was absent or malformed.
    #[error("{path}: {what}")]
    Vocab {
        /// The file at fault.
        path: PathBuf,
        /// What was wrong with it.
        what: String,
    },
}
