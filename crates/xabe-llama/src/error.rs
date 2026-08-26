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

    /// Grouped-query attention, which this schema does not describe.
    ///
    /// Llama-2 13B has as many key-value heads as query heads, so `k_proj` and
    /// `q_proj` are the same shape. A checkpoint with fewer would bind here
    /// with the wrong expected shape and be refused for the wrong reason, so
    /// it is refused for the right one instead.
    #[error("{kv} key-value heads against {q} query heads; this schema assumes they match")]
    GroupedQuery {
        /// Key-value heads.
        kv: usize,
        /// Query heads.
        q: usize,
    },

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
