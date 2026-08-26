//! What this crate refuses, and why each refusal exists.

use std::path::PathBuf;
use thiserror::Error;

/// A checkpoint this crate would not load, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum WhisperError {
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

    /// `d_model` does not divide evenly by the head count.
    #[error("{what}: d_model {d_model} does not divide by {heads} heads")]
    RaggedHeads {
        /// Which half of the model.
        what: &'static str,
        /// The width that failed to divide.
        d_model: usize,
        /// The head count it failed to divide by.
        heads: usize,
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

    /// A tokenizer file was absent or malformed.
    #[error("{path}: {what}")]
    Vocab {
        /// The file at fault.
        path: PathBuf,
        /// What was wrong with it.
        what: String,
    },
}
