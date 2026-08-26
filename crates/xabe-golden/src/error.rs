//! Errors raised while reading a captured oracle.
//!
//! A golden directory is regenerable, so the useful failure here is not "it
//! broke" but "which stage, and how". Every variant names the tensor it is
//! talking about for that reason.

use std::path::PathBuf;

/// A capture directory could not be opened, parsed, or trusted.
#[derive(Debug, thiserror::Error)]
pub enum GoldenError {
    /// The manifest or one of its `.bin` files could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// `manifest.json` is not valid JSON, or does not have the expected shape.
    #[error("{path} is not a valid capture manifest: {source}")]
    Manifest {
        /// The manifest that could not be parsed.
        path: PathBuf,
        /// The underlying deserialisation error.
        #[source]
        source: serde_json::Error,
    },

    /// A stage the caller asked for is not in this capture. Listing what *is*
    /// present turns a typo into an immediately obvious mistake.
    #[error("{capture} has no tensor {name}; it holds: {available}")]
    NoSuchTensor {
        /// The capture directory.
        capture: PathBuf,
        /// The tensor that was asked for.
        name: String,
        /// Comma-separated list of the tensors that do exist.
        available: String,
    },

    /// The manifest declares a dtype this reader does not handle.
    #[error("{name} has dtype {dtype}, which is not one of f32, i64, i32")]
    UnknownDtype {
        /// The offending tensor.
        name: String,
        /// The dtype string as it appeared in the manifest.
        dtype: String,
    },

    /// The caller asked for a dtype the tensor is not stored in.
    #[error("{name} is stored as {actual}, not {wanted}")]
    WrongDtype {
        /// The offending tensor.
        name: String,
        /// What the manifest says it is.
        actual: String,
        /// What the caller asked for.
        wanted: String,
    },

    /// The file's length disagrees with the shape and dtype in the manifest.
    #[error("{name} is {actual} bytes but its shape {shape:?} implies {expected}")]
    SizeMismatch {
        /// The offending tensor.
        name: String,
        /// The tensor's declared shape.
        shape: Vec<usize>,
        /// The byte count the shape implies.
        expected: usize,
        /// The byte count actually on disk.
        actual: usize,
    },

    /// The file's contents do not match the checksum recorded at capture time.
    ///
    /// This is the variant that catches a half-written capture - a truncated
    /// `.bin` otherwise reads as a perfectly plausible shorter tensor.
    #[error(
        "{name} does not match its recorded checksum; the capture is damaged, re-run capture.py"
    )]
    Corrupt {
        /// The offending tensor.
        name: String,
    },

    /// A comparison was asked for between differently sized tensors.
    #[error("{name}: computed {actual} values, the oracle holds {expected}")]
    LengthMismatch {
        /// The stage being compared.
        name: String,
        /// How many values the oracle holds.
        expected: usize,
        /// How many values the implementation produced.
        actual: usize,
    },
}
