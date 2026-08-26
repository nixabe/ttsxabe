//! What the VAD refuses.

use thiserror::Error;

/// A checkpoint or a request the VAD could not accept.
#[derive(Debug, Error)]
pub enum VadError {
    /// The container could not be opened or a tensor could not be addressed.
    #[error(transparent)]
    Container(#[from] xabe_st::StError),

    /// The checkpoint's geometry is not the one this implementation computes.
    ///
    /// The graph below is written for exactly one network. A checkpoint with
    /// different channel counts would load, run, and produce numbers that mean
    /// nothing, so the shapes are checked at bind time instead.
    #[error("{what}: expected {expected}, found {found}")]
    Geometry {
        /// Which property disagreed.
        what: &'static str,
        /// What this implementation requires.
        expected: String,
        /// What the checkpoint declares.
        found: String,
    },

    /// The `__metadata__` block is missing something the schema needs.
    ///
    /// The converter writes the ggml header into it precisely so the geometry
    /// can be checked against the tensors. A file without it was produced by
    /// something else and should not be trusted to have the same layout.
    #[error("checkpoint metadata has no {0}; was it produced by tools/vad/ggml_to_safetensors.py?")]
    MissingMetadata(&'static str),
}
