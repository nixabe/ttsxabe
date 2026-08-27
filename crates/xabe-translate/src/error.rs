//! What this crate refuses, and why each refusal exists.

use thiserror::Error;

/// A translation that could not be produced, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum TranslateError {
    /// The device could not be opened, or a kernel failed.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// The checkpoint could not be read or bound.
    #[error(transparent)]
    Llama(#[from] xabe_llama::LlamaError),

    /// The checkpoint could not be opened.
    #[error(transparent)]
    St(#[from] xabe_st::StError),

    /// The GGUF container failed to read.
    #[error(transparent)]
    Gguf(#[from] xabe_gguf::GgufError),

    /// The sequence outran the positions RoPE was trained over.
    ///
    /// Llama-2's context is 4096 and this checkpoint has no rope scaling. Past
    /// it the model does not fail - it degrades, fluently - which is why this
    /// is a refusal rather than a warning.
    #[error("position {at} is past the {max} this checkpoint was trained over")]
    PastTheEnd {
        /// The position asked for.
        at: usize,
        /// The last one that exists.
        max: usize,
    },
}
