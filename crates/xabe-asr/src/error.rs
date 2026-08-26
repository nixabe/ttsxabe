//! What this crate refuses, and why each refusal exists.

use thiserror::Error;

/// A transcription that could not be produced, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum AsrError {
    /// The device could not be opened, or a kernel failed.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// The checkpoint could not be read or bound.
    #[error(transparent)]
    Whisper(#[from] xabe_whisper::WhisperError),

    /// The checkpoint could not be opened.
    #[error(transparent)]
    St(#[from] xabe_st::StError),

    /// The decoder was asked for more tokens than its positions cover.
    ///
    /// Whisper's decoder has 448 learned positions and no extrapolation. Past
    /// them the position embedding would be read out of bounds, and a model
    /// that silently wrapped would produce fluent nonsense.
    #[error("position {at} is past the {max} the decoder was trained on")]
    PastTheEnd {
        /// The position asked for.
        at: usize,
        /// The last one that exists.
        max: usize,
    },
}
