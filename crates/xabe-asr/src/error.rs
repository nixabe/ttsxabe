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

    /// A cross-attention projection whose bias is not where the batched
    /// cache build expects it.
    ///
    /// Every layer's key and value projections go out as one product each,
    /// which carries one bias for the whole batch - so the value biases are
    /// added in the head split, and the key projection is assumed to have
    /// none, as Whisper's does. A checkpoint that broke either assumption
    /// would bind cleanly and build a cache that is quietly off by a bias, so
    /// it is refused here by layer instead.
    #[error("decoder layer {layer}: {what}, which the batched cross-attention cache does not add")]
    CrossBias {
        /// Which layer.
        layer: usize,
        /// What was found.
        what: &'static str,
    },
}
