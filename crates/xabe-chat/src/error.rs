//! What this crate refuses, and why each refusal exists.

use thiserror::Error;

/// A reply that could not be produced, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum ChatError {
    /// The device could not be opened, or a kernel failed.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// The checkpoint could not be read or bound.
    #[error(transparent)]
    Llama(#[from] xabe_llama::LlamaError),

    /// The GGUF container failed to read.
    #[error(transparent)]
    Gguf(#[from] xabe_gguf::GgufError),

    /// The checkpoint was not a GGUF.
    ///
    /// Unlike the translator, which exists on this disk in both containers,
    /// the chat model exists as a GGUF and nothing else - its vocabulary lives
    /// inside the file. A 🤗 directory would load its weights and then have no
    /// tokenizer, so it is refused at open rather than most of the way in.
    #[error("{0} is not a .gguf; the chat model is only published as one")]
    NotGguf(std::path::PathBuf),

    /// The sequence outran the positions the model was trained over.
    ///
    /// Past it a Llama does not fail, it degrades fluently, which is why this
    /// is a refusal rather than a warning.
    #[error("position {at} is past the {max} this checkpoint was trained over")]
    PastTheEnd {
        /// The position asked for.
        at: usize,
        /// The last one that exists.
        max: usize,
    },

    /// A sampler parameter outside the range it means anything over.
    #[error("{what} is {got}, which is outside {range}")]
    BadSampler {
        /// Which knob.
        what: &'static str,
        /// What was passed.
        got: f32,
        /// What it has to be in.
        range: &'static str,
    },

    /// A conversation with nothing in it.
    ///
    /// A prompt of only a system message and headers produces a reply to
    /// nothing, which reads as the model being broken rather than as the
    /// caller having sent an empty turn.
    #[error("the conversation has no user turn to answer")]
    NothingToAnswer,
}
