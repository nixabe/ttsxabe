//! What this crate refuses, and why each refusal exists.

use thiserror::Error;

/// A CosyVoice stage that could not be loaded or run.
#[derive(Debug, Error)]
pub enum CosyError {
    /// The device could not be opened, or a kernel failed.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// A checkpoint could not be opened or a tensor read.
    #[error(transparent)]
    St(#[from] xabe_st::StError),

    /// A geometry whose divisions do not come out whole.
    #[error("{what}: {got} against {want}")]
    Geometry {
        /// Which division.
        what: &'static str,
        /// The numerator, or the value that was wrong.
        got: usize,
        /// The denominator, or the value it should have agreed with.
        want: usize,
    },

    /// A tensor the schema requires is not in the checkpoint.
    #[error("no tensor named {0}")]
    MissingTensor(String),

    /// A tensor is present but not the shape the geometry implies.
    #[error("{name} is {found:?}, expected {want:?}")]
    Shape {
        /// The tensor that disagreed.
        name: String,
        /// What the checkpoint declares.
        found: Vec<usize>,
        /// What the geometry implies.
        want: Vec<usize>,
    },

    /// The prompt does not carry `<|endofprompt|>`.
    ///
    /// Upstream asserts this and it is worth keeping: the instruct string is
    /// what carries the marker, so a prompt without it is one the model was
    /// never trained to answer. It does not fail — it produces confident
    /// nonsense, which is the failure mode a refusal is cheapest against.
    #[error("the prompt has no <|endofprompt|> (token {0}); the instruct text is what carries it")]
    NoEndOfPrompt(u32),

    /// The speaker tensors were absent or the wrong shape.
    ///
    /// They come from two ONNX models this crate deliberately does not run,
    /// captured once from the reference clip. Missing, they cannot be derived
    /// here, so this names the script that produces them.
    #[error("{what}; run tools/oracle/capture_cosyvoice.py to produce it")]
    Speaker {
        /// What was missing or malformed.
        what: String,
    },

    /// Generation ran past the length the text can justify.
    #[error("produced {got} speech tokens for {text} text tokens, past the ratio of {max}")]
    RanAway {
        /// Speech tokens produced.
        got: usize,
        /// Text tokens given.
        text: usize,
        /// The ratio upstream allows.
        max: usize,
    },
}
