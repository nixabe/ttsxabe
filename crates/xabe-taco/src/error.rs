//! What this crate refuses, and what each refusal prevents.

use thiserror::Error;

/// Everything that can go wrong opening or running Tacotron2 + WaveGlow.
#[derive(Debug, Error)]
pub enum TacoError {
    /// A checkpoint file is absent.
    ///
    /// Named rather than reported as a bare I/O error because the two halves
    /// live in separate files and "not found" is otherwise ambiguous about
    /// which one.
    #[error("{what} is missing at {path}")]
    Missing {
        /// Which of the three files.
        what: &'static str,
        /// Where it was looked for.
        path: String,
    },

    /// `tacotron2.json` could not be read or parsed.
    #[error("reading {path}: {source}")]
    Config {
        /// The config path.
        path: String,
        /// The underlying parse or I/O failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The config declares a geometry this crate does not implement.
    ///
    /// The forward pass hard-codes the shapes the published checkpoint has -
    /// one frame per step, eight-wide groups, a two-layer prenet. A config
    /// saying otherwise would run and produce wrong audio, so it stops here.
    #[error("unsupported geometry: {0}")]
    Geometry(String),

    /// A tensor was absent or the wrong shape.
    #[error(transparent)]
    Weights(#[from] xabe_st::StError),

    /// A device operation failed.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// An invertible 1x1 convolution's weight is singular.
    ///
    /// WaveGlow initialises these as orthonormal and training keeps them
    /// invertible, so this cannot happen with a healthy checkpoint - which is
    /// exactly why it is worth saying out loud rather than dividing by zero and
    /// emitting noise.
    #[error("convinv.{flow} is singular and cannot be inverted")]
    Singular {
        /// Which flow.
        flow: usize,
    },
}
