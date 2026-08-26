//! The on-disk shape of a capture.
//!
//! `manifest.json` records two things: how the capture was produced, and what
//! is in it. The provenance half exists because the reference implementation
//! has changed shape before - a golden file whose `transformers` version is
//! unknown cannot be trusted to still describe the current reference, and the
//! only way to notice is to have written the version down.

use serde::Deserialize;

/// The whole of `manifest.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// The 🤗 model id the capture was taken from.
    pub model: String,
    /// The input text, verbatim.
    pub text: String,
    /// The RNG seed. Recorded for reproducibility, but the draws themselves are
    /// captured too - see `noise_dur` and `noise_prior`.
    pub seed: u64,
    /// Output sample rate, in Hz.
    pub sampling_rate: u32,
    /// Standard deviation multiplier applied to the prior's noise.
    pub noise_scale: f32,
    /// Standard deviation multiplier applied to the duration predictor's noise.
    pub noise_scale_duration: f32,
    /// Frames per symbol multiplier; the reciprocal of the length scale.
    pub speaking_rate: f32,
    /// The `transformers` version that produced this capture.
    pub transformers: String,
    /// The `torch` version that produced this capture.
    pub torch: String,
    /// Always `cpu`; a GPU capture would fold PyTorch's kernel choices into the
    /// definition of correct.
    pub device: String,
    /// Always `float32`.
    pub dtype: String,
    /// Thread count during capture. One, because float32 reduction order is not
    /// thread-invariant and the last bits of every tensor move if it is not.
    pub threads: usize,
    /// One entry per captured stage, keyed by stage name.
    pub tensors: std::collections::BTreeMap<String, TensorInfo>,
}

/// One captured tensor's description.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    /// The `.bin` file's name, relative to the capture directory.
    pub file: String,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// One of `f32`, `i64`, `i32`.
    pub dtype: String,
    /// The file's length in bytes, as written.
    pub bytes: usize,
    /// SHA-256 of the file, as written.
    pub sha256: String,
}

impl TensorInfo {
    /// Total number of values, the product of the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes per value for this tensor's dtype.
    pub fn item_size(&self) -> Option<usize> {
        match self.dtype.as_str() {
            "f32" | "i32" => Some(4),
            "i64" => Some(8),
            _ => None,
        }
    }
}
