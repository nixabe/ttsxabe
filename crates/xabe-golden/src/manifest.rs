//! The on-disk shape of a capture.
//!
//! `manifest.json` records two things: how the capture was produced, and what
//! is in it. The provenance half exists because the reference implementation
//! has changed shape before - a golden file whose `transformers` version is
//! unknown cannot be trusted to still describe the current reference, and the
//! only way to notice is to have written the version down.
//!
//! # Two references write these
//!
//! `tools/oracle/capture.py` records 🤗 `VitsModel` and
//! `tools/oracle/capture_coqui.py` records Coqui's own `Vits`. The tensor half
//! is identical - same names, same layouts - because it is the same
//! architecture. The provenance half is not: one names a `transformers`
//! version and the other a `coqui_tts` version, and only the second has
//! phonemes to record. The fields that belong to one dialect default rather
//! than being required, and [`Manifest::input`] is what a test should read
//! instead of choosing between them.

use serde::Deserialize;

/// The whole of `manifest.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    /// The 🤗 model id the capture was taken from.
    pub model: String,
    /// The input text, verbatim.
    pub text: String,
    /// Which reference produced this capture: `coqui`, or absent for 🤗.
    #[serde(default)]
    pub dialect: Option<String>,
    /// The phonemes the text became, when the reference phonemised it.
    ///
    /// Present only for a Coqui capture, whose model is trained on IPA rather
    /// than on letters. It is what the engine is actually fed - see
    /// [`Manifest::input`].
    #[serde(default)]
    pub phonemes: Option<String>,
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
    /// The `transformers` version that produced this capture. Empty for a
    /// Coqui capture, which records [`Manifest::coqui_tts`] instead.
    #[serde(default)]
    pub transformers: String,
    /// The `coqui-tts` version that produced this capture. Empty for a 🤗 one.
    #[serde(default)]
    pub coqui_tts: String,
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

impl Manifest {
    /// What the engine should be given to reproduce this capture.
    ///
    /// The phonemes where the reference phonemised, the text otherwise. A test
    /// that reads `text` directly passes on one dialect and silently compares
    /// two different sentences on the other, so it should read this instead.
    pub fn input(&self) -> &str {
        self.phonemes.as_deref().unwrap_or(&self.text)
    }
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
