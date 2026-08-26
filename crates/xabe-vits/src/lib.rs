//! VITS model geometry and weight binding.
//!
//! # Why this crate exists
//!
//! To make a wrong checkpoint fail loudly. VITS keeps producing speech when its
//! weights are misread — a transposed kernel, a coupling block bound at the
//! wrong width — so the only defence is to check every shape at load time
//! against a validated configuration, and name the tensor that disagrees.
//!
//! # What is (and isn't) here
//!
//! [`VitsConfig`] holds the geometry and rejects one it cannot express.
//! [`VitsWeights`] binds every tensor the inference path reads, borrowed from
//! the mapping and shape-checked. Neither does arithmetic; that is `xabe-dsp`.
//!
//! Start at [`VitsConfig::from_json_path`], then [`VitsWeights::load`].

mod config;
mod error;
mod weights;

pub use config::VitsConfig;
pub use error::{ConfigError, WeightError};
pub use weights::{
    Conv, DdsConv, Decoder, DurationFlow, DurationPredictor, EncoderLayer, FlowBlock, Norm,
    ResBlock, TextEncoder, VitsWeights, WaveNetLayer, WnConv,
};
