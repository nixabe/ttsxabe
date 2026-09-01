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
//! # Two published checkpoints, one geometry
//!
//! `facebook/mms-tts-nan` and `neurlang/coqui-vits-suisiann-minnan-hokkien` are
//! the same architecture from different trainers, and this crate reads both.
//! [`VitsConfig`] is what they have in common; [`CoquiConfig`] is the second
//! one's own configuration file, which is a whole training run rather than a
//! model, and [`CoquiConfig::to_vits`] is the conversion. [`VitsWeights::load`]
//! binds a 🤗 safetensors export and [`VitsWeights::load_coqui`] a torch
//! `.pth`, and they produce the identical structure - so the forward pass in
//! `xabe-tts` never learns which it is running.
//!
//! They differ in the container, in every tensor name, in the symbol table, and
//! in whether the decoder's weight norm was fused before saving - see
//! [`MaybeWn`] for the last, which is the only one that is not cosmetic.
//! `docs/MODEL.md` has all five.
//!
//! Start at [`VitsConfig::from_json_path`], then [`VitsWeights::load`].

mod config;
mod coqui;
mod error;
mod tokenizer;
mod weights;

pub use config::VitsConfig;
pub use coqui::{CoquiAudio, CoquiCharacters, CoquiConfig, CoquiModelArgs, CoquiTokenizer};
pub use error::{ConfigError, TokenizerError, WeightError};
pub use tokenizer::Tokenizer;
pub use weights::{
    Conv, DdsConv, Decoder, DurationFlow, DurationPredictor, EncoderLayer, FlowBlock, MaybeWn,
    Norm, ResBlock, TextEncoder, VitsWeights, WaveNetLayer, WnConv,
};
