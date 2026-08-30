//! Tacotron2 and WaveGlow, in this engine.
//!
//! A third synthesiser alongside `xabe-tts`'s VITS and `xabe-cosy`'s
//! CosyVoice3, and the oldest design of the three: an autoregressive mel
//! decoder with location-sensitive attention, and a normalising-flow vocoder.
//!
//! Start at [`Taco`].
//!
//! # One crate, not two
//!
//! Every other model here is a geometry crate that refuses to do arithmetic
//! plus a crate that runs it. This follows `xabe-cosy` instead and keeps both
//! inside one crate with the boundary drawn between modules: [`config`] and
//! `weights` know the shapes and never touch an activation, `model` and
//! `vocoder` do the arithmetic and never parse a file.
//!
//! # It reads converted weights, which is a first
//!
//! Every other stage reads the checkpoint as published. This one cannot:
//! WaveGlow ships as a pickled `nn.Module` object graph in the pre-1.6 torch
//! format, which will not parse without PyTorch and the model's own class
//! definitions. `tools/convert_tacotron2.py` does that once, offline, and
//! writes safetensors. The claim that this workspace reads published
//! checkpoints directly holds everywhere except here.
//!
//! # It is stochastic
//!
//! Twice, and both are the model rather than an implementation choice: the
//! prenet keeps its dropout at inference, and WaveGlow starts from noise. Two
//! calls on the same text give two renderings.

mod clock;
mod config;
mod error;
mod model;
mod pipeline;
mod text;
mod vocoder;
mod weights;

pub use clock::Timings;
pub use config::Config;
pub use error::TacoError;
pub use model::Rng;
pub use pipeline::{FILES, Taco};
pub use text::{Tokenizer, poj_to_tlpa};
