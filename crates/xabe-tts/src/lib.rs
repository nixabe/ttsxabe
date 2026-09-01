//! The forward pass, assembled.
//!
//! `xabe-vits` says what the tensors are, `xabe-dsp` says what the arithmetic
//! is, and this crate is where they meet: it runs the stages in order and owns
//! the shapes that flow between them.
//!
//! Stages land here one milestone at a time, each diffed against the captured
//! PyTorch oracle before the next is started. See `docs/MILESTONES.md` for what
//! is finished and `docs/ORACLE.md` for what it is being checked against.
//!
//! # It runs two checkpoints, and the stages do not know it
//!
//! [`Synthesizer::open`] reads the 🤗 export of `mms-tts-nan` and
//! [`Synthesizer::open_coqui`] a Coqui trainer's save of the same architecture;
//! [`GpuModel`] has the same pair. Everything below the constructor is shared,
//! because by then both are a [`xabe_vits::VitsWeights`] and a
//! [`xabe_vits::VitsConfig`]. What was decided at the constructor is kept in
//! [`Source`] and [`Symbols`].
//!
//! The one place the difference reaches arithmetic is the decoder, whose
//! convolutions arrive fused from one and weight-normalised from the other -
//! `decoder.rs` fuses when it has to, on both devices.

mod decoder;
mod duration;
mod flow;
mod gpu;
mod prior;
mod rng;
mod source;
mod synthesize;
mod text_encoder;

pub use decoder::decoder;
pub use duration::duration_predictor;
pub use flow::flow_reverse;
pub use gpu::GpuModel;
pub use prior::{Prior, expand_prior};
pub use rng::Rng;
pub use source::{Source, Symbols};
pub use synthesize::{Prepared, Prosody, SynthesisError, Synthesizer};
pub use text_encoder::{EncoderOutput, text_encoder};
