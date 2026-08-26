//! The forward pass, assembled.
//!
//! `xabe-vits` says what the tensors are, `xabe-dsp` says what the arithmetic
//! is, and this crate is where they meet: it runs the stages in order and owns
//! the shapes that flow between them.
//!
//! Stages land here one milestone at a time, each diffed against the captured
//! PyTorch oracle before the next is started. See `docs/MILESTONES.md` for what
//! is finished and `docs/ORACLE.md` for what it is being checked against.

mod decoder;
mod duration;
mod flow;
mod prior;
mod rng;
mod synthesize;
mod text_encoder;

pub use decoder::decoder;
pub use duration::duration_predictor;
pub use flow::flow_reverse;
pub use prior::{Prior, expand_prior};
pub use rng::Rng;
pub use synthesize::{Prepared, Prosody, SynthesisError, Synthesizer};
pub use text_encoder::{EncoderOutput, text_encoder};
