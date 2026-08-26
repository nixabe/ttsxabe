//! The forward pass, assembled.
//!
//! `xabe-vits` says what the tensors are, `xabe-dsp` says what the arithmetic
//! is, and this crate is where they meet: it runs the stages in order and owns
//! the shapes that flow between them.
//!
//! Stages land here one milestone at a time, each diffed against the captured
//! PyTorch oracle before the next is started. See `docs/MILESTONES.md` for what
//! is finished and `docs/ORACLE.md` for what it is being checked against.

mod text_encoder;

pub use text_encoder::{EncoderOutput, text_encoder};
