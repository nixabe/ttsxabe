//! Whisper's forward pass, assembled.
//!
//! `xabe-whisper` says what the tensors are, `xabe-cuda` says what the
//! arithmetic is, and this crate is where they meet: it runs the stages in
//! order and owns the shapes that flow between them. The same split as
//! `xabe-vits` and `xabe-tts`, for the same reason.
//!
//! The reference is 🤗 `WhisperForConditionalGeneration` in float32 on CPU,
//! captured stage by stage. See `docs/ORACLE.md`.

mod error;
mod model;

pub use error::AsrError;
pub use model::{AsrModel, Cache};
