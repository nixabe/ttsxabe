//! Llama-2's forward pass, assembled.
//!
//! `xabe-llama` says what the tensors are, `xabe-cuda` says what the
//! arithmetic is, and this crate is where they meet. The same split as
//! `xabe-whisper`/`xabe-asr` and `xabe-vits`/`xabe-tts`.
//!
//! The reference is 🤗 `LlamaForCausalLM` in float32 on CPU, captured stage by
//! stage. See `docs/ORACLE.md`.

mod error;
mod model;

pub use error::TranslateError;
pub use model::{Cache, TEMPLATE, Translator};
