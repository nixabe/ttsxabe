//! Llama-2's forward pass, assembled.
//!
//! `llmtie-llama` says what the tensors are, `llmtie-cuda` says what the
//! arithmetic is, and this crate is where they meet. The same split as
//! `llmtie-whisper`/`llmtie-asr` and `llmtie-vits`/`llmtie-tts`.
//!
//! The reference is 🤗 `LlamaForCausalLM` in float32 on CPU, captured stage by
//! stage. See `docs/ORACLE.md`.

mod error;
mod model;

pub use error::TranslateError;
pub use model::{Cache, Packing, TEMPLATE, TranslationBatch, Translator};
