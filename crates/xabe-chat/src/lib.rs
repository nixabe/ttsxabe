//! Llama-3 on CUDA: the chat model, in this engine.
//!
//! # What this retracts
//!
//! The plan said the chat LLM stays in llama.cpp and there is no
//! `--llm-model`. This is the half of that retraction which runs weights.
//! `docs/MILESTONES.md` carries the reasoning; the short version is that
//! `--llm-url` remained the only way to reach a chat model long after the
//! loader could read one, and delegating the last stage kept a second runtime,
//! a second copy of the weights and a second GPU allocation alive for it.
//!
//! # What is here and what is next door
//!
//! This crate owns the arithmetic and the sampler. It does **not** own the
//! prompt format: `xabe-serve`'s `config.rs` builds the transcript
//! `gateway.py` established, and this takes a prompt string exactly as
//! llama-server's `/completion` does. One opinion about prompt format, in one
//! place.
//!
//! Start at [`ChatModel::open`], then [`ChatModel::complete`].
//!
//! # The three things that differ from [`xabe_translate`]
//!
//! Same architecture family, and each difference is silent when missed - the
//! model keeps producing fluent text from wrong arithmetic, which is why each
//! has a test rather than a comment:
//!
//! - **Grouped-query attention**, 32 query heads over 8 key-value heads.
//! - **A rope base of 500000**, not 10000.
//! - **`rope_freqs.weight`**, Llama-3.1's per-pair divisor, which is not all
//!   ones on this checkpoint.

mod error;
mod generate;
mod model;
mod sample;

pub use error::ChatError;
pub use generate::{Completion, Stop};
pub use model::{Cache, ChatModel};
pub use sample::{Rng, Sampling, sample};
