//! Llama-2: the geometry, the weight schema and the tokenizer.
//!
//! Everything here is *typed and shape-checked, before any arithmetic* - the
//! same shape as milestone 2 of the VITS work, applied to a model forty times
//! larger. It is built whether or not the forward pass ever is, because none
//! of it is wasted either way: the dtype work is what the tensor-core path
//! needs, and a schema that binds all 363 tensors is a real proof the geometry
//! is understood.
//!
//! # Why the forward pass may never arrive
//!
//! `run.sh` sets `DIRECT_TAIGI=1` by default, which takes the translator out
//! of the reply path entirely and was measured at 3.8 s -> 1.6 s end to end.
//! So the 13 B translator is not used in the default configuration, and
//! writing a second large-model engine for a bypassed stage would be poor
//! value. The decision is deferred until everything else runs, with the loader
//! already sitting here if the answer turns out to be yes.
//!
//! Start at [`LlamaConfig`], then [`LlamaWeights`], then [`Tokenizer`].

mod bpe;
mod config;
mod error;
pub mod gguf;
mod tokenizer;
mod weights;

pub use bpe::{Bpe, pre_tokenize};
pub use config::LlamaConfig;
pub use error::LlamaError;
pub use tokenizer::{Kind, Piece, Tokenizer, UNDERLINE};
pub use weights::{Attention, Bound, Layer, LlamaWeights, Mlp};
