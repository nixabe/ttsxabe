//! Whisper: the geometry, the frontend, the tokenizer and the forward pass.
//!
//! The reference is 🤗 `WhisperForConditionalGeneration` in float32 on CPU, not
//! whisper.cpp. That choice is not neutral - whisper.cpp's tokenizer is a
//! greedy longest-match over a `std::regex` in which `[[:alpha:]]` is not
//! `\p{L}`, so it already disagrees with the reference on Han input, which is
//! all this engine ever transcribes. `whisper-server`'s transcripts remain a
//! cross-check, not the definition of correct. See `docs/ORACLE.md`.
//!
//! # Deliberately absent
//!
//! The live pipeline runs `-nt`, greedy, at a fixed `language=zh`, on
//! VAD-gated utterances of a few seconds. That makes DTW and token timestamps,
//! the grammar engine, beam search and `whisper_full_parallel` dead weight -
//! roughly 1,200 lines of C++ that buy this engine nothing. Each omission is
//! recorded in `docs/MODEL.md` with its reason, so re-adding one is a decision
//! rather than a discovery.
//!
//! Start at [`WhisperConfig`], then [`Frontend`], then [`WhisperWeights`].

mod config;
mod error;
mod frontend;
mod weights;

pub use config::WhisperConfig;
pub use error::WhisperError;
pub use frontend::{DYNAMIC_RANGE, F_MAX, Frontend};
pub use weights::{
    Attention, Conv1d, DecoderLayer, EncoderLayer, LayerNorm, Linear, WhisperWeights,
};
