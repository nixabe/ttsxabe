//! Silero voice activity detection, from scratch.
//!
//! Fifteen tensors and about a hundred lines of arithmetic: a length-256
//! convolution standing in for an STFT, four small convolutions, one LSTM cell
//! and a 1×1 convolution. One probability per 512 samples, then a hysteresis
//! segmenter that turns those probabilities into spans of speech.
//!
//! It is the smallest whole model in the pipeline, which is why the engine
//! absorbed it first: it can be verified frame by frame against whisper.cpp in
//! about a day, so if the approach were going to fail it would fail here rather
//! than after the 1.5 B ASR had been written.
//!
//! **This is not optional in the pipeline.** Without it, Whisper invents speech
//! out of silence: digital silence transcribed as 我…, faint hiss as
//! 我現在在醫院, room noise as (我會陪你一起走) - and the assistant then answers
//! the hallucination. See `docs/MODEL.md`.
//!
//! Start at [`Vad::probabilities`], then [`segments`].

mod error;
mod forward;
mod segment;
mod weights;

pub use error::VadError;
pub use forward::Vad;
pub use segment::{Segment, SegmentParams, segments};
pub use weights::{
    BINS, Conv, ENCODER, GATES, HIDDEN, PAD, STFT_HOP, STFT_KERNEL, STFT_ROWS, VadWeights, WINDOW,
};

/// Opens a converted checkpoint and binds it.
pub fn open(path: impl AsRef<std::path::Path>) -> Result<Vad, VadError> {
    let file = xabe_st::StFile::open(path)?;
    Ok(Vad::new(VadWeights::load(&file)?))
}
