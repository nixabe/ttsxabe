//! Safetensors container reading for `ttsxabe`.
//!
//! # Why this crate exists
//!
//! The VITS checkpoint is a safetensors file: an 8-byte length, a JSON header,
//! and a flat data segment. Nothing about that needs a dependency, and pulling
//! one in would hand a third party the decision of what happens when a
//! checkpoint is truncated. This crate answers that question itself, in
//! [`StError`], with one variant per way a file can lie about its contents.
//!
//! # What is (and isn't) here
//!
//! Addressing and validation only. This crate knows a tensor is 512×192×7 f32
//! at byte 4096; it has no idea it is a convolution kernel. Meaning lives in
//! `xabe-vits`, which depends on this crate and not the other way round.
//!
//! A checkpoint above a few gigabytes ships split across several files with an
//! index naming which holds what. [`StSet`] addresses those as one checkpoint;
//! a single-file checkpoint is a set of one, so callers never branch on it.
//!
//! Start at [`StFile::open`], then [`StSet::open`].

mod error;
mod file;
mod shard;

pub use error::StError;
pub use file::{Dtype, StFile, TensorInfo};
pub use shard::StSet;
