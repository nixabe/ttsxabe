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
//! Start at [`StFile::open`].

mod error;
mod file;

pub use error::StError;
pub use file::{Dtype, StFile, TensorInfo};
