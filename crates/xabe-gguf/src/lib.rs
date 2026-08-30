//! GGUF container reading for `ttsxabe`.
//!
//! # Why this crate exists
//!
//! The chat model ships as a GGUF, not a safetensors checkpoint, and GGUF is
//! a different container with a different byte layout, a different metadata
//! model and a different idea of what a tensor's shape is. This crate is the
//! only place that knows any of that, exactly as `xabe-st` is the only place
//! that knows safetensors. Everything downstream reads through [`GgufFile`],
//! so a layout mistake surfaces in one file rather than thirty.
//!
//! # What is (and isn't) here
//!
//! Addressing and validation only, the same contract `xabe-st` keeps. This
//! crate knows a tensor is 4096x14336 f16 at byte 2101362944; it has no idea
//! it is a feed-forward gate. Meaning lives in `xabe-llama`.
//!
//! Nine block formats are decoded alongside the three unquantized widths:
//! `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` and the K-quants `Q2_K` through
//! `Q6_K`. Each is unpacked to f32 on read, so nothing above this crate has to
//! know a tensor was packed at all.
//!
//! **Unpacking is not the same as running quantized.** The weights land in
//! memory, and on the device, at full width: a 4-bit 13 B is a 7 GB file and
//! still 26.5 GB of f16 once loaded. What this buys is disk and load
//! bandwidth. Keeping blocks packed on the device would mean teaching every
//! matmul the block layouts, which is a different and much larger piece of
//! work - see `docs/MODEL.md`.
//!
//! Two deliberate limits remain:
//!
//! - **No `IQ*`, `TQ*` or `Q8_K`.** The first two are importance-weighted and
//!   ternary families this workspace has no file in; `Q8_K` is an intermediate
//!   used while quantizing and never appears in a stored tensor. All are
//!   refused by name and id rather than mis-sized.
//! - **v3 only.** v1 and v2 differ in the width of the count fields, so
//!   guessing would read the tensor directory at the wrong offset and report
//!   plausible nonsense.
//!
//! # Shapes are transposed, and that is the trap
//!
//! GGUF stores dimensions fastest-varying first, which is the reverse of a
//! safetensors header. The same matrix a safetensors file calls
//! `[128256, 4096]` appears here as `[4096, 128256]`. [`TensorInfo::dims`] is
//! what the file said; [`TensorInfo::shape`] is the row-major reading, and is
//! what should be compared against a geometry.
//!
//! # Provenance
//!
//! Adapted from `llmxabe/crates/xabe-gguf`, the same author's LLM engine,
//! which has been reading GGUF on this hardware for a while. The cursor, the
//! value model and the parse order came from there. The accessors were
//! reshaped to mirror `xabe-st`, [`TensorInfo::shape`] is new, and the
//! dequantizers are written against `gguf-py` and checked against it. See
//! `docs/TOOLCHAIN.md`.
//!
//! Start at [`GgufFile::open`].

mod dequant;
mod error;
mod file;
mod reader;
mod types;
mod value;

pub use dequant::dequantize_blocks;
pub use error::GgufError;
pub use file::{GgufFile, TensorInfo};
pub use types::GgmlType;
pub use value::{GgufArray, GgufValue};
