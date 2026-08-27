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
//! Two deliberate reductions against the format as published:
//!
//! - **No quantized types.** F32, F16 and BF16 are read; every block format
//!   is refused by name and id. Decoding one needs a dequantizer per format,
//!   and this workspace runs f16 throughout, so a `Q4_K` tensor has no
//!   correct downstream interpretation. Refusing beats mis-sizing: the block
//!   formats pack a scale per 32 or 256 elements, so reading one as raw
//!   values consumes the right number of bytes and yields the wrong numbers.
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
//! value model and the parse order came from there; the quantized type table
//! was dropped, the accessors were reshaped to mirror `xabe-st`, and
//! [`TensorInfo::shape`] is new. See `docs/TOOLCHAIN.md`.
//!
//! Start at [`GgufFile::open`].

mod error;
mod file;
mod reader;
mod types;
mod value;

pub use error::GgufError;
pub use file::{GgufFile, TensorInfo};
pub use types::GgmlType;
pub use value::{GgufArray, GgufValue};
