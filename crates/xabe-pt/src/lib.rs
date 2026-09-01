//! Torch checkpoint reading for `ttsxabe`.
//!
//! # Why this crate exists
//!
//! The Coqui VITS checkpoint is a `.pth`: a zip archive holding a pickle that
//! describes the tensors and one stored entry per storage. That is a third
//! container beside safetensors and GGUF, and it gets the same treatment -
//! one crate that knows the byte layout, and nothing above it that does.
//!
//! # This is not "unpickling"
//!
//! Unpickling in general executes what the stream names, which is why the
//! WaveGlow checkpoint in `xabe-taco` has to be converted offline: it is a
//! pickled `nn.Module` object graph and rebuilding it needs the model's own
//! class definitions.
//!
//! A **state dict** is not that. It is a mapping of strings to tensors, and its
//! stream names exactly three things: `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2`, and a storage class. This crate
//! implements those three and refuses everything else by name, so the
//! distinction between a file it can honestly read and one it cannot is an
//! error message rather than a guess.
//!
//! # What is (and isn't) here
//!
//! Addressing and validation only, the same contract `xabe-st` and `xabe-gguf`
//! keep. This crate knows a tensor is 512x192x7 f32 at byte 489728; it has no
//! idea it is a convolution kernel. Meaning lives in `xabe-vits`.
//!
//! Two limits are deliberate:
//!
//! - **Stored entries only.** Torch writes every entry uncompressed and aligns
//!   the storages to 64 bytes so they can be mapped; an archive that compressed
//!   them could not be borrowed from and is refused rather than inflated.
//! - **Contiguous tensors only.** A saved view would read its elements in the
//!   wrong order while keeping a plausible shape, so the stride is checked.
//!
//! Start at [`PtFile::open_section`].

mod error;
mod file;
mod pickle;
mod zip;

pub use error::PtError;
pub use file::{Dtype, PtFile, TensorInfo};
