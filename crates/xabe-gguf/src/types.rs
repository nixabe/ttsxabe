//! The `ggml_type` ids this reader decodes, and the ones it refuses.
//!
//! Ids are `enum ggml_type` in `ggml/include/ggml.h`. Only the three
//! unquantized widths are here, and that is a decision rather than an
//! omission: this workspace runs f16 throughout and has no dequantizer for
//! any block format, so a `Q4_K` tensor has no correct interpretation
//! downstream. Meeting one is an error with its name and id attached
//! ([`GgufError::UnsupportedGgmlType`]), not a silent mis-sizing - the block
//! formats pack a scale per 32 or 256 elements, so treating them as raw
//! values would read the right number of bytes and the wrong numbers.
//!
//! [`GgufError::UnsupportedGgmlType`]: crate::GgufError::UnsupportedGgmlType

/// A ggml element type this crate can size and read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// IEEE-754 single precision.
    F32 = 0,
    /// IEEE-754 half precision.
    F16 = 1,
    /// Brain float: an f32 with the low 16 mantissa bits cut off.
    Bf16 = 30,
}

impl GgmlType {
    /// Maps a raw `enum ggml_type` id, or `None` if this reader will not
    /// decode it.
    pub(crate) fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Bytes per element.
    ///
    /// Every type here has a block size of one, which is why this is a plain
    /// multiply rather than the block arithmetic a quantized reader needs.
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}
