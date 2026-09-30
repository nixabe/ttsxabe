//! The `ggml_type` ids this reader decodes, and the ones it refuses.
//!
//! Ids and block geometry are `enum ggml_type` and `GGML_QUANT_SIZES` in
//! upstream llama.cpp (`ggml/include/ggml.h`, `gguf-py/gguf/constants.py`).
//!
//! Three unquantized widths and the nine block formats a llama.cpp checkpoint
//! is actually shipped in. What is deliberately absent is the `IQ*` family,
//! the ternary `TQ*` formats, and `Q8_K` - the last because it is an
//! intermediate used while quantizing and never appears in a stored tensor.
//! Meeting any of them is [`GgufError::UnsupportedGgmlType`] with the name and
//! the id, not a silent mis-sizing.
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
    /// 4-bit, centred, one f16 scale per 32.
    Q4_0 = 2,
    /// 4-bit, offset, an f16 scale and minimum per 32.
    Q4_1 = 3,
    /// 5-bit, centred, one f16 scale per 32.
    Q5_0 = 6,
    /// 5-bit, offset, an f16 scale and minimum per 32.
    Q5_1 = 7,
    /// 8-bit, one f16 scale per 32.
    Q8_0 = 8,
    /// "K-quant" 2-bit: 256-element superblock, 4-bit scales and minimums.
    Q2K = 10,
    /// "K-quant" 3-bit: 6-bit scales, plus a high-bit mask.
    Q3K = 11,
    /// "K-quant" 4-bit: 6-bit scales and minimums, eight sub-blocks of 32.
    Q4K = 12,
    /// "K-quant" 5-bit: `Q4_K` plus one high bit per element.
    Q5K = 13,
    /// "K-quant" 6-bit: 8-bit signed scales, no minimum.
    Q6K = 14,
    /// Brain float 16: truncated f32 mantissa.
    Bf16 = 30,
}

impl GgmlType {
    /// Maps a raw `enum ggml_type` id, or `None` if this reader will not
    /// decode it.
    pub(crate) fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    /// Elements per block. One for the unquantized widths.
    pub const fn block_size(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes per block.
    pub const fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Whether the values are packed in blocks rather than stored outright.
    pub const fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16)
    }

    /// Bytes needed to hold `n` elements.
    ///
    /// Not `n * type_size`: for a block format that would be six times too
    /// large, which is the difference between a tensor directory that
    /// validates and one that rejects every file it is given.
    pub const fn bytes_for(self, n: u64) -> u64 {
        n / self.block_size() * self.type_size()
    }
}
