//! GGUF metadata values.
//!
//! The format's key-value store carries twelve scalar types plus one level of
//! array nesting. This mirrors that shape rather than collapsing everything to
//! something JSON-like, so asking for a `u32` on a key that holds a string
//! returns `None` instead of a silently truncating cast.

/// One GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// Unsigned byte.
    U8(u8),
    /// Signed byte.
    I8(i8),
    /// Unsigned 16-bit.
    U16(u16),
    /// Signed 16-bit.
    I16(i16),
    /// Unsigned 32-bit.
    U32(u32),
    /// Signed 32-bit.
    I32(i32),
    /// Single precision.
    F32(f32),
    /// Boolean, stored as one byte.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Unsigned 64-bit.
    U64(u64),
    /// Signed 64-bit.
    I64(i64),
    /// Double precision.
    F64(f64),
    /// A homogeneous array.
    Array(GgufArray),
}

/// A homogeneous GGUF metadata array.
///
/// One type tag covers every element, so this is a `Vec` per element type
/// rather than a `Vec<GgufValue>` - it cannot represent a mixed array because
/// the format cannot either.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufArray {
    /// Unsigned bytes.
    U8(Vec<u8>),
    /// Signed bytes.
    I8(Vec<i8>),
    /// Unsigned 16-bit.
    U16(Vec<u16>),
    /// Signed 16-bit.
    I16(Vec<i16>),
    /// Unsigned 32-bit.
    U32(Vec<u32>),
    /// Signed 32-bit.
    I32(Vec<i32>),
    /// Single precision.
    F32(Vec<f32>),
    /// Booleans.
    Bool(Vec<bool>),
    /// Strings. The tokenizer's vocabulary and merge table arrive this way.
    String(Vec<String>),
    /// Unsigned 64-bit.
    U64(Vec<u64>),
    /// Signed 64-bit.
    I64(Vec<i64>),
    /// Double precision.
    F64(Vec<f64>),
}

impl GgufArray {
    /// How many elements it holds.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether it holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
