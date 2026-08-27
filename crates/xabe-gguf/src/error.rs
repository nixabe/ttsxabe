//! One variant per way a GGUF file can lie about its contents.

use std::path::PathBuf;

/// Everything that can go wrong reading a GGUF container.
#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    /// The first four bytes were not `GGUF`.
    #[error("not a GGUF file: magic is {0:?}, wanted \"GGUF\"")]
    BadMagic([u8; 4]),

    /// A version this crate has not been checked against.
    ///
    /// Refused rather than attempted: v1 and v2 differ in the width of the
    /// count fields, so guessing would read the tensor directory at the wrong
    /// offset and report plausible nonsense.
    #[error("GGUF version {0} is not supported; this reader is v3 only")]
    UnsupportedVersion(u32),

    /// A read ran off the end of the mapping.
    ///
    /// The reason every read is bounds-checked instead of transmuting the
    /// mapping: a 16 GB file truncated mid-download must fail with a message,
    /// not a segfault inside a slice operation.
    #[error("unexpected end of file reading {context} at byte {offset}")]
    UnexpectedEof {
        /// What was being read.
        context: &'static str,
        /// Where the cursor was.
        offset: u64,
    },

    /// A string field was not UTF-8.
    #[error("invalid UTF-8 in the metadata")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    /// A metadata value tag outside the 13 the format defines.
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),

    /// An array whose element type is itself an array.
    ///
    /// GGUF has exactly one level of nesting. A file claiming otherwise is
    /// corrupt, and following it would recurse without bound.
    #[error("nested arrays are not part of GGUF")]
    NestedArray,

    /// The same key twice.
    #[error("duplicate metadata key `{0}`")]
    DuplicateKey(String),

    /// The same tensor name twice.
    #[error("duplicate tensor `{0}`")]
    DuplicateTensor(String),

    /// `general.alignment` was present but not a power of two.
    #[error("alignment {0} is not a power of two")]
    BadAlignment(u32),

    /// More dimensions than ggml supports.
    #[error("tensor `{name}` claims {n_dims} dimensions; ggml allows at most 4")]
    TooManyDimensions {
        /// The tensor.
        name: String,
        /// What it claimed.
        n_dims: u32,
    },

    /// A ggml element type this crate will not read.
    ///
    /// Quantized types land here on purpose. Reading one needs a dequantizer
    /// per format, and this workspace runs f16 throughout by decision - so a
    /// quantized checkpoint is refused by name rather than loaded into
    /// arithmetic that would treat its packed blocks as raw values.
    #[error("tensor `{name}` has ggml type {ggml_type}, which this reader does not decode")]
    UnsupportedGgmlType {
        /// The tensor.
        name: String,
        /// The raw `enum ggml_type` id.
        ggml_type: u32,
    },

    /// A tensor's declared extent is not inside the file.
    #[error("tensor `{name}` spans {start}..{end} but the file is {file_len} bytes")]
    TensorOutOfBounds {
        /// The tensor.
        name: String,
        /// First byte.
        start: u64,
        /// One past the last.
        end: u64,
        /// How big the file actually is.
        file_len: u64,
    },

    /// A row that is not a whole number of quantization blocks.
    #[error("tensor `{name}` has rows of {row}, which is not a multiple of {block}")]
    RaggedBlocks {
        /// The tensor.
        name: String,
        /// The fastest-varying dimension.
        row: u64,
        /// The format's block size.
        block: u64,
    },

    /// An offset computation overflowed.
    #[error("offset arithmetic overflowed for tensor `{0}`")]
    OffsetOverflow(String),

    /// A tensor was asked for by a name the file does not have.
    #[error("no tensor named `{0}`")]
    MissingTensor(String),

    /// A tensor was read at a width it is not stored at.
    #[error("tensor `{name}` is {found:?}, not {wanted}")]
    WrongDtype {
        /// The tensor.
        name: String,
        /// What it is.
        found: crate::GgmlType,
        /// What the caller asked for.
        wanted: &'static str,
    },

    /// The file could not be opened or mapped.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// Why.
        source: std::io::Error,
    },
}
