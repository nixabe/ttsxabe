//! Errors raised while opening or reading a safetensors file.
//!
//! Every variant names a specific way a file can be wrong. They exist so that a
//! malformed model produces a sentence naming the offending tensor, rather than
//! a panic inside a slice index.

use std::path::PathBuf;

/// A safetensors container could not be opened, parsed, or trusted.
#[derive(Debug, thiserror::Error)]
pub enum StError {
    /// The file could not be opened or memory-mapped.
    #[error("cannot open {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// The file is shorter than the 8-byte header-length prefix.
    #[error("{path} is {len} bytes, too short to hold a safetensors header")]
    TooShort {
        /// The file that is too short.
        path: PathBuf,
        /// Its actual length in bytes.
        len: u64,
    },

    /// The declared header length runs past the end of the file. Checked before
    /// slicing so a corrupt length cannot cause an out-of-bounds read.
    #[error("{path} declares a {header_len}-byte header but is only {len} bytes")]
    HeaderOverrun {
        /// The file whose header length is impossible.
        path: PathBuf,
        /// The declared header length.
        header_len: u64,
        /// The actual file length.
        len: u64,
    },

    /// The header is not valid UTF-8 JSON, or not a JSON object.
    #[error("{path} has a malformed JSON header: {source}")]
    Header {
        /// The file whose header failed to parse.
        path: PathBuf,
        /// The `serde_json` failure.
        #[source]
        source: serde_json::Error,
    },

    /// A tensor's byte range lies partly or wholly outside the data segment.
    /// This is the check that turns a truncated download into a message instead
    /// of a segfault.
    #[error("tensor {name} spans bytes {start}..{end} but the data segment is {data_len} bytes")]
    TensorOutOfBounds {
        /// The offending tensor.
        name: String,
        /// Start of the declared range, relative to the data segment.
        start: u64,
        /// End of the declared range.
        end: u64,
        /// Size of the data segment.
        data_len: u64,
    },

    /// A tensor's declared shape does not match the bytes it claims to occupy.
    #[error("tensor {name} has shape {shape:?} of {dtype} ({expected} bytes) but spans {actual}")]
    TensorSizeMismatch {
        /// The offending tensor.
        name: String,
        /// Its declared shape.
        shape: Vec<usize>,
        /// Its declared dtype.
        dtype: String,
        /// Bytes implied by shape × dtype width.
        expected: u64,
        /// Bytes the offset range actually covers.
        actual: u64,
    },

    /// A dtype this reader does not decode. The VITS checkpoints are pure F32;
    /// anything else is rejected loudly rather than reinterpreted.
    #[error("tensor {name} has unsupported dtype {dtype}")]
    UnsupportedDtype {
        /// The offending tensor.
        name: String,
        /// The dtype string from the header.
        dtype: String,
    },

    /// The data segment, or a tensor within it, does not start on a 4-byte
    /// boundary.
    ///
    /// The safetensors format does not *guarantee* alignment, but every real
    /// producer pads the JSON header so the data segment lands on 8 bytes.
    /// Reading unaligned bytes as `f32` is undefined behaviour, so this reader
    /// refuses the file rather than casting a misaligned pointer. Caught by the
    /// container tests, which is why they build files byte by byte.
    #[error("{what} starts at byte {offset}, which is not 4-byte aligned")]
    Misaligned {
        /// What is misaligned - the data segment, or a named tensor.
        what: String,
        /// The offending offset.
        offset: u64,
    },

    /// A tensor was requested by name and is not in the file.
    #[error("tensor {name} is {dtype}; borrow it with tensor_f32, which widens")]
    NotBorrowable {
        /// The tensor asked for.
        name: String,
        /// What the file actually stores.
        dtype: &'static str,
    },

    /// A tensor the schema expected is not in the file.
    #[error("no tensor named {name}")]
    MissingTensor {
        /// The name that was requested.
        name: String,
    },

    /// A tensor was requested with a shape the caller depends on, and the file
    /// disagrees. Weight loading uses this so a wrong-sized checkpoint fails at
    /// load rather than producing silent numerical nonsense.
    #[error("tensor {name} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// The tensor whose shape is wrong.
        name: String,
        /// The shape the caller required.
        expected: Vec<usize>,
        /// The shape the file declares.
        actual: Vec<usize>,
    },
}
