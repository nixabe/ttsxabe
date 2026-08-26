//! What this crate refuses, and why each refusal exists.

use thiserror::Error;

/// A container this crate could not read, named precisely enough to fix.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The file is shorter than the chunk it claims to contain.
    ///
    /// A truncated download and a valid short file differ only in the declared
    /// sizes, so the length has to be checked against the bytes actually
    /// present rather than trusted.
    #[error("{what} needs {needed} bytes at offset {at} but the file holds {available}")]
    Truncated {
        /// Which structure was being read.
        what: &'static str,
        /// Byte offset the read started at.
        at: usize,
        /// Bytes the declared size required.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// The four-byte tag at the start of a chunk was not the one expected.
    #[error("expected the {expected} tag at offset {at}, found {found:?}")]
    BadTag {
        /// The tag that should have been there.
        expected: &'static str,
        /// Byte offset it should have been at.
        at: usize,
        /// What was there instead, as written.
        found: [u8; 4],
    },

    /// A required chunk was absent.
    ///
    /// `fmt ` and `data` are not at fixed offsets: a WAV may carry `LIST`,
    /// `fact` or padding chunks between them, so assuming the canonical
    /// 44-byte header reads metadata as samples.
    #[error("the file has no {0} chunk")]
    MissingChunk(&'static str),

    /// The sample format is one this crate deliberately does not handle.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// The channel count is zero, or more than this crate will downmix.
    #[error("{0} channels; expected mono or stereo")]
    UnsupportedChannels(u16),

    /// The declared sample rate is zero, which makes every duration undefined.
    #[error("sample rate is zero")]
    ZeroSampleRate,

    /// The data chunk does not hold a whole number of frames.
    #[error("data chunk is {bytes} bytes, not a multiple of {block_align} per frame")]
    RaggedData {
        /// Size of the data chunk.
        bytes: usize,
        /// Bytes per frame across all channels.
        block_align: usize,
    },

    /// The file could not be read at all.
    #[error("reading {path}: {source}")]
    Io {
        /// The path that failed.
        path: String,
        /// The underlying failure.
        source: std::io::Error,
    },
}
