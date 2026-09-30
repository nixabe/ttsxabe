//! Errors raised while opening or reading a torch checkpoint.
//!
//! Every variant names a specific way a `.pth` can be wrong, or a specific
//! thing this reader deliberately refuses. They exist so that a checkpoint this
//! crate cannot honestly read says which byte, tensor or opcode defeated it,
//! rather than producing a state dict that is quietly missing half its weights.

use std::path::PathBuf;

/// A torch checkpoint could not be opened, parsed, or trusted.
#[derive(Debug, thiserror::Error)]
pub enum PtError {
    /// The file could not be opened or memory-mapped.
    #[error("cannot open {path}: {source}")]
    Io {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// The file has no zip end-of-central-directory record, so it is not a
    /// zip at all.
    ///
    /// This is what a **pre-1.6** torch file produces: those are a bare pickle
    /// stream with the storages appended, not an archive. They cannot be read
    /// here and the message says so, because the difference is invisible from
    /// the extension - both are called `.pth`.
    #[error(
        "{path} is not a zip archive: torch files written before 1.6 are a bare pickle stream and cannot be read here"
    )]
    NotAnArchive {
        /// The file that is not an archive.
        path: PathBuf,
    },

    /// The archive is structurally malformed: a header runs past the end of the
    /// file, or a record does not carry its own signature.
    #[error("{path} is a malformed zip: {what}")]
    MalformedArchive {
        /// The file whose structure is wrong.
        path: PathBuf,
        /// Which record, and how.
        what: String,
    },

    /// An entry is compressed. Torch writes every entry stored, precisely so
    /// that storages can be mapped rather than inflated; a compressed one means
    /// the file was rewritten by something else and its tensors cannot be
    /// borrowed from the mapping.
    #[error(
        "{name} is compressed (method {method}); this reader maps entries and cannot inflate them"
    )]
    Compressed {
        /// The offending entry.
        name: String,
        /// The zip compression method it declares.
        method: u16,
    },

    /// The archive holds no `data.pkl`, so it is a zip but not a torch save.
    #[error("{path} contains no data.pkl entry, so it is not a torch checkpoint")]
    NoPickle {
        /// The archive that is missing it.
        path: PathBuf,
    },

    /// The archive declares big-endian storages. Every tensor here would need
    /// a byte swap, which would mean copying rather than borrowing, so it is
    /// refused rather than silently mis-read.
    #[error("{path} declares {byteorder}-endian storages; only little-endian is read")]
    Byteorder {
        /// The archive.
        path: PathBuf,
        /// What it declared.
        byteorder: String,
    },

    /// The pickle stream ended without a `STOP`, or ran off its own end.
    #[error("data.pkl is truncated at byte {at}")]
    PickleTruncated {
        /// How far the reader got.
        at: usize,
    },

    /// An opcode this reader does not implement. Named by byte and by mnemonic
    /// where one is known, because the fix is always to implement it rather
    /// than to guess what it would have pushed.
    #[error("data.pkl uses unsupported pickle opcode {opcode:#04x} ({name}) at byte {at}")]
    PickleOpcode {
        /// The opcode byte.
        opcode: u8,
        /// Its mnemonic, or `unknown`.
        name: &'static str,
        /// Where it appeared.
        at: usize,
    },

    /// The pickle stack, memo or mark stack was in a state the opcode could not
    /// act on. A well-formed pickle never does this, so it means the stream is
    /// corrupt or this reader has a bug.
    #[error("data.pkl is inconsistent at byte {at}: {what}")]
    PickleState {
        /// Where it went wrong.
        at: usize,
        /// What was expected and what was found.
        what: String,
    },

    /// A `GLOBAL` naming something this reader will not call.
    ///
    /// Reconstructing a state dict needs exactly three: `collections.OrderedDict`,
    /// `torch._utils._rebuild_tensor_v2` and a storage class. Anything else is a
    /// pickled *object graph* rather than a state dict, and executing it would
    /// need the model's own class definitions - which is the line this crate
    /// does not cross.
    #[error("data.pkl calls {module}.{name}, which is not part of a state dict")]
    UnsupportedGlobal {
        /// The module it was imported from.
        module: String,
        /// The attribute name.
        name: String,
    },

    /// A storage class whose element type this reader does not decode.
    #[error("storage {key} is a {class}, whose element type is not read here")]
    UnsupportedStorage {
        /// The storage's key within `data/`.
        key: String,
        /// The class the pickle named.
        class: String,
    },

    /// The requested section is absent, or the root object is not a mapping.
    #[error("{path} has no {section} section holding a state dict")]
    NoSection {
        /// The archive.
        path: PathBuf,
        /// The section that was asked for.
        section: String,
    },

    /// A value in the state dict is not a tensor.
    #[error("{name} in the state dict is {found}, not a tensor")]
    NotATensor {
        /// The offending key.
        name: String,
        /// What was there instead.
        found: &'static str,
    },

    /// A tensor names a storage the archive does not hold.
    #[error("tensor {name} refers to storage {key}, which the archive does not contain")]
    MissingStorage {
        /// The tensor.
        name: String,
        /// The storage key it wanted.
        key: String,
    },

    /// A tensor is a strided *view* rather than a contiguous block.
    ///
    /// Reading one as if it were contiguous would silently permute or repeat
    /// elements, so it is refused. No published checkpoint saves views - torch
    /// makes tensors contiguous on save - which is why this is an error rather
    /// than a copying slow path.
    #[error("tensor {name} has shape {shape:?} with stride {stride:?}, which is not contiguous")]
    NotContiguous {
        /// The tensor.
        name: String,
        /// Its shape.
        shape: Vec<usize>,
        /// The stride it declared.
        stride: Vec<usize>,
    },

    /// A tensor's window lies partly outside the storage it borrows from.
    #[error("tensor {name} spans elements {start}..{end} of a {numel}-element storage")]
    OutOfBounds {
        /// The tensor.
        name: String,
        /// First element of its window.
        start: usize,
        /// One past the last.
        end: usize,
        /// Elements the storage holds.
        numel: usize,
    },

    /// A storage entry's data does not start on a boundary its element type can
    /// be read from.
    ///
    /// Torch aligns storages to 64 bytes and records that in
    /// `.storage_alignment`, so this should never fire - but nothing in the zip
    /// format forces it, and casting a misaligned pointer is undefined
    /// behaviour rather than a wrong number. Checked, not assumed.
    #[error("storage {key} starts at byte {offset}, which is not {width}-byte aligned")]
    Misaligned {
        /// The storage.
        key: String,
        /// Where its data begins in the file.
        offset: usize,
        /// The alignment its element type needs.
        width: usize,
    },

    /// A tensor was asked for by a name the checkpoint does not have.
    #[error("no tensor named {name}")]
    MissingTensor {
        /// The name that was asked for.
        name: String,
    },

    /// A tensor's declared shape is not the one the caller required.
    #[error("tensor {name} is {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// The offending tensor.
        name: String,
        /// The shape the caller required.
        expected: Vec<usize>,
        /// The shape the file declares.
        actual: Vec<usize>,
    },

    /// A tensor stored narrower than `f32` cannot be borrowed without copying.
    #[error("tensor {name} is {dtype} and cannot be borrowed as f32; read it with tensor_f32")]
    NotBorrowable {
        /// The offending tensor.
        name: String,
        /// The width it is stored at.
        dtype: &'static str,
    },
}
