//! The safetensors container: an 8-byte little-endian header length, that many
//! bytes of JSON, then a data segment addressed by byte offsets relative to its
//! own start.
//!
//! # What is (and isn't) here
//!
//! This module owns *addressing*: where a tensor's bytes are and whether the
//! file's own claims about them are self-consistent. It does not own meaning —
//! it has no idea what `decoder.conv_post.weight` is for. That belongs to
//! [`xabe_vits`](../../xabe_vits/index.html).
//!
//! The file is memory-mapped and never copied. [`StFile::tensor`] hands back a
//! borrowed `&[f32]` pointing into the mapping, so loading a 139 MB checkpoint
//! costs one `mmap` call and no allocation.
//!
//! Start at [`StFile::open`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::StError;

/// Byte width of the header-length prefix that opens every safetensors file.
const HEADER_LEN_PREFIX: usize = 8;

/// Element types this reader decodes.
///
/// The enum exists so an unexpected dtype is named in an error instead of being
/// reinterpreted as float and producing noise. Only [`Dtype::F32`] can be
/// borrowed without copying; the narrower two are widened on read, because this
/// card computes in f32 and every kernel in the workspace takes `&[f32]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 half precision. The Silero VAD's convolutions are stored this way.
    F16,
    /// Brain float: an f32 with the low 16 mantissa bits cut off.
    ///
    /// This card is Turing and has no bf16 arithmetic at all, so a bf16
    /// checkpoint can only be read by widening it here. The widening is exact -
    /// bf16 *is* the top half of an f32 - so nothing is lost by it.
    Bf16,
}

impl Dtype {
    /// Parses the header's dtype string, or `None` if unsupported.
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Width of one element in bytes.
    pub const fn width(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// The name safetensors uses, for error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
        }
    }
}

/// Where one tensor lives and what shape it claims to be.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Dimensions, outermost first, exactly as the header lists them.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: Dtype,
    /// Start offset within the data segment.
    start: u64,
    /// End offset within the data segment, exclusive.
    end: u64,
}

impl TensorInfo {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A memory-mapped safetensors file whose offsets have all been validated.
///
/// Construction is the only place bounds are checked; every accessor afterwards
/// can slice without re-verifying, because [`StFile::open`] has already refused
/// any file whose tensors do not fit.
pub struct StFile {
    path: PathBuf,
    map: memmap2::Mmap,
    /// Offset of the data segment within the mapping.
    data_start: usize,
    /// `BTreeMap` rather than a hash map so `tensors()` iterates in a stable
    /// order; weight-schema tests compare inventories and flapping order makes
    /// their failures unreadable.
    tensors: BTreeMap<String, TensorInfo>,
    /// The producer's `__metadata__` block, which is free-form string pairs.
    metadata: BTreeMap<String, String>,
}

/// Prints what the file *is*, not its 139 MB of contents: a panic in a test
/// that formats an unexpected `Ok(..)` should name the file, not dump a mapping.
impl std::fmt::Debug for StFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StFile")
            .field("path", &self.path)
            .field("tensors", &self.tensors.len())
            .field("data_bytes", &(self.map.len() - self.data_start))
            .finish()
    }
}

impl StFile {
    /// Opens and validates a safetensors file.
    ///
    /// Every tensor's byte range is checked against the data segment, and its
    /// declared shape against the bytes it spans, before this returns. A file
    /// that opens successfully cannot later panic on a slice.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|source| StError::Io {
            path: path.clone(),
            source,
        })?;
        // SAFETY: the mapping lives as long as `self`, and the file is opened
        // read-only. A concurrent truncation would be undefined, which is the
        // same contract every mmap-based model loader accepts.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| StError::Io {
            path: path.clone(),
            source,
        })?;

        let len = map.len() as u64;
        if map.len() < HEADER_LEN_PREFIX {
            return Err(StError::TooShort { path, len });
        }

        let header_len = u64::from_le_bytes(
            map[..HEADER_LEN_PREFIX]
                .try_into()
                .expect("slice is exactly 8 bytes"),
        );
        let data_start = HEADER_LEN_PREFIX as u64 + header_len;
        if data_start > len {
            return Err(StError::HeaderOverrun {
                path,
                header_len,
                len,
            });
        }

        // Reading the data segment as `f32` requires 4-byte alignment. The mmap
        // base is page-aligned, so only the segment offset can break it.
        if !data_start.is_multiple_of(4) {
            return Err(StError::Misaligned {
                what: "data segment".to_string(),
                offset: data_start,
            });
        }

        let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(
            &map[HEADER_LEN_PREFIX..data_start as usize],
        )
        .map_err(|source| StError::Header {
            path: path.clone(),
            source,
        })?;

        let data_len = len - data_start;
        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, value) in header {
            // Written by the producer to carry format metadata; not a tensor.
            // Kept rather than skipped: a converter that records the source
            // geometry there is the only thing that lets a schema check the
            // shapes it binds against what the original file declared.
            if name == "__metadata__" {
                if let serde_json::Value::Object(map) = value {
                    for (k, v) in map {
                        if let serde_json::Value::String(v) = v {
                            metadata.insert(k, v);
                        }
                    }
                }
                continue;
            }
            let info = parse_tensor(&name, &value, data_len)?;
            tensors.insert(name, info);
        }

        tracing::debug!(
            path = %path.display(),
            tensors = tensors.len(),
            data_bytes = data_len,
            "opened safetensors file"
        );

        Ok(Self {
            path,
            map,
            data_start: data_start as usize,
            tensors,
            metadata,
        })
    }

    /// The path this file was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of tensors in the file.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file declares no tensors at all.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterates tensor names and metadata in sorted order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Metadata for one tensor, or `None` if absent.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// One entry of the producer's `__metadata__` block.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Every `__metadata__` entry, in sorted order.
    pub fn metadata(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Borrows a tensor's data as `f32`, without copying.
    ///
    /// F32 only. A narrower tensor is refused by name rather than widened here,
    /// because widening allocates and this function promises not to - see
    /// [`StFile::tensor_f32`] for the copying version.
    pub fn tensor(&self, name: &str) -> Result<&[f32], StError> {
        let info = self.require(name)?;
        self.require_f32(name, info)?;
        Ok(self.slice(info))
    }

    /// Reads a tensor as `f32`, widening it if the file stores it narrower.
    ///
    /// Unlike [`StFile::tensor`] this always allocates, which is why it is a
    /// separate function rather than the default: a 6 GiB F32 checkpoint should
    /// not be copied just because some other checkpoint is F16.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, StError> {
        let info = self.require(name)?;
        Ok(self.widen(info))
    }

    /// Reads a tensor as `f32` and asserts its shape.
    pub fn tensor_f32_shaped(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, StError> {
        let info = self.require(name)?;
        Self::require_shape(name, info, expected)?;
        Ok(self.widen(info))
    }

    /// Looks a tensor up, or names the one that is missing.
    fn require(&self, name: &str) -> Result<&TensorInfo, StError> {
        self.tensors
            .get(name)
            .ok_or_else(|| StError::MissingTensor {
                name: name.to_string(),
            })
    }

    /// Refuses a tensor that cannot be borrowed as `f32`.
    fn require_f32(&self, name: &str, info: &TensorInfo) -> Result<(), StError> {
        if info.dtype != Dtype::F32 {
            return Err(StError::NotBorrowable {
                name: name.to_string(),
                dtype: info.dtype.name(),
            });
        }
        Ok(())
    }

    /// Checks a declared shape against an expected one.
    fn require_shape(name: &str, info: &TensorInfo, expected: &[usize]) -> Result<(), StError> {
        if info.shape != expected {
            return Err(StError::ShapeMismatch {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: info.shape.clone(),
            });
        }
        Ok(())
    }

    /// Copies a tensor into `f32`, widening F16 or BF16 on the way.
    fn widen(&self, info: &TensorInfo) -> Vec<f32> {
        let start = self.data_start + info.start as usize;
        let end = self.data_start + info.end as usize;
        let bytes = &self.map[start..end];
        match info.dtype {
            Dtype::F32 => self.slice(info).to_vec(),
            // Read pairwise rather than cast: a 2-byte element only needs
            // 2-byte alignment, and requiring 4 here would refuse files that
            // are perfectly readable.
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(half::f16::from_le_bytes(*b)))
                .collect(),
            // bf16 is the top half of an f32, so this is exact and needs no
            // rounding decision at all.
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
        }
    }

    /// Borrows a tensor and asserts its shape.
    ///
    /// Weight loading goes through this so a checkpoint of the wrong geometry
    /// fails while loading, named, instead of producing plausible-sounding
    /// noise at synthesis time.
    pub fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<&[f32], StError> {
        let info = self.require(name)?;
        self.require_f32(name, info)?;
        Self::require_shape(name, info, expected)?;
        Ok(self.slice(info))
    }

    /// Reinterprets a validated byte range as `f32`.
    fn slice(&self, info: &TensorInfo) -> &[f32] {
        let start = self.data_start + info.start as usize;
        let end = self.data_start + info.end as usize;
        let bytes = &self.map[start..end];
        // SAFETY: `open` verified that `end - start == numel * 4`, that the data
        // segment and this tensor both start 4-byte aligned, and the mmap base is
        // page-aligned - so the cast pointer is aligned. The mapping outlives the
        // returned slice. safetensors data is little-endian, which matches every
        // platform this is built for; a big-endian host would need a byte swap
        // here and is not supported.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
    }
}

/// Validates one header entry and converts it to a [`TensorInfo`].
fn parse_tensor(
    name: &str,
    value: &serde_json::Value,
    data_len: u64,
) -> Result<TensorInfo, StError> {
    #[derive(serde::Deserialize)]
    struct Raw {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [u64; 2],
    }

    let raw: Raw = serde_json::from_value(value.clone()).map_err(|source| StError::Header {
        path: PathBuf::from(name),
        source,
    })?;

    let dtype = Dtype::parse(&raw.dtype).ok_or_else(|| StError::UnsupportedDtype {
        name: name.to_string(),
        dtype: raw.dtype.clone(),
    })?;

    let [start, end] = raw.data_offsets;
    if end < start || end > data_len {
        return Err(StError::TensorOutOfBounds {
            name: name.to_string(),
            start,
            end,
            data_len,
        });
    }

    if !start.is_multiple_of(dtype.width()) {
        return Err(StError::Misaligned {
            what: format!("tensor {name}"),
            offset: start,
        });
    }

    let expected = raw.shape.iter().product::<usize>() as u64 * dtype.width();
    let actual = end - start;
    if expected != actual {
        return Err(StError::TensorSizeMismatch {
            name: name.to_string(),
            shape: raw.shape,
            dtype: raw.dtype,
            expected,
            actual,
        });
    }

    Ok(TensorInfo {
        shape: raw.shape,
        dtype,
        start,
        end,
    })
}
