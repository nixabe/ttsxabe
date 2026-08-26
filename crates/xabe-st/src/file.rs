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

/// Element types this reader decodes. The VITS checkpoints are pure F32; the
/// enum exists so an unexpected dtype is named in an error instead of being
/// reinterpreted as float and producing noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE-754 single precision.
    F32,
}

impl Dtype {
    /// Parses the header's dtype string, or `None` if unsupported.
    fn parse(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            _ => None,
        }
    }

    /// Width of one element in bytes.
    const fn width(self) -> u64 {
        match self {
            Self::F32 => 4,
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
        for (name, value) in header {
            // Written by the producer to carry format metadata; not a tensor.
            if name == "__metadata__" {
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

    /// Borrows a tensor's data as `f32`, without copying.
    pub fn tensor(&self, name: &str) -> Result<&[f32], StError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::MissingTensor {
                name: name.to_string(),
            })?;
        Ok(self.slice(info))
    }

    /// Borrows a tensor and asserts its shape.
    ///
    /// Weight loading goes through this so a checkpoint of the wrong geometry
    /// fails while loading, named, instead of producing plausible-sounding
    /// noise at synthesis time.
    pub fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<&[f32], StError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::MissingTensor {
                name: name.to_string(),
            })?;
        if info.shape != expected {
            return Err(StError::ShapeMismatch {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: info.shape.clone(),
            });
        }
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
