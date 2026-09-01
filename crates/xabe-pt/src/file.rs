//! A torch checkpoint, mapped and addressed.
//!
//! # What is (and isn't) here
//!
//! This module owns *addressing*: which archive entry a tensor's elements live
//! in, where in it they start, and whether the checkpoint's own claims about
//! them hold together. It has no idea what `waveform_decoder.ups.0` is for -
//! that belongs to `xabe-vits`.
//!
//! The file is memory-mapped and never copied. [`PtFile::tensor`] hands back a
//! borrowed `&[f32]` pointing into the mapping, exactly as `xabe-st` does for
//! safetensors, so the two containers are interchangeable to a weight schema.
//!
//! # A checkpoint is not only a state dict
//!
//! A `.pth` written by a trainer holds the optimiser, the scheduler and the
//! run's own config beside the weights, and the weights are one entry in that
//! mapping rather than the whole file. [`PtFile::open_section`] names which
//! entry to read; the 949 tensors of the Coqui VITS checkpoint sit under
//! `model`, beside 2847 more belonging to the optimiser that inference never
//! touches.
//!
//! Start at [`PtFile::open_section`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::PtError;
use crate::pickle::{self, Value};
use crate::zip;

pub use crate::pickle::Dtype;

/// Where one tensor lives and what shape it claims to be.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: Dtype,
    /// Start offset within the whole mapping.
    start: usize,
    /// End offset, exclusive.
    end: usize,
}

impl TensorInfo {
    /// Total element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A memory-mapped torch checkpoint whose tensors have all been validated.
///
/// Construction is the only place bounds and alignment are checked; every
/// accessor afterwards can slice without re-verifying, because
/// [`PtFile::open_section`] has already refused any file whose tensors do not
/// fit the storages they name.
pub struct PtFile {
    path: PathBuf,
    map: memmap2::Mmap,
    /// `BTreeMap` rather than a hash map so `tensors()` iterates in a stable
    /// order; weight-schema tests compare inventories and flapping order makes
    /// their failures unreadable.
    tensors: BTreeMap<String, TensorInfo>,
}

/// Prints what the file *is*, not its 950 MB of contents.
impl std::fmt::Debug for PtFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtFile")
            .field("path", &self.path)
            .field("tensors", &self.tensors.len())
            .field("bytes", &self.map.len())
            .finish()
    }
}

impl PtFile {
    /// Opens a checkpoint whose root object is itself the state dict.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PtError> {
        Self::read(path.as_ref(), None)
    }

    /// Opens a checkpoint and reads the state dict stored under one key.
    ///
    /// A trainer's checkpoint keeps the weights beside the optimiser state, so
    /// the section has to be named: `model` for a Coqui save. The optimiser's
    /// own tensors are never bound, and their storages are never touched.
    pub fn open_section(path: impl AsRef<Path>, section: &str) -> Result<Self, PtError> {
        Self::read(path.as_ref(), Some(section))
    }

    /// Maps the archive, runs its pickle, and binds every tensor it names.
    fn read(path: &Path, section: Option<&str>) -> Result<Self, PtError> {
        let path = path.to_path_buf();
        let file = std::fs::File::open(&path).map_err(|source| PtError::Io {
            path: path.clone(),
            source,
        })?;
        // SAFETY: the mapping lives as long as `self`, and the file is opened
        // read-only. A concurrent truncation would be undefined, which is the
        // same contract every mmap-based model loader accepts.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| PtError::Io {
            path: path.clone(),
            source,
        })?;

        let entries = zip::entries(&path, &map)?;
        let by_name: BTreeMap<&str, &zip::Entry> =
            entries.iter().map(|e| (e.name.as_str(), e)).collect();

        // Torch names the archive after the file it was saved to, so the
        // directory prefix is whatever precedes `data.pkl` rather than a
        // constant.
        let pickle_entry = by_name
            .keys()
            .find(|n| **n == "data.pkl" || n.ends_with("/data.pkl"))
            .copied()
            .ok_or_else(|| PtError::NoPickle { path: path.clone() })?;
        let prefix = &pickle_entry[..pickle_entry.len() - "data.pkl".len()];

        // Written since torch 1.7. Absent means little-endian, which every
        // platform this builds for is; present and anything else would need a
        // byte swap per element and is refused rather than mis-read.
        if let Some(entry) = by_name.get(format!("{prefix}byteorder").as_str()) {
            let order = String::from_utf8_lossy(&map[entry.start..entry.start + entry.len]);
            let order = order.trim();
            if order != "little" {
                return Err(PtError::Byteorder {
                    path,
                    byteorder: order.to_string(),
                });
            }
        }

        let entry = by_name[pickle_entry];
        let root = pickle::load(&map[entry.start..entry.start + entry.len])?;

        let state = match section {
            None => root,
            Some(name) => root.get(name).ok_or_else(|| PtError::NoSection {
                path: path.clone(),
                section: name.to_string(),
            })?,
        };
        let entries_of_state = state.as_dict().ok_or_else(|| PtError::NoSection {
            path: path.clone(),
            section: section.unwrap_or("root").to_string(),
        })?;

        let mut tensors = BTreeMap::new();
        for (key, value) in entries_of_state.iter() {
            let Value::Str(name) = key else {
                continue;
            };
            let info = bind(name, value, prefix, &by_name)?;
            tensors.insert(name.to_string(), info);
        }
        drop(entries_of_state);

        tracing::debug!(
            path = %path.display(),
            tensors = tensors.len(),
            section = section.unwrap_or("root"),
            "opened torch checkpoint",
        );

        Ok(Self { path, map, tensors })
    }

    /// The path this file was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of tensors bound.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the section held no tensors at all.
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
    ///
    /// F32 only. A narrower tensor is refused by name rather than widened here,
    /// because widening allocates and this function promises not to.
    pub fn tensor(&self, name: &str) -> Result<&[f32], PtError> {
        let info = self.require(name)?;
        self.require_f32(name, info)?;
        Ok(self.slice(info))
    }

    /// Borrows a tensor and asserts its shape.
    ///
    /// Weight loading goes through this so a checkpoint of the wrong geometry
    /// fails while loading, named, instead of producing plausible-sounding
    /// noise at synthesis time.
    pub fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<&[f32], PtError> {
        let info = self.require(name)?;
        self.require_f32(name, info)?;
        Self::require_shape(name, info, expected)?;
        Ok(self.slice(info))
    }

    /// Reads a tensor as `f32`, widening it if the file stores it narrower.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, PtError> {
        let info = self.require(name)?;
        let bytes = &self.map[info.start..info.end];
        Ok(match info.dtype {
            Dtype::F32 => self.slice(info).to_vec(),
            // Read pairwise rather than cast: a 2-byte element only needs
            // 2-byte alignment, and requiring 4 here would refuse tensors that
            // are perfectly readable.
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(half::f16::from_le_bytes(*b)))
                .collect(),
            // bf16 is the top half of an f32, so this is exact.
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
        })
    }

    /// Looks a tensor up, or names the one that is missing.
    fn require(&self, name: &str) -> Result<&TensorInfo, PtError> {
        self.tensors
            .get(name)
            .ok_or_else(|| PtError::MissingTensor {
                name: name.to_string(),
            })
    }

    /// Refuses a tensor that cannot be borrowed as `f32`.
    fn require_f32(&self, name: &str, info: &TensorInfo) -> Result<(), PtError> {
        if info.dtype != Dtype::F32 {
            return Err(PtError::NotBorrowable {
                name: name.to_string(),
                dtype: info.dtype.name(),
            });
        }
        Ok(())
    }

    /// Checks a declared shape against an expected one.
    fn require_shape(name: &str, info: &TensorInfo, expected: &[usize]) -> Result<(), PtError> {
        if info.shape != expected {
            return Err(PtError::ShapeMismatch {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: info.shape.clone(),
            });
        }
        Ok(())
    }

    /// Reinterprets a validated byte range as `f32`.
    fn slice(&self, info: &TensorInfo) -> &[f32] {
        let bytes = &self.map[info.start..info.end];
        // SAFETY: `read` verified that this range lies inside the entry, that
        // the entry's data begins 4-byte aligned, and that the tensor's own
        // element offset is a whole number of elements - so the cast pointer is
        // aligned. The mmap base is page-aligned and the mapping outlives the
        // returned slice. Torch storages are little-endian, which `read` also
        // checked.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) }
    }
}

/// Resolves one state-dict entry to a byte range, or says why it cannot.
fn bind(
    name: &str,
    value: &Value,
    prefix: &str,
    by_name: &BTreeMap<&str, &zip::Entry>,
) -> Result<TensorInfo, PtError> {
    let Value::Tensor(t) = value else {
        return Err(PtError::NotATensor {
            name: name.to_string(),
            found: value.kind(),
        });
    };

    // A saved tensor is contiguous because torch makes it so on the way out.
    // A view would still have a plausible shape and would read its elements in
    // the wrong order, so the stride is checked rather than assumed. Axes of
    // length one are skipped: their stride is arbitrary and unobservable.
    let mut expected = 1usize;
    for axis in (0..t.shape.len()).rev() {
        if t.shape[axis] > 1 && t.stride[axis] != expected {
            return Err(PtError::NotContiguous {
                name: name.to_string(),
                shape: t.shape.clone(),
                stride: t.stride.clone(),
            });
        }
        expected *= t.shape[axis];
    }
    let numel: usize = t.shape.iter().product();

    let end = t.offset + numel;
    if end > t.storage.numel {
        return Err(PtError::OutOfBounds {
            name: name.to_string(),
            start: t.offset,
            end,
            numel: t.storage.numel,
        });
    }

    let entry = by_name
        .get(format!("{prefix}data/{}", t.storage.key).as_str())
        .ok_or_else(|| PtError::MissingStorage {
            name: name.to_string(),
            key: t.storage.key.clone(),
        })?;

    let width = t.dtype_width();
    if !entry.start.is_multiple_of(width) {
        return Err(PtError::Misaligned {
            key: t.storage.key.clone(),
            offset: entry.start,
            width,
        });
    }
    // The storage's own entry has to be long enough for what the pickle said it
    // holds; a truncated download shortens the entry, not the claim.
    if entry.len < t.storage.numel * width {
        return Err(PtError::OutOfBounds {
            name: name.to_string(),
            start: t.offset,
            end,
            numel: entry.len / width,
        });
    }

    Ok(TensorInfo {
        shape: t.shape.clone(),
        dtype: t.storage.dtype,
        start: entry.start + t.offset * width,
        end: entry.start + end * width,
    })
}

impl pickle::Tensor {
    /// Width of one of this tensor's elements, in bytes.
    fn dtype_width(&self) -> usize {
        self.storage.dtype.width()
    }
}
