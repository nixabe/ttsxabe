//! Checkpoints split across several files.
//!
//! Anything above a few gigabytes ships sharded: an index naming which file
//! each tensor lives in, and `model-0000N-of-0000M.safetensors` beside it. The
//! ASR is two shards and the translator six, so this is not optional for
//! anything past the synthesiser.
//!
//! The index is the *only* thing consulted for placement. A tensor named in the
//! index but absent from its shard is an error, and so is a tensor present in a
//! shard but absent from the index - a checkpoint that half-agrees with its own
//! manifest is one that will load and be wrong somewhere specific.
//!
//! Every shard is mapped, so a 6 GiB checkpoint costs `M` `mmap` calls and no
//! allocation. Start at [`StSet::open`].

use crate::error::StError;
use crate::file::{Dtype, StFile, TensorInfo};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The manifest a sharded checkpoint ships with.
const INDEX: &str = "model.safetensors.index.json";

/// The single-file name a checkpoint uses when it is not sharded.
const SINGLE: &str = "model.safetensors";

/// One or more safetensors files addressed as a single checkpoint.
///
/// A single-file checkpoint is one shard, so callers do not branch on it.
pub struct StSet {
    /// Where the checkpoint was opened from.
    root: PathBuf,
    /// The shards, in the order the index first mentions them.
    shards: Vec<StFile>,
    /// Tensor name to the shard holding it.
    placement: BTreeMap<String, usize>,
}

impl std::fmt::Debug for StSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StSet")
            .field("root", &self.root)
            .field("shards", &self.shards.len())
            .field("tensors", &self.placement.len())
            .finish()
    }
}

impl StSet {
    /// Opens a checkpoint directory, sharded or not.
    ///
    /// Given a directory this looks for the index and falls back to the
    /// single-file name. Given a file it opens exactly that file, so a caller
    /// that already knows the path does not have to construct a directory.
    pub fn open(path: impl AsRef<Path>) -> Result<StSet, StError> {
        let path = path.as_ref();
        if path.is_file() {
            let shard = StFile::open(path)?;
            return Ok(Self::single(path.to_path_buf(), shard));
        }

        let index = path.join(INDEX);
        if !index.is_file() {
            let single = path.join(SINGLE);
            let shard = StFile::open(&single)?;
            return Ok(Self::single(path.to_path_buf(), shard));
        }
        Self::from_index(path, &index)
    }

    /// Wraps one already-opened file as a set of one.
    fn single(root: PathBuf, shard: StFile) -> StSet {
        let placement = shard
            .tensors()
            .map(|(name, _)| (name.to_string(), 0))
            .collect();
        StSet {
            root,
            shards: vec![shard],
            placement,
        }
    }

    /// Opens every shard the index names.
    fn from_index(root: &Path, index: &Path) -> Result<StSet, StError> {
        let text = std::fs::read_to_string(index).map_err(|source| StError::Io {
            path: index.to_path_buf(),
            source,
        })?;
        let parsed: Index = serde_json::from_str(&text).map_err(|source| StError::Header {
            path: index.to_path_buf(),
            source,
        })?;

        // Opened in first-mention order rather than sorted, so the shard
        // indices in any diagnostic match the order the index reads in.
        let mut files: Vec<String> = Vec::new();
        for file in parsed.weight_map.values() {
            if !files.iter().any(|f| f == file) {
                files.push(file.clone());
            }
        }

        let mut shards = Vec::with_capacity(files.len());
        for file in &files {
            shards.push(StFile::open(root.join(file))?);
        }

        let mut placement = BTreeMap::new();
        for (name, file) in &parsed.weight_map {
            let at = files
                .iter()
                .position(|f| f == file)
                .expect("every file in the map is in the list built from it");
            if shards[at].info(name).is_none() {
                return Err(StError::MissingTensor {
                    name: format!("{name} (the index places it in {file})"),
                });
            }
            placement.insert(name.clone(), at);
        }

        // A tensor in a shard but not in the index would be silently
        // unreachable, and a schema that expected it would report it missing
        // from a checkpoint that in fact contains it.
        for (at, shard) in shards.iter().enumerate() {
            for (name, _) in shard.tensors() {
                if !placement.contains_key(name) {
                    return Err(StError::UnindexedTensor {
                        name: name.to_string(),
                        file: files[at].clone(),
                    });
                }
            }
        }

        // The index declares the total; a mismatch means shards from different
        // exports, which loads fine and produces a model that is subtly not the
        // one that was published.
        if let Some(declared) = parsed.metadata.and_then(|m| m.total_size) {
            let actual: u64 = shards
                .iter()
                .flat_map(|s| s.tensors())
                .map(|(_, i)| i.numel() as u64 * i.dtype.width())
                .sum();
            if declared != actual {
                return Err(StError::SizeDisagreement { declared, actual });
            }
        }

        tracing::debug!(
            root = %root.display(),
            shards = shards.len(),
            tensors = placement.len(),
            "opened a sharded checkpoint",
        );

        Ok(StSet {
            root: root.to_path_buf(),
            shards,
            placement,
        })
    }

    /// Where this checkpoint was opened from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// How many files the checkpoint is split across.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// How many tensors it holds.
    pub fn len(&self) -> usize {
        self.placement.len()
    }

    /// Whether it holds no tensors at all.
    pub fn is_empty(&self) -> bool {
        self.placement.is_empty()
    }

    /// Every tensor name and its metadata, in sorted order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.placement.iter().map(|(name, at)| {
            (
                name.as_str(),
                self.shards[*at]
                    .info(name)
                    .expect("placement was checked at open"),
            )
        })
    }

    /// Metadata for one tensor, or `None` if absent.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let at = *self.placement.get(name)?;
        self.shards[at].info(name)
    }

    /// One `__metadata__` entry, from whichever shard carries it.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.shards.iter().find_map(|s| s.meta(key))
    }

    /// Borrows a tensor's data as `f32`, without copying. F32 only.
    pub fn tensor(&self, name: &str) -> Result<&[f32], StError> {
        self.shard_of(name)?.tensor(name)
    }

    /// Borrows a tensor and asserts its shape. F32 only.
    pub fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<&[f32], StError> {
        self.shard_of(name)?.tensor_shaped(name, expected)
    }

    /// Reads a tensor as `f32`, widening F16 or BF16.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, StError> {
        self.shard_of(name)?.tensor_f32(name)
    }

    /// Reads a tensor as raw f16 bits. See [`StFile::tensor_f16`].
    pub fn tensor_f16(&self, name: &str) -> Result<Vec<u16>, StError> {
        self.shard_of(name)?.tensor_f16(name)
    }

    /// Reads a tensor as `f32` and asserts its shape.
    pub fn tensor_f32_shaped(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, StError> {
        self.shard_of(name)?.tensor_f32_shaped(name, expected)
    }

    /// Total elements across every tensor.
    pub fn total_elements(&self) -> usize {
        self.tensors().map(|(_, i)| i.numel()).sum()
    }

    /// The set of dtypes present, for reporting what a checkpoint costs.
    pub fn dtypes(&self) -> Vec<Dtype> {
        let mut seen: Vec<Dtype> = Vec::new();
        for (_, info) in self.tensors() {
            if !seen.contains(&info.dtype) {
                seen.push(info.dtype);
            }
        }
        seen
    }

    /// The shard holding a tensor, or an error naming the tensor.
    fn shard_of(&self, name: &str) -> Result<&StFile, StError> {
        let at = self
            .placement
            .get(name)
            .ok_or_else(|| StError::MissingTensor {
                name: name.to_string(),
            })?;
        Ok(&self.shards[*at])
    }
}

/// `model.safetensors.index.json`.
#[derive(serde::Deserialize)]
struct Index {
    #[serde(default)]
    metadata: Option<IndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

/// The index's own summary of what it points at.
#[derive(serde::Deserialize)]
struct IndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}
