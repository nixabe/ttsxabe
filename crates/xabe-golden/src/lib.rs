//! Reads the golden captures produced by `tools/oracle/capture.py`.
//!
//! The reference implementation is the definition of correct for this project,
//! and it is *captured*, not described. This crate is the reading half of that
//! arrangement: it opens a capture directory, validates it against its own
//! manifest, and hands stages back as plain slices for a differential test to
//! diff.
//!
//! # Why the checksums are verified on read
//!
//! A capture is a directory of headerless binary files whose meaning lives
//! entirely in a sidecar JSON. Nothing about a truncated `.bin` looks wrong: it
//! reads as a shorter tensor, and a shorter tensor produces a length mismatch
//! at some later, unrelated stage. Verifying the recorded SHA-256 on read turns
//! that into [`GoldenError::Corrupt`], naming the file, at the moment it is
//! opened. Captures are small - the largest here is 160 kB - so this costs
//! nothing worth measuring.
//!
//! # Layout
//!
//! ```text
//! .golden/base/
//!   manifest.json     provenance + one entry per tensor
//!   input_ids.bin     raw little-endian, C order, no header
//!   enc_out.bin
//!   ...
//! ```
//!
//! `.golden/` is gitignored: it is regenerable and it is large.

mod compare;
mod error;
mod manifest;

pub use compare::Comparison;
pub use error::GoldenError;
pub use manifest::{Manifest, TensorInfo};

use sha2::Digest;
use std::path::{Path, PathBuf};

/// One capture directory, opened and validated.
#[derive(Debug)]
pub struct Golden {
    dir: PathBuf,
    manifest: Manifest,
}

impl Golden {
    /// Opens a capture directory and parses its manifest.
    ///
    /// The individual `.bin` files are not read here - only when a stage is
    /// asked for - so opening a capture to check its provenance is cheap.
    pub fn open(dir: &Path) -> Result<Self, GoldenError> {
        let path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&path).map_err(|source| GoldenError::Io {
            path: path.clone(),
            source,
        })?;
        let manifest: Manifest =
            serde_json::from_str(&text).map_err(|source| GoldenError::Manifest { path, source })?;

        tracing::debug!(
            dir = %dir.display(),
            model = %manifest.model,
            transformers = %manifest.transformers,
            seed = manifest.seed,
            stages = manifest.tensors.len(),
            "opened capture",
        );

        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    /// Opens the capture named by `XABE_GOLDEN`, or `.golden/base` beside the
    /// workspace root.
    ///
    /// Returns `None` rather than an error when the directory is absent:
    /// captures are regenerable artefacts that a fresh checkout will not have,
    /// and a test that cannot find one should say so and skip rather than fail
    /// for a reason unrelated to the code under test.
    pub fn open_default() -> Option<Self> {
        let dir = match std::env::var("XABE_GOLDEN") {
            Ok(p) => PathBuf::from(p),
            Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".golden/base"),
        };
        if !dir.join("manifest.json").is_file() {
            return None;
        }
        match Self::open(&dir) {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!(error = %e, "capture present but unreadable");
                None
            }
        }
    }

    /// The capture's provenance and inventory.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The directory this capture was read from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The stage names present, in sorted order.
    pub fn stages(&self) -> Vec<&str> {
        self.manifest.tensors.keys().map(String::as_str).collect()
    }

    /// One stage's description.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, GoldenError> {
        self.manifest
            .tensors
            .get(name)
            .ok_or_else(|| GoldenError::NoSuchTensor {
                capture: self.dir.clone(),
                name: name.to_string(),
                available: self
                    .manifest
                    .tensors
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// One stage's shape.
    pub fn shape(&self, name: &str) -> Result<&[usize], GoldenError> {
        Ok(&self.info(name)?.shape)
    }

    /// Reads one stage's file, checking its length and checksum.
    fn bytes(&self, name: &str) -> Result<Vec<u8>, GoldenError> {
        let info = self.info(name)?;
        let path = self.dir.join(&info.file);
        let raw = std::fs::read(&path).map_err(|source| GoldenError::Io { path, source })?;

        let item = info.item_size().ok_or_else(|| GoldenError::UnknownDtype {
            name: name.to_string(),
            dtype: info.dtype.clone(),
        })?;
        let expected = info.numel() * item;
        if raw.len() != expected {
            return Err(GoldenError::SizeMismatch {
                name: name.to_string(),
                shape: info.shape.clone(),
                expected,
                actual: raw.len(),
            });
        }

        let digest = sha2::Sha256::digest(&raw);
        if format!("{digest:x}") != info.sha256 {
            return Err(GoldenError::Corrupt {
                name: name.to_string(),
            });
        }

        Ok(raw)
    }

    /// Reads one stage as `f32`.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>, GoldenError> {
        let info = self.info(name)?;
        if info.dtype != "f32" {
            return Err(GoldenError::WrongDtype {
                name: name.to_string(),
                actual: info.dtype.clone(),
                wanted: "f32".to_string(),
            });
        }
        let raw = self.bytes(name)?;
        // Built by copy from explicit little-endian chunks rather than by
        // transmuting the buffer: `Vec<u8>` carries no alignment guarantee, and
        // the capture is little-endian by definition, not by host convention.
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect())
    }

    /// Reads one stage as `i64`. Only `input_ids` is stored this way.
    pub fn i64s(&self, name: &str) -> Result<Vec<i64>, GoldenError> {
        let info = self.info(name)?;
        if info.dtype != "i64" {
            return Err(GoldenError::WrongDtype {
                name: name.to_string(),
                actual: info.dtype.clone(),
                wanted: "i64".to_string(),
            });
        }
        let raw = self.bytes(name)?;
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .copied()
            .map(i64::from_le_bytes)
            .collect())
    }

    /// Diffs a computed stage against the captured one.
    ///
    /// Judged at `atol + rtol * |expected|`; see [`Comparison`] for what comes
    /// back and why it carries more than a boolean.
    pub fn compare(
        &self,
        name: &str,
        actual: &[f32],
        atol: f32,
        rtol: f32,
    ) -> Result<Comparison, GoldenError> {
        let expected = self.f32s(name)?;
        if expected.len() != actual.len() {
            return Err(GoldenError::LengthMismatch {
                name: name.to_string(),
                expected: expected.len(),
                actual: actual.len(),
            });
        }
        Ok(Comparison::new(name, &expected, actual, atol, rtol))
    }
}
