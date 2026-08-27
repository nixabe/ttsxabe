//! The container: header, metadata store, tensor directory, tensor bytes.
//!
//! Field order is transcribed from the format comment in `ggml/src/gguf.cpp`.
//! A v3 file is: the magic, a `u32` version, a `u64` tensor count, a `u64`
//! metadata count, that many key-value pairs, that many tensor-info records,
//! then padding up to `general.alignment` and the data section.

use crate::error::GgufError;
use crate::reader::Cursor;
use crate::types::GgmlType;
use crate::value::{GgufArray, GgufValue};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// What the format uses when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u32 = 32;

/// The only version this reader has been checked against.
const GGUF_VERSION: u32 = 3;

/// ggml's fixed maximum rank.
const GGML_MAX_DIMS: u32 = 4;

/// One entry from the tensor directory.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// The name, e.g. `blk.3.attn_q.weight`.
    pub name: String,
    /// Dimensions in **ggml order**: `dims[0]` varies fastest.
    ///
    /// This is the transpose of how the same matrix is written in a
    /// safetensors header, and it is the single easiest thing to get wrong
    /// about this format. A `[4096, 128256]` here is the `[128256, 4096]`
    /// row-major matrix a safetensors file would declare. [`Self::shape`]
    /// does the flip; this field is what the file said.
    pub dims: Vec<u64>,
    /// The element type.
    pub ggml_type: GgmlType,
    /// Byte offset within the data section, not an absolute file position.
    pub offset: u64,
    /// Product of `dims`.
    pub n_elements: u64,
    /// `n_elements * ggml_type.byte_size()`.
    pub n_bytes: u64,
}

impl TensorInfo {
    /// The shape in row-major order, so it can be compared against a
    /// safetensors shape or a geometry written the way the reference is.
    ///
    /// Purely `dims` reversed. It exists so that every caller does not have to
    /// remember to reverse, and so that forgetting is a compile-time absence
    /// rather than a silently transposed weight.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

enum Backing {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Owned(v) => v,
        }
    }
}

/// A parsed GGUF file with zero-copy access to each tensor's bytes.
///
/// Holds no owned copy of the weights: [`Self::open`] maps the file, so
/// opening a 16 GB checkpoint touches the pages the metadata scan needs and
/// nothing else.
pub struct GgufFile {
    backing: Backing,
    path: PathBuf,
    version: u32,
    alignment: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    data_offset: u64,
}

struct Parsed {
    version: u32,
    alignment: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

impl GgufFile {
    /// Maps and parses the GGUF file at `path`.
    ///
    /// # Safety of the mapping
    ///
    /// `Mmap::map` is unsafe because the OS cannot promise the file is not
    /// truncated or rewritten by another process while mapped, which would
    /// raise `SIGBUS` on access. That is the same trade `xabe-st` makes and
    /// the same one llama.cpp makes; the alternative is copying 16 GB.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| GgufError::Io {
            path: path.clone(),
            source,
        })?;
        Self::from_backing(Backing::Mmap(mmap), path)
    }

    /// Parses a GGUF image already in memory.
    ///
    /// Only the tests use this, to build a small file by hand rather than
    /// writing and mapping a temporary one.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        Self::from_backing(Backing::Owned(bytes), PathBuf::from("<memory>"))
    }

    fn from_backing(backing: Backing, path: PathBuf) -> Result<Self, GgufError> {
        let parsed = parse(&backing)?;
        let file_len = backing.len() as u64;

        // Every tensor is checked against the file's real length here, once,
        // so `tensor_bytes` can slice without re-validating and without
        // returning an Option nobody would know how to handle.
        for t in &parsed.tensors {
            let start = parsed
                .data_offset
                .checked_add(t.offset)
                .ok_or_else(|| GgufError::OffsetOverflow(t.name.clone()))?;
            let end = start
                .checked_add(t.n_bytes)
                .ok_or_else(|| GgufError::OffsetOverflow(t.name.clone()))?;
            if end > file_len {
                return Err(GgufError::TensorOutOfBounds {
                    name: t.name.clone(),
                    start,
                    end,
                    file_len,
                });
            }
        }

        let index = parsed
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        tracing::debug!(
            tensors = parsed.tensors.len(),
            metadata = parsed.metadata.len(),
            alignment = parsed.alignment,
            "opened a GGUF file"
        );

        Ok(Self {
            backing,
            path,
            version: parsed.version,
            alignment: parsed.alignment,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            index,
            data_offset: parsed.data_offset,
        })
    }

    /// Where it was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The format version. Always 3.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// The data section's alignment.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// How many tensors the directory holds.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the directory is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Every tensor, in directory order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// One tensor's directory entry.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Every metadata key.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    /// One metadata value.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// A metadata value as `u32`, widening the narrower unsigned types.
    ///
    /// Writers are inconsistent about which width they use for a count, so a
    /// reader that insisted on exactly `U32` would reject valid files.
    /// Narrowing is refused rather than truncated.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::U8(v) => Some(u32::from(*v)),
            GgufValue::U16(v) => Some(u32::from(*v)),
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => u32::try_from(*v).ok(),
            GgufValue::I32(v) => u32::try_from(*v).ok(),
            GgufValue::I64(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// A metadata value as `f32`, widening `f64` and the integers.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            GgufValue::U32(v) => Some(*v as f32),
            GgufValue::I32(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// A metadata value as a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// A metadata value as a string array.
    pub fn get_strings(&self, key: &str) -> Option<&[String]> {
        match self.metadata.get(key)? {
            GgufValue::Array(GgufArray::String(v)) => Some(v),
            _ => None,
        }
    }

    /// A metadata value as an `f32` array.
    pub fn get_f32s(&self, key: &str) -> Option<&[f32]> {
        match self.metadata.get(key)? {
            GgufValue::Array(GgufArray::F32(v)) => Some(v),
            _ => None,
        }
    }

    /// A metadata value as an `i32` array.
    pub fn get_i32s(&self, key: &str) -> Option<&[i32]> {
        match self.metadata.get(key)? {
            GgufValue::Array(GgufArray::I32(v)) => Some(v),
            _ => None,
        }
    }

    /// One tensor's raw bytes, borrowed from the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self
            .info(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let start = (self.data_offset + info.offset) as usize;
        let end = start + info.n_bytes as usize;
        Ok(&self.backing[start..end])
    }

    /// One tensor as f32, whatever it is stored as.
    ///
    /// F32 is copied, F16 and BF16 are widened - both exactly, since every
    /// value of either fits an f32. This is the accessor that costs memory:
    /// a `[128256, 4096]` embedding is 2 GB as f16 and 4 GB through here.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self
            .info(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let raw = self.tensor_bytes(name)?;
        // Read element-wise rather than casting the mapping: a 2-byte element
        // needs only 2-byte alignment, and a tensor's offset carries no
        // promise of more than the file's 32-byte section alignment.
        Ok(match info.ggml_type {
            GgmlType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            GgmlType::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from(half::f16::from_le_bytes(*b)))
                .collect(),
            // bf16 is the top half of an f32, so this is exact.
            GgmlType::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
        })
    }

    /// One tensor as raw f16 bits, whatever it is stored as.
    ///
    /// The mirror of `xabe_st::StFile::tensor_f16`, and deliberately the same
    /// shape of promise: F16 is copied bit for bit, F32 and BF16 are rounded
    /// to nearest even. Unlike that one this cannot overflow to an infinity
    /// from BF16 - bf16 and f16 have different exponent ranges, so a large
    /// bf16 does saturate, and this checkpoint has no bf16 in it to do so.
    pub fn tensor_f16(&self, name: &str) -> Result<Vec<u16>, GgufError> {
        let info = self
            .info(name)
            .ok_or_else(|| GgufError::MissingTensor(name.to_string()))?;
        let raw = self.tensor_bytes(name)?;
        Ok(match info.ggml_type {
            // Already f16: the bytes are the answer, so no conversion runs at
            // all and the result is bit-identical by construction.
            GgmlType::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect(),
            GgmlType::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| half::f16::from_f32(f32::from_le_bytes(*b)).to_bits())
                .collect(),
            GgmlType::Bf16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| {
                    let wide = f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16);
                    half::f16::from_f32(wide).to_bits()
                })
                .collect(),
        })
    }
}

/// Reads one metadata value of the given tag.
fn value(cur: &mut Cursor, tag: u32, nested: bool) -> Result<GgufValue, GgufError> {
    Ok(match tag {
        0 => GgufValue::U8(cur.u8()?),
        1 => GgufValue::I8(cur.i8()?),
        2 => GgufValue::U16(cur.u16()?),
        3 => GgufValue::I16(cur.i16()?),
        4 => GgufValue::U32(cur.u32()?),
        5 => GgufValue::I32(cur.i32()?),
        6 => GgufValue::F32(cur.f32()?),
        7 => GgufValue::Bool(cur.bool()?),
        8 => GgufValue::String(cur.string()?),
        9 => {
            if nested {
                return Err(GgufError::NestedArray);
            }
            GgufValue::Array(array(cur)?)
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.i64()?),
        12 => GgufValue::F64(cur.f64()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

/// Reads one array: an element tag, a count, then that many elements.
fn array(cur: &mut Cursor) -> Result<GgufArray, GgufError> {
    let tag = cur.u32()?;
    let n = cur.u64()?;
    let cap = cur.hint(n);

    /// Fills a `Vec` without trusting `n` for the allocation.
    macro_rules! collect {
        ($variant:ident, $read:ident) => {{
            let mut v = Vec::with_capacity(cap);
            for _ in 0..n {
                v.push(cur.$read()?);
            }
            GgufArray::$variant(v)
        }};
    }

    Ok(match tag {
        0 => collect!(U8, u8),
        1 => collect!(I8, i8),
        2 => collect!(U16, u16),
        3 => collect!(I16, i16),
        4 => collect!(U32, u32),
        5 => collect!(I32, i32),
        6 => collect!(F32, f32),
        7 => collect!(Bool, bool),
        8 => collect!(String, string),
        9 => return Err(GgufError::NestedArray),
        10 => collect!(U64, u64),
        11 => collect!(I64, i64),
        12 => collect!(F64, f64),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

fn parse(bytes: &[u8]) -> Result<Parsed, GgufError> {
    let mut cur = Cursor::new(bytes);

    let magic = cur.magic()?;
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic(magic));
    }
    let version = cur.u32()?;
    if version != GGUF_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let n_tensors = cur.u64()?;
    let n_kv = cur.u64()?;

    let mut metadata = HashMap::with_capacity(cur.hint(n_kv));
    for _ in 0..n_kv {
        let key = cur.string()?;
        let tag = cur.u32()?;
        let val = value(&mut cur, tag, false)?;
        if metadata.insert(key.clone(), val).is_some() {
            return Err(GgufError::DuplicateKey(key));
        }
    }

    // `general.alignment` governs where the data section starts. A non-power
    // of two would make the padding arithmetic below meaningless, so it is
    // refused rather than rounded to something plausible.
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(v)) => *v,
        Some(_) => return Err(GgufError::BadAlignment(0)),
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::BadAlignment(alignment));
    }

    let mut tensors = Vec::with_capacity(cur.hint(n_tensors));
    let mut seen: HashMap<String, ()> = HashMap::with_capacity(cur.hint(n_tensors));
    for _ in 0..n_tensors {
        let name = cur.string()?;
        let n_dims = cur.u32()?;
        if n_dims > GGML_MAX_DIMS {
            return Err(GgufError::TooManyDimensions { name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(cur.u64()?);
        }
        let raw_type = cur.u32()?;
        let ggml_type =
            GgmlType::from_id(raw_type).ok_or_else(|| GgufError::UnsupportedGgmlType {
                name: name.clone(),
                ggml_type: raw_type,
            })?;
        let offset = cur.u64()?;

        let n_elements: u64 = dims.iter().product::<u64>();
        let n_bytes = n_elements
            .checked_mul(ggml_type.byte_size())
            .ok_or_else(|| GgufError::OffsetOverflow(name.clone()))?;

        if seen.insert(name.clone(), ()).is_some() {
            return Err(GgufError::DuplicateTensor(name));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            n_elements,
            n_bytes,
        });
    }

    // The data section starts at the first multiple of `alignment` at or
    // after the end of the directory.
    let end = cur.pos();
    let align = u64::from(alignment);
    let data_offset = end.div_ceil(align) * align;

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        data_offset,
    })
}
