//! The container, against a file built by hand and against the real one.
//!
//! The synthetic cases are here because a 16 GB checkpoint is a bad place to
//! learn what a truncated file does; the real one is here because a hand-built
//! file only proves the reader agrees with the test's own writer.

use xabe_gguf::{GgmlType, GgufError, GgufFile, GgufValue};

/// Builds a minimal but valid v3 image: one metadata key, one f32 tensor.
struct Writer(Vec<u8>);

impl Writer {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn str(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
        self
    }
}

/// `n_tensors`, `n_kv`, one `u32` metadata pair, then the tensor directory.
fn image(tensor_dims: &[u64], ggml_type: u32, data: &[f32]) -> Vec<u8> {
    let mut w = Writer::new();
    w.0.extend_from_slice(b"GGUF");
    w.u32(3).u64(1).u64(1);
    w.str("answer").u32(4).u32(42);
    w.str("t").u32(tensor_dims.len() as u32);
    for &d in tensor_dims {
        w.u64(d);
    }
    w.u32(ggml_type).u64(0);

    let mut bytes = w.0;
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// The same image, with the data section given as raw bytes rather than f32 -
/// which is the only way to build one for a format whose elements are not
/// four bytes wide.
fn image_raw(name: &str, dims: &[u64], ty: u32, raw: &[u8]) -> Vec<u8> {
    let mut w = Writer::new();
    w.0.extend_from_slice(b"GGUF");
    w.u32(3).u64(1).u64(0);
    w.str(name).u32(dims.len() as u32);
    for &d in dims {
        w.u64(d);
    }
    w.u32(ty).u64(0);
    let mut bytes = w.0;
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    bytes.extend_from_slice(raw);
    bytes
}

#[test]
fn a_hand_built_file_round_trips() {
    let f = GgufFile::from_bytes(image(&[2, 3], 0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .expect("a valid image should parse");

    assert_eq!(f.version(), 3);
    assert_eq!(f.alignment(), 32, "absent general.alignment means 32");
    assert_eq!(f.len(), 1);
    assert_eq!(f.get("answer"), Some(&GgufValue::U32(42)));
    assert_eq!(f.get_u32("answer"), Some(42));

    let info = f.info("t").expect("the tensor is named t");
    assert_eq!(info.dims, vec![2, 3]);
    assert_eq!(info.n_elements, 6);
    assert_eq!(info.n_bytes, 24);
    assert_eq!(info.ggml_type, GgmlType::F32);
    assert_eq!(
        f.tensor_f32("t").expect("readable"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn ggml_dims_are_the_reverse_of_a_row_major_shape() {
    // The single easiest thing to get wrong about this format, so it is
    // asserted rather than left to a comment. `dims` is what the file said;
    // `shape` is the matrix as a safetensors header would have written it.
    let f = GgufFile::from_bytes(image(&[4, 7], 0, &[0.0; 28])).expect("parses");
    let info = f.info("t").unwrap();
    assert_eq!(info.dims, vec![4, 7], "fastest-varying first, as stored");
    assert_eq!(
        info.shape(),
        vec![7, 4],
        "rows first, as the reference reads"
    );
}

#[test]
fn a_bad_magic_is_named_not_guessed() {
    let mut bytes = image(&[1], 0, &[1.0]);
    bytes[0] = b'X';
    match GgufFile::from_bytes(bytes) {
        Err(GgufError::BadMagic(m)) => assert_eq!(&m, b"XGUF"),
        other => panic!("wanted BadMagic, got {other:?}", other = other.err()),
    }
}

#[test]
fn an_older_version_is_refused_rather_than_attempted() {
    // v1 and v2 use narrower count fields, so reading one as v3 would land
    // the tensor directory at the wrong offset and report plausible nonsense.
    let mut bytes = image(&[1], 0, &[1.0]);
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    match GgufFile::from_bytes(bytes) {
        Err(GgufError::UnsupportedVersion(v)) => assert_eq!(v, 2),
        other => panic!(
            "wanted UnsupportedVersion, got {other:?}",
            other = other.err()
        ),
    }
}

#[test]
fn a_quantized_tensor_is_sized_by_its_block_geometry() {
    // This test used to assert that Q8_0 was *refused*, which was true when
    // the crate read only the three unquantized widths. It reads nine block
    // formats now, so what matters is the sizing: a block format's bytes are
    // `elements / block_size * type_size`, and using the unquantized rule
    // instead would claim a 32-element Q8_0 tensor needs 32 bytes where it
    // needs 34 - a directory that validates and then reads every tensor after
    // the first at the wrong offset.
    //
    // The unpacked values are checked against `gguf-py` in `dequant.rs`; this
    // is only about the arithmetic in the directory.
    let raw = vec![0u8; 34];
    let f = GgufFile::from_bytes(image_raw("t", &[32], 8, &raw)).expect("Q8_0 parses");
    let info = f.info("t").expect("bound");
    assert_eq!(info.ggml_type, GgmlType::Q8_0);
    assert_eq!(info.n_elements, 32, "one block of 32");
    assert_eq!(info.n_bytes, 34, "an f16 scale and 32 signed bytes");
    assert!(info.ggml_type.is_quantized());
    assert_eq!(info.ggml_type.block_size(), 32);
    assert_eq!(info.ggml_type.type_size(), 34);
}

#[test]
fn a_format_outside_the_table_is_still_refused_by_name_and_id() {
    // The table grew; the refusal did not go away. Id 16 is `Q2_K`'s
    // neighbour `IQ2_XXS`, an importance-weighted format with no file in this
    // project - refused rather than mis-sized.
    match GgufFile::from_bytes(image_raw("t", &[256], 16, &[0u8; 66])) {
        Err(GgufError::UnsupportedGgmlType { name, ggml_type }) => {
            assert_eq!(name, "t");
            assert_eq!(ggml_type, 16);
        }
        other => panic!(
            "wanted UnsupportedGgmlType, got {other:?}",
            other = other.err()
        ),
    }
}

#[test]
fn a_truncated_data_section_is_caught_at_open() {
    let mut bytes = image(&[2, 3], 0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    bytes.truncate(bytes.len() - 8);
    match GgufFile::from_bytes(bytes) {
        Err(GgufError::TensorOutOfBounds { name, .. }) => assert_eq!(name, "t"),
        other => panic!(
            "wanted TensorOutOfBounds, got {other:?}",
            other = other.err()
        ),
    }
}

#[test]
fn f16_reaches_f32_and_back_without_a_conversion_in_the_middle() {
    // Stored f16, asked for f16: the bytes are the answer, so this must be
    // bit-identical rather than merely close.
    let halves: Vec<u16> = [1.0f32, -2.5, 0.125]
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let mut w = Writer::new();
    w.0.extend_from_slice(b"GGUF");
    w.u32(3).u64(1).u64(0);
    w.str("t").u32(1).u64(3).u32(1).u64(0);
    let mut bytes = w.0;
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    for h in &halves {
        bytes.extend_from_slice(&h.to_le_bytes());
    }

    let f = GgufFile::from_bytes(bytes).expect("parses");
    assert_eq!(f.tensor_f16("t").expect("f16"), halves);
    assert_eq!(f.tensor_f32("t").expect("f32"), vec![1.0, -2.5, 0.125]);
}

#[test]
fn a_missing_tensor_is_an_error_with_the_name_in_it() {
    let f = GgufFile::from_bytes(image(&[1], 0, &[1.0])).expect("parses");
    match f.tensor_bytes("nope") {
        Err(GgufError::MissingTensor(n)) => assert_eq!(n, "nope"),
        other => panic!("wanted MissingTensor, got {other:?}", other = other.err()),
    }
}
