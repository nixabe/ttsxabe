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
fn a_quantized_tensor_is_refused_by_name_and_id() {
    // Q8_0 is id 8. Reading its packed blocks as raw values would consume the
    // right number of bytes and produce the wrong numbers, which is the
    // failure this refusal exists to prevent.
    match GgufFile::from_bytes(image(&[32], 8, &[0.0; 32])) {
        Err(GgufError::UnsupportedGgmlType { name, ggml_type }) => {
            assert_eq!(name, "t");
            assert_eq!(ggml_type, 8);
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
