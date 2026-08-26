//! Container-level tests against synthetic files.
//!
//! These build safetensors files byte by byte so every rejection path can be
//! exercised without a 139 MB checkpoint. The real-weights test lives in
//! `real_model.rs` and skips loudly when the checkpoint is absent.

use std::io::Write;

use xabe_st::{StError, StFile};

/// Builds a safetensors file, padding the header with spaces so the data
/// segment lands on an 8-byte boundary. Real producers do this; the reader
/// requires it, and `rejects_a_misaligned_data_segment` covers the case where
/// a producer did not.
fn write_file(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
    let mut header = header.to_string();
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    write_file_raw(name, &header, data)
}

/// Builds a safetensors file with the header written exactly as given.
fn write_file_raw(name: &str, header: &str, data: &[u8]) -> std::path::PathBuf {
    // One file per test: these run in parallel, and a shared name meant one test
    // truncating a file another still had mapped, which surfaced as SIGBUS.
    let path = tmpdir().join(format!("{name}.safetensors"));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write header length");
    f.write_all(header.as_bytes()).expect("write header");
    f.write_all(data).expect("write data");
    path
}

fn tmpdir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("xabe-st-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("create temp dir");
    d
}

#[test]
fn reads_a_well_formed_file() {
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let path = write_file("reads_a_well_formed_file", header, &data);

    let f = StFile::open(&path).expect("open");
    assert_eq!(f.len(), 1);
    assert_eq!(f.tensor("a").expect("tensor a"), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(f.info("a").expect("info a").shape, vec![2, 2]);
    assert_eq!(f.info("a").expect("info a").numel(), 4);
}

#[test]
fn metadata_entry_is_not_a_tensor() {
    let data: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let header =
        r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let path = write_file("metadata_entry_is_not_a_tensor", header, &data);

    let f = StFile::open(&path).expect("open");
    assert_eq!(f.len(), 1, "__metadata__ must not be counted as a tensor");
}

#[test]
fn rejects_a_tensor_running_past_the_data_segment() {
    let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    // only 8 bytes of data for a tensor that claims 16
    let path = write_file(
        "rejects_a_tensor_running_past_the_data_segment",
        header,
        &[0u8; 8],
    );

    match StFile::open(&path) {
        Err(StError::TensorOutOfBounds {
            name,
            end,
            data_len,
            ..
        }) => {
            assert_eq!(name, "a");
            assert_eq!(end, 16);
            assert_eq!(data_len, 8);
        }
        other => panic!("expected TensorOutOfBounds, got {other:?}"),
    }
}

#[test]
fn rejects_a_shape_that_disagrees_with_its_byte_range() {
    let header = r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}"#;
    let path = write_file(
        "rejects_a_shape_that_disagrees_with_its_byte_range",
        header,
        &[0u8; 16],
    );

    match StFile::open(&path) {
        Err(StError::TensorSizeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 12);
            assert_eq!(actual, 16);
        }
        other => panic!("expected TensorSizeMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_an_unsupported_dtype() {
    // BF16 used to be here. It is supported now - the Silero VAD is F16 and the
    // translator is BF16 - so the case moved to a dtype that really is not
    // handled. Integer tensors appear in real checkpoints (token ids, offsets),
    // which is exactly why reading one as float has to be refused rather than
    // reinterpreted.
    let header = r#"{"a":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}}"#;
    let path = write_file("rejects_an_unsupported_dtype", header, &[0u8; 16]);

    match StFile::open(&path) {
        Err(StError::UnsupportedDtype { dtype, .. }) => assert_eq!(dtype, "I64"),
        other => panic!("expected UnsupportedDtype, got {other:?}"),
    }
}

#[test]
fn a_half_precision_tensor_is_widened_rather_than_reinterpreted() {
    // 1.0 and -2.0 as IEEE binary16.
    let header = r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
    let bytes = [0x00, 0x3C, 0x00, 0xC0];
    let path = write_file("half_precision", header, &bytes);
    let f = StFile::open(&path).expect("open");

    // Borrowing must be refused: two halves reinterpreted as one f32 is a
    // number, just not this one, and nothing downstream would notice.
    let err = f.tensor("a").unwrap_err();
    assert!(err.to_string().contains("F16"), "{err}");

    assert_eq!(f.tensor_f32("a").expect("widen"), vec![1.0, -2.0]);
}

#[test]
fn brain_float_widening_is_exact_because_it_is_the_top_half_of_an_f32() {
    // bf16 keeps the sign, the exponent and 7 mantissa bits, so widening is a
    // shift and cannot round. 1.0 is 0x3F80, -2.0 is 0xC000.
    let header = r#"{"a":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
    let bytes = [0x80, 0x3F, 0x00, 0xC0, 0x49, 0x40];
    let path = write_file("brain_float", header, &bytes);
    let f = StFile::open(&path).expect("open");

    let got = f.tensor_f32("a").expect("widen");
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1], -2.0);
    // 0x4049 << 16 is exactly f32 3.140625, not an approximation of it.
    assert_eq!(got[2], f32::from_bits(0x4049_0000));
}

#[test]
fn rejects_a_header_longer_than_the_file() {
    let path = tmpdir().join("short.safetensors");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(&9_999_999u64.to_le_bytes()).expect("write len");
    f.write_all(b"{}").expect("write body");

    match StFile::open(&path) {
        Err(StError::HeaderOverrun { header_len, .. }) => assert_eq!(header_len, 9_999_999),
        other => panic!("expected HeaderOverrun, got {other:?}"),
    }
}

#[test]
fn tensor_shaped_rejects_the_wrong_geometry() {
    let data: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let path = write_file("tensor_shaped_rejects_the_wrong_geometry", header, &data);

    let f = StFile::open(&path).expect("open");
    assert!(f.tensor_shaped("a", &[2]).is_ok());
    match f.tensor_shaped("a", &[1, 2]) {
        Err(StError::ShapeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(actual, vec![2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn missing_tensor_is_named() {
    let path = write_file("missing_tensor_is_named", r#"{}"#, &[]);
    let f = StFile::open(&path).expect("open");
    match f.tensor("nope") {
        Err(StError::MissingTensor { name }) => assert_eq!(name, "nope"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn rejects_a_misaligned_data_segment() {
    // 57 bytes of header + 8 = 65, not a multiple of 4: casting the data segment
    // to f32 here would be undefined behaviour.
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    assert!(
        !(8 + header.len()).is_multiple_of(4),
        "test premise: header is odd-sized"
    );
    let path = write_file_raw("rejects_a_misaligned_data_segment", header, &[0u8; 4]);

    match StFile::open(&path) {
        Err(StError::Misaligned { what, .. }) => assert_eq!(what, "data segment"),
        other => panic!("expected Misaligned, got {other:?}"),
    }
}

#[test]
fn narrowing_to_f16_refuses_what_f16_cannot_hold() {
    // bf16 carries f32's exponent range and f16 does not: bf16 reaches 3.4e38,
    // f16 stops at 65504. This card has no bf16 at all and f32 would be 52 GB
    // against a 48 GB card, so the narrowing is mandatory - which makes the
    // check mandatory too. Saturating silently puts an infinity in one weight
    // of four hundred million and produces a model that loads without
    // complaint and speaks nonsense.
    //
    // 0x7F00 as bf16 is 2^127, about 1.7e38.
    let header = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let bytes = [0x80, 0x3F, 0x00, 0x7F];
    let path = write_file("narrowing_overflow", header, &bytes);
    let f = StFile::open(&path).expect("open");

    match f.tensor_f16("a") {
        Err(StError::NotRepresentable { at, value, .. }) => {
            assert_eq!(at, 1, "the second value is the large one");
            assert!(value > 1e38, "{value:e}");
        }
        other => panic!("expected NotRepresentable, got {other:?}"),
    }
}

#[test]
fn narrowing_to_f16_lets_underflow_through_and_says_so() {
    // The asymmetry is deliberate. A weight of 1e-9 rounding to zero
    // contributes nothing either way; an infinity poisons every product it
    // touches. 0x3080 as bf16 is about 9.3e-10, well under f16's smallest
    // subnormal of 6e-8.
    let header = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let bytes = [0x80, 0x3F, 0x80, 0x30];
    let path = write_file("narrowing_underflow", header, &bytes);
    let f = StFile::open(&path).expect("open");

    let got = f.tensor_f16("a").expect("narrow");
    assert_eq!(got[0], half::f16::from_f32(1.0).to_bits());
    assert_eq!(got[1], 0, "flushed to zero rather than refused");
}

#[test]
fn narrowing_an_f16_tensor_copies_its_bits_unchanged() {
    // No conversion means no rounding and no way to fail. Reading it back
    // through f32 would round twice and is the obvious thing to have written.
    let header = r#"{"a":{"dtype":"F16","shape":[3],"data_offsets":[0,6]}}"#;
    // 1.0, -2.0, and the smallest subnormal - which a round trip through f32
    // preserves, but only because f32 is wider. The point is that nothing
    // converts it at all.
    let bytes = [0x00, 0x3C, 0x00, 0xC0, 0x01, 0x00];
    let path = write_file("narrowing_identity", header, &bytes);
    let f = StFile::open(&path).expect("open");

    assert_eq!(
        f.tensor_f16("a").expect("narrow"),
        vec![0x3C00, 0xC000, 0x0001]
    );
}

#[test]
fn narrowing_an_f32_tensor_rounds_to_nearest_even() {
    // The same rounding the CUDA staging does, so a weight narrowed here and
    // one narrowed by the kernel are the same bits. 2049 is exactly between
    // two f16 values - the spacing is 2 up there - and ties go to even.
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2049.0f32.to_le_bytes());
    bytes.extend_from_slice(&2051.0f32.to_le_bytes());
    let path = write_file("narrowing_round", header, &bytes);
    let f = StFile::open(&path).expect("open");

    let got: Vec<f32> = f
        .tensor_f16("a")
        .expect("narrow")
        .into_iter()
        .map(|b| f32::from(half::f16::from_bits(b)))
        .collect();
    assert_eq!(got, vec![2048.0, 2052.0], "ties to even, both directions");
}
