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
