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
    let header = r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let path = write_file("rejects_an_unsupported_dtype", header, &[0u8; 4]);

    match StFile::open(&path) {
        Err(StError::UnsupportedDtype { dtype, .. }) => assert_eq!(dtype, "BF16"),
        other => panic!("expected UnsupportedDtype, got {other:?}"),
    }
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
