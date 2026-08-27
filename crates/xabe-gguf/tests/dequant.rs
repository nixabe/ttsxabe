//! Every block format against `gguf-py`'s own dequantization.
//!
//! The capture holds packed bytes and the f32 the reference unpacks them to,
//! so what is asserted is **exact equality**, not a tolerance. This side never
//! quantizes; it only has to reproduce an unpacking, and an unpacking either
//! agrees bit for bit or has an indexing bug.
//!
//! The corpus is pseudo-random encodings rather than a quantizer's output, for
//! two reasons written out in `tools/oracle/capture_quants.py`: Python `gguf`
//! can only quantize the five legacy formats, and random encodings reach every
//! nibble and every packed six-bit scale where a quantizer only reaches the
//! well-conditioned subset.

use std::path::{Path, PathBuf};

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    id: u32,
    block_size: usize,
    type_size: usize,
    rows: usize,
    cols: usize,
    elements: usize,
    packed_bytes: usize,
    round_trip: bool,
}

#[derive(serde::Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

fn corpus() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/gguf/quants");
    p.join("manifest.json").is_file().then_some(p)
}

/// Wraps `raw` in a one-tensor GGUF so the reader reaches it the way it would
/// in a real file - through the directory, the alignment padding and the
/// offset arithmetic, not through a private entry point.
fn image(name: &str, dims: &[u64], ty: u32, raw: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&ty.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.extend_from_slice(raw);
    b
}

fn read_f32(p: &Path) -> Vec<f32> {
    let bytes = std::fs::read(p).expect("read the expected values");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Runs one capture through the reader and compares element for element.
fn check(dir: &Path, c: &Case, suffix: &str) {
    let raw = std::fs::read(dir.join(format!("{}{suffix}.bin", c.name))).expect("packed");
    let want = read_f32(&dir.join(format!("{}{suffix}.f32", c.name)));

    assert_eq!(raw.len(), c.rows * c.cols / c.block_size * c.type_size);
    assert_eq!(want.len(), c.elements);

    // ggml order: the fastest-varying dimension first, so a [rows, cols]
    // tensor is written [cols, rows].
    let f =
        xabe_gguf::GgufFile::from_bytes(image("t", &[c.cols as u64, c.rows as u64], c.id, &raw))
            .unwrap_or_else(|e| panic!("{}{suffix}: {e}", c.name));

    let info = f.info("t").expect("bound");
    assert_eq!(info.n_elements as usize, c.elements, "{}", c.name);
    assert_eq!(info.n_bytes as usize, c.packed_bytes, "{}", c.name);

    let got = f
        .tensor_f32("t")
        .unwrap_or_else(|e| panic!("{}{suffix}: {e}", c.name));
    assert_eq!(got.len(), want.len(), "{}", c.name);

    let mut wrong = 0usize;
    let mut first = None;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if g.to_bits() != w.to_bits() {
            wrong += 1;
            if first.is_none() {
                first = Some((i, *g, *w));
            }
        }
    }
    assert_eq!(
        wrong,
        0,
        "{}{suffix}: {wrong} of {} elements differ, first at {:?}",
        c.name,
        want.len(),
        first
    );
}

#[test]
fn every_block_format_matches_the_reference_exactly() {
    let Some(dir) = corpus() else {
        println!("SKIP: run tools/oracle/capture_quants.py first");
        return;
    };
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).expect("read"))
            .expect("parse");

    assert_eq!(m.cases.len(), 10, "five legacy formats and five K-quants");
    for c in &m.cases {
        check(&dir, c, "");
        println!("  {:6} {} elements, exact", c.name, c.elements);
    }
}

#[test]
fn the_quantizable_formats_also_match_on_a_real_round_trip() {
    // Random encodings reach corners a quantizer never emits; a quantizer
    // reaches the distribution a real checkpoint has. Both are worth running,
    // and only five formats can do the second in Python.
    let Some(dir) = corpus() else {
        println!("SKIP: run tools/oracle/capture_quants.py first");
        return;
    };
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).expect("read"))
            .expect("parse");

    let mut ran = 0;
    for c in m.cases.iter().filter(|c| c.round_trip) {
        check(&dir, c, ".q");
        ran += 1;
    }
    assert_eq!(ran, 5, "Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0");
}

#[test]
fn block_sizes_and_type_sizes_are_the_reference_ones() {
    // Transcribed constants, checked against the capture rather than trusted.
    // Getting one wrong sizes every tensor of that format incorrectly, which
    // shows up as a directory that will not validate - loud, but only if the
    // number is wrong in the direction that overflows the file.
    let Some(dir) = corpus() else {
        println!("SKIP: run tools/oracle/capture_quants.py first");
        return;
    };
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).expect("read"))
            .expect("parse");

    for c in &m.cases {
        let raw = vec![0u8; c.type_size];
        let f = xabe_gguf::GgufFile::from_bytes(image("t", &[c.block_size as u64], c.id, &raw))
            .unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let info = f.info("t").expect("bound");
        assert_eq!(
            info.n_bytes as usize, c.type_size,
            "{} bytes per block",
            c.name
        );
        assert_eq!(
            info.n_elements as usize, c.block_size,
            "{} elements per block",
            c.name
        );
    }
}

#[test]
fn a_row_that_is_not_a_whole_number_of_blocks_is_refused() {
    // ggml never writes one, so this is about a corrupt or hand-built file:
    // reading it would slide every subsequent row against its scale.
    let raw = vec![0u8; 34];
    match xabe_gguf::GgufFile::from_bytes(image("t", &[20, 1], 8, &raw)) {
        Err(xabe_gguf::GgufError::RaggedBlocks { name, row, block }) => {
            assert_eq!(name, "t");
            assert_eq!(row, 20);
            assert_eq!(block, 32);
        }
        other => panic!("wanted RaggedBlocks, got {:?}", other.err()),
    }
}

#[test]
fn a_format_this_reader_does_not_decode_is_still_refused_by_id() {
    // The table grew; the refusal did not go away. Q8_K is id 15 and is an
    // intermediate that never appears in a stored tensor, so it stays out.
    match xabe_gguf::GgufFile::from_bytes(image("t", &[256], 15, &vec![0u8; 292])) {
        Err(xabe_gguf::GgufError::UnsupportedGgmlType { name, ggml_type }) => {
            assert_eq!(name, "t");
            assert_eq!(ggml_type, 15);
        }
        other => panic!("wanted UnsupportedGgmlType, got {:?}", other.err()),
    }
}
