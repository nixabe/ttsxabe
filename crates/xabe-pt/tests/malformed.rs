//! What the reader refuses, and whether it says why.
//!
//! These need no checkpoint: the archives are built here, byte by byte, which
//! is the only way to produce the shapes a real file will not have. The point
//! of each is that the failure is a sentence naming the problem rather than a
//! panic inside a slice index - `xabe-pt` is the layer between a downloaded
//! file and a pointer cast, and every one of these is a way a download can be
//! wrong.

use std::io::Write;
use xabe_pt::PtFile;

/// Writes a zip holding one stored entry, and returns its path.
///
/// Hand-assembled rather than taken from a crate: a dependency that produced
/// only well-formed archives could not write any of the cases below.
fn zip_with(dir: &std::path::Path, name: &str, entry: &str, body: &[u8]) -> std::path::PathBuf {
    let mut out: Vec<u8> = Vec::new();
    let entry_bytes = entry.as_bytes();

    // Local file header: signature, version, flags, method 0 (stored), time,
    // date, CRC (unchecked here), both sizes, name and extra lengths.
    let local = out.len() as u32;
    out.extend(0x0403_4b50u32.to_le_bytes());
    out.extend([20u8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    out.extend(0u32.to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend((entry_bytes.len() as u16).to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(entry_bytes);
    out.extend(body);

    // Central directory record for it.
    let dir_start = out.len() as u32;
    out.extend(0x0201_4b50u32.to_le_bytes());
    out.extend([20u8, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    out.extend(0u32.to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend((body.len() as u32).to_le_bytes());
    out.extend((entry_bytes.len() as u16).to_le_bytes());
    out.extend(0u16.to_le_bytes()); // extra
    out.extend(0u16.to_le_bytes()); // comment
    out.extend(0u16.to_le_bytes()); // disk
    out.extend(0u16.to_le_bytes()); // internal attributes
    out.extend(0u32.to_le_bytes()); // external attributes
    out.extend(local.to_le_bytes());
    out.extend(entry_bytes);
    let dir_len = out.len() as u32 - dir_start;

    // End of central directory.
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(dir_len.to_le_bytes());
    out.extend(dir_start.to_le_bytes());
    out.extend(0u16.to_le_bytes());

    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create archive");
    f.write_all(&out).expect("write archive");
    path
}

/// A scratch directory that removes itself.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("xabe-pt-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch");
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_file_that_is_not_a_zip_names_the_pre_1_6_format() {
    let scratch = Scratch::new("not-a-zip");
    let path = scratch.0.join("legacy.pth");
    // What a pre-1.6 torch save opens with: a bare pickle stream, no archive.
    std::fs::write(&path, b"\x80\x02}q\x00.").expect("write file");

    let err = PtFile::open(&path).expect_err("a bare pickle must be refused");
    let text = err.to_string();
    assert!(text.contains("not a zip archive"), "{text}");
    // The message has to name the format, because the extension does not
    // distinguish it from a file this reader handles.
    assert!(text.contains("1.6"), "{text}");
}

#[test]
fn a_zip_without_a_pickle_is_not_a_checkpoint() {
    let scratch = Scratch::new("no-pickle");
    let path = zip_with(&scratch.0, "empty.pth", "archive/version", b"3\n");

    let err = PtFile::open(&path).expect_err("an archive with no data.pkl must be refused");
    assert!(err.to_string().contains("data.pkl"), "{err}");
}

#[test]
fn a_missing_file_names_the_path() {
    let err = PtFile::open("/nonexistent/definitely-not-here.pth").expect_err("must be refused");
    assert!(err.to_string().contains("definitely-not-here.pth"), "{err}");
}

#[test]
fn an_unsupported_opcode_is_named_rather_than_guessed() {
    let scratch = Scratch::new("bad-opcode");
    // PROTO 2, then EMPTY_SET - a real opcode this reader does not implement.
    // The distinction matters: an unimplemented opcode is a bug report, and
    // silently stopping would produce a state dict missing everything after it.
    let path = zip_with(&scratch.0, "odd.pth", "archive/data.pkl", b"\x80\x02\x8f.");

    let err = PtFile::open(&path).expect_err("an unimplemented opcode must be refused");
    let text = err.to_string();
    assert!(text.contains("0x8f"), "{text}");
    assert!(text.contains("EMPTY_SET"), "{text}");
}

#[test]
fn a_pickle_that_is_an_object_graph_names_what_it_wanted_to_call() {
    let scratch = Scratch::new("object-graph");
    // GLOBAL torch.nn.Module, EMPTY_TUPLE, REDUCE, STOP: the shape a pickled
    // `nn.Module` opens with, and the thing this reader will not do.
    let path = zip_with(
        &scratch.0,
        "graph.pth",
        "archive/data.pkl",
        b"\x80\x02ctorch.nn\nModule\n)R.",
    );

    let err = PtFile::open(&path).expect_err("an object graph must be refused");
    let text = err.to_string();
    assert!(text.contains("torch.nn.Module"), "{text}");
    assert!(text.contains("state dict"), "{text}");
}

#[test]
fn a_section_that_is_not_a_mapping_is_refused() {
    let scratch = Scratch::new("no-section");
    // An empty dict as the root: no `model` key to read a state dict from.
    let path = zip_with(&scratch.0, "empty.pth", "archive/data.pkl", b"\x80\x02}.");

    let err = PtFile::open_section(&path, "model").expect_err("an absent section must be refused");
    assert!(err.to_string().contains("model"), "{err}");
}

#[test]
fn an_empty_state_dict_opens_and_holds_nothing() {
    let scratch = Scratch::new("empty-dict");
    let path = zip_with(&scratch.0, "empty.pth", "archive/data.pkl", b"\x80\x02}.");

    // Not an error: a checkpoint may legitimately hold no tensors, and it is
    // the weight schema above that decides whether that is a problem.
    let f = PtFile::open(&path).expect("an empty state dict is still a state dict");
    assert!(f.is_empty());
    assert_eq!(f.len(), 0);
}
