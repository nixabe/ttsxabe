//! Per-layer block outputs against `llama-eval-callback` reading the same GGUF.
//!
//! # The oracle this model was said not to have
//!
//! `tests/llama_server.rs` says the chat model has one reference and it is the
//! weak kind - a product comparison, because there is no HuggingFace checkpoint
//! for it on this machine. That was true of the comparison and not of the
//! model. llama.cpp's `eval-callback` prints every node of its graph with a
//! scalar sum, which is a per-layer tap on the same file, and
//! `tools/oracle/capture_chat_layers.py` captures it.
//!
//! A sum is a weaker tap than a tensor: it can hide a permutation, and one that
//! cancels turns a small absolute error into a large percentage. So this
//! compares against the *magnitude the layer works at* rather than against the
//! sum, and it is a floor on wrongness rather than a proof of rightness.
//!
//! # What it is actually for
//!
//! Localisation. A reply that diverges says only that something is wrong; this
//! says which layer it entered at, and that is what turned "the chat model
//! disagrees with llama-server" from a mystery into a one-line cause - see
//! `docs/TESTING.md`.
//!
//! It is not a tight tolerance and should not become one. Two independent
//! implementations of this model at f16 differ by about 1% at the last layer
//! *when nothing is wrong*, because the network amplifies operand rounding by
//! about 1.15x a layer. The bound here is set well above that, to catch a
//! wiring mistake rather than to police arithmetic.

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Golden {
    tokens: Option<usize>,
    /// The file the capture was taken from. See [`Model`].
    model: Option<Model>,
    nodes: HashMap<String, f64>,
}

/// Which GGUF `eval-callback` read, as the capture recorded it.
///
/// Checked before anything is compared, because the failure it prevents does
/// not look like a mismatch - it looks like a wiring fault. A capture of the
/// Q4_K build against the engine reading the f16 build of the same model sits
/// at 0.36 of the layer magnitude from block 1 onward and stays flat, which is
/// exactly the shape of a real defect: an offset that enters at one layer and
/// rides the residual stream. It is two different models being compared.
///
/// Name and size rather than a hash, so the check costs a `stat` on a file
/// that can be 16 GB. Two quantizations of one model differ in both.
#[derive(serde::Deserialize)]
struct Model {
    name: String,
    bytes: u64,
}

/// Ratio of the disagreement to the layer's own working magnitude.
///
/// Read off the measurement rather than chosen: the same comparison on the f16
/// checkpoint, where both sides do the same arithmetic, sits at 0.01 by the last
/// layer, and on the quantized one at 0.04 because the two multiply packed
/// weights differently. A wiring mistake is not 5%, it is 50%.
const BOUND: f64 = 0.25;

fn golden() -> Option<Golden> {
    let p = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.golden/chat/layers.json"
    ));
    match std::fs::read(&p) {
        Ok(b) => serde_json::from_slice(&b).ok(),
        Err(_) => {
            eprintln!(
                "SKIP: no {} (tools/oracle/capture_chat_layers.py)",
                p.display()
            );
            None
        }
    }
}

#[test]
fn every_block_output_tracks_the_reference_graph() {
    let (Some(m), Some(d)) = (
        std::env::var("XABE_CHAT_MODEL").ok().map(PathBuf::from),
        std::env::var("XABE_CHAT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    ) else {
        eprintln!("SKIP: set XABE_CHAT_MODEL and XABE_CHAT_DEVICE");
        return;
    };
    let Some(g) = golden() else { return };

    // Refuse a capture of a different file before comparing anything to it.
    let want = g.model.as_ref().unwrap_or_else(|| {
        panic!(
            "the capture predates recording which model it came from; \
             re-take it with tools/oracle/capture_chat_layers.py"
        )
    });
    let name = m
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let bytes = std::fs::metadata(&m).map(|x| x.len()).unwrap_or(0);
    assert!(
        name == want.name && bytes == want.bytes,
        "the capture is of {} ({} bytes) and XABE_CHAT_MODEL is {name} ({bytes} bytes); \
         these are different models and the comparison would be meaningless",
        want.name,
        want.bytes,
    );

    let model = xabe_chat::ChatModel::open(&m, d).expect("open the chat model");
    // The capture's own prompt, tokenized here. Its token count is recorded so
    // a capture taken against a different prompt fails loudly rather than
    // comparing two different graphs.
    let tok = model.tokenizer();
    let mut ids = vec![tok.bos()];
    ids.extend(tok.encode("hi", false));
    assert_eq!(
        Some(ids.len()),
        g.tokens,
        "the capture was taken on a different prompt"
    );

    let n = model.config().num_hidden_layers;
    let mut cache = model.cache();
    let (_, taps) = model.forward_tapped(&ids, &mut cache, n).expect("forward");

    // The last layer is skipped: llama.cpp computes only the row it needs for
    // the output there, so its `l_out-31` is one token wide where ours is all
    // of them, and the two sums are not the same quantity.
    let mut worst = 0.0f64;
    for (i, t) in taps.iter().enumerate().take(n - 1) {
        let Some(&want) = g.nodes.get(&format!("l_out-{i}")) else {
            continue;
        };
        let got: f64 = t.iter().map(|&v| f64::from(v)).sum();
        // Against the magnitude the layer works at, not against the sum: a
        // residual stream that cancels to near zero would otherwise make any
        // difference look total.
        let mag = t.iter().map(|&v| f64::from(v).abs()).sum::<f64>() / t.len() as f64
            * (t.len() as f64).sqrt();
        let rel = (want - got).abs() / mag.max(1e-9);
        worst = worst.max(rel);
        assert!(
            rel < BOUND,
            "l_out-{i}: reference {want}, ours {got}, {rel:.4} of the layer's magnitude",
        );
    }
    println!("  worst block-output divergence {worst:.4} of the layer's magnitude");
}
