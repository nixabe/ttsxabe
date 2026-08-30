//! The packed path against the unpacked one, on the same quantized file.
//!
//! # What this adds to the kernel tests
//!
//! `xabe-cuda`'s `tests/quant.rs` proves the matmul unpacks every block format
//! element for element. It says nothing about the *wiring*: that the ggml type
//! is mapped to the right layout, that the rope permutation is applied to the
//! packed bytes as well as the f16 ones, and that the packed operand reaches
//! the kernel for every projection rather than most of them. Each of those
//! produces a model that loads, runs, and is wrong.
//!
//! So this loads the same file twice - [`Packing::Packed`] and
//! [`Packing::F16`] - and compares the logits. The weights are identical by
//! construction, because both come from the same blocks; only where they are
//! unpacked differs. A wiring mistake moves the logits far more than the
//! rounding does.
//!
//! # Why they are close rather than identical
//!
//! On the tiled kernel they should agree exactly: the packed path unpacks to
//! f32 and `gemm_pack` rounds to f16, while the F16 path rounds the same
//! dequantized value to f16 at load, and both roundings are to nearest even.
//!
//! On the scalar kernel they cannot. That path is exact f32, so the packed
//! operand is the *dequantized* value and the F16 operand is that value
//! rounded - a real difference, in the packed path's favour. Decode runs there,
//! so a single-token step is where any divergence shows up.
//!
//! # Sequentially, because two copies do not fit
//!
//! The f16 copy of the 8 B is 16 GB. Loading it beside the packed one would be
//! 21 GB, which fits, but the second model is built after the first is dropped
//! anyway - there is nothing to compare until both sets of logits exist, and
//! logits are kilobytes.

use std::path::PathBuf;
use xabe_chat::{ChatModel, Packing};

/// A quantized copy of the chat model, if one has been made.
///
/// `XABE_QUANT_DIR` has no default on purpose - the files are gigabytes and
/// derived rather than downloaded, so `docs/TESTING.md` names the one command
/// that reproduces any of them rather than pinning a path a test depends on.
fn quantized() -> Option<PathBuf> {
    let dir = std::env::var("XABE_QUANT_DIR").ok()?;
    // The same naming `xabe-gguf`'s `quantized_model.rs` uses, so one
    // directory serves both and `docs/TESTING.md` names one command.
    let name =
        std::env::var("XABE_QUANT_FILE").unwrap_or_else(|_| "breeze-Q4_K_M.gguf".to_string());
    let p = PathBuf::from(dir).join(name);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIP: no {} under XABE_QUANT_DIR", p.display());
        None
    }
}

/// The card to load 8 GB onto. No default: this host has three and two of them
/// are running somebody's pipeline.
fn device() -> Option<usize> {
    match std::env::var("XABE_CHAT_DEVICE").ok() {
        Some(v) => v.parse().ok(),
        None => {
            eprintln!("SKIP: set XABE_CHAT_DEVICE to the card to load onto");
            None
        }
    }
}

#[test]
fn packed_weights_agree_with_unpacked_ones_on_the_same_file() {
    let (Some(path), Some(dev)) = (quantized(), device()) else {
        return;
    };

    // Long enough to run the tiled kernel on prefill and to exercise every
    // projection in all 32 layers; short enough that the comparison is about
    // the weights rather than about a cache.
    let prompt = "台灣的天氣真好，今天想去海邊走走。";

    let logits = |packing: Packing| -> (Vec<f32>, usize) {
        let m = ChatModel::open_with(&path, dev, packing).expect("the quantized GGUF");
        let ids = m.tokenizer().encode(prompt, true);
        let mut cache = m.cache();
        let out = m.forward(&ids, &mut cache).expect("a forward pass");
        let v = m.gpu().download(&out).expect("the logits");
        let vocab = m.config().vocab_size;
        // The last position's row is the one a sampler would read.
        let last = v.len() - vocab;
        (v[last..].to_vec(), ids.len())
    };

    let (packed, n) = logits(Packing::Packed);
    let (wide, n2) = logits(Packing::F16);
    assert_eq!(n, n2, "the same prompt must tokenize the same way");
    assert_eq!(packed.len(), wide.len(), "the same vocabulary");

    // The decision is what matters: a sampler reads the argmax, and a wiring
    // mistake moves it. Rounding does not.
    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("a non-empty vocabulary")
    };
    assert_eq!(
        arg(&packed),
        arg(&wide),
        "packed and unpacked chose different tokens",
    );

    // And the whole distribution, judged relative to the spread of the logits
    // rather than to each one: a logit near zero carries no information and a
    // relative tolerance on it would be measuring noise.
    let span =
        wide.iter().fold(f32::MIN, |a, &b| a.max(b)) - wide.iter().fold(f32::MAX, |a, &b| a.min(b));
    let worst = packed
        .iter()
        .zip(&wide)
        .map(|(p, w)| (p - w).abs())
        .fold(0.0f32, f32::max);
    println!("worst logit difference {worst:.6} against a span of {span:.3}");
    assert!(
        worst <= 0.02 * span,
        "packed and unpacked logits differ by {worst}, which is more than \
         rounding accounts for against a span of {span}",
    );
}
