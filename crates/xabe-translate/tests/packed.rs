//! The packed path against the unpacked one, on the same quantized translator.
//!
//! The sibling of `xabe-chat`'s `tests/packed.rs`, and it exists separately
//! rather than being trusted by analogy for one reason: the two crates pick the
//! head count for the rope unpermutation differently. `xabe-chat` compares the
//! tensor's row count against `hidden_size`; this one matches on the tensor's
//! *name*. Both mirror their own crate's `f16` path, and neither is checked by
//! the other's test.
//!
//! It happens not to matter on this checkpoint - 40 query heads over 40
//! key-value heads, so both branches give 40 - which is exactly why it is worth
//! a test rather than a reading. A grouped-query translator would make the two
//! disagree, silently, and the shapes would still check out.
//!
//! Loads sequentially: the f16 half is 26.5 GB and the packed half about 8, so
//! holding both would be 34.5 GB. Sequential also keeps this honest about what
//! it is comparing, which is two loads of one file rather than two models.

use std::path::PathBuf;
use xabe_translate::{Packing, Translator};

/// A quantized copy of the translator, if one has been made.
fn quantized() -> Option<PathBuf> {
    let dir = std::env::var("XABE_QUANT_DIR").ok()?;
    let name = std::env::var("XABE_TRANSLATOR_QUANT")
        .unwrap_or_else(|_| "taigi-translator-13b-Q4_K_M.gguf".to_string());
    let p = PathBuf::from(dir).join(name);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!("SKIP: no {} under XABE_QUANT_DIR", p.display());
        None
    }
}

/// The card to load 26.5 GB onto. No default, for the same reason
/// `XABE_CHAT_DEVICE` has none: this host is shared.
fn device() -> Option<usize> {
    match std::env::var("XABE_TRANSLATOR_DEVICE").ok() {
        Some(v) => v.parse().ok(),
        None => {
            eprintln!("SKIP: set XABE_TRANSLATOR_DEVICE to the card to load onto");
            None
        }
    }
}

#[test]
fn packed_weights_agree_with_unpacked_ones_on_the_same_file() {
    let (Some(path), Some(dev)) = (quantized(), device()) else {
        return;
    };

    // A sentence from the model card's own domain, so the prompt is the shape
    // the checkpoint was fine-tuned on rather than arbitrary text.
    let source = "今天天氣真好，我想去海邊走走。";

    let run = |packing: Packing| -> (Vec<f32>, String) {
        let m = Translator::open_with(&path, dev, packing).expect("the quantized GGUF");
        let ids = m.prompt_ids(source, "HAN");
        let mut cache = m.cache();
        let out = m.forward(&ids, &mut cache).expect("a forward pass");
        let v = m.gpu().download(&out).expect("the logits");
        let vocab = m.config().vocab_size;
        let last = v.len() - vocab;
        // The product-level answer as well as the logits: greedy, so it is a
        // function of the weights alone and a wiring mistake changes it.
        let text = m.translate(source, "HAN", 64, 1.1).expect("a translation");
        (v[last..].to_vec(), text)
    };

    let (packed, packed_text) = run(Packing::Packed);
    let (wide, wide_text) = run(Packing::F16);
    assert_eq!(packed.len(), wide.len(), "the same vocabulary");

    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("a non-empty vocabulary")
    };
    println!("packed:   {packed_text}");
    println!("unpacked: {wide_text}");
    assert_eq!(
        arg(&packed),
        arg(&wide),
        "packed and unpacked chose different first tokens",
    );

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
        "packed and unpacked logits differ by {worst}, more than rounding \
         accounts for against a span of {span}",
    );
}
