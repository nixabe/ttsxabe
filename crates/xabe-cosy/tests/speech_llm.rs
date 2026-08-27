//! The speech LLM against CosyVoice3's own, position by position.
//!
//! # Why the tokens are not the target
//!
//! `ras_sampling` draws with `torch.multinomial`, so upstream's token sequence
//! is a function of PyTorch's RNG as much as of the weights. Measured on the
//! capture, the greedy argmax agrees with the sampled run at **21 of 143**
//! positions - so two correct implementations would produce visibly different
//! token sequences, and comparing them would prove nothing in either
//! direction.
//!
//! So the captured tokens are fed back in and the *log-probabilities* are
//! compared at every position. That is deterministic, it is 143 observations
//! rather than one, and it keeps measuring past the first divergence. Same
//! reason `xabe-chat` is compared that way, and here it is not a preference:
//! there is no other option.
//!
//! # What is compared, and to what tolerance
//!
//! Upstream runs float32 on a card, and so does this, but the reduction orders
//! differ - so this is not an equality test. Three statistics, because each
//! catches something the others do not:
//!
//! - **Argmax agreement**, which is what actually decides a token. A layout
//!   mistake moves the argmax; a rounding difference almost never does.
//! - **Max absolute error** over the log-probabilities, which catches a
//!   systematic offset that leaves the ordering intact.
//! - **Correlation per row**, which catches a *permutation* - the failure a
//!   transposed weight produces, where the values are all right and in the
//!   wrong places. Mean error alone passes that; correlation does not.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A minimal `.npy` reader: magic, version, a header dict, then C-order data.
///
/// Written here rather than pulled in as a dependency because it is twenty
/// lines for the one shape this capture writes - float32, C order - and it
/// refuses everything else by name rather than mis-reading it.
fn npy_f32(p: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not a .npy", p.display());
    let (major, hlen_at) = (b[6], 8usize);
    let (hlen, data_at) = match major {
        1 => (
            u16::from_le_bytes([b[hlen_at], b[hlen_at + 1]]) as usize,
            10,
        ),
        2 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        v => panic!("{}: .npy version {v}", p.display()),
    };
    let head = std::str::from_utf8(&b[data_at..data_at + hlen]).expect("header is ascii");
    assert!(
        head.contains("'<f4'") || head.contains("\"<f4\""),
        "{}: not little-endian float32: {head}",
        p.display()
    );
    assert!(
        head.contains("'fortran_order': False"),
        "{}: fortran order: {head}",
        p.display()
    );

    let open = head.find('(').expect("a shape tuple");
    let close = head[open..].find(')').expect("a shape tuple") + open;
    let shape: Vec<usize> = head[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("a dimension"))
        .collect();

    let data = &b[data_at + hlen..];
    let v: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(v.len(), shape.iter().product::<usize>(), "{}", p.display());
    (shape, v)
}

fn npy_ids(p: &Path) -> Vec<u32> {
    // The capture saves everything through `.float()`, so ids arrive as f32.
    // Rounding rather than truncating: 2715.9999 is token 2716.
    let (_, v) = npy_f32(p);
    v.iter().map(|&x| x.round() as u32).collect()
}

fn capture() -> Option<PathBuf> {
    let d = root().join(".golden/cosyvoice");
    d.join("forced_logprobs.npy").is_file().then_some(d)
}

fn model_path() -> Option<PathBuf> {
    let p = root().join("models/tts/cosyvoice3-0.5b/llm.safetensors");
    p.is_file().then_some(p)
}

/// See `docs/TESTING.md`: no default, because two of this box's three cards
/// are running somebody's pipeline and this model is not small.
fn device() -> Option<usize> {
    std::env::var("XABE_COSY_DEVICE").ok()?.parse().ok()
}

/// Pearson correlation, which is what separates "rounding" from "permuted".
fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Log-softmax, which is what upstream compares against - `llm_decoder`'s
/// output goes through `.log_softmax(-1)` before the sampler sees it.
fn log_softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    for v in row.iter_mut() {
        *v -= lse;
    }
}

#[test]
fn the_forced_logprobs_match_cosyvoice_at_every_position() {
    let (Some(dir), Some(model), Some(dev)) = (capture(), model_path(), device()) else {
        println!(
            "SKIP: needs models/tts/cosyvoice3-0.5b/llm.safetensors \
             (tools/convert_cosyvoice.py), .golden/cosyvoice \
             (tools/oracle/capture_cosyvoice.py) and XABE_COSY_DEVICE=<free card>"
        );
        return;
    };

    let llm = xabe_cosy::SpeechLlm::open(&model, dev).expect("open the speech llm");
    let cfg = *llm.config();
    let vocab = cfg.speech_vocab_size;

    let text = npy_ids(&dir.join("text.npy"));
    let instruct = npy_ids(&dir.join("prompt_text.npy"));
    let speech = npy_ids(&dir.join("speech_token.npy"));
    let (want_shape, want) = npy_f32(&dir.join("forced_logprobs.npy"));
    assert_eq!(want_shape, vec![1, speech.len(), vocab], "capture shape");

    // Upstream concatenates the instruct's ids and the utterance's *before*
    // embedding, and the instruct is what carries `<|endofprompt|>`.
    let mut all = instruct.clone();
    all.extend(&text);
    let prompt = llm.prompt(&all, text.len()).expect("prompt");

    // Teacher forcing: the prompt, then every speech token but the last. The
    // last one's own prediction is the step after it and has no target.
    let mut h = llm
        .gpu()
        .zeros((prompt.len + speech.len() - 1) * cfg.hidden_size)
        .expect("scratch");
    llm.gpu()
        .copy_into(&mut h, &prompt.h, 0, prompt.len * cfg.hidden_size)
        .expect("prompt into place");
    for (i, &t) in speech[..speech.len() - 1].iter().enumerate() {
        let e = llm.speech_step(t).expect("speech embedding");
        llm.gpu()
            .copy_into(
                &mut h,
                &e,
                (prompt.len + i) * cfg.hidden_size,
                cfg.hidden_size,
            )
            .expect("forced token into place");
    }

    let n = prompt.len + speech.len() - 1;
    let mut cache = llm.cache();
    let logits = llm.forward(&h, n, &mut cache).expect("forward");
    let all_logits = llm.gpu().download(&logits).expect("download");
    assert_eq!(all_logits.len(), n * vocab);

    // Position `prompt.len - 1` predicts speech token 0; one per forced token
    // after that. Off by one here compares every row against its neighbour,
    // which correlates well enough to look like a rounding problem - so the
    // agreement count below is what actually pins it.
    let mut agree = 0usize;
    let mut worst_abs = 0.0f32;
    let mut worst_corr = 1.0f64;
    let mut first_bad = None;
    for i in 0..speech.len() {
        let at = (prompt.len - 1 + i) * vocab;
        let mut got = all_logits[at..at + vocab].to_vec();
        log_softmax(&mut got);
        let w = &want[i * vocab..(i + 1) * vocab];

        let am = |r: &[f32]| {
            r.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(j, _)| j)
        };
        if am(&got) == am(w) {
            agree += 1;
        } else if first_bad.is_none() {
            first_bad = Some((i, am(&got), am(w)));
        }
        worst_abs = worst_abs.max(
            got.iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max),
        );
        worst_corr = worst_corr.min(correlation(&got, w));
    }

    println!(
        "  {agree}/{} argmax agree, worst |dlogp| {worst_abs:.5}, worst correlation {worst_corr:.6}",
        speech.len()
    );
    if let Some((i, a, b)) = first_bad {
        println!("  first disagreement at position {i}: ours {a}, theirs {b}");
    }

    // Correlation first, because it is the one that separates "the arithmetic
    // is wrong" from "the arithmetic rounds differently". A transposed weight
    // or a missing bias lands here, far from 1.
    assert!(
        worst_corr > 0.9999,
        "worst row correlates {worst_corr:.6} with the reference"
    );
    assert!(
        worst_abs < 0.05,
        "worst log-probability differs by {worst_abs:.5}"
    );
    // Argmax is what decides a token, so it is held to near-exact.
    assert!(
        agree * 100 >= speech.len() * 99,
        "only {agree} of {} argmaxes agree",
        speech.len()
    );
}

#[test]
fn a_prompt_without_the_end_of_prompt_marker_is_refused() {
    // Upstream asserts this and it is worth keeping. The instruct string is
    // what carries `<|endofprompt|>`, so a prompt without it is one the model
    // was never trained to answer - and it does not fail, it produces
    // confident nonsense, which is the case a refusal is cheapest against.
    let (Some(model), Some(dev)) = (model_path(), device()) else {
        println!("SKIP: see the test above");
        return;
    };
    let llm = xabe_cosy::SpeechLlm::open(&model, dev).expect("open");
    match llm.prompt(&[1, 2, 3], 3) {
        Err(xabe_cosy::CosyError::NoEndOfPrompt(t)) => {
            assert_eq!(t, xabe_cosy::LlmConfig::ENDOFPROMPT);
        }
        other => panic!("wanted a refusal, got {:?}", other.map(|p| p.len)),
    }
}
