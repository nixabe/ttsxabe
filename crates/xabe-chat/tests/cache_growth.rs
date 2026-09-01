//! A decode that runs past the cache's capacity computes the same thing as one
//! that never had to grow.
//!
//! `consistency.rs` asks whether prefill and stepwise decode agree, and they do,
//! at 199 tokens. That is under the 256 the cache is first allocated at, so it
//! never grows a cache, and what growth has to preserve is the one thing it did
//! not check.
//!
//! The cache is **head-major**: keys are `[kv_head, cap, head_dim]` and values
//! are `[kv_head, head_dim, cap]`, so `cap` is a *stride* in both, not just a
//! length. Doubling it moves every head but the first. A growth that copies the
//! live prefix flat - which is what a position-major cache would want - leaves
//! head 0 correct and scrambles the other seven, and the model answers one
//! fluent sentence off the still-correct head before collapsing into noise.
//!
//! That is why this compares *logits at every position* rather than the reply:
//! the failure is silent, gradual, and reads as the model being bad rather than
//! as arithmetic being wrong.

use std::path::PathBuf;

/// How far apart two paths through the same function may be, relative to the
/// logit span.
///
/// Looser than `consistency.rs`'s 0.02, and measured rather than chosen. The
/// two paths share no reduction order - a 320-row tiled prefill against a
/// mat-vec over one row - and on this checkpoint that floor runs to 2.1%: the
/// worst five positions come in at 2.1, 1.8, 1.8, 1.7 and 1.5%, at positions
/// 202, 205, 293, 259 and 221. Scattered on both sides of the capacity, which
/// is what rounding looks like.
///
/// The bug this guards against does not sit anywhere near that floor. It put
/// 65 of 120 positions over 2% in an unbroken run from 256 to the end, and the
/// worst of them at 30.4 on a span of 37.9 - 80%. So 4% is twice the measured
/// noise and twenty times below the failure, and no threshold in that gap
/// decides anything.
const BOUND: f32 = 0.04;

/// Tokens prefilled before stepping, chosen to sit under the 256-token floor
/// the cache is first allocated at, so that the steps after it force a growth.
const PREFILL: usize = 200;

/// Tokens stepped one at a time, enough to cross 256 and keep going past it.
const STEPS: usize = 120;

fn model() -> Option<xabe_chat::ChatModel> {
    let (Some(m), Some(d)) = (
        std::env::var("XABE_CHAT_MODEL").ok().map(PathBuf::from),
        std::env::var("XABE_CHAT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    ) else {
        eprintln!("SKIP: set XABE_CHAT_MODEL and XABE_CHAT_DEVICE");
        return None;
    };
    Some(xabe_chat::ChatModel::open(&m, d).expect("open the chat model"))
}

/// Real text, long enough to step past a capacity boundary.
///
/// The prompt the consistency capture uses, repeated until it is long enough.
/// Repetition is fine here - nothing is being asked about the *quality* of the
/// continuation, only that two ways of computing it agree.
fn ids(model: &xabe_chat::ChatModel, want: usize) -> Vec<u32> {
    let tok = model.tokenizer();
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.golden/chat/consistency_prompt.txt"
    ))
    .unwrap_or_else(|_| {
        "台灣上好食的物件是啥物？小助理講，台灣有真濟好食的物件，親像滷肉飯、牛肉麵佮珍珠奶茶。"
            .to_string()
    });
    let mut ids = vec![tok.bos()];
    while ids.len() < want {
        ids.extend(tok.encode(&text, false));
    }
    ids.truncate(want);
    ids
}

#[test]
fn a_decode_past_the_cache_capacity_matches_one_that_never_grew() {
    let Some(model) = model() else { return };
    let vocab = model.config().vocab_size;
    let ids = ids(&model, PREFILL + STEPS);

    // The control. One pass, so the cache is allocated at 512 up front and
    // growth never runs. Every position's logits, not only the last.
    let mut whole = model.cache();
    let batched = model
        .gpu()
        .download(&model.forward(&ids, &mut whole).expect("batched forward"))
        .expect("download");
    assert_eq!(whole.len(), ids.len());

    // The generation path: prefill, then one token at a time. This is the one
    // that crosses 256 and grows.
    let mut step = model.cache();
    model
        .forward(&ids[..PREFILL], &mut step)
        .expect("prefill forward");
    let cap_before = step.len();
    assert_eq!(cap_before, PREFILL);

    // The difference at every position, relative to that position's span.
    let mut rel: Vec<(usize, f32)> = Vec::with_capacity(STEPS);
    for (t, &id) in ids.iter().enumerate().skip(PREFILL) {
        let row_step = model
            .gpu()
            .download(&model.forward(&[id], &mut step).expect("step forward"))
            .expect("download");
        // Position `t` is predicted by the row *at* `t`, which the step that
        // consumed `ids[t]` produced.
        let row = &batched[t * vocab..(t + 1) * vocab];
        let span = row.iter().fold(f32::MIN, |a, &v| a.max(v))
            - row.iter().fold(f32::MAX, |a, &v| a.min(v));
        let w = row
            .iter()
            .zip(&row_step[..vocab])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        rel.push((t, w / span));
    }

    // Where the differences are, not only how big. The signature of the bug is
    // an unbroken run starting at the capacity - scattered singletons are the
    // rounding floor of a tiled prefill against a mat-vec step.
    let mut worst = rel.clone();
    worst.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let over: Vec<usize> = rel
        .iter()
        .filter(|&&(_, r)| r > BOUND)
        .map(|&(t, _)| t)
        .collect();
    println!("  worst five positions: {:?}", &worst[..5.min(worst.len())]);
    println!("  over {BOUND}: {over:?}");

    let (at, r) = worst[0];
    assert!(
        r <= BOUND,
        "position {at} differs by {:.1}% of its logit span, over the {:.0}% \
         this comparison is allowed. Positions over: {over:?}. An unbroken run \
         from 256 to the end is the cache growing without re-striding a \
         head-major cache; anything else is not.",
        100.0 * r,
        100.0 * BOUND,
    );
}
