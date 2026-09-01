//! The batched forward against the same tokens fed one at a time.
//!
//! Both are this engine, so no oracle is involved and nothing about llama.cpp
//! is being asserted. The two must agree: prefilling `n` positions in one pass
//! and decoding them one by one compute the same function, and the only licence
//! between them is arithmetic - a different kernel, a different reduction order.
//!
//! It runs at 199 tokens, which is **under** the 256 positions the cache is
//! first allocated at, so nothing here ever grows one. That is not an oversight
//! to fix by lengthening this prompt - the length is load-bearing for the
//! comparison below - but it does mean this test says nothing about a decode
//! that outgrows its cache, and one that did was wrong for months.
//! `cache_growth.rs` is that case; see `docs/TESTING.md`.
//!
//! It exists because the two disagreed in a way precision could not explain.
//! The batched path picked a different token from llama-server at 10 of 105
//! teacher-forced decisions and the stepwise path at 1, and *replacing the
//! batched matmul entirely* - f16 tensor cores for an int8 integer kernel - did
//! not move a single one of them. Two unrelated arithmetics agreeing to the
//! token is not what rounding looks like.

use std::path::PathBuf;

/// How far apart the two paths may be, relative to the logit span.
///
/// Generous: one runs a 128x128 tiled matmul over 200 rows and the other a
/// mat-vec over one, and their reduction orders share nothing. What it is
/// looking for is a fork, not a rounding difference.
const BOUND: f32 = 0.02;

#[test]
fn a_batched_prefill_matches_the_same_tokens_one_at_a_time() {
    let (Some(m), Some(d)) = (
        std::env::var("XABE_CHAT_MODEL").ok().map(PathBuf::from),
        std::env::var("XABE_CHAT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    ) else {
        eprintln!("SKIP: set XABE_CHAT_MODEL and XABE_CHAT_DEVICE");
        return;
    };
    let model = xabe_chat::ChatModel::open(&m, d).expect("open the chat model");
    let tok = model.tokenizer();
    let vocab = model.config().vocab_size;

    // Past GEMV_MAX_M, so the batched run takes the tiled matmul and the
    // stepwise one cannot.
    let mut ids = vec![tok.bos()];
    // The capture's own prompt when it is there, because length is the thing
    // this is testing: the two paths were measured to agree at 46 tokens and
    // the comparison that disagreed runs at 199.
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.golden/chat/consistency_prompt.txt"
    ))
    .unwrap_or_else(|_| {
        "台灣上好食的物件是啥物？小助理講，台灣有真濟好食的物件，親像滷肉飯、牛肉麵佮珍珠奶茶。"
            .to_string()
    });
    ids.extend(tok.encode(&text, false));
    assert!(ids.len() > xabe_cuda::GEMV_MAX_M, "the batch must be tiled");

    let mut c1 = model.cache();
    let batched = model
        .gpu()
        .download(&model.forward(&ids, &mut c1).unwrap())
        .unwrap();

    // Every position, not only the last. A causal model computes the same
    // function at each of them, and a fork that only shows in the middle of a
    // sequence is exactly what a mask off by one looks like.
    let mut c2 = model.cache();
    let mut worst_at = (0usize, 0.0f32, 0.0f32);
    let mut forks = 0usize;
    for (t, &id) in ids.iter().enumerate() {
        let step = model
            .gpu()
            .download(&model.forward(&[id], &mut c2).unwrap())
            .unwrap();
        let row = &batched[t * vocab..(t + 1) * vocab];
        let span = row.iter().fold(f32::MIN, |a, &v| a.max(v))
            - row.iter().fold(f32::MAX, |a, &v| a.min(v));
        let w = row
            .iter()
            .zip(&step[..vocab])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i)
        };
        if top(row) != top(&step[..vocab]) {
            forks += 1;
        }
        if w / span > worst_at.1 / worst_at.2.max(1e-9) {
            worst_at = (t, w, span);
        }
    }
    println!(
        "  worst position {} at {:.4} of a span of {:.4} ({:.2}%); {forks} of {} argmaxes fork",
        worst_at.0,
        worst_at.1,
        worst_at.2,
        100.0 * worst_at.1 / worst_at.2,
        ids.len(),
    );
    let last = (ids.len() - 1) * vocab;
    let batched = &batched[last..last + vocab];
    let mut c3 = model.cache();
    let mut stepwise = Vec::new();
    for &id in &ids {
        stepwise = model
            .gpu()
            .download(&model.forward(&[id], &mut c3).unwrap())
            .unwrap();
    }

    let span = batched.iter().fold(f32::MIN, |a, &v| a.max(v))
        - batched.iter().fold(f32::MAX, |a, &v| a.min(v));
    let worst = batched
        .iter()
        .zip(&stepwise[..vocab])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    println!(
        "  worst logit difference {worst:.4} of a span of {span:.4} ({:.2}%), \
         argmax {} against {}",
        100.0 * worst / span,
        top(batched),
        top(&stepwise[..vocab]),
    );
    assert!(
        worst <= BOUND * span,
        "the two paths fork: {worst} of a span of {span}",
    );
}
