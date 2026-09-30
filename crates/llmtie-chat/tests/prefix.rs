//! A prompt prefilled on top of a kept prefix against the same prompt cold.
//!
//! Both are this engine, so no oracle is involved. The serving thread keeps
//! one [`llmtie_chat::Prefix`] across turns and prefills only from the first
//! token a turn's prompt does not share with what the cache holds; this is
//! the check that doing so computes the same function as prefilling the whole
//! prompt into an empty cache. The licence between them is arithmetic - a
//! prefill of the tail runs its matmuls at a different row count from a
//! prefill of the whole - and what is being looked for is a fork: a cache
//! that holds the wrong positions, or a truncation that leaves a stale one
//! where attention reads it.

use std::path::PathBuf;

/// How far apart the two may be, relative to the logit span. The same bound
/// as `consistency.rs`, for the same reason.
const BOUND: f32 = 0.02;

/// The shape of the gateway's transcript: a fixed opening, then turns.
const OPENING: &str = "以下是使用者佮小助理的對話。小助理用台語回答，回答愛簡單、親切。\n\
使用者: 你好\n小助理: 你好，我是你的小助理。\n\
使用者: 今仔日天氣按怎？\n小助理: 今仔日天氣真好。\n";

fn model() -> Option<llmtie_chat::ChatModel> {
    let (Some(m), Some(d)) = (
        std::env::var("LLMTIE_CHAT_MODEL").ok().map(PathBuf::from),
        std::env::var("LLMTIE_CHAT_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    ) else {
        eprintln!("SKIP: set LLMTIE_CHAT_MODEL and LLMTIE_CHAT_DEVICE");
        return None;
    };
    Some(llmtie_chat::ChatModel::open(&m, d).expect("open the chat model"))
}

fn span(v: &[f32]) -> f32 {
    v.iter().fold(f32::MIN, |a, &x| a.max(x)) - v.iter().fold(f32::MAX, |a, &x| a.min(x))
}

fn top(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

#[test]
fn a_tail_prefilled_after_a_truncated_cache_matches_the_whole_prompt() {
    let Some(model) = model() else { return };
    let tok = model.tokenizer();
    let vocab = model.config().vocab_size;
    let ids = |text: &str| {
        let mut v = vec![tok.bos()];
        v.extend(tok.encode(text, false));
        v
    };
    // The first turn leaves more in the cache than the second shares with
    // it, so the second has to cut it back before it prefills - which is
    // the case a stale position would show in.
    let a = ids(&format!("{OPENING}使用者: 台北有啥物好耍的？\n小助理:"));
    let b = ids(&format!("{OPENING}使用者: 你會曉講台語無？\n小助理:"));
    let shared = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    assert!(
        shared > 16 && shared < a.len() && shared < b.len(),
        "the two prompts must share a real prefix and then differ: {shared} of {} and {}",
        a.len(),
        b.len(),
    );

    let mut cold = model.cache();
    let whole = model
        .gpu()
        .download(&model.forward_last(&b, &mut cold).expect("whole prefill"))
        .expect("download");

    let mut warm = model.cache();
    model.forward_last(&a, &mut warm).expect("first prompt");
    // A decoded token on top, as a finished turn leaves behind.
    model.forward_last(&[a[1]], &mut warm).expect("one step");
    warm.truncate(shared);
    assert_eq!(warm.len(), shared);
    let tail = model
        .gpu()
        .download(
            &model
                .forward_last(&b[shared..], &mut warm)
                .expect("tail prefill"),
        )
        .expect("download");
    assert_eq!(warm.len(), b.len());

    let (whole, tail) = (&whole[..vocab], &tail[..vocab]);
    let s = span(whole);
    let worst = whole
        .iter()
        .zip(tail)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  {shared} of {} tokens reused; worst logit difference {worst:.4} of a span of {s:.4} \
         ({:.2}%), argmax {} against {}",
        b.len(),
        100.0 * worst / s,
        top(whole),
        top(tail),
    );
    assert!(worst <= BOUND * s, "the two fork: {worst} of a span of {s}");
    assert_eq!(top(whole), top(tail), "the next token differs");
}

#[test]
fn a_completion_through_a_kept_prefix_matches_one_from_empty() {
    let Some(model) = model() else { return };
    let greedy = llmtie_chat::Sampling::greedy(24);
    let stops = vec!["使用者:".to_string(), "\n".to_string()];
    let first = format!("{OPENING}使用者: 台北有啥物好耍的？\n小助理:");
    let second = format!("{OPENING}使用者: 你會曉講台語無？\n小助理:");

    let mut prefix = model.prefix();
    let one = model
        .complete_after(&mut prefix, &first, &greedy, &stops, &mut |_| true)
        .expect("first turn");
    let held = prefix.len();
    assert!(held > 0, "a completion leaves its prompt in the cache");
    // The next turn carries the first in its history, which is how a
    // conversation grows and why the kept run is most of every prompt.
    let third = format!(
        "{OPENING}使用者: 台北有啥物好耍的？\n小助理:{}\n使用者: 你會曉講台語無？\n小助理:",
        one.text
    );
    for prompt in [&second, &third] {
        let kept = model
            .complete_after(&mut prefix, prompt, &greedy, &stops, &mut |_| true)
            .expect("through the kept prefix");
        let cold = model
            .complete(prompt, &greedy, &stops, &mut |_| true)
            .expect("from empty");
        println!("  kept {:?}\n  cold {:?}", kept.text, cold.text);
        assert_eq!(kept.text, cold.text, "the reply changed with the cache");
    }
}
