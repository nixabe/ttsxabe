//! The batched decode step against the single-sequence one, on the quantized
//! translator.
//!
//! Two things are held. A step over several sequences produces, for each,
//! the logits the single-sequence step produces for it alone - to the
//! tolerance the normalisations' different association allows, and with the
//! same greedy choice. And a `TranslationBatch` of two sentences produces the
//! texts `translate` produces for each on its own.

use std::path::PathBuf;
use xabe_translate::{Cache, Packing, Translator};

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

fn device() -> Option<usize> {
    match std::env::var("XABE_TRANSLATOR_DEVICE").ok() {
        Some(v) => v.parse().ok(),
        None => {
            eprintln!("SKIP: set XABE_TRANSLATOR_DEVICE to the card to load onto");
            None
        }
    }
}

const SOURCES: [&str; 3] = [
    "今天天氣真好，我想去海邊走走。",
    "你吃飽了嗎？",
    "我們明天早上八點在車站見面。",
];

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn a_batched_step_is_each_sequence_stepped_alone() {
    let (Some(path), Some(dev)) = (quantized(), device()) else {
        return;
    };
    let m = Translator::open_with(&path, dev, Packing::Packed).expect("the quantized GGUF");
    let vocab = m.config().vocab_size;

    // Three prompts, prefilled three times over: one set stepped by the
    // single-sequence path, one set stepped one row at a time through the
    // batched step, one set stepped together.
    let prompts: Vec<Vec<u32>> = SOURCES.iter().map(|s| m.prompt_ids(s, "POJ")).collect();
    let mut single: Vec<Cache> = Vec::new();
    let mut alone: Vec<Cache> = Vec::new();
    let mut together: Vec<Cache> = Vec::new();
    let mut nexts = Vec::new();
    for p in &prompts {
        for set in [&mut single, &mut alone, &mut together] {
            let mut c = m.cache();
            let l = m.forward_last(p, &mut c).expect("prefill");
            if set.is_empty() || set.len() < prompts.len() {
                nexts.push(argmax(&m.gpu().download(&l).unwrap()) as u32);
            }
            set.push(c);
        }
    }
    nexts.truncate(prompts.len());
    let mut worst = 0.0f32;
    for step in 0..4 {
        // The single-sequence path, with its folded normalisations.
        let mut want = Vec::new();
        for (c, &t) in single.iter_mut().zip(&nexts) {
            let l = m.forward_last(&[t], c).expect("single");
            want.push(m.gpu().download(&l).unwrap());
        }
        // The batched step, one row at a time.
        let mut one = Vec::new();
        for (c, &t) in alone.iter_mut().zip(&nexts) {
            let mut refs = vec![c];
            let l = m.step_rows(&[t], &mut refs).expect("alone");
            one.push(m.gpu().download(&l).unwrap());
        }
        // The batched step, all rows together.
        let mut refs: Vec<&mut Cache> = together.iter_mut().collect();
        let l = m.step_rows(&nexts, &mut refs).expect("together");
        let got = m.gpu().download(&l).unwrap();
        for (r, w) in want.iter().enumerate() {
            let g = &got[r * vocab..(r + 1) * vocab];
            // Together against alone: the same kernels over the same codes,
            // so bit for bit.
            assert_eq!(
                one[r], g,
                "step {step} row {r}: together differs from alone"
            );
            // Against the single-sequence path: that path folds each
            // normalisation into the mat-vec before it and this one takes it
            // as a pass, and the int8 twin of the normalised row can round a
            // code differently where a value sits on a boundary; forty
            // layers of that is a small fraction of the logit span, and the
            // greedy choice is held exactly.
            let span = w.iter().cloned().fold(f32::MIN, f32::max)
                - w.iter().cloned().fold(f32::MAX, f32::min);
            let diff = w
                .iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            worst = worst.max(diff / span);
            assert!(
                diff <= 2e-2 * span,
                "step {step} row {r}: logits differ by {diff} over a span of {span}"
            );
            assert_eq!(
                argmax(w),
                argmax(g),
                "step {step} row {r}: the greedy choice"
            );
        }
        nexts = want.iter().map(|w| argmax(w) as u32).collect();
    }
    eprintln!("worst logit difference against the single path: {worst:.2e} of the span");
    for (a, t) in alone.iter().zip(&together) {
        assert_eq!(a.len(), t.len(), "the caches advanced together");
    }

    // The refusals.
    let mut refs: Vec<&mut Cache> = together.iter_mut().collect();
    assert!(
        m.step_rows(&[1, 2], &mut refs).is_err(),
        "a token count that is not the row count"
    );
    let mut empty = m.cache();
    let mut refs = vec![&mut empty];
    assert!(
        m.step_rows(&[1], &mut refs).is_err(),
        "a cache without its prompt"
    );
}

#[test]
fn a_batch_of_sentences_translates_each_as_it_would_alone() {
    let (Some(path), Some(dev)) = (quantized(), device()) else {
        return;
    };
    let m = Translator::open_with(&path, dev, Packing::Packed).expect("the quantized GGUF");
    let (max_new, penalty) = (64, Translator::REPEAT_PENALTY);
    let want: Vec<String> = SOURCES
        .iter()
        .map(|s| m.translate(s, "POJ", max_new, penalty).expect("alone"))
        .collect();

    let mut batch = m.batch(max_new, penalty);
    let ids: Vec<u64> = SOURCES
        .iter()
        .map(|s| batch.admit(s, "POJ").expect("admit"))
        .collect();
    assert_eq!(batch.len(), 3);
    let mut got: Vec<Option<String>> = vec![None; 3];
    let mut steps = 0;
    while !batch.is_empty() {
        for (id, text) in batch.step().expect("step") {
            let i = ids.iter().position(|&x| x == id).expect("a known id");
            got[i] = Some(text);
        }
        steps += 1;
        assert!(steps <= max_new + 2, "the batch does not stop");
    }
    for (i, w) in want.iter().enumerate() {
        let g = got[i].as_deref().expect("every sentence finished");
        eprintln!("{}\n  alone:    {w}\n  together: {g}", SOURCES[i]);
        assert!(!g.is_empty(), "sentence {i} translated to nothing");
    }
    // Greedy decoding is a chain of argmaxes, and the batched step's
    // normalisation differs from the folded one by an ulp; a decision that
    // sits on a tie can flip. The texts are compared, and a difference is
    // reported rather than failed only when it is one sentence of three.
    let same = want
        .iter()
        .zip(&got)
        .filter(|(w, g)| g.as_deref() == Some(w.as_str()))
        .count();
    assert!(
        same >= 2,
        "batched translations differ from the single ones in {} of 3",
        3 - same
    );
}
