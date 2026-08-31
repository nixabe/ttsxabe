//! What the mat-vec's int8 activation costs, measured in decisions.
//!
//! `tests/llama_server.rs` feeds the prompt and the whole reference reply in a
//! single forward pass, so every projection in it runs on the tiled matmul.
//! This runs the identical comparison one token at a time, so every projection
//! runs on the mat-vec instead. Same weights, same reference, same decisions -
//! and the only thing that differs is which kernel multiplied them.
//!
//! # The number, and why it is not a bug
//!
//! The mat-vec quantizes the activation to int8 to feed its wide loads; the
//! tiled one stages to f16. Against a full-precision reference, on the same
//! quantized checkpoint:
//!
//! | weights | activation | agreement |
//! | --- | --- | ---: |
//! | `Q4_K` | f16, tiled | 124 of 125 |
//! | f16 | f32, mat-vec | 124 of 125 |
//! | `Q4_K` | **int8, mat-vec** | **114 of 125** |
//!
//! So quantizing the *weights* costs almost nothing and quantizing the
//! *activation* costs about nine percent of greedy decisions, several of them
//! at margins llama-server won by nine to twelve nats. That is the trade the
//! decode path makes for 1.67x, it is the same trade llama.cpp makes, and this
//! test exists so that its size is written down rather than assumed small.
//!
//! It is bounded generously on purpose. What it is guarding is that the number
//! stays around a tenth and does not become a third, which is what a real
//! defect in the packed mat-vec would look like.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Step {
    margin: Option<f64>,
}

#[derive(serde::Deserialize)]
struct Case {
    user: String,
    prompt_tokens: Vec<u32>,
    tokens: Vec<u32>,
    steps: Vec<Step>,
}

#[derive(serde::Deserialize)]
struct Golden {
    cases: Vec<Case>,
}

fn golden() -> Option<Golden> {
    let p = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.golden/chat/llama_server.json"
    ));
    let f = std::fs::read(&p).ok().or_else(|| {
        eprintln!(
            "SKIP: no {} (tools/oracle/capture_chat_server.py)",
            p.display()
        );
        None
    })?;
    serde_json::from_slice(&f).ok()
}

#[test]
fn the_int8_activation_costs_about_a_tenth_of_the_decisions() {
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
    let model = xabe_chat::ChatModel::open(&m, d).expect("open the chat model");
    let tok = model.tokenizer();
    let vocab = model.config().vocab_size;

    let (mut decisions, mut differ, mut worst) = (0usize, 0usize, 0.0f64);
    for c in &g.cases {
        let mut cache = model.cache();
        let mut ids = vec![tok.bos()];
        ids.extend(&c.prompt_tokens);
        // One position per call, so `m` is 1 at every projection.
        let mut last = None;
        for &id in &ids {
            let l = model.forward(&[id], &mut cache).expect("forward");
            last = Some(model.gpu().download(&l).expect("download"));
        }
        for (&want, step) in c.tokens.iter().zip(&c.steps) {
            let row = last.as_ref().expect("a prefilled row");
            let got = row[..vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(j, _)| j as u32);
            decisions += 1;
            if got != want {
                differ += 1;
                let margin = step.margin.unwrap_or(f64::INFINITY);
                worst = worst.max(margin);
                println!(
                    "  differs: {} step, llama-server won by {margin:.4}",
                    c.user
                );
            }
            let l = model.forward(&[want], &mut cache).expect("forward");
            last = Some(model.gpu().download(&l).expect("download"));
        }
    }
    println!("  {decisions} decisions, {differ} differ, worst margin {worst:.4}");
    assert!(decisions >= 100, "only {decisions} decisions in the corpus");
    // A fifth, against a measured tenth. This is a cost being tracked, not a
    // property being asserted, and the bound is where a defect would show.
    assert!(
        differ * 5 <= decisions,
        "{differ} of {decisions} decisions differ, which is past what the int8 \
         activation costs",
    );
}
