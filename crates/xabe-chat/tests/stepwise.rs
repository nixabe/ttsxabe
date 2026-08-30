//! Teacher forcing one token at a time, which is the decode path.
//!
//! `tests/llama_server.rs` feeds the prompt and the whole reference reply in a
//! single forward pass, so every projection in it runs on the tiled matmul. This
//! runs the identical comparison one token at a time, so every projection runs
//! on the mat-vec instead. Same weights, same reference, same decisions - and
//! the only thing that differs is which kernel multiplied them.
//!
//! It exists because those two kernels approximate a packed weight differently:
//! the mat-vec hands the checkpoint's blocks to an integer dot product against
//! an int8 activation, and the tiled one dequantizes to f16 first. This test is
//! how much that is worth, in decisions.

use std::path::PathBuf;

/// How close llama-server's own decision has to have been for a disagreement
/// there to be a rounding difference rather than a bug. See `llama_server.rs`,
/// which reads it off the same capture.
const TIE: f64 = 0.25;

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
fn one_token_at_a_time_is_the_mat_vec_and_agrees_better() {
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
    // The same threshold `llama_server.rs` uses, and the point of the test is
    // that this path meets it where the batched one does not: a packed weight
    // multiplied as integers against an int8 activation is llama.cpp's own
    // arithmetic, and a packed weight dequantized to f16 is not.
    assert!(
        worst < TIE,
        "a disagreement at {worst:.4} nats, which is not a tie",
    );
}
