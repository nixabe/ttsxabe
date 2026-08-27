//! The chat model against `llama-server` running the same GGUF.
//!
//! # What this proves, and what it does not
//!
//! It proves the replacement is a replacement: the same prompt through this
//! engine produces the text llama-server produces, so swapping `--llm-url` for
//! `--llm-model` does not change what the pipeline says.
//!
//! It does **not** prove either of them computes the reference arithmetic.
//! Every other model in this workspace has a float32 🤗 oracle with per-layer
//! taps beside its product comparison; this one has no 🤗 checkpoint on this
//! machine at all - it exists as a GGUF and nothing else. So there is one
//! reference here rather than two, and it is the weaker kind. That is a real
//! gap and is named rather than papered over; `docs/ORACLE.md` carries it.
//!
//! # One test, because the model is 16 GB
//!
//! `cargo test` runs a file's tests on separate threads, and each of these
//! would load its own copy of the weights - four at once, 64 GB, onto a 48 GB
//! card. So this is one test with sections rather than four tests, and the
//! sections say what they are checking. Requiring `--test-threads=1` instead
//! would have been an invisible condition that fails as an out-of-memory error
//! rather than as a message.
//!
//! # Everything runs greedy
//!
//! `gateway.py` samples at 0.3, and a sampled reply is not comparable - two
//! correct implementations drawing from the same distribution give different
//! text. So the capture pins `temperature: 0` with no repetition penalty, and
//! this runs `Sampling::greedy` against it, which makes the reply a function of
//! the prompt alone. The sampler is tested separately, against the
//! distribution rather than against a draw.
//!
//! # Comparing replies is the weak test; comparing decisions is the strong one
//!
//! Free-running text is what the product does and it is a poor measurement.
//! One token going the other way forks the rest of the sentence, so a single
//! near-tie reads as a whole reply disagreeing - and eight replies is eight
//! observations no matter how long they are.
//!
//! So the main check here is **teacher forcing**: the reference reply is fed
//! back in and every position is asked what this engine would have chosen
//! there. That is 125 decisions across the same eight prompts instead of
//! eight, it keeps measuring past the first divergence, and a fork stops
//! hiding everything behind it.
//!
//! It also makes the near-ties answerable rather than arguable. The capture
//! records `n_probs = 2`, so it carries **how much llama-server's chosen token
//! won by** at every step. A disagreement at a step it won by three nats is an
//! arithmetic bug; a disagreement at a step it won by five hundredths is two
//! f16 implementations with different reduction orders, and there is no
//! version of this engine that would not have some. The threshold below is
//! read off that recorded distribution rather than invented.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Step {
    token: String,
    /// How much llama-server's choice beat the runner-up by, in nats.
    margin: Option<f64>,
}

#[derive(serde::Deserialize)]
struct Case {
    user: String,
    prompt: String,
    /// The prompt's ids, from the server that consumed them.
    prompt_tokens: Vec<u32>,
    text: String,
    /// The ids llama-server actually generated.
    ///
    /// Not the re-encoding of the reply text: a generated sequence is **not**
    /// the canonical BPE segmentation of its own output, and re-encoding gives
    /// a different, equally valid cut. Measured on the first case, 17 pieces
    /// against the 15 that were generated - so a per-position comparison built
    /// on re-encoding would be comparing two differently-cut sequences and
    /// calling the mismatch an error.
    tokens: Vec<u32>,
    steps: Vec<Step>,
}

#[derive(serde::Deserialize)]
struct Golden {
    n_predict: usize,
    stops: Vec<String>,
    cases: Vec<Case>,
}

/// How close llama-server's own decision has to have been for a disagreement
/// there to be a rounding difference rather than a bug.
///
/// Read off the capture, not chosen: the recorded margins run from 0.01 nats
/// up to 3.0, and the tight end is a handful of steps in the first tenth. A
/// quarter of a nat is comfortably above every recorded near-tie and far below
/// every confident decision, so it separates the two populations rather than
/// splitting either.
const TIE: f64 = 0.25;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The device to run on.
///
/// Not defaulted to 0: this box has three cards and two of them are running
/// somebody's pipeline. `run.sh` says to check `nvidia-smi` before taking one,
/// and a test that silently lands on a busy card is exactly what that is
/// warning about. So the test is skipped unless a card is named.
fn device() -> Option<usize> {
    std::env::var("XABE_CHAT_DEVICE").ok()?.parse().ok()
}

fn model() -> Option<PathBuf> {
    let p = root().join("models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf");
    p.is_file().then_some(p)
}

fn golden() -> Option<Golden> {
    let s = std::fs::read_to_string(root().join(".golden/chat/llama_server.json")).ok()?;
    Some(serde_json::from_str(&s).expect("parse the capture"))
}

#[test]
fn the_engine_reproduces_llama_server_and_streams_it_correctly() {
    let (Some(m), Some(g), Some(d)) = (model(), golden(), device()) else {
        println!(
            "SKIP: needs the chat GGUF, .golden/chat/llama_server.json \
             (tools/oracle/capture_chat_server.py) and XABE_CHAT_DEVICE=<free card>"
        );
        return;
    };
    let model = xabe_chat::ChatModel::open(&m, d).expect("open the chat model");
    let tok = model.tokenizer();
    let vocab = model.config().vocab_size;
    let greedy = xabe_chat::Sampling::greedy(g.n_predict);

    // 1. Teacher forcing: every decision, not every reply.
    let (mut decisions, mut disagreed, mut tightest) = (0usize, Vec::new(), f64::INFINITY);
    for c in &g.cases {
        // Our own encoding of the prompt against the server's, which is a
        // real check and not a formality - every position below is indexed off
        // this length, so a one-token disagreement would silently compare each
        // row against the wrong reference token.
        assert_eq!(
            tok.encode(&c.prompt, false),
            c.prompt_tokens,
            "{}: prompt tokenization",
            c.user
        );

        let mut ids = vec![tok.bos()];
        ids.extend(&c.prompt_tokens);
        let head = ids.len();
        let reply = c.tokens.clone();
        assert_eq!(reply.len(), c.steps.len(), "{}: capture is ragged", c.user);
        ids.extend(&reply);

        let mut cache = model.cache();
        let logits = model.forward(&ids, &mut cache).expect("forward");
        let all = model.gpu().download(&logits).expect("download");

        for (i, (&want, step)) in reply.iter().zip(&c.steps).enumerate() {
            // Position `head - 1 + i` is the one whose row predicts reply
            // token `i`: the last prompt token predicts the first reply token.
            let row = &all[(head - 1 + i) * vocab..(head + i) * vocab];
            let got = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(j, _)| j as u32);
            decisions += 1;
            if got == want {
                continue;
            }
            let margin = step.margin.unwrap_or(f64::INFINITY);
            tightest = tightest.min(margin);
            disagreed.push((c.user.clone(), i, step.token.clone(), margin));
        }
    }

    for (user, i, token, margin) in &disagreed {
        println!("  differs: {user} step {i} ({token:?}), llama-server won by {margin:.4}");
    }
    println!(
        "  {} decisions, {} differ, all at margins below {:.4}",
        decisions,
        disagreed.len(),
        if disagreed.is_empty() { 0.0 } else { tightest }
    );

    let solid: Vec<_> = disagreed.iter().filter(|(.., m)| *m >= TIE).collect();
    assert!(
        solid.is_empty(),
        "disagreements where llama-server was not close: {solid:?}"
    );
    assert!(decisions >= 100, "only {decisions} decisions in the corpus");
    // A handful of coin-flips is expected; a majority means something else.
    assert!(
        disagreed.len() * 20 <= decisions,
        "{} of {decisions} decisions differ, which is too many to be ties",
        disagreed.len()
    );

    // 2. Free-running replies, which is what the product does. Held to a
    //    weaker bar for the reason in the module docs: one near-tie forks the
    //    rest of the sentence, so this measures forks and not correctness.
    //
    //    Streamed and returned at once, because the browser plays what it is
    //    streamed and never sees `Completion::text`: a mismatch between the two
    //    is a reply spoken differently from how it is displayed, and is
    //    invisible to a test that only checks the return value.
    let mut exact = 0;
    for c in &g.cases {
        let mut streamed = String::new();
        let out = model
            .complete(&c.prompt, &greedy, &g.stops, &mut |chunk| {
                streamed.push_str(chunk);
                true
            })
            .expect("complete");

        assert_eq!(streamed, out.text, "{}: stream against return", c.user);
        // A replacement character is what a mid-character emit looks like from
        // the browser's side, and a byte-level BPE reaches one Han character
        // over more than one token.
        assert!(!streamed.contains('\u{FFFD}'), "{}: {streamed:?}", c.user);

        let got = out.text.trim();
        if got == c.text {
            exact += 1;
        } else {
            println!(
                "  forked: {}\n    want {:?}\n    got  {:?}",
                c.user, c.text, got
            );
        }
    }
    println!("  {exact} of {} replies identical", g.cases.len());
    assert!(
        exact * 4 >= g.cases.len() * 3,
        "only {exact} of {} replies matched",
        g.cases.len()
    );
    assert!(g.cases.len() >= 5, "the corpus should not have shrunk");

    // 3. Cancellation, which is how the gateway drops a reply when the user
    //    talks over it. It has to stop at the next token rather than at the end
    //    of the sentence, or a barged-in turn keeps a card busy generating text
    //    nobody will hear - and the callback must not be called again after it
    //    has said stop, which is what the held-back tail would otherwise do.
    let c = &g.cases[0];
    let mut chunks = 0;
    let out = model
        .complete(&c.prompt, &greedy, &g.stops, &mut |_| {
            chunks += 1;
            false
        })
        .expect("complete");
    assert_eq!(out.stop, xabe_chat::Stop::Cancelled);
    assert_eq!(chunks, 1, "the callback was asked again after saying stop");
    assert!(out.tokens < 8, "{} tokens after one chunk", out.tokens);

    // 4. The seed, which is about the sampler rather than the model. A reply
    //    that cannot be reproduced cannot be diffed against anything, and "it
    //    said something different that time" is not a bug report.
    let s = xabe_chat::Sampling {
        max_tokens: 32,
        ..Default::default()
    };
    let run = |s: &xabe_chat::Sampling| {
        model
            .complete(&c.prompt, s, &g.stops, &mut |_| true)
            .expect("complete")
            .text
    };
    assert_eq!(run(&s), run(&s), "the same seed gave two replies");
    // And a different seed is *allowed* to differ - without this the equality
    // above would also pass on a sampler that ignores the seed entirely.
    let other = xabe_chat::Sampling { seed: 7, ..s };
    println!(
        "  seed {} -> {:?}\n  seed 7 -> {:?}",
        s.seed,
        run(&s),
        run(&other)
    );

    // 5. A sampler setting that means nothing is refused rather than run. Both
    //    of these produce output rather than an error if left unchecked, which
    //    makes a units mistake look like a model problem.
    for bad in [
        xabe_chat::Sampling {
            temperature: -1.0,
            ..s
        },
        xabe_chat::Sampling { top_p: 1.5, ..s },
    ] {
        assert!(matches!(
            model.complete(&c.prompt, &bad, &g.stops, &mut |_| true),
            Err(xabe_chat::ChatError::BadSampler { .. })
        ));
    }
}
