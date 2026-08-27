//! Completes one prompt with the chat model, for a look at the output.
//!
//!     cargo run --release -p xabe-chat --example chat -- \
//!         models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf 2 "使用者: 你好\n小助理:"
//!
//! Not a test - there is nothing here to assert against. It is the thing to
//! run when a test says the model is wrong and the next question is what it
//! actually said.

use std::io::Write;

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: chat <model.gguf> <device> [prompt]");
    let device: usize = a.next().expect("device").parse().expect("a device number");
    let prompt = a.next().unwrap_or_else(|| "使用者: 你好\n小助理:".into());
    let prompt = prompt.replace("\\n", "\n");

    let t0 = std::time::Instant::now();
    let m = xabe_chat::ChatModel::open(std::path::Path::new(&path), device).expect("open");
    let cfg = m.config();
    println!(
        "loaded in {:.1}s: {} layers, {} heads over {} kv heads, rope {}",
        t0.elapsed().as_secs_f32(),
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.rope_theta,
    );

    let stops = [
        "使用者:".to_string(),
        "小助理:".to_string(),
        "\n\n".to_string(),
    ];
    let s = xabe_chat::Sampling::default();

    let t1 = std::time::Instant::now();
    let mut first = None;
    let out = m
        .complete(&prompt, &s, &stops, &mut |chunk| {
            first.get_or_insert_with(|| t1.elapsed());
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            true
        })
        .expect("complete");

    let dt = t1.elapsed();
    println!(
        "\n\n{} tokens in {:.2}s ({:.1} tok/s), first at {:.2}s, stopped on {:?}",
        out.tokens,
        dt.as_secs_f32(),
        out.tokens as f32 / dt.as_secs_f32(),
        first.unwrap_or(dt).as_secs_f32(),
        out.stop,
    );
}
