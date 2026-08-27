//! Synthesises one sentence with the whole in-process pipeline.
//!
//! ```sh
//! cargo run --release -p xabe-cosy --example say -- \
//!     2 "台北今仔日好天。" /tmp/out.wav
//! ```
//!
//! The device is the first argument and has no default on purpose: two of this
//! box's three cards are running somebody's pipeline. See `docs/TESTING.md`.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let dev: usize = args
        .next()
        .expect("usage: say <device> <text> [out.wav]")
        .parse()
        .expect("a device number");
    let text = args.next().expect("usage: say <device> <text> [out.wav]");
    let out = args.next().unwrap_or_else(|| "cosy.wav".into());

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("models/tts/cosyvoice3-0.5b");
    let instruct = "You are a helpful assistant. 請用閩南話表達。<|endofprompt|>";

    let started = std::time::Instant::now();
    let cosy = xabe_cosy::Cosy::open(
        &dir,
        &dir.join("voices/taigi-ref.safetensors"),
        instruct,
        dev,
    )
    .expect("open cosyvoice");
    let loaded = started.elapsed();

    let started = std::time::Instant::now();
    let wav = cosy.synthesize(&text).expect("synthesize");
    let took = started.elapsed();

    let seconds = wav.len() as f32 / cosy.sample_rate() as f32;
    println!(
        "  loaded in {:.1}s, {seconds:.2}s of audio in {:.2}s ({:.2}x realtime)",
        loaded.as_secs_f32(),
        took.as_secs_f32(),
        seconds / took.as_secs_f32()
    );

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("create"));
    xabe_audio::write_wav(&mut f, &wav, cosy.sample_rate() as u32).expect("write");
    println!("  wrote {out}");
}
