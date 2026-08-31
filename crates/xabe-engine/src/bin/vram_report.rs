//! What the pipeline actually occupies on one card, stage by stage.
//!
//! The question this answers is not "how big is the checkpoint" but "does the
//! whole pipeline fit on one card", and those differ by whichever container
//! the weights arrived in. A `Q4_K_M` GGUF is a 7.9 GB file that used to land
//! as 26.5 GB of f16 because the loader unpacked on read; with the packed
//! matmul it lands as 7.9 GB. Nothing but a measurement distinguishes those
//! two situations from outside, so this measures.
//!
//! Stages are loaded **cumulatively and in one process**, because that is the
//! configuration being asked about. Loading them one at a time and adding up
//! the peaks would answer a different question and give a smaller number.
//!
//! ```sh
//! cargo run --release -p xabe-engine --bin xabe-vram -- \
//!     --device 0 \
//!     --llm        models/breeze2-8b-Q4_K_M.gguf \
//!     --translator models/taigi-translator-13b-Q4_K_M.gguf \
//!     --asr        models/asr/breeze-asr-26 \
//!     --tts        models/tts/mms-tts-nan \
//!     --cosyvoice  models/tts/cosyvoice3-0.5b
//! ```
//!
//! The figure comes from `nvidia-smi`, not from the allocator: it is the
//! number a neighbour sees, it includes the CUDA context and the runtime's own
//! reservations, and it is what "fits on one card" is actually about.

use clap::Parser;
use std::path::PathBuf;
use std::process::Command;

/// Loads stages onto one card and reports what each one costs.
#[derive(Parser, Debug)]
#[command(name = "xabe-vram")]
struct Args {
    /// Which card. Check `nvidia-smi` first - this allocates for real.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// The chat GGUF, quantized or f16.
    #[arg(long)]
    llm: Option<PathBuf>,

    /// The translator, as a GGUF file or a 🤗 directory.
    #[arg(long)]
    translator: Option<PathBuf>,

    /// The Whisper checkpoint directory.
    #[arg(long)]
    asr: Option<PathBuf>,

    /// The VITS checkpoint directory.
    #[arg(long)]
    tts: Option<PathBuf>,

    /// The converted Tacotron2 + WaveGlow directory, as the other alternative
    /// synthesiser.
    ///
    /// Loaded before the ASR so that a run naming it and no VITS still charges
    /// the CUDA context to a synthesiser rather than to the ASR.
    #[arg(long)]
    tacotron2: Option<PathBuf>,

    /// The converted CosyVoice directory, as the alternative synthesiser.
    ///
    /// Its three GPU-resident sub-models are opened directly rather than
    /// through `Cosy::open`, which additionally wants a voice bundle from
    /// `tools/make_cosyvoice_voice.py`. The bundle is four small tensors and
    /// occupies nothing worth reporting; these three are the weights.
    #[arg(long)]
    cosyvoice: Option<PathBuf>,
}

/// Megabytes in use on `device`, as `nvidia-smi` reports them.
///
/// Whole-card rather than per-process: the CUDA context, the driver's own
/// reservations and any fragmentation all count against the 48 GiB, and a
/// per-process figure would quietly omit them.
fn used_mib(device: usize) -> u64 {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            &format!("--id={device}"),
        ])
        .output()
        .expect("nvidia-smi should be on PATH to measure anything");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("nvidia-smi should report a number")
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let args = Args::parse();
    let dev = args.device;

    let base = used_mib(dev);
    println!("card {dev}: {base} MiB already in use before this process");
    println!();
    println!("{:<14} {:>12} {:>12}", "stage", "delta MiB", "total MiB");

    let mut last = base;
    let mut step = |name: &str| {
        let now = used_mib(dev);
        println!("{:<14} {:>12} {:>12}", name, now - last, now - base);
        last = now;
    };

    // Held in scope to the end of main: dropping one would free its weights
    // and make every later figure a measurement of a different configuration.
    //
    // TTS goes first, and that is not arbitrary. The CUDA context is created by
    // whichever stage opens the device first and costs a few hundred MiB that
    // belong to no stage, so it is charged to the smallest one - 36 M
    // parameters - where it is obviously the context rather than the weights.
    let _tts = args.tts.as_ref().map(|p| {
        let m = xabe_tts::GpuModel::open(p, dev).expect("the TTS checkpoint");
        step("tts + ctx");
        m
    });
    let _taco = args.tacotron2.as_ref().map(|p| {
        // `sigma` and the seed steer synthesis, not residency; the defaults
        // from `taco_bench` keep this the same object the pipeline builds.
        let m = xabe_taco::Taco::open(p, dev, None, 0).expect("the Tacotron2 checkpoint");
        step("tacotron2");
        m
    });
    let _asr = args.asr.as_ref().map(|p| {
        let m = xabe_asr::AsrModel::open(p, dev).expect("the ASR checkpoint");
        step("asr");
        m
    });
    let _llm = args.llm.as_ref().map(|p| {
        let m = xabe_chat::ChatModel::open(p, dev).expect("the chat GGUF");
        step("llm");
        m
    });
    let _tr = args.translator.as_ref().map(|p| {
        let m = xabe_translate::Translator::open(p, dev).expect("the translator");
        step("translator");
        m
    });

    let _cosy = args.cosyvoice.as_ref().map(|p| {
        let llm = xabe_cosy::SpeechLlm::open(&p.join("llm.safetensors"), dev)
            .expect("the CosyVoice speech LM");
        let flow =
            xabe_cosy::Flow::open(&p.join("flow.safetensors"), dev).expect("the CosyVoice flow");
        let hift = xabe_cosy::Vocoder::open(&p.join("hift.safetensors"), dev)
            .expect("the CosyVoice vocoder");
        step("cosyvoice");
        (llm, flow, hift)
    });

    // The VAD is deliberately absent: `xabe_vad::open` takes no device
    // ordinal, because Silero is 1.8 M parameters of CPU arithmetic and never
    // reaches the card. A stage that occupies no VRAM is worth saying once
    // rather than leaving a reader to wonder which stage was forgotten.
    println!("{:<14} {:>12} {:>12}", "vad (cpu)", 0, last - base);

    let total = used_mib(dev) - base;
    let capacity = 49152u64;
    println!();
    println!(
        "everything resident: {total} MiB ({:.1} GiB) of a {} MiB card",
        total as f64 / 1024.0,
        capacity,
    );
    println!(
        "headroom for caches and activations: {} MiB",
        capacity.saturating_sub(base + total),
    );
}
