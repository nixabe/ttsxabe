//! Times CosyVoice3 stage by stage: the speech LLM, the flow, the excitation
//! and the vocoder, each synchronised before its clock stops. The totals are
//! therefore slightly pessimistic against `Cosy::synthesize`, which does not
//! synchronise between stages; the split is what this is for.
//!
//! ```sh
//! cargo run --release -p xabe-cosy --example stages -- 0 "台北今仔日好天。" [runs] [mel.f32]
//! ```
//!
//! With a fourth argument the last run's mel is written to it as raw little
//! endian f32, `[80, frames]`, so two builds can be compared at the flow's
//! output rather than at the waveform, whose phase the vocoder's excitation
//! makes a poor witness.

use std::path::PathBuf;
use std::time::Instant;
use xabe_cosy::{
    Dither, F0Predictor, Flow, SourceConfig, SpeechLlm, Tokenizer, Vocoder, Voice, excitation,
};
use xabe_st::StFile;

fn main() {
    let mut args = std::env::args().skip(1);
    let dev: usize = args
        .next()
        .expect("usage: stages <device> <text> [runs]")
        .parse()
        .expect("a device");
    let text = args.next().expect("usage: stages <device> <text> [runs]");
    let runs: usize = args.next().map_or(3, |r| r.parse().expect("runs"));
    let mel_out = args.next();

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("models/tts/cosyvoice3-0.5b");
    let instruct = "You are a helpful assistant. 請用閩南話表達。<|endofprompt|>";

    let t = Instant::now();
    let tok = Tokenizer::from_dir(&dir.join("CosyVoice-BlankEN")).expect("tokenizer");
    let _llm = SpeechLlm::open(&dir.join("llm.safetensors"), dev).expect("llm");
    let t_llm = t.elapsed();
    let t = Instant::now();
    let flow = Flow::open(&dir.join("flow.safetensors"), dev).expect("flow");
    let t_flow = t.elapsed();
    let t = Instant::now();
    let voc = Vocoder::open(&dir.join("hift.safetensors"), dev).expect("vocoder");
    let hift = StFile::open(dir.join("hift.safetensors")).expect("hift");
    let f0 = F0Predictor::bind(&hift, voc.gpu()).expect("f0");
    let source_w = hift
        .tensor_shaped("m_source.l_linear.weight", &[1, xabe_cosy::HARMONICS])
        .expect("w")
        .to_vec();
    let source_b = hift
        .tensor_shaped("m_source.l_linear.bias", &[1])
        .expect("b")[0];
    let voice = Voice::open(
        &dir.join("voices/taigi-ref.safetensors"),
        flow.config().mel_dim,
    )
    .expect("voice");
    let t_voc = t.elapsed();
    println!(
        "open: llm {:.1}s, flow {:.1}s, vocoder+voice {:.1}s",
        t_llm.as_secs_f32(),
        t_flow.as_secs_f32(),
        t_voc.as_secs_f32()
    );

    let cosy = xabe_cosy::Cosy::open(
        &dir,
        &dir.join("voices/taigi-ref.safetensors"),
        instruct,
        dev,
    )
    .expect("cosy");
    let ids = tok.encode(&text);
    for run in 0..runs {
        let t = Instant::now();
        let tokens = cosy.speech_tokens(&ids).expect("tokens");
        let t_tok = t.elapsed();

        let t = Instant::now();
        let (mel, frames) = flow
            .mel(
                &voice.prompt_token,
                &tokens,
                &voice.prompt_feat,
                &voice.embedding,
                &voice.cfm_noise,
            )
            .expect("mel");
        flow.gpu().synchronize().expect("sync");
        let t_mel = t.elapsed();

        let gpu = voc.gpu();
        let t = Instant::now();
        let gmel = gpu.upload(&mel).expect("upload");
        let f0v = f0.predict(gpu, &gmel, frames).expect("f0");
        let samples = frames * voc.config().hop();
        let dither = Dither::seeded(samples, 0x5F1E_C0DE);
        let src = excitation(&f0v, &SourceConfig::default(), &dither, &source_w, source_b)
            .expect("source");
        let gsrc = gpu.upload(&src).expect("upload");
        gpu.synchronize().expect("sync");
        let t_src = t.elapsed();

        let t = Instant::now();
        let wav = voc.decode(&gmel, frames, &gsrc, samples).expect("decode");
        let t_dec = t.elapsed();
        let secs = wav.len() as f32 / 24000.0;
        if run + 1 == runs
            && let Some(path) = &mel_out
        {
            let bytes: Vec<u8> = mel.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(path, bytes).expect("write the mel");
        }
        println!(
            "run {run}: {} text ids, {} speech tokens, {frames} frames, {secs:.2}s audio | llm {:.0} ms ({:.2} ms/token) | flow {:.0} ms | f0+source {:.0} ms | vocoder {:.0} ms | total {:.0} ms",
            ids.len(),
            tokens.len(),
            t_tok.as_secs_f32() * 1e3,
            t_tok.as_secs_f32() * 1e3 / tokens.len().max(1) as f32,
            t_mel.as_secs_f32() * 1e3,
            t_src.as_secs_f32() * 1e3,
            t_dec.as_secs_f32() * 1e3,
            (t_tok + t_mel + t_src + t_dec).as_secs_f32() * 1e3
        );
    }
}
