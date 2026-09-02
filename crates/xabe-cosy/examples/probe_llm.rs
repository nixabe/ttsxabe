//! The speech LLM's teacher-forced log-probabilities, written out for a
//! build-to-build comparison in the terms the oracle test uses.
//!
//! `probe_llm <device> <out dir> [tokens.u32]`: without a token file the
//! model samples the utterance itself and writes `tokens.u32`; with one it
//! forces those. Either way it writes `prefill.f32` - every position's
//! log-probs from one forward over the whole sequence - and `decode.f32`,
//! the same positions produced one token at a time through the cache, so
//! both paths of `forward` are compared.
use std::path::PathBuf;
use xabe_cosy::{LlmConfig, Rng, SpeechLlm, Tokenizer, ras_sample};

fn log_softmax(row: &mut [f32]) {
    let m = row.iter().cloned().fold(f32::MIN, f32::max);
    let s: f32 = row.iter().map(|v| (v - m).exp()).sum();
    let l = m + s.ln();
    row.iter_mut().for_each(|v| *v -= l);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dev: usize = args.next().expect("device").parse().expect("device");
    let out = args.next().expect("out dir");
    let forced = args.next();
    std::fs::create_dir_all(&out).expect("dir");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("models/tts/cosyvoice3-0.5b");
    let tok = Tokenizer::from_dir(&dir.join("CosyVoice-BlankEN")).expect("tokenizer");
    let llm = SpeechLlm::open(&dir.join("llm.safetensors"), dev).expect("llm");
    let cfg = *llm.config();
    let (h, vocab) = (cfg.hidden_size, cfg.speech_vocab_size);
    let gpu = llm.gpu();

    let mut all = tok.encode("You are a helpful assistant. 請用閩南話表達。<|endofprompt|>");
    let text = tok.encode("台北今仔日好天，咱來去公園行行咧。");
    all.extend(&text);
    let prompt = llm.prompt(&all, text.len()).expect("prompt");

    let tokens: Vec<u32> = match forced {
        Some(p) => std::fs::read(&p)
            .expect("tokens")
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        None => {
            let mut cache = llm.cache();
            let mut logits = llm
                .forward(&prompt.h, prompt.len, &mut cache)
                .expect("forward");
            let mut at = prompt.len;
            let mut rng = Rng::new(xabe_cosy::RasConfig::default().seed);
            let ras = xabe_cosy::RasConfig::default();
            let mut out_t = Vec::new();
            for i in 0..text.len() * 20 {
                let row = gpu
                    .copy_range(&logits, (at - 1) * vocab, vocab)
                    .expect("row");
                let mut row = gpu.download(&row).expect("download");
                if i < text.len() * 2 {
                    row[cfg.speech_token_size] = f32::NEG_INFINITY;
                }
                let id = ras_sample(&row, &out_t, &ras, &mut rng);
                if id as usize >= cfg.speech_token_size {
                    break;
                }
                out_t.push(id);
                let e = llm.speech_step(id).expect("embed");
                logits = llm.forward(&e, 1, &mut cache).expect("step");
                at = 1;
            }
            let bytes: Vec<u8> = out_t.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{out}/tokens.u32"), bytes).expect("write");
            out_t
        }
    };
    println!("{} speech tokens", tokens.len());

    // Teacher forcing in one forward.
    let n = prompt.len + tokens.len();
    let mut hs = gpu.zeros(n * h).expect("scratch");
    gpu.copy_into(&mut hs, &prompt.h, 0, prompt.len * h)
        .expect("prompt");
    for (i, &t) in tokens.iter().enumerate() {
        let e = llm.speech_step(t).expect("embed");
        gpu.copy_into(&mut hs, &e, (prompt.len + i) * h, h)
            .expect("token");
    }
    let mut cache = llm.cache();
    let logits = gpu
        .download(&llm.forward(&hs, n, &mut cache).expect("forward"))
        .expect("dl");
    let mut pre = Vec::with_capacity((tokens.len() + 1) * vocab);
    for i in 0..=tokens.len() {
        let mut row = logits[(prompt.len - 1 + i) * vocab..(prompt.len + i) * vocab].to_vec();
        log_softmax(&mut row);
        pre.extend(row);
    }
    let bytes: Vec<u8> = pre.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out}/prefill.f32"), bytes).expect("write");

    // The same positions, one token at a time through the cache.
    let mut cache = llm.cache();
    let mut logits = llm
        .forward(&prompt.h, prompt.len, &mut cache)
        .expect("forward");
    let mut at = prompt.len;
    let mut dec = Vec::with_capacity((tokens.len() + 1) * vocab);
    for i in 0..=tokens.len() {
        let row = gpu
            .copy_range(&logits, (at - 1) * vocab, vocab)
            .expect("row");
        let mut row = gpu.download(&row).expect("download");
        log_softmax(&mut row);
        dec.extend(row);
        if i < tokens.len() {
            let e = llm.speech_step(tokens[i]).expect("embed");
            logits = llm.forward(&e, 1, &mut cache).expect("step");
            at = 1;
        }
    }
    let bytes: Vec<u8> = dec.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out}/decode.f32"), bytes).expect("write");
    let _ = LlmConfig::SOS;
    println!(
        "wrote {out}/prefill.f32 and decode.f32, {} rows of {vocab}",
        tokens.len() + 1
    );
}
