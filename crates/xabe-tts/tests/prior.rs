//! Differential test: length regulation against the captured alignment.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_st::StFile;
use xabe_tts::{duration_predictor, expand_prior, text_encoder};
use xabe_vits::{Tokenizer, VitsConfig, VitsWeights};

const ATOL: f32 = 2e-4;
const RTOL: f32 = 2e-4;

fn find_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("XABE_TTS_MODEL") {
        return PathBuf::from(p).parent().map(Into::into);
    }
    // The consolidated model tree is the canonical home. The HuggingFace cache
    // is kept as a fallback so a checkout that never ran the move still tests.
    let local =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tts/mms-tts-nan");
    if local.join("model.safetensors").is_file() {
        return Some(local);
    }
    let home = std::env::var("HOME").ok()?;
    let root = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--facebook--mms-tts-nan/snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    snap.join("model.safetensors").is_file().then_some(snap)
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/base; see docs/ORACLE.md");
}

#[test]
fn the_expanded_prior_matches_the_oracle() {
    let Some(snap) = find_snapshot() else {
        skip();
        return;
    };
    let Some(g) = Golden::open_default() else {
        skip();
        return;
    };
    let file = StFile::open(snap.join("model.safetensors")).expect("open checkpoint");
    let cfg = VitsConfig::from_json_path(snap.join("config.json")).expect("read config");
    let tok = Tokenizer::load(&snap).expect("load tokenizer");
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");

    let ids = tok.encode(&g.manifest().text);
    let enc = text_encoder(&ids, &w.text_encoder, &cfg);
    let noise_dur = g.f32s("noise_dur").expect("noise_dur");
    let log_d = duration_predictor(&enc.hidden, &noise_dur, &w.duration_predictor, &cfg);
    let noise_prior = g.f32s("noise_prior").expect("noise_prior");

    let prior = expand_prior(&enc.m_p, &enc.logs_p, &log_d, &noise_prior, &cfg);

    // The alignment first: if the frame count or the symbol boundaries differ,
    // every value below is being compared at the wrong position and the
    // tolerance failure would say nothing useful.
    assert_eq!(prior.frames, g.shape("z_p").unwrap()[2]);
    let attn = prior.attention_matrix(ids.len());
    let c = g.compare("attn", &attn, 0.0, 0.0).expect("compare attn");
    assert!(c.passed(), "{c}");

    let c = g
        .compare("z_p", &prior.z_p, ATOL, RTOL)
        .expect("compare z_p");
    assert!(c.passed(), "{c}");
    eprintln!("z_p: max abs {:.3e}", c.max_abs);
}

#[test]
fn every_symbol_gets_at_least_one_frame() {
    let Some(snap) = find_snapshot() else {
        skip();
        return;
    };
    let Some(g) = Golden::open_default() else {
        skip();
        return;
    };
    let file = StFile::open(snap.join("model.safetensors")).expect("open checkpoint");
    let cfg = VitsConfig::from_json_path(snap.join("config.json")).expect("read config");
    let tok = Tokenizer::load(&snap).expect("load tokenizer");
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");

    let ids = tok.encode(&g.manifest().text);
    let enc = text_encoder(&ids, &w.text_encoder, &cfg);
    let noise_dur = g.f32s("noise_dur").expect("noise_dur");
    let log_d = duration_predictor(&enc.hidden, &noise_dur, &w.duration_predictor, &cfg);
    let noise_prior = g.f32s("noise_prior").expect("noise_prior");
    let prior = expand_prior(&enc.m_p, &enc.logs_p, &log_d, &noise_prior, &cfg);

    // `ceil` is what guarantees this. Truncating instead would delete the
    // shortest symbols outright - a mispronunciation, not a distortion, and one
    // that a waveform comparison would report as a length mismatch rather than
    // as the missing sound it is.
    let mut seen = vec![false; ids.len()];
    for &s in &prior.alignment {
        seen[s] = true;
    }
    assert!(seen.iter().all(|&v| v), "a symbol was allocated no frames");

    // Monotone and contiguous: frame f reads a symbol no earlier than frame
    // f-1's, and advances by at most one.
    for pair in prior.alignment.windows(2) {
        assert!(pair[1] == pair[0] || pair[1] == pair[0] + 1, "{pair:?}");
    }
}
