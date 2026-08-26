//! Differential test: the text encoder against the captured PyTorch oracle.
//!
//! Stage by stage, not just at the output. The capture holds the scaled
//! embedding and all six layer outputs precisely so that a failure says which
//! layer, and the assertions below walk them in order and stop at the first
//! one that disagrees - a later stage's error is a consequence of an earlier
//! one, and reporting both would bury the cause.
//!
//! Skips when the checkpoint or the capture is absent.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_st::StFile;
use xabe_tts::text_encoder;
use xabe_vits::{Tokenizer, VitsConfig, VitsWeights};

/// Absolute tolerance. The encoder is six layers of f32 accumulation in a
/// different order than PyTorch's BLAS, so the last few bits differ; nothing
/// here is expected to match exactly. The observed maximum is 3.8e-6 at
/// `enc_out`, so this leaves an order of magnitude of headroom for a different
/// BLAS without leaving room for an actual mistake.
const ATOL: f32 = 5e-5;

/// Relative tolerance, which is what actually judges the large values.
const RTOL: f32 = 5e-5;

fn find_snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("XABE_TTS_MODEL") {
        return PathBuf::from(p).parent().map(Into::into);
    }
    let home = std::env::var("HOME").ok()?;
    let root = std::path::Path::new(&home)
        .join(".cache/huggingface/hub/models--facebook--mms-tts-nan/snapshots");
    let snap = std::fs::read_dir(root).ok()?.flatten().next()?.path();
    snap.join("model.safetensors").is_file().then_some(snap)
}

struct Fixture {
    file: StFile,
    cfg: VitsConfig,
    tok: Tokenizer,
    golden: Golden,
}

fn fixture() -> Option<Fixture> {
    let snap = find_snapshot()?;
    let golden = Golden::open_default()?;
    Some(Fixture {
        file: StFile::open(snap.join("model.safetensors")).expect("open checkpoint"),
        cfg: VitsConfig::from_json_path(snap.join("config.json")).expect("read config"),
        tok: Tokenizer::load(&snap).expect("load tokenizer"),
        golden,
    })
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/base; see docs/ORACLE.md");
}

#[test]
fn every_stage_of_the_text_encoder_matches_the_oracle() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");
    let ids = f.tok.encode(&f.golden.manifest().text);
    let out = text_encoder(&ids, &w.text_encoder, &f.cfg);

    // In order, and stop at the first disagreement: a later stage's error is
    // downstream of an earlier one, so reporting both hides the cause.
    let mut stages: Vec<(String, &Vec<f32>)> = vec![("embed".to_string(), &out.embed)];
    for (i, h) in out.layers.iter().enumerate() {
        stages.push((format!("enc_layer_{i}"), h));
    }
    stages.push(("enc_out".to_string(), &out.hidden));
    stages.push(("m_p".to_string(), &out.m_p));
    stages.push(("logs_p".to_string(), &out.logs_p));

    for (name, computed) in stages {
        let c = f
            .golden
            .compare(&name, computed, ATOL, RTOL)
            .unwrap_or_else(|e| panic!("comparing {name}: {e}"));
        assert!(c.passed(), "{c}");
        eprintln!("{name}: max abs {:.3e}", c.max_abs);
    }
}

#[test]
fn the_shapes_are_what_the_prior_expects() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");
    let ids = f.tok.encode(&f.golden.manifest().text);
    let out = text_encoder(&ids, &w.text_encoder, &f.cfg);

    assert_eq!(out.t, ids.len());
    assert_eq!(out.layers.len(), f.cfg.num_hidden_layers);
    assert_eq!(out.hidden.len(), ids.len() * f.cfg.hidden_size);
    // The projection produces `2 * flow_size` channels and the split must take
    // the first half as the mean - reversing them is a shape-preserving error
    // that turns the prior inside out.
    assert_eq!(out.m_p.len(), ids.len() * f.cfg.flow_size);
    assert_eq!(out.logs_p.len(), ids.len() * f.cfg.flow_size);
}

#[test]
fn the_embedding_scaling_is_present() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");
    let ids = f.tok.encode(&f.golden.manifest().text);
    let out = text_encoder(&ids, &w.text_encoder, &f.cfg);

    // Dropping the sqrt(hidden_size) crashes nothing and is partly absorbed by
    // the layer norms, so it is worth asserting against the raw table directly
    // rather than trusting the stage comparison to notice.
    let raw = f.golden.f32s("embed_raw").expect("read embed_raw");
    let scale = (f.cfg.hidden_size as f32).sqrt();
    let expected: Vec<f32> = raw.iter().map(|v| v * scale).collect();
    let c = xabe_golden::Comparison::new("embed", &expected, &out.embed, ATOL, RTOL);
    assert!(c.passed(), "{c}");
}
