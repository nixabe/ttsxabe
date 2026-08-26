//! Differential test: the stochastic duration predictor on the captured noise.
//!
//! "On fixed noise" is the whole point. The predictor is a sampler, so it has
//! no output to compare against unless the draw is pinned - and pinning a
//! *seed* would only work if Rust and PyTorch generated identical normals from
//! it, which is not something to assume. The oracle captured the draw itself,
//! so this feeds the reference's exact `noise_dur` in and expects the
//! reference's exact `log_duration` out.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_st::StFile;
use xabe_tts::{duration_predictor, text_encoder};
use xabe_vits::{Tokenizer, VitsConfig, VitsWeights};

/// The observed maximum is 1.0e-5. The headroom is wider than the text
/// encoder's because the spline's bin search is a comparison chain: a value
/// sitting almost exactly on a knot can land one bin either side of the
/// reference, and the resulting jump is not small. Nothing observed does that
/// here, but a different sentence might.
const ATOL: f32 = 2e-4;
const RTOL: f32 = 2e-4;

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
    Some(Fixture {
        file: StFile::open(snap.join("model.safetensors")).expect("open checkpoint"),
        cfg: VitsConfig::from_json_path(snap.join("config.json")).expect("read config"),
        tok: Tokenizer::load(&snap).expect("load tokenizer"),
        golden: Golden::open_default()?,
    })
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/base; see docs/ORACLE.md");
}

/// Runs the encoder and the predictor, returning the log durations.
fn predict(f: &Fixture, w: &VitsWeights<'_>) -> Vec<f32> {
    let ids = f.tok.encode(&f.golden.manifest().text);
    let enc = text_encoder(&ids, &w.text_encoder, &f.cfg);
    let noise = f.golden.f32s("noise_dur").expect("read noise_dur");
    duration_predictor(&enc.hidden, &noise, &w.duration_predictor, &f.cfg)
}

#[test]
fn the_log_durations_match_the_oracle() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");
    let got = predict(&f, &w);

    let c = f
        .golden
        .compare("log_duration", &got, ATOL, RTOL)
        .expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!("log_duration: max abs {:.3e}", c.max_abs);
}

#[test]
fn the_durations_are_positive_and_sum_to_the_captured_frame_count() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");
    let log_d = predict(&f, &w);

    // The reference turns log durations into frames with
    // `ceil(exp(log_d) / speaking_rate)`. Reproducing that here and checking it
    // against the frame count the capture actually used is a stronger claim
    // than the tolerance comparison: the durations are *rounded up*, so a
    // near-miss on any symbol changes the total by a whole frame and shows up
    // as a mismatch rather than as a small error.
    let frames: f32 = log_d
        .iter()
        .map(|v| (v.exp() / f.cfg.speaking_rate).ceil())
        .sum();
    let expected = f.golden.shape("z_p").unwrap()[2];
    assert_eq!(
        frames as usize, expected,
        "durations expand to {frames} frames, the oracle used {expected}",
    );
    assert!(log_d.iter().all(|v| v.is_finite()));
}

#[test]
fn dropping_the_skipped_flow_matters() {
    let Some(f) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&f.file, &f.cfg).expect("bind weights");

    // The reference's reverse pass runs four of its five flow blocks, skipping
    // `flows[1]`. That is bizarre enough to look like a transcription error, so
    // this asserts the shape of the list it actually uses: five blocks are
    // stored, and one affine plus `num_flows` convolutional ones were bound.
    // If a future change "fixes" the omission, `the_log_durations_match_the_
    // oracle` fails - this test exists to say the omission was deliberate.
    assert_eq!(
        w.duration_predictor.flows.len(),
        f.cfg.duration_predictor_num_flows + 1,
        "one elementwise-affine block plus the convolutional flows",
    );
}
