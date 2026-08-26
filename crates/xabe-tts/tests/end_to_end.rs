//! Milestone 9: the whole pipeline, text in and waveform out, against the
//! oracle.
//!
//! Every earlier differential test fed a stage the oracle's own input, which
//! isolates that stage but also hides anything that goes wrong *between*
//! stages: a layout that two adjacent stages agree on and the reference does
//! not, a transpose that cancels itself. This one starts from the manifest's
//! text and the two captured draws, and nothing else.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_tts::Synthesizer;

/// End of the chain, so the error is everything upstream accumulated. The
/// observed maximum is far below this.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

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

fn fixture() -> Option<(Synthesizer, Golden)> {
    let snap = find_snapshot()?;
    let g = Golden::open_default()?;
    Some((Synthesizer::open(&snap).expect("open model"), g))
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/base; see docs/ORACLE.md");
}

#[test]
fn text_in_waveform_out_matches_the_reference() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };

    let prepared = s.prepare(&g.manifest().text).expect("prepare");
    let noise_dur = g.f32s("noise_dur").expect("noise_dur");
    let prosody = s.durations(&prepared, &noise_dur).expect("durations");

    // The staged API's contract, checked against the capture: the frame count
    // the durations imply must be the one the reference's second draw was
    // shaped for. If these disagree the pipeline has already diverged, and the
    // waveform comparison below would be comparing different lengths.
    assert_eq!(
        prosody.prior_noise_len(s.config()),
        g.f32s("noise_prior").expect("noise_prior").len(),
    );

    let noise_prior = g.f32s("noise_prior").expect("noise_prior");
    let audio = s.render(&prepared, &prosody, &noise_prior).expect("render");

    let c = g.compare("waveform", &audio, ATOL, RTOL).expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!(
        "end to end: {} samples, max abs {:.3e}, mean abs {:.3e}",
        audio.len(),
        c.max_abs,
        c.mean_abs,
    );
}

#[test]
fn synthesis_from_a_seed_is_reproducible() {
    let Some((s, _)) = fixture() else {
        skip();
        return;
    };

    // The seeded path does not reproduce PyTorch - two RNGs agreeing across
    // languages is not something to assume - so all it can promise is that it
    // reproduces itself. That promise is the one users depend on, so it is
    // worth a test.
    let a = s.synthesize("li ho", 7).expect("synthesize");
    let b = s.synthesize("li ho", 7).expect("synthesize");
    assert_eq!(a, b, "the same seed must give the same audio");

    let c = s.synthesize("li ho", 8).expect("synthesize");
    assert_ne!(a, c, "a different seed must give different audio");
    assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}

#[test]
fn text_with_no_speakable_symbols_is_an_error() {
    let Some((s, _)) = fixture() else {
        skip();
        return;
    };

    // Han text and bare punctuation both tokenise to nothing, and synthesising
    // would return a zero-length waveform - which a caller would reasonably
    // treat as success and play as silence. Naming it is the whole point.
    for text in ["", "   ", "你好", "..."] {
        let err = s.synthesize(text, 0).expect_err("should be refused");
        assert!(
            err.to_string().contains("no symbols"),
            "{text:?} gave: {err}",
        );
    }
}
