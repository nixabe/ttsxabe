//! The Coqui checkpoint, stage by stage, against its own captured oracle.
//!
//! Same discipline as `end_to_end.rs` and for the same reason: this
//! architecture fails quietly, and a waveform that sounds like speech is not
//! evidence of anything in a language the author does not speak. What is new
//! here is the *loader* - a torch container, a different naming scheme, and a
//! decoder whose weight norm has not been fused - so the stages below are
//! ordered to say which of those went wrong when one of them does.
//!
//! # The input is phonemes
//!
//! The manifest records the IPA string the reference's phonemiser produced, and
//! that is what is fed back in here. Reading `text` instead would tokenise Han
//! characters to nothing and compare an empty utterance against a real one.
//!
//! Capture with:
//!
//! ```text
//! .venv-coqui/bin/python tools/oracle/capture_coqui.py \
//!     --out .golden/coqui-base --seed 0 --text "..."
//! ```

use std::path::{Path, PathBuf};
use xabe_golden::Golden;
use xabe_tts::Synthesizer;

/// End of the chain, so the error is everything upstream accumulated. The same
/// thresholds the 🤗 path is held to.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

/// Locates the Coqui model directory.
fn find_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("XABE_COQUI_MODEL") {
        let p = PathBuf::from(p);
        return p.join("best_model.pth").is_file().then_some(p);
    }
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tts/coqui-vits-suisiann");
    local.join("best_model.pth").is_file().then_some(local)
}

/// Locates the Coqui capture, which is a different directory from the 🤗 one.
fn find_golden() -> Option<Golden> {
    let dir = match std::env::var("XABE_COQUI_GOLDEN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".golden/coqui-base"),
    };
    if !dir.join("manifest.json").is_file() {
        return None;
    }
    Golden::open(&dir).ok()
}

fn fixture() -> Option<(Synthesizer, Golden)> {
    let dir = find_model()?;
    let g = find_golden()?;
    Some((Synthesizer::open_coqui(&dir).expect("open model"), g))
}

fn skip() {
    eprintln!("SKIP: need coqui-vits-suisiann and .golden/coqui-base; see docs/ORACLE.md");
}

#[test]
fn the_capture_is_a_coqui_one() {
    let Some((_, g)) = fixture() else {
        skip();
        return;
    };
    let m = g.manifest();
    assert_eq!(m.dialect.as_deref(), Some("coqui"));
    assert_eq!(m.sampling_rate, 22_050);
    assert!(
        m.phonemes.is_some(),
        "a Coqui capture must record what the phonemiser produced",
    );
    assert_ne!(
        m.input(),
        m.text,
        "the model is fed phonemes, not the Han text",
    );
}

#[test]
fn tokenisation_matches_the_reference() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };
    let ids = s.tokenizer().encode(g.manifest().input());
    let want = g.i64s("input_ids").expect("input_ids");
    assert_eq!(ids, want, "symbol ids differ from the reference's");
    // Blank, symbol, blank, ... blank: an odd length, and both ends the blank.
    assert!(ids.len() % 2 == 1);
    assert_eq!(ids[0], 3, "the blank is at id 3 in this vocabulary");
}

#[test]
fn the_text_encoder_matches_the_reference() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };
    let p = s.prepare(g.manifest().input()).expect("prepare");

    // In order, and stop at the first disagreement: a later stage's error is
    // downstream of an earlier one, so reporting both hides the cause. The
    // per-layer entries turn "the text encoder is wrong" into "layer 3 is
    // wrong" for free, and here that matters more than on the 🤗 path - a
    // mis-bound tensor from a different naming scheme shows up in exactly one
    // layer.
    let mut stages: Vec<(String, &Vec<f32>)> = vec![("embed".to_string(), &p.encoded.embed)];
    for (i, h) in p.encoded.layers.iter().enumerate() {
        stages.push((format!("enc_layer_{i}"), h));
    }
    stages.push(("enc_out".to_string(), &p.encoded.hidden));
    stages.push(("m_p".to_string(), &p.encoded.m_p));
    stages.push(("logs_p".to_string(), &p.encoded.logs_p));

    for (name, computed) in stages {
        let c = g
            .compare(&name, computed, ATOL, RTOL)
            .unwrap_or_else(|e| panic!("comparing {name}: {e}"));
        assert!(c.passed(), "{c}");
        eprintln!("{name}: max abs {:.3e}", c.max_abs);
    }
}

#[test]
fn the_duration_predictor_matches_the_reference() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };
    let p = s.prepare(g.manifest().input()).expect("prepare");
    let noise = g.f32s("noise_dur").expect("noise_dur");
    let prosody = s.durations(&p, &noise).expect("durations");

    let c = g
        .compare("log_duration", &prosody.log_duration, ATOL, RTOL)
        .expect("compare");
    assert!(c.passed(), "{c}");

    // The frame count is where a duration error becomes a length error, and a
    // length error would make every later comparison fail for the wrong reason.
    let frames = g.shape("z_p").expect("z_p")[2];
    assert_eq!(prosody.frames, frames, "expanded to the wrong length");
}

#[test]
fn text_in_waveform_out_matches_the_reference() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };

    let prepared = s.prepare(g.manifest().input()).expect("prepare");
    let noise_dur = g.f32s("noise_dur").expect("noise_dur");
    let prosody = s.durations(&prepared, &noise_dur).expect("durations");

    // The staged API's contract, checked against the capture: the frame count
    // the durations imply must be the one the reference's second draw was
    // shaped for.
    let noise_prior = g.f32s("noise_prior").expect("noise_prior");
    assert_eq!(prosody.prior_noise_len(s.config()), noise_prior.len());

    let audio = s.render(&prepared, &prosody, &noise_prior).expect("render");

    let c = g.compare("waveform", &audio, ATOL, RTOL).expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!(
        "coqui end to end: {} samples at {} Hz, max abs {:.3e}, mean abs {:.3e}",
        audio.len(),
        s.config().sampling_rate,
        c.max_abs,
        c.mean_abs,
    );
}

#[test]
fn synthesis_from_a_seed_is_reproducible() {
    let Some((s, g)) = fixture() else {
        skip();
        return;
    };
    // The seeded path does not reproduce PyTorch - two RNGs agreeing across
    // languages is not something to assume - so all it can promise is that it
    // reproduces itself.
    //
    // Deliberately a short input: this runs the vocoder three times and the CPU
    // path is not fast, while what is being checked - that the same seed gives
    // the same samples - does not need a long utterance to be true or false.
    let full = g.manifest().input();
    let input: String = full.chars().take(8).collect();
    let input = input.as_str();
    let a = s.synthesize(input, 7).expect("synthesize");
    let b = s.synthesize(input, 7).expect("synthesize");
    assert_eq!(a, b);
    let c = s.synthesize(input, 8).expect("synthesize");
    assert_ne!(a, c, "a different seed must give different audio");
}
