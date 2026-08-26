//! Differential test: the HiFi-GAN decoder against the captured waveform.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_st::StFile;
use xabe_tts::decoder;
use xabe_vits::{VitsConfig, VitsWeights};

/// The decoder is the deepest stack here - four transposed convolutions and
/// twelve residual blocks - and it ends in `tanh`, which compresses error near
/// the rails. Judged mostly by the relative term.
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

fn fixture() -> Option<(StFile, VitsConfig, Golden)> {
    let snap = find_snapshot()?;
    Some((
        StFile::open(snap.join("model.safetensors")).expect("open checkpoint"),
        VitsConfig::from_json_path(snap.join("config.json")).expect("read config"),
        Golden::open_default()?,
    ))
}

fn skip() {
    eprintln!("SKIP: need mms-tts-nan and .golden/base; see docs/ORACLE.md");
}

#[test]
fn the_decoder_matches_the_oracle_waveform() {
    let Some((file, cfg, g)) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");

    // Fed the oracle's own `z`, so this isolates the decoder from everything
    // upstream of it.
    let z = g.f32s("z").expect("read z");
    let audio = decoder(&z, &w.decoder, &cfg);

    assert_eq!(audio.len(), g.shape("waveform").unwrap()[1]);
    let c = g.compare("waveform", &audio, ATOL, RTOL).expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!("waveform: max abs {:.3e}", c.max_abs);
}

#[test]
fn the_upsample_stages_multiply_to_the_hop_length() {
    let Some((_, cfg, g)) = fixture() else {
        skip();
        return;
    };
    // The frame-to-sample ratio is the product of the strides. Getting one
    // transposed convolution's padding wrong changes the length by a few
    // samples, which is a mismatch the comparison would report as a length
    // error rather than as the padding bug it is - so it is asserted here,
    // where the message says what it means.
    let frames = g.shape("z").unwrap()[2];
    let samples = g.shape("waveform").unwrap()[1];
    assert_eq!(samples, frames * cfg.hop_length());
    assert_eq!(
        cfg.upsample_rates.iter().product::<usize>(),
        cfg.hop_length(),
    );
}

#[test]
fn the_output_is_bounded_and_not_clipped() {
    let Some((file, cfg, g)) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");
    let z = g.f32s("z").expect("read z");
    let audio = decoder(&z, &w.decoder, &cfg);

    // `tanh` bounds the output whatever happens upstream, so being in range
    // proves nothing on its own. Being *comfortably* inside it does: forgetting
    // to divide the fused resblocks by three makes the signal three times too
    // loud, and the symptom is a waveform pinned near the rails rather than an
    // out-of-range value.
    let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak < 1.0, "peak {peak} is at the tanh rail");
    let pinned = audio.iter().filter(|v| v.abs() > 0.99).count();
    assert!(
        pinned * 100 < audio.len(),
        "{pinned} of {} samples are pinned near +/-1; the fusion may not be averaged",
        audio.len(),
    );
}
