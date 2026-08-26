//! Differential test: the prior flow, reversed, against the oracle.
//!
//! The capture holds `z_p` and `z` separately for exactly this reason. The flow
//! is four coupling blocks that trade halves between them, and a swapped half
//! or a flip on the wrong side is invisible in the waveform - the audio is
//! still audio - while being obvious here.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_st::StFile;
use xabe_tts::flow_reverse;
use xabe_vits::{VitsConfig, VitsWeights};

/// The flow is four blocks of four WaveNet layers, each a dilated convolution
/// over 192 channels, so this is the deepest accumulation so far.
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
fn the_reversed_flow_matches_the_oracle() {
    let Some((file, cfg, g)) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");

    // Fed the oracle's own `z_p`, so this isolates the flow: an error here is
    // the flow's, not something inherited from the prior expansion.
    let z_p = g.f32s("z_p").expect("read z_p");
    let z = flow_reverse(&z_p, &w.flow, &cfg);

    let c = g.compare("z", &z, ATOL, RTOL).expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!("z: max abs {:.3e}", c.max_abs);
}

#[test]
fn the_flow_actually_changes_its_input() {
    let Some((file, cfg, g)) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");
    let z_p = g.f32s("z_p").expect("read z_p");
    let z = flow_reverse(&z_p, &w.flow, &cfg);

    // Half the channels pass through each coupling block untouched, so a flow
    // that silently did nothing at all would still leave half the tensor
    // correct. Requiring a large change over the whole tensor catches the
    // degenerate "every block was a no-op" case that a tolerance would not.
    let moved = z_p
        .iter()
        .zip(&z)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    assert!(
        moved > z.len() / 2,
        "only {moved} of {} values moved; the flow may be a no-op",
        z.len(),
    );
}

#[test]
fn the_coupling_is_mean_only() {
    let Some((file, cfg, _)) = fixture() else {
        skip();
        return;
    };
    let w = VitsWeights::load(&file, &cfg).expect("bind weights");

    // `conv_post` emits `flow_size / 2` channels, not `flow_size`. A coupling
    // that also predicted a scale would emit both, and inverting it would need
    // a division this implementation does not do. Pinning the width is how a
    // future checkpoint with scaled coupling gets caught here rather than
    // producing quietly wrong audio.
    for block in &w.flow {
        assert_eq!(block.conv_post.out_ch, cfg.flow_half());
        assert_eq!(block.conv_pre.in_ch, cfg.flow_half());
    }
    assert_eq!(w.flow.len(), cfg.prior_encoder_num_flows);
}
