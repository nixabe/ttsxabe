//! Milestone 12, correctness half: the CUDA pipeline against the oracle.
//!
//! The per-kernel tests in `xabe-cuda` establish that each kernel agrees with
//! its scalar twin. They say nothing about whether the *pipeline* wires them
//! together the way the CPU path does - a transposed layout that two adjacent
//! GPU stages agree on would pass every kernel test and produce different
//! audio. So this runs the whole thing from text and the captured noise, and
//! compares against the same captured waveform the CPU path is held to.

use std::path::PathBuf;
use xabe_golden::Golden;
use xabe_tts::GpuModel;

/// Looser than the CPU path's, and for a reason worth stating: the GPU fuses
/// multiply-add, so its arithmetic is not the CPU's rearranged but genuinely
/// different - more accurate per operation, and differently rounded. Through
/// the decoder's twelve residual blocks that difference compounds.
const ATOL: f32 = 2e-3;
const RTOL: f32 = 2e-2;

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

/// Which device to use. GPU 2 on this host runs somebody else's job.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn fixture() -> Option<(GpuModel, Golden)> {
    let snap = find_snapshot()?;
    let g = Golden::open_default()?;
    // Skip only when there is genuinely no device. Skipping on any error hides
    // defects as absent hardware - which is exactly what happened once here.
    match GpuModel::open(&snap, ordinal()) {
        Ok(m) => Some((m, g)),
        Err(xabe_tts::SynthesisError::Cuda(xabe_cuda::CudaError::NoDevice(why))) => {
            eprintln!("SKIP: no CUDA device ({why})");
            None
        }
        Err(e) => panic!("the device is present but unusable: {e}"),
    }
}

#[test]
fn the_cuda_pipeline_matches_the_reference_waveform() {
    let Some((m, g)) = fixture() else { return };

    let noise_dur = g.f32s("noise_dur").expect("noise_dur");
    let noise_prior = g.f32s("noise_prior").expect("noise_prior");

    // The prior's noise is requested by length, because the length depends on
    // the durations this pipeline predicts. Asserting the length inside the
    // callback is what proves the GPU duration predictor agreed with the
    // reference's - without it, a mismatch would surface as a confusing
    // waveform-length error.
    let audio = m
        .synthesize_with_noise(&g.manifest().text, &noise_dur, &|n| {
            assert_eq!(
                n,
                noise_prior.len(),
                "the GPU predicted different durations"
            );
            noise_prior.clone()
        })
        .expect("synthesize");

    let c = g.compare("waveform", &audio, ATOL, RTOL).expect("compare");
    assert!(c.passed(), "{c}");
    eprintln!(
        "cuda end to end: {} samples, max abs {:.3e}, mean abs {:.3e}",
        audio.len(),
        c.max_abs,
        c.mean_abs,
    );
}

#[test]
fn cuda_synthesis_from_a_seed_is_reproducible() {
    let Some((m, _)) = fixture() else { return };
    let a = m.synthesize("li ho", 7).expect("synthesize");
    let b = m.synthesize("li ho", 7).expect("synthesize");
    assert_eq!(a, b, "the same seed must give the same audio");
    assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}

#[test]
fn cuda_refuses_text_with_no_speakable_symbols() {
    let Some((m, _)) = fixture() else { return };
    let err = m.synthesize("你好", 0).expect_err("should be refused");
    assert!(err.to_string().contains("no symbols"), "{err}");
}
