//! The Coqui checkpoint on CUDA, against the same oracle the CPU path uses.
//!
//! The per-kernel tests in `xabe-cuda` establish that each kernel agrees with
//! its scalar twin, and `coqui_end_to_end.rs` establishes that the CPU pipeline
//! agrees with the reference. Neither says the *device* pipeline is wired the
//! way the CPU one is, and this checkpoint adds a new way for that to go wrong:
//! its decoder arrives weight-normalised, so the fusion happens on the card,
//! once, at upload - where the 🤗 path has nothing to fuse at all.

use std::path::{Path, PathBuf};
use xabe_golden::Golden;
use xabe_tts::GpuModel;

/// Looser than the CPU path's, for the reason `gpu_end_to_end.rs` gives: the
/// GPU fuses multiply-add, so its arithmetic is not the CPU's rearranged but
/// genuinely different, and the difference compounds through twelve residual
/// blocks.
const ATOL: f32 = 2e-3;
const RTOL: f32 = 2e-2;

fn find_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("XABE_COQUI_MODEL") {
        let p = PathBuf::from(p);
        return p.join("best_model.pth").is_file().then_some(p);
    }
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tts/coqui-vits-suisiann");
    local.join("best_model.pth").is_file().then_some(local)
}

fn find_golden() -> Option<Golden> {
    let dir = match std::env::var("XABE_COQUI_GOLDEN") {
        Ok(p) => PathBuf::from(p),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".golden/coqui-base"),
    };
    dir.join("manifest.json")
        .is_file()
        .then(|| Golden::open(&dir).ok())
        .flatten()
}

/// Which device to use. GPU 2 on this host runs somebody else's job.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn fixture() -> Option<(GpuModel, Golden)> {
    let dir = find_model()?;
    let g = find_golden()?;
    // Skip only when there is genuinely no device. Skipping on any error hides
    // defects as absent hardware - which is exactly what happened once here.
    match GpuModel::open_coqui(&dir, ordinal()) {
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

    // Asserting the length inside the callback is what proves the GPU duration
    // predictor agreed with the reference's - without it, a mismatch would
    // surface as a confusing waveform-length error.
    let audio = m
        .synthesize_with_noise(g.manifest().input(), &noise_dur, &|n| {
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
        "coqui cuda end to end: {} samples at {} Hz, max abs {:.3e}, mean abs {:.3e}",
        audio.len(),
        m.config().sampling_rate,
        c.max_abs,
        c.mean_abs,
    );
}

#[test]
fn cuda_synthesis_from_a_seed_is_reproducible() {
    let Some((m, g)) = fixture() else { return };
    let input: String = g.manifest().input().chars().take(8).collect();
    let a = m.synthesize(&input, 7).expect("synthesize");
    let b = m.synthesize(&input, 7).expect("synthesize");
    assert_eq!(a, b, "the same seed must give the same audio");
    assert!(a.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}

#[test]
fn cuda_refuses_input_with_no_speakable_symbols() {
    let Some((m, _)) = fixture() else { return };
    // Han text is what this model is *about*, and it is still not what it eats:
    // the symbol table holds IPA, so unphonemised text tokenises to nothing.
    let err = m.synthesize("你好", 0).expect_err("should be refused");
    assert!(err.to_string().contains("no symbols"), "{err}");
}
