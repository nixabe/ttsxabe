//! Synthesise Taigi, then transcribe it back.
//!
//! This is the test that caught the POJ/Tâi-lô error originally, and it is the
//! only objective check on the synthesiser this project has: the author cannot
//! evaluate Taigi by ear, and a differential test against PyTorch proves the
//! arithmetic matches without saying anything about whether the *text* going
//! in was in the right orthography. Passing a stage-by-stage diff while
//! producing intelligible nonsense is a real failure mode, and it happened.
//!
//! Both models are this engine's own now, which makes the test cheaper than it
//! was and slightly weaker: two stages that agree with each other could in
//! principle be wrong together. They cannot be wrong *quietly* together,
//! because each also matches its own captured reference - so this asks the
//! remaining question, which is whether the two compose into something a
//! listener would understand.

use std::path::{Path, PathBuf};

/// The TTS checkpoint, or `None` if it is not on this machine.
fn tts_model() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tts/mms-tts-nan");
    p.join("model.safetensors").is_file().then_some(p)
}

/// The ASR checkpoint, or `None` if it is not on this machine.
fn asr_model() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/asr/breeze-asr-26");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

/// Which device to use. See `docs/TESTING.md`; check `nvidia-smi` first.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

#[test]
fn poj_synthesised_here_is_transcribed_back_with_its_meaning_intact() {
    let (Some(tts_dir), Some(asr_dir)) = (tts_model(), asr_model()) else {
        println!("SKIP: models/tts/mms-tts-nan or models/asr/breeze-asr-26 is missing");
        return;
    };
    let tts = match xabe_tts::GpuModel::open(&tts_dir, ordinal()) {
        Ok(m) => m,
        Err(e) if e.to_string().contains("no usable CUDA device") => {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        Err(e) => panic!("the TTS checkpoint is present but unusable: {e}"),
    };
    let asr = xabe_asr::AsrModel::open(&asr_dir, ordinal()).expect("open the ASR");

    // POJ, not Tâi-lô: `chin` and not `tsin`, `ji̍t` and not `ji̍t`'s Tâi-lô
    // spelling. The vocabulary of `mms-tts-nan` has `c` and U+0358 in its 48
    // symbols, which is what settles it. See docs/MODEL.md.
    let cases: &[(&str, &[&str])] = &[
        // (POJ in, substrings the transcript must contain)
        ("lí hó, kin-á-ji̍t thinn-khì chin hó.", &["今天", "好"]),
        ("góa beh khì chhī-tiûⁿ bé mi̍h-kiāⁿ.", &["市場"]),
    ];

    for &(poj, wanted) in cases {
        // A fixed seed, so a failure is reproducible: the duration predictor
        // samples, and an unseeded run would make this test flaky in a way
        // that looks like an ASR problem.
        let audio = tts.synthesize(poj, 20_250_827).expect("synthesise");
        assert_eq!(tts.config().sampling_rate, 16_000, "the ASR wants 16 kHz");

        let text = asr.transcribe(&audio, "zh").expect("transcribe");
        println!("  {poj:?}\n    -> {text:?}");
        for want in wanted {
            assert!(
                text.contains(want),
                "{poj:?} came back as {text:?}, which does not contain {want:?}",
            );
        }
    }
}
