//! The VAD against whisper.cpp's, frame by frame.
//!
//! The captures come from `tools/oracle/capture_vad.sh`, which runs the
//! reference over a deterministic corpus and writes its probabilities and
//! segments as binary. Nothing here is hand-transcribed; if a number in a
//! comment disagrees with a capture, the capture is right.
//!
//! Four of the clips are the pipeline's known hallucination triggers. The VAD
//! is the only thing standing between those and an assistant that answers a
//! sentence Whisper invented out of silence, so agreeing with the reference on
//! them is the point of the whole exercise.

use std::path::{Path, PathBuf};
use xabe_vad::{SegmentParams, VadWeights, segments};

/// The converted checkpoint, or `None` if it has not been produced.
fn checkpoint() -> Option<PathBuf> {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/vad/silero-v5.1.2.safetensors");
    p.is_file().then_some(p)
}

/// The capture directory, or `None` if nothing has been captured.
fn golden() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/vad");
    p.join("manifest.json").is_file().then_some(p)
}

/// Reads a `.bin` of little-endian f32.
fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// One clip and everything the reference said about it.
struct Clip {
    /// Directory name, which is also the clip's name.
    name: String,
    /// The audio, mono at 16 kHz.
    samples: Vec<f32>,
    /// One reference probability per 512 samples.
    probs: Vec<f32>,
    /// Reference segment edges, in seconds, as (start, end) pairs.
    segments: Vec<(f32, f32)>,
}

/// Every captured clip.
fn corpus() -> Option<Vec<Clip>> {
    let root = golden()?;
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&root)
        .expect("read .golden/vad")
        .flatten()
    {
        let dir = entry.path();
        if !dir.is_dir() || dir.file_name().is_some_and(|n| n == "clips") {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let wav = root.join("clips").join(format!("{name}.wav"));
        if !wav.is_file() || !dir.join("probs.bin").is_file() {
            continue;
        }
        let audio = xabe_audio::read_wav(&wav).expect("read clip");
        assert_eq!(audio.sample_rate, 16_000, "{name} is not 16 kHz");
        out.push(Clip {
            name,
            samples: audio.samples,
            probs: read_f32(&dir.join("probs.bin")),
            segments: read_f32(&dir.join("segments.bin"))
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (c[0], c[1]))
                .collect(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Some(out)
}

/// The parameters `capture_vad.sh` used, which are the pipeline's own.
fn params() -> SegmentParams {
    SegmentParams {
        threshold: 0.6,
        min_speech_duration_ms: 250,
        min_silence_duration_ms: 200,
        ..SegmentParams::default()
    }
}

fn skip(what: &str) {
    eprintln!("SKIP: {what}; see docs/ORACLE.md");
}

#[test]
fn the_schema_binds_every_tensor_in_the_checkpoint() {
    let Some(path) = checkpoint() else {
        skip("no converted VAD checkpoint - run tools/vad/ggml_to_safetensors.py");
        return;
    };
    let f = xabe_st::StFile::open(&path).expect("open checkpoint");
    let w = VadWeights::load(&f).expect("bind weights");

    assert_eq!(f.len(), 15, "the checkpoint has 15 tensors");

    // A tensor the schema forgets to read does not raise an error - it simply
    // never appears - so the parameter count is the only thing that notices.
    let in_file: usize = f.tensors().map(|(_, i)| i.numel()).sum();
    assert_eq!(
        w.total_elements(),
        in_file,
        "the schema reads every parameter in the file",
    );
}

#[test]
fn the_checkpoint_is_f16_and_is_widened_on_read() {
    let Some(path) = checkpoint() else {
        skip("no converted VAD checkpoint");
        return;
    };
    let f = xabe_st::StFile::open(&path).expect("open checkpoint");

    // Silero's convolutions are stored half precision. This was not in the
    // plan - it was found by converting the file - and it is why xabe-st reads
    // more than F32.
    let stft = f
        .info("_model.stft.forward_basis_buffer")
        .expect("the stft basis");
    assert_eq!(stft.dtype, xabe_st::Dtype::F16);

    // Borrowing it as f32 must be refused by name rather than reinterpreting
    // pairs of halves as single floats, which would produce plausible noise.
    let err = f.tensor("_model.stft.forward_basis_buffer").unwrap_err();
    assert!(err.to_string().contains("F16"), "{err}");

    // The biases really are F32, so the file is genuinely mixed.
    let bias = f
        .info("_model.encoder.0.reparam_conv.bias")
        .expect("a bias");
    assert_eq!(bias.dtype, xabe_st::Dtype::F32);
}

#[test]
fn per_frame_probabilities_match_the_reference() {
    let (Some(path), Some(clips)) = (checkpoint(), corpus()) else {
        skip("no checkpoint or no captures - run tools/oracle/capture_vad.sh");
        return;
    };
    assert!(!clips.is_empty(), "the capture directory holds no clips");

    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for clip in &clips {
        let (name, reference) = (&clip.name, &clip.probs);
        let mut vad = xabe_vad::open(&path).expect("load vad");
        // Reset is implicit in a fresh detector, and the reference resets too:
        // whisper_vad_detect_speech clears the LSTM state before it starts.
        let got = vad.probabilities(&clip.samples);

        assert_eq!(
            got.len(),
            reference.len(),
            "{name}: {} probabilities against the reference's {}",
            got.len(),
            reference.len(),
        );
        for (i, (g, r)) in got.iter().zip(reference).enumerate() {
            let d = (g - r).abs();
            if d > worst {
                worst = d;
                worst_at = format!("{name} frame {i}: {g} against {r}");
            }
        }
    }

    // Measured worst case across the corpus is 6.8e-3 on a quantity bounded in
    // [0, 1], and that number has an explanation rather than a tolerance.
    //
    // whisper.cpp stores this checkpoint's convolutions as F16 and runs ggml's
    // F16 kernels, which round the *activations* to half precision at the input
    // of every convolution. This implementation widens the weights once at load
    // and keeps activations in f32 throughout. Rounding this implementation's
    // conv inputs through f16 as an experiment drops the worst disagreement
    // from 6.8e-3 to 1.8e-4, a 37x reduction, which locates the whole of the
    // difference in that one choice.
    //
    // F32 is kept: it is the more accurate of the two, it is what upstream
    // Silero computes in, and the experiment shows the reference's extra error
    // is ggml's storage format rather than anything about the model. What is
    // asserted below instead is the property the pipeline actually depends on -
    // that no frame lands on a different side of a threshold - which
    // `segments_match_the_reference_on_every_clip` then confirms end to end.
    assert!(
        worst < 1e-2,
        "worst probability disagreement {worst:.3e} at {worst_at}",
    );
    eprintln!("worst probability disagreement across the corpus: {worst:.3e}");
}

#[test]
fn no_frame_lands_on_a_different_side_of_a_threshold() {
    let (Some(path), Some(clips)) = (checkpoint(), corpus()) else {
        skip("no checkpoint or no captures");
        return;
    };

    // This, not the raw disagreement, is what the segmenter consumes. A
    // probability that differs by 6.8e-3 changes nothing unless it differs
    // *across* a threshold, and the two thresholds are the only two numbers the
    // hysteresis ever compares against.
    let p = params();
    let neg = (p.threshold - 0.15).max(0.01);

    let mut checked = 0usize;
    for clip in &clips {
        let (name, reference) = (&clip.name, &clip.probs);
        let mut vad = xabe_vad::open(&path).expect("load vad");
        let got = vad.probabilities(&clip.samples);
        for (i, (g, r)) in got.iter().zip(reference).enumerate() {
            for (label, t) in [("threshold", p.threshold), ("neg_threshold", neg)] {
                assert_eq!(
                    *g >= t,
                    *r >= t,
                    "{name} frame {i} straddles {label} {t}: {g} against {r}",
                );
            }
            checked += 1;
        }
    }
    eprintln!("{checked} frames agree on both thresholds");
}

#[test]
fn the_four_hallucination_triggers_produce_no_speech() {
    let (Some(path), Some(clips)) = (checkpoint(), corpus()) else {
        skip("no checkpoint or no captures");
        return;
    };

    // Each of these is a case where the ASR invented a sentence and the
    // assistant answered it: digital silence as 我…, faint hiss as 我現在在醫院,
    // room noise as (我會陪你一起走), and a transient opening a turn on its own.
    for trigger in ["silence", "hiss", "room", "click"] {
        let clip = clips
            .iter()
            .find(|c| c.name == trigger)
            .unwrap_or_else(|| panic!("the corpus has no {trigger} clip"));

        let mut vad = xabe_vad::open(&path).expect("load vad");
        let probs = vad.probabilities(&clip.samples);
        let got = segments(&probs, params());

        assert!(
            clip.segments.is_empty(),
            "the reference itself found speech in {trigger}; the corpus is wrong",
        );
        assert!(
            got.is_empty(),
            "{trigger} produced {} segments; the pipeline would hallucinate here",
            got.len(),
        );
        let peak = probs.iter().copied().fold(0.0f32, f32::max);
        let ref_peak = clip.probs.iter().copied().fold(0.0f32, f32::max);
        eprintln!("{trigger}: peak probability {peak:.4} (reference {ref_peak:.4})");
    }
}

#[test]
fn segments_match_the_reference_on_every_clip() {
    let (Some(path), Some(clips)) = (checkpoint(), corpus()) else {
        skip("no checkpoint or no captures");
        return;
    };

    for clip in &clips {
        let name = &clip.name;
        let mut vad = xabe_vad::open(&path).expect("load vad");
        let probs = vad.probabilities(&clip.samples);
        let got = segments(&probs, params());

        let want = &clip.segments;
        assert_eq!(
            got.len(),
            want.len(),
            "{name}: {} segments against the reference's {}: {got:?} vs {want:?}",
            got.len(),
            want.len(),
        );

        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            // The reference reports centiseconds, so its edges are quantised to
            // 10 ms and cannot be compared more finely than that.
            assert!(
                (g.start_s() as f32 - w.0).abs() <= 0.01,
                "{name} segment {i}: start {:.3} against {:.3}",
                g.start_s(),
                w.0,
            );
            assert!(
                (g.end_s() as f32 - w.1).abs() <= 0.01,
                "{name} segment {i}: end {:.3} against {:.3}",
                g.end_s(),
                w.1,
            );
        }
        if !want.is_empty() {
            eprintln!("{name}: {} segments, matching", want.len());
        }
    }
}

#[test]
fn the_recurrent_state_carries_across_frames_and_reset_clears_it() {
    let Some(path) = checkpoint() else {
        skip("no converted VAD checkpoint");
        return;
    };
    // This is the property that makes the VAD a detector rather than a
    // classifier, and the one that makes carrying state across two independent
    // clips a silent correctness bug rather than a performance detail.
    let mut vad = xabe_vad::open(&path).expect("load vad");
    let noise: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();

    let first = vad.probabilities(&noise);
    let carried = vad.probabilities(&noise);
    vad.reset();
    let after_reset = vad.probabilities(&noise);

    assert_eq!(
        first, after_reset,
        "reset must return the detector to its start"
    );
    assert_ne!(
        first, carried,
        "state must carry across calls, or the LSTM is doing nothing",
    );
}
