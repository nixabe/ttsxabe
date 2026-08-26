//! Reads a real capture produced by `tools/oracle/capture.py`.
//!
//! These tests are about the capture *format* and the capture's internal
//! consistency, not about any Rust kernel - none exist yet. What they establish
//! is that the oracle is addressable from Rust and that it holds what the
//! later milestones will need, so that a stage failing later means the stage is
//! wrong and not that the golden file was never what it claimed to be.
//!
//! Skipped, not failed, when no capture is present: `.golden/` is gitignored
//! and a fresh checkout will not have one. Regenerate with the command in
//! `docs/ORACLE.md`.

use xabe_golden::{Comparison, Golden, GoldenError};

/// Frames to samples. The decoder's upsample rates multiply to this.
const HOP_LENGTH: usize = 256;

/// The flow's channel count.
const FLOW_SIZE: usize = 192;

fn capture() -> Option<Golden> {
    let g = Golden::open_default();
    if g.is_none() {
        eprintln!("SKIP: no capture; see docs/ORACLE.md to regenerate");
    }
    g
}

#[test]
fn every_stage_the_milestones_need_is_present() {
    let Some(g) = capture() else { return };

    // One entry per stage of the inference path. A missing name here means a
    // milestone downstream has nothing to diff against, which is worth failing
    // over now rather than discovering four milestones later.
    let required = [
        "input_ids",
        "embed_raw",
        "embed",
        "enc_out",
        "m_p",
        "logs_p",
        "log_duration",
        "noise_dur",
        "noise_prior",
        "attn",
        "z_p",
        "z",
        "waveform",
    ];
    let present = g.stages();
    let missing: Vec<_> = required
        .iter()
        .filter(|n| !present.contains(n))
        .copied()
        .collect();
    assert!(missing.is_empty(), "capture is missing {missing:?}");
}

#[test]
fn the_shapes_agree_with_each_other() {
    let Some(g) = capture() else { return };

    // Every shape in the capture is a function of two numbers: the symbol count
    // and the frame count. Deriving both from one tensor each and then checking
    // the rest against them catches a capture taken from a modified model far
    // more cheaply than hardcoding the dimensions of a particular sentence.
    let symbols = g.shape("input_ids").unwrap()[1];
    let frames = g.shape("z_p").unwrap()[2];
    assert!(symbols > 0 && frames > symbols);

    for name in ["embed_raw", "embed", "enc_out", "m_p", "logs_p"] {
        assert_eq!(
            g.shape(name).unwrap(),
            [1, symbols, FLOW_SIZE],
            "{name} is not [1, symbols, flow]",
        );
    }
    assert_eq!(g.shape("log_duration").unwrap(), [1, 1, symbols]);
    assert_eq!(g.shape("noise_dur").unwrap(), [1, 2, symbols]);

    // The expansion matrix is the hinge between the two: symbols in, frames out.
    assert_eq!(g.shape("attn").unwrap(), [1, frames, symbols]);

    for name in ["noise_prior", "z_p", "z"] {
        assert_eq!(g.shape(name).unwrap(), [1, FLOW_SIZE, frames], "{name}");
    }
    assert_eq!(
        g.shape("waveform").unwrap(),
        [1, frames * HOP_LENGTH],
        "the decoder's upsample stages must multiply to {HOP_LENGTH}",
    );
}

#[test]
fn the_expansion_matrix_is_one_symbol_per_frame() {
    let Some(g) = capture() else { return };

    let symbols = g.shape("input_ids").unwrap()[1];
    let frames = g.shape("z_p").unwrap()[2];
    let attn = g.f32s("attn").unwrap();

    // `attn` is built by differencing a cumulative-duration mask, which is an
    // indirect enough construction that it is worth asserting the property it
    // is supposed to have: each frame reads exactly one symbol, and every
    // symbol is read by at least one frame.
    let mut per_symbol = vec![0usize; symbols];
    for f in 0..frames {
        let row = &attn[f * symbols..(f + 1) * symbols];
        let hot: Vec<usize> = row
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hot.len(), 1, "frame {f} reads {} symbols", hot.len());
        assert_eq!(row[hot[0]], 1.0, "frame {f} has a non-unit weight");
        per_symbol[hot[0]] += 1;
    }
    assert!(
        per_symbol.iter().all(|&n| n >= 1),
        "some symbol was allocated no frames",
    );
    assert_eq!(per_symbol.iter().sum::<usize>(), frames);
}

#[test]
fn the_embedding_scaling_survived_the_capture() {
    let Some(g) = capture() else { return };

    // `embed` is the table lookup times sqrt(hidden_size). Capturing both sides
    // of that multiply is the point - the scaling is easy to forget when
    // reimplementing, and this pins the factor as a measured quantity rather
    // than a remembered one.
    let raw = g.f32s("embed_raw").unwrap();
    let scaled = g.f32s("embed").unwrap();
    let expected: Vec<f32> = raw.iter().map(|v| v * (FLOW_SIZE as f32).sqrt()).collect();

    let c = Comparison::new("embed", &scaled, &expected, 1e-5, 1e-5);
    assert!(c.passed(), "{c}");
}

#[test]
fn the_waveform_is_audio_and_not_silence() {
    let Some(g) = capture() else { return };

    let w = g.f32s("waveform").unwrap();
    let peak = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let energy = w.iter().map(|v| f64::from(v * v)).sum::<f64>() / w.len() as f64;

    // A capture that ran without error but produced silence would satisfy every
    // shape assertion above and be worthless. These bounds are deliberately
    // loose: the claim is "this is speech", not "this is a particular speech".
    assert!(
        (0.01..=1.0).contains(&peak),
        "peak amplitude {peak} is not plausible audio",
    );
    assert!(energy > 1e-6, "waveform RMS^2 {energy} is silence");
    assert!(
        w.iter().all(|v| v.is_finite()),
        "waveform contains non-finite samples",
    );
}

#[test]
fn the_provenance_is_recorded() {
    let Some(g) = capture() else { return };
    let m = g.manifest();

    // The reference has changed shape before. A capture that does not say which
    // version produced it cannot be checked against the current reference, so
    // an empty version field is a defect in the capture tooling.
    assert!(
        !m.transformers.is_empty(),
        "no transformers version recorded"
    );
    assert!(!m.torch.is_empty(), "no torch version recorded");
    assert_eq!(m.device, "cpu", "a GPU capture is not the oracle");
    assert_eq!(m.dtype, "float32");
    assert_eq!(
        m.threads, 1,
        "float32 reduction order is not thread-invariant"
    );
    assert_eq!(m.sampling_rate, 16_000);
    assert!(m.noise_scale > 0.0 && m.noise_scale_duration > 0.0);
}

#[test]
fn a_damaged_capture_is_refused_rather_than_read() {
    let Some(g) = capture() else { return };

    // A truncated or flipped `.bin` reads as a perfectly plausible tensor -
    // there is no header to disagree with. The recorded checksum is the only
    // thing standing between a damaged capture and a mystifying failure five
    // milestones downstream, so it is worth proving it actually fires.
    let scratch = std::env::temp_dir().join(format!("xabe-golden-damaged-{}", std::process::id(),));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    for entry in std::fs::read_dir(g.dir()).unwrap() {
        let p = entry.unwrap().path();
        std::fs::copy(&p, scratch.join(p.file_name().unwrap())).unwrap();
    }

    let victim = scratch.join("log_duration.bin");
    let mut bytes = std::fs::read(&victim).unwrap();
    bytes[0] ^= 0xff;
    std::fs::write(&victim, &bytes).unwrap();

    let damaged = Golden::open(&scratch).unwrap();
    let err = damaged.f32s("log_duration").unwrap_err();
    assert!(
        matches!(err, GoldenError::Corrupt { .. }),
        "expected Corrupt, got {err}",
    );
    // Every other stage still reads, so the error names the damage rather than
    // condemning the whole capture.
    assert!(damaged.f32s("enc_out").is_ok());

    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn a_missing_stage_names_what_is_available() {
    let Some(g) = capture() else { return };

    let err = g.f32s("enc_ouput").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("enc_out"),
        "the message should list real stages: {msg}"
    );
}
