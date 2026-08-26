//! What the WAV reader must survive.
//!
//! The writer had no test at all until the reader existed to check it against.
//! Most of these are the cases that make the naive "samples start at byte 44"
//! reader wrong, because that reader is the one everybody writes first.

use xabe_audio::{AudioError, Wav, parse_wav, wav_bytes};

/// Builds a WAV with arbitrary chunks between `fmt ` and `data`.
fn build(fmt: &[u8], chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(fmt);
    for (id, data) in chunks {
        body.extend_from_slice(*id);
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            body.push(0);
        }
    }
    let mut out = Vec::from(*b"RIFF");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// A 16-bit PCM `fmt ` chunk.
fn fmt_pcm16(channels: u16, rate: u32) -> Vec<u8> {
    let block = 2 * channels;
    let mut v = Vec::new();
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    v.extend_from_slice(&block.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v
}

#[test]
fn what_the_writer_writes_the_reader_reads_back() {
    let want: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
    let got = parse_wav(&wav_bytes(&want, 16_000)).expect("parse");
    assert_eq!(got.sample_rate, 16_000);
    assert_eq!(got.samples.len(), want.len());

    // The two directions use deliberately different constants, and the bound
    // has to account for it. The writer scales by 32767 so that +1.0 lands on
    // i16::MAX rather than wrapping to i16::MIN; the reader divides by 32768,
    // which is the true inverse of the i16 range. So for a written value
    // `round(w * 32767)` read back as `/ 32768`, the error is
    //
    //     w - (w * 32767 + e) / 32768  =  (w - e) / 32768,   |e| <= 1/2
    //
    // giving |error| <= (|w| + 1/2) / 32768. Assuming a flat half-LSB instead
    // fails on loud samples, where the scale term dominates the rounding one.
    for (i, (g, w)) in got.samples.iter().zip(&want).enumerate() {
        let bound = (w.abs() + 0.5) / 32768.0;
        assert!(
            (g - w).abs() <= bound,
            "sample {i}: {g} vs {w}, bound {bound}"
        );
    }
}

#[test]
fn a_chunk_between_fmt_and_data_is_skipped_rather_than_read_as_audio() {
    // The canonical header is 44 bytes. This file's data starts at 68, which
    // is exactly the case a fixed-offset reader gets wrong: it would return
    // the LIST chunk's text as samples and be off by 12 frames thereafter.
    let pcm: Vec<u8> = (0..8i16).flat_map(|v| (v * 4096).to_le_bytes()).collect();
    let file = build(
        &fmt_pcm16(1, 16_000),
        &[
            (b"LIST", b"INFOISFT written by a test".to_vec()),
            (b"data", pcm),
        ],
    );
    let got = parse_wav(&file).expect("parse");
    assert_eq!(got.samples.len(), 8);
    assert!((got.samples[1] - 4096.0 / 32768.0).abs() < 1e-6);
}

#[test]
fn an_odd_sized_chunk_is_followed_by_a_pad_byte_that_the_walk_must_step_over() {
    let pcm: Vec<u8> = (0..4i16).flat_map(|v| (v * 1000).to_le_bytes()).collect();
    let file = build(
        &fmt_pcm16(1, 8_000),
        &[(b"fact", vec![1, 2, 3]), (b"data", pcm)],
    );
    let got = parse_wav(&file).expect("parse");
    assert_eq!(got.sample_rate, 8_000);
    assert_eq!(got.samples.len(), 4);
}

#[test]
fn stereo_is_averaged_not_left_channel_only() {
    // Left is silent, right carries the signal - a clip with one dead channel.
    // Taking the left channel alone would return silence, which downstream
    // looks like a VAD failure rather than a reader bug.
    let mut pcm = Vec::new();
    for _ in 0..4 {
        pcm.extend_from_slice(&0i16.to_le_bytes());
        pcm.extend_from_slice(&16384i16.to_le_bytes());
    }
    let file = build(&fmt_pcm16(2, 16_000), &[(b"data", pcm)]);
    let got = parse_wav(&file).expect("parse");
    assert_eq!(got.samples.len(), 4);
    assert!((got.samples[0] - 0.25).abs() < 1e-4, "{}", got.samples[0]);
}

#[test]
fn the_extensible_header_is_resolved_to_the_format_it_wraps() {
    // Windows capture stacks emit WAVE_FORMAT_EXTENSIBLE for plain 16-bit PCM.
    let mut fmt = fmt_pcm16(1, 48_000);
    fmt[0..2].copy_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&22u16.to_le_bytes()); // extension size
    fmt.extend_from_slice(&16u16.to_le_bytes()); // valid bits
    fmt.extend_from_slice(&4u32.to_le_bytes()); // channel mask
    fmt.extend_from_slice(&1u16.to_le_bytes()); // the real format tag
    fmt.extend_from_slice(&[0u8; 14]); // rest of the GUID
    let file = build(&fmt, &[(b"data", 0i16.to_le_bytes().to_vec())]);
    let got = parse_wav(&file).expect("parse");
    assert_eq!(got.sample_rate, 48_000);
}

#[test]
fn float_samples_are_taken_verbatim() {
    let mut fmt = fmt_pcm16(1, 16_000);
    fmt[0..2].copy_from_slice(&3u16.to_le_bytes());
    fmt[14..16].copy_from_slice(&32u16.to_le_bytes());
    let pcm: Vec<u8> = [0.5f32, -0.25, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let file = build(&fmt, &[(b"data", pcm)]);
    let got = parse_wav(&file).expect("parse");
    assert_eq!(got.samples, vec![0.5, -0.25, 1.0]);
}

#[test]
fn a_truncated_file_is_named_rather_than_panicking() {
    let full = wav_bytes(&[0.0; 64], 16_000);
    let cut = &full[..full.len() - 40];
    match parse_wav(cut) {
        Err(AudioError::Truncated { what, .. }) => assert_eq!(what, "a chunk body"),
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn an_unsupported_width_is_refused_by_name_rather_than_misread() {
    // 24-bit PCM is common and is not handled. Reading it as 16-bit would
    // produce plausible noise, which is the failure this refusal prevents.
    let mut fmt = fmt_pcm16(1, 16_000);
    fmt[14..16].copy_from_slice(&24u16.to_le_bytes());
    let file = build(&fmt, &[(b"data", vec![0; 9])]);
    match parse_wav(&file) {
        Err(AudioError::UnsupportedFormat(m)) => assert!(m.contains("24"), "{m}"),
        other => panic!("expected UnsupportedFormat, got {other:?}"),
    }
}

#[test]
fn a_file_that_is_not_riff_is_refused_at_the_first_tag() {
    match parse_wav(b"OggS____________________________") {
        Err(AudioError::BadTag { expected, at, .. }) => {
            assert_eq!((expected, at), ("RIFF", 0));
        }
        other => panic!("expected BadTag, got {other:?}"),
    }
}

#[test]
fn duration_is_reported_in_seconds() {
    let w = Wav {
        samples: vec![0.0; 8_000],
        sample_rate: 16_000,
    };
    assert!((w.seconds() - 0.5).abs() < 1e-6);
}
