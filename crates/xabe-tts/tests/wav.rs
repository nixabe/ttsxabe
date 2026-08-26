//! The WAV container, checked byte for byte.
//!
//! Needs no model and no capture: a header is a header. It is worth testing
//! because a wrong field here produces a file that some players open and others
//! reject, which is a far more annoying failure than one that never opens.

use xabe_tts::wav_bytes;

/// Reads a little-endian `u32` at `off`.
fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Reads a little-endian `u16` at `off`.
fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[test]
fn the_header_says_what_the_file_contains() {
    let samples = vec![0.0f32; 1000];
    let b = wav_bytes(&samples, 16_000);

    assert_eq!(&b[0..4], b"RIFF");
    assert_eq!(&b[8..12], b"WAVE");
    assert_eq!(&b[12..16], b"fmt ");
    assert_eq!(&b[36..40], b"data");

    assert_eq!(u32_at(&b, 16), 16, "PCM header length");
    assert_eq!(u16_at(&b, 20), 1, "format must be PCM");
    assert_eq!(u16_at(&b, 22), 1, "mono");
    assert_eq!(u32_at(&b, 24), 16_000, "sample rate");
    assert_eq!(u32_at(&b, 28), 32_000, "bytes per second");
    assert_eq!(u16_at(&b, 32), 2, "block align");
    assert_eq!(u16_at(&b, 34), 16, "bits per sample");

    // The two length fields are the ones that actually get miscomputed: the
    // RIFF size excludes its own eight bytes, the data size counts only the
    // samples.
    assert_eq!(u32_at(&b, 4) as usize, b.len() - 8, "RIFF chunk size");
    assert_eq!(u32_at(&b, 40) as usize, samples.len() * 2, "data size");
    assert_eq!(b.len(), 44 + samples.len() * 2);
}

#[test]
fn full_scale_does_not_wrap() {
    // Scaling by 32768 rather than 32767 sends +1.0 to -32768, which is a loud
    // click at exactly the loudest moment - the one place nobody is listening
    // for a bug.
    let b = wav_bytes(&[1.0, -1.0, 0.0], 16_000);
    let s: Vec<i16> = b[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(i16::from_le_bytes)
        .collect();
    assert_eq!(s, vec![32767, -32767, 0]);
}

#[test]
fn out_of_range_input_is_clamped_not_wrapped() {
    // `tanh` bounds the model's own output, but this function is public and a
    // caller can hand it anything.
    let b = wav_bytes(&[1.5, -2.0], 16_000);
    let s: Vec<i16> = b[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(i16::from_le_bytes)
        .collect();
    assert_eq!(s, vec![32767, -32767]);
}

#[test]
fn an_empty_waveform_is_a_valid_empty_file() {
    let b = wav_bytes(&[], 16_000);
    assert_eq!(b.len(), 44);
    assert_eq!(u32_at(&b, 40), 0);
    assert_eq!(u32_at(&b, 4) as usize, 36);
}
