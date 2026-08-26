//! Writing a mono WAV.
//!
//! Sixteen-bit PCM, because that is what every player and every ASR front end
//! accepts without asking. The model produces f32 in [-1, 1]; the conversion
//! scales by 32767 and rounds, which is the convention `soundfile` and
//! `torchaudio` use, so a file written here and one written by the reference
//! pipeline are byte-comparable.

use std::io::Write;

/// Serialises samples in [-1, 1] as a 16-bit mono WAV.
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM header size
    out.extend_from_slice(&1u16.to_le_bytes()); // format: PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // bytes per second
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());

    for &v in samples {
        // Clamped before scaling: `tanh` bounds the model's own output, but a
        // caller can hand this anything, and wrapping a stray 1.2 to -32768
        // would be a loud click rather than a quiet error.
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Writes a mono WAV to `w`.
pub fn write_wav(w: &mut dyn Write, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    w.write_all(&wav_bytes(samples, sample_rate))
}
