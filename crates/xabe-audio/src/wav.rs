//! Reading and writing mono WAV.
//!
//! Writing is sixteen-bit PCM, because that is what every player and every ASR
//! front end accepts without asking. The model produces f32 in [-1, 1]; the
//! conversion scales by 32767 and rounds, which is the convention `soundfile`
//! and `torchaudio` use, so a file written here and one written by the
//! reference pipeline are byte-comparable.
//!
//! Reading is deliberately less trusting than writing. A WAV is a RIFF chunk
//! list, not a 44-byte header: real files carry `LIST`, `fact` and padding
//! chunks between `fmt ` and `data`, and the widespread assumption that samples
//! begin at offset 44 reads that metadata as audio. So the chunks are walked.
//!
//! What it refuses: anything but 16-bit PCM or 32-bit float, and anything
//! wider than stereo. Those two formats cover the whole pipeline, and a format
//! silently mishandled is worse than one named in an error.

use crate::AudioError;
use std::io::Write;
use std::path::Path;

/// Format tag for integer PCM, from the RIFF spec.
const FORMAT_PCM: u16 = 1;
/// Format tag for IEEE 754 float samples.
const FORMAT_FLOAT: u16 = 3;
/// Format tag for the extensible header, whose real format is in a sub-field.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Decoded mono audio and the rate it was recorded at.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav {
    /// Samples in [-1, 1], one channel, downmixed if the file was stereo.
    pub samples: Vec<f32>,
    /// Frames per second, as declared by the file.
    pub sample_rate: u32,
}

impl Wav {
    /// Duration in seconds.
    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

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

/// Reads a WAV file into mono f32 samples.
pub fn read_wav(path: impl AsRef<Path>) -> Result<Wav, AudioError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| AudioError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_wav(&bytes)
}

/// Parses a WAV already in memory.
///
/// Separate from [`read_wav`] so the multipart upload the server receives does
/// not have to be written to a file first.
pub fn parse_wav(bytes: &[u8]) -> Result<Wav, AudioError> {
    tag(bytes, 0, b"RIFF", "RIFF")?;
    // Bytes 4..8 are the RIFF size. It is routinely wrong - streamed writers
    // cannot know it in advance and leave it at 0 or 0xFFFFFFFF - so the chunk
    // walk below is bounded by the real file length instead.
    tag(bytes, 8, b"WAVE", "WAVE")?;

    let mut fmt: Option<Fmt> = None;
    let mut data: Option<&[u8]> = None;

    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
        let size = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
            as usize;
        let body = at + 8;
        let available = bytes.len().saturating_sub(body);
        if size > available {
            return Err(AudioError::Truncated {
                what: "a chunk body",
                at: body,
                needed: size,
                available,
            });
        }
        match &id {
            b"fmt " => fmt = Some(parse_fmt(&bytes[body..body + size], body)?),
            b"data" => data = Some(&bytes[body..body + size]),
            _ => {}
        }
        // Chunks are word-aligned: an odd size is followed by a pad byte that
        // is not counted in the size field. Missing this walks into the middle
        // of the next chunk and reads its tag as garbage.
        at = body + size + (size & 1);
    }

    let fmt = fmt.ok_or(AudioError::MissingChunk("fmt "))?;
    let data = data.ok_or(AudioError::MissingChunk("data"))?;
    decode(&fmt, data)
}

/// The parts of a `fmt ` chunk this crate acts on.
struct Fmt {
    format: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

fn parse_fmt(body: &[u8], at: usize) -> Result<Fmt, AudioError> {
    if body.len() < 16 {
        return Err(AudioError::Truncated {
            what: "the fmt chunk",
            at,
            needed: 16,
            available: body.len(),
        });
    }
    let mut format = u16::from_le_bytes([body[0], body[1]]);
    // WAVE_FORMAT_EXTENSIBLE keeps the real tag in the first two bytes of the
    // GUID that follows the extension size. Files from Windows capture stacks
    // routinely use it for plain 16-bit PCM.
    if format == FORMAT_EXTENSIBLE && body.len() >= 26 {
        format = u16::from_le_bytes([body[24], body[25]]);
    }
    Ok(Fmt {
        format,
        channels: u16::from_le_bytes([body[2], body[3]]),
        sample_rate: u32::from_le_bytes([body[4], body[5], body[6], body[7]]),
        bits: u16::from_le_bytes([body[14], body[15]]),
    })
}

fn decode(fmt: &Fmt, data: &[u8]) -> Result<Wav, AudioError> {
    if fmt.sample_rate == 0 {
        return Err(AudioError::ZeroSampleRate);
    }
    let channels = match fmt.channels {
        1 | 2 => fmt.channels as usize,
        n => return Err(AudioError::UnsupportedChannels(n)),
    };
    let width = match (fmt.format, fmt.bits) {
        (FORMAT_PCM, 16) => 2,
        (FORMAT_FLOAT, 32) => 4,
        (f, b) => {
            return Err(AudioError::UnsupportedFormat(format!(
                "format tag {f} at {b} bits; expected 16-bit PCM or 32-bit float"
            )));
        }
    };

    let block_align = width * channels;
    if !data.len().is_multiple_of(block_align) {
        return Err(AudioError::RaggedData {
            bytes: data.len(),
            block_align,
        });
    }

    let frames = data.len() / block_align;
    let mut samples = Vec::with_capacity(frames);
    for frame in data.chunks_exact(block_align) {
        // Stereo is averaged rather than left-channel-only: a clip recorded
        // with one live channel and one dead one is common, and dropping the
        // wrong one yields silence that looks like a VAD failure.
        let mut acc = 0.0f32;
        for lane in frame.chunks_exact(width) {
            acc += match width {
                2 => f32::from(i16::from_le_bytes([lane[0], lane[1]])) / 32768.0,
                _ => f32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]),
            };
        }
        samples.push(acc / channels as f32);
    }

    Ok(Wav {
        samples,
        sample_rate: fmt.sample_rate,
    })
}

fn tag(bytes: &[u8], at: usize, want: &[u8; 4], name: &'static str) -> Result<(), AudioError> {
    if bytes.len() < at + 4 {
        return Err(AudioError::Truncated {
            what: name,
            at,
            needed: 4,
            available: bytes.len().saturating_sub(at),
        });
    }
    let found = [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]];
    if &found != want {
        return Err(AudioError::BadTag {
            expected: name,
            at,
            found,
        });
    }
    Ok(())
}
