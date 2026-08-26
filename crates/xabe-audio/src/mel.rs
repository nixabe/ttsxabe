//! Mel spectrograms: the filter bank, and the short-time transform it weights.
//!
//! Everything here stops at linear mel power. The logarithm, the dynamic-range
//! floor and the affine rescale that a particular model wants on top of it are
//! that model's convention, not audio's, and they live in the crate that owns
//! the model. See `xabe_whisper::log_mel` for the one this engine uses.

use xabe_dsp::{Fft, reflect_pad};

/// How a spectrogram is framed.
///
/// The defaults are Whisper's, because that is what the engine's ASR asks for,
/// but nothing here reads them as anything other than numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelConfig {
    /// Transform length, and therefore the window length.
    pub n_fft: usize,
    /// Samples between consecutive frames.
    pub hop: usize,
    /// Number of mel filters.
    pub n_mels: usize,
    /// Sample rate the frequency axis is calibrated against.
    pub sample_rate: u32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop: 160,
            n_mels: 80,
            sample_rate: 16_000,
        }
    }
}

impl MelConfig {
    /// Number of unique frequency bins, `n_fft/2 + 1`.
    pub fn n_freq(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Hertz to Slaney mels: linear below 1 kHz, logarithmic above.
///
/// This is the "slaney" scale of `torchaudio` and 🤗 `mel_filter_bank`, not
/// the "htk" one. They disagree by a few percent across the whole band, which
/// is enough to move every filter edge and produce a spectrogram that looks
/// entirely reasonable and transcribes badly.
fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    // 6.4 is the span in hertz of the top log-spaced decade divided by
    // 1000 Hz, and 27 the number of filters across it, in Slaney's original.
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

/// The inverse of [`hz_to_mel`].
fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        F_SP * mel
    }
}

/// Slaney-normalised triangular filters, `[n_freq, n_mels]` row-major.
///
/// The layout is the reference's: frequency-major, so the projection is a
/// matrix on the left of the spectrogram rather than a transpose at every
/// frame. It is also the layout the captured `mel_filters.bin` has, which is
/// how this function is tested.
///
/// Computed rather than shipped. The bank is a closed-form function of four
/// numbers, and a stored one is a file that can go missing, go stale, or be
/// silently the wrong variant - three failure modes traded for forty lines.
pub fn mel_filters(cfg: &MelConfig, f_max: f64) -> Vec<f32> {
    let (n_freq, n_mels) = (cfg.n_freq(), cfg.n_mels);

    // `n_mels + 2` edges in mel, so every filter has a left, a peak and a
    // right, and adjacent filters overlap at half amplitude.
    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(f_max));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    // The reference spaces the FFT axis over `sample_rate // 2`, integer
    // division included.
    let nyquist = (cfg.sample_rate / 2) as f64;
    let bin_hz: Vec<f64> = (0..n_freq)
        .map(|i| nyquist * i as f64 / (n_freq - 1) as f64)
        .collect();

    let mut out = vec![0.0f32; n_freq * n_mels];
    for (i, &f) in bin_hz.iter().enumerate() {
        for m in 0..n_mels {
            let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            let down = (f - lo) / (mid - lo);
            let up = (hi - f) / (hi - mid);
            // Slaney normalisation: each filter integrates to the same area,
            // so a wide high-frequency filter does not simply outweigh a
            // narrow low-frequency one.
            let enorm = 2.0 / (hi - lo);
            out[i * n_mels + m] = (down.min(up).max(0.0) * enorm) as f32;
        }
    }
    out
}

/// A periodic Hann window of length `n`.
///
/// Periodic, not symmetric: `torch.hann_window` divides by `n`, and the
/// symmetric variant every textbook writes divides by `n - 1`. The difference
/// is one sample of taper and it is audible in the top bins.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .map(|v| v as f32)
        .collect()
}

/// Mel power, `[n_mels, frames]` row-major.
///
/// Centred: the signal is reflected by `n_fft/2` on each side first, so frame
/// `t` is centred on sample `t * hop`. `frames` is `1 + len / hop`, and the
/// caller decides what to do with the last one - Whisper drops it, because the
/// reference's `stft[..., :-1]` does.
///
/// # Panics
///
/// If `filters` is not `n_freq * n_mels` long, or the signal is shorter than
/// the reflection.
pub fn mel_power(samples: &[f32], cfg: &MelConfig, filters: &[f32]) -> Vec<f32> {
    let (n_freq, n_mels) = (cfg.n_freq(), cfg.n_mels);
    assert_eq!(
        filters.len(),
        n_freq * n_mels,
        "filter bank is not [{n_freq}, {n_mels}]"
    );

    let padded = reflect_pad(samples, cfg.n_fft / 2);
    let frames = 1 + samples.len() / cfg.hop;
    let window = hann(cfg.n_fft);
    let fft = Fft::new(cfg.n_fft);

    let mut out = vec![0.0f32; n_mels * frames];
    let mut frame = vec![0.0f32; cfg.n_fft];
    let mut bins = vec![0.0f32; 2 * n_freq];
    let mut power = vec![0.0f32; n_freq];

    for t in 0..frames {
        let start = t * cfg.hop;
        // A frame of digital silence has a zero spectrum, so its contribution
        // to every mel bin is exactly zero and `out` already holds that. This
        // is not an approximation, and it is not a micro-optimisation either:
        // a model with a fixed 30-second window spends most of its frontend on
        // padding, and on a 2.7-second clip 91% of the frames are zeros. It
        // took the frontend from 171 ms to a rounding error.
        let src = &padded[start.min(padded.len())..(start + cfg.n_fft).min(padded.len())];
        if src.iter().all(|&v| v == 0.0) {
            continue;
        }
        for (i, s) in frame.iter_mut().enumerate() {
            *s = padded.get(start + i).copied().unwrap_or(0.0) * window[i];
        }
        fft.forward_real(&frame, &mut bins);
        for (p, c) in power.iter_mut().zip(bins.as_chunks::<2>().0) {
            *p = c[0] * c[0] + c[1] * c[1];
        }
        for (i, &p) in power.iter().enumerate() {
            let row = &filters[i * n_mels..(i + 1) * n_mels];
            for (m, &w) in row.iter().enumerate() {
                out[m * frames + t] += w * p;
            }
        }
    }
    out
}
