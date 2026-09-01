//! Audio to the `[n_mels, 3000]` block the encoder eats.

use crate::WhisperConfig;
use xabe_audio::{MelConfig, mel_filters, mel_power};

/// Whisper's mel band ceiling, in hertz.
pub const F_MAX: f64 = 8000.0;

/// Dynamic range kept below the loudest frame, in log10 units - 80 dB.
pub const DYNAMIC_RANGE: f32 = 8.0;

/// The frontend, holding the filter bank so it is built once.
#[derive(Debug, Clone)]
pub struct Frontend {
    mel: MelConfig,
    filters: Vec<f32>,
    /// Samples in one window: 30 s at 16 kHz.
    n_samples: usize,
    /// Frames the encoder wants, which is one fewer than the transform gives.
    n_frames: usize,
}

impl Frontend {
    /// Builds the frontend `cfg`'s geometry implies.
    pub fn new(cfg: &WhisperConfig) -> Self {
        let mel = MelConfig {
            n_mels: cfg.num_mel_bins,
            ..MelConfig::default()
        };
        Self {
            filters: mel_filters(&mel, F_MAX),
            mel,
            n_samples: cfg.n_samples(),
            n_frames: cfg.n_frames(),
        }
    }

    /// The filter bank, `[n_freq, n_mels]` row-major.
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    /// Frames one window produces.
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Log-mel features, `[n_mels, n_frames]` row-major.
    ///
    /// `samples` is zero-padded or truncated to exactly one 30-second window
    /// first, because the encoder's position embedding has no other length.
    /// Truncation is a real limit and the engine avoids it by VAD-gating
    /// utterances rather than by chunking here; a chunked frontend would have
    /// to answer the normalisation question below, and there is no good answer.
    ///
    /// # The normalisation is global, and that is load-bearing
    ///
    /// The floor is `max - 8` over the *entire* window, so the value at second
    /// one depends on the loudest frame anywhere in the file - including in the
    /// silence that was padded on. Reproducing that exactly is why this takes a
    /// whole window rather than a stream: any design that normalises per chunk
    /// drifts away from the reference in a way that gets worse the quieter the
    /// speech is.
    pub fn log_mel(&self, samples: &[f32]) -> Vec<f32> {
        let mut window = vec![0.0f32; self.n_samples];
        let n = samples.len().min(self.n_samples);
        window[..n].copy_from_slice(&samples[..n]);

        let power = mel_power(&window, &self.mel, &self.filters);
        let frames = power.len() / self.mel.n_mels;
        debug_assert_eq!(frames, self.n_frames + 1);

        // The reference drops the last frame - `stft[..., :-1]` - because a
        // centred transform of N samples yields N/hop + 1 frames and the model
        // was trained on N/hop of them.
        // `power` is *exactly* zero wherever the frame was digital silence -
        // `mel_power` leaves those rows as it allocated them - and on a
        // three-second clip inside this model's fixed thirty-second window
        // that is nine bins in ten. `0f32.max(1e-10).log10()` is a constant,
        // so it is evaluated once here instead of a quarter of a million
        // times: the same expression on the same input, so the same bits, and
        // `log10` is thirty-odd cycles that this frontend was spending on
        // padding.
        let silent = (1e-10f32).log10();
        let mut out = vec![0.0f32; self.mel.n_mels * self.n_frames];
        let mut peak = f32::NEG_INFINITY;
        for m in 0..self.mel.n_mels {
            for t in 0..self.n_frames {
                let p = power[m * frames + t];
                let v = if p == 0.0 {
                    silent
                } else {
                    p.max(1e-10).log10()
                };
                out[m * self.n_frames + t] = v;
                peak = peak.max(v);
            }
        }

        let floor = peak - DYNAMIC_RANGE;
        for v in &mut out {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        out
    }
}
