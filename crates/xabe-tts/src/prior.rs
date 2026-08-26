//! Length regulation: symbols to frames.
//!
//! The duration predictor says how many frames each symbol lasts. This turns
//! that into the alignment matrix and uses it to stretch the prior's mean and
//! log-variance from one value per symbol to one per frame, then samples.
//!
//! # Durations are rounded up, and that is load-bearing
//!
//! The reference computes `ceil(exp(log_duration) / speaking_rate)`. Rounding
//! rather than truncating means every symbol gets at least one frame even when
//! its predicted duration is tiny, so no symbol is silently deleted. It also
//! means the total frame count is not a smooth function of the log durations: a
//! difference of 1e-6 in one symbol can change the output length by a whole
//! frame, and every tensor downstream with it. That is why the duration
//! comparison against the oracle is done on the frame count and not only on a
//! tolerance.
//!
//! # The alignment is hard, not soft
//!
//! `attn` is built by differencing a cumulative-duration mask, and the result
//! is exactly one symbol per frame with weight 1 - not a distribution. The
//! matrix multiplication the reference uses to apply it is therefore a gather
//! in disguise, and that is how it is written here.

use xabe_vits::VitsConfig;

/// The expanded prior, and the alignment that produced it.
#[derive(Debug, Clone)]
pub struct Prior {
    /// Which symbol each frame reads, `[frames]`.
    pub alignment: Vec<usize>,
    /// The sampled prior latents, `[flow_size, frames]`.
    pub z_p: Vec<f32>,
    /// Frames in the output.
    pub frames: usize,
}

impl Prior {
    /// Rebuilds the dense `[frames, symbols]` alignment matrix.
    ///
    /// Only the tests want this - the forward pass uses [`Self::alignment`]
    /// directly - but the oracle captured the dense form, so this is what makes
    /// the two comparable.
    pub fn attention_matrix(&self, symbols: usize) -> Vec<f32> {
        let mut m = vec![0.0; self.frames * symbols];
        for (f, &s) in self.alignment.iter().enumerate() {
            m[f * symbols + s] = 1.0;
        }
        m
    }
}

/// Expands the prior from one value per symbol to one per frame, and samples.
///
/// `m_p` and `logs_p` are `[symbols, flow_size]` as the text encoder produces
/// them; `log_duration` is `[symbols]`; `noise` is a `[flow_size, frames]`
/// standard normal draw whose frame count must match what the durations imply.
pub fn expand_prior(
    m_p: &[f32],
    logs_p: &[f32],
    log_duration: &[f32],
    noise: &[f32],
    cfg: &VitsConfig,
) -> Prior {
    let symbols = log_duration.len();
    let ch = cfg.flow_size;

    // `speaking_rate` is a rate, so the reference divides by it: larger is
    // faster and gives fewer frames.
    let durations: Vec<usize> = log_duration
        .iter()
        .map(|v| (v.exp() / cfg.speaking_rate).ceil().max(0.0) as usize)
        .collect();
    let frames = durations.iter().sum::<usize>().max(1);

    let mut alignment = Vec::with_capacity(frames);
    for (s, &d) in durations.iter().enumerate() {
        alignment.extend(std::iter::repeat_n(s, d));
    }
    alignment.truncate(frames);
    // A degenerate all-zero duration vector would leave this short; the
    // reference's clamp to at least one frame has the same effect.
    while alignment.len() < frames {
        alignment.push(symbols.saturating_sub(1));
    }

    let mut z_p = vec![0.0; ch * frames];
    for (f, &s) in alignment.iter().enumerate() {
        for c in 0..ch {
            let mean = m_p[s * ch + c];
            let logs = logs_p[s * ch + c];
            z_p[c * frames + f] = mean + noise[c * frames + f] * logs.exp() * cfg.noise_scale;
        }
    }

    Prior {
        alignment,
        z_p,
        frames,
    }
}
