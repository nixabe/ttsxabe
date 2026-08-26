//! The forward pass: 512 samples in, one probability out.
//!
//! ```text
//!   frame[512]
//!     └ reflect-pad 64 each end            → [640]
//!     └ conv1d, 258 kernels of 256, hop 128 → [258, 4]
//!     └ sqrt(re² + im²) over the two halves → [129, 4]
//!     └ conv 129→128, stride 1, ReLU        → [128, 4]
//!     └ conv 128→64,  stride 2, ReLU        → [64, 2]
//!     └ conv 64→64,   stride 2, ReLU        → [64, 1]
//!     └ conv 64→128,  stride 1, ReLU        → [128, 1]
//!     └ LSTM cell, hidden 128               → [128]
//!     └ ReLU, dot with 128 weights, + bias
//!     └ sigmoid                             → one probability
//! ```
//!
//! The LSTM state carries **across frames** and is what makes this a detector
//! rather than a classifier: the probability for one 32 ms frame depends on
//! everything before it. [`Vad::reset`] is therefore not optional between
//! independent clips - carrying state across a clip boundary is a silent
//! correctness bug, not a performance detail.
//!
//! One deliberate divergence from upstream Silero is recorded in
//! [`Vad::probabilities`].

use crate::weights::{
    BINS, GATES, HIDDEN, PAD, STFT_HOP, STFT_KERNEL, STFT_ROWS, VadWeights, WINDOW,
};
use xabe_dsp::conv1d_strided;

/// A detector, holding the recurrent state between frames.
#[derive(Debug)]
pub struct Vad {
    weights: VadWeights,
    /// Hidden state, `[128]`.
    h: Vec<f32>,
    /// Cell state, `[128]`.
    c: Vec<f32>,
}

impl Vad {
    /// Wraps bound weights in a detector whose state is zeroed.
    pub fn new(weights: VadWeights) -> Vad {
        Vad {
            weights,
            h: vec![0.0; HIDDEN],
            c: vec![0.0; HIDDEN],
        }
    }

    /// Zeroes the recurrent state.
    ///
    /// Required between independent clips. The state is the memory of what was
    /// said a moment ago, and carrying it from one recording into the next
    /// makes the first frames of the second depend on the end of the first.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// The weights, for tests that want to count them.
    pub fn weights(&self) -> &VadWeights {
        &self.weights
    }

    /// One probability per [`WINDOW`] samples.
    ///
    /// The last frame is zero-padded if the audio does not divide evenly, which
    /// is what whisper.cpp does, so the probability count is
    /// `ceil(len / WINDOW)` and the final value covers a partly silent frame.
    ///
    /// **Divergence, chosen on purpose.** Upstream Silero prepends 64 *real*
    /// samples of context to each frame - the `n_context` the checkpoint
    /// declares - so a frame sees the tail of its predecessor. whisper.cpp
    /// parses `n_context` and then ignores it, substituting a symmetric
    /// reflective pad. This follows whisper.cpp, because whisper.cpp is what
    /// produced every threshold in `TurnPolicy` and every hallucination
    /// measurement in the pipeline; matching Python silero-vad instead would
    /// invalidate the tuning that the rest of the system is built on. The
    /// consequence is that these probabilities do not bit-match upstream
    /// silero-vad, and are not meant to.
    pub fn probabilities(&mut self, samples: &[f32]) -> Vec<f32> {
        let frames = samples.len().div_ceil(WINDOW).max(1);
        let mut out = Vec::with_capacity(frames);
        let mut window = vec![0.0f32; WINDOW];

        for i in 0..frames {
            let start = i * WINDOW;
            let end = (start + WINDOW).min(samples.len());
            window.fill(0.0);
            if start < samples.len() {
                window[..end - start].copy_from_slice(&samples[start..end]);
            }
            out.push(self.frame(&window));
        }
        out
    }

    /// One frame, advancing the recurrent state.
    pub fn frame(&mut self, window: &[f32]) -> f32 {
        debug_assert_eq!(window.len(), WINDOW);

        let padded = reflect_pad(window, PAD);
        let mag = self.stft(&padded);
        let encoded = self.encode(mag);
        let hidden = self.lstm(&encoded);
        self.head(&hidden)
    }

    /// The convolution that stands in for a short-time Fourier transform.
    ///
    /// The basis is 258 rows of 256: 129 cosines followed by 129 sines, each
    /// already multiplied by the analysis window. So a plain convolution
    /// produces the real and imaginary parts, and the magnitude is one
    /// `sqrt(re² + im²)` per bin.
    fn stft(&self, padded: &[f32]) -> Vec<f32> {
        let positions = (padded.len() - STFT_KERNEL) / STFT_HOP + 1;
        let spectrum = conv1d_strided(
            padded,
            1,
            padded.len(),
            &self.weights.stft_basis,
            None,
            STFT_ROWS,
            STFT_KERNEL,
            STFT_HOP,
            0,
            0,
            1,
        );

        let mut mag = vec![0.0f32; BINS * positions];
        for bin in 0..BINS {
            for t in 0..positions {
                let re = spectrum[bin * positions + t];
                let im = spectrum[(bin + BINS) * positions + t];
                mag[bin * positions + t] = (re * re + im * im).sqrt();
            }
        }
        mag
    }

    /// Four convolutions with ReLU between them, ending at one time position.
    fn encode(&self, mut cur: Vec<f32>) -> Vec<f32> {
        let mut t = cur.len() / BINS;
        for conv in &self.weights.encoder {
            let out_t = (t + 2 * conv.pad - conv.k) / conv.stride + 1;
            let mut next = conv1d_strided(
                &cur,
                conv.in_ch,
                t,
                &conv.weight,
                None,
                conv.out_ch,
                conv.k,
                conv.stride,
                conv.pad,
                conv.pad,
                1,
            );
            for ch in 0..conv.out_ch {
                let b = conv.bias[ch];
                for v in &mut next[ch * out_t..(ch + 1) * out_t] {
                    *v = (*v + b).max(0.0);
                }
            }
            cur = next;
            t = out_t;
        }
        debug_assert_eq!(t, 1, "the encoder must reduce to a single time position");
        cur
    }

    /// One LSTM step, updating `h` and `c`.
    ///
    /// Gate order is **i, f, g, o**, which is PyTorch's stacking and is not the
    /// only convention in use: getting it wrong produces a detector that runs,
    /// converges to plausible-looking probabilities, and is wrong everywhere.
    fn lstm(&mut self, x: &[f32]) -> Vec<f32> {
        let mut gates = vec![0.0f32; GATES * HIDDEN];
        for (row, gate) in gates.iter_mut().enumerate() {
            let ih = &self.weights.lstm_ih[row * HIDDEN..(row + 1) * HIDDEN];
            let hh = &self.weights.lstm_hh[row * HIDDEN..(row + 1) * HIDDEN];
            let mut acc = self.weights.lstm_bias_ih[row] + self.weights.lstm_bias_hh[row];
            for k in 0..HIDDEN {
                acc += ih[k] * x[k] + hh[k] * self.h[k];
            }
            *gate = acc;
        }

        let mut out = vec![0.0f32; HIDDEN];
        for j in 0..HIDDEN {
            let i = sigmoid(gates[j]);
            let f = sigmoid(gates[HIDDEN + j]);
            let g = gates[2 * HIDDEN + j].tanh();
            let o = sigmoid(gates[3 * HIDDEN + j]);
            self.c[j] = f * self.c[j] + i * g;
            out[j] = o * self.c[j].tanh();
        }
        self.h.copy_from_slice(&out);
        out
    }

    /// ReLU, a 1×1 convolution over 128 channels, and a sigmoid.
    fn head(&self, hidden: &[f32]) -> f32 {
        let mut acc = self.weights.head_bias;
        for (w, h) in self.weights.head_weight.iter().zip(hidden) {
            acc += w * h.max(0.0);
        }
        sigmoid(acc)
    }
}

/// Mirrors `pad` samples from each end, excluding the edge sample itself.
///
/// `[a b c d]` with `pad = 2` becomes `[c b | a b c d | c b]`. This is
/// `torch.nn.functional.pad(mode="reflect")` and `ggml_pad_reflect_1d`.
pub fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    for i in 0..pad {
        out.push(x[pad - i]);
    }
    out.extend_from_slice(x);
    for i in 0..pad {
        out.push(x[x.len() - 2 - i]);
    }
    out
}

/// The logistic function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
