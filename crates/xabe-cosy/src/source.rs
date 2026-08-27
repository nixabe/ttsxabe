//! The excitation HiFTNet filters: an F0 predictor and a bank of oscillators.
//!
//! # Why a vocoder needs a source at all
//!
//! HiFTNet is a *neural source filter*. The network does not invent a waveform
//! from a mel; it shapes an excitation that already has the right pitch. That
//! excitation is a sum of harmonics at the predicted F0, plus noise where the
//! frame is unvoiced, and getting it wrong gives speech with the right timbre
//! and the wrong pitch - which sounds like a different person rather than like
//! a bug.
//!
//! # Three buffers that are not in the checkpoint
//!
//! `SineGen2` and `SourceModuleHnNSF` each call `torch.rand` in `__init__` and
//! keep the result as a plain attribute - not a parameter, not a registered
//! buffer - so none of it reaches `hift.pt`. They are regenerated from torch's
//! global RNG every time the model is constructed, which means **upstream does
//! not reproduce across load orderings either**.
//!
//! They are dither: an initial phase offset per harmonic, a bank of noise for
//! the harmonics, and the unvoiced noise. Perceptually they are nothing.
//! Numerically they decide every sample, so this engine loads the captured
//! ones rather than drawing its own - see [`Dither`] and
//! `tools/oracle/capture_cosyvoice.py`.
//!
//! One of the three turns out not to matter, and it is worth saying which:
//! `rand_ini` is added to phase row **0 only**, and the very next operation
//! decimates by 480 sampling at 239.5 - so row 0 is never read. It is loaded
//! and applied anyway, because "this input has no effect" is a property of the
//! current arithmetic and not something to bake in.
//!
//! # This runs on the host
//!
//! Deliberately. It is 137,000 samples of elementwise work, one prefix sum and
//! a nine-wide dot product - microseconds either way against a language model
//! and a 22-layer transformer. The prefix sum is the part that would want
//! care on a device, and it is exactly the part where a wrong answer is a
//! wrong pitch.

use crate::CosyError;
use crate::vocoder::{Conv, causal_pad};
use xabe_cuda::{CudaSlice, Gpu};
use xabe_st::StFile;

/// Harmonics the oscillator bank produces: the fundamental and eight above it.
pub const HARMONICS: usize = 9;

/// The constants `cosyvoice3.yaml` gives the source module.
#[derive(Debug, Clone, Copy)]
pub struct SourceConfig {
    /// Output rate.
    pub sample_rate: usize,
    /// Samples per mel frame.
    pub upsample: usize,
    /// Amplitude of the harmonics, `nsf_alpha`.
    pub sine_amp: f32,
    /// Noise amplitude where the frame is voiced, `nsf_sigma`.
    pub noise_std: f32,
    /// F0 above which a frame counts as voiced, `nsf_voiced_threshold`.
    ///
    /// **Ten hertz, not zero.** Upstream's default for this class is zero and
    /// the yaml overrides it; at zero every frame with any predicted pitch at
    /// all counts as voiced, which turns silence into a hum.
    pub voiced_threshold: f32,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            upsample: 480,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 10.0,
        }
    }
}

/// The three constructed-not-trained buffers, as captured.
pub struct Dither {
    /// One phase offset per harmonic, `[HARMONICS]`.
    pub rand_ini: Vec<f32>,
    /// Noise for the harmonics, `[n, HARMONICS]` in row-major order.
    pub sine_waves: Vec<f32>,
    /// How many samples of it there are.
    pub len: usize,
}

impl Dither {
    /// Draws a fresh dither of `samples` samples.
    ///
    /// Upstream's is `torch.rand(1, 300 * 24000, 9)` taken from the *global*
    /// RNG during construction, so it is not in the checkpoint and does not
    /// reproduce across load orderings - see the module header. There is
    /// therefore nothing to match, and shipping 259 MB of somebody else's
    /// arbitrary draw with every voice would be pretending otherwise.
    ///
    /// So the engine draws its own, from a named seed, and is reproducible on
    /// its own terms. The differential test still loads the captured one,
    /// because there the point is to compare against upstream rather than to
    /// sound right.
    pub fn seeded(samples: usize, seed: u64) -> Self {
        let mut rng = crate::Rng::new(seed);
        // Uniform, not normal: `torch.rand`, not `torch.randn`. The buffer is
        // used as additive noise and a normal draw here would be a louder hiss
        // with a different mean.
        let sine_waves = (0..samples * HARMONICS).map(|_| rng.unit()).collect();
        let mut rand_ini: Vec<f32> = (0..HARMONICS).map(|_| rng.unit()).collect();
        rand_ini[0] = 0.0;
        Self {
            rand_ini,
            sine_waves,
            len: samples,
        }
    }

    /// Refuses a bundle that cannot cover `samples`.
    ///
    /// Named rather than silently truncated: too short a buffer would make the
    /// tail of a long utterance silent noise, which is audible as a fade and
    /// is very hard to attribute back to a capture that was taken on a shorter
    /// sentence.
    pub fn check(&self, samples: usize) -> Result<(), CosyError> {
        if self.rand_ini.len() != HARMONICS {
            return Err(CosyError::Speaker {
                what: format!(
                    "rand_ini has {} values, expected {HARMONICS}",
                    self.rand_ini.len()
                ),
            });
        }
        if self.len < samples {
            return Err(CosyError::Speaker {
                what: format!(
                    "the dither covers {} samples and this utterance needs {samples}",
                    self.len
                ),
            });
        }
        Ok(())
    }
}

/// The F0 predictor: five causal convolutions and a linear head.
pub struct F0Predictor {
    convs: Vec<Conv>,
    head_w: CudaSlice<f32>,
    head_b: Vec<f32>,
}

impl F0Predictor {
    /// Binds the predictor out of an open `hift.safetensors`.
    pub fn bind(f: &StFile, gpu: &Gpu) -> Result<Self, CosyError> {
        // Kernel four then four of three, and the first is the only one that
        // pads on the *right*: it may look ahead, the rest may not.
        let shapes = [
            (512usize, 80usize, 4usize),
            (512, 512, 3),
            (512, 512, 3),
            (512, 512, 3),
            (512, 512, 3),
        ];
        let mut convs = Vec::new();
        for (i, &(out, inp, k)) in shapes.iter().enumerate() {
            // The layers are 0, 2, 4, 6, 8: the odd indices are the ELUs, and
            // `nn.Sequential` numbers them too.
            let p = format!("f0_predictor.condnet.{}", i * 2);
            convs.push(Conv::bind_wn(f, gpu, &p, out, inp, k)?);
        }
        Ok(Self {
            head_w: gpu.upload(f.tensor_shaped("f0_predictor.classifier.weight", &[1, 512])?)?,
            head_b: f
                .tensor_shaped("f0_predictor.classifier.bias", &[1])?
                .to_vec(),
            convs,
        })
    }

    /// Predicts one F0 per mel frame.
    ///
    /// Upstream runs this in **float64** - "precision is crucial for causal
    /// inference" - and this runs float32. That is a deliberate difference and
    /// the test measures it rather than assuming it away: the streaming path
    /// the comment is about does not exist here, and a whole-utterance pass
    /// has no cache boundary for an error to accumulate across.
    pub fn predict(
        &self,
        gpu: &Gpu,
        mel: &CudaSlice<f32>,
        frames: usize,
    ) -> Result<Vec<f32>, CosyError> {
        let mut x = gpu.zeros(80 * frames)?;
        gpu.copy_into(&mut x, mel, 0, 80 * frames)?;

        for (i, c) in self.convs.iter().enumerate() {
            let pad = causal_pad(c.k, 1);
            // Only the first may look ahead; the rest pad on the left.
            let (l, r) = if i == 0 { (0, pad) } else { (pad, 0) };
            let (y, t) = gpu.conv1d(
                &x,
                &c.w,
                Some(&c.bias),
                c.in_ch,
                frames,
                c.out_ch,
                c.k,
                l,
                r,
                1,
            )?;
            debug_assert_eq!(t, frames, "a causal convolution changed the length");
            x = y;
            gpu.elu(&mut x, c.out_ch * frames)?;
        }

        // A 512-to-1 projection, then the absolute value: F0 is a frequency
        // and upstream takes `abs` rather than a softplus or a clamp, so a
        // strongly negative activation becomes a strongly *positive* pitch.
        let h = gpu.download(&x)?;
        let w = gpu.download(&self.head_w)?;
        let mut out = vec![0.0f32; frames];
        for (t, o) in out.iter_mut().enumerate() {
            let mut acc = self.head_b[0];
            for (c, wc) in w.iter().enumerate() {
                acc += wc * h[c * frames + t];
            }
            *o = acc.abs();
        }
        Ok(out)
    }
}

/// Turns one F0 per frame into one excitation sample per output sample.
///
/// Faithful to `SineGen2` with `causal = True`, which is what a 24 kHz
/// checkpoint selects. Two things in it look like mistakes and are not:
///
/// - The phase is accumulated at the **frame** rate and then held constant
///   across each frame's 480 samples (`mode="nearest"`). The non-causal path
///   interpolates instead, and the trained weights were fitted against this
///   one.
/// - The decimation before the prefix sum is a linear resample by 1/480, which
///   on a signal that is already piecewise-constant over 480 samples returns
///   the block's value - and never reads sample 0, which is why `rand_ini`
///   has no effect.
pub fn excitation(
    f0: &[f32],
    cfg: &SourceConfig,
    dither: &Dither,
    linear_w: &[f32],
    linear_b: f32,
) -> Result<Vec<f32>, CosyError> {
    let frames = f0.len();
    let n = frames * cfg.upsample;
    dither.check(n)?;

    // Accumulated in f64: this is a prefix sum over thousands of terms whose
    // result is fed to a sine, so drift here is drift in pitch.
    let mut phase = vec![0.0f64; frames];
    let mut out = vec![0.0f32; n];

    for (h, &lw) in linear_w.iter().enumerate().take(HARMONICS) {
        let mult = (h + 1) as f64;
        let mut acc = 0.0f64;
        for (i, p) in phase.iter_mut().enumerate() {
            // The block value, which is what the 1/480 linear decimation
            // returns for a signal held constant across the block.
            let rad = (f64::from(f0[i]) * mult / cfg.sample_rate as f64).rem_euclid(1.0);
            acc += rad;
            *p = acc * std::f64::consts::TAU;
        }
        for i in 0..frames {
            // `phase * upsample`, held across the frame. The multiply is
            // upstream's, and it is what makes each frame a whole number of
            // cycles rather than a fraction of one.
            let s = (phase[i] * cfg.upsample as f64).sin() as f32 * cfg.sine_amp;
            let voiced = f0[i] > cfg.voiced_threshold;
            let amp = if voiced {
                cfg.noise_std
            } else {
                cfg.sine_amp / 3.0
            };
            for j in 0..cfg.upsample {
                let t = i * cfg.upsample + j;
                let noise = amp * dither.sine_waves[t * HARMONICS + h];
                let v = if voiced { s } else { 0.0 } + noise;
                out[t] += lw * v;
            }
        }
    }

    for v in &mut out {
        *v = (*v + linear_b).tanh();
    }
    Ok(out)
}
