//! CosyVoice3's HiFT vocoder: mel and an excitation signal in, waveform out.
//!
//! # Not the HiFi-GAN next door
//!
//! `xabe-tts` already runs a HiFi-GAN decoder, and this looks like it: a
//! `conv_pre`, three upsampling stages each followed by three dilated
//! residual blocks, a `conv_post`. Four things differ, and the first is the
//! one that would have cost the most:
//!
//! - **`ups` is not a transposed convolution.** The name is HiFi-GAN's, the
//!   weight shape `[out, in, k]` is a plain conv's, and upstream's
//!   `CausalConv1dUpsample` is *nearest-neighbour upsampling by `stride`
//!   followed by a causal conv*. A transposed convolution with the same weight
//!   runs, produces the right output length, and is a different function.
//! - **Every convolution is causal**, padded on one side rather than
//!   symmetrically. `conv_pre` pads *right* and everything else pads *left*.
//! - **Snake, not leaky ReLU**, inside the residual blocks - `x + sin²(αx)/α`
//!   with a trained α per channel. The leaky ReLUs remain between stages.
//! - **The head is an inverse STFT**, not a convolution to a waveform. The
//!   last convolution produces 18 channels which are read as a log-magnitude
//!   and a phase over nine bins, and 4× more samples come out than frames went
//!   in.
//!
//! # The excitation is the other half
//!
//! HiFTNet is a *neural source filter*: the network shapes an excitation
//! signal rather than inventing a waveform. That signal is the caller's, and
//! its own path - an F0 predictor and a bank of sine oscillators - is
//! [`crate::source`]. Here it arrives as samples, is transformed once, and is
//! mixed into each stage at that stage's rate.

use crate::CosyError;
use xabe_cuda::{CudaSlice, Gpu};
use xabe_st::StFile;

/// A convolution whose weight normalisation has been fused at load.
pub(crate) struct Conv {
    pub(crate) w: CudaSlice<f32>,
    pub(crate) bias: CudaSlice<f32>,
    pub(crate) in_ch: usize,
    pub(crate) out_ch: usize,
    pub(crate) k: usize,
}

impl Conv {
    /// Binds one weight-normalised convolution.
    ///
    /// The checkpoint stores a direction and a magnitude; the convolution
    /// wants their product, and it does not change between utterances - so it
    /// is fused once here rather than at every call.
    pub(crate) fn bind_wn(
        f: &StFile,
        gpu: &Gpu,
        prefix: &str,
        out: usize,
        inp: usize,
        k: usize,
    ) -> Result<Self, CosyError> {
        let v = gpu.upload(f.tensor_shaped(&format!("{prefix}.weight_v"), &[out, inp, k])?)?;
        let g = gpu.upload(f.tensor_shaped(&format!("{prefix}.weight_g"), &[out, 1, 1])?)?;
        Ok(Self {
            w: gpu.fuse_weight_norm(&v, &g, out, inp, k)?,
            bias: gpu.upload(f.tensor_shaped(&format!("{prefix}.bias"), &[out])?)?,
            in_ch: inp,
            out_ch: out,
            k,
        })
    }
}

/// One residual block: three (Snake, conv, Snake, conv) pairs, each added back.
struct ResBlock {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    alphas1: Vec<CudaSlice<f32>>,
    alphas2: Vec<CudaSlice<f32>>,
    dilations: Vec<usize>,
    ch: usize,
}

/// The vocoder's geometry, transcribed from `cosyvoice3.yaml`.
#[derive(Debug, Clone, Copy)]
pub struct HiftConfig {
    /// Mel bands in.
    pub in_channels: usize,
    /// Width after `conv_pre`; halved at each stage.
    pub base_channels: usize,
    /// Output rate.
    pub sample_rate: usize,
    /// Points in the inverse transform.
    pub n_fft: usize,
    /// Its hop, which is also the last upsampling factor.
    pub hop_len: usize,
    /// How far `conv_pre` may look ahead.
    pub look_right: usize,
    /// Slope of the leaky ReLUs between stages.
    pub lrelu_slope: f32,
    /// The waveform is clamped to this.
    pub audio_limit: f32,
}

impl Default for HiftConfig {
    fn default() -> Self {
        Self {
            in_channels: 80,
            base_channels: 512,
            sample_rate: 24_000,
            n_fft: 16,
            hop_len: 4,
            look_right: 4,
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        }
    }
}

impl HiftConfig {
    /// Upsampling factors, in order.
    pub const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
    /// Their kernel widths.
    pub const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
    /// Residual block kernel widths, three per stage.
    pub const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
    /// The excitation branch's, which are not the same.
    pub const SOURCE_KERNELS: [usize; 3] = [7, 7, 11];
    /// Dilations inside every residual block.
    pub const DILATIONS: [usize; 3] = [1, 3, 5];

    /// Samples per mel frame: the three upsamplings and then the hop.
    pub fn hop(&self) -> usize {
        Self::UPSAMPLE_RATES.iter().product::<usize>() * self.hop_len
    }
}

/// The vocoder, resident on one card.
pub struct Vocoder {
    cfg: HiftConfig,
    gpu: Gpu,
    conv_pre: Conv,
    ups: Vec<Conv>,
    source_downs: Vec<Conv>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv,
    window: CudaSlice<f32>,
}

/// The left padding a causal convolution needs to keep its length.
///
/// Upstream's expression, transcribed rather than simplified: it is
/// `int((k*d - d)/2)*2 + (k+1)%2`, which is `(k-1)*d` for odd `k` and one more
/// than that rounded down for even `k`. Simplifying it to `(k-1)*d` agrees on
/// every odd kernel in this model and is wrong on `conv_pre`'s even one.
pub(crate) fn causal_pad(k: usize, dilation: usize) -> usize {
    (k * dilation - dilation) / 2 * 2 + (k + 1) % 2
}

impl Vocoder {
    /// The slope of the *last* leaky ReLU, which is not the configured one.
    ///
    /// See the call site: upstream defaults it and configures every other one.
    pub const POST_LRELU_SLOPE: f32 = 0.01;

    /// Loads `hift.safetensors` onto CUDA device `ordinal`.
    pub fn open(path: &std::path::Path, ordinal: usize) -> Result<Self, CosyError> {
        let cfg = HiftConfig::default();
        let f = StFile::open(path)?;
        let gpu = Gpu::open(ordinal)?;
        Self::from_parts(cfg, f, gpu)
    }

    /// The same, on a device already open.
    pub fn from_parts(cfg: HiftConfig, f: StFile, gpu: Gpu) -> Result<Self, CosyError> {
        // Weight norm is fused once here rather than at every call. The
        // checkpoint stores a direction and a magnitude; the convolution wants
        // their product, and it does not change between utterances.
        let wn = |p: &str, out: usize, inp: usize, k: usize| -> Result<Conv, CosyError> {
            Conv::bind_wn(&f, &gpu, p, out, inp, k)
        };
        // The excitation's downsamplers are the one place with no weight norm.
        let plain = |p: &str, out: usize, inp: usize, k: usize| -> Result<Conv, CosyError> {
            Ok(Conv {
                w: gpu.upload(f.tensor_shaped(&format!("{p}.weight"), &[out, inp, k])?)?,
                bias: gpu.upload(f.tensor_shaped(&format!("{p}.bias"), &[out])?)?,
                in_ch: inp,
                out_ch: out,
                k,
            })
        };
        let block = |p: &str, ch: usize, k: usize| -> Result<ResBlock, CosyError> {
            let mut r = ResBlock {
                convs1: Vec::new(),
                convs2: Vec::new(),
                alphas1: Vec::new(),
                alphas2: Vec::new(),
                dilations: HiftConfig::DILATIONS.to_vec(),
                ch,
            };
            for j in 0..3 {
                r.convs1.push(wn(&format!("{p}.convs1.{j}"), ch, ch, k)?);
                r.convs2.push(wn(&format!("{p}.convs2.{j}"), ch, ch, k)?);
                r.alphas1.push(
                    gpu.upload(f.tensor_shaped(&format!("{p}.activations1.{j}.alpha"), &[ch])?)?,
                );
                r.alphas2.push(
                    gpu.upload(f.tensor_shaped(&format!("{p}.activations2.{j}.alpha"), &[ch])?)?,
                );
            }
            Ok(r)
        };

        let base = cfg.base_channels;
        let spec = cfg.n_fft + 2;

        let mut ups = Vec::new();
        for (i, (&_u, &k)) in HiftConfig::UPSAMPLE_RATES
            .iter()
            .zip(&HiftConfig::UPSAMPLE_KERNELS)
            .enumerate()
        {
            ups.push(wn(&format!("ups.{i}"), base >> (i + 1), base >> i, k)?);
        }

        // The excitation is downsampled to each stage's rate. The rates are
        // the *reverse* cumulative product of the upsampling rates, so the
        // first stage sees the excitation decimated by fifteen and the last
        // sees it untouched - a one-tap convolution rather than a stride.
        let down_rates = [15usize, 3, 1];
        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();
        for (i, (&u, &k)) in down_rates
            .iter()
            .zip(&HiftConfig::SOURCE_KERNELS)
            .enumerate()
        {
            let out = base >> (i + 1);
            source_downs.push(if u == 1 {
                plain(&format!("source_downs.{i}"), out, spec, 1)?
            } else {
                plain(&format!("source_downs.{i}"), out, spec, u * 2)?
            });
            source_resblocks.push(block(&format!("source_resblocks.{i}"), out, k)?);
        }

        let mut resblocks = Vec::new();
        for i in 0..HiftConfig::UPSAMPLE_RATES.len() {
            let ch = base >> (i + 1);
            for &k in &HiftConfig::RESBLOCK_KERNELS {
                resblocks.push(block(&format!("resblocks.{}", resblocks.len()), ch, k)?);
            }
            let _ = ch;
        }
        let last = base >> HiftConfig::UPSAMPLE_RATES.len();

        let window = gpu.upload(&xabe_dsp::hann_periodic(cfg.n_fft))?;
        Ok(Self {
            conv_pre: wn("conv_pre", base, cfg.in_channels, cfg.look_right + 1)?,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post: wn("conv_post", spec, last, 7)?,
            window,
            cfg,
            gpu,
        })
    }

    /// The geometry this vocoder was bound against.
    pub fn config(&self) -> &HiftConfig {
        &self.cfg
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// One causal convolution, padded on the side its type says.
    fn causal(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        c: &Conv,
        dilation: usize,
        left: bool,
    ) -> Result<(CudaSlice<f32>, usize), CosyError> {
        let pad = causal_pad(c.k, dilation);
        let (l, r) = if left { (pad, 0) } else { (0, pad) };
        Ok(self.gpu.conv1d(
            x,
            &c.w,
            Some(&c.bias),
            c.in_ch,
            t,
            c.out_ch,
            c.k,
            l,
            r,
            dilation,
        )?)
    }

    /// One residual block, in place over its own output.
    fn resblock(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        b: &ResBlock,
    ) -> Result<CudaSlice<f32>, CosyError> {
        let mut h = self.gpu.zeros(b.ch * t)?;
        self.gpu.copy_into(&mut h, x, 0, b.ch * t)?;
        for j in 0..b.convs1.len() {
            let mut xt = self.gpu.zeros(b.ch * t)?;
            self.gpu.copy_into(&mut xt, &h, 0, b.ch * t)?;
            self.gpu.snake(&mut xt, &b.alphas1[j], b.ch, t)?;
            let (xt2, t2) = self.causal(&xt, t, &b.convs1[j], b.dilations[j], true)?;
            debug_assert_eq!(t2, t, "a causal convolution changed the length");

            let mut xt2 = xt2;
            self.gpu.snake(&mut xt2, &b.alphas2[j], b.ch, t)?;
            // The second convolution of each pair is dilation 1 regardless of
            // the first's - upstream hard-codes it, and copying the dilation
            // across is a change that still runs and still sounds like speech.
            let (xt3, t3) = self.causal(&xt2, t, &b.convs2[j], 1, true)?;
            debug_assert_eq!(t3, t);
            self.gpu.add_inplace(&mut h, &xt3, b.ch * t)?;
        }
        Ok(h)
    }

    /// Turns a mel and its excitation into a waveform.
    ///
    /// `source` is the excitation at the output rate - one sample per output
    /// sample - which is what the `source` module produces.
    pub fn decode(
        &self,
        mel: &CudaSlice<f32>,
        frames: usize,
        source: &CudaSlice<f32>,
        source_len: usize,
    ) -> Result<Vec<f32>, CosyError> {
        let mut taps = self.decode_tapped(mel, frames, source, source_len)?;
        Ok(taps.pop().expect("decode always produces a waveform").1)
    }

    /// The same, returning every stage boundary on the host.
    ///
    /// On the public surface for the same reason the language models' taps
    /// are: "the waveform is wrong" localises to nothing, and "stage 1 is
    /// wrong" localises to twelve tensors. The last entry is the waveform.
    pub fn decode_tapped(
        &self,
        mel: &CudaSlice<f32>,
        frames: usize,
        source: &CudaSlice<f32>,
        source_len: usize,
    ) -> Result<Vec<(String, Vec<f32>)>, CosyError> {
        let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
        let cfg = &self.cfg;
        let spec_ch = cfg.n_fft + 2;

        // The excitation is transformed once, at the output rate, and then
        // decimated per stage. Real and imaginary parts are stacked as
        // channels - eighteen of them, which is where `n_fft + 2` comes from.
        let (re, im, sframes) =
            self.gpu
                .stft(source, &self.window, source_len, cfg.n_fft, cfg.hop_len)?;
        let bins = cfg.n_fft / 2 + 1;
        let mut s_stft = self.gpu.zeros(spec_ch * sframes)?;
        self.gpu.copy_into(&mut s_stft, &re, 0, bins * sframes)?;
        self.gpu
            .copy_into(&mut s_stft, &im, bins * sframes, bins * sframes)?;

        // `conv_pre` is the one convolution that pads on the *right*: it is
        // allowed to look ahead by `look_right`, which is what makes the
        // vocoder causal-with-lookahead rather than strictly causal.
        taps.push(("s_stft".into(), self.gpu.download(&s_stft)?));

        let (mut x, mut t) = self.causal(mel, frames, &self.conv_pre, 1, false)?;
        assert_eq!(t, frames, "conv_pre changed the frame count");
        taps.push(("conv_pre".into(), self.gpu.download(&x)?));

        // The excitation's decimation per stage: the reverse cumulative product
        // of the upsampling rates, so the first stage sees it decimated by
        // fifteen and the last sees it untouched.
        const DOWN_RATES: [usize; 3] = [15, 3, 1];
        debug_assert_eq!(DOWN_RATES.len(), self.ups.len());
        for (i, &down) in DOWN_RATES.iter().enumerate() {
            self.gpu
                .leaky_relu(&mut x, self.ups[i].in_ch * t, cfg.lrelu_slope)?;

            // Nearest-neighbour upsample, then a causal convolution. **Not** a
            // transposed convolution: same weight shape, same output length,
            // different function.
            let u = HiftConfig::UPSAMPLE_RATES[i];
            let up_t = t * u;
            let repeated = self.gpu.upsample_nearest(&x, self.ups[i].in_ch, t, u)?;
            let (xu, t2) = self.causal(&repeated, up_t, &self.ups[i], 1, true)?;
            debug_assert_eq!(t2, up_t);
            x = xu;
            t = up_t;

            // A one-sample reflection pad before the last stage, which is what
            // makes the frame count come out at `frames * hop / hop_len + 1`
            // so the inverse transform lands on exactly `frames * hop`
            // samples. Without it the waveform is four samples short.
            if i == self.ups.len() - 1 {
                let ch = self.ups[i].out_ch;
                let mut padded = self.gpu.zeros(ch * (t + 1))?;
                // Reflecting one sample on the left means copying index 1 into
                // index 0 - the edge sample is not repeated.
                for c in 0..ch {
                    self.gpu.copy_into(
                        &mut padded,
                        &self.gpu.copy_range(&x, c * t + 1, 1)?,
                        c * (t + 1),
                        1,
                    )?;
                    self.gpu.copy_into(
                        &mut padded,
                        &self.gpu.copy_range(&x, c * t, t)?,
                        c * (t + 1) + 1,
                        t,
                    )?;
                }
                x = padded;
                t += 1;
            }

            // The excitation, decimated to this stage's rate and shaped.
            let d = &self.source_downs[i];
            let (si, st) = if down == 1 {
                // A one-tap convolution: this stage runs at the excitation's
                // own rate, so there is nothing to decimate.
                self.gpu.conv1d(
                    &s_stft,
                    &d.w,
                    Some(&d.bias),
                    d.in_ch,
                    sframes,
                    d.out_ch,
                    1,
                    0,
                    0,
                    1,
                )?
            } else {
                // A strided convolution with `stride - 1` of left padding,
                // which is upstream's `CausalConv1dDownSample`.
                let pad = down - 1;
                self.gpu.strided_conv1d(
                    &s_stft,
                    &d.w,
                    Some(&d.bias),
                    d.in_ch,
                    sframes,
                    d.out_ch,
                    d.k,
                    down,
                    pad,
                )?
            };
            taps.push((format!("source_down{i}"), self.gpu.download(&si)?));
            let si = self.resblock(&si, st, &self.source_resblocks[i])?;
            debug_assert_eq!(st, t, "the excitation and the stage disagree on length");
            self.gpu.add_inplace(&mut x, &si, self.ups[i].out_ch * t)?;

            // Three residual blocks, summed and averaged. Dropping the average
            // scales the residual by three and is inaudible for one stage and
            // ruinous over three.
            let ch = self.ups[i].out_ch;
            let mut sum = self.gpu.zeros(ch * t)?;
            for j in 0..HiftConfig::RESBLOCK_KERNELS.len() {
                let out = self.resblock(&x, t, &self.resblocks[i * 3 + j])?;
                self.gpu.add_inplace(&mut sum, &out, ch * t)?;
            }
            self.gpu.scale_inplace(
                &mut sum,
                ch * t,
                1.0 / HiftConfig::RESBLOCK_KERNELS.len() as f32,
            )?;
            x = sum;
            taps.push((format!("stage{i}"), self.gpu.download(&x)?));
        }

        // **0.01, not `lrelu_slope`.** Upstream writes `F.leaky_relu(x)` here
        // with no second argument, so this one takes PyTorch's default while
        // every leaky ReLU between the stages takes the configured 0.1. It
        // reads like an oversight upstream and it is what the trained weights
        // were fitted against, so it is reproduced rather than tidied.
        //
        // Passing 0.1 here costs almost nothing until `conv_post`'s output is
        // exponentiated into a magnitude: measured, every stage stayed exact,
        // `conv_post` correlated 0.962, and the waveform came out with ten
        // times the energy it should have.
        self.gpu
            .leaky_relu(&mut x, self.conv_post.in_ch * t, Self::POST_LRELU_SLOPE)?;
        let (x, t) = self.causal(&x, t, &self.conv_post, 1, true)?;
        taps.push(("conv_post".into(), self.gpu.download(&x)?));

        // The head: nine log-magnitudes and nine phases, read as a spectrum.
        // The magnitude is clipped at 1e2 before the transform, which is
        // upstream's guard against `exp` of a large activation - without it one
        // frame can carry the whole waveform.
        let host = self.gpu.download(&x)?;
        let (mut re, mut im) = (vec![0.0f32; bins * t], vec![0.0f32; bins * t]);
        for b in 0..bins {
            for f in 0..t {
                let mag = host[b * t + f].exp().min(1e2);
                let phase = host[(bins + b) * t + f].sin();
                re[b * t + f] = mag * phase.cos();
                im[b * t + f] = mag * phase.sin();
            }
        }

        let wave = self.gpu.istft(
            &self.gpu.upload(&re)?,
            &self.gpu.upload(&im)?,
            &self.window,
            t,
            cfg.n_fft,
            cfg.hop_len,
        )?;
        let mut out = self.gpu.download(&wave)?;
        for v in &mut out {
            *v = v.clamp(-cfg.audio_limit, cfg.audio_limit);
        }
        taps.push(("wav".into(), out));
        Ok(taps)
    }
}
