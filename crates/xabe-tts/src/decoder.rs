//! The HiFi-GAN decoder: latents to a waveform.
//!
//! Four transposed convolutions upsample by 8, 8, 2 and 2 - 256 samples per
//! frame in total - and after each one a multi-receptive-field fusion averages
//! three residual blocks with kernels 3, 7 and 11. A final width-7 convolution
//! collapses to one channel and `tanh` bounds it to [-1, 1].
//!
//! # The last leaky ReLU has a different slope
//!
//! Every leaky ReLU in this decoder uses `config.leaky_relu_slope`, which is
//! 0.1 - except the last one, immediately before `conv_post`, where the
//! reference writes
//!
//! ```python
//! hidden_states = nn.functional.leaky_relu(hidden_states)
//! ```
//!
//! with no slope argument, so it takes PyTorch's default of **0.01**. Whether
//! that is deliberate or an oversight upstream does not matter: it is what
//! produced the weights, so it is what correct means. Using 0.1 there changes
//! only the negative half of one activation, which is audible as a slight
//! roughness and visible as nothing at all in a shape check.
//!
//! # The fusion averages, it does not sum
//!
//! The three resblock outputs are added and then divided by three. Dropping the
//! division makes the waveform three times too loud, which `tanh` then clips
//! into distortion rather than into an obvious error.

use std::borrow::Cow;

use xabe_dsp::{conv1d, fuse_weight_norm, leaky_relu, transposed_conv1d};
use xabe_vits::{Decoder, MaybeWn, ResBlock, VitsConfig};

/// PyTorch's default negative slope, which the final activation inherits by
/// being called without an argument.
const TORCH_DEFAULT_SLOPE: f32 = 0.01;

/// The kernel to convolve with, fusing the weight norm if one is stored.
///
/// The 🤗 export of `mms-tts-nan` fused the decoder's parameterisation before
/// saving and the Coqui checkpoint does not, so the same convolution arrives
/// two ways. Fusing here rather than at load keeps `xabe-vits` free of
/// arithmetic, and the borrowed case still copies nothing.
///
/// The row count comes from the magnitude's own length rather than from
/// `out_ch`. That is not a shortcut: weight norm normalises over every axis but
/// the first, and a *transposed* convolution stores `[in, out, k]` - so its
/// magnitude has one entry per input channel, and reading `out_ch` here would
/// divide 512 rows by 256 norms.
fn kernel<'a>(c: &MaybeWn<'a>) -> Cow<'a, [f32]> {
    match c {
        MaybeWn::Fused(c) => Cow::Borrowed(c.weight),
        MaybeWn::Normalised(w) => {
            let rows = w.weight_g.len();
            let cols = w.weight_v.len() / (rows * w.k);
            Cow::Owned(fuse_weight_norm(w.weight_v, w.weight_g, rows, cols, w.k))
        }
    }
}

/// Synthesises a waveform from the flow's output.
///
/// `z` is `[flow_size, frames]`; the result is `[frames * hop_length]` samples
/// in [-1, 1].
pub fn decoder(z: &[f32], w: &Decoder<'_>, cfg: &VitsConfig) -> Vec<f32> {
    let frames = z.len() / cfg.flow_size;
    let per_stage = cfg.resblocks_per_stage();

    let mut h = conv1d(
        z,
        cfg.flow_size,
        frames,
        w.conv_pre.weight,
        w.conv_pre.bias,
        cfg.upsample_initial_channel,
        w.conv_pre.k,
        w.conv_pre.k / 2,
        w.conv_pre.k / 2,
        1,
    );
    let mut t = frames;
    let mut ch = cfg.upsample_initial_channel;

    for (stage, up) in w.upsampler.iter().enumerate() {
        leaky_relu(&mut h, cfg.leaky_relu_slope);

        let stride = cfg.upsample_rates[stage];
        let k = up.k();
        let pad = (k - stride) / 2;
        let weight = kernel(up);
        h = transposed_conv1d(&h, ch, t, &weight, up.bias(), up.out_ch(), k, stride, pad);
        t = (t - 1) * stride + k - 2 * pad;
        ch = up.out_ch();

        // Multi-receptive-field fusion: the same input through three different
        // kernel sizes, averaged.
        let mut fused = vec![0.0; ch * t];
        for j in 0..per_stage {
            let block = &w.resblocks[stage * per_stage + j];
            let out = resblock(&h, ch, t, block, &cfg.resblock_dilation_sizes[j], cfg);
            for (dst, src) in fused.iter_mut().zip(&out) {
                *dst += src;
            }
        }
        let scale = 1.0 / per_stage as f32;
        for v in &mut fused {
            *v *= scale;
        }
        h = fused;
    }

    // No slope argument in the reference, so this one is 0.01 rather than 0.1.
    leaky_relu(&mut h, TORCH_DEFAULT_SLOPE);

    let out = conv1d(
        &h,
        ch,
        t,
        w.conv_post.weight,
        // `conv_post` is the one convolution in the checkpoint with no bias.
        w.conv_post.bias,
        1,
        w.conv_post.k,
        w.conv_post.k / 2,
        w.conv_post.k / 2,
        1,
    );
    out.iter().map(|v| v.tanh()).collect()
}

/// One multi-receptive-field residual block.
fn resblock(
    x: &[f32],
    ch: usize,
    t: usize,
    block: &ResBlock<'_>,
    dilations: &[usize],
    cfg: &VitsConfig,
) -> Vec<f32> {
    let mut h = x.to_vec();
    for (i, dilation) in dilations.iter().copied().enumerate() {
        let residual = h.clone();

        let c1 = &block.convs1[i];
        leaky_relu(&mut h, cfg.leaky_relu_slope);
        let k1 = c1.k();
        let pad = (k1 * dilation - dilation) / 2;
        h = conv1d(
            &h,
            ch,
            t,
            &kernel(c1),
            c1.bias(),
            ch,
            k1,
            pad,
            pad,
            dilation,
        );

        let c2 = &block.convs2[i];
        leaky_relu(&mut h, cfg.leaky_relu_slope);
        // The second convolution of each pair is never dilated, whatever the
        // first one's dilation was.
        let k2 = c2.k();
        let pad = (k2 - 1) / 2;
        h = conv1d(&h, ch, t, &kernel(c2), c2.bias(), ch, k2, pad, pad, 1);

        for (dst, src) in h.iter_mut().zip(&residual) {
            *dst += src;
        }
    }
    h
}
