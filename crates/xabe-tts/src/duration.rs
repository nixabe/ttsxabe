//! The stochastic duration predictor.
//!
//! Given the text encoder's hidden states and a noise draw, produces one log
//! duration per symbol. It is a normalising flow run backwards: the noise is
//! the latent, and the coupling blocks transform it into durations conditioned
//! on the text.
//!
//! # The flow list is not run in reverse order
//!
//! This is the trap in this file, and it is a quiet one. The predictor holds an
//! elementwise-affine block followed by four convolutional flows. Inverting a
//! flow means running the blocks backwards, so one expects
//! `[conv4, conv3, conv2, conv1, affine]`. The reference instead computes
//!
//! ```python
//! flows = list(reversed(self.flows))
//! flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
//! ```
//!
//! which is `[conv4, conv3, conv2, affine]` - four blocks, and the one dropped
//! is `conv1`, the *first* convolutional flow, not the last. Running all five
//! produces durations that are wrong but entirely plausible: the audio is
//! speech, at the wrong pace, with no error anywhere. There is nothing to
//! notice unless you are diffing against the reference.
//!
//! # Noise is passed in, not drawn
//!
//! Sampling is the caller's business. Passing the draw in is what makes this
//! testable against a capture at all - two RNGs agreeing on a seed across
//! languages is not something to assume - and it is also what will let the CLI
//! offer a reproducible `--seed`.

use xabe_dsp::{conv1d, depthwise_conv1d, gelu, layer_norm, spline_inverse, transpose};
use xabe_vits::{DdsConv, DurationFlow, DurationPredictor, VitsConfig};

/// Epsilon in the depthwise stack's layer norms.
///
/// The reference constructs these as bare `nn.LayerNorm(channels)`, so they use
/// PyTorch's default rather than the config's `layer_norm_eps`. The two happen
/// to be equal for this checkpoint; they are not the same setting.
const DDS_NORM_EPS: f32 = 1e-5;

/// Predicts one log duration per symbol.
///
/// `hidden` is the text encoder's output in `[t, hidden_size]`; `noise` is a
/// `[2, t]` standard normal draw. Returns `[t]`.
pub fn duration_predictor(
    hidden: &[f32],
    noise: &[f32],
    w: &DurationPredictor<'_>,
    cfg: &VitsConfig,
) -> Vec<f32> {
    let ch = cfg.hidden_size;
    let t = hidden.len() / ch;
    debug_assert_eq!(noise.len(), 2 * t);

    // The conditioning path: the text, projected and mixed along time. It is
    // computed once and handed to every coupling block.
    let cond = transpose(hidden, t, ch);
    let cond = conv1d(
        &cond,
        ch,
        t,
        w.conv_pre.weight,
        w.conv_pre.bias,
        ch,
        1,
        0,
        0,
        1,
    );
    let cond = dds(&cond, ch, t, &w.conv_dds, None, cfg);
    let cond = conv1d(
        &cond,
        ch,
        t,
        w.conv_proj.weight,
        w.conv_proj.bias,
        ch,
        1,
        0,
        0,
        1,
    );

    let mut z: Vec<f32> = noise.iter().map(|v| v * cfg.noise_scale_duration).collect();

    for &i in &reverse_order(w.flows.len()) {
        // The flip is *before* each block, and it is a channel swap: the two
        // channels of `z` trade places so that each block conditions on the
        // half the previous one transformed.
        let (a, b) = z.split_at(t);
        z = b.iter().chain(a).copied().collect();
        apply_reverse(&mut z, t, &w.flows[i], &cond, cfg);
    }

    z.truncate(t);
    z
}

/// The block order the reference's reverse pass actually uses.
///
/// `reversed(flows)` then `[:-2] + [flows[-1]]`, which drops the second entry
/// of the original list. Written as index arithmetic so the omission is
/// visible rather than buried in slice syntax.
fn reverse_order(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (2..n).rev().collect();
    order.push(0);
    order
}

/// Applies one coupling block in the inverse direction, in place.
fn apply_reverse(z: &mut [f32], t: usize, flow: &DurationFlow<'_>, cond: &[f32], cfg: &VitsConfig) {
    match flow {
        DurationFlow::Affine {
            log_scale,
            translate,
        } => {
            for c in 0..2 {
                let inv = (-log_scale[c]).exp();
                for v in &mut z[c * t..(c + 1) * t] {
                    *v = (*v - translate[c]) * inv;
                }
            }
        }
        DurationFlow::Spline {
            conv_pre,
            conv_dds,
            conv_proj,
        } => {
            let ch = cfg.hidden_size;
            let bins = cfg.duration_predictor_flow_bins;
            let half = cfg.depth_separable_channels / 2;
            // The network emits `bins` widths, `bins` heights and `bins - 1`
            // interior derivatives. The two boundary derivatives are fixed, not
            // predicted - see `xabe_dsp::spline_inverse`.
            let per = bins * 3 - 1;

            let (first, second) = z.split_at_mut(half * t);
            let h = conv1d(
                first,
                half,
                t,
                conv_pre.weight,
                conv_pre.bias,
                ch,
                1,
                0,
                0,
                1,
            );
            let h = dds(&h, ch, t, conv_dds, Some(cond), cfg);
            let h = conv1d(
                &h,
                ch,
                t,
                conv_proj.weight,
                conv_proj.bias,
                half * per,
                1,
                0,
                0,
                1,
            );

            // The widths and heights are divided by sqrt(filter_channels) but
            // the derivatives are not. An easy line to apply uniformly, and
            // doing so shifts every knot slope.
            let scale = 1.0 / (ch as f32).sqrt();
            let mut widths = vec![0.0; bins];
            let mut heights = vec![0.0; bins];
            let mut derivs = vec![0.0; bins - 1];

            for c in 0..half {
                for pos in 0..t {
                    let row = |k: usize| h[(c * per + k) * t + pos];
                    for k in 0..bins {
                        widths[k] = row(k) * scale;
                        heights[k] = row(bins + k) * scale;
                    }
                    for (k, d) in derivs.iter_mut().enumerate() {
                        *d = row(2 * bins + k);
                    }
                    let idx = c * t + pos;
                    second[idx] = spline_inverse(
                        second[idx],
                        &widths,
                        &heights,
                        &derivs,
                        cfg.duration_predictor_tail_bound,
                    );
                }
            }
        }
    }
}

/// The dilated depthwise-separable convolution stack.
///
/// Each layer is a depthwise dilated convolution, a norm over channels, GELU, a
/// pointwise convolution, a second norm, a second GELU, and a residual add. The
/// dilation grows as `kernel^i`, which is what gives three layers a receptive
/// field of 27 frames.
fn dds(
    x: &[f32],
    ch: usize,
    t: usize,
    w: &DdsConv<'_>,
    cond: Option<&[f32]>,
    cfg: &VitsConfig,
) -> Vec<f32> {
    let k = cfg.duration_predictor_kernel_size;
    let mut inputs: Vec<f32> = match cond {
        Some(c) => x.iter().zip(c).map(|(a, b)| a + b).collect(),
        None => x.to_vec(),
    };

    for i in 0..cfg.depth_separable_num_layers {
        let dilation = k.pow(i as u32);
        // The reference writes `(kernel * dilation - dilation) // 2`, which is
        // the same-padding for a dilated kernel, symmetric on both sides.
        let pad = (k * dilation - dilation) / 2;

        let h = depthwise_conv1d(
            &inputs,
            ch,
            t,
            w.dilated[i].weight,
            w.dilated[i].bias,
            k,
            pad,
            pad,
            dilation,
        );
        let mut h = norm_channels(&h, ch, t, &w.norms_1[i]);
        gelu(&mut h);
        let h = conv1d(
            &h,
            ch,
            t,
            w.pointwise[i].weight,
            w.pointwise[i].bias,
            ch,
            1,
            0,
            0,
            1,
        );
        let mut h = norm_channels(&h, ch, t, &w.norms_2[i]);
        gelu(&mut h);

        for (dst, src) in inputs.iter_mut().zip(&h) {
            *dst += src;
        }
    }
    inputs
}

/// Layer-norms across channels for each time step, in convolution layout.
///
/// The reference transposes into `[t, channels]`, norms, and transposes back.
/// The transposes are what make this a norm over *channels* rather than over
/// time, so they are done here too rather than being optimised into the index
/// arithmetic - `xabe-dsp` is the crate that chooses readable.
fn norm_channels(x: &[f32], ch: usize, t: usize, n: &xabe_vits::Norm<'_>) -> Vec<f32> {
    let tc = transpose(x, ch, t);
    let normed = layer_norm(&tc, t, ch, n.weight, n.bias, DDS_NORM_EPS);
    transpose(&normed, t, ch)
}
