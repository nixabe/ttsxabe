//! The prior flow, run backwards.
//!
//! Four residual coupling blocks, each transforming half the channels using a
//! WaveNet conditioned on the other half. Inverting them is cheap because the
//! coupling is *mean-only*: the reference computes a `log_stddev` and it is
//! identically zero, so the forward step is `second += mean` and the inverse is
//! `second -= mean`. Nothing needs to be solved.
//!
//! # What is easy to get wrong
//!
//! **The flip is before the block, not after.** Forward, the reference flips
//! *after* each coupling; backward, it flips *before*. Getting the order right
//! matters because the flip is what makes successive blocks transform
//! different channels - do it on the wrong side and every block transforms the
//! same half, which is still invertible and still produces audio.
//!
//! **The flip reverses the whole channel axis, not the two halves.**
//! `torch.flip(x, [1])` on `[1, 192, T]` gives channel order 191, 190, ... 0.
//! Reading it as "swap the halves" is right at two channels - which is where
//! the duration predictor uses it - and wrong at 192. This cost a debugging
//! round: the duration predictor passed, and the flow came out with 30,134 of
//! its 30,144 values wrong.
//!
//! **The WaveNet's kernels are weight-normalised.** These are the only tensors
//! in the inference path stored as `weight_g` and `weight_v` rather than a
//! plain `weight`; see `docs/MODEL.md`.
//!
//! **The last WaveNet layer is narrower.** Every layer but the last emits
//! `2 * hidden` channels - half fed back as a residual, half accumulated into
//! the output - while the last emits only `hidden`, because there is nothing
//! left to feed back. The reference's comment is "last one is not necessary".

use xabe_dsp::{conv1d, flip_channels, fuse_weight_norm, gated_activation};
use xabe_vits::{FlowBlock, VitsConfig};

/// Runs the prior flow in reverse: `z_p` to `z`, both `[flow_size, frames]`.
pub fn flow_reverse(z_p: &[f32], flows: &[FlowBlock<'_>], cfg: &VitsConfig) -> Vec<f32> {
    let ch = cfg.flow_size;
    let frames = z_p.len() / ch;
    let half = cfg.flow_half();

    let mut x = z_p.to_vec();
    for block in flows.iter().rev() {
        // Flip first, and note it reverses the *whole* channel axis - it is
        // not a swap of the two halves. The two are the same operation only at
        // two channels, which is where the duration predictor uses it, so a
        // half-swap passes there and is wrong here.
        x = flip_channels(&x, ch, frames);

        let (first, second) = x.split_at_mut(half * frames);
        let h = conv1d(
            first,
            half,
            frames,
            block.conv_pre.weight,
            block.conv_pre.bias,
            cfg.hidden_size,
            1,
            0,
            0,
            1,
        );
        let h = wavenet(&h, frames, block, cfg);
        let mean = conv1d(
            &h,
            cfg.hidden_size,
            frames,
            block.conv_post.weight,
            block.conv_post.bias,
            half,
            1,
            0,
            0,
            1,
        );

        // Mean-only coupling: the reference's log_stddev is zeros, so the
        // inverse is a subtraction and not a division.
        for (dst, m) in second.iter_mut().zip(&mean) {
            *dst -= m;
        }
    }
    x
}

/// The WaveNet stack inside one coupling block.
fn wavenet(x: &[f32], frames: usize, block: &FlowBlock<'_>, cfg: &VitsConfig) -> Vec<f32> {
    let ch = cfg.hidden_size;
    let k = cfg.wavenet_kernel_size;
    let n = block.wavenet.len();

    let mut inputs = x.to_vec();
    let mut outputs = vec![0.0; ch * frames];

    for (i, layer) in block.wavenet.iter().enumerate() {
        let dilation = cfg.wavenet_dilation_rate.pow(i as u32);
        let pad = (k * dilation - dilation) / 2;

        let w = fuse_weight_norm(
            layer.in_layer.weight_v,
            layer.in_layer.weight_g,
            2 * ch,
            ch,
            k,
        );
        let h = conv1d(
            &inputs,
            ch,
            frames,
            &w,
            Some(layer.in_layer.bias),
            2 * ch,
            k,
            pad,
            pad,
            dilation,
        );
        // No speaker conditioning: this checkpoint is single-speaker, so the
        // reference adds a zero tensor here.
        let acts = gated_activation(&h, ch, frames);

        let out_ch = layer.res_skip.out_ch;
        let w = fuse_weight_norm(
            layer.res_skip.weight_v,
            layer.res_skip.weight_g,
            out_ch,
            ch,
            1,
        );
        let res_skip = conv1d(
            &acts,
            ch,
            frames,
            &w,
            Some(layer.res_skip.bias),
            out_ch,
            1,
            0,
            0,
            1,
        );

        if i < n - 1 {
            for c in 0..ch * frames {
                inputs[c] += res_skip[c];
                outputs[c] += res_skip[ch * frames + c];
            }
        } else {
            for c in 0..ch * frames {
                outputs[c] += res_skip[c];
            }
        }
    }
    outputs
}
