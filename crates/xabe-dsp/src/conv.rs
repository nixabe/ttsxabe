//! One-dimensional convolution.

/// Convolves `[in_c, t]` input with `[out_c, in_c, k]` weights.
///
/// Padding is explicit and asymmetric because the reference's is: for an even
/// kernel it pads `(k-1)/2` left and `k/2` right, which are different numbers.
/// Dilation is here because the duration predictor's depthwise-separable stack
/// uses it; everything else passes 1.
///
/// This is cross-correlation, not true convolution - the kernel is not flipped.
/// That is what every deep-learning framework calls "conv", and flipping it
/// would produce audio that is wrong in a way no shape check would catch.
// Shapes are arguments, not types - see the crate essay. Grouping them into a
// descriptor struct would satisfy the lint and make every call site say less
// about what it is actually doing.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &[f32],
    in_c: usize,
    t: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out_c: usize,
    k: usize,
    pad_left: usize,
    pad_right: usize,
    dilation: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_c * t);
    debug_assert_eq!(w.len(), out_c * in_c * k);

    let span = dilation * (k - 1) + 1;
    let out_t = (t + pad_left + pad_right).saturating_sub(span) + 1;
    let mut out = vec![0.0; out_c * out_t];

    for o in 0..out_c {
        let b = bias.map_or(0.0, |b| b[o]);
        for p in 0..out_t {
            let mut acc = b;
            for i in 0..in_c {
                for tap in 0..k {
                    // Position in the *unpadded* input. Negative or past the
                    // end means the pad, which is zero, so it contributes
                    // nothing and is skipped rather than materialised.
                    let pos = (p + tap * dilation) as isize - pad_left as isize;
                    if pos < 0 || pos as usize >= t {
                        continue;
                    }
                    acc += x[i * t + pos as usize] * w[(o * in_c + i) * k + tap];
                }
            }
            out[o * out_t + p] = acc;
        }
    }
    out
}

/// The reference's `same`-style padding for a kernel of size `k`.
///
/// Asymmetric for even kernels, which is why it is a function rather than a
/// number computed at each call site.
pub fn same_padding(k: usize) -> (usize, usize) {
    ((k - 1) / 2, k / 2)
}

/// Depthwise convolution: each channel is convolved with its own kernel.
///
/// The weights are `[channels, 1, k]` - PyTorch stores a grouped convolution as
/// `[out_channels, in_channels / groups, k]`, and with `groups == channels`
/// that middle dimension is 1. Nothing about the shape distinguishes this from
/// an ordinary convolution with one input channel, which is why it is a
/// separate function rather than a flag.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv1d(
    x: &[f32],
    ch: usize,
    t: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    pad_left: usize,
    pad_right: usize,
    dilation: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), ch * t);
    debug_assert_eq!(w.len(), ch * k);

    let span = dilation * (k - 1) + 1;
    let out_t = (t + pad_left + pad_right).saturating_sub(span) + 1;
    let mut out = vec![0.0; ch * out_t];

    for c in 0..ch {
        let b = bias.map_or(0.0, |b| b[c]);
        for p in 0..out_t {
            let mut acc = b;
            for tap in 0..k {
                let pos = (p + tap * dilation) as isize - pad_left as isize;
                if pos < 0 || pos as usize >= t {
                    continue;
                }
                acc += x[c * t + pos as usize] * w[c * k + tap];
            }
            out[c * out_t + p] = acc;
        }
    }
    out
}

/// Transposed convolution, the decoder's upsampler.
///
/// The weights are `[in_channels, out_channels, k]` - the reverse of every
/// other convolution here - while the bias stays per *output* channel. That
/// asymmetry is PyTorch's, and it is why this cannot share [`conv1d`].
///
/// Written as scatter rather than gather: each input position contributes to
/// `k` output positions, at stride `stride`. The gather form exists and is
/// faster, but it needs the output-to-input index inverted, which is where the
/// off-by-ones live.
#[allow(clippy::too_many_arguments)]
pub fn transposed_conv1d(
    x: &[f32],
    in_ch: usize,
    t: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out_ch: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_ch * t);
    debug_assert_eq!(w.len(), in_ch * out_ch * k);

    let out_t = (t - 1) * stride + k - 2 * padding;
    let mut out = vec![0.0; out_ch * out_t];
    if let Some(b) = bias {
        for o in 0..out_ch {
            out[o * out_t..(o + 1) * out_t].fill(b[o]);
        }
    }

    for i in 0..in_ch {
        for n in 0..t {
            let v = x[i * t + n];
            if v == 0.0 {
                continue;
            }
            for tap in 0..k {
                let pos = (n * stride + tap) as isize - padding as isize;
                if pos < 0 || pos as usize >= out_t {
                    continue;
                }
                for o in 0..out_ch {
                    out[o * out_t + pos as usize] += v * w[(i * out_ch + o) * k + tap];
                }
            }
        }
    }
    out
}
