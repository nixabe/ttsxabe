//! Layer normalisation.

/// Normalises each of `t` rows of `c` values, then scales and shifts.
///
/// The variance is the biased one - divided by `c`, not `c - 1` - because that
/// is what `torch.nn.LayerNorm` uses. Getting this wrong produces an error that
/// shrinks as the channel count grows, which is exactly the kind of small
/// consistent bias that survives a loose tolerance and ruins a waveform.
pub fn layer_norm(
    x: &[f32],
    t: usize,
    c: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * c);
    debug_assert_eq!(weight.len(), c);
    debug_assert_eq!(bias.len(), c);

    let mut out = vec![0.0; x.len()];
    for row in 0..t {
        let src = &x[row * c..(row + 1) * c];
        let mean = src.iter().sum::<f32>() / c as f32;
        let var = src.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            out[row * c + i] = (src[i] - mean) * inv * weight[i] + bias[i];
        }
    }
    out
}

/// The residual sum and the normalisation of it, the reference for
/// `Gpu::layer_norm_add`.
///
/// `h` becomes `h + res` and the return is [`layer_norm`] of that. It exists
/// as its own function rather than as a composition because the kernel it
/// mirrors folds the two together, and a differential test compares against
/// what the kernel is meant to compute rather than against what it happens to
/// be equal to.
pub fn layer_norm_add(
    h: &mut [f32],
    res: &[f32],
    t: usize,
    c: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(h.len(), t * c);
    debug_assert_eq!(res.len(), t * c);
    for (v, r) in h.iter_mut().zip(res) {
        *v += r;
    }
    layer_norm(h, t, c, weight, bias, eps)
}
