//! Rotary position embedding, and the two kernels Llama needs beside it.

/// Root-mean-square normalisation, over rows of `dim`.
///
/// Not layer normalisation: there is no mean subtraction and no bias, only a
/// scale. Substituting one for the other passes every shape check and shifts
/// every activation by the row's mean, which on a residual stream is not
/// small.
///
/// The reference computes the variance in f32 whatever the model's dtype is,
/// and this is f32 throughout, so the two agree by construction.
///
/// # Panics
///
/// If `x` is not `rows * dim` long, or `weight` is not `dim`.
pub fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim, "rms_norm wants [rows, dim]");
    assert_eq!(weight.len(), dim, "rms_norm wants one scale per column");
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let scale = (mean_sq + eps).sqrt().recip();
        for (i, (o, v)) in out[r * dim..(r + 1) * dim].iter_mut().zip(row).enumerate() {
            *o = v * scale * weight[i];
        }
    }
    out
}

/// The sigmoid linear unit, `x * sigmoid(x)`, in place.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// `a = silu(a) * b`, which is the SwiGLU feed-forward's gate.
///
/// # Panics
///
/// If the two are not the same length.
pub fn silu_mul(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "silu_mul wants matching lengths");
    for (v, &g) in a.iter_mut().zip(b) {
        *v = *v * (1.0 / (1.0 + (-*v).exp())) * g;
    }
}

/// Rotary position embedding, in place, over `[t, heads * head_dim]`.
///
/// `first` is the absolute position of row zero, so a decode step past a KV
/// cache rotates by where the token really is rather than by zero.
///
/// # The halves convention
///
/// 🤗 pairs dimension `i` with `i + head_dim/2`, not `2i` with `2i+1`. The two
/// conventions are a permutation of each other and both are called "RoPE";
/// picking the wrong one produces a model that is coherent for four or five
/// tokens and then drifts, which is the hardest possible thing to debug. The
/// weights were trained with this one.
///
/// # Panics
///
/// If `x` is not `t * heads * head_dim` long, or `head_dim` is odd.
pub fn rope(x: &mut [f32], t: usize, heads: usize, head_dim: usize, theta: f32, first: usize) {
    rope_scaled(x, t, heads, head_dim, theta, first, None);
}

/// RoPE with an optional per-dimension frequency divisor.
///
/// Llama-2 has none and passes `None`. Llama-3.1 carries one as
/// `rope_freqs.weight`: `head_dim / 2` factors that divide the inverse
/// frequency of each rotating pair, stretching the low-frequency dimensions so
/// a model trained at 8k reaches 128k. On Breeze2 the factors run 1.0 for the
/// first 29 pairs, rise through 1.21, 1.55, 2.03, 2.69, 3.68, 5.26, and sit at
/// 8.0 for the rest.
///
/// Passing `None` for a checkpoint that ships the tensor is the failure this
/// exists to prevent, and it has no shape to catch it: the model stays fluent
/// for a sentence and drifts once the context grows past what the unscaled
/// frequencies cover.
pub fn rope_scaled(
    x: &mut [f32],
    t: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
    first: usize,
    freq_div: Option<&[f32]>,
) {
    assert_eq!(
        x.len(),
        t * heads * head_dim,
        "rope wants [t, heads*head_dim]"
    );
    assert!(head_dim.is_multiple_of(2), "head_dim {head_dim} is odd");
    let half = head_dim / 2;
    if let Some(d) = freq_div {
        assert_eq!(d.len(), half, "one divisor per rotating pair");
    }
    for p in 0..t {
        for h in 0..heads {
            let base = (p * heads + h) * head_dim;
            for i in 0..half {
                // inv_freq[i] = theta ** (-2i / head_dim), in f32.
                //
                // f64 here would be *more* accurate than the reference and so
                // less useful: 🤗 computes the frequencies and the angle in
                // f32, and at position 4095 an f32 angle is only good to about
                // 2.4e-4 radians because that is the spacing of f32 up there.
                // A twin that is better than what it is a twin for turns a
                // faithful implementation into a failing test.
                let mut inv = theta.powf(-2.0 * i as f32 / head_dim as f32);
                if let Some(d) = freq_div {
                    inv /= d[i];
                }
                let angle = (first + p) as f32 * inv;
                let (sin, cos) = (angle.sin(), angle.cos());
                let (a, b) = (x[base + i], x[base + i + half]);
                x[base + i] = a * cos - b * sin;
                x[base + i + half] = b * cos + a * sin;
            }
        }
    }
}
