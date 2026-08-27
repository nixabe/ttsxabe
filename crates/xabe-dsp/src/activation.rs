//! Pointwise activations and the softmax.

/// Rectified linear unit, in place.
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}

/// Leaky rectified linear unit, in place. The decoder uses slope 0.1.
pub fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// Softmax over each of `rows` rows of `cols` values, in place.
///
/// Subtracts the row maximum first. That is not an optimisation: the attention
/// logits here are unbounded, and `exp` of a large positive logit is infinity,
/// which turns the whole row into NaN.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(x.len(), rows * cols);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Snake in place: `x + sin^2(a*x)/a`, one alpha per channel.
///
/// HiFTNet's activation. `x` is `[channels, t]`, `alpha` is one value per
/// channel, and the guard sits on the *divisor* exactly where upstream puts
/// it - an alpha trained to zero makes this the identity plus a very large
/// term, which is the reference's behaviour and not something to improve
/// quietly.
pub fn snake(x: &mut [f32], alpha: &[f32], ch: usize, t: usize) {
    debug_assert_eq!(x.len(), ch * t);
    debug_assert_eq!(alpha.len(), ch);
    for (c, a) in alpha.iter().enumerate() {
        for v in &mut x[c * t..(c + 1) * t] {
            let s = (*v * a).sin();
            *v += s * s / (a + 1e-9);
        }
    }
}
