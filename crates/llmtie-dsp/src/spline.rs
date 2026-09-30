//! Monotonic rational-quadratic splines, inverted.
//!
//! The duration predictor's coupling blocks transform one channel through a
//! learned monotonic function of the other. That function is a piecewise
//! rational quadratic on `[-B, B]` and the identity outside it, parameterised
//! by per-bin widths, heights and knot derivatives that the network predicts.
//!
//! Only the inverse is here. Synthesis runs the flow backwards and the forward
//! direction exists solely for training, which this project does not do.
//!
//! # The two things that are easy to get wrong
//!
//! **The derivative array is padded to `bins + 1`, and the padding is not
//! zero.** The network predicts `3 * bins - 1` numbers: `bins` widths, `bins`
//! heights and `bins - 1` interior knot derivatives. The two boundary
//! derivatives are then *fixed* at a constant chosen so that
//! `min_derivative + softplus(constant)` is exactly 1 - which is what makes the
//! spline join the identity tail smoothly. Reading `3 * bins - 1` and using it
//! directly gives an index-out-of-range at best, and a discontinuous transform
//! at worst.
//!
//! **The bin search uses the output axis when inverting.** Going forward the
//! input lies on the width axis; going backward it lies on the height axis.
//! Searching the wrong one produces a plausible number from the wrong bin.

/// Smallest permitted bin width, as a fraction of the interval.
const MIN_BIN_WIDTH: f64 = 1e-3;

/// Smallest permitted bin height.
const MIN_BIN_HEIGHT: f64 = 1e-3;

/// Floor on the knot derivatives, keeping the transform invertible.
const MIN_DERIVATIVE: f64 = 1e-3;

/// Inverts the spline for one value.
///
/// `widths`, `heights` and `derivs` are the network's raw outputs for this
/// position: `bins`, `bins` and `bins - 1` values respectively. `tail` is the
/// bound beyond which the transform is the identity.
pub fn spline_inverse(y: f32, widths: &[f32], heights: &[f32], derivs: &[f32], tail: f32) -> f32 {
    let bins = widths.len();
    debug_assert_eq!(heights.len(), bins);
    debug_assert_eq!(derivs.len(), bins - 1);

    let y = f64::from(y);
    let bound = f64::from(tail);
    if y < -bound || y > bound {
        return y as f32;
    }

    // Widths and heights are softmaxed, floored, then scaled to span the
    // interval. The cumulative arrays have `bins + 1` entries: the knots.
    let cumwidths = knots(widths, bound, MIN_BIN_WIDTH);
    let cumheights = knots(heights, bound, MIN_BIN_HEIGHT);

    // The boundary derivatives are fixed so that the spline meets the identity
    // tail with slope 1; `softplus(CONSTANT) == 1 - MIN_DERIVATIVE` by
    // construction.
    let constant = ((1.0 - MIN_DERIVATIVE).exp() - 1.0).ln();
    let mut d = Vec::with_capacity(bins + 1);
    d.push(MIN_DERIVATIVE + softplus(constant));
    d.extend(
        derivs
            .iter()
            .map(|&v| MIN_DERIVATIVE + softplus(f64::from(v))),
    );
    d.push(MIN_DERIVATIVE + softplus(constant));

    // Inverting, so the search is on the *height* axis. The reference nudges
    // the last knot by 1e-6 so that a value sitting exactly on the upper bound
    // lands in the last bin rather than one past it.
    let idx = bin_index(y, &cumheights);

    let x_lo = cumwidths[idx];
    let w = cumwidths[idx + 1] - cumwidths[idx];
    let y_lo = cumheights[idx];
    let h = cumheights[idx + 1] - cumheights[idx];
    let delta = h / w;
    let d_lo = d[idx];
    let d_hi = d[idx + 1];

    // Solve the rational quadratic for theta. Written as `2c / (-b - sqrt(D))`
    // rather than the textbook `(-b + sqrt(D)) / 2a`: the two are algebraically
    // equal, and this form is the numerically stable one when `b` is large and
    // positive, which is exactly where the textbook form cancels catastrophically.
    let dy = y - y_lo;
    let common = dy * (d_lo + d_hi - 2.0 * delta);
    let a = h * (delta - d_lo) + common;
    let b = h * d_lo - common;
    let c = -delta * dy;

    let discriminant = b * b - 4.0 * a * c;
    debug_assert!(discriminant >= 0.0, "the spline is not monotonic");
    let theta = (2.0 * c) / (-b - discriminant.max(0.0).sqrt());

    (theta * w + x_lo) as f32
}

/// Builds the `bins + 1` knot positions along one axis.
fn knots(raw: &[f32], bound: f64, floor: f64) -> Vec<f64> {
    let bins = raw.len();
    let mut p: Vec<f64> = raw.iter().map(|&v| f64::from(v)).collect();
    softmax(&mut p);
    for v in &mut p {
        *v = floor + (1.0 - floor * bins as f64) * *v;
    }

    let mut cum = Vec::with_capacity(bins + 1);
    cum.push(-bound);
    let mut acc = 0.0;
    for v in &p {
        acc += v;
        cum.push(2.0 * bound * acc - bound);
    }
    // The reference overwrites both ends rather than trusting the sum to land
    // exactly on them, and so does this: the accumulated error is small but the
    // tail join is not a place to have any.
    cum[0] = -bound;
    let last = cum.len() - 1;
    cum[last] = bound;
    cum
}

/// Which bin `y` falls in, given `bins + 1` knots.
fn bin_index(y: f64, cum: &[f64]) -> usize {
    let bins = cum.len() - 1;
    // Count knots at or below `y`, minus one - with the reference's 1e-6 nudge
    // on the top knot so that `y == bound` stays inside the last bin.
    let mut n: usize = 0;
    for (i, &edge) in cum.iter().enumerate() {
        let edge = if i == cum.len() - 1 {
            edge + 1e-6
        } else {
            edge
        };
        if y >= edge {
            n += 1;
        }
    }
    n.saturating_sub(1).min(bins - 1)
}

/// Softmax in place, over the whole slice.
fn softmax(x: &mut [f64]) {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// `log(1 + exp(x))`, computed the way that does not overflow.
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
