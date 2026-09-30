//! Layout helpers.
//!
//! Two layouts appear in this model and the reference permutes between them
//! constantly: `[T, C]` for anything transformer-shaped, `[C, T]` for anything
//! convolution-shaped. Nothing here is clever; it exists so that the call sites
//! say which layout they are in rather than leaving it to be inferred from the
//! index arithmetic.

/// Transposes a row-major `[rows, cols]` matrix into `[cols, rows]`.
pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), rows * cols);
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// Reverses the channel order of a `[channels, t]` tensor.
///
/// This is `torch.flip(x, [1])` on a `[batch, channels, t]` tensor: it reverses
/// the whole channel axis. It is *not* a swap of two halves - the two coincide
/// only at two channels, which is exactly where the duration predictor uses it,
/// so a half-swap passes there and fails on the 192-channel flow.
pub fn flip_channels(x: &[f32], ch: usize, t: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), ch * t);
    let mut out = vec![0.0; x.len()];
    for c in 0..ch {
        let src = (ch - 1 - c) * t;
        out[c * t..(c + 1) * t].copy_from_slice(&x[src..src + t]);
    }
    out
}

/// Reflects `pad` samples off each end, excluding the endpoints themselves.
///
/// This is `torch.nn.functional.pad(mode="reflect")`, which is what both
/// `torch.stft(center=True)` and Silero's frontend use. The endpoints are not
/// repeated - the left pad starts at `x[pad]` and walks *down* to `x[1]` - and
/// getting that wrong shifts every frame by one sample, which reads as a model
/// that is slightly and inexplicably worse rather than as a bug.
///
/// # Panics
///
/// If `pad >= x.len()`, the point at which the reflection runs off the far end
/// and the operation stops being defined.
pub fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    assert!(
        pad < x.len(),
        "reflecting {pad} off a signal of {}",
        x.len()
    );
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    out.extend((1..=pad).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((1..=pad).map(|i| x[x.len() - 1 - i]));
    out
}
