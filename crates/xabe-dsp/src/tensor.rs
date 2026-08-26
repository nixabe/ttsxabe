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
