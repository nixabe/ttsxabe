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
