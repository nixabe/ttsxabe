//! Dense projection.

/// Computes `x @ w.T + b` for row-major `x` of `[t, in_c]` and `w` of
/// `[out_c, in_c]`.
///
/// The weight layout is PyTorch's `nn.Linear`: output channel major, which is
/// the transpose of what the multiply wants. Storing it that way and indexing
/// it transposed is the reference's arrangement, so this does the same rather
/// than transposing the weights at load time - a load-time transpose would make
/// the kernel faster and the comparison against the reference harder to read,
/// and this crate is the one that chooses readable.
pub fn linear(
    x: &[f32],
    t: usize,
    in_c: usize,
    w: &[f32],
    bias: Option<&[f32]>,
    out_c: usize,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * in_c);
    debug_assert_eq!(w.len(), out_c * in_c);

    let mut out = vec![0.0; t * out_c];
    for row in 0..t {
        for o in 0..out_c {
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for i in 0..in_c {
                acc += x[row * in_c + i] * w[o * in_c + i];
            }
            out[row * out_c + o] = acc;
        }
    }
    out
}
