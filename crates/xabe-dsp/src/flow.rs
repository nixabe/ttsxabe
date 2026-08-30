//! The inverse of a normalising flow's affine coupling, scalar.
//!
//! WaveGlow runs backwards at inference: it starts from noise and undoes twelve
//! couplings, so the operation that matters is the inverse one. The forward
//! direction is only ever used for training and is not here.

/// `out = (x - b) / exp(s)`, over `[half, t]`.
///
/// `st` is `[2 * half, t]`: the coupling network emits the shift `b` as its
/// first half and the log scale `s` as its second. That order is the
/// checkpoint's, not a convention worth changing - reading it backwards
/// produces audio, just wrong audio.
///
/// A division by `exp(s)` rather than a multiply by `exp(-s)`. The two are the
/// same arithmetic and not the same floats, and twelve flows compose this.
pub fn coupling_inverse(x: &[f32], st: &[f32], out: &mut [f32], half: usize, t: usize) {
    assert_eq!(x.len(), half * t, "x is not [half, t]");
    assert_eq!(st.len(), 2 * half * t, "st is not [2 * half, t]");
    assert_eq!(out.len(), half * t, "out is not [half, t]");

    for c in 0..half {
        for p in 0..t {
            let b = st[c * t + p];
            let s = st[(half + c) * t + p];
            out[c * t + p] = (x[c * t + p] - b) / s.exp();
        }
    }
}
