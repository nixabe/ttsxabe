//! Runtime int8 quantization of an activation.
//!
//! The reference for the mat-vec's fast path. It exists because that path
//! cannot use wide loads while the activation is f32: a lane that reads sixteen
//! bytes of packed weight covers 32 elements, and fetching 32 f32 activations
//! for them costs more than the wide load wins. At int8 they are two loads and
//! the dot product becomes four `dp4a`.
//!
//! This is lossy, and visibly so. Everything else in this crate is a reference
//! for an exact computation; this one defines an approximation, and the
//! differential test against the CUDA twin is a bit-for-bit comparison of the
//! *same* approximation, not a tolerance on the original.

/// Quantises `x` to int8 in groups of `GROUP`, returning the codes and one
/// scale a group.
///
/// Scale is `max|x| / 127` over the group, so zero maps to zero and the range
/// is symmetric - an asymmetric scale would need a zero point, and a zero point
/// would cost the mat-vec another term per block for no accuracy that matters
/// at this width.
///
/// A group that is entirely zero gets scale zero and quantises to zero, which
/// is exact rather than a special case.
///
/// Rounding is to nearest, ties away from zero, matching `__float2int_rn`'s
/// behaviour on the values this ever sees - a tie is representable here only
/// when the scale is a power of two, and the test covers it.
pub const GROUP: usize = 32;

/// See [`GROUP`].
pub fn quantize_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    debug_assert!(x.len().is_multiple_of(GROUP), "ragged group");
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; x.len() / GROUP];
    for (g, chunk) in x.as_chunks::<GROUP>().0.iter().enumerate() {
        let mx = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = mx / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales[g] = d;
        for (j, v) in chunk.iter().enumerate() {
            codes[g * GROUP + j] = (v * inv).round() as i8;
        }
    }
    (codes, scales)
}
