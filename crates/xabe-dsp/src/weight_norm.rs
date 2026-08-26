//! Weight normalisation, undone.
//!
//! PyTorch's `weight_norm` reparameterises a kernel as `g · v / ‖v‖`, storing
//! the direction `v` and the magnitude `g` separately so that training can move
//! them independently. At inference the two are always recombined the same way,
//! so this fuses them once at load rather than on every call.
//!
//! Only part of this checkpoint is stored unfused - the flow's WaveNet layers -
//! while the decoder carries plain `weight` tensors. That asymmetry is real and
//! is documented in `docs/MODEL.md`; assuming either rule holds everywhere
//! fails, in one direction loudly and in the other silently.

/// Fuses `weight_v` and `weight_g` into an ordinary convolution kernel.
///
/// The norm is taken per output channel over the input and kernel axes
/// together, which is what `dim=0` means to `weight_norm`. Normalising over the
/// wrong axes produces a kernel of the right shape and the wrong scale.
pub fn fuse_weight_norm(v: &[f32], g: &[f32], out_ch: usize, in_ch: usize, k: usize) -> Vec<f32> {
    debug_assert_eq!(v.len(), out_ch * in_ch * k);
    debug_assert_eq!(g.len(), out_ch);

    let per = in_ch * k;
    let mut w = vec![0.0; v.len()];
    for o in 0..out_ch {
        let row = &v[o * per..(o + 1) * per];
        // Accumulated in f64: the sum runs over up to a few thousand squares,
        // and the result divides every weight in the row.
        let norm = row
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let scale = (f64::from(g[o]) / norm) as f32;
        for (dst, src) in w[o * per..(o + 1) * per].iter_mut().zip(row) {
            *dst = src * scale;
        }
    }
    w
}

/// WaveNet's gated activation: `tanh(first half) * sigmoid(second half)`.
///
/// The input carries `2 * ch` channels and the output `ch`. Splitting it the
/// other way round - sigmoid first - is a shape-preserving mistake.
pub fn gated_activation(x: &[f32], ch: usize, t: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), 2 * ch * t);
    let mut out = vec![0.0; ch * t];
    for c in 0..ch {
        for i in 0..t {
            let a = x[c * t + i];
            let b = x[(ch + c) * t + i];
            out[c * t + i] = a.tanh() * (1.0 / (1.0 + (-b).exp()));
        }
    }
    out
}
