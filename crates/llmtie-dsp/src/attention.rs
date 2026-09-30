//! Multi-head self-attention with relative positional representation.
//!
//! # The relative-position trick, unwound
//!
//! The reference implements the relative bias with three pad-reshape-slice
//! manoeuvres (`_get_relative_embeddings`,
//! `_relative_position_to_absolute_position`, and its inverse). They are the
//! standard trick for computing a relative bias as one dense matmul on a GPU,
//! and they are almost unreadable. Composing all three gives something much
//! simpler, and the derivation is worth writing down because the simplification
//! is the entire content of this file.
//!
//! Let `W` be the window (4 here), `L` the sequence length, and `E` the stored
//! embedding table of shape `[2W+1, D]`.
//!
//! `_get_relative_embeddings` zero-pads `E` by `p = max(L - W - 1, 0)` on both
//! sides and then slices `[s, s + 2L - 1)` where `s = max(W + 1 - L, 0)`. So
//! entry `r` of the result is `E[r - p + s]`, or zero when that is out of
//! range. Exactly one of `p` and `s` is non-zero, and in both cases
//! `s - p = W + 1 - L`.
//!
//! `_relative_position_to_absolute_position` maps a `[L, 2L-1]` matrix to
//! `[L, L]` by `out[i][j] = x[i][L - 1 + j - i]`. Working the padded-flatten
//! through by hand gives that, and the index is always in range.
//!
//! Composing: the bias at `(i, j)` reads embedding entry
//!
//! ```text
//!   (L - 1 + j - i) - p + s  =  (L - 1 + j - i) + (W + 1 - L)  =  W + j - i
//! ```
//!
//! which is in `[0, 2W]` exactly when `|i - j| <= W`, and out of range - hence
//! zero - otherwise. So the whole apparatus is: **a bias of `q_i · E[W + j - i]`
//! inside a window of ±W, and nothing outside it.** The same derivation on
//! `_absolute_position_to_relative_position` gives the value side:
//! `out_i += Σ_j a[i][j] · E_v[W + j - i]` over the same window.
//!
//! Note this is not a *mask*: positions outside the window still attend
//! normally through the ordinary `q·k` term. Only the positional bias is
//! windowed. Reading the reference as sliding-window attention - an easy
//! mistake, since the machinery looks like one - would silently change what the
//! model attends to.

use crate::activation::softmax_rows;
use crate::linear::linear;

/// Runs one self-attention block.
///
/// `x` is `[t, embed]`; the four projections are `[embed, embed]` PyTorch
/// `nn.Linear` weights; `emb_rel_k` and `emb_rel_v` are `[2W+1, head_dim]`.
/// Returns `[t, embed]`.
///
/// No attention mask: this project synthesises one utterance at a time with no
/// padding, so the reference's mask is identically one and its only effect
/// would be to add zero.
#[allow(clippy::too_many_arguments)]
pub fn self_attention(
    x: &[f32],
    t: usize,
    embed: usize,
    heads: usize,
    window: usize,
    q_w: &[f32],
    q_b: Option<&[f32]>,
    k_w: &[f32],
    k_b: Option<&[f32]>,
    v_w: &[f32],
    v_b: Option<&[f32]>,
    out_w: &[f32],
    out_b: Option<&[f32]>,
    emb_rel_k: &[f32],
    emb_rel_v: &[f32],
) -> Vec<f32> {
    let head_dim = embed / heads;
    let scaling = (head_dim as f32).powf(-0.5);
    let span = 2 * window + 1;
    debug_assert_eq!(emb_rel_k.len(), span * head_dim);
    debug_assert_eq!(emb_rel_v.len(), span * head_dim);

    let q = linear(x, t, embed, q_w, q_b, embed);
    let k = linear(x, t, embed, k_w, k_b, embed);
    let v = linear(x, t, embed, v_w, v_b, embed);

    // `[t, embed]` laid out head-major would be more convenient, but keeping the
    // reference's `[t, heads, head_dim]` means every index below reads the same
    // as the tensor it is checked against.
    let mut context = vec![0.0; t * embed];

    for h in 0..heads {
        let base = h * head_dim;
        let mut logits = vec![0.0; t * t];

        for i in 0..t {
            for j in 0..t {
                let mut acc = 0.0;
                for d in 0..head_dim {
                    // The reference scales the query once, before both the
                    // key product and the relative product, so the scale
                    // applies to the bias too.
                    acc += q[i * embed + base + d] * scaling * k[j * embed + base + d];
                }
                // The windowed positional bias derived above.
                let r = window as isize + j as isize - i as isize;
                if r >= 0 && (r as usize) < span {
                    let e = r as usize * head_dim;
                    for d in 0..head_dim {
                        acc += q[i * embed + base + d] * scaling * emb_rel_k[e + d];
                    }
                }
                logits[i * t + j] = acc;
            }
        }

        softmax_rows(&mut logits, t, t);

        for i in 0..t {
            for j in 0..t {
                let a = logits[i * t + j];
                for d in 0..head_dim {
                    context[i * embed + base + d] += a * v[j * embed + base + d];
                }
                let r = window as isize + j as isize - i as isize;
                if r >= 0 && (r as usize) < span {
                    let e = r as usize * head_dim;
                    for d in 0..head_dim {
                        context[i * embed + base + d] += a * emb_rel_v[e + d];
                    }
                }
            }
        }
    }

    linear(&context, t, embed, out_w, out_b, embed)
}
