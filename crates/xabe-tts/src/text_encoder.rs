//! The text encoder: symbol ids to the prior's parameters.
//!
//! Six pre-norm-shaped layers - though they are in fact *post*-norm, which is
//! worth saying out loud because the diagram in the VITS paper is not - each
//! being relative-position self-attention, a residual add, a norm, a
//! convolutional feed-forward, a second residual add and a second norm. Then a
//! width-1 convolution projects to `2 * flow_size` and splits into the prior's
//! mean and log-variance.
//!
//! # Two details that do not look like details
//!
//! **The embedding is scaled by `sqrt(hidden_size)`.** One line in the
//! reference, easy to drop, and dropping it does not crash anything: it makes
//! every subsequent layer see inputs 14× too small, which the layer norms
//! partly absorb, so the output is plausible and wrong.
//!
//! **The feed-forward transposes into convolution layout and back.** It is a
//! convolution over time with kernel 3, not a position-wise linear - so a
//! symbol's feed-forward output depends on its neighbours. Implementing it as
//! the position-wise MLP that transformers usually have would match every shape
//! and be wrong at every position.
//!
//! There is no attention mask and no padding mask here: this project
//! synthesises one utterance at a time with nothing padded, so both are
//! identically one in the reference and their only effect is to multiply by it.

use xabe_dsp::{conv1d, layer_norm, relu, same_padding, self_attention, transpose};
use xabe_vits::{TextEncoder, VitsConfig};

/// What the text encoder produces.
///
/// Carries the per-layer intermediates as well as the result. They are kept
/// because the oracle captured them: with them, a differential failure says
/// "layer 3", and without them it says "the text encoder", which is six layers
/// and four kernels to bisect by hand. Six sequences of `[t, 192]` is a few
/// hundred kilobytes for a sentence - nothing next to the decoder, which
/// materialises 256 samples per frame.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// The scaled embedding lookup, `[t, hidden_size]`.
    pub embed: Vec<f32>,
    /// Each layer's output, `[t, hidden_size]`, in order.
    pub layers: Vec<Vec<f32>>,
    /// The last layer's hidden states, `[t, hidden_size]`.
    pub hidden: Vec<f32>,
    /// The prior's mean, `[t, flow_size]`.
    pub m_p: Vec<f32>,
    /// The prior's log standard deviation, `[t, flow_size]`.
    pub logs_p: Vec<f32>,
    /// Number of symbols.
    pub t: usize,
}

/// Runs the text encoder over a sequence of symbol ids.
pub fn text_encoder(ids: &[i64], w: &TextEncoder<'_>, cfg: &VitsConfig) -> EncoderOutput {
    let t = ids.len();
    let hidden = cfg.hidden_size;

    // Embedding lookup, then the scaling that is easy to forget.
    let scale = (hidden as f32).sqrt();
    let mut h = vec![0.0; t * hidden];
    for (pos, &id) in ids.iter().enumerate() {
        let row = id as usize * hidden;
        for c in 0..hidden {
            h[pos * hidden + c] = w.embed[row + c] * scale;
        }
    }

    let embed = h.clone();
    let mut layers = Vec::with_capacity(w.layers.len());

    for layer in &w.layers {
        let attn = self_attention(
            &h,
            t,
            hidden,
            cfg.num_attention_heads,
            cfg.window_size,
            layer.q.weight,
            layer.q.bias,
            layer.k.weight,
            layer.k.bias,
            layer.v.weight,
            layer.v.bias,
            layer.out.weight,
            layer.out.bias,
            layer.emb_rel_k,
            layer.emb_rel_v,
        );
        for (dst, src) in h.iter_mut().zip(&attn) {
            *dst += src;
        }
        h = layer_norm(
            &h,
            t,
            hidden,
            layer.norm.weight,
            layer.norm.bias,
            cfg.layer_norm_eps,
        );

        let ff = feed_forward(&h, t, cfg, layer);
        for (dst, src) in h.iter_mut().zip(&ff) {
            *dst += src;
        }
        h = layer_norm(
            &h,
            t,
            hidden,
            layer.final_norm.weight,
            layer.final_norm.bias,
            cfg.layer_norm_eps,
        );
        layers.push(h.clone());
    }

    // The projection is a width-1 convolution over `[hidden, t]`, so it needs
    // the convolution layout even though nothing about it is convolutional.
    let stats = conv1d(
        &transpose(&h, t, hidden),
        hidden,
        t,
        w.project.weight,
        w.project.bias,
        cfg.flow_size * 2,
        1,
        0,
        0,
        1,
    );

    // Split along channels *before* transposing back, since the split is on the
    // channel axis and the two halves are contiguous there.
    let flow = cfg.flow_size;
    let m_p = transpose(&stats[..flow * t], flow, t);
    let logs_p = transpose(&stats[flow * t..], flow, t);

    EncoderOutput {
        embed,
        layers,
        hidden: h,
        m_p,
        logs_p,
        t,
    }
}

/// The convolutional feed-forward block.
fn feed_forward(
    h: &[f32],
    t: usize,
    cfg: &VitsConfig,
    layer: &xabe_vits::EncoderLayer<'_>,
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let k = layer.ffn_1.k;
    let (pl, pr) = same_padding(k);

    let x = transpose(h, t, hidden);
    let mut y = conv1d(
        &x,
        hidden,
        t,
        layer.ffn_1.weight,
        layer.ffn_1.bias,
        cfg.ffn_dim,
        k,
        pl,
        pr,
        1,
    );
    relu(&mut y);
    let z = conv1d(
        &y,
        cfg.ffn_dim,
        t,
        layer.ffn_2.weight,
        layer.ffn_2.bias,
        hidden,
        k,
        pl,
        pr,
        1,
    );
    transpose(&z, hidden, t)
}
