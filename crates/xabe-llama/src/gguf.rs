//! The one place a GGUF's weights are not the checkpoint's weights.
//!
//! # llama.cpp permutes the query and key projections
//!
//! Both conventions compute the same rotation; they disagree about which two
//! elements of a head form a rotating pair.
//!
//! - 🤗 pairs element `i` with element `i + head_dim/2` — the **halves**
//!   convention, and what `xabe_dsp::rope` implements.
//! - ggml pairs `2i` with `2i+1` — **interleaved**.
//!
//! Rather than carry two rope kernels, llama.cpp's converter bakes the
//! difference into the weights: it permutes the *rows* of `attn_q` and
//! `attn_k` on the way into a GGUF, so that an interleaved rotation over the
//! permuted rows equals a halves rotation over the original ones.
//!
//! `attn_v` is untouched, because no rotation is applied to values. That
//! asymmetry is the fingerprint: reading a GGUF Llama without undoing this
//! gives a model whose `v` is right, whose norms and feed-forward are right,
//! and whose `q` and `k` are shuffled within every head. Every shape checks
//! out. The output is fluent and wrong.
//!
//! Measured on `taigi-translator-13b-f16.gguf` against the safetensors
//! checkpoint it was converted from: `attn_q` and `attn_k` differ in about
//! 25.78 M of 26.21 M elements, and [`unpermute_rope`] takes both to
//! **bit-identical**. Every other tensor in the file is bit-identical without
//! it.
//!
//! This crate undoes the permutation at load so that everything downstream
//! sees the 🤗 layout and one rope kernel serves both containers.

/// Undoes llama.cpp's rope permutation on one `[rows, cols]` tensor.
///
/// `heads` is the number of heads *this* tensor is divided into: the query
/// head count for `attn_q`, the **key-value** head count for `attn_k`. Passing
/// the wrong one silently produces a different shuffle, which is why the
/// caller names it rather than this deriving it.
///
/// Row `2i + k` of each head goes back to row `k * head_dim/2 + i`.
pub fn unpermute_rope(w: &[u16], rows: usize, cols: usize, heads: usize) -> Vec<u16> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert!(heads > 0 && rows.is_multiple_of(heads));
    let head_dim = rows / heads;
    let half = head_dim / 2;
    let mut out = vec![0u16; w.len()];
    for h in 0..heads {
        for i in 0..half {
            for k in 0..2 {
                let src = h * head_dim + 2 * i + k;
                let dst = h * head_dim + k * half + i;
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

/// The same permutation as [`unpermute_rope`], on packed blocks.
///
/// This is possible at all because [`unpermute_rope`] never looks *inside* a
/// row: it moves `cols` contiguous elements at a time, so it is a permutation
/// of whole rows and nothing else. A quantized row is a whole number of
/// blocks - GGUF guarantees the fastest-varying dimension is - so moving
/// `row_bytes` at a time moves exactly the same rows without unpacking
/// anything, and the result is still a valid packed tensor.
///
/// That is what lets `attn_q` and `attn_k` stay packed in VRAM. The
/// alternative would be to unpack them, permute, and repack, which would need
/// a quantizer this workspace does not have and would not be bit-identical if
/// it did.
///
/// `unpermuting_bytes_agrees_with_unpermuting_elements` pins the two against
/// each other on the same data.
pub fn unpermute_rope_bytes(w: &[u8], rows: usize, row_bytes: usize, heads: usize) -> Vec<u8> {
    debug_assert_eq!(w.len(), rows * row_bytes);
    debug_assert!(heads > 0 && rows.is_multiple_of(heads));
    let head_dim = rows / heads;
    let half = head_dim / 2;
    let mut out = vec![0u8; w.len()];
    for h in 0..heads {
        for i in 0..half {
            for k in 0..2 {
                let src = h * head_dim + 2 * i + k;
                let dst = h * head_dim + k * half + i;
                out[dst * row_bytes..(dst + 1) * row_bytes]
                    .copy_from_slice(&w[src * row_bytes..(src + 1) * row_bytes]);
            }
        }
    }
    out
}

/// Applies llama.cpp's rope permutation, the inverse of [`unpermute_rope`].
///
/// Only the tests need this: it is what lets them start from the safetensors
/// tensor and reproduce the GGUF's bits, which is a stronger statement than
/// showing the two agree after both are transformed.
pub fn permute_rope(w: &[u16], rows: usize, cols: usize, heads: usize) -> Vec<u16> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert!(heads > 0 && rows.is_multiple_of(heads));
    let head_dim = rows / heads;
    let half = head_dim / 2;
    let mut out = vec![0u16; w.len()];
    for h in 0..heads {
        for i in 0..half {
            for k in 0..2 {
                let src = h * head_dim + k * half + i;
                let dst = h * head_dim + 2 * i + k;
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

/// Whether a GGUF tensor name is one of the two the permutation touches.
///
/// Named rather than matched inline so that the list is in one place: getting
/// it wrong in either direction - missing `attn_k`, or including `attn_v` -
/// produces a model that loads cleanly and speaks nonsense.
pub fn is_rope_permuted(name: &str) -> bool {
    name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight")
}
