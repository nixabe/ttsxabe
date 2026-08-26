//! Every tensor in the checkpoint, bound by name and checked against the
//! geometry `config.json` declares.
//!
//! 1,259 of them. Nothing here does arithmetic; the point is that a shape
//! disagreement is a named error at load time rather than a transcript that is
//! subtly wrong at 3 a.m. This is the same discipline that caught the
//! weight-norm mistake in `xabe-vits`, applied to a model forty times larger.
//!
//! # Why the weights are borrowed
//!
//! The checkpoint is 6.12 GiB of F32 and it is memory-mapped. Copying it into
//! owned `Vec`s would double that for no gain: the tensors are read-only for
//! the whole life of the model, and the CUDA path uploads from these slices
//! directly. So every field is a `&'a [f32]` into the mapping, and the
//! `WhisperWeights` borrows the [`StSet`] it came from.

use crate::{WhisperConfig, WhisperError};
use xabe_st::StSet;

/// A projection: `[out, in]` row-major, with an optional bias.
///
/// The bias is optional because Whisper's `k_proj` genuinely has none - a
/// detail worth a type rather than a comment, since a zero bias silently
/// invented here would be indistinguishable from the real thing until the
/// tensor count came up two hundred short.
#[derive(Debug, Clone, Copy)]
pub struct Linear<'a> {
    /// `[out_dim, in_dim]`, row-major.
    pub weight: &'a [f32],
    /// `[out_dim]`, when the reference has one.
    pub bias: Option<&'a [f32]>,
    /// Columns of `weight`.
    pub in_dim: usize,
    /// Rows of `weight`.
    pub out_dim: usize,
}

/// Scale and shift for a layer normalisation.
#[derive(Debug, Clone, Copy)]
pub struct LayerNorm<'a> {
    /// `[d]`.
    pub weight: &'a [f32],
    /// `[d]`.
    pub bias: &'a [f32],
}

/// A convolution over the mel axis: `[out, in, k]`.
#[derive(Debug, Clone, Copy)]
pub struct Conv1d<'a> {
    /// `[out_ch, in_ch, k]`, row-major.
    pub weight: &'a [f32],
    /// `[out_ch]`.
    pub bias: &'a [f32],
    /// Input channels.
    pub in_ch: usize,
    /// Output channels.
    pub out_ch: usize,
    /// Kernel width.
    pub k: usize,
    /// Stride.
    pub stride: usize,
}

/// The four projections of one attention block.
#[derive(Debug, Clone, Copy)]
pub struct Attention<'a> {
    /// Query projection, with bias.
    pub q: Linear<'a>,
    /// Key projection, which has no bias in any Whisper checkpoint.
    pub k: Linear<'a>,
    /// Value projection, with bias.
    pub v: Linear<'a>,
    /// Output projection, with bias.
    pub out: Linear<'a>,
}

/// One encoder block: self-attention, then a feed-forward, both pre-normed.
#[derive(Debug, Clone, Copy)]
pub struct EncoderLayer<'a> {
    /// Normalisation before self-attention.
    pub attn_ln: LayerNorm<'a>,
    /// Self-attention over the whole 1500-position sequence.
    pub attn: Attention<'a>,
    /// Normalisation before the feed-forward.
    pub ffn_ln: LayerNorm<'a>,
    /// Expansion, `d_model` to `encoder_ffn_dim`.
    pub fc1: Linear<'a>,
    /// Contraction, back to `d_model`.
    pub fc2: Linear<'a>,
}

/// One decoder block: causal self-attention, cross-attention, feed-forward.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayer<'a> {
    /// Normalisation before self-attention.
    pub attn_ln: LayerNorm<'a>,
    /// Causal self-attention over the tokens emitted so far.
    pub attn: Attention<'a>,
    /// Normalisation before cross-attention.
    pub cross_ln: LayerNorm<'a>,
    /// Cross-attention into the encoder's output.
    pub cross: Attention<'a>,
    /// Normalisation before the feed-forward.
    pub ffn_ln: LayerNorm<'a>,
    /// Expansion, `d_model` to `decoder_ffn_dim`.
    pub fc1: Linear<'a>,
    /// Contraction, back to `d_model`.
    pub fc2: Linear<'a>,
}

/// The whole checkpoint, bound.
#[derive(Debug)]
pub struct WhisperWeights<'a> {
    /// The geometry every shape here was checked against.
    pub cfg: &'a WhisperConfig,
    /// `[d_model, n_mels, 3]`, stride 1.
    pub conv1: Conv1d<'a>,
    /// `[d_model, d_model, 3]`, stride 2 - this is what halves the frames.
    pub conv2: Conv1d<'a>,
    /// `[max_source_positions, d_model]`, sinusoidal and frozen.
    pub enc_pos: &'a [f32],
    /// The encoder blocks, in order.
    pub enc_layers: Vec<EncoderLayer<'a>>,
    /// The encoder's final normalisation.
    pub enc_ln: LayerNorm<'a>,
    /// `[vocab_size, d_model]`, also the output projection - Whisper ties them.
    pub embed_tokens: &'a [f32],
    /// `[max_target_positions, d_model]`, learned.
    pub dec_pos: &'a [f32],
    /// The decoder blocks, in order.
    pub dec_layers: Vec<DecoderLayer<'a>>,
    /// The decoder's final normalisation.
    pub dec_ln: LayerNorm<'a>,
}

/// Binds one tensor and checks its shape.
fn get<'a>(st: &'a StSet, name: &str, want: &[usize]) -> Result<&'a [f32], WhisperError> {
    let info = st
        .info(name)
        .ok_or_else(|| WhisperError::MissingTensor(name.to_string()))?;
    if info.shape != want {
        return Err(WhisperError::Shape {
            name: name.to_string(),
            found: info.shape.clone(),
            want: want.to_vec(),
        });
    }
    Ok(st.tensor(name)?)
}

/// Binds a projection, with a bias if `bias` says the reference has one.
fn linear<'a>(
    st: &'a StSet,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> Result<Linear<'a>, WhisperError> {
    Ok(Linear {
        weight: get(st, &format!("{prefix}.weight"), &[out_dim, in_dim])?,
        bias: bias
            .then(|| get(st, &format!("{prefix}.bias"), &[out_dim]))
            .transpose()?,
        in_dim,
        out_dim,
    })
}

/// Binds a layer normalisation.
fn norm<'a>(st: &'a StSet, prefix: &str, d: usize) -> Result<LayerNorm<'a>, WhisperError> {
    Ok(LayerNorm {
        weight: get(st, &format!("{prefix}.weight"), &[d])?,
        bias: get(st, &format!("{prefix}.bias"), &[d])?,
    })
}

/// Binds the four projections of an attention block.
///
/// `k_proj` is bound without a bias, which is the reference's shape and not an
/// omission: Whisper's key projection is the one linear layer in the model that
/// has none.
fn attention<'a>(st: &'a StSet, prefix: &str, d: usize) -> Result<Attention<'a>, WhisperError> {
    Ok(Attention {
        q: linear(st, &format!("{prefix}.q_proj"), d, d, true)?,
        k: linear(st, &format!("{prefix}.k_proj"), d, d, false)?,
        v: linear(st, &format!("{prefix}.v_proj"), d, d, true)?,
        out: linear(st, &format!("{prefix}.out_proj"), d, d, true)?,
    })
}

impl<'a> WhisperWeights<'a> {
    /// Binds every tensor in the checkpoint against `cfg`.
    pub fn load(st: &'a StSet, cfg: &'a WhisperConfig) -> Result<Self, WhisperError> {
        let d = cfg.d_model;

        let conv1 = Conv1d {
            weight: get(st, "model.encoder.conv1.weight", &[d, cfg.num_mel_bins, 3])?,
            bias: get(st, "model.encoder.conv1.bias", &[d])?,
            in_ch: cfg.num_mel_bins,
            out_ch: d,
            k: 3,
            stride: 1,
        };
        let conv2 = Conv1d {
            weight: get(st, "model.encoder.conv2.weight", &[d, d, 3])?,
            bias: get(st, "model.encoder.conv2.bias", &[d])?,
            in_ch: d,
            out_ch: d,
            k: 3,
            stride: 2,
        };

        let mut enc_layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            let p = format!("model.encoder.layers.{i}");
            enc_layers.push(EncoderLayer {
                attn_ln: norm(st, &format!("{p}.self_attn_layer_norm"), d)?,
                attn: attention(st, &format!("{p}.self_attn"), d)?,
                ffn_ln: norm(st, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear(st, &format!("{p}.fc1"), d, cfg.encoder_ffn_dim, true)?,
                fc2: linear(st, &format!("{p}.fc2"), cfg.encoder_ffn_dim, d, true)?,
            });
        }

        let mut dec_layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            let p = format!("model.decoder.layers.{i}");
            dec_layers.push(DecoderLayer {
                attn_ln: norm(st, &format!("{p}.self_attn_layer_norm"), d)?,
                attn: attention(st, &format!("{p}.self_attn"), d)?,
                cross_ln: norm(st, &format!("{p}.encoder_attn_layer_norm"), d)?,
                cross: attention(st, &format!("{p}.encoder_attn"), d)?,
                ffn_ln: norm(st, &format!("{p}.final_layer_norm"), d)?,
                fc1: linear(st, &format!("{p}.fc1"), d, cfg.decoder_ffn_dim, true)?,
                fc2: linear(st, &format!("{p}.fc2"), cfg.decoder_ffn_dim, d, true)?,
            });
        }

        let w = Self {
            cfg,
            conv1,
            conv2,
            enc_pos: get(
                st,
                "model.encoder.embed_positions.weight",
                &[cfg.max_source_positions, d],
            )?,
            enc_layers,
            enc_ln: norm(st, "model.encoder.layer_norm", d)?,
            embed_tokens: get(
                st,
                "model.decoder.embed_tokens.weight",
                &[cfg.vocab_size, d],
            )?,
            dec_pos: get(
                st,
                "model.decoder.embed_positions.weight",
                &[cfg.max_target_positions, d],
            )?,
            dec_layers,
            dec_ln: norm(st, "model.decoder.layer_norm", d)?,
        };

        tracing::info!(
            tensors = w.tensor_count(),
            parameters = w.parameter_count(),
            "bound the checkpoint",
        );
        Ok(w)
    }

    /// How many tensors the schema binds.
    ///
    /// Counted from the geometry, not from the file, so it can be compared
    /// against the file's own inventory. Per encoder layer: two norms at two
    /// tensors each, four projections at seven tensors (`k_proj` has no bias),
    /// and two feed-forward layers at two each - 15. A decoder layer adds a
    /// third norm and a second attention: 15 + 2 + 7 = 24.
    pub fn tensor_count(&self) -> usize {
        let fixed = 4 /* two convolutions */ + 1 /* encoder positions */
            + 2 /* encoder norm */ + 1 /* token embedding */
            + 1 /* decoder positions */ + 2 /* decoder norm */;
        fixed + 15 * self.cfg.encoder_layers + 24 * self.cfg.decoder_layers
    }

    /// How many parameters the bound tensors hold.
    pub fn parameter_count(&self) -> usize {
        let lin = |l: &Linear| l.weight.len() + l.bias.map_or(0, <[f32]>::len);
        let nrm = |n: &LayerNorm| n.weight.len() + n.bias.len();
        let att = |a: &Attention| lin(&a.q) + lin(&a.k) + lin(&a.v) + lin(&a.out);
        self.conv1.weight.len()
            + self.conv1.bias.len()
            + self.conv2.weight.len()
            + self.conv2.bias.len()
            + self.enc_pos.len()
            + nrm(&self.enc_ln)
            + self.embed_tokens.len()
            + self.dec_pos.len()
            + nrm(&self.dec_ln)
            + self
                .enc_layers
                .iter()
                .map(|l| {
                    nrm(&l.attn_ln) + att(&l.attn) + nrm(&l.ffn_ln) + lin(&l.fc1) + lin(&l.fc2)
                })
                .sum::<usize>()
            + self
                .dec_layers
                .iter()
                .map(|l| {
                    nrm(&l.attn_ln)
                        + att(&l.attn)
                        + nrm(&l.cross_ln)
                        + att(&l.cross)
                        + nrm(&l.ffn_ln)
                        + lin(&l.fc1)
                        + lin(&l.fc2)
                })
                .sum::<usize>()
    }
}
