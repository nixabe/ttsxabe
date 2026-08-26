//! Every tensor in the checkpoint, bound by name and checked against the
//! geometry `config.json` declares.
//!
//! 363 of them, 13.26 billion parameters, and **not one byte of them is read
//! here**. The schema binds names to the shapes the header declares and checks
//! those against the geometry; materialising 26 GB to prove a name exists
//! would be an odd way to spend a load.
//!
//! That is the whole of milestone 19. The point of doing it before any
//! arithmetic is that a shape disagreement becomes a named error at load time
//! rather than fluent nonsense at 3 a.m., and that binding all 363 by name is
//! a real proof the geometry is understood - the same test that caught the
//! weight-norm mistake in `xabe-vits`.

use crate::{LlamaConfig, LlamaError};
use xabe_st::{Dtype, StSet};

/// A tensor the schema has bound: what it is called, and what the file says
/// about it.
#[derive(Debug, Clone)]
pub struct Bound {
    /// The name in the checkpoint.
    pub name: String,
    /// The shape the header declares, already checked against the geometry.
    pub shape: Vec<usize>,
    /// The width it is stored at.
    pub dtype: Dtype,
}

impl Bound {
    /// How many parameters it holds.
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The four projections of one attention block.
///
/// All four are square here, because this checkpoint has as many key-value
/// heads as query heads. A grouped-query model would narrow `k` and `v`, and
/// [`LlamaConfig::check`] refuses one rather than binding it wrongly.
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection.
    pub q: Bound,
    /// Key projection.
    pub k: Bound,
    /// Value projection.
    pub v: Bound,
    /// Output projection.
    pub o: Bound,
}

/// One SwiGLU feed-forward block.
///
/// Three matrices, not two: the gate and the up projection are separate and
/// multiplied elementwise after `silu`, which is what makes the inner width
/// 13824 rather than 4x the hidden size.
#[derive(Debug, Clone)]
pub struct Mlp {
    /// The half that goes through `silu`.
    pub gate: Bound,
    /// The half that does not.
    pub up: Bound,
    /// The contraction back to `hidden_size`.
    pub down: Bound,
}

/// One transformer block.
#[derive(Debug, Clone)]
pub struct Layer {
    /// RMS normalisation before attention.
    pub attn_norm: Bound,
    /// Self-attention.
    pub attn: Attention,
    /// RMS normalisation before the feed-forward.
    pub ffn_norm: Bound,
    /// The feed-forward.
    pub mlp: Mlp,
}

/// The whole checkpoint, bound.
#[derive(Debug)]
pub struct LlamaWeights {
    /// `[vocab_size, hidden_size]`.
    pub embed_tokens: Bound,
    /// The blocks, in order.
    pub layers: Vec<Layer>,
    /// The final RMS normalisation.
    pub norm: Bound,
    /// `[vocab_size, hidden_size]`, separate because the embeddings are untied.
    pub lm_head: Bound,
}

/// Binds one tensor and checks its shape.
fn get(st: &StSet, name: &str, want: &[usize]) -> Result<Bound, LlamaError> {
    let info = st
        .info(name)
        .ok_or_else(|| LlamaError::MissingTensor(name.to_string()))?;
    if info.shape != want {
        return Err(LlamaError::Shape {
            name: name.to_string(),
            found: info.shape.clone(),
            want: want.to_vec(),
        });
    }
    Ok(Bound {
        name: name.to_string(),
        shape: info.shape.clone(),
        dtype: info.dtype,
    })
}

impl LlamaWeights {
    /// Binds every tensor in the checkpoint against `cfg`.
    pub fn load(st: &StSet, cfg: &LlamaConfig) -> Result<Self, LlamaError> {
        let (h, i, v) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for n in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{n}");
            layers.push(Layer {
                attn_norm: get(st, &format!("{p}.input_layernorm.weight"), &[h])?,
                attn: Attention {
                    q: get(st, &format!("{p}.self_attn.q_proj.weight"), &[h, h])?,
                    k: get(st, &format!("{p}.self_attn.k_proj.weight"), &[h, h])?,
                    v: get(st, &format!("{p}.self_attn.v_proj.weight"), &[h, h])?,
                    o: get(st, &format!("{p}.self_attn.o_proj.weight"), &[h, h])?,
                },
                ffn_norm: get(st, &format!("{p}.post_attention_layernorm.weight"), &[h])?,
                mlp: Mlp {
                    gate: get(st, &format!("{p}.mlp.gate_proj.weight"), &[i, h])?,
                    up: get(st, &format!("{p}.mlp.up_proj.weight"), &[i, h])?,
                    down: get(st, &format!("{p}.mlp.down_proj.weight"), &[h, i])?,
                },
            });
        }

        let w = Self {
            embed_tokens: get(st, "model.embed_tokens.weight", &[v, h])?,
            layers,
            norm: get(st, "model.norm.weight", &[h])?,
            lm_head: get(st, "lm_head.weight", &[v, h])?,
        };
        tracing::info!(
            tensors = w.tensor_count(),
            parameters = w.parameter_count(),
            "bound the checkpoint",
        );
        Ok(w)
    }

    /// Every tensor the schema bound, in binding order.
    pub fn tensors(&self) -> impl Iterator<Item = &Bound> {
        let per_layer = self.layers.iter().flat_map(|l| {
            [
                &l.attn_norm,
                &l.attn.q,
                &l.attn.k,
                &l.attn.v,
                &l.attn.o,
                &l.ffn_norm,
                &l.mlp.gate,
                &l.mlp.up,
                &l.mlp.down,
            ]
        });
        std::iter::once(&self.embed_tokens)
            .chain(per_layer)
            .chain([&self.norm, &self.lm_head])
    }

    /// How many tensors the schema binds.
    pub fn tensor_count(&self) -> usize {
        self.tensors().count()
    }

    /// How many parameters the bound tensors hold.
    pub fn parameter_count(&self) -> usize {
        self.tensors().map(Bound::elements).sum()
    }
}
