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
/// `q` and `o` are square. `k` and `v` are `[kv_dim, hidden]`, which equals
/// square only when the model is not grouped-query - Llama-3's 8 key-value
/// heads of 128 make them `[1024, 4096]` against a `[4096, 4096]` query. The
/// binding takes that from [`LlamaConfig::kv_dim`] rather than assuming.
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
    /// Llama-3's per-frequency rope scaling, when the checkpoint carries it.
    ///
    /// `None` for Llama-2, which has no such tensor. Present as 64 f32 for a
    /// 128-wide head on Llama-3.1, one factor per rotating pair.
    pub rope_freqs: Option<Bound>,
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

/// Binds one GGUF tensor and checks its shape.
///
/// The shape compared is [`xabe_gguf::TensorInfo::shape`], the row-major
/// reading, **not** the `dims` the file stores - those are reversed. Comparing
/// the stored order against a geometry written the reference's way would pass
/// only for square matrices, which is every projection in this schema except
/// the two that matter.
fn get_gguf(f: &xabe_gguf::GgufFile, name: &str, want: &[usize]) -> Result<Bound, LlamaError> {
    let info = f
        .info(name)
        .ok_or_else(|| LlamaError::MissingTensor(name.to_string()))?;
    let shape = info.shape();
    if shape != want {
        return Err(LlamaError::Shape {
            name: name.to_string(),
            found: shape,
            want: want.to_vec(),
        });
    }
    Ok(Bound {
        name: name.to_string(),
        shape,
        dtype: match info.ggml_type {
            xabe_gguf::GgmlType::F32 => Dtype::F32,
            xabe_gguf::GgmlType::F16 => Dtype::F16,
            xabe_gguf::GgmlType::Bf16 => Dtype::Bf16,
        },
    })
}

impl LlamaWeights {
    /// Binds every tensor in a GGUF checkpoint against `cfg`.
    ///
    /// The same schema as [`Self::load`] over a different container, and the
    /// differences are all naming: `blk.N.attn_q.weight` for
    /// `model.layers.N.self_attn.q_proj.weight`, `token_embd` for
    /// `model.embed_tokens`, `output` for `lm_head`, `output_norm` for
    /// `model.norm`.
    ///
    /// One tensor has no counterpart on the safetensors side at all.
    /// `rope_freqs.weight` is Llama-3's per-frequency rope scaling, 64 values
    /// for a head width of 128, and it is bound rather than ignored: leaving
    /// it out would make the tensor count reconcile at 291 against a file that
    /// says 292, and a schema that cannot account for every tensor is not a
    /// proof the geometry is understood.
    pub fn from_gguf(f: &xabe_gguf::GgufFile, cfg: &LlamaConfig) -> Result<Self, LlamaError> {
        let (h, i, v) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size);
        let kv = cfg.kv_dim();

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for n in 0..cfg.num_hidden_layers {
            let p = format!("blk.{n}");
            layers.push(Layer {
                attn_norm: get_gguf(f, &format!("{p}.attn_norm.weight"), &[h])?,
                attn: Attention {
                    q: get_gguf(f, &format!("{p}.attn_q.weight"), &[h, h])?,
                    k: get_gguf(f, &format!("{p}.attn_k.weight"), &[kv, h])?,
                    v: get_gguf(f, &format!("{p}.attn_v.weight"), &[kv, h])?,
                    o: get_gguf(f, &format!("{p}.attn_output.weight"), &[h, h])?,
                },
                ffn_norm: get_gguf(f, &format!("{p}.ffn_norm.weight"), &[h])?,
                mlp: Mlp {
                    gate: get_gguf(f, &format!("{p}.ffn_gate.weight"), &[i, h])?,
                    up: get_gguf(f, &format!("{p}.ffn_up.weight"), &[i, h])?,
                    down: get_gguf(f, &format!("{p}.ffn_down.weight"), &[h, i])?,
                },
            });
        }

        let embed_tokens = get_gguf(f, "token_embd.weight", &[v, h])?;
        // Tied embeddings mean there is no `output.weight`; the embedding
        // serves as both, so it is bound twice rather than left absent.
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            get_gguf(f, "output.weight", &[v, h])?
        };

        let w = Self {
            embed_tokens,
            layers,
            norm: get_gguf(f, "output_norm.weight", &[h])?,
            lm_head,
            rope_freqs: match f.info("rope_freqs.weight") {
                Some(_) => Some(get_gguf(f, "rope_freqs.weight", &[cfg.head_dim() / 2])?),
                None => None,
            },
        };
        tracing::info!(
            tensors = w.tensor_count(),
            parameters = w.parameter_count(),
            "bound the GGUF checkpoint",
        );
        Ok(w)
    }

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
            rope_freqs: None,
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
            .chain(self.rope_freqs.iter())
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
