//! Llama-3 on CUDA, with grouped-query attention.

use crate::ChatError;
use std::path::Path;
use xabe_cuda::{Batch, CudaSlice, Gpu, Operand};
use xabe_gguf::GgufFile;
use xabe_llama::{Bound, Bpe, LlamaConfig, LlamaWeights};

/// One `[out, in]` matrix on the device, at f16.
struct GLinear {
    w: CudaSlice<u16>,
    in_dim: usize,
    out_dim: usize,
}

/// One decoder block.
struct GLayer {
    attn_norm: CudaSlice<f32>,
    q: GLinear,
    k: GLinear,
    v: GLinear,
    o: GLinear,
    ffn_norm: CudaSlice<f32>,
    gate: GLinear,
    up: GLinear,
    down: GLinear,
}

/// The chat model, resident on one card.
///
/// # How this differs from `xabe_translate`
///
/// Same architecture family, three changes, and every one of them is silent if
/// missed - the model keeps producing fluent text from wrong arithmetic:
///
/// - **Grouped-query attention.** 32 query heads over 8 key-value heads, so
///   `k` and `v` project to 1024 rather than 4096 and one key-value head
///   serves four query heads. The cache is a quarter the size; the attention
///   is not, so the heads are expanded on the way in.
/// - **A rope base of 500000**, not 10000. A defaulted base gives a model
///   fluent for one sentence and drifting after it.
/// - **`rope_freqs.weight`**, Llama-3.1's per-pair frequency divisor, which
///   has no safetensors counterpart and is *not* all ones on this checkpoint -
///   1.0 for the first 29 pairs, a ramp through six, then 8.0 for the rest.
///   Ignoring it is the same failure as the rope base and shows up at the same
///   place: long context.
///
/// # Why it is GGUF-only
///
/// The translator exists on this disk in both containers, so it reads either.
/// This model exists as a GGUF and nothing else, and its vocabulary lives
/// inside the file - there is no `tokenizer.json` beside it. A safetensors
/// path would load 16 GB of weights and then have nothing to tokenize with.
pub struct ChatModel {
    cfg: LlamaConfig,
    tokenizer: Bpe,
    gpu: Gpu,
    embed: CudaSlice<f32>,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The output projection. Llama-3 does not tie it to the embedding.
    lm_head: CudaSlice<u16>,
    /// Llama-3.1's per-pair rope divisor, if the checkpoint carries one.
    rope_freqs: Option<CudaSlice<f32>>,
}

/// The keys and values a decode step reuses.
///
/// Held at `kv_dim` - the *narrow* width - rather than expanded. Storing the
/// expansion would quadruple the cache for no information: the four query
/// heads in a group read the same key-value head, so what is saved is
/// 3/4 of 32 layers' worth of context, which at 8k tokens is 6 GB against 1.5.
pub struct Cache {
    k: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    len: usize,
}

impl Cache {
    /// How many tokens it holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been decoded into it yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drops everything, so the next forward pass starts from position zero.
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.len = 0;
    }
}

impl ChatModel {
    /// Loads a GGUF onto CUDA device `ordinal`.
    ///
    /// About 16 GB of transfers at f16, and it takes as long as that sounds.
    /// A quantized copy of the same checkpoint loads faster because the file
    /// is smaller - and lands at the same 16 GB, because this engine unpacks
    /// on read rather than running packed blocks. See `docs/MODEL.md`.
    pub fn open(path: &Path, ordinal: usize) -> Result<Self, ChatError> {
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            return Err(ChatError::NotGguf(path.to_path_buf()));
        }
        let f = GgufFile::open(path)?;
        let cfg = LlamaConfig::from_gguf(&f)?;
        // Deliberately *not* `refuse_grouped_query`. The translator calls that
        // because its forward pass has no head mapping; this one is written
        // for the mapping, so the shape it refuses is the shape this exists to
        // run.
        let tokenizer = Bpe::from_gguf(&f)?;
        let w = LlamaWeights::from_gguf(&f, &cfg)?;
        let gpu = Gpu::open(ordinal)?;

        let narrow = |b: &Bound| -> Result<CudaSlice<u16>, ChatError> {
            Ok(gpu.upload_u16(&Self::f16(&f, b, &cfg)?)?)
        };
        let wide = |b: &Bound| -> Result<CudaSlice<f32>, ChatError> {
            Ok(gpu.upload(&f.tensor_f32(&b.name)?)?)
        };
        let lin = |b: &Bound| -> Result<GLinear, ChatError> {
            Ok(GLinear {
                w: narrow(b)?,
                in_dim: b.shape[1],
                out_dim: b.shape[0],
            })
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layers.push(GLayer {
                attn_norm: wide(&l.attn_norm)?,
                q: lin(&l.attn.q)?,
                k: lin(&l.attn.k)?,
                v: lin(&l.attn.v)?,
                o: lin(&l.attn.o)?,
                ffn_norm: wide(&l.ffn_norm)?,
                gate: lin(&l.mlp.gate)?,
                up: lin(&l.mlp.up)?,
                down: lin(&l.mlp.down)?,
            });
        }

        let rope_freqs = match &w.rope_freqs {
            Some(b) => Some(wide(b)?),
            None => None,
        };
        if rope_freqs.is_none() {
            // Worth saying out loud rather than defaulting silently: a
            // Llama-3.1 checkpoint has one, and its absence changes what long
            // contexts do.
            tracing::warn!("no rope_freqs.weight; rope runs unscaled");
        }

        let model = Self {
            embed: wide(&w.embed_tokens)?,
            layers,
            norm: wide(&w.norm)?,
            lm_head: narrow(&w.lm_head)?,
            rope_freqs,
            cfg,
            tokenizer,
            gpu,
        };
        tracing::info!(
            device = ordinal,
            heads = model.cfg.num_attention_heads,
            kv_heads = model.cfg.num_key_value_heads,
            "chat model on the device"
        );
        Ok(model)
    }

    /// A tensor at f16, undoing llama.cpp's rope permutation where it applies.
    ///
    /// `attn_q` and `attn_k` are stored with ggml's interleaved rope pairing
    /// baked in; every other tensor is untouched. `attn_v` is the fingerprint
    /// of that asymmetry - values are not rotated, so it is never permuted.
    fn f16(f: &GgufFile, b: &Bound, cfg: &LlamaConfig) -> Result<Vec<u16>, ChatError> {
        let raw = f.tensor_f16(&b.name)?;
        if !xabe_llama::gguf::is_rope_permuted(&b.name) {
            return Ok(raw);
        }
        // Rows are the output width, so `q` unpermutes over the query heads
        // and `k` over the key-value heads. Using the query count for both
        // would scramble `k` into a valid-looking tensor of the wrong rows.
        let heads = if b.shape[0] == cfg.hidden_size {
            cfg.num_attention_heads
        } else {
            cfg.num_key_value_heads
        };
        Ok(xabe_llama::gguf::unpermute_rope(
            &raw, b.shape[0], b.shape[1], heads,
        ))
    }

    /// The geometry this model was bound against.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// The tokenizer, for building prompts and reading answers.
    pub fn tokenizer(&self) -> &Bpe {
        &self.tokenizer
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// An empty cache for one sequence.
    pub fn cache(&self) -> Cache {
        Cache {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
        }
    }

    /// One projection.
    fn project(
        &self,
        x: Operand<'_>,
        l: &GLinear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.gpu.gemm_batched(
            x,
            Operand::F16(&l.w),
            None,
            Batch::single(rows * l.out_dim),
            rows,
            l.in_dim,
            l.out_dim,
        )?)
    }

    /// Runs `ids` through the model and returns the logits, `[n, vocab]`.
    pub fn forward(&self, ids: &[u32], cache: &mut Cache) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.forward_tapped(ids, cache, 0)?.0)
    }

    /// The same, also returning the first `taps` block outputs on the host.
    ///
    /// On the public surface for the same reason the translator's is: "the
    /// model is wrong" is not a fact anyone can act on, and "layer 7 is wrong"
    /// is.
    pub fn forward_tapped(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
    ) -> Result<(CudaSlice<f32>, Vec<Vec<f32>>), ChatError> {
        let h_dim = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let kv_dim = self.cfg.kv_dim();
        let (n, past) = (ids.len(), cache.len);
        if past + n > self.cfg.max_position_embeddings {
            return Err(ChatError::PastTheEnd {
                at: past + n,
                max: self.cfg.max_position_embeddings,
            });
        }

        let ids64: Vec<i64> = ids.iter().map(|&i| i64::from(i)).collect();
        // Llama does not scale its embeddings, unlike the models that do.
        let mut h =
            self.gpu
                .embed_scaled(&self.embed, &self.gpu.upload_i64(&ids64)?, n, h_dim, 1.0)?;

        let first = cache.k.is_empty();
        let mut tapped = Vec::with_capacity(taps);
        for (i, l) in self.layers.iter().enumerate() {
            let x = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.attn_norm, self.cfg.rms_norm_eps)?;
            let mut q = self.project(Operand::F32(&x), &l.q, n)?;
            let mut k = self.project(Operand::F32(&x), &l.k, n)?;
            let v = self.project(Operand::F32(&x), &l.v, n)?;

            // Rotated before caching, because the position is absolute: a key
            // stored unrotated would be rotated again by the wrong offset on
            // every later step.
            //
            // `k` rotates over **`kv_heads`**, not `heads`. The kernel walks
            // the tensor as `heads * head_dim` per position, so passing the
            // query count here would read 4096 floats out of a 1024-wide row
            // and rotate four positions' keys as if they were one position's
            // heads. There is no shape check that catches it - the buffer is
            // the right length in total.
            let d = self.rope_freqs.as_ref();
            self.gpu
                .rope_scaled(&mut q, d, n, heads, hd, self.cfg.rope_theta, past)?;
            self.gpu
                .rope_scaled(&mut k, d, n, kv_heads, hd, self.cfg.rope_theta, past)?;

            if first {
                cache.k.push(k);
                cache.v.push(v);
            } else {
                for (slot, new) in [(&mut cache.k[i], k), (&mut cache.v[i], v)] {
                    let mut grown = self.gpu.zeros((past + n) * kv_dim)?;
                    self.gpu.copy_into(&mut grown, slot, 0, past * kv_dim)?;
                    self.gpu
                        .copy_into(&mut grown, &new, past * kv_dim, n * kv_dim)?;
                    *slot = grown;
                }
            }
            let tk = past + n;

            // Split at the narrow head count, then expand. `repeat_kv` works
            // on whole heads, so it is indifferent to whether the head's block
            // is `[t, hd]` or the transposed `[hd, t]` that the value side
            // wants - both are `t * hd` contiguous floats per head.
            let qh = self.gpu.split_heads(&q, n, heads, hd)?;
            let kh = self.gpu.split_heads(&cache.k[i], tk, kv_heads, hd)?;
            let kh = self.gpu.repeat_kv(&kh, heads, kv_heads, tk, hd)?;
            let vt = self.gpu.split_heads_t(&cache.v[i], tk, kv_heads, hd)?;
            let vt = self.gpu.repeat_kv(&vt, heads, kv_heads, tk, hd)?;

            let mut scores = self.gpu.gemm_batched(
                Operand::F32(&qh),
                Operand::F32(&kh),
                None,
                Batch {
                    count: heads,
                    a: n * hd,
                    w: tk * hd,
                    out: n * tk,
                },
                n,
                hd,
                tk,
            )?;
            // Llama scales the *scores*, not the query - the opposite of
            // Whisper, and the same algebra. Copy each where it belongs.
            self.gpu
                .scale_inplace(&mut scores, heads * n * tk, (hd as f32).powf(-0.5))?;
            self.gpu.causal_mask(&mut scores, heads, n, tk, tk - n)?;
            self.gpu.softmax_rows(&mut scores, heads * n, tk)?;

            let ctx = self.gpu.gemm_batched(
                Operand::F32(&scores),
                Operand::F32(&vt),
                None,
                Batch {
                    count: heads,
                    a: n * tk,
                    w: hd * tk,
                    out: n * hd,
                },
                n,
                tk,
                hd,
            )?;
            let ctx = self.gpu.merge_heads(&ctx, n, heads, hd)?;
            let out = self.project(Operand::F32(&ctx), &l.o, n)?;
            self.gpu.add_inplace(&mut h, &out, n * h_dim)?;

            let x = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.ffn_norm, self.cfg.rms_norm_eps)?;
            let mut gate = self.project(Operand::F32(&x), &l.gate, n)?;
            let up = self.project(Operand::F32(&x), &l.up, n)?;
            self.gpu
                .silu_mul(&mut gate, &up, n * self.cfg.intermediate_size)?;
            let down = self.project(Operand::F32(&gate), &l.down, n)?;
            self.gpu.add_inplace(&mut h, &down, n * h_dim)?;

            if i < taps {
                tapped.push(self.gpu.download(&h)?);
            }
        }
        cache.len = past + n;

        let h = self
            .gpu
            .rms_norm(&h, n, h_dim, &self.norm, self.cfg.rms_norm_eps)?;
        if taps > 0 {
            tapped.push(self.gpu.download(&h)?);
        }
        Ok((
            self.gpu.gemm_batched(
                Operand::F32(&h),
                Operand::F16(&self.lm_head),
                None,
                Batch::single(n * self.cfg.vocab_size),
                n,
                h_dim,
                self.cfg.vocab_size,
            )?,
            tapped,
        ))
    }
}
