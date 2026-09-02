//! Llama-3 on CUDA, with grouped-query attention.

use crate::ChatError;
use std::path::Path;
use xabe_cuda::{Batch, CudaSlice, DecodeScratch, Gpu, NormScratch, Operand, Q8, Quant};
use xabe_gguf::GgufFile;
use xabe_llama::{Bound, Bpe, LlamaConfig, LlamaWeights};

/// Whether a block-quantized checkpoint stays packed on the card.
///
/// The default is [`Packing::Packed`], which is what makes a quantized file
/// occupy its own size in VRAM rather than its unpacked size. [`Packing::F16`]
/// is what this engine did before the packed matmul existed, and it is kept
/// for two reasons rather than removed: it is the control the packed path is
/// tested against on identical weights, and it is the answer if a format ever
/// turns out to cost more accuracy than a stage can afford.
///
/// It has no effect on an unquantized checkpoint, which has no blocks to keep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Packing {
    /// Keep the checkpoint's blocks; unpack inside the matmul.
    #[default]
    Packed,
    /// Unpack at load and store one f16 per element.
    F16,
}

/// How one matrix is held on the device.
///
/// The distinction is memory and nothing else: both reach the same kernel and
/// both are staged to f16 inside it. What differs is what sits in VRAM between
/// calls - one number per element, or the checkpoint's own packed blocks.
enum GWeight {
    /// Rounded to f16 at load. What an f16 or f32 checkpoint gives.
    F16(CudaSlice<u16>),
    /// The checkpoint's blocks, byte for byte, unpacked inside the matmul.
    Packed {
        /// The packed bytes.
        data: CudaSlice<u8>,
        /// Which layout reads them.
        ty: Quant,
    },
}

/// The token table on the device.
///
/// Gathered rather than multiplied, so it is the one packed tensor the
/// matmuls never see. It used to be widened to f32 at load regardless of the
/// file - 2 GB at this vocabulary for a table a decode step reads one row of -
/// and now stays in its blocks when the file has them; `Gpu::embed_packed`
/// unpacks the rows it gathers. See docs/BENCHMARKS.md.
enum GEmbed {
    /// Full width, what an f16 or f32 checkpoint gives.
    F32(CudaSlice<f32>),
    /// The checkpoint's blocks, unpacked a row at a time on lookup.
    Packed {
        /// The packed bytes.
        data: CudaSlice<u8>,
        /// Which layout reads them.
        ty: Quant,
    },
}

/// One `[out, in]` matrix on the device.
struct GLinear {
    w: GWeight,
    in_dim: usize,
    out_dim: usize,
}

impl GLinear {
    /// This matrix as the matmul's right operand.
    fn operand(&self) -> Operand<'_> {
        match &self.w {
            GWeight::F16(w) => Operand::F16(w),
            GWeight::Packed { data, ty } => Operand::Q { data, ty: *ty },
        }
    }
}

/// The attention projections, grouped into as few products as the checkpoint
/// allows.
///
/// Batching needs identical `(in_dim, out_dim, block format)`, and which of q,
/// k and v qualify is a property of the file rather than a choice. Q4_K_M
/// stores `attn_v` as Q6_K in half the layers of both models, and a
/// grouped-query model gives `attn_q` a different width from `attn_k` - so
/// this model usually fuses k with v and the translator usually fuses q with
/// k, and a layer whose three disagree runs three products.
///
/// The point is block count. One 128-token prompt is one row of tiles, so a
/// 5120-wide projection is 40 blocks on a machine that runs 144 at once, and
/// the answer used to be splitting the contraction - which buys blocks at the
/// cost of a partial buffer and a reduction pass. Three projections issued as
/// one product buy the same blocks with neither.
struct GAttn {
    /// The projections' rows stacked in q, k, v order, one allocation per
    /// run of the same width and block format: `[q; k; v]` when all three
    /// agree, `[q; k]` and `[v]` when `attn_v` is the odd one out.
    ///
    /// A decode step projects a whole part in one launch - the rows of one
    /// activation against every row of the stack - and reads q, k and v out
    /// of that one output at their row offsets. A prompt wants each
    /// projection's rows as `[n, out]`, so it runs the members one at a time
    /// from their first row; see `Gpu::gemm_batched_from`. What the stack
    /// buys is block count at one row: the chat model's k and v were two
    /// launches of 128 blocks each, and are now rows of a 768-block launch.
    parts: Vec<GAttnPart>,
}

/// One stacked allocation of attention projections; see [`GAttn`].
struct GAttnPart {
    /// The stack as one matrix of `rows.iter().sum()` output rows.
    w: GLinear,
    /// Which of q (0), k (1), v (2) the first member is; the rest follow.
    first: usize,
    /// Each member's output rows, in order.
    rows: Vec<usize>,
}

/// One batched product: `count` matrices of identical shape in one allocation.
/// The gate and the up projection, together where the checkpoint allows it.
///
/// They are the same shape and, in this one, the same block format - so they
/// are one batched product over one activation, and the SiLU gate reads the two
/// halves of its output. That is a launch a layer on each side, and at a single
/// decoded row a launch is most of what a kernel this size costs. `Split` is
/// the fallback for a checkpoint that quantizes them differently, which
/// `Q4_K_M` does do to other pairs - `attn_v` and `ffn_down` get `Q6_K` here
/// while their neighbours get `Q4_K`.
enum GMlp {
    Fused(GGroup),
    Split(GLinear, GLinear),
}

struct GGroup {
    w: GLinear,
    count: usize,
}

/// One decoder block.
struct GLayer {
    attn_norm: CudaSlice<f32>,
    attn: GAttn,
    o: GLinear,
    ffn_norm: CudaSlice<f32>,
    mlp: GMlp,
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
    embed: GEmbed,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The output projection. Llama-3 does not tie it to the embedding.
    lm_head: GLinear,
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
    /// f16, and that is the whole point of it.
    ///
    /// The weights a token reads are a fixed 4.6 GB; the cache is the term
    /// that *grows*, and at 2048 positions it is 537 MB re-read for every
    /// token on top of them. Halving its width halves that, and halves what a
    /// long conversation costs to hold. llama.cpp's cache is f16 by default,
    /// so this moves the arithmetic toward the reference rather than away from
    /// it - `layer_taps` compares against that reference and its bound did not
    /// move. See docs/BENCHMARKS.md.
    k: Vec<CudaSlice<u16>>,
    v: Vec<CudaSlice<u16>>,
    /// Tokens the buffers have room for, which is not how many they hold.
    ///
    /// The first version of this grew by exactly the tokens being added, which
    /// meant an allocation, a zeroing and a full copy of the whole cache for
    /// every layer of every token - 128 allocations and 16 MB of copying a
    /// token at 64 of context, and quadratic in the context after that. It is
    /// doubled instead, so a decode of any length pays for the growth a
    /// logarithmic number of times and appends in place the rest.
    cap: usize,
    len: usize,
    /// The fused decode attention's partials and counters; see
    /// `Gpu::attn_decode_f16`. Per cache rather than per model because two
    /// conversations decoding by turns must not share a counter.
    scratch: DecodeScratch,
    /// The norm-fused mat-vec's partials and counter; see
    /// `Gpu::gemv_norm`. Per cache for the same reason.
    norm_scratch: NormScratch,
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
    ///
    /// The attention scratch is kept: it holds nothing about the sequence,
    /// and its counters are already at rest.
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.cap = 0;
        self.len = 0;
    }
}

impl ChatModel {
    /// Loads a GGUF onto CUDA device `ordinal`.
    ///
    /// A quantized checkpoint keeps its own blocks and occupies about what the
    /// file occupies; an f16 or f32 one is rounded to f16 and occupies two
    /// bytes an element. This used to say a quantized copy "lands at the same
    /// 16 GB, because this engine unpacks on read", which was true before
    /// `Packing::Packed` existed and is the default now. Use
    /// [`Self::open_with`] to load the same file the other way. See
    /// `docs/MODEL.md`.
    pub fn open(path: &Path, ordinal: usize) -> Result<Self, ChatError> {
        Self::open_with(path, ordinal, Packing::default())
    }

    /// The same, choosing how a quantized checkpoint is held.
    pub fn open_with(path: &Path, ordinal: usize, packing: Packing) -> Result<Self, ChatError> {
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
        // A matrix stays in whatever the file stores it as, when the kernel has
        // a path for that layout. This is the whole difference between a
        // quantized checkpoint that loads faster and one that also *fits*: the
        // f16 branch is 2 bytes an element, the packed branch is whatever the
        // block format is - 4.5 bits for Q4_K.
        let lin = |b: &Bound| -> Result<GLinear, ChatError> {
            let packed = (packing == Packing::Packed)
                .then(|| b.packed.and_then(|t| Quant::from_id(t as u32)))
                .flatten();
            let w = match packed {
                Some(ty) => GWeight::Packed {
                    data: gpu.upload_quant(ty, &Self::packed(&f, b, &cfg, ty)?)?,
                    ty,
                },
                None => GWeight::F16(narrow(b)?),
            };
            Ok(GLinear {
                w,
                in_dim: b.shape[1],
                out_dim: b.shape[0],
            })
        };

        // q, k and v as one product each where they must be and as one product
        // together where they may be. See [`GAttn`]: identical shape and
        // identical block format, which is the whole condition.
        let fused = |bs: &[&Bound]| -> Result<GGroup, ChatError> {
            let head = bs[0];
            let packed = (packing == Packing::Packed)
                .then(|| head.packed.and_then(|t| Quant::from_id(t as u32)))
                .flatten();
            let w = match packed {
                Some(ty) => {
                    let mut all = Vec::new();
                    for b in bs {
                        all.extend_from_slice(&Self::packed(&f, b, &cfg, ty)?);
                    }
                    GWeight::Packed {
                        data: gpu.upload_quant(ty, &all)?,
                        ty,
                    }
                }
                None => {
                    let mut all = Vec::new();
                    for b in bs {
                        all.extend_from_slice(&Self::f16(&f, b, &cfg)?);
                    }
                    GWeight::F16(gpu.upload_u16(&all)?)
                }
            };
            Ok(GGroup {
                w: GLinear {
                    w,
                    in_dim: head.shape[1],
                    out_dim: head.shape[0],
                },
                count: bs.len(),
            })
        };
        let attn = |a: &xabe_llama::Attention| -> Result<GAttn, ChatError> {
            let bs = [&a.q, &a.k, &a.v];
            // Stacked where the width and the block format agree; the row
            // count may differ, which is the whole point for a grouped-query
            // model whose k and v are a quarter of q.
            let key = |b: &Bound| {
                (
                    b.shape[1],
                    (packing == Packing::Packed).then_some(b.packed).flatten(),
                )
            };
            let mut runs: Vec<Vec<usize>> = Vec::new();
            for i in 0..3 {
                let joins = runs
                    .last()
                    .and_then(|r| r.last())
                    .is_some_and(|&j| key(bs[i]) == key(bs[j]));
                match joins {
                    true => runs.last_mut().expect("a run to join").push(i),
                    false => runs.push(vec![i]),
                }
            }
            let mut parts = Vec::with_capacity(runs.len());
            for r in &runs {
                let members: Vec<&Bound> = r.iter().map(|&i| bs[i]).collect();
                let g = fused(&members)?;
                parts.push(GAttnPart {
                    w: GLinear {
                        w: g.w.w,
                        in_dim: g.w.in_dim,
                        out_dim: members.iter().map(|b| b.shape[0]).sum(),
                    },
                    first: r[0],
                    rows: members.iter().map(|b| b.shape[0]).collect(),
                });
            }
            Ok(GAttn { parts })
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layers.push(GLayer {
                attn_norm: wide(&l.attn_norm)?,
                attn: attn(&l.attn)?,
                o: lin(&l.attn.o)?,
                ffn_norm: wide(&l.ffn_norm)?,
                mlp: {
                    // The same test the attention grouping uses to decide a
                    // run: identical shape and, when packed, identical format.
                    let key = |b: &Bound| {
                        (
                            b.shape.clone(),
                            (packing == Packing::Packed).then_some(b.packed).flatten(),
                        )
                    };
                    match key(&l.mlp.gate) == key(&l.mlp.up) {
                        true => GMlp::Fused(fused(&[&l.mlp.gate, &l.mlp.up])?),
                        false => GMlp::Split(lin(&l.mlp.gate)?, lin(&l.mlp.up)?),
                    }
                },
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

        // The table stays packed on the same condition a matrix does. It is
        // never rope-permuted, so its bytes are the file's.
        let embed = match (packing == Packing::Packed)
            .then(|| w.embed_tokens.packed.and_then(|t| Quant::from_id(t as u32)))
            .flatten()
        {
            Some(ty) => GEmbed::Packed {
                data: gpu.upload_quant(ty, &Self::packed(&f, &w.embed_tokens, &cfg, ty)?)?,
                ty,
            },
            None => GEmbed::F32(wide(&w.embed_tokens)?),
        };
        let model = Self {
            embed,
            layers,
            norm: wide(&w.norm)?,
            lm_head: lin(&w.lm_head)?,
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

    /// A tensor's blocks, byte for byte, with the rope permutation undone.
    ///
    /// The sibling of [`Self::f16`] and it has to undo the same permutation,
    /// or `q` and `k` are shuffled within every head and the model is fluent
    /// and wrong - see `xabe_llama::gguf`. It can do so *without unpacking*
    /// only because that permutation moves whole rows: a quantized row is a
    /// whole number of blocks, so the same shuffle applies to byte ranges.
    fn packed(f: &GgufFile, b: &Bound, cfg: &LlamaConfig, ty: Quant) -> Result<Vec<u8>, ChatError> {
        let raw = f.tensor_bytes(&b.name)?;
        if !xabe_llama::gguf::is_rope_permuted(&b.name) {
            return Ok(raw.to_vec());
        }
        let heads = if b.shape[0] == cfg.hidden_size {
            cfg.num_attention_heads
        } else {
            cfg.num_key_value_heads
        };
        Ok(xabe_llama::gguf::unpermute_rope_bytes(
            raw,
            b.shape[0],
            ty.bytes(b.shape[1]),
            heads,
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
            cap: 0,
            len: 0,
            scratch: DecodeScratch::new(),
            norm_scratch: NormScratch::new(),
        }
    }

    /// One projection.
    /// Every matrix of a group against one activation, in one launch.
    ///
    /// The output is `[count, rows, out_dim]`. `a` is zero because the whole
    /// point of a group is that its members share the activation - and share
    /// its int8 twin, so it is quantized once rather than once a projection.
    fn project_group(
        &self,
        x: Operand<'_>,
        g: &GGroup,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.gpu.gemm_batched(
            x,
            g.w.operand(),
            None,
            Batch {
                count: g.count,
                a: 0,
                w: g.w.in_dim * g.w.out_dim,
                out: rows * g.w.out_dim,
                w_row: 0,
            },
            rows,
            g.w.in_dim,
            g.w.out_dim,
        )?)
    }

    fn project(
        &self,
        x: Operand<'_>,
        l: &GLinear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.gpu.gemm_batched(
            x,
            l.operand(),
            None,
            Batch::single(rows * l.out_dim),
            rows,
            l.in_dim,
            l.out_dim,
        )?)
    }

    /// Normalises, and takes the int8 twin at the same time when the shape
    /// admits it.
    ///
    /// `None` is not a failure: `gemm_batched` quantises for itself when it has
    /// to, so the only thing lost is the sharing. But the sharing is worth
    /// having at every length now that the tiled matmul reads int8 too: this
    /// activation feeds three projections at attention and two at the MLP, and
    /// without the twin each of them quantises it again. That was 4.4% of the
    /// prefill in one kernel, and it was the same numbers five times.
    fn normed(
        &self,
        h: &mut CudaSlice<f32>,
        add: Option<&CudaSlice<f32>>,
        rows: usize,
        k: usize,
        weight: &CudaSlice<f32>,
    ) -> Result<(CudaSlice<f32>, Option<Q8>), ChatError> {
        let eps = self.cfg.rms_norm_eps;
        if !k.is_multiple_of(256) {
            if let Some(a) = add {
                self.gpu.add_inplace(h, a, rows * k)?;
            }
            return Ok((self.gpu.rms_norm(h, rows, k, weight, eps)?, None));
        }
        let (x, q8) = self.gpu.rms_norm_q(h, add, rows, k, weight, eps)?;
        Ok((x, Some(q8)))
    }

    /// Pairs an activation with its twin, when there is one.
    fn operand<'a>(x: &'a CudaSlice<f32>, q8: Option<&'a Q8>) -> Operand<'a> {
        match q8 {
            Some(q8) => Operand::F32Q { data: x, q8 },
            None => Operand::F32(x),
        }
    }

    /// Runs `ids` through the model and returns the logits, `[n, vocab]`.
    pub fn forward(&self, ids: &[u32], cache: &mut Cache) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.run(ids, cache, 0, false)?.0)
    }

    /// The same, but only the last position's logits, `[1, vocab]`.
    ///
    /// What generation wants: the other rows are run so the cache is filled,
    /// and their logits are thrown away. Projecting them through a
    /// 128256-wide head is not free - see the note at the end of `run`.
    pub fn forward_last(
        &self,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, ChatError> {
        Ok(self.run(ids, cache, 0, true)?.0)
    }

    /// The same as [`Self::forward`], also returning the first `taps` block
    /// outputs on the host.
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
        self.run(ids, cache, taps, false)
    }

    fn run(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
        last_only: bool,
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
        let dids = self.gpu.upload_i64(&ids64)?;
        let mut h = match &self.embed {
            GEmbed::F32(t) => self.gpu.embed_scaled(t, &dids, n, h_dim, 1.0)?,
            GEmbed::Packed { data, ty } => {
                self.gpu.embed_packed(data, *ty, &dids, n, h_dim, 1.0)?
            }
        };

        // Room for `past + n` before any layer touches it, so the loop below
        // never allocates. Doubling from a floor of 256 means a 64-token decode
        // grows once and a 4096-token one grows five times.
        // The block output that the next normalisation still has to add. See
        // the note where it is set: the residual add and the normalisation read
        // the same row, so they are one pass.
        let mut residual: Option<CudaSlice<f32>> = None;
        // The next normalised row and its twin, when the projection that
        // closed the last sub-layer produced them in its tail - which at
        // one row it does, see `Gpu::gemv_norm`. Then `residual` is `None`
        // and `h` is already settled.
        let mut pending: Option<(CudaSlice<f32>, Q8)> = None;
        let first = cache.k.is_empty();
        if cache.cap < past + n {
            let want = (past + n).next_power_of_two().max(256);
            let was = cache.cap;
            // Re-strided, not copied. The cache is head-major and `cap` is the
            // stride between heads in both layouts, so a flat copy of the live
            // prefix moves head 0 correctly and buries every other head inside
            // its own earlier positions - a reply that is fluent for one
            // sentence and noise after it, at whatever token first crosses a
            // power of two. See `Gpu::cache_grow`.
            let keys = cache.k.iter_mut().map(|s| (s, false));
            let values = cache.v.iter_mut().map(|s| (s, true));
            for (slot, transposed) in keys.chain(values) {
                let mut grown = self.gpu.zeros_f16(want * kv_dim)?;
                // A fresh conversation grows from empty, and sixty-four
                // zero-byte copies are still sixty-four launches.
                if past > 0 {
                    self.gpu.cache_grow_f16(
                        slot, &mut grown, kv_heads, hd, was, want, past, transposed,
                    )?;
                }
                *slot = grown;
            }
            cache.cap = want;
        }
        let mut tapped = Vec::with_capacity(taps);
        for (i, l) in self.layers.iter().enumerate() {
            // One int8 twin for three projections, taken by the normalisation
            // that produced the activation. The packed mat-vec would otherwise
            // take the same one three times, in three launches that each re-read
            // a row this kernel had just written.
            let (x, xq) = match pending.take() {
                Some((x, q)) => (x, Some(q)),
                None => self.normed(&mut h, residual.take().as_ref(), n, h_dim, &l.attn_norm)?,
            };
            let xo = Self::operand(&x, xq.as_ref());
            // One product per group rather than one per projection. Each
            // element of a batched product writes a contiguous block of the
            // same output, so q, k and v are located by an offset into one of
            // these buffers and nothing is copied to separate them.
            // At one row a part is one product over its whole stack, and q,
            // k and v are row offsets into its output. A prompt runs the
            // members one at a time so each comes out `[n, out]`. See `GAttn`.
            let mut proj = Vec::with_capacity(3);
            let mut at = [(0usize, 0usize); 3];
            for part in &l.attn.parts {
                let mut before = 0usize;
                if n == 1 {
                    let idx = proj.len();
                    proj.push(self.gpu.gemm_batched(
                        xo,
                        part.w.operand(),
                        None,
                        Batch::single(part.w.out_dim),
                        1,
                        part.w.in_dim,
                        part.w.out_dim,
                    )?);
                    for (mi, &rows) in part.rows.iter().enumerate() {
                        at[part.first + mi] = (idx, before);
                        before += rows;
                    }
                } else {
                    for (mi, &rows) in part.rows.iter().enumerate() {
                        proj.push(self.gpu.gemm_batched_from(
                            xo,
                            part.w.operand(),
                            before,
                            None,
                            Batch::single(n * rows),
                            n,
                            part.w.in_dim,
                            rows,
                        )?);
                        at[part.first + mi] = (proj.len() - 1, 0);
                        before += rows;
                    }
                }
            }
            let (qg, q_off) = at[0];
            let (kg, k_off) = at[1];
            let (vg, v_off) = at[2];
            debug_assert_eq!((qg, q_off), (0, 0), "the query leads its group");

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
            // Appended in place. `grow` has already made room for `past + n`
            // and copied what was there, so at steady state this is two copies
            // a layer and no allocation at all.
            // Scattered straight into the layout attention reads, rather than
            // appended and rearranged. See `Gpu::cache_append`.
            if first {
                cache.k.push(self.gpu.zeros_f16(cache.cap * kv_dim)?);
                cache.v.push(self.gpu.zeros_f16(cache.cap * kv_dim)?);
            }
            let cap = cache.cap;
            if n == 1 {
                // One decoded position: the two rotations and the two appends
                // are one launch, and the rotated key goes straight to the
                // cache. See `Gpu::rope_cache_f16`.
                self.gpu.rope_cache_f16(
                    &mut proj,
                    (qg, q_off),
                    (kg, k_off),
                    (vg, v_off),
                    d,
                    heads,
                    kv_heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                    &mut cache.k[i],
                    &mut cache.v[i],
                    cap,
                )?;
            } else {
                self.gpu.rope_scaled(
                    &mut proj[qg],
                    q_off,
                    d,
                    n,
                    heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                )?;
                self.gpu.rope_scaled(
                    &mut proj[kg],
                    k_off,
                    d,
                    n,
                    kv_heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                )?;
                self.gpu.cache_append_f16(
                    &proj[kg],
                    k_off,
                    &mut cache.k[i],
                    n,
                    kv_heads,
                    hd,
                    cap,
                    past,
                    false,
                )?;
                self.gpu.cache_append_f16(
                    &proj[vg],
                    v_off,
                    &mut cache.v[i],
                    n,
                    kv_heads,
                    hd,
                    cap,
                    past,
                    true,
                )?;
            }
            let tk = past + n;

            // The grouped heads *are* the batch. A batch of `kv_heads`
            // products, each covering the `group` query heads that share one
            // key head, replaces expanding the keys and values to the query
            // head count: `repeat_kv` materialised four identical copies of
            // every cached head, every layer, every token, and this reads the
            // one copy four times instead.
            //
            // The query rows line up for free. `split_heads` lays them out
            // `[head][t][d]`, so the `group * n` rows one key head serves are
            // contiguous, and for a single step they are contiguous already -
            // which is why the split is skipped there.
            let group = heads / kv_heads;
            // The query is at the start of its buffer, so `split_heads` reads
            // it from zero and a single step reads it as the row it is.
            let q = &proj[qg];

            // A prompt takes the fused attention: scores, mask, softmax and
            // the value product in one kernel, reading the query buffer and
            // the caches in place - no head split, no score matrix, no merge,
            // and the grouped heads read the one cached copy directly. A
            // single step takes its own fused kernel, which does the same
            // three things for one row without ever writing the score row -
            // three launches a layer became one, and the grouped rows share
            // every cache read. The chain below is what is left for a width
            // neither kernel is instantiated at, which this checkpoint's
            // never is.
            if n == 1 && (hd == 128 || hd == 64) {
                // The context and its int8 twin in one pass; the projection
                // that reads the twin is packed in every checkpoint here, and
                // an unpacked one ignores it.
                let (ctx, cq) = self.gpu.attn_decode_f16_q(
                    q,
                    &cache.k[i],
                    &cache.v[i],
                    heads,
                    kv_heads,
                    hd,
                    tk,
                    cap,
                    (hd as f32).powf(-0.5),
                    false,
                    &mut cache.scratch,
                )?;
                match l.o.w {
                    GWeight::Packed {
                        ty: Quant::Q4K | Quant::Q6K,
                        ..
                    } => {
                        // The output projection with the residual add and the
                        // MLP's normalisation in its tail: one launch where
                        // there were three.
                        pending = Some(self.gpu.gemv_norm(
                            l.o.operand(),
                            &cq,
                            l.o.in_dim,
                            l.o.out_dim,
                            &mut h,
                            &l.ffn_norm,
                            self.cfg.rms_norm_eps,
                            &mut cache.norm_scratch,
                        )?);
                    }
                    _ => {
                        let out = self.project(
                            Operand::F32Q {
                                data: &ctx,
                                q8: &cq,
                            },
                            &l.o,
                            1,
                        )?;
                        residual = Some(out);
                    }
                }
            } else if n > 1 && hd == 128 {
                let ctx = self.gpu.flash_attn_f16(
                    q,
                    &cache.k[i],
                    &cache.v[i],
                    n,
                    past,
                    heads,
                    kv_heads,
                    hd,
                    cap,
                    (hd as f32).powf(-0.5),
                    true,
                )?;
                let out = self.project(Operand::F32(&ctx), &l.o, n)?;
                residual = Some(out);
            } else {
                let qh = if n == 1 {
                    None
                } else {
                    Some(self.gpu.split_heads(q, n, heads, hd)?)
                };
                let mut scores = self.gpu.gemm_batched(
                    Operand::F32(qh.as_ref().unwrap_or(q)),
                    Operand::F16(&cache.k[i]),
                    None,
                    Batch {
                        count: kv_heads,
                        a: group * n * hd,
                        w: cap * hd,
                        out: group * n * tk,
                        w_row: 0,
                    },
                    group * n,
                    hd,
                    tk,
                )?;
                // Llama scales the *scores*, not the query - the opposite of
                // Whisper, and the same algebra. Copy each where it belongs.
                self.gpu.softmax_causal(
                    &mut scores,
                    heads * n,
                    tk,
                    n,
                    tk - n,
                    (hd as f32).powf(-0.5),
                )?;

                // `w_row` is `cap`, not `tk`: the values sit in a buffer with room
                // for more positions than are in it, so a row of the operand is a
                // capacity apart. Contracting over `tk` of it is what makes the
                // untouched tail of the cache irrelevant rather than wrong.
                let ctx = self.gpu.gemm_batched(
                    Operand::F32(&scores),
                    Operand::F16(&cache.v[i]),
                    None,
                    Batch {
                        count: kv_heads,
                        a: group * n * tk,
                        w: hd * cap,
                        out: group * n * hd,
                        w_row: cap,
                    },
                    group * n,
                    tk,
                    hd,
                )?;
                // `[kv_head][group * n][hd]` is `[head][t][hd]`, which for a single
                // step is already `[heads * hd]` - the shape the output projection
                // wants. Only a multi-row pass has anything to merge.
                // The merge takes the context's int8 twin in the same pass when
                // the output projection is packed and will read it - the same
                // reasoning that gives the normalisation and the gating theirs.
                let packed_o = matches!(l.o.w, GWeight::Packed { .. });
                let (ctx, cq) = match n {
                    1 => (ctx, None),
                    _ if packed_o && (heads * hd).is_multiple_of(256) => {
                        let (c, q) = self.gpu.merge_heads_q(&ctx, n, heads, hd)?;
                        (c, Some(q))
                    }
                    _ => (self.gpu.merge_heads(&ctx, n, heads, hd)?, None),
                };
                // Not added here. The next normalisation reads `h + out` and
                // nothing between now and then does, so the sum is left for it to
                // take in the pass it was going to make anyway.
                let out = self.project(Self::operand(&ctx, cq.as_ref()), &l.o, n)?;
                residual = Some(out);
            }

            let (x, xq) = match pending.take() {
                Some((x, q)) => (x, Some(q)),
                None => self.normed(&mut h, residual.take().as_ref(), n, h_dim, &l.ffn_norm)?,
            };
            let xo = Self::operand(&x, xq.as_ref());
            let inter = self.cfg.intermediate_size;
            // One product over both halves where the checkpoint allows it, and
            // then the gate reads its own output; see `GMlp`.
            let (mut gate, up) = match &l.mlp {
                GMlp::Fused(g) => (self.project_group(xo, g, n)?, None),
                GMlp::Split(gw, uw) => (self.project(xo, gw, n)?, Some(self.project(xo, uw, n)?)),
            };
            // The gate is projected back down by a packed weight, so its twin
            // comes from the gating rather than from a kernel of its own - at
            // every row count, because the tiled integer kernel reads the same
            // codes the mat-vec does. Only for a packed `down`; an f16 one
            // would leave the codes unread.
            let packed_down = matches!(l.down.w, GWeight::Packed { .. });
            let want_q = packed_down && inter.is_multiple_of(256);
            let gq = match (&up, want_q) {
                (Some(u), true) => Some(self.gpu.silu_mul_q(&mut gate, u, n, inter)?),
                (Some(u), false) => {
                    self.gpu.silu_mul(&mut gate, u, n * inter)?;
                    None
                }
                // Fused: the two operands are the two halves of `gate`.
                (None, true) => Some(self.gpu.silu_mul_pair(&mut gate, n, inter)?),
                (None, false) => {
                    self.gpu.silu_mul_halves(&mut gate, n * inter)?;
                    None
                }
            };
            let down_kq = matches!(
                l.down.w,
                GWeight::Packed {
                    ty: Quant::Q4K | Quant::Q6K,
                    ..
                }
            );
            match (n, &gq, down_kq) {
                (1, Some(q), true) => {
                    // The down projection with the residual add and the *next*
                    // normalisation in its tail - the next layer's, or the
                    // model's final one after the last layer.
                    let next = self
                        .layers
                        .get(i + 1)
                        .map_or(&self.norm, |nl| &nl.attn_norm);
                    pending = Some(self.gpu.gemv_norm(
                        l.down.operand(),
                        q,
                        l.down.in_dim,
                        l.down.out_dim,
                        &mut h,
                        next,
                        self.cfg.rms_norm_eps,
                        &mut cache.norm_scratch,
                    )?);
                }
                _ => {
                    let down = self.project(Self::operand(&gate, gq.as_ref()), &l.down, n)?;
                    residual = Some(down);
                }
            }

            // A tap is the block's output, which is the residual stream *with*
            // this block's contribution - so the pending add has to be settled
            // before it is read. Only when a tap is asked for.
            if i < taps {
                if let Some(r) = residual.take() {
                    self.gpu.add_inplace(&mut h, &r, n * h_dim)?;
                }
                tapped.push(self.gpu.download(&h)?);
            }
        }
        cache.len = past + n;

        if let Some(r) = residual.take() {
            self.gpu.add_inplace(&mut h, &r, n * h_dim)?;
        }
        // The final normalisation, unless the last layer's down projection
        // already produced it - with a twin the head's packed mat-vec reads.
        let (h, hq) = match pending.take() {
            Some((x, q)) => (x, Some(q)),
            None => (
                self.gpu
                    .rms_norm(&h, n, h_dim, &self.norm, self.cfg.rms_norm_eps)?,
                None,
            ),
        };
        if taps > 0 {
            tapped.push(self.gpu.download(&h)?);
        }
        // Only the last row predicts the next token. A 128-token prompt
        // projected all 128 of them through a 128256-wide head and threw 127
        // away: 7.8 ms of a 95 ms prefill, measured. The rows are still *run*
        // through every block - that is what fills the cache - they are just
        // not projected.
        let (rows, h) = match last_only && n > 1 {
            true => (1, self.gpu.copy_range(&h, (n - 1) * h_dim, h_dim)?),
            false => (n, h),
        };
        Ok((
            self.gpu.gemm_batched(
                Self::operand(&h, hq.as_ref()),
                self.lm_head.operand(),
                None,
                Batch::single(rows * self.cfg.vocab_size),
                rows,
                h_dim,
                self.cfg.vocab_size,
            )?,
            tapped,
        ))
    }
}
