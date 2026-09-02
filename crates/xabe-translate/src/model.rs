//! The Llama-2 forward pass on CUDA.

use crate::TranslateError;
use std::path::Path;
use xabe_cuda::{Batch, CudaSlice, DecodeScratch, Gpu, NormScratch, Operand, Q8, Quant};
use xabe_gguf::GgufFile;
use xabe_llama::{Bound, LlamaConfig, LlamaWeights, Tokenizer};
use xabe_st::StSet;

/// The checkpoint, in whichever container it happens to be.
///
/// The 13 B translator exists on this machine twice: as the 🤗 safetensors
/// directory it was published as, and as the f16 GGUF `llama-server` runs.
/// They hold the same 363 tensors and the same 13,261,870,080 parameters, and
/// this engine reads either.
///
/// Two things differ, and only two:
///
/// - **Width.** The safetensors is bf16 and is rounded to f16 on the way in;
///   the GGUF is already f16, so the same call is a byte copy. Measured
///   bit-identical on every tensor that is not permuted, which is what makes
///   the containers interchangeable rather than merely similar.
/// - **Layout of `q` and `k`.** llama.cpp bakes its interleaved rope
///   convention into those two tensors. [`xabe_llama::gguf`] explains it and
///   undoes it here, so nothing downstream learns which container it came
///   from.
enum Source {
    /// A 🤗 checkpoint directory.
    Safetensors(StSet),
    /// A single GGUF file.
    Gguf(Box<GgufFile>),
}

impl Source {
    /// Opens whichever container `path` names.
    ///
    /// A `.gguf` extension picks the GGUF reader and anything else is treated
    /// as a checkpoint directory. Dispatching on the extension rather than
    /// sniffing the magic keeps the error useful: a mistyped directory should
    /// say the config is missing, not that the magic was not `GGUF`.
    fn open(path: &Path) -> Result<Self, TranslateError> {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            Ok(Self::Gguf(Box::new(GgufFile::open(path)?)))
        } else {
            Ok(Self::Safetensors(StSet::open(path)?))
        }
    }

    fn config(&self) -> Result<LlamaConfig, TranslateError> {
        Ok(match self {
            Self::Safetensors(st) => LlamaConfig::from_dir(st.root())?,
            Self::Gguf(f) => LlamaConfig::from_gguf(f)?,
        })
    }

    fn tokenizer(&self) -> Result<Tokenizer, TranslateError> {
        Ok(match self {
            Self::Safetensors(st) => Tokenizer::from_dir(st.root())?,
            Self::Gguf(f) => Tokenizer::from_gguf(f)?,
        })
    }

    fn weights(&self, cfg: &LlamaConfig) -> Result<LlamaWeights, TranslateError> {
        Ok(match self {
            Self::Safetensors(st) => LlamaWeights::load(st, cfg)?,
            Self::Gguf(f) => LlamaWeights::from_gguf(f, cfg)?,
        })
    }

    /// One tensor as f16, with the GGUF's rope permutation undone.
    fn f16(&self, b: &Bound, cfg: &LlamaConfig) -> Result<Vec<u16>, TranslateError> {
        Ok(match self {
            Self::Safetensors(st) => st.tensor_f16(&b.name)?,
            Self::Gguf(f) => {
                let raw = f.tensor_f16(&b.name)?;
                if xabe_llama::gguf::is_rope_permuted(&b.name) {
                    // `q` is divided into query heads and `k` into key-value
                    // heads. They are the same number on this checkpoint, so
                    // the distinction costs nothing here and is the whole
                    // difference on a grouped-query one.
                    let heads = if b.name.ends_with(".attn_q.weight") {
                        cfg.num_attention_heads
                    } else {
                        cfg.num_key_value_heads
                    };
                    xabe_llama::gguf::unpermute_rope(&raw, b.shape[0], b.shape[1], heads)
                } else {
                    raw
                }
            }
        })
    }

    /// One tensor's blocks, byte for byte, if it is stored in blocks at all.
    ///
    /// `None` for a safetensors checkpoint, which has no block formats, and
    /// for the unquantized GGUF widths. The rope permutation is undone here
    /// too and without unpacking, because it moves whole rows and a quantized
    /// row is a whole number of blocks - see
    /// [`xabe_llama::gguf::unpermute_rope_bytes`].
    fn packed(
        &self,
        b: &Bound,
        cfg: &LlamaConfig,
    ) -> Result<Option<(Vec<u8>, Quant)>, TranslateError> {
        let Self::Gguf(f) = self else {
            return Ok(None);
        };
        let Some(ty) = b.packed.and_then(|t| Quant::from_id(t as u32)) else {
            return Ok(None);
        };
        let raw = f.tensor_bytes(&b.name)?;
        let bytes = if xabe_llama::gguf::is_rope_permuted(&b.name) {
            let heads = if b.name.ends_with(".attn_q.weight") {
                cfg.num_attention_heads
            } else {
                cfg.num_key_value_heads
            };
            xabe_llama::gguf::unpermute_rope_bytes(raw, b.shape[0], ty.bytes(b.shape[1]), heads)
        } else {
            raw.to_vec()
        };
        Ok(Some((bytes, ty)))
    }

    /// One tensor as f32. Never a permuted one - the embedding, the norms and
    /// nothing else reach this.
    fn f32(&self, b: &Bound) -> Result<Vec<f32>, TranslateError> {
        debug_assert!(
            !xabe_llama::gguf::is_rope_permuted(&b.name),
            "a permuted tensor must go through f16, which is where the permutation is undone"
        );
        Ok(match self {
            Self::Safetensors(st) => st.tensor_f32(&b.name)?,
            Self::Gguf(f) => f.tensor_f32(&b.name)?,
        })
    }
}

/// The prompt this checkpoint was fine-tuned on.
///
/// From the model card, and reproduced exactly including both newlines: the
/// trailing one is load-bearing, and the card says so. The `{BOS}` it shows is
/// the tokenizer's, added by [`Translator::prompt_ids`].
pub const TEMPLATE: &str = "[TRANS]\n{src}\n[/TRANS]\n[{tgt}]\n";

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
/// Both reach the same kernel and both are staged to f16 inside it; what
/// differs is what occupies VRAM between calls. At 13 B that difference is the
/// difference between 26.5 GB and 7.9 GB, which is the difference between
/// needing a card to itself and sharing one with the rest of the pipeline.
enum GWeight {
    /// Rounded to f16 at load. What a safetensors or f16 checkpoint gives.
    F16(CudaSlice<u16>),
    /// The checkpoint's blocks, byte for byte, unpacked inside the matmul.
    Packed {
        /// The packed bytes.
        data: CudaSlice<u8>,
        /// Which layout reads them.
        ty: Quant,
    },
}

/// The token table on the device, full width or in the file's blocks.
///
/// The chat model has the same pair for the same reason; see `xabe-chat`.
enum GEmbed {
    /// Full width, what a safetensors or unquantized GGUF gives.
    F32(CudaSlice<f32>),
    /// The checkpoint's blocks, unpacked a row at a time on lookup.
    Packed {
        /// The packed bytes.
        data: CudaSlice<u8>,
        /// Which layout reads them.
        ty: Quant,
    },
}

/// A projection, on the device.
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

/// One transformer block, on the device.
/// The attention projections, grouped into as few products as the checkpoint
/// allows.
///
/// Batching needs identical `(in_dim, out_dim, block format)`. This model is
/// multi-head, so all three projections are the same width and the only thing
/// that can separate them is the format - and Q4_K_M stores `attn_v` as Q6_K
/// in half the layers, so half fuse all three and half fuse q with k and run v
/// alone. `xabe-chat` has the same type for the same reason and reaches a
/// different grouping, because a grouped-query model gives `attn_q` a
/// different width from `attn_k`.
///
/// The point is block count. One 128-token prompt is one row of tiles, so a
/// 5120-wide projection is 40 blocks on a machine that runs 144 at once, and
/// the answer used to be splitting the contraction - which buys blocks at the
/// cost of a partial buffer and a reduction pass. Three projections issued as
/// one product buy the same blocks with neither.
struct GAttn {
    /// One entry per product, in q, k, v order.
    groups: Vec<GGroup>,
    /// Where each of q, k, v lands: which group, and which element of it.
    ///
    /// `q` is always `(0, 0)`, which is what lets the query keep reading the
    /// output buffer from its start with no offset of its own.
    at: [(usize, usize); 3],
}

/// One batched product: `count` matrices of identical shape in one allocation.
struct GGroup {
    w: GLinear,
    count: usize,
}

/// The gate and the up projection, together where the checkpoint allows it.
///
/// They are the same shape and, in every checkpoint here, the same block
/// format - so they are one batched product over one activation, and the SiLU
/// gate reads the two halves of its output. That is one launch a layer instead
/// of two on each side, and at a single decoded row a launch is most of what a
/// kernel this size costs. `Split` is the fallback for a checkpoint that
/// quantizes them differently, which is a thing `Q4_K_M` does to other pairs.
enum GMlp {
    Fused(GGroup),
    Split(GLinear, GLinear),
}

struct GLayer {
    attn_norm: CudaSlice<f32>,
    attn: GAttn,
    o: GLinear,
    ffn_norm: CudaSlice<f32>,
    mlp: GMlp,
    down: GLinear,
}

/// The model, loaded onto one card.
///
/// # Why the weights are f16 and this is not a choice
///
/// The checkpoint is bf16 and `sm_75` has no bf16 at all. Widening to f32
/// would be 53 GB against a 48 GB card. f16 is the only width the model fits
/// in, which is why `xabe_st::StFile::tensor_f16` exists and why its range
/// check is not optional - see `docs/MODEL.md`.
pub struct Translator {
    gpu: Gpu,
    cfg: LlamaConfig,
    tokenizer: Tokenizer,
    /// `[vocab, hidden]`, for the lookup.
    ///
    /// Read rather than multiplied: a token's vector goes straight into the
    /// residual stream. Full width from a safetensors or f16 file; from a
    /// quantized GGUF it stays in its blocks and the gather unpacks the rows
    /// it reads - which is 1.1 GB against 160 MB at this vocabulary, and the
    /// same numbers either way, since the blocks decode to exactly the f32
    /// that used to be uploaded. See docs/BENCHMARKS.md.
    embed: GEmbed,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The output projection, which this checkpoint does *not* tie to the
    /// embedding - `tie_word_embeddings` is false.
    lm_head: GLinear,
}

/// The keys and values a decode step reuses.
pub struct Cache {
    /// Per layer, head-major and rotated before storing: keys are
    /// `[heads, capacity, head_dim]` and values `[heads, head_dim, capacity]`,
    /// which are the two shapes attention reads. Rearranging them per step was
    /// four kernels and four allocations a layer.
    /// f16, which on this model is the larger of the two savings it makes.
    ///
    /// The translator has no grouped-query attention - 40 key heads for 40
    /// query heads - so its cache is five times the chat model's per token and
    /// 6.25 GiB at a doubled 4 096 capacity. Halving that was named in
    /// docs/BENCHMARKS.md as the first thing to try if the card ever got
    /// tight, and it also decodes with a *single* row, which is the shape the
    /// f16 mat-vec is fastest at. See docs/BENCHMARKS.md for both numbers.
    k: Vec<CudaSlice<u16>>,
    v: Vec<CudaSlice<u16>>,
    /// Tokens the buffers have room for, which is not how many they hold.
    ///
    /// Doubled on growth. Growing by exactly the tokens added meant an
    /// allocation, a zeroing and a full copy of the cache for every layer of
    /// every token - quadratic in the context.
    cap: usize,
    len: usize,
    /// The fused decode attention's partials and counters; see
    /// `Gpu::attn_decode_f16`. Per cache, so two sequences decoding by turns
    /// never share a counter.
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
}

impl Translator {
    /// Loads a checkpoint directory onto CUDA device `ordinal`.
    ///
    /// About 27 GB of transfers, and it takes as long as that sounds. The
    /// alternative - mapping and letting the kernels fault pages in - would
    /// move the cost to the first translation instead of the load, which is
    /// the wrong place for a service that is started once.
    pub fn open(path: &Path, ordinal: usize) -> Result<Self, TranslateError> {
        Self::open_with(path, ordinal, Packing::default())
    }

    /// The same, choosing how a quantized checkpoint is held.
    pub fn open_with(
        path: &Path,
        ordinal: usize,
        packing: Packing,
    ) -> Result<Self, TranslateError> {
        let gpu = Gpu::open(ordinal)?;
        let src = Source::open(path)?;
        let cfg = src.config()?;
        // The schema binds grouped-query checkpoints; this forward pass does
        // not map several query heads onto one key-value head, so it refuses
        // one here rather than indexing off the end of the cache later. The
        // check lives at the engine and not in `check()` because a shape is a
        // fact about the file and this is a fact about the arithmetic.
        cfg.refuse_grouped_query()?;
        let tokenizer = src.tokenizer()?;
        let w = src.weights(&cfg)?;

        let narrow = |b: &Bound| -> Result<CudaSlice<u16>, TranslateError> {
            Ok(gpu.upload_u16(&src.f16(b, &cfg)?)?)
        };
        let wide =
            |b: &Bound| -> Result<CudaSlice<f32>, TranslateError> { Ok(gpu.upload(&src.f32(b)?)?) };
        // Packed when the container packs it and the kernel reads that layout;
        // f16 otherwise. A safetensors checkpoint always takes the second
        // branch, so this changes nothing about how the 🤗 directory loads.
        let lin = |b: &Bound| -> Result<GLinear, TranslateError> {
            let w = match if packing == Packing::Packed {
                src.packed(b, &cfg)?
            } else {
                None
            } {
                Some((bytes, ty)) => GWeight::Packed {
                    data: gpu.upload_quant(ty, &bytes)?,
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

        // q, k and v as one product where their shapes and formats allow it.
        // See [`GAttn`].
        let fused = |bs: &[&Bound]| -> Result<GGroup, TranslateError> {
            let head = bs[0];
            let packed = if packing == Packing::Packed {
                src.packed(head, &cfg)?.map(|(_, ty)| ty)
            } else {
                None
            };
            let w = match packed {
                Some(ty) => {
                    let mut all = Vec::new();
                    for b in bs {
                        let (bytes, bt) = src.packed(b, &cfg)?.expect("a packed sibling");
                        debug_assert_eq!(bt, ty, "a group with two formats");
                        all.extend_from_slice(&bytes);
                    }
                    GWeight::Packed {
                        data: gpu.upload_quant(ty, &all)?,
                        ty,
                    }
                }
                None => {
                    let mut all = Vec::new();
                    for b in bs {
                        all.extend_from_slice(&src.f16(b, &cfg)?);
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
        let attn = |a: &xabe_llama::Attention| -> Result<GAttn, TranslateError> {
            let bs = [&a.q, &a.k, &a.v];
            let key = |b: &Bound| {
                (
                    b.shape.clone(),
                    (packing == Packing::Packed).then_some(b.packed).flatten(),
                )
            };
            let mut runs: Vec<Vec<usize>> = Vec::new();
            let mut at = [(0usize, 0usize); 3];
            for i in 0..3 {
                let joins = runs
                    .last()
                    .and_then(|r| r.last())
                    .is_some_and(|&j| key(bs[i]) == key(bs[j]));
                match joins {
                    true => runs.last_mut().expect("a run to join").push(i),
                    false => runs.push(vec![i]),
                }
                at[i] = (runs.len() - 1, runs[runs.len() - 1].len() - 1);
            }
            let mut groups = Vec::with_capacity(runs.len());
            for r in &runs {
                let bs: Vec<&Bound> = r.iter().map(|&i| bs[i]).collect();
                groups.push(fused(&bs)?);
            }
            Ok(GAttn { groups, at })
        };

        let mut layers = Vec::with_capacity(w.layers.len());
        for l in &w.layers {
            layers.push(GLayer {
                attn_norm: wide(&l.attn_norm)?,
                attn: attn(&l.attn)?,
                o: lin(&l.attn.o)?,
                ffn_norm: wide(&l.ffn_norm)?,
                mlp: {
                    // The same test `attn` uses to decide a run: identical
                    // shape and, when packed, identical block format.
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

        let model = Self {
            embed: match if packing == Packing::Packed {
                src.packed(&w.embed_tokens, &cfg)?
            } else {
                None
            } {
                Some((bytes, ty)) => GEmbed::Packed {
                    data: gpu.upload_quant(ty, &bytes)?,
                    ty,
                },
                None => GEmbed::F32(wide(&w.embed_tokens)?),
            },
            layers,
            norm: wide(&w.norm)?,
            lm_head: lin(&w.lm_head)?,
            cfg,
            tokenizer,
            gpu,
        };
        tracing::info!(device = ordinal, "translator on the device");
        Ok(model)
    }

    /// The geometry this model was bound against.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// The tokenizer, for building prompts and reading answers.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The token ids for one translation request, `<s>` included.
    pub fn prompt_ids(&self, source: &str, target: &str) -> Vec<u32> {
        let prompt = TEMPLATE.replace("{src}", source).replace("{tgt}", target);
        let mut ids = vec![self.tokenizer.bos()];
        ids.extend(self.tokenizer.encode(&prompt));
        ids
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
    ) -> Result<(CudaSlice<f32>, Option<Q8>), TranslateError> {
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
    ) -> Result<CudaSlice<f32>, TranslateError> {
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
    ) -> Result<CudaSlice<f32>, TranslateError> {
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

    /// Runs `ids` through the model and returns the logits, `[n, vocab]`.
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, TranslateError> {
        Ok(self.run(ids, cache, 0, false)?.0)
    }

    /// The same, but only the last position's logits, `[1, vocab]`.
    ///
    /// What generation wants: the other rows are run so the cache is filled,
    /// and their logits are thrown away. See the note at the end of `run`.
    pub fn forward_last(
        &self,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, TranslateError> {
        Ok(self.run(ids, cache, 0, true)?.0)
    }

    /// The same as [`Self::forward`], also returning the first `taps` block
    /// outputs on the host.
    ///
    /// On the public surface for the same reason the ASR's are: "the model is
    /// wrong" is not a fact anyone can act on, and "layer 7 is wrong" is.
    pub fn forward_tapped(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
    ) -> Result<(CudaSlice<f32>, Vec<Vec<f32>>), TranslateError> {
        self.run(ids, cache, taps, false)
    }

    fn run(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
        last_only: bool,
    ) -> Result<(CudaSlice<f32>, Vec<Vec<f32>>), TranslateError> {
        let (h_dim, heads) = (self.cfg.hidden_size, self.cfg.num_attention_heads);
        let hd = self.cfg.head_dim();
        let (n, past) = (ids.len(), cache.len);
        if past + n > self.cfg.max_position_embeddings {
            return Err(TranslateError::PastTheEnd {
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

        // The block output the next normalisation still has to add; see where
        // it is set. The residual add and the normalisation read the same row,
        // so they are one pass.
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
            // Re-strided, not copied - the cache is head-major and `cap` is the
            // stride between heads. See `Gpu::cache_grow`, and the chat model,
            // which has the same growth and had the same bug.
            let keys = cache.k.iter_mut().map(|s| (s, false));
            let values = cache.v.iter_mut().map(|s| (s, true));
            for (slot, transposed) in keys.chain(values) {
                let mut grown = self.gpu.zeros_f16(want * h_dim)?;
                // A fresh conversation grows from empty, and eighty zero-byte
                // copies are still eighty launches.
                if past > 0 {
                    self.gpu
                        .cache_grow_f16(slot, &mut grown, heads, hd, was, want, past, transposed)?;
                }
                *slot = grown;
            }
            cache.cap = want;
        }
        let mut tapped = Vec::with_capacity(taps);
        for (i, l) in self.layers.iter().enumerate() {
            // One int8 twin for three projections, taken by the normalisation
            // that produced the activation.
            let (x, xq) = match pending.take() {
                Some((x, q)) => (x, Some(q)),
                None => self.normed(&mut h, residual.take().as_ref(), n, h_dim, &l.attn_norm)?,
            };
            let xo = Self::operand(&x, xq.as_ref());
            // One product per group rather than one per projection. Each
            // element of a batched product writes a contiguous block of the
            // same output, so q, k and v are located by an offset into one of
            // these buffers and nothing is copied to separate them.
            let mut proj = Vec::with_capacity(l.attn.groups.len());
            for g in &l.attn.groups {
                proj.push(self.gpu.gemm_batched(
                    xo,
                    g.w.operand(),
                    None,
                    Batch {
                        count: g.count,
                        // Zero: every matrix of the group multiplies the same
                        // normalised activation, which is what lets it be
                        // quantized once instead of once a projection.
                        a: 0,
                        w: g.w.in_dim * g.w.out_dim,
                        out: n * g.w.out_dim,
                        w_row: 0,
                    },
                    n,
                    g.w.in_dim,
                    g.w.out_dim,
                )?);
            }
            let block = |j: usize| -> (usize, usize) {
                let (gi, ei) = l.attn.at[j];
                (gi, ei * n * l.attn.groups[gi].w.out_dim)
            };
            let (qg, q_off) = block(0);
            let (kg, k_off) = block(1);
            let (vg, v_off) = block(2);
            debug_assert_eq!((qg, q_off), (0, 0), "the query leads its group");

            // Rotated before caching, because the position is absolute: a key
            // stored unrotated would be rotated again by the wrong offset on
            // every later step.
            // Scattered straight into the layout attention reads.
            if first {
                cache.k.push(self.gpu.zeros_f16(cache.cap * h_dim)?);
                cache.v.push(self.gpu.zeros_f16(cache.cap * h_dim)?);
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
                    None,
                    heads,
                    heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                    &mut cache.k[i],
                    &mut cache.v[i],
                    cap,
                )?;
            } else {
                self.gpu.rope(
                    &mut proj[qg],
                    q_off,
                    n,
                    heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                )?;
                self.gpu.rope(
                    &mut proj[kg],
                    k_off,
                    n,
                    heads,
                    hd,
                    self.cfg.rope_theta,
                    past,
                )?;
                self.gpu.cache_append_f16(
                    &proj[kg],
                    k_off,
                    &mut cache.k[i],
                    n,
                    heads,
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
                    heads,
                    hd,
                    cap,
                    past,
                    true,
                )?;
            }
            let tk = past + n;

            // A single step's queries are already `[head][1][d]`; only a
            // multi-row pass has anything to split.
            // The query is at the start of its buffer, so `split_heads` reads
            // it from zero and a single step reads it as the row it is.
            let q = &proj[qg];

            // A prompt takes the fused attention: scores, mask, softmax and
            // the value product in one kernel, reading the query buffer and
            // the caches in place - no head split, no score matrix, no merge.
            // A single step takes its own fused kernel, which does the same
            // for one row without writing the score row: three launches a
            // layer became one. The chain below is what is left for a
            // geometry neither kernel covers, which this checkpoint's never
            // is.
            if n == 1 && (hd == 128 || hd == 64) {
                // The context and its int8 twin in one pass; the projection
                // that reads the twin is packed in every checkpoint here, and
                // an unpacked one ignores it.
                let (ctx, cq) = self.gpu.attn_decode_f16_q(
                    q,
                    &cache.k[i],
                    &cache.v[i],
                    heads,
                    heads,
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
                    heads,
                    hd,
                    cap,
                    (hd as f32).powf(-0.5),
                    true,
                )?;
                let out = self.project(Operand::F32(&ctx), &l.o, n)?;
                residual = Some(out);
            } else {
                let qh = match n {
                    1 => None,
                    _ => Some(self.gpu.split_heads(q, n, heads, hd)?),
                };

                let mut scores = self.gpu.gemm_batched(
                    Operand::F32(qh.as_ref().unwrap_or(q)),
                    Operand::F16(&cache.k[i]),
                    None,
                    Batch {
                        count: heads,
                        a: n * hd,
                        w: cap * hd,
                        out: n * tk,
                        w_row: 0,
                    },
                    n,
                    hd,
                    tk,
                )?;
                // Llama scales the *scores*, not the query - the opposite of
                // Whisper, and the same algebra. The scale, the mask and the
                // softmax are one pass; see `Gpu::softmax_causal`.
                self.gpu.softmax_causal(
                    &mut scores,
                    heads * n,
                    tk,
                    n,
                    tk - n,
                    (hd as f32).powf(-0.5),
                )?;

                let ctx = self.gpu.gemm_batched(
                    Operand::F32(&scores),
                    Operand::F16(&cache.v[i]),
                    None,
                    // `w_row` is `cap`, not `tk`: the values sit in a buffer with
                    // room for more positions than are in it.
                    Batch {
                        count: heads,
                        a: n * tk,
                        w: hd * cap,
                        out: n * hd,
                        w_row: cap,
                    },
                    n,
                    tk,
                    hd,
                )?;
                // The merge takes the context's int8 twin in the same pass when
                // the output projection is packed and will read it - the same
                // reasoning that gives the normalisation and the gating theirs. A
                // single step has nothing to merge, and its twin is taken by
                // `project` as before.
                let packed_o = matches!(l.o.w, GWeight::Packed { .. });
                let (ctx, cq) = match n {
                    1 => (ctx, None),
                    _ if packed_o && (heads * hd).is_multiple_of(256) => {
                        let (c, q) = self.gpu.merge_heads_q(&ctx, n, heads, hd)?;
                        (c, Some(q))
                    }
                    _ => (self.gpu.merge_heads(&ctx, n, heads, hd)?, None),
                };
                // Not added here: the next normalisation reads `h + out` and
                // nothing between now and then does.
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
            // then the gate reads its own output; see `GMlp`. `pair` says which
            // shape the SiLU below has to take, because the fused buffer holds
            // `[2, n, inter]` and the split one holds two of `[n, inter]`.
            let (mut gate, up) = match &l.mlp {
                GMlp::Fused(g) => (self.project_group(xo, g, n)?, None),
                GMlp::Split(gw, uw) => (self.project(xo, gw, n)?, Some(self.project(xo, uw, n)?)),
            };
            // The twin is taken at every row count, not only decode's: the
            // tiled integer kernel reads the same codes the mat-vec does, and
            // quantizing inside `project` instead re-reads the whole
            // activation - 28 MB a layer at 512 tokens. Only for a packed
            // `down`, because an f16 one would leave the codes unread.
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

            // A tap is the block's output, so the pending add has to be settled
            // before it is read.
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
        // Only the last row predicts the next token. A prompt projected every
        // row through a 32000-wide head and threw all but one away; the rows
        // are still *run* through every block - that is what fills the cache -
        // they are just not projected.
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

impl Translator {
    /// How many recent tokens the repetition penalty looks back over.
    ///
    /// llama.cpp's `repeat_last_n` default, which is what the pipeline's
    /// translator has been running with.
    pub const REPEAT_LAST_N: usize = 64;

    /// llama.cpp's `repeat_penalty` default for this stage, from `gateway.py`.
    pub const REPEAT_PENALTY: f32 = 1.1;

    /// Greedy decoding from a prompt, returning the tokens produced.
    ///
    /// `penalty` is llama.cpp's `repeat_penalty`; 1.0 turns it off. It is a
    /// parameter rather than a constant because the two things this has to
    /// agree with disagree with each other: the captured 🤗 oracle is pure
    /// greedy, and the `llama-server` the pipeline runs today passes 1.1. Both
    /// comparisons are worth making, so both are reachable.
    /// `stops` are substrings of the *decoded* answer that end it. They are the
    /// same ones the caller would cut at afterwards, and checking them here is
    /// the difference between paying for the tokens after the answer and not:
    /// this checkpoint closes its tag and then keeps going, and nothing in the
    /// loop below notices, because `</s>` never arrives. With `max_new` at 256
    /// that is most of a translation's cost spent on text about to be thrown
    /// away. Decoding the answer again each step is O(n) on n <= 256 tokens
    /// against a 28 ms forward pass, so it does not register.
    pub fn generate(
        &self,
        ids: &[u32],
        max_new: usize,
        penalty: f32,
        stops: &[&str],
    ) -> Result<Vec<u32>, TranslateError> {
        let mut cache = self.cache();
        let mut pending = ids.to_vec();
        let mut out = Vec::new();
        // `<pad>` terminates as well as `</s>`. That is this checkpoint's own
        // convention, from the model card, and a decoder that stops only on
        // `</s>` runs to the token limit on every second translation.
        let stop = [
            self.tokenizer.eos(),
            self.tokenizer.special("<pad>").unwrap_or(u32::MAX),
        ];

        for _ in 0..max_new {
            let logits = self.forward_last(&pending, &mut cache)?;
            let mut row = self.gpu.download(&logits)?;
            if penalty != 1.0 {
                // llama.cpp's rule, and it is not the obvious one: a positive
                // logit is *divided* by the penalty and a negative one
                // multiplied, so that both move towards zero. Dividing both
                // would make an already-unlikely token more likely.
                let seen = ids.iter().chain(&out);
                let start = (ids.len() + out.len()).saturating_sub(Self::REPEAT_LAST_N);
                // The newline is exempt. That is llama.cpp's `penalize_nl`,
                // which defaults to false - "consider newlines as a repeatable
                // token" - and it is not cosmetic here: this prompt template is
                // four lines, so a penalised newline is a newline the model has
                // already seen three times, and it answers by reaching for
                // punctuation instead of ending the line.
                let newline = self.tokenizer.byte_id(b'\n');
                for &id in seen.skip(start) {
                    if Some(id) == newline {
                        continue;
                    }
                    let v = &mut row[id as usize];
                    *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
                }
            }
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
                .map(|(i, _)| i as u32)
                .expect("the vocabulary is not empty");
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            if !stops.is_empty() {
                let so_far = self.tokenizer.decode(&out, true);
                if stops.iter().any(|s| so_far.contains(s)) {
                    break;
                }
            }
            pending = vec![next];
        }
        Ok(out)
    }

    /// Translates one sentence into `target`.
    ///
    /// `target` is the model card's language code: `ZH`, `EN`, `POJ`, `HL` or
    /// `HAN`. The answer is cut at `[/` or at a newline followed by `[`, which
    /// are `gateway.py`'s two stop strings: the model is trained to close its
    /// own tag and there is nothing useful after it. Both are needed, not one -
    /// the model sometimes opens the next block instead of closing this one.
    pub fn translate(
        &self,
        source: &str,
        target: &str,
        max_new: usize,
        penalty: f32,
    ) -> Result<String, TranslateError> {
        const STOPS: [&str; 2] = ["[/", "\n["];
        let ids = self.prompt_ids(source, target);
        let out = self.generate(&ids, max_new, penalty, &STOPS)?;
        let text = self.tokenizer.decode(&out, true);
        let cut = STOPS
            .iter()
            .filter_map(|s| text.find(s))
            .min()
            .unwrap_or(text.len());
        tracing::debug!(tokens = out.len(), "translated");
        Ok(text[..cut].trim().to_string())
    }
}
