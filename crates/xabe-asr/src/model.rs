//! The Whisper forward pass on CUDA.

use crate::AsrError;
use std::path::Path;
use xabe_cuda::{Batch, CudaSlice, Gpu, Operand};
use xabe_st::StSet;
use xabe_whisper::{
    Attention, Conv1d, DecoderLayer, EncoderLayer, Frontend, GenerationConfig, LayerNorm, Linear,
    Tokenizer, WhisperConfig, WhisperWeights,
};

/// Layer-normalisation epsilon. `torch.nn.LayerNorm`'s default, and Whisper
/// takes it as it comes.
const EPS: f32 = 1e-5;

/// A projection, on the device.
///
/// The weight is stored as f16. That is not a precision decision taken lightly
/// and it is barely a decision at all on the tiled path: the matmul rounds an
/// F32 weight to f16 on the way into shared memory on every trip, so storing
/// F32 bought exactly nothing and cost twice the global traffic. It *is* a
/// real decision on the decode shapes, where the scalar path would otherwise
/// accumulate the weight exactly - and that is measured, in `oracle.rs` and in
/// the transcripts, rather than assumed.
struct GLinear {
    w: CudaSlice<u16>,
    b: Option<CudaSlice<f32>>,
    in_dim: usize,
    out_dim: usize,
}

/// A layer normalisation's scale and shift, on the device.
struct GNorm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// The four projections of one attention block, on the device.
struct GAttention {
    q: GLinear,
    k: GLinear,
    v: GLinear,
    out: GLinear,
}

/// One encoder block, on the device.
struct GEncoderLayer {
    attn_ln: GNorm,
    attn: GAttention,
    ffn_ln: GNorm,
    fc1: GLinear,
    fc2: GLinear,
}

/// One decoder block, on the device.
struct GDecoderLayer {
    attn_ln: GNorm,
    attn: GAttention,
    cross_ln: GNorm,
    cross: GAttention,
    ffn_ln: GNorm,
    fc1: GLinear,
    fc2: GLinear,
}

/// A convolution, on the device, already flattened for the matmul.
struct GConv {
    /// `[out_ch, in_ch * k]` - the same bytes as `[out_ch, in_ch, k]`, which is
    /// exactly the contraction `im2col` produces.
    w: CudaSlice<u16>,
    b: CudaSlice<f32>,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
}

/// The model, loaded onto one card.
///
/// # Why there is no CPU path
///
/// One 30-second window is about 2.2 TFLOP through the encoder alone. The
/// scalar kernels in `xabe-dsp` run at something under 2 GFLOP/s, which is
/// twenty minutes per utterance - not a slow option but a fictional one. So
/// the ASR stage is GPU-only, refused by name at preflight rather than offered
/// and then unusable, and its differential proof is the captured oracle
/// directly rather than a scalar twin of the whole model. The *kernels* still
/// have their twins; it is only the assembly of them that does not.
pub struct AsrModel {
    gpu: Gpu,
    cfg: WhisperConfig,
    decoding: GenerationConfig,
    frontend: Frontend,
    tokenizer: Tokenizer,
    conv1: GConv,
    conv2: GConv,
    enc_pos: CudaSlice<f32>,
    enc_layers: Vec<GEncoderLayer>,
    enc_ln: GNorm,
    /// `[vocab, d_model]`, for the embedding lookup.
    ///
    /// Kept at full precision because it is *read*, not multiplied: a token's
    /// vector goes straight into the residual stream, where rounding it would
    /// perturb the input rather than the arithmetic.
    embed_tokens: CudaSlice<f32>,
    /// The same table as the tied output projection, rounded once.
    ///
    /// 133 MB against 265, and the decoder reads all of it for every token it
    /// emits - which at one token a step is pure bandwidth.
    embed_logits: CudaSlice<u16>,
    dec_pos: CudaSlice<f32>,
    dec_layers: Vec<GDecoderLayer>,
    dec_ln: GNorm,
}

/// The keys and values a decode step reuses.
///
/// Cross-attention's halves are computed once for the whole utterance: they
/// depend only on the encoder's output, so recomputing them per token would be
/// 32 layers of a 1500x1280 projection per token, which is most of the cost of
/// the decoder. Self-attention's grow by one row a step.
pub struct Cache {
    /// Per layer, `[heads, cap, head_dim]` and `[heads, head_dim, cap]`.
    ///
    /// Head-major and transposed on the value side, which is the layout
    /// attention reads. The first version of this stored what the projection
    /// produced - `[len, d_model]` - and rearranged it into this shape every
    /// step, with a fresh allocation, a zeroing and a full copy of the whole
    /// cache to grow it. That is four allocations and six launches a layer,
    /// 128 and 192 a token across the stack, all producing tensors thrown away
    /// before the next one. The chat model was found to have the same fault
    /// and fixed the same way; docs/BENCHMARKS.md has both.
    self_k: Vec<CudaSlice<f32>>,
    self_v: Vec<CudaSlice<f32>>,
    /// Per layer, `[heads, 1500, head_dim]` and `[heads, head_dim, 1500]`,
    /// packed - every decode step reads all of both.
    cross_k: Vec<CudaSlice<u16>>,
    cross_v: Vec<CudaSlice<u16>>,
    /// How many tokens the self-attention halves hold.
    len: usize,
    /// How many they have room for, which is not the same number.
    ///
    /// Allocated once at `max_target_positions` rather than doubled as the
    /// chat model's is: 448 positions of 1280 floats is 2.3 MB a layer and
    /// 73 MB across the stack, which is small enough that growing it would be
    /// machinery bought with nothing.
    cap: usize,
}

impl Cache {
    /// How many tokens have been decoded into it.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been decoded yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsrModel {
    /// Loads a checkpoint directory onto CUDA device `ordinal`.
    pub fn open(dir: &Path, ordinal: usize) -> Result<Self, AsrError> {
        let gpu = Gpu::open(ordinal)?;
        let cfg = WhisperConfig::from_dir(dir)?;
        let decoding = GenerationConfig::from_dir(dir)?;
        let frontend = Frontend::new(&cfg);
        let tokenizer = Tokenizer::from_dir(dir)?;

        let st = StSet::open(dir)?;
        let w = WhisperWeights::load(&st, &cfg)?;
        let up = |x: &[f32]| gpu.upload(x);
        let up16 = |x: &[f32]| gpu.upload_f16(x);
        let lin = |l: &Linear| -> Result<GLinear, AsrError> {
            Ok(GLinear {
                w: up16(l.weight)?,
                b: l.bias.map(up).transpose()?,
                in_dim: l.in_dim,
                out_dim: l.out_dim,
            })
        };
        let nrm = |n: &LayerNorm| -> Result<GNorm, AsrError> {
            Ok(GNorm {
                w: up(n.weight)?,
                b: up(n.bias)?,
            })
        };
        let att = |a: &Attention| -> Result<GAttention, AsrError> {
            Ok(GAttention {
                q: lin(&a.q)?,
                k: lin(&a.k)?,
                v: lin(&a.v)?,
                out: lin(&a.out)?,
            })
        };
        let cnv = |c: &Conv1d| -> Result<GConv, AsrError> {
            Ok(GConv {
                w: up16(c.weight)?,
                b: up(c.bias)?,
                in_ch: c.in_ch,
                out_ch: c.out_ch,
                k: c.k,
                stride: c.stride,
            })
        };

        let mut enc_layers = Vec::with_capacity(w.enc_layers.len());
        for l in &w.enc_layers {
            let EncoderLayer {
                attn_ln,
                attn,
                ffn_ln,
                fc1,
                fc2,
            } = l;
            enc_layers.push(GEncoderLayer {
                attn_ln: nrm(attn_ln)?,
                attn: att(attn)?,
                ffn_ln: nrm(ffn_ln)?,
                fc1: lin(fc1)?,
                fc2: lin(fc2)?,
            });
        }
        let mut dec_layers = Vec::with_capacity(w.dec_layers.len());
        for l in &w.dec_layers {
            let DecoderLayer {
                attn_ln,
                attn,
                cross_ln,
                cross,
                ffn_ln,
                fc1,
                fc2,
            } = l;
            dec_layers.push(GDecoderLayer {
                attn_ln: nrm(attn_ln)?,
                attn: att(attn)?,
                cross_ln: nrm(cross_ln)?,
                cross: att(cross)?,
                ffn_ln: nrm(ffn_ln)?,
                fc1: lin(fc1)?,
                fc2: lin(fc2)?,
            });
        }

        let model = Self {
            conv1: cnv(&w.conv1)?,
            conv2: cnv(&w.conv2)?,
            enc_pos: up(w.enc_pos)?,
            enc_layers,
            enc_ln: nrm(&w.enc_ln)?,
            embed_tokens: up(w.embed_tokens)?,
            embed_logits: up16(w.embed_tokens)?,
            dec_pos: up(w.dec_pos)?,
            dec_layers,
            dec_ln: nrm(&w.dec_ln)?,
            cfg,
            decoding,
            frontend,
            tokenizer,
            gpu,
        };
        tracing::info!(device = ordinal, "asr model on the device");
        Ok(model)
    }

    /// The geometry this model was bound against.
    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    /// The decoding parameters the checkpoint ships.
    pub fn generation(&self) -> &GenerationConfig {
        &self.decoding
    }

    /// The mel frontend, so a caller can produce features without the model.
    pub fn frontend(&self) -> &Frontend {
        &self.frontend
    }

    /// The tokenizer, for building prefixes and reading transcripts.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// One projection.
    fn project(
        &self,
        x: Operand<'_>,
        l: &GLinear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, AsrError> {
        Ok(self.gpu.gemm_batched(
            x,
            Operand::F16(&l.w),
            l.b.as_ref(),
            Batch::single(rows * l.out_dim),
            rows,
            l.in_dim,
            l.out_dim,
        )?)
    }

    // Activations stay F32 on the way in. Rounding them first was implemented
    // and measured: 5 ms of 256, because the tiles a projection re-reads
    // already fit in this card's 6 MB L2, so the halved traffic was traffic
    // that never reached memory. See docs/BENCHMARKS.md.

    /// One layer normalisation, over rows of `d_model`.
    fn norm(&self, x: &CudaSlice<f32>, n: &GNorm, rows: usize) -> Result<CudaSlice<f32>, AsrError> {
        Ok(self
            .gpu
            .layer_norm(x, rows, self.cfg.d_model, &n.w, &n.b, EPS)?)
    }

    /// [`Self::norm`] or [`Self::norm_add`], depending on whether a sub-layer's
    /// output is still waiting to be added.
    ///
    /// Only the first normalisation of a stack has nothing waiting.
    fn normed(
        &self,
        h: &mut CudaSlice<f32>,
        res: Option<&CudaSlice<f32>>,
        n: &GNorm,
        rows: usize,
    ) -> Result<CudaSlice<f32>, AsrError> {
        match res {
            Some(r) => self.norm_add(h, r, n, rows),
            None => self.norm(h, n, rows),
        }
    }

    /// The residual sum and the normalisation that always follows it.
    ///
    /// Every normalisation in a block reads the residual stream immediately
    /// after a sub-layer added to it, and nothing in between reads it - so the
    /// sum is left for the pass that was going to be made anyway. `h` is
    /// updated in place because it is what the next sub-layer adds to.
    fn norm_add(
        &self,
        h: &mut CudaSlice<f32>,
        res: &CudaSlice<f32>,
        n: &GNorm,
        rows: usize,
    ) -> Result<CudaSlice<f32>, AsrError> {
        Ok(self
            .gpu
            .layer_norm_add(h, res, rows, self.cfg.d_model, &n.w, &n.b, EPS)?)
    }

    /// Attention, given queries already projected and split, and keys and
    /// values already split.
    ///
    /// `q` is `[heads, tq, head_dim]`, `k` is `[heads, cap, head_dim]` and `v`
    /// is `[heads, head_dim, cap]` - the transpose, because the context
    /// product reads the values down their time axis. Returns `[tq, d_model]`.
    ///
    /// `cap` is how many positions the key and value operands are *laid out*
    /// for, which for the self-attention cache is more than the `tk` in them.
    /// It is a stride, not a bound: the products contract over `tk`, so the
    /// untouched tail of the cache is skipped rather than zero-weighted. Pass
    /// `tk` where the operand is exactly as long as it is used, which is both
    /// attentions in the encoder and cross-attention in the decoder.
    // Shapes are arguments, not types - the same convention as the `xabe-dsp`
    // and `xabe-cuda` kernels this composes.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: Operand<'_>,
        k: Operand<'_>,
        v: Operand<'_>,
        tq: usize,
        tk: usize,
        cap: usize,
        heads: usize,
        causal: bool,
    ) -> Result<CudaSlice<f32>, AsrError> {
        let hd = self.cfg.d_model / heads;
        let mut scores = self.gpu.gemm_batched(
            q,
            k,
            None,
            Batch {
                count: heads,
                a: tq * hd,
                w: cap * hd,
                out: tq * tk,
                w_row: 0,
            },
            tq,
            hd,
            tk,
        )?;
        if causal {
            // Mask and softmax in one pass. With `tk - tq` keys already
            // cached, query `i` really sits at position `i + (tk - tq)`, which
            // is what the offset says. The scale is 1: Whisper scales the
            // query before the product and not the scores after it, for the
            // reason [`Self::queries`] gives.
            self.gpu
                .softmax_causal(&mut scores, heads * tq, tk, tq, tk - tq, 1.0)?;
        } else {
            self.gpu.softmax_rows(&mut scores, heads * tq, tk)?;
        }
        let ctx = self.gpu.gemm_batched(
            // The probabilities are *not* converted. They are read exactly
            // once - the context product has a single column tile at
            // `head_dim` 64 - so a conversion pass would cost 270 MB to save
            // 90. Every other operand here is read a dozen times, which is
            // what makes the trade go the other way for them.
            Operand::F32(&scores),
            v,
            None,
            Batch {
                count: heads,
                a: tq * tk,
                w: hd * cap,
                out: tq * hd,
                // Zero where the values are exactly as long as they are used,
                // which is the layout the packed cross-attention halves are in
                // and the only one the f16 path takes.
                w_row: if cap == tk { 0 } else { cap },
            },
            tq,
            tk,
            hd,
        )?;
        // `[head][tq][hd]` is `[tq][heads * hd]` when `tq` is 1, which is
        // every step of a greedy decode after the prefix. There is nothing to
        // merge and the launch is skipped.
        match tq {
            1 => Ok(ctx),
            _ => Ok(self.gpu.merge_heads(&ctx, tq, heads, hd)?),
        }
    }

    /// Queries, projected, scaled and split.
    ///
    /// The scale goes on the query before the product and not on the scores
    /// after it. Algebraically the same; not the same rounding, and the
    /// reference says so out loud - "Scaling is susceptible to floating point
    /// arithmetics' imprecisions which can lead to different results (this is
    /// dependent from model to model, e.g. whisper is one such case)". So it
    /// is copied where it belongs rather than moved somewhere tidier.
    fn queries(
        &self,
        x: Operand<'_>,
        a: &GAttention,
        t: usize,
        heads: usize,
    ) -> Result<CudaSlice<f32>, AsrError> {
        let hd = self.cfg.d_model / heads;
        let mut q = self.project(x, &a.q, t)?;
        self.gpu
            .scale_inplace(&mut q, t * self.cfg.d_model, (hd as f32).powf(-0.5))?;
        // One row is already `[head][1][hd]`, so the split has nothing to do.
        match t {
            1 => Ok(q),
            _ => Ok(self.gpu.split_heads(&q, t, heads, hd)?),
        }
    }

    /// A feed-forward block, applied in place on the residual stream.
    ///
    /// `res` is what the attention sub-layer produced and has not been added
    /// yet: the normalisation takes it, for the reason [`Self::norm_add`]
    /// gives. What this returns is this block's own output, unadded, for the
    /// next normalisation to take the same way - so a residual add survives
    /// only at the very end of a stack, where the last one is folded into the
    /// final normalisation instead.
    fn feed_forward(
        &self,
        h: &mut CudaSlice<f32>,
        res: &CudaSlice<f32>,
        ln: &GNorm,
        fc1: &GLinear,
        fc2: &GLinear,
        t: usize,
    ) -> Result<CudaSlice<f32>, AsrError> {
        let x = self.norm_add(h, res, ln, t)?;
        let mut inner = self.project(Operand::F32(&x), fc1, t)?;
        self.gpu.gelu(&mut inner, t * fc1.out_dim)?;
        self.project(Operand::F32(&inner), fc2, t)
    }

    /// The encoder, from log-mel features to `[1500, d_model]` on the device.
    ///
    /// `mel` is `[n_mels, n_frames]` row-major, which is what
    /// [`Frontend::log_mel`] produces.
    pub fn encode(&self, mel: &[f32]) -> Result<CudaSlice<f32>, AsrError> {
        Ok(self.encode_tapped(mel, 0)?.0)
    }

    /// The encoder, also returning the first `taps` block outputs on the host.
    ///
    /// This exists for the differential tests and for debugging, and it is on
    /// the public surface deliberately: "the encoder is wrong" is not a fact
    /// anyone can act on, and "layer 7 is wrong" is. Each tap costs a 7.7 MB
    /// download, so `taps` is a count rather than a flag.
    pub fn encode_tapped(
        &self,
        mel: &[f32],
        taps: usize,
    ) -> Result<(CudaSlice<f32>, Vec<Vec<f32>>), AsrError> {
        let (d, heads) = (self.cfg.d_model, self.cfg.encoder_attention_heads);
        let frames = self.cfg.n_frames();
        assert_eq!(mel.len(), self.cfg.num_mel_bins * frames, "mel shape");

        // The convolutions want time-major input, and so does everything after
        // them, so the one transpose happens here on the smallest tensor in the
        // pass rather than on a 1500x1280 activation later.
        let x = xabe_dsp::transpose(mel, self.cfg.num_mel_bins, frames);
        let mut h = self.gpu.upload(&x)?;

        for c in [&self.conv1, &self.conv2] {
            let t = h.len() / c.in_ch;
            // Width 3 with one of padding on each side: "same" at stride 1, and
            // exactly half the frames at stride 2.
            let (col, out_t) = self.gpu.im2col(&h, t, c.in_ch, c.k, c.stride, 1, 1)?;
            h = self.gpu.gemm_batched(
                Operand::F32(&col),
                Operand::F16(&c.w),
                Some(&c.b),
                Batch::single(out_t * c.out_ch),
                out_t,
                c.in_ch * c.k,
                c.out_ch,
            )?;
            self.gpu.gelu(&mut h, out_t * c.out_ch)?;
        }

        let t = self.cfg.max_source_positions;
        self.gpu.add_inplace(&mut h, &self.enc_pos, t * d)?;

        let mut tapped = Vec::with_capacity(taps);
        let hd = d / heads;
        // The encoder's window is 1500 positions, so one layer's scores are
        // 20 x 1500 x 1500 floats - 180 MB written by the score product, read
        // and written again by the softmax, and read once more by the context
        // product. Across 32 layers that is 23 GB of traffic carrying no
        // arithmetic. `flash_attn` never writes a score anywhere; see
        // docs/BENCHMARKS.md for what it was worth here.
        let fused = self.gpu.supports_flash(hd, heads, heads);
        // What the previous sub-layer produced and has not been added to the
        // residual stream yet: the next normalisation takes it, for the reason
        // `norm_add` gives. Only the first normalisation of the stack finds
        // nothing here.
        let mut res: Option<CudaSlice<f32>> = None;
        for (i, l) in self.enc_layers.iter().enumerate() {
            let x = self.normed(&mut h, res.take().as_ref(), &l.attn_ln, t)?;
            let k = self.gpu.split_heads(
                &self.project(Operand::F32(&x), &l.attn.k, t)?,
                t,
                heads,
                hd,
            )?;
            let v = self.gpu.split_heads_t(
                &self.project(Operand::F32(&x), &l.attn.v, t)?,
                t,
                heads,
                hd,
            )?;
            let ctx = if fused {
                // The scale stays on the query rather than moving to the
                // scores, for the reason [`Self::queries`] gives - so the
                // kernel is handed 1.0 and the rounding is the chain's. The
                // fused path also skips the query's head split and the
                // context's merge: it reads the projection buffer's layout
                // and writes it back.
                let mut q = self.project(Operand::F32(&x), &l.attn.q, t)?;
                self.gpu
                    .scale_inplace(&mut q, t * d, (hd as f32).powf(-0.5))?;
                self.gpu
                    .flash_attn(&q, &k, &v, t, 0, heads, heads, hd, t, 1.0, false)?
            } else {
                let q = self.queries(Operand::F32(&x), &l.attn, t, heads)?;
                self.attend(
                    Operand::F32(&q),
                    Operand::F32(&k),
                    Operand::F32(&v),
                    t,
                    t,
                    t,
                    heads,
                    false,
                )?
            };
            let out = self.project(Operand::F32(&ctx), &l.attn.out, t)?;
            res = Some(self.feed_forward(&mut h, &out, &l.ffn_ln, &l.fc1, &l.fc2, t)?);
            if i < taps {
                // A tap is the block's output, so the deferred sum is taken
                // here. Same arithmetic as deferring it - the same two floats
                // added in the same order - so a tapped run and a production
                // one agree bit for bit, and the tap means what its name says.
                let r = res.take().expect("the feed-forward always leaves one");
                self.gpu.add_inplace(&mut h, &r, t * d)?;
                tapped.push(self.gpu.download(&h)?);
            }
        }

        Ok((
            self.normed(&mut h, res.take().as_ref(), &self.enc_ln, t)?,
            tapped,
        ))
    }

    /// Builds the cache for one utterance from the encoder's output.
    pub fn cache(&self, encoded: &CudaSlice<f32>) -> Result<Cache, AsrError> {
        let (d, heads) = (self.cfg.d_model, self.cfg.decoder_attention_heads);
        let (hd, t) = (d / heads, self.cfg.max_source_positions);
        let mut cross_k = Vec::with_capacity(self.dec_layers.len());
        let mut cross_v = Vec::with_capacity(self.dec_layers.len());

        // The encoder's output is read *raw*. `encoder_attn_layer_norm` belongs
        // to the decoder's own stream and is applied to the queries in
        // `decode`; normalising the keys and values with it as well would be
        // an easy symmetry to assume and is not what the reference does.
        for l in &self.dec_layers {
            // Stored packed: every decode step reads all 32 layers of both, so
            // this is 160 MB of traffic a token rather than 320. The split
            // writes f16 directly - it is already touching every element, and
            // a `to_f16` pass after it read and wrote the same 7.7 MB tensor
            // again to change nothing but its width.
            cross_k.push(self.gpu.split_heads_f16(
                &self.project(Operand::F32(encoded), &l.cross.k, t)?,
                t,
                heads,
                hd,
            )?);
            cross_v.push(self.gpu.split_heads_t_f16(
                &self.project(Operand::F32(encoded), &l.cross.v, t)?,
                t,
                heads,
                hd,
            )?);
        }
        // The self-attention halves are allocated here rather than on the
        // first step, so a decode never allocates at all.
        let cap = self.cfg.max_target_positions;
        let mut self_k = Vec::with_capacity(self.dec_layers.len());
        let mut self_v = Vec::with_capacity(self.dec_layers.len());
        for _ in &self.dec_layers {
            self_k.push(self.gpu.zeros(cap * d)?);
            self_v.push(self.gpu.zeros(cap * d)?);
        }
        Ok(Cache {
            self_k,
            self_v,
            cross_k,
            cross_v,
            len: 0,
            cap,
        })
    }

    /// Runs `ids` through the decoder and returns the logits, `[n, vocab]`.
    ///
    /// The cache is extended by `ids.len()` tokens, so calling this with the
    /// whole prefix and then one token at a time is the same computation as
    /// calling it once with all of them.
    pub fn decode(&self, ids: &[u32], cache: &mut Cache) -> Result<CudaSlice<f32>, AsrError> {
        Ok(self.decode_tapped(ids, cache, 0)?.0)
    }

    /// The same, also returning the first `taps` block outputs on the host.
    ///
    /// See [`AsrModel::encode_tapped`] for why this is public.
    pub fn decode_tapped(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
    ) -> Result<(CudaSlice<f32>, Vec<Vec<f32>>), AsrError> {
        let (d, heads) = (self.cfg.d_model, self.cfg.decoder_attention_heads);
        let hd = d / heads;
        let (n, past) = (ids.len(), cache.len);
        let enc_t = self.cfg.max_source_positions;
        if past + n > self.cfg.max_target_positions {
            return Err(AsrError::PastTheEnd {
                at: past + n,
                max: self.cfg.max_target_positions,
            });
        }

        let ids64: Vec<i64> = ids.iter().map(|&i| i64::from(i)).collect();
        let mut h =
            self.gpu
                .embed_scaled(&self.embed_tokens, &self.gpu.upload_i64(&ids64)?, n, d, 1.0)?;
        // `scale_embedding` is false on this checkpoint, so the scale is 1.
        let pos = self.gpu.copy_range(&self.dec_pos, past * d, n * d)?;
        self.gpu.add_inplace(&mut h, &pos, n * d)?;

        let mut tapped = Vec::with_capacity(taps);
        // As in the encoder: a sub-layer's output waits for the next
        // normalisation to add it, so a decode step spends three launches a
        // layer rather than six on what is one pass either way.
        let mut res: Option<CudaSlice<f32>> = None;
        for (i, l) in self.dec_layers.iter().enumerate() {
            let x = self.normed(&mut h, res.take().as_ref(), &l.attn_ln, n)?;
            let q = self.queries(Operand::F32(&x), &l.attn, n, heads)?;
            let k_new = self.project(Operand::F32(&x), &l.attn.k, n)?;
            let v_new = self.project(Operand::F32(&x), &l.attn.v, n)?;

            // Scattered straight into the layout attention reads, in a buffer
            // that already has room for the whole utterance. This used to
            // allocate a larger pair, copy the whole cache into it and then
            // permute both into head order every step; the comment that stood
            // here argued the permutation was cheaper than a scattered append,
            // and it was measuring the wrong thing - the append and the
            // permutation are the same kernel, so keeping the cache in the
            // read layout costs nothing and saves all of it.
            let cap = cache.cap;
            self.gpu.cache_append(
                &k_new,
                0,
                &mut cache.self_k[i],
                n,
                heads,
                hd,
                cap,
                past,
                false,
            )?;
            self.gpu.cache_append(
                &v_new,
                0,
                &mut cache.self_v[i],
                n,
                heads,
                hd,
                cap,
                past,
                true,
            )?;
            let tk = past + n;
            let ctx = self.attend(
                Operand::F32(&q),
                Operand::F32(&cache.self_k[i]),
                Operand::F32(&cache.self_v[i]),
                n,
                tk,
                cap,
                heads,
                true,
            )?;
            let out = self.project(Operand::F32(&ctx), &l.attn.out, n)?;

            // Cross-attention. Only the queries come from the decoder; the
            // keys and values were built once from the encoder's output, which
            // is what makes a decode step cheap.
            let x = self.norm_add(&mut h, &out, &l.cross_ln, n)?;
            let q = self.queries(Operand::F32(&x), &l.cross, n, heads)?;
            let ctx = self.attend(
                Operand::F32(&q),
                Operand::F16(&cache.cross_k[i]),
                Operand::F16(&cache.cross_v[i]),
                n,
                enc_t,
                enc_t,
                heads,
                false,
            )?;
            let out = self.project(Operand::F32(&ctx), &l.cross.out, n)?;
            res = Some(self.feed_forward(&mut h, &out, &l.ffn_ln, &l.fc1, &l.fc2, n)?);
            if i < taps {
                // The deferred sum, taken here so the tap is the block's
                // output. See the encoder for why this is the same arithmetic.
                let r = res.take().expect("the feed-forward always leaves one");
                self.gpu.add_inplace(&mut h, &r, n * d)?;
                tapped.push(self.gpu.download(&h)?);
            }
        }
        cache.len = past + n;

        let h = self.normed(&mut h, res.take().as_ref(), &self.dec_ln, n)?;
        if taps > 0 {
            // The decoder's final normalisation, tapped under its own name so
            // a test can tell "the last block is wrong" from "the last norm
            // is wrong".
            tapped.push(self.gpu.download(&h)?);
        }
        // The output projection is the token embedding, transposed. Whisper
        // ties them, so there is no second matrix to bind or to upload.
        Ok((
            self.gpu.gemm_batched(
                Operand::F32(&h),
                Operand::F16(&self.embed_logits),
                None,
                Batch::single(n * self.cfg.vocab_size),
                n,
                d,
                self.cfg.vocab_size,
            )?,
            tapped,
        ))
    }
}

impl AsrModel {
    /// Greedy decoding, from log-mel features to token ids.
    ///
    /// Returns the tokens the model produced - prefix and end-of-transcript
    /// excluded - which is what the reference's `generate` hands back.
    ///
    /// # What this deliberately is not
    ///
    /// No beam search, no `best_of`, no temperature ladder, no compression or
    /// log-probability retry. The live pipeline runs greedy at a fixed
    /// language on VAD-gated utterances of a few seconds, and the fallback
    /// machinery exists for long-form transcription this engine does not do.
    /// Each omission is in `docs/MODEL.md` with its reason, so re-adding one
    /// is a decision rather than a discovery.
    pub fn generate(
        &self,
        mel: &[f32],
        language: &str,
        max_new: usize,
    ) -> Result<Vec<u32>, AsrError> {
        let encoded = self.encode(mel)?;
        let mut cache = self.cache(&encoded)?;

        let prefix = self.decoding.prefix(language, "transcribe")?;
        let mut pending = prefix.clone();
        let mut out = Vec::new();
        let budget = max_new.min(self.decoding.max_length.saturating_sub(prefix.len()));

        for step in 0..budget {
            let logits = self.decode(&pending, &mut cache)?;
            // Only the last row matters: every earlier one predicts a token
            // that is already in the prefix.
            let row = self.gpu.download(&self.gpu.copy_range(
                &logits,
                (pending.len() - 1) * self.cfg.vocab_size,
                self.cfg.vocab_size,
            )?)?;
            let next = self.pick(&row, step == 0);
            if next == self.decoding.eos_token_id {
                break;
            }
            out.push(next);
            pending = vec![next];
        }
        Ok(out)
    }

    /// Transcribes 16 kHz mono samples.
    pub fn transcribe(&self, samples: &[f32], language: &str) -> Result<String, AsrError> {
        let mel = self.frontend.log_mel(samples);
        let ids = self.generate(&mel, language, self.decoding.max_length)?;
        Ok(self.tokenizer.decode(&ids, true))
    }

    /// The highest-scoring token, with the suppression the checkpoint asks for.
    ///
    /// `suppress_tokens` is every control token plus the pieces OpenAI found
    /// the model hallucinating; `begin_suppress_tokens` is a leading space and
    /// an immediate end of transcript, and applies only at the first generated
    /// position. Skipping them does not fail loudly - it produces a transcript
    /// that begins with a space, or an empty one, on some utterances and not
    /// others.
    fn pick(&self, row: &[f32], first: bool) -> u32 {
        let suppressed = |id: u32| {
            self.decoding.suppress_tokens.contains(&id)
                || (first && self.decoding.begin_suppress_tokens.contains(&id))
        };
        row.iter()
            .enumerate()
            .filter(|&(i, _)| !suppressed(i as u32))
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are finite"))
            .map(|(i, _)| i as u32)
            .expect("the vocabulary is not empty")
    }
}
