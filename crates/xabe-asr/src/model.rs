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
    /// Per layer, `[len, d_model]`, capacity `max_target_positions`.
    self_k: Vec<CudaSlice<f32>>,
    self_v: Vec<CudaSlice<f32>>,
    /// Per layer, `[heads, 1500, head_dim]` and `[heads, head_dim, 1500]`,
    /// packed - every decode step reads all of both.
    cross_k: Vec<CudaSlice<u16>>,
    cross_v: Vec<CudaSlice<u16>>,
    /// How many tokens the self-attention halves hold.
    len: usize,
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

    /// Attention, given queries already projected and split, and keys and
    /// values already split.
    ///
    /// `q` is `[heads, tq, head_dim]`, `k` is `[heads, tk, head_dim]` and `v`
    /// is `[heads, head_dim, tk]` - the transpose, because the context product
    /// reads the values down their time axis. Returns `[tq, d_model]`.
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
                w: tk * hd,
                out: tq * tk,
            },
            tq,
            hd,
            tk,
        )?;
        if causal {
            // With `tk - tq` keys already cached, query `i` really sits at
            // position `i + (tk - tq)`.
            self.gpu.causal_mask(&mut scores, heads, tq, tk, tk - tq)?;
        }
        self.gpu.softmax_rows(&mut scores, heads * tq, tk)?;
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
                w: hd * tk,
                out: tq * hd,
            },
            tq,
            tk,
            hd,
        )?;
        Ok(self.gpu.merge_heads(&ctx, tq, heads, hd)?)
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
        Ok(self.gpu.split_heads(&q, t, heads, hd)?)
    }

    /// A feed-forward block, applied in place on the residual stream.
    fn feed_forward(
        &self,
        h: &mut CudaSlice<f32>,
        ln: &GNorm,
        fc1: &GLinear,
        fc2: &GLinear,
        t: usize,
    ) -> Result<(), AsrError> {
        let x = self.norm(h, ln, t)?;
        let mut inner = self.project(Operand::F32(&x), fc1, t)?;
        self.gpu.gelu(&mut inner, t * fc1.out_dim)?;
        let out = self.project(Operand::F32(&inner), fc2, t)?;
        self.gpu.add_inplace(h, &out, t * self.cfg.d_model)?;
        Ok(())
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
        for (i, l) in self.enc_layers.iter().enumerate() {
            let x = self.norm(&h, &l.attn_ln, t)?;
            let q = self.queries(Operand::F32(&x), &l.attn, t, heads)?;
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
            let ctx = self.attend(
                Operand::F32(&q),
                Operand::F32(&k),
                Operand::F32(&v),
                t,
                t,
                heads,
                false,
            )?;
            let out = self.project(Operand::F32(&ctx), &l.attn.out, t)?;
            self.gpu.add_inplace(&mut h, &out, t * d)?;

            self.feed_forward(&mut h, &l.ffn_ln, &l.fc1, &l.fc2, t)?;
            if i < taps {
                tapped.push(self.gpu.download(&h)?);
            }
        }

        Ok((self.norm(&h, &self.enc_ln, t)?, tapped))
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
            let k = self.gpu.split_heads(
                &self.project(Operand::F32(encoded), &l.cross.k, t)?,
                t,
                heads,
                hd,
            )?;
            let v = self.gpu.split_heads_t(
                &self.project(Operand::F32(encoded), &l.cross.v, t)?,
                t,
                heads,
                hd,
            )?;
            // Stored packed: every decode step reads all 32 layers of both, so
            // this is 160 MB of traffic a token rather than 320.
            cross_k.push(self.gpu.to_f16(&k, t * d)?);
            cross_v.push(self.gpu.to_f16(&v, t * d)?);
        }
        Ok(Cache {
            self_k: Vec::new(),
            self_v: Vec::new(),
            cross_k,
            cross_v,
            len: 0,
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

        let first = cache.self_k.is_empty();
        let mut tapped = Vec::with_capacity(taps);
        for (i, l) in self.dec_layers.iter().enumerate() {
            let x = self.norm(&h, &l.attn_ln, n)?;
            let q = self.queries(Operand::F32(&x), &l.attn, n, heads)?;
            let k_new = self.project(Operand::F32(&x), &l.attn.k, n)?;
            let v_new = self.project(Operand::F32(&x), &l.attn.v, n)?;

            // The cache holds `[t, d_model]` and is split per step. Keeping it
            // already split would save the permutation and cost an append that
            // scatters across `heads` separate runs; at a few hundred tokens
            // of 1280 the permutation is the cheaper of the two by far.
            if first {
                cache.self_k.push(k_new);
                cache.self_v.push(v_new);
            } else {
                let (ck, cv) = (&mut cache.self_k[i], &mut cache.self_v[i]);
                let mut grown_k = self.gpu.zeros((past + n) * d)?;
                let mut grown_v = self.gpu.zeros((past + n) * d)?;
                self.gpu.copy_into(&mut grown_k, ck, 0, past * d)?;
                self.gpu.copy_into(&mut grown_v, cv, 0, past * d)?;
                self.gpu.copy_into(&mut grown_k, &k_new, past * d, n * d)?;
                self.gpu.copy_into(&mut grown_v, &v_new, past * d, n * d)?;
                *ck = grown_k;
                *cv = grown_v;
            }
            let tk = past + n;
            let k = self.gpu.split_heads(&cache.self_k[i], tk, heads, hd)?;
            let v = self.gpu.split_heads_t(&cache.self_v[i], tk, heads, hd)?;
            let ctx = self.attend(
                Operand::F32(&q),
                Operand::F32(&k),
                Operand::F32(&v),
                n,
                tk,
                heads,
                true,
            )?;
            let out = self.project(Operand::F32(&ctx), &l.attn.out, n)?;
            self.gpu.add_inplace(&mut h, &out, n * d)?;

            // Cross-attention. Only the queries come from the decoder; the
            // keys and values were built once from the encoder's output, which
            // is what makes a decode step cheap.
            let x = self.norm(&h, &l.cross_ln, n)?;
            let q = self.queries(Operand::F32(&x), &l.cross, n, heads)?;
            let ctx = self.attend(
                Operand::F32(&q),
                Operand::F16(&cache.cross_k[i]),
                Operand::F16(&cache.cross_v[i]),
                n,
                enc_t,
                heads,
                false,
            )?;
            let out = self.project(Operand::F32(&ctx), &l.cross.out, n)?;
            self.gpu.add_inplace(&mut h, &out, n * d)?;

            self.feed_forward(&mut h, &l.ffn_ln, &l.fc1, &l.fc2, n)?;
            if i < taps {
                tapped.push(self.gpu.download(&h)?);
            }
        }
        cache.len = past + n;

        let h = self.norm(&h, &self.dec_ln, n)?;
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
