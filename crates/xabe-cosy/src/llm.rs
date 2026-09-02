//! CosyVoice3's speech language model: Qwen2 0.5 B, text in, speech tokens out.

use crate::{CosyError, LlmConfig};
use xabe_cuda::{Batch, CudaSlice, DecodeScratch, Gpu, Operand};
use xabe_st::StFile;

/// One `[out, in]` matrix on the device, with the bias Qwen2 puts on `q`, `k`
/// and `v` but not on `o`.
///
/// The matrix is held at f16. A decode step is one row against every
/// weight, so the step is the weight stream and nothing else - 1.98 GB a
/// token at f32, 4.6 ms on this card, against 0.99 GB at f16 - and the
/// prompt's tiled matmul stages its operands as f16 regardless, so the
/// prefill's arithmetic does not change at all. The decode mat-vec's does,
/// from an exact f32 weight to an f16 one against an f32 activation;
/// `examples/probe_llm.rs` measures what that moves the teacher-forced
/// log-probabilities by against the f32 build, and `docs/BENCHMARKS.md`
/// has the figures. The biases stay f32.
struct GLinear {
    w: CudaSlice<u16>,
    bias: Option<CudaSlice<f32>>,
    in_dim: usize,
    out_dim: usize,
}

/// One decoder block.
///
/// The three attention inputs are one `[h + 2 kv, h]` weight and the two
/// MLP inputs one `[2 inter, h]`, so a decode step projects each set in one
/// launch; the prompt reads the same allocations a projection at a time
/// through `Gpu::gemm_batched_from`. The three attention biases are kept
/// apart as well as stacked, because the prompt's three products each want
/// their own.
struct GLayer {
    attn_norm: CudaSlice<f32>,
    qkv: GLinear,
    q_bias: CudaSlice<f32>,
    k_bias: CudaSlice<f32>,
    v_bias: CudaSlice<f32>,
    o: GLinear,
    ffn_norm: CudaSlice<f32>,
    gate_up: GLinear,
    down: GLinear,
}

/// The speech LLM, resident on one card.
///
/// # How this differs from the chat model next door
///
/// Both are grouped-query decoder transformers and the family resemblance is
/// the hazard. Three differences, each of which leaves the model producing
/// confident nonsense rather than failing:
///
/// - **Qwen2 puts a bias on `q`, `k` and `v`**, and none on `o`. Llama has no
///   attention biases at all. A schema that skipped them binds every tensor it
///   looks for and is quietly wrong at every layer.
/// - **A rope base of 1000000**, which is neither Llama-2's 10000 nor
///   Llama-3's 500000, and there is no `rope_freqs` scaling.
/// - **The head is not the embedding.** Text comes in through
///   `model.embed_tokens` (151936 wide) and speech goes out through
///   `llm_decoder` (6761 wide), with `speech_embedding` feeding generated
///   tokens back in. Three separate matrices where a chat model has two, and
///   the checkpoint's own `lm_head` - tied to the text embedding - is *unused*
///   and is dropped at conversion.
///
/// # The prompt layout
///
/// ```text
/// [sos] embed(prompt_text ++ text) [task_id]
/// ```
///
/// `sos` and `task_id` are indices into `speech_embedding`, not into the text
/// vocabulary - they are speech-side markers standing at the boundary. There
/// is **no audio prompt**: `frontend_instruct2` deletes it, so the speaker is
/// carried entirely by the flow stage. Wiring one in here because the
/// zero-shot path has it there gives a model that runs and matches on no
/// stage.
pub struct SpeechLlm {
    cfg: LlmConfig,
    gpu: Gpu,
    /// Qwen2's text embedding, `[vocab, hidden]`, at f16: 151936 rows of
    /// which a prompt reads a few dozen, and 544 MB at f32.
    embed: CudaSlice<u16>,
    /// The speech-token embedding, `[speech_vocab, hidden]`, at f16 like the
    /// text one. Also the source of the `sos` and `task_id` markers. It was
    /// tried at f32 as well - every decoded token enters through it - and
    /// the teacher-forced log-probabilities came out no closer to the f32
    /// build's, so it is held the way every other matrix here is.
    speech_embed: CudaSlice<u16>,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The speech head, `[speech_vocab, hidden]`, at f16 like every other
    /// matrix here.
    decoder: CudaSlice<u16>,
}

/// The keys and values a decode step reuses.
///
/// Held at f16 in the layout the attention kernels read - keys
/// `[kv_heads, cap, head_dim]`, values `[kv_heads, head_dim, cap]` - with
/// a capacity that doubles, exactly as the chat model's is; see
/// `xabe_chat`'s `Cache` for why the first version of that, which grew by
/// the token being added, was quadratic. A single position appends into
/// it from the projection's rotation kernel and reads it with the fused
/// decode attention, so a step's attention is two launches a layer where
/// the f32 chain was eleven and rebuilt the whole cache for every token.
pub struct Cache {
    k: Vec<CudaSlice<u16>>,
    v: Vec<CudaSlice<u16>>,
    /// Positions the buffers have room for, not how many they hold.
    cap: usize,
    len: usize,
    /// The fused decode attention's partials and counter; per cache so two
    /// utterances decoding by turns never share one.
    scratch: DecodeScratch,
}

impl Cache {
    /// How many positions it holds.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been decoded into it yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One assembled prompt, as embeddings rather than ids.
///
/// The prompt cannot be a token sequence: it mixes the *text* vocabulary with
/// two markers from the *speech* vocabulary, and there is no single table that
/// holds both. Upstream builds it by concatenating embeddings for exactly this
/// reason, and so does this.
pub struct Prompt {
    /// `[n, hidden]` on the device.
    pub h: CudaSlice<f32>,
    /// How many positions it holds.
    pub len: usize,
    /// How many of those are the utterance's own text, which is what the
    /// length limits are computed from - not the instruct's share.
    pub text_len: usize,
}

impl SpeechLlm {
    /// Loads `llm.safetensors` onto CUDA device `ordinal`.
    pub fn open(path: &std::path::Path, ordinal: usize) -> Result<Self, CosyError> {
        let cfg = LlmConfig::default();
        cfg.check()?;
        let f = StFile::open(path)?;
        let gpu = Gpu::open(ordinal)?;

        // Every tensor goes through `tensor_shaped`, which refuses a shape that
        // is not the one asked for. That is not belt and braces here: this
        // model has **no `config.json`**, so every number in `LlmConfig` is
        // transcribed from a `hyperpyyaml` file that cannot be parsed without
        // running it. The checkpoint is the only thing that can confirm the
        // transcription, and a mistake has to surface as a named tensor rather
        // than as a model that loads and sounds wrong.
        let want = |name: &str, shape: &[usize]| -> Result<CudaSlice<f32>, CosyError> {
            Ok(gpu.upload(f.tensor_shaped(name, shape)?)?)
        };

        let (h, kv) = (cfg.hidden_size, cfg.kv_dim());
        let lin =
            |prefix: &str, out: usize, inp: usize, bias: bool| -> Result<GLinear, CosyError> {
                Ok(GLinear {
                    w: gpu
                        .upload_f16(f.tensor_shaped(&format!("{prefix}.weight"), &[out, inp])?)?,
                    bias: match bias {
                        true => Some(want(&format!("{prefix}.bias"), &[out])?),
                        false => None,
                    },
                    in_dim: inp,
                    out_dim: out,
                })
            };

        // Several `[out_i, in]` tensors stacked along `out`, at f16.
        let stacked = |names: &[(String, usize)], inp: usize| -> Result<GLinear, CosyError> {
            let out: usize = names.iter().map(|(_, o)| o).sum();
            let mut rows = Vec::with_capacity(out * inp);
            for (name, o) in names {
                rows.extend_from_slice(f.tensor_shaped(name, &[*o, inp])?);
            }
            Ok(GLinear {
                w: gpu.upload_f16(&rows)?,
                bias: None,
                in_dim: inp,
                out_dim: out,
            })
        };

        let inter = cfg.intermediate_size;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            // The biases are the Qwen2 tell. `o_proj` has none, and asking
            // for one fails here rather than at the first odd utterance.
            let q_bias = want(&format!("{p}.self_attn.q_proj.bias"), &[h])?;
            let k_bias = want(&format!("{p}.self_attn.k_proj.bias"), &[kv])?;
            let v_bias = want(&format!("{p}.self_attn.v_proj.bias"), &[kv])?;
            let mut qkv = stacked(
                &[
                    (format!("{p}.self_attn.q_proj.weight"), h),
                    (format!("{p}.self_attn.k_proj.weight"), kv),
                    (format!("{p}.self_attn.v_proj.weight"), kv),
                ],
                h,
            )?;
            let mut qkv_bias = Vec::with_capacity(h + 2 * kv);
            for b in [&q_bias, &k_bias, &v_bias] {
                qkv_bias.extend_from_slice(&gpu.download(b)?);
            }
            qkv.bias = Some(gpu.upload(&qkv_bias)?);
            layers.push(GLayer {
                attn_norm: want(&format!("{p}.input_layernorm.weight"), &[h])?,
                qkv,
                q_bias,
                k_bias,
                v_bias,
                o: lin(&format!("{p}.self_attn.o_proj"), h, h, false)?,
                ffn_norm: want(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                gate_up: stacked(
                    &[
                        (format!("{p}.mlp.gate_proj.weight"), inter),
                        (format!("{p}.mlp.up_proj.weight"), inter),
                    ],
                    h,
                )?,
                down: lin(
                    &format!("{p}.mlp.down_proj"),
                    h,
                    cfg.intermediate_size,
                    false,
                )?,
            });
        }

        let half = |name: &str, shape: &[usize]| -> Result<CudaSlice<u16>, CosyError> {
            Ok(gpu.upload_f16(f.tensor_shaped(name, shape)?)?)
        };
        let model = Self {
            embed: half("model.embed_tokens.weight", &[cfg.vocab_size, h])?,
            speech_embed: half("speech_embedding.weight", &[cfg.speech_vocab_size, h])?,
            layers,
            norm: want("model.norm.weight", &[h])?,
            decoder: half("llm_decoder.weight", &[cfg.speech_vocab_size, h])?,
            cfg,
            gpu,
        };
        tracing::info!(
            device = ordinal,
            layers = cfg.num_hidden_layers,
            heads = cfg.num_attention_heads,
            kv_heads = cfg.num_key_value_heads,
            "cosyvoice speech llm on the device"
        );
        Ok(model)
    }

    /// The geometry this model was bound against.
    pub fn config(&self) -> &LlmConfig {
        &self.cfg
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// An empty cache for one utterance.
    pub fn cache(&self) -> Cache {
        Cache {
            k: Vec::new(),
            v: Vec::new(),
            cap: 0,
            len: 0,
            scratch: DecodeScratch::new(),
        }
    }

    /// Assembles `[sos] embed(text) [task_id]` on the device.
    ///
    /// `text` is the instruct ids followed by the utterance's, already
    /// concatenated - upstream concatenates before embedding, and `text_len`
    /// is the utterance's share alone because that is what the length limits
    /// are computed from.
    pub fn prompt(&self, text: &[u32], text_len: usize) -> Result<Prompt, CosyError> {
        if !text.contains(&LlmConfig::ENDOFPROMPT) {
            return Err(CosyError::NoEndOfPrompt(LlmConfig::ENDOFPROMPT));
        }
        let h = self.cfg.hidden_size;
        let n = text.len() + 2;
        let mut out = self.gpu.zeros(n * h)?;

        let ids: Vec<i64> = text.iter().map(|&i| i64::from(i)).collect();
        let body = self.gpu.embed_scaled_f16(
            &self.embed,
            &self.gpu.upload_i64(&ids)?,
            text.len(),
            h,
            1.0,
        )?;
        // The two markers come from the *speech* table, not the text one.
        let markers = self.gpu.embed_scaled_f16(
            &self.speech_embed,
            &self
                .gpu
                .upload_i64(&[i64::from(LlmConfig::SOS), i64::from(LlmConfig::TASK_ID)])?,
            2,
            h,
            1.0,
        )?;
        // `copy_into` has no *source* offset - it always reads from the start
        // of `src` - so the second marker has to be sliced out first. Passing
        // `markers` twice writes `sos` into both slots, which is a prompt that
        // is one token wrong at its very last position: the model still
        // answers, most of the sequence still agrees, and the first few tokens
        // are decided by the wrong context.
        let task = self.gpu.copy_range(&markers, h, h)?;
        self.gpu.copy_into(&mut out, &markers, 0, h)?;
        self.gpu.copy_into(&mut out, &body, h, text.len() * h)?;
        self.gpu.copy_into(&mut out, &task, (n - 1) * h, h)?;
        Ok(Prompt {
            h: out,
            len: n,
            text_len,
        })
    }

    /// Embeds one generated speech token, to be fed back in.
    pub fn speech_step(&self, id: u32) -> Result<CudaSlice<f32>, CosyError> {
        Ok(self.gpu.embed_scaled_f16(
            &self.speech_embed,
            &self.gpu.upload_i64(&[i64::from(id)])?,
            1,
            self.cfg.hidden_size,
            1.0,
        )?)
    }

    /// One projection, with Qwen2's bias where there is one.
    fn project(
        &self,
        x: Operand<'_>,
        l: &GLinear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, CosyError> {
        Ok(self.gpu.gemm_batched(
            x,
            Operand::F16(&l.w),
            l.bias.as_ref(),
            Batch::single(rows * l.out_dim),
            rows,
            l.in_dim,
            l.out_dim,
        )?)
    }

    /// Runs `n` positions of embeddings and returns `[n, speech_vocab]` logits.
    ///
    /// Takes embeddings rather than ids because the prompt mixes two
    /// vocabularies; see [`Prompt`].
    pub fn forward(
        &self,
        x: &CudaSlice<f32>,
        n: usize,
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, CosyError> {
        let h_dim = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let kv_dim = self.cfg.kv_dim();
        let past = cache.len;
        let first = cache.k.is_empty();
        let scale = (hd as f32).powf(-0.5);
        // Several positions read the cache at full width through the f32
        // chain below, which only the first call can do: after that the
        // earlier positions exist only at f16 in the attention layout. No
        // caller here does otherwise - a prompt, then one token at a time -
        // and refusing is better than an attention that ignores its past.
        if n > 1 && past > 0 {
            return Err(CosyError::Geometry {
                what: "a multi-position forward after decoding began (past positions)",
                got: past,
                want: 0,
            });
        }

        // Room for this call, doubling so a decode of any length pays for
        // growth a logarithmic number of times. The first call sizes the
        // buffers for its prompt and a good stretch of speech after it.
        if first {
            cache.cap = (past + n + 256).next_power_of_two();
        } else if past + n > cache.cap {
            let (was, want) = (cache.cap, (past + n).next_power_of_two());
            let keys = cache.k.iter_mut().map(|s| (s, false));
            let values = cache.v.iter_mut().map(|s| (s, true));
            for (slot, transposed) in keys.chain(values) {
                let mut grown = self.gpu.zeros_f16(want * kv_dim)?;
                self.gpu
                    .cache_grow_f16(slot, &mut grown, kv_heads, hd, was, want, past, transposed)?;
                *slot = grown;
            }
            cache.cap = want;
        }
        let cap = cache.cap;

        // The residual stream, a copy of the input because it is added to
        // in place. SAFETY: `copy_into` writes all `n * h_dim` of it.
        let mut h = unsafe { self.gpu.uninit(n * h_dim) }?;
        self.gpu.copy_into(&mut h, x, 0, n * h_dim)?;

        for (i, l) in self.layers.iter().enumerate() {
            let xn = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.attn_norm, LlmConfig::RMS_EPS)?;
            if first {
                cache.k.push(self.gpu.zeros_f16(cap * kv_dim)?);
                cache.v.push(self.gpu.zeros_f16(cap * kv_dim)?);
            }
            let tk = past + n;

            let ctx = if n == 1 {
                // One position: the three projections as one mat-vec over
                // the stacked weight, then the query rotated in place, the
                // key rotated and the value stored straight into the caches
                // from the same buffer, then the fused decode attention over
                // the grouped heads, reading the caches where they are. See
                // `Gpu::rope_cache_f16` and `Gpu::attn_decode_f16`.
                let mut proj = [self.project(Operand::F32(&xn), &l.qkv, 1)?];
                self.gpu.rope_cache_f16(
                    &mut proj,
                    (0, 0),
                    (0, h_dim),
                    (0, h_dim + kv_dim),
                    None,
                    heads,
                    kv_heads,
                    hd,
                    LlmConfig::ROPE_THETA,
                    past,
                    &mut cache.k[i],
                    &mut cache.v[i],
                    cap,
                )?;
                self.gpu.attn_decode_f16(
                    &proj[0],
                    &cache.k[i],
                    &cache.v[i],
                    heads,
                    kv_heads,
                    hd,
                    tk,
                    cap,
                    scale,
                    false,
                    &mut cache.scratch,
                )?
            } else {
                let from = |first: usize, bias: &CudaSlice<f32>, out: usize| {
                    self.gpu.gemm_batched_from(
                        Operand::F32(&xn),
                        Operand::F16(&l.qkv.w),
                        first,
                        Some(bias),
                        Batch::single(n * out),
                        n,
                        h_dim,
                        out,
                    )
                };
                let mut q = from(0, &l.q_bias, h_dim)?;
                let mut k = from(h_dim, &l.k_bias, kv_dim)?;
                let v = from(h_dim + kv_dim, &l.v_bias, kv_dim)?;
                // `k` rotates over `kv_heads`, not `heads`: the kernel walks
                // the tensor as `heads * head_dim` per position, so the query
                // count would read 896 floats out of a 128-wide row. The
                // buffer is the right length in total, so nothing checks it
                // and nothing crashes.
                self.gpu
                    .rope(&mut q, 0, n, heads, hd, LlmConfig::ROPE_THETA, past)?;
                self.gpu
                    .rope(&mut k, 0, n, kv_heads, hd, LlmConfig::ROPE_THETA, past)?;
                // Into the f16 caches for the steps that follow, and through
                // the f32 chain for this call: the prompt is the first call
                // and the only multi-position one, so the chain's operands
                // are the rows just projected and nothing has to be read
                // back out of the cache at full width. The fused f16 prefill
                // attention is instantiated at a head width of 128 and this
                // model's is 64, which is why the chain stays.
                self.gpu.cache_append_f16(
                    &k,
                    0,
                    &mut cache.k[i],
                    n,
                    kv_heads,
                    hd,
                    cap,
                    past,
                    false,
                )?;
                self.gpu.cache_append_f16(
                    &v,
                    0,
                    &mut cache.v[i],
                    n,
                    kv_heads,
                    hd,
                    cap,
                    past,
                    true,
                )?;

                let qh = self.gpu.split_heads(&q, n, heads, hd)?;
                let kh = self.gpu.split_heads(&k, n, kv_heads, hd)?;
                let kh = self.gpu.repeat_kv(&kh, heads, kv_heads, n, hd)?;
                let vt = self.gpu.split_heads_t(&v, n, kv_heads, hd)?;
                let vt = self.gpu.repeat_kv(&vt, heads, kv_heads, n, hd)?;
                let mut scores = self.gpu.gemm_batched(
                    Operand::F32(&qh),
                    Operand::F32(&kh),
                    None,
                    Batch {
                        count: heads,
                        a: n * hd,
                        w: n * hd,
                        out: n * n,
                        w_row: 0,
                    },
                    n,
                    hd,
                    n,
                )?;
                self.gpu.scale_inplace(&mut scores, heads * n * n, scale)?;
                self.gpu.causal_mask(&mut scores, heads, n, n, 0)?;
                self.gpu.softmax_rows(&mut scores, heads * n, n)?;
                let ctx = self.gpu.gemm_batched(
                    Operand::F32(&scores),
                    Operand::F32(&vt),
                    None,
                    Batch {
                        count: heads,
                        a: n * n,
                        w: hd * n,
                        out: n * hd,
                        w_row: 0,
                    },
                    n,
                    n,
                    hd,
                )?;
                self.gpu.merge_heads(&ctx, n, heads, hd)?
            };
            let out = self.project(Operand::F32(&ctx), &l.o, n)?;
            self.gpu.add_inplace(&mut h, &out, n * h_dim)?;

            let xn = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.ffn_norm, LlmConfig::RMS_EPS)?;
            // Gate and up as one batched product over the stacked weight -
            // `[2, n, inter]` out - and the gating written over the first
            // half, which is where the down projection reads.
            let inter = self.cfg.intermediate_size;
            let mut gu = self.gpu.gemm_batched(
                Operand::F32(&xn),
                Operand::F16(&l.gate_up.w),
                None,
                Batch {
                    count: 2,
                    a: 0,
                    w: inter * h_dim,
                    out: n * inter,
                    w_row: 0,
                },
                n,
                h_dim,
                inter,
            )?;
            let _twin = self.gpu.silu_mul_pair(&mut gu, n, inter)?;
            let down = self.project(Operand::F32(&gu), &l.down, n)?;
            self.gpu.add_inplace(&mut h, &down, n * h_dim)?;
        }
        cache.len = past + n;

        let h = self
            .gpu
            .rms_norm(&h, n, h_dim, &self.norm, LlmConfig::RMS_EPS)?;
        Ok(self.gpu.gemm_batched(
            Operand::F32(&h),
            Operand::F16(&self.decoder),
            None,
            Batch::single(n * self.cfg.speech_vocab_size),
            n,
            h_dim,
            self.cfg.speech_vocab_size,
        )?)
    }
}
