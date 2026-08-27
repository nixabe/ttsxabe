//! CosyVoice3's speech language model: Qwen2 0.5 B, text in, speech tokens out.

use crate::{CosyError, LlmConfig};
use xabe_cuda::{Batch, CudaSlice, Gpu, Operand};
use xabe_st::StFile;

/// One `[out, in]` matrix on the device, with the bias Qwen2 puts on `q`, `k`
/// and `v` but not on `o`.
struct GLinear {
    w: CudaSlice<f32>,
    bias: Option<CudaSlice<f32>>,
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
    /// Qwen2's text embedding, `[vocab, hidden]`.
    embed: CudaSlice<f32>,
    /// The speech-token embedding, `[speech_vocab, hidden]`. Also the source
    /// of the `sos` and `task_id` markers.
    speech_embed: CudaSlice<f32>,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The speech head, `[speech_vocab, hidden]`.
    decoder: CudaSlice<f32>,
}

/// The keys and values a decode step reuses.
pub struct Cache {
    k: Vec<CudaSlice<f32>>,
    v: Vec<CudaSlice<f32>>,
    len: usize,
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
                    w: want(&format!("{prefix}.weight"), &[out, inp])?,
                    bias: match bias {
                        true => Some(want(&format!("{prefix}.bias"), &[out])?),
                        false => None,
                    },
                    in_dim: inp,
                    out_dim: out,
                })
            };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(GLayer {
                attn_norm: want(&format!("{p}.input_layernorm.weight"), &[h])?,
                // The biases are the Qwen2 tell. `o_proj` has none, and asking
                // for one fails here rather than at the first odd utterance.
                q: lin(&format!("{p}.self_attn.q_proj"), h, h, true)?,
                k: lin(&format!("{p}.self_attn.k_proj"), kv, h, true)?,
                v: lin(&format!("{p}.self_attn.v_proj"), kv, h, true)?,
                o: lin(&format!("{p}.self_attn.o_proj"), h, h, false)?,
                ffn_norm: want(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                gate: lin(
                    &format!("{p}.mlp.gate_proj"),
                    cfg.intermediate_size,
                    h,
                    false,
                )?,
                up: lin(&format!("{p}.mlp.up_proj"), cfg.intermediate_size, h, false)?,
                down: lin(
                    &format!("{p}.mlp.down_proj"),
                    h,
                    cfg.intermediate_size,
                    false,
                )?,
            });
        }

        let model = Self {
            embed: want("model.embed_tokens.weight", &[cfg.vocab_size, h])?,
            speech_embed: want("speech_embedding.weight", &[cfg.speech_vocab_size, h])?,
            layers,
            norm: want("model.norm.weight", &[h])?,
            decoder: want("llm_decoder.weight", &[cfg.speech_vocab_size, h])?,
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
            len: 0,
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
        let body =
            self.gpu
                .embed_scaled(&self.embed, &self.gpu.upload_i64(&ids)?, text.len(), h, 1.0)?;
        // The two markers come from the *speech* table, not the text one.
        let markers = self.gpu.embed_scaled(
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
        Ok(self.gpu.embed_scaled(
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
            Operand::F32(&l.w),
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

        let mut h = self.gpu.zeros(n * h_dim)?;
        self.gpu.copy_into(&mut h, x, 0, n * h_dim)?;

        let first = cache.k.is_empty();
        for (i, l) in self.layers.iter().enumerate() {
            let xn = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.attn_norm, LlmConfig::RMS_EPS)?;
            let mut q = self.project(Operand::F32(&xn), &l.q, n)?;
            let mut k = self.project(Operand::F32(&xn), &l.k, n)?;
            let v = self.project(Operand::F32(&xn), &l.v, n)?;

            // `k` rotates over `kv_heads`, not `heads`: the kernel walks the
            // tensor as `heads * head_dim` per position, so the query count
            // would read 896 floats out of a 128-wide row. The buffer is the
            // right length in total, so nothing checks it and nothing crashes.
            self.gpu
                .rope(&mut q, n, heads, hd, LlmConfig::ROPE_THETA, past)?;
            self.gpu
                .rope(&mut k, n, kv_heads, hd, LlmConfig::ROPE_THETA, past)?;

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

            let xn = self
                .gpu
                .rms_norm(&h, n, h_dim, &l.ffn_norm, LlmConfig::RMS_EPS)?;
            let mut gate = self.project(Operand::F32(&xn), &l.gate, n)?;
            let up = self.project(Operand::F32(&xn), &l.up, n)?;
            self.gpu
                .silu_mul(&mut gate, &up, n * self.cfg.intermediate_size)?;
            let down = self.project(Operand::F32(&gate), &l.down, n)?;
            self.gpu.add_inplace(&mut h, &down, n * h_dim)?;
        }
        cache.len = past + n;

        let h = self
            .gpu
            .rms_norm(&h, n, h_dim, &self.norm, LlmConfig::RMS_EPS)?;
        Ok(self.gpu.gemm_batched(
            Operand::F32(&h),
            Operand::F32(&self.decoder),
            None,
            Batch::single(n * self.cfg.speech_vocab_size),
            n,
            h_dim,
            self.cfg.speech_vocab_size,
        )?)
    }
}
