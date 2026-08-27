//! The Llama-2 forward pass on CUDA.

use crate::TranslateError;
use std::path::Path;
use xabe_cuda::{Batch, CudaSlice, Gpu, Operand};
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

/// A projection, on the device, stored narrow.
struct GLinear {
    w: CudaSlice<u16>,
    in_dim: usize,
    out_dim: usize,
}

/// One transformer block, on the device.
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
    /// `[vocab, hidden]` at full width, for the lookup.
    ///
    /// Read rather than multiplied: a token's vector goes straight into the
    /// residual stream, where rounding it perturbs the input rather than the
    /// arithmetic. 1.1 GB, against a model that is already 26.5.
    embed: CudaSlice<f32>,
    layers: Vec<GLayer>,
    norm: CudaSlice<f32>,
    /// The output projection, which this checkpoint does *not* tie to the
    /// embedding - `tie_word_embeddings` is false.
    lm_head: CudaSlice<u16>,
}

/// The keys and values a decode step reuses.
pub struct Cache {
    /// Per layer, `[len, hidden]`, rotated before storing.
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
}

impl Translator {
    /// Loads a checkpoint directory onto CUDA device `ordinal`.
    ///
    /// About 27 GB of transfers, and it takes as long as that sounds. The
    /// alternative - mapping and letting the kernels fault pages in - would
    /// move the cost to the first translation instead of the load, which is
    /// the wrong place for a service that is started once.
    pub fn open(path: &Path, ordinal: usize) -> Result<Self, TranslateError> {
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
        let lin = |b: &Bound| -> Result<GLinear, TranslateError> {
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

        let model = Self {
            embed: wide(&w.embed_tokens)?,
            layers,
            norm: wide(&w.norm)?,
            lm_head: narrow(&w.lm_head)?,
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
            len: 0,
        }
    }

    /// One projection.
    fn project(
        &self,
        x: Operand<'_>,
        l: &GLinear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, TranslateError> {
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
    pub fn forward(
        &self,
        ids: &[u32],
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, TranslateError> {
        Ok(self.forward_tapped(ids, cache, 0)?.0)
    }

    /// The same, also returning the first `taps` block outputs on the host.
    ///
    /// On the public surface for the same reason the ASR's are: "the model is
    /// wrong" is not a fact anyone can act on, and "layer 7 is wrong" is.
    pub fn forward_tapped(
        &self,
        ids: &[u32],
        cache: &mut Cache,
        taps: usize,
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
            self.gpu
                .rope(&mut q, n, heads, hd, self.cfg.rope_theta, past)?;
            self.gpu
                .rope(&mut k, n, heads, hd, self.cfg.rope_theta, past)?;

            if first {
                cache.k.push(k);
                cache.v.push(v);
            } else {
                for (slot, new) in [(&mut cache.k[i], k), (&mut cache.v[i], v)] {
                    let mut grown = self.gpu.zeros((past + n) * h_dim)?;
                    self.gpu.copy_into(&mut grown, slot, 0, past * h_dim)?;
                    self.gpu
                        .copy_into(&mut grown, &new, past * h_dim, n * h_dim)?;
                    *slot = grown;
                }
            }
            let tk = past + n;

            let qh = self.gpu.split_heads(&q, n, heads, hd)?;
            let kh = self.gpu.split_heads(&cache.k[i], tk, heads, hd)?;
            let vt = self.gpu.split_heads_t(&cache.v[i], tk, heads, hd)?;

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
    pub fn generate(
        &self,
        ids: &[u32],
        max_new: usize,
        penalty: f32,
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
            let logits = self.forward(&pending, &mut cache)?;
            let mut row = self.gpu.download(&self.gpu.copy_range(
                &logits,
                (pending.len() - 1) * self.cfg.vocab_size,
                self.cfg.vocab_size,
            )?)?;
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
        let ids = self.prompt_ids(source, target);
        let out = self.generate(&ids, max_new, penalty)?;
        let text = self.tokenizer.decode(&out, true);
        let cut = ["[/", "\n["]
            .iter()
            .filter_map(|s| text.find(s))
            .min()
            .unwrap_or(text.len());
        Ok(text[..cut].trim().to_string())
    }
}
