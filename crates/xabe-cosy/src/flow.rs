//! CosyVoice3's flow: speech tokens in, mel out, through a DiT and a solver.
//!
//! # Two pieces with different jobs
//!
//! **The conditioning.** Speech tokens are embedded, passed through a small
//! look-ahead convolution, and repeated to the mel rate - two mel frames per
//! token. The speaker's 192-wide embedding is normalised and projected to 80.
//! The reference clip's own mel is laid into the first `mel_len1` frames of a
//! condition tensor, and the rest is zeros. That prefix is *the whole reason
//! the output sounds like the reference speaker*: the flow is asked to
//! continue a mel it has been shown the start of.
//!
//! **The solver.** Ten Euler steps of a conditional flow-matching ODE, each
//! one estimator call on a batch of two - conditioned and unconditioned - so
//! that classifier-free guidance can extrapolate between them. The estimator
//! is a 22-layer DiT.
//!
//! # The starting point is captured, not drawn
//!
//! `CausalConditionalCFM.__init__` seeds torch and draws `randn([1, 80, 15000])`
//! once, keeping it as a plain attribute. Every utterance starts from a prefix
//! of that same buffer. It is deterministic given the seed and it is still not
//! something to reproduce in Rust, so it is captured - and a solver compared
//! against a different starting point is not a comparison at all.
//!
//! # The traps
//!
//! - **`AdaLayerNormZero_Final` chunks `(scale, shift)`**, and every other
//!   modulation in the file chunks shift before scale. Swapping them leaves a
//!   model that runs and sounds wrong.
//! - **The rope is partial and interleaved.** Only the first 64 of 1024 dims
//!   are rotated, before the heads are split - so one head of sixteen carries
//!   position - and the pairs are `(2j, 2j+1)`, not halves.
//! - **The feed-forward's GELU is the tanh approximation**, selected by
//!   `approximate="tanh"`, not the exact erf form the rest of this workspace
//!   uses.

use crate::CosyError;
use xabe_cuda::{Batch, CudaSlice, Gpu, Operand};
use xabe_st::StFile;

/// A row-major `[out, in]` matrix with an optional bias.
struct Linear {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    in_dim: usize,
    out_dim: usize,
}

/// One DiT block.
struct Block {
    /// `[6 * dim, dim]`: shift, scale and gate for attention and then for the
    /// feed-forward, in that order.
    attn_norm: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ff_in: Linear,
    ff_out: Linear,
}

/// The flow's geometry, transcribed from `cosyvoice3.yaml`.
#[derive(Debug, Clone, Copy)]
pub struct FlowConfig {
    /// Mel bands.
    pub mel_dim: usize,
    /// The speech codebook the tokens index.
    pub vocab_size: usize,
    /// Width of the speaker embedding as it arrives.
    pub spk_embed_dim: usize,
    /// Mel frames per speech token.
    pub token_mel_ratio: usize,
    /// DiT width.
    pub dim: usize,
    /// DiT blocks.
    pub depth: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one head, and of the rotary part.
    pub dim_head: usize,
    /// Feed-forward multiplier.
    pub ff_mult: usize,
    /// Euler steps the solver takes.
    pub n_timesteps: usize,
    /// How far classifier-free guidance extrapolates.
    pub cfg_rate: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            mel_dim: 80,
            vocab_size: 6561,
            spk_embed_dim: 192,
            token_mel_ratio: 2,
            dim: 1024,
            depth: 22,
            heads: 16,
            dim_head: 64,
            ff_mult: 2,
            n_timesteps: 10,
            cfg_rate: 0.7,
        }
    }
}

impl FlowConfig {
    /// The sinusoidal timestep embedding's width, before the MLP.
    pub const FREQ_EMBED_DIM: usize = 256;
    /// The scale `SinusPositionEmbedding` multiplies the timestep by.
    pub const TIME_SCALE: f32 = 1000.0;
    /// LayerNorm epsilon, which is 1e-6 here and not torch's 1e-5 default.
    pub const NORM_EPS: f32 = 1e-6;
    /// Width of the conv positional embedding's kernel.
    pub const POS_KERNEL: usize = 31;
    /// Its group count.
    pub const POS_GROUPS: usize = 16;
    /// How far the look-ahead convolution may see, `pre_lookahead_len`.
    pub const LOOKAHEAD: usize = 3;
    /// The slope of the look-ahead's leaky ReLU: torch's default, not 0.1.
    pub const LOOKAHEAD_SLOPE: f32 = 0.01;
}

/// The flow, resident on one card.
pub struct Flow {
    cfg: FlowConfig,
    gpu: Gpu,
    token_embed: Vec<f32>,
    spk_affine: Linear,
    look1: (CudaSlice<f32>, CudaSlice<f32>),
    look2: (CudaSlice<f32>, CudaSlice<f32>),
    time_mlp0: Linear,
    time_mlp2: Linear,
    input_proj: Linear,
    pos1: (CudaSlice<f32>, CudaSlice<f32>),
    pos2: (CudaSlice<f32>, CudaSlice<f32>),
    blocks: Vec<Block>,
    /// `[2 * dim, dim]`, chunked as **scale then shift**.
    norm_out: Linear,
    proj_out: Linear,
    inv_freq: CudaSlice<f32>,
}

impl Flow {
    /// Loads `flow.safetensors` onto CUDA device `ordinal`.
    pub fn open(path: &std::path::Path, ordinal: usize) -> Result<Self, CosyError> {
        let cfg = FlowConfig::default();
        let f = StFile::open(path)?;
        let gpu = Gpu::open(ordinal)?;
        Self::from_parts(cfg, f, gpu)
    }

    /// The same, on a device already open.
    pub fn from_parts(cfg: FlowConfig, f: StFile, gpu: Gpu) -> Result<Self, CosyError> {
        let lin = |p: &str, out: usize, inp: usize, bias: bool| -> Result<Linear, CosyError> {
            Ok(Linear {
                w: gpu.upload(f.tensor_shaped(&format!("{p}.weight"), &[out, inp])?)?,
                b: match bias {
                    true => Some(gpu.upload(f.tensor_shaped(&format!("{p}.bias"), &[out])?)?),
                    false => None,
                },
                in_dim: inp,
                out_dim: out,
            })
        };
        let conv = |p: &str,
                    out: usize,
                    inp: usize,
                    k: usize|
         -> Result<(CudaSlice<f32>, CudaSlice<f32>), CosyError> {
            Ok((
                gpu.upload(f.tensor_shaped(&format!("{p}.weight"), &[out, inp, k])?)?,
                gpu.upload(f.tensor_shaped(&format!("{p}.bias"), &[out])?)?,
            ))
        };

        let d = cfg.dim;
        let est = "decoder.estimator";
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = format!("{est}.transformer_blocks.{i}");
            blocks.push(Block {
                attn_norm: lin(&format!("{p}.attn_norm.linear"), 6 * d, d, true)?,
                q: lin(&format!("{p}.attn.to_q"), d, d, true)?,
                k: lin(&format!("{p}.attn.to_k"), d, d, true)?,
                v: lin(&format!("{p}.attn.to_v"), d, d, true)?,
                o: lin(&format!("{p}.attn.to_out.0"), d, d, true)?,
                ff_in: lin(&format!("{p}.ff.ff.0.0"), d * cfg.ff_mult, d, true)?,
                ff_out: lin(&format!("{p}.ff.ff.2"), d, d * cfg.ff_mult, true)?,
            });
        }

        let model = Self {
            // On the host: the only thing ever done with it is a gather of
            // a few hundred rows, which is a memcpy either way, and keeping it
            // here saves a download per utterance.
            token_embed: f
                .tensor_shaped("input_embedding.weight", &[cfg.vocab_size, cfg.mel_dim])?
                .to_vec(),
            spk_affine: lin(
                "spk_embed_affine_layer",
                cfg.mel_dim,
                cfg.spk_embed_dim,
                true,
            )?,
            // The look-ahead pair: 80 to 1024 over four frames, then back to
            // 80 over three. Both are ordinary convolutions with left padding
            // - the "lookahead" is in the *training*, not in the padding.
            look1: conv("pre_lookahead_layer.conv1", 1024, cfg.mel_dim, 4)?,
            look2: conv("pre_lookahead_layer.conv2", cfg.mel_dim, 1024, 3)?,
            time_mlp0: lin(
                &format!("{est}.time_embed.time_mlp.0"),
                d,
                FlowConfig::FREQ_EMBED_DIM,
                true,
            )?,
            time_mlp2: lin(&format!("{est}.time_embed.time_mlp.2"), d, d, true)?,
            // 320 in: the noisy mel, the condition, the token embedding and
            // the speaker, each 80 wide.
            input_proj: lin(&format!("{est}.input_embed.proj"), d, 4 * cfg.mel_dim, true)?,
            pos1: conv(
                &format!("{est}.input_embed.conv_pos_embed.conv1.0"),
                d,
                d / FlowConfig::POS_GROUPS,
                FlowConfig::POS_KERNEL,
            )?,
            pos2: conv(
                &format!("{est}.input_embed.conv_pos_embed.conv2.0"),
                d,
                d / FlowConfig::POS_GROUPS,
                FlowConfig::POS_KERNEL,
            )?,
            blocks,
            norm_out: lin(&format!("{est}.norm_out.linear"), 2 * d, d, true)?,
            proj_out: lin(&format!("{est}.proj_out"), cfg.mel_dim, d, true)?,
            inv_freq: gpu.upload(
                f.tensor_shaped(&format!("{est}.rotary_embed.inv_freq"), &[cfg.dim_head / 2])?,
            )?,
            cfg,
            gpu,
        };
        tracing::info!(
            depth = cfg.depth,
            dim = cfg.dim,
            "cosyvoice flow on the device"
        );
        Ok(model)
    }

    /// The geometry this flow was bound against.
    pub fn config(&self) -> &FlowConfig {
        &self.cfg
    }

    /// The device, for tests that want to read an intermediate back.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// `[rows, in]` times `[out, in]^T`, plus the bias.
    fn linear(
        &self,
        x: &CudaSlice<f32>,
        l: &Linear,
        rows: usize,
    ) -> Result<CudaSlice<f32>, CosyError> {
        Ok(self.gpu.gemm_batched(
            Operand::F32(x),
            Operand::F32(&l.w),
            l.b.as_ref(),
            Batch::single(rows * l.out_dim),
            rows,
            l.in_dim,
            l.out_dim,
        )?)
    }
}

impl Flow {
    /// The sinusoidal half of the timestep embedding.
    ///
    /// `scale * t` against `exp(-log(10000) * i / (half - 1))`, sines then
    /// cosines. Note the divisor is `half - 1` and not `half`: an off-by-one
    /// here shifts every frequency slightly and is invisible until the mel is
    /// compared.
    fn timestep_embedding(&self, t: f32) -> Vec<f32> {
        let half = FlowConfig::FREQ_EMBED_DIM / 2;
        let step = (10_000f32).ln() / (half - 1) as f32;
        let mut out = vec![0.0; FlowConfig::FREQ_EMBED_DIM];
        for i in 0..half {
            let a = FlowConfig::TIME_SCALE * t * (-(i as f32) * step).exp();
            out[i] = a.sin();
            out[half + i] = a.cos();
        }
        out
    }

    /// `time_mlp(sinusoid(t))`, one `[dim]` vector per batch row.
    fn time_embed(&self, t: f32) -> Result<CudaSlice<f32>, CosyError> {
        let e = self.gpu.upload(&self.timestep_embedding(t))?;
        let mut h = self.linear(&e, &self.time_mlp0, 1)?;
        self.gpu.silu(&mut h, self.cfg.dim)?;
        self.linear(&h, &self.time_mlp2, 1)
    }

    /// LayerNorm with no affine, which is what `elementwise_affine=False` is.
    fn norm(&self, x: &CudaSlice<f32>, rows: usize) -> Result<CudaSlice<f32>, CosyError> {
        let ones = self.gpu.upload(&vec![1.0f32; self.cfg.dim])?;
        let zeros = self.gpu.zeros(self.cfg.dim)?;
        Ok(self
            .gpu
            .layer_norm(x, rows, self.cfg.dim, &ones, &zeros, FlowConfig::NORM_EPS)?)
    }

    /// One estimator evaluation over a batch of two.
    ///
    /// `x`, `mu` and `cond` arrive as `[2, mel_dim, n]` - channel-major, which
    /// is how the reference holds them - and are transposed here because every
    /// linear wants `[position, feature]`.
    #[allow(clippy::too_many_arguments)]
    fn estimator(
        &self,
        x: &[f32],
        mu: &[f32],
        cond: &[f32],
        spk: &[f32],
        t: f32,
        n: usize,
        taps: &mut Vec<(String, Vec<f32>)>,
    ) -> Result<Vec<f32>, CosyError> {
        let (d, m) = (self.cfg.dim, self.cfg.mel_dim);
        let temb = self.gpu.download(&self.time_embed(t)?)?;
        taps.push(("dit_temb".into(), temb.clone()));

        let mut out = vec![0.0f32; 2 * m * n];
        for b in 0..2 {
            // `[4 * mel_dim, n]` on the host, transposed into `[n, 4 * mel_dim]`.
            // The order is the reference's: noisy mel, condition, token
            // embedding, speaker.
            let mut cat = vec![0.0f32; n * 4 * m];
            for c in 0..m {
                for p in 0..n {
                    cat[p * 4 * m + c] = x[(b * m + c) * n + p];
                    cat[p * 4 * m + m + c] = cond[(b * m + c) * n + p];
                    cat[p * 4 * m + 2 * m + c] = mu[(b * m + c) * n + p];
                    // The speaker is one vector repeated at every position.
                    cat[p * 4 * m + 3 * m + c] = spk[b * m + c];
                }
            }
            if b == 0 {
                taps.push(("dit_cat".into(), cat.clone()));
            }
            let h = self.linear(&self.gpu.upload(&cat)?, &self.input_proj, n)?;
            if b == 0 {
                taps.push(("dit_proj".into(), self.gpu.download(&h)?));
            }

            // The convolutional positional embedding, added as a residual.
            // Both convolutions are grouped and pad `kernel - 1` on the left,
            // and both are followed by Mish. `[n, dim]` has to become
            // `[dim, n]` for a convolution and back again afterwards.
            let hc = self.gpu.transpose(&h, n, d)?;
            let pad = FlowConfig::POS_KERNEL - 1;
            let (mut p1, t1) = self.gpu.grouped_conv1d(
                &hc,
                &self.pos1.0,
                &self.pos1.1,
                d,
                n,
                d,
                FlowConfig::POS_KERNEL,
                FlowConfig::POS_GROUPS,
                pad,
            )?;
            debug_assert_eq!(t1, n);
            self.gpu.mish(&mut p1, d * n)?;
            let (mut p2, t2) = self.gpu.grouped_conv1d(
                &p1,
                &self.pos2.0,
                &self.pos2.1,
                d,
                n,
                d,
                FlowConfig::POS_KERNEL,
                FlowConfig::POS_GROUPS,
                pad,
            )?;
            debug_assert_eq!(t2, n);
            self.gpu.mish(&mut p2, d * n)?;
            if b == 0 {
                taps.push((
                    "dit_pos".into(),
                    self.gpu.download(&self.gpu.transpose(&p2, d, n)?)?,
                ));
            }

            let mut h = h;
            let back = self.gpu.transpose(&p2, d, n)?;
            self.gpu.add_inplace(&mut h, &back, n * d)?;
            if b == 0 {
                taps.push(("dit_input_embed".into(), self.gpu.download(&h)?));
            }

            for (bi, blk) in self.blocks.iter().enumerate() {
                // The six modulation vectors, from the timestep. Each block
                // has its own projection, so this is per block and not hoisted.
                let mut te = self.gpu.upload(&temb)?;
                self.gpu.silu(&mut te, d)?;
                let mods = self.gpu.download(&self.linear(&te, &blk.attn_norm, 1)?)?;
                let (shift_a, scale_a, gate_a) = (&mods[..d], &mods[d..2 * d], &mods[2 * d..3 * d]);
                let (shift_f, scale_f, gate_f) =
                    (&mods[3 * d..4 * d], &mods[4 * d..5 * d], &mods[5 * d..]);

                // Attention, modulated in.
                let normed = self.gpu.download(&self.norm(&h, n)?)?;
                let mut xin = vec![0.0f32; n * d];
                for p in 0..n {
                    for c in 0..d {
                        xin[p * d + c] = normed[p * d + c] * (1.0 + scale_a[c]) + shift_a[c];
                    }
                }
                let xin = self.gpu.upload(&xin)?;

                let mut q = self.linear(&xin, &blk.q, n)?;
                let mut k = self.linear(&xin, &blk.k, n)?;
                let v = self.linear(&xin, &blk.v, n)?;
                // Partial and interleaved: the first `dim_head` of each 1024
                // wide row, before the heads are split - so one head of
                // sixteen carries position.
                self.gpu
                    .rope_gptj(&mut q, &self.inv_freq, n, d, self.cfg.dim_head)?;
                self.gpu
                    .rope_gptj(&mut k, &self.inv_freq, n, d, self.cfg.dim_head)?;

                let hd = self.cfg.dim_head;
                let heads = self.cfg.heads;
                let qh = self.gpu.split_heads(&q, n, heads, hd)?;
                let kh = self.gpu.split_heads(&k, n, heads, hd)?;
                let vt = self.gpu.split_heads_t(&v, n, heads, hd)?;

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
                self.gpu
                    .scale_inplace(&mut scores, heads * n * n, (hd as f32).powf(-0.5))?;
                // No causal mask: the flow sees the whole utterance at once,
                // which is what makes it a flow rather than a decoder.
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
                let ctx = self.gpu.merge_heads(&ctx, n, heads, hd)?;
                let attn = self.gpu.download(&self.linear(&ctx, &blk.o, n)?)?;

                let mut hh = self.gpu.download(&h)?;
                for p in 0..n {
                    for c in 0..d {
                        hh[p * d + c] += gate_a[c] * attn[p * d + c];
                    }
                }
                h = self.gpu.upload(&hh)?;

                // Feed-forward, modulated in the same way.
                let normed = self.gpu.download(&self.norm(&h, n)?)?;
                let mut fin = vec![0.0f32; n * d];
                for p in 0..n {
                    for c in 0..d {
                        fin[p * d + c] = normed[p * d + c] * (1.0 + scale_f[c]) + shift_f[c];
                    }
                }
                let mut ff = self.linear(&self.gpu.upload(&fin)?, &blk.ff_in, n)?;
                // The tanh approximation, not the erf form.
                self.gpu.gelu_tanh(&mut ff, n * d * self.cfg.ff_mult)?;
                let ff = self.gpu.download(&self.linear(&ff, &blk.ff_out, n)?)?;

                let mut hh = self.gpu.download(&h)?;
                for p in 0..n {
                    for c in 0..d {
                        hh[p * d + c] += gate_f[c] * ff[p * d + c];
                    }
                }
                h = self.gpu.upload(&hh)?;
                if b == 0 && matches!(bi, 0 | 1 | 7 | 14 | 21) {
                    taps.push((format!("dit_block{bi}"), self.gpu.download(&h)?));
                }
            }

            // The final modulation chunks **(scale, shift)** - the reverse of
            // every other one in the file.
            let mut te = self.gpu.upload(&temb)?;
            self.gpu.silu(&mut te, d)?;
            let fin = self.gpu.download(&self.linear(&te, &self.norm_out, 1)?)?;
            let (scale, shift) = (&fin[..d], &fin[d..]);

            let normed = self.gpu.download(&self.norm(&h, n)?)?;
            let mut last = vec![0.0f32; n * d];
            for p in 0..n {
                for c in 0..d {
                    last[p * d + c] = normed[p * d + c] * (1.0 + scale[c]) + shift[c];
                }
            }
            let y =
                self.gpu
                    .download(&self.linear(&self.gpu.upload(&last)?, &self.proj_out, n)?)?;

            // Back to channel-major, which is what the solver works in.
            for c in 0..m {
                for p in 0..n {
                    out[(b * m + c) * n + p] = y[p * m + c];
                }
            }
        }
        Ok(out)
    }

    /// The estimator's own boundaries, for batch row 0.
    ///
    /// Returns `(name, values)` in evaluation order, so a probe can say which
    /// module diverged rather than that the mel did. The last entry is the
    /// whole two-row output.
    pub fn estimate_tapped(
        &self,
        x: &[f32],
        mu: &[f32],
        cond: &[f32],
        spk: &[f32],
        t: f32,
        n: usize,
    ) -> Result<Vec<(String, Vec<f32>)>, CosyError> {
        let mut taps = Vec::new();
        let out = self.estimator(x, mu, cond, spk, t, n, &mut taps)?;
        taps.push(("dit_step0".into(), out));
        Ok(taps)
    }

    /// One estimator evaluation, exposed so a test can compare it alone.
    ///
    /// The solver runs ten of these and a mistake in any one of them looks the
    /// same from the outside.
    pub fn estimate(
        &self,
        x: &[f32],
        mu: &[f32],
        cond: &[f32],
        spk: &[f32],
        t: f32,
        n: usize,
    ) -> Result<Vec<f32>, CosyError> {
        self.estimator(x, mu, cond, spk, t, n, &mut Vec::new())
    }

    /// The speaker vector the DiT sees: L2-normalised, then projected 192 to 80.
    ///
    /// The normalisation is `F.normalize`'s, which divides by the norm
    /// **clamped below at 1e-12** rather than by the norm - so a zero
    /// embedding gives zeros instead of NaNs, and a caller who forgets to
    /// provide one gets a flat voice rather than a mel full of NaN.
    fn speaker(&self, embedding: &[f32]) -> Result<Vec<f32>, CosyError> {
        if embedding.len() != self.cfg.spk_embed_dim {
            return Err(CosyError::Geometry {
                what: "speaker embedding width",
                got: embedding.len(),
                want: self.cfg.spk_embed_dim,
            });
        }
        let norm = embedding
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        let unit: Vec<f32> = embedding.iter().map(|v| v / norm).collect();
        let g = self.gpu.upload(&unit)?;
        Ok(self.gpu.download(&self.linear(&g, &self.spk_affine, 1)?)?)
    }

    /// Speech tokens to the DiT's `mu`: embed, look ahead, repeat.
    ///
    /// `pre_lookahead_layer` is two convolutions with a residual, and the
    /// asymmetry is the whole point of the name: the first pads **three on the
    /// right**, so a frame is allowed to see three tokens into the future, and
    /// the second pads two on the left so nothing after that does. Upstream's
    /// streaming path passes those three tokens in as `context` instead of
    /// padding with zeros; a whole utterance has nothing to pass, and pads.
    ///
    /// The activation between them is `F.leaky_relu` with **no slope given**,
    /// so 0.01 - the same default that the vocoder's last activation turned on,
    /// and worth naming twice because it looks like the 0.1 used everywhere
    /// else in this checkpoint.
    fn front(
        &self,
        ids: &[u32],
        taps: &mut Vec<(String, Vec<f32>)>,
    ) -> Result<(Vec<f32>, usize), CosyError> {
        let (m, nt) = (self.cfg.mel_dim, ids.len());
        if nt == 0 {
            return Err(CosyError::Geometry {
                what: "the flow was given no speech tokens",
                got: 0,
                want: 1,
            });
        }

        let mut emb = vec![0.0f32; nt * m];
        for (i, &t) in ids.iter().enumerate() {
            let t = t as usize;
            if t >= self.cfg.vocab_size {
                return Err(CosyError::Geometry {
                    what: "speech token outside the codebook",
                    got: t,
                    want: self.cfg.vocab_size,
                });
            }
            emb[i * m..(i + 1) * m].copy_from_slice(&self.token_embed[t * m..(t + 1) * m]);
        }
        taps.push(("flow_token_embed".into(), emb.clone()));

        // `[nt, m]` for the residual, `[m, nt]` for the convolutions.
        let x = self.gpu.upload(&emb)?;
        let xc = self.gpu.transpose(&x, nt, m)?;
        let look = FlowConfig::LOOKAHEAD;
        let (mut h1, t1) = self.gpu.conv1d(
            &xc,
            &self.look1.0,
            Some(&self.look1.1),
            m,
            nt,
            self.cfg.dim,
            look + 1,
            0,
            look,
            1,
        )?;
        debug_assert_eq!(t1, nt, "the look-ahead convolution changed the length");
        self.gpu
            .leaky_relu(&mut h1, self.cfg.dim * nt, FlowConfig::LOOKAHEAD_SLOPE)?;
        let (h2, t2) = self.gpu.conv1d(
            &h1,
            &self.look2.0,
            Some(&self.look2.1),
            self.cfg.dim,
            nt,
            m,
            3,
            2,
            0,
            1,
        )?;
        debug_assert_eq!(t2, nt, "the look-ahead convolution changed the length");

        let mut back = self.gpu.transpose(&h2, m, nt)?;
        self.gpu.add_inplace(&mut back, &x, nt * m)?;
        let look = self.gpu.download(&back)?;
        taps.push(("pre_lookahead".into(), look.clone()));

        // `repeat_interleave(token_mel_ratio)` along position, transposed into
        // the channel-major `[m, n]` the estimator wants. Two mel frames per
        // speech token: 25 Hz in, 50 Hz out.
        let r = self.cfg.token_mel_ratio;
        let n = nt * r;
        let mut mu = vec![0.0f32; m * n];
        for p in 0..n {
            let src = p / r;
            for c in 0..m {
                mu[c * n + p] = look[src * m + c];
            }
        }
        Ok((mu, n))
    }

    /// The Euler solver: `n_timesteps` steps of classifier-free guidance.
    ///
    /// Both halves of the batch carry the same noisy mel; the second gets a
    /// zero `mu`, zero condition and zero speaker, and the two are extrapolated
    /// apart by `cfg_rate`. That is why the estimator runs a batch of two and
    /// why the batch is not an optimisation to be removed.
    ///
    /// The step size is recomputed as `t_span[step + 1] - t` rather than as a
    /// difference of two entries of the schedule. On a cosine schedule those
    /// are the same number in exact arithmetic and not in float32, and this is
    /// the one upstream takes.
    pub fn solve(
        &self,
        mu: &[f32],
        cond: &[f32],
        spk: &[f32],
        noise: &[f32],
        n: usize,
    ) -> Result<Vec<f32>, CosyError> {
        let m = self.cfg.mel_dim;
        for (what, got) in [
            ("mu", mu.len()),
            ("cond", cond.len()),
            ("noise", noise.len()),
        ] {
            if got < m * n {
                return Err(CosyError::Geometry {
                    what,
                    got,
                    want: m * n,
                });
            }
        }
        if spk.len() != m {
            return Err(CosyError::Geometry {
                what: "projected speaker",
                got: spk.len(),
                want: m,
            });
        }

        let ts = self.t_span();
        let mut x = noise[..m * n].to_vec();
        let mut x2 = vec![0.0f32; 2 * m * n];
        let mut mu2 = vec![0.0f32; 2 * m * n];
        mu2[..m * n].copy_from_slice(&mu[..m * n]);
        let mut cond2 = vec![0.0f32; 2 * m * n];
        cond2[..m * n].copy_from_slice(&cond[..m * n]);
        let mut spk2 = vec![0.0f32; 2 * m];
        spk2[..m].copy_from_slice(spk);

        let (mut t, mut dt) = (ts[0], ts[1] - ts[0]);
        for step in 1..ts.len() {
            x2[..m * n].copy_from_slice(&x);
            x2[m * n..].copy_from_slice(&x);
            let d = self.estimator(&x2, &mu2, &cond2, &spk2, t, n, &mut Vec::new())?;
            let r = self.cfg.cfg_rate;
            for (i, xi) in x.iter_mut().enumerate() {
                *xi += dt * ((1.0 + r) * d[i] - r * d[m * n + i]);
            }
            t += dt;
            if step + 1 < ts.len() {
                dt = ts[step + 1] - t;
            }
        }
        Ok(x)
    }

    /// The whole flow: speech tokens in, the generated mel out.
    ///
    /// `prompt_feat` is the speaker's own mel, `[frames, mel_dim]`, and it does
    /// two jobs at once: it is the condition for its own stretch of the
    /// timeline, and its length is where the generated part starts. The
    /// returned mel is **only** that generated part, `[mel_dim, frames]`, which
    /// is what the vocoder takes.
    pub fn mel(
        &self,
        prompt_tokens: &[u32],
        tokens: &[u32],
        prompt_feat: &[f32],
        embedding: &[f32],
        noise: &[f32],
    ) -> Result<(Vec<f32>, usize), CosyError> {
        self.mel_inner(
            prompt_tokens,
            tokens,
            prompt_feat,
            embedding,
            noise,
            &mut Vec::new(),
        )
    }

    /// The same, keeping every intermediate for `examples/probe_flow.rs`.
    pub fn mel_tapped(
        &self,
        prompt_tokens: &[u32],
        tokens: &[u32],
        prompt_feat: &[f32],
        embedding: &[f32],
        noise: &[f32],
    ) -> Result<Vec<(String, Vec<f32>)>, CosyError> {
        let mut taps = Vec::new();
        let (mel, _) = self.mel_inner(
            prompt_tokens,
            tokens,
            prompt_feat,
            embedding,
            noise,
            &mut taps,
        )?;
        taps.push(("mel".into(), mel));
        Ok(taps)
    }

    fn mel_inner(
        &self,
        prompt_tokens: &[u32],
        tokens: &[u32],
        prompt_feat: &[f32],
        embedding: &[f32],
        noise: &[f32],
        taps: &mut Vec<(String, Vec<f32>)>,
    ) -> Result<(Vec<f32>, usize), CosyError> {
        let m = self.cfg.mel_dim;
        let spk = self.speaker(embedding)?;
        taps.push(("spk80".into(), spk.clone()));

        let mut ids = prompt_tokens.to_vec();
        ids.extend_from_slice(tokens);
        let (mu, n) = self.front(&ids, taps)?;
        taps.push(("flow_mu".into(), mu.clone()));

        if !prompt_feat.len().is_multiple_of(m) {
            return Err(CosyError::Geometry {
                what: "the prompt mel is not a whole number of frames",
                got: prompt_feat.len(),
                want: m,
            });
        }
        let head = prompt_feat.len() / m;
        if head >= n {
            return Err(CosyError::Geometry {
                what: "the prompt mel is as long as the whole utterance",
                got: head,
                want: n,
            });
        }

        // The condition is the prompt's own mel laid at the front of an
        // otherwise zero timeline, transposed to channel-major.
        let mut cond = vec![0.0f32; m * n];
        for p in 0..head {
            for c in 0..m {
                cond[c * n + p] = prompt_feat[p * m + c];
            }
        }
        taps.push(("flow_cond".into(), cond.clone()));

        let x = self.solve(&mu, &cond, &spk, noise, n)?;

        // Only the part after the prompt is new; the rest is the solver
        // reconstructing audio the caller already has.
        let tail = n - head;
        let mut out = vec![0.0f32; m * tail];
        for c in 0..m {
            out[c * tail..(c + 1) * tail].copy_from_slice(&x[c * n + head..c * n + n]);
        }
        Ok((out, tail))
    }

    /// The solver's timesteps: a cosine schedule over `[0, 1]`.
    pub fn t_span(&self) -> Vec<f32> {
        (0..=self.cfg.n_timesteps)
            .map(|i| {
                let u = i as f32 / self.cfg.n_timesteps as f32;
                1.0 - (u * 0.5 * std::f32::consts::PI).cos()
            })
            .collect()
    }
}
