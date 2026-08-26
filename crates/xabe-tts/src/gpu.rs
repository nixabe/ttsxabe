//! The forward pass on CUDA.
//!
//! A deliberate mirror of the CPU path rather than a replacement for it. The
//! scalar version in the sibling modules stays the readable definition of what
//! this model computes - `docs/ARCHITECTURE.md` is explicit that `xabe-dsp` is
//! written to be read against the reference line by line - and this one is
//! checked against it, stage by stage, on the same input.
//!
//! # Weights are uploaded once
//!
//! [`GpuModel::open`] copies all 662 inference tensors to the device and fuses
//! the flow's weight normalisation there, so synthesis touches no host memory
//! except the two noise draws and the final download. At 145 MB that is one
//! transfer of about 20 ms against a synthesis measured in tens of
//! milliseconds - worth doing once and never again, and the reason this is a
//! loaded model rather than a function.
//!
//! # What stays on the host
//!
//! The duration predictor's rational-quadratic spline. It runs on one channel
//! over a few dozen symbol positions, four times; a kernel for it would cost
//! more in launches and round trips than the arithmetic is worth. Everything
//! around it - the depthwise-separable stacks that are the actual work - is on
//! the device, so the round trips are of a `[1, T]` vector.
//!
//! The alignment is host-side for the same reason: it is a cumulative sum over
//! symbols, and it has to come back to the host anyway to size the prior's
//! noise.

use crate::rng::Rng;
use crate::synthesize::SynthesisError;
use std::path::Path;
use xabe_cuda::{CudaError, CudaSlice, Gpu};
use xabe_dsp::spline_inverse;
use xabe_st::StFile;
use xabe_vits::{Conv, DdsConv, DurationFlow, Norm, Tokenizer, VitsConfig, VitsWeights, WnConv};

/// A convolution or dense projection, on the device.
struct GConv {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    /// Output channels. Needed because the WaveNet's last residual-skip layer
    /// is half the width of the others.
    out_ch: usize,
    /// Kernel width, so the call sites need not re-derive the padding rule.
    k: usize,
}

/// A layer norm's scale and shift, on the device.
struct GNorm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// A dilated depthwise-separable stack, on the device.
struct GDds {
    dilated: Vec<GConv>,
    pointwise: Vec<GConv>,
    norms_1: Vec<GNorm>,
    norms_2: Vec<GNorm>,
}

/// One duration coupling block.
enum GDurFlow {
    /// Elementwise affine. Two channels, so it stays on the host.
    Affine {
        log_scale: Vec<f32>,
        translate: Vec<f32>,
    },
    /// A convolutional flow ending in a spline.
    Spline(Box<GSplineFlow>),
}

/// A convolutional duration flow's three parameter groups.
///
/// Boxed inside [`GDurFlow`] because it is far larger than the affine variant,
/// and the enum lives in a `Vec` where every element would otherwise pay for
/// the largest.
struct GSplineFlow {
    conv_pre: GConv,
    conv_dds: GDds,
    conv_proj: GConv,
}

/// The per-stage timings [`GpuModel::synthesize_timed`] reports.
type Timings = Vec<(&'static str, f64)>;

/// What the text encoder leaves on the device: the hidden states in convolution
/// layout, and the prior's mean and log standard deviation in `[t, channels]`.
struct Encoded {
    hidden: CudaSlice<f32>,
    m_p: CudaSlice<f32>,
    logs_p: CudaSlice<f32>,
}

/// One text encoder layer.
struct GEncoderLayer {
    q: GConv,
    k: GConv,
    v: GConv,
    out: GConv,
    emb_rel_k: CudaSlice<f32>,
    emb_rel_v: CudaSlice<f32>,
    norm: GNorm,
    ffn_1: GConv,
    ffn_2: GConv,
    final_norm: GNorm,
}

/// One prior coupling block, with its weight normalisation already fused.
struct GFlowBlock {
    conv_pre: GConv,
    conv_post: GConv,
    wavenet: Vec<(GConv, GConv)>,
}

/// One multi-receptive-field residual block.
struct GResBlock {
    convs1: Vec<GConv>,
    convs2: Vec<GConv>,
}

/// Everything the forward pass reads, resident on the device.
struct GpuWeights {
    embed: CudaSlice<f32>,
    layers: Vec<GEncoderLayer>,
    project: GConv,

    dur_pre: GConv,
    dur_dds: GDds,
    dur_proj: GConv,
    dur_flows: Vec<GDurFlow>,

    flow: Vec<GFlowBlock>,

    dec_pre: GConv,
    upsampler: Vec<GConv>,
    resblocks: Vec<GResBlock>,
    dec_post: GConv,
}

/// A model loaded onto a CUDA device.
pub struct GpuModel {
    gpu: Gpu,
    cfg: VitsConfig,
    tok: Tokenizer,
    w: GpuWeights,
}

impl std::fmt::Debug for GpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuModel")
            .field("layers", &self.w.layers.len())
            .field("flow", &self.w.flow.len())
            .field("resblocks", &self.w.resblocks.len())
            .finish()
    }
}

impl GpuModel {
    /// Loads a model directory onto CUDA device `ordinal`.
    pub fn open(dir: &Path, ordinal: usize) -> Result<Self, SynthesisError> {
        let gpu = Gpu::open(ordinal)?;
        let cfg = VitsConfig::from_json_path(dir.join("config.json"))?;
        let tok = Tokenizer::load(dir)?;
        let file = StFile::open(dir.join("model.safetensors"))?;
        let host = VitsWeights::load(&file, &cfg)?;
        let w = upload(&gpu, &host, &cfg)?;
        tracing::info!(ordinal, "model resident on device");
        Ok(Self { gpu, cfg, tok, w })
    }

    /// The model's geometry.
    pub fn config(&self) -> &VitsConfig {
        &self.cfg
    }

    /// The model's geometry, for the CLI's sampling overrides.
    pub fn config_mut(&mut self) -> &mut VitsConfig {
        &mut self.cfg
    }

    /// Synthesises audio, drawing noise from `seed`.
    pub fn synthesize(&self, text: &str, seed: u64) -> Result<Vec<f32>, SynthesisError> {
        let mut rng = Rng::new(seed);
        let ids = self.tok.encode(text);
        if ids.is_empty() {
            return Err(SynthesisError::NoSymbols {
                text: text.to_string(),
            });
        }
        let noise_dur = rng.normals(2 * ids.len());
        let enc = self.text_encoder(&ids)?;
        let log_duration = self.duration_predictor(&enc.hidden, ids.len(), &noise_dur)?;

        let (alignment, frames) = alignment(&log_duration, self.cfg.speaking_rate);
        let noise_prior = rng.normals(self.cfg.flow_size * frames);
        self.render(&enc, &alignment, frames, &noise_prior)
    }

    /// Synthesises, reporting how long each stage took.
    ///
    /// Costs a device synchronisation between stages, so the total is a little
    /// higher than [`Self::synthesize`] would give. It exists because
    /// `docs/OPTIMIZATION.md` refuses to accept a guess about where the time
    /// goes, and the answer turned out to be worth writing down.
    pub fn synthesize_timed(
        &self,
        text: &str,
        seed: u64,
    ) -> Result<(Vec<f32>, Timings), SynthesisError> {
        use std::time::Instant;
        let mut stages: Timings = Vec::new();
        let mark = |name: &'static str, t: Instant, s: &mut Timings| {
            s.push((name, t.elapsed().as_secs_f64() * 1000.0));
        };

        let mut rng = Rng::new(seed);
        let ids = self.tok.encode(text);
        if ids.is_empty() {
            return Err(SynthesisError::NoSymbols {
                text: text.to_string(),
            });
        }
        let noise_dur = rng.normals(2 * ids.len());

        let t0 = Instant::now();
        let enc = self.text_encoder(&ids)?;
        self.gpu.synchronize()?;
        mark("text_encoder", t0, &mut stages);

        let t0 = Instant::now();
        let log_duration = self.duration_predictor(&enc.hidden, ids.len(), &noise_dur)?;
        self.gpu.synchronize()?;
        mark("duration", t0, &mut stages);

        let (alignment, frames) = alignment(&log_duration, self.cfg.speaking_rate);
        let noise = rng.normals(self.cfg.flow_size * frames);
        let ch = self.cfg.flow_size;

        let t0 = Instant::now();
        let dalign = self.gpu.upload_i32(&alignment)?;
        let dnoise = self.gpu.upload(&noise)?;
        let z_p = self.gpu.expand_prior(
            &enc.m_p,
            &enc.logs_p,
            &dalign,
            &dnoise,
            ch,
            frames,
            self.cfg.noise_scale,
        )?;
        self.gpu.synchronize()?;
        mark("prior", t0, &mut stages);

        let t0 = Instant::now();
        let z = self.flow_reverse(&z_p, frames)?;
        self.gpu.synchronize()?;
        mark("flow", t0, &mut stages);

        let t0 = Instant::now();
        let audio = self.decoder(&z, frames)?;
        self.gpu.synchronize()?;
        mark("decoder", t0, &mut stages);

        let t0 = Instant::now();
        let out = self.gpu.download(&audio)?;
        mark("download", t0, &mut stages);
        Ok((out, stages))
    }

    /// Synthesises from noise the caller supplies, for differential testing.
    ///
    /// Public because the whole point of the CPU path is to be the definition
    /// of correct, and checking that requires feeding both the same draws.
    pub fn synthesize_with_noise(
        &self,
        text: &str,
        noise_dur: &[f32],
        prior_noise: &dyn Fn(usize) -> Vec<f32>,
    ) -> Result<Vec<f32>, SynthesisError> {
        let ids = self.tok.encode(text);
        if ids.is_empty() {
            return Err(SynthesisError::NoSymbols {
                text: text.to_string(),
            });
        }
        let enc = self.text_encoder(&ids)?;
        let log_duration = self.duration_predictor(&enc.hidden, ids.len(), noise_dur)?;
        let (alignment, frames) = alignment(&log_duration, self.cfg.speaking_rate);
        let noise = prior_noise(self.cfg.flow_size * frames);
        self.render(&enc, &alignment, frames, &noise)
    }

    /// Tokenised text to hidden states and the prior's parameters.
    ///
    /// Returns `hidden` as `[hidden_size, t]` - convolution layout, which is
    /// what the duration predictor wants - and `m_p`/`logs_p` as `[t, ch]`.
    fn text_encoder(&self, ids: &[i64]) -> Result<Encoded, SynthesisError> {
        let g = &self.gpu;
        let t = ids.len();
        let ch = self.cfg.hidden_size;
        let eps = self.cfg.layer_norm_eps;

        let dids = g.upload_i64(ids)?;
        let mut h = g.embed_scaled(&self.w.embed, &dids, t, ch, (ch as f32).sqrt())?;

        for layer in &self.w.layers {
            let q = g.linear(&h, &layer.q.w, layer.q.b.as_ref(), t, ch, ch)?;
            let k = g.linear(&h, &layer.k.w, layer.k.b.as_ref(), t, ch, ch)?;
            let v = g.linear(&h, &layer.v.w, layer.v.b.as_ref(), t, ch, ch)?;
            let mut scores = g.attention_scores(
                &q,
                &k,
                &layer.emb_rel_k,
                t,
                ch,
                self.cfg.num_attention_heads,
                self.cfg.window_size,
            )?;
            g.softmax_rows(&mut scores, self.cfg.num_attention_heads * t, t)?;
            let ctx = g.attention_context(
                &scores,
                &v,
                &layer.emb_rel_v,
                t,
                ch,
                self.cfg.num_attention_heads,
                self.cfg.window_size,
            )?;
            let attn = g.linear(&ctx, &layer.out.w, layer.out.b.as_ref(), t, ch, ch)?;
            g.add_inplace(&mut h, &attn, t * ch)?;
            h = g.layer_norm(&h, t, ch, &layer.norm.w, &layer.norm.b, eps)?;

            // The feed-forward is a convolution over time, so it needs the
            // other layout and back.
            let x = g.transpose(&h, t, ch)?;
            let k3 = layer.ffn_1.k;
            let (pl, pr) = xabe_dsp::same_padding(k3);
            let (mut y, _) = g.conv1d(
                &x,
                &layer.ffn_1.w,
                layer.ffn_1.b.as_ref(),
                ch,
                t,
                self.cfg.ffn_dim,
                k3,
                pl,
                pr,
                1,
            )?;
            g.relu(&mut y, self.cfg.ffn_dim * t)?;
            let (z, _) = g.conv1d(
                &y,
                &layer.ffn_2.w,
                layer.ffn_2.b.as_ref(),
                self.cfg.ffn_dim,
                t,
                ch,
                k3,
                pl,
                pr,
                1,
            )?;
            let ff = g.transpose(&z, ch, t)?;
            g.add_inplace(&mut h, &ff, t * ch)?;
            h = g.layer_norm(&h, t, ch, &layer.final_norm.w, &layer.final_norm.b, eps)?;
        }

        let hc = g.transpose(&h, t, ch)?;
        let flow = self.cfg.flow_size;
        let (stats, _) = g.conv1d(
            &hc,
            &self.w.project.w,
            self.w.project.b.as_ref(),
            ch,
            t,
            flow * 2,
            1,
            0,
            0,
            1,
        )?;
        // Split on the channel axis, which is contiguous there, then transpose
        // each half into `[t, ch]`.
        let m_c = g.copy_range(&stats, 0, flow * t)?;
        let l_c = g.copy_range(&stats, flow * t, flow * t)?;
        Ok(Encoded {
            hidden: hc,
            m_p: g.transpose(&m_c, flow, t)?,
            logs_p: g.transpose(&l_c, flow, t)?,
        })
    }

    /// The stochastic duration predictor. `hidden` is `[hidden_size, t]`.
    fn duration_predictor(
        &self,
        hidden: &CudaSlice<f32>,
        t: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>, SynthesisError> {
        let g = &self.gpu;
        let ch = self.cfg.hidden_size;

        let (cond, _) = g.conv1d(
            hidden,
            &self.w.dur_pre.w,
            self.w.dur_pre.b.as_ref(),
            ch,
            t,
            ch,
            1,
            0,
            0,
            1,
        )?;
        let cond = self.dds(&cond, ch, t, &self.w.dur_dds, None)?;
        let (cond, _) = g.conv1d(
            &cond,
            &self.w.dur_proj.w,
            self.w.dur_proj.b.as_ref(),
            ch,
            t,
            ch,
            1,
            0,
            0,
            1,
        )?;

        // Two channels over a few dozen positions: the flow itself is host
        // arithmetic, and only the conditioning stacks inside each block are
        // worth a launch.
        let mut z: Vec<f32> = noise
            .iter()
            .map(|v| v * self.cfg.noise_scale_duration)
            .collect();

        for &i in &reverse_order(self.w.dur_flows.len()) {
            z = xabe_dsp::flip_channels(&z, 2, t);
            match &self.w.dur_flows[i] {
                GDurFlow::Affine {
                    log_scale,
                    translate,
                } => {
                    for c in 0..2 {
                        let inv = (-log_scale[c]).exp();
                        for v in &mut z[c * t..(c + 1) * t] {
                            *v = (*v - translate[c]) * inv;
                        }
                    }
                }
                GDurFlow::Spline(sf) => {
                    let (conv_pre, conv_dds, conv_proj) =
                        (&sf.conv_pre, &sf.conv_dds, &sf.conv_proj);
                    let bins = self.cfg.duration_predictor_flow_bins;
                    let half = self.cfg.depth_separable_channels / 2;
                    let per = bins * 3 - 1;

                    let first = g.upload(&z[..half * t])?;
                    let (h, _) = g.conv1d(
                        &first,
                        &conv_pre.w,
                        conv_pre.b.as_ref(),
                        half,
                        t,
                        ch,
                        1,
                        0,
                        0,
                        1,
                    )?;
                    let h = self.dds(&h, ch, t, conv_dds, Some(&cond))?;
                    let (h, _) = g.conv1d(
                        &h,
                        &conv_proj.w,
                        conv_proj.b.as_ref(),
                        ch,
                        t,
                        half * per,
                        1,
                        0,
                        0,
                        1,
                    )?;
                    let h = g.download(&h)?;

                    let scale = 1.0 / (ch as f32).sqrt();
                    let mut widths = vec![0.0; bins];
                    let mut heights = vec![0.0; bins];
                    let mut derivs = vec![0.0; bins - 1];
                    for c in 0..half {
                        for pos in 0..t {
                            let row = |k: usize| h[(c * per + k) * t + pos];
                            for k in 0..bins {
                                widths[k] = row(k) * scale;
                                heights[k] = row(bins + k) * scale;
                            }
                            for (k, d) in derivs.iter_mut().enumerate() {
                                *d = row(2 * bins + k);
                            }
                            let idx = half * t + c * t + pos;
                            z[idx] = spline_inverse(
                                z[idx],
                                &widths,
                                &heights,
                                &derivs,
                                self.cfg.duration_predictor_tail_bound,
                            );
                        }
                    }
                }
            }
        }
        z.truncate(t);
        Ok(z)
    }

    /// The dilated depthwise-separable stack.
    fn dds(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
        w: &GDds,
        cond: Option<&CudaSlice<f32>>,
    ) -> Result<CudaSlice<f32>, SynthesisError> {
        let g = &self.gpu;
        let k = self.cfg.duration_predictor_kernel_size;
        let mut inputs = g.copy_range(x, 0, ch * t)?;
        if let Some(c) = cond {
            g.add_inplace(&mut inputs, c, ch * t)?;
        }

        for i in 0..self.cfg.depth_separable_num_layers {
            let dilation = k.pow(i as u32);
            let pad = (k * dilation - dilation) / 2;
            let (h, _) = g.depthwise_conv1d(
                &inputs,
                &w.dilated[i].w,
                w.dilated[i].b.as_ref(),
                ch,
                t,
                k,
                pad,
                pad,
                dilation,
            )?;
            let mut h = self.norm_channels(&h, ch, t, &w.norms_1[i])?;
            g.gelu(&mut h, ch * t)?;
            let (h, _) = g.conv1d(
                &h,
                &w.pointwise[i].w,
                w.pointwise[i].b.as_ref(),
                ch,
                t,
                ch,
                1,
                0,
                0,
                1,
            )?;
            let mut h = self.norm_channels(&h, ch, t, &w.norms_2[i])?;
            g.gelu(&mut h, ch * t)?;
            g.add_inplace(&mut inputs, &h, ch * t)?;
        }
        Ok(inputs)
    }

    /// Layer-norms across channels for each position, in convolution layout.
    fn norm_channels(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
        n: &GNorm,
    ) -> Result<CudaSlice<f32>, SynthesisError> {
        let g = &self.gpu;
        let tc = g.transpose(x, ch, t)?;
        // Bare `nn.LayerNorm` in the reference, so PyTorch's default epsilon
        // rather than the config's - see the CPU twin.
        let normed = g.layer_norm(&tc, t, ch, &n.w, &n.b, 1e-5)?;
        Ok(g.transpose(&normed, t, ch)?)
    }

    /// Expands the prior, inverts the flow, decodes.
    fn render(
        &self,
        enc: &Encoded,
        alignment: &[i32],
        frames: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>, SynthesisError> {
        let g = &self.gpu;
        let ch = self.cfg.flow_size;

        let dalign = g.upload_i32(alignment)?;
        let dnoise = g.upload(noise)?;
        let z_p = g.expand_prior(
            &enc.m_p,
            &enc.logs_p,
            &dalign,
            &dnoise,
            ch,
            frames,
            self.cfg.noise_scale,
        )?;

        let z = self.flow_reverse(&z_p, frames)?;
        let audio = self.decoder(&z, frames)?;
        Ok(g.download(&audio)?)
    }

    /// The prior flow, inverted.
    fn flow_reverse(
        &self,
        z_p: &CudaSlice<f32>,
        frames: usize,
    ) -> Result<CudaSlice<f32>, SynthesisError> {
        let g = &self.gpu;
        let ch = self.cfg.flow_size;
        let half = self.cfg.flow_half();
        let hidden = self.cfg.hidden_size;
        let k = self.cfg.wavenet_kernel_size;

        let mut x = g.copy_range(z_p, 0, ch * frames)?;
        for block in self.w.flow.iter().rev() {
            x = g.flip_channels(&x, ch, frames)?;
            let first = g.copy_range(&x, 0, half * frames)?;

            let (h, _) = g.conv1d(
                &first,
                &block.conv_pre.w,
                block.conv_pre.b.as_ref(),
                half,
                frames,
                hidden,
                1,
                0,
                0,
                1,
            )?;

            let mut inputs = h;
            let mut outputs = g.zeros(hidden * frames)?;
            let n = block.wavenet.len();
            for (i, (in_layer, res_skip)) in block.wavenet.iter().enumerate() {
                let dilation = self.cfg.wavenet_dilation_rate.pow(i as u32);
                let pad = (k * dilation - dilation) / 2;
                let (gated, _) = g.conv1d(
                    &inputs,
                    &in_layer.w,
                    in_layer.b.as_ref(),
                    hidden,
                    frames,
                    2 * hidden,
                    k,
                    pad,
                    pad,
                    dilation,
                )?;
                let acts = g.gated_activation(&gated, hidden, frames)?;
                let (rs, _) = g.conv1d(
                    &acts,
                    &res_skip.w,
                    res_skip.b.as_ref(),
                    hidden,
                    frames,
                    res_skip.out_ch,
                    1,
                    0,
                    0,
                    1,
                )?;
                if i < n - 1 {
                    let res = g.copy_range(&rs, 0, hidden * frames)?;
                    let skip = g.copy_range(&rs, hidden * frames, hidden * frames)?;
                    g.add_inplace(&mut inputs, &res, hidden * frames)?;
                    g.add_inplace(&mut outputs, &skip, hidden * frames)?;
                } else {
                    g.add_inplace(&mut outputs, &rs, hidden * frames)?;
                }
            }

            let (mean, _) = g.conv1d(
                &outputs,
                &block.conv_post.w,
                block.conv_post.b.as_ref(),
                hidden,
                frames,
                half,
                1,
                0,
                0,
                1,
            )?;
            // Mean-only coupling, so the inverse is a subtraction.
            let mut second = g.copy_range(&x, half * frames, half * frames)?;
            g.sub_inplace(&mut second, &mean, half * frames)?;

            // Concatenate the untouched half with the transformed one.
            let mut next = g.zeros(ch * frames)?;
            g.copy_into(&mut next, &first, 0, half * frames)?;
            g.copy_into(&mut next, &second, half * frames, half * frames)?;
            x = next;
        }
        Ok(x)
    }

    /// The HiFi-GAN decoder.
    fn decoder(&self, z: &CudaSlice<f32>, frames: usize) -> Result<CudaSlice<f32>, SynthesisError> {
        let g = &self.gpu;
        let per_stage = self.cfg.resblocks_per_stage();
        let kp = self.w.dec_pre.k;

        let (mut h, _) = g.conv1d(
            z,
            &self.w.dec_pre.w,
            self.w.dec_pre.b.as_ref(),
            self.cfg.flow_size,
            frames,
            self.cfg.upsample_initial_channel,
            kp,
            kp / 2,
            kp / 2,
            1,
        )?;
        let mut t = frames;
        let mut ch = self.cfg.upsample_initial_channel;

        for (stage, up) in self.w.upsampler.iter().enumerate() {
            g.leaky_relu(&mut h, ch * t, self.cfg.leaky_relu_slope)?;
            let stride = self.cfg.upsample_rates[stage];
            let pad = (up.k - stride) / 2;
            let (u, out_t) = g.transposed_conv1d(
                &h,
                &up.w,
                up.b.as_ref(),
                ch,
                t,
                up.out_ch,
                up.k,
                stride,
                pad,
            )?;
            h = u;
            t = out_t;
            ch = up.out_ch;

            let mut fused = g.zeros(ch * t)?;
            for j in 0..per_stage {
                let block = &self.w.resblocks[stage * per_stage + j];
                let out = self.resblock(&h, ch, t, block, &self.cfg.resblock_dilation_sizes[j])?;
                g.add_inplace(&mut fused, &out, ch * t)?;
            }
            g.scale_inplace(&mut fused, ch * t, 1.0 / per_stage as f32)?;
            h = fused;
        }

        // No slope argument in the reference, so 0.01 rather than 0.1.
        g.leaky_relu(&mut h, ch * t, 0.01)?;
        let kq = self.w.dec_post.k;
        let (mut out, _) = g.conv1d(
            &h,
            &self.w.dec_post.w,
            self.w.dec_post.b.as_ref(),
            ch,
            t,
            1,
            kq,
            kq / 2,
            kq / 2,
            1,
        )?;
        g.tanh(&mut out, t)?;
        Ok(out)
    }

    /// One multi-receptive-field residual block.
    fn resblock(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
        block: &GResBlock,
        dilations: &[usize],
    ) -> Result<CudaSlice<f32>, SynthesisError> {
        let g = &self.gpu;
        let mut h = g.copy_range(x, 0, ch * t)?;
        for (i, dilation) in dilations.iter().copied().enumerate() {
            let residual = g.copy_range(&h, 0, ch * t)?;

            let c1 = &block.convs1[i];
            g.leaky_relu(&mut h, ch * t, self.cfg.leaky_relu_slope)?;
            let pad = (c1.k * dilation - dilation) / 2;
            let (y, _) = g.conv1d(
                &h,
                &c1.w,
                c1.b.as_ref(),
                ch,
                t,
                ch,
                c1.k,
                pad,
                pad,
                dilation,
            )?;
            h = y;

            let c2 = &block.convs2[i];
            g.leaky_relu(&mut h, ch * t, self.cfg.leaky_relu_slope)?;
            let pad = (c2.k - 1) / 2;
            let (y, _) = g.conv1d(&h, &c2.w, c2.b.as_ref(), ch, t, ch, c2.k, pad, pad, 1)?;
            h = y;

            g.add_inplace(&mut h, &residual, ch * t)?;
        }
        Ok(h)
    }
}

/// The reverse order the duration flow actually uses: reversed, then with the
/// second entry of the original list dropped. See the CPU twin.
fn reverse_order(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (2..n).rev().collect();
    order.push(0);
    order
}

/// Frames per symbol, and the total.
fn alignment(log_duration: &[f32], speaking_rate: f32) -> (Vec<i32>, usize) {
    let mut out = Vec::new();
    for (s, v) in log_duration.iter().enumerate() {
        let d = (v.exp() / speaking_rate).ceil().max(0.0) as usize;
        out.extend(std::iter::repeat_n(s as i32, d));
    }
    if out.is_empty() {
        out.push(0);
    }
    let frames = out.len();
    (out, frames)
}

/// Copies every inference tensor to the device.
fn upload(g: &Gpu, w: &VitsWeights<'_>, cfg: &VitsConfig) -> Result<GpuWeights, SynthesisError> {
    let conv = |c: &Conv<'_>| -> Result<GConv, CudaError> {
        Ok(GConv {
            w: g.upload(c.weight)?,
            b: match c.bias {
                Some(b) => Some(g.upload(b)?),
                None => None,
            },
            out_ch: c.out_ch,
            k: c.k,
        })
    };
    let norm = |n: &Norm<'_>| -> Result<GNorm, CudaError> {
        Ok(GNorm {
            w: g.upload(n.weight)?,
            b: g.upload(n.bias)?,
        })
    };
    // Weight normalisation is fused on the device, using the same kernel the
    // differential tests cover.
    let wn = |c: &WnConv<'_>| -> Result<GConv, CudaError> {
        let v = g.upload(c.weight_v)?;
        let gg = g.upload(c.weight_g)?;
        Ok(GConv {
            w: g.fuse_weight_norm(&v, &gg, c.out_ch, c.in_ch, c.k)?,
            b: Some(g.upload(c.bias)?),
            out_ch: c.out_ch,
            k: c.k,
        })
    };
    let dds = |d: &DdsConv<'_>| -> Result<GDds, CudaError> {
        Ok(GDds {
            dilated: d.dilated.iter().map(&conv).collect::<Result<_, _>>()?,
            pointwise: d.pointwise.iter().map(&conv).collect::<Result<_, _>>()?,
            norms_1: d.norms_1.iter().map(&norm).collect::<Result<_, _>>()?,
            norms_2: d.norms_2.iter().map(&norm).collect::<Result<_, _>>()?,
        })
    };

    let mut layers = Vec::with_capacity(w.text_encoder.layers.len());
    for l in &w.text_encoder.layers {
        layers.push(GEncoderLayer {
            q: conv(&l.q)?,
            k: conv(&l.k)?,
            v: conv(&l.v)?,
            out: conv(&l.out)?,
            emb_rel_k: g.upload(l.emb_rel_k)?,
            emb_rel_v: g.upload(l.emb_rel_v)?,
            norm: norm(&l.norm)?,
            ffn_1: conv(&l.ffn_1)?,
            ffn_2: conv(&l.ffn_2)?,
            final_norm: norm(&l.final_norm)?,
        });
    }

    let mut dur_flows = Vec::with_capacity(w.duration_predictor.flows.len());
    for f in &w.duration_predictor.flows {
        dur_flows.push(match f {
            DurationFlow::Affine {
                log_scale,
                translate,
            } => GDurFlow::Affine {
                log_scale: log_scale.to_vec(),
                translate: translate.to_vec(),
            },
            DurationFlow::Spline {
                conv_pre,
                conv_dds,
                conv_proj,
            } => GDurFlow::Spline(Box::new(GSplineFlow {
                conv_pre: conv(conv_pre)?,
                conv_dds: dds(conv_dds)?,
                conv_proj: conv(conv_proj)?,
            })),
        });
    }

    let mut flow = Vec::with_capacity(w.flow.len());
    for b in &w.flow {
        let mut wavenet = Vec::with_capacity(b.wavenet.len());
        for l in &b.wavenet {
            wavenet.push((wn(&l.in_layer)?, wn(&l.res_skip)?));
        }
        flow.push(GFlowBlock {
            conv_pre: conv(&b.conv_pre)?,
            conv_post: conv(&b.conv_post)?,
            wavenet,
        });
    }

    let mut resblocks = Vec::with_capacity(w.decoder.resblocks.len());
    for r in &w.decoder.resblocks {
        resblocks.push(GResBlock {
            convs1: r.convs1.iter().map(&conv).collect::<Result<_, _>>()?,
            convs2: r.convs2.iter().map(&conv).collect::<Result<_, _>>()?,
        });
    }

    let _ = cfg;
    Ok(GpuWeights {
        embed: g.upload(w.text_encoder.embed)?,
        layers,
        project: conv(&w.text_encoder.project)?,
        dur_pre: conv(&w.duration_predictor.conv_pre)?,
        dur_dds: dds(&w.duration_predictor.conv_dds)?,
        dur_proj: conv(&w.duration_predictor.conv_proj)?,
        dur_flows,
        flow,
        dec_pre: conv(&w.decoder.conv_pre)?,
        upsampler: w
            .decoder
            .upsampler
            .iter()
            .map(&conv)
            .collect::<Result<_, _>>()?,
        resblocks,
        dec_post: conv(&w.decoder.conv_post)?,
    })
}
