//! Binding a checkpoint to a validated [`VitsConfig`].
//!
//! # Why this is a whole module
//!
//! VITS fails quietly. A tensor read at the wrong shape still produces audio,
//! just wrong audio, and no listening test on a language you do not speak will
//! catch it. So every tensor is fetched through
//! [`StFile::tensor_shaped`](xabe_st::StFile::tensor_shaped) with the geometry
//! the config implies: a checkpoint that disagrees fails here, naming the
//! tensor, rather than at synthesis time naming nothing.
//!
//! # What is (and isn't) here
//!
//! Borrowed slices and their shapes. No arithmetic — not even the weight-norm
//! fusion, which needs a division and therefore belongs to `xabe-dsp`.
//!
//! `posterior_encoder` is not read. It encodes ground-truth spectrograms during
//! training and has no role in synthesis; skipping it leaves 100 of the
//! checkpoint's 762 tensors untouched.
//!
//! Start at [`VitsWeights::load`].

use xabe_st::StFile;

use crate::config::VitsConfig;
use crate::error::WeightError;

/// A convolution's weight and bias, borrowed from the mapping.
#[derive(Debug, Clone, Copy)]
pub struct Conv<'a> {
    /// Kernel, laid out `[out_channels, in_channels, kernel]`.
    pub weight: &'a [f32],
    /// Per-output-channel bias, or `None` where the layer has none.
    pub bias: Option<&'a [f32]>,
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Kernel width.
    pub k: usize,
}

/// A weight-normalised convolution, stored unfused.
///
/// PyTorch's `weight_norm` keeps the direction `v` and the magnitude `g`
/// separately, and the effective kernel is `g * v / ||v||`, normalised over all
/// axes but the first. Fusing it is a division, so it happens in `xabe-dsp`;
/// this type only carries the parts.
///
/// In this checkpoint **only the flow's WaveNet layers are stored unfused**.
/// The decoder's upsamplers and resblocks carry a plain `weight`, because the
/// export removed their weight-norm parameterisation. Do not assume either
/// convention holds across the whole file - the loader checks which it finds.
#[derive(Debug, Clone, Copy)]
pub struct WnConv<'a> {
    /// Direction, laid out `[out_channels, in_channels, kernel]`.
    pub weight_v: &'a [f32],
    /// Magnitude, laid out `[out_channels, 1, 1]`.
    pub weight_g: &'a [f32],
    /// Per-output-channel bias.
    pub bias: &'a [f32],
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Kernel width.
    pub k: usize,
}

/// A convolution the checkpoint may store either fused or weight-normalised.
///
/// The two published checkpoints disagree here and nowhere else. The 🤗 export
/// of `mms-tts-nan` removed the weight-norm parameterisation from the decoder's
/// upsamplers and resblocks, so they arrive as a plain `weight`; the Coqui
/// trainer's own save keeps it, so the same convolutions arrive as `original0`
/// and `original1`. They are the same convolution once fused, so the difference
/// belongs here - in what was read - and not in the forward pass, which asks
/// for a kernel and does not care which form it came from.
#[derive(Debug, Clone, Copy)]
pub enum MaybeWn<'a> {
    /// Stored fused: one `weight` tensor, ready to convolve with.
    Fused(Conv<'a>),
    /// Stored the way `weight_norm` keeps it: a direction and a magnitude,
    /// which multiply to the kernel after a division `xabe-dsp` performs.
    Normalised(WnConv<'a>),
}

impl<'a> MaybeWn<'a> {
    /// Output channels.
    ///
    /// For a transposed convolution this is still the *output* count, which is
    /// the second dimension of the stored kernel rather than the first - see
    /// [`Decoder::upsampler`].
    pub fn out_ch(&self) -> usize {
        match self {
            Self::Fused(c) => c.out_ch,
            Self::Normalised(c) => c.out_ch,
        }
    }

    /// Input channels.
    pub fn in_ch(&self) -> usize {
        match self {
            Self::Fused(c) => c.in_ch,
            Self::Normalised(c) => c.in_ch,
        }
    }

    /// Kernel width.
    pub fn k(&self) -> usize {
        match self {
            Self::Fused(c) => c.k,
            Self::Normalised(c) => c.k,
        }
    }

    /// Per-output-channel bias, if the layer has one.
    pub fn bias(&self) -> Option<&'a [f32]> {
        match self {
            Self::Fused(c) => c.bias,
            Self::Normalised(c) => Some(c.bias),
        }
    }

    /// Elements bound, counting the magnitude when there is one.
    fn elements(&self) -> usize {
        match self {
            Self::Fused(c) => c.elements(),
            Self::Normalised(c) => c.elements(),
        }
    }
}

/// A layer normalisation's learned scale and shift.
#[derive(Debug, Clone, Copy)]
pub struct Norm<'a> {
    /// Scale.
    pub weight: &'a [f32],
    /// Shift.
    pub bias: &'a [f32],
}

/// One text-encoder layer: relative self-attention then a convolutional FFN.
#[derive(Debug, Clone, Copy)]
pub struct EncoderLayer<'a> {
    /// Query projection, `[hidden, hidden]`.
    pub q: Conv<'a>,
    /// Key projection.
    pub k: Conv<'a>,
    /// Value projection.
    pub v: Conv<'a>,
    /// Output projection.
    pub out: Conv<'a>,
    /// Relative key embeddings, `[1, 2*window+1, head_dim]`.
    pub emb_rel_k: &'a [f32],
    /// Relative value embeddings, `[1, 2*window+1, head_dim]`.
    pub emb_rel_v: &'a [f32],
    /// Post-attention norm.
    pub norm: Norm<'a>,
    /// First FFN convolution, widening to `ffn_dim`.
    pub ffn_1: Conv<'a>,
    /// Second FFN convolution, narrowing back.
    pub ffn_2: Conv<'a>,
    /// Post-FFN norm.
    pub final_norm: Norm<'a>,
}

/// The text encoder: embedding, transformer stack, and the projection that
/// splits into the prior's mean and log-variance.
#[derive(Debug)]
pub struct TextEncoder<'a> {
    /// Symbol embedding table, `[vocab, hidden]`.
    pub embed: &'a [f32],
    /// Transformer layers.
    pub layers: Vec<EncoderLayer<'a>>,
    /// Projection to `2 * flow_size` channels: mean then log-variance.
    pub project: Conv<'a>,
}

/// One WaveNet layer inside a coupling block.
#[derive(Debug, Clone, Copy)]
pub struct WaveNetLayer<'a> {
    /// Dilated input convolution, producing gate and filter halves.
    pub in_layer: WnConv<'a>,
    /// Residual and skip projection.
    pub res_skip: WnConv<'a>,
}

/// One affine coupling block of the normalising flow.
#[derive(Debug)]
pub struct FlowBlock<'a> {
    /// Projection from half-width up to full width.
    pub conv_pre: Conv<'a>,
    /// The WaveNet stack that conditions the coupling.
    pub wavenet: Vec<WaveNetLayer<'a>>,
    /// Projection back down to half width.
    pub conv_post: Conv<'a>,
}

/// A depthwise-separable dilated convolution stack, used throughout the
/// duration predictor.
#[derive(Debug)]
pub struct DdsConv<'a> {
    /// Depthwise convolutions, one per depth.
    pub dilated: Vec<Conv<'a>>,
    /// Pointwise convolutions, one per depth.
    pub pointwise: Vec<Conv<'a>>,
    /// Norms applied after the depthwise stage.
    pub norms_1: Vec<Norm<'a>>,
    /// Norms applied after the pointwise stage.
    pub norms_2: Vec<Norm<'a>>,
}

/// One coupling block of the duration predictor's flow.
///
/// The first block is an elementwise affine (`log_scale` and `translate` only);
/// the rest are spline couplings whose projection emits `3 * bins - 1`
/// parameters per channel.
#[derive(Debug)]
pub enum DurationFlow<'a> {
    /// Elementwise affine transform.
    Affine {
        /// Log scale, `[2, 1]`.
        log_scale: &'a [f32],
        /// Translation, `[2, 1]`.
        translate: &'a [f32],
    },
    /// Rational-quadratic spline coupling.
    Spline {
        /// Projection from one channel up to model width.
        conv_pre: Conv<'a>,
        /// The conditioning stack.
        conv_dds: DdsConv<'a>,
        /// Projection to spline parameters, `[3 * bins - 1, hidden, 1]`.
        conv_proj: Conv<'a>,
    },
}

/// The stochastic duration predictor.
#[derive(Debug)]
pub struct DurationPredictor<'a> {
    /// Input projection.
    pub conv_pre: Conv<'a>,
    /// Conditioning stack.
    pub conv_dds: DdsConv<'a>,
    /// Output projection.
    pub conv_proj: Conv<'a>,
    /// The flow run in reverse to sample durations.
    pub flows: Vec<DurationFlow<'a>>,
    /// Posterior input projection, used by the `post_flows` branch.
    pub post_conv_pre: Conv<'a>,
    /// Posterior conditioning stack.
    pub post_conv_dds: DdsConv<'a>,
    /// Posterior output projection.
    pub post_conv_proj: Conv<'a>,
    /// Posterior flow.
    pub post_flows: Vec<DurationFlow<'a>>,
}

/// One multi-receptive-field resblock of the HiFi-GAN decoder.
#[derive(Debug)]
pub struct ResBlock<'a> {
    /// Dilated convolutions.
    pub convs1: Vec<MaybeWn<'a>>,
    /// Undilated convolutions paired with them.
    pub convs2: Vec<MaybeWn<'a>>,
}

/// The HiFi-GAN decoder.
#[derive(Debug)]
pub struct Decoder<'a> {
    /// Input projection from flow width to `upsample_initial_channel`.
    pub conv_pre: Conv<'a>,
    /// Transposed convolutions, laid out `[in_channels, out_channels, kernel]`
    /// — the opposite order to every other convolution in the model, and the
    /// reason `Conv::out_ch` here is the *second* dimension.
    pub upsampler: Vec<MaybeWn<'a>>,
    /// Resblocks, flattened stage-major as the checkpoint stores them.
    pub resblocks: Vec<ResBlock<'a>>,
    /// Final projection to one channel. Has no bias.
    pub conv_post: Conv<'a>,
}

/// Every tensor the inference path reads, bound and shape-checked.
#[derive(Debug)]
pub struct VitsWeights<'a> {
    /// Text encoder.
    pub text_encoder: TextEncoder<'a>,
    /// Stochastic duration predictor.
    pub duration_predictor: DurationPredictor<'a>,
    /// Normalising flow.
    pub flow: Vec<FlowBlock<'a>>,
    /// HiFi-GAN decoder.
    pub decoder: Decoder<'a>,
}

/// Fetches a convolution with an explicit shape and a bias.
fn conv<'a>(
    f: &'a StFile,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<Conv<'a>, WeightError> {
    let shape: Vec<usize> = if k == 0 {
        vec![out_ch, in_ch]
    } else {
        vec![out_ch, in_ch, k]
    };
    Ok(Conv {
        weight: f.tensor_shaped(&format!("{prefix}.weight"), &shape)?,
        bias: Some(f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?),
        out_ch,
        in_ch,
        k: k.max(1),
    })
}

/// Fetches a transposed convolution.
///
/// These are stored `[in_channels, out_channels, kernel]` - the reverse of every
/// other convolution here - while the bias is still per *output* channel. That
/// asymmetry is why this cannot go through [`conv`].
fn conv_transposed<'a>(
    f: &'a StFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
) -> Result<Conv<'a>, WeightError> {
    Ok(Conv {
        weight: f.tensor_shaped(&format!("{prefix}.weight"), &[in_ch, out_ch, k])?,
        bias: Some(f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?),
        out_ch,
        in_ch,
        k,
    })
}

/// Fetches a weight-normalised convolution, unfused.
fn wn_conv<'a>(
    f: &'a StFile,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<WnConv<'a>, WeightError> {
    Ok(WnConv {
        weight_v: f.tensor_shaped(&format!("{prefix}.weight_v"), &[out_ch, in_ch, k])?,
        weight_g: f.tensor_shaped(&format!("{prefix}.weight_g"), &[out_ch, 1, 1])?,
        bias: f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?,
        out_ch,
        in_ch,
        k,
    })
}

/// Fetches a layer norm.
fn norm<'a>(f: &'a StFile, prefix: &str, ch: usize) -> Result<Norm<'a>, WeightError> {
    Ok(Norm {
        weight: f.tensor_shaped(&format!("{prefix}.weight"), &[ch])?,
        bias: f.tensor_shaped(&format!("{prefix}.bias"), &[ch])?,
    })
}

/// Fetches a depthwise-separable dilated stack of `depth` levels.
fn dds<'a>(
    f: &'a StFile,
    prefix: &str,
    depth: usize,
    ch: usize,
    k: usize,
) -> Result<DdsConv<'a>, WeightError> {
    let mut dilated = Vec::with_capacity(depth);
    let mut pointwise = Vec::with_capacity(depth);
    let mut norms_1 = Vec::with_capacity(depth);
    let mut norms_2 = Vec::with_capacity(depth);
    for i in 0..depth {
        // Depthwise: one input channel per group, so the stored `in_ch` is 1.
        dilated.push(conv(f, &format!("{prefix}.convs_dilated.{i}"), ch, 1, k)?);
        pointwise.push(conv(
            f,
            &format!("{prefix}.convs_pointwise.{i}"),
            ch,
            ch,
            1,
        )?);
        norms_1.push(norm(f, &format!("{prefix}.norms_1.{i}"), ch)?);
        norms_2.push(norm(f, &format!("{prefix}.norms_2.{i}"), ch)?);
    }
    Ok(DdsConv {
        dilated,
        pointwise,
        norms_1,
        norms_2,
    })
}

/// Fetches a duration-predictor flow branch.
///
/// Index 0 is the elementwise affine; the remainder are spline couplings. The
/// checkpoint stores `num_flows + 1` entries for `num_flows` couplings.
fn duration_flows<'a>(
    f: &'a StFile,
    prefix: &str,
    cfg: &VitsConfig,
) -> Result<Vec<DurationFlow<'a>>, WeightError> {
    let mut flows = Vec::new();
    flows.push(DurationFlow::Affine {
        log_scale: f.tensor_shaped(&format!("{prefix}.0.log_scale"), &[2, 1])?,
        translate: f.tensor_shaped(&format!("{prefix}.0.translate"), &[2, 1])?,
    });

    let hidden = cfg.hidden_size;
    let k = cfg.duration_predictor_kernel_size;
    let depth = cfg.depth_separable_channels + 1;
    for i in 1..=cfg.duration_predictor_num_flows {
        let p = format!("{prefix}.{i}");
        let conv_proj_out = f
            .info(&format!("{p}.conv_proj.weight"))
            .map_or(0, |t| t.shape[0]);
        flows.push(DurationFlow::Spline {
            conv_pre: conv(f, &format!("{p}.conv_pre"), hidden, 1, 1)?,
            conv_dds: dds(f, &format!("{p}.conv_dds"), depth, hidden, k)?,
            conv_proj: conv(f, &format!("{p}.conv_proj"), conv_proj_out, hidden, 1)?,
        });
    }
    Ok(flows)
}

impl<'a> VitsWeights<'a> {
    /// Binds every inference tensor, checking each shape against `cfg`.
    ///
    /// Returns on the first disagreement, naming the tensor. Nothing is copied:
    /// each field borrows from the file's mapping.
    pub fn load(f: &'a StFile, cfg: &VitsConfig) -> Result<Self, WeightError> {
        let hidden = cfg.hidden_size;
        let flow_half = cfg.flow_half();

        // ---- text encoder ------------------------------------------------
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("text_encoder.encoder.layers.{i}");
            let rel = [1, cfg.rel_window(), cfg.head_dim()];
            layers.push(EncoderLayer {
                q: conv(f, &format!("{p}.attention.q_proj"), hidden, hidden, 0)?,
                k: conv(f, &format!("{p}.attention.k_proj"), hidden, hidden, 0)?,
                v: conv(f, &format!("{p}.attention.v_proj"), hidden, hidden, 0)?,
                out: conv(f, &format!("{p}.attention.out_proj"), hidden, hidden, 0)?,
                emb_rel_k: f.tensor_shaped(&format!("{p}.attention.emb_rel_k"), &rel)?,
                emb_rel_v: f.tensor_shaped(&format!("{p}.attention.emb_rel_v"), &rel)?,
                norm: norm(f, &format!("{p}.layer_norm"), hidden)?,
                ffn_1: conv(
                    f,
                    &format!("{p}.feed_forward.conv_1"),
                    cfg.ffn_dim,
                    hidden,
                    cfg.ffn_kernel_size,
                )?,
                ffn_2: conv(
                    f,
                    &format!("{p}.feed_forward.conv_2"),
                    hidden,
                    cfg.ffn_dim,
                    cfg.ffn_kernel_size,
                )?,
                final_norm: norm(f, &format!("{p}.final_layer_norm"), hidden)?,
            });
        }
        let text_encoder = TextEncoder {
            embed: f.tensor_shaped(
                "text_encoder.embed_tokens.weight",
                &[cfg.vocab_size, hidden],
            )?,
            layers,
            project: conv(f, "text_encoder.project", 2 * cfg.flow_size, hidden, 1)?,
        };

        // ---- duration predictor ------------------------------------------
        let depth = cfg.depth_separable_channels + 1;
        let dk = cfg.duration_predictor_kernel_size;
        let duration_predictor = DurationPredictor {
            conv_pre: conv(f, "duration_predictor.conv_pre", hidden, hidden, 1)?,
            conv_dds: dds(f, "duration_predictor.conv_dds", depth, hidden, dk)?,
            conv_proj: conv(f, "duration_predictor.conv_proj", hidden, hidden, 1)?,
            flows: duration_flows(f, "duration_predictor.flows", cfg)?,
            post_conv_pre: conv(f, "duration_predictor.post_conv_pre", hidden, 1, 1)?,
            post_conv_dds: dds(f, "duration_predictor.post_conv_dds", depth, hidden, dk)?,
            post_conv_proj: conv(f, "duration_predictor.post_conv_proj", hidden, hidden, 1)?,
            post_flows: duration_flows(f, "duration_predictor.post_flows", cfg)?,
        };

        // ---- flow ---------------------------------------------------------
        let mut flow = Vec::with_capacity(cfg.prior_encoder_num_flows);
        for i in 0..cfg.prior_encoder_num_flows {
            let p = format!("flow.flows.{i}");
            let mut wavenet = Vec::with_capacity(cfg.prior_encoder_num_wavenet_layers);
            for j in 0..cfg.prior_encoder_num_wavenet_layers {
                wavenet.push(WaveNetLayer {
                    in_layer: wn_conv(
                        f,
                        &format!("{p}.wavenet.in_layers.{j}"),
                        2 * hidden,
                        hidden,
                        cfg.wavenet_kernel_size,
                    )?,
                    // The last layer projects to residual only; every earlier
                    // one emits residual and skip, hence the doubled width.
                    res_skip: wn_conv(
                        f,
                        &format!("{p}.wavenet.res_skip_layers.{j}"),
                        if j + 1 < cfg.prior_encoder_num_wavenet_layers {
                            2 * hidden
                        } else {
                            hidden
                        },
                        hidden,
                        1,
                    )?,
                });
            }
            flow.push(FlowBlock {
                conv_pre: conv(f, &format!("{p}.conv_pre"), hidden, flow_half, 1)?,
                wavenet,
                conv_post: conv(f, &format!("{p}.conv_post"), flow_half, hidden, 1)?,
            });
        }

        // ---- decoder -------------------------------------------------------
        let stages = cfg.num_upsample_stages();
        let mut upsampler = Vec::with_capacity(stages);
        for s in 0..stages {
            upsampler.push(MaybeWn::Fused(conv_transposed(
                f,
                &format!("decoder.upsampler.{s}"),
                cfg.upsample_in_channels(s),
                cfg.upsample_out_channels(s),
                cfg.upsample_kernel_sizes[s],
            )?));
        }

        let mut resblocks = Vec::with_capacity(cfg.num_resblocks());
        for s in 0..stages {
            let ch = cfg.upsample_out_channels(s);
            for (r, &k) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let idx = s * cfg.resblocks_per_stage() + r;
                let n = cfg.resblock_dilation_sizes[r].len();
                let mut convs1 = Vec::with_capacity(n);
                let mut convs2 = Vec::with_capacity(n);
                for c in 0..n {
                    convs1.push(MaybeWn::Fused(conv(
                        f,
                        &format!("decoder.resblocks.{idx}.convs1.{c}"),
                        ch,
                        ch,
                        k,
                    )?));
                    convs2.push(MaybeWn::Fused(conv(
                        f,
                        &format!("decoder.resblocks.{idx}.convs2.{c}"),
                        ch,
                        ch,
                        k,
                    )?));
                }
                resblocks.push(ResBlock { convs1, convs2 });
            }
        }

        let last_ch = cfg.upsample_out_channels(stages - 1);
        let decoder = Decoder {
            conv_pre: conv(
                f,
                "decoder.conv_pre",
                cfg.upsample_initial_channel,
                cfg.flow_size,
                7,
            )?,
            upsampler,
            resblocks,
            // conv_post carries no bias in this checkpoint.
            conv_post: Conv {
                weight: f.tensor_shaped("decoder.conv_post.weight", &[1, last_ch, 7])?,
                bias: None,
                out_ch: 1,
                in_ch: last_ch,
                k: 7,
            },
        };

        tracing::debug!(
            layers = cfg.num_hidden_layers,
            flows = cfg.prior_encoder_num_flows,
            resblocks = resblocks_len(&decoder),
            "bound VITS weights"
        );

        Ok(Self {
            text_encoder,
            duration_predictor,
            flow,
            decoder,
        })
    }
}

/// Resblock count, for the load-time trace.
fn resblocks_len(d: &Decoder<'_>) -> usize {
    d.resblocks.len()
}

impl Conv<'_> {
    /// Elements in this convolution's weight and bias.
    fn elements(&self) -> usize {
        self.weight.len() + self.bias.map_or(0, <[f32]>::len)
    }
}

impl WnConv<'_> {
    /// Elements in this convolution's direction, magnitude and bias.
    fn elements(&self) -> usize {
        self.weight_v.len() + self.weight_g.len() + self.bias.len()
    }
}

impl Norm<'_> {
    /// Elements in this norm's scale and shift.
    fn elements(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

impl DdsConv<'_> {
    /// Elements across the whole depthwise-separable stack.
    fn elements(&self) -> usize {
        self.dilated.iter().map(Conv::elements).sum::<usize>()
            + self.pointwise.iter().map(Conv::elements).sum::<usize>()
            + self.norms_1.iter().map(Norm::elements).sum::<usize>()
            + self.norms_2.iter().map(Norm::elements).sum::<usize>()
    }
}

impl DurationFlow<'_> {
    /// Elements in this coupling block.
    fn elements(&self) -> usize {
        match self {
            Self::Affine {
                log_scale,
                translate,
            } => log_scale.len() + translate.len(),
            Self::Spline {
                conv_pre,
                conv_dds,
                conv_proj,
            } => conv_pre.elements() + conv_dds.elements() + conv_proj.elements(),
        }
    }
}

impl VitsWeights<'_> {
    /// Total parameters bound by this schema.
    ///
    /// Exists so a test can compare against the checkpoint's own inference
    /// subset. A tensor the schema forgets to read does not raise an error -
    /// it simply never appears - so counting is the only way to notice. This is
    /// how the unfused-weight-norm mistake in the decoder was caught.
    pub fn total_elements(&self) -> usize {
        let te = &self.text_encoder;
        let text = te.embed.len()
            + te.project.elements()
            + te.layers
                .iter()
                .map(|l| {
                    l.q.elements()
                        + l.k.elements()
                        + l.v.elements()
                        + l.out.elements()
                        + l.emb_rel_k.len()
                        + l.emb_rel_v.len()
                        + l.norm.elements()
                        + l.ffn_1.elements()
                        + l.ffn_2.elements()
                        + l.final_norm.elements()
                })
                .sum::<usize>();

        let dp = &self.duration_predictor;
        let dur = dp.conv_pre.elements()
            + dp.conv_dds.elements()
            + dp.conv_proj.elements()
            + dp.post_conv_pre.elements()
            + dp.post_conv_dds.elements()
            + dp.post_conv_proj.elements()
            + dp.flows.iter().map(DurationFlow::elements).sum::<usize>()
            + dp.post_flows
                .iter()
                .map(DurationFlow::elements)
                .sum::<usize>();

        let flow = self
            .flow
            .iter()
            .map(|b| {
                b.conv_pre.elements()
                    + b.conv_post.elements()
                    + b.wavenet
                        .iter()
                        .map(|w| w.in_layer.elements() + w.res_skip.elements())
                        .sum::<usize>()
            })
            .sum::<usize>();

        let d = &self.decoder;
        let dec = d.conv_pre.elements()
            + d.conv_post.elements()
            + d.upsampler.iter().map(MaybeWn::elements).sum::<usize>()
            + d.resblocks
                .iter()
                .map(|r| {
                    r.convs1.iter().map(MaybeWn::elements).sum::<usize>()
                        + r.convs2.iter().map(MaybeWn::elements).sum::<usize>()
                })
                .sum::<usize>();

        text + dur + flow + dec
    }
}
