//! Binding both checkpoints, and the three things that are done once at load.
//!
//! Nothing here does arithmetic on activations. What it does do is fold every
//! constant the forward pass would otherwise recompute per utterance:
//!
//! - **Batch norm is folded into the convolution before it.** At inference a
//!   `BatchNorm1d` is an affine map with constant coefficients, so
//!   `BN(conv(x))` is a convolution with scaled weights and a shifted bias.
//!   Tacotron2 has eight of them and none survive to the device.
//! - **Weight norm is fused.** The checkpoint stores a direction and a
//!   magnitude; every WaveGlow convolution wants their product and it does not
//!   change between utterances.
//! - **The invertible 1x1 convolutions are inverted.** WaveGlow runs backwards
//!   at inference, and the reference inverts them lazily on the first call.
//!   Doing it at load means a singular matrix is a startup error naming the
//!   flow rather than a first turn of noise.

use crate::{Config, TacoError};
use xabe_cuda::{CudaSlice, Gpu, Operand};
use xabe_st::StFile;

/// `BatchNorm1d`'s default epsilon. Not stored in the checkpoint.
const BN_EPS: f32 = 1e-5;

/// How a convolution's weight is held on the device.
///
/// The tiled matmul rounds both operands to f16 inside the kernel regardless,
/// so a weight destined only for that path is stored rounded and the rounding
/// stops being per-call work and per-call bandwidth. It is not an accuracy
/// choice: the numbers reaching the tensor cores are the same either way.
/// `conv1d` reads f32, so anything on that path stays f32 - as does anything
/// whose contraction is odd, which the matmul refuses in half precision.
pub(crate) enum Weight {
    Full(CudaSlice<f32>),
    Half(CudaSlice<u16>),
}

impl Weight {
    pub(crate) fn operand(&self) -> Operand<'_> {
        match self {
            Weight::Full(v) => Operand::F32(v),
            Weight::Half(v) => Operand::F16(v),
        }
    }

    /// The f32 form, for the callers that convolve rather than multiply.
    pub(crate) fn full(&self) -> &CudaSlice<f32> {
        match self {
            Weight::Full(v) => v,
            Weight::Half(_) => unreachable!("this weight was bound for the matmul path"),
        }
    }
}

/// A convolution, bias always present after folding.
pub(crate) struct Conv {
    pub(crate) w: Weight,
    pub(crate) bias: CudaSlice<f32>,
    pub(crate) in_ch: usize,
    pub(crate) out_ch: usize,
    pub(crate) k: usize,
}

/// Uploads a weight as f16 when the matmul will accept it, f32 otherwise.
///
/// The contraction is `in_ch * k`; the tiled kernel stages two elements at a
/// time and refuses an odd one, which is why WaveGlow's three-channel start
/// projection stays full width.
fn upload_weight(gpu: &Gpu, w: &[f32], in_ch: usize, k: usize) -> Result<Weight, TacoError> {
    if (in_ch * k).is_multiple_of(2) {
        let f32s = gpu.upload(w)?;
        Ok(Weight::Half(gpu.to_f16(&f32s, w.len())?))
    } else {
        Ok(Weight::Full(gpu.upload(w)?))
    }
}

/// A dense projection. Tacotron2's `LinearNorm` is bias-free almost everywhere.
pub(crate) struct Dense {
    pub(crate) w: CudaSlice<f32>,
    pub(crate) bias: Option<CudaSlice<f32>>,
    pub(crate) in_c: usize,
    pub(crate) out_c: usize,
}

impl Dense {
    fn bind(
        f: &StFile,
        gpu: &Gpu,
        name: &str,
        out: usize,
        inp: usize,
        bias: Option<&str>,
    ) -> Result<Self, TacoError> {
        Ok(Self {
            w: gpu.upload(f.tensor_shaped(name, &[out, inp])?)?,
            bias: match bias {
                Some(b) => Some(gpu.upload(f.tensor_shaped(b, &[out])?)?),
                None => None,
            },
            in_c: inp,
            out_c: out,
        })
    }
}

/// One LSTM's four parameter tensors.
///
/// The two biases are kept apart rather than summed because they are summed
/// on different sides of the recurrence: `bias_ih` folds into the projection of
/// the whole input sequence, `bias_hh` into the per-step one.
///
/// The weights are f32 or f16 by `half`, and the two LSTMs of the encoder and
/// the two of the decoder choose differently. The decoder's are streamed in
/// full for every frame - 71 MB a frame at f32, two thirds of a frame's time
/// on the card, at the card's bandwidth - and the frame is one row, so the
/// mat-vec is the memory system and nothing else. Halving the width halves
/// the frame's largest cost.
///
/// It is also lossless, and that is a property of the checkpoint rather than
/// of f16: NVIDIA trained this model under AMP, and every LSTM weight in the
/// published file is exactly representable in f16 (checked over all 19.4 M
/// of them). What the half-width path changes is accumulation order, and the
/// test in `pipeline` measures that on the real checkpoint at 6e-6 on mels
/// of span 10. The encoder's two are held at f32 anyway: they run once a
/// line, and the encoder is held to its reference at 1e-5, which is a bound
/// worth not spending on nothing.
pub(crate) struct Lstm {
    pub(crate) w_ih: Weight,
    pub(crate) w_hh: Weight,
    pub(crate) b_ih: CudaSlice<f32>,
    pub(crate) b_hh: CudaSlice<f32>,
    pub(crate) hidden: usize,
}

impl Lstm {
    fn bind(
        f: &StFile,
        gpu: &Gpu,
        prefix: &str,
        suffix: &str,
        input: usize,
        hidden: usize,
        half: bool,
    ) -> Result<Self, TacoError> {
        let g = 4 * hidden;
        let up = |name: &str, shape: &[usize]| -> Result<Weight, TacoError> {
            let t = f.tensor_shaped(name, shape)?;
            if half {
                upload_weight(gpu, t, shape[0], shape[1])
            } else {
                Ok(Weight::Full(gpu.upload(t)?))
            }
        };
        Ok(Self {
            w_ih: up(&format!("{prefix}weight_ih{suffix}"), &[g, input])?,
            w_hh: up(&format!("{prefix}weight_hh{suffix}"), &[g, hidden])?,
            b_ih: gpu.upload(f.tensor_shaped(&format!("{prefix}bias_ih{suffix}"), &[g])?)?,
            b_hh: gpu.upload(f.tensor_shaped(&format!("{prefix}bias_hh{suffix}"), &[g])?)?,
            hidden,
        })
    }
}

/// Folds a `BatchNorm1d` into the convolution feeding it.
///
/// `y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta` is a convolution
/// whose weights are scaled per output channel and whose bias absorbs the rest.
fn fold_bn(
    f: &StFile,
    gpu: &Gpu,
    conv: &str,
    bn: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<Conv, TacoError> {
    let w = f.tensor_shaped(&format!("{conv}.weight"), &[out_ch, in_ch, k])?;
    let b = f.tensor_shaped(&format!("{conv}.bias"), &[out_ch])?;
    let gamma = f.tensor_shaped(&format!("{bn}.weight"), &[out_ch])?;
    let beta = f.tensor_shaped(&format!("{bn}.bias"), &[out_ch])?;
    let mean = f.tensor_shaped(&format!("{bn}.running_mean"), &[out_ch])?;
    let var = f.tensor_shaped(&format!("{bn}.running_var"), &[out_ch])?;

    let per = in_ch * k;
    let mut fw = vec![0.0f32; out_ch * per];
    let mut fb = vec![0.0f32; out_ch];
    for o in 0..out_ch {
        let scale = gamma[o] / (var[o] + BN_EPS).sqrt();
        for j in 0..per {
            fw[o * per + j] = w[o * per + j] * scale;
        }
        fb[o] = (b[o] - mean[o]) * scale + beta[o];
    }
    Ok(Conv {
        w: Weight::Full(gpu.upload(&fw)?),
        bias: gpu.upload(&fb)?,
        in_ch,
        out_ch,
        k,
    })
}

/// Binds a weight-normalised convolution, fusing the two halves.
fn bind_wn(
    f: &StFile,
    gpu: &Gpu,
    prefix: &str,
    out: usize,
    inp: usize,
    k: usize,
) -> Result<Conv, TacoError> {
    let v = f.tensor_shaped(&format!("{prefix}.weight_v"), &[out, inp, k])?;
    let g = f.tensor_shaped(&format!("{prefix}.weight_g"), &[out, 1, 1])?;
    let fused = xabe_dsp::fuse_weight_norm(v, g, out, inp, k);
    Ok(Conv {
        w: upload_weight(gpu, &fused, inp, k)?,
        bias: gpu.upload(f.tensor_shaped(&format!("{prefix}.bias"), &[out])?)?,
        in_ch: inp,
        out_ch: out,
        k,
    })
}

/// Binds one row-slice of a weight-normalised convolution.
///
/// The checkpoint stores the conditioning projection as one `[2 * ch * layers,
/// cond]` matrix and each residual/skip layer as one `[2 * ch, ch]`, because
/// the reference slices their *outputs*. In `[steps, channels]` an output slice
/// is a stride rather than a range, so the split is done here instead - on the
/// rows, which are output channels and are contiguous.
///
/// Weight norm is fused on the host for these: the device kernel normalises a
/// whole tensor, and what is wanted is a normalised slice of one.
// The slice is named by index and count rather than by a range type, which is
// what makes the two call sites read as "piece 1 of 2" and "piece i of layers".
#[allow(clippy::too_many_arguments)]
fn bind_wn_rows(
    f: &StFile,
    gpu: &Gpu,
    prefix: &str,
    out_total: usize,
    inp: usize,
    k: usize,
    piece: usize,
    pieces: usize,
) -> Result<Conv, TacoError> {
    let v = f.tensor_shaped(&format!("{prefix}.weight_v"), &[out_total, inp, k])?;
    let g = f.tensor_shaped(&format!("{prefix}.weight_g"), &[out_total, 1, 1])?;
    let b = f.tensor_shaped(&format!("{prefix}.bias"), &[out_total])?;
    let fused = xabe_dsp::fuse_weight_norm(v, g, out_total, inp, k);

    let out = out_total / pieces;
    let per = inp * k;
    let lo = piece * out;
    Ok(Conv {
        w: upload_weight(gpu, &fused[lo * per..(lo + out) * per], inp, k)?,
        bias: gpu.upload(&b[lo..lo + out])?,
        in_ch: inp,
        out_ch: out,
        k,
    })
}

/// A plain convolution with a bias and no normalisation.
fn bind_plain(
    f: &StFile,
    gpu: &Gpu,
    prefix: &str,
    out: usize,
    inp: usize,
    k: usize,
    matmul: bool,
) -> Result<Conv, TacoError> {
    let raw = f.tensor_shaped(&format!("{prefix}.weight"), &[out, inp, k])?;
    Ok(Conv {
        w: match matmul {
            true => upload_weight(gpu, raw, inp, k)?,
            false => Weight::Full(gpu.upload(raw)?),
        },
        bias: gpu.upload(f.tensor_shaped(&format!("{prefix}.bias"), &[out])?)?,
        in_ch: inp,
        out_ch: out,
        k,
    })
}

/// Gauss-Jordan inverse of a small dense matrix, with partial pivoting.
///
/// `c` is 8, 6 or 4 here, so this is a load-time formality rather than a
/// numerical exercise - but a singular pivot is reported instead of producing
/// infinities that would only show up as noise in the output.
fn invert(m: &[f32], c: usize, flow: usize) -> Result<Vec<f32>, TacoError> {
    let mut a: Vec<f64> = m.iter().map(|&v| v as f64).collect();
    let mut inv = vec![0.0f64; c * c];
    for i in 0..c {
        inv[i * c + i] = 1.0;
    }
    for col in 0..c {
        let (mut best, mut mag) = (col, a[col * c + col].abs());
        for r in col + 1..c {
            if a[r * c + col].abs() > mag {
                best = r;
                mag = a[r * c + col].abs();
            }
        }
        if mag < 1e-12 {
            return Err(TacoError::Singular { flow });
        }
        if best != col {
            for j in 0..c {
                a.swap(col * c + j, best * c + j);
                inv.swap(col * c + j, best * c + j);
            }
        }
        let p = a[col * c + col];
        for j in 0..c {
            a[col * c + j] /= p;
            inv[col * c + j] /= p;
        }
        for r in 0..c {
            if r == col {
                continue;
            }
            let factor = a[r * c + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..c {
                a[r * c + j] -= factor * a[col * c + j];
                inv[r * c + j] -= factor * inv[col * c + j];
            }
        }
    }
    Ok(inv.into_iter().map(|v| v as f32).collect())
}

/// Tacotron2's weights, batch norm already gone.
pub(crate) struct Taco2 {
    /// Kept on the host: 71 rows is a gather, not a matmul, and doing it here
    /// costs one upload of `[tokens, 512]` instead of a kernel.
    pub(crate) embedding: Vec<f32>,
    pub(crate) enc_convs: Vec<Conv>,
    pub(crate) enc_fwd: Lstm,
    pub(crate) enc_rev: Lstm,
    pub(crate) prenet: Vec<Dense>,
    pub(crate) attention_rnn: Lstm,
    pub(crate) query: Dense,
    pub(crate) memory: Dense,
    pub(crate) v: Dense,
    pub(crate) location_conv: Conv,
    pub(crate) location_dense: Dense,
    pub(crate) decoder_rnn: Lstm,
    /// The frame projection and the stop gate stacked into one
    /// `[n_mel + 1, dhac]` with their biases, so a frame's mel and its stop
    /// logit are one mat-vec. Both have a bias in every Tacotron2 checkpoint,
    /// so the stack adds nothing that was not there.
    pub(crate) proj_gate: Dense,
    pub(crate) postnet: Vec<Conv>,
}

impl Taco2 {
    pub(crate) fn open(f: &StFile, gpu: &Gpu, c: &Config) -> Result<Self, TacoError> {
        Self::open_with(f, gpu, c, true)
    }

    /// As [`Taco2::open`], with the decoder's LSTM weights at f16 or f32.
    /// The engine always takes f16 - see [`Lstm`] - and the f32 form exists
    /// so the test in `pipeline` can measure the difference on the real
    /// checkpoint rather than assert it.
    pub(crate) fn open_with(
        f: &StFile,
        gpu: &Gpu,
        c: &Config,
        decoder_half: bool,
    ) -> Result<Self, TacoError> {
        let (e, mel) = (c.encoder_dim, c.n_mel);
        let n = c.symbols.len();

        let mut enc_convs = Vec::with_capacity(c.encoder_convs);
        for i in 0..c.encoder_convs {
            enc_convs.push(fold_bn(
                f,
                gpu,
                &format!("encoder.convolutions.{i}.0.conv"),
                &format!("encoder.convolutions.{i}.1"),
                e,
                e,
                c.encoder_kernel,
            )?);
        }

        let mut prenet = Vec::with_capacity(2);
        for (i, inp) in [mel, c.prenet_dim].into_iter().enumerate() {
            prenet.push(Dense::bind(
                f,
                gpu,
                &format!("decoder.prenet.layers.{i}.linear_layer.weight"),
                c.prenet_dim,
                inp,
                None,
            )?);
        }

        let mut postnet = Vec::with_capacity(c.postnet_convs);
        for i in 0..c.postnet_convs {
            let inp = if i == 0 { mel } else { c.postnet_dim };
            let out = if i == c.postnet_convs - 1 {
                mel
            } else {
                c.postnet_dim
            };
            postnet.push(fold_bn(
                f,
                gpu,
                &format!("postnet.convolutions.{i}.0.conv"),
                &format!("postnet.convolutions.{i}.1"),
                out,
                inp,
                c.postnet_kernel,
            )?);
        }

        let dhac = c.decoder_rnn_dim + e;
        Ok(Self {
            embedding: f.tensor_shaped("embedding.weight", &[n, e])?.to_vec(),
            enc_convs,
            enc_fwd: Lstm::bind(f, gpu, "encoder.lstm.", "_l0", e, c.lstm_hidden, false)?,
            enc_rev: Lstm::bind(
                f,
                gpu,
                "encoder.lstm.",
                "_l0_reverse",
                e,
                c.lstm_hidden,
                false,
            )?,
            prenet,
            attention_rnn: Lstm::bind(
                f,
                gpu,
                "decoder.attention_rnn.",
                "",
                c.prenet_dim + e,
                c.attention_rnn_dim,
                decoder_half,
            )?,
            query: Dense::bind(
                f,
                gpu,
                "decoder.attention_layer.query_layer.linear_layer.weight",
                c.attention_dim,
                c.attention_rnn_dim,
                None,
            )?,
            memory: Dense::bind(
                f,
                gpu,
                "decoder.attention_layer.memory_layer.linear_layer.weight",
                c.attention_dim,
                e,
                None,
            )?,
            v: Dense::bind(
                f,
                gpu,
                "decoder.attention_layer.v.linear_layer.weight",
                1,
                c.attention_dim,
                None,
            )?,
            // Two input channels: the previous attention weights and their
            // running sum. Bias-free in the reference.
            location_conv: Conv {
                w: Weight::Full(gpu.upload(f.tensor_shaped(
                    "decoder.attention_layer.location_layer.location_conv.conv.weight",
                    &[c.location_filters, 2, c.location_kernel],
                )?)?),
                bias: gpu.zeros(c.location_filters)?,
                in_ch: 2,
                out_ch: c.location_filters,
                k: c.location_kernel,
            },
            location_dense: Dense::bind(
                f,
                gpu,
                "decoder.attention_layer.location_layer.location_dense.linear_layer.weight",
                c.attention_dim,
                c.location_filters,
                None,
            )?,
            decoder_rnn: Lstm::bind(
                f,
                gpu,
                "decoder.decoder_rnn.",
                "",
                c.attention_rnn_dim + e,
                c.decoder_rnn_dim,
                decoder_half,
            )?,
            proj_gate: {
                let mut w = f
                    .tensor_shaped(
                        "decoder.linear_projection.linear_layer.weight",
                        &[mel, dhac],
                    )?
                    .to_vec();
                w.extend_from_slice(
                    f.tensor_shaped("decoder.gate_layer.linear_layer.weight", &[1, dhac])?,
                );
                let mut b = f
                    .tensor_shaped("decoder.linear_projection.linear_layer.bias", &[mel])?
                    .to_vec();
                b.extend_from_slice(f.tensor_shaped("decoder.gate_layer.linear_layer.bias", &[1])?);
                Dense {
                    w: gpu.upload(&w)?,
                    bias: Some(gpu.upload(&b)?),
                    in_c: dhac,
                    out_c: mel + 1,
                }
            },
            postnet,
        })
    }
}

/// One coupling network, with every output slice already its own matrix.
pub(crate) struct Wn {
    pub(crate) start: Conv,
    pub(crate) end: Conv,
    /// The conditioning projection for every layer at once, `[2 * ch * layers,
    /// cond]` - which is how the checkpoint stores it.
    ///
    /// Kept whole rather than split per layer because the conditioning does not
    /// depend on the audio being transformed, so all `layers` of it are one
    /// matmul. On this checkpoint's shape that measured 2.94 ms against 2.26 -
    /// 11.8 TFLOP/s against 15.3 - and a layer's share is then a column range
    /// of every row, which [`Gpu::add_strided`] adds without materialising.
    pub(crate) cond: Conv,
    pub(crate) in_layers: Vec<Conv>,
    /// The residual half, absent on the last layer, which feeds only the skip.
    pub(crate) res: Vec<Option<Conv>>,
    /// The skip half, `[ch, ch]`.
    pub(crate) skip: Vec<Conv>,
}

/// One flow: an inverted 1x1 mixing convolution and a coupling network.
pub(crate) struct Flow {
    /// `[channels, channels, 1]`, already inverted.
    pub(crate) inv: CudaSlice<f32>,
    pub(crate) channels: usize,
    pub(crate) wn: Wn,
}

/// WaveGlow's weights.
pub(crate) struct Glow {
    pub(crate) upsample: Conv,
    pub(crate) flows: Vec<Flow>,
}

impl Glow {
    pub(crate) fn open(f: &StFile, gpu: &Gpu, c: &Config) -> Result<Self, TacoError> {
        let upsample = bind_plain(f, gpu, "upsample", c.n_mel, c.n_mel, c.filter_length, false)?;
        let ch = c.wn_channels;
        let cond_in = c.n_mel * c.n_group;

        let mut flows = Vec::with_capacity(c.n_flows);
        for (k, &channels) in c.flow_channels().iter().enumerate() {
            let half = channels / 2;
            let raw = f.tensor_shaped(
                &format!("convinv.{k}.conv.weight"),
                &[channels, channels, 1],
            )?;
            let inv = gpu.upload(&invert(raw, channels, k)?)?;

            let mut in_layers = Vec::with_capacity(c.wn_layers);
            let mut res = Vec::with_capacity(c.wn_layers);
            let mut skip = Vec::with_capacity(c.wn_layers);
            // One projection for all the layers: see `Wn::cond`.
            let cond = bind_wn_rows(
                f,
                gpu,
                &format!("WN.{k}.cond_layer"),
                2 * ch * c.wn_layers,
                cond_in,
                1,
                0,
                1,
            )?;
            for i in 0..c.wn_layers {
                in_layers.push(bind_wn(
                    f,
                    gpu,
                    &format!("WN.{k}.in_layers.{i}"),
                    2 * ch,
                    ch,
                    c.wn_kernel,
                )?);
                // The last layer feeds only the skip path, so it has no
                // residual half. Getting this backwards runs, and mixes the
                // residual into the output.
                let name = format!("WN.{k}.res_skip_layers.{i}");
                if i == c.wn_layers - 1 {
                    res.push(None);
                    skip.push(bind_wn_rows(f, gpu, &name, ch, ch, 1, 0, 1)?);
                } else {
                    res.push(Some(bind_wn_rows(f, gpu, &name, 2 * ch, ch, 1, 0, 2)?));
                    skip.push(bind_wn_rows(f, gpu, &name, 2 * ch, ch, 1, 1, 2)?);
                }
            }

            flows.push(Flow {
                inv,
                channels,
                wn: Wn {
                    start: bind_wn(f, gpu, &format!("WN.{k}.start"), ch, half, 1)?,
                    // The only convolution in the coupling network without
                    // weight norm; the reference leaves it plain.
                    end: bind_plain(f, gpu, &format!("WN.{k}.end"), 2 * half, ch, 1, true)?,
                    cond,
                    in_layers,
                    res,
                    skip,
                },
            });
        }
        Ok(Self { upsample, flows })
    }
}
