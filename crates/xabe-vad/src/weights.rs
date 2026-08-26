//! The geometry, and every tensor bound against it.
//!
//! Fifteen tensors. The whole network is a length-256 convolution standing in
//! for an STFT, four small convolutions, one LSTM cell and a 1×1 convolution -
//! about a hundred lines of real arithmetic, which is exactly why this is the
//! first model the engine absorbs. If the approach is going to fail it fails
//! here, for a tenth of what the ASR would cost.
//!
//! The geometry is *constants*, not parameters. This implementation computes
//! one specific network; a checkpoint with different channel counts would load,
//! run, and produce numbers that mean nothing. So every shape is asserted at
//! bind time and a checkpoint that disagrees is refused by name.

use crate::error::VadError;
use xabe_st::StFile;

/// Samples per frame. One probability comes out per frame.
pub const WINDOW: usize = 512;

/// Reflective padding applied to each end of a frame before the STFT.
pub const PAD: usize = 64;

/// Length of the STFT basis, and so of its convolution kernel.
pub const STFT_KERNEL: usize = 256;

/// Hop of the STFT convolution.
pub const STFT_HOP: usize = 128;

/// Rows of the STFT basis: 129 real followed by 129 imaginary.
pub const STFT_ROWS: usize = 258;

/// Frequency bins after the real and imaginary halves are combined.
pub const BINS: usize = STFT_ROWS / 2;

/// The four encoder convolutions: (in, out, kernel, stride, pad).
///
/// The strides are the reason the encoder ends with a single time position:
/// four STFT frames go to four, then two, then one, and the last layer holds
/// it there. Everything downstream is a single 128-vector.
pub const ENCODER: [(usize, usize, usize, usize, usize); 4] = [
    (BINS, 128, 3, 1, 1),
    (128, 64, 3, 2, 1),
    (64, 64, 3, 2, 1),
    (64, 128, 3, 1, 1),
];

/// Width of the LSTM's input and of its hidden state.
pub const HIDDEN: usize = 128;

/// The LSTM's four gates, stacked: input, forget, cell, output.
pub const GATES: usize = 4;

/// One encoder convolution's weights.
#[derive(Debug)]
pub struct Conv {
    /// `[out, in, k]`, row-major.
    pub weight: Vec<f32>,
    /// One per output channel.
    pub bias: Vec<f32>,
    /// Input channels.
    pub in_ch: usize,
    /// Output channels.
    pub out_ch: usize,
    /// Kernel width.
    pub k: usize,
    /// Stride.
    pub stride: usize,
    /// Padding applied to each end.
    pub pad: usize,
}

/// Every tensor the forward pass reads.
#[derive(Debug)]
pub struct VadWeights {
    /// `[258, 256]`: the real and imaginary halves of a windowed DFT basis.
    pub stft_basis: Vec<f32>,
    /// The four encoder convolutions, in order.
    pub encoder: Vec<Conv>,
    /// `[512, 128]` input-to-hidden.
    pub lstm_ih: Vec<f32>,
    /// `[512, 128]` hidden-to-hidden.
    pub lstm_hh: Vec<f32>,
    /// `[512]`.
    pub lstm_bias_ih: Vec<f32>,
    /// `[512]`.
    pub lstm_bias_hh: Vec<f32>,
    /// `[128]`: the final 1×1 convolution, which is a dot product.
    pub head_weight: Vec<f32>,
    /// The scalar the head adds before the sigmoid.
    pub head_bias: f32,
}

impl VadWeights {
    /// Binds every tensor, checking each against the geometry above.
    pub fn load(f: &StFile) -> Result<VadWeights, VadError> {
        // The converter records the ggml header. When it is present the file's
        // own claim about the frame length is checked against the constant this
        // implementation is written for, so a future Silero with a different
        // window is refused rather than silently mis-framed.
        if let Some(declared) = f.meta("n_window")
            && declared != WINDOW.to_string()
        {
            return Err(VadError::Geometry {
                what: "n_window",
                expected: WINDOW.to_string(),
                found: declared.to_string(),
            });
        }

        // The STFT basis is stored `[258, 1, 256]` - ggml's singleton input
        // channel survived the conversion. It is one row per basis function.
        let stft_basis = f.tensor_f32_shaped(
            "_model.stft.forward_basis_buffer",
            &[STFT_ROWS, 1, STFT_KERNEL],
        )?;

        let mut encoder = Vec::with_capacity(ENCODER.len());
        for (i, &(in_ch, out_ch, k, stride, pad)) in ENCODER.iter().enumerate() {
            let weight = f.tensor_f32_shaped(
                &format!("_model.encoder.{i}.reparam_conv.weight"),
                &[out_ch, in_ch, k],
            )?;
            let bias =
                f.tensor_f32_shaped(&format!("_model.encoder.{i}.reparam_conv.bias"), &[out_ch])?;
            encoder.push(Conv {
                weight,
                bias,
                in_ch,
                out_ch,
                k,
                stride,
                pad,
            });
        }

        let gate_rows = GATES * HIDDEN;
        let lstm_ih = f.tensor_f32_shaped("_model.decoder.rnn.weight_ih", &[gate_rows, HIDDEN])?;
        let lstm_hh = f.tensor_f32_shaped("_model.decoder.rnn.weight_hh", &[gate_rows, HIDDEN])?;
        let lstm_bias_ih = f.tensor_f32_shaped("_model.decoder.rnn.bias_ih", &[gate_rows])?;
        let lstm_bias_hh = f.tensor_f32_shaped("_model.decoder.rnn.bias_hh", &[gate_rows])?;

        let head_weight = f.tensor_f32_shaped("_model.decoder.decoder.2.weight", &[HIDDEN])?;
        // The bias is a rank-0 tensor: shape `[]`, one element. ggml wrote it
        // with n_dims = 0, and reversing an empty dimension list gives an empty
        // shape rather than `[1]`.
        let head_bias = f.tensor_f32_shaped("_model.decoder.decoder.2.bias", &[])?;
        let head_bias = *head_bias
            .first()
            .ok_or(VadError::MissingMetadata("a scalar head bias"))?;

        tracing::debug!(
            tensors = 15,
            parameters = stft_basis.len()
                + encoder
                    .iter()
                    .map(|c| c.weight.len() + c.bias.len())
                    .sum::<usize>()
                + lstm_ih.len()
                + lstm_hh.len()
                + lstm_bias_ih.len()
                + lstm_bias_hh.len()
                + head_weight.len()
                + 1,
            "bound the VAD checkpoint",
        );

        Ok(VadWeights {
            stft_basis,
            encoder,
            lstm_ih,
            lstm_hh,
            lstm_bias_ih,
            lstm_bias_hh,
            head_weight,
            head_bias,
        })
    }

    /// Total parameters bound by this schema.
    ///
    /// A tensor the schema forgets to read does not raise an error - it simply
    /// never appears - so counting is the only way to notice.
    pub fn total_elements(&self) -> usize {
        self.stft_basis.len()
            + self
                .encoder
                .iter()
                .map(|c| c.weight.len() + c.bias.len())
                .sum::<usize>()
            + self.lstm_ih.len()
            + self.lstm_hh.len()
            + self.lstm_bias_ih.len()
            + self.lstm_bias_hh.len()
            + self.head_weight.len()
            + 1
    }
}
