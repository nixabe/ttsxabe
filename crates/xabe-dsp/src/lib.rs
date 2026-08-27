//! Scalar reference kernels.
//!
//! Every function here is written to be read against 🤗 Transformers'
//! `modeling_vits.py` line by line. They are the definition of correct that the
//! CUDA kernels will be tested against, so they optimise for being *evidently*
//! correct rather than for throughput: no blocking, no unrolling, no fused
//! anything. A clever reference is a reference you cannot trust.
//!
//! Two conventions run through the whole crate:
//!
//! - **Shapes are arguments, not types.** Every function takes flat `&[f32]`
//!   plus explicit dimensions. Encoding shapes in the type system would be
//!   pleasant right up to the point where the duration predictor reshapes a
//!   tensor four times in six lines.
//! - **Layout is named at the call site.** `[T, C]` for transformer-shaped
//!   data, `[C, T]` for convolution-shaped. The reference permutes between them
//!   constantly and silently; here the permutes are visible.

mod activation;
mod attention;
mod conv;
mod erf;
mod fft;
mod linear;
mod norm;
mod rope;
mod spline;
mod tensor;
mod weight_norm;

pub use activation::{elu, gelu_tanh, leaky_relu, mish, relu, snake, softmax_rows};
pub use attention::self_attention;
pub use conv::{conv1d, conv1d_strided, depthwise_conv1d, same_padding, transposed_conv1d};
pub use erf::{erf, gelu};
pub use fft::{Fft, dft, hann_periodic, istft, stft};
pub use linear::linear;
pub use norm::layer_norm;
pub use rope::{rms_norm, rope, rope_scaled, silu, silu_mul};
pub use spline::spline_inverse;
pub use tensor::{flip_channels, reflect_pad, transpose};
pub use weight_norm::{fuse_weight_norm, gated_activation};
