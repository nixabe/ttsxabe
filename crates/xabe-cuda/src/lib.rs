//! CUDA kernels for the VITS forward pass.
//!
//! Every kernel here has a scalar twin in `xabe-dsp` and is tested against it
//! per kernel, on the same inputs, before anything is assembled from them. That
//! ordering is the point: a GPU pipeline that is wrong somewhere is nearly
//! impossible to bisect after the fact, and trivially bisected before it exists.
//!
//! # No feature flag
//!
//! `cudarc` is used with `fallback-dynamic-loading`, so the driver is resolved
//! at runtime and this crate links on a machine with no CUDA toolkit and no
//! card. [`Gpu::open`] returns [`CudaError::NoDevice`] there. GPU-ness is a
//! runtime skip, not a compile-time `cfg` - which means the CUDA code is
//! type-checked by every ordinary `cargo build`, rather than rotting behind a
//! flag nobody enables.
//!
//! # What is not here
//!
//! No fusion, no shared-memory tiling, no tensor cores. The kernels are direct
//! implementations, one thread per output element. That is enough to establish
//! correctness and to measure, and `docs/OPTIMIZATION.md` is explicit that
//! optimisation arrives with a measurement rather than ahead of one.

mod device;
mod error;
mod kernels;

pub use cudarc::driver::CudaSlice;
pub use device::{Batch, GEMV_MAX_M, Gpu, Operand, Quant};
pub use error::CudaError;
pub use kernels::SOURCE;
