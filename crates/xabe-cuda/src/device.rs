//! Device handle, module loading, and one method per kernel.
//!
//! [`Gpu::open`] is fallible in the ordinary way and returns
//! [`CudaError::NoDevice`] when there is no card or no driver, so callers can
//! fall back to the CPU path rather than being unable to start. Nothing in this
//! crate is behind a feature flag.

use crate::error::CudaError;
use crate::kernels::SOURCE;
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

/// Threads per block for the flat element-wise kernels.
const BLOCK: u32 = 256;

/// Threads per block for the reduction kernels, which use shared memory sized
/// to the block.
const REDUCE_BLOCK: u32 = 256;

/// Output positions per block in the tiled convolution.
const CONV_BLOCK: u32 = 128;

/// Output channels each convolution thread accumulates. Must match `OC_TILE`
/// in the device source.
const CONV_OC_TILE: u32 = 8;

/// Time positions each convolution thread accumulates. Must match `T_REG`.
const CONV_T_REG: u32 = 4;

/// An open CUDA device with the kernels compiled and loaded.
pub struct Gpu {
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    funcs: HashMap<&'static str, CudaFunction>,
}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu")
            .field("kernels", &self.funcs.len())
            .finish()
    }
}

/// Every kernel in [`SOURCE`], looked up once at load.
/// Rows at or below which [`Gpu::gemm`] takes the exact scalar path.
///
/// The tensor-core kernel's M dimension is 16 rows wide and its block tile is
/// 128, so at one token it fills 1/128 of the tile: measured 0.02 TFLOP/s
/// against 23.8 at encoder width. Sixteen is where the tiled kernel first has a
/// whole instruction's worth of rows to work with.
///
/// The two paths do not have the same precision - the scalar one is exact f32
/// and the tiled one rounds its operands to f16 - so this constant is exported
/// rather than hidden. A test that compares `gemm` against a reference has to
/// know which side of it a shape falls on.
pub const GEMV_MAX_M: usize = 16;

const NAMES: &[&str] = &[
    "conv1d",
    "conv1d_short",
    "depthwise_conv1d",
    "transposed_conv1d",
    "linear",
    "gemm",
    "gemv",
    "layer_norm",
    "softmax_rows",
    "act_relu",
    "act_leaky_relu",
    "act_tanh",
    "act_gelu",
    "gated_activation",
    "add_inplace",
    "sub_inplace",
    "scale_inplace",
    "copy_range",
    "copy_into",
    "transpose",
    "flip_channels",
    "embed_scaled",
    "fuse_weight_norm",
    "attention_scores",
    "attention_context",
    "expand_prior",
];

impl Gpu {
    /// Opens device `ordinal` and compiles the kernels.
    pub fn open(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal).map_err(|e| CudaError::NoDevice(format!("{e:?}")))?;
        let stream = ctx.default_stream();

        // Compiled for the development target. NVRTC will happily target a
        // newer architecture, but pinning it keeps the generated code the same
        // on every machine that runs the differential tests.
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_75"),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)
            .map_err(|e| CudaError::Compile(format!("{e:?}")))?;
        let module = ctx.load_module(ptx).map_err(|source| CudaError::Driver {
            what: "loading the compiled module",
            source,
        })?;

        let mut funcs = HashMap::with_capacity(NAMES.len());
        for &name in NAMES {
            let f = module
                .load_function(name)
                .map_err(|_| CudaError::MissingKernel {
                    name: name.to_string(),
                })?;
            funcs.insert(name, f);
        }

        tracing::info!(ordinal, kernels = funcs.len(), "opened CUDA device");
        Ok(Self {
            stream,
            module,
            funcs,
        })
    }

    /// Opens the first device that works, or reports why none did.
    pub fn open_default() -> Result<Self, CudaError> {
        Self::open(0)
    }

    /// Copies a slice to the device.
    pub fn upload(&self, x: &[f32]) -> Result<CudaSlice<f32>, CudaError> {
        self.stream
            .clone_htod(x)
            .map_err(|source| CudaError::Driver {
                what: "uploading",
                source,
            })
    }

    /// Copies a slice of ids to the device.
    pub fn upload_i64(&self, x: &[i64]) -> Result<CudaSlice<i64>, CudaError> {
        self.stream
            .clone_htod(x)
            .map_err(|source| CudaError::Driver {
                what: "uploading ids",
                source,
            })
    }

    /// Copies a slice of indices to the device.
    pub fn upload_i32(&self, x: &[i32]) -> Result<CudaSlice<i32>, CudaError> {
        self.stream
            .clone_htod(x)
            .map_err(|source| CudaError::Driver {
                what: "uploading indices",
                source,
            })
    }

    /// Copies a buffer back to the host, synchronising.
    pub fn download(&self, x: &CudaSlice<f32>) -> Result<Vec<f32>, CudaError> {
        self.stream
            .clone_dtoh(x)
            .map_err(|source| CudaError::Driver {
                what: "downloading",
                source,
            })
    }

    /// Allocates a zeroed device buffer.
    pub fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, CudaError> {
        self.stream
            .alloc_zeros::<f32>(n)
            .map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })
    }

    /// Blocks until every queued kernel has finished.
    ///
    /// Only needed for timing: the copies back are already synchronising.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.stream
            .synchronize()
            .map_err(|source| CudaError::Driver {
                what: "synchronising",
                source,
            })
    }

    /// Launches a kernel over `n` flat elements with the given arguments
    /// already pushed.
    ///
    /// Looks up a kernel that was loaded at open.
    fn func(&self, name: &'static str) -> &CudaFunction {
        self.funcs
            .get(name)
            .unwrap_or_else(|| panic!("{name} was in NAMES but not loaded"))
    }

    /// A flat launch over `n` elements.
    fn flat(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// One block per row, with shared memory for the reduction.
    fn per_row(rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: REDUCE_BLOCK * 4,
        }
    }
}

/// Turns a launch result into ours.
///
/// `launch` returns the events it recorded for cross-stream ordering; this
/// crate uses one stream, so they are discarded.
fn launched<T>(
    what: &'static str,
    r: Result<T, cudarc::driver::DriverError>,
) -> Result<(), CudaError> {
    r.map(|_| ())
        .map_err(|source| CudaError::Driver { what, source })
}

/// The kernels.
///
/// Each is a thin wrapper: allocate the output, push the arguments in the order
/// the device code declares them, launch. The shapes are the same arguments the
/// `xabe-dsp` twin takes, in the same order, so the two can be read side by
/// side - which is the point, since the differential tests call both.
///
/// A **null bias pointer means no bias**. `conv_post` in the decoder is the one
/// convolution in the checkpoint without one; passing a null `u64` where the
/// kernel expects `const float*` is how that is expressed, since there is no
/// device pointer for an empty allocation.
impl Gpu {
    /// General 1-D convolution. Mirrors [`xabe_dsp::conv1d`].
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        in_ch: usize,
        t: usize,
        out_ch: usize,
        k: usize,
        pad_left: usize,
        pad_right: usize,
        dilation: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let span = dilation * (k - 1) + 1;
        let out_t = (t + pad_left + pad_right).saturating_sub(span) + 1;
        // Which specialisation: the four-deep time tile pays for itself only
        // when the sequence can fill it. Below that the short kernel wins,
        // because three quarters of every thread's arithmetic would otherwise
        // fall past the end of the sequence.
        let long_form = out_t >= (CONV_BLOCK * CONV_T_REG) as usize;
        let t_reg = if long_form { CONV_T_REG } else { 1 };
        let threads = if long_form {
            CONV_BLOCK
        } else {
            (out_t as u32).next_multiple_of(32).clamp(32, CONV_BLOCK)
        };
        let per_block = threads * t_reg;
        let mut out = self.zeros(out_ch * out_t)?;
        let (a, b_, c, d, e, g, h) = (
            in_ch as i32,
            t as i32,
            out_ch as i32,
            k as i32,
            pad_left as i32,
            dilation as i32,
            out_t as i32,
        );
        let null: u64 = 0;
        let f = self.func(if long_form { "conv1d" } else { "conv1d_short" });
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&g)
            .arg(&h);
        // The tiled kernel's launch shape: one block per (output-channel tile,
        // time tile), with the input window in dynamic shared memory.
        let span = dilation * (k - 1) + 1;
        let tile = per_block as usize + span - 1;
        let cfg = LaunchConfig {
            grid_dim: (
                (out_ch as u32).div_ceil(CONV_OC_TILE),
                (out_t as u32).div_ceil(per_block),
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (tile * 4) as u32,
        };
        launched("conv1d", unsafe { lb.launch(cfg) })?;
        Ok((out, out_t))
    }

    /// Depthwise convolution. Mirrors [`xabe_dsp::depthwise_conv1d`].
    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv1d(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        ch: usize,
        t: usize,
        k: usize,
        pad_left: usize,
        pad_right: usize,
        dilation: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let span = dilation * (k - 1) + 1;
        let out_t = (t + pad_left + pad_right).saturating_sub(span) + 1;
        let mut out = self.zeros(ch * out_t)?;
        let (a, b_, c, d, e, g) = (
            ch as i32,
            t as i32,
            k as i32,
            pad_left as i32,
            dilation as i32,
            out_t as i32,
        );
        let null: u64 = 0;
        let f = self.func("depthwise_conv1d");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&g);
        launched("depthwise_conv1d", unsafe {
            lb.launch(Self::flat(ch * out_t))
        })?;
        Ok((out, out_t))
    }

    /// Transposed convolution. Mirrors [`xabe_dsp::transposed_conv1d`].
    #[allow(clippy::too_many_arguments)]
    pub fn transposed_conv1d(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        in_ch: usize,
        t: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        padding: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let out_t = (t - 1) * stride + k - 2 * padding;
        let mut out = self.zeros(out_ch * out_t)?;
        let (a, b_, c, d, e, g, h) = (
            in_ch as i32,
            t as i32,
            out_ch as i32,
            k as i32,
            stride as i32,
            padding as i32,
            out_t as i32,
        );
        let null: u64 = 0;
        let f = self.func("transposed_conv1d");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&g)
            .arg(&h);
        launched("transposed_conv1d", unsafe {
            lb.launch(Self::flat(out_ch * out_t))
        })?;
        Ok((out, out_t))
    }

    /// Dense projection. Mirrors [`xabe_dsp::linear`].
    pub fn linear(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        rows: usize,
        in_c: usize,
        out_c: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(rows * out_c)?;
        let (a, b_, c) = (rows as i32, in_c as i32, out_c as i32);
        let null: u64 = 0;
        let f = self.func("linear");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out).arg(&a).arg(&b_).arg(&c);
        launched("linear", unsafe { lb.launch(Self::flat(rows * out_c)) })?;
        Ok(out)
    }

    /// `out[m][n] = sum_k a[m][k] * w[n][k]`, on the tensor cores.
    ///
    /// The same contract as [`Gpu::linear`] and a different implementation:
    /// this one stages both operands as f16 and accumulates in f32, which is
    /// worth 17.9 to 99 TFLOP/s on this card and costs one rounding of each
    /// operand. `linear` stays for the places that want exact f32 and are small
    /// enough not to care - the choice is per call site, which is why both
    /// exist rather than one replacing the other.
    ///
    /// `k` must be a multiple of 8, which the instruction fixes. Whisper's
    /// contractions are 1280 and 5120, so this is a check rather than a
    /// limitation; padding a ragged contraction silently would be worse.
    pub fn gemm(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        if !k.is_multiple_of(8) {
            return Err(CudaError::RaggedContraction { k });
        }
        if m <= GEMV_MAX_M {
            return self.gemv(a, w, bias, m, k, n);
        }
        let mut out = self.zeros(m * n)?;
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let null: u64 = 0;
        let f = self.func("gemm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out).arg(&mi).arg(&ki).arg(&ni);

        // 128 rows of `a` and 128 of `w` per block, across 8 warps. The tile is
        // chosen by global traffic rather than by shared capacity - see the
        // derivation in kernels.rs, which is also why these three numbers must
        // move together with GEMM_MT, GEMM_NT and GEMM_WARPS.
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every (m, n) exactly once, `out` is m*n
        // elements, and every global read and write inside the kernel is bounds
        // checked against m, k and n.
        launched("gemm", unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// The same product for a handful of rows, one warp per output channel.
    ///
    /// Reached through [`Gpu::gemm`] rather than called directly, so a caller
    /// never has to know which shape it has. It is **exact f32** - no tensor
    /// cores, so no operand rounding - which means precision here is a function
    /// of `m`. See [`GEMV_MAX_M`].
    fn gemv(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(m * n)?;
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let null: u64 = 0;
        let f = self.func("gemv");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(w);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out).arg(&mi).arg(&ki).arg(&ni);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(8) as u32, m as u32, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one warp per (row, col), both bounds checked in the kernel;
        // `out` is m*n elements and every lane writes at most one of them.
        launched("gemv", unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// Layer normalisation over each row. Mirrors [`xabe_dsp::layer_norm`].
    pub fn layer_norm(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(rows * cols)?;
        let c = cols as i32;
        let f = self.func("layer_norm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(weight)
            .arg(bias)
            .arg(&mut out)
            .arg(&c)
            .arg(&eps);
        launched("layer_norm", unsafe { lb.launch(Self::per_row(rows)) })?;
        Ok(out)
    }

    /// Softmax over each row, in place. Mirrors [`xabe_dsp::softmax_rows`].
    pub fn softmax_rows(
        &self,
        x: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        let c = cols as i32;
        let f = self.func("softmax_rows");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&c);
        launched("softmax_rows", unsafe { lb.launch(Self::per_row(rows)) })
    }

    /// Applies a pointwise activation in place.
    fn activate(
        &self,
        name: &'static str,
        x: &mut CudaSlice<f32>,
        n: usize,
        slope: Option<f32>,
    ) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&len);
        let s = slope.unwrap_or(0.0);
        if slope.is_some() {
            lb.arg(&s);
        }
        launched(name, unsafe { lb.launch(Self::flat(n)) })
    }

    /// ReLU, in place.
    pub fn relu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_relu", x, n, None)
    }

    /// Leaky ReLU, in place. Mirrors [`xabe_dsp::leaky_relu`].
    pub fn leaky_relu(
        &self,
        x: &mut CudaSlice<f32>,
        n: usize,
        slope: f32,
    ) -> Result<(), CudaError> {
        self.activate("act_leaky_relu", x, n, Some(slope))
    }

    /// Hyperbolic tangent, in place.
    pub fn tanh(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_tanh", x, n, None)
    }

    /// Exact GELU, in place. Mirrors [`xabe_dsp::gelu`].
    ///
    /// The device has an IEEE-accurate `erff`, so this needs none of the
    /// rational approximation the CPU twin carries.
    pub fn gelu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_gelu", x, n, None)
    }

    /// WaveNet's gated activation. Mirrors [`xabe_dsp::gated_activation`].
    pub fn gated_activation(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(ch * t)?;
        let (a, b_) = (ch as i32, t as i32);
        let f = self.func("gated_activation");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_);
        launched("gated_activation", unsafe { lb.launch(Self::flat(ch * t)) })?;
        Ok(out)
    }

    /// Element-wise `a += b`.
    pub fn add_inplace(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func("add_inplace");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b).arg(&len);
        launched("add_inplace", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Element-wise `a -= b`.
    pub fn sub_inplace(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func("sub_inplace");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b).arg(&len);
        launched("sub_inplace", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Element-wise `a *= s`.
    pub fn scale_inplace(&self, a: &mut CudaSlice<f32>, n: usize, s: f32) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func("scale_inplace");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(&len).arg(&s);
        launched("scale_inplace", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Copies `n` values starting at `offset` into a fresh buffer.
    pub fn copy_range(
        &self,
        x: &CudaSlice<f32>,
        offset: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(n)?;
        let (a, b_) = (offset as i32, n as i32);
        let f = self.func("copy_range");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_);
        launched("copy_range", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(out)
    }

    /// Writes `src` into `dst` starting at `offset`.
    pub fn copy_into(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        offset: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        let (a, b_) = (offset as i32, n as i32);
        let f = self.func("copy_into");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(dst).arg(src).arg(&a).arg(&b_);
        launched("copy_into", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Transposes a row-major `[rows, cols]`. Mirrors [`xabe_dsp::transpose`].
    pub fn transpose(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(rows * cols)?;
        let (a, b_) = (rows as i32, cols as i32);
        let f = self.func("transpose");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_);
        launched("transpose", unsafe { lb.launch(Self::flat(rows * cols)) })?;
        Ok(out)
    }

    /// Reverses the channel axis. Mirrors [`xabe_dsp::flip_channels`].
    pub fn flip_channels(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(ch * t)?;
        let (a, b_) = (ch as i32, t as i32);
        let f = self.func("flip_channels");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_);
        launched("flip_channels", unsafe { lb.launch(Self::flat(ch * t)) })?;
        Ok(out)
    }

    /// Embedding lookup with the `sqrt(hidden_size)` scaling folded in.
    pub fn embed_scaled(
        &self,
        table: &CudaSlice<f32>,
        ids: &CudaSlice<i64>,
        t: usize,
        ch: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(t * ch)?;
        let (a, b_) = (t as i32, ch as i32);
        let f = self.func("embed_scaled");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(table)
            .arg(ids)
            .arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&scale);
        launched("embed_scaled", unsafe { lb.launch(Self::flat(t * ch)) })?;
        Ok(out)
    }

    /// Fuses weight normalisation. Mirrors [`xabe_dsp::fuse_weight_norm`].
    pub fn fuse_weight_norm(
        &self,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        out_ch: usize,
        in_ch: usize,
        k: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let per = in_ch * k;
        let mut out = self.zeros(out_ch * per)?;
        let p = per as i32;
        let f = self.func("fuse_weight_norm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(v).arg(g).arg(&mut out).arg(&p);
        launched("fuse_weight_norm", unsafe {
            lb.launch(Self::per_row(out_ch))
        })?;
        Ok(out)
    }

    /// Attention logits with the windowed relative bias, `[heads, t, t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_scores(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        emb_rel_k: &CudaSlice<f32>,
        t: usize,
        embed: usize,
        heads: usize,
        window: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let head_dim = embed / heads;
        let mut out = self.zeros(heads * t * t)?;
        let (a, b_, c, d, e) = (
            t as i32,
            embed as i32,
            heads as i32,
            head_dim as i32,
            window as i32,
        );
        let scaling = (head_dim as f32).powf(-0.5);
        let f = self.func("attention_scores");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(q)
            .arg(k)
            .arg(emb_rel_k)
            .arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&scaling);
        launched("attention_scores", unsafe {
            lb.launch(Self::flat(heads * t * t))
        })?;
        Ok(out)
    }

    /// Attention output with the windowed relative value term, `[t, embed]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_context(
        &self,
        probs: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        emb_rel_v: &CudaSlice<f32>,
        t: usize,
        embed: usize,
        heads: usize,
        window: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let head_dim = embed / heads;
        let mut out = self.zeros(t * embed)?;
        let (a, b_, c, d, e) = (
            t as i32,
            embed as i32,
            heads as i32,
            head_dim as i32,
            window as i32,
        );
        let f = self.func("attention_context");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(probs)
            .arg(v)
            .arg(emb_rel_v)
            .arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e);
        launched("attention_context", unsafe {
            lb.launch(Self::flat(t * embed))
        })?;
        Ok(out)
    }

    /// Length regulation and prior sampling. Mirrors [`xabe_tts`'s
    /// `expand_prior`], with the alignment already computed on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_prior(
        &self,
        m_p: &CudaSlice<f32>,
        logs_p: &CudaSlice<f32>,
        alignment: &CudaSlice<i32>,
        noise: &CudaSlice<f32>,
        ch: usize,
        frames: usize,
        noise_scale: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(ch * frames)?;
        let (a, b_) = (ch as i32, frames as i32);
        let f = self.func("expand_prior");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(m_p)
            .arg(logs_p)
            .arg(alignment)
            .arg(noise)
            .arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&noise_scale);
        launched("expand_prior", unsafe {
            lb.launch(Self::flat(ch * frames))
        })?;
        Ok(out)
    }
}
