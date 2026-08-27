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

/// One side of a matmul, in whichever precision it is stored.
///
/// The tiled kernel rounds *both* operands to f16 on the way into shared
/// memory, on every trip. So storing either side as F32 buys no accuracy at
/// all and costs twice the global traffic - and global traffic is what this
/// kernel is limited by. [`Operand::F16`] is the same arithmetic at half the
/// bytes, and it halves what a large model occupies on the card as well.
///
/// The scalar path *is* different: it accumulates an F32 operand exactly and
/// rounds an f16 one. That is a real precision decision on the decode shapes,
/// and it is the caller's to make - which is why this is a type rather than
/// something [`Gpu::upload`] decides.
#[derive(Debug, Clone, Copy)]
pub enum Operand<'a> {
    /// Full precision, as it comes out of the checkpoint.
    F32(&'a CudaSlice<f32>),
    /// Rounded once. Two halves to a 32-bit word, so the contraction must be
    /// even - every one in a transformer is.
    F16(&'a CudaSlice<u16>),
}

impl Operand<'_> {
    /// Whether this side is packed, as the kernel's flag.
    fn half(self) -> i32 {
        i32::from(matches!(self, Operand::F16(_)))
    }
}

/// How a batched matmul steps between its products.
///
/// Three separate strides because attention needs two different shapes from
/// the same call: the score matrices step `tq*head_dim` through the queries
/// and `tk*head_dim` through the keys, and the contexts step `tq*tk` through
/// the probabilities. One stride would fit neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Batch {
    /// How many independent products.
    pub count: usize,
    /// Elements between consecutive left operands.
    pub a: usize,
    /// Elements between consecutive right operands.
    ///
    /// Counted in elements of the logical matrix in both precisions, so a
    /// caller does not have to know which one it handed over.
    pub w: usize,
    /// Elements between consecutive outputs.
    pub out: usize,
}

impl Batch {
    /// One product, which is what [`Gpu::gemm`] is.
    ///
    /// The strides are never read at `count == 1`, but naming the output size
    /// keeps them meaningful rather than arbitrary.
    pub fn single(out: usize) -> Self {
        Self {
            count: 1,
            a: 0,
            w: 0,
            out,
        }
    }
}

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
    "act_snake",
    "act_elu",
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
    "im2col",
    "pack_f16",
    "rms_norm",
    "silu_mul",
    "rope",
    "repeat_kv",
    "stft_dft",
    "istft_ola",
    "upsample_nearest",
    "strided_conv1d",
    "split_heads",
    "split_heads_t",
    "merge_heads",
    "causal_mask",
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

    /// Copies a packed tensor back, as raw halves.
    ///
    /// For tests: the only claim worth making about an f16 tensor is which
    /// bits it holds, and a comparison in f32 would hide a rounding-mode
    /// disagreement behind a conversion.
    pub fn download_u16(&self, x: &CudaSlice<u16>) -> Result<Vec<u16>, CudaError> {
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

    /// Allocates without zeroing.
    ///
    /// # Safety
    ///
    /// The contents are whatever the last owner of that memory left there.
    /// Only sound when the kernel that follows writes every element - which is
    /// worth checking rather than assuming: a tiled kernel that predicates its
    /// stores can leave the ragged edge of a tile untouched, and the resulting
    /// garbage is *plausible* garbage, because it is somebody else's
    /// activations.
    ///
    /// Used where the alternative is a real cost: one attention score matrix
    /// is 45 M floats, and zeroing it 32 times is 5.7 GB of writes that the
    /// matmul immediately overwrites.
    pub unsafe fn uninit(&self, n: usize) -> Result<CudaSlice<f32>, CudaError> {
        // SAFETY: the caller has promised every element is written before it is
        // read. That promise is what this function's own safety comment is
        // about; there is nothing further to check here.
        unsafe { self.stream.alloc::<f32>(n) }.map_err(|source| CudaError::Driver {
            what: "allocating",
            source,
        })
    }

    /// Copies already-packed f16 bits to the device.
    ///
    /// For weights a checkpoint stores narrow, where rounding has already
    /// happened at load - see `xabe_st::StFile::tensor_f16`, which is also
    /// where the range check lives.
    pub fn upload_u16(&self, x: &[u16]) -> Result<CudaSlice<u16>, CudaError> {
        self.stream
            .clone_htod(x)
            .map_err(|source| CudaError::Driver {
                what: "uploading f16 weights",
                source,
            })
    }

    /// Copies a slice to the device as f16, rounding once.
    ///
    /// Round-to-nearest-even on the host, which is what `cvt.rn.f16.f32`
    /// does - so a weight uploaded this way is bit-identical to what the
    /// tiled kernel would have produced from the F32 original.
    pub fn upload_f16(&self, x: &[f32]) -> Result<CudaSlice<u16>, CudaError> {
        let packed: Vec<u16> = x
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        self.stream
            .clone_htod(&packed)
            .map_err(|source| CudaError::Driver {
                what: "uploading f16 weights",
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
    /// Any `k` is accepted, including an odd one. That is worth saying because
    /// it twice was not - first "a multiple of 8", which was wrong about the
    /// instruction, then "even", which was right about the `float2` staging
    /// and still refused the decoder's own arithmetic.
    pub fn gemm(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        self.gemm_batched(
            Operand::F32(a),
            Operand::F32(w),
            bias,
            Batch::single(m * n),
            m,
            k,
            n,
        )
    }

    /// A batch of independent products, one per `blockIdx.z`.
    ///
    /// Attention is twenty of these per layer - one score matrix and one
    /// context per head - and the strides differ between the two, so they are
    /// arguments rather than a shape convention. Everything else is
    /// [`Gpu::gemm`], including the choice between the two kernels.
    // Shapes are arguments, not types - the same convention as the `xabe-dsp`
    // twins these mirror. Bundling m, k and n into a descriptor would satisfy
    // the lint and make every call site say less about what it is doing.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self,
        a: Operand<'_>,
        w: Operand<'_>,
        bias: Option<&CudaSlice<f32>>,
        batch: Batch,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        if (a.half() == 1 || w.half() == 1) && !k.is_multiple_of(2) {
            return Err(CudaError::RaggedContraction { k });
        }
        // SAFETY: both kernels write every element of the tile they own, with
        // the predication covering exactly the (m, n) range - see the store
        // loop in kernels.rs, and `every_output_element_is_written_exactly_once`
        // in the tests, which is the check this relies on.
        let mut out = unsafe { self.uninit(batch.count * m * n) }?;
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let (sa, sw, so) = (batch.a as i64, batch.w as i64, batch.out as i64);
        let (a_half, w_half) = (a.half(), w.half());
        let null: u64 = 0;

        // 128 rows of `a` and 128 of `w` per block, across 8 warps, or one warp
        // per output channel when there are too few rows to fill a tile. The
        // tile is chosen by global traffic rather than by shared capacity - see
        // the derivation in kernels.rs, which is also why those three numbers
        // must move together with GEMM_MT, GEMM_NT and GEMM_WARPS.
        let small = m <= GEMV_MAX_M;
        let f = self.func(if small { "gemv" } else { "gemm" });
        let mut lb = self.stream.launch_builder(f);
        match a {
            Operand::F32(v) => lb.arg(v),
            Operand::F16(v) => lb.arg(v),
        };
        match w {
            Operand::F32(v) => lb.arg(v),
            Operand::F16(v) => lb.arg(v),
        };
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out)
            .arg(&mi)
            .arg(&ki)
            .arg(&ni)
            .arg(&sa)
            .arg(&sw)
            .arg(&so)
            .arg(&a_half)
            .arg(&w_half);

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: if small {
                (n.div_ceil(8) as u32, m as u32, batch.count as u32)
            } else {
                (
                    n.div_ceil(128) as u32,
                    m.div_ceil(128) as u32,
                    batch.count as u32,
                )
            },
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every (batch, m, n) exactly once, `out` is
        // batch*m*n elements, and every global read and write inside the kernel
        // is bounds checked against m, k and n.
        launched(if small { "gemv" } else { "gemm" }, unsafe {
            lb.launch(cfg)
        })?;
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
    /// ELU in place, with alpha 1 - torch's default and the only one used.
    pub fn elu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_elu", x, n, None)
    }

    /// Snake in place: `x + sin^2(a*x)/a`, one alpha per channel.
    ///
    /// HiFTNet's activation and the reason its vocoder is not a plain
    /// HiFi-GAN. Periodic, so the network can represent a harmonic signal
    /// without learning one - which is also why `alpha` is trained per channel
    /// rather than fixed.
    pub fn snake(
        &self,
        x: &mut CudaSlice<f32>,
        alpha: &CudaSlice<f32>,
        ch: usize,
        t: usize,
    ) -> Result<(), CudaError> {
        let (c, tt) = (ch as i32, t as i32);
        let f = self.func("act_snake");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(alpha).arg(&c).arg(&tt);
        launched("act_snake", unsafe { lb.launch(Self::flat(ch * t)) })
    }

    /// A centred real STFT, as a direct DFT. Returns `(real, imag)`, each
    /// `[n_fft / 2 + 1, frames]`.
    ///
    /// A direct transform rather than an FFT because `n_fft` is 16 in the only
    /// caller: sixteen points is below the size at which a butterfly pays for
    /// its plan and its twiddle table.
    pub fn stft(
        &self,
        x: &CudaSlice<f32>,
        window: &CudaSlice<f32>,
        n: usize,
        n_fft: usize,
        hop: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>, usize), CudaError> {
        // torch's `center=True`: the signal is reflect-padded by `n_fft / 2`
        // either side, which is what makes this the frame count.
        let frames = n / hop + 1;
        let bins = n_fft / 2 + 1;
        let mut re = self.zeros(bins * frames)?;
        let mut im = self.zeros(bins * frames)?;
        let (a, b, c, d) = (n as i32, n_fft as i32, hop as i32, frames as i32);
        let f = self.func("stft_dft");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(&mut re)
            .arg(&mut im)
            .arg(window)
            .arg(&a)
            .arg(&b)
            .arg(&c)
            .arg(&d);
        launched("stft_dft", unsafe { lb.launch(Self::flat(bins * frames)) })?;
        Ok((re, im, frames))
    }

    /// The inverse, as overlap-add with torch's window-envelope division.
    ///
    /// One thread per *output* sample, gathering the frames that cover it,
    /// rather than one per frame scattering with atomics. The gather is
    /// deterministic; a scatter's summation order is whatever the scheduler
    /// chose that run, and a vocoder whose output moves between runs cannot be
    /// diffed against anything.
    pub fn istft(
        &self,
        re: &CudaSlice<f32>,
        im: &CudaSlice<f32>,
        window: &CudaSlice<f32>,
        frames: usize,
        n_fft: usize,
        hop: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let out_n = (frames - 1) * hop;
        let mut out = self.zeros(out_n)?;
        let (a, b, c, d) = (out_n as i32, n_fft as i32, hop as i32, frames as i32);
        let f = self.func("istft_ola");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(re)
            .arg(im)
            .arg(&mut out)
            .arg(window)
            .arg(&a)
            .arg(&b)
            .arg(&c)
            .arg(&d);
        launched("istft_ola", unsafe { lb.launch(Self::flat(out_n)) })?;
        Ok(out)
    }

    /// Repeats each timestep `factor` times: `[ch, t]` to `[ch, t * factor]`.
    ///
    /// The half of HiFT's `ups` that is not a convolution. Upstream upsamples
    /// by repetition and *then* convolves; a transposed convolution with the
    /// same weight interleaves zeros instead. Same output length, different
    /// function - which is exactly the kind of difference a shape check will
    /// never catch.
    pub fn upsample_nearest(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
        factor: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let n = ch * t * factor;
        let mut out = self.zeros(n)?;
        let (a, b, c) = (ch as i32, t as i32, factor as i32);
        let f = self.func("upsample_nearest");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b).arg(&c);
        launched("upsample_nearest", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(out)
    }

    /// A strided convolution with left padding only. Returns the output and
    /// its length.
    ///
    /// [`Gpu::conv1d`] is stride-one; this exists for the vocoder's excitation
    /// branch, which decimates one spectrum to three different rates.
    #[allow(clippy::too_many_arguments)]
    pub fn strided_conv1d(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        in_ch: usize,
        t: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        pad_left: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let out_t = (t + pad_left).saturating_sub(k) / stride + 1;
        let mut out = self.zeros(out_ch * out_t)?;
        let dummy = self.zeros(1)?;
        let has = i32::from(bias.is_some());
        let b = bias.unwrap_or(&dummy);
        let (p, q, r, s, u, v, y) = (
            in_ch as i32,
            t as i32,
            out_ch as i32,
            k as i32,
            stride as i32,
            pad_left as i32,
            out_t as i32,
        );
        let f = self.func("strided_conv1d");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(w)
            .arg(b)
            .arg(&mut out)
            .arg(&p)
            .arg(&q)
            .arg(&r)
            .arg(&s)
            .arg(&u)
            .arg(&v)
            .arg(&y)
            .arg(&has);
        launched("strided_conv1d", unsafe {
            lb.launch(Self::flat(out_ch * out_t))
        })?;
        Ok((out, out_t))
    }

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

    /// Root-mean-square normalisation. Mirrors [`xabe_dsp::rms_norm`].
    pub fn rms_norm(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        weight: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // SAFETY: the kernel writes every element of every row it owns, and
        // the grid is one block per row.
        let mut out = unsafe { self.uninit(rows * dim) }?;
        let d = dim as i32;
        let f = self.func("rms_norm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(weight).arg(&mut out).arg(&d).arg(&eps);
        // A power of two, because the reduction halves the block each step.
        let threads = (dim as u32).next_power_of_two().clamp(32, 1024);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        launched("rms_norm", unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// `a = silu(a) * b`. Mirrors [`xabe_dsp::silu_mul`].
    pub fn silu_mul(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func("silu_mul");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b).arg(&len);
        launched("silu_mul", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Rotary position embedding, in place. Mirrors [`xabe_dsp::rope`].
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        first: usize,
    ) -> Result<(), CudaError> {
        self.rope_scaled(x, None, t, heads, head_dim, theta, first)
    }

    /// RoPE with an optional per-pair frequency divisor.
    ///
    /// `freq_div` is Llama-3's `rope_freqs.weight`, `head_dim / 2` long.
    /// Mirrors [`xabe_dsp::rope_scaled`]. Llama-2 passes `None`; passing
    /// `None` for a checkpoint that ships the tensor is a model that stays
    /// fluent for a sentence and drifts after, with no shape to catch it.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_scaled(
        &self,
        x: &mut CudaSlice<f32>,
        freq_div: Option<&CudaSlice<f32>>,
        t: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        first: usize,
    ) -> Result<(), CudaError> {
        let n = t * heads * head_dim / 2;
        let (a, b_, c, d) = (t as i32, heads as i32, head_dim as i32, first as i32);
        let f = self.func("rope");
        let mut lb = self.stream.launch_builder(f);
        // A flag, not a null pointer: every launch argument has to point at
        // something real, so the no-scaling case passes a one-element dummy
        // the kernel is told never to read.
        let dummy = self.zeros(1)?;
        let has = i32::from(freq_div.is_some());
        let div = freq_div.unwrap_or(&dummy);
        lb.arg(x)
            .arg(div)
            .arg(&has)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&theta)
            .arg(&d);
        launched("rope", unsafe { lb.launch(Self::flat(n)) })
    }

    /// Widens a grouped-query key or value cache to the full head count.
    ///
    /// `src` is `[kv_heads, t, head_dim]`, `dst` is `[heads, t, head_dim]`,
    /// and head `h` reads from `h / group`.
    pub fn repeat_kv(
        &self,
        src: &CudaSlice<f32>,
        heads: usize,
        kv_heads: usize,
        t: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let n = heads * t * head_dim;
        let mut dst = unsafe { self.uninit(n)? };
        let group = (heads / kv_heads) as i32;
        let (h, tt, hd) = (heads as i32, t as i32, head_dim as i32);
        let f = self.func("repeat_kv");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(src)
            .arg(&mut dst)
            .arg(&h)
            .arg(&group)
            .arg(&tt)
            .arg(&hd);
        launched("repeat_kv", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(dst)
    }

    /// Lays a convolution out as a matrix: `[t, in_ch]` to `[out_t, in_ch*k]`.
    ///
    /// Whisper's encoder stem is two width-3 convolutions over 3000 positions
    /// at 80 and then 1280 channels. As convolutions they would want a kernel
    /// tuned for channel counts two orders of magnitude past what the VITS
    /// decoder's is written for; as `im2col` then [`Gpu::gemm`] they are two of
    /// the products this card is fastest at, and this gather is the only new
    /// code. Returns the matrix and `out_t`.
    #[allow(clippy::too_many_arguments)]
    pub fn im2col(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        in_ch: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let out_t = (t + 2 * pad - k) / stride + 1;
        let cols = in_ch * k;
        let mut out = self.zeros(out_t * cols)?;
        let (a, b_, c, d, e, g) = (
            t as i32,
            in_ch as i32,
            k as i32,
            stride as i32,
            pad as i32,
            out_t as i32,
        );
        let f = self.func("im2col");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(&mut out)
            .arg(&a)
            .arg(&b_)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&g);
        launched("im2col", unsafe { lb.launch(Self::flat(out_t * cols)) })?;
        Ok((out, out_t))
    }

    /// Rounds a tensor to f16 on the device, so the matmul reads half of it.
    ///
    /// Worth a pass of its own when the result is read many times: a
    /// projection's input is re-read once per column tile - twelve times at
    /// encoder width - so converting it once trades one pass for eleven halved
    /// ones.
    pub fn to_f16(&self, x: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<u16>, CudaError> {
        let mut out = self
            .stream
            .alloc_zeros::<u16>(n)
            .map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })?;
        let len = n as i32;
        let f = self.func("pack_f16");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&len);
        launched("pack_f16", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(out)
    }

    /// `[t, heads*head_dim]` to `[heads, t, head_dim]`.
    pub fn split_heads(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        self.reshape_heads("split_heads", x, t, heads, head_dim)
    }

    /// `[t, heads*head_dim]` to `[heads, head_dim, t]`.
    ///
    /// The value tensor's layout, because the context product reads it down
    /// its time axis and [`Gpu::gemm`] takes its right operand as `[n, k]`.
    pub fn split_heads_t(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        self.reshape_heads("split_heads_t", x, t, heads, head_dim)
    }

    /// `[heads, t, head_dim]` back to `[t, heads*head_dim]`.
    pub fn merge_heads(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        self.reshape_heads("merge_heads", x, t, heads, head_dim)
    }

    /// The body of the three head permutations, which differ only in kernel.
    fn reshape_heads(
        &self,
        name: &'static str,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let n = t * heads * head_dim;
        let mut out = self.zeros(n)?;
        let (a, b_, c) = (t as i32, heads as i32, head_dim as i32);
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_).arg(&c);
        launched(name, unsafe { lb.launch(Self::flat(n)) })?;
        Ok(out)
    }

    /// Sets a batch of score matrices to negative infinity above the diagonal.
    ///
    /// `offset` is how many cached keys precede the queries. Decoding one
    /// token at a time makes the mask a no-op, which is why getting `offset`
    /// wrong survives until something feeds the decoder two tokens at once.
    pub fn causal_mask(
        &self,
        scores: &mut CudaSlice<f32>,
        batch: usize,
        tq: usize,
        tk: usize,
        offset: usize,
    ) -> Result<(), CudaError> {
        let n = batch * tq * tk;
        let (a, b_, c, d) = (batch as i32, tq as i32, tk as i32, offset as i32);
        let f = self.func("causal_mask");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(scores).arg(&a).arg(&b_).arg(&c).arg(&d);
        launched("causal_mask", unsafe { lb.launch(Self::flat(n)) })
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
