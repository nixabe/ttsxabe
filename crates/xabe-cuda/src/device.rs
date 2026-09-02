//! Device handle, module loading, and one method per kernel.
//!
//! [`Gpu::open`] is fallible in the ordinary way and returns
//! [`CudaError::NoDevice`] when there is no card or no driver, so callers can
//! fall back to the CPU path rather than being unable to start. Nothing in this
//! crate is behind a feature flag.

use crate::error::CudaError;
use crate::kernels::{self, SOURCE};
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

/// Set once if `libcuda` turned out not to exist; see [`Gpu::context`].
static DRIVER_MISSING: OnceLock<String> = OnceLock::new();

/// Threads per block for the flat element-wise kernels.
const BLOCK: u32 = 256;

/// Threads per block for the reduction kernels, which use shared memory sized
/// to the block.
const REDUCE_BLOCK: u32 = 256;

/// Output positions per block in the tiled convolution.
const CONV_BLOCK: u32 = 128;

/// Blocks a `gemm` aims to put on the card before it stops splitting `k`.
///
/// The card has 72 SMs and the tile is register-heavy enough that one resident
/// block an SM is what it gets, so this is "fill the machine twice over" -
/// enough to cover the tail of a wave without making the slices so short that
/// the staging stops paying for itself.
const SM_TARGET: usize = 144;

/// The shortest contraction a split slice is allowed.
///
/// A slice reads the whole `GEMM_MT x GEMM_NT` tile footprint however little of
/// `k` it covers, so a short slice is nearly all staging.
const KSLICE_MIN: usize = 512;

/// The most ways `k` is ever split, capping the reduction pass and the
/// `ksplit * batch * m * n` scratch it reads.
const KSPLIT_MAX: usize = 8;

/// The shortest slice a *tail* split is allowed, against [`KSLICE_MIN`] for a
/// fill split.
///
/// The two regimes tolerate different slices, and the sweep that says so is in
/// `docs/BENCHMARKS.md`: a machine-filling split of an under-one-wave launch
/// keeps winning down to 1024-element slices, where a tail split of a
/// multi-wave launch falls off a cliff below about 2048 - at slices of 1707
/// and shorter it measured slower than not splitting at all.
const KSLICE_TAIL: usize = 2048;

/// Past this idle fraction of the last wave, a tail split pays.
///
/// 160 blocks on 144 slots idles 44% of two waves and wants the split; 448
/// blocks idles 22% of four and measured *worse* split. The boundary is
/// between those two measurements, not derived.
const TAIL_IDLE_MIN: f64 = 0.3;

/// How many ways `gemm` splits the contraction at this shape. One is no split.
///
/// A function rather than an expression at the call site so the geometry can be
/// asserted without a device: what it returns decides whether a launch writes
/// `out` or a scratch buffer, and getting it wrong is silent.
///
/// Two regimes, split on whether the launch fills one wave of `SM_TARGET`
/// resident blocks:
///
/// * **Under a wave**, splitting turns idle SMs into concurrent slices and the
///   old rule stands: fill the machine, keep a slice at least `KSLICE_MIN`.
/// * **Over a wave**, more blocks do not buy concurrency - they buy a shorter
///   *tail*. A launch of 1.11 waves runs its last 16 blocks on an otherwise
///   idle machine for a whole block's k-loop, and splitting s ways cuts that
///   straggler's loop by s. That is worth having exactly when the idle
///   fraction is large - the translator's 5120-wide projections at 512 tokens,
///   160 blocks, 44% idle, measured 15-25% faster split four ways - and worth
///   avoiding when the waves are already full: the same model's 13824-wide
///   projections, three exact waves, measured 80% *slower* split in two. The
///   slice floor is higher here too; see `KSLICE_TAIL`.
pub fn ksplit_for(m: usize, k: usize, n: usize, batch: usize) -> usize {
    let blocks =
        n.div_ceil(kernels::GEMM_NT as usize) * m.div_ceil(kernels::GEMM_MT as usize) * batch;
    let blocks = blocks.max(1);
    if blocks <= SM_TARGET {
        return (SM_TARGET / blocks)
            .min(k / KSLICE_MIN)
            .clamp(1, KSPLIT_MAX);
    }
    let waves = blocks.div_ceil(SM_TARGET);
    let idle = (waves * SM_TARGET - blocks) as f64 / (waves * SM_TARGET) as f64;
    if idle < TAIL_IDLE_MIN {
        return 1;
    }
    (2..=4usize)
        .rev()
        .find(|&s| k / s >= KSLICE_TAIL)
        .unwrap_or(1)
}

/// Output channels each convolution thread accumulates. Must match `OC_TILE`
/// in the device source.
const CONV_OC_TILE: u32 = 8;

/// Time positions each convolution thread accumulates. Must match `T_REG`.
const CONV_T_REG: u32 = 4;

/// What the fused decode attention keeps between calls.
///
/// A block per chunk of keys writes its partial - a running maximum, a sum and
/// an unnormalised context - here, and the last block to finish for a head
/// merges them. The counters that decide which block is last start at zero
/// and are reset by the block that reads them, so this is allocated once and
/// never zeroed again; the partials are never read before they are written.
///
/// Held by the caller rather than by [`Gpu`] because the device handle is
/// shared by every stage, and two stages decoding by turns must not share a
/// counter. It grows when a longer context needs more chunks, which is a
/// logarithmic number of allocations over a conversation and none in steady
/// state - the same reasoning the KV caches follow.
/// The partials and the counter `Gpu::gemv_norm` merges through.
///
/// One float a block and one counter, owned by the caller for the same
/// reason [`DecodeScratch`] is: the kernel runs twice a layer for every
/// decoded token, and an allocation a launch is the cost it exists to
/// remove. Grows once when a wider projection needs more blocks.
pub struct NormScratch {
    part: Option<CudaSlice<f32>>,
    ctr: Option<CudaSlice<u32>>,
    blocks: usize,
}

impl NormScratch {
    /// Empty; the first call allocates.
    pub fn new() -> Self {
        Self {
            part: None,
            ctr: None,
            blocks: 0,
        }
    }
}

impl Default for NormScratch {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DecodeScratch {
    part: Option<CudaSlice<f32>>,
    ctr: Option<CudaSlice<u32>>,
    kv_heads: usize,
    head_dim: usize,
    chunks: usize,
}

impl DecodeScratch {
    /// Empty; the first call allocates.
    pub fn new() -> Self {
        Self {
            part: None,
            ctr: None,
            kv_heads: 0,
            head_dim: 0,
            chunks: 0,
        }
    }
}

impl Default for DecodeScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Where [`Gpu::gemv_into`] puts each column of its one output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutLayout {
    /// `out[col]`, the plain row.
    Row,
    /// Column `h * head_dim + j` at `(h * cap + pos) * head_dim + j` - one
    /// position of a head-major `[heads, cap, head_dim]` key cache, which is
    /// what `cache_append` scatters a projected row into.
    KeyCache {
        /// Elements a head.
        head_dim: usize,
        /// Positions the cache has room for.
        cap: usize,
        /// The position being written.
        pos: usize,
    },
    /// Column `c` at `c * cap + pos` - one position of a transposed
    /// `[heads, head_dim, cap]` value cache, `cache_append_t`'s layout.
    ValueCache {
        /// Positions the cache has room for.
        cap: usize,
        /// The position being written.
        pos: usize,
    },
}

/// Which cache the fused decode attention reads.
enum KvCache<'a> {
    F32(&'a CudaSlice<f32>, &'a CudaSlice<f32>),
    F16(&'a CudaSlice<u16>, &'a CudaSlice<u16>),
}

/// An open CUDA device with the kernels compiled and loaded.
pub struct Gpu {
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    funcs: HashMap<&'static str, CudaFunction>,
    /// One zeroed float, for launch arguments that must point at something.
    ///
    /// Every kernel argument has to be a real pointer, so an optional input is
    /// passed as a flag plus a buffer the kernel is told never to read. Made
    /// once: allocating and zeroing it per call was 64 allocations and 64
    /// memsets a decoded token, for a float nothing ever looks at.
    dummy: CudaSlice<f32>,
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
pub const GEMV_MAX_M: usize = 4;

/// The widest row [`Gpu::gemv_ln`] normalises: its last block holds the row
/// in registers, `GN_NL` float4 a thread.
pub const GEMV_LN_MAX_N: usize = 4 * kernels::GN_NL as usize * kernels::GEMV_WARPS as usize * 32;

/// A block-quantized weight format, by its ggml type id.
///
/// This mirrors `xabe_gguf::GgmlType`'s quantized half and deliberately does
/// not reuse it: the crate map has `xabe-cuda` depending on `xabe-dsp` alone,
/// and a GPU crate that had to open a GGUF to name a block layout would be the
/// wrong edge. The duplication is three small tables, and
/// `quant_sizes_match_the_container_crate` pins them together so a drift is a
/// test failure rather than a wrong answer.
///
/// The unquantized widths are absent on purpose. F32 and F16 are
/// [`Operand::F32`] and [`Operand::F16`], which have their own kernel paths;
/// a `Quant::F16` would be a second spelling of one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// 4-bit, centred, one f16 scale per 32.
    Q4_0,
    /// 4-bit, offset, an f16 scale and minimum per 32.
    Q4_1,
    /// 5-bit, centred, one f16 scale per 32.
    Q5_0,
    /// 5-bit, offset, an f16 scale and minimum per 32.
    Q5_1,
    /// 8-bit, one f16 scale per 32.
    Q8_0,
    /// "K-quant" 2-bit: 256-element superblock, 4-bit scales and minimums.
    Q2K,
    /// "K-quant" 3-bit: 6-bit scales, plus a high-bit mask.
    Q3K,
    /// "K-quant" 4-bit: 6-bit scales and minimums, eight sub-blocks of 32.
    Q4K,
    /// "K-quant" 5-bit: `Q4_K` plus one high bit per element.
    Q5K,
    /// "K-quant" 6-bit: 8-bit signed scales, no minimum.
    Q6K,
}

impl Quant {
    /// The ggml type id, which is what the kernel switches on.
    pub const fn id(self) -> i32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }

    /// The format for a ggml type id, or `None` if no kernel path reads it.
    ///
    /// Takes a bare id rather than a container type so that a caller holding a
    /// `GgmlType` can map it without this crate depending on the crate that
    /// defines one. `None` covers both the unquantized widths - which have
    /// their own operands - and the families this workspace refuses, `IQ*`,
    /// `TQ*` and `Q8_K`.
    pub const fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            _ => return None,
        })
    }

    /// Elements per block.
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Bytes per block.
    pub const fn type_size(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Bytes between blocks once the tensor is on the device.
    ///
    /// The same as [`Quant::type_size`] for every format but Q6_K, which is
    /// padded from 210 to 224. A 16-byte load has to be 16-byte aligned and 210
    /// is not a multiple of 16, so consecutive blocks in the file land at every
    /// alignment in turn and the wide mat-vec cannot read them. 224 is the next
    /// multiple of 16, costing 6.7% of the bytes of a Q6_K tensor.
    ///
    /// This is the one place the engine's copy of a checkpoint is not
    /// byte-for-byte the file's. It began as a stride alone; the blocks are now
    /// also re-packed on the way - same 210 bytes of payload, reordered so a
    /// staged run of elements is one aligned read per operand. See
    /// [`Gpu::upload_quant`] for the layout and why.
    pub const fn device_stride(self) -> usize {
        match self {
            Self::Q6K => 224,
            _ => self.type_size(),
        }
    }

    /// Bytes a tensor of `elements` occupies in this format.
    ///
    /// `elements` must be a whole number of blocks; the callers that size an
    /// upload have already checked that against the tensor's own shape.
    pub const fn bytes(self, elements: usize) -> usize {
        elements / self.block_size() * self.type_size()
    }
}

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
    /// Block-quantized, and unpacked *inside* the matmul rather than at load.
    ///
    /// This is the variant that changes what fits on a card. The others store
    /// one number per element; this one stores the checkpoint's own packed
    /// blocks and pays a dozen integer ops per element to read them, which on
    /// a bandwidth-bound kernel is the cheaper half of the trade. A weight and
    /// only a weight can be stored this way - see [`CudaError::QuantizedActivation`].
    Q {
        /// The packed blocks, byte for byte as they sit in the container.
        data: &'a CudaSlice<u8>,
        /// Which layout to read them with.
        ty: Quant,
    },
    /// Full precision, with the int8 twin the packed mat-vec wants already
    /// taken.
    ///
    /// Identical to [`Operand::F32`] in what it means and in what every path
    /// but one does with it. The difference is bookkeeping: a transformer layer
    /// feeds one normed activation to three projections and another to two, and
    /// quantizing it once per *projection* is four fifths of a kernel launch
    /// and an allocation wasted. The caller that knows an activation is about
    /// to be used more than once takes [`Gpu::quantize_activation`] itself and
    /// hands the result to each.
    ///
    /// A mismatched `q8` is a caller error the kernel cannot see - it would
    /// read another tensor's codes and produce plausible numbers - so
    /// [`Gpu::gemm_batched`] checks its shape against `m` and `k` and refuses.
    F32Q {
        /// The activation, unchanged.
        data: &'a CudaSlice<f32>,
        /// Its int8 twin.
        q8: &'a Q8,
    },
}

/// An activation quantized to int8, ready for the packed mat-vec.
///
/// Codes and per-group scales share one allocation: the codes first, then the
/// scales at [`Q8::scale_offset`] bytes. Two allocations would be clearer and
/// there are enough of these a token that clearer is not what this needs.
#[derive(Debug)]
pub struct Q8 {
    /// `rows * k` codes, then `rows * k / 32` scales as f32.
    buf: CudaSlice<i8>,
    /// Rows quantized, counting every row of every batch element.
    rows: usize,
    /// The contraction length these codes were taken along.
    k: usize,
}

impl Q8 {
    /// Bytes from the start of the buffer to the first scale.
    ///
    /// A multiple of four, because `k` is a multiple of 1024 - which is what
    /// makes reading the scales as f32 from the same allocation sound.
    pub const fn scale_offset(&self) -> usize {
        self.rows * self.k
    }

    /// The shape these codes were taken at.
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.k)
    }
}

impl<'a> Operand<'a> {
    /// Whether this side is packed, as the kernel's flag.
    fn half(self) -> i32 {
        i32::from(matches!(self, Operand::F16(_)))
    }

    /// The int8 twin a caller took ahead of time, if any.
    fn q8(self) -> Option<&'a Q8> {
        match self {
            Operand::F32Q { q8, .. } => Some(q8),
            _ => None,
        }
    }

    /// The block format, if this side is quantized.
    fn quant(self) -> Option<Quant> {
        match self {
            Operand::Q { ty, .. } => Some(ty),
            _ => None,
        }
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
    /// Elements between consecutive *rows* of the right operand, or 0 for `k`.
    ///
    /// Only the value cache needs it, and it needs it because the cache is one
    /// buffer with room for more positions than are in it: a head's values are
    /// `[head_dim, capacity]` and the contraction is over the `tk` of them that
    /// exist. Zero rather than `Option` because zero is not a stride any real
    /// operand has, and because every other caller would have to spell out
    /// `None`.
    pub w_row: usize,
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
            w_row: 0,
        }
    }
}

/// Threads a fused normalise-and-quantise block runs.
///
/// A multiple of 32, so a warp owns four whole 32-column scale groups, and the
/// largest the hardware allows, because at one block a row this block is all
/// the parallelism a decode step has.
///
/// Both smaller choices were measured and are worse. At 256 with scalar loads
/// it lost to the unfused pair outright; at 256 with `float4`, chosen so that
/// 5120 divides exactly instead of leaving a quarter-full last iteration, the
/// 13 B translator went from 55.5 to 55.1 tok/s. Starving the block costs more
/// than the ragged iteration it avoids.
const RMS_THREADS: u32 = 1024;

const NAMES: &[&str] = &[
    "conv1d",
    "conv1d_short",
    "depthwise_conv1d",
    "transposed_conv1d",
    "linear",
    "gemm",
    "gemm_i8_q4k",
    "gemm_i8_q6k",
    "gemm_i8_q4k_narrow",
    "gemm_i8_q6k_narrow",
    "gemm_reduce",
    "flash_attn",
    "flash_attn_64",
    "gemv",
    "gemv_norm",
    "gemv_q_rows2",
    "gemv_q_rows3",
    "gemv_q_rows4",
    "gemv_ln",
    "gemv_qkv_f16",
    "gemv_rows",
    "cache_append_f16",
    "cache_append_t_f16",
    "rope_cache_f16",
    "cache_grow_f16",
    "flash_attn_h",
    "quantize_q8",
    "cache_append",
    "cache_append_t",
    "cache_grow",
    "softmax_causal",
    "layer_norm",
    "layer_norm_add",
    "silu_mul_pair",
    "layer_norm_add_f16",
    "layer_norm_mod",
    "gate_add",
    "act_gelu_f16",
    "softmax_rows",
    "act_relu",
    "act_leaky_relu",
    "act_snake",
    "act_elu",
    "act_mish",
    "act_silu",
    "act_gelu_tanh",
    "act_tanh",
    "act_gelu",
    "gated_activation",
    "gated_activation_rows",
    "add_inplace",
    "sub_inplace",
    "mul_inplace",
    "add_strided",
    "scale_inplace",
    "copy_range",
    "copy_into",
    "copy_from_into",
    "concat2",
    "relu_mask",
    "attn_weights_update",
    "taco_energies",
    "taco_context",
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
    "rope_gptj",
    "grouped_conv1d",
    "split_heads",
    "split_heads_t",
    "split_heads_f16",
    "split_heads_t_f16",
    "merge_heads",
    "causal_mask",
    "lstm_gates",
    "coupling_inverse",
    "attn_decode_h128_c32",
    "attn_decode_h128",
    "attn_decode_h128_c128",
    "attn_decode_h64",
    "attn_decode_h64_g8",
    "attn_decode_h64_g8_c32",
    "attn_decode_f64",
    "embed_q",
];

impl Gpu {
    /// Opens device `ordinal` and compiles the kernels.
    pub fn open(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = Self::context(ordinal)?;
        let stream = ctx.default_stream();

        // Keep freed stream-ordered allocations in the pool across
        // synchronises. The default release threshold is zero, which hands
        // every freed block back to the OS at the next sync - so the first
        // sizeable allocation after one is a real `cuMemCreate`, and a
        // multi-megabyte scratch buffer costs most of a millisecond that the
        // pool exists to not charge. Measured, not read off a manual: with
        // the threshold at zero a 19 MB per-launch scratch put a constant
        // ~0.9 ms floor under every tiled matmul in a synchronise-per-round
        // benchmark, and raising it removed the floor. Best effort: a driver
        // too old to have pools still runs everything else.
        unsafe {
            use cudarc::driver::{result, sys};
            if let Ok(dev) = result::device::get(ordinal as i32)
                && let Ok(pool) = result::device::get_default_mem_pool(dev)
            {
                let threshold: u64 = u64::MAX;
                let _ = result::mem_pool::set_attribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut core::ffi::c_void,
                );
            }
        }

        // Compiled for the development target. NVRTC will happily target a
        // newer architecture, but pinning it keeps the generated code the same
        // on every machine that runs the differential tests.
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_75"),
            ..Default::default()
        };
        // Caught for the same reason as the driver above, and it is not the
        // same machine that hits it: a box with the NVIDIA *driver* and no
        // CUDA *toolkit* loads `libcuda` and then panics on `libnvrtc`. That
        // is an ordinary install, not a broken one, so it gets an error.
        let ptx =
            match std::panic::catch_unwind(|| cudarc::nvrtc::compile_ptx_with_opts(SOURCE, opts)) {
                Ok(r) => r.map_err(|e| CudaError::Compile(format!("{e:?}")))?,
                Err(_) => {
                    return Err(CudaError::NoDevice(
                        "the NVRTC library is not on this machine; \
                     the driver is installed but the CUDA toolkit is not"
                            .to_string(),
                    ));
                }
            };
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
        let dummy = stream
            .alloc_zeros::<f32>(1)
            .map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })?;
        Ok(Self {
            stream,
            module,
            funcs,
            dummy,
        })
    }

    /// Creates the context, turning "there is no CUDA here" into an error.
    ///
    /// `cudarc` is built with `fallback-dynamic-loading`, which resolves
    /// `libcuda` at first use and **panics** if it is not on the machine -
    /// `panic_no_lib_found` in `cudarc/src/lib.rs`, which returns `!`. A panic
    /// is the wrong shape for that answer. "This box has no CUDA" is an
    /// ordinary configuration, not a bug: every caller in this workspace
    /// already handles [`CudaError::NoDevice`], and the tests skip on it.
    ///
    /// Unwinding across the FFI boundary is not the risk it looks like here:
    /// the panic is raised by Rust code in `cudarc`, before any call into the
    /// driver, because the library it would call was never loaded.
    ///
    /// The verdict is remembered because it cannot change while the process
    /// lives, and because catching the same panic once per test fills a log
    /// with the same backtrace thirty-four times.
    fn context(ordinal: usize) -> Result<Arc<CudaContext>, CudaError> {
        if let Some(why) = DRIVER_MISSING.get() {
            return Err(CudaError::NoDevice(why.clone()));
        }
        match std::panic::catch_unwind(|| CudaContext::new(ordinal)) {
            Ok(Ok(ctx)) => Ok(ctx),
            // A driver that is present and refuses: no such ordinal, no
            // device, a version mismatch. Already the right shape.
            Ok(Err(e)) => Err(CudaError::NoDevice(format!("{e:?}"))),
            Err(_) => {
                let why = "the CUDA driver library is not on this machine";
                let _ = DRIVER_MISSING.set(why.to_string());
                Err(CudaError::NoDevice(why.to_string()))
            }
        }
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

    /// Allocates a zeroed f16 device buffer, of `n` *elements* rather than
    /// words.
    ///
    /// A cache is the caller: it is written a token at a time and read in
    /// full, so the positions past the live length are read before anything
    /// writes them and must be zero rather than whatever was there.
    pub fn zeros_f16(&self, n: usize) -> Result<CudaSlice<u16>, CudaError> {
        self.stream
            .alloc_zeros::<u16>(n)
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

    /// Allocates an int8 buffer without zeroing.
    ///
    /// # Safety
    ///
    /// As [`Gpu::uninit`]. Only used for the activation `quantize_q8` writes,
    /// which writes every element of it.
    unsafe fn uninit_i8(&self, n: usize) -> Result<CudaSlice<i8>, CudaError> {
        // SAFETY: the caller has promised every element is written first.
        unsafe { self.stream.alloc::<i8>(n) }.map_err(|source| CudaError::Driver {
            what: "allocating",
            source,
        })
    }

    /// Quantises an activation to int8 in groups of 32, one scale a group.
    ///
    /// `rows` counts every row of every batch element, laid out `[batch, m, k]`
    /// with `sa` between batch elements in the input and nothing between them
    /// in the output - the output is dense, because nothing but the mat-vec
    /// reads it.
    ///
    /// Mirrors `xabe_dsp::quantize_q8`, which is the reference the
    /// differential test compares against.
    fn quantize_into(
        &self,
        a: &CudaSlice<f32>,
        k: usize,
        rows: usize,
        sa: i64,
        m: usize,
    ) -> Result<Q8, CudaError> {
        let groups = k / 32;
        // SAFETY: one lane per element and one group per warp covers every code
        // and every scale of both regions; the grid is rounded up and the
        // excess returns before writing.
        let mut buf = unsafe { self.uninit_i8(rows * k + rows * groups * 4) }?;
        let off = (rows * k) as i32;
        let (ki, ri, mi) = (k as i32, rows as i32, m as i32);
        let f = self.func("quantize_q8");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a)
            .arg(&mut buf)
            .arg(&off)
            .arg(&ki)
            .arg(&ri)
            .arg(&sa)
            .arg(&mi);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((rows * groups).div_ceil(8) as u32, 1, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every group of every row exactly once, and
        // the buffer is sized to it.
        launched("quantize_q8", unsafe { lb.launch(cfg) })?;
        Ok(Q8 { buf, rows, k })
    }

    /// Takes the int8 twin of an activation that several matmuls will read.
    ///
    /// The point of doing it here rather than inside [`Gpu::gemm_batched`] is
    /// that a transformer layer reuses one normed activation across three
    /// projections and another across two. Quantizing per projection is correct
    /// and wasteful; this is the same work done once. Hand the result back as
    /// [`Operand::F32Q`].
    ///
    /// Rejects a shape the packed mat-vec would not use anyway, rather than
    /// silently producing codes nothing reads.
    /// An int8 twin of `rows` rows of `k`, zeroed, for a kernel that writes
    /// it a row at a time - [`Self::attn_decode_f16_q_row`]. `k` must be a
    /// multiple of 32, one scale a group.
    pub fn q8_zeros(&self, rows: usize, k: usize) -> Result<Q8, CudaError> {
        if !k.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock { k, block: 32 });
        }
        let buf = self
            .stream
            .alloc_zeros::<i8>(rows * k + rows * (k / 32) * 4)
            .map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })?;
        Ok(Q8 { buf, rows, k })
    }

    pub fn quantize_activation(
        &self,
        a: &CudaSlice<f32>,
        rows: usize,
        k: usize,
    ) -> Result<Q8, CudaError> {
        if !k.is_multiple_of(256) {
            return Err(CudaError::RaggedBlock { k, block: 256 });
        }
        self.quantize_into(a, k, rows, (rows * k) as i64, rows)
    }

    /// Quantises an activation and copies both halves back.
    ///
    /// For the differential test against `xabe_dsp::quantize_q8`. The engine
    /// never needs the codes on the host - the mat-vec consumes them where they
    /// are - so this exists only to make the CUDA half comparable.
    pub fn quantize_q8_for_test(
        &self,
        a: &CudaSlice<f32>,
        k: usize,
        rows: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), CudaError> {
        let q = self.quantize_into(a, k, rows, (rows * k) as i64, rows)?;
        self.q8_parts_for_test(&q)
    }

    /// A twin's codes and scales, back on the host, for the differential
    /// tests only.
    pub fn q8_parts_for_test(&self, q: &Q8) -> Result<(Vec<i8>, Vec<f32>), CudaError> {
        let raw = self
            .stream
            .clone_dtoh(&q.buf)
            .map_err(|source| CudaError::Driver {
                what: "downloading",
                source,
            })?;
        let (codes, tail) = raw.split_at(q.scale_offset());
        let scales = tail
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8]))
            .collect();
        Ok((codes.to_vec(), scales))
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

    /// Copies packed quantization blocks to the device, byte for byte.
    ///
    /// No conversion of any kind: the bytes that sit in the GGUF are the bytes
    /// that sit in VRAM, and `q_elem` in the kernel is the only thing that ever
    /// interprets them. That is the point - a conversion here would put the
    /// weights back at full width and give the whole exercise away.
    pub fn upload_u8(&self, x: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        self.stream
            .clone_htod(x)
            .map_err(|source| CudaError::Driver {
                what: "uploading quantized weights",
                source,
            })
    }

    /// Copies packed blocks to the device at [`Quant::device_stride`].
    ///
    /// The path every model loader takes. For all but Q6_K it is
    /// [`Gpu::upload_u8`] with a length check; for Q6_K it re-strides *and*
    /// re-packs the blocks on the way - see `Gpu::q6k_device_block` - which
    /// is why the check is a rejection and not a `debug_assert`: a `bytes`
    /// that is not a whole number of blocks would be rebuilt into garbage that
    /// still decodes to plausible numbers.
    pub fn upload_quant(&self, q: Quant, bytes: &[u8]) -> Result<CudaSlice<u8>, CudaError> {
        let ts = q.type_size();
        if !bytes.len().is_multiple_of(ts) {
            return Err(CudaError::RaggedBlockBytes {
                len: bytes.len(),
                ty: ts,
            });
        }
        let stride = q.device_stride();
        if stride == ts {
            return self.upload_u8(bytes);
        }
        let blocks = bytes.len() / ts;
        let mut wide = vec![0u8; blocks * stride];
        for (b, src) in bytes.chunks_exact(ts).enumerate() {
            let dst = &mut wide[b * stride..(b + 1) * stride];
            match q {
                Quant::Q6K => Self::q6k_device_block(src, dst),
                // No other format is re-strided today; if one ever is, it gets
                // a plain copy until its kernels ask for more.
                _ => dst[..ts].copy_from_slice(src),
            }
        }
        self.upload_u8(&wide)
    }

    /// One Q6_K block, file layout to device layout.
    ///
    /// The file pairs a `ql` byte's nibbles 64 elements apart and scatters a
    /// `qh` byte's four fields 32 apart, which makes every kernel that stages
    /// a *run* of elements fetch twice the bytes it decodes. The device block
    /// pairs nibbles 32 apart - exactly Q4_K's shape, elements `j` and
    /// `j + 32` in one byte - and packs the 2-bit high fields one 16-element
    /// run to a word, element `e` at bits `8 * (e % 4) + 2 * (e / 4)`, so a
    /// staged run is one aligned read per operand and every fetched byte is
    /// used. Scales and `d` keep their offsets; the tail of the 224 pads.
    ///
    /// Every device-side reader of a Q6_K block decodes this layout and only
    /// this layout - `q_elem` documents it on the kernel side - and nothing
    /// downloads a block back, so the file's own order exists on the card
    /// nowhere at all.
    fn q6k_device_block(src: &[u8], dst: &mut [u8]) {
        let lo4 = |e: usize| (src[(e >> 7) * 64 + (e & 63)] >> (4 * ((e >> 6) & 1))) & 0x0F;
        let hi2 = |e: usize| (src[128 + (e >> 7) * 32 + (e & 31)] >> (2 * ((e >> 5) & 3))) & 0x03;
        for p in 0..4 {
            for b in 0..32 {
                let e = 64 * p + b;
                dst[32 * p + b] = lo4(e) | (lo4(e + 32) << 4);
            }
            for h in 0..2 {
                for s in 0..2 {
                    for v in 0..4 {
                        let e0 = 64 * p + 32 * s + 16 * h + v;
                        dst[128 + 16 * p + 8 * h + 4 * s + v] =
                            hi2(e0) | (hi2(e0 + 4) << 2) | (hi2(e0 + 8) << 4) | (hi2(e0 + 12) << 6);
                    }
                }
            }
        }
        dst[192..210].copy_from_slice(&src[192..210]);
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
    /// General 1-D convolution. Mirrors `xabe_dsp::conv1d`.
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

    /// Depthwise convolution. Mirrors `xabe_dsp::depthwise_conv1d`.
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

    /// Transposed convolution. Mirrors `xabe_dsp::transposed_conv1d`.
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

    /// Dense projection. Mirrors `xabe_dsp::linear`.
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
    /// worth 180x over `linear` at the ASR's encoder shapes and costs one
    /// rounding of each operand. It measures 22.4 TFLOP/s there against an
    /// instruction ceiling of 102.3 on this card, so the headroom is in the
    /// staging rather than the arithmetic; docs/KERNELS.md has both numbers
    /// and what is and is not known about the difference. `linear` stays for the places that want exact f32 and are small
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
        self.gemm_batched_from(a, w, 0, bias, batch, m, k, n)
    }

    /// [`Self::gemm_batched`] with the weight read from row `w_first` of
    /// its allocation rather than from its start.
    ///
    /// This is what lets several projections of one activation live in one
    /// allocation and still be run one at a time: the chat model holds its
    /// q, k and v as one `[4096 + 1024 + 1024, 4096]` block so that a decode
    /// step projects all three in one launch, and a prompt - whose rows must
    /// come out `[n, out]` a projection - starts each product at that
    /// projection's first row. `w_first` counts rows, so the offset is a
    /// whole number of blocks for a packed weight and of elements otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched_from(
        &self,
        a: Operand<'_>,
        w: Operand<'_>,
        w_first: usize,
        bias: Option<&CudaSlice<f32>>,
        batch: Batch,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // The offset in the weight's own units, and the bytes it has to have.
        let w_skip = w_first * k;
        let (w_len, w_need) = match w {
            Operand::Q { data, ty } => {
                let bs = ty.block_size();
                if !w_skip.is_multiple_of(bs) {
                    return Err(CudaError::RaggedBlock {
                        k: w_skip,
                        block: bs,
                    });
                }
                (data.len(), (w_skip / bs + n * k / bs) * ty.device_stride())
            }
            Operand::F16(v) => (v.len(), w_skip + n * k),
            Operand::F32(v) | Operand::F32Q { data: v, .. } => (v.len(), w_skip + n * k),
        };
        if w_first > 0 && w_len < w_need {
            return Err(CudaError::SliceOverrun {
                at: w_need,
                len: w_len,
            });
        }
        // The f16 paths that take an odd contraction are the two mat-vecs, and
        // they take it because they have to: between them they contract an f16
        // value cache over however many positions have been decoded, and half
        // of those are odd. A grouped-query model arrives with `group` rows and
        // a multi-head one with a single row, so both shapes occur. Each reads
        // the last element as a lone half rather than as part of a pair - see
        // `gemv_rows` and the tail of `gemv`'s f16 branch. The tiled kernel
        // addresses whole words throughout and an odd `k` has no layout in it
        // at all, so it keeps the refusal.
        let odd_ok = matches!(w, Operand::F16(_))
            && a.half() == 0
            && m <= GEMV_MAX_M
            && matches!(a, Operand::F32(_) | Operand::F32Q { .. });
        if (a.half() == 1 || w.half() == 1) && !k.is_multiple_of(2) && !odd_ok {
            return Err(CudaError::RaggedContraction { k });
        }
        // Only a weight is ever stored packed. An activation is produced by the
        // previous kernel at f32 and consumed by this one, so there is nothing
        // to quantize and no kernel path that would read it.
        if a.quant().is_some() {
            return Err(CudaError::QuantizedActivation);
        }
        // A row must start on a block boundary, or `q_at` would read across the
        // edge into the previous row's last block and be wrong rather than out
        // of bounds. GGUF guarantees it for the fastest-varying dimension,
        // which is this `k`; the check is here because "guaranteed by the
        // format" is exactly the assumption worth failing loudly on.
        if let Some(q) = w.quant()
            && !k.is_multiple_of(q.block_size())
        {
            return Err(CudaError::RaggedBlock {
                k,
                block: q.block_size(),
            });
        }
        // SAFETY: both kernels write every element of the tile they own, with
        // the predication covering exactly the (m, n) range - see the store
        // loop in kernels.rs, and `every_output_element_is_written_exactly_once`
        // in the tests, which is the check this relies on.
        let mut out = unsafe { self.uninit(batch.count * m * n) }?;
        // A row stride is meaningful only where a row is read as a row. The
        // packed paths derive it from the block layout, so a caller asking for
        // one there is asking for something that would be silently ignored.
        // The f16 path reads words of two elements and honours it halved,
        // which is what an f16 value cache needs and why it must be even.
        if batch.w_row != 0 && !matches!(w, Operand::F32(_) | Operand::F16(_)) {
            return Err(CudaError::StridedNonF32Weight {
                stride: batch.w_row,
            });
        }
        if let Operand::F16(_) = w
            && !batch.w_row.is_multiple_of(2)
        {
            return Err(CudaError::OddCacheCapacity { cap: batch.w_row });
        }
        let (mi, ki, ni) = (m as i32, k as i32, n as i32);
        let w_rs = batch.w_row as i32;
        let (sa, sw, so) = (batch.a as i64, batch.w as i64, batch.out as i64);
        let (a_half, w_half) = (a.half(), w.half());
        let (w_quant, q_bs, q_ts) = match w.quant() {
            Some(q) => (q.id(), q.block_size() as i32, q.device_stride() as i32),
            None => (0, 0, 0),
        };
        let null: u64 = 0;

        // `GEMM_MT` rows of `a` and `GEMM_NT` of `w` per block, across
        // `GEMM_WARPS` warps, or one warp per output channel when there are too
        // few rows to fill a tile. Those three are read out of the kernel's own
        // `#define`s rather than written again here - see `kernels::define`.
        let small = m <= GEMV_MAX_M;

        // The integer tensor cores, when the weight is a K-quant.
        //
        // Both operands are quantized on this path where the f16 kernel
        // rounded only the operands and kept f32 accumulation, so it is a
        // second deliberate approximation and a larger one - see `gemm_i8`. It
        // is what llama.cpp does for the same shapes, and the integer
        // instruction runs at twice the f16 one on this card, measured - not
        // the four times an earlier note claimed from a half-rate f32
        // accumulate this Quadro does not have. See docs/KERNELS.md.
        //
        // `k` must be a multiple of 256 because that is what the quantiser
        // wants; the block check above already requires it of these two
        // formats, so the condition costs nothing and says what it needs.
        let use_i8 = !small
            && matches!(a, Operand::F32(_) | Operand::F32Q { .. })
            && matches!(w.quant(), Some(Quant::Q4K | Quant::Q6K))
            && q_bs == 256
            && k.is_multiple_of(256);

        // The wide packed mat-vec reads the activation as int8, because a lane
        // that loads sixteen bytes of quants covers 32 elements and 32 f32
        // activations cost more to fetch than the wide load saves. Quantising
        // is a kernel of its own rather than a change to any call site: the
        // buffer lives exactly as long as this launch.
        //
        // Restricted to what the fast path actually handles - four super-blocks
        // to a warp, so `k` must be a multiple of 1024 - and to an f32
        // activation, which every caller of a quantized matmul passes.
        let wide = small
            && matches!(a, Operand::F32(_) | Operand::F32Q { .. })
            && matches!(w.quant(), Some(Quant::Q4K | Quant::Q6K))
            && k.is_multiple_of(256);
        // A zero left-operand stride means every product of the batch reads the
        // same activation - the attention projections are three matrices
        // against one normalised input - so it is quantized once and the
        // kernel is told to stop advancing between products.
        let shared_a = batch.count > 1 && batch.a == 0;
        let q_rows = if shared_a { m } else { batch.count * m };
        // What both packed paths add per batch element when they index the
        // codes, which are dense `[batch, a_rows, k]`.
        let a_rows = if shared_a { 0i32 } else { m as i32 };

        // A caller's own twin has to have been taken at this shape. Nothing in
        // the kernel could notice otherwise: it would index another tensor's
        // codes, in bounds, and return numbers.
        if let Some(q8) = a.q8()
            && q8.shape() != (q_rows, k)
        {
            let (rows, kk) = q8.shape();
            return Err(CudaError::MismatchedQ8 {
                rows,
                k: kk,
                want_rows: q_rows,
                want_k: k,
            });
        }
        // The mat-vec's fast path and the tiled integer kernel read the same
        // codes in the same layout, so they share one quantisation.
        //
        // Quantizing `count * m` rows for a shared activation would be the
        // same numbers two or three times over; `q_rows` above says so.
        let int8 = wide || use_i8;
        let taken = match (int8, a) {
            (true, Operand::F32(v)) => Some(self.quantize_into(v, k, q_rows, sa, m)?),
            _ => None,
        };
        let q8 = match (int8, a.q8()) {
            (true, Some(q)) => Some(q),
            _ => taken.as_ref(),
        };

        // Split the contraction when the tile leaves the machine idle.
        //
        // A 128-token prefill is one tile of `m`, so a 1024-wide projection is
        // eight blocks on 72 SMs and the tile size cannot fix it: shrinking
        // `GEMM_MT` to make more blocks makes each weight dequantized once per
        // block instead of once. Splitting `k` adds blocks without adding that
        // redundancy - every weight is still read exactly once, by whichever
        // slice owns its part of the contraction - at the cost of one pass over
        // `ksplit * batch * m * n` floats to sum the slices.
        //
        // Kept to a whole number of staged trips each, and only while a slice
        // still has enough contraction to amortise its staging; below that the
        // split costs more than the idle SMs do.
        let ksplit = if small {
            1
        } else {
            ksplit_for(m, k, n, batch.count)
        };
        // Uninitialised, not zeroed. Every slice assigns every element of the
        // tile it owns - a slice with no contraction left assigns the zero it
        // started with - so the memset was one pass over `ksplit * m * n`
        // floats that nothing ever read.
        //
        // SAFETY: the same coverage argument as `out` above, once per slice.
        let mut partial = if ksplit > 1 {
            Some(unsafe { self.uninit(ksplit * batch.count * m * n) }?)
        } else {
            None
        };
        let ks = ksplit as i32;

        // Round an f32 activation once rather than on every trip.
        //
        // The tiled kernel converts both operands to f16 going into shared
        // memory anyway, and re-reads the activation once per column tile - a
        // 14336-wide projection reads it 112 times. Converting it up front is
        // one pass in exchange for halving all of those, and `pack_f16` uses
        // the same `cvt.rn.f16.f32` the staging does, so the arithmetic is not
        // approximated differently - it is the same bits.
        //
        // Only for the tiled path: `gemv` accumulates an f32 operand exactly,
        // and rounding its activation would be a precision change rather than
        // a layout one. Two halves share a 32-bit word, so an odd `k` has no
        // f16 layout at all - decoding attends over the 1, 2, 3, ... tokens
        // emitted so far and half of those are odd. The half pointer is
        // `sa >> 1` for the same reason, so an odd stride would land every
        // batch after the first half an element out.
        //
        // Only against a *packed* weight, and that restriction is measured
        // rather than principled-sounding. A block-quantized weight is a
        // quarter the bytes of the activation it multiplies, so the activation
        // is most of what the staging reads and halving it is most of the
        // saving. Against an f16 or f32 weight the weight dominates, the
        // conversion is a launch and a pass for a few percent of the traffic,
        // and it measured as a loss: the ASR lost 9.6% of a transcription to
        // it, and `bench-gemm`'s f32 encoder shapes ran at 19.3 TFLOP/s with
        // it against 20.8 without.
        //
        // And only over what the kernel reads. `v` can be a slice of a larger
        // allocation, and converting all of it was the other half of that cost.
        let want = (batch.count - 1) * batch.a + m * k;
        let widened = match a {
            Operand::F32(v) | Operand::F32Q { data: v, .. }
                if !small
                    && !use_i8
                    && w.quant().is_some()
                    && k.is_multiple_of(2)
                    && batch.a.is_multiple_of(2) =>
            {
                Some(self.to_f16(v, want.min(v.len()))?)
            }
            _ => None,
        };
        let a_half = if widened.is_some() { 1 } else { a_half };

        if use_i8 {
            let q = q8.expect("`use_i8` implies the activation was quantized");
            let off = q.scale_offset() as i32;
            // Which row tile. A block computes its tile's rows whether `m` has
            // them or not, so the narrow one is right exactly when the wide one
            // would be computing a majority of nothing - which is every prefill
            // this pipeline runs, because a clause is twenty-odd tokens.
            let narrow = m <= kernels::GEMM_I8_MT_NARROW as usize;
            let mt = if narrow {
                kernels::GEMM_I8_MT_NARROW
            } else {
                kernels::GEMM_I8_MT
            };
            // One kernel per block format: the staging differs entirely and
            // compiling both into one entry point cost registers on both.
            let name = match w.quant() {
                // The narrow row tile where the wide one would spend most of
                // its arithmetic on rows the prompt does not have: a block
                // computes `GEMM_I8_MT` rows either way, so a 24-token prefill
                // against 128 of them is five sixths padding. See the note
                // beside `GEMM_I8_ENTRY`.
                Some(Quant::Q6K) if narrow => "gemm_i8_q6k_narrow",
                Some(Quant::Q6K) => "gemm_i8_q6k",
                _ if narrow => "gemm_i8_q4k_narrow",
                _ => "gemm_i8_q4k",
            };
            let mut lb = self.stream.launch_builder(self.func(name));
            lb.arg(&q.buf).arg(&off);
            let wq_view;
            match w {
                Operand::Q { data, ty } if w_first > 0 => {
                    wq_view = data.slice(w_skip / ty.block_size() * ty.device_stride()..);
                    lb.arg(&wq_view)
                }
                Operand::Q { data, .. } => lb.arg(data),
                // Refused by `use_i8`, but the arm has to exist.
                Operand::F32(v) | Operand::F32Q { data: v, .. } => lb.arg(v),
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
                .arg(&sw)
                .arg(&so)
                .arg(&q_ts)
                .arg(&a_rows)
                .arg(&ks);
            match &mut partial {
                Some(p) => lb.arg(p),
                None => lb.arg(&null),
            };
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (
                    (m as u32).div_ceil(mt),
                    (n as u32).div_ceil(kernels::GEMM_I8_NT),
                    (batch.count * ksplit) as u32,
                ),
                block_dim: (32, kernels::GEMM_I8_WARPS, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the grid covers every (batch, m, n) exactly once, `out`
            // is batch*m*n elements, and every global read and write inside
            // the kernel is bounds checked against m, k and n.
            launched(name, unsafe { lb.launch(cfg) })?;
            if let Some(p) = &partial {
                self.reduce_partials(p, bias, &mut out, batch.count, m, n, ksplit)?;
            }
            return Ok(out);
        }

        // Several rows against an unpacked weight read that weight once. Only
        // attention arrives here - the KV cache is the one f32 "weight" in the
        // model, and its rows are a grouped-query group - and only above one
        // row, where there is something to share. See `gemv_rows`.
        if small
            && m > 1
            && w_first == 0
            && matches!(w, Operand::F32(_) | Operand::F16(_))
            && matches!(a, Operand::F32(_) | Operand::F32Q { .. })
        {
            debug_assert!(m <= kernels::GEMV_ROWS_MAX as usize, "gemv_rows is bounded");
            let (Operand::F32(av) | Operand::F32Q { data: av, .. }) = a else {
                unreachable!("matched above")
            };
            let f = self.func("gemv_rows");
            let mut lb = self.stream.launch_builder(f);
            lb.arg(av);
            match w {
                Operand::F32(v) => lb.arg(v),
                Operand::F16(v) => lb.arg(v),
                _ => unreachable!("matched above"),
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
                .arg(&w_rs)
                .arg(&w_half);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n.div_ceil(8) as u32, 1, batch.count as u32),
                block_dim: (32, kernels::GEMV_WARPS, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the grid covers every (batch, n) once and the kernel
            // walks rows 0..m inside, `out` is batch*m*n elements, and every
            // read is bounds checked against k and n.
            launched("gemv_rows", unsafe { lb.launch(cfg) })?;
            return Ok(out);
        }

        // Several rows against a *packed* weight share its stream too. This
        // is what several sequences decoding together come to: each row is
        // one token of one sequence, the weight is the whole traffic, and
        // `gemv` at `blockIdx.y` would stream it once a row. See
        // `gemv_q_rows`, whose rows are `gemv`'s bit for bit.
        if small && m > 1 && wide {
            debug_assert!(m <= kernels::GEMV_Q_ROWS as usize, "gemv_q_rows is bounded");
            let q = q8.expect("`wide` implies the activation was quantized");
            let off = q.scale_offset() as i32;
            let name = match m {
                2 => "gemv_q_rows2",
                3 => "gemv_q_rows3",
                _ => "gemv_q_rows4",
            };
            let mut lb = self.stream.launch_builder(self.func(name));
            let wq_view;
            match w {
                Operand::Q { data, ty } if w_first > 0 => {
                    wq_view = data.slice(w_skip / ty.block_size() * ty.device_stride()..);
                    lb.arg(&wq_view)
                }
                Operand::Q { data, .. } => lb.arg(data),
                _ => unreachable!("`wide` implies a packed weight"),
            };
            lb.arg(&w_quant)
                .arg(&q_ts)
                .arg(&q.buf)
                .arg(&off)
                .arg(&a_rows);
            match bias {
                Some(v) => lb.arg(v),
                None => lb.arg(&null),
            };
            lb.arg(&mut out)
                .arg(&mi)
                .arg(&ki)
                .arg(&ni)
                .arg(&sw)
                .arg(&so);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n.div_ceil(8) as u32, 1, batch.count as u32),
                block_dim: (32, kernels::GEMV_WARPS, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the grid covers every (batch, n) once and the kernel
            // writes rows 0..m of each column; `out` is batch*m*n elements,
            // the codes were taken at `(q_rows, k)` above, and the weight's
            // blocks are bounds checked against n and k.
            launched(name, unsafe { lb.launch(cfg) })?;
            return Ok(out);
        }

        let f = self.func(if small { "gemv" } else { "gemm" });
        let mut lb = self.stream.launch_builder(f);
        match (&widened, a) {
            (Some(h), _) => lb.arg(h),
            (None, Operand::F32(v) | Operand::F32Q { data: v, .. }) => lb.arg(v),
            (None, Operand::F16(v)) => lb.arg(v),
            // Refused above, but the arm has to exist. Passing the pointer
            // keeps this a rejected input rather than an unreachable panic.
            (None, Operand::Q { data, .. }) => lb.arg(data),
        };
        let (wf_view, wh_view, wq_view);
        match w {
            Operand::F32(v) | Operand::F32Q { data: v, .. } if w_first > 0 => {
                wf_view = v.slice(w_skip..);
                lb.arg(&wf_view)
            }
            Operand::F16(v) if w_first > 0 => {
                wh_view = v.slice(w_skip..);
                lb.arg(&wh_view)
            }
            Operand::Q { data, ty } if w_first > 0 => {
                wq_view = data.slice(w_skip / ty.block_size() * ty.device_stride()..);
                lb.arg(&wq_view)
            }
            Operand::F32(v) | Operand::F32Q { data: v, .. } => lb.arg(v),
            Operand::F16(v) => lb.arg(v),
            Operand::Q { data, .. } => lb.arg(data),
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
            .arg(&w_half)
            .arg(&w_quant)
            .arg(&q_bs)
            .arg(&q_ts)
            .arg(&w_rs);
        if !small {
            match &mut partial {
                Some(p) => lb.arg(&ks).arg(p),
                None => lb.arg(&ks).arg(&null),
            };
        }
        let asc_off = q8.map_or(0, |q| q.scale_offset() as i32);
        // The plain epilogue: no activation, a fresh `[batch, m, n]` output.
        let (epi_act, o_cs, o_hs, o_hd, o_off) = (0i32, 1i32, 0i32, 0i32, 0i64);
        if small {
            match q8 {
                Some(q) => lb.arg(&q.buf).arg(&asc_off).arg(&a_rows),
                None => lb.arg(&null).arg(&asc_off).arg(&a_rows),
            };
            lb.arg(&epi_act)
                .arg(&o_cs)
                .arg(&o_hs)
                .arg(&o_hd)
                .arg(&o_off);
        }

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: if small {
                (n.div_ceil(8) as u32, m as u32, batch.count as u32)
            } else {
                (
                    (n as u32).div_ceil(kernels::GEMM_NT),
                    (m as u32).div_ceil(kernels::GEMM_MT),
                    (batch.count * ksplit) as u32,
                )
            },
            block_dim: (32, kernels::GEMM_WARPS, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every (batch, m, n) exactly once, `out` is
        // batch*m*n elements, and every global read and write inside the kernel
        // is bounds checked against m, k and n.
        launched(if small { "gemv" } else { "gemm" }, unsafe {
            lb.launch(cfg)
        })?;

        if let Some(p) = &partial {
            self.reduce_partials(p, bias, &mut out, batch.count, m, n, ksplit)?;
        }
        Ok(out)
    }

    /// Sums a split contraction's slices and adds the bias.
    ///
    /// Shared by both tiled kernels: they disagree about everything else and
    /// agree about what a partial looks like.
    #[allow(clippy::too_many_arguments)]
    fn reduce_partials(
        &self,
        partial: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        m: usize,
        n: usize,
        ksplit: usize,
    ) -> Result<(), CudaError> {
        let null: u64 = 0;
        let total = batch * m * n;
        let (mn, ni, bi, ks) = ((m * n) as i32, n as i32, batch as i32, ksplit as i32);
        let mut lb = self.stream.launch_builder(self.func("gemm_reduce"));
        lb.arg(partial);
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(out).arg(&mn).arg(&ni).arg(&bi).arg(&ks);
        // SAFETY: one thread per output element, bounds checked in the kernel,
        // and `partial` holds exactly `ksplit * batch * m * n`.
        launched("gemm_reduce", unsafe {
            lb.launch(cudarc::driver::LaunchConfig::for_num_elems(total as u32))
        })?;
        Ok(())
    }

    /// [`Self::cache_append`] into an f16 cache.
    ///
    /// A twin rather than a type parameter: the two differ only in the width
    /// they store and in nothing a caller has to reason about, and one of them
    /// has to name a kernel either way.
    ///
    /// `cap` must be even. Everything that reads this cache reads it as pairs
    /// of halves in a word - the value layout puts a head's row `cap` apart,
    /// and an odd `cap` would put every other row's pairs across a word
    /// boundary. It is a constructor's job to guarantee, not a reader's to
    /// survive, so it is refused here.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_append_f16(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        dst: &mut CudaSlice<u16>,
        n: usize,
        kv_heads: usize,
        head_dim: usize,
        cap: usize,
        past: usize,
        transposed: bool,
    ) -> Result<(), CudaError> {
        if past + n > cap {
            return Err(CudaError::CacheOverrun { at: past + n, cap });
        }
        if !cap.is_multiple_of(2) {
            return Err(CudaError::OddCacheCapacity { cap });
        }
        let total = n * kv_heads * head_dim;
        if src_off + total > src.len() {
            return Err(CudaError::SliceOverrun {
                at: src_off + total,
                len: src.len(),
            });
        }
        let name = if transposed {
            "cache_append_t_f16"
        } else {
            "cache_append_f16"
        };
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        let (ni, kh, hd, ca, pa) = (
            n as i32,
            kv_heads as i32,
            head_dim as i32,
            cap as i32,
            past as i32,
        );
        let so = src_off as i64;
        lb.arg(src)
            .arg(&so)
            .arg(dst)
            .arg(&ni)
            .arg(&kh)
            .arg(&hd)
            .arg(&ca)
            .arg(&pa);
        // SAFETY: as `cache_append`, with a half-width destination whose
        // element count is the same.
        launched(name, unsafe { lb.launch(Self::flat(total)) })?;
        Ok(())
    }

    /// The rotation and both cache writes for one decoded position, in one
    /// launch: `q` rotated in place, `k` rotated and stored at `pos` of the
    /// key cache, `v` stored at `pos` of the transposed value cache. Exactly
    /// what `rope_scaled` twice and `cache_append_f16` twice would do at
    /// `t = 1`, bit for bit; see `rope_cache_f16` in the kernels.
    ///
    /// The three projections are named as `(buffer index, offset)` into
    /// `proj`, the outputs of the batched attention products, because they
    /// may share an allocation - the translator's do - and a `&mut` to one
    /// and a `&` to another would then be the same slice. Every range is
    /// checked against its buffer before the launch.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_cache_f16(
        &self,
        proj: &mut [CudaSlice<f32>],
        q: (usize, usize),
        k: (usize, usize),
        v: (usize, usize),
        freq_div: Option<&CudaSlice<f32>>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        pos: usize,
        kc: &mut CudaSlice<u16>,
        vc: &mut CudaSlice<u16>,
        cap: usize,
    ) -> Result<(), CudaError> {
        if pos >= cap {
            return Err(CudaError::CacheOverrun { at: pos + 1, cap });
        }
        if !cap.is_multiple_of(2) {
            return Err(CudaError::OddCacheCapacity { cap });
        }
        if !head_dim.is_multiple_of(2) {
            return Err(CudaError::OddHeadDim { head_dim });
        }
        let spans = [
            (q, heads * head_dim),
            (k, kv_heads * head_dim),
            (v, kv_heads * head_dim),
        ];
        for ((bi, off), len) in spans {
            let have = proj.get(bi).map_or(0, |b| b.len());
            if off + len > have {
                return Err(CudaError::SliceOverrun {
                    at: off + len,
                    len: have,
                });
            }
        }
        let want = cap * kv_heads * head_dim;
        if kc.len() < want || vc.len() < want {
            return Err(CudaError::SliceOverrun {
                at: want,
                len: kc.len().min(vc.len()),
            });
        }
        let dummy = &self.dummy;
        let has = i32::from(freq_div.is_some());
        let div = freq_div.unwrap_or(dummy);
        let (qo, ko, vo) = (q.1 as i64, k.1 as i64, v.1 as i64);
        let (hi, kh, hd, ps, ca) = (
            heads as i32,
            kv_heads as i32,
            head_dim as i32,
            pos as i32,
            cap as i32,
        );
        let f = self.func("rope_cache_f16");
        let mut lb = self.stream.launch_builder(f);
        // The query is written through a shared borrow, because the key may
        // be the same allocation and Rust has no way to say "disjoint ranges
        // of one device buffer". The kernel touches only the ranges checked
        // above, and `proj` is held `&mut` here so nothing else reads or
        // writes any of them until this returns.
        lb.arg(&proj[q.0])
            .arg(&qo)
            .arg(&proj[k.0])
            .arg(&ko)
            .arg(&proj[v.0])
            .arg(&vo)
            .arg(div)
            .arg(&has)
            .arg(&hi)
            .arg(&kh)
            .arg(&hd)
            .arg(&theta)
            .arg(&ps)
            .arg(kc)
            .arg(vc)
            .arg(&ca);
        let total = heads * head_dim / 2 + kv_heads * head_dim / 2 + kv_heads * head_dim;
        // SAFETY: every read and write is bounds checked above against the
        // buffers it lands in, and the grid covers `total` threads that each
        // touch one pair or one element and nothing past `total`.
        launched("rope_cache_f16", unsafe { lb.launch(Self::flat(total)) })
    }

    /// Writes one step's keys or values into a head-major cache.
    ///
    /// `src` is `[n, kv_heads * head_dim]` as the projection produced it. With
    /// `transposed`, the destination is `[kv_heads, head_dim, cap]` - the shape
    /// the context matmul contracts over; without, it is
    /// `[kv_heads, cap, head_dim]`, the shape the score matmul reads.
    ///
    /// Refuses a write that would run off the end, because the destination is
    /// one long buffer and a `past` past its capacity would land in the next
    /// head's keys: in bounds, and wrong for every step after it.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_append(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        dst: &mut CudaSlice<f32>,
        n: usize,
        kv_heads: usize,
        head_dim: usize,
        cap: usize,
        past: usize,
        transposed: bool,
    ) -> Result<(), CudaError> {
        if past + n > cap {
            return Err(CudaError::CacheOverrun { at: past + n, cap });
        }
        let total = n * kv_heads * head_dim;
        // The source may be one product of a batched projection rather than a
        // whole allocation, so the read runs from `src_off` and has to fit.
        if src_off + total > src.len() {
            return Err(CudaError::SliceOverrun {
                at: src_off + total,
                len: src.len(),
            });
        }
        let name = if transposed {
            "cache_append_t"
        } else {
            "cache_append"
        };
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        let (ni, kh, hd, ca, pa) = (
            n as i32,
            kv_heads as i32,
            head_dim as i32,
            cap as i32,
            past as i32,
        );
        let so = src_off as i64;
        lb.arg(src)
            .arg(&so)
            .arg(dst)
            .arg(&ni)
            .arg(&kh)
            .arg(&hd)
            .arg(&ca)
            .arg(&pa);
        // SAFETY: one thread per source element, bounds checked in the kernel
        // against `n * kv_heads * head_dim` from `src_off`, the destination
        // range is checked above, and the caller's `src_off` is checked here.
        launched(name, unsafe { lb.launch(Self::flat(total)) })?;
        Ok(())
    }

    /// Moves the `live` tokens of a head-major cache into one of a larger
    /// capacity.
    ///
    /// The companion of [`Gpu::cache_append`], and it exists because `cap` is a
    /// **stride** in both of that method's layouts rather than only a length. A
    /// head's data begins at a multiple of the capacity, so raising the
    /// capacity moves every head but the first, and copying the live prefix
    /// flat - the whole cache is one buffer, so it is tempting - lands heads 1
    /// upward inside their own earlier positions.
    ///
    /// That failure is silent in every way a failure can be: the buffer is the
    /// right length, every read stays in bounds, and attention keeps producing
    /// fluent text off the one head that did not move. Hence a kernel that is
    /// told the layout instead of a copy that assumes one.
    ///
    /// `transposed` means the same thing it does in `cache_append`: the values'
    /// `[kv_heads, head_dim, cap]` rather than the keys' `[kv_heads, cap,
    /// head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_grow(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        kv_heads: usize,
        head_dim: usize,
        old_cap: usize,
        new_cap: usize,
        live: usize,
        transposed: bool,
    ) -> Result<(), CudaError> {
        if live > old_cap || new_cap < old_cap {
            return Err(CudaError::CacheOverrun {
                at: live.max(old_cap),
                cap: new_cap.min(old_cap),
            });
        }
        // Both layouts are `rows` runs of `len` contiguous floats, a source
        // capacity apart and a destination capacity apart. Only what a run is
        // differs: a whole head's positions for the keys, one head's single
        // dimension across positions for the values.
        let (rows, len, src_stride, dst_stride) = match transposed {
            false => (
                kv_heads,
                live * head_dim,
                old_cap * head_dim,
                new_cap * head_dim,
            ),
            true => (kv_heads * head_dim, live, old_cap, new_cap),
        };
        if rows * src_stride > src.len() {
            return Err(CudaError::SliceOverrun {
                at: rows * src_stride,
                len: src.len(),
            });
        }
        if rows * dst_stride > dst.len() {
            return Err(CudaError::SliceOverrun {
                at: rows * dst_stride,
                len: dst.len(),
            });
        }
        let total = rows * len;
        if total == 0 {
            return Ok(());
        }
        let (r, l, ss, ds) = (
            rows as i32,
            len as i32,
            src_stride as i32,
            dst_stride as i32,
        );
        let f = self.func("cache_grow");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(src).arg(dst).arg(&r).arg(&l).arg(&ss).arg(&ds);
        // SAFETY: one thread per copied element, bounds checked in the kernel
        // against `rows * len`, and both allocations are checked above to hold
        // `rows` runs at their own stride.
        launched("cache_grow", unsafe { lb.launch(Self::flat(total)) })?;
        Ok(())
    }

    /// [`Self::cache_grow`] for an f16 cache. Same runs, same strides,
    /// half the width - and the same reason it is a kernel and not a
    /// memcpy, which the note on that one gives.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_grow_f16(
        &self,
        src: &CudaSlice<u16>,
        dst: &mut CudaSlice<u16>,
        kv_heads: usize,
        head_dim: usize,
        old_cap: usize,
        new_cap: usize,
        live: usize,
        transposed: bool,
    ) -> Result<(), CudaError> {
        if live > old_cap || new_cap < old_cap {
            return Err(CudaError::CacheOverrun {
                at: live.max(old_cap),
                cap: new_cap.min(old_cap),
            });
        }
        // Both layouts are `rows` runs of `len` contiguous halves, a source
        // capacity apart and a destination capacity apart. Only what a run is
        // differs: a whole head's positions for the keys, one head's single
        // dimension across positions for the values.
        let (rows, len, src_stride, dst_stride) = match transposed {
            false => (
                kv_heads,
                live * head_dim,
                old_cap * head_dim,
                new_cap * head_dim,
            ),
            true => (kv_heads * head_dim, live, old_cap, new_cap),
        };
        if rows * src_stride > src.len() {
            return Err(CudaError::SliceOverrun {
                at: rows * src_stride,
                len: src.len(),
            });
        }
        if rows * dst_stride > dst.len() {
            return Err(CudaError::SliceOverrun {
                at: rows * dst_stride,
                len: dst.len(),
            });
        }
        let total = rows * len;
        if total == 0 {
            return Ok(());
        }
        let (r, l, ss, ds) = (
            rows as i32,
            len as i32,
            src_stride as i32,
            dst_stride as i32,
        );
        let f = self.func("cache_grow_f16");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(src).arg(dst).arg(&r).arg(&l).arg(&ss).arg(&ds);
        // SAFETY: one thread per copied element, bounds checked in the kernel
        // against `rows * len`, and both allocations are checked above to hold
        // `rows` runs at their own stride.
        launched("cache_grow_f16", unsafe { lb.launch(Self::flat(total)) })?;
        Ok(())
    }

    /// Layer normalisation over each row. Mirrors `xabe_dsp::layer_norm`.
    pub fn layer_norm(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // SAFETY: one block a row writes every element of its row.
        let mut out = unsafe { self.uninit(rows * cols) }?;
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

    /// A DiT block's adaptive normalisation: `layer_norm` with no affine,
    /// then `* (1 + scale) + shift`, with `scale` and `shift` read as
    /// `cols`-wide segments of `mods` at `scale_off` and `shift_off`. What
    /// `layer_norm` with weight `1 + scale` and bias `shift` computes, in one
    /// launch and without the two vectors being built; the test holds it to
    /// `xabe_dsp::layer_norm` on exactly that weight and bias.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_mod(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        mods: &CudaSlice<f32>,
        shift_off: usize,
        scale_off: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let need = shift_off.max(scale_off) + cols;
        if mods.len() < need {
            return Err(CudaError::SliceOverrun {
                at: need,
                len: mods.len(),
            });
        }
        if x.len() < rows * cols {
            return Err(CudaError::SliceOverrun {
                at: rows * cols,
                len: x.len(),
            });
        }
        if !shift_off.is_multiple_of(4) || !scale_off.is_multiple_of(4) {
            return Err(CudaError::Misaligned {
                what: "a modulation segment",
                offset: if shift_off.is_multiple_of(4) {
                    scale_off
                } else {
                    shift_off
                },
                align: 4,
            });
        }
        // SAFETY: one block a row writes every element of its row.
        let mut out = unsafe { self.uninit(rows * cols) }?;
        let (c, so, sc) = (cols as i32, shift_off as i32, scale_off as i32);
        let f = self.func("layer_norm_mod");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(mods)
            .arg(&so)
            .arg(&sc)
            .arg(&mut out)
            .arg(&c)
            .arg(&eps);
        // SAFETY: both segments were checked to lie inside `mods` above.
        launched("layer_norm_mod", unsafe { lb.launch(Self::per_row(rows)) })?;
        Ok(out)
    }

    /// The gated residual of a DiT block: `h[p, c] += mods[gate_off + c] * x[p, c]`
    /// over `rows` rows of `cols`.
    pub fn gate_add(
        &self,
        h: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        mods: &CudaSlice<f32>,
        gate_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), CudaError> {
        let n = rows * cols;
        if h.len() < n || x.len() < n {
            return Err(CudaError::SliceOverrun {
                at: n,
                len: h.len().min(x.len()),
            });
        }
        if mods.len() < gate_off + cols {
            return Err(CudaError::SliceOverrun {
                at: gate_off + cols,
                len: mods.len(),
            });
        }
        let (go, c, ni) = (gate_off as i32, cols as i32, n as i32);
        let f = self.func("gate_add");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(h).arg(x).arg(mods).arg(&go).arg(&c).arg(&ni);
        // SAFETY: every index is bounded by `n` and the gate segment by the
        // check above.
        launched("gate_add", unsafe { lb.launch(Self::flat(n)) })
    }

    /// The residual sum and the normalisation of it, in one pass.
    ///
    /// `h` becomes `h + res` - which is what the next sub-layer adds to, so it
    /// has to survive - and the return is the normalisation of that. Mirrors
    /// `xabe_dsp::layer_norm_add`.
    ///
    /// Every normalisation in a transformer block reads the residual stream
    /// just after something was added to it, and nothing between the two reads
    /// it. As two kernels that is five passes and two launches where this is
    /// four and one: on the encoder's 1500 rows the passes are what costs, and
    /// on a single decode step the row is five kilobytes and the launch is.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_add(
        &self,
        h: &mut CudaSlice<f32>,
        res: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // SAFETY: one block a row writes every element of its row.
        let mut out = unsafe { self.uninit(rows * cols) }?;
        let c = cols as i32;
        let f = self.func("layer_norm_add");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(h)
            .arg(res)
            .arg(weight)
            .arg(bias)
            .arg(&mut out)
            .arg(&c)
            .arg(&eps);
        launched("layer_norm_add", unsafe { lb.launch(Self::per_row(rows)) })?;
        Ok(out)
    }

    /// [`Self::layer_norm_add`], returning the normalisation at f16.
    ///
    /// For the case where the result is a matmul's left operand and nothing
    /// else, which is every normalisation in the Whisper encoder. The matmul
    /// stages its left operand as f16 regardless, and `f32_to_f16` is the
    /// rounding it applies, so this returns the same bits the f32 path would
    /// have produced and halves the stream the matmul re-reads once per column
    /// tile. `h` is still f32: it is the residual stream.
    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm_add_f16(
        &self,
        h: &mut CudaSlice<f32>,
        res: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<u16>, CudaError> {
        // Every element is written; see `reshape_heads`.
        let mut out = unsafe { self.stream.alloc::<u16>(rows * cols) }.map_err(|source| {
            CudaError::Driver {
                what: "allocating",
                source,
            }
        })?;
        let c = cols as i32;
        let f = self.func("layer_norm_add_f16");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(h)
            .arg(res)
            .arg(weight)
            .arg(bias)
            .arg(&mut out)
            .arg(&c)
            .arg(&eps);
        launched("layer_norm_add_f16", unsafe {
            lb.launch(Self::per_row(rows))
        })?;
        Ok(out)
    }

    /// GELU reading f32 and writing f16 elsewhere, for a matmul's left operand.
    ///
    /// The out-of-place twin of [`Self::gelu`], and bit-identical to calling
    /// that and letting the matmul stage the result. It exists for the
    /// encoder's MLP, where the inner activation is 30.7 MB that the second
    /// projection re-reads once per column tile.
    pub fn gelu_f16(&self, x: &CudaSlice<f32>, n: usize) -> Result<CudaSlice<u16>, CudaError> {
        let mut out =
            unsafe { self.stream.alloc::<u16>(n) }.map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })?;
        let len = n as i32;
        let f = self.func("act_gelu_f16");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&len);
        launched("act_gelu_f16", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(out)
    }

    /// Softmax over each row, in place. Mirrors `xabe_dsp::softmax_rows`.
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

    /// Softmax over each row, scaling and causal-masking on the way.
    ///
    /// `tq` rows to a query block and `offset` the position the first of them
    /// sits at, which is `tk - tq` for the usual append-only attention. Replaces
    /// a scale, a mask and a softmax with one pass - see the kernel.
    pub fn softmax_causal(
        &self,
        x: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        tq: usize,
        offset: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        let (c, q, o) = (cols as i32, tq as i32, offset as i32);
        let f = self.func("softmax_causal");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&c).arg(&q).arg(&o).arg(&scale);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            // A multiple of 32, so the shuffle reductions have full warps.
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launched("softmax_causal", unsafe { lb.launch(cfg) })
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

    /// Leaky ReLU, in place. Mirrors `xabe_dsp::leaky_relu`.
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

    /// Exact GELU, in place. Mirrors `xabe_dsp::gelu`.
    ///
    /// The device has an IEEE-accurate `erff`, so this needs none of the
    /// rational approximation the CPU twin carries.
    pub fn gelu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_gelu", x, n, None)
    }

    /// WaveNet's gated activation. Mirrors `xabe_dsp::gated_activation`.
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

    /// WaveNet's gated activation for `[t, 2 * ch]` data.
    ///
    /// The row-major twin of [`Gpu::gated_activation`]: same arithmetic, the
    /// other layout. Both exist because the two callers keep their data
    /// differently, and a transpose to share one kernel costs more than the
    /// kernel does.
    pub fn gated_activation_rows(
        &self,
        x: &CudaSlice<f32>,
        ch: usize,
        t: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(ch * t)?;
        let (a, b_) = (ch as i32, t as i32);
        let f = self.func("gated_activation_rows");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_);
        launched("gated_activation_rows", unsafe {
            lb.launch(Self::flat(ch * t))
        })?;
        Ok(out)
    }

    /// One LSTM step. Mirrors `xabe_dsp::lstm_gates`.
    ///
    /// `gi` and `gh` are the input-side and hidden-side pre-activations, each
    /// `[4 * hidden]` in PyTorch's input/forget/cell/output order. `c` is
    /// updated in place and `h` written; both are `[hidden]`.
    ///
    /// Split that way because the input side does not depend on the recurrence:
    /// a bidirectional encoder can project its whole sequence with one
    /// [`Gpu::linear`] and then loop over nothing but this.
    pub fn lstm_gates(
        &self,
        gi: &CudaSlice<f32>,
        gh: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        h: &mut CudaSlice<f32>,
        hidden: usize,
    ) -> Result<(), CudaError> {
        let n = hidden as i32;
        let f = self.func("lstm_gates");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(gi).arg(gh).arg(c).arg(h).arg(&n);
        launched("lstm_gates", unsafe { lb.launch(Self::flat(hidden)) })
    }

    /// WaveGlow's affine coupling, inverted. Mirrors `xabe_dsp::coupling_inverse`.
    ///
    /// `x` is `[half, t]`, `st` is `[2 * half, t]` holding the shift then the
    /// log scale, and the result is `(x - b) / exp(s)`.
    pub fn coupling_inverse(
        &self,
        x: &CudaSlice<f32>,
        st: &CudaSlice<f32>,
        half: usize,
        t: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let mut out = self.zeros(half * t)?;
        let (a, b_) = (half as i32, t as i32);
        let f = self.func("coupling_inverse");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(st).arg(&mut out).arg(&a).arg(&b_);
        launched("coupling_inverse", unsafe {
            lb.launch(Self::flat(half * t))
        })?;
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

    /// `a[r * cols + j] += b[r * stride + off + j]`.
    ///
    /// The strided twin of [`Gpu::add_inplace`]. WaveGlow's conditioning is one
    /// `[steps, 2 * ch * layers]` product rather than one matmul per layer,
    /// which measured 1.30x on the shape, and a layer's share of it is a column
    /// range of every row instead of a contiguous run.
    pub fn add_strided(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        cols: usize,
        stride: usize,
        off: usize,
        rows: usize,
    ) -> Result<(), CudaError> {
        let (c, s, o, r) = (cols as i32, stride as i32, off as i32, rows as i32);
        let f = self.func("add_strided");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b).arg(&c).arg(&s).arg(&o).arg(&r);
        launched("add_strided", unsafe { lb.launch(Self::flat(cols * rows)) })
    }

    /// Element-wise `a *= b`.
    ///
    /// Sits next to the add and the subtract because Tacotron2's prenet keeps
    /// its dropout on at inference - the mask is a vector, not a scalar, so
    /// `scale_inplace` cannot express it.
    pub fn mul_inplace(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let len = n as i32;
        let f = self.func("mul_inplace");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b).arg(&len);
        launched("mul_inplace", unsafe { lb.launch(Self::flat(n)) })
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

    /// `dst[doff..doff + n] = src[soff..soff + n]`, both ranges checked.
    pub fn copy_from_into(
        &self,
        dst: &mut CudaSlice<f32>,
        doff: usize,
        src: &CudaSlice<f32>,
        soff: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        if dst.len() < doff + n || src.len() < soff + n {
            return Err(CudaError::SliceOverrun {
                at: (doff + n).max(soff + n),
                len: dst.len().min(src.len()),
            });
        }
        let (d, s, ni) = (doff as i32, soff as i32, n as i32);
        let f = self.func("copy_from_into");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(dst).arg(&d).arg(src).arg(&s).arg(&ni);
        // SAFETY: both ranges are checked above and the grid covers `n`.
        launched("copy_from_into", unsafe { lb.launch(Self::flat(n)) })
    }

    /// `dst[0..na] = a[0..na]` and `dst[na..na + nb] = b[0..nb]` in one
    /// launch; see `concat2` in the kernels.
    pub fn concat2(
        &self,
        dst: &mut CudaSlice<f32>,
        a: &CudaSlice<f32>,
        na: usize,
        b: &CudaSlice<f32>,
        nb: usize,
    ) -> Result<(), CudaError> {
        if dst.len() < na + nb || a.len() < na || b.len() < nb {
            return Err(CudaError::SliceOverrun {
                at: na + nb,
                len: dst.len().min(a.len() + nb).min(b.len() + na),
            });
        }
        let (nai, nbi) = (na as i32, nb as i32);
        let f = self.func("concat2");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(dst).arg(a).arg(&nai).arg(b).arg(&nbi);
        // SAFETY: every range is checked above and the grid covers `na + nb`.
        launched("concat2", unsafe { lb.launch(Self::flat(na + nb)) })
    }

    /// `y = relu(y) * mask[off..off + n]`, the prenet's dropout in one pass;
    /// exactly `relu` then `mul_inplace`. See `relu_mask` in the kernels.
    pub fn relu_mask(
        &self,
        y: &mut CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        off: usize,
        n: usize,
    ) -> Result<(), CudaError> {
        if y.len() < n || mask.len() < off + n {
            return Err(CudaError::SliceOverrun {
                at: off + n,
                len: mask.len().min(y.len() + off),
            });
        }
        let (o, ni) = (off as i32, n as i32);
        let f = self.func("relu_mask");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(y).arg(mask).arg(&o).arg(&ni);
        // SAFETY: both ranges are checked above and the grid covers `n`.
        launched("relu_mask", unsafe { lb.launch(Self::flat(n)) })
    }

    /// The location attention's `[alignment; cumulative]` of `t` each,
    /// updated with a new alignment: the first half replaced, the second
    /// added to. See `attn_weights_update` in the kernels.
    pub fn attn_weights_update(
        &self,
        cat: &mut CudaSlice<f32>,
        alignment: &CudaSlice<f32>,
        t: usize,
    ) -> Result<(), CudaError> {
        if cat.len() < 2 * t || alignment.len() < t {
            return Err(CudaError::SliceOverrun {
                at: 2 * t,
                len: cat.len().min(2 * alignment.len()),
            });
        }
        let ti = t as i32;
        let f = self.func("attn_weights_update");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(cat).arg(alignment).arg(&ti);
        // SAFETY: `cat` holds `2 t` and `alignment` `t`, checked above.
        launched("attn_weights_update", unsafe { lb.launch(Self::flat(t)) })
    }

    /// Tacotron2's location-sensitive attention for one decoder step, in two
    /// launches: the energies and scores of the `t` encoder positions from
    /// the location features `loc` (`[f, t]`, as `conv1d` leaves them), the
    /// location dense weight `wl` (`[a, f]`), the `query` (`[a]`) and the
    /// `processed` memory (`[t, a]`) against `v` (`[a]`); then the softmax,
    /// the context (`[e]`, from `memory` `[t, e]`) and the running weights
    /// `cat` (`[2 t]`, the alignment then the cumulative sum). See
    /// `taco_energies` and `taco_context` in the kernels; the test holds
    /// both to a CPU chain of the seven kernels they replace.
    ///
    /// `a` is a thread count and must be a power of two up to 1024; `t` and
    /// `e` are bounded by shared memory and the block, at 4096 and 1024.
    #[allow(clippy::too_many_arguments)]
    pub fn taco_attention(
        &self,
        loc: &CudaSlice<f32>,
        wl: &CudaSlice<f32>,
        query: &CudaSlice<f32>,
        processed: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        memory: &CudaSlice<f32>,
        t: usize,
        f: usize,
        a: usize,
        e: usize,
        cat: &mut CudaSlice<f32>,
        context: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        if a == 0 || !a.is_power_of_two() || a > 1024 || t == 0 || t > 4096 || e > 1024 {
            return Err(CudaError::UnsupportedAttention {
                head_dim: a,
                heads: t,
                kv_heads: e,
            });
        }
        let checks = [
            (loc.len(), f * t),
            (wl.len(), a * f),
            (query.len(), a),
            (processed.len(), t * a),
            (v.len(), a),
            (memory.len(), t * e),
            (cat.len(), 2 * t),
            (context.len(), e),
        ];
        for (len, need) in checks {
            if len < need {
                return Err(CudaError::SliceOverrun { at: need, len });
            }
        }
        // SAFETY: every score is written by its block below.
        let mut score = unsafe { self.uninit(t) }?;
        let (ti, fi, ai, ei) = (t as i32, f as i32, a as i32, e as i32);
        let fun = self.func("taco_energies");
        let mut lb = self.stream.launch_builder(fun);
        lb.arg(loc)
            .arg(wl)
            .arg(query)
            .arg(processed)
            .arg(v)
            .arg(&mut score)
            .arg(&ti)
            .arg(&fi)
            .arg(&ai);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (a as u32, 1, 1),
            shared_mem_bytes: (a * 4) as u32,
        };
        // SAFETY: a block a position and a thread a unit, every read bounded
        // by the lengths checked above.
        launched("taco_energies", unsafe { lb.launch(cfg) })?;

        let n = e.max(t.min(1024)).max(32).next_power_of_two().min(1024);
        let fun = self.func("taco_context");
        let mut lb = self.stream.launch_builder(fun);
        lb.arg(&score)
            .arg(memory)
            .arg(context)
            .arg(cat)
            .arg(&ti)
            .arg(&ei);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n as u32, 1, 1),
            shared_mem_bytes: ((t + n) * 4) as u32,
        };
        // SAFETY: one block; the shared buffer holds `t` alignments and `n`
        // partials, and every global access is bounded by `t` and `e`.
        launched("taco_context", unsafe { lb.launch(cfg) })
    }

    /// Transposes a row-major `[rows, cols]`. Mirrors `xabe_dsp::transpose`.
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

    /// Reverses the channel axis. Mirrors `xabe_dsp::flip_channels`.
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

    /// One row through a mat-vec, with the result placed and finished by the
    /// kernel rather than by two more launches.
    ///
    /// `x` is one row of `k`, `w` an f32 or f16 `[n, k]` weight, and the
    /// output row is written into `out` at the positions `layout` names, with
    /// the exact GELU applied first when `gelu` is set. The arithmetic is
    /// `gemm_batched`'s mat-vec to the bit - same kernel, same order - so a
    /// projection followed by `cache_append` and one followed by `gelu` each
    /// become this and produce the same numbers. What is saved is a launch or
    /// two a layer, which at one decoded row is what a layer costs; see
    /// docs/BENCHMARKS.md.
    ///
    /// Refuses a layout that does not fit `out` - the failure it guards is a
    /// scatter into the next head's positions, in bounds and wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_into(
        &self,
        x: &CudaSlice<f32>,
        w: Operand<'_>,
        bias: Option<&CudaSlice<f32>>,
        k: usize,
        n: usize,
        gelu: bool,
        layout: OutLayout,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let w_half = match w {
            Operand::F32(_) => 0i32,
            Operand::F16(_) => 1i32,
            Operand::F32Q { .. } | Operand::Q { .. } => {
                return Err(CudaError::QuantizedActivation);
            }
        };
        if x.len() < k {
            return Err(CudaError::SliceOverrun {
                at: k,
                len: x.len(),
            });
        }
        let wlen = match w {
            Operand::F32(v) | Operand::F32Q { data: v, .. } => v.len(),
            Operand::F16(v) => v.len(),
            Operand::Q { data, .. } => data.len(),
        };
        if wlen < n * k {
            return Err(CudaError::SliceOverrun {
                at: n * k,
                len: wlen,
            });
        }
        if w_half == 1 && !k.is_multiple_of(2) {
            return Err(CudaError::RaggedContraction { k });
        }
        let (o_cs, o_hs, o_hd, o_off, last) = match layout {
            OutLayout::Row => (1usize, 0usize, 0usize, 0usize, n - 1),
            OutLayout::KeyCache { head_dim, cap, pos } => {
                if head_dim == 0 || !n.is_multiple_of(head_dim) || pos >= cap {
                    return Err(CudaError::CacheOverrun { at: pos + 1, cap });
                }
                let heads = n / head_dim;
                (
                    1,
                    (cap - 1) * head_dim,
                    head_dim,
                    pos * head_dim,
                    ((heads - 1) * cap + pos) * head_dim + head_dim - 1,
                )
            }
            OutLayout::ValueCache { cap, pos } => {
                if pos >= cap {
                    return Err(CudaError::CacheOverrun { at: pos + 1, cap });
                }
                (cap, 0, 0, pos, (n - 1) * cap + pos)
            }
        };
        if last >= out.len() {
            return Err(CudaError::SliceOverrun {
                at: last + 1,
                len: out.len(),
            });
        }
        let null: u64 = 0;
        let (mi, ki, ni) = (1i32, k as i32, n as i32);
        let (sa, sw, so) = (0i64, 0i64, 0i64);
        let (a_half, w_quant, q_bs, q_ts, w_rs) = (0i32, 0i32, 0i32, 0i32, 0i32);
        let (asc_off, a_rows) = (0i32, 1i32);
        let epi_act = i32::from(gelu);
        let (cs, hs, hd, off) = (o_cs as i32, o_hs as i32, o_hd as i32, o_off as i64);
        let f = self.func("gemv");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x);
        match w {
            Operand::F32(v) | Operand::F32Q { data: v, .. } => lb.arg(v),
            Operand::F16(v) => lb.arg(v),
            Operand::Q { data, .. } => lb.arg(data),
        };
        match bias {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(out)
            .arg(&mi)
            .arg(&ki)
            .arg(&ni)
            .arg(&sa)
            .arg(&sw)
            .arg(&so)
            .arg(&a_half)
            .arg(&w_half)
            .arg(&w_quant)
            .arg(&q_bs)
            .arg(&q_ts)
            .arg(&w_rs)
            .arg(&null)
            .arg(&asc_off)
            .arg(&a_rows)
            .arg(&epi_act)
            .arg(&cs)
            .arg(&hs)
            .arg(&hd)
            .arg(&off);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(8) as u32, 1, 1),
            block_dim: (32, kernels::GEMV_WARPS, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one warp a column, columns past `n` return; every store
        // lands at or before `last`, which is checked against `out` above.
        launched("gemv", unsafe { lb.launch(cfg) })
    }
    /// The counter the norm-fused mat-vecs arrive on, allocated on first use.
    fn norm_counter<'s>(
        &self,
        scratch: &'s mut NormScratch,
    ) -> Result<&'s mut CudaSlice<u32>, CudaError> {
        if scratch.ctr.is_none() {
            scratch.ctr =
                Some(
                    self.stream
                        .alloc_zeros::<u32>(1)
                        .map_err(|source| CudaError::Driver {
                            what: "allocating",
                            source,
                        })?,
                );
        }
        Ok(scratch.ctr.as_mut().expect("allocated above"))
    }

    /// [`Self::gemv_norm`] for an f16 weight and a layer normalisation: the
    /// product of `a` with `w` plus `bias` is added into `h`, and `h` is then
    /// normalised - mean and variance - with `weight` and `shift` into the
    /// returned row. Exactly `gemv` then `layer_norm_add`, with `h` bit for
    /// bit and the row within an ulp; see `gemv_ln` in the kernels.
    ///
    /// Refuses an odd `k`, an `n` that is not a multiple of four or is longer
    /// than [`GEMV_LN_MAX_N`] - the last block holds the row in registers -
    /// and every operand shorter than its shape.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_ln(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<u16>,
        bias: Option<&CudaSlice<f32>>,
        k: usize,
        n: usize,
        h: &mut CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        shift: &CudaSlice<f32>,
        eps: f32,
        scratch: &mut NormScratch,
    ) -> Result<CudaSlice<f32>, CudaError> {
        if !k.is_multiple_of(2) {
            return Err(CudaError::RaggedContraction { k });
        }
        if !n.is_multiple_of(4) {
            return Err(CudaError::RaggedBlock { k: n, block: 4 });
        }
        if n > GEMV_LN_MAX_N {
            return Err(CudaError::NormFusion {
                what: "a row longer than the last block can hold",
            });
        }
        if a.len() < k {
            return Err(CudaError::SliceOverrun {
                at: k,
                len: a.len(),
            });
        }
        if w.len() < n * k {
            return Err(CudaError::SliceOverrun {
                at: n * k,
                len: w.len(),
            });
        }
        let short = [
            h.len(),
            weight.len(),
            shift.len(),
            bias.map_or(n, |b| b.len()),
        ]
        .into_iter()
        .min()
        .expect("four lengths");
        if short < n {
            return Err(CudaError::SliceOverrun { at: n, len: short });
        }
        let ctr = self.norm_counter(scratch)?;
        // SAFETY: the last block writes every one of the `n` outputs.
        let mut x = unsafe { self.uninit(n) }?;
        let null: u64 = 0;
        let (ki, ni) = (k as i32, n as i32);
        let f = self.func("gemv_ln");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(w);
        match bias {
            Some(b) => lb.arg(b),
            None => lb.arg(&null),
        };
        lb.arg(&ki)
            .arg(&ni)
            .arg(h)
            .arg(weight)
            .arg(shift)
            .arg(&eps)
            .arg(&mut x)
            .arg(ctr);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n.div_ceil(kernels::GEMV_WARPS as usize) as u32, 1, 1),
            block_dim: (32, kernels::GEMV_WARPS, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one warp a column, columns past `n` return before storing;
        // the last block reads and writes `n` elements of `h`, `weight`,
        // `shift` and `x`, all checked above; the counter is one word and is
        // reset by the last block.
        launched("gemv_ln", unsafe { lb.launch(cfg) })?;
        Ok(x)
    }

    /// The three attention projections of one row in one launch, each placed
    /// where the attention reads it: the queries into `q`, the keys into `kc`
    /// at `pos` of a head-major cache laid out for `cap` positions, the values
    /// into `vc` at `pos` of the transposed one. `w` is the three `[d, k]`
    /// weights stacked into `[3 d, k]` and `bias` their three biases, each
    /// optional. What [`Self::gemv_into`] three times would produce, bit for
    /// bit; see `gemv_qkv_f16` in the kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_qkv_f16(
        &self,
        a: &CudaSlice<f32>,
        w: &CudaSlice<u16>,
        bias: [Option<&CudaSlice<f32>>; 3],
        k: usize,
        d: usize,
        head_dim: usize,
        cap: usize,
        pos: usize,
        q: &mut CudaSlice<f32>,
        kc: &mut CudaSlice<f32>,
        vc: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        if !k.is_multiple_of(2) {
            return Err(CudaError::RaggedContraction { k });
        }
        if head_dim == 0 || !d.is_multiple_of(head_dim) {
            return Err(CudaError::RaggedBlock {
                k: d,
                block: head_dim,
            });
        }
        if pos >= cap {
            return Err(CudaError::CacheOverrun { at: pos + 1, cap });
        }
        if a.len() < k {
            return Err(CudaError::SliceOverrun {
                at: k,
                len: a.len(),
            });
        }
        if w.len() < 3 * d * k {
            return Err(CudaError::SliceOverrun {
                at: 3 * d * k,
                len: w.len(),
            });
        }
        if q.len() < d {
            return Err(CudaError::SliceOverrun {
                at: d,
                len: q.len(),
            });
        }
        if kc.len() < d * cap || vc.len() < d * cap {
            return Err(CudaError::SliceOverrun {
                at: d * cap,
                len: kc.len().min(vc.len()),
            });
        }
        for b in bias.into_iter().flatten() {
            if b.len() < d {
                return Err(CudaError::SliceOverrun {
                    at: d,
                    len: b.len(),
                });
            }
        }
        let null: u64 = 0;
        let (ki, di, hd, ca, ps) = (k as i32, d as i32, head_dim as i32, cap as i32, pos as i32);
        let f = self.func("gemv_qkv_f16");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(w);
        for b in bias {
            match b {
                Some(b) => lb.arg(b),
                None => lb.arg(&null),
            };
        }
        lb.arg(&ki)
            .arg(&di)
            .arg(&hd)
            .arg(&ca)
            .arg(&ps)
            .arg(q)
            .arg(kc)
            .arg(vc);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((3 * d).div_ceil(kernels::GEMV_WARPS as usize) as u32, 1, 1),
            block_dim: (32, kernels::GEMV_WARPS, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: one warp a column, columns past `3 d` return; each store
        // lands inside `q`, or at position `pos < cap` of a cache of `d * cap`
        // elements, all checked above.
        launched("gemv_qkv_f16", unsafe { lb.launch(cfg) })
    }

    /// A single-row packed mat-vec whose tail is the residual add and the
    /// next normalisation: `h += w · a`, then `x = rms_norm(h) * weight` and
    /// its int8 twin, in one launch. See `gemv_norm` in the kernels.
    ///
    /// Only the shape the two Llama stages decode with: a K-quant weight
    /// (`Q4_K` or `Q6_K`), the activation already quantized as one row of
    /// `k`, `k` a multiple of 256 and `n` of 32. Anything else is refused
    /// rather than routed to the chain, so a caller that asked for the fusion
    /// and did not get it hears about it.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_norm(
        &self,
        w: Operand<'_>,
        a: &Q8,
        k: usize,
        n: usize,
        h: &mut CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        eps: f32,
        scratch: &mut NormScratch,
    ) -> Result<(CudaSlice<f32>, Q8), CudaError> {
        let Operand::Q { data, ty } = w else {
            return Err(CudaError::NormFusion {
                what: "an unpacked weight",
            });
        };
        if !matches!(ty, Quant::Q4K | Quant::Q6K) {
            return Err(CudaError::NormFusion {
                what: "a block format other than Q4_K or Q6_K",
            });
        }
        if !k.is_multiple_of(256) {
            return Err(CudaError::RaggedBlock { k, block: 256 });
        }
        if !n.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock { k: n, block: 32 });
        }
        if a.shape() != (1, k) {
            return Err(CudaError::NormFusion {
                what: "an activation twin that is not one row of k",
            });
        }
        let nb = k / 256;
        let need = n * nb * ty.device_stride();
        if data.len() < need {
            return Err(CudaError::SliceOverrun {
                at: need,
                len: data.len(),
            });
        }
        if h.len() < n || weight.len() < n {
            return Err(CudaError::SliceOverrun {
                at: n,
                len: h.len().min(weight.len()),
            });
        }
        let blocks = n.div_ceil(kernels::GEMV_WARPS as usize);
        self.norm_counter(scratch)?;
        if scratch.blocks < blocks || scratch.part.is_none() {
            // SAFETY: every partial is written by its block before the last
            // block reads it, and only `blocks` of them are read.
            scratch.part = Some(unsafe { self.uninit(blocks) }?);
            scratch.blocks = blocks;
        }
        let part = scratch.part.as_mut().expect("allocated above");
        let ctr = scratch.ctr.as_mut().expect("allocated above");

        // SAFETY: the last block writes every one of the `n` outputs, every
        // code, and every scale.
        let mut x = unsafe { self.uninit(n) }?;
        let mut xq = Q8 {
            buf: unsafe { self.uninit_i8(n + (n / 32) * 4) }?,
            rows: 1,
            k: n,
        };
        let (wq, ts, ki, ni) = (ty.id(), ty.device_stride() as i32, k as i32, n as i32);
        let asc_off = a.scale_offset() as i32;
        let xasc_off = xq.scale_offset() as i32;
        let f = self.func("gemv_norm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(data)
            .arg(&wq)
            .arg(&ts)
            .arg(&a.buf)
            .arg(&asc_off)
            .arg(&ki)
            .arg(&ni)
            .arg(h)
            .arg(weight)
            .arg(&eps)
            .arg(&mut x)
            .arg(&mut xq.buf)
            .arg(&xasc_off)
            .arg(part)
            .arg(ctr);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (32, kernels::GEMV_WARPS, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every column once, every read is bounds
        // checked above, and the counter is left at zero for the next call.
        launched("gemv_norm", unsafe { lb.launch(cfg) })?;
        Ok((x, xq))
    }

    /// [`Self::embed_scaled`] off a table that stays in its checkpoint's
    /// blocks.
    ///
    /// `table` is `vocab` rows of `ch` elements in `ty`'s blocks, as
    /// [`Self::upload_quant`] laid them out; each gathered row is unpacked on
    /// the way out. Refuses a row that is not a whole number of blocks, since
    /// a row that started mid-block would decode plausibly and wrongly.
    pub fn embed_packed(
        &self,
        table: &CudaSlice<u8>,
        ty: Quant,
        ids: &CudaSlice<i64>,
        t: usize,
        ch: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let bs = ty.block_size();
        if !ch.is_multiple_of(bs) {
            return Err(CudaError::RaggedBlock { k: ch, block: bs });
        }
        if ids.len() < t {
            return Err(CudaError::SliceOverrun {
                at: t,
                len: ids.len(),
            });
        }
        // SAFETY: one block a row writes every one of the row's `ch`
        // elements, and the grid has `t` blocks.
        let mut out = unsafe { self.uninit(t * ch) }?;
        if t == 0 {
            return Ok(out);
        }
        let (tyi, bsi, tsi, ti, chi) = (
            ty.id(),
            bs as i32,
            ty.device_stride() as i32,
            t as i32,
            ch as i32,
        );
        let f = self.func("embed_q");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(table)
            .arg(&tyi)
            .arg(&bsi)
            .arg(&tsi)
            .arg(ids)
            .arg(&mut out)
            .arg(&ti)
            .arg(&chi)
            .arg(&scale);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        launched("embed_q", unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// Fuses weight normalisation. Mirrors `xabe_dsp::fuse_weight_norm`.
    /// SiLU in place, unfused.
    pub fn silu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_silu", x, n, None)
    }

    /// GELU's tanh approximation, which is `nn.GELU(approximate="tanh")`.
    ///
    /// A different function from [`Gpu::gelu`]'s exact erf form: they agree to
    /// about 1e-3, which is far more than rounding and far less than an
    /// obvious break. Callers ask for the one their checkpoint was fitted
    /// against.
    pub fn gelu_tanh(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_gelu_tanh", x, n, None)
    }

    /// A grouped convolution with left padding. Returns the output and its
    /// length.
    ///
    /// The weight's second axis is the channel *within* the group, so it is
    /// `[out_ch, in_ch / groups, k]` and not `[out_ch, in_ch, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_conv1d(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        in_ch: usize,
        t: usize,
        out_ch: usize,
        k: usize,
        groups: usize,
        pad_left: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let out_t = (t + pad_left).saturating_sub(k) + 1;
        let mut out = self.zeros(out_ch * out_t)?;
        let (a, b, c, d, e, g, h) = (
            in_ch as i32,
            t as i32,
            out_ch as i32,
            k as i32,
            groups as i32,
            pad_left as i32,
            out_t as i32,
        );
        let f = self.func("grouped_conv1d");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x)
            .arg(w)
            .arg(bias)
            .arg(&mut out)
            .arg(&a)
            .arg(&b)
            .arg(&c)
            .arg(&d)
            .arg(&e)
            .arg(&g)
            .arg(&h);
        launched("grouped_conv1d", unsafe {
            lb.launch(Self::flat(out_ch * out_t))
        })?;
        Ok((out, out_t))
    }

    /// Mish in place: `x * tanh(softplus(x))`.
    pub fn mish(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.activate("act_mish", x, n, None)
    }

    /// GPT-J style partial rotary embedding over interleaved pairs.
    ///
    /// Not [`Gpu::rope`]: that one rotates every head and pairs `i` with
    /// `i + head_dim / 2`, which is what a HuggingFace layout wants. This
    /// rotates only the first `rot_dim` of each row and pairs `2j` with
    /// `2j + 1`. Both produce fluent speech from the wrong weights.
    pub fn rope_gptj(
        &self,
        x: &mut CudaSlice<f32>,
        inv_freq: &CudaSlice<f32>,
        positions: usize,
        dim: usize,
        rot_dim: usize,
    ) -> Result<(), CudaError> {
        let n = positions * rot_dim / 2;
        let (a, b, c) = (positions as i32, dim as i32, rot_dim as i32);
        let f = self.func("rope_gptj");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(inv_freq).arg(&a).arg(&b).arg(&c);
        launched("rope_gptj", unsafe { lb.launch(Self::flat(n)) })
    }

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
        let dummy = &self.dummy;
        let has = i32::from(bias.is_some());
        let b = bias.unwrap_or(dummy);
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

    /// Root-mean-square normalisation. Mirrors `xabe_dsp::rms_norm`.
    pub fn rms_norm(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        weight: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self
            .rms_norm_inner(x, None, rows, dim, weight, eps, false)?
            .0)
    }

    /// The same, also emitting the int8 twin the packed mat-vec wants.
    ///
    /// The activation a normalisation produces is, in every transformer here,
    /// immediately projected two or three times by a packed weight. Taking the
    /// twin here rather than in a kernel of its own saves a launch, an
    /// allocation and a re-read of the row that was just written.
    ///
    /// Refuses the shapes the fused mapping does not cover rather than falling
    /// back silently: a caller that asked for a twin and got none would work,
    /// slowly, and nothing would say why.
    pub fn rms_norm_q(
        &self,
        x: &mut CudaSlice<f32>,
        add: Option<&CudaSlice<f32>>,
        rows: usize,
        dim: usize,
        weight: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<(CudaSlice<f32>, Q8), CudaError> {
        // Whole 32-column scale groups, which is also what makes a group a
        // whole number of threads. The last iteration may be ragged; the kernel
        // predicates it, and the boundary falls on a group either way.
        if !dim.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock { k: dim, block: 32 });
        }
        let (out, q8) = self.rms_norm_inner(x, add, rows, dim, weight, eps, true)?;
        Ok((out, q8.expect("asked for the twin")))
    }

    /// Both of the above. `quantize` decides whether the twin is written.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_inner(
        &self,
        x: &CudaSlice<f32>,
        add: Option<&CudaSlice<f32>>,
        rows: usize,
        dim: usize,
        weight: &CudaSlice<f32>,
        eps: f32,
        quantize: bool,
    ) -> Result<(CudaSlice<f32>, Option<Q8>), CudaError> {
        // SAFETY: the kernel writes every element of every row it owns, and
        // the grid is one block per row.
        let mut out = unsafe { self.uninit(rows * dim) }?;
        // SAFETY: the same loop writes every code, and its last lane every
        // scale, over the same range.
        let mut q8 = match quantize {
            true => Some(Q8 {
                buf: unsafe { self.uninit_i8(rows * dim + rows * (dim / 32) * 4) }?,
                rows,
                k: dim,
            }),
            false => None,
        };
        let d = dim as i32;
        let off = q8.as_ref().map_or(0, |q| q.scale_offset() as i32);
        let null: u64 = 0;
        let f = self.func("rms_norm");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x);
        match add {
            Some(a) => lb.arg(a),
            None => lb.arg(&null),
        };
        lb.arg(weight).arg(&mut out);
        match &mut q8 {
            Some(q) => lb.arg(&mut q.buf),
            None => lb.arg(&null),
        };
        lb.arg(&off).arg(&d).arg(&eps);
        // A multiple of 32, so that one warp covers one scale group.
        let threads = match quantize {
            true => RMS_THREADS,
            false => (dim as u32).next_power_of_two().clamp(32, 1024),
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        launched("rms_norm", unsafe { lb.launch(cfg) })?;
        Ok((out, q8))
    }

    /// `a = silu(a) * b`. Mirrors `xabe_dsp::silu_mul`.
    pub fn silu_mul(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        self.silu_mul_inner(a, b, 1, n, false).map(|_| ())
    }

    /// The same, also emitting the int8 twin the packed mat-vec wants.
    ///
    /// The gated result is projected straight back down by a packed weight, so
    /// this is the same saving [`Gpu::rms_norm_q`] makes at the other end of
    /// the block.
    /// `rows` and `k` are given separately even though the kernel walks a flat
    /// index, because the twin is addressed as `[rows, k]` by the matmul that
    /// reads it. Passing `rows * k` as the width produces a twin that is the
    /// right length and the wrong shape - which `gemm_batched` refuses, and
    /// which nothing downstream would have noticed.
    pub fn silu_mul_q(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        rows: usize,
        k: usize,
    ) -> Result<Q8, CudaError> {
        if !(rows * k).is_multiple_of(BLOCK as usize) || !k.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock {
                k: rows * k,
                block: BLOCK as usize,
            });
        }
        Ok(self
            .silu_mul_inner(a, b, rows, k, true)?
            .expect("asked for the twin"))
    }

    /// SiLU-gates one buffer against its own second half, and quantises.
    ///
    /// `x` is `[2, rows, k]` - the output of one batched product over the gate
    /// and up weights - and the result is `silu(gate) * up` written over the
    /// first half, which is where the down projection then reads it from. One
    /// buffer rather than two because that is what a batched product produces,
    /// and the point of producing it that way is one launch a layer instead of
    /// two. At a single decoded row a launch is most of what a kernel this size
    /// costs.
    ///
    /// The arithmetic is `silu_mul`'s exactly: same expression, same order,
    /// same group quantiser.
    pub fn silu_mul_pair(
        &self,
        x: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
    ) -> Result<Q8, CudaError> {
        let n = rows * k;
        if !n.is_multiple_of(BLOCK as usize) || !k.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock {
                k: n,
                block: BLOCK as usize,
            });
        }
        Ok(self
            .silu_mul_pair_inner(x, rows, k, true)?
            .expect("asked for the twin"))
    }

    /// [`Self::silu_mul_pair`] without the int8 twin, for an f16 down
    /// projection that would never read the codes.
    pub fn silu_mul_halves(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), CudaError> {
        self.silu_mul_pair_inner(x, 1, n, false)?;
        Ok(())
    }

    fn silu_mul_pair_inner(
        &self,
        x: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        quantize: bool,
    ) -> Result<Option<Q8>, CudaError> {
        let n = rows * k;
        // SAFETY: one thread per element writes every code, and its group's
        // first lane every scale.
        let mut q8 = match quantize {
            true => Some(Q8 {
                buf: unsafe { self.uninit_i8(n + (n / 32) * 4) }?,
                rows,
                k,
            }),
            false => None,
        };
        let off = q8.as_ref().map_or(0, |q| q.scale_offset() as i32);
        let null: u64 = 0;
        let ni = n as i32;
        let f = self.func("silu_mul_pair");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x);
        match &mut q8 {
            Some(q) => lb.arg(&mut q.buf),
            None => lb.arg(&null),
        };
        lb.arg(&off).arg(&ni);
        launched("silu_mul_pair", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(q8)
    }

    /// Both of the above.
    fn silu_mul_inner(
        &self,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        rows: usize,
        k: usize,
        quantize: bool,
    ) -> Result<Option<Q8>, CudaError> {
        let n = rows * k;
        // SAFETY: one thread per element writes every code, and its group's
        // first lane every scale.
        let mut q8 = match quantize {
            true => Some(Q8 {
                buf: unsafe { self.uninit_i8(n + (n / 32) * 4) }?,
                rows,
                k,
            }),
            false => None,
        };
        let off = q8.as_ref().map_or(0, |q| q.scale_offset() as i32);
        let null: u64 = 0;
        let ni = n as i32;
        let f = self.func("silu_mul");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(a).arg(b);
        match &mut q8 {
            Some(q) => lb.arg(&mut q.buf),
            None => lb.arg(&null),
        };
        lb.arg(&off).arg(&ni);
        launched("silu_mul", unsafe { lb.launch(Self::flat(n)) })?;
        Ok(q8)
    }

    /// Rotary position embedding, in place. Mirrors `xabe_dsp::rope`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut CudaSlice<f32>,
        off: usize,
        t: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        first: usize,
    ) -> Result<(), CudaError> {
        self.rope_scaled(x, off, None, t, heads, head_dim, theta, first)
    }

    /// RoPE with an optional per-pair frequency divisor.
    ///
    /// `freq_div` is Llama-3's `rope_freqs.weight`, `head_dim / 2` long.
    /// Mirrors `xabe_dsp::rope_scaled`. Llama-2 passes `None`; passing
    /// `None` for a checkpoint that ships the tensor is a model that stays
    /// fluent for a sentence and drifts after, with no shape to catch it.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_scaled(
        &self,
        x: &mut CudaSlice<f32>,
        off: usize,
        freq_div: Option<&CudaSlice<f32>>,
        t: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        first: usize,
    ) -> Result<(), CudaError> {
        let n = t * heads * head_dim / 2;
        // `off` exists because the attention projections are issued as one
        // batched product: `q` and `k` are contiguous blocks of one output,
        // and rotating `k` in place is cheaper than copying it out to rotate.
        if off + t * heads * head_dim > x.len() {
            return Err(CudaError::SliceOverrun {
                at: off + t * heads * head_dim,
                len: x.len(),
            });
        }
        let (a, b_, c, d) = (t as i32, heads as i32, head_dim as i32, first as i32);
        let f = self.func("rope");
        let mut lb = self.stream.launch_builder(f);
        // A flag, not a null pointer: every launch argument has to point at
        // something real, so the no-scaling case passes a one-element dummy
        // the kernel is told never to read.
        let dummy = &self.dummy;
        let has = i32::from(freq_div.is_some());
        let div = freq_div.unwrap_or(dummy);
        let o = off as i64;
        lb.arg(x)
            .arg(&o)
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
        dilation: usize,
    ) -> Result<(CudaSlice<f32>, usize), CudaError> {
        let span = dilation * (k - 1) + 1;
        let out_t = (t + 2 * pad - span) / stride + 1;
        let cols = in_ch * k;
        let mut out = self.zeros(out_t * cols)?;
        let (a, b_, c, d, e, dl, g) = (
            t as i32,
            in_ch as i32,
            k as i32,
            stride as i32,
            pad as i32,
            dilation as i32,
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
            .arg(&dl)
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

    /// [`Self::split_heads`], writing f16.
    ///
    /// For a tensor that is built once and then read whole many times - the
    /// cross-attention cache is the case this exists for - where the split and
    /// a `to_f16` pass after it read and write the same tensor twice to change
    /// nothing but its width. The rounding is `f32_to_f16`'s, which is the
    /// round-to-nearest-even the tiled matmul's own staging does, so an
    /// operand converted here and one staged from f32 are the same bits.
    pub fn split_heads_f16(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<u16>, CudaError> {
        self.reshape_heads_f16("split_heads_f16", x, 0, None, t, heads, head_dim)
    }

    /// [`Self::split_heads_t`], writing f16.
    pub fn split_heads_t_f16(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<u16>, CudaError> {
        self.reshape_heads_f16("split_heads_t_f16", x, 0, None, t, heads, head_dim)
    }

    /// [`Self::split_heads_f16`] of the `[t, heads * head_dim]` block that
    /// starts `x_off` elements into `x`, adding the row `bias` - given as a
    /// buffer and an offset into it - before rounding.
    ///
    /// For a cache built from one batched projection over every layer: the
    /// batch carries a single bias where each layer has its own, so the bias
    /// moves here. It is the same f32 add the matmul's epilogue makes, so the
    /// cache holds the same bits it did when the layers were projected one at
    /// a time - `the_packed_head_splits_are_the_f32_ones_converted` says so.
    #[allow(clippy::too_many_arguments)]
    pub fn split_heads_f16_at(
        &self,
        x: &CudaSlice<f32>,
        x_off: usize,
        bias: Option<(&CudaSlice<f32>, usize)>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<u16>, CudaError> {
        self.reshape_heads_f16("split_heads_f16", x, x_off, bias, t, heads, head_dim)
    }

    /// [`Self::split_heads_t_f16`] with an offset and a bias, as
    /// [`Self::split_heads_f16_at`].
    #[allow(clippy::too_many_arguments)]
    pub fn split_heads_t_f16_at(
        &self,
        x: &CudaSlice<f32>,
        x_off: usize,
        bias: Option<(&CudaSlice<f32>, usize)>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<u16>, CudaError> {
        self.reshape_heads_f16("split_heads_t_f16", x, x_off, bias, t, heads, head_dim)
    }

    /// `[heads, t, head_dim]` back to `[t, heads*head_dim]`.
    pub fn merge_heads(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        Ok(self.merge_heads_inner(x, t, heads, head_dim, false)?.0)
    }

    /// Causal attention over a whole prompt, fused: scores, mask, softmax and
    /// the value product in one kernel, nothing materialised.
    ///
    /// `q` is the projection buffer itself, `[tq, heads * 128]` with the
    /// query first - no head split - and the merged context comes back in the
    /// same shape, so neither `split_heads` nor `merge_heads` runs. `k` and
    /// `v` are the caches in their own layouts, `[kv_head][pos][128]` and
    /// `[kv_head][128][cap]`, holding `past + tq` valid positions. A
    /// grouped-query model passes `kv_heads < heads`.
    ///
    /// The head dimension is fixed at 128 - both Llama stages' - because the
    /// kernel's tile shapes assume it; anything else is refused, and the
    /// caller keeps the unfused chain for it. The arithmetic is the chain's
    /// exactly: scores accumulate in f32 from f16-rounded operands,
    /// `__expf` like `softmax_causal`, probabilities rounded to f16 on their
    /// way into the value product - where the chain rounded them too.
    #[allow(clippy::too_many_arguments)]
    /// Whether [`Self::flash_attn`] covers this attention shape.
    ///
    /// A predicate rather than a caught error, because choosing the path is
    /// not an exceptional case: a model whose head width the kernel is not
    /// instantiated at runs the unfused chain and is correct, and asking with
    /// an error would mean allocating an output buffer to throw away.
    pub fn supports_flash(&self, head_dim: usize, heads: usize, kv_heads: usize) -> bool {
        matches!(head_dim, 64 | 128) && heads.is_multiple_of(kv_heads.max(1))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        tq: usize,
        past: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        cap: usize,
        scale: f32,
        causal: bool,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // Only the widths the kernel is instantiated at. A width it is not
        // instantiated at would index across heads *in bounds* and return
        // plausible context, so it is refused by name and the caller falls
        // back to the unfused chain.
        // The query rows a block owns, which is the kernel's own `QT` and
        // therefore its grid stride. It is not the same at both widths: see
        // the kernel's header for why the encoder's instantiation takes 64.
        let (name, qt) = match head_dim {
            128 => ("flash_attn", 32),
            64 => ("flash_attn_64", 64),
            _ => {
                return Err(CudaError::UnsupportedAttention {
                    head_dim,
                    heads,
                    kv_heads,
                });
            }
        };
        if !heads.is_multiple_of(kv_heads.max(1)) {
            return Err(CudaError::UnsupportedAttention {
                head_dim,
                heads,
                kv_heads,
            });
        }
        // SAFETY: every (row, column) of the output is written by exactly one
        // lane of the store loop below; rows past `tq` are predicated off.
        let mut out = unsafe { self.uninit(tq * heads * head_dim) }?;
        let (tqi, pi, hi, kvi, ci) = (
            tq as i32,
            past as i32,
            heads as i32,
            kv_heads as i32,
            cap as i32,
        );
        let causal_i = i32::from(causal);
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut out)
            .arg(&tqi)
            .arg(&pi)
            .arg(&hi)
            .arg(&kvi)
            .arg(&ci)
            .arg(&scale)
            .arg(&causal_i);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((tq as u32).div_ceil(qt), heads as u32, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        launched(name, unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// [`Self::flash_attn`] against an f16 cache.
    ///
    /// Only at a head width of 128, which is the only width any stage holds a
    /// cache that way at, and only with an even `cap` - the value layout is
    /// read as words of two positions, so an odd capacity would straddle them.
    /// Both are refused by name rather than producing plausible context.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_f16(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        tq: usize,
        past: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        cap: usize,
        scale: f32,
        causal: bool,
    ) -> Result<CudaSlice<f32>, CudaError> {
        // Only the widths the kernel is instantiated at. A width it is not
        // instantiated at would index across heads *in bounds* and return
        // plausible context, so it is refused by name and the caller falls
        // back to the unfused chain.
        // The query rows a block owns, which is the kernel's own `QT` and
        // therefore its grid stride. It is not the same at both widths: see
        // the kernel's header for why the encoder's instantiation takes 64.
        let (name, qt) = match head_dim {
            128 => ("flash_attn_h", 32),
            _ => {
                return Err(CudaError::UnsupportedAttention {
                    head_dim,
                    heads,
                    kv_heads,
                });
            }
        };
        if !cap.is_multiple_of(2) {
            return Err(CudaError::OddCacheCapacity { cap });
        }
        if !heads.is_multiple_of(kv_heads.max(1)) {
            return Err(CudaError::UnsupportedAttention {
                head_dim,
                heads,
                kv_heads,
            });
        }
        // SAFETY: every (row, column) of the output is written by exactly one
        // lane of the store loop below; rows past `tq` are predicated off.
        let mut out = unsafe { self.uninit(tq * heads * head_dim) }?;
        let (tqi, pi, hi, kvi, ci) = (
            tq as i32,
            past as i32,
            heads as i32,
            kv_heads as i32,
            cap as i32,
        );
        let causal_i = i32::from(causal);
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(q)
            .arg(k)
            .arg(v)
            .arg(&mut out)
            .arg(&tqi)
            .arg(&pi)
            .arg(&hi)
            .arg(&kvi)
            .arg(&ci)
            .arg(&scale)
            .arg(&causal_i);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((tq as u32).div_ceil(qt), heads as u32, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        launched(name, unsafe { lb.launch(cfg) })?;
        Ok(out)
    }

    /// Attention for one query position, in one launch, off an f16 cache.
    ///
    /// `q` is `[heads, head_dim]` with the query heads of a grouped-query
    /// group adjacent, which is the layout one decode row already has. The
    /// caches are the append kernels' layouts: keys `[kv_heads, cap,
    /// head_dim]`, values `[kv_heads, head_dim, cap]`, `tk` positions of each
    /// live. Returns `[heads, head_dim]`, which for one row is `[heads *
    /// head_dim]` and needs no merge.
    ///
    /// Replaces the score product, the softmax and the value product - three
    /// launches and a score row written and read back twice - with one kernel
    /// that keeps the scores in shared memory. `scale_q` puts the scale on
    /// the query before the product, as Whisper does; otherwise it goes on
    /// the scores, as Llama does. See the kernel for the arithmetic.
    ///
    /// Only the widths the kernel is instantiated at: 64 and 128.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_f16(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let (out, _) = self.attn_decode_f16_inner(
            q, k, v, heads, kv_heads, head_dim, tk, cap, scale, scale_q, scratch, false,
        )?;
        Ok(out)
    }

    /// [`Self::attn_decode_f16`] with the context's int8 twin taken in the
    /// same pass - `quantize_q8`'s arithmetic over the row the merging block
    /// has in registers, for the packed output projection that reads it.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_f16_q(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
    ) -> Result<(CudaSlice<f32>, Q8), CudaError> {
        let (out, q8) = self.attn_decode_f16_inner(
            q, k, v, heads, kv_heads, head_dim, tk, cap, scale, scale_q, scratch, true,
        )?;
        Ok((out, q8.expect("asked for the twin")))
    }

    /// [`Self::attn_decode_f16_q`] for one of several sequences decoding
    /// together: the query is read from element `q_off` of `q`, and the
    /// context and its twin are written to row `row` of `out` and `twin`,
    /// which are `[rows, heads * head_dim]` shared by the batch and filled a
    /// row at a time - `twin` from [`Self::q8_zeros`]. The same kernel and
    /// the same arithmetic as the single-row call, which is what the test
    /// holds it to.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_f16_q_row(
        &self,
        q: &CudaSlice<f32>,
        q_off: usize,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
        out: &mut CudaSlice<f32>,
        twin: &mut Q8,
        row: usize,
    ) -> Result<(), CudaError> {
        let name = Self::attn_decode_f16_name(heads, kv_heads, head_dim, tk, cap)?;
        let n = heads * head_dim;
        self.attn_decode_launch(
            name,
            q,
            q_off,
            KvCache::F16(k, v),
            heads,
            kv_heads,
            head_dim,
            tk,
            cap,
            scale,
            scale_q,
            scratch,
            out,
            row * n,
            Some(twin),
        )
    }

    /// The f16 decode kernel for a shape, by head width and context.
    /// How many query rows a key-value head carries in the kernel `name`:
    /// the `G` its entry was instantiated with.
    fn ad_gmax(name: &str) -> usize {
        if name.contains("_g8") {
            8
        } else {
            kernels::AD_GMAX as usize
        }
    }

    fn attn_decode_f16_name(
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
    ) -> Result<&'static str, CudaError> {
        if !cap.is_multiple_of(2) {
            return Err(CudaError::OddCacheCapacity { cap });
        }
        // The chunk width by context, from the sweep in `docs/BENCHMARKS.md`:
        // narrow chunks win while the context is short or the model has no
        // query groups to share a head's reads, wide ones once the merge over
        // many partials is what the last block waits on.
        Ok(match head_dim {
            128 if tk <= 256 || heads == kv_heads => "attn_decode_h128_c32",
            128 if tk >= 2048 => "attn_decode_h128_c128",
            128 => "attn_decode_h128",
            // A query group wider than `AD_GMAX` takes the instantiation
            // sized for eight: CosyVoice3's speech LLM, 14 heads over 2.
            // With two key-value heads the grid is `chunks * 2` blocks, so
            // at a short context the narrow chunk is what fills the card.
            64 if heads / kv_heads.max(1) > kernels::AD_GMAX as usize && tk <= 1024 => {
                "attn_decode_h64_g8_c32"
            }
            64 if heads / kv_heads.max(1) > kernels::AD_GMAX as usize => "attn_decode_h64_g8",
            64 => "attn_decode_h64",
            _ => {
                return Err(CudaError::UnsupportedAttention {
                    head_dim,
                    heads,
                    kv_heads,
                });
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_decode_f16_inner(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<u16>,
        v: &CudaSlice<u16>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
        twin: bool,
    ) -> Result<(CudaSlice<f32>, Option<Q8>), CudaError> {
        let name = Self::attn_decode_f16_name(heads, kv_heads, head_dim, tk, cap)?;
        self.attn_decode_inner(
            name,
            q,
            KvCache::F16(k, v),
            heads,
            kv_heads,
            head_dim,
            tk,
            cap,
            scale,
            scale_q,
            scratch,
            twin,
        )
    }

    /// [`Self::attn_decode_f16`] off an f32 cache, at head width 64 - the
    /// ASR's self-attention cache, which is the one f32 cache still decoded
    /// against.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
    ) -> Result<CudaSlice<f32>, CudaError> {
        if head_dim != 64 {
            return Err(CudaError::UnsupportedAttention {
                head_dim,
                heads,
                kv_heads,
            });
        }
        let (out, _) = self.attn_decode_inner(
            "attn_decode_f64",
            q,
            KvCache::F32(k, v),
            heads,
            kv_heads,
            head_dim,
            tk,
            cap,
            scale,
            scale_q,
            scratch,
            false,
        )?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_decode_inner(
        &self,
        name: &'static str,
        q: &CudaSlice<f32>,
        kv: KvCache<'_>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
        twin: bool,
    ) -> Result<(CudaSlice<f32>, Option<Q8>), CudaError> {
        let n = heads * head_dim;
        // SAFETY: every element is written by exactly one thread - of the only
        // block when there is one chunk, of the merging block otherwise.
        let mut out = unsafe { self.uninit(n) }?;
        // SAFETY: the same thread writes the code, and its warp's first lane
        // the scale, for every element and every group.
        let mut q8 = match twin {
            true => Some(Q8 {
                buf: unsafe { self.uninit_i8(n + (n / 32) * 4) }?,
                rows: 1,
                k: n,
            }),
            false => None,
        };
        self.attn_decode_launch(
            name,
            q,
            0,
            kv,
            heads,
            kv_heads,
            head_dim,
            tk,
            cap,
            scale,
            scale_q,
            scratch,
            &mut out,
            0,
            q8.as_mut(),
        )?;
        Ok((out, q8))
    }

    /// The checks and the launch behind every decode attention: the query
    /// from `q_off`, the context to `out_off` of `out`, and its twin to the
    /// same row of `twin` when there is one.
    #[allow(clippy::too_many_arguments)]
    fn attn_decode_launch(
        &self,
        name: &'static str,
        q: &CudaSlice<f32>,
        q_off: usize,
        kv: KvCache<'_>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        tk: usize,
        cap: usize,
        scale: f32,
        scale_q: bool,
        scratch: &mut DecodeScratch,
        out: &mut CudaSlice<f32>,
        out_off: usize,
        twin: Option<&mut Q8>,
    ) -> Result<(), CudaError> {
        // The kernel carries at most `gmax` query rows a key-value head -
        // `AD_GMAX`, or eight for the one instantiation named for it - and
        // a group of zero is a division by it.
        let gmax = Self::ad_gmax(name);
        if kv_heads == 0 || !heads.is_multiple_of(kv_heads) || heads / kv_heads > gmax {
            return Err(CudaError::UnsupportedAttention {
                head_dim,
                heads,
                kv_heads,
            });
        }
        if tk == 0 || tk > cap {
            return Err(CudaError::CacheOverrun { at: tk, cap });
        }
        // A value row is `cap` elements, and the kernel loads it eight bytes
        // at a time at least - so the capacity must keep every row at that
        // alignment. Every capacity here is a multiple of 64 or 1500.
        let esz = match &kv {
            KvCache::F32(..) => 4,
            KvCache::F16(..) => 2,
        };
        if !(cap * esz).is_multiple_of(8) {
            return Err(CudaError::OddCacheCapacity { cap });
        }
        let (klen, vlen) = match &kv {
            KvCache::F32(k, v) => (k.len(), v.len()),
            KvCache::F16(k, v) => (k.len(), v.len()),
        };
        let need = kv_heads * cap * head_dim;
        if klen < need || vlen < need {
            return Err(CudaError::SliceOverrun {
                at: need,
                len: klen.min(vlen),
            });
        }
        let n = heads * head_dim;
        if q.len() < q_off + n {
            return Err(CudaError::SliceOverrun {
                at: q_off + n,
                len: q.len(),
            });
        }
        if out.len() < out_off + n {
            return Err(CudaError::SliceOverrun {
                at: out_off + n,
                len: out.len(),
            });
        }
        // The twin's scales are one a group of 32, so a row must start on a
        // group; every head width here is a multiple of 32.
        if !out_off.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock {
                k: out_off,
                block: 32,
            });
        }
        if let Some(t) = &twin {
            let (rows, k) = t.shape();
            if k != n || rows * k < out_off + n {
                return Err(CudaError::MismatchedQ8 {
                    rows,
                    k,
                    want_rows: out_off / n + 1,
                    want_k: n,
                });
            }
        }
        let group = heads / kv_heads;
        let ch = match name {
            "attn_decode_h128_c32" | "attn_decode_h64_g8_c32" => 32,
            "attn_decode_h128_c128" => 128,
            _ => kernels::AD_CH as usize,
        };
        let chunks = tk.div_ceil(ch);

        // Grow the scratch to this call's shape. Doubling the chunk count
        // keeps the allocations logarithmic in the context; a change of head
        // geometry, which no stage ever makes, starts over.
        let stride = gmax * (head_dim + 2);
        if scratch.kv_heads != kv_heads || scratch.head_dim != head_dim {
            scratch.part = None;
            scratch.ctr = None;
            scratch.chunks = 0;
            scratch.kv_heads = kv_heads;
            scratch.head_dim = head_dim;
        }
        if scratch.ctr.is_none() {
            scratch.ctr = Some(self.stream.alloc_zeros::<u32>(kv_heads).map_err(|source| {
                CudaError::Driver {
                    what: "allocating",
                    source,
                }
            })?);
        }
        if scratch.chunks < chunks || scratch.part.is_none() {
            let want = chunks.next_power_of_two().max(4);
            // SAFETY: a partial is read only by the block that merges a head,
            // and only for the chunks and rows that this same launch wrote.
            scratch.part = Some(unsafe { self.uninit(kv_heads * want * stride) }?);
            scratch.chunks = want;
        }
        let part = scratch.part.as_mut().expect("allocated above");
        let ctr = scratch.ctr.as_mut().expect("allocated above");

        let null: u64 = 0;
        let mut q8 = twin;
        let asc_off = q8.as_ref().map_or(0, |q| q.scale_offset() as i32);
        let (qo, oo) = (q_off as i64, out_off as i64);
        let (tki, gi, ci, sqi, chi) = (
            tk as i32,
            group as i32,
            cap as i32,
            i32::from(scale_q),
            chunks as i32,
        );
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(q);
        match kv {
            KvCache::F32(k, v) => {
                lb.arg(k).arg(v);
            }
            KvCache::F16(k, v) => {
                lb.arg(k).arg(v);
            }
        }
        lb.arg(out)
            .arg(part)
            .arg(ctr)
            .arg(&tki)
            .arg(&gi)
            .arg(&ci)
            .arg(&scale)
            .arg(&sqi)
            .arg(&chi);
        match &mut q8 {
            Some(q) => lb.arg(&mut q.buf).arg(&asc_off),
            None => lb.arg(&null).arg(&asc_off),
        };
        lb.arg(&qo).arg(&oo);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (chunks as u32, kv_heads as u32, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the grid covers every (chunk, head) once; the caches are
        // checked above to hold `kv_heads * cap * head_dim`, the chunk's keys
        // are bounded by `tk <= cap` inside the kernel, the scratch holds
        // `chunks` partials a head, and the query, the output row and the
        // twin's row are bounds checked above.
        launched(name, unsafe { lb.launch(cfg) })
    }

    /// The merge and the int8 twin of its output, in one pass.
    ///
    /// The merged context is what the output projection multiplies, and both
    /// packed matmul paths read it as int8 - so quantizing it in a pass of its
    /// own re-reads the whole row for numbers the merge is already holding.
    /// The same reasoning gave `rms_norm` and `silu_mul` their twins.
    pub fn merge_heads_q(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(CudaSlice<f32>, Q8), CudaError> {
        let k = heads * head_dim;
        if !k.is_multiple_of(32) {
            return Err(CudaError::RaggedBlock { k, block: 32 });
        }
        let (out, q8) = self.merge_heads_inner(x, t, heads, head_dim, true)?;
        Ok((out, q8.expect("asked for the twin")))
    }

    fn merge_heads_inner(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        heads: usize,
        head_dim: usize,
        quantize: bool,
    ) -> Result<(CudaSlice<f32>, Option<Q8>), CudaError> {
        let k = heads * head_dim;
        let n = t * k;
        let mut out = self.zeros(n)?;
        // SAFETY: one thread per element writes every code, and its group's
        // first lane every scale.
        let mut q8 = match quantize {
            true => Some(Q8 {
                buf: unsafe { self.uninit_i8(n + (n / 32) * 4) }?,
                rows: t,
                k,
            }),
            false => None,
        };
        let off = q8.as_ref().map_or(0, |q| q.scale_offset() as i32);
        let null: u64 = 0;
        let (a, b_, c) = (t as i32, heads as i32, head_dim as i32);
        let f = self.func("merge_heads");
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out);
        match &mut q8 {
            Some(q) => lb.arg(&mut q.buf),
            None => lb.arg(&null),
        };
        lb.arg(&off).arg(&a).arg(&b_).arg(&c);
        launched("merge_heads", unsafe { lb.launch(Self::flat(n)) })?;
        Ok((out, q8))
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
        // Every element is written by the kernel, transposed or not, so there
        // is nothing for a zeroing pass to establish - and at encoder width
        // that pass is 7.7 MB.
        let mut out = unsafe { self.uninit(n) }?;
        let (a, b_, c) = (t as i32, heads as i32, head_dim as i32);
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(x).arg(&mut out).arg(&a).arg(&b_).arg(&c);
        launched(name, unsafe {
            lb.launch(Self::reshape_cfg(name, t, heads * head_dim))
        })?;
        Ok(out)
    }

    /// The launch shape one of the reshape kernels wants.
    ///
    /// The transposing pair stages a 32x32 tile and is launched over that grid;
    /// the others are element-at-a-time and take the flat one. Getting this
    /// wrong is not a subtle failure - a tiled kernel under a flat grid covers
    /// a thirty-second of its output - so the two are chosen by name here
    /// rather than by a flag a caller could forget.
    fn reshape_cfg(name: &str, t: usize, d: usize) -> LaunchConfig {
        if name == "split_heads_t" || name == "split_heads_t_f16" {
            LaunchConfig {
                grid_dim: (d.div_ceil(32) as u32, t.div_ceil(32) as u32, 1),
                block_dim: (32, 8, 1),
                shared_mem_bytes: 0,
            }
        } else {
            Self::flat(t * d)
        }
    }

    /// [`Self::reshape_heads`] for the kernels that write f16.
    #[allow(clippy::too_many_arguments)]
    fn reshape_heads_f16(
        &self,
        name: &'static str,
        x: &CudaSlice<f32>,
        x_off: usize,
        bias: Option<(&CudaSlice<f32>, usize)>,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<u16>, CudaError> {
        let n = t * heads * head_dim;
        if x_off + n > x.len() {
            return Err(CudaError::SliceOverrun {
                at: x_off + n,
                len: x.len(),
            });
        }
        if let Some((b, off)) = bias
            && off + heads * head_dim > b.len()
        {
            return Err(CudaError::SliceOverrun {
                at: off + heads * head_dim,
                len: b.len(),
            });
        }
        let mut out = self
            .stream
            .alloc_zeros::<u16>(n)
            .map_err(|source| CudaError::Driver {
                what: "allocating",
                source,
            })?;
        let (a, b_, c) = (t as i32, heads as i32, head_dim as i32);
        let null: u64 = 0;
        let xv = x.slice(x_off..);
        let bv = bias.map(|(b, off)| b.slice(off..));
        let f = self.func(name);
        let mut lb = self.stream.launch_builder(f);
        lb.arg(&xv);
        match &bv {
            Some(v) => lb.arg(v),
            None => lb.arg(&null),
        };
        lb.arg(&mut out).arg(&a).arg(&b_).arg(&c);
        launched(name, unsafe {
            lb.launch(Self::reshape_cfg(name, t, heads * head_dim))
        })?;
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
