//! The CUDA source, compiled at runtime by NVRTC.
//!
//! Kept as a string rather than a `.cu` file compiled by `build.rs` so that the
//! workspace builds with no CUDA toolkit present - see `docs/TOOLCHAIN.md`.
//! GPU-ness is a runtime skip, not a compile-time `cfg`.
//!
//! Every kernel here has a scalar twin in `xabe-dsp` and is tested against it
//! per kernel, on the same inputs, before anything is assembled from them.
//!
//! Two conventions:
//!
//! - **Tensors are `[channels, time]` or `[time, channels]`, flat, row-major**,
//!   exactly as on the CPU side. No padding, no swizzling. A layout change is
//!   an optimisation, and optimisations arrive with a measurement.
//! - **A null bias pointer means no bias.** `conv_post` in the decoder is the
//!   one convolution in the checkpoint without one, and threading an
//!   `Option` through to the device is worse than checking a pointer.

/// All device code, as one translation unit.
pub const SOURCE: &str = r#"

// ------------------------------------------------------------------- matmul
//
// A tiled matrix multiply on the tensor cores, for the transformer stages.
//
// `out[m][n] = sum_k a[m][k] * w[n][k]`, which is a linear layer with weights
// stored `[out, in]` - so `a` is row-major and `w` is column-major from the
// instruction's point of view, and `mma.sync.aligned.m16n8k8.row.col` is
// exactly that shape with no transpose anywhere.
//
// Why the tensor cores rather than a scalar fp32 tile: measured on this card,
// scalar fp32 runs at 17.9 TFLOP/s and `m16n8k8` at 99. The encoder is 32
// layers of 1500x1280x1280 and 1500x1280x5120, about 1.9 TFLOP in total, so
// the difference is the difference between a usable ASR and an unusable one.
//
// **Only two MMA shapes are reachable at sm_75**: `m8n8k16.s32.s8.s8.s32` and
// `m16n8k8.f32.f16.f16.f32`. `m16n8k16` assembles under NVRTC and is then
// rejected by ptxas - NVRTC success is not evidence of reachability. That, the
// fragment layouts below and the shared-memory stride argument are adapted from
// `llmxabe`, which has been running them on this card; see docs/KERNELS.md.
//
// **Operands are f16, accumulation is f32.** That is a precision decision, not
// an oversight: fp16 *accumulation* looks safe on random data at every depth
// and then breaks on adversarial input by two orders of magnitude, growing
// monotonically with the contraction length. The operand rounding this does
// cost is what the differential test against `xabe_dsp::linear` measures.

// Keys one `attn_decode` block covers, and the query heads a key-value head
// may serve there. Defined up here, beside the matmul's, because the Rust side
// finds a `#define` by scanning this string at compile time and a name at the
// far end of it costs more constant evaluation than rustc allows.
#define AD_CH 64
#define AD_GMAX 4
#define AD_CMAX 256

#define GEMM_WARPS 8
#define GEMM_MT    128
#define GEMM_NT    128
#define GEMM_KC    32
#define GEMM_MSTEPS (GEMM_MT / 16)
#define GEMM_KSTEPS (GEMM_KC / 8)
#define GEMM_NPW   (GEMM_NT / (GEMM_WARPS * 8))   // 8-wide n tiles per warp

// Elements of the contraction one thread unpacks from a K-quant per trip. The
// header decode is per call, so a longer run is fewer decodes for the same
// weights; sixteen is the ceiling, because a Q6_K scale group is sixteen
// elements and a run that crossed one would need two scales.
#define GEMM_QRUN  (((GEMM_KC % 16) == 0) ? 16 : 8)

// Weight-staging items one thread takes per trip. A compile-time count so the
// per-thread header cache below is indexed by a constant and stays in
// registers.
#define GEMM_BITER ((GEMM_NT * (GEMM_KC / GEMM_QRUN) + GEMM_WARPS * 32 - 1) \
                    / (GEMM_WARPS * 32))

// The block tile is chosen by *global* traffic, not by shared-memory capacity.
//
// A block reads the whole contraction for GEMM_MT rows of `a` and GEMM_NT rows
// of `w`, and does 2*MT*NT*k flops with them, so it performs
// `2*MT*NT/(MT+NT)` flops per float it reads. At 64x32 that is 43, and against
// 672 GB/s it caps the kernel at about 7 TFLOP/s - which is exactly where the
// first version measured. At 128x128 it is 128, capping it near 21.
//
// Measured on one Quadro RTX 8000, medians of 20, at 1500x1280x1280:
//
//   64x32,  4 warps    1.60 ms    3.1 TFLOP/s
//   128x128, 8 warps   see docs/BENCHMARKS.md
//
// This is why the tile is not a tuning knob to be swept blindly: the first
// arrangement was three times off the bandwidth its shape implied, and the
// arithmetic said so before any measurement did.

// Shared rows are addressed as 32-bit words, because a fragment load is two
// halves at a time and the k index of a fragment is always even.
//
// The stride carries four words of padding beyond the KC/2 a row needs, and
// that padding is the whole point. A fragment load has lane `l` read row
// `l >> 2` at word `l & 3`, so consecutive groups of four lanes are one row
// apart. With no padding at KC=32 the stride is 16 words, the eight groups
// land on banks {0,16,0,16,...}, and the load is four-way conflicted.
//
// Four words of padding fixes it at both tile depths that matter, which is why
// it is `+ 4` rather than a number chosen for one of them:
//
//   KC=32 -> stride 20 -> groups at banks 0,20,8,28,16,4,24,12
//   KC=64 -> stride 36 -> groups at banks 0,4,8,12,16,20,24,28
//
// Either way the eight groups of four cover all 32 banks exactly once, so the
// load is conflict-free.
#define GEMM_WSTRIDE (GEMM_KC / 2 + 4)

// NVRTC has no include path, so <cuda_fp16.h> is unreachable and every
// conversion is inline PTX. Packs two floats into one b32 as {lo, hi}, which is
// the order a fragment register wants: the low half is the smaller k.
__device__ __forceinline__ unsigned gemm_pack(float lo, float hi) {
    unsigned r;
    asm("{ .reg .f16 x, y; cvt.rn.f16.f32 x, %1; cvt.rn.f16.f32 y, %2; mov.b32 %0, {x, y}; }"
        : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

// One float to one half, round to nearest even - the same rounding
// `gemm_pack` does, so a tensor converted here and one staged from F32 are the
// same bits.
__device__ __forceinline__ unsigned short f32_to_f16(float v) {
    unsigned short r;
    asm("{ .reg .f16 x; cvt.rn.f16.f32 x, %1; mov.b16 %0, x; }" : "=h"(r) : "f"(v));
    return r;
}

// The inverse, for the scalar path. `gemm_pack`'s output and a pair of halves
// stored contiguously are the same 32 bits, which is the whole reason an f16
// weight needs no conversion at all on the tiled path.
__device__ __forceinline__ void gemm_unpack(unsigned p, float& lo, float& hi) {
    asm("{ .reg .f16 x, y;\n"
        "  mov.b32 {x, y}, %2;\n"
        "  cvt.f32.f16 %0, x;\n"
        "  cvt.f32.f16 %1, y; }"
        : "=f"(lo), "=f"(hi) : "r"(p));
}

// The two shared loads that feed `mma`, as one instruction each.
//
// `ldmatrix` exists to do exactly this: every lane hands it the address of one
// *row* of an 8x8 half tile, and it returns the tile already distributed in
// the register layout `mma` wants. The scalar loads it replaces were one
// 32-bit `ld.shared` per lane per fragment - 72 of them per lane per staged
// trip against 64 `mma`, in a loop measured at 41 ms of a 95 ms prefill with
// the staging deleted around it.
//
// The address a lane supplies is its row's base plus the k step, and both are
// 16-byte aligned: `GEMM_WSTRIDE` is a multiple of four words so a row starts
// aligned, and a k step is four words into it. `ldmatrix` requires that.
//
// Only the first 8 lanes' addresses matter for `.x1` and the first 16 for
// `.x2`; the rest are ignored, and are given an in-range address anyway rather
// than an out-of-bounds one that happens not to be read.
__device__ __forceinline__ void gemm_ld_a(
    const unsigned* row, unsigned& a0, unsigned& a1)
{
    unsigned p = (unsigned)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(a0), "=r"(a1) : "r"(p));
}

__device__ __forceinline__ unsigned gemm_ld_b(const unsigned* row)
{
    unsigned p = (unsigned)__cvta_generic_to_shared(row);
    unsigned b;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(b) : "r"(p));
    return b;
}

__device__ __forceinline__ void gemm_mma_step(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned b0)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

// The same product when there is only a handful of rows.
//
// `m16n8k8` has a 16-row M dimension, so at one token the tiled kernel fills
// one row of 128 and throws the rest away - measured 0.02 TFLOP/s against 23.8
// at encoder width. Decode is exactly that shape, one token at a time, tens of
// thousands of times per utterance, so it gets its own kernel rather than a
// tolerance for the waste.
//
// One warp per output channel, striding the contraction and reducing across
// the lanes. The weight reads are coalesced - lane `l` reads `w[col][l]` - and
// the activation row is broadcast, so it stays in cache across the whole grid.
//
// This path is **exact f32**: there are no tensor cores involved and so no
// operand rounding. That makes precision a function of shape, which is worth
// knowing rather than hiding - see `GEMV_MAX_M`.

// -------------------------------------------------------- quantized weights
//
// Block-quantized weights, unpacked *inside* the matmul instead of on the way
// onto the card. That is the whole point: `xabe-gguf` already decodes these
// formats, but it decodes them to f32 at load, so a 4-bit 13 B still occupied
// 26.5 GB of f16 once resident. Unpacking per use makes the resident copy the
// packed one - 7.9 GB for that same model - which is what lets the pipeline
// fit on a single card.
//
// The trade is ALU for bandwidth, and these kernels are bandwidth-bound, so it
// is the right way round: a Q4_K element is 4.5 bits against f16's 16, so the
// weight traffic falls by 3.6x while the arithmetic per element grows by a
// dozen integer ops that overlap with the loads.
//
// **Every layout below is transcribed from `xabe_gguf::dequant`**, which is
// itself read off `gguf-py/gguf/quants.py` - the code that wrote these files -
// and checked against it at exact equality on all ten formats. Transcribing a
// second time is a second chance to get the element *ordering* wrong, which is
// the trap `docs/MODEL.md` names and which produces a permuted tensor rather
// than an error, so the differential tests in `tests/quant.rs` compare against
// that same decoder rather than against reasoning.

#define QT_Q4_0  2
#define QT_Q4_1  3
#define QT_Q5_0  6
#define QT_Q5_1  7
#define QT_Q8_0  8
#define QT_Q2_K 10
#define QT_Q3_K 11
#define QT_Q4_K 12
#define QT_Q5_K 13
#define QT_Q6_K 14

// A little-endian f16 at `b[i]`, as f32. NVRTC has no <cuda_fp16.h>, so this
// is inline PTX like every other conversion in this file.
__device__ __forceinline__ float q_f16(const unsigned char* b, int i) {
    unsigned short h = (unsigned short)b[i] | ((unsigned short)b[i + 1] << 8);
    float r;
    asm("{ .reg .f16 x; mov.b16 x, %1; cvt.f32.f16 %0, x; }" : "=f"(r) : "h"(h));
    return r;
}

// The 6-bit scale and minimum pair `q` of eight, shared by Q4_K and Q5_K.
// Mirrors `xabe_gguf::dequant::scale_min`, which unrolls the same packing as
// four assignments per iteration over 0..4.
__device__ __forceinline__ void q_scale_min(
    const unsigned char* s, int q, unsigned char& sc, unsigned char& mn)
{
    if (q < 4) {
        sc = s[q] & 0x3F;
        mn = s[q + 4] & 0x3F;
    } else {
        int i = q - 4;
        sc = (s[i + 8] & 0x0F) | ((s[i] >> 2) & 0x30);
        mn = (s[i + 8] >> 4) | ((s[i + 4] >> 2) & 0x30);
    }
}

// The two halves of a 32-bit word, as floats.
__device__ __forceinline__ float q_half_lo(unsigned w) {
    float r;
    asm("{ .reg .f16 x; mov.b16 x, %1; cvt.f32.f16 %0, x; }"
        : "=f"(r) : "h"((unsigned short)(w & 0xFFFF)));
    return r;
}

__device__ __forceinline__ float q_half_hi(unsigned w) {
    float r;
    asm("{ .reg .f16 x; mov.b16 x, %1; cvt.f32.f16 %0, x; }"
        : "=f"(r) : "h"((unsigned short)(w >> 16)));
    return r;
}

// `q_scale_min` again, reading the twelve scale bytes out of the three words a
// single header load produced rather than out of memory. Same packing, same
// answer; `the_wide_kquant_matvec_agrees_with_the_f32_product` and the
// block-format tests hold both to the Rust in `xabe_gguf::dequant`.
__device__ __forceinline__ void q_scale_min_words(
    unsigned y, unsigned z, unsigned w, int q, unsigned char& sc, unsigned char& mn)
{
    // byte 0..3 in `y`, 4..7 in `z`, 8..11 in `w`.
    #define Q_SB(i) ((unsigned char)(((i) < 4 ? y : (i) < 8 ? z : w) >> ((((i) & 3)) << 3)))
    if (q < 4) {
        sc = Q_SB(q) & 0x3F;
        mn = Q_SB(q + 4) & 0x3F;
    } else {
        int i = q - 4;
        sc = (Q_SB(i + 8) & 0x0F) | ((Q_SB(i) >> 2) & 0x30);
        mn = (Q_SB(i + 8) >> 4) | ((Q_SB(i + 4) >> 2) & 0x30);
    }
    #undef Q_SB
}

// Element `j` of the block at `blk`. One switch, one case per format, in the
// same order as the Rust.
__device__ __forceinline__ float q_elem(int ty, const unsigned char* blk, int j) {
    switch (ty) {
    // d(2) + 16 packed nibbles. Low nibbles of all 16 bytes first, then high -
    // not low-then-high of each byte, which is the permutation trap.
    case QT_Q4_0: {
        float d = q_f16(blk, 0);
        const unsigned char* qs = blk + 2;
        unsigned char nib = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
        return d * (float)((int)nib - 8);
    }
    // Offset rather than centred: no -8.
    case QT_Q4_1: {
        float d = q_f16(blk, 0), m = q_f16(blk, 2);
        const unsigned char* qs = blk + 4;
        unsigned char nib = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
        return d * (float)nib + m;
    }
    // d(2) + qh(4, one bit per element) + nibbles.
    case QT_Q5_0: {
        float d = q_f16(blk, 0);
        unsigned qh = (unsigned)blk[2] | ((unsigned)blk[3] << 8)
                    | ((unsigned)blk[4] << 16) | ((unsigned)blk[5] << 24);
        const unsigned char* qs = blk + 6;
        unsigned char lo = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
        unsigned char hi = (unsigned char)((qh >> j) & 1);
        return d * (float)((int)(lo | (hi << 4)) - 16);
    }
    case QT_Q5_1: {
        float d = q_f16(blk, 0), m = q_f16(blk, 2);
        unsigned qh = (unsigned)blk[4] | ((unsigned)blk[5] << 8)
                    | ((unsigned)blk[6] << 16) | ((unsigned)blk[7] << 24);
        const unsigned char* qs = blk + 8;
        unsigned char lo = (j < 16) ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
        unsigned char hi = (unsigned char)((qh >> j) & 1);
        return d * (float)(lo | (hi << 4)) + m;
    }
    // The simple one: a scale and 32 signed bytes.
    case QT_Q8_0: {
        float d = q_f16(blk, 0);
        return d * (float)((int)(signed char)blk[2 + j]);
    }
    // scales(16) + qs(64) + d(2) + dmin(2).
    case QT_Q2_K: {
        const unsigned char* scales = blk;
        const unsigned char* qs = blk + 16;
        float d = q_f16(blk, 80), dmin = q_f16(blk, 82);
        int g = j / 16;
        float dl = d * (float)(scales[g] & 0x0F);
        float ml = dmin * (float)(scales[g] >> 4);
        int hi = j / 128, r = j % 128, s = r / 32, kk = r % 32;
        unsigned char q = (qs[hi * 32 + kk] >> (2 * s)) & 3;
        return dl * (float)q - ml;
    }
    // hmask(32) + qs(64) + scales(12) + d(2). The high bit's sense is
    // inverted - the offset applies when the mask bit is zero.
    case QT_Q3_K: {
        const unsigned char* hmask = blk;
        const unsigned char* qs = blk + 32;
        const unsigned char* sc = blk + 96;
        float d = q_f16(blk, 108);
        int g = j / 16;
        unsigned char low  = (g < 8) ? (sc[g] & 0x0F) : (sc[g - 8] >> 4);
        unsigned char high = (sc[8 + (g % 4)] >> (2 * (g / 4))) & 0x03;
        int scale = (int)(low | (high << 4)) - 32;
        int hi = j / 128, r = j % 128, s = r / 32, kk = r % 32;
        int ql = (int)((qs[hi * 32 + kk] >> (2 * s)) & 3);
        int qh = (int)(((hmask[kk] >> (j / 32)) & 1) ^ 1);
        return d * (float)scale * (float)(ql - (qh << 2));
    }
    // d(2) + dmin(2) + scales(12) + qs(128). Eight sub-blocks of 32.
    case QT_Q4_K: {
        float d = q_f16(blk, 0), dmin = q_f16(blk, 2);
        unsigned char sc, mn;
        q_scale_min(blk + 4, j / 32, sc, mn);
        const unsigned char* qs = blk + 16;
        int hi = j / 64, r = j % 64, s = r / 32, kk = r % 32;
        unsigned char q = (qs[hi * 32 + kk] >> (4 * s)) & 0x0F;
        return d * (float)sc * (float)q - dmin * (float)mn;
    }
    // Q4_K plus one high bit per element.
    case QT_Q5_K: {
        float d = q_f16(blk, 0), dmin = q_f16(blk, 2);
        int jj = j / 32;
        unsigned char sc, mn;
        q_scale_min(blk + 4, jj, sc, mn);
        const unsigned char* qh = blk + 16;
        const unsigned char* qs = blk + 48;
        int hi = j / 64, r = j % 64, s = r / 32, kk = r % 32;
        unsigned char lo = (qs[hi * 32 + kk] >> (4 * s)) & 0x0F;
        unsigned char bit = (qh[j % 32] >> jj) & 1;
        return d * (float)sc * (float)(lo | (bit << 4)) - dmin * (float)mn;
    }
    // ql(128) + qh(64) + scales(16, signed) + d(2), in the *device* layout,
    // not the file's. `Gpu::upload_quant` re-packs a Q6_K block on the way to
    // the card: the low nibbles are paired 32 elements apart the way Q4_K's
    // are, and the 2-bit high fields are packed one 16-element run to a word,
    // element `e` at bits `8*(e%4) + 2*(e/4)`. The file's own grouping - low
    // halves 64 apart, high fields four to a byte across 128 elements - made
    // every kernel that stages a run fetch twice what it used.
    case QT_Q6_K: {
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const unsigned char* scales = blk + 192;
        float d = q_f16(blk, 208);
        int g = j / 16;
        int p = j / 64, r = j % 64;
        int s = r / 32, h = (r / 16) & 1, e = r % 16;
        unsigned char lo = (ql[p * 32 + (r % 32)] >> (4 * s)) & 0x0F;
        unsigned char bits =
            (qh[p * 16 + h * 8 + s * 4 + (e % 4)] >> (2 * (e / 4))) & 0x03;
        int q = (int)(lo | (bits << 4)) - 32;
        return d * (float)((int)(signed char)scales[g]) * (float)q;
    }
    }
    return 0.0f;
}

// Element (row, kk) of a quantized `[n, k]` weight.
//
// `k` is a whole number of blocks, so a row starts on a block boundary and the
// division below is exact. GGUF guarantees that for the fastest-varying
// dimension, and `Gpu::gemm_batched` refuses the shape when it does not hold
// rather than reading across a row edge.
// Eight elements of one K-quant super-block, dotted with eight activations.
//
// `q_elem` re-derives a block's header for every element it returns - two f16
// scales, a six-bit sub-block scale, and four integer divisions - because it is
// written to be read against the format tables one case at a time. At 256
// elements to a super-block that is 256 header decodes where one is needed, and
// it is why the packed path measured 47 GB/s against a card that streams 672.
//
// These two hoist it. Every divisor below is a power of two and every quotient
// is loop-invariant, so the inner eight are a nibble extract and a fused
// multiply-add. Thirty-two lanes at eight elements each is one super-block per
// warp.
//
// Which eight differs between the formats, and for Q4_K that is the point. A
// Q4_K byte packs two elements 32 apart, so handing each lane eight *adjacent*
// elements fetches every byte twice - once per lane wanting one of its nibbles -
// and discards half of each shift-and-mask. `q4k_pair` gives a lane four whole
// bytes instead: one aligned 32-bit load and two float4 activation loads for
// the same eight elements, against eight byte loads before. It costs a second
// sub-block scale pair, because the two nibbles of a byte land in adjacent
// sub-blocks, and that is the whole of the cost. Measured standalone on this
// card at n=14336, k=4096: 384 us to 88 us.
//
// Q4_K and Q6_K only: between them they are every weight byte in both
// checkpoints this pipeline loads. Anything else still goes through `q_at`.
//
// `AVEC` says the activation row is 16-byte aligned. It is a template parameter
// rather than a test because the branch is loop-invariant and warp-uniform, and
// leaving it inside the loop measured 27% slower than hoisting it.
template <bool AVEC>
__device__ __forceinline__ float q4k_pair(
    const unsigned char* blk, const float* ap, int hi, int kk)
{
    float d = q_f16(blk, 0), dmin = q_f16(blk, 2);
    unsigned char s0, m0, s1, m1;
    // The low nibbles of these four bytes are elements hi * 64 + kk .. + 3 and
    // the high nibbles are those plus 32, which is the next sub-block along.
    q_scale_min(blk + 4, hi << 1, s0, m0);
    q_scale_min(blk + 4, (hi << 1) | 1, s1, m1);
    float ds0 = d * (float)s0, dm0 = dmin * (float)m0;
    float ds1 = d * (float)s1, dm1 = dmin * (float)m1;

    // Aligned: `blk` is a multiple of 144 bytes from a device allocation and
    // `kk` is a multiple of four, so the 32-bit read is on a 4-byte boundary.
    unsigned v = *(const unsigned*)(blk + 16 + (hi << 5) + kk);
    float al[4], ah[4];
    if (AVEC) {
        *(float4*)al = *(const float4*)ap;
        *(float4*)ah = *(const float4*)(ap + 32);
    } else {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            al[t] = ap[t];
            ah[t] = ap[t + 32];
        }
    }

    float acc = 0.0f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        unsigned byte = (v >> (t << 3)) & 0xFFu;
        acc += al[t] * (ds0 * (float)(byte & 0x0Fu) - dm0);
        acc += ah[t] * (ds1 * (float)(byte >> 4) - dm1);
    }
    return acc;
}

// Eight consecutive elements of a Q6_K super-block, in the device layout.
//
// `Gpu::upload_quant` re-packs Q6_K so that eight consecutive elements are
// eight consecutive `ql` bytes' nibbles of one rank and one word of `qh` -
// two word loads and one, all aligned, every fetched byte used. The file's
// own grouping needed a lane to own two columns across all four `qh` fields
// to avoid re-reading bytes, and even then the reads were 16-bit.
//
// `j` is the first element and must be a multiple of eight, so the run stays
// inside one 16-element scale group and one nibble rank.
__device__ __forceinline__ float q6k_dot8(
    const unsigned char* blk, const float* ap, int j)
{
    float d = q_f16(blk, 208);
    const signed char* scales = (const signed char*)(blk + 192);
    const int p = j >> 6, s = (j >> 5) & 1, h = (j >> 4) & 1, e0 = j & 15;
    // Both words are 4-byte aligned: the device stride is 224, the allocation
    // is 256-aligned, and `j` is a multiple of eight.
    const unsigned char* qlp = blk + (p << 5) + (h << 4) + e0;
    unsigned lw[2];
    lw[0] = *(const unsigned*)qlp;
    lw[1] = *(const unsigned*)(qlp + 4);
    const unsigned W =
        *(const unsigned*)(blk + 128 + (p << 4) + (h << 3) + (s << 2));
    const float ds = d * (float)((int)scales[j >> 4]);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int lo = ((int)(lw[i >> 2] >> ((i & 3) << 3)) >> (s << 2)) & 0x0F;
        const int e = e0 + i;
        int b = (int)(W >> (((e & 3) << 3) + ((e >> 2) << 1))) & 3;
        acc += ap[i] * (ds * (float)((lo | (b << 4)) - 32));
    }
    return acc;
}

// Eight consecutive elements of one K-quant super-block, header decoded once.
//
// `gemm` stages the weight tile two elements per thread and was reaching them
// `N` words of a byte run, read as one vector when it is aligned for one.
//
// The runs this reads are 16-byte aligned for every layout that reaches here -
// `Quant::device_stride` is a multiple of sixteen, the allocation is
// 256-aligned, and the element offset is a multiple of the run - but that is
// an argument, not a guarantee, so it is tested. The byte-at-a-time fallback
// is why nothing here casts a pointer it has not measured, the same rule
// `xabe-st` applies to a mapped header.
template <int N>
__device__ __forceinline__ void q_words(const unsigned char* p, unsigned* w)
{
    if (N == 4 && (((size_t)p) & 15) == 0) {
        *reinterpret_cast<uint4*>(w) = *reinterpret_cast<const uint4*>(p);
    } else if (N == 8 && (((size_t)p) & 15) == 0) {
        *reinterpret_cast<uint4*>(w)     = *reinterpret_cast<const uint4*>(p);
        *reinterpret_cast<uint4*>(w + 4) = *reinterpret_cast<const uint4*>(p + 16);
    } else if ((((size_t)p) & 3) == 0) {
        #pragma unroll
        for (int t = 0; t < N; ++t) {
            w[t] = *reinterpret_cast<const unsigned*>(p + 4 * t);
        }
    } else {
        #pragma unroll
        for (int t = 0; t < N; ++t) {
            w[t] = p[4 * t] | (p[4 * t + 1] << 8)
                 | (p[4 * t + 2] << 16) | (p[4 * t + 3] << 24);
        }
    }
}

// through `q_at`, which re-derives the block header for each - the same waste
// `q4k_pair` exists to remove from `gemv`, left behind in the tiled kernel
// because only the decode path had been measured. It is most of prefill.
//
// `RUN` *RUN-aligned* elements stay inside one 32-element sub-block and one
// 16-element Q6_K scale group, so the scales, the shift and the byte pointer
// are all shared and the inner RUN are a nibble extract and a multiply-add.
// Same addressing as `q6k_dot8` before it was regrouped; `gemm` wants a run of
// the contraction where `gemv` wants a run of whole bytes.
//
// The run length is a parameter because the header decode is per *call*, not
// per element: at eight, four threads decoded the same super-block header to
// cover one row of a `GEMM_KC = 32` trip. Sixteen halves that and reads the
// quants as one 16-byte load instead of two 8-byte ones. It cannot go higher
// while a Q6_K scale group is sixteen elements.
// The same, for a caller that already holds the header.
//
// A Q4_K super-block is 256 elements and a staged trip covers `GEMM_KC` of
// them, so eight consecutive trips read the same sixteen bytes - and a thread
// keeps the same weight row throughout, so it can read them once. Before this
// the weight side fetched 24 bytes for every 16 elements whose payload is 9,
// and the loads were 13 ms of a 69 ms prefill.
template <int RUN>
__device__ __forceinline__ void q4k_run_hdr(
    uint4 h, const unsigned char* blk, int j, float* e)
{
    float d = q_half_lo(h.x), dmin = q_half_hi(h.x);
    unsigned char sc, mn;
    q_scale_min_words(h.y, h.z, h.w, j >> 5, sc, mn);
    const unsigned char* qs = blk + 16 + ((j >> 6) << 5) + (j & 31);
    int shift = ((j >> 5) & 1) << 2;
    float ds = d * (float)sc, dm = dmin * (float)mn;
    unsigned w[RUN / 4];
    q_words<RUN / 4>(qs, w);
    #pragma unroll
    for (int t = 0; t < RUN; ++t) {
        unsigned byte = (w[t >> 2] >> ((t & 3) << 3)) & 0xFF;
        e[t] = ds * (float)((byte >> shift) & 0x0F) - dm;
    }
}

template <int RUN>
__device__ __forceinline__ void q4k_run(
    const unsigned char* blk, int j, float* e)
{
    // The whole header in one load. A Q4_K block opens with `d`, `dmin` and
    // the twelve packed scale bytes - exactly sixteen - and a block is
    // 16-byte aligned, so `d`, `dmin` and both 6-bit fields come out of one
    // `uint4` instead of the eight separate byte loads they used to cost.
    //
    // That mattered more than it looks: staging the weights was measured at 40
    // ms of an 88 ms prefill against 13 for the activations, and only 3 of it
    // was the arithmetic. The rest was this - narrow, scattered loads, two
    // threads of every warp re-reading the same header 144 bytes away from
    // its neighbours.
    float d, dmin;
    unsigned char sc, mn;
    if ((((size_t)blk) & 15) == 0) {
        uint4 h = *reinterpret_cast<const uint4*>(blk);
        d    = q_half_lo(h.x);
        dmin = q_half_hi(h.x);
        q_scale_min_words(h.y, h.z, h.w, j >> 5, sc, mn);
    } else {
        d = q_f16(blk, 0);
        dmin = q_f16(blk, 2);
        q_scale_min(blk + 4, j >> 5, sc, mn);
    }
    const unsigned char* qs = blk + 16 + ((j >> 6) << 5) + (j & 31);
    int shift = ((j >> 5) & 1) << 2;
    float ds = d * (float)sc, dm = dmin * (float)mn;
    // Eight nibbles come from eight consecutive bytes, which is eight byte
    // loads - one per element produced. Two word loads do the same work when
    // the run is word-aligned, and it is for every layout that reaches here:
    // `Quant::device_stride` is a multiple of sixteen, the allocation is
    // 256-aligned, and `j` is a multiple of eight. That is an argument, not a
    // guarantee, so the alignment is tested rather than assumed - the same
    // reason `xabe-st` refuses to cast a header it has not measured.
    unsigned w[RUN / 4];
    q_words<RUN / 4>(qs, w);
    #pragma unroll
    for (int t = 0; t < RUN; ++t) {
        unsigned byte = (w[t >> 2] >> ((t & 3) << 3)) & 0xFF;
        e[t] = ds * (float)((byte >> shift) & 0x0F) - dm;
    }
}

template <int RUN>
__device__ __forceinline__ void q6k_run(
    const unsigned char* blk, int j, float* e)
{
    const signed char* scales = (const signed char*)(blk + 192);
    float d = q_f16(blk, 208);
    // Device layout: the run's low nibbles are `RUN` consecutive bytes of one
    // rank, and its 2-bit high fields are one word, element `e` of the
    // 16-element run at bits `8*(e%4) + 2*(e/4)`. The file's own grouping
    // needed a second full-width read of `qh` per run.
    const int p = j >> 6, s = (j >> 5) & 1;
    const unsigned char* qlp = blk + (p << 5) + (j & 31);
    const unsigned W = *(const unsigned*)(
        blk + 128 + (p << 4) + (((j >> 4) & 1) << 3) + (s << 2));
    float dsc = d * (float)((int)scales[j >> 4]);
    unsigned lw[RUN / 4];
    q_words<RUN / 4>(qlp, lw);
    #pragma unroll
    for (int t = 0; t < RUN; ++t) {
        const int sft = (t & 3) << 3;
        const int ee = (j & 15) + t;
        int lo = (((int)(lw[t >> 2] >> sft) & 0xFF) >> (s << 2)) & 0x0F;
        int b = (int)(W >> (((ee & 3) << 3) + ((ee >> 2) << 1))) & 3;
        e[t] = dsc * (float)((lo | (b << 4)) - 32);
    }
}

// Quantises an activation row to int8 in groups of 32, with one f32 scale a
// group.
//
// This is the first thing in this engine that quantizes at *runtime*, and it
// exists for one reason: the packed mat-vec cannot use wide loads while the
// activation is f32. Measured on this card, a lane loading sixteen bytes of
// Q4_K quants reaches 578 GB/s against 440 for four bytes - but sixteen bytes
// is 32 elements, and 32 f32 activations is 128 bytes of scattered reads that
// cost more than the wide load wins. At int8 they are 32 bytes and two loads,
// and the dot product becomes four `dp4a` instead of 32 conversions and 64
// fused multiply-adds.
//
// Scale is max|a| / 127 over the group, so zero maps to zero and the sign is
// symmetric. A group that is entirely zero gets scale zero and quantises to
// zero, which is exact.
extern "C" __global__ void quantize_q8(
    const float* __restrict__ a, signed char* __restrict__ qa,
    int asc_off, int k, int rows, long sa, int m)
{
    float* asc = (float*)(qa + asc_off);
    const int groups = k >> 5;
    const int g = blockIdx.x * blockDim.y + threadIdx.y;
    if (g >= rows * groups) {
        return;
    }
    const int r = g / groups, j = g - r * groups;
    const int z = r / m, row = r - z * m;
    const float* src = a + (size_t)z * sa + (size_t)row * k + (j << 5);
    const int lane = threadIdx.x;
    const float v = src[lane];
    float mx = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    const float d = mx * (1.0f / 127.0f);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    qa[(size_t)r * k + (j << 5) + lane] = (signed char)__float2int_rn(v * inv);
    if (lane == 0) {
        asc[(size_t)r * groups + j] = d;
    }
}

// Sixteen bytes of one Q4_K super-block's quants against an int8 activation.
//
// Eight lanes cover a block's 128 quant bytes, so a warp spans four consecutive
// super-blocks and a lane produces 32 elements: the sixteen low nibbles and the
// sixteen high ones, 32 apart in the contraction. `slot` picks the 16-byte run,
// and the two runs of elements land in adjacent 32-element sub-blocks, so two
// scale pairs cover them - the same shape `q4k_pair` uses, four times as wide.
//
// The nibble masks are the point: `v & 0x0F0F0F0F` is already four int8 weights
// in one word, which is exactly what `dp4a` wants, and no element is ever
// converted to float. The block's minimum comes out as `dmin * mn * sum(a)`,
// and `sum(a)` is another `dp4a` against a word of ones.
__device__ __forceinline__ float q4k_wide(
    const unsigned char* blk, const signed char* xa, const float* asc,
    int slot, int jlo, int q0, int j0)
{
    uint4 q = *(const uint4*)(blk + 16 + (slot << 4));
    float d = q_f16(blk, 0), dmin = q_f16(blk, 2);
    unsigned char s0, m0, s1, m1;
    q_scale_min(blk + 4, q0, s0, m0);
    q_scale_min(blk + 4, q0 + 1, s1, m1);
    uint4 xl = *(const uint4*)(xa + j0);
    uint4 xh = *(const uint4*)(xa + j0 + 32);
    const unsigned* qw = (const unsigned*)&q;
    const unsigned* xlw = (const unsigned*)&xl;
    const unsigned* xhw = (const unsigned*)&xh;
    int dot0 = 0, sum0 = 0, dot1 = 0, sum1 = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        unsigned v = qw[w];
        dot0 = __dp4a((int)(v & 0x0F0F0F0Fu), (int)xlw[w], dot0);
        sum0 = __dp4a(0x01010101, (int)xlw[w], sum0);
        dot1 = __dp4a((int)((v >> 4) & 0x0F0F0F0Fu), (int)xhw[w], dot1);
        sum1 = __dp4a(0x01010101, (int)xhw[w], sum1);
    }
    float a0 = asc[j0 >> 5], a1 = asc[(j0 + 32) >> 5];
    return a0 * (d * (float)s0 * (float)dot0 - dmin * (float)m0 * (float)sum0)
         + a1 * (d * (float)s1 * (float)dot1 - dmin * (float)m1 * (float)sum1);
}

// Sixteen bytes of one Q6_K super-block's low quants and eight of its high
// bits, against an int8 activation.
//
// The same shape as `q4k_wide` and it needs the same thing to work: a 16-byte
// load has to be 16-byte aligned, and a Q6_K block is 210 bytes, so consecutive
// blocks in a file are aligned to 2 and nothing more. `Gpu::upload_quant` pads
// the stride to 224 and re-packs the block on the way - the device layout
// `q_elem` documents - which is the only place in the engine where what sits
// in VRAM is not byte-for-byte what sits in the file.
//
// Eight lanes cover a block. In the device layout a lane owns sixteen `ql`
// bytes whose two nibble ranks are the elements at `j0` and 32 further along -
// the same pairing Q4_K ships with - and the eight `qh` bytes beside them,
// one word per 16-element run, element `e` at bits `8*(e%4) + 2*(e/4)`. Two
// of the block's sixteen signed scales cover them. The file's own grouping
// made this read sixteen `qh` bytes and use half of each.
//
// The -32 bias is not applied per element. `sum(x)` comes out of a second
// `dp4a` against a word of ones and the bias is one multiply at the end, which
// is what keeps the inner loop to eight instructions.
__device__ __forceinline__ float q6k_wide(
    const unsigned char* blk, const signed char* xa, const float* asc,
    int qlo, int qho, int sc_lo, int j0)
{
    uint4 v = *(const uint4*)(blk + qlo);
    uint2 u = *(const uint2*)(blk + qho);
    float d = q_f16(blk, 208);
    const signed char* sc = (const signed char*)(blk + 192);
    float slo = (float)sc[sc_lo], shi = (float)sc[sc_lo + 2];
    uint4 xl = *(const uint4*)(xa + j0);
    uint4 xh = *(const uint4*)(xa + j0 + 32);
    const unsigned* vw = (const unsigned*)&v;
    const unsigned* xlw = (const unsigned*)&xl;
    const unsigned* xhw = (const unsigned*)&xh;
    int dot0 = 0, sum0 = 0, dot1 = 0, sum1 = 0;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        unsigned lo = (vw[w] & 0x0F0F0F0Fu)
                    | (((u.x >> (w << 1)) & 0x03030303u) << 4);
        unsigned hi = ((vw[w] >> 4) & 0x0F0F0F0Fu)
                    | (((u.y >> (w << 1)) & 0x03030303u) << 4);
        dot0 = __dp4a((int)lo, (int)xlw[w], dot0);
        sum0 = __dp4a(0x01010101, (int)xlw[w], sum0);
        dot1 = __dp4a((int)hi, (int)xhw[w], dot1);
        sum1 = __dp4a(0x01010101, (int)xhw[w], sum1);
    }
    float a0 = asc[j0 >> 5], a1 = asc[(j0 + 32) >> 5];
    return a0 * d * slo * (float)(dot0 - 32 * sum0)
         + a1 * d * shi * (float)(dot1 - 32 * sum1);
}

__device__ __forceinline__ float q_at(
    const unsigned char* w, int ty, int bs, int ts, long row, int k, int kk)
{
    long b = row * (long)(k / bs) + (long)(kk / bs);
    return q_elem(ty, w + b * (long)ts, kk % bs);
}

#define GEMV_WARPS 8

// `w` is either `const float*` or a packed `const __half*`, selected by
// `w_half`. The strides `sa`, `sw` and `so` count elements of the logical
// matrix in both cases, so a caller does not have to know which precision it
// handed over.
extern "C" __global__ __launch_bounds__(GEMV_WARPS * 32) void gemv(
    const void* __restrict__ a,
    const void* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int m, int k, int n,
    long sa, long sw, long so,
    int a_half, int w_half,
    // Zero when `w` is F32 or F16. Otherwise the ggml type id, and `w` points
    // at packed blocks: `q_bs` elements to a block, `q_ts` bytes to a block.
    int w_quant, int q_bs, int q_ts,
    // Elements between consecutive rows of `w`, when that is not `k`.
    //
    // The value cache is the reason this exists: it is one buffer of
    // `[kv_heads, head_dim, capacity]` and attention contracts over `tk`
    // positions of it, so a row of the operand is `capacity` apart and not
    // `tk`. F32 only - a packed weight is a checkpoint tensor and those are
    // always tight.
    int w_rs,
    // The activation again, int8 in groups of 32, or null. `quantize_q8` writes
    // it densely as `[batch, m, k]`, so it has its own addressing rather than
    // `sa`, and it writes the per-group scales into the same allocation at
    // `asc_off` bytes - one allocation rather than two, because at 225 of these
    // a token the allocation is a cost worth counting. Only the two K-quant
    // paths read either.
    const signed char* __restrict__ qa, int asc_off, int a_rows,
    // The epilogue, for a single-row product whose result goes somewhere
    // other than a fresh `[m, n]` buffer. `epi_act` 1 applies the exact GELU
    // to the sum; the rest place column `col` at
    // `o_off + col * o_cs + (col / o_hd) * o_hs` when `o_hd` is nonzero, which
    // is a head-major key cache when `o_hs` is a head's stride less one
    // position, and at `o_off + col * o_cs` otherwise, which is a transposed
    // value cache when `o_cs` is the capacity. The defaults `0, 1, 0, 0, 0`
    // are the plain store. See `Gpu::gemv_into`.
    int epi_act, int o_cs, int o_hs, int o_hd, long o_off)
{
    const int lane = threadIdx.x;
    const int col  = blockIdx.x * GEMV_WARPS + threadIdx.y;
    const int row  = blockIdx.y;
    if (col >= n || row >= m) {
        return;
    }

    // One independent product per blockIdx.z, at a fixed stride in each
    // operand. Attention is twenty of these per layer and the alternative is
    // twenty launches; the strides are separate arguments because a batched
    // score matrix and a batched context share no single stride.
    out += (size_t)blockIdx.z * so;

    // The activation is one row, broadcast across the whole grid, so it stays
    // in cache and its precision costs nothing in bandwidth. `a_half` is
    // accepted for symmetry with the tiled kernel and unpacked the same way.
    const float* af = (const float*)a + (size_t)blockIdx.z * sa + (size_t)row * k;
    const unsigned* ahr =
        (const unsigned*)a + (size_t)blockIdx.z * (sa >> 1) + (size_t)row * (k >> 1);
    const float* asc = qa ? (const float*)(qa + asc_off) : (const float*)0;
    float acc = 0.0f;
    if (w_quant) {
        // Blocks tile along the contraction, so a row is `k / q_bs` of them.
        // The stride `sw` counts elements of the logical matrix like every
        // other path, and is converted to bytes here rather than at the call.
        const unsigned char* wq = (const unsigned char*)w
            + (size_t)blockIdx.z * (size_t)(sw / q_bs) * (size_t)q_ts;
        // The two K-quants the checkpoints actually use get a path that decodes
        // a header once per eight elements instead of once per element. One
        // super-block per warp, eight contiguous elements per lane.
        if (qa && q_bs == 256 && w_quant == QT_Q4_K) {
            // Four super-blocks a warp, sixteen quant bytes a lane. See
            // `q4k_wide` for why this needs the int8 activation to pay.
            const int nb = k >> 8;
            const int sub = lane >> 3, slot = lane & 7;
            const int jlo = (slot >> 1) * 64 + (slot & 1) * 16;
            const int q0 = jlo >> 5;
            // `a_rows` and not `m`: the codes are dense `[batch, a_rows, k]`,
            // and a batch that shares one activation - the attention
            // projections do - quantizes it once and passes zero here.
            const size_t r = (size_t)blockIdx.z * (size_t)a_rows + row;
            const signed char* xa = qa + r * k;
            const float* xs = asc + r * (k >> 5);
            const unsigned char* wc = wq + (size_t)col * nb * (size_t)q_ts;
            // A warp covers four super-blocks a trip, and a row is not always a
            // multiple of four of them - the 13 B translator's down projection
            // contracts over 13824, which is 54. The lanes past the end sit out
            // rather than the row being refused: their contribution is a
            // separate term of the warp reduction, so skipping it is exact.
            for (int b = 0; b < nb; b += 4) {
                if (b + sub < nb) {
                    acc += q4k_wide(wc + (size_t)(b + sub) * (size_t)q_ts,
                                    xa, xs, slot, jlo, q0, ((b + sub) << 8) + jlo);
                }
            }
        } else if (!a_half && q_bs == 256 && w_quant == QT_Q4_K) {
            const int nb = k >> 8;
            const int hi = lane >> 3, kk = (lane & 7) << 2;
            const float* av = af + (hi << 6) + kk;
            // Two loops rather than a test inside one: see `q4k_pair`.
            if ((((unsigned long long)af) & 15ull) == 0ull) {
                for (int b = 0; b < nb; ++b) {
                    acc += q4k_pair<true>(
                        wq + ((size_t)col * nb + b) * (size_t)q_ts,
                        av + (b << 8), hi, kk);
                }
            } else {
                for (int b = 0; b < nb; ++b) {
                    acc += q4k_pair<false>(
                        wq + ((size_t)col * nb + b) * (size_t)q_ts,
                        av + (b << 8), hi, kk);
                }
            }
        } else if (qa && q_bs == 256 && w_quant == QT_Q6_K) {
            const int nb = k >> 8;
            const int sub = lane >> 3, slot = lane & 7;
            // The device layout indexes like Q4_K: a lane's sixteen `ql` bytes
            // carry elements `jlo` and `jlo + 32`, and its eight `qh` bytes
            // sit beside them.
            const int pp = slot >> 1, hh = slot & 1;
            const int qlo = (pp << 5) + (hh << 4);
            const int qho = 128 + (pp << 4) + (hh << 3);
            const int sc_lo = (pp << 2) + hh;
            const int jlo = (pp << 6) + (hh << 4);
            const size_t r = (size_t)blockIdx.z * (size_t)a_rows + row;
            const signed char* xa = qa + r * k;
            const float* xs = asc + r * (k >> 5);
            const unsigned char* wc = wq + (size_t)col * nb * (size_t)q_ts;
            for (int b = 0; b < nb; b += 4) {
                if (b + sub < nb) {
                    acc += q6k_wide(wc + (size_t)(b + sub) * (size_t)q_ts,
                                    xa, xs, qlo, qho, sc_lo,
                                    ((b + sub) << 8) + jlo);
                }
            }
        } else if (!a_half && q_bs == 256 && w_quant == QT_Q6_K) {
            const int nb = k >> 8;
            const int j = lane << 3;
            const float* av = af + j;
            for (int b = 0; b < nb; ++b) {
                acc += q6k_dot8(wq + ((size_t)col * nb + b) * (size_t)q_ts,
                                av + (b << 8), j);
            }
        } else {
            for (int i = lane; i < k; i += 32) {
                float av;
                if (a_half) {
                    float lo, hi;
                    gemm_unpack(ahr[i >> 1], lo, hi);
                    av = (i & 1) ? hi : lo;
                } else {
                    av = af[i];
                }
                acc += av * q_at(wq, w_quant, q_bs, q_ts, col, k, i);
            }
        }
    } else if (w_half) {
        // Two halves to a word, and `k` is even whenever a weight is stored
        // this way - every contraction in the model is.
        const int kh = k >> 1;
        // `w_rs` is honoured here as well as on the f32 path: an f16 value
        // cache is a row of `capacity` halves and the contraction is `tk` of
        // them. Both are even, so the word stride is the element stride
        // halved.
        const unsigned* wv = (const unsigned*)w + (size_t)blockIdx.z * (sw >> 1)
                           + (size_t)col * (size_t)(w_rs ? (w_rs >> 1) : kh);
        for (int i = lane; i < kh; i += 32) {
            float lo, hi, alo, ahi;
            gemm_unpack(wv[i], lo, hi);
            if (a_half) {
                gemm_unpack(ahr[i], alo, ahi);
            } else {
                alo = af[2 * i];
                ahi = af[2 * i + 1];
            }
            acc += alo * lo + ahi * hi;
        }
        // The odd tail, for the same reason `gemv_rows` has one: a multi-head
        // model decodes with one row, and its value cache is contracted over
        // however many positions exist. An f16 *weight* is never odd; an f16
        // cache is, half the time. Refused when the activation is also f16,
        // which has no layout for it either way.
        if ((k & 1) && !a_half && lane == 0) {
            float lo, hi;
            gemm_unpack(wv[kh], lo, hi);
            acc += af[k - 1] * lo;
        }
    } else {
        const float* wv =
            (const float*)w + (size_t)blockIdx.z * sw + (size_t)col * (w_rs ? w_rs : k);
        for (int i = lane; i < k; i += 32) {
            float av = a_half ? 0.0f : af[i];
            if (a_half) {
                float lo, hi;
                gemm_unpack(ahr[i >> 1], lo, hi);
                av = (i & 1) ? hi : lo;
            }
            acc += av * wv[i];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        float v = acc + (bias ? bias[col] : 0.0f);
        if (epi_act == 1) {
            // `act_gelu`'s expression, character for character.
            v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        }
        size_t idx;
        if (o_hd) {
            idx = (size_t)o_off + (size_t)col * o_cs + (size_t)(col / o_hd) * o_hs;
        } else if (o_cs != 1) {
            idx = (size_t)o_off + (size_t)col * o_cs;
        } else {
            idx = (size_t)o_off + (size_t)row * n + col;
        }
        out[idx] = v;
    }
}


// The rows of a mat-vec against an *unpacked* weight, carried by one warp
// instead of one block each.
//
// `gemv` puts row `r` at `blockIdx.y`, so `m` rows read the weight `m` times.
// For a checkpoint tensor that is the right trade - the weight is the whole
// traffic and `m` is one - but attention is the case where it is not: the
// "weight" is the KV cache, and the `m` rows are the query heads of a
// grouped-query group, which exist precisely because they share it. Measured
// on this card at the 8 B model's decode shapes, four rows cost 2.4x one row
// rather than the 1.0x sharing would give, so L2 was not absorbing the
// re-reads. docs/BENCHMARKS.md has the sweep.
//
// So this reads `wv[i]` once and spends it on every row. The arithmetic is
// element-for-element what `gemv` does - the same products accumulated in the
// same order into the same per-lane partial, and the same reduction - which is
// why the differential test can demand exact equality between the two.
//
// f32 both sides, because that is what the caches are and because a packed
// weight has no case here: it is a checkpoint tensor read once a token.
#define GEMV_ROWS_MAX 4
extern "C" __global__ __launch_bounds__(GEMV_WARPS * 32) void gemv_rows(
    const float* __restrict__ a,
    const void* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int m, int k, int n,
    long sa, long sw, long so,
    // Elements between consecutive rows of `w`. See the note in `gemv`: the
    // value cache is why it exists.
    int w_rs,
    // Whether `w` is f16, two halves to a word. `k` and `w_rs` are then both
    // even - a cache is head_dim wide or capacity apart and both are.
    int w_half)
{
    const int lane = threadIdx.x;
    const int col  = blockIdx.x * GEMV_WARPS + threadIdx.y;
    if (col >= n) {
        return;
    }
    out += (size_t)blockIdx.z * so;
    const float* af = a + (size_t)blockIdx.z * sa;

    float acc[GEMV_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
        acc[r] = 0.0f;
    }
    // `m` is a runtime count but never above GEMV_ROWS_MAX, so the row loop is
    // unrolled against the bound and predicated on the count. An unrolled body
    // is what makes the single weight load pay: the four products issue back
    // to back off one register.
    if (w_half) {
        // A word is two consecutive elements of the contraction, so a lane
        // covers 64 of them a trip rather than 32 - which is the point, and
        // why this is not the f32 loop with a conversion in it.
        const int kh = k >> 1;
        const unsigned* wv = (const unsigned*)w + (size_t)blockIdx.z * (sw >> 1)
                           + (size_t)col * (size_t)(w_rs ? (w_rs >> 1) : kh);
        for (int i = lane; i < kh; i += 32) {
            float lo, hi;
            gemm_unpack(wv[i], lo, hi);
            #pragma unroll
            for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
                if (r < m) {
                    const float* ar = af + (size_t)r * k + 2 * i;
                    acc[r] += ar[0] * lo + ar[1] * hi;
                }
            }
        }
        // An odd contraction, which the other f16 paths refuse and this one
        // has to take: decoding contracts the value cache over the 1, 2, 3...
        // positions emitted so far and half of those are odd. The last
        // element is the low half of a word whose high half is a position the
        // activation does not have, so it is read alone rather than as a pair.
        // In bounds because a capacity is even and `k` never exceeds it.
        if ((k & 1) && lane == 0) {
            float lo, hi;
            gemm_unpack(wv[kh], lo, hi);
            #pragma unroll
            for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
                if (r < m) {
                    acc[r] += af[(size_t)r * k + k - 1] * lo;
                }
            }
        }
    } else {
        const float* wv = (const float*)w + (size_t)blockIdx.z * sw
                        + (size_t)col * (size_t)(w_rs ? w_rs : k);
        for (int i = lane; i < k; i += 32) {
            const float wi = wv[i];
            #pragma unroll
            for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
                if (r < m) {
                    acc[r] += af[(size_t)r * k + i] * wi;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
        if (r >= m) {
            break;
        }
        float v = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (lane == 0) {
            out[(size_t)r * n + col] = v + (bias ? bias[col] : 0.0f);
        }
    }
}


extern "C" __global__ __launch_bounds__(GEMM_WARPS * 32) void gemm(
    const void* __restrict__ a,
    const void* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int m, int k, int n,
    long sa, long sw, long so,
    int a_half, int w_half,
    // Zero when `w` is F32 or F16. Otherwise the ggml type id, and `w` points
    // at packed blocks: `q_bs` elements to a block, `q_ts` bytes to a block.
    int w_quant, int q_bs, int q_ts,
    // Elements between consecutive rows of `w`, when that is not `k`. See the
    // note in `gemv`; f32 only.
    int w_rs,
    // How many ways the contraction is split across `blockIdx.z`. One is the
    // ordinary matmul and the only case that writes `out` directly; above one,
    // each slice writes its own partial into `partial` and `gemm_reduce` sums
    // them. See the note above `partial`.
    int ksplit,
    // `[ksplit, batch, m, n]`, written instead of `out` when `ksplit > 1`.
    float* __restrict__ partial)
{
    // Aligned for the quad-wide staging stores: `GEMM_WSTRIDE` is a multiple
    // of four words, so a row start is 16-byte aligned when the array is.
    __shared__ __align__(16) unsigned as[GEMM_MT * GEMM_WSTRIDE];
    __shared__ __align__(16) unsigned bs[GEMM_NT * GEMM_WSTRIDE];

    // See the note in `gemv`: blockIdx.z selects one product of a batch, and
    // above it one slice of the contraction. Slice is the slower axis so that
    // a batch's blocks stay adjacent, as they were before the split existed.
    const int slice = (int)(blockIdx.z / (gridDim.z / ksplit));
    const int bat   = (int)(blockIdx.z % (gridDim.z / ksplit));

    out += (size_t)bat * so;
    const float*    af = (const float*)a    + (size_t)bat * sa;
    const unsigned* ah = (const unsigned*)a + (size_t)bat * (sa >> 1);
    const float*    wf = (const float*)w    + (size_t)bat * sw;
    const unsigned* wh = (const unsigned*)w + (size_t)bat * (sw >> 1);
    // Guarded because `q_bs` is zero on the unquantized paths, where this
    // pointer is never read.
    const unsigned char* wq = (const unsigned char*)w
        + (size_t)bat * (size_t)(q_bs ? (sw / q_bs) * (long)q_ts : 0);

    // Each slice takes a whole number of staged trips, so no slice boundary
    // falls inside a `GEMM_KC` tile and the staging loop is unchanged. The last
    // slice is short, or empty when `k` does not divide evenly - an empty slice
    // still writes its zeroed accumulator, which the reduction needs.
    const int kstep = ((k + ksplit - 1) / ksplit + GEMM_KC - 1) / GEMM_KC * GEMM_KC;
    const int kbeg  = slice * kstep;
    const int kend  = min(k, kbeg + kstep);

    const int lane = threadIdx.x;          // 0..31
    const int warp = threadIdx.y;          // 0..GEMM_WARPS-1
    const int tid  = warp * 32 + lane;
    const int g    = lane >> 2;            // groupID
    const int tg   = lane & 3;             // threadID_in_group

    const int m0 = blockIdx.y * GEMM_MT;
    const int n0 = blockIdx.x * GEMM_NT;

    float acc[GEMM_MSTEPS][GEMM_NPW][4];
    #pragma unroll
    for (int i = 0; i < GEMM_MSTEPS; ++i) {
        #pragma unroll
        for (int j = 0; j < GEMM_NPW; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
    }

    // One header per weight row per *super-block*, not per trip. `q_fast` and
    // the row a thread stages are both fixed for the whole loop, so the only
    // thing that changes is which 256-element super-block `kk` falls in - once
    // every eight trips at `GEMM_KC = 32`. See `q4k_run_hdr`.
    uint4 bhdr[GEMM_BITER];
    long  bhdr_sb[GEMM_BITER];
    #pragma unroll
    for (int u = 0; u < GEMM_BITER; ++u) {
        bhdr_sb[u] = -1;
    }

    for (int kc = kbeg; kc < kend; kc += GEMM_KC) {
        // Stage both tiles as f16. Out-of-range rows and columns are zeroed
        // rather than clamped: a zero contributes nothing to the dot product,
        // where a clamped duplicate would contribute the wrong thing.
        //
        // The pair of floats is read as one `float2` on the common path. Two
        // scalar loads is two memory transactions where one would do, and the
        // staging is on the critical path of every trip - the scalar fallback
        // exists for the ragged tail, where the second float is past the end of
        // the contraction, and for an odd `k`.
        //
        // An odd `k` is not hypothetical: decoding attends over the tokens
        // emitted so far, so the contraction is 1, 2, 3, ... and half of those
        // are odd. `float2` wants an 8-byte alignment and the offset is
        // `row * k + kk` with `kk` even, so an odd `k` misaligns every row
        // after the first. Taking the scalar path for the whole trip costs
        // nothing that can be measured - every contraction big enough to care
        // about (1280, 5120, 1500, 240, 3840) is even.
        const bool whole = (kc + GEMM_KC <= kend) && ((k & 1) == 0);

        // Four staged words at a time.
        //
        // A row is `GEMM_KC / 2` words and that is a multiple of four, so a
        // quad never straddles two rows, and `GEMM_WSTRIDE` is a multiple of
        // four so every row starts 16-byte aligned. The eight separate loads
        // and eight separate stores this replaces were, together with the
        // weight side, the largest single cost in the kernel: with the whole
        // staging removed - wrong results, timing only - the same mma loop ran
        // a prefill in 41 ms against 107 with it.
        //
        // The f16 source is read as one `uint4` when it is aligned and the
        // quad is wholly inside the contraction; otherwise word at a time. The
        // f32 source needs two `float4` loads for the same four output words,
        // and falls back the same way. Alignment is tested, not assumed.
        for (int q = tid; q < GEMM_MT * (GEMM_KC / 8); q += GEMM_WARPS * 32) {
            const int row = q / (GEMM_KC / 8);
            const int j   = (q % (GEMM_KC / 8)) * 4;
            const int kk  = kc + 2 * j;
            uint4 v = make_uint4(0, 0, 0, 0);
            unsigned* out4 = &as[row * GEMM_WSTRIDE + j];
            if (m0 + row < m) {
                if (a_half) {
                    const int kh = k >> 1;
                    const int aj = (kc >> 1) + j;
                    const unsigned* src = ah + (size_t)(m0 + row) * kh + aj;
                    if (aj + 3 < kh && (((size_t)src) & 15) == 0) {
                        v = *reinterpret_cast<const uint4*>(src);
                    } else {
                        unsigned* w = &v.x;
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            if (aj + t < kh) w[t] = src[t];
                        }
                    }
                } else {
                    const float* src = af + (size_t)(m0 + row) * k + kk;
                    float e[8];
                    if (whole && (((size_t)src) & 15) == 0) {
                        const float4* s4 = reinterpret_cast<const float4*>(src);
                        *reinterpret_cast<float4*>(e)     = s4[0];
                        *reinterpret_cast<float4*>(e + 4) = s4[1];
                    } else {
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            e[t] = (kk + t < k) ? src[t] : 0.0f;
                        }
                    }
                    unsigned* w = &v.x;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        w[t] = gemm_pack(e[2 * t], e[2 * t + 1]);
                    }
                }
            }
            *reinterpret_cast<uint4*>(out4) = v;
        }
        // The two K-quants every checkpoint here uses stage `GEMM_QRUN`
        // elements a thread so the header is decoded once for all of them.
        // `GEMM_QRUN` divides `GEMM_KC` and `kc` is a multiple of `GEMM_KC`,
        // so `kk` is run-aligned and the run cannot straddle a sub-block.
        const bool q_fast = w_quant && q_bs == 256
            && (w_quant == QT_Q4_K || w_quant == QT_Q6_K);
        if (q_fast) {
            #pragma unroll
            for (int u = 0; u < GEMM_BITER; ++u) {
                const int i = tid + u * (GEMM_WARPS * 32);
                if (i >= GEMM_NT * (GEMM_KC / GEMM_QRUN)) continue;
                int row = i / (GEMM_KC / GEMM_QRUN);
                int jq  = i % (GEMM_KC / GEMM_QRUN);
                int kk  = kc + GEMM_QRUN * jq;
                float e[GEMM_QRUN];
                #pragma unroll
                for (int t = 0; t < GEMM_QRUN; ++t) {
                    e[t] = 0.0f;
                }
                if (n0 + row < n) {
                    if (kk + GEMM_QRUN - 1 < k) {
                        long nb = k / 256;
                        const unsigned char* blk = wq
                            + ((size_t)(n0 + row) * nb + (kk >> 8)) * (size_t)q_ts;
                        if (w_quant == QT_Q4_K) {
                            const long sb = (long)(n0 + row) * nb + (kk >> 8);
                            if (sb != bhdr_sb[u]) {
                                bhdr[u] = (((size_t)blk) & 15) == 0
                                    ? *reinterpret_cast<const uint4*>(blk)
                                    : make_uint4(0, 0, 0, 0);
                                bhdr_sb[u] = ((((size_t)blk) & 15) == 0) ? sb : -1;
                            }
                            if (bhdr_sb[u] == sb) {
                                q4k_run_hdr<GEMM_QRUN>(bhdr[u], blk, kk & 255, e);
                            } else {
                                // Unaligned block: no cache, the general path.
                                q4k_run<GEMM_QRUN>(blk, kk & 255, e);
                            }
                        } else {
                            q6k_run<GEMM_QRUN>(blk, kk & 255, e);
                        }
                    } else {
                        // The tail of a contraction that is not a whole tile.
                        #pragma unroll
                        for (int t = 0; t < GEMM_QRUN; ++t) {
                            if (kk + t < k) {
                                e[t] = q_at(wq, w_quant, q_bs, q_ts, n0 + row, k, kk + t);
                            }
                        }
                    }
                }
                // The words a K-quant run produces are contiguous and
                // quad-aligned, for the same reason the activation's are.
                #pragma unroll
                for (int h = 0; h < GEMM_QRUN / 8; ++h) {
                    uint4 v;
                    unsigned* w = &v.x;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        w[t] = gemm_pack(e[8 * h + 2 * t], e[8 * h + 2 * t + 1]);
                    }
                    *reinterpret_cast<uint4*>(
                        &bs[row * GEMM_WSTRIDE + (GEMM_QRUN / 2) * jq + 4 * h]) = v;
                }
            }
        }
        for (int i = q_fast ? GEMM_NT * (GEMM_KC / 2) : tid;
             i < GEMM_NT * (GEMM_KC / 2); i += GEMM_WARPS * 32) {
            int row = i / (GEMM_KC / 2);
            int j   = i % (GEMM_KC / 2);
            int kk  = kc + 2 * j;
            unsigned packed = 0;
            if (w_quant) {
                // Two elements per staged word, each unpacked from its block.
                // Out of range stays zero, for the same reason the other paths
                // zero rather than clamp: zero contributes nothing.
                float lo = 0.0f, hi = 0.0f;
                if (n0 + row < n) {
                    if (kk     < k) lo = q_at(wq, w_quant, q_bs, q_ts, n0 + row, k, kk);
                    if (kk + 1 < k) hi = q_at(wq, w_quant, q_bs, q_ts, n0 + row, k, kk + 1);
                }
                packed = gemm_pack(lo, hi);
            } else if (w_half) {
                // No conversion at all. `gemm_pack` produces {f16(lo), f16(hi)}
                // in one b32, and two halves stored contiguously little-endian
                // are those same 32 bits - so an f16 weight is staged with a
                // plain 32-bit load. Half the global traffic of the F32 path
                // for arithmetic that was identical anyway, because the F32
                // path rounds to f16 here regardless.
                const int kh = k >> 1;
                const int wj = (kc >> 1) + j;
                if (n0 + row < n && wj < kh) {
                    packed = wh[(size_t)(n0 + row) * kh + wj];
                }
            } else {
                float lo = 0.0f, hi = 0.0f;
                if (n0 + row < n) {
                    const float* src = wf + (size_t)(n0 + row) * (w_rs ? w_rs : k) + kk;
                    if (whole) {
                        float2 v = *reinterpret_cast<const float2*>(src);
                        lo = v.x;
                        hi = v.y;
                    } else {
                        if (kk     < k) lo = src[0];
                        if (kk + 1 < k) hi = src[1];
                    }
                }
                packed = gemm_pack(lo, hi);
            }
            bs[row * GEMM_WSTRIDE + j] = packed;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < GEMM_KSTEPS; ++ks) {
            // Each B fragment is reused across every m tile and each A fragment
            // across every n tile, so one shared load feeds several
            // instructions. That reuse is what keeps this off the shared-load
            // limit once the global one has been raised.
            // `ldmatrix` wants a row index per lane, where the scalar load
            // wanted an element index: lanes 0-7 name the eight rows of one
            // 8x8 tile, lanes 8-15 the next.
            const int lrow = lane & 15;
            unsigned b0[GEMM_NPW];
            #pragma unroll
            for (int nt = 0; nt < GEMM_NPW; ++nt) {
                b0[nt] = gemm_ld_b(
                    &bs[((warp * GEMM_NPW + nt) * 8 + (lane & 7)) * GEMM_WSTRIDE + 4 * ks]);
            }
            #pragma unroll
            for (int ms = 0; ms < GEMM_MSTEPS; ++ms) {
                unsigned a0, a1;
                gemm_ld_a(&as[(16 * ms + lrow) * GEMM_WSTRIDE + 4 * ks], a0, a1);
                #pragma unroll
                for (int nt = 0; nt < GEMM_NPW; ++nt) {
                    gemm_mma_step(acc[ms][nt][0], acc[ms][nt][1],
                                  acc[ms][nt][2], acc[ms][nt][3], a0, a1, b0[nt]);
                }
            }
        }
        __syncthreads();
    }

    // D is 16x8: d0 is row `g` column `2*tg`, d2 is row `g+8`. The lane's `g`
    // means the N index when loading B and the M index when storing D; that is
    // the instruction redistributing, not a mistake.
    // A split writes its raw partial sum: the bias is a constant, so adding it
    // in every slice would add it `ksplit` times. `gemm_reduce` adds it once.
    if (ksplit > 1) {
        out = partial + ((size_t)slice * (gridDim.z / ksplit) + bat) * (size_t)m * n;
    }
    #pragma unroll
    for (int nt = 0; nt < GEMM_NPW; ++nt) {
        const int col0 = n0 + (warp * GEMM_NPW + nt) * 8 + 2 * tg;
        const float bias0 = (bias && ksplit == 1 && col0     < n) ? bias[col0]     : 0.0f;
        const float bias1 = (bias && ksplit == 1 && col0 + 1 < n) ? bias[col0 + 1] : 0.0f;
        #pragma unroll
        for (int ms = 0; ms < GEMM_MSTEPS; ++ms) {
            int row0 = m0 + 16 * ms + g;
            int row1 = row0 + 8;
            if (row0 < m) {
                if (col0     < n) out[(size_t)row0 * n + col0]     = acc[ms][nt][0] + bias0;
                if (col0 + 1 < n) out[(size_t)row0 * n + col0 + 1] = acc[ms][nt][1] + bias1;
            }
            if (row1 < m) {
                if (col0     < n) out[(size_t)row1 * n + col0]     = acc[ms][nt][2] + bias0;
                if (col0 + 1 < n) out[(size_t)row1 * n + col0 + 1] = acc[ms][nt][3] + bias1;
            }
        }
    }
}

// Sums the slices a split-k `gemm` produced and adds the bias.
//
// The sum is ordered - slice 0 first - rather than an `atomicAdd` race, so the
// result does not depend on how the blocks happened to be scheduled. A matmul
// that returns slightly different numbers from one run to the next would make
// every differential threshold in the workspace a coin toss.
extern "C" __global__ void gemm_reduce(
    const float* __restrict__ partial, const float* __restrict__ bias,
    float* __restrict__ out, int mn, int n, int batch, int ksplit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mn * batch) return;
    float acc = 0.0f;
    for (int s = 0; s < ksplit; ++s) {
        acc += partial[(size_t)s * mn * batch + i];
    }
    out[i] = acc + (bias ? bias[i % n] : 0.0f);
}
// ------------------------------------------------------- the integer matmul
//
// The same product as `gemm`, on the int8 tensor cores instead of the f16 ones.
//
// Measured on this card, `m16n8k8.f32.f16.f16.f32` runs at 102.3 TFLOP/s and
// `m8n8k16.s32.s8.s8` at twice that - not the 65.3 and four times this comment
// used to claim, which assumed a half-rate f32 accumulate that Quadro Turing
// does not have. Twice the arithmetic rate is still the largest lever on this
// card and llama.cpp takes it, which is why this path exists; what the old
// numbers also implied - that the f16 kernel had nothing left - does not
// follow. See docs/KERNELS.md.
//
// This is **the engine's second deliberate approximation**, and a
// larger one than the mat-vec's: both operands are quantized, where the f16
// kernel rounded a weight that was already 4-bit and left the arithmetic exact
// to f32 accumulation.
//
// # Why the shape works out
//
// `GEMM_I8_KC` is 32 elements a trip, and that is the number that makes the
// scales constant rather than a convenience:
//
// - A Q4_K sub-block is 32 elements with one `(d*sc, dmin*mn)` pair, so a trip
//   sees exactly one pair per weight row.
// - `xabe_dsp::quantize_q8` quantizes activations in groups of 32, so a trip
//   sees exactly one scale per activation row.
//
// The integer accumulator therefore runs a whole `mma` step - sixteen
// elements, the finest group Q6_K has - before anything is scaled.
//
// # The minimum is a rank-one correction, not a matmul
//
// Q4_K is `w = ds*q - dm` with `q` in 0..15, so
//
//     sum_k a_k w_k  =  ds * sum_k(a_k q_k)  -  dm * sum_k(a_k)
//
// The first term is what the integer `mma` computes. The second needs only
// `sum_k a_k` over the step, which depends on the activation row and not on the
// weight column - so it is computed once per row while staging and applied as
// one multiply-add per output. Q6_K carries no minimum at all: its value is
// `d * sc * (q - 32)`, and the -32 - but not `sc` - folds into the code.
//
// # Fragments
//
// `m8n8k16` wants four int8 in a lane's register: lane `l` holds row `l>>2` and
// contraction `4*(l&3) .. +3` for both operands, and holds `D[l>>2][2*(l&3)]`
// and the column after it. Shared memory is therefore staged as *words* of four
// codes, which is also how the staging writes them, so no lane ever addresses a
// byte.

#define GEMM_I8_WARPS  8
#define GEMM_I8_MT     128     // rows of `a` per block
#define GEMM_I8_NT     128     // rows of `w` per block
// 64 of contraction a trip, which is two Q4_K sub-blocks - and the pairing is
// the point. Q4_K stores the low nibble of a byte at element `j` and the high
// nibble at `j + 32`, so a trip of 32 reads sixteen bytes and uses half of each
// one. A trip of 64 uses both, which halves what this kernel reads from global
// memory for four fifths of the weights. It also halves the barriers, and it
// gives the activation staging one 32-byte row per thread instead of leaving
// half the block idle - that staging measured 15% of the kernel at 32.
#define GEMM_I8_KC     64
#define GEMM_I8_SUB    (GEMM_I8_KC / 32)             // sub-blocks a trip
#define GEMM_I8_KS     2                             // `mma` steps a sub-block
#define GEMM_I8_KG     16                            // elements an `mma` step
#define GEMM_I8_WM     2       // warps down the tile
#define GEMM_I8_WN     4       // and across it
#define GEMM_I8_MR     (GEMM_I8_MT / GEMM_I8_WM)     // rows one warp owns
#define GEMM_I8_NR     (GEMM_I8_NT / GEMM_I8_WN)     // columns one warp owns
#define GEMM_I8_MS     (GEMM_I8_MR / 8)              // m8n8k16 is 8 rows
#define GEMM_I8_NPW    (GEMM_I8_NR / 8)
#define GEMM_I8_WORDS  (GEMM_I8_KC / 4)              // 4 codes to a word
// A multiple of four, so every 16-byte fragment row is 16-byte aligned -
// `ldmatrix` requires it - and twenty rather than sixteen, so that eight
// consecutive rows cover all 32 banks: `20r mod 32` runs 0, 20, 8, 28, 16, 4,
// 24, 12, which is the eight distinct multiples of four.
#define GEMM_I8_STRIDE (GEMM_I8_WORDS + 4)
#define GEMM_I8_THREADS (GEMM_I8_WARPS * 32)
// Weight runs one thread stages, as a compile-time trip count: the header
// cache below is a register array and a runtime bound would put it in local
// memory, which is the thing it exists to avoid.
#define GEMM_I8_BITER \
    ((GEMM_I8_NT * GEMM_I8_KS + GEMM_I8_THREADS - 1) / GEMM_I8_THREADS)

__device__ __forceinline__ void mma_s8(
    int& d0, int& d1, unsigned a, unsigned b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
}

// Sum of the four signed codes in a word. `dp4a` against ones, which is one
// instruction where the shifts and sign extensions are eight.
__device__ __forceinline__ int sum4(unsigned w, int acc) {
    return __dp4a((int)w, 0x01010101, acc);
}

// Four `m8n8k16` int8 fragments in one instruction.
//
// `ldmatrix` is a 16-bit instruction and this is an 8-bit matmul, and they fit
// anyway: an 8x8 tile of b16 is 8 rows of 16 bytes, and a lane ends up holding
// bytes `4*(l&3) .. +3` of row `l>>2` - which is the int8 fragment layout for
// both operands, exactly. So the same instruction that feeds the f16 kernel
// feeds this one, and the alternative is 32 scalar shared loads a trip.
//
// Lane `l` supplies the address of row `l&7` of matrix `l>>3`.
__device__ __forceinline__ void ld_i8_x4(unsigned* d, const unsigned* row)
{
    unsigned p = (unsigned)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(p));
}

// One body, two entry points. The format is a template parameter and not an
// argument because it decides which staging code exists at all and whether the
// two `mma` steps of a sub-block share an accumulator: as a runtime branch,
// ptxas allocated registers for the union of both and scheduled for neither.
template <int QT, int MT>
__device__ __forceinline__ void gemm_i8_body(
    const signed char* __restrict__ qa,
    int asc_off,
    const unsigned char* __restrict__ wq,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int m, int k, int n,
    long sw, long so,
    int q_ts, int a_rows,
    int ksplit, float* __restrict__ partial)
{
    // The row tile is a template parameter so a short prefill can have a
    // smaller one. A block computes `MT` rows whether or not `m` has them, so
    // a 24-token prompt against `MT = 128` does five times the arithmetic it
    // needs; the padding is the whole cost of a short prefill, and halving the
    // tile halves it.
    //
    // `MR` is what one warp owns down the tile and `MS` how many `m8n8k16`
    // rows that is. Four is the floor and not a tuning choice: the fragment
    // load below is an `ldmatrix .x4`, which takes four row groups at once.
    constexpr int MR = MT / GEMM_I8_WM;
    constexpr int MS = MR / 8;
    static_assert(MS % 4 == 0, "the fragment load takes four row groups");

    __shared__ __align__(16) unsigned au[MT * GEMM_I8_STRIDE];
    __shared__ __align__(16) unsigned bu[GEMM_I8_NT * GEMM_I8_STRIDE];
    // Scales, paired so that a lane fetches both halves of what it needs in one
    // eight-byte load. This is the difference between the scales costing more
    // shared traffic than the fragments and costing a fraction of them.
    //
    // `asx` is (scale, sum of codes) per row per sub-block; `bds` is `d * sc`
    // per `mma` step, because Q6_K changes scale every sixteen weights; `bdm`
    // is `dmin * mn` per sub-block, which Q6_K does not have at all.
    __shared__ __align__(8) float asx[MT * GEMM_I8_SUB * 2];
    __shared__ __align__(8) float bds[GEMM_I8_SUB][GEMM_I8_KS][GEMM_I8_NT];
    __shared__ __align__(8) float bdm[GEMM_I8_SUB][GEMM_I8_NT];

    // Codes then scales in one allocation, the layout `quantize_q8` writes.
    const float* __restrict__ ascale = (const float*)(qa + asc_off);

    const int slice = (int)(blockIdx.z / (gridDim.z / ksplit));
    const int bat   = (int)(blockIdx.z % (gridDim.z / ksplit));

    out += (size_t)bat * so;
    const unsigned char* wb = wq + (size_t)bat * (size_t)((sw / 256) * (long)q_ts);
    // `quantize_q8` writes one dense row per (batch, row), so the batch stride
    // here is a row count and not the activation's own stride - and it is zero
    // when every product of the batch reads the *same* activation, which is
    // what the attention projections do.
    const size_t arow = (size_t)bat * (size_t)a_rows;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int g    = lane >> 2;
    const int tg   = lane & 3;

    // `x` is the row tile, not the column tile, and that ordering is the point:
    // the blocks that share a weight tile are the ones that differ in `m`, so
    // making them consecutive puts them on the machine together and lets L2
    // serve the weight to all but the first.
    const int m0 = blockIdx.x * MT;
    const int n0 = blockIdx.y * GEMM_I8_NT;
    // A square-ish warp grid, because a warp's shared traffic is
    // `(MS + NPW) * KS` words and `MS * NPW` is fixed by the `mma` count. One
    // warp per column strip made that 18 words a trip; two by four makes it 12.
    const int mb0 = (warp / GEMM_I8_WN) * MR;
    const int nb0 = (warp % GEMM_I8_WN) * GEMM_I8_NR;

    const int kstep = ((k + ksplit - 1) / ksplit + GEMM_I8_KC - 1)
                      / GEMM_I8_KC * GEMM_I8_KC;
    const int kbeg  = slice * kstep;
    const int kend  = min(k, kbeg + kstep);

    float acc[MS][GEMM_I8_NPW][2];
    #pragma unroll
    for (int i = 0; i < MS; ++i) {
        #pragma unroll
        for (int j = 0; j < GEMM_I8_NPW; ++j) {
            acc[i][j][0] = acc[i][j][1] = 0.0f;
        }
    }

    const long nb = k / 256;
    const int groups = k >> 5;

    // A staging thread reads the same weight row on every trip, so it crosses a
    // super-block only once in four - and the scales it needs live in the
    // sixteen bytes at the front of one (Q4_K) or at byte 192 of one (Q6_K).
    // Re-reading those every trip measured 17.5% of this kernel; caching them
    // per thread is the same trick `gemm` uses, for the same reason.
    uint4 whdr[GEMM_I8_BITER];
    float wd[GEMM_I8_BITER];
    long  wsb[GEMM_I8_BITER];
    #pragma unroll
    for (int u = 0; u < GEMM_I8_BITER; ++u) {
        wsb[u] = -1;
        wd[u] = 0.0f;
    }

    for (int kc = kbeg; kc < kend; kc += GEMM_I8_KC) {
        // The activation is already int8: `gemm_batched` quantizes it once for
        // the whole launch. Doing it here instead - which the first version of
        // this kernel did - repeats the maximum, the reciprocal and the
        // rounding once per column tile, 32 times over on a 4096-wide
        // projection, and measured 640 tok/s against the f16 kernel's 1926. So
        // this is a copy: one sub-block a thread, 32 bytes in and 32 out.
        for (int i = tid; i < MT * GEMM_I8_SUB; i += GEMM_I8_THREADS) {
            const int r = i / GEMM_I8_SUB, sb = i % GEMM_I8_SUB;
            const int row = m0 + r;
            uint4 v0 = make_uint4(0u, 0u, 0u, 0u), v1 = v0;
            float d = 0.0f;
            if (row < m) {
                const uint4* src = reinterpret_cast<const uint4*>(
                    qa + (arow + row) * (size_t)k + kc + 32 * sb);
                v0 = src[0];
                v1 = src[1];
                d = ascale[(arow + row) * (size_t)groups + (kc >> 5) + sb];
            }
            uint4* dst = reinterpret_cast<uint4*>(
                &au[r * GEMM_I8_STRIDE + 8 * sb]);
            dst[0] = v0;
            dst[1] = v1;
            // The code sum is wanted per sub-block rather than per step: it is
            // only ever multiplied by `dmin * mn`, which Q4_K holds constant
            // across a sub-block and Q6_K does not have.
            int t = sum4(v0.w, sum4(v0.z, sum4(v0.y, sum4(v0.x, 0))));
            t = sum4(v1.w, sum4(v1.z, sum4(v1.y, sum4(v1.x, t))));
            asx[2 * (GEMM_I8_SUB * r + sb)] = d;
            asx[2 * (GEMM_I8_SUB * r + sb) + 1] = (float)t;
        }

        // The weights. One thread stages sixteen bytes and both nibbles of
        // them, which is one `mma` step of each of the trip's two sub-blocks -
        // and for Q4_K the nibbles need no unpacking loop at all, because
        // `w & 0x0F0F0F0F` already *is* four int8 codes in the order the
        // tensor core reads them.
        #pragma unroll
        for (int u = 0; u < GEMM_I8_BITER; ++u) {
            const int i = tid + u * GEMM_I8_THREADS;
            if (i >= GEMM_I8_NT * GEMM_I8_KS) break;
            const int r = i / GEMM_I8_KS, h = i % GEMM_I8_KS;
            const int col = n0 + r;
            unsigned lo[GEMM_I8_KG / 4], hi[GEMM_I8_KG / 4];
            #pragma unroll
            for (int t = 0; t < GEMM_I8_KG / 4; ++t) lo[t] = hi[t] = 0u;
            float ds0 = 0.0f, ds1 = 0.0f, dm0 = 0.0f, dm1 = 0.0f;
            if (col < n) {
                const long sb = (long)col * nb + (kc >> 8);
                const unsigned char* blk = wb + (size_t)sb * (size_t)q_ts;
                const int j = (kc & 255) + GEMM_I8_KG * h;
                if (sb != wsb[u]) {
                    wsb[u] = sb;
                    if (QT == QT_Q4_K) {
                        whdr[u] = *reinterpret_cast<const uint4*>(blk);
                    } else {
                        whdr[u] = *reinterpret_cast<const uint4*>(blk + 192);
                        wd[u] = q_f16(blk, 208);
                    }
                }
                if (QT == QT_Q4_K) {
                    // One sixteen-byte run, low nibbles for elements `j` and
                    // high nibbles for `j + 32` - the trip's other sub-block.
                    unsigned char sc, mn;
                    q_scale_min_words(whdr[u].y, whdr[u].z, whdr[u].w, j >> 5, sc, mn);
                    ds0 = q_half_lo(whdr[u].x) * (float)sc;
                    dm0 = q_half_hi(whdr[u].x) * (float)mn;
                    q_scale_min_words(whdr[u].y, whdr[u].z, whdr[u].w,
                                      (j >> 5) + 1, sc, mn);
                    ds1 = q_half_lo(whdr[u].x) * (float)sc;
                    dm1 = q_half_hi(whdr[u].x) * (float)mn;
                    unsigned w[GEMM_I8_KG / 4];
                    q_words<GEMM_I8_KG / 4>(blk + 16 + ((j >> 6) << 5) + (j & 31), w);
                    #pragma unroll
                    for (int t = 0; t < GEMM_I8_KG / 4; ++t) {
                        lo[t] = w[t] & 0x0F0F0F0Fu;
                        hi[t] = (w[t] >> 4) & 0x0F0F0F0Fu;
                    }
                } else {
                    // Q6_K: `d * scales[j/16] * (q - 32)`, no minimum. In the
                    // device layout its low nibbles are paired 32 apart the
                    // way Q4_K's are, so one sixteen-byte run is both of the
                    // trip's sub-blocks, and their 2-bit high fields are the
                    // two words beside it - 24 bytes a step where the file's
                    // own grouping cost 48 and used half of every one.
                    //
                    // The scale is one of the sixteen cached bytes, picked with
                    // selects rather than a subscript: `whdr` is a register
                    // array and indexing it by a loop variable would spill the
                    // whole thing to local memory.
                    // The two sub-scales go to shared as *integers*, bit-cast
                    // through the float array, and `d` goes to `bdm` beside
                    // them: the mma loop folds the scales into the integer
                    // accumulation and converts once per sub-block, like
                    // Q4_K's merged path, instead of paying a quarter-rate
                    // I2F per step. `|sc * dot|` is at most 128 * 65024 and
                    // the folded sum at most 16,646,144 - under 2^24, so the
                    // one conversion is still exact.
                    const unsigned w0 = ((j >> 4) & 4) ? whdr[u].y : whdr[u].x;
                    const unsigned w1 = ((j >> 4) & 4) ? whdr[u].w : whdr[u].z;
                    const unsigned ws = ((j >> 4) & 8) ? w1 : w0;
                    ds0 = __int_as_float((int)(signed char)
                          ((ws >> (((j >> 4) & 3) << 3)) & 0xFF));
                    const int j1 = j + 32;
                    const unsigned x0 = ((j1 >> 4) & 4) ? whdr[u].y : whdr[u].x;
                    const unsigned x1 = ((j1 >> 4) & 4) ? whdr[u].w : whdr[u].z;
                    const unsigned xs = ((j1 >> 4) & 8) ? x1 : x0;
                    ds1 = __int_as_float((int)(signed char)
                          ((xs >> (((j1 >> 4) & 3) << 3)) & 0xFF));
                    // `bdm` is indexed by sub-block and both of the trip's
                    // sub-blocks sit in one super-block, so `d` is the same
                    // value twice.
                    dm0 = wd[u];
                    dm1 = wd[u];

                    const int pq = j >> 6, hq = (j >> 4) & 1;
                    unsigned w[GEMM_I8_KG / 4], hw[2];
                    q_words<GEMM_I8_KG / 4>(blk + (pq << 5) + (hq << 4), w);
                    q_words<2>(blk + 128 + (pq << 4) + (hq << 3), hw);
                    #pragma unroll
                    for (int t = 0; t < GEMM_I8_KG / 4; ++t) {
                        const unsigned a0 = w[t] & 0x0F0F0F0Fu;
                        const unsigned a1 = (w[t] >> 4) & 0x0F0F0F0Fu;
                        const unsigned b0 = (hw[0] >> (t << 1)) & 0x03030303u;
                        const unsigned b1 = (hw[1] >> (t << 1)) & 0x03030303u;
                        lo[t] = __vsub4(a0 | (b0 << 4), 0x20202020u);
                        hi[t] = __vsub4(a1 | (b1 << 4), 0x20202020u);
                    }
                }
            }
            unsigned* dst = &bu[r * GEMM_I8_STRIDE + (GEMM_I8_KG / 4) * h];
            #pragma unroll
            for (int t = 0; t < GEMM_I8_KG / 4; ++t) {
                dst[t] = lo[t];
                dst[8 + t] = hi[t];
            }
            bds[0][h][r] = ds0;
            bds[1][h][r] = ds1;
            if (h == 0) {
                bdm[0][r] = dm0;
                bdm[1][r] = dm1;
            }
        }
        __syncthreads();

        // Deliberately not unrolled: the two sub-blocks each want a full set
        // of fragments and scales in registers, and holding both sets at once
        // spills 160 bytes to local memory.
        #pragma unroll 1
        for (int sb = 0; sb < GEMM_I8_SUB; ++sb) {
            // Both `mma` steps of a sub-block, and their scales, hoisted out of
            // the row loop: neither depends on which row is being multiplied.
            unsigned bfr[GEMM_I8_KS][GEMM_I8_NPW];
            float2 ds2[GEMM_I8_KS][GEMM_I8_NPW], dm2[GEMM_I8_NPW];
            #pragma unroll
            for (int ks = 0; ks < GEMM_I8_KS; ++ks) {
                ld_i8_x4(bfr[ks], &bu[(nb0 + 8 * (lane >> 3) + (lane & 7))
                                      * GEMM_I8_STRIDE + 8 * sb + 4 * ks]);
                #pragma unroll
                for (int nt = 0; nt < GEMM_I8_NPW; ++nt) {
                    const int c = nb0 + 8 * nt + 2 * tg;
                    ds2[ks][nt] = *reinterpret_cast<const float2*>(&bds[sb][ks][c]);
                    if (ks == 0) dm2[nt] = *reinterpret_cast<const float2*>(&bdm[sb][c]);
                }
            }
            // Q4_K holds one scale across all 32 elements of a sub-block, so
            // both steps can run into the same integer accumulator and be
            // converted once. Q6_K changes scale every sixteen and has to
            // convert twice.
            const bool merged = (QT == QT_Q4_K);

            #pragma unroll
            for (int m4 = 0; m4 < MS / 4; ++m4) {
                unsigned afr[GEMM_I8_KS][4];
                #pragma unroll
                for (int ks = 0; ks < GEMM_I8_KS; ++ks) {
                    ld_i8_x4(afr[ks],
                             &au[(mb0 + 8 * (4 * m4 + (lane >> 3)) + (lane & 7))
                                 * GEMM_I8_STRIDE + 8 * sb + 4 * ks]);
                }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int ms = 4 * m4 + q;
                    // (scale, sum of codes) for this row and sub-block, in one
                    // load.
                    const float2 a2 = *reinterpret_cast<const float2*>(
                        &asx[2 * (GEMM_I8_SUB * (mb0 + 8 * ms + g) + sb)]);
                    #pragma unroll
                    for (int nt = 0; nt < GEMM_I8_NPW; ++nt) {
                        // `sum a*w = as*ds*sum(ia*q) - as*dm*sum(ia)`, where
                        // the second term is per sub-block: Q4_K's minimum is
                        // constant over one and Q6_K has none.
                        if (merged) {
                            int d0 = 0, d1 = 0;
                            #pragma unroll
                            for (int ks = 0; ks < GEMM_I8_KS; ++ks) {
                                mma_s8(d0, d1, afr[ks][q], bfr[ks][nt]);
                            }
                            acc[ms][nt][0] +=
                                a2.x * (ds2[0][nt].x * (float)d0 - dm2[nt].x * a2.y);
                            acc[ms][nt][1] +=
                                a2.x * (ds2[0][nt].y * (float)d1 - dm2[nt].y * a2.y);
                        } else {
                            // Q6_K's scale changes every sixteen elements, so
                            // its two steps cannot share a bare accumulator -
                            // but the scales are 8-bit integers, so they fold
                            // into the *integer* sum instead: two full-rate
                            // IMADs and one exact conversion, where a
                            // conversion per step was a quarter-rate I2F each.
                            // `ds2` holds the scales as bit-cast ints and
                            // `dm2` holds `d`; the staging says why the fold
                            // stays under 2^24 and therefore exact.
                            int dA0 = 0, dA1 = 0, dB0 = 0, dB1 = 0;
                            mma_s8(dA0, dA1, afr[0][q], bfr[0][nt]);
                            mma_s8(dB0, dB1, afr[1][q], bfr[1][nt]);
                            const int t0 = __float_as_int(ds2[0][nt].x) * dA0
                                         + __float_as_int(ds2[1][nt].x) * dB0;
                            const int t1 = __float_as_int(ds2[0][nt].y) * dA1
                                         + __float_as_int(ds2[1][nt].y) * dB1;
                            acc[ms][nt][0] += a2.x * dm2[nt].x * (float)t0;
                            acc[ms][nt][1] += a2.x * dm2[nt].y * (float)t1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ksplit > 1) {
        out = partial + ((size_t)slice * (gridDim.z / ksplit) + bat) * (size_t)m * n;
    }
    #pragma unroll
    for (int ms = 0; ms < MS; ++ms) {
        const int row = m0 + mb0 + 8 * ms + g;
        if (row >= m) continue;
        #pragma unroll
        for (int nt = 0; nt < GEMM_I8_NPW; ++nt) {
            const int col0 = n0 + nb0 + 8 * nt + 2 * tg;
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int col = col0 + u;
                if (col < n) {
                    const float b = (bias && ksplit == 1) ? bias[col] : 0.0f;
                    out[(size_t)row * n + col] = acc[ms][nt][u] + b;
                }
            }
        }
    }
}

#define GEMM_I8_ENTRY(name, qt, mt)                                           \
    extern "C" __global__ __launch_bounds__(GEMM_I8_THREADS, 2) void name(    \
        const signed char* __restrict__ qa, int asc_off,                      \
        const unsigned char* __restrict__ wq, const float* __restrict__ bias, \
        float* __restrict__ out, int m, int k, int n, long sw, long so,       \
        int q_ts, int a_rows, int ksplit, float* __restrict__ partial)        \
    {                                                                         \
        gemm_i8_body<qt, mt>(qa, asc_off, wq, bias, out, m, k, n, sw, so,     \
                             q_ts, a_rows, ksplit, partial);                  \
    }

// Two row tiles, and the narrow one is not a tuning knob. A prefill computes
// `MT` rows whether the prompt has them or not, and a translator's prompt is
// twenty-odd tokens - so at 128 rows nine tenths of the arithmetic is padding.
// 64 is the narrowest this kernel's `ldmatrix .x4` allows; see `gemm_i8_body`.
#define GEMM_I8_MT_NARROW 64
GEMM_I8_ENTRY(gemm_i8_q4k, QT_Q4_K, GEMM_I8_MT)
GEMM_I8_ENTRY(gemm_i8_q6k, QT_Q6_K, GEMM_I8_MT)
GEMM_I8_ENTRY(gemm_i8_q4k_narrow, QT_Q4_K, GEMM_I8_MT_NARROW)
GEMM_I8_ENTRY(gemm_i8_q6k_narrow, QT_Q6_K, GEMM_I8_MT_NARROW)

// ------------------------------------------------------------ flash attention
//
// Attention for a whole prompt - or a whole encoder window - in one kernel:
// scores, mask, softmax and the value product, with nothing materialised. The
// unfused chain writes the score matrix, reads it back to softmax it, writes
// the probabilities, and reads them again for the value product -
// `heads * tq * tk` floats three times over, plus a head split and a merge. At
// 512 tokens on the 13 B that chain measured about 27 ms a prefill, and the
// Whisper encoder's 20 x 1500 x 1500 scores are 180 MB a layer; the score
// tensor exists only so the softmax can find its row maximum, and the
// running-maximum trick removes that need.
//
// One block owns 32 query rows of one head and walks the keys 32 at a time,
// keeping the output accumulator in registers. Per tile: scores by `m16n8k8`
// into f32, the row maximum folded into a running one, the accumulator
// rescaled by `exp(m_old - m_new)`, probabilities rounded to f16 - exactly
// where the unfused chain rounded them, on their way into the value product -
// and one more `m16n8k8` against the values. `__expf`, because that is what
// `softmax_causal` and `softmax_rows` both use; this kernel replaces them and
// must not be a precision change.
//
// **The one place it is not the chain's arithmetic** is where the normaliser
// divides. `softmax_rows` scales the probabilities by `1/l` before rounding
// them to f16; an online softmax cannot, because `l` is not known until the
// last tile, so it rounds `exp(s - m)` instead and divides the f32 accumulator
// at the end. Both roundings are one f16 step on a positive number, so the
// relative error is the same size - and on a 1500-key encoder row the online
// form is the better conditioned of the two, because `exp(s - m)` sits near 1
// where `p / l` sits near 1/1500. It is still a change, so it is stated here
// and measured against the captured oracle rather than assumed harmless.
//
// **Causality is a flag, not a shape.** A decoder prompt masks the upper
// triangle and the loop bound doubles as a free skip of it; the Whisper
// encoder attends over the whole window and passes `causal = 0`, which turns
// the per-row limit into the key count and nothing else.
//
// Layouts are the caches' own: K is `[kv_head][pos][hd]`, V is
// `[kv_head][hd][cap]`, and the queries are read straight out of the
// projection buffer at `[tq, heads * hd]` - no `split_heads` - with the
// merged context written the same way, so no `merge_heads` either.
// Grouped-query models map `head / (heads / kv_heads)`.
//
// **`HD` is a template parameter, and the two widths that exist are
// instantiated by name below.** The fragment layout is what depends on it: a
// warp owns `HD / 32` of the output's n8 column fragments, so 128 gives four
// and 64 gives two, and the shared-memory strides follow. Every other tile
// shape - 32 query rows, 32 keys a trip, eight warps as two query groups by
// four column groups - is independent of the head width and is not repeated
// per instantiation. A width that is not instantiated is refused by name in
// the wrapper rather than indexing across heads in bounds.

#define FA_WARPS 8

// One block owns QT query rows of one head and walks the keys KT at a time.
//
// **QT was the knob that mattered while this kernel was moving bytes.** A
// block stages every key and value it walks past, so the whole of K and V is
// re-staged once per query block: at 1500 encoder positions and QT 32 that is
// 47 trips through 0.8 MB. That is why QT is 64 here rather than 32, and it is
// no longer what limits the kernel - holding the cache at f16, which halves
// that traffic exactly, changed nothing at all, because at 768 KB a head the
// re-reads were already inside a 6 MB L2.
//
// **KT is the knob now, and what it buys is `mma` per shared load.** A warp
// issues KF * 1 products in the score phase for one `a` fragment and KF `b`
// fragments, so a wider key tile gives a warp more column fragments to spend
// each loaded query fragment on - KT 64 puts KF and NT at 4 where KT 32 left
// them at 2, and takes the kernel from 1.75 words loaded per `mma` to 1.5.
// It also halves the number of trips, and with them the barriers.
//
// KT 64 was measured before and rejected, and that measurement was
// confounded: with the score tile still in shared memory it cost 45.8 KB, one
// resident block, and half the threads - which is exactly enough to cancel
// what it gains. Once the scores stay in registers it fits in 28.25 KB, keeps
// two blocks, and pays. docs/BENCHMARKS.md has the whole sweep.
//
// The warp grid follows from QT: QG query groups of 16 rows, and the remaining
// FA_WARPS / QG warps spread across the key fragments of the score product and
// the column fragments of the value product. Every count below is derived, so
// a new (QT, KT, HD) is a template argument rather than an edit.
// `KVH` says the caches are f16. Both staging loops already round what they
// read to f16 on the way into shared memory, so an f16 cache does not cost a
// conversion here - it removes one, and halves what the loop fetches. It is a
// template argument and not a flag because these two loops run once a key tile
// and a branch inside them is a branch in the hot path.
template <int HD, int KT, int QT, bool KVH>
__device__ __forceinline__ void flash_attn_impl(
    const float* __restrict__ q,
    const void* __restrict__ kc,
    const void* __restrict__ vc,
    float* __restrict__ out,
    int tq, int past, int heads, int kv_heads, int cap, float scale, int causal)
{
    // Words a Qs/Ks row: HD/2 packed pairs, plus four to spread the banks.
    constexpr int QSTR = HD / 2 + 4;
    constexpr int KSTR = HD / 2 + 4;
    // Words a Vs or Ps row: KT/2 packed pairs, padded the same way.
    constexpr int VSTR = KT / 2 + 4;
    constexpr int PSTR = KT / 2 + 4;
    // Query groups of 16 rows, and the warp columns left over for them.
    constexpr int QG = QT / 16;
    constexpr int CG = FA_WARPS / QG;
    // n8 output column fragments a warp owns: CG column groups cover HD.
    constexpr int NT = (HD / 8) / CG;
    // n8 key fragments a warp owns in the score product: CG warp columns
    // cover KT.
    constexpr int KF = (KT / 8) / CG;
    constexpr int KVS = (HD * VSTR > KT * KSTR) ? HD * VSTR : KT * KSTR;

    // Every count above divides exactly or the warp grid silently drops work.
    static_assert(QT % 16 == 0 && FA_WARPS % QG == 0, "QT does not tile");
    static_assert(NT * CG * 8 == HD, "the column groups do not cover HD");
    static_assert(KF * CG * 8 == KT, "the warp columns do not cover KT");

    // sm_75 gives a block 48 KB of *static* shared memory - the 64 KB needs a
    // dynamic opt-in this kernel does not take. It gives an *SM* 64 KB, so
    // what this comes to also decides how many blocks are resident, and that
    // has decided more of this kernel's speed than anything else: every shape
    // measured at one resident block came in at 114 ms or worse and every one
    // at two or three came in under 116. docs/BENCHMARKS.md has the sweep.
    constexpr int SHARED = (QT * QSTR + KVS + QT * PSTR + QT * CG + 3 * QT) * 4;
    static_assert(SHARED <= 48 * 1024, "the tile exceeds 48 KB of shared memory");

    __shared__ __align__(16) unsigned qs[QT * QSTR];
    __shared__ __align__(16) unsigned kvs[KVS];
    __shared__ __align__(16) unsigned ps[QT * PSTR];
    // One slot a row per warp column. This is all that is left of a `[QT][KT]`
    // score tile: the scores stay in the registers the `mma` left them in, and
    // what the warps have to tell each other is not the scores but the
    // reduction over them - CG numbers a row, twice a tile. See the score
    // product below.
    __shared__ float sm_part[QT * CG];
    __shared__ float sm_m[QT], sm_l[QT], sm_fac[QT];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int g = lane >> 2, tg = lane & 3;

    const int qt0 = blockIdx.x * QT;
    const int h = blockIdx.y;
    const int kh = h / (heads / kv_heads);
    const int dq = heads * HD;

    // Elements either way; the f16 pointers are words, so a head's offset is
    // halved with it. `cap` is even for exactly this reason - see `Cache`.
    const float* kb = KVH ? (const float*)0 : (const float*)kc + (size_t)kh * cap * HD;
    const float* vb = KVH ? (const float*)0 : (const float*)vc + (size_t)kh * HD * cap;
    const unsigned* kbh =
        KVH ? (const unsigned*)kc + (size_t)kh * cap * (HD / 2) : (const unsigned*)0;
    const unsigned* vbh =
        KVH ? (const unsigned*)vc + (size_t)kh * HD * (cap / 2) : (const unsigned*)0;

    // Queries once, rounded to f16 - the same rounding the tiled gemm applied
    // to its operands. A row past `tq` stages zeros and is never stored.
    for (int i = tid; i < QT * (HD / 2); i += FA_WARPS * 32) {
        const int r = i / (HD / 2), j = i % (HD / 2);
        unsigned w = 0u;
        if (qt0 + r < tq) {
            const float2 v =
                *reinterpret_cast<const float2*>(q + (size_t)(qt0 + r) * dq
                                                 + (size_t)h * HD + 2 * j);
            w = gemm_pack(v.x, v.y);
        }
        qs[r * QSTR + j] = w;
    }
    if (tid < QT) {
        sm_m[tid] = -1.0f / 0.0f;
        sm_l[tid] = 0.0f;
    }

    // The output accumulator: warp `w` owns query group `w / CG` (16 rows)
    // and value columns `(w % CG) * NT * 8 .. + NT * 8 - 1`, as NT fragments.
    const int mg = warp / CG;
    const int ng0 = (warp % CG) * NT;
    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    __syncthreads();

    // Causal: the last key a row of this block may see is
    // `past + qt0 + QT - 1`, so the loop stops there and never touches the
    // upper triangle. Non-causal: every row sees every key.
    const int ktot = past + tq;
    const int kend = causal ? min(ktot, past + qt0 + QT) : ktot;
    for (int kv0 = 0; kv0 < kend; kv0 += KT) {
        // Keys, `[pos][d]`, rounded like the queries. Positions past `kend`
        // stage zeros; the mask discards whatever the mma made of them.
        for (int i = tid; i < KT * (HD / 2); i += FA_WARPS * 32) {
            const int r = i / (HD / 2), j = i % (HD / 2);
            unsigned w = 0u;
            if (kv0 + r < kend) {
                if (KVH) {
                    // Already the packed pair this wants.
                    w = kbh[(size_t)(kv0 + r) * (HD / 2) + j];
                } else {
                    const float2 v = *reinterpret_cast<const float2*>(
                        kb + (size_t)(kv0 + r) * HD + 2 * j);
                    w = gemm_pack(v.x, v.y);
                }
            }
            kvs[r * KSTR + j] = w;
        }
        __syncthreads();

        // S = Q K^T for this tile. Each warp covers one query group and KF
        // of the tile's n8 key fragments. The `a` fragment is loaded once per
        // contraction step and reused across them.
        //
        // **The scores never leave the registers the `mma` writes them to.**
        // They used to go to a `[QT][KT]` shared tile, be read back for the
        // running maximum and read again for the exponential - three passes
        // over 9 KB a tile, and a third of the block's shared budget, which is
        // its residency. What crosses warps instead is the reduction rather
        // than the scores: the `m16n8k8` accumulator gives a lane two rows of
        // its fragment, the four lanes sharing a `g` hold the rest of those
        // rows, so an xor butterfly folds a warp's own columns and only CG
        // partials a row reach shared memory.
        const int sng = warp % CG;
        float d[KF][4];
        {
            #pragma unroll
            for (int f = 0; f < KF; ++f) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    d[f][j] = 0.0f;
                }
            }
            #pragma unroll
            for (int ks = 0; ks < HD / 8; ++ks) {
                unsigned a0, a1;
                gemm_ld_a(&qs[(mg * 16 + (lane & 15)) * QSTR + 4 * ks],
                          a0, a1);
                #pragma unroll
                for (int f = 0; f < KF; ++f) {
                    const unsigned b0 = gemm_ld_b(
                        &kvs[((sng * KF + f) * 8 + (lane & 7)) * KSTR + 4 * ks]);
                    gemm_mma_step(d[f][0], d[f][1], d[f][2], d[f][3], a0, a1, b0);
                }
            }
            // Scaled here rather than on the way out to shared memory, which
            // is the same arithmetic in the same place: the scale was always
            // applied to the accumulator before anything read it.
            #pragma unroll
            for (int f = 0; f < KF; ++f) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    d[f][j] *= scale;
                }
            }
        }

        // This warp's maximum for each of its two rows, masked. `lim` is the
        // last key a row may attend to - its own position when causal, the
        // last key otherwise.
        const int r0 = mg * 16 + g;
        const int lim0 = causal ? (past + qt0 + r0) : (ktot - 1);
        const int lim1 = causal ? (past + qt0 + r0 + 8) : (ktot - 1);
        {
            float a = -1.0f / 0.0f, b = -1.0f / 0.0f;
            #pragma unroll
            for (int f = 0; f < KF; ++f) {
                const int c = kv0 + (sng * KF + f) * 8 + 2 * tg;
                if (c <= lim0) {
                    a = fmaxf(a, d[f][0]);
                }
                if (c + 1 <= lim0) {
                    a = fmaxf(a, d[f][1]);
                }
                if (c <= lim1) {
                    b = fmaxf(b, d[f][2]);
                }
                if (c + 1 <= lim1) {
                    b = fmaxf(b, d[f][3]);
                }
            }
            #pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
                b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, o));
            }
            if (tg == 0) {
                sm_part[r0 * CG + sng] = a;
                sm_part[(r0 + 8) * CG + sng] = b;
            }
        }
        __syncthreads();

        // The running maximum, one thread a row: it folds the CG warp columns
        // into the row's own and leaves the rescale factor for everyone. One
        // thread because `sm_m` is read and written here, and a second writer
        // would race with it rather than agree with it.
        if (tid < QT) {
            float mx = -1.0f / 0.0f;
            #pragma unroll
            for (int c = 0; c < CG; ++c) {
                mx = fmaxf(mx, sm_part[tid * CG + c]);
            }
            const float m_new = fmaxf(sm_m[tid], mx);
            sm_fac[tid] = __expf(sm_m[tid] - m_new);
            sm_m[tid] = m_new;
        }
        __syncthreads();

        // Probabilities, rounded to f16 on their way into the value product -
        // where the unfused chain rounded them too - and the row sums, folded
        // the way the maxima were. A lane's two accumulator columns are
        // adjacent and the first is even, so the pair it holds is exactly one
        // packed word of `ps` and the layout the value product's `a` fragment
        // wants falls out of the score product's own.
        {
            const float m0 = sm_m[r0], m1 = sm_m[r0 + 8];
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int f = 0; f < KF; ++f) {
                const int c = kv0 + (sng * KF + f) * 8 + 2 * tg;
                const float p00 = (c <= lim0) ? __expf(d[f][0] - m0) : 0.0f;
                const float p01 = (c + 1 <= lim0) ? __expf(d[f][1] - m0) : 0.0f;
                const float p10 = (c <= lim1) ? __expf(d[f][2] - m1) : 0.0f;
                const float p11 = (c + 1 <= lim1) ? __expf(d[f][3] - m1) : 0.0f;
                s0 += p00 + p01;
                s1 += p10 + p11;
                const int w = (sng * KF + f) * 4 + tg;
                ps[r0 * PSTR + w] = gemm_pack(p00, p01);
                ps[(r0 + 8) * PSTR + w] = gemm_pack(p10, p11);
            }
            #pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                s0 += __shfl_xor_sync(0xffffffff, s0, o);
                s1 += __shfl_xor_sync(0xffffffff, s1, o);
            }
            if (tg == 0) {
                sm_part[r0 * CG + sng] = s0;
                sm_part[(r0 + 8) * CG + sng] = s1;
            }
        }

        // Values, `[d][pos]` - the cache's own transposed layout, which is
        // already the `[n][k]` shape the B fragment wants.
        for (int i = tid; i < HD * (KT / 2); i += FA_WARPS * 32) {
            const int r = i / (KT / 2), j = i % (KT / 2);
            unsigned w = 0u;
            if (kv0 + 2 * j + 1 < kend) {
                if (KVH) {
                    // `cap` and `kv0` are both even, so the pair this word
                    // wants is one word of the cache and not two halves of
                    // neighbouring ones.
                    w = vbh[((size_t)r * cap + kv0) / 2 + j];
                } else {
                    const float2 v = *reinterpret_cast<const float2*>(
                        vb + (size_t)r * cap + kv0 + 2 * j);
                    w = gemm_pack(v.x, v.y);
                }
            } else if (kv0 + 2 * j < kend) {
                if (KVH) {
                    // The odd tail: keep the low half, zero the one the mask
                    // would have discarded anyway.
                    w = vbh[((size_t)r * cap + kv0) / 2 + j] & 0x0000ffffu;
                } else {
                    w = gemm_pack(vb[(size_t)r * cap + kv0 + 2 * j], 0.0f);
                }
            }
            kvs[r * VSTR + j] = w;
        }
        __syncthreads();

        // The running sum, one thread a row, for the reason the maximum is.
        // Nothing reads `sm_l` before the final store, so it is folded in
        // beside the value product rather than costing a barrier of its own;
        // the loop's closing barrier is what stops the next tile overwriting
        // `sm_part` underneath it.
        if (tid < QT) {
            float sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < CG; ++c) {
                sum += sm_part[tid * CG + c];
            }
            sm_l[tid] = sm_l[tid] * sm_fac[tid] + sum;
        }

        // O = O * fac + P V, with the `a` fragment again loaded once per
        // contraction step and reused across the column fragments. The
        // rescale factor is per row, and a lane's fragment rows are `g` and
        // `g + 8` of its group.
        {
            float d[NT][4];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    d[nt][j] = 0.0f;
                }
            }
            #pragma unroll
            for (int ks = 0; ks < KT / 8; ++ks) {
                unsigned a0, a1;
                gemm_ld_a(&ps[(mg * 16 + (lane & 15)) * PSTR + 4 * ks], a0, a1);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const unsigned b0 = gemm_ld_b(
                        &kvs[((ng0 + nt) * 8 + (lane & 7)) * VSTR + 4 * ks]);
                    gemm_mma_step(d[nt][0], d[nt][1], d[nt][2], d[nt][3],
                                  a0, a1, b0);
                }
            }
            const float f0 = sm_fac[mg * 16 + g];
            const float f1 = sm_fac[mg * 16 + g + 8];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                acc[nt][0] = acc[nt][0] * f0 + d[nt][0];
                acc[nt][1] = acc[nt][1] * f0 + d[nt][1];
                acc[nt][2] = acc[nt][2] * f1 + d[nt][2];
                acc[nt][3] = acc[nt][3] * f1 + d[nt][3];
            }
        }
        __syncthreads();
    }

    // O / l, written merged: `[tq, heads * hd]`, `split_heads` and
    // `merge_heads` both gone.
    const int r0 = mg * 16 + g, r1 = r0 + 8;
    const float inv0 = sm_l[r0] > 0.0f ? 1.0f / sm_l[r0] : 0.0f;
    const float inv1 = sm_l[r1] > 0.0f ? 1.0f / sm_l[r1] : 0.0f;
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int col = (size_t)h * HD + (ng0 + nt) * 8 + 2 * tg;
        if (qt0 + r0 < tq) {
            float* o = out + (size_t)(qt0 + r0) * dq + col;
            o[0] = acc[nt][0] * inv0;
            o[1] = acc[nt][1] * inv0;
        }
        if (qt0 + r1 < tq) {
            float* o = out + (size_t)(qt0 + r1) * dq + col;
            o[0] = acc[nt][2] * inv1;
            o[1] = acc[nt][3] * inv1;
        }
    }
}

// The two head widths that exist in this engine: 128 for both Llama stages,
// 64 for every Whisper size (large-v2's 1280 over 20 heads, and the smaller
// ones' too - Whisper holds the head width fixed and varies the count). Each
// takes the query tile its own traffic wants: the Whisper encoder walks 1500
// keys and 64 rows a block halves what it re-stages, where a Llama prefill's
// causal loop stops at the block's own diagonal and 32 is already enough.
// Both were measured, and docs/BENCHMARKS.md has the numbers.
extern "C" __global__ __launch_bounds__(FA_WARPS * 32, 2) void flash_attn(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    float* __restrict__ out,
    int tq, int past, int heads, int kv_heads, int cap, float scale, int causal)
{
    flash_attn_impl<128, 32, 32, false>(q, kc, vc, out, tq, past, heads,
                                        kv_heads, cap, scale, causal);
}

// The same at 128, reading an f16 cache. Only this width, because the chat
// model is the only stage that holds its cache that way - the ASR's is a
// 64-wide encoder cache it re-reads inside L2, where an f16 copy was measured
// and bought nothing. See docs/BENCHMARKS.md.
extern "C" __global__ __launch_bounds__(FA_WARPS * 32, 2) void flash_attn_h(
    const float* __restrict__ q,
    const unsigned short* __restrict__ kc,
    const unsigned short* __restrict__ vc,
    float* __restrict__ out,
    int tq, int past, int heads, int kv_heads, int cap, float scale, int causal)
{
    flash_attn_impl<128, 32, 32, true>(q, kc, vc, out, tq, past, heads,
                                       kv_heads, cap, scale, causal);
}

extern "C" __global__ __launch_bounds__(FA_WARPS * 32, 2) void flash_attn_64(
    const float* __restrict__ q,
    const float* __restrict__ kc,
    const float* __restrict__ vc,
    float* __restrict__ out,
    int tq, int past, int heads, int kv_heads, int cap, float scale, int causal)
{
    flash_attn_impl<64, 64, 64, false>(q, kc, vc, out, tq, past, heads,
                                       kv_heads, cap, scale, causal);
}

// ---------------------------------------------------------------- convolution

// Cross-correlation over time. `w` is [out_ch, in_ch, k].
//
// This is where essentially all of the model's arithmetic is - the decoder's
// residual blocks are 99% of the FLOP count and every one of them is this
// kernel - so it is the one place that is written for speed rather than for
// obviousness. Two things make it fast, and the first matters far more:
//
//   1. **Each thread computes OC_TILE channels x T_REG time positions.** The
//      naive form does one multiply-add per value loaded from `x` and one per
//      weight loaded, which leaves it entirely load-bound. The channel tile
//      reuses each `x` value OC_TILE times; the time tile reuses each weight
//      T_REG times. Both were needed: the channel tile alone took the decoder
//      from 4.6% of the card's fp32 peak to 13.7%, and it stalled there
//      because the weight loads had become the limit.
//   2. **The input window lives in shared memory.** Adjacent output positions
//      read overlapping windows, so without this each value is fetched `k`
//      times.
//
// Its differential twin is `xabe_dsp::conv1d`, which stays the plain triple
// loop. That is the division of labour the whole project is built on: the
// readable one defines correct, the fast one is tested against it.
// The body, parameterised so that one implementation serves both the long
// sequences the decoder produces and the short ones everything else works on.
template <int OC_TILE, int T_REG>
__device__ __forceinline__ void conv1d_body(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int pad_left, int dilation, int out_t,
    float* sx)
{
    int oc0 = blockIdx.x * OC_TILE;
    int per_block = blockDim.x * T_REG;
    int p0 = blockIdx.y * per_block;

    float acc[OC_TILE][T_REG];
    #pragma unroll
    for (int a = 0; a < OC_TILE; ++a) {
        float b0 = (oc0 + a < out_ch && bias) ? bias[oc0 + a] : 0.0f;
        #pragma unroll
        for (int j = 0; j < T_REG; ++j) acc[a][j] = b0;
    }

    int span = (k - 1) * dilation + 1;
    int tile = per_block + span - 1;
    // Positions rise with j, so a thread whose first one is past the end has
    // nothing to do at all. Without this the tail block does a full block's
    // worth of arithmetic and stores none of it.
    bool active = (p0 + (int)threadIdx.x) < out_t;

    for (int i = 0; i < in_ch; ++i) {
        for (int sIdx = threadIdx.x; sIdx < tile; sIdx += blockDim.x) {
            int pos = p0 + sIdx - pad_left;
            sx[sIdx] = (pos >= 0 && pos < t) ? x[(size_t)i * t + pos] : 0.0f;
        }
        __syncthreads();

        if (active) {
            const float* wi = w + ((size_t)oc0 * in_ch + i) * k;
            for (int tap = 0; tap < k; ++tap) {
                float xv[T_REG];
                #pragma unroll
                for (int j = 0; j < T_REG; ++j) {
                    xv[j] = sx[threadIdx.x + j * blockDim.x + tap * dilation];
                }
                #pragma unroll
                for (int a = 0; a < OC_TILE; ++a) {
                    // One weight load, T_REG multiply-adds. Without the time
                    // register tile this loaded a weight per multiply-add,
                    // which is what held the decoder at a seventh of peak.
                    float wv = wi[(size_t)a * in_ch * k + tap];
                    #pragma unroll
                    for (int j = 0; j < T_REG; ++j) {
                        acc[a][j] = fmaf(xv[j], wv, acc[a][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) return;
    #pragma unroll
    for (int j = 0; j < T_REG; ++j) {
        int p = p0 + threadIdx.x + j * blockDim.x;
        if (p >= out_t) continue;
        #pragma unroll
        for (int a = 0; a < OC_TILE; ++a) {
            int o = oc0 + a;
            if (o < out_ch) out[(size_t)o * out_t + p] = acc[a][j];
        }
    }
}

extern "C" __global__ void conv1d(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int pad_left, int dilation, int out_t)
{
    extern __shared__ float sx[];
    conv1d_body<8, 4>(x, w, bias, out, in_ch, t, out_ch, k, pad_left, dilation, out_t, sx);
}

// For sequences too short to fill a four-deep time tile. The text encoder runs
// over 69 symbols and the flow over a couple of hundred frames; at T_REG = 4
// three quarters of every thread's arithmetic would fall past the end.
extern "C" __global__ void conv1d_short(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int pad_left, int dilation, int out_t)
{
    extern __shared__ float sx[];
    conv1d_body<8, 1>(x, w, bias, out, in_ch, t, out_ch, k, pad_left, dilation, out_t, sx);
}

extern "C" {

// One kernel per channel; `w` is [ch, k].
__global__ void depthwise_conv1d(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int ch, int t, int k, int pad_left, int dilation, int out_t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ch * out_t) return;
    int c = idx / out_t;
    int p = idx - c * out_t;

    float acc = bias ? bias[c] : 0.0f;
    for (int tap = 0; tap < k; ++tap) {
        int pos = p + tap * dilation - pad_left;
        if (pos < 0 || pos >= t) continue;
        acc = fmaf(x[(size_t)c * t + pos], w[(size_t)c * k + tap], acc);
    }
    out[idx] = acc;
}

// Transposed convolution, written as a gather so no atomics are needed.
//
// The CPU twin scatters: each input contributes to `k` outputs. Inverting that
// is where the off-by-ones live, so it is spelled out. An output position `p`
// is reached only by taps congruent to `(p + pad) mod stride`, and the input
// index for such a tap is `(p + pad - tap) / stride`.
//
// `w` is [in_ch, out_ch, k]; the bias is per *output* channel.
__global__ void transposed_conv1d(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int stride, int pad, int out_t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * out_t) return;
    int o = idx / out_t;
    int p = idx - o * out_t;

    float acc = bias ? bias[o] : 0.0f;
    int shifted = p + pad;
    int first = shifted % stride;
    for (int i = 0; i < in_ch; ++i) {
        const float* xi = x + (size_t)i * t;
        const float* wi = w + ((size_t)i * out_ch + o) * k;
        for (int tap = first; tap < k; tap += stride) {
            int n = (shifted - tap) / stride;
            if (n < 0 || n >= t) continue;
            acc = fmaf(xi[n], wi[tap], acc);
        }
    }
    out[idx] = acc;
}

// -------------------------------------------------------------------- dense

// y[t][o] = bias[o] + sum_i x[t][i] * w[o][i]. PyTorch's nn.Linear layout.
__global__ void linear(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int rows, int in_c, int out_c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * out_c) return;
    int r = idx / out_c;
    int o = idx - r * out_c;

    float acc = bias ? bias[o] : 0.0f;
    const float* xr = x + (size_t)r * in_c;
    const float* wo = w + (size_t)o * in_c;
    for (int i = 0; i < in_c; ++i) acc = fmaf(xr[i], wo[i], acc);
    out[idx] = acc;
}

// -------------------------------------------------------------- normalisation

// One block per row. Two shared reductions: sum, then sum of squares.
// One block a row. The two reductions - the mean, then the variance about it,
// which is the two-pass form `torch.nn.LayerNorm` computes - go through warp
// shuffles and one barrier each rather than a shared-memory tree with a
// barrier at every level, and the row stays in registers between the passes
// when it fits. That is the same shape `rms_norm` has, and for the same
// reason: at one decoded row this block is all the parallelism there is, and
// what it costs is the length of its dependency chain rather than what it
// reads. A whole row of 1280 was 11 us beyond the launch floor on the old
// tree; docs/BENCHMARKS.md has what this one measures.
//
// Four floats a thread when the row is a whole number of them, which every
// row here is; the scalar path is the general contract. The register cache
// holds `LN_REG` float4 a thread, so rows up to `4 * LN_REG * blockDim.x` are
// read once - past that the later passes re-read the row, which is a
// correctness-neutral slow path rather than a refusal.
#define LN_REG 8

__device__ __forceinline__ float ln_block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            t += __shfl_xor_sync(0xffffffff, t, o);
        }
        if (lane == 0) {
            red[0] = t;
        }
    }
    __syncthreads();
    const float r = red[0];
    // The next reduction writes `red` again; nobody may still be reading it.
    __syncthreads();
    return r;
}

} // extern "C" - an overload set and a template cannot have C linkage.

__device__ __forceinline__ void norm_store(float* p, int i, float v) {
    p[i] = v;
}
__device__ __forceinline__ void norm_store(unsigned short* p, int i, float v) {
    p[i] = f32_to_f16(v);
}

// `ADD` folds the residual sum in: `h += res` on the way through the first
// pass, so `h` holds the sum for every pass below and for whatever adds to the
// residual stream next. Without it `h` is read and never written.
template <typename OUT, bool ADD>
__device__ __forceinline__ void layer_norm_impl(
    float* __restrict__ h, const float* __restrict__ res,
    const float* __restrict__ weight, const float* __restrict__ bias,
    OUT* __restrict__ out, int cols, float eps)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    const size_t base = (size_t)row * cols;
    float* hr = h + base;
    const float* rr = res + base;
    OUT* outr = out + base;
    const bool wide = (cols & 3) == 0;
    const int n4 = cols >> 2;

    float4 keep[LN_REG];
    float sum = 0.0f;
    if (wide) {
        float4* h4 = (float4*)hr;
        const float4* r4 = (const float4*)rr;
        int slot = 0;
        for (int i = tid; i < n4; i += nt, ++slot) {
            float4 v = h4[i];
            if (ADD) {
                const float4 r = r4[i];
                v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
                h4[i] = v;
            }
            if (slot < LN_REG) {
                keep[slot] = v;
            }
            sum += (v.x + v.y) + (v.z + v.w);
        }
    } else {
        for (int i = tid; i < cols; i += nt) {
            float v = hr[i];
            if (ADD) {
                v += rr[i];
                hr[i] = v;
            }
            sum += v;
        }
    }
    const float mean = ln_block_sum(sum, red) / (float)cols;

    float sq = 0.0f;
    if (wide) {
        const float4* h4 = (const float4*)hr;
        int slot = 0;
        for (int i = tid; i < n4; i += nt, ++slot) {
            const float4 v = (slot < LN_REG) ? keep[slot] : h4[i];
            const float a = v.x - mean, b = v.y - mean, c = v.z - mean, d = v.w - mean;
            sq += (a * a + b * b) + (c * c + d * d);
        }
    } else {
        for (int i = tid; i < cols; i += nt) {
            const float d = hr[i] - mean;
            sq += d * d;
        }
    }
    // The biased variance, matching torch.nn.LayerNorm.
    const float inv = rsqrtf(ln_block_sum(sq, red) / (float)cols + eps);

    if (wide) {
        const float4* h4 = (const float4*)hr;
        const float4* w4 = (const float4*)weight;
        const float4* b4 = (const float4*)bias;
        int slot = 0;
        for (int i = tid; i < n4; i += nt, ++slot) {
            const float4 v = (slot < LN_REG) ? keep[slot] : h4[i];
            const float4 w = w4[i], b = b4[i];
            norm_store(outr, 4 * i + 0, (v.x - mean) * inv * w.x + b.x);
            norm_store(outr, 4 * i + 1, (v.y - mean) * inv * w.y + b.y);
            norm_store(outr, 4 * i + 2, (v.z - mean) * inv * w.z + b.z);
            norm_store(outr, 4 * i + 3, (v.w - mean) * inv * w.w + b.w);
        }
    } else {
        for (int i = tid; i < cols; i += nt) {
            norm_store(outr, i, (hr[i] - mean) * inv * weight[i] + bias[i]);
        }
    }
}

extern "C" {

__global__ void layer_norm(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ out,
    int cols, float eps)
{
    // `x` is not written: `ADD` is off, and the only store to `h` is under it.
    layer_norm_impl<float, false>((float*)x, x, weight, bias, out, cols, eps);
}

// The same, taking the residual sum on the way in.
//
// Every normalisation in a transformer block reads `h + out` where `out` is
// what the previous sub-layer produced, and nothing between the two reads `h`.
// Done as two kernels that is a pass to add, a pass to write `h`, and then the
// normalisation reading `h` again - five passes and two launches where this is
// four and one. On a decode step the passes are five kilobytes and it is the
// launch that costs; on the encoder's 1500 rows it is the passes.
//
// `h` is updated in place because the residual stream is what the next
// sub-layer adds to, so the sum has to survive; `out` is dead afterwards.
// The output width is a template parameter because every consumer of a
// normalisation here is a matmul, and a matmul stages its left operand as f16
// whatever it is handed. Rounding in this kernel, where the value is already
// in a register, is therefore free and *bit-identical*: `f32_to_f16` is the
// same round-to-nearest-even `gemm_pack` applies during staging, so an operand
// narrowed here and one narrowed inside the matmul are the same bits. What it
// buys is the re-reads - a projection's activation is read once per column
// tile, ten times at encoder width and forty at the MLP's - and halving that
// stream measured about 5% of each of those matmuls.
//
// `h` stays f32 throughout: it is the residual stream, it is added to rather
// than multiplied by, and narrowing it would be a real approximation.
// The implementation is `layer_norm_impl<OUT, true>` above.

__global__ void layer_norm_add(
    float* __restrict__ h, const float* __restrict__ res,
    const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ out, int cols, float eps)
{
    layer_norm_impl<float, true>(h, res, weight, bias, out, cols, eps);
}

__global__ void layer_norm_add_f16(
    float* __restrict__ h, const float* __restrict__ res,
    const float* __restrict__ weight, const float* __restrict__ bias,
    unsigned short* __restrict__ out, int cols, float eps)
{
    layer_norm_impl<unsigned short, true>(h, res, weight, bias, out, cols, eps);
}

// One block per row; subtracts the row max before exponentiating.
__global__ void softmax_rows(float* __restrict__ x, int cols)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* xr = x + (size_t)row * cols;

    // NVRTC compiles without the host math headers, so `INFINITY` is not
    // defined here; this is its bit pattern.
    float m = __int_as_float(0xff800000);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    sdata[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    m = sdata[0];
    __syncthreads();

    float partial = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = __expf(xr[i] - m);
        xr[i] = e;
        partial += e;
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] *= inv;
}

// Softmax over each row, with the scale and the causal mask folded in.
//
// Three kernels and three passes over the score matrix became one. The mask
// costs a comparison rather than a pass, and at a single decode step it costs
// nothing at all - the one query is the last position, so no column is masked -
// which is the case the unfused version paid a full launch and a full pass for.
//
// Warp shuffles rather than a shared-memory tree, for the reason `rms_norm`
// gives: at one row of scores this block is all the parallelism there is.
extern "C" __global__ void softmax_causal(
    float* __restrict__ x, int cols, int tq, int offset, float scale)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const int lim = (row % tq) + offset;
    float* xr = x + (size_t)row * cols;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int warps = nt >> 5;
    // NVRTC compiles without the host math headers, so this is -INFINITY's bit
    // pattern.
    const float ninf = __int_as_float(0xff800000);

    float m = ninf;
    for (int i = tid; i < cols; i += nt) {
        m = fmaxf(m, (i > lim) ? ninf : xr[i] * scale);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (lane == 0) {
        red[warp] = m;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < warps) ? red[lane] : ninf;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        if (lane == 0) {
            red[0] = v;
        }
    }
    __syncthreads();
    m = red[0];
    __syncthreads();

    float partial = 0.0f;
    for (int i = tid; i < cols; i += nt) {
        float e = (i > lim) ? 0.0f : __expf(xr[i] * scale - m);
        xr[i] = e;
        partial += e;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, o);
    }
    if (lane == 0) {
        red[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        if (lane == 0) {
            red[0] = v;
        }
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int i = tid; i < cols; i += nt) {
        xr[i] *= inv;
    }
}

// --------------------------------------------------------------- activations

__global__ void act_relu(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

__global__ void act_leaky_relu(float* x, int n, float slope)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] *= slope;
}

/* Snake: x + sin^2(a*x)/a, with one alpha per channel.
 *
 * HiFTNet's activation, and the reason the vocoder is not a plain HiFi-GAN.
 * It is periodic, which is what lets the network represent a harmonic signal
 * without having to learn one - and it is why `alpha` is a trained parameter
 * per channel rather than a constant.
 *
 * `1e-9` guards the division exactly where upstream puts it: on the divisor,
 * not on the result. An alpha that has trained to zero makes this the identity
 * plus a very large term, which is upstream's behaviour and not something to
 * quietly improve. */
__global__ void act_snake(float* x, const float* __restrict__ alpha, int ch, int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ch * t) {
        float a = alpha[i / t];
        float s = __sinf(x[i] * a);
        x[i] += s * s / (a + 1e-9f);
    }
}

/* ELU with alpha 1, which is torch's default and the only one used here. */
__global__ void act_elu(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = __expf(x[i]) - 1.0f;
}

/* Mish: x * tanh(softplus(x)).
 *
 * `log1pf(expf(x))` rather than `logf(1 + expf(x))`: the second loses every
 * bit of a small `x` to the addition, and overflows to infinity for x above
 * about 88 where softplus should simply be x. The threshold below keeps the
 * large branch exact instead of relying on `log1p` to recover it. */
__global__ void act_mish(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float sp = v > 20.0f ? v : log1pf(expf(v));
        x[i] = v * tanhf(sp);
    }
}

/* SiLU, on its own rather than fused with a gate. */
__global__ void act_silu(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + __expf(-x[i]));
}

/* GELU's tanh approximation, which is `nn.GELU(approximate="tanh")`.
 *
 * A different function from `act_gelu`'s exact erf form - they agree to about
 * 1e-3, which is far more than a rounding difference and far less than an
 * obvious break. The DiT's feed-forward asks for this one by name. */
__global__ void act_gelu_tanh(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ void act_tanh(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// The exact erf GELU, matching torch's default. `erff` is IEEE-accurate on the
// device, so unlike the CPU twin this needs no rational approximation.
__global__ void act_gelu(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    }
}

// The same function, reading f32 and writing f16 somewhere else.
//
// For the one place the result is a matmul's left operand and nothing else -
// the encoder's MLP, whose inner activation is 30.7 MB that `fc2` re-reads once
// per column tile. Out of place rather than in place because the widths differ;
// bit-identical to running `act_gelu` and letting the matmul stage the result,
// for the reason `norm_store` gives.
__global__ void act_gelu_f16(
    const float* __restrict__ x, unsigned short* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = f32_to_f16(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

// tanh(first half) * sigmoid(second half), 2*ch in and ch out.
// The same gate, for data laid out `[t, 2 * ch]` rather than `[2 * ch, t]`.
//
// WaveGlow's coupling network is a chain of matmuls, and a matmul wants the
// contracted axis last - which puts the channels innermost. Transposing into
// the channel-major form just to gate would undo that once per layer.
__global__ void gated_activation_rows(
    const float* __restrict__ x, float* __restrict__ out, int ch, int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch * t) return;
    int p = i / ch;
    int c = i - p * ch;
    float a = x[(size_t)p * 2 * ch + c];
    float b = x[(size_t)p * 2 * ch + ch + c];
    out[i] = tanhf(a) * (1.0f / (1.0f + __expf(-b)));
}

__global__ void gated_activation(
    const float* __restrict__ x, float* __restrict__ out, int ch, int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch * t) return;
    int c = i / t;
    int p = i - c * t;
    float a = x[(size_t)c * t + p];
    float b = x[(size_t)(ch + c) * t + p];
    out[i] = tanhf(a) * (1.0f / (1.0f + __expf(-b)));
}

// -------------------------------------------------------------- element-wise

__global__ void add_inplace(float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// `a[r * cols + j] += b[r * stride + off + j]`.
//
// The strided twin of `add_inplace`, for adding one column block of a wide
// matrix to a narrow one. WaveGlow's conditioning is a single `[steps, 2 * ch *
// layers]` product - one matmul rather than one per layer, which measured
// 1.30x on the shape - and a layer's share of it is a column range of every
// row rather than a contiguous run.
__global__ void add_strided(
    float* a, const float* b, int cols, int stride, int off, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols * rows) return;
    int r = i / cols, j = i - r * cols;
    a[i] += b[(size_t)r * stride + off + j];
}

__global__ void sub_inplace(float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] -= b[i];
}

__global__ void mul_inplace(float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= b[i];
}

__global__ void scale_inplace(float* a, int n, float s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

// Copies a contiguous range out of a larger buffer.
//
// Exists so that a channel split - the text encoder's projection into a mean
// and a log-variance, the WaveNet's residual and skip halves - is an explicit
// copy rather than a device view threaded through every signature.
__global__ void copy_range(
    const float* __restrict__ x, float* __restrict__ out, int offset, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[offset + i];
}

// Writes `n` values into `dst` starting at `offset`. The inverse of
// `copy_range`, and what makes a channel concatenation one launch.
__global__ void copy_into(
    float* __restrict__ dst, const float* __restrict__ src, int offset, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[offset + i] = src[i];
}

__global__ void transpose(
    const float* __restrict__ x, float* __restrict__ out, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols;
    int c = i - r * cols;
    out[(size_t)c * rows + r] = x[i];
}

// Reverses the channel axis. Not a swap of halves - see docs/MODEL.md.
__global__ void flip_channels(
    const float* __restrict__ x, float* __restrict__ out, int ch, int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch * t) return;
    int c = i / t;
    int p = i - c * t;
    out[i] = x[(size_t)(ch - 1 - c) * t + p];
}

// ------------------------------------------------------------------ specific

// Embedding lookup with the sqrt(hidden) scaling folded in.
__global__ void embed_scaled(
    const float* __restrict__ table, const long long* __restrict__ ids,
    float* __restrict__ out, int t, int ch, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * ch) return;
    int pos = i / ch;
    int c = i - pos * ch;
    out[i] = table[(size_t)ids[pos] * ch + c] * scale;
}

// Fuses weight_norm's direction and magnitude into a plain kernel.
// One block per output channel.
__global__ void fuse_weight_norm(
    const float* __restrict__ v, const float* __restrict__ g,
    float* __restrict__ out, int per)
{
    extern __shared__ float sdata[];
    int o = blockIdx.x;
    const float* vo = v + (size_t)o * per;
    float* oo = out + (size_t)o * per;

    float partial = 0.0f;
    for (int i = threadIdx.x; i < per; i += blockDim.x) partial += vo[i] * vo[i];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float scale = g[o] / sqrtf(sdata[0]);
    for (int i = threadIdx.x; i < per; i += blockDim.x) oo[i] = vo[i] * scale;
}

// Attention logits with the windowed relative bias, per head.
// q, k are [t, embed]; emb_rel_k is [2*window+1, head_dim]; out is [heads, t, t].
__global__ void attention_scores(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ emb_rel_k, float* __restrict__ out,
    int t, int embed, int heads, int head_dim, int window, float scaling)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= heads * t * t) return;
    int h = idx / (t * t);
    int rem = idx - h * t * t;
    int i = rem / t;
    int j = rem - i * t;

    int base = h * head_dim;
    const float* qi = q + (size_t)i * embed + base;
    const float* kj = k + (size_t)j * embed + base;

    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) acc = fmaf(qi[d] * scaling, kj[d], acc);

    int r = window + j - i;
    if (r >= 0 && r < 2 * window + 1) {
        const float* e = emb_rel_k + (size_t)r * head_dim;
        for (int d = 0; d < head_dim; ++d) acc = fmaf(qi[d] * scaling, e[d], acc);
    }
    out[idx] = acc;
}

// Attention output with the windowed relative value term.
__global__ void attention_context(
    const float* __restrict__ probs, const float* __restrict__ v,
    const float* __restrict__ emb_rel_v, float* __restrict__ out,
    int t, int embed, int heads, int head_dim, int window)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t * embed) return;
    int i = idx / embed;
    int c = idx - i * embed;
    int h = c / head_dim;
    int d = c - h * head_dim;

    const float* pr = probs + ((size_t)h * t + i) * t;
    float acc = 0.0f;
    for (int j = 0; j < t; ++j) {
        float a = pr[j];
        acc = fmaf(a, v[(size_t)j * embed + c], acc);
        int r = window + j - i;
        if (r >= 0 && r < 2 * window + 1) {
            acc = fmaf(a, emb_rel_v[(size_t)r * head_dim + d], acc);
        }
    }
    out[idx] = acc;
}

// Length regulation and prior sampling in one pass.
// m_p and logs_p are [symbols, ch]; alignment is [frames]; out is [ch, frames].
__global__ void expand_prior(
    const float* __restrict__ m_p, const float* __restrict__ logs_p,
    const int* __restrict__ alignment, const float* __restrict__ noise,
    float* __restrict__ out, int ch, int frames, float noise_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ch * frames) return;
    int c = i / frames;
    int f = i - c * frames;
    int s = alignment[f];
    float mean = m_p[(size_t)s * ch + c];
    float logs = logs_p[(size_t)s * ch + c];
    out[i] = fmaf(noise[i] * __expf(logs), noise_scale, mean);
}

}
// -------------------------------------------------------------------- llama

// Root-mean-square normalisation, one block per row.
//
// Not layer normalisation: no mean subtraction and no bias. Substituting one
// for the other passes every shape check and shifts every activation by the
// row's mean, which on a residual stream is not small.
// Root-mean-square normalisation, optionally emitting the int8 twin as well.
//
// One block a row, because the reduction is over the whole row. That is only a
// block's worth of parallelism at a single decode step, so the two things that
// cost are latency and synchronisation: the reduction is warp shuffles with one
// `__syncthreads` between the warp results and their combination, rather than a
// shared-memory tree with a barrier at every halving.
//
// With `qa`, it also writes the codes and scales `quantize_q8` would have -
// which is the same arithmetic over data this kernel already holds, against a
// second launch that re-reads the row it just wrote. The scale is per 32
// columns, and a warp covers exactly 32 consecutive columns per iteration, so
// the group reduction is a shuffle and needs no memory at all. That mapping is
// why `dim` must be a multiple of the block: a ragged last iteration would
// leave lanes inactive inside a full-mask shuffle.
extern "C" __global__ void rms_norm(
    float* __restrict__ x,
    const float* __restrict__ add,
    const float* __restrict__ weight,
    float* __restrict__ out,
    signed char* __restrict__ qa,
    int asc_off,
    int dim, float eps)
{
    __shared__ float red[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int warps = nt >> 5;
    const size_t base = (size_t)row * dim;
    const int n4 = dim >> 2;

    // Four floats a thread, not one. A whole hidden row is 16 KB and one block
    // is all the parallelism a single decode step has, so what decides this
    // kernel is how many loads a thread keeps in flight - the same finding the
    // packed mat-vec turned on, and the same fix.
    float4* xr = (float4*)(x + base);
    const float4* ar = add ? (const float4*)(add + base) : (const float4*)0;
    const float4* wr = (const float4*)weight;
    float4* orow = (float4*)(out + base);

    // Four floats a thread needs a row that is a whole number of them. Every
    // model shape here is; the scalar path below is for the general contract,
    // and for it alone - the quantized path is a multiple of 1024 by
    // construction and never takes it.
    const bool wide = (dim & 3) == 0;

    float acc = 0.0f;
    if (!wide) {
        const float* xs = x + base;
        for (int i = tid; i < dim; i += nt) {
            float v = xs[i];
            if (add) {
                v += add[base + i];
                x[base + i] = v;
            }
            acc += v * v;
        }
    }
    for (int i = tid; wide && i < n4; i += nt) {
        float4 v = xr[i];
        // The residual, when there is one. Folded in here rather than left to
        // `add_inplace` because the sum is what this kernel reads anyway, and
        // `x` is updated in place because the residual stream is what the
        // *next* block adds to.
        if (add) {
            float4 r = ar[i];
            v.x += r.x;
            v.y += r.y;
            v.z += r.z;
            v.w += r.w;
            xr[i] = v;
        }
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, o);
    }
    if (lane == 0) {
        red[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        float s = (lane < warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, o);
        }
        if (lane == 0) {
            red[0] = s;
        }
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float)dim + eps);

    // A thread owns four columns, so a scale group of 32 is eight threads and
    // the group reduction is three shuffles inside a warp. The loop runs to a
    // multiple of the block rather than to `n4` so that every lane reaches every
    // shuffle - a thread past the end contributes zero rather than not arriving.
    signed char* qrow = qa + base;
    float* asc = qa ? (float*)(qa + asc_off) + (size_t)row * (dim >> 5) : (float*)0;
    if (!wide) {
        const float* xs = x + base;
        float* os = out + base;
        for (int i = tid; i < dim; i += nt) {
            os[i] = xs[i] * scale * weight[i];
        }
        return;
    }
    const int ceil4 = ((n4 + nt - 1) / nt) * nt;
    for (int i = tid; i < ceil4; i += nt) {
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < n4) {
            float4 v = xr[i], w = wr[i];
            y.x = v.x * scale * w.x;
            y.y = v.y * scale * w.y;
            y.z = v.z * scale * w.z;
            y.w = v.w * scale * w.w;
            orow[i] = y;
        }
        if (!qa) {
            continue;
        }
        float mx = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
        #pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        }
        const float d = mx * (1.0f / 127.0f);
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        if (i < n4) {
            char4 c;
            c.x = (signed char)__float2int_rn(y.x * inv);
            c.y = (signed char)__float2int_rn(y.y * inv);
            c.z = (signed char)__float2int_rn(y.z * inv);
            c.w = (signed char)__float2int_rn(y.w * inv);
            *(char4*)(qrow + (i << 2)) = c;
            if ((tid & 7) == 0) {
                asc[i >> 3] = d;
            }
        }
    }
}



// a = silu(a) * b, the SwiGLU gate.
// `a = silu(a) * b`, optionally emitting the int8 twin as well.
//
// The twin is here for the same reason it is in `rms_norm`: the result goes
// straight into a packed projection, and quantizing it in a kernel of its own
// means a launch and a second read of the row this one just wrote. A block is a
// multiple of 32 threads over a flat index, so a warp owns exactly one
// 32-element scale group and the group reduction is a shuffle.
//
// `n` must be a multiple of the block for the twin, or the last block's tail
// lanes would sit inside a full-mask shuffle without being in the group.
// The same gate, over one buffer that holds both halves.
//
// `gate` and `up` are the same shape and, in every checkpoint here, the same
// block format - so they are one batched product whose output is
// `[2, rows, inter]`, and the two operands this needs are the two halves of it.
// Written as `x[i] = silu(x[i]) * x[i + n]` rather than as two pointers because
// they alias one allocation, and one `&mut` is the only shape Rust will hand
// over. What it buys is a launch a layer, and at one row a launch is most of
// what a small kernel costs.
extern "C" __global__ void silu_mul_pair(
    float* __restrict__ x,
    signed char* __restrict__ qa,
    int asc_off,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n && !qa) {
        return;
    }
    float y = 0.0f;
    if (i < n) {
        // `expf` for the reason `silu_mul` gives.
        float v = x[i];
        y = v * (1.0f / (1.0f + expf(-v))) * x[i + n];
        x[i] = y;
    }
    if (!qa) {
        return;
    }
    // The same group quantiser `silu_mul` ends with: one scale a warp, taken
    // from the largest magnitude in it.
    float* asc = (float*)(qa + asc_off);
    float mx = fabsf(y);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    const float d = mx * (1.0f / 127.0f);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    if (i < n) {
        qa[i] = (signed char)__float2int_rn(y * inv);
        if ((threadIdx.x & 31) == 0) {
            asc[i >> 5] = d;
        }
    }
}

extern "C" __global__ void silu_mul(
    float* __restrict__ a,
    const float* __restrict__ b,
    signed char* __restrict__ qa,
    int asc_off,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n && !qa) {
        return;
    }
    float y = 0.0f;
    if (i < n) {
        float v = a[i];
        // `expf`, not `__expf`. The fast intrinsic is about 2^-21 accurate and
        // this one is a few ulp; the difference costs nothing measurable here
        // and it keeps the differential test against the scalar twin tight
        // enough to catch a real mistake.
        y = v * (1.0f / (1.0f + expf(-v))) * b[i];
        a[i] = y;
    }
    if (!qa) {
        return;
    }
    float* asc = (float*)(qa + asc_off);
    float mx = fabsf(y);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    const float d = mx * (1.0f / 127.0f);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    if (i < n) {
        qa[i] = (signed char)__float2int_rn(y * inv);
        if ((threadIdx.x & 31) == 0) {
            asc[i >> 5] = d;
        }
    }
}

// Rotary position embedding, in place over [t, heads * head_dim].
//
// 🤗 pairs dimension i with i + head_dim/2, not 2i with 2i+1. The two are a
// permutation of each other, both are called RoPE, and picking the wrong one
// gives a model that is coherent for four or five tokens and then drifts -
// the hardest possible thing to debug. `first` is the absolute position of row
// zero, so a decode step past a KV cache rotates by where the token really is.
extern "C" __global__ void rope(
    float* __restrict__ x, long off,
    const float* __restrict__ freq_div, int has_div,
    int t, int heads, int head_dim, float theta, int first)
{
    const int half = head_dim >> 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * heads * half) {
        return;
    }
    const int j = i % half;
    const int h = (i / half) % heads;
    const int p = i / (half * heads);

    // Accurate `powf` and `sincosf`, not the `__` intrinsics. `__sincosf` does
    // no real range reduction, and at position 4095 the angle for the lowest
    // frequency is 4095 radians - measured 6e-4 of full scale out, against
    // 6e-6 here. The reference computes this in f32 too, so the scalar twin
    // does the same rather than being more accurate than what it is a
    // reference for.
    // `freq_div` is Llama-3's per-pair frequency divisor, null for Llama-2.
    // Dividing here rather than pre-scaling `theta` because the factors are
    // not uniform: on Breeze2 they are 1.0 for the first 29 pairs and 8.0 for
    // the last 29, with six interpolated between.
    float inv = powf(theta, -2.0f * (float)j / (float)head_dim);
    // A flag rather than a null check: the host always passes a real pointer,
    // because a launch argument has to point at something and a one-element
    // dummy is cheaper to reason about than a null that must never be read.
    if (has_div) {
        inv /= freq_div[j];
    }
    const float angle = (float)(first + p) * inv;
    float sn, cs;
    sincosf(angle, &sn, &cs);

    // `off` is where this tensor starts. The attention projections are issued
    // as one batched product, so `q` and `k` are two contiguous blocks of one
    // allocation rather than two allocations, and rotating `k` means starting
    // at its block instead of copying it out first.
    const size_t base = (size_t)off + ((size_t)p * heads + h) * head_dim + j;
    const float a = x[base];
    const float b = x[base + half];
    x[base]        = a * cs - b * sn;
    x[base + half] = b * cs + a * sn;
}

// ------------------------------------------------------------------ whisper

// Convolution rewritten as a matrix, so it can use the tensor cores.
//
// Whisper's encoder stem is two convolutions of width 3 over 3000 positions,
// 80 to 1280 channels and then 1280 to 1280 at stride 2. Written as a
// convolution they would want a kernel tuned for a channel count two orders of
// magnitude past what the VITS decoder's is written for. Written as
// `im2col` + `gemm` they are two of the products this card is fastest at, and
// the only new code is this gather.
//
// `x` is [t, in_ch] - time major, the transformer layout - and the output is
// [out_t, in_ch * k], which is exactly the contraction a weight stored as
// [out_ch, in_ch, k] wants. Positions off either end read as zero, which is
// what padding means; a clamp would repeat the first frame instead.
extern "C" __global__ void im2col(
    const float* __restrict__ x,
    float* __restrict__ out,
    int t, int in_ch, int k, int stride, int pad, int dilation, int out_t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cols = in_ch * k;
    if (i >= out_t * cols) {
        return;
    }
    int ti = i / cols;
    int r  = i % cols;
    int c  = r / k;
    int kk = r % k;
    // The column index is `c * k + kk`, which is exactly how a `[out, in, k]`
    // weight flattens - so the matmul that follows needs no reshaping.
    int src = ti * stride + kk * dilation - pad;
    out[i] = (src >= 0 && src < t) ? x[(size_t)src * in_ch + c] : 0.0f;
}

// [t, heads * head_dim] -> [heads, t, head_dim].
//
// Attention is a batch of small products, one per head, and every one of them
// wants its head contiguous. The projections produce the heads interleaved,
// so somebody has to move them; doing it in one pass here is cheaper than
// teaching the matmul a stride it would pay for on every tile.
extern "C" __global__ void split_heads(
    const float* __restrict__ x,
    float* __restrict__ out,
    int t, int heads, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = heads * head_dim;
    if (i >= t * d) {
        return;
    }
    int ti = i / d;
    int h  = (i % d) / head_dim;
    int j  = i % head_dim;
    out[((size_t)h * t + ti) * head_dim + j] = x[i];
}

// [t, heads * head_dim] -> [heads, head_dim, t], the transpose of the above.
//
// The value tensor is the one operand read down its time axis: the context is
// `probs [tq, tk] x V [tk, head_dim]`, and the matmul takes its right operand
// as [n, k]. So V arrives already transposed rather than being transposed
// again inside the product.
// A tiled transpose, because that is all this is.
//
// `[t, heads*head_dim]` to `[heads, head_dim, t]` sends element `(ti, c)` to
// `(c, ti)` for `c = h * head_dim + j`. The head structure never enters the
// address arithmetic - it is the transpose of a `[t, d]` matrix and nothing
// more, and writing it that way is what makes the staging tile obvious.
//
// One element a thread reads coalesced and writes scattered: consecutive lanes
// land `t` floats apart, so a warp's store is 32 sectors where a coalesced one
// is four. It measured 141 GB/s on a card that copies at about 500, which is
// 3.5 ms across the encoder's 32 layers for a pass that carries no arithmetic.
// Staging a 32x32 tile in shared makes both halves coalesced.
//
// The tile row is 33 wide, and that padding is the whole trick: the write-back
// reads a *column* of the tile, and at a stride of 32 a column is one bank and
// the read is 32-way conflicted. At 33 it walks all 32.
#define TR_TILE 32
#define TR_ROWS 8

extern "C" __global__ void split_heads_t(
    const float* __restrict__ x,
    float* __restrict__ out,
    int t, int heads, int head_dim)
{
    __shared__ float tile[TR_TILE][TR_TILE + 1];
    const int d  = heads * head_dim;
    const int c0 = blockIdx.x * TR_TILE;
    const int r0 = blockIdx.y * TR_TILE;

    const int c = c0 + (int)threadIdx.x;
    #pragma unroll
    for (int o = 0; o < TR_TILE; o += TR_ROWS) {
        const int r = r0 + (int)threadIdx.y + o;
        if (r < t && c < d) {
            tile[threadIdx.y + o][threadIdx.x] = x[(size_t)r * d + c];
        }
    }
    __syncthreads();

    // `x` runs down the tile's first axis now, so the lane index picks the row
    // of the source and the store is contiguous in `t`.
    const int rt = r0 + (int)threadIdx.x;
    #pragma unroll
    for (int o = 0; o < TR_TILE; o += TR_ROWS) {
        const int cc = c0 + (int)threadIdx.y + o;
        if (rt < t && cc < d) {
            out[(size_t)cc * t + rt] = tile[threadIdx.x][threadIdx.y + o];
        }
    }
}

// The two above, writing f16 instead of f32.
//
// The cross-attention cache is built once an utterance and then read whole by
// every decode step, so it is held packed - which used to mean a `split_heads`
// pass at f32 and a `pack_f16` pass after it, reading and writing the same 7.7
// MB tensor twice to change nothing but its width. The split is already
// touching every element and `f32_to_f16` rounds the way `gemm_pack` does, so
// the conversion is free where the element is already in a register.
//
// `bias`, when not null, is a `[heads * head_dim]` row added before the
// rounding. It is there so the cross-attention projections of every layer can
// go out as one batched matmul, which carries one bias for the whole batch,
// while each layer keeps its own: the add is the same f32 add the matmul's
// epilogue would have made, so the bits do not change.
extern "C" __global__ void split_heads_f16(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    unsigned short* __restrict__ out,
    int t, int heads, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = heads * head_dim;
    if (i >= t * d) {
        return;
    }
    int ti = i / d;
    int h  = (i % d) / head_dim;
    int j  = i % head_dim;
    const float v = x[i] + (bias ? bias[i % d] : 0.0f);
    out[((size_t)h * t + ti) * head_dim + j] = f32_to_f16(v);
}

extern "C" __global__ void split_heads_t_f16(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    unsigned short* __restrict__ out,
    int t, int heads, int head_dim)
{
    // The f32 twin's tile, and its note on why the row is 33 wide. The store is
    // two bytes a lane rather than four, which is half a coalesced warp's
    // worth - still four sectors against the scattered form's 32.
    __shared__ float tile[TR_TILE][TR_TILE + 1];
    const int d  = heads * head_dim;
    const int c0 = blockIdx.x * TR_TILE;
    const int r0 = blockIdx.y * TR_TILE;

    const int c = c0 + (int)threadIdx.x;
    #pragma unroll
    for (int o = 0; o < TR_TILE; o += TR_ROWS) {
        const int r = r0 + (int)threadIdx.y + o;
        if (r < t && c < d) {
            tile[threadIdx.y + o][threadIdx.x] = x[(size_t)r * d + c] + (bias ? bias[c] : 0.0f);
        }
    }
    __syncthreads();

    const int rt = r0 + (int)threadIdx.x;
    #pragma unroll
    for (int o = 0; o < TR_TILE; o += TR_ROWS) {
        const int cc = c0 + (int)threadIdx.y + o;
        if (rt < t && cc < d) {
            out[(size_t)cc * t + rt] = f32_to_f16(tile[threadIdx.x][threadIdx.y + o]);
        }
    }
}

// [heads, t, head_dim] -> [t, heads * head_dim]. The inverse of `split_heads`.
//
// `qa` non-null asks for the int8 twin beside the merge, exactly as
// `silu_mul` and `rms_norm` produce one: the merged row is what the output
// projection multiplies, and quantizing it in a pass of its own re-reads the
// whole context - 10 MB a layer at 512 tokens - for numbers this kernel is
// already holding. A group of 32 is one warp because consecutive threads
// write consecutive output elements.
extern "C" __global__ void merge_heads(
    const float* __restrict__ x,
    float* __restrict__ out,
    signed char* __restrict__ qa,
    int asc_off,
    int t, int heads, int head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = heads * head_dim;
    if (i >= t * d && !qa) {
        return;
    }
    float y = 0.0f;
    if (i < t * d) {
        int ti = i / d;
        int h  = (i % d) / head_dim;
        int j  = i % head_dim;
        y = x[((size_t)h * t + ti) * head_dim + j];
        out[i] = y;
    }
    if (!qa) {
        return;
    }
    float* asc = (float*)(qa + asc_off);
    float mx = fabsf(y);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    const float dd = mx * (1.0f / 127.0f);
    const float inv = dd > 0.0f ? 1.0f / dd : 0.0f;
    if (i < t * d) {
        qa[i] = (signed char)__float2int_rn(y * inv);
        if ((threadIdx.x & 31) == 0) {
            asc[i >> 5] = dd;
        }
    }
}

// Masks a batch of score matrices so a query cannot see a later key.
//
// `offset` is how many cached keys precede the queries: with a KV cache the
// query at index `i` is really at position `offset + i`, and masking without
// it would hide the entire cache. Decoding one token at a time makes the mask
// a no-op, which is exactly why the bug survives until something feeds the
// decoder two tokens at once.
// One tensor from f32 to packed f16, so the matmul can read half the bytes.
extern "C" __global__ void pack_f16(
    const float* __restrict__ x,
    unsigned short* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = f32_to_f16(x[i]);
}

extern "C" __global__ void causal_mask(
    float* __restrict__ scores,
    int batch, int tq, int tk, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * tq * tk) {
        return;
    }
    int j  = i % tk;
    int qi = (i / tk) % tq;
    if (j > qi + offset) {
        // NVRTC has no <math.h>, so INFINITY is not reliably a macro here.
        scores[i] = __int_as_float(0xff800000);
    }
}

// Repeats each key/value head `group` times, widening a grouped-query cache
// to the full query-head count.
//
// Input is `[kv_heads, t, head_dim]` as `split_heads` leaves it; output is
// `[heads, t, head_dim]` with head `h` reading from `h / group`. The
// alternative - teaching the batched matmul to advance its weight pointer
// every `group` batches - would be faster and is a kernel change with a
// differential test of its own; this is the same arithmetic with one extra
// read of the cache, and the cache is small next to the projections.
// Writes one step's keys into a head-major cache.
//
// `src` is the projection's output, `[n, kv_heads * head_dim]`; `dst` is
// `[kv_heads, capacity, head_dim]`. The scatter is the whole point: written
// this way, attention reads a head's keys as `[tk, head_dim]` at a fixed offset
// and needs no transpose, no head split and no expansion of the grouped heads.
//
// The cache used to be stored the way it arrives and rearranged every step -
// four kernels and four allocations a layer, all of them producing a tensor
// that was thrown away before the next token. This is that work done once, at
// the only moment the data is small.
extern "C" __global__ void cache_append(
    const float* __restrict__ src, long src_off, float* __restrict__ dst,
    int n, int kv_heads, int head_dim, int cap, int past)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int d = kv_heads * head_dim;
    if (i >= (size_t)n * d) {
        return;
    }
    const int ti = (int)(i / d);
    const int h = (int)((i % d) / head_dim);
    const int j = (int)(i % head_dim);
    dst[((size_t)h * cap + past + ti) * head_dim + j] = src[(size_t)src_off + i];
}

// The same for values, which attention contracts over rather than along.
//
// `dst` is `[kv_heads, head_dim, capacity]`, so a head's values are already the
// transpose the second matmul wants. That its rows are `capacity` apart rather
// than `tk` is what `w_rs` in `gemv` and `gemm` is for.
extern "C" __global__ void cache_append_t(
    const float* __restrict__ src, long src_off, float* __restrict__ dst,
    int n, int kv_heads, int head_dim, int cap, int past)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int d = kv_heads * head_dim;
    if (i >= (size_t)n * d) {
        return;
    }
    const int ti = (int)(i / d);
    const int h = (int)((i % d) / head_dim);
    const int j = (int)(i % head_dim);
    dst[((size_t)h * head_dim + j) * cap + past + ti] = src[(size_t)src_off + i];
}

// The same two appends, writing an f16 cache.
//
// The cache is the one buffer in a decode whose size is the *context* rather
// than the checkpoint, so it is the one place where halving a width halves
// something that grows. Attention re-reads all of it for every token, and at
// 2048 positions that is 537 MB against the weights' 4.6 GB - see
// docs/BENCHMARKS.md. Rounding is `f32_to_f16`'s, which is round-to-nearest-
// even and the same rounding the tiled kernels do when they stage.
extern "C" __global__ void cache_append_f16(
    const float* __restrict__ src, long src_off, unsigned short* __restrict__ dst,
    int n, int kv_heads, int head_dim, int cap, int past)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int d = kv_heads * head_dim;
    if (i >= (size_t)n * d) {
        return;
    }
    const int ti = (int)(i / d);
    const int h = (int)((i % d) / head_dim);
    const int j = (int)(i % head_dim);
    dst[((size_t)h * cap + past + ti) * head_dim + j] =
        f32_to_f16(src[(size_t)src_off + i]);
}

extern "C" __global__ void cache_append_t_f16(
    const float* __restrict__ src, long src_off, unsigned short* __restrict__ dst,
    int n, int kv_heads, int head_dim, int cap, int past)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int d = kv_heads * head_dim;
    if (i >= (size_t)n * d) {
        return;
    }
    const int ti = (int)(i / d);
    const int h = (int)((i % d) / head_dim);
    const int j = (int)(i % head_dim);
    dst[((size_t)h * head_dim + j) * cap + past + ti] =
        f32_to_f16(src[(size_t)src_off + i]);
}

// One launch for everything between the attention projections and the
// attention itself, at a single decoded position: the query rotated in place,
// the key rotated and stored, the value stored - into the f16 caches at `pos`.
//
// At one row these were four launches a layer - `rope` twice, `cache_append_f16`
// and its transpose - each moving a few kilobytes, so each costing what a
// launch costs and nothing else. The arithmetic is theirs character for
// character: the same `powf` and `sincosf`, the same pairing of `j` with
// `j + half`, the same `f32_to_f16`. The rotated key is never written back to
// the projection buffer, because at one row nothing reads it from there: the
// attention reads the cache.
//
// `q`, `k` and `v` may be three offsets into one allocation - the translator
// issues its projections as one batched product - or three allocations, so
// none of them is `__restrict__`. The ranges they name never overlap.
extern "C" __global__ void rope_cache_f16(
    float* q, long q_off,
    const float* k, long k_off,
    const float* v, long v_off,
    const float* __restrict__ freq_div, int has_div,
    int heads, int kv_heads, int head_dim, float theta, int pos,
    unsigned short* __restrict__ kc, unsigned short* __restrict__ vc, int cap)
{
    const int half = head_dim >> 1;
    const int nq = heads * half;
    const int nk = kv_heads * half;
    const int nv = kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nq + nk) {
        const bool is_q = i < nq;
        const int e = is_q ? i : i - nq;
        const int j = e % half;
        const int h = e / half;
        float inv = powf(theta, -2.0f * (float)j / (float)head_dim);
        if (has_div) {
            inv /= freq_div[j];
        }
        float sn, cs;
        sincosf((float)pos * inv, &sn, &cs);
        if (is_q) {
            const size_t base = (size_t)q_off + (size_t)h * head_dim + j;
            const float a = q[base];
            const float b = q[base + half];
            q[base]        = a * cs - b * sn;
            q[base + half] = b * cs + a * sn;
        } else {
            const size_t base = (size_t)k_off + (size_t)h * head_dim + j;
            const float a = k[base];
            const float b = k[base + half];
            unsigned short* d = kc + ((size_t)h * cap + pos) * head_dim;
            d[j]        = f32_to_f16(a * cs - b * sn);
            d[j + half] = f32_to_f16(b * cs + a * sn);
        }
    } else if (i < nq + nk + nv) {
        const int e = i - nq - nk;
        const int h = e / head_dim;
        const int j = e % head_dim;
        vc[((size_t)h * head_dim + j) * cap + pos] = f32_to_f16(v[(size_t)v_off + e]);
    }
}

// Re-strides a head-major cache into a larger one.
//
// `cap` is a *stride* in both cache layouts, not only a length: keys are
// `[kv_heads, cap, head_dim]` and values are `[kv_heads, head_dim, cap]`, so a
// head's data begins at a multiple of `cap` and doubling the capacity moves
// every head but the first. A flat copy of the live prefix - which is what a
// position-major cache would want - leaves head 0 where it belongs and lands
// the rest inside their own earlier positions.
//
// Nothing catches that downstream. The buffer is the right length, every read
// is in bounds, and attention goes on producing fluent text off one correct
// head. It is the shape of bug this file exists to make impossible, so the
// growth gets a kernel that knows the layout rather than a memcpy that does
// not.
//
// One kernel for both layouts: the caller turns the layout into `rows` runs of
// `len` contiguous floats, a source stride and a destination stride.
extern "C" __global__ void cache_grow(
    const float* __restrict__ src, float* __restrict__ dst,
    int rows, int len, int src_stride, int dst_stride)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * len) {
        return;
    }
    const int r = (int)(i / len);
    const int j = (int)(i % len);
    dst[(size_t)r * dst_stride + j] = src[(size_t)r * src_stride + j];
}

// `cache_grow` for an f16 cache. Same run-and-stride argument, half the width;
// see the note above it for why a memcpy is not what this is.
extern "C" __global__ void cache_grow_f16(
    const unsigned short* __restrict__ src, unsigned short* __restrict__ dst,
    int rows, int len, int src_stride, int dst_stride)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * len) {
        return;
    }
    const int r = (int)(i / len);
    const int j = (int)(i % len);
    dst[(size_t)r * dst_stride + j] = src[(size_t)r * src_stride + j];
}

extern "C" __global__ void repeat_kv(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int heads, int group, int t, int head_dim)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t per_head = (size_t)t * head_dim;
    if (i >= (size_t)heads * per_head) {
        return;
    }
    const int h = (int)(i / per_head);
    const size_t within = i - (size_t)h * per_head;
    dst[i] = src[(size_t)(h / group) * per_head + within];
}

// ------------------------------------------------------------------- spectra

/* A real forward STFT, computed as a direct DFT.
 *
 * `n_fft` is **16** here, so an FFT would be the wrong tool: sixteen points is
 * below the size at which a butterfly beats a straight sum, and a direct
 * transform needs no plan, no twiddle table and no second code path for the
 * odd frame at the end. One thread per (bin, frame).
 *
 * `center` is torch's default and is reproduced: the signal is reflect-padded
 * by `n_fft / 2` on both sides, so frame `f` is centred on sample `f * hop`
 * rather than starting there. Getting this wrong shifts every frame by eight
 * samples, which sounds like a delay and measures like noise. */
extern "C" __global__ void stft_dft(
    const float* __restrict__ x, float* __restrict__ re, float* __restrict__ im,
    const float* __restrict__ window, int n, int n_fft, int hop, int frames)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bins = n_fft / 2 + 1;
    if (i >= bins * frames) return;
    int bin = i / frames, f = i % frames;

    float sr = 0.0f, si = 0.0f;
    int half = n_fft / 2;
    for (int j = 0; j < n_fft; ++j) {
        // Reflect padding, as `torch.stft(pad_mode='reflect')` does it: the
        // edge sample is not repeated, so index -1 is sample 1.
        int at = f * hop + j - half;
        if (at < 0) at = -at;
        if (at >= n) at = 2 * (n - 1) - at;
        // A degenerate signal shorter than the window would fold past both
        // ends; clamping keeps the read in bounds rather than reading nothing.
        if (at < 0) at = 0;
        if (at >= n) at = n - 1;

        float v = x[at] * window[j];
        float ang = -6.283185307179586f * (float)bin * (float)j / (float)n_fft;
        sr += v * __cosf(ang);
        si += v * __sinf(ang);
    }
    re[bin * frames + f] = sr;
    im[bin * frames + f] = si;
}

/* The inverse, as overlap-add with torch's window-envelope normalisation.
 *
 * One thread per output sample, gathering every frame that covers it, rather
 * than one thread per frame scattering with atomics. The gather is
 * deterministic; the scatter's summation order is whatever the scheduler
 * chose that run, and a vocoder whose output changes between runs cannot be
 * diffed against anything.
 *
 * The envelope is the sum of the squared window over the frames covering each
 * sample, which is what `torch.istft` divides by. It is constant in the middle
 * for a Hann window at this hop and is *not* constant at the edges, so it is
 * computed rather than assumed. */
extern "C" __global__ void istft_ola(
    const float* __restrict__ re, const float* __restrict__ im,
    float* __restrict__ out, const float* __restrict__ window,
    int out_n, int n_fft, int hop, int frames)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= out_n) return;

    int half = n_fft / 2;
    int bins = n_fft / 2 + 1;
    // `center` again: output sample `s` sits at `s + half` in the padded
    // signal that the frames tile.
    int p = s + half;

    float acc = 0.0f, env = 0.0f;
    int first = (p - n_fft + 1 + hop - 1) / hop;
    if (first < 0) first = 0;
    for (int f = first; f <= p / hop && f < frames; ++f) {
        int j = p - f * hop;
        if (j < 0 || j >= n_fft) continue;

        // The inverse DFT of a half-spectrum: bin 0 and, for even `n_fft`, the
        // Nyquist bin appear once; every other bin stands for a conjugate pair
        // and so counts twice. Treating all bins alike doubles the DC term and
        // is a constant offset on the waveform.
        float v = 0.0f;
        for (int b = 0; b < bins; ++b) {
            float ang = 6.283185307179586f * (float)b * (float)j / (float)n_fft;
            float c = __cosf(ang), sn = __sinf(ang);
            float term = re[b * frames + f] * c - im[b * frames + f] * sn;
            bool edge = (b == 0) || (n_fft % 2 == 0 && b == bins - 1);
            v += edge ? term : 2.0f * term;
        }
        v /= (float)n_fft;

        float w = window[j];
        acc += v * w;
        env += w * w;
    }
    out[s] = env > 1e-11f ? acc / env : 0.0f;
}

/* Nearest-neighbour upsampling along time: `out[c][t*u + j] = x[c][t]`.
 *
 * The other half of what makes HiFT's `ups` not a transposed convolution.
 * Upstream upsamples by repetition and *then* convolves; a transposed
 * convolution with the same weight interleaves zeros instead and learns a
 * different function. Both give the same output length, which is why this is
 * worth its own kernel rather than a comment. */
extern "C" __global__ void upsample_nearest(
    const float* __restrict__ x, float* __restrict__ out,
    int ch, int t, int factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_t = t * factor;
    if (i >= ch * out_t) return;
    int c = i / out_t, j = i % out_t;
    out[i] = x[c * t + j / factor];
}

/* A strided 1-D convolution with left padding only.
 *
 * `conv1d` asserts stride one, and the excitation branch needs a stride: the
 * source spectrum is decimated by fifteen and then by three to reach each
 * stage's rate. One thread per output element, which is what the shapes here
 * want - the widest is 256 channels over a few thousand frames. */
extern "C" __global__ void strided_conv1d(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int stride, int pad_left, int out_t,
    int has_bias)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_ch * out_t) return;
    int o = i / out_t, s = i % out_t;

    float acc = has_bias ? bias[o] : 0.0f;
    for (int c = 0; c < in_ch; ++c) {
        const float* wr = w + ((size_t)o * in_ch + c) * k;
        const float* xr = x + (size_t)c * t;
        for (int j = 0; j < k; ++j) {
            int at = s * stride + j - pad_left;
            if (at >= 0 && at < t) acc += wr[j] * xr[at];
        }
    }
    out[i] = acc;
}

/* GPT-J style partial rotary embedding, over the *interleaved* pairs.
 *
 * Two things make this not the `rope` next door, and both are easy to get
 * wrong in a way that still produces speech:
 *
 * - **Partial.** Only the first `rot_dim` of each `dim`-wide row is rotated
 *   and the rest passes through. In this model `dim` is 1024 and `rot_dim` is
 *   64, and the rotation happens *before* the heads are split - so of sixteen
 *   heads exactly one carries position information.
 * - **Interleaved.** Element `2j` pairs with `2j + 1`, which is ggml's
 *   convention rather than the one `rope` implements for HuggingFace layouts.
 *
 * The frequencies come from the checkpoint rather than from a base, because
 * `x_transformers` stores `inv_freq` as a buffer. */
extern "C" __global__ void rope_gptj(
    float* x, const float* __restrict__ inv_freq,
    int positions, int dim, int rot_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = rot_dim / 2;
    if (i >= positions * pairs) return;
    int pos = i / pairs, j = i % pairs;

    float ang = (float)pos * inv_freq[j];
    float c = __cosf(ang), s = __sinf(ang);
    size_t at = (size_t)pos * dim + 2 * j;
    float a = x[at], b = x[at + 1];
    x[at]     = a * c - b * s;
    x[at + 1] = b * c + a * s;
}

/* A grouped 1-D convolution with left padding only.
 *
 * The DiT's positional embedding is `groups = 16` over 1024 channels, so each
 * group sees 64 in-channels and produces 64 out. Running it as sixteen
 * ungrouped convolutions would mean sixteen slices per call and forty
 * launches per solver step; running it as one ungrouped convolution would be
 * a different, much larger, function. */
extern "C" __global__ void grouped_conv1d(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ bias, float* __restrict__ out,
    int in_ch, int t, int out_ch, int k, int groups, int pad_left, int out_t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_ch * out_t) return;
    int o = i / out_t, s = i % out_t;

    int in_per = in_ch / groups, out_per = out_ch / groups;
    int g = o / out_per;

    float acc = bias[o];
    for (int c = 0; c < in_per; ++c) {
        // The weight is `[out_ch, in_ch / groups, k]`: the second axis is the
        // channel *within* the group, not the absolute channel.
        const float* wr = w + ((size_t)o * in_per + c) * k;
        const float* xr = x + (size_t)(g * in_per + c) * t;
        for (int j = 0; j < k; ++j) {
            int at = s + j - pad_left;
            if (at >= 0 && at < t) acc += wr[j] * xr[at];
        }
    }
    out[i] = acc;
}

// PyTorch lays an LSTM's four gates along one `4H` axis in the order input,
// forget, cell, output, and splits the pre-activation into an input-side and a
// hidden-side half so the input side can be computed for every step at once.
// Both halves arrive here rather than being summed first: one launch instead of
// two, and the sum is a register add either way.
extern "C" __global__ void lstm_gates(
    const float* __restrict__ gi, const float* __restrict__ gh,
    float* __restrict__ c, float* __restrict__ h, int hidden)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden) return;
    float ig = 1.0f / (1.0f + __expf(-(gi[j] + gh[j])));
    float fg = 1.0f / (1.0f + __expf(-(gi[hidden + j] + gh[hidden + j])));
    float gg = tanhf(gi[2 * hidden + j] + gh[2 * hidden + j]);
    float og = 1.0f / (1.0f + __expf(-(gi[3 * hidden + j] + gh[3 * hidden + j])));
    float cn = fg * c[j] + ig * gg;
    c[j] = cn;
    h[j] = og * tanhf(cn);
}

// WaveGlow's affine coupling run backwards: `x = (x - b) / exp(s)`, where the
// coupling network emits `b` as the first half of `st` and `s` as the second.
//
// A division by `expf` rather than a multiply by `__expf(-s)`, which is the
// same arithmetic and not the same numbers: twelve flows compose this, so the
// cheaper intrinsic's error compounds where the reference's does not.
extern "C" __global__ void coupling_inverse(
    const float* __restrict__ x, const float* __restrict__ st,
    float* __restrict__ out, int half, int t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half * t) return;
    int c = i / t;
    int p = i - c * t;
    float b = st[(size_t)c * t + p];
    float s = st[(size_t)(half + c) * t + p];
    out[i] = (x[i] - b) / expf(s);
}

// ------------------------------------------------------- decode attention
//
// One query position against a head-major cache, in one launch.
//
// A decode step used to reach attention as three kernels a layer - a score
// mat-vec, a softmax and a value mat-vec - with the score row written to
// global memory between them and re-read twice. At one query the score row is
// tiny and the launches are what cost: three launches, three tails, and a
// kernel whose whole job is fifty microseconds of exponentials. This kernel
// does the chain in one pass and never materialises the scores at all.
//
// Shape. `q` is `[heads, HD]`, the query heads grouped so that query head
// `h * group + g` reads key-value head `h` - which is how `split_heads` lays a
// grouped-query model out, and how a multi-head one arrives with `group` 1.
// The caches are the layouts the append kernels write: keys `[kv_heads, cap,
// HD]` and values `[kv_heads, HD, cap]`, f16 or f32 by instantiation.
//
// Grid `(chunks, kv_heads)`, a block per `AD_CH` keys of one head, `HD`
// threads a block. Nothing is staged: a lane reads 16 bytes of a key row and
// the lanes that share a key reduce their partial dot products with shuffles,
// and a thread reads one value row's run of positions straight from the
// transposed cache, whose rows are contiguous along exactly that axis. Every
// load a thread makes is issued before anything waits on one - eight key
// loads, then eight value loads before the softmax they do not depend on.
//
// That is the second shape this kernel had. The first staged the keys and
// then the values through a padded shared tile, and measured *slower* than
// the three launches it replaced at short contexts and barely ahead at long
// ones: a block was a chain of a dozen dependent round trips - a row of
// loads, a barrier, a phase, a barrier, another row of loads - and with a
// few dozen blocks in flight nothing hid the chain. What a single-query
// kernel is short of is not bandwidth or launches but the length of its
// critical path, and this shape has three round trips on it.
//
// Chunks are combined by the last block to finish, which is the standard
// fenced-counter pattern: each block writes its running maximum, its sum and
// its unnormalised context to `part`, bumps the head's counter, and the block
// that sees the count reach `chunks - 1` merges them with the usual
// `exp(m_c - m)` rescaling and resets the counter for the next call. So the
// counters must start at zero and are left at zero - `Gpu::attn_decode` owns
// that. At one chunk the block writes the output directly.
//
// Arithmetic. Scores are f32 sums of f16-or-f32 cache elements against f32
// queries, the softmax is `__expf` as in `softmax_causal`, and the context is
// an f32 sum - the same operations the three-kernel chain performs, in a
// different association. `scale_q` says where the scale goes: on the query
// before the product, as Whisper does and as the `scale_inplace` launch this
// replaces did, or on the scores after it, as Llama does. Algebraically
// identical, not the same rounding, and each model keeps its own.

// A global load that bypasses L1 - for a value another SM wrote.
__device__ __forceinline__ float ld_cg(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float4 ld_cg4(const float* p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

// One output element of the context, and its int8 code when a twin was asked
// for. The `HD` threads of a block own `HD` consecutive elements of one head,
// so a warp is exactly one scale group of 32 and the group maximum is five
// shuffles - `quantize_q8`'s arithmetic on the row this block just produced,
// which saves the launch that would re-read it.
__device__ __forceinline__ void ad_emit(
    float* __restrict__ out, signed char* __restrict__ qa, float* __restrict__ asc,
    size_t idx, float y)
{
    out[idx] = y;
    if (qa) {
        float mx = fabsf(y);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        }
        const float d = mx * (1.0f / 127.0f);
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        qa[idx] = (signed char)__float2int_rn(y * inv);
        if ((threadIdx.x & 31) == 0) {
            asc[idx >> 5] = d;
        }
    }
}

template <int HD, bool KVH, int CH>
__device__ __forceinline__ void attn_decode_impl(
    const float* __restrict__ q,
    const unsigned* __restrict__ kc,
    const unsigned* __restrict__ vc,
    float* __restrict__ out,
    float* __restrict__ part,
    unsigned* __restrict__ ctr,
    int tk, int group, int cap, float scale, int scale_q, int chunks,
    signed char* __restrict__ qa, int asc_off)
{
    float* asc = qa ? (float*)(qa + asc_off) : (float*)0;
    constexpr int T    = HD;                    // threads a block
    constexpr int W    = KVH ? HD / 2 : HD;     // words in a key row
    constexpr int LPK  = W / 4;                 // lanes a key, 16 bytes each
    constexpr int KPW  = 32 / LPK;              // keys a warp a trip
    constexpr int WARPS = T / 32;
    constexpr int KPWARP = CH / WARPS;       // keys a warp covers
    constexpr int KTRIPS = KPWARP / KPW;
    constexpr int EPL  = KVH ? 8 : 4;           // elements a lane a trip
    constexpr int VW   = KVH ? CH / 2 : CH; // words in a value row's chunk
    static_assert(LPK * KPW == 32, "a key is a whole number of lanes and a warp of keys");
    static_assert(KPWARP % KPW == 0, "a warp's keys are whole trips");
    static_assert(W % 4 == 0, "a key row is whole 16-byte loads");

    __shared__ float qs[AD_GMAX * HD];
    __shared__ float sc[AD_GMAX * CH];
    __shared__ float m_s[AD_GMAX], l_s[AD_GMAX];
    __shared__ int last;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int h = blockIdx.y;
    const int c = blockIdx.x;
    const int kv0 = c * CH;
    const int nk = min(CH, tk - kv0);
    const float ninf = __int_as_float(0xff800000);

    // Every key load a lane will make, issued before anything waits on one.
    // A lane owns 16 bytes of `KPW` keys a trip, `KTRIPS` trips - eight
    // independent loads in flight rather than a chain of round trips, which
    // is what the first version of this kernel paid for staging the chunk
    // through shared memory a row at a time.
    const int ks = lane / LPK, wo = (lane - ks * LPK) * 4;
    const unsigned* kb = kc + ((size_t)h * cap + kv0) * W;
    uint4 kr[KTRIPS];
    #pragma unroll
    for (int t = 0; t < KTRIPS; ++t) {
        const int r = warp * KPWARP + t * KPW + ks;
        kr[t] = make_uint4(0u, 0u, 0u, 0u);
        if (r < nk) {
            kr[t] = *reinterpret_cast<const uint4*>(kb + (size_t)r * W + wo);
        }
    }

    // The group's queries, scaled here when the scale belongs on the query.
    const float qf = scale_q ? scale : 1.0f;
    for (int i = tid; i < group * HD; i += T) {
        qs[i] = q[(size_t)h * group * HD + i] * qf;
    }
    __syncthreads();

    // The lane's slice of each query, in registers: the elements its 16
    // bytes of key cover.
    float qr[AD_GMAX][EPL];
    #pragma unroll
    for (int g = 0; g < AD_GMAX; ++g) {
        #pragma unroll
        for (int e = 0; e < EPL; ++e) {
            qr[g][e] = (g < group) ? qs[g * HD + (KVH ? 2 * wo : wo) + e] : 0.0f;
        }
    }

    // Scores: the lane's partial dot products, then a reduction across the
    // LPK lanes that share a key. The lane with the key's first word writes
    // the score; a key past `nk` gets minus infinity so the softmax drops it.
    const float sf = scale_q ? 1.0f : scale;
    #pragma unroll
    for (int t = 0; t < KTRIPS; ++t) {
        float acc[AD_GMAX];
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            acc[g] = 0.0f;
        }
        const unsigned wv[4] = {kr[t].x, kr[t].y, kr[t].z, kr[t].w};
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (KVH) {
                float lo, hi;
                gemm_unpack(wv[w], lo, hi);
                #pragma unroll
                for (int g = 0; g < AD_GMAX; ++g) {
                    acc[g] += qr[g][2 * w] * lo + qr[g][2 * w + 1] * hi;
                }
            } else {
                const float v = __uint_as_float(wv[w]);
                #pragma unroll
                for (int g = 0; g < AD_GMAX; ++g) {
                    acc[g] += qr[g][w] * v;
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            #pragma unroll
            for (int o = LPK / 2; o > 0; o >>= 1) {
                acc[g] += __shfl_xor_sync(0xffffffff, acc[g], o);
            }
        }
        if (lane == ks * LPK) {
            const int r = warp * KPWARP + t * KPW + ks;
            #pragma unroll
            for (int g = 0; g < AD_GMAX; ++g) {
                if (g < group) {
                    sc[g * CH + r] = (r < nk) ? acc[g] * sf : ninf;
                }
            }
        }
    }

    // The value loads go out now, before the softmax, because they do not
    // depend on it: thread `d` owns value row `d` and reads its chunk of
    // positions as a run of 16-byte loads - 8 bytes when the capacity keeps
    // rows only so aligned, which the ASR's 1500 encoder positions do. A
    // vector that would run past the live positions is taken a word at a
    // time, so the tail of a chunk never reads past `tk`; the words past it
    // are zero and so are their probabilities.
    const unsigned* vb = vc + ((size_t)h * HD * cap + kv0) / (KVH ? 2 : 1);
    const int nw = KVH ? (nk + 1) / 2 : nk;
    const int rs = KVH ? cap / 2 : cap;
    const bool v16 = ((cap * (KVH ? 2 : 4)) & 15) == 0;
    unsigned vr[VW];
    {
        const unsigned* src = vb + (size_t)tid * rs;
        #pragma unroll
        for (int w0 = 0; w0 < VW; w0 += 4) {
            vr[w0] = vr[w0 + 1] = vr[w0 + 2] = vr[w0 + 3] = 0u;
            if (w0 >= nw) {
                continue;
            }
            if (w0 + 4 <= nw && v16) {
                const uint4 x = *reinterpret_cast<const uint4*>(src + w0);
                vr[w0] = x.x; vr[w0 + 1] = x.y; vr[w0 + 2] = x.z; vr[w0 + 3] = x.w;
            } else if (w0 + 4 <= nw) {
                const uint2 x0 = *reinterpret_cast<const uint2*>(src + w0);
                const uint2 x1 = *reinterpret_cast<const uint2*>(src + w0 + 2);
                vr[w0] = x0.x; vr[w0 + 1] = x0.y; vr[w0 + 2] = x1.x; vr[w0 + 3] = x1.y;
            } else {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (w0 + e < nw) {
                        vr[w0 + e] = src[w0 + e];
                    }
                }
            }
        }
    }
    __syncthreads();

    // The softmax over the chunk, by warp 0: `SPL` scores a lane, 32 apart.
    constexpr int SPL = CH / 32;
    static_assert(SPL >= 1 && CH % 32 == 0, "a chunk is whole warps of scores");
    if (tid < 32) {
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            if (g < group) {
                float* row = sc + g * CH;
                float v[SPL], p[SPL];
                float m = row[lane];
                v[0] = m;
                #pragma unroll
                for (int e = 1; e < SPL; ++e) {
                    v[e] = row[lane + 32 * e];
                    m = fmaxf(m, v[e]);
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
                }
                float l = 0.0f;
                #pragma unroll
                for (int e = 0; e < SPL; ++e) {
                    p[e] = (lane + 32 * e < nk) ? __expf(v[e] - m) : 0.0f;
                    l += p[e];
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    l += __shfl_xor_sync(0xffffffff, l, o);
                }
                #pragma unroll
                for (int e = 0; e < SPL; ++e) {
                    row[lane + 32 * e] = p[e];
                }
                if (lane == 0) {
                    m_s[g] = m;
                    l_s[g] = l;
                }
            }
        }
    }
    __syncthreads();

    // The context: one thread an output element, the probabilities read as a
    // broadcast and the value row already in registers.
    float o[AD_GMAX];
    #pragma unroll
    for (int g = 0; g < AD_GMAX; ++g) {
        o[g] = 0.0f;
    }
    #pragma unroll
    for (int w = 0; w < VW; ++w) {
        if (KVH) {
            float lo, hi;
            gemm_unpack(vr[w], lo, hi);
            #pragma unroll
            for (int g = 0; g < AD_GMAX; ++g) {
                if (g < group) {
                    o[g] += sc[g * CH + 2 * w] * lo + sc[g * CH + 2 * w + 1] * hi;
                }
            }
        } else {
            const float v = __uint_as_float(vr[w]);
            #pragma unroll
            for (int g = 0; g < AD_GMAX; ++g) {
                if (g < group) {
                    o[g] += sc[g * CH + w] * v;
                }
            }
        }
    }

    if (chunks == 1) {
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            if (g < group) {
                ad_emit(out, qa, asc, ((size_t)h * group + g) * HD + tid, o[g] / l_s[g]);
            }
        }
        return;
    }

    // Several chunks: publish this one's partial and let the last block merge.
    float* mine = part + ((size_t)(h * chunks + c) * AD_GMAX) * (HD + 2);
    #pragma unroll
    for (int g = 0; g < AD_GMAX; ++g) {
        if (g < group) {
            mine[g * (HD + 2) + tid] = o[g];
            if (tid == 0) {
                mine[g * (HD + 2) + HD] = m_s[g];
                mine[g * (HD + 2) + HD + 1] = l_s[g];
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned prev = atomicAdd(ctr + h, 1u);
        last = (prev == (unsigned)(chunks - 1));
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    // The merge, for every group at once and with as few round trips as it
    // can be given: one pass loading every chunk's maximum and sum, a warp a
    // group reducing them out of shared memory, one pass of independent
    // loads for the context. The first version of this was one thread's
    // serial loop of volatile loads, and the second reduced each group in
    // turn through block-wide barriers - twelve dependent round trips for a
    // grouped-query head, which was most of what the kernel cost at a short
    // context. `.cg` loads because the partials were written by other SMs
    // and must not be served from this one's L1. Past `AD_CMAX` chunks the
    // shared arrays are too small and the serial form takes over; that is a
    // context of sixteen thousand and more, and correct rather than fast.
    const float* all = part + ((size_t)h * chunks * AD_GMAX) * (HD + 2);
    if (chunks <= AD_CMAX) {
        __shared__ float mf[AD_GMAX * AD_CMAX];
        __shared__ float ls[AD_GMAX * AD_CMAX];
        __shared__ float L_s[AD_GMAX];
        for (int i = tid; i < group * chunks; i += T) {
            const int g = i / chunks, cc = i - g * chunks;
            const float* pc = all + ((size_t)cc * AD_GMAX + g) * (HD + 2);
            mf[g * AD_CMAX + cc] = ld_cg(pc + HD);
            ls[g * AD_CMAX + cc] = ld_cg(pc + HD + 1);
        }
        __syncthreads();
        for (int g = warp; g < group; g += WARPS) {
            float m = ninf;
            for (int cc = lane; cc < chunks; cc += 32) {
                m = fmaxf(m, mf[g * AD_CMAX + cc]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
            }
            float l = 0.0f;
            for (int cc = lane; cc < chunks; cc += 32) {
                const float f = __expf(mf[g * AD_CMAX + cc] - m);
                mf[g * AD_CMAX + cc] = f;
                l += ls[g * AD_CMAX + cc] * f;
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                l += __shfl_xor_sync(0xffffffff, l, o);
            }
            if (lane == 0) {
                L_s[g] = l;
            }
        }
        __syncthreads();
        float acc[AD_GMAX];
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            acc[g] = 0.0f;
        }
        #pragma unroll 4
        for (int cc = 0; cc < chunks; ++cc) {
            const float* pc = all + (size_t)cc * AD_GMAX * (HD + 2) + tid;
            #pragma unroll
            for (int g = 0; g < AD_GMAX; ++g) {
                if (g < group) {
                    acc[g] += ld_cg(pc + g * (HD + 2)) * mf[g * AD_CMAX + cc];
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            if (g < group) {
                ad_emit(out, qa, asc, ((size_t)h * group + g) * HD + tid, acc[g] / L_s[g]);
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < AD_GMAX; ++g) {
            if (g < group) {
                float m = ninf;
                for (int cc = 0; cc < chunks; ++cc) {
                    m = fmaxf(m, ld_cg(all + ((size_t)cc * AD_GMAX + g) * (HD + 2) + HD));
                }
                float l = 0.0f, acc = 0.0f;
                for (int cc = 0; cc < chunks; ++cc) {
                    const float* pc = all + ((size_t)cc * AD_GMAX + g) * (HD + 2);
                    const float f = __expf(ld_cg(pc + HD) - m);
                    l += ld_cg(pc + HD + 1) * f;
                    acc += ld_cg(pc + tid) * f;
                }
                ad_emit(out, qa, asc, ((size_t)h * group + g) * HD + tid, acc / l);
            }
        }
    }
    if (tid == 0) {
        ctr[h] = 0u;
    }
}

// The chunk width is the one knob measured to matter across contexts, and no
// single value wins: 32 keys is ahead at short contexts and for a
// multi-head model, where the merge over many chunks is cheap against the
// work each does; 64 is ahead in the middle for a grouped-query model; 128
// is ahead at 2048 positions, where the merge of thirty-three partials had
// become the critical path. `docs/BENCHMARKS.md` has the sweep. Each width
// is an instantiation, and `Gpu::attn_decode` picks by the context.
#define AD_ENTRY(NAME, HD, KVH, CH, KT, LB)                                    \
extern "C" __global__ __launch_bounds__(HD, LB) void NAME(                     \
    const float* __restrict__ q,                                               \
    const KT* __restrict__ kc,                                                 \
    const KT* __restrict__ vc,                                                 \
    float* __restrict__ out, float* __restrict__ part, unsigned* __restrict__ ctr, \
    int tk, int group, int cap, float scale, int scale_q, int chunks,         \
    signed char* __restrict__ qa, int asc_off)                                 \
{                                                                              \
    attn_decode_impl<HD, KVH, CH>(q, (const unsigned*)kc, (const unsigned*)vc, \
                                  out, part, ctr, tk, group, cap, scale,       \
                                  scale_q, chunks, qa, asc_off);               \
}

AD_ENTRY(attn_decode_h128_c32,  128, true,  32,  unsigned short, 4)
AD_ENTRY(attn_decode_h128,      128, true,  64,  unsigned short, 4)
AD_ENTRY(attn_decode_h128_c128, 128, true,  128, unsigned short, 4)
AD_ENTRY(attn_decode_h64,       64,  true,  64,  unsigned short, 8)
AD_ENTRY(attn_decode_f64,       64,  false, 64,  float,          8)


// ------------------------------------------------- the mat-vec with a norm
//
// The mat-vec that closes a sub-layer, with the residual add and the next
// normalisation in its tail.
//
// At one row the two projections that close a block - the attention output
// and the MLP down - are each followed by a normalisation that reads the row
// they wrote plus the residual stream, and writes the row the next projections
// read plus its int8 twin: two more launches a layer, each under four
// microseconds and each mostly its own floor. This kernel is `gemv`'s packed
// K-quant path with that tail in it. Every block adds its columns into `h`
// and publishes the sum of their squares; the last block to finish sums the
// partials in a fixed order - so the reduction is the same run to run, which
// an atomic float sum would not be - and then normalises the whole row.
//
// The sum is not associated the way `rms_norm` associates it, so the scale
// can differ from that kernel's by an ulp; both are held to the CPU twin at
// the same tolerance. `h` is read back by the last block through the L2 -
// `ld_cg` - because other blocks wrote it, and each block fences its writes
// before its arrival is counted. The column product is `gemv`'s, character
// for character, and lands in `h` bit for bit as `gemv` then `add` would.
extern "C" __global__ __launch_bounds__(GEMV_WARPS * 32) void gemv_norm(
    const unsigned char* __restrict__ w, int w_quant, int q_ts,
    const signed char* __restrict__ qa, int asc_off,
    int k, int n,
    float* h,
    const float* __restrict__ weight, float eps,
    float* __restrict__ x,
    signed char* __restrict__ xq, int xasc_off,
    float* __restrict__ part, unsigned* __restrict__ ctr)
{
    constexpr int T = GEMV_WARPS * 32;
    __shared__ float red[GEMV_WARPS];
    __shared__ float tree[T];
    __shared__ int last;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int col = blockIdx.x * GEMV_WARPS + warp;
    const int nb = k >> 8;
    const float* asc = (const float*)(qa + asc_off);

    // The residual element this warp will add to, fetched before the
    // contraction so its round trip hides under the weight stream rather
    // than following the reduction.
    const float h0 = (lane == 0 && col < n) ? h[col] : 0.0f;
    float acc = 0.0f;
    if (col < n) {
        const unsigned char* wc = w + (size_t)col * nb * (size_t)q_ts;
        const int sub = lane >> 3, slot = lane & 7;
        if (w_quant == QT_Q4_K) {
            const int jlo = (slot >> 1) * 64 + (slot & 1) * 16;
            const int q0 = jlo >> 5;
            for (int b = 0; b < nb; b += 4) {
                if (b + sub < nb) {
                    acc += q4k_wide(wc + (size_t)(b + sub) * (size_t)q_ts,
                                    qa, asc, slot, jlo, q0, ((b + sub) << 8) + jlo);
                }
            }
        } else {
            const int pp = slot >> 1, hh = slot & 1;
            const int qlo = (pp << 5) + (hh << 4);
            const int qho = 128 + (pp << 4) + (hh << 3);
            const int sc_lo = (pp << 2) + hh;
            const int jlo = (pp << 6) + (hh << 4);
            for (int b = 0; b < nb; b += 4) {
                if (b + sub < nb) {
                    acc += q6k_wide(wc + (size_t)(b + sub) * (size_t)q_ts,
                                    qa, asc, qlo, qho, sc_lo,
                                    ((b + sub) << 8) + jlo);
                }
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    // The residual add, and this column's share of the sum of squares.
    float sq = 0.0f;
    if (lane == 0 && col < n) {
        const float s = h0 + acc;
        h[col] = s;
        sq = s * s;
    }
    if (lane == 0) {
        red[warp] = sq;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < GEMV_WARPS; ++i) {
            s += red[i];
        }
        part[blockIdx.x] = s;
        __threadfence();
        last = (atomicAdd(ctr, 1u) == gridDim.x - 1);
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();

    // Thread `t` owns partials `t, t + T, ...` and the tree over the threads
    // is fixed, so the total does not depend on which block arrived last.
    // Every load below is issued before anything waits on one: this block
    // runs alone at the end of the launch, so its critical path is the
    // launch's tail, and a loop of dependent round trips here measured as
    // ten microseconds a launch - more than the kernel it replaced.
    float s = 0.0f;
    {
        constexpr int PL = 4;
        for (int i0 = tid; i0 < (int)gridDim.x; i0 += PL * T) {
            float pv[PL];
            #pragma unroll
            for (int e = 0; e < PL; ++e) {
                const int i = i0 + e * T;
                pv[e] = (i < (int)gridDim.x) ? ld_cg(part + i) : 0.0f;
            }
            #pragma unroll
            for (int e = 0; e < PL; ++e) {
                s += pv[e];
            }
        }
    }
    tree[tid] = s;
    __syncthreads();
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tree[tid] += tree[tid + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(tree[0] / (float)n + eps);

    // The normalised row and its int8 twin, four columns a thread so that a
    // scale group of 32 is eight lanes and three shuffles - `rms_norm`'s
    // mapping - and `NL` groups of four a thread loaded before any is used.
    // `n` is a multiple of 32, so a group never straddles a warp.
    float* xasc = (float*)(xq + xasc_off);
    const int n4 = n >> 2;
    constexpr int NL = 4;
    for (int i0 = tid; i0 < n4; i0 += NL * T) {
        float4 hv[NL], wv[NL];
        #pragma unroll
        for (int e = 0; e < NL; ++e) {
            const int i = i0 + e * T;
            if (i < n4) {
                hv[e] = ld_cg4(h + (i << 2));
                wv[e] = *((const float4*)weight + i);
            }
        }
        #pragma unroll
        for (int e = 0; e < NL; ++e) {
            const int i = i0 + e * T;
            float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (i < n4) {
                y.x = hv[e].x * scale * wv[e].x;
                y.y = hv[e].y * scale * wv[e].y;
                y.z = hv[e].z * scale * wv[e].z;
                y.w = hv[e].w * scale * wv[e].w;
                *((float4*)x + i) = y;
            }
            float mx = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            }
            const float d = mx * (1.0f / 127.0f);
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            if (i < n4) {
                char4 c;
                c.x = (signed char)__float2int_rn(y.x * inv);
                c.y = (signed char)__float2int_rn(y.y * inv);
                c.z = (signed char)__float2int_rn(y.z * inv);
                c.w = (signed char)__float2int_rn(y.w * inv);
                *(char4*)(xq + (i << 2)) = c;
                if ((lane & 7) == 0) {
                    xasc[i >> 3] = d;
                }
            }
        }
    }
    if (tid == 0) {
        *ctr = 0u;
    }
}

// Gathers rows of a block-quantized embedding table, unpacking as it goes.
//
// The one tensor of a packed checkpoint that is not a matmul operand. It was
// widened to f32 at load for want of this kernel, which at the 8 B chat
// model's 128 256-row vocabulary was 2 GB on the card for a table the decode
// reads one row of - see docs/BENCHMARKS.md. The unpacking is `q_elem`'s,
// element by element: a step gathers a handful of rows of a few thousand
// elements, so the per-element header decode the matmuls could not afford
// costs nothing measurable here and buys every format at once.
//
// One block a row; `table` holds `ch / bs` blocks of `ts` bytes a row, in the
// device layout `Gpu::upload_quant` wrote.
extern "C" __global__ void embed_q(
    const unsigned char* __restrict__ table, int ty, int bs, int ts,
    const long long* __restrict__ ids, float* __restrict__ out,
    int t, int ch, float scale)
{
    const int pos = blockIdx.x;
    if (pos >= t) return;
    const unsigned char* row = table + (size_t)ids[pos] * (size_t)(ch / bs) * (size_t)ts;
    for (int c = threadIdx.x; c < ch; c += blockDim.x) {
        out[(size_t)pos * ch + c] = q_elem(ty, row + (size_t)(c / bs) * ts, c % bs) * scale;
    }
}

"#;

/// The integer a `#define` in [`SOURCE`] binds to, read at compile time.
///
/// The launch geometry and the kernel's block tile are the same three numbers,
/// and writing them twice is how they drift: the grid stayed at `div_ceil(128)`
/// through a tile change once, which under-covers the output rather than
/// failing, so the matmul silently returns whatever the buffer already held.
/// Reading them out of the source makes the `#define` the only copy.
///
/// Panics at compile time if the name is not defined, so a rename is a build
/// error rather than a wrong grid.
const fn define(key: &str) -> u32 {
    let (s, k, tag) = (SOURCE.as_bytes(), key.as_bytes(), b"#define ");
    let mut i = 0;
    while i + tag.len() + k.len() < s.len() {
        let mut hit = true;
        let mut j = 0;
        while j < tag.len() {
            if s[i + j] != tag[j] {
                hit = false;
                break;
            }
            j += 1;
        }
        j = 0;
        while hit && j < k.len() {
            if s[i + tag.len() + j] != k[j] {
                hit = false;
                break;
            }
            j += 1;
        }
        // The name has to end here, or `GEMM_KC` would match `GEMM_KSTEPS`.
        let mut p = i + tag.len() + k.len();
        if hit && (s[p] == b' ' || s[p] == b'\t') {
            while s[p] == b' ' || s[p] == b'\t' {
                p += 1;
            }
            let mut v = 0;
            let mut any = false;
            while p < s.len() && s[p] >= b'0' && s[p] <= b'9' {
                v = v * 10 + (s[p] - b'0') as u32;
                p += 1;
                any = true;
            }
            if any {
                return v;
            }
        }
        i += 1;
    }
    panic!("no such #define in the CUDA source")
}

/// Warps per `gemm` block; the launch's `block_dim.y`.
pub const GEMM_WARPS: u32 = define("GEMM_WARPS");
/// Warps in a `gemv` or `gemv_rows` block; one output column each.
pub const GEMV_WARPS: u32 = define("GEMV_WARPS");
/// Rows of the activation one `gemm` block covers; the grid's `y` step.
pub const GEMM_MT: u32 = define("GEMM_MT");
/// Rows of the weight one `gemm` block covers; the grid's `x` step.
pub const GEMM_NT: u32 = define("GEMM_NT");

/// Warps per `gemm_i8` block.
pub const GEMM_I8_WARPS: u32 = define("GEMM_I8_WARPS");
/// Rows of the activation one `gemm_i8` block covers.
pub const GEMM_I8_MT: u32 = define("GEMM_I8_MT");

/// The narrow row tile, for a prefill with fewer rows than the wide one
/// computes. See the note beside `GEMM_I8_ENTRY`.
pub const GEMM_I8_MT_NARROW: u32 = define("GEMM_I8_MT_NARROW");
/// Rows `gemv_rows` will carry in one warp. `GEMV_MAX_M` must not exceed it.
pub const GEMV_ROWS_MAX: u32 = define("GEMV_ROWS_MAX");
/// Rows of the weight one `gemm_i8` block covers.
pub const GEMM_I8_NT: u32 = define("GEMM_I8_NT");
/// Keys one `attn_decode` block covers; the grid's `x` step.
pub const AD_CH: u32 = define("AD_CH");
/// Query heads a key-value head may serve in `attn_decode`.
pub const AD_GMAX: u32 = define("AD_GMAX");
