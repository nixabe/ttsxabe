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

#define GEMM_WARPS 8
#define GEMM_MT    128     // rows of `a` per block
#define GEMM_NT    128     // rows of `w` per block
#define GEMM_KC    32      // contraction staged per trip
#define GEMM_MSTEPS (GEMM_MT / 16)
#define GEMM_KSTEPS (GEMM_KC / 8)
#define GEMM_NPW   (GEMM_NT / (GEMM_WARPS * 8))   // 8-wide n tiles per warp

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
    // ql(128) + qh(64) + scales(16, signed) + d(2). The low part runs in
    // groups of 64 and the high part in groups of 32.
    case QT_Q6_K: {
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const unsigned char* scales = blk + 192;
        float d = q_f16(blk, 208);
        int g = j / 16;
        int hi = j / 128, r = j % 128;
        int sl = r / 64, k64 = r % 64;
        int sh = r / 32, k32 = r % 32;
        unsigned char lo = (ql[hi * 64 + k64] >> (4 * sl)) & 0x0F;
        unsigned char bits = (qh[hi * 32 + k32] >> (2 * sh)) & 0x03;
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

// Eight elements of a Q6_K super-block, chosen so no packed byte is read twice.
//
// A Q6_K element takes a nibble of `ql` and a 2-bit field of `qh`. The nibbles
// of one `ql` byte are 64 apart and the four fields of one `qh` byte are 32
// apart, so eight *adjacent* elements touch eight `ql` bytes and eight `qh`
// bytes and use an eighth of what they fetch: the warp reads every byte twice
// over for `ql` and four times over for `qh`.
//
// A lane instead owns two adjacent columns across all four fields - elements
// hi * 128 + sh * 32 + b + c for sh in 0..3 and c in 0..1. Those eight need
// three 16-bit reads, two of `ql` and one of `qh`, and across the warp they
// cover ql[0..127] and qh[0..63] exactly once. Sixteen byte loads become three
// short ones. Measured standalone on this card at n=14336, k=4096: 410 us to
// 129 us, agreeing with the layout above to 2.8e-07 relative.
//
// Shorts and not words because a Q6_K block is 210 bytes: the stride is even
// but not a multiple of four, so successive blocks are 2-byte aligned and
// nothing wider is safe.
__device__ __forceinline__ float q6k_dot8(
    const unsigned char* blk, const float* ap, int hi, int bb, int g0)
{
    float d = q_f16(blk, 208);
    const signed char* scales = (const signed char*)(blk + 192);
    unsigned h  = *(const unsigned short*)(blk + 128 + (hi << 5) + bb);
    unsigned l0 = *(const unsigned short*)(blk + (hi << 6) + bb);
    unsigned l1 = *(const unsigned short*)(blk + (hi << 6) + 32 + bb);

    // Scale group j / 16 is hi * 8 + sh * 2 + b / 16, and the last term is the
    // same for both columns because `bb` is even.
    float ds[4];
    #pragma unroll
    for (int sh = 0; sh < 4; ++sh) {
        ds[sh] = d * (float)((int)scales[g0 + (sh << 1)]);
    }

    float acc = 0.0f;
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        unsigned hb = (h >> (c << 3)) & 0xFFu;
        unsigned b0 = (l0 >> (c << 3)) & 0xFFu;
        unsigned b1 = (l1 >> (c << 3)) & 0xFFu;
        // sh 0 and 2 share a `ql` byte, as do sh 1 and 3; the low nibble is the
        // lower field of the pair.
        unsigned nib[4] = { b0 & 0x0Fu, b1 & 0x0Fu, b0 >> 4, b1 >> 4 };
        #pragma unroll
        for (int sh = 0; sh < 4; ++sh) {
            int q = (int)(nib[sh] | (((hb >> (sh << 1)) & 3u) << 4)) - 32;
            acc += ap[(sh << 5) + c] * (ds[sh] * (float)q);
        }
    }
    return acc;
}

// Eight consecutive elements of one K-quant super-block, header decoded once.
//
// `gemm` stages the weight tile two elements per thread and was reaching them
// through `q_at`, which re-derives the block header for each - the same waste
// `q4k_pair` exists to remove from `gemv`, left behind in the tiled kernel
// because only the decode path had been measured. It is most of prefill.
//
// Eight *eight-aligned* elements stay inside one 32-element sub-block and one
// 16-element Q6_K scale group, so the scales, the shift and the byte pointer
// are all shared and the inner eight are a nibble extract and a multiply-add.
// Same addressing as `q6k_dot8` before it was regrouped; `gemm` wants a run of
// the contraction where `gemv` wants a run of whole bytes.
__device__ __forceinline__ void q4k_eight(
    const unsigned char* blk, int j, float* e)
{
    float d = q_f16(blk, 0), dmin = q_f16(blk, 2);
    unsigned char sc, mn;
    q_scale_min(blk + 4, j >> 5, sc, mn);
    const unsigned char* qs = blk + 16 + ((j >> 6) << 5) + (j & 31);
    int shift = ((j >> 5) & 1) << 2;
    float ds = d * (float)sc, dm = dmin * (float)mn;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        e[t] = ds * (float)((qs[t] >> shift) & 0x0F) - dm;
    }
}

__device__ __forceinline__ void q6k_eight(
    const unsigned char* blk, int j, float* e)
{
    const unsigned char* qh = blk + 128;
    const signed char* scales = (const signed char*)(blk + 192);
    float d = q_f16(blk, 208);
    int g = j >> 7, r = j & 127;
    int sl = (r >> 6) & 1, sh = (r >> 5) & 3;
    const unsigned char* qlp = blk + (g << 6) + (r & 63);
    const unsigned char* qhp = qh + (g << 5) + (r & 31);
    float dsc = d * (float)((int)scales[j >> 4]);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int lo = (qlp[t] >> (sl << 2)) & 0x0F;
        int b = (qhp[t] >> (sh << 1)) & 0x03;
        e[t] = dsc * (float)((lo | (b << 4)) - 32);
    }
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
    int w_quant, int q_bs, int q_ts)
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
        if (!a_half && q_bs == 256 && w_quant == QT_Q4_K) {
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
        } else if (!a_half && q_bs == 256 && w_quant == QT_Q6_K) {
            const int nb = k >> 8;
            const int hi = lane >> 4, bb = (lane & 15) << 1;
            const int g0 = (hi << 3) + (bb >> 4);
            const float* av = af + (hi << 7) + bb;
            for (int b = 0; b < nb; ++b) {
                acc += q6k_dot8(wq + ((size_t)col * nb + b) * (size_t)q_ts,
                                av + (b << 8), hi, bb, g0);
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
        const unsigned* wv = (const unsigned*)w + (size_t)blockIdx.z * (sw >> 1)
                           + (size_t)col * kh;
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
    } else {
        const float* wv = (const float*)w + (size_t)blockIdx.z * sw + (size_t)col * k;
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
        out[(size_t)row * n + col] = acc + (bias ? bias[col] : 0.0f);
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
    int w_quant, int q_bs, int q_ts)
{
    __shared__ unsigned as[GEMM_MT * GEMM_WSTRIDE];
    __shared__ unsigned bs[GEMM_NT * GEMM_WSTRIDE];

    // See the note in `gemv`: blockIdx.z selects one product of a batch.
    out += (size_t)blockIdx.z * so;
    const float*    af = (const float*)a    + (size_t)blockIdx.z * sa;
    const unsigned* ah = (const unsigned*)a + (size_t)blockIdx.z * (sa >> 1);
    const float*    wf = (const float*)w    + (size_t)blockIdx.z * sw;
    const unsigned* wh = (const unsigned*)w + (size_t)blockIdx.z * (sw >> 1);
    // Guarded because `q_bs` is zero on the unquantized paths, where this
    // pointer is never read.
    const unsigned char* wq = (const unsigned char*)w
        + (size_t)blockIdx.z * (size_t)(q_bs ? (sw / q_bs) * (long)q_ts : 0);

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

    for (int kc = 0; kc < k; kc += GEMM_KC) {
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
        const bool whole = (kc + GEMM_KC <= k) && ((k & 1) == 0);
        for (int i = tid; i < GEMM_MT * (GEMM_KC / 2); i += GEMM_WARPS * 32) {
            int row = i / (GEMM_KC / 2);
            int j   = i % (GEMM_KC / 2);
            int kk  = kc + 2 * j;
            unsigned packed = 0;
            if (a_half) {
                const int kh = k >> 1;
                const int aj = (kc >> 1) + j;
                if (m0 + row < m && aj < kh) {
                    packed = ah[(size_t)(m0 + row) * kh + aj];
                }
            } else {
                float lo = 0.0f, hi = 0.0f;
                if (m0 + row < m) {
                    const float* src = af + (size_t)(m0 + row) * k + kk;
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
            as[row * GEMM_WSTRIDE + j] = packed;
        }
        // The two K-quants every checkpoint here uses stage eight elements a
        // thread so the header is decoded once for all of them. `GEMM_KC` is a
        // multiple of eight and `kc` a multiple of `GEMM_KC`, so `kk` is
        // eight-aligned and the run cannot straddle a sub-block.
        const bool q_fast = w_quant && q_bs == 256
            && (w_quant == QT_Q4_K || w_quant == QT_Q6_K);
        if (q_fast) {
            for (int i = tid; i < GEMM_NT * (GEMM_KC / 8); i += GEMM_WARPS * 32) {
                int row = i / (GEMM_KC / 8);
                int jq  = i % (GEMM_KC / 8);
                int kk  = kc + 8 * jq;
                float e[8];
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    e[t] = 0.0f;
                }
                if (n0 + row < n) {
                    if (kk + 7 < k) {
                        long nb = k / 256;
                        const unsigned char* blk = wq
                            + ((size_t)(n0 + row) * nb + (kk >> 8)) * (size_t)q_ts;
                        if (w_quant == QT_Q4_K) {
                            q4k_eight(blk, kk & 255, e);
                        } else {
                            q6k_eight(blk, kk & 255, e);
                        }
                    } else {
                        // The tail of a contraction that is not a whole tile.
                        #pragma unroll
                        for (int t = 0; t < 8; ++t) {
                            if (kk + t < k) {
                                e[t] = q_at(wq, w_quant, q_bs, q_ts, n0 + row, k, kk + t);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    bs[row * GEMM_WSTRIDE + 4 * jq + t] =
                        gemm_pack(e[2 * t], e[2 * t + 1]);
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
                    const float* src = wf + (size_t)(n0 + row) * k + kk;
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
            unsigned b0[GEMM_NPW];
            #pragma unroll
            for (int nt = 0; nt < GEMM_NPW; ++nt) {
                b0[nt] = bs[((warp * GEMM_NPW + nt) * 8 + g) * GEMM_WSTRIDE + 4 * ks + tg];
            }
            #pragma unroll
            for (int ms = 0; ms < GEMM_MSTEPS; ++ms) {
                unsigned a0 = as[(16 * ms + g)     * GEMM_WSTRIDE + 4 * ks + tg];
                unsigned a1 = as[(16 * ms + g + 8) * GEMM_WSTRIDE + 4 * ks + tg];
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
    #pragma unroll
    for (int nt = 0; nt < GEMM_NPW; ++nt) {
        const int col0 = n0 + (warp * GEMM_NPW + nt) * 8 + 2 * tg;
        const float bias0 = (bias && col0     < n) ? bias[col0]     : 0.0f;
        const float bias1 = (bias && col0 + 1 < n) ? bias[col0 + 1] : 0.0f;
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
__global__ void layer_norm(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ out,
    int cols, float eps)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * cols;
    float* outr = out + (size_t)row * cols;

    float partial = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) partial += xr[i];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;
    __syncthreads();

    partial = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float d = xr[i] - mean;
        partial += d * d;
    }
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    // The biased variance, matching torch.nn.LayerNorm.
    float inv = rsqrtf(sdata[0] / (float)cols + eps);

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        outr[i] = (xr[i] - mean) * inv * weight[i] + bias[i];
    }
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
extern "C" __global__ void rms_norm(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int dim, float eps)
{
    extern __shared__ float partial[];
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* orow = out + (size_t)row * dim;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        acc += xr[i] * xr[i];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        orow[i] = xr[i] * scale * weight[i];
    }
}

// a = silu(a) * b, the SwiGLU gate.
extern "C" __global__ void silu_mul(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = a[i];
    // `expf`, not `__expf`. The fast intrinsic is about 2^-21 accurate and
    // this one is a few ulp; the difference costs nothing measurable here and
    // it keeps the differential test against the scalar twin tight enough to
    // catch a real mistake.
    a[i] = v * (1.0f / (1.0f + expf(-v))) * b[i];
}

// Rotary position embedding, in place over [t, heads * head_dim].
//
// 🤗 pairs dimension i with i + head_dim/2, not 2i with 2i+1. The two are a
// permutation of each other, both are called RoPE, and picking the wrong one
// gives a model that is coherent for four or five tokens and then drifts -
// the hardest possible thing to debug. `first` is the absolute position of row
// zero, so a decode step past a KV cache rotates by where the token really is.
extern "C" __global__ void rope(
    float* __restrict__ x,
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

    const size_t base = ((size_t)p * heads + h) * head_dim + j;
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
extern "C" __global__ void split_heads_t(
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
    out[((size_t)h * head_dim + j) * t + ti] = x[i];
}

// [heads, t, head_dim] -> [t, heads * head_dim]. The inverse of `split_heads`.
extern "C" __global__ void merge_heads(
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
    out[i] = x[((size_t)h * t + ti) * head_dim + j];
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

"#;
