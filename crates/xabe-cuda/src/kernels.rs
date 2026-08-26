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

__global__ void sub_inplace(float* a, const float* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] -= b[i];
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
"#;
