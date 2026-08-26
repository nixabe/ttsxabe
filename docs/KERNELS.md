# Kernels

Every entry needs a CPU reference in `xabe-dsp` and a differential test before
it is done. Status here is the truth; a row is not ticked because the code
exists.

## Inventory

| kernel | used by | reference | CUDA | differential |
| --- | --- | --- | --- | --- |
| embedding lookup | text encoder | `xabe-dsp` (inline) | `embed_scaled` | `xabe-tts` text_encoder |
| layer norm | text encoder | `xabe_dsp::layer_norm` | `layer_norm` | `xabe-tts` text_encoder |
| relative-position self-attention | text encoder (window 4) | `xabe_dsp::self_attention` | `attention_scores` + `attention_context` | `xabe-dsp` relative_position + `xabe-tts` |
| conv1d, kernel 3 | text encoder FFN | `xabe_dsp::conv1d` | `conv1d` | `xabe-tts` text_encoder |
| conv1d, general | flow, duration predictor, decoder | `xabe_dsp::conv1d` | `conv1d` | `xabe-tts` text_encoder |
| depthwise-separable conv | duration predictor | `xabe_dsp::depthwise_conv1d` | `depthwise_conv1d` | `xabe-tts` duration |
| transposed conv1d | decoder upsamplers | `xabe_dsp::transposed_conv1d` | `transposed_conv1d` | `xabe-tts` decoder |
| leaky ReLU | decoder | `xabe_dsp::leaky_relu` | `act_leaky_relu` | `xabe-tts` decoder |
| WaveNet residual block | flow coupling, posterior | `xabe-tts` flow::wavenet | `gated_activation` + `conv1d` | `xabe-tts` flow |
| affine coupling | flow | `xabe-tts` flow_reverse | `sub_inplace` | `xabe-tts` flow |
| stochastic duration flow | duration predictor | `xabe_dsp::spline_inverse` | host (69 positions) | `xabe-tts` duration |
| length regulation / attention expansion | prior → frames | `xabe-tts` expand_prior | `expand_prior` | `xabe-tts` prior |
| HiFi-GAN resblock (MRF) | decoder | `xabe-tts` decoder::resblock | `conv1d` + `act_leaky_relu` | `xabe-tts` decoder |
| tanh output | decoder | `xabe-tts` decoder | `act_tanh` | `xabe-tts` decoder |
| strided conv1d | VAD stft + encoder | `xabe_dsp::conv1d_strided` | (cpu only) | `xabe-vad` reference |
| magnitude from re/im halves | VAD stft | `xabe-vad` stft | (cpu only) | `xabe-vad` reference |
| LSTM cell, gates i f g o | VAD decoder | `xabe-vad` lstm | (cpu only) | `xabe-vad` reference |

## Also implemented

Two kernels the original inventory did not name, because reading the reference
is what turned them up:

| kernel | used by | reference | CUDA | differential |
| --- | --- | --- | --- | --- |
| exact GELU (needs `erf`) | duration predictor | `xabe_dsp::gelu` | `act_gelu` | `xabe-dsp` gelu |
| softmax | attention, spline knots | `xabe_dsp::softmax_rows` | `softmax_rows` | via attention |

GELU is the one kernel here that *approximates* the reference rather than
rearranging it: Rust has no `erf`, so `xabe-dsp` carries Cody's rational
approximation. PyTorch's default GELU is the exact erf form, and the tanh
approximation - the obvious substitute - differs by up to 4.7e-4 near
`|x| = 2.7`, an order of magnitude above the tolerances here.

## Notes that will bite

- **Transposed convolutions store `[in, out, kernel]`**, the opposite of the
  ordinary convolutions beside them. `decoder.upsampler.0.weight` is
  `[512, 256, 16]`: 512 in, 256 out.
- **Relative-position attention uses a window of 4**, so the relative embedding
  table is `2 * 4 + 1` wide and the indexing is not the same as absolute RoPE-
  style attention. This is the most commonly mis-ported piece of VITS.
- **The flow is run in reverse at inference**, which means the coupling halves
  and the order of the four blocks both invert. A forward-order implementation
  produces audio.
- **Dilations in the decoder resblocks are per-kernel**: `[[1,3,5], [1,3,5],
  [1,3,5]]` against kernels `[3, 7, 11]`, three convolution pairs each.

## The CUDA column

Every kernel named there lives in `xabe-cuda`'s single NVRTC translation unit
and is tested against its `xabe-dsp` twin in
`crates/xabe-cuda/tests/kernels.rs`, per kernel, before anything is assembled
from it. A GPU pipeline that is wrong somewhere is nearly impossible to bisect
after the fact and trivially bisected before it exists.

Two entries deserve a note:

- **`transposed_conv1d` is not the CPU code ported.** The scalar twin scatters -
  each input contributes to `k` outputs - and the kernel gathers, because a
  scatter needs atomics. They are an inverse pair rather than one algorithm
  written twice, and the inversion is where the off-by-ones live. That is what
  its differential test is really checking.
- **`act_gelu` uses the device's `erff`,** which is IEEE-accurate, while the CPU
  twin carries Cody's rational approximation because Rust has no `erf`. Their
  test compares two different implementations of the same function, so
  agreement is evidence about both.

The stochastic duration flow has no CUDA entry on purpose: it is a rational
quadratic spline evaluated at one channel over a few dozen symbol positions,
four times. Moving it would cost more in launches and transfers than it saves.

## Reference implementations are scalar on purpose

`xabe-dsp` kernels are written to be read against the PyTorch source line by
line. They are not vectorised, not blocked, and not clever. A reference you have
to reason about is not a reference.

## The matmul, added for the ASR

| kernel | used by | CPU twin | notes |
| --- | --- | --- | --- |
| `gemm` | encoder and decoder projections | `xabe_dsp::linear` | tensor cores, `m16n8k8`, f16 operands, f32 accumulate |
| `gemv` | the same, at decode width | `xabe_dsp::linear` | one warp per output channel, exact f32 |

`Gpu::gemm` dispatches between them on `m`, at `GEMV_MAX_M = 16`. The two do
not have the same precision, which is why the constant is public: a test that
compares against a reference has to know which side of it a shape falls on.

### What the tensor-core path is worth

Medians of 20, one Quadro RTX 8000, against the one-thread-per-output `linear`:

| shape | what it is | `gemm` | `linear` | | |
| --- | --- | --- | --- | --- | --- |
| 1500x1280x1280 | encoder q/k/v/o | 0.21 ms | 37.6 ms | 182x | 23.8 TFLOP/s |
| 1500x1280x5120 | encoder mlp up | 0.94 ms | 151.4 ms | 161x | 20.9 TFLOP/s |
| 1500x5120x1280 | encoder mlp down | 0.83 ms | 151.6 ms | 183x | 23.7 TFLOP/s |
| 1x1280x1280 | decode projection | 0.02 ms | 0.23 ms | 12x | — |
| 1x1280x51864 | decode output head | 0.45 ms | 1.83 ms | 4x | 590 GB/s |

The output head is bandwidth-bound and running at 88% of the card's 672 GB/s,
which is about as good as a GEMV gets.

### Why the tile is 128x128

Chosen by arithmetic first and confirmed by measurement. A block reads the whole
contraction for `MT` rows of the activations and `NT` rows of the weights, and
does `2*MT*NT*k` flops with them - so it performs `2*MT*NT/(MT+NT)` flops per
float it reads. At 64x32 that is 43, capping the kernel near 7 TFLOP/s against
672 GB/s. At 128x128 it is 128, capping it near 21, which is where it measures.

The sweep, all at KC=32 unless stated, TFLOP/s for (q/k/v/o, mlp up, mlp down):

| warps | MT | NT | KC | | | |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 128 | 128 | 32 | **23.6** | **21.3** | **24.1** |
| 8 | 128 | 64 | 32 | 16.6 | 17.0 | 16.4 |
| 8 | 64 | 128 | 32 | 14.6 | 17.5 | 15.1 |
| 8 | 128 | 128 | 64 | 18.9 | 17.9 | 19.3 |
| 4 | 64 | 64 | 32 | 21.0 | 14.4 | 20.4 |
| 8 | 256 | 64 | 32 | 20.1 | 18.5 | 18.8 |
| 8 | 64 | 64 | 64 | 19.5 | 14.6 | 18.7 |

### WHY NOT

- **Do not time a GPU kernel by downloading its result.** The first version of
  `bench-gemm` called `download` to force the queue to drain, and for the wider
  shapes the PCIe copy was *most* of the measurement: 1500x5120 floats is 31 MB,
  about 5 ms at 6 GB/s, against a kernel that runs in under one. Every number in
  the first sweep was the bus, which is why the tile appeared not to matter -
  all seven configurations "measured" 2.9 to 3.5 TFLOP/s. `synchronize`.
- **KC=64 is slower than KC=32**, at every tile shape tried. More contraction
  per staging trip is fewer barriers and also a larger shared footprint; the
  second wins here.

### Reachability at sm_75

Only two MMA shapes assemble on this card: `m8n8k16.s32.s8.s8.s32` and
`m16n8k8.f32.f16.f16.f32`. `m16n8k16` is accepted by NVRTC and then rejected by
ptxas - **NVRTC success is not evidence of reachability**. That constraint, the
fragment layouts, and the shared-memory stride argument are adapted from
`llmxabe`, which has been running them on this hardware.

### Why f32 accumulation is not caution

fp16 *operands* are safe; fp16 *accumulation* is not. `llmxabe` records the
measurement: on IID-random data fp16 accumulation looks safe at every depth with
26-30x headroom, and it then broke an adversarial differential test by 209x,
with constant-input error growing monotonically 3.2e-2 at 8K to 7.3e-1 at 131K.
Rescale cadence does not help. So `m16n8k8.f32...f32` is the shape used, and the
operand rounding it does cost is measured rather than assumed: 6.5e-5 of full
scale on a k=1280 contraction.
