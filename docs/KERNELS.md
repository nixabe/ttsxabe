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
