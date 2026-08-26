# Kernels

Every entry needs a CPU reference in `xabe-dsp` and a differential test before
it is done. Status here is the truth; a row is not ticked because the code
exists.

## Inventory

| kernel | used by | reference | CUDA | differential |
| --- | --- | --- | --- | --- |
| embedding lookup | text encoder | `xabe-dsp` (inline) |  | `xabe-tts` text_encoder |
| layer norm | text encoder | `xabe_dsp::layer_norm` |  | `xabe-tts` text_encoder |
| relative-position self-attention | text encoder (window 4) | `xabe_dsp::self_attention` |  | `xabe-dsp` relative_position + `xabe-tts` |
| conv1d, kernel 3 | text encoder FFN | `xabe_dsp::conv1d` |  | `xabe-tts` text_encoder |
| conv1d, general | flow, duration predictor, decoder | `xabe_dsp::conv1d` |  | `xabe-tts` text_encoder |
| depthwise-separable conv | duration predictor | `xabe_dsp::depthwise_conv1d` |  | `xabe-tts` duration |
| transposed conv1d | decoder upsamplers | `xabe_dsp::transposed_conv1d` |  | `xabe-tts` decoder |
| leaky ReLU | decoder | `xabe_dsp::leaky_relu` |  | `xabe-tts` decoder |
| WaveNet residual block | flow coupling, posterior | `xabe-tts` flow::wavenet |  | `xabe-tts` flow |
| affine coupling | flow | `xabe-tts` flow_reverse |  | `xabe-tts` flow |
| stochastic duration flow | duration predictor | `xabe_dsp::spline_inverse` |  | `xabe-tts` duration |
| length regulation / attention expansion | prior → frames | `xabe-tts` expand_prior |  | `xabe-tts` prior |
| HiFi-GAN resblock (MRF) | decoder | `xabe-tts` decoder::resblock |  | `xabe-tts` decoder |
| tanh output | decoder | `xabe-tts` decoder |  | `xabe-tts` decoder |

## Also implemented

Two kernels the original inventory did not name, because reading the reference
is what turned them up:

| kernel | used by | reference | CUDA | differential |
| --- | --- | --- | --- | --- |
| exact GELU (needs `erf`) | duration predictor | `xabe_dsp::gelu` | | `xabe-dsp` gelu |
| softmax | attention, spline knots | `xabe_dsp::softmax_rows` | | via attention |

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

## Reference implementations are scalar on purpose

`xabe-dsp` kernels are written to be read against the PyTorch source line by
line. They are not vectorised, not blocked, and not clever. A reference you have
to reason about is not a reference.
