# Kernels

Every entry needs a CPU reference in `xabe-dsp` and a differential test before
it is done. Status here is the truth; a row is not ticked because the code
exists.

## Inventory

| kernel | used by | reference | CUDA | differential |
| --- | --- | --- | --- | --- |
| embedding lookup | text encoder | | | |
| layer norm | text encoder | | | |
| relative-position self-attention | text encoder (window 4) | | | |
| conv1d, kernel 3 | text encoder FFN | | | |
| conv1d, general | flow, duration predictor, decoder | | | |
| depthwise-separable conv | duration predictor | | | |
| transposed conv1d | decoder upsamplers | | | |
| leaky ReLU | decoder | | | |
| WaveNet residual block | flow coupling, posterior | | | |
| affine coupling | flow | | | |
| stochastic duration flow | duration predictor | | | |
| length regulation / attention expansion | prior → frames | | | |
| HiFi-GAN resblock (MRF) | decoder | | | |
| tanh output | decoder | | | |

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
