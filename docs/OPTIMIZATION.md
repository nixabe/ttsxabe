# The performance model

Written before the kernels, so that the first implementation is not shaped by a
wrong intuition. **Nothing here is measured on this implementation yet**; the
numbers are hardware facts and reference-implementation observations.

## 1. The card

Quadro RTX 8000, Turing, sm_75, 48 GB, 672 GB/s.

- fp16 tensor cores: yes
- bf16: **no**
- fp8: **no**
- int8 tensor cores: yes

The absent bf16 matters more than it sounds. It removes the usual "cast
everything to bf16 and stop worrying about range" option, so mixed precision
here means fp16 with real overflow risk, or staying in fp32.

## 2. Where the time goes

The model is 36 M parameters — 145 MB in f32. At 672 GB/s, reading every weight
once costs ~0.2 ms. Synthesis of a short clause takes ~90 ms in PyTorch.

**This is not a bandwidth-bound problem.** Weight traffic is negligible; the
cost is in the number of small kernel launches and in the decoder's arithmetic.
That points the optimisation work at fusion and launch overhead, not at
quantisation or clever memory layout.

Quantising a 145 MB model to save bandwidth would be solving a problem this
workload does not have, while adding a numerics risk it cannot afford.

## 3. Where the arithmetic is

The HiFi-GAN decoder is 39.5% of the parameters and produces 256 output samples
per input frame through four transposed convolutions, each followed by three
multi-receptive-field resblocks with dilations `[1,3,5]`. That is where the FLOPs
are, and it is the first place to look.

The text encoder runs over symbols, not frames — tens of positions, not
thousands. It will not be the bottleneck at any realistic utterance length, and
optimising it first would be optimising the wrong end.

## 4. What transfers from `llmxabe`, and what does not

**Transfers:** the NVRTC-at-runtime approach, the arena discipline (no
allocation on the hot path), the differential-test-per-kernel standard, and the
rule that a measurement lands with its optimisation.

**Does not transfer:** anything about KV caches, prefix reuse, speculative
decoding, or request scheduling. This is a feed-forward network with no
autoregressive loop over a cache. The one autoregressive-shaped thing is the
duration predictor's flow, which is four small steps over symbols.

Also does not transfer: the assumption that fp16 helps. It was measured
*slower* on these cards for the neighbouring TTS model in the same pipeline —
see WHY NOT in [BENCHMARKS.md](BENCHMARKS.md).

## 5. The ceiling

If the whole synthesis reduces to arithmetic on the decoder and every launch is
fused perfectly, the floor is set by the transposed convolutions' FLOP count at
sm_75 fp32 rates. That bound has not been computed yet, and until it is, "how
much faster can this get" has no honest answer.

Compute it before promising a factor.
