# The performance model

Written before the kernels, so that the first implementation is not shaped by a
wrong intuition. **Nothing in sections 1 to 5 was measured on this
implementation when it was written**; the numbers were hardware facts and
reference-implementation observations. Section 6 was added after the
synthesiser was measured and section 7 after the engine grew five more models.

**Everything up to section 6 is about the synthesiser**, which was the whole
project when this was written. Section 7 is there because the scope widened to
an ASR, two Llama stages and two more synthesisers, and three of this file's
conclusions invert on those.

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

Now computed. The decoder is **6.15e8 FLOPs per input frame**, or 2.4e6 per
output sample - dominated by the residual blocks, not by the transposed
convolutions, which are under 1% of it. For a 163-frame utterance that is
100.2 GFLOP.

At the card's 16.3 TFLOPS fp32 peak that is a 6.1 ms floor. The current kernels
reach 2.94 TFLOPS, 18% of peak. A well-tuned dense kernel reaching 40-60% would
land at 10-15 ms, so the honest remaining headroom is **2-3x, all of it in the
convolution kernel**.

## 6. What was actually true

Written after the measurements, against section 2's predictions.

**Right:** the decoder is where the time is (72%), and the text encoder is not
the bottleneck (13% even at 69 symbols). Bandwidth is not the constraint.

**Wrong, or at least misleading:** section 2 said the cost was "in the number of
small kernel launches and in the decoder's arithmetic". Launch overhead turned
out to be negligible - synthesis time is linear in output length with an
intercept indistinguishable from zero, across a 15x range of utterance
lengths - so allocation and launch discipline bought nothing, and the arena
work section 4 expected to transfer from `llmxabe` was not needed. The whole
gain came from arithmetic intensity inside one kernel:

| change | decoder |
| --- | --- |
| one thread per output element | 4.6% of peak |
| + 8 output channels per thread (register tile) | 13.7% |
| + 4 time positions per thread (weight reuse) | 18.0% |

The second step is the one worth remembering: after the channel tile the kernel
was loading one weight per multiply-add, so it was still load-bound - just on a
different operand than before. **Fixing an arithmetic-intensity problem on one
operand can leave you with the same problem on the other.**

## 7. What inverted when the scope widened

Sections 2 and 4 are right about VITS and wrong about most of what the engine
now contains. Recorded here rather than edited above, because the reasoning was
sound for the model it was about and the failure is one of scope:

**"Quantising would be solving a problem this workload does not have."** True of
36 M parameters in 145 MB. The chat model is 4.9 GB and the translator 7.9 GB,
both read once per decoded token, and there decode *is* bandwidth bound - the
matmul streams 4.62 GB in 8.4 ms, which is 93% of what the card can read. The
weights stay in the checkpoint's own blocks and are unpacked inside the matmul.
See `docs/KERNELS.md`.

**"Anything about KV caches does not transfer."** Three stages have one: the
ASR's decoder, and both Llama stages, which hold theirs at f16 and grow it by
doubling. What still does not transfer is the rest of that sentence - prefix
reuse, speculative decoding and request scheduling are all still absent, and
`docs/ARCHITECTURE.md` says why.

**"Launch overhead turned out to be negligible."** True of the synthesiser, and
measured there. It is not true of a Llama decode, which issues about 390
kernels for one token; a seventh of a 13 B step is small kernels that cost what
a launch costs rather than what they read, and that is what `silu_mul_pair`
exists for. It is still not the *binding* constraint - the CPU queues those 390
launches in 2.03 ms of a 5.09 ms run and then waits, which is why capturing the
step in a CUDA graph was measured as pointless and not built.

What did transfer, and held everywhere: NVRTC at runtime, a differential test
per kernel, and the rule that a measurement lands with its optimisation.
