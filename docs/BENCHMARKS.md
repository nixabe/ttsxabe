# Benchmarks

## Current standing

One Quadro RTX 8000, `facebook/mms-tts-nan`, the sentence
`lí hó, kin-á-ji̍t thinn-khì chin hó.` (69 symbols, ~2.6 s of audio at 16 kHz).
Twenty timed synthesis calls after five warm-up, medians, alternated in pairs.

| implementation | median | x realtime |
| --- | --- | --- |
| PyTorch, CUDA, fp32 | 65.6 ms / 2.85 s | 43.2 |
| `xabe-tts`, CUDA, fp32 | 48.4 ms / 2.61 s | 53.9 |
| `xabe-tts`, CPU, scalar | ~120 s / 2.67 s | 0.02 |

**1.24x faster than PyTorch** per second of audio, stable to within 0.2 across
three interleaved rounds. The utterance lengths differ because both sample
their own durations, which is why the comparison is made on time per second of
audio rather than on the raw medians.

Where that time goes, measured with `xabe-tts-bench --stages`:

| stage | ms | share |
| --- | --- | --- |
| decoder | 34.1 | 72% |
| text encoder | 6.0 | 13% |
| flow | 5.1 | 11% |
| duration predictor | 1.9 | 4% |
| prior, download | 0.1 | <1% |

The CPU row is the scalar reference and is not a target: it exists to be read
and to be correct.

This section holds the current numbers and nothing else. When a measurement
supersedes a cell, replace the cell — never append a dated note, a before/after
delta, or an "improved from X" narrative. The change story belongs in the commit
message; durable reasoning belongs in WHY below, and measured rejections in
WHY NOT.

## Headroom

The decoder is 100.2 GFLOP for this utterance - 6.15e8 per input frame, 2.4e6
per output sample - and it runs in 34.1 ms, which is 2.94 TFLOPS, or **18% of
this card's 16.3 TFLOPS fp32 peak**. At 100% of peak the decoder would take
6.1 ms; nothing reaches that, but a well-tuned dense kernel reaching 40-60%
would put it at 10-15 ms.

So the honest statement of remaining headroom is roughly 2-3x on the decoder,
and it is in the convolution kernel rather than anywhere else. This number is
computed rather than guessed because
[OPTIMIZATION.md](OPTIMIZATION.md) refuses to promise a factor without it.

## The baseline to beat

Measured on the pipeline this project exists to replace, on the target hardware,
`facebook/mms-tts-nan` under PyTorch on one Quadro RTX 8000:

| input | wall clock |
| --- | --- |
| short clause (~7 syllables) | ~90 ms synthesis |
| whole `/tts` request, short | ~274 ms including translation |

For context, the alternative engine in that pipeline (Fun-CosyVoice3, 0.5 B,
24 kHz) takes ~820 ms on the same clause. This project targets the 36 M VITS
model, so ~90 ms is the number to beat, not 820 ms.

Comparisons must be against PyTorch's best settings on this card, not its
defaults. A ratio measured against a badly configured baseline is not a result.

## How to measure

- Release build, `cargo test --workspace --release` green first.
- Warm the model, then time N ≥ 20 synthesis calls, report the **median** and
  the spread.
- Alternate implementations in pairs rather than running all of A then all of B.
  This card thermally drifts, and a 5% difference measured back-to-back is
  indistinguishable from drift measured in blocks.
- State the utterance length. Synthesis time scales with output frames, so a
  number without its input is not a number.

## Correctness gates

No performance number is admissible unless `cargo test --workspace --release` is
green on the same commit. A fast wrong kernel is not a result; see
[TESTING.md](TESTING.md).

## WHY NOT

Measured rejections. Things that looked like they should help and did not.

### `torch.backends.cudnn.benchmark = True` makes the baseline 13x slower

The obvious knob to turn when comparing against PyTorch, and turning it moved
the baseline from 65 ms to 1023 ms. cuDNN's autotuner caches its algorithm
choice per input shape, and every utterance has a different frame count because
the durations are sampled - so it re-tunes on every call and reuses nothing.

The general lesson: **autotuning is a bet on shape stability, and a model that
samples its own output length has none.** Left off, which makes PyTorch's
defaults its best settings for this workload.

### fp16 was already rejected upstream

Measured slower on these cards for the neighbouring TTS model in the same
pipeline. Turing has fp16 tensor cores but no bf16, so mixed precision here
means real overflow risk for no measured gain. Not attempted.

# WHY

Durable reasoning. Things learned that outlive the change that taught them.

## The vocabulary is POJ, and getting that wrong is inaudible to the author

`facebook/mms-tts-nan` has `c` and U+0358 in its 48 symbols, which makes it POJ,
not Tâi-lô. Converting POJ to Tâi-lô before synthesis moves the text out of
distribution: on a sentence differing only in `chin`/`tsin`, ASR read the POJ
version back as 你好 今天天氣很好 and the Tâi-lô version as B號今天登記正好.

The general lesson: **the round trip through an ASR model is a usable objective
metric for TTS correctness** when you cannot evaluate the output language by
ear. It is not a substitute for a differential test, but it catches whole
classes of orthography and prosody bugs that no unit test would.

## A fifth of the checkpoint is never read

`posterior_encoder` (100 tensors, 7.24 M parameters) exists for training. Load
time and VRAM budgets should be computed on the inference subset.

## WHY NOT

Measured rejections. Things that looked like they should help and did not.

### `torch.backends.cudnn.benchmark = True` makes the baseline 13x slower

The obvious knob to turn when comparing against PyTorch, and turning it moved
the baseline from 65 ms to 1023 ms. cuDNN's autotuner caches its algorithm
choice per input shape, and every utterance has a different frame count because
the durations are sampled - so it re-tunes on every call and reuses nothing.

The general lesson: **autotuning is a bet on shape stability, and a model that
samples its own output length has none.** Left off, which makes PyTorch's
defaults its best settings for this workload.

### fp16 was already rejected upstream

Measured slower on these cards for the neighbouring TTS model in the same
pipeline. Turing has fp16 tensor cores but no bf16, so mixed precision here
means real overflow risk for no measured gain. Not attempted.

# WHY NOT

Measured rejections. Each entry saves someone the same week.

## fp16 on the flow/decoder — rejected in the reference implementation

Not yet measured for this implementation, but the neighbouring engine in the
same pipeline (Fun-CosyVoice3) was measurably **slower** with `fp16=True` on
these cards: 3.48 s vs 2.27 s on a short utterance, 7.56 s vs 5.61 s on a longer
one. Turing has fp16 tensor cores but the cast overhead dominates when the
kernels are memory-bound.

Do not assume fp16 is free here. Measure it, and expect it to lose.
