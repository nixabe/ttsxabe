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

### ASR

One Quadro RTX 8000, `Breeze-ASR-26` (large-v2, 1.54 B), greedy, `language=zh`.
Twenty timed transcriptions after three warm-up, medians, alternated in pairs
against a `whisper-server` started **without** `--vad` so that both sides do
the same job. Both produced identical transcripts on every clip.

| clip | `xabe-asr`, CUDA | `whisper-server`, f16 | ratio |
| --- | --- | --- | --- |
| 2.67 s | 264 ms | 144 ms | 0.55x |
| 3.90 s | 296 ms | 177 ms | 0.60x |

**This is the milestone's target missed, not met.** `docs/MILESTONES.md` asked
for an ASR faster than `whisper-server`; it is 1.8x slower, and the section
below computes what closing that would take rather than promising a factor.

Where the 264 ms goes, measured with `xabe-asr-bench --stages`:

| stage | ms | share |
| --- | --- | --- |
| encoder | 173 | 66% |
| decode loop, 6 tokens | 57 | 22% |
| cross-attention KV | 21 | 8% |
| mel frontend (CPU) | 17 | 6% |

The encoder is 2.26 TFLOP for a 30-second window, so 173 ms is 13.1 TFLOP/s -
55% of what this workspace's own matmul reaches standalone, and 13% of the
card's 99 TFLOP/s f16 tensor-core peak. Note that the window is fixed: a 2.67 s
clip and a 29 s one cost the same encoder.

### Translator

Measured now that it runs. It is on the reply path whenever `--direct-taigi` is
absent, which is how the pipeline is served when Taigi output is wanted from a
Mandarin-speaking chat model, and the paragraph that used to sit here said the
measurement to take was decode tokens per second against `llama-server`. That
number is below, with the chat model beside it.

What is known: the weights are 26.5 GB at f16, and the three-test oracle binary
takes 113 s end to end on one card with most of that a single load — which is
why those tests share one `OnceLock<Mutex<_>>` instance rather than loading per
test. Three concurrent loads is 80 GB and an out-of-memory that reads like a
broken loader. If the translator ever returns to the
reply path, the measurement to take is decode tokens per second against
`llama-server` on the f16 GGUF, alternated in pairs on the same card, exactly
as the ASR is measured above.

This section holds the current numbers and nothing else. When a measurement
supersedes a cell, replace the cell — never append a dated note, a before/after
delta, or an "improved from X" narrative. The change story belongs in the commit
message; durable reasoning belongs in WHY below, and measured rejections in
WHY NOT.

## The two Llama stages: 6.4x, and the ceiling that is left

One Quadro RTX 8000, `xabe-llm-bench`, 128 prompt tokens then 64 decoded,
medians over five rounds after one warm-up. Decode is what a listener waits
through - a reply of N tokens is one prefill and N decodes - and it is what
`llama-server` reports, so it is the number that can be compared.

| Stage | Checkpoint | Prefill | Decode before | Decode after |
| --- | --- | ---: | ---: | ---: |
| chat | Breeze2 8 B Q4_K_M | 303 tok/s | 9.5 tok/s | **61.0 tok/s** |
| translator | Taigi 13 B Q4_K_M | 210 tok/s | 5.6 tok/s | **35.1 tok/s** |

**6.4x and 6.3x.** Effective bandwidth against the file on disk went from 47
and 45 GB/s to 300 and 282.

### What was wrong: a header decoded once per element

Decode is a `gemv` per projection and should be bound by streaming the weights
once. It was bound by unpacking them. `q_elem` re-derives a block's header for
every element it returns - two f16 scales, a six-bit sub-block scale, and four
integer divisions - because it is written to be read against the format tables
one case at a time. At 256 elements to a K-quant super-block that is 256 header
decodes where one is needed.

The fix is a specialised path for **Q4_K and Q6_K**, which between them are every
weight byte in both checkpoints - 74% and 26% of the chat file, 77% and 23% of
the translator. Eight elements per lane makes every divisor a power of two and
every quotient loop-invariant, so thirty-two lanes at eight elements is one
super-block per warp. Anything else still goes through `q_at`.

### What was also wrong: every packed byte fetched more than once

Hoisting the header left the decode at 116 GB/s, and the second half of the gap
was in *which* eight elements a lane took. Adjacent ones are the obvious choice
and the wrong one, because a K-quant byte does not hold adjacent elements. A
Q4_K byte packs two elements 32 apart, so a lane wanting eight adjacent elements
loads eight bytes and throws away a nibble of each - and the byte it half-used
is loaded again by the lane that wanted the other half. Q6_K is worse: its `ql`
nibbles are 64 apart and its `qh` 2-bit fields 32 apart, so eight adjacent
elements cost sixteen byte loads and every byte is fetched two or four times.

Regrouping fixes both. A Q4_K lane takes four whole bytes - one aligned 32-bit
load and two `float4` activation loads for the same eight elements, against ten
loads before - at the cost of a second sub-block scale pair, because the two
nibbles of a byte land in adjacent sub-blocks. A Q6_K lane takes two adjacent
columns across all four 2-bit fields, which is three 16-bit loads against
sixteen 8-bit ones, and across a warp covers `ql[0..127]` and `qh[0..63]`
exactly once. Standalone on this card at n=14336, k=4096:

| Format | Adjacent eight | Regrouped eight |
| --- | ---: | ---: |
| Q4_K | 384 us, 86 GB/s | **88 us, 372 GB/s** |
| Q6_K | 410 us, 117 GB/s | **129 us, 373 GB/s** |

Q6_K reads shorts and not words because its block is 210 bytes: the stride is
even but not a multiple of four, so successive blocks are only 2-byte aligned.
Q4_K's 144-byte block is a multiple of four and its load is a word.

The `float4` activation read needs the row to start on a 16-byte boundary, which
depends on strides the kernel is handed rather than anything it controls, so the
kernel tests and keeps a scalar path for when it does not hold. The test is a
template parameter and not a branch inside the loop: it is loop-invariant and
warp-uniform either way, and leaving it in the loop measured 27% slower.

### The ceiling that is left, and what it is not

300 GB/s against a card that streams 672, and the isolated kernel reaches 372.
The gap between those two is the rest of a decode - the small key and value
projections, attention, the norms, and a launch per projection - not the matmul.

`--packing f16` used to be the faster option and is not any more. It reads 2.6x
more bytes per token for the same weights, and now that the packed path is no
longer wasting its loads, that costs what it should:

| Same file, same weights | Decode | Prefill | Residency |
| --- | ---: | ---: | ---: |
| `--packing packed` | **16.4 ms/tok, 61.0 tok/s** | 302 tok/s | 4.9 GB |
| `--packing f16` | 28.5 ms/tok, 35.0 tok/s | **469 tok/s** | ~16 GB |

So packed is 1.74x faster at decode *and* 3.3x smaller, and the trade-off that
made f16 worth considering is gone. f16 still wins prefill by 1.55x, because
prefill is a `gemm` with as many rows as there are prompt tokens and reaches the
tensor cores, which the packed path does not - that one is real and unaddressed.

## A spoken turn: where the 7 s went

`xabe-engine --serve`, one typed turn, the reply chunked as it streams and each
clause translated then synthesised. Measured over the WebSocket, so these are
what a listener waits through rather than what a stage costs in isolation.
Three clauses, Tacotron2, Breeze2 8 B chat, Taigi 13 B translator; medians
of three runs of the same prompt.

| | one card | translator on card 1 |
| --- | ---: | ---: |
| first audio | 2272 ms | **1930 ms** |
| whole turn | 6784 ms | **5620 ms** |

The one-card column is after the Tacotron2 second pass below; the two-card one
predates it and is that much better again in practice.

Read that as the cost of the single-card constraint rather than as a speedup
available everywhere: **everything below applies only across cards.** On one
card the numbers are what they always were, and deliberately so - see the second
change.

Two changes, and the order matters because the first one is a flag.

**The translator was sharing a card with the chat model that feeds it.** Not
idly: the reply is chunked as it streams, so the first clause is translated
while the second is still being written, and the two decode loops interleave on
one set of SMs. Moving the translator to `--translator-device 1` took first
audio to 2000 ms on its own. The later clauses did not move, which is the tell -
by then the chat model has finished and there was nothing to contend with.

**Translation and synthesis were strictly sequential.** They are different
models on now-different cards, so clause N+1 is translated while clause N is
still becoming a waveform. Synthesis stays a single ordered consumer, because
audio has to reach the browser in playback order.

**Overlapping them on one card is worse than not.** Measured before the overlap
was made conditional: synthesis went from about 400 ms a clause to 950-1200,
first audio from 2659 ms to 2919, and the whole turn no faster. Two GPU jobs on
one set of SMs do not run in half the time; they run in the same total time and
delay whichever finishes first, which here is the clause the listener is waiting
for. `xabe-engine` therefore compares the translator's resolved device with the
synthesiser's and only overlaps when they differ - `translate_ahead` in the
startup line says which it chose.

### The split, and why the synthesiser is not the thing to optimise

Per clause, with the translator on its own card:

| clause | translate | synthesise |
| --- | ---: | ---: |
| 9 characters | 1145 ms | 214 ms |
| 15 characters | 1373 ms | 389 ms |
| 16 characters | 1883 ms | 581 ms |

Synthesis is a sixth to a quarter of a clause and runs at about twelve times
realtime. Halving it would take roughly 200 ms off a turn; halving the
translator would take nearly a second. The remaining lever on a turn is the
13 B translator's decode rate, not Tacotron2.

## Residency: the whole pipeline on one card

One Quadro RTX 8000 (49152 MiB), measured with `xabe-vram`, which reads
`nvidia-smi` rather than the allocator - the CUDA context and the driver's own
reservations count against the card, and a per-process figure would omit them.
Stages are loaded **cumulatively in one process**, because that is the
configuration being asked about; loading them separately and adding the peaks
would answer a different question and give a smaller number.

| stage | container | delta MiB | cumulative |
| --- | --- | --- | --- |
| TTS, VITS 36 M, + the CUDA context | safetensors, f32 | 297 | 297 |
| ASR, Whisper large-v2 1.54 B | safetensors → f16 | 3 200 | 3 497 |
| chat, Breeze2 8 B | GGUF `Q4_K_M`, packed | 6 400 | 9 897 |
| translator, Llama-2 13 B | GGUF `Q4_K_M`, packed | 8 608 | 18 505 |
| CosyVoice3, LM + flow + vocoder | safetensors | 3 266 | **21 771** |

**21 771 MiB — 21.3 GiB of a 48 GiB card**, 44% of it, leaving 27 375 MiB for
KV caches and activations. Without CosyVoice, which is the alternative
synthesiser rather than a second stage, the four remaining are 18 505 MiB.

The context is charged to the TTS row because it is created by whichever stage
opens the device first, and 36 M parameters is where it is obviously the
context rather than the weights. CosyVoice is measured as its three
GPU-resident sub-models opened directly; `Cosy::open` additionally wants a
voice bundle, which is four small tensors and occupies nothing worth
reporting.

### What the packing is worth, same file loaded both ways

`Packing::F16` unpacks at load, which is what this engine did before
`Operand::Q`. Same bytes on disk, same arithmetic, different residency.

| model | file | `Packing::F16` | `Packing::Packed` | ratio |
| --- | --- | --- | --- | --- |
| Breeze2 8 B `Q4_K_M` | 4 685 MiB | 16 489 MiB | 6 400 MiB | 2.58x |
| Taigi Llama-2 13 B `Q4_K_M` | 7 663 MiB | 26 025 MiB | 8 608 MiB | 3.02x |

The packed figures exceed the file sizes by 1 715 and 945 MiB, and that gap is
the **embedding table**: a gather rather than a matmul, so it has its own kernel
and is still widened to f32 at load. The gap is what widening costs *over*
storing it packed - at 8 B, 128 256 x 4 096 as f32 is 2 004 MiB against about
282 MiB as `Q4_K`, a difference of 1 722 MiB, which is the 1 715 measured. The
13 B's smaller vocabulary gives 940 MiB by the same arithmetic against 945
measured.

It is also why the 8 B ratio is the worse of the two despite identical
quantization: a larger vocabulary over a smaller model puts more of the
checkpoint into the one tensor that is not packed.

### Why this is the difference between fitting and not

At f16 the same four stages come to **46 011 MiB** - 297 + 3 200 + 16 489 +
26 025 - against a 49 152 MiB card. That is 93.6% full and leaves 3 141 MiB,
which is *less than the 13 B's own KV cache* at any useful context length.

Add CosyVoice, which is unquantized either way, and f16 comes to **49 277
MiB** against a 49 152 MiB card: it does not merely leave too little headroom,
it exceeds the card by 125 MiB and fails to load at all. Packed, the same five
stages are 21 771 MiB and use 44% of one card.

So the honest statement is not "f16 is tight". At f16 these stages do not share
a card; packed, they share one with 27 GB to spare.

### This is residency, not speed

Nothing here was timed. The unpacking feeds the same f16 tensor-core path an
f32 weight always fed, so `Q4_K_M` buys memory and the int8 path that would
make `Q8_0` *faster* rather than merely smaller is still not in this workspace.
Whether unpacking per use costs measurable time on the decode loop has not been
measured and is not claimed either way.

## Tacotron2 + WaveGlow: 3.07x, and where it went

Measured on card 0, Quadro RTX 8000, medians over nine rounds after two warmups,
with `xabe-taco-bench`. Synthesis is stochastic, so the frame count moves
between runs on the same text and a mean would mostly be measuring that.

| Text | Audio | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `li2 ho2` | 5.57 s | 1381.6 ms | 453.3 ms | 3.05x |
| `gua2 si7 tai5-uan5-lang5` | 1.60 s | 407.9 ms | 133.0 ms | 3.07x |
| a two-clause line | 5.57 s | 1399.4 ms | 467.4 ms | 2.99x |

**3.90x realtime to 12.04x realtime** on the middle line.

### A second pass: 1.28x more, from one thread per output element

`nsys` rather than the built-in breakdown, because the breakdown synchronises
per stage and the decoder has seven stages inside a loop that runs once per mel
frame - it charges its own syncs to whatever it is timing. The kernel summary
does not:

| kernel | share of GPU time | calls |
| --- | ---: | ---: |
| `gemm` | 37.5% | 1980 |
| `linear` | **25.6%** | 5190 |
| `gemv` | 14.9% | 7645 |

A quarter of the time in `linear`, which is the plain one-thread-per-output
kernel, at 46 us a call. Tacotron2's decoder called it directly for the
projections it runs every frame, and for `m` of one that is as bad as it sounds:
the gate is `n` of one, so **one thread** walked 1536 weights while the other 71
SMs idled, and the query got 128 threads. `Gpu::gemm` dispatches a single row to
`gemv` - a warp per output column and a shuffle reduction - and keeps f32 either
way, so the three single-row projections moved to it.

| Text | Audio | Before | After | Realtime |
| --- | ---: | ---: | ---: | ---: |
| `Tâi-lâm ū chiok chē hó-chia̍h--ê,` | 2.40 s | 203.1 ms | **158.5 ms** | 15.16x |
| `chhin-chhiūⁿ:Khah-sú môa-lî,...` | 4.62 s | 374.2 ms | **296.9 ms** | 15.56x |
| `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó,...` | 7.98 s | 643.7 ms | **506.8 ms** | 15.74x |

**1.28x, and 12.0x realtime to 15.5x.** It costs what a different summation
order costs: same sample count, cosine 0.9999973, error 52.2 dB below the
signal.

The conditioning is also one matmul per flow now instead of one per layer -
WaveGlow's `cond_layer` does not depend on the audio being transformed, and the
checkpoint already stores it as a single `[2 * ch * layers, cond]` matrix that
was being sliced apart at load. In isolation that shape is 2.94 ms against 2.26,
11.8 TFLOP/s against 15.3; end to end it is worth 2.6%, measured 158.5 ms
against 162.8 with everything else equal. Bit-identical output, so it stays, but
it is not where the 1.28x came from.

Through a warm server, against the engine already running beside it:

| Engine | Round trip | Audio | Realtime |
| --- | ---: | ---: | ---: |
| mms (VITS) | 36.7 ms | 1.38 s @ 16 kHz | 37.4x |
| tacotron2 | 132.6 ms | 1.59 s @ 22.05 kHz | 12.0x |

Those two rows predate the second pass. Through the socket after it, on the
single-card layout, synthesis of a clause is 271-318 ms where it had been
338-447, and the turn it sits in moved with it:

| one card, three clauses | before | after |
| --- | ---: | ---: |
| first audio | 2738 ms | **2272 ms** |
| whole turn | 7206 ms | **6784 ms** |

Still 3.6x behind the synthesiser this repository started with, which is what an
autoregressive decoder and an 87.9 M-parameter flow vocoder cost against a
one-shot 36.3 M-parameter VITS. It is not what decides a turn.

### What the four changes were worth

Every one of them is the same observation: the work was being done by a general
kernel where a specialised one already existed.

| Change | Median | Note |
| --- | ---: | --- |
| baseline | 407.9 ms | |
| 1x1 convolutions to `gemm` | 296.8 ms | `wn cond` alone, 117.8 to 35.6 ms |
| the decode loop's one-row projections to `gemm` | 215.0 ms | dispatches to `gemv` |
| the dilated convolution to `im2col` + `gemm` | 176.4 ms | |
| the coupling network kept in `[steps, channels]` | 133.9 ms | |
| the matmul path's weights stored f16 | 132.2 ms | ~1%, kept for the width |

**A 1x1 convolution is a matmul.** `conv1d` is a windowed kernel that stages a
halo in shared memory, for a window of one. Four of WaveGlow's five projections
per layer are 1x1.

**One row is a `gemv`.** The decoder's per-step projections were going through
`linear`; `gemm` dispatches to `gemv` below seventeen rows, which keeps f32 end
to end - only the *tiled* kernel stages f16. So this one was free of any
accuracy question, and the encoder's agreement with the reference improved
slightly, from 1.252e-6 to 1.222e-6.

**The layout was the last third of it.** Every operation in a coupling network
is a matmul, and a matmul wants its contracted axis last. Holding the data
channel-major meant transposing around each of them - about thirty per flow,
three hundred and sixty per utterance. Keeping the whole network in `[steps,
channels]` leaves two, at the boundary with the flow. That needed the
conditioning and residual/skip weights split at load, because in this layout
their output slices are strides rather than ranges, and a `gated_activation`
that splits along the inner axis.

**f16 weights bought about one percent.** Kept anyway: the tiled kernel rounds
both operands to f16 inside itself regardless, so storing them rounded is
strictly less work and half the width at identical numerics. The measured effect
on speed was inside the noise, and is reported as such rather than rounded up.

### What it cost in accuracy: -54 dB

The vocoder now reaches the tensor cores, which round both operands to f16.
Against the original f32 path, on the same seed:

- identical length, 36096 samples
- correlation **0.999998**
- rms difference **1.94e-3 of rms signal, -54.3 dB**

Below WaveGlow's own sampling noise. The mel is bit-identical either way - the
decoder's arithmetic did not change precision, only kernel - so this is the
vocoder alone.

### The profile that is left, and why it stops here

Of 156 ms on the timed run of the middle line: the coupling networks are 75 ms
and are now real tensor-core work, and the decode loop is 69 ms and is **launch
bound**. Thirty-five launches a step, 138 steps, at ten to fifteen microseconds
of latency each - the arithmetic in a step is a handful of `gemv`s over
kilobytes. Fusing them, or replaying the step as a CUDA graph, is the next
lever and is a project rather than a change.

### A measurement trap in the harness itself

The per-stage breakdown attributed 23.9 ms to `coupling_inverse`, a kernel that
measures 6.6 us in isolation on the same shapes - a factor of three hundred. It
was not that kernel. Timing a stage means synchronising after it, and the
transposes being timed elsewhere were queueing work that some later sync had to
drain; the breakdown moved it to whichever stage happened to sync. It vanished
when the transposes did.

So the totals in these tables are from runs with timing **off**, and the
breakdown is only ever used to decide where to look next. `taco_bench` prints
both and labels the timed one as not comparable, which is the honest way to
show a number that is useful and wrong.

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

### The ASR's headroom, and what it would cost

Two things account for the encoder's 173 ms, and both are known quantities.

**34 ms of it carries no arithmetic at all.** The attention score matrix is
20 heads x 1500 x 1500 floats - 180 MB - and it is written by the score
product, read and written again by the softmax, and read by the context
product. That is 23 GB of traffic across 32 layers, at 672 GB/s. Flash
attention removes it entirely by never materialising the matrix.

**The matmul runs at 23.8 of 99 TFLOP/s.** The projections and feed-forwards
are 1.89 TFLOP, which is 79 ms at that rate. `ldmatrix` and a double-buffered
global-to-shared pipeline are the standard route to 40-50% of peak on this
architecture; at 40% the same work is 47 ms.

Both together put the encoder near 100 ms and the whole transcription near
190 ms - still above `whisper-server`'s 144. A third lever would be needed, and
what it is has not been established, which is exactly why this section does not
name one.

## CosyVoice3 in-engine: a preliminary figure, and why it is only that

| implementation | median | seconds of audio | s per s of audio |
| --- | --- | --- | --- |
| Python `taigi_tts_daemon.py`, `POST /tts` | 3.57 s | 3.64 | 0.98 |
| `xabe-engine`, `--tts-engine cosyvoice=<dir>` | 4.61 s | 6.08 | 0.76 |

**1.29x faster per second of audio.** The utterance lengths differ because both
sample their own speech tokens, which is the same reason the VITS comparison
above is made per second of audio rather than on the raw medians.

This does **not** meet this document's own bar and should not be quoted as if it
did. Five timed calls, not twenty. More importantly the two are not on the same
card: the Python service shares GPU 1 with a 26 GB `llama-server`, and the
engine has GPU 2 to itself. That confound points the same way as the result, so
the real figure is somewhere below 1.29x and the honest statement is that the
port is *not slower*. A proper paired run belongs in `xabe-tts-bench` alongside
the VITS one, on one card, and has not been done.

The `examples/say` path, measured on its own: 3.1 s to load all three networks,
then 6.08 s of audio in 5.06 s — 1.20x realtime, one utterance at a time, no
batching and no streaming.

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
  number without its input is not a number. The ASR is the opposite case and
  needs saying for the opposite reason: its window is a fixed 30 seconds, so
  the encoder costs the same whatever the clip.
- Do not compare across separate invocations. This card drifts about 6% between
  runs of identical code - measured, while chasing a change that turned out to
  be noise - which is more than most optimisations are worth. Only numbers from
  the same alternated run are comparable.

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

### f16 activations: implemented, measured at 5%, removed

Rounding the *weights* to f16 was worth 447 ms to 264 - the tiled matmul rounds
both operands on the way into shared memory anyway, so storing F32 bought no
accuracy and cost twice the traffic. Doing the same to the *activations* looked
like the identical argument and was not: about 5%, at the cost of a conversion
pass and an `Act` type threaded through every layer.

The reason is the card, not the arithmetic. A projection's left operand is
re-read once per column tile - ten times, forty for the feed-forward
expansion - but a 1280x1280 f16 weight tile is 3.3 MB and this card's L2 is
6 MB. The re-reads were already being served by L2, so halving them halved
traffic that never reached memory.

The general lesson: **an operand small enough to sit in L2 is not on the
bandwidth budget, however many times the kernel reads it.** Arithmetic
intensity computed against DRAM overstates the cost of any tile that fits.

The kernel keeps its symmetric `Operand` support, with a differential test that
asserts the f16 and F32 staging paths are *bit-identical* rather than merely
close. The cross-attention KV cache uses it: every decode step reads all 32
layers of both halves in full, and that does not fit in L2.

### The mel frontend was 38% of the ASR, and 91% of it was silence

Not a rejection - a result, recorded here because it is the shape of mistake
that is easy to repeat. Whisper's window is a fixed 30 seconds and utterances
in this pipeline are two or three, so most of the frontend was transforming
zero-padding. A frame of digital silence has a zero spectrum, so skipping it is
exact rather than approximate: 171 ms to 17.

The general lesson: **a fixed-size window makes the padding part of the cost**,
and the padding is the part with a closed-form answer.

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

### Stopping the translator at its stop string saved nothing measurable

`Translator::translate` generates up to `max_new` and then cuts the answer at
`[/` or a newline followed by `[`, so any token after the stop string is decoded
and thrown away. With `max_new` at 256 and decode at 28 ms a token that looked
like most of a translation's cost.

It is not. Checking the stop strings inside the loop instead of after it left
translation at 1145, 1373 and 1883 ms on the three clauses that had cost 1154,
1361 and 1865 - flat, because this checkpoint emits `</s>` or `<pad>` at the end
of the answer and the loop already stopped there. The answers measured 24, 34
and 53 tokens, which is what the text is worth.

The check was kept anyway, and this is why it is recorded here rather than as a
speedup: the cut in `translate` exists precisely because the model *sometimes*
closes its tag instead of ending, and on those turns the loop had no way to
know. It bounds a tail that does not show up in a median.

### Two micro-optimisations of the K-quant gemv, both measured flat

Once the header decode was hoisted, the two obvious next steps both changed
nothing and were reverted rather than kept for the look of the thing.

**An eight-byte vectorised load.** A Q4_K block is 144 bytes and every offset
into its quants is a multiple of eight, so a lane's eight bytes are one `uint2`
rather than eight `unsigned char`. 23.8 to 23.7 tok/s.

**Two accumulators over an unrolled pair of super-blocks**, to break what is
otherwise one dependent chain of fused multiply-adds down the whole row. 23.7
tok/s, unchanged. Still flat after the regrouping above, and by then it also
cost 8%, so it stayed out.

The latency chain really is not the limit. The loads are, and the first of these
is the more useful failure: it cut the *number* of load instructions without
touching how many times each byte was fetched, because it kept one nibble per
lane. Fetching each byte once needed the lane-to-element map to change, not the
load width - see the regrouping above, which is 4.3x. A vectorised load over the
wrong grouping is a faster way to do the redundant work.

This entry used to end by concluding from those two flat results that the loop
was arithmetic-bound and that occupancy and register pressure were what to look
at next. Both halves were wrong. `ptxas -v` puts `gemv` at 64 registers with no
spills, which is full occupancy on sm_75, and the variant that removed the
arithmetic while keeping the loads ran at 431 GB/s against the shipped kernel's
86. Two flat results are evidence about the two things tried and not about
everything else.

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
