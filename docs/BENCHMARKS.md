# Benchmarks

## Current standing

The one-line version: the synthesiser is 1.24x faster than PyTorch, the ASR is
0.99x against `whisper-server` on three seconds of speech and 1.05x to 1.15x
from five seconds up - level on the briefest clip, by a margin inside both
sides' own spread, and ahead on every other, alternated in one sitting against
a `whisper-server` built here from the same checkpoint - and both Llama
stages are **level with or
ahead of llama.cpp on every number measured** - the chat model ahead on all
three of its rows (1.08x and 1.17x prefill, 1.06x decode), the translator
ahead on decode, level on 512-token prefill, and at 0.94x of a 128-token
llama.cpp median that swings 20% between its own runs, which the table below
reads as inside llama.cpp's noise and no stronger claim. Each of those has its
own section; this paragraph is not the evidence for any of them.

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
audio rather than on the raw medians. The engine has since gained 1.085x on
this shape from the transposed convolution alone - 46.25 ms to 42.64, 61x
realtime - measured against its own previous binary and not against PyTorch
again; see "The transposed convolution" under Tacotron2.

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
Medians of twenty rounds after three warm-up, each round running one of each,
against a `whisper-server` started **without** `--vad` so that both sides do
the same job. Both produced identical transcripts on both clips.

**Both halves of every ratio here were measured in one sitting**, on the same
card, against a `whisper-server` built from this repository's own
`whisper.cpp` checkout with `GGML_CUDA=ON` and `CMAKE_CUDA_ARCHITECTURES=75`,
reading the same checkpoint converted by that tree's own
`models/convert-h5-to-ggml.py`. The version of this table that stood before
was arithmetic across two sittings and said so; it is superseded rather than
appended to.

| clip | `xabe-asr`, CUDA | `whisper-server`, f16 | ratio | transcripts |
| --- | --- | --- | --- | --- |
| 2.93 s | 185.9 ms | 189.4 ms | **1.02x** | identical |
| 4.98 s | 220.8 ms | 239.8 ms | **1.09x** | identical |
| 7.28 s | 243.5 ms | 266.6 ms | **1.09x** | differ |
| 9.95 s | 291.8 ms | 353.8 ms | **1.21x** | differ |

The row before this one was 0.99x / 1.05x / 1.05x / 1.15x, and the one before
that 0.94x / 1.00x / 0.99x / 1.08x; both moved from the decoder and neither
from the encoder, and what moved this one is in "Eight launches a layer"
below: the decode loop went from 68.8 ms to 64.3 on ten tokens and from 127.0
to 118.0 on twenty. On the 2.93 s clip the twenty rounds spread 184.4 to
187.1 ms here against 186.8 to 190.9 there, so the 3.5 ms between the medians
is outside both - the first time the shortest clip has been. The
`whisper-server` column is within 1 ms of the previous sitting's on every
clip, which is the check that the sitting was quiet.

`whisper-server` is started `-nf -bo 1 -bs -1` as well as without `--vad`, so
both sides are strictly single-pass greedy with no temperature fallback -
which is what this engine does and all it does. That turned out not to matter
(the fallback was not firing and the numbers are unchanged either way) but it
is the difference between a matched comparison and one that happened to be
matched.

**The milestone's target is met from five seconds of speech up and level at
three, and the shape of that is the useful result.** The two engines have
opposite cost structures. The encoder is a fixed 30-second window for both and
ours is about 19 ms slower at it, so every transcription starts that far
behind; the decode is cheaper here, by about 2 ms a token. From 2.93 s to
9.95 s our total grows 117 ms against their 165, and the fixed deficit is paid
off at about eleven tokens - one more than the shortest clip produces, which
is where its 1.3 ms comes from.

**The workload this engine exists for is the short end.** The pipeline runs
greedy over VAD-gated utterances of a few seconds, which is the 0.99x row - so
the honest reading is that the item is level rather than won for the case that
matters, by less than either side moves between its own rounds, and the 1.15x
is a fact about where the curve goes rather than the number to quote.

The two longer clips are also weaker evidence for a second reason: the
transcripts diverge. Both engines are single-pass greedy on the same weights,
so the paths separate as they lengthen and neither is wrong relative to the
other - but they are no longer doing quite the same work, and the two rows
where the transcripts are identical are the ones to trust.

The clips are synthesised rather than recorded, so the row is reproducible on
any box that has the checkpoints - `bench/` is gitignored and the two lines are
`lí hó, guá sī tâi-uân-lâng, tsin hoaⁿ-hí bat lí.` and `kin-á-jit ê thiⁿ-khì
tsin hó, lán lâi khì kong-hn̂g sàn-pōo, hó bô?`, spoken by `mms-tts-nan` at
16 kHz. They transcribe to ten and sixteen tokens.

Where the 211.9 ms goes, measured with `xabe-asr-bench --stages`:

| stage | ms | share |
| --- | --- | --- |
| encoder | 111 | 52% |
| decode loop, 10 tokens | 77 | 36% |
| cross-attention KV | 17 | 8% |
| mel frontend (CPU) | 6 | 3% |

The encoder is 2.26 TFLOP for a 30-second window, so 111 ms is 20.4 TFLOP/s.
Note that the window is fixed: a 2.93 s clip and a 29 s one cost the same
encoder, which is why the longer clip's ratio is the better one.

**The encoder is the whole of the remaining gap.** `whisper-bench` on the same
build and the same weights puts `whisper.cpp`'s encoder at 83.3 ms against this
one's 111, which is 28 of the 33 ms between the two columns on the 2.93 s clip.
Everything else is level - the decode loop, the cross-attention cache and the
frontend together are 100 ms against their 102.

Of those 111 ms, 86 are the tiled `gemm` at 22.4 TFLOP/s. **It is not the
accumulator type**, which is what this section used to say: measured on this
card, `mma.m16n8k8.f32.f16.f16.f32` runs at 102.3 TFLOP/s and the f16-accumulate
form at 103.0, so the trade whisper.cpp makes is worth 0.7% here. `whisper.cpp`
gets 27.3 TFLOP/s across its whole encoder from cuBLAS, and closing this is a
question of a tiled matmul that competes with it. Neither of the obvious
ceilings is the binding one - the gemm is at 78% of what its 128x128 tile's
arithmetic intensity allows against 672 GB/s, 22% of what the instruction
allows, and halving the larger operand stream by rounding the activations to
f16 was worth 5%.

**The binding one is the register file, and it is now measured.** `ncu` cannot
be run on this machine (`ERR_NVGPUCTRPERM`), but `ptxas -v` did not need to be:
the kernel compiles to exactly 128 registers a thread with no spill, which is
exactly the budget for the two blocks a SM that `65536 / (2 * 256)` allows, and
64 of those 128 are the accumulators a 128x128 tile over 256 threads cannot give
up. Software-pipelining the staging - the standard fix for a `stage; sync; mma;
sync` loop on an architecture without `cp.async` - was written and measured at
six tile-and-occupancy arrangements, and **every one of them lost**: 184
registers and one block a SM at the shipped tile for half the throughput, 108
bytes of spill when two blocks are forced back, and a fitting pipeline at 128x64
that gives up more arithmetic intensity than it recovers. `docs/KERNELS.md` has
the table and the arithmetic. The gap to cuBLAS is an architecture this kernel
shape has run out of room on, not a missing trick in the staging loop.

### The round that found the encoder's other half

The encoder had been treated as "the tiled `gemm` and some noise" for three
rounds. Timing every kernel it runs, at the encoder's own shapes, said
otherwise - and two of the rows were nothing like what the assumption implied.
Medians of nine on one Quadro RTX 8000, one call and then the 32 layers:

| kernel | one call | x32 layers | share |
| --- | ---: | ---: | ---: |
| `gemm` q/k/v/out, 1500x1280x1280 | 0.264 ms | 33.7 ms | 24.8% |
| `gemm` fc1, 1500x1280x5120 | 1.020 ms | 32.6 ms | 24.0% |
| `gemm` fc2, 1500x5120x1280 | 0.946 ms | 30.3 ms | 22.3% |
| `flash_attn`, 20 heads over 1500 | 0.740 ms | **23.7 ms** | 17.4% |
| `layer_norm_add` | 0.087 ms | 5.6 ms | 4.1% |
| `gelu` | 0.117 ms | 3.7 ms | 2.8% |
| `split_heads_t` | 0.109 ms | **3.5 ms** | 2.6% |
| `split_heads` | 0.051 ms | 1.6 ms | 1.2% |
| `scale_inplace` | 0.035 ms | 1.1 ms | 0.8% |

The arithmetic that reframed the problem is in the first four rows against
`whisper.cpp`'s column. The encoder's 32 layers are 2256 GFLOP; `whisper.cpp`
does them in 83.3 ms, which is **27.1 TFLOP/s across its whole encoder** -
barely above the 22.4 this engine's `gemm` reaches in isolation. cuBLAS was not
running away with it. Roughly half the gap was everything that is not a matmul.

**`split_heads_t` was a transpose written as a scatter.** It moves 15.4 MB in
109 us, which is 141 GB/s of a card that copies at about 500: the read is
coalesced and the write is not, because consecutive lanes land `t` floats apart
and a warp stores 32 sectors where it could store four. It is also, on
inspection, not a head permutation at all - the head structure cancels out of
the address arithmetic and it is the transpose of a `[t, d]` matrix. A 32x32
staging tile, padded to 33 words so the write-back reads a conflict-free
column, took it to **0.042 ms**, and dropping the zeroing of a buffer the
kernel fully overwrites took `split_heads` to 0.039.

**The activations were f32 into every matmul that stages them as f16 anyway.**
Rounding a projection's left operand before the call was measured at 5% of each
matmul and had been rejected, correctly, because a rounding *pass* costs more
than one matmul saves. Fused into the kernel that already has the value in a
register it is free, so `layer_norm_add` and `gelu` grew f16-emitting twins and
the encoder's projections read half the stream. This is bit-identical, not
approximately so: `f32_to_f16` is the same round-to-nearest-even `gemm_pack`
applies during staging, and the oracle test passes layer by layer unchanged.

The same trick pays a second time in front of the cross-attention cache, where
one 7.7 MB conversion serves **sixty-four** projections of the encoder output.

Measured on the 2.93 s clip, `xabe-asr-bench --stages`, medians of nine:

| stage | before | after |
| --- | ---: | ---: |
| mel frontend (CPU) | 5.2 ms | 5.5 ms |
| encoder | 110.9 ms | **101.9 ms** |
| cross-attention KV | 17.0 ms | **14.8 ms** |
| decode loop | 78.0 ms | 76.3 ms |

### The frontend was spending its time on silence

The mel frontend is CPU work the card sits idle through, and on the shortest
clip it was 5.3 ms of 199. Three things it was doing to padding rather than to
audio, all of them exactly equivalent to remove:

- **`log10` on a quarter of a million zeros.** `mel_power` leaves a digitally
  silent frame's row as it allocated it, so `power` is *exactly* zero there,
  and on a 2.93 s clip inside a fixed 30-second window that is nine bins in
  ten. `0f32.max(1e-10).log10()` is a constant; evaluating it once and
  comparing against zero is the same bits and not thirty cycles a bin.
- **Transposing frames that are all zero.** `mel_power`'s output transpose ran
  over all 3001 frames writing `frames` apart, which is 12 KB, so no two writes
  shared a cache line. Everything outside the live range is zero in the source
  and already zero in the destination.
- **Three threads of eight left idle.** The spawn threshold was 64 frames a
  thread, and 294 live frames is five threads. At 32 it is eight.

5.3 ms to about 3.6, and the 4.98 s clip crossed to 1.00x with it. Nothing
here touched a number the model sees.

### The decode step, kernel by kernel

The same profile for the other half, at `n = 1`. Every row here was measured
with a synchronise after the call, so every row carries the same **8.0 us**
floor of a launch and a round trip - measured with a do-nothing kernel and
printed as its own row, because without it every number below reads high:

| kernel | measured | less the floor | per token | GB/s |
| --- | ---: | ---: | ---: | ---: |
| `layer_norm_add` `[1,1280]`, x3 | 19.0 us | **11.0 us** | 1.06 ms | - |
| `gemv` 1x1280x1280, x6 | 16.0 us | 8.0 us | 1.54 ms | 410 |
| `gemv` fc1 1x1280x5120 | 35.0 us | 27.0 us | 0.86 ms | 485 |
| `gemv` fc2 1x5120x1280 | 35.3 us | 27.3 us | 0.87 ms | 480 |
| cross scores, 20x(1x64x1500) f16 | 26.6 us | **18.6 us** | 0.60 ms | 206 |
| cross context, 20x(1x1500x64) f16 | 17.7 us | 9.7 us | 0.31 ms | 396 |
| `split_heads` `[1,1280]`, x2 | 11.4 us | 3.4 us | 0.22 ms | - |
| self-attention, three kernels | ~12 us | ~4 us | 0.30 ms | - |
| `gelu` `[1,5120]` | 8.2 us | 0.2 us | 0.01 ms | - |

Those sum to about 6.0 ms a token against a measured 7.6, and the difference is
launch overhead in the real pipeline, which is **3.2 us a launch** on this
machine (measured over 5000 back-to-back launches of a kernel that does
nothing). A decode step is roughly 24 launches a layer and 775 a token, so
about 2.5 ms of queueing against 7.6 ms of GPU work - the CPU stays ahead, and
collapsing launches is not the lever it looks like.

Two rows are anomalies and both are recorded rather than fixed:

- **`layer_norm_add` at one row costs 11 us to move 10 KB.** It is launched one
  block per row, so at `n = 1` it is a single block on a single SM running two
  dependent tree reductions with `__syncthreads` at every level. It is 17% of
  the decode's GPU time. Shortening the tree with warp shuffles would help, and
  would change the order the mean and the variance are summed in - which is a
  reassociation in a kernel every model here shares, including the chat model
  whose agreement with llama-server is a documented 1 of 125 decisions. Not
  worth 3 ms of one ASR clip.
- **The cross-attention score product reads at 206 GB/s.** Twenty batched
  mat-vecs of one row against 1500 columns, and the shape is the problem rather
  than the code.

### Measured and rejected: caching the normalisation's row in registers

A normalisation reads its row three times - to sum it, to accumulate the
variance, and to scale it - and at 256 threads over 1280 columns that is five
values a thread, which fit in registers. Holding them, with the summation order
and the single accumulator untouched so the result is bit-identical, removes two
reads of 7.7 MB a call.

It measured **nothing**: encoder 101.8 ms to 101.9, decode 76.6 to 75.9, both
inside their own spread. The kernel was already at 535 GB/s and the re-reads
were being served out of L2, which the traffic arithmetic did not account for
and the measurement did. Reverted.

### Measured and rejected: batching the gemv's loads

`gemv` gives one warp an output column and walks the weight row lane-strided,
one load and one multiply-add at a time, so a warp has a single memory request
in flight. Fetching four words before using any of them - bit-identical, since
the summation order and the single accumulator are unchanged - was worth 7% of
the kernel in isolation (8.3 us to 7.7 us at 1x1280x1280) and **nothing at all
end to end**: the decode loop measured 76.3 ms before and 76.6 after, which is
inside its own spread. Reverted. The isolated kernel is not the critical path
it looks like when the pipeline overlaps it with everything else.

### Wave quantisation is real here, and it is measurable

Chasing the rest of the encoder turned up a ceiling worth writing down, because
it is a property of the shapes rather than of the kernel. A 128x128 tile puts
1500x1280 at 12 x 10 = 120 blocks, and two blocks a SM over 72 SMs is 144
concurrent slots - so the launch is one wave that leaves a sixth of the machine
idle. Sweeping the column count at a fixed 1500x1280 says so directly:

| shape | blocks | wave fill | TFLOP/s |
| --- | ---: | ---: | ---: |
| 1500x1280x1280 | 120 | 83% | 18.8 |
| 1500x1280x1536 | 144 | **100%** | **21.3** |
| 1500x1280x1664 | 156 | 54% | 13.4 |
| 1500x1280x1920 | 180 | 62% | 15.1 |

Throughput tracks the fill and nothing else - 1664 columns is *more* work than
1536 and takes 72% longer, because 156 blocks is one full wave plus a twelfth
of another.

That last row is also what says the fill is **not** a lever on the encoder, and
it is worth being precise about why, because the obvious reading of the table is
the wrong one. Dispatch rounds up to whole waves rather than trickling: cost is
`ceil(blocks / 144)` wave-times. A 120-block projection is therefore *already*
one wave, which is the minimum any amount of that work can take - the idle sixth
is capacity going unused, not time being spent. Batching q, k and v into one
360-block launch is three waves and so is issuing them separately, and the same
holds for overlapping them across streams. There is nothing there.

Where it does bite is `fc1`, which is 480 blocks and so pays four waves for
3.33 waves of work - about 17% of 32 ms. Recovering it needs 432 blocks or
fewer at the same total work, which means a taller row tile, which means more
accumulators, which is the register wall above. And it bites the
cross-attention cache, where 64 independent projections of 120 blocks are 64
waves apart and 7680 blocks together are 54 - a real 16%, about 2.4 ms, and the
only unspent item on this list that survives its own arithmetic.

### The round that took the ASR from 0.74x to 0.83x

Three changes, all of them outside the encoder, and none of them arithmetic.
Alternated against `whisper-server` in the same sitting throughout, so the
column on the right is a control as well as a target:

| end to end, 2.93 s clip | `xabe-asr` | `whisper-server` | ratio |
| --- | ---: | ---: | ---: |
| before | 249.4 ms | 184.2 ms | 0.74x |
| after | 222.3 ms | 185.3 ms | **0.83x** |

The three changes were measured by stage rather than end to end, because two of
them are in the frontend and one is in the decoder and an end-to-end median
cannot tell them apart. `xabe-asr-bench --stages`, medians of nine:

| stage | before | after |
| --- | ---: | ---: |
| decode loop, 10 tokens | 88.3 ms | 76.4 ms |
| mel frontend (CPU) | 18.0 ms | 10.7 ms |
| encoder | 116.8 ms | 115.2 ms |
| cross-attention KV | 19.1 ms | 19.2 ms |

The encoder and the cross-attention row are the control: nothing in this round
touched either, and neither moved.

**The cache** is the chat model's finding arrived at again. The decoder's
self-attention cache was stored the way the projection produced it,
`[t, d_model]`, and grown
by allocating a larger pair, zeroing them, copying the whole cache in and then
permuting both into head order - four allocations and six launches a layer,
128 and 192 a token, every one of them producing a tensor thrown away before
the next token. `Gpu::cache_append` already existed for exactly this and was
already scattering straight into the layout attention reads; the ASR simply was
not using it. With the cache head-major the two head splits go too, the mask
and softmax fuse into `softmax_causal`, and a single decode step's head split
and merge become no-ops. Decode loop 88.3 ms to 76.4 on ten tokens.

The comment that stood over the old code argued the permutation was cheaper
than a scattered append. It was comparing against the wrong alternative: the
append and the permutation are the same kernel, so writing the cache in the
read layout costs nothing and saves all of it.

**The frontend** is two changes, and both were found by splitting it: 10.2 ms
of the 15.7 was the transform and 5.5 was everything else.

The transform's inner butterfly indexed its twiddle table with
`tw[step * ((j * (q * m + k)) % n)]`. `n` is a runtime value, so that `%` is a
hardware division, and a 400-point transform executes a few thousand of them.
The exponent can be carried instead: `j*q*m mod n` is `((j*q) mod r) * m`
because `m * r == n`, and `j*k` never reaches `n`, so the running index stays
below `2n` and one conditional subtraction reduces it. Making `j` the innermost
loop is what allows the carry, and accumulating into the destination rather
than a scalar is what allows that - which also lets `j == 0` be a copy and
removes the zeroing pass.

The rest was the filter bank, which accumulated straight into the output:
`out[m * frames + t] += w * p` for 201 frequency bins times 80 mel bins, which
is sixteen thousand strided read-modify-writes a frame over 940 KB, none of it
staying in cache. Accumulating a contiguous 80-float row and storing it once
took that half from 5.5 ms to 1.3.

`Fft::forward_real` also allocated three `n`-long arrays per call, which is
three per frame of a spectrogram. `Fft::forward_real_with` takes a `Scratch`
the caller owns instead - not interior mutability, because the plan is shared
and has to stay `Sync` for two threads to transform different frames against
it.

### The encoder's fused attention: 117.5 ms to 112.5, and why the sweep matters

The kernel was 25 ms of the encoder at 7.3 TFLOP/s while the tiled `gemm`
beside it managed 22 on the same tensor cores. Bandwidth and occupancy had
both been tried and rejected (WHY NOT, below), which left the instruction mix:
a warp issuing `MT * NF` products for `MT` row fragments and `NF` column
fragments spends `(2*MT + NF) / (MT*NF)` shared words a product, and the fused
attention sat at 1.75 where `gemm` sits at 1.125.

Two changes, and the second is only reachable because of the first.

**The scores stay in the registers the `mma` wrote them to.** They used to go
to a `[QT][KT]` f32 tile in shared memory and be read back twice, for the row
maximum and the exponential. What warps actually have to exchange is the
reduction, not the scores: an xor butterfly over the four lanes that share a
fragment row folds a warp's own columns for free, and only one partial a row
per warp column reaches memory. `[QT][CG]` replaces `[QT][KT]` - 512 bytes for
9216 - and three passes over 9 KB a trip go with it.

**KT 64 then fits, and it is what raises the ratio.** At `hd` 64 a warp's
column fragment count is `KT / 16`: 2 at KT 32, 4 at KT 64, which is 1.5 words
a product instead of 1.75. The trip count halves too, and its barriers with it.

Three rounds alternated against the previous build, nine timed runs each:

| round | before | after |
| --- | ---: | ---: |
| 1 | 117.1 ms | 112.4 ms |
| 2 | 117.9 ms | 112.5 ms |
| 3 | 117.5 ms | 112.8 ms |

**KT 64 had been measured before and recorded here as changing nothing.** That
measurement was not wrong, it was confounded: with the score tile still in
shared memory a KT 64 block wanted 45.8 KB, which is one resident block on an
SM's 64 KB against two at 28.75 - and half the threads is about what the wider
tile gains. The two effects cancelled. **A parameter that measures flat may be
paying for itself in a currency you are not measuring**, and the currency here
was residency.

The sweep says how much residency dominates. Thirteen shapes, encoder median of
nine each, `(KT, QT, warps, row fragments a warp)`:

| shape | words per `mma` | blocks/SM | threads/SM | encoder |
| --- | ---: | ---: | ---: | ---: |
| 32, 64, 8, 1 *(the shipped one, with the score tile)* | 1.75 | 2 | 512 | 115.2 |
| 32, 64, 4, 2 *(with the score tile)* | 1.25 | 2 | 256 | 122.8 |
| 64, 64, 8, 1 | 1.50 | 2 | 512 | **111.2** |
| 64, 64, 8, 2 | 1.50 | 2 | 512 | 111.5 |
| 32, 64, 8, 1 | 1.75 | 3 | 768 | 112.5 |
| 32, 64, 8, 2 | 2.00 | 3 | 768 | 114.5 |
| 64, 32, 8, 1 | 1.50 | 3 | 768 | 115.1 |
| 32, 64, 4, 2 | 1.25 | 3 | 384 | 115.4 |
| 64, 128, 8, 2 | 1.00 | 1 | 256 | 114.4 |
| 32, 128, 8, 2 | 1.25 | 1 | 256 | 116.2 |
| 64, 64, 4, 2 | 1.00 | 2 | 256 | 120.6 |
| 128, 64, 8, 2 | — | 1 | 256 | 121.2 |
| 96, 64, 8, 1 | — | 1 | 256 | 121.8 |

Everything at one resident block is 114 ms or worse and everything at two or
three is under 116, whatever its load ratio - the two best load ratios in the
table are the third and fourth *worst* rows. Within a fixed residency the ratio
then orders things cleanly: at 768 threads, 1.75 beats 2.00 by 2.0 ms; at 512,
1.50 beats 1.75 by 1.3.

**Giving a warp two row fragments - the obvious way to raise the ratio - is not
what did it, and was dropped.** For a fixed warp count and query tile, doubling
the row fragments a warp owns doubles the warp *columns* and so halves the
column fragments, which is why `64, 64, 8, 2` measures no better than
`64, 64, 8, 1` at the same ratio and `32, 64, 8, 2` measures worse than
`32, 64, 8, 1`. Escaping that needs QT 128, which costs a resident block and
lands at 114.4. The template parameter was removed rather than shipped at 1.

Both Llama stages take the same kernel at `hd` 128 and were checked for
regression the same way - three rounds alternated, 512-token prefill on the
chat model: 3058.8, 3028.4, 3010.4 tok/s before against 3038.2, 3013.7, 3005.8
after, which is inside the drift each column shows on its own. Their tile is
unchanged; what reaches them is the smaller shared footprint, and it does not
cross a residency boundary for them.

### The round that took the ASR from 0.85x to 0.89x

Three changes, all outside the encoder, and the pattern is the same one the
fused attention showed: the frontend and the small kernels around the matmuls
had more in them than the matmuls did. `--stages`, medians of nine:

| stage | before | after |
| --- | ---: | ---: |
| mel frontend (CPU) | 11.1 ms | 5.9 ms |
| cross-attention KV | 19.5 ms | 17.0 ms |
| encoder | 111.6 ms | 110.8 ms |
| decode loop, 10 tokens | 77.9 ms | 77.5 ms |

**The frontend threads across frames, and which frames is the whole trick.**
Splitting all 3001 evenly is the obvious thing and measured *slower than one
thread* - 12.5 ms against 11.1. The window is a fixed 30 seconds, a clip is a
few, and the silent frames that cost nothing are all at the end, so an even
split hands the first thread every non-empty frame and the other seven a run of
zeros; all it buys is the spawns. Splitting the range that has signal in it,
found from the first and last non-zero sample rather than by testing each
frame, gives 5.9.

**`layer_norm_add` takes the residual sum on the way in.** Every normalisation
in a transformer block reads the stream immediately after a sub-layer added to
it and nothing between reads it, so the sum is left for the pass that was going
to be made anyway - four passes and one launch against five and two. That
removes an `add_inplace` from every block of both stacks, 1132 launches and 4.1
ms of kernel time a transcription, of which about 1.7 is net once the wider
normalisation is paid for.

**`split_heads_f16` writes the packed cross-attention cache directly**, where
the split ran at f32 and a `pack_f16` pass after it read and wrote the same 7.7
MB tensor again to change nothing but its width.

### What the remaining gap is, and what it is not

The encoder is 111 ms against `whisper.cpp`'s 83 and everything else is level:
the decode loop, the cross-attention cache and the frontend together are 100 ms
against their 102. So the gap is the encoder matmul and nothing else, and 86 of
those 111 ms are the tiled `gemm` at 22.4 TFLOP/s.

**It is not the accumulator type, and this document said for a long time that
it was.** Measured back to back out of registers on this card:

| instruction | rate |
| --- | ---: |
| `mma.m16n8k8.row.col.f32.f16.f16.f32` | 102.3 TFLOP/s |
| `mma.m16n8k8.row.col.f16.f16.f16.f16` | 103.0 TFLOP/s |
| `mma.m8n8k16.row.col.s32.s8.s8.s32` | 203 TOPS |

Flat across one, two, four and eight independent accumulator chains a thread,
so a throughput ceiling rather than a latency artefact, and 78-87% of the
card's 130.5 TFLOP/s and 261 TOPS. The half-rate f32 accumulate Turing is known
for is a GeForce restriction and this is a Quadro: **f32 accumulation costs
0.7% here.** Adopting the fp16 accumulation whisper.cpp's cuBLAS path uses
would buy under one percent, whatever anyone thinks of its accuracy, so the
refusal recorded in `docs/KERNELS.md` costs this engine nothing and the "only
way past" framing that stood here was wrong.

What is short is the staging, and **what limits it is now established.**
22.4 TFLOP/s is 22% of the instruction ceiling but 78% of what the 128x128
tile's arithmetic intensity allows against 672 GB/s - and it is not simply
bandwidth either, because rounding the activations to f16, which halves the
larger of the two operand streams, was worth 5%. Neither of those is the
binding ceiling. The register file is: `ptxas -v` puts the kernel at exactly
128 registers a thread with no spill, which is exactly `65536 / (2 * 256)`, the
budget for the two resident blocks that provide all of this architecture's
latency hiding - and 64 of the 128 are accumulators the tile cannot give up.
`ncu` still cannot be run here (`ERR_NVGPUCTRPERM` - the account has no GPU
performance-counter permission), and it turned out not to be needed.

This paragraph used to assert that register prefetch cost the second resident
block without saying by how much. It does, and the number is 184 registers
against a budget of 128: **half the throughput, 11.7 TFLOP/s against 22.5.**
Forcing two blocks back with `__launch_bounds__` spills 108 bytes a thread and
reaches 15.3; shrinking the tile until a pipeline fits reaches 18.6 at 128x64
and loses more arithmetic intensity than it recovers. Six arrangements, all
below the shipped one, tabulated in `docs/KERNELS.md`. The other standard
escapes stand as recorded there: KC 64 loses to KC 32, and wider tiles lose to
128x128.

So the honest statement of the remaining 13 ms on the shortest clip is that it
is a tiled matmul competing with cuBLAS on an architecture with no `cp.async`,
not an arithmetic or accuracy limit.

**The gap is now bounded from both ends, and it does not close.** The encoder
is 101.9 ms against `whisper.cpp`'s 83.3, and everything else in the engine is
5.6 ms *ahead* of everything else in theirs - so the whole 13 ms is the
encoder, and of the encoder's 102 ms about 74 are the tiled `gemm` and 18 are
`flash_attn`. The `gemm` is register-file bound, which is measured above. The
`flash_attn` is bounded by the same shared-memory budget its own tile sweep
already explored, which found two and three resident blocks indistinguishable.
What is left outside those two is nine milliseconds of small kernels, and three
separate attempts to take time out of them - the gemv's loads, the
normalisation's re-reads, and the software pipeline - each measured neutral or
worse, for the same reason each time: the dominant kernels are not the ones
being changed, and the caches were already covering the traffic being saved.

**What is left is smaller than it looked, and this list has been corrected
downward twice.** The frontend was on it at 5.3 ms and has been spent, for 1.7
of that. Overlapping the projections across streams was on it at about 5 ms and
is worth *nothing*, for the reason the wave table above now spells out - a
120-block launch is already one wave. What survives is batching the
cross-attention cache's 64 projections into one launch (2.4 ms), reading the
keys straight out of the projection buffer instead of permuting them (0.8 ms),
and folding the query scale into `flash_attn` (0.9 ms): **about 4 ms against
the 11 still needed.**

So the shortest clip is not reachable by any accounting available here.
Reaching parity on it needs a matmul nearer cuBLAS's rate on sm_75, and that is
the thing this architecture has been measured to refuse.

### The decoder's round: 0.94x to 0.99x on the shortest clip, from the stage the accounting called level

The paragraph above was wrong, and the way it was wrong is worth keeping. It
costed the encoder side of the pipeline to 4 ms against the 11 needed and
concluded the shortest clip was out of reach, and the 7.7 ms it did not see
were in the decode loop - which it had set aside as "level" at 100 ms against
`whisper.cpp`'s 102. Level with the other engine is not the same as done, and
the decoder turned out to be spending a launch on almost everything it did.

Two clips, `xabe-asr-bench --stages`, nine-round medians, both binaries
alternated in one sitting:

| stage | 2.93 s, 10 tokens, before | after | 7.28 s, 20 tokens, before | after |
| --- | ---: | ---: | ---: | ---: |
| mel frontend (CPU) | 3.6 ms | 4.2 | 5.5 | 7.4 |
| encoder | 105.0 | 105.0 | 105.3 | 104.9 |
| cross-attention KV | 15.3 | **13.5** | 15.3 | **13.6** |
| decode loop | 74.3 | **68.5** | 136.4 | **126.2** |

The frontend is CPU work and swings between 3.5 and 7.6 ms across runs of
either binary; nothing in it changed. The encoder did not change and did not
move. What moved is the decoder, by 0.58 ms a token at ten tokens and 0.51 at
twenty, and the cache build by 1.8 ms - against the 2.4 the accounting above
had costed for it, which is the one thing on that list that was tried.

Three changes, all of them launch count rather than arithmetic:

- **One launch for each attention.** Self-attention over the f32 cache and
  cross-attention over the packed f16 encoder cache were each a head split, a
  batched score matmul, a softmax and a value product; each is now `attn_decode`,
  reading its caches in place. `bench-attn` at the Whisper decoder's shape - 20
  heads of 64 - puts the self-attention at 0.18 ms per 32 layers where the
  chain was 0.47, and the cross-attention over 1 500 encoder positions at 0.76
  where it was 0.91. `docs/KERNELS.md` has the kernel and the register table.
- **The projections write where their output lives.** The mat-vec grew an
  epilogue that places a row: the key projection lands in the self-attention
  key cache at its position and the value projection in the transposed value
  cache, so `cache_append` and its twin are not launched at all in the decode
  loop, and the feed-forward's inner projection applies its GELU on the way out
  rather than in a kernel of its own. The test for it is exact equality against
  the three kernels it replaces, run in turn.
- **The cross-attention cache in two matmuls, not sixty-four.** Every layer's
  key and value projections of the encoder output are batched into one call
  each, and the head split that writes the packed cache folds the value bias in
  as it converts. This is the 2.4 ms item from the accounting above, measured
  at 1.8.

Nine launches a layer are gone from the decode step - two query scalings,
four of the attention chains' six, two cache appends and the GELU - which is
twenty-two down to thirteen, and at 32 layers and ten tokens 2 880 launches;
at the few microseconds each costs, that is the 5.8 ms the decode loop lost. The layer normalisation was also rewritten on warp
shuffles with one barrier in place of the two-pass reduction it had, and is
measured at nothing on this clip; it is kept because it is simpler and because
the differential test still passes at the same threshold.

What is left after this round is the same encoder - 105 ms in this sitting
against `whisper.cpp`'s 83 in the one that measured it, and the two sittings'
clocks are not the same - a cache build at 13.5 ms, and a decoder that spends
thirteen launches a layer where it spent twenty-two. The shortest clip is 1.3 ms
behind with both engines' spreads overlapping, and the next 1.3 ms is in none
of the three places this section names.

### Eight launches a layer: the decoder's second round, and the shortest clip

The next 1.3 ms was in the decoder again, and the paragraph above had named
it without noticing: thirteen launches a layer at one token, and the rule the
Llama stages' round had established - under about ten microseconds a kernel
is mostly its own floor, and the floor is paid per grid - had not yet been
applied here. Five of the thirteen were seams. The three input projections
of the self-attention were three mat-vecs over one row, each already placing
its output where the attention reads it; they are one launch over the three
weights stacked, `gemv_qkv_f16`, with the same placements in its epilogue.
And each of the three sub-layers closed with a projection followed by a
layer normalisation that was a launch reading five kilobytes; each of those
projections now carries the residual add and the normalisation in its tail,
`gemv_ln`, which is `gemv_norm`'s design with a two-pass mean-and-variance
tail because a layer norm's one-pass form cancels on a residual stream.
`docs/KERNELS.md` has both. Nothing else changed: the encoder, the cache
build and the prefix path are the same code, and `nsys` counts a decode
step at 267 launches - 32 layers of 8, and 11 outside them - where it was
427.

Two clips, `xabe-asr-bench --stages`, nine-round medians, both binaries
alternated twice in one sitting with the box held quiet by the other
sessions; both pairs of both clips agree to 0.2 ms, so one of each is given:

| stage | 2.93 s, 10 tokens, before | after | 7.28 s, 20 tokens, before | after |
| --- | ---: | ---: | ---: | ---: |
| encoder | 103.2 ms | 103.6 | 104.3 | 104.3 |
| cross-attention KV | 13.3 | 13.3 | 13.5 | 13.4 |
| decode loop | 68.8 | **64.3** | 126.9 | **118.0** |

That is 0.45 ms a token on both clips, for 160 launches a token removed:
2.8 microseconds a launch, which is the per-grid floor the Llama round
measured, on a different model with different kernels. The transcripts are
the ones the previous sitting produced, token for token: `h` lands bit for
bit and the normalised rows differ from the old kernel's by an ulp, which the
argmax never saw.

Against `whisper-server`, twenty rounds interleaved in the same sitting, the
table at the top: 185.9 against 189.4 ms on the 2.93 s clip, with the spreads
184.4-187.1 and 186.8-190.9 not overlapping. The other three clips moved by
the same decode saving and read 1.09x, 1.09x and 1.21x. The encoder is still
about 20 ms behind `whisper.cpp`'s, so the decoder is still paying that off -
now in about nine tokens rather than eleven, and the shortest clip has ten.

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

### `GEMV_MAX_M` was 16 and the crossover is 4

The mat-vec kernel gives a warp one output column and reads that column's whole
weight row, so a product of `m` rows reads **the entire weight set `m` times**.
The tiled kernel reads it once and pays a fixed cost instead. `GEMV_MAX_M` picks
between them, and it was set at 16 on the reasoning that the tiled path wants a
whole `m16n8k8` instruction's worth of rows to be worth using - which is true
about the instruction and wrong about the cost.

Prefill on the 13 B translator, across the threshold, medians of five:

| prompt | path | median |
| ---: | --- | ---: |
| 8 tok | mat-vec | 106.6 ms |
| 16 tok | mat-vec | **209.5 ms** |
| 17 tok | tiled | **68.8 ms** |
| 32 tok | tiled | 70.9 ms |

Sixteen rows cost three times what seventeen do, and the mat-vec side grows
linearly because it is re-reading 8 GB a row. Forcing the tiled path all the way
down puts it at 66.3 ms at two rows and 69.2 at sixteen - near enough flat,
because one 128-row tile covers all of them. So the mat-vec wins only while
`m * 9 ms < 31 ms` of marginal cost, which is **four rows**, and that is what
the constant now says.

`GEMV_MAX_M` is public precisely so a test that asserts the scalar path's exact
f32 can sit on the scalar side of it; the one that hard-coded 16 in a comment
saying "which is `GEMV_MAX_M`" now reads the constant.

### Where a translation's time goes

One clause in, Taigi out, on one Quadro RTX 8000 with the `Q4_K_M` file,
medians of five, greedy with `repeat_penalty` 1.1:

| clause | prompt | answer | median | per token |
| --- | ---: | ---: | ---: | ---: |
| `你好，我是台灣人。` | 24 tok | 23 tok | 399 ms | 17.33 ms |
| `今天天氣很好，我們去公園散步好嗎？` | 29 tok | 49 tok | 814 ms | 16.60 ms |
| `我很愛看花，也愛聽鳥兒唱歌，你呢？` | 33 tok | 46 tok | 767 ms | 16.68 ms |

**A translation is its decode loop and nothing else.** Timing the loop's four
parts separately - the forward pass, the logits download, the repeat penalty,
the argmax and the stop-string check - puts `forward_last` at **99.1%** of it.
The 56024-wide logits download is 0.4%, the CPU argmax over them 0.4%, and
re-decoding the whole answer every token to test two stop strings - which looks
quadratic and is - 0.003%. None of those is worth touching.

### A prefill was computing five times the rows it had

The integer matmul's block owns 128 rows of the activation and computes all of
them, `m` having that many or not. A translator's prompt is twenty-odd tokens.
So a clause's prefill was doing five sixths of its arithmetic on rows that did
not exist, and that is nearly all of what a short prefill cost - `m = 24` and
`m = 128` produce the same `mma` work and measured 69.8 ms against 90.5, so the
per-row part is small and the padded part is about 65 ms of the 70.

The row tile is a template parameter now, with a second pair of entry points at
64. Sixty-four is the floor rather than a choice: the fragment load is an
`ldmatrix .x4`, which takes four row groups at once, so a warp's share has to be
at least 32 rows and two warps cover the tile.

| prompt | wide tile, 128 | narrow tile, 64 |
| ---: | ---: | ---: |
| 24 tok | 69.8 ms | **51.1 ms** |
| 32 tok | 70.7 ms | **52.4 ms** |
| 64 tok | 75.9 ms | **57.8 ms** |
| 128 tok | 90.5 ms | 86.2 ms (unchanged path) |

Those figures carry about 35 ms of first-call cache zeroing each, so the
prefill itself went from roughly 35 ms to 16. The narrow entry costs 121-124
registers against the wide one's 128 and 19456 bytes of shared against 25600,
so it holds the same two blocks a SM.

### The decode is within 4% of the memory it has to move

At one row a decode reads every weight in the model, so its floor is bytes over
bandwidth. The `--packing f16` flag exists to separate the two, and on this
model it says:

| | streamed a token | per token | effective |
| --- | ---: | ---: | ---: |
| packed `Q4_K_M` | 8.0 GB | 15.99 ms | 579 GB/s |
| the same weights at f16 | 25.9 GB | 45.30 ms | 602 GB/s |

Both figures are the weight stream alone, with the token embedding excluded -
it is gathered a row at a time, not streamed - and with the 2.3 ms a token of
non-weight kernels taken off both sides. **Unpacking `Q4_K` costs 4%**, and
against a ceiling of 599 GB/s measured on the largest mat-vec this card runs,
that is most of what there is.

What is left is the 2.3 ms a token, which is about a seventh of the step and is
fifteen small kernels a layer: `rms_norm_q` twice (0.67 ms a token), the
three-kernel attention chain (0.54), the SiLU gate (0.16), RoPE twice (0.08),
and the cache append twice (0.03). None of them reads enough to matter and all
of them cost what a launch costs. The only lever on that is *fewer* of them.

This section holds the current numbers and nothing else. When a measurement
supersedes a cell, replace the cell — never append a dated note, a before/after
delta, or an "improved from X" narrative. The change story belongs in the commit
message; durable reasoning belongs in WHY below, and measured rejections in
WHY NOT.

## Several clauses, one weight stream: the translator's batched decode

A translation is the weight stream: 8 GB a token at 555 GB/s on the 13 B
translator, 15.3 ms a token, and the section above says why that is close to
done. What a spoken turn hands the translator, though, is not one sequence.
The reply is chunked into clauses as it streams, and the second clause is
waiting before the first is half translated; the third arrives while the
second waits. Translating them one at a time streams the weights once a
token *a clause*. Decoding them together streams the weights once a token
for all of them.

That is `Translator::step_rows`, one token for up to four sequences with
their own caches, and under it `gemv_q_rows` - the packed mat-vec over
several int8 rows with each weight byte fetched once, `docs/KERNELS.md` -
and a query offset and destination row on the decode attention, so each
sequence attends over its own cache into one shared operand. The rows are
bit for bit what the single-row kernels produce; against the single path,
which folds each normalisation into a mat-vec's tail, the batched step's
logits differ by under 1% of their span over forty layers and the greedy
choice is the same, and a batch of three sentences produced the three
translations the single path produces, character for character.

`xabe-llm-bench --kind translate --rows N`, 128-token prompts, 32 decoded
tokens, nine-round medians, one card held quiet:

| rows in the step | ms a step | tokens/s, one row | tokens/s across the rows |
| ---: | ---: | ---: | ---: |
| 1 | 15.3 | 65.3 | 65.3 |
| 2 | 17.5 | 57.2 | 114.5 |
| 3 | 21.5 | 46.5 | 139.4 |
| 4 | 25.9 | 38.6 | 154.5 |

Two rows for 1.14x the time of one, three for 1.41x, four for 1.69x. What
the extra rows cost is inside the mat-vec, not around it: the profile puts
the single step's mat-vecs at 14.4 ms and the three-row kernel at 18.1, with
the per-row attentions, rotations and the unfused normalisations adding 1.7
between them. The three-row kernel streams the same 8 GB at 442 GB/s where
the single-row one reaches 555, and what has changed is the instructions
between loads - two activation loads, two scales and eight `dp4a` a row for
every sixteen bytes of weight - so this is an issue-rate cost on a kernel
that was memory-bound at one row. Staging the activation once a block or
precomputing the per-group code sums that the `-32` bias currently spends
four of the eight `dp4a` on are the two things that would take it down, and
neither has been tried; the trade-off for a turn is settled below without
them.

### What it is worth on a turn, and the policy that came out of it

`tools/bench/turn_bench.py`, one typed turn over the WebSocket - the reply
chunked as it streams, each clause translated then synthesised - on one
card with the chat model, the translator, the ASR and Tacotron2 all
resident. The reply is three clauses of 15, 12 and 30 characters, the same
every run; four runs each, medians. `--translate-ahead 0` translates the
clauses one at a time in step with the synthesiser, which is what the
previous binary did and what the batched thread does with one clause in it;
`--translate-ahead 1` hands every clause to the translator as it is cut.

| | first audio | whole turn | clause 1 translated | clause 3 translated |
| --- | ---: | ---: | ---: | ---: |
| one at a time | **1 488 ms** | 3 880 ms | 866 ms | 1 172 ms |
| every clause as it arrives | 1 654 ms | **3 613 ms** | 1 020 ms | 1 310 ms |

The whole turn is 267 ms shorter and the first audio 166 ms later, and both
for the same reason: the first clause was sharing its steps. Its translation
went from 866 ms to 1 020 with the second and third clauses decoding beside
it, and the first clause is the one the listener is waiting through. The
third clause is the other half of the story - 30 characters, 1.2 s alone,
and it arrives last, so its translation plus its synthesis is the tail of
the turn whichever way the clauses are scheduled, and batching cannot
shorten a single long clause. Synthesis of the short second clause also
went from 129 ms to 380 with the translator decoding beside it on the same
card, which is the contention `translate_ahead` was introduced to avoid,
now off the critical path.

So the first clause of a turn is translated alone, and every clause after
it is handed over as it arrives. That was tried next, and it lost on both
counts - first audio 1 918 ms, the whole turn 3 949 - for a reason the
first attempt had hidden: with the first clause finished, the second and
third decode *beside its synthesis*, and on one card two GPU jobs do not
run in half the time. Clause 1's synthesis went from 380 ms to 780 with the
translator streaming next to it, and every later stage paid the same.

The card, not the translator, is the contended resource on one card. So a
translator and a synthesiser on one device now take turns on it: synthesis
holds the card while it runs, and the batched translator - which decodes
several clauses a step and loses nothing by pausing a few hundred
milliseconds between steps - steps only while it is free (`xabe-engine`'s
`card` module; the synthesiser never waits, so there is nothing to
deadlock). With that, the same turn and the same four-run medians, and then
a five-clause reply of 84 tokens, three runs:

| one card, synthesis holding it | first audio | whole turn |
| --- | ---: | ---: |
| three clauses, in step | 1 498 ms | 3 908 ms |
| three clauses, first alone then batched | 1 563 ms | 3 781 ms |
| five clauses, in step | 1 664 ms | 7 118 ms |
| five clauses, first alone then batched | 1 527 ms | 7 128 ms |

Inside the run-to-run spread on both counts, both replies. The reason is
worth keeping: on one card the tail of a turn is the *last* clause, which
is cut last and is often the longest, and its translation plus its
synthesis is the critical path whichever way the earlier clauses are
scheduled. Batching lengthens that clause's own steps by the 1.14-1.41x
above and cannot start it any sooner, so what it gains on the earlier
clauses it gives back on the last. The 84-token reply's last clause was 49
characters: 1 954 ms alone, 2 788-3 063 ms batched behind three others.

**With the translator on its own card it is a different turn.** Chat model,
ASR and Tacotron2 on card 0, the translator on card 1, the three-clause
reply, three runs each:

| translator on card 1 | first audio | whole turn |
| --- | ---: | ---: |
| in step (`--translate-ahead 0`) | 1 136 ms | 3 445 ms |
| first alone, then batched (`--translate-ahead 1`) | 1 145 ms | **2 816 ms** |

18% off the whole turn and first audio unmoved, which is what the batch
was built for: the second and third clauses decode together the moment the
first is done, nothing is contending with synthesis, and the clause the
listener is waiting on keeps its own steps. (The first clause also
translates in 643 ms here against 855 on the shared card, where the chat
model is still writing beside it.)

So the defaults stand where they were - in step on one card, ahead on two -
and what changed is what "ahead" does: every clause after the first is
handed over as it is cut and decoded together, rather than one clause of
overlap. `--translate-ahead` is an explicit flag as well as the
device-derived default, for a shared card that turns out to have the room,
and the shared-card priority applies whenever the two stages resolve to one
device, whichever way the flag is set.

## First speech, with llama.cpp behind both Llama stages

The table below says llama.cpp is ahead on both stages. This is what that is
worth to a listener. Same card, same checkpoints, `--llm-url` and
`--translator-url` pointed at two `llama-server` processes on 127.0.0.1 with
everything else - ASR, VAD, the synthesisers, the socket - still in this engine.

| one card, three-clause turn | in-process | llama.cpp behind both |
| --- | ---: | ---: |
| first audio | 1798 ms | **1400 ms** |
| whole turn | 5638 ms | **3525 ms** |
| resident | 22.4 GB | 21.7 GB |

**1.28x to first speech.** Medians of seven runs against three; the turn total is
the softer of the two numbers because the two chat models sample differently and
the replies are not the same length, where first audio is one clause either way.
Per clause, translation went from 1091-1164 ms to 798-859.

It costs a process boundary and two more things to supervise, and it wins. The
in-process stages stay - they are what the oracle tests compare against, and
they are what makes the packed-weight work measurable at all - but nothing here
should claim the pipeline is fastest with them in the reply path, because it is
not.

## Against llama.cpp: level or ahead on every row

`llama-bench` on the same card and the same two files, `-ngl 99`, against
`xabe-llm-bench` at the same shapes.

The protocol got stricter than one sitting, because the llama.cpp column will
not hold still across sittings - see below. **Three alternated rounds**, each
round one `llama-bench -r 9` followed by one `xabe-llm-bench --rounds 9`, both
models, both prompt lengths, on an otherwise idle card. Each cell is the
median of its three rounds, and the spread beside a llama.cpp figure is
`llama-bench`'s own, from the median round.

**The repeat count is part of the measurement.** At five rounds this engine
once reported 2760 tok/s on a cell where nine rounds said 2677, because the
first rounds run on a boosted clock. Every engine figure below is a nine-round
one.

| Checkpoint | | llama.cpp | this engine | |
| --- | --- | ---: | ---: | ---: |
| Breeze2 8 B Q4_K_M | prefill, 128 tok | 2259 +/- 46 | **2447 tok/s** | 1.08x |
| | prefill, 512 tok | 2513 +/- 91 | **2928 tok/s** | 1.17x |
| | decode, 64 tok | 95.3 +/- 0.3 | **100.9 tok/s** | 1.06x |
| Taigi 13 B Q4_K_M | prefill, 128 tok | **1428 +/- 34** | 1339 tok/s | 0.94x |
| | prefill, 512 tok | 1647 +/- 7 | 1636 tok/s | level |
| | decode, 64 tok | 60.0 +/- 1.4 | **61.4 tok/s** | 1.02x |

**The translator's 128-token row is the one cell not won, and it is also the
one where llama.cpp cannot repeat its own number.** Its three nine-round
medians were 1453, 1428 and 1161 - a 20% span between full `-r 9` runs minutes
apart - while the engine's three were 1339.4, 1336.7 and 1341.6. The 0.94x is
against the middle of a swing the engine's whole spread fits inside three
times over. The claim this table makes for that cell is "inside llama.cpp's
own noise", and nothing stronger; the claim it declines to make is that 1428
is what llama.cpp reliably runs at.

**Why alternated rounds, and why the llama.cpp column moved.** The previous
version of this table, one sitting on the same card and the same files,
recorded llama.cpp at 2263/2835 on the chat model, 1401/1560 on the
translator, and 100.8/61.2 decode. Re-measured for this table its chat prefill
is down 11% at 512 tokens, its translator prefill is up 6%, and both decodes
are down 5% - every one of those a larger move than most rows' spread. The
engine's own figures repeat across sittings to about 1%. So the only
comparison this file trusts is the two tools interleaved in one sitting, and
the verdicts above come from that and nothing older.

Both prefill columns are the same measurement: the engine's `forward` projects
one row through the output head, which is what `llama-bench`'s pp does. The
engine's decode-64 is taken after a 128-token prompt where `llama-bench`'s
tg64 generates from an empty context, a difference that can only flatter the
llama.cpp side.

What moved the engine column from 2341/2677/1306/1438 - level at one prompt
length, 0.92x to 0.94x at the other three - to the numbers above is the round
below: "The round that closed prefill".

Decode was 60.8 and 35.8 when this table was first written, so the two stages
are 1.66x and 1.71x faster than that. **Read the decode margins carefully:
the engine did not get faster this round.** It reads 100.9 and 61.4 where the
previous sitting said 101.3 and 61.7 - unchanged within noise, which is what
the prefill round had to preserve and did. The 1.06x and 1.02x exist because
llama.cpp's own decode came in 5% and 2% lower this sitting than last. A
margin that appears when the other column moves is reported as what it is.

The translator's last 10% was not a kernel. A warp of the wide mat-vec covers
four super-blocks a trip, and the path was gated on `k` being a multiple of
1024 - four of them. 13 B Llama-2 contracts its down projection over 13824,
which is 54 super-blocks, so **22% of every layer's weights fell silently onto
the four-byte path**. Letting the lanes past the end of a row sit out is exact,
because their contribution is a separate term of the warp reduction, and it is
worth 55.4 to 61.2 tok/s.

Decode is bandwidth, and the weights are now read at close to the roof: the chat
model's `gemv` moves 5.0 GB per token in 8.7 ms, which is 567 GB/s against a
streaming ceiling of 587 measured on this card. That is 97%, and it is why the
last two thirds of this work were spent on everything that is *not* the matmul.

The earlier note here said llama.cpp's advantage was structural - `mmvq.cu`
splitting one output row across a thread block, because on Turing a thread with
too few k-iterations "cannot keep enough loads in flight to reach the bandwidth
roof". Half of that reading was right and half was wrong. The diagnosis was
right: loads in flight was the constraint. The prescription was not - a
block-per-row shape was implemented and measured worse (WHY NOT below), and what
actually closed the gap was widening each thread's load from 4 bytes to 16, which
needs the *activation* narrow enough to keep up. See "Sixteen bytes a lane"
below.

Prefill is arithmetic, and it was the larger gap for most of this file's
history - 3.5x behind at its worst. Three rounds of work closed it: the f16
tiling round ("Prefill: what four changes bought"), the move to the integer
tensor cores ("Prefill on the integer tensor cores"), and the round after
that, which is the one that ended level-or-ahead and comes next.

## The round that closed prefill

Where the last section's table came from. The starting point is the previous
table's engine column - 2341 and 2677 tok/s on the chat model at 128 and 512
tokens, 1306 and 1438 on the translator - against a llama.cpp lead of 6% to
8% everywhere but chat-128. Four changes closed it, and the widest cell moved
14%. Single runs at each step, all four cells, taken with the same binary
back to back; single runs read a few percent above the nine-round settled
figures, so the rows compare to each other and the settled protocol table
above is the claim:

| | chat 128 | chat 512 | trans 128 | trans 512 |
| --- | ---: | ---: | ---: | ---: |
| session start, nine-round figures | 2341 | 2677 | 1306 | 1438 |
| Q6_K device layout, scale fold, two-regime split | 2350 | 2741 | 1295 | 1490 |
| the int8 twin fused into three more writers | 2362 | 2759 | 1307 | 1513 |
| the memory pool told to keep its pages | 2379 | 2772 | 1328 | 1564 |
| attention fused into one kernel | 2488 | 2989 | 1355 | 1697 |

### A stale binary, and three changes measured as nothing

The first row's gain was nearly written off, because the first three changes
were each measured at zero effect - 1412.8, then 1401.6, 1403.5, 1401.9
tok/s on the translator's 512-token cell. All three measurements were of an
unchanged binary: `cargo build --release -p xabe-cuda` rebuilds the library,
and `xabe-llm-bench` lives in `xabe-engine` and does not get relinked. The
kernel source is a compile-time string, so nothing failed - the old string
ran, correctly, at the old speed. Rebuilding the workspace surfaced all three
changes at once, which is why the table's first row is one combined step: the
per-change split was never validly measured, and this file does not invent
it. The rule this buys: **measure with a binary the build just wrote, and if
a kernel change measures as exactly nothing, suspect the link before the
theory.**

The three changes in that row, design in `docs/KERNELS.md`:

- **Q6_K is restrided at upload** into a device layout whose low nibbles pair
  32 elements apart, Q4_K-style, and whose high bits land one 16-element run
  per word - one shift against the file layout's eight-way byte gather.
- **Its two sub-scales fold into integer arithmetic** - `sc0*dot0 + sc1*dot1`
  is exact in int32, bounded under 2^24 - so the kernel converts once per
  sub-block pair instead of twice, on a card whose I2F runs at quarter rate.
- **The split-k rule grew a second regime.** The old rule split only
  projections too small to fill the card. The translator's wide projections
  are the opposite shape: 160 or 216 blocks against 144 slots, one full wave
  and a straggler tail that leaves most of the card idle for half the time.
  The rule now measures that idle fraction and splits when it exceeds 0.3,
  taking the largest split in 2..4 that keeps 2048 of contraction per slice.
  The earlier sweep's conclusion that the old rule "was already at the
  optimum" was true of the constants it swept and not of the rule shape; the
  5% the sweep found and declined is taken here without a constant that hurts
  the other cells. `nsys` puts the Q6_K projections at 47.2 ms of a
  translator-512 prefill where they had been 65.2.

### Three more writers for the int8 twin

The second row is `quantize_q8` almost disappearing from the profile. It ran
once per projection group when the activation was produced by a kernel with
no quantizing tail. Three got one: `silu_mul` already quantized for decode
and now does for prefill whenever the down projection is packed (the gate
was a row-count check left over from before the tiled kernel read codes),
`merge_heads` quantizes the merged context on its way out for the o
projection, and growing the KV cache no longer copies at all when there is
no past to copy. Each fused tail has a differential test against
`quantize_activation` at exact equality - same codes, same sums.

### The memory pool told to keep its pages

The third row is not a kernel and was found with a standalone benchmark that
refused to go below 0.9 ms however small the work got. CUDA's default async
memory pool has a release threshold of zero: every synchronize returns the
pool's free pages to the driver, so every buffer allocated after a
synchronize is a real allocation, at real-allocation cost, on every token
and every prefill forever. One attribute at device open -
`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` to `u64::MAX` - lets the pool keep what
it has been given, and is worth 0.5% to 3.3% depending on the cell, the
translator's 512-token cell most. It also retroactively explained a phantom
cliff at large split factors in the ksplit sweep - more slices means a bigger
partial buffer, and the "kernel time" of every micro-benchmark that allocated
after a synchronize was part allocator. The two-regime rule's in-model gains
were re-confirmed with the pool fixed; the raw sweep numbers were not
re-taken and are read with that caveat.

### Attention fused into one kernel

The last row is the largest single step and the only new kernel of the
round: `flash_attn`, an online-softmax fused attention for prefill, design
and constraints in `docs/KERNELS.md`. What it replaces, per layer per
prefill: `split_heads`, the batched QK^T against every past position, the
causal softmax materialising the full score matrix, the batched PV product,
`merge_heads`, and on the chat model `repeat_kv` - six launches and two
score-matrix round trips through global memory become one kernel that never
writes a score anywhere. `nsys` had the unfused chain at 27 ms of a
translator-512 prefill; the fused kernel reads the KV cache in place, takes
Q straight from the projection buffer, and hands the merged context to the o
projection. It refuses any head width but 128 rather than silently indexing
across heads, and the model falls back to the unfused chain - which is why
that chain is still here and still tested. Its differential test drives
peaked and flat softmax rows against a scalar reference that rounds where
the unfused chain rounds.

What it does *not* change: decode. A single-token step attends one query
row; the fused kernel's 32-row tile would run 31/32 empty, so decode keeps
the mat-vec chain, and the decode rows above moved by nothing - 100.9 and
61.4 against 101.3 and 61.7 the sitting before.

### WHY NOT, again

Implemented, measured, reverted, same as every entry in the older lists:

**Stream-K, the way llama.cpp schedules the same tensor cores.** Fixed
blocks, each walking a contiguous span of tile-trips, partial tiles reduced
through a scratch buffer with a fixup pass - no wave quantization by
construction, and llama.cpp's `mul_mat_q` does it this way on this card. Built,
correct, and slower than the two-regime split rule at nearly every shape:
0.308 against 0.233 ms on the translator's 128-token o/v shape, 0.582
against 0.474 on the chat model's 512-token o shape (and before the pool
fix it had measured 0.867 - the scratch buffer was being really allocated
every launch, which is the pool finding above). The arithmetic of why: at
these shapes a block's span is 56-60 trips while a whole tile is 64-216, so
nearly every tile is split across blocks and takes the scratch-and-fixup
path that stream-K is supposed to reserve for stragglers. llama.cpp's tiles
are shaped so a span usually covers whole tiles; this kernel's are not, and
reshaping them is the f16 sweep's conclusion again - 128x128 won for
arithmetic-intensity reasons that have not changed.

**A format-aware slice floor,** raising `KSLICE_MIN` to 4096 for Q6_K on the
theory that its heavier staging amortises worse across slices: -1% on the
model (1282 against 1294 on the translator's 128-token cell), inside noise
everywhere else, reverted.

## Prefill on the integer tensor cores

The f16 kernel ran out of room - or so this said. At 1836 tok/s it was measured
at 86% of its own ceiling, 32 ms of `mma` against a believed 65.3 TFLOP/s for
`m16n8k8.f32.f16.f16.f32`, with llama.cpp 1.32x ahead. **The 65.3 was wrong.**
Measured back to back out of registers the instruction runs at 102.3 TFLOP/s on
this card, so the same 32 ms of `mma` is about 20 and the fraction is 55%, not
86%. `docs/KERNELS.md` has the microbenchmark and the two other numbers it
corrects.

The decision survives the correction and the numbers below are unaffected: the
integer shape is twice the f16 rate rather than four times, which is still the
largest arithmetic lever this card has, and llama.cpp uses it. What does not
survive is "no amount of work on the staging reaches that" - the staging had
more room in it than was recorded, and nobody has established how much of that
is reachable.

**This is the engine's second deliberate approximation and a larger one than the
first.** The f16 path rounds a weight that was already four bits and keeps the
arithmetic exact to f32 accumulation. This one quantizes the activation as well.
What that costs is measured below rather than argued about.

Chat model, `--prompt 512`, single runs at each step, prefill tok/s:

| | chat prefill |
| --- | ---: |
| f16 tiled `gemm`, the previous best | 1926 |
| `gemm_i8`, activation quantized inside the kernel | 640 |
| reading `Gpu::quantize_activation`'s codes instead | 1634 |
| `ldmatrix` for the fragments, a 2x4 warp grid, scales in pairs | 1657 |
| the block header cached per staging thread | 1869 |
| both `mma` steps of a Q4_K sub-block into one accumulator | 1980 |
| the row tile on `blockIdx.x`, so weight-sharing blocks co-reside | 1993 |
| 64 of contraction a trip, using both nibbles of every byte read | 2292 |
| the int8 twin shared across a block's five projections | 2305 |
| one kernel per block format instead of a runtime branch | 2666 |
| back to a 128x128 tile at two blocks an SM | 2766 |
| the split-k partial buffer left uninitialised | 2702-2766 |

The last two rows are inside the run-to-run spread of this benchmark, which is
about 2% at 512 tokens; the settled figures are the three-repeat ones in the
table above. Everything before them is larger than that spread.

### The two that mattered most were not the tensor cores

**Quantizing the activation inside the kernel cost 3x.** The first version
computed the maximum, the reciprocal and the rounding for its own 128 rows on
every trip - which is once per column tile, 32 times over on a 4096-wide
projection, and the same numbers every time. `Gpu::quantize_activation` already
existed for the mat-vec's fast path. Reading its output instead turned the
activation staging into a copy: 640 to 1634 tok/s.

**One kernel per block format was worth 20%.** Q4_K and Q6_K stage completely
differently, and a Q4_K sub-block holds one scale across all 32 elements where
Q6_K changes scale every 16 - so one merges its two `mma` steps into a single
integer accumulator and the other cannot. As a runtime branch on a kernel
argument, ptxas allocated registers for the union of both and scheduled for
neither, and spilled eight bytes doing it. As a template parameter with two
`extern "C"` entry points: 2305 to 2666, no spills.

### Where the time was, and what it is not

The A/B is the same one the f16 work used: delete a piece, accept wrong results,
time it. Taken at 222.2 ms per 512-token prefill, after the 64-element trip and
before the format split:

| piece | ms | share |
| --- | ---: | ---: |
| staging, all of it | 68.5 | 31% |
| - the weight's global read and scale decode | 28 | 13% |
| - the activation's global read | 25.3 | 11% |
| - the code sums, `dp4a` against ones | 2.1 | 1% |
| converting int32 sums to f32 and scaling them | 29 | 13% |
| `mma` | ~42 | 19% |
| everything else, including the other kernels | ~50 | 22% |

Re-taken later on the **13 B translator** at 512 tokens, where the gap was then
widest, against a 352.6 ms baseline - the shares are not the same model's:

| piece | ms | share |
| --- | ---: | ---: |
| staging, all of it | 110.9 | 31% |
| - the activation's global read | 55.8 | 16% |
| - the weight's global read and scale decode | 40.7 | 12% |
| - the shared stores | 17.5 | 5% |
| converting int32 sums to f32 and scaling them | 73.4 | 21% |
| the two `__syncthreads()` a trip | 64.2 | 18% |

The conversion is a fifth of this kernel here against an eighth on the chat
model, and the reason is Q6_K: it changes scale every sixteen weights where
Q4_K changes every thirty-two, so it converts twice a sub-block where Q4_K
converts once, and it is 15.3% of these weights. The barrier row is measured by
replacing both with `__syncwarp` - wrong results, timing only - and the entry
above explains why that number is not headroom.

The conversion is the floor this design has. Per output per 32-element
sub-block it is one `I2F` and three multiply-adds against one `mma`, and that
ratio does not change with the tile: it is set by how often the scales change,
which is a property of Q4_K and not of the kernel. Making it cheaper means
quantizing the activation in groups of 256 rather than 32 so the scale factors
out of the contraction, which is a real accuracy loss for about 3%, and it was
not taken.

### WHY NOT, this round

Every one of these was implemented, measured against the run beside it, and
reverted.

| | measured | against |
| --- | ---: | ---: |
| `__launch_bounds__(256, 2)` to force two blocks an SM at 152 registers | 1624 | 1634 |
| a 256-row tile, 16 warps, one block an SM | 2568 | 2766 |
| a 256-column tile, 16 warps, one block an SM | 2666 | 2766 |
| a 64-column tile at three blocks an SM (512 tok) | 2188 | 2766 |
| a 64-column tile at three blocks an SM (128 tok) | 1823 | 2261 |
| a 4x2 warp grid - 4 rows, 8 columns a warp | 2068 | 2766 |
| `SM_TARGET` 288 for the split (128 tok) | 1953 | 2261 |
| `SM_TARGET` 216 with 256-element slices (128 tok) | 2159 | 2261 |
| `SM_TARGET` 108 / 72 / 36 (128 tok) | 1875 / 2251 / 1796 | 2261 |
| rounding the split up rather than down (128 tok) | 1986 | 2261 |
| software-pipelined loads, 2 blocks an SM (translator 512) | 1213 | 1438 |
| the same at 1 block an SM, 214 registers, no spills | 1273 | 1438 |
| the packed blocks transposed to `[k-block][n]` (chat 512) | 2597 | 2677 |
| the same (translator 512) | 1389 | 1438 |
| the same (chat 128) | 2370 | 2341 |
| the same (translator 128) | 1341 | 1306 |

The occupancy one is worth stating plainly because it was the first guess and
it was wrong. The kernel used 152 registers, which is one block an SM, and
forcing 128 to get two changed nothing - 1624 against 1634. It was never
occupancy-bound. The tile sweeps say the same thing the f16 sweep did: 128x128
wins, and it wins for the arithmetic-intensity reason rather than by accident.

**The pipelining result is the one that changed how to read the profile.** A
fresh decomposition of the translator's 512-token prefill said the two
`__syncthreads()` a trip were worth 64 ms of 353 - 18% - and the two global
reads another 96 ms. The standard answer is to issue the next trip's loads
right after the barrier and let them land in registers while the tensor cores
work on the trip already staged. That was built, and it is 16% *slower*.

Twenty-one registers of staging is what it costs, and at two blocks an SM there
is no room for them: 80 bytes of spill, 1213 tok/s against 1438. Given the room
- one block an SM, 214 registers, no spill at all - it is still slower, 1273.
Which answers the question the profile could not: **the second resident block
was already covering those barriers.** Eight warps pipelined by hand beat
nothing; sixteen warps interleaved by the scheduler beat both. The 64 ms is
what a barrier costs when you delete it and keep everything else, not headroom
sitting there to be recovered.

The block transpose is the interesting one because it is a *split* result, and
the reason is locality in two directions at once. A staging thread reads sixteen
bytes of one output channel's super-block, and consecutive threads are
consecutive channels - which in the file's own layout are `k/256 * 144` bytes
apart, 2880 on the translator. Transposing to `[k-block][n]` puts them 144
apart, which is why both 128-token figures improve. But it also puts the *next*
trip's bytes for the same channel `n * 144` away instead of adjacent, and a
512-token prefill runs four times as many trips per weight read. The trips win.
Neither layout is right for both, and a layout that is right for both means
reordering inside the blocks as well - the shuffle llama.cpp's MMQ does at load,
which is a larger change than this file has measured.

The split-k rule was swept in both directions, at both prompt lengths and on
both models, and the existing one was at the optimum *of its own shape* - no
constant beat its constants, which was not obvious, since it was tuned for the
f16 kernel and the integer one only happens to share its tile. A later round
changed the rule's shape rather than its constants and won - the two-regime
rule in "The round that closed prefill" above.

The sweep did find one thing worth writing down, because it is a real 5% that
was deliberately not taken. At 512 tokens the two models want *opposite*
things:

| | chat 128 | trans 128 | chat 512 | trans 512 |
| --- | ---: | ---: | ---: | ---: |
| `SM_TARGET` 144, `KSLICE_MIN` 512 | **2360** | **1294** | **2760** | 1395 |
| `SM_TARGET` 432, `KSLICE_MIN` 2048 | 2032 | 1197 | 2665 | **1471** |
| `SM_TARGET` 432, `KSLICE_MIN` 1024 | 2145 | 1201 | 2637 | 1459 |
| `SM_TARGET` 288, `KSLICE_MIN` 2048 | 2028 | 1197 | 2668 | 1407 |

The translator's 512-token prefill gains 5.4% from splitting more, and every
other column loses - up to 14%. The cause is wave quantization and it is
visible in the block counts: the translator's 5120-wide projections are
40 x 4 = 160 blocks against 144 concurrent slots, which is one full wave plus
sixteen stragglers, while the chat model's 4096-wide ones are 128 blocks and
already under one wave. A constant that splits the first also splits the
second, and the second does not want it.

**Fitting a rule to two models is fitting a rule to two points.** The gain is
real and shape-dependent, and the way to take it is to stop the shape from
arising rather than to tune a constant around it. That is the next section, and
it recovers the same 5% without a constant at all.

### The projections, grouped

q, k and v multiply the same normalised activation, so they can be one batched
product instead of three launches - and then the shape that wanted more
splitting does not arise. The translator's 5120-wide projections go from three
lots of 160 blocks to one lot of 480: 3.3 waves instead of three times 1.11,
with no partial buffer and no reduction pass.

Batching needs identical `(in_dim, out_dim, block format)`, and which of the
three qualify is a property of the checkpoint rather than a choice:

- Q4_K_M stores `attn_v` as **Q6_K in half the layers** of both models, so
  half of them fuse all three and half fuse what they can.
- A grouped-query model gives `attn_q` a different width from `attn_k`, so the
  8 B chat model fuses k with v and leaves q alone, where the multi-head 13 B
  translator fuses q with k.

So the grouping is computed per layer from the shapes and formats, and a layer
whose three disagree runs three products, exactly as before.

Nothing is copied to take the three apart afterwards. Each element of a batched
product writes a contiguous block of one output, so `q` is that output's prefix
and `k` and `v` are located by an offset - which is why `rope` and
`cache_append` grew one, and why `gemv` and `gemm_i8` grew a row stride for the
activation. A zero left-operand stride now means *shared*, and the activation is
quantized once for the group instead of once a projection.

Measured against the same build with the grouping disabled, three runs of nine
rounds each:

| | grouping off | grouping on | |
| --- | ---: | ---: | ---: |
| chat, 128 tok | 2345 | 2341 | level |
| chat, 512 tok | 2665 | 2677 | +0.5% |
| translator, 128 tok | 1286 | 1306 | +1.6% |
| translator, 512 tok | 1384 | **1438** | **+3.9%** |
| translator, decode | 60.9 | **61.7** | +1.3% |

The translator gets what the split-k sweep found and the chat model does not,
and both are what the block counts predict: the translator's three projections
are the same width so all three fuse, while the chat model's q is four times
its k and only k and v can go together - which are its two *small* projections
and were never the ones leaving the machine idle.

The decode row is the one that was not predicted. A single-token step is
launch-bound, and grouping removes two launches a layer.

### What it costs in accuracy

Every number here is measured, and the comparison that matters is at the end.

| | |
| --- | --- |
| kernel against a CPU twin of the same approximation | passes at 1e-5 relative |
| chat, packed against f16 in this engine | 0.175 of a 25.32 logit span (0.69%) |
| translator, packed against f16 | 0.121 of a 28.66 span (0.42%) |
| chat against llama-server on f16 | 1 of 125 decisions, margin 0.056 |
| | 7 of 8 replies identical |
| chat block sums against llama.cpp on the same Q4_K file | worst 0.099 of the layer magnitude |

**The llama-server agreement did not move.** It was 1 of 125 at margin 0.056
and 7 of 8 replies before this work and it is the same after, which is the
result that mattered: the integer path is a different arithmetic, not a worse
model.

### Whose integer path is closer

"Different from llama.cpp" is not the same as "worse than llama.cpp", and the
difference is measurable. `llama-eval-callback` prints a scalar sum per graph
node, so both engines can be read at every block output on the *same* Q4_K file,
and both can be compared against that file's weights computed more precisely -
dequantized and run through this engine's f16 path, which rounds operands but
quantizes no activation.

Mean absolute difference from that reference over 31 block outputs, prompt
`hi`:

| | mean \|block sum - the same weights at f16\| |
| --- | ---: |
| this engine, `gemm_i8` | **0.234** |
| llama.cpp CUDA, MMQ | 0.337 |
| llama.cpp CPU, `-ngl 0` | 2.048 |

So on this file the engine's integer matmul sits closer to the exact
computation than llama.cpp's does, and both sit far closer than llama.cpp's own
CPU backend - which quantizes activations in groups of 256 where the two CUDA
paths use 32. That last row is the useful calibration: the spread between
llama.cpp's own two backends is six times the spread between the two CUDA
implementations.

One caveat, stated rather than buried: the reference is this engine's f16 path,
so it shares an implementation with the row being judged. It is a different
computation - f16 tensor cores, f32 accumulation, no activation quantization -
but it is not a neutral third party. The ordering it produces is independently
sensible, which is the only reason it is quoted.

Against `llama.cpp`'s f16 build the picture is different and it is not an
arithmetic result: this engine is 9.402 from it, llama.cpp CUDA on Q4_K is
9.298, and llama.cpp CPU on Q4_K is 7.618. All three are dominated by the same
thing - the weights really are four bits - and the 0.1 between the first two is
the whole arithmetic disagreement.

## Prefill: what four changes bought, and what two did not

Prefill went from 553 to 1414 tok/s on the chat model and 400 to 841 on the
translator. The route was one diagnostic repeated at each step: delete the
staging from the tiled `gemm` - wrong results, timing only - and time the mma
loop on its own. At the start that said 547 against 3344; at the end, 1414
against 3103. Everything below is the distance between those two numbers.

| | chat prefill | translator |
| --- | ---: | ---: |
| start | 553 | 400 |
| split the contraction across blocks | 926 | 578 |
| round the activation to f16 once | 982 | 632 |
| read K-quant nibbles by the word | 1198 | 733 |
| stage in quads, unpack sixteen a call | 1355 | 832 |
| project only the row that predicts | 1414 | 841 |
| `ldmatrix` instead of scalar shared loads | 1469 | 889 |
| one load for a Q4_K header instead of eight | 1836 | 1025 |
| *the same kernel with no staging at all* | *3341* | - |

**The tile was not the problem, and a broken measurement said it was.** The
first thing tried was a tile sweep, and it reported 1441 tok/s at 32x64 against
551 at 128x128. That was wrong: the launch geometry was written `div_ceil(128)`
in `device.rs` beside a tile that is a `#define` in `kernels.rs`, so every
non-128 tile launched a grid that covered a fraction of the output and the
kernel "ran" faster by not computing most of it. The ASR oracle caught it; the
LLM benchmark checks no numbers and would not have. `kernels::define` now reads
the tile out of the CUDA source at compile time so there is one copy. Swept
again with a correct grid, 128x128x32 is the best tile on every shape measured,
which is what the arithmetic-intensity derivation in `docs/KERNELS.md` said
before the sweep started.

**What was actually wrong was that one tile of `m` is the whole prefill.** A
128-token prompt at `GEMM_MT = 128` is one row of blocks, so a 1024-wide
projection was eight blocks on 72 SMs. Shrinking the tile to make more blocks
makes each weight dequantized once per block instead of once, which is why the
corrected sweep hates it. Splitting `k` instead adds blocks without that
redundancy. That was the largest single change, 1.67x.

**The rest was instruction count in the staging loop**, in three parts: the
activation was staged from f32 and re-read once per column tile though the
kernel rounds it to f16 anyway; the K-quant nibbles were read one byte at a
time, one global load per element produced; and the activation quads and the
K-quant runs were both narrower than the alignment allowed. None of these is
clever. All three were sitting in the hottest loop in the engine.

### Where the last two came from, and where the time is now

Both came out of the same diagnostic run twice: delete one half of the staging
and time what is left. That said the *weights* were 40 ms of an 88 ms prefill
and the activations 13, and an earlier test had already put the
dequantisation arithmetic at 3 ms. So 37 ms was neither arithmetic nor the
activation - it was how the weight header was read.

A Q4_K block opens with `d`, `dmin` and twelve packed scale bytes: exactly
sixteen, and 16-byte aligned. Reading them cost eight separate byte loads a
run, scattered 144 bytes apart across a warp. One `uint4` is the whole header,
and that alone was 1469 to 1836 tok/s.

`ldmatrix` is the other. The mma loop issued 72 `ld.shared` a lane per staged
trip against 64 `mma` - each lane fetching its own 32 bits of a tile the
hardware will fetch whole - and `ldmatrix` takes that to 40.

Split again at the end, a 69 ms prefill is now roughly 38 ms of mma loop, 21
of weight staging and 9 of activation staging. The mma loop is the largest
piece and is about 72% of what this card can issue: 1.8 TFLOP of prefill
against 512 f32-accumulate FLOP a cycle on 72 SMs at 1.77 GHz is a 27.6 ms
floor, and the loop with all staging deleted measures 38.

### The two that did not work, and why

Both are the textbook answer to "the staging and the mma never overlap, because
a `__syncthreads()` separates them in each direction". Both lose on this card,
and they lose for the same reason.

- **Register prefetch.** Fetch trip i+1 into registers before the mma loop, so
  the global loads issue underneath the tensor instructions. Registers went 117
  to 158, which drops residency from two blocks an SM to one, and prefill went
  940 to 687. Pinning `__launch_bounds__(..., 2)` capped it at 128 registers
  and spilled 12 bytes, which gave 935 - a wash. Prefetching only the weights
  was worse still (171 registers, 699).
- **Shared double buffering.** Stage trip i+1 into a second pair of buffers
  while the mma reads the first, one barrier a trip instead of two. No register
  cost, but 40960 bytes of shared memory a block against the SM's 64 KB, so
  again one resident block: 1178 against 1355, and the encoder shapes fell from
  19.4 to 11.8 TFLOP/s. Shrinking the tile until two blocks fit does not
  recover it - 128x64 double-buffered is 1128.

The common cause is that sm_75 has no `cp.async`: a global-to-shared copy
occupies registers or a second buffer, and this kernel has no room for either.
The second resident block was already providing the overlap, and both schemes
pay for a better overlap by removing it.

**Register prefetch was tried again** after the staging got cheaper, on the
weight side alone, and it is a *wash* rather than a loss: 1854 against 1855.
Worth recording twice over. The first attempt indexed the register file by a
loop variable the compiler could not prove constant, which put it in local
memory - a 32-byte stack frame and 1521 tok/s, slower than no prefetch at all.
Unrolled over a compile-time trip count it holds 124 registers, no stack
frame, and buys nothing: the two resident blocks were already covering that
latency. It is not in the tree.

**`ldmatrix.x4`** takes the mma loop's shared loads from 40 to 18 and measures
worse - 1469 against 1477 on the chat model, 882 against 897 on the
translator, 122 registers against 117. Four results have to stay live across
the `nt` loop.

**Keeping the weight tile in registers instead of shared** is the idea this
stopped short of. The B tile has *no cross-warp reuse* - warp `w` reads rows
`16w..16w+15` and nothing else touches them - so its shared round trip is pure
overhead. What makes it not worth doing is that `bs` is a statically sized
`__shared__` array and the format is a runtime value, so the 10 KB cannot be
given back and only the instructions can: about 5% of the loop, against a
rewrite of the fragment addressing.

### Two things that are not worth trying again

The **dequantization arithmetic** is 3 ms of a 90 ms prefill. Replacing the
scale and nibble arithmetic with a bit-twiddle that keeps the same loads gives
1472 against 1422. Measure it before optimising it: an earlier version of this
same test replaced `q4k_run` with a byte-at-a-time loop and reported the
dequant as *negative* cost, because it had swapped an efficient path for an
inefficient one rather than removing anything.

**Widening each warp's n-slice** to cut the ratio of shared loads to mma
instructions - 18 loads per 16 mma - measured worse at every tile tried
(`GEMM_WARPS = 4` at 128x128: 819; 128x256 at eight warps: 828). The
accumulator array grows with it and the registers come from the same budget.

### The ASR, and a regression that hid inside an end-to-end wash

The same `gemm` runs the Whisper encoder, so the encoder shapes went from
16.8/15.2/16.2 to 20.8/18.0/19.7 TFLOP/s. The first end-to-end check said the
ASR had not moved at all - 293.4 ms before the work, 294.3 after - which was
read as "the encoder matmul is not what the ASR waits on" and written up as
such. That reading was wrong, and the way it was wrong is the point: the
measurement compared *all* of the session's changes at once, and inside it one
change was giving back what the others won.

The activation-widening pass was the culprit. It rounds an f32 activation to
f16 once instead of on every staging trip, which is free accuracy-wise and
worth 1% on the chat model's prefill - and it was costing the ASR 9.6% of a
transcription, 262 ms against 288. Two reasons: it converted `v.len()` rather
than the `(count - 1) * sa + m * k` the kernel actually reads, and it ran
against every weight format. A block-quantized weight is a quarter the bytes of
the activation it multiplies, so halving the activation is most of the saving;
an f16 or f32 weight dominates its own staging and the pass buys a few percent
of the traffic for a launch. Gated on a packed weight and sized to what is
read, the ASR is 266 ms - 9.4% faster than before any of this work, not level
with it.

`bench-gemm` shows the same thing from the other side: its f32 encoder shapes
ran at 19.3 TFLOP/s with the widening and 20.8 without.

The ASR was still 0.55x against `whisper-server` after this work, and a miss
is what it stayed. The lesson is narrower and worth more: **an end-to-end number that does not
move is not evidence that nothing changed.** Two changes of opposite sign
inside one A/B look exactly like no change at all, and the only way this
surfaced was profiling the ASR and finding a kernel at 10.4% of its GPU time
that had not existed that morning.

## One launch for a decode step's attention, and the embedding held packed

A decoded token's attention on both Llama stages was three launches a layer -
a batched score matmul, a softmax and a value product - on top of the KV
append, and the value product's one-row shape ran the cache at 200 to
400 GB/s. `attn_decode` does the three in one, reading the caches in place:
a block per 64 keys of one KV head, every load issued before anything waits on
one, and a last-block merge across the chunks in the same launch.
`docs/KERNELS.md` has the design, the six arrangements that lost to it, and the
register table.

`bench-attn`, ms per 32 layers, the chain against the fused kernel at each
model's own shape (32 query heads over 8 KV heads of 128 for the chat model;
40 over 40 for the translator):

| shape | context | chain | fused |
| --- | ---: | ---: | ---: |
| chat | 128 | 0.40 | 0.42 |
| chat | 512 | 0.64 | **0.45** |
| chat | 1024 | 0.91 | **0.60** |
| chat | 2048 | 1.51 | **1.17** |
| translator | 128 | 0.51 | 0.44 |
| translator | 1024 | 1.62 | 1.61 |

End to end, nine-round medians, both binaries alternated in one sitting,
decode in ms a token:

| stage | context | before | after |
| --- | ---: | ---: | ---: |
| chat | 128 | 9.65 | 9.74 |
| chat | 1024 | 10.22 | **10.09** |
| chat | 2048 | 10.82 | **10.58** |
| translator | 128 | 15.92 | 15.90 |
| translator | 1024 | 17.77 | 17.88 |

Prefill did not move on any row - 49.4 ms against 49.4 at 128 tokens, 333
against 334 at 1024, 706 against 707 at 2048 on the chat model; 89.0 against
89.0 and 616 against 620 on the translator - because a prefill takes
`flash_attn` and never sees this kernel.

**1.3% at 1024 and 2.2% at 2048 on the chat model, and 1% the other way at
128.** The slope is what moved: 0.62 ms more per 1024 tokens of context before,
0.45 after. The 128-token row is 0.09 ms a token worse, which is more than the
microbenchmark's 0.02 and is recorded as a cost, not noise: at 128 positions a
head is two chunks and the merge is a second pass over very little. The rows a
conversation decodes at are the 1024 and 2048 ones, and that is where this
kernel was aimed.

**The translator does not move**, and the microbenchmark said it would not:
1.61 ms against 1.62 at 1024. With 40 KV heads for 40 query heads there is no
group of queries sharing a head's cache for one launch to serve, so the chain
was already reading each head once per token, and the fused kernel reads the
same bytes in one launch instead of three. The launch count saved is worth
0.07 ms per step at 128 tokens on the microbenchmark and nothing measurable end
to end; the 0.6% at 1024 is inside the row's run-to-run spread and is not
claimed either way.

**The embedding tables stay packed.** `embed_q` gathers a token's row straight
out of the file's own blocks and unpacks each element on the way, so the table
is no longer widened to f32 at load: the chat model's residency went from
6 400 MiB to 4 768 and the translator's from 8 608 to 7 778, and the gather
returns exactly the f32 the widened table held - the differential test is at
equality against the CPU dequantizer. This is residency and not speed: one row
of a few kilobytes a token is not where a step's time goes, and the timing
tables above were measured with it in place.

## A decode step in 243 launches, not 469

The round after the fused attention was about launch count, and the rule it
ran on was the one the sibling engine's session had already measured and
said out loud: at one token, any kernel under about ten microseconds is
mostly its own floor, and the floor is paid per grid, not per body. So a
seam in `blockIdx` that puts two bodies under one grid is free and bit for
bit the same, and anything that adds work to a body is not. Every fold below
that removed a grid paid; the one that added a tail to a body came out level
on kernel time and paid only in the launches it removed.

`nsys` on the chat model's decode step at a 1024-token context, sixteen
steps averaged, before this round and after it:

| | launches a step | kernel time | gaps between launches | span |
| --- | ---: | ---: | ---: | ---: |
| before | 469 | 9 669 us | 411 us | 10 080 us |
| after | 243 | 9 384 us | 240 us | 9 624 us |

What the 226 launches were, and what took each:

- **Four to one after the projections.** `rope` on q, `rope` on k, the key
  append and the value append were four launches of a few kilobytes each;
  `rope_cache_f16` is the four under one grid, and the rotated key goes
  straight to the cache. Bit for bit the chain's output.
- **The normalisation into the tail of the projection before it.** The
  attention output and the MLP down are each followed by a norm that reads
  the row they wrote plus the residual stream; `gemv_norm` does the add,
  publishes a partial sum of squares a block, and the last block to arrive
  sums the partials in a fixed order and normalises the row with its int8
  twin. The first tail walked the row a scalar at a time behind L2 loads and
  cost ten microseconds a launch - more than the kernel it replaced;
  widened to four columns a thread with the loads issued ahead, it costs
  about three over the plain mat-vec, which is what the norm kernel cost on
  its own. So the kernel time is level and the two launches and their gaps
  are the gain. The reduction is deterministic on purpose: an atomic float
  sum would have made every step's arithmetic a function of scheduling.
- **The attention writes its own twin.** The output projection used to spend
  a launch quantising the context on its way in; the decode attention now
  takes `quantize_q8`'s five shuffles over the row it has in registers. The
  attention kernel is 1.4 us longer for it and the launch is gone.
- **q, k and v under one grid.** The chat model's k and v were two launches
  of 128 blocks each at about 430 GB/s; with the three projections held as
  one `[6144, 4096]` stack they are one launch of 768 blocks at 26.7 us
  against 19.0 + 11.1 apart. Where `attn_v` is `Q6_K` the stack is `[q; k]`
  and v stays its own launch. A prompt still runs them one at a time, from
  each projection's first row, because its rows have to come out `[n, out]`.
- **The chunk width by context.** `attn_decode`'s 64-key chunk was the right
  width in the middle of the range and the wrong one at both ends; the
  kernel is now instantiated at 32, 64 and 128 and the launcher picks by the
  context. `docs/KERNELS.md` has the sweep. About 1% of a step at each end.

End to end, nine-round medians, both binaries alternated in one sitting with
the box quiet, decode in ms a token:

| stage | context | before | after | |
| --- | ---: | ---: | ---: | ---: |
| chat | 128 | 9.67 | **9.20** | -4.9% |
| chat | 1024 | 10.24 | **9.64** | -5.9% |
| chat | 2048 | 10.83 | **10.07** | -7.0% |
| translator | 128 | 15.93 | **15.34** | -3.7% |
| translator | 1024 | 17.77 | **17.20** | -3.2% |

Prefill did not move on any row - 49.3 against 49.8 ms at 128 tokens, 332
against 335 at 1024, 705 against 707 at 2048 on the chat model, 89.3 against
89.0 and 617 against 619 on the translator - because none of this touches a
prompt: a prompt takes the tiled kernels, and the folds are all at one row.
The context slope on the chat model is 0.46 ms per 1024 tokens now, from
0.62. The ASR's decode loop is 69.2 ms against the same baseline's 73.7 on
the 2.93 s clip, which is the previous round's figure again: nothing in this
round touches it.

The llama.cpp columns were **not** re-measured in this sitting, so the
"level or ahead on every row" table above stands as it was and this round's
decode figures are against this engine's own previous binary only.

What is left of the step at 1024, from the same profile: the mat-vecs are
8 436 us of the 9 384 - 3 635 in gate/up at 585 GB/s, 3 245 in the two
norm-fused projections, 762 in the head, 794 in q/k/v at 528 GB/s - the
attention 635, and everything else under 200. The weight stream is 4.92 GB
in about 8.4 ms, which is 585 GB/s against the card's 672, and the next
lever there is the 4096-wide projections' launch floor, not any kernel's
body.

## Decode at a context length worth having

Every decode figure above was taken at a 128-token prompt, and that is the
wrong place to look at this model. A conversation carries a system prompt and a
history, so the context a reply is actually decoded against is hundreds to
thousands of tokens, and decode gets slower as it grows: the weights are a
fixed 4.92 GB a token but the KV cache is read in full for every token, and at
2048 positions that cache is 537 MB - 10% again on top of the weights.

Swept, nine-round medians:

| context | decode, before | after |
| --- | --- | --- |
| 128 | 9.77 ms/tok | 9.72 |
| 512 | 10.14 | 9.98 |
| 1024 | 10.66 | 10.32 |
| 2048 | 11.71 | 10.98 |

**6.2% at 2048 and 3.2% at 1024**, and the slope is what moved: decode used to
cost 0.63 ms more per 1024 tokens of context and now costs 0.42. Both binaries
alternated in one sitting, twice through, because the 128-token column moves by
more than this change is worth between sittings - the prefill rows drifted 2-4%
across two sittings here while decode reproduced to 0.02 ms. The section
before this one has since taken the 2048 row down a further 2.2%.

### The grouped-query rows were each reading the whole cache

Timing the decode attention on its own, at the 8 B model's shapes and across
all 32 layers, said where it was going:

| context | scores | softmax | context product | total |
| --- | --- | --- | --- | --- |
| 512 | 0.416 ms (161 GB/s) | 0.132 | 0.323 (208 GB/s) | 0.871 |
| 1024 | 0.731 (184 GB/s) | 0.134 | 0.503 (267 GB/s) | 1.368 |
| 2048 | 1.430 (188 GB/s) | 0.167 | 0.871 (308 GB/s) | 2.467 |

161 to 188 GB/s against a card that streams at 585. The suspect was `gemv`'s
grid: it puts output row `r` at `blockIdx.y`, so `m` rows are `m` blocks and
each one reads the weight for itself. For a checkpoint tensor that is the right
trade - `m` is one and the weight is the whole traffic - but here the "weight"
is the KV cache and the `m` rows are the four query heads of a grouped-query
group, which exist *precisely because* they share it.

The obvious objection is that L2 should absorb the re-reads: one key head's
cache at 2048 positions is 1 MB and the L2 is 6 MB, which is the same argument
that correctly explained why an f16 cache did nothing for the ASR encoder. So
it was measured rather than assumed - the identical call at one row instead of
four:

| context | four rows | one row |
| --- | --- | --- |
| 512 | 0.416 ms | 0.245 |
| 1024 | 0.731 | 0.353 |
| 2048 | 1.430 | 0.603 |

Four rows cost **2.4x** one row, not the 1.0x sharing would give. L2 was not
absorbing them, and the ASR precedent did not transfer: there the re-reads were
one head's 768 KB walked repeatedly by one kernel, here they are eight heads
against 32 layers with a 4.92 GB weight stream evicting everything between.

`gemv_rows` reads the column once and spends it on every row. Same shapes:

| context | scores | softmax | context product | total |
| --- | --- | --- | --- | --- |
| 512 | 0.269 ms (249 GB/s) | 0.132 | 0.280 (240 GB/s) | 0.681 |
| 1024 | 0.415 (324 GB/s) | 0.134 | 0.406 (331 GB/s) | 0.955 |
| 2048 | 0.704 (381 GB/s) | 0.166 | 0.661 (406 GB/s) | 1.531 |

Four rows now cost 1.17x one row rather than 2.4x, and the chain is 1.6x
faster at 2048. What is left is the softmax, which at 0.166 ms over 32 launches
is a launch cost and not a bandwidth one, and the gap from 406 to 585 GB/s.

**This is the chat model's kernel and nothing else's.** The translator and the
ASR are multi-head, not grouped-query, so their decode arrives with one row and
stays on `gemv`.

### Measured and rejected: capturing the decode step in a CUDA graph

The step issues about 390 kernels a token, and at roughly 3.3 us of queueing
each that is 1.3 ms of CPU against a 9.7 ms token - close enough to the 1.26 ms
that separates decode from its bandwidth floor to be worth ruling in or out
before building anything.

Ruled out. Issuing 390 launches and timing the submission separately from the
completion:

```
390 launches: issued in 2.03 ms, finished in 5.09 ms
```

The CPU finishes queueing the step less than halfway through it and then waits,
so the launches are already hidden behind the GPU and a graph would remove a
cost that is not being paid. `cudarc` has `CudaGraph` and it was not used.

## An f16 KV cache: 4 GiB, and a speed result that depends on the row count

The cache is the one buffer whose size is the *context* rather than the
checkpoint. Halving its width was named in the residency section above as the
first thing to try if the card got tight, and the machinery is the same for
both Llama stages: two append kernels, a growth kernel, both mat-vecs and the
fused prefill attention, each with an f16 twin.

**What it bought in memory,** peak resident during a 2048-token prefill and 32
decodes, polled from `nvidia-smi`:

| | f32 cache | f16 cache | saved |
| --- | ---: | ---: | ---: |
| chat 8 B | 8 070 MiB | 7 558 MiB | 512 MiB |
| translator 13 B | 17 190 MiB | **13 126 MiB** | **4 064 MiB** |

The chat saving is exactly the 512 MiB the arithmetic predicts. The
translator's is a quarter more than the 3.1 GiB predicted, because the prefill
also stages from the cache and its working set narrows with it.

**What it bought in speed depends on which model,** and the split is the whole
finding. Nine-round medians, alternated:

| | 128 | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: | ---: |
| chat, f32 cache | 9.66 | 9.96 | 10.28 | 10.94 |
| chat, f16 cache | 9.68 | 9.95 | 10.25 | **10.84** |
| translator, f32 cache | 15.98 | 17.35 | 19.00 | — |
| translator, f16 cache | 15.89 | **16.67** | **17.75** | — |

ms a token. **The chat model is a wash and the translator gains 3.9% at 512 and
6.6% at 1024.** Prefill moved once, on the chat model at 2048: 719.0 ms to
708.5.

### Why the same change wins on one model and not the other

Timed standalone across 32 layers at a 2048 cache, the two decode products
separately:

| | scores, f32 | scores, f16 | context, f32 | context, f16 |
| --- | ---: | ---: | ---: | ---: |
| four rows (chat) | 0.736 ms | **0.811** | 0.654 ms | 0.583 |
| one row (translator) | 0.601 ms | 0.483 | 0.592 ms | 0.357 |

At one row both products gain, 20% and 40%. At four rows the *score* product
gets **slower**, and it cancels what the value product gains.

The reason is the contraction, not the traffic. A score product contracts over
one head width - 128 - so at f32 a warp makes four trips and at f16 it makes
**two**, and two loads is not enough outstanding work to cover their latency;
the reduction and the four predicated row products are then most of what the
warp does. The value product contracts over the whole context, which at 2048 is
sixty-four trips either way, and there halving the bytes halves the time.

A grouped-query model reaches the score product with four rows and a multi-head
one with one, so the same kernel is latency bound on the first and bandwidth
bound on the second. Nothing here is a *loss* on the chat model, and it keeps
512 MiB, so both stages hold their caches at f16 - but the speed claim is the
translator's alone.

## Sixteen bytes a lane, and the token that was 40% not-matmul

Two separate problems, found in that order, and the second was the larger.

### The mat-vec was capped by load width, not by block layout

The packed mat-vec had a lane owning four whole bytes of a Q4_K block. Six
attempts to beat that are recorded in WHY NOT, all worse, and the conclusion
drawn from them - that the block layout was the cap - was wrong.

The measurement that settled it was a kernel that read the same 144-byte blocks
with the same header gaps, sixteen bytes a lane, and did *no arithmetic at all*:
**57.2 us, 577.8 GB/s**, against a streaming ceiling of 586.5 for the same
buffer. The layout was never the problem. Four bytes a lane was.

Sixteen bytes of Q4_K is 32 elements, and 32 f32 activations is 128 bytes of
scattered reads that cost more than the wide load wins - measured at 128.6 us,
worse than the 88.7 it replaced, with `ptxas` reporting 64 registers and no
spills, so it was the reads and not the pressure. Staging the activations in
shared memory instead was 143.3 us, killed by 16-way bank conflicts.

What works is quantizing the activation to int8. Then 32 of them are two 16-byte
loads, and the dot product is four `__dp4a` against nibble masks that are
already four int8 weights in a word:

| n | k | before | wide + int8 | |
| --- | --- | ---: | ---: | ---: |
| 14336 | 4096 | 88.7 us / 373 GB/s | **58.9 us / 561 GB/s** | 1.50x |
| 4096 | 4096 | - | 488 GB/s | 1.52x |
| 4096 | 14336 | - | 541 GB/s | 1.60x |

Neither half works alone: `dp4a` on four-byte loads was worth 4%, and wide loads
without it went backwards.

Q6_K needed one more thing. Its blocks are 210 bytes, so consecutive blocks in a
file sit at every alignment in turn and a 16-byte load is not legal on any of
them. They are re-strided to 224 on upload - 6.7% more VRAM on Q6_K tensors, and
the only place in this engine where what sits in the card is not byte-for-byte
what sits in the file.

In the model: chat decode 60.8 -> 76.4 tok/s with Q4_K alone, 80.9 with Q6_K too.

### The rest of the token was 40% of it

At 80.9 tok/s a token was 12.37 ms, of which `nsys` put 8.8 in the mat-vec. The
other 3.6 ms was 1.7 ms of small kernels and 1.9 ms the GPU spent idle, waiting
for a CPU issuing **1154 launches and 1126 allocations a token**. Nothing in
that number is arithmetic.

What was actually wrong, in the order it was found and fixed:

| | change | decode |
| --- | --- | ---: |
| | starting point | 80.9 tok/s |
| 1 | one int8 twin per activation, not per projection | 82.3 |
| 2 | KV cache appended in place, capacity doubled | 84.9 |
| 3 | head-major cache, grouped heads as the batch | **95.2** |
| 4 | normalise and quantise in one kernel | 96.4 |
| 5 | residual add folded into the normalisation; scale, mask and softmax into one pass | 97.5 |
| 6 | the gate's twin taken by the gating; `up` sharing the twin `gate` already had | 98.9 |
| 7 | `float4` in the normalisation | 100.1 |
| 8 | the rope's dummy argument allocated once, not per call | **101.5** |

(3) is the structural one. The cache was stored the way the projection produced
it and rearranged into attention's layout every step: a head split for the
queries, one for the keys, a transposed one for the values, and two `repeat_kv`
expansions that materialised four identical copies of every cached head - six
kernels and six allocations a layer, all of them producing a tensor thrown away
before the next token. Storing the cache head-major makes every one of them
disappear, and the grouped heads become the batch dimension of a matmul that was
already batched: eight products of four query rows each, instead of thirty-two
products against a key tensor expanded to match.

(2) is worth naming separately because it was quadratic. The cache grew by
exactly the tokens being added, so every layer of every token allocated a new
buffer, zeroed it, and copied the whole cache into it - 128 allocations and
16 MB of copying a token at 64 of context, and worse from there.

(8) is the smallest change here and worth 1.4 tok/s: `rope` took an optional
argument as a flag plus a buffer the kernel never reads, and allocated and
zeroed that one float on every call.

Per decoded token, chat: 1154 launches and 1126 allocations became 866 and
roughly 480, and the GPU-idle gap went from 1.9 ms to 0.15.

### What it cost in accuracy, which is more than it first looked

The int8 activation was, when this section was written, the engine's one
deliberate approximation. Against the same model with f16 weights, worst logit
difference **0.167 against a span of 25.3**, or 0.66%.

It is not the only one any more - `gemm_i8` quantizes activations in prefill
too, and the figure for both together is 0.69% on this model and 0.42% on the
translator. The numbers below are the ones measured when only the mat-vec read
int8; the agreement with llama-server they report is unchanged by the second
kernel, which is the point of quoting them here rather than restating them.

That number understated it, and the sentence that used to follow it - that
greedy decoding picks the same token at every position - was measured against a
capture that turned out to be the wrong reference. Against llama-server running
the **f16** build, on the same quantized checkpoint:

| weights | activation | agreement |
| --- | --- | ---: |
| `Q4_K` | f16, tiled | 124 of 125 |
| f16 | f32, mat-vec | 124 of 125 |
| `Q4_K` | **int8, mat-vec** | **114 of 125** |

**Quantizing the weights costs almost nothing and quantizing the activation
costs about a tenth of greedy decisions**, several at margins the reference won
by nine to twelve nats. It is the same trade llama.cpp makes for the same
reason, and it is what this section's 1.67x is bought with. `docs/TESTING.md`
has the rest.

### The translator is 1.55x faster and still behind

Same changes, same order, 35.8 -> 55.3 tok/s against llama.cpp's 61.5. It is
further from the roof than the chat model - 500 GB/s in the mat-vec against 567 -
and the reason is its attention rather than its projections: 13 B Llama-2 is
multi-head, so there are 80 small attention mat-vecs a token against the chat
model's 64, over a KV cache four times the size and held at f32 where llama.cpp
holds it at f16. Halving that cache is the next thing to try and it has not been
tried.

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
| `--packing packed` | **16.4 ms/tok, 61.0 tok/s** | **399 tok/s** | 4.9 GB |
| `--packing f16` | 28.5 ms/tok, 35.0 tok/s | 322 tok/s | ~16 GB |

So packed is 1.74x faster at decode *and* 3.3x smaller, and the trade-off that
made f16 worth considering is gone.

Prefill used to be f16's one remaining win, by 1.55x, and it is not any more -
see below. The reason it was is worth keeping: prefill is a `gemm` with as many
rows as there are prompt tokens, and the tiled kernel *stages* its operands to
f16 before touching the tensor cores. Reading a packed weight means unpacking
during that staging, and the staging had never been measured.

### Prefill was decoding a block header for every element it staged

`gemv` got the one-header-per-eight-elements treatment when decode was measured.
`gemm` did not, because prefill was not on anyone's list, so its staging loop
still reached each weight through `q_at` - which re-derives two f16 scales, a
six-bit sub-block scale and four divisions per element. Two elements a thread,
two full header decodes.

Eight elements a thread instead, one header for all of them. Eight *eight-
aligned* elements stay inside one 32-element sub-block and one Q6_K scale group,
and `GEMM_KC` is a multiple of eight, so nothing straddles.

| Taigi 13 B, prompt tokens | prefill | |
| --- | ---: | ---: |
| packed, per element | 515.9 ms / 32 tok | 62.0 tok/s |
| packed, one header per pair | 298.9 ms / 32 tok | 107.1 tok/s |
| packed, one header per eight | **238.7 ms / 32 tok** | **134.0 tok/s** |

**2.16x.** At 128 prompt tokens it passes the f16 path outright - 398.8 tok/s
against 322.4 - so packed is now ahead of f16 on both halves of a forward pass
as well as 3.3x smaller. There is no shape left where widening the weights
helps.

## A spoken turn: where the 7 s went

`xabe-engine --serve`, one typed turn, the reply chunked as it streams and each
clause translated then synthesised. Measured over the WebSocket, so these are
what a listener waits through rather than what a stage costs in isolation.
Three clauses, Tacotron2, Breeze2 8 B chat, Taigi 13 B translator; medians
of three runs of the same prompt.

| | one card | translator on card 1 |
| --- | ---: | ---: |
| first audio | **1798 ms** | 1930 ms |
| whole turn | **5638 ms** | 5620 ms |

The one-card column is current. The two-card one predates both the Tacotron2
second pass and the prefill fix, which is why one card now reads better than
two: splitting the cards is still worth what it was worth, on top of numbers
that have since moved.

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

Since then the translator has learned to decode several clauses over one
weight stream, which is the lever above applied to what a turn actually
hands it; see "Several clauses, one weight stream" for what it bought and
the policy it settled.

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
| ASR, Whisper large-v2 1.54 B | safetensors → f16 | 3 202 | 3 499 |
| chat, Breeze2 8 B | GGUF `Q4_K_M`, packed | 4 768 | 8 267 |
| translator, Llama-2 13 B | GGUF `Q4_K_M`, packed | 7 778 | 16 045 |
| CosyVoice3, LM + flow + vocoder | safetensors → f16 where read as f16 | 1 958 | **18 005** |

**18 005 MiB — 17.6 GiB of a 48 GiB card**, 37% of it, leaving 31 141 MiB for
KV caches and activations. Without CosyVoice, which is the alternative
synthesiser rather than a second stage, the four remaining are 16 047 MiB.
The CosyVoice row stood at 3 266 MiB, and then at 3 469 when it was measured
again after the speech LLM's weights were halved - which is the trap
described under "CosyVoice3's residency" below. The whole table was
re-measured in one process after it was closed, which is where these
figures are from; CosyVoice alone, with the context charged to it, is
2 157 MiB.
This table stood at 21 771 MiB until the embedding tables were held packed -
"One launch for a decode step's attention, and the embedding held packed"
above - which is 1 632 MiB on the chat row and 830 on the translator's against
the table as it stood, and 1 730 and 928 when the same binary is loaded both
ways in one sitting, which is the figure the arithmetic below predicts.

The VAD is absent from that table because it occupies nothing: `xabe_vad::open`
takes no device ordinal, Silero being 1.8 M parameters of CPU arithmetic. The
report prints a zero row for it rather than leaving a reader to wonder which
stage was forgotten.

### The same card with Tacotron2 as the only synthesiser

Neither VITS nor CosyVoice, which is the configuration to read if the reply
path is Han text through Tacotron2. The Tacotron2 row read 489 when it was
measured with its weights converted on the card and 459 on the day the
loader was changed to convert on the host, which is the same trap as
CosyVoice's below at a smaller scale; it is 427 now, and the rows after
it are carried from the earlier measurement:

| stage | container | delta MiB | cumulative |
| --- | --- | --- | --- |
| Tacotron2 + WaveGlow 116 M, + the CUDA context | safetensors → f16 on the host | 427 | 427 |
| ASR, Whisper large-v2 1.54 B | safetensors → f16 | 3 202 | 3 629 |
| chat, Breeze2 8 B | GGUF `Q4_K_M`, packed | 4 768 | 8 397 |
| translator, Llama-2 13 B | GGUF `Q4_K_M`, packed | 7 778 | **16 175** |

**16 175 MiB — 15.8 GiB**, leaving 32 977 MiB. Tacotron2 costs 130 MiB more
than VITS did in the row above, which is the two models' parameter counts and
not the context; dropping CosyVoice is what saves the 3 266 MiB.

### The caches are the part that grows, and the translator has no GQA

Weights are what the table measures. The KV cache's capacity doubles from 256,
so a 2 100-token conversation has allocated 4 096 slots and pays the 4k column.
**The cache was f32 when this was written and is f16 now** - see "An f16 KV
cache" below - so the figures here are the f32 ones the rest of this section
reasons about, and each is now half what it says:

| | kv heads | per token | 1k ctx | 4k ctx |
| --- | ---: | ---: | ---: | ---: |
| chat 8 B | 8 of 32 | 256 KiB | 256 MiB | 1.00 GiB |
| translator 13 B | 40 of 40 | 1 600 KiB | 1 600 MiB | **6.25 GiB** |

That arithmetic is checked against a measurement rather than left as
arithmetic: the translator alone, a 2 048-token prefill and 64 decodes on an
otherwise empty card, peaks at **18 031 MiB** - about 8.9 GiB of weights and
context, 6.25 GiB of KV at the doubled 4 096 capacity, and the rest prefill
activations.

So the four stages at a 4k context on both models came to roughly 26 GiB of
the 48. The headroom is real, and the translator's cache spends it four times
faster than the chat model's: this entry ended by saying an f16 KV cache there
was worth 3.1 GiB and was the first thing to try if it ever got tight. **It has
since been done**, and measured at more than that - 4.0 GiB - because the
prefill's own working set narrows with it.

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
| Breeze2 8 B `Q4_K_M` | 4 685 MiB | 16 489 MiB | 4 768 MiB | 3.46x |
| Taigi Llama-2 13 B `Q4_K_M` | 7 663 MiB | 26 025 MiB | 7 778 MiB | 3.35x |

The packed figures exceed the file sizes by 83 and 115 MiB, under 2%, and
that is the f32 norms and the allocator's rounding; it is not itemised. This
table used to read 6 400 and 8 608 MiB, 1 715 and 945 over the files, and the
whole of that was the **embedding table**: a gather rather than a matmul, so
it had its own kernel and was widened to f32 at load - at 8 B, 128 256 x 4 096
as f32 is 2 004 MiB against about 282 as `Q4_K`, a difference of 1 722, which
was the 1 715 measured. `embed_q` now gathers out of the packed blocks and the
difference is gone: 1 730 MiB on the 8 B and 928 on the 13 B, the same
binary loaded both ways in one sitting, against 1 722 and 940 predicted. The 8 B's ratio was the worse of the two for exactly that
reason - a larger vocabulary over a smaller model put more of the checkpoint
into the one tensor that was not packed - and now it is the better one.

### Why this is the difference between fitting and not

At f16 the same four stages come to **46 011 MiB** - 297 + 3 200 + 16 489 +
26 025 - against a 49 152 MiB card. That is 93.6% full and leaves 3 141 MiB,
which is *less than the 13 B's own KV cache* at any useful context length.

Add CosyVoice, which is unquantized either way, and f16 comes to **49 277
MiB** against a 49 152 MiB card: it does not merely leave too little headroom,
it exceeds the card by 125 MiB and fails to load at all. Packed, the same five
stages are 19 311 MiB and use 39% of one card.

So the honest statement is not "f16 is tight". At f16 these stages do not share
a card; packed, they share one with 29 GB to spare.

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
lever and is a project rather than a change. (It has since been pulled; the
next section is that round, and it found the loop was only half launches.)

### The decode loop, fused: 1.21x, and a checkpoint that was f16 all along

That paragraph named the next lever and this is the round that pulled it.
Two things were done to the loop and one was found about the weights.

**The loop was CPU-bound, not launch-bound.** `nsys` on the shortest line
put the decode loop at 39.2 operations a frame and 380 µs a frame, with the
card busy for 174 of them - 46%. What the other half was: a stop-gate
download every frame, a dropout mask uploaded every frame, an allocation
and a `memset` for every intermediate. Folding what could be folded -
`relu_mask`, `concat2`, `attn_weights_update`, `copy_from_into`, the
projection and the gate as one stacked mat-vec, the masks drawn once and
the gate read in batches of eight (`GATE_LOOKAHEAD`, bit-identical frames
against reading it every frame) - took it to 26.8 operations and 298 µs.
The location attention was then nine of the 27, and it is two now:
`taco_energies` and `taco_context`, held to a CPU chain of the seven
kernels they replace at 1e-4. That is 19.3 operations and 265 µs.

**Two thirds of the card's time was the LSTMs streaming f32.** With the
launches out of the way the profile was readable: four mat-vecs a frame
over the two decoder LSTMs, 71 MB of f32 weight a frame, 30 µs each, and
between them 107 of the 158 µs the card was busy. That is 666 GB/s, which
is the card. The checkpoint was trained under AMP, and **every LSTM weight
in the published file is exactly representable in f16** - all 19.4 M of
them, checked - so holding the decoder's two at f16 is not a rounding, it
is the same numbers at half the bytes. Measured on the real checkpoint,
same seed and masks, the mel moves by 5.7e-6 on a span of 10, which is the
half-width mat-vec's accumulation order and nothing else; the test in
`xabe-taco`'s `pipeline` holds it at 1e-4. Every other checkpoint in
`models/` was scanned the same way afterwards and none of them is: under
0.3% of sampled f32 values are f16-exact in the ASR, both VITS, CosyVoice3,
WaveGlow and Silero, which is what random f32 looks like. The finding is
Tacotron2's alone, and it came from how NVIDIA trained it. The mat-vec went from 30.3 to
16.5 µs and the frame from 265 to 233 µs, with the card busy for 110.

Same sitting, alternated in pairs, the previous binary against this one,
the two pairs' worse figure on each side. Synthesis is seeded, so the
frame counts are identical across the four runs of each line.

| Text | Frames | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `Tâi-lâm ū chiok chē hó-chia̍h--ê,` | 206 | 149.5 ms | 123.5 ms | 1.21x |
| `chhin-chhiūⁿ: Khah-sú môa-lî, ke-nn̄g-ko, ah-bah-mī` | 397 | 278.1 ms | 224.6 ms | 1.24x |
| `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó, Chhiah-khàm-lâu` | 513 | 358.5 ms | 287.3 ms | 1.25x |

The better pair reads 148.6 → 122.0, 268.9 → 223.2 and 345.8 → 284.7, which
is 1.20x to 1.22x; **1.21x** is the figure to quote. 16.1x realtime to
19.6x on the shortest line.

**What is left is the CPU again, and it is a different project.** At 19
operations a frame the card is busy 47% of the time; `cuLaunchKernel` has a
median of 5.4 µs in the API trace and the loop issues one every 12 µs, so
the card finishes each frame's work and waits for the next to be described.
The encoder's per-token LSTM steps are the same shape at a much smaller
weight and were left at f32, because that stage is held to its reference
at 1e-5 and runs once a line. Replaying a frame as a CUDA graph would
remove the issue cost, and every offset a frame's kernels take is
data-dependent on the frame index, so that is the kernels taking their
position from device memory rather than the wrapper - a design change
rather than a fold, and where the next round would start.

### Sixteen launches and no allocation: 1.03x, and a graph that bought nothing

The paragraph above ends by naming the CPU as what was left and a CUDA
graph as the lever. This is the round that pulled it, and the diagnosis
was wrong in a way worth recording.

**The graph was built, and it measured as nothing.** The frame was
captured once and replayed - `Gpu::capture` around the same launches,
every offset that depends on the frame index read from a device-side
counter so the recording is the frame - and the first replay was 124.5
against 122.7 ms on the shortest line. `nsys` at node level said why: the
frame allocated eleven temporaries, a capture turns each into a memory
node, and a graph holding memory nodes replays at about the cost of
issuing it. So the frame was made allocation-free - `conv1d_into`, the
mat-vec's `gemv_into`, a caller-owned scratch for the attention's
energies, every temporary a field of one state - and replayed again: 184
µs a frame under `nsys` against **166 issued launch by launch**, and on
the bench, alternated three times, 119.0 against 119.5 ms. Level. The
graph is not in the tree.

**What the trace showed instead is the card's front end.** One frame,
both modes, the gap before each kernel:

| Kernel | Runs | Gap before, replayed | Gap before, issued |
| --- | ---: | ---: | ---: |
| `gemv` (prenet) | 2.1 µs | 7.9 µs | 7.7 µs |
| `relu_mask` | 1.3 µs | 1.9 µs | 2.9 µs |
| `gemv` (attention cell, input side) | 12.9 µs | 3.9 µs | 4.2 µs |
| `gemv` (recurrent side) | 16.4 µs | 0.5 µs | 0.6 µs |
| `lstm_gates` | 1.9 µs | 0.6 µs | 0.5 µs |
| `conv1d_short` (location) | 15.6 µs | 7.6 µs | 8.1 µs |
| `gemv` (query) | 2.9 µs | 0.5 µs | 0.5 µs |
| `taco_energies` | 6.8 µs | 0.5 µs | 0.5 µs |
| `taco_context` | 5.6 µs | 0.5 µs | 0.5 µs |
| `gemv` (decoder cell, input side) | 23.3 µs | 8.3 µs | 8.5 µs |
| `gemv` (projection) | 3.9 µs | 7.0 µs | 8.3 µs |

After a kernel that runs fifteen microseconds the next several launches
start half a microsecond apart, whatever they are; after a run of short
ones, each launch costs three to eight. Two idempotent kernels inserted
after `relu_mask` cost 7.1 and 3.6 µs of gap, and one inserted after the
15 µs convolution cost 0.5, so it is the run-ahead of the card's launch
pipeline that hides the cost and not anything about the kernels. The
same pattern, the same places, in both modes. The 5.4 µs
`cuLaunchKernel` median the previous round read as the CPU falling
behind was `nsys` inflating the API it was tracing, and the chat model's
step - the peer session's decode is a graph, and reports the same floor
inside it - never showed the gaps because its kernels are long.

**So the lever is the launch count, and the frame is sixteen now.** The
three `concat2` launches are gone by layout: each LSTM's hidden state
lives at the head of the buffer the next projection reads, the prenet's
last layer writes into the head of the attention cell's input, and
`taco_context` writes the context after all three, at their offsets, in
the pass that computes it. The two copies and the counter step that ended
a frame are one `taco_emit`, and the copy of the frame into the next
step's input is gone because the prenet reads the projection's output
where it lies. The same kernels do the same arithmetic on the same
numbers, the stop lands on the same frame of every line, and the tests
that hold the batched gate read and the half-width decoder to their
references pass unchanged; the mel was not diffed against the previous
binary.

Same sitting, alternated in pairs, the previous binary against this one,
the worse pair on each side:

| Text | Frames | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `Tâi-lâm ū chiok chē hó-chia̍h--ê,` | 206 | 123.4 ms | 119.9 ms | 1.03x |
| `chhin-chhiūⁿ: Khah-sú môa-lî, ke-nn̄g-ko, ah-bah-mī` | 397 | 224.6 ms | 218.9 ms | 1.03x |
| `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó, Chhiah-khàm-lâu` | 513 | 287.3 ms | 279.1 ms | 1.03x |

The better pair reads 122.6 → 119.1, 223.5 → 217.9 and 286.6 → 277.8.
**1.03x**, about 20 µs a frame, which is what six launches cost at this
floor.

**Where the utterance is now, and why the loop stops here.** The decoder
is 206 frames at about 165 µs, some 34 ms of the 119; the card is busy
for 105 of each frame's 166. What follows the decoder is 99 ms of wall
and 87 of kernel time, and by kernel: `gemm` 51.6 ms - the dilated
convolutions 24.1 at 21 TFLOP/s, the conditioning 16.6 at 25, and the
residual and skip projections **10.6 as two `n = 256` products a layer at
16 TFLOP/s and 106 blocks on a card with 144 slots** - then the
conditioning add 6.9, `im2col` 6.1, the residual and skip adds 6.1, 445
`memset`s 5.0, the mel upsample 4.6 at 0.6 TFLOP/s, the gated activation
3.6, and the postnet's five convolutions 2.1 at 1.3 TFLOP/s. Every one of
those but the first two is a fold or a shape: the add into the
activation that reads its sum, the two projections as one, the adds into
the projection's epilogue, the `memset`s under outputs the kernel writes
whole. That is the next round, and it is worth several times what the
loop has left.

### The vocoder's folds: 1.07x to 1.11x, and an epilogue that cost double before it cost half

The accounting above named the adds, the `im2col`, the `memset`s and the
two half-width projections as folds and shapes. This is the round that
did the first three of those four, and what it found in the matmul on the
way is worth more than the folds.

**What changed.** Each layer's residual and skip projections were two
products of `n = 256` over the same activation, each into a fresh buffer
and each followed by an `add_inplace` into a running sum. The checkpoint
stores the two as one `[512, 256]` tensor, residual rows then skip rows,
so they are bound as one and run as one batched product with a shared
left operand, a bias of the batch's own, and a store that *adds* to the
buffer it writes - `Gpu::gemm_batched_into` - with the two running sums
laid out as one `[2, steps, 256]` so a batch's outputs are a stride
apart. The conditioning add was a read, a write and a launch over 27 MB
a layer to leave a sum the gate then read; `gated_cond_rows` reads the
conditioning on the way into the gate instead. And the two allocations
the loop zeroed every layer are written whole by their kernels, so they
are not zeroed: 445 `memset`s an utterance are 265, and the survivors
are the small ones.

**The accumulating store cost double before it cost half.** The first
form of the epilogue loaded, added and stored each element in turn, and
the batched product ran at 224 µs where the two it replaced ran 55 each
- the single-product form at 108. The compiler cannot prove that a store
to `out` does not alias the next load from it, so the sixty-four loads a
thread were serialised behind the sixty-four stores. Written as two
passes - the running values folded into the accumulators in a pass that
stores nothing, then the plain store - the batched product is 159 µs,
which is 80 a product: the plain kernel plus the read of its output,
where the two adds it replaces were 34 each on top of the products. Even
the plain store loop had slowed, 55 to 70 µs, from a per-element branch
added to it, because at `k = 256` this kernel is mostly its epilogue;
the plain loop is untouched now and the accumulating one is its own
branch.

By kernel, after the decoder, one utterance of 206 frames:

| Kernel | Before | After |
| --- | ---: | ---: |
| residual and skip products | 10.6 ms | 14.0 ms |
| their two adds | 6.1 ms | - |
| conditioning add | 6.9 ms | - |
| gated activation | 3.6 ms | 6.0 ms |
| `memset` | 5.0 ms | 0.75 ms |
| everything else | 54.7 ms | 56.6 ms |
| kernel time | 86.9 ms | 77.4 ms |
| wall | 99.1 ms | 84.7 ms |

The audio is **bit-identical** to the previous commit's - the engine's
one-shot output on the shortest line, compared sample for sample - which
is the ordering rule in the epilogue: `(product + bias)` first and the
running value added to that, exactly what the add kernel computed.

Same sitting, alternated in pairs, the previous binary against this one,
the worse pair on each side:

| Text | Frames | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `Tâi-lâm ū chiok chē hó-chia̍h--ê,` | 206 | 119.3 ms | 111.3 ms | 1.07x |
| `chhin-chhiūⁿ: Khah-sú môa-lî, ke-nn̄g-ko, ah-bah-mī` | 397 | 218.6 ms | 197.3 ms | 1.11x |
| `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó, Chhiah-khàm-lâu` | 513 | 279.9 ms | 252.6 ms | 1.11x |

The better pair reads 117.9 → 110.1, 217.2 → 195.2 and 277.9 → 251.9.
The longer lines gain more because the vocoder's share grows with the
line; 20.0x realtime to 21.5x on the shortest and 21.4x to 23.6x on the
longest.

**What is left there.** The dilated products at 24.7 ms and the
conditioning at 17 ms are the tiled kernel at 21 and 25 TFLOP/s, its
ceiling on this card. `im2col` is 6.1 ms of gather that an A-loader
reading three taps of the residual stream would not need; the gate is
6.0 ms at 545 GB/s, bandwidth. The mel upsample is 4.6 ms at 0.6
TFLOP/s, and an integer division per tap per channel that looked like
the reason was hoisted and measured as nothing; the kernel is read-bound
on a weight each thread walks 320 elements of, and the fix is threads
that share the walk. The postnet's five convolutions are 2.1 ms at 1.3
TFLOP/s through the direct kernel.

### The transposed convolution: a thread per eight windows, 1.04x on Tacotron2 and 1.085x on VITS

The mel upsample was 4.6 ms for 2.7 GFLOP, 0.6 TFLOP/s, and the paragraph
above had just found that the integer division per tap per channel was
not why. The kernel was a thread per output sample walking 320 weights -
80 channels, four taps - and every sample in a window of 256 read a
different 320, so the weight was read once per sample: 5.5 GB an
utterance through L2 for a 26 MB tensor. The taps that touch a sample
are fixed by its phase within the stride, so samples one stride apart
share every tap. A thread now carries eight consecutive windows of one
channel at one phase, reads each weight once for the eight, and each
output's sum is still channel by channel and taps ascending - the same
to the bit, held to the scatter form by the existing test, and both
synthesisers' audio compared sample for sample against the previous
commit's engine.

The upsample is 0.97 ms. The same kernel is the VITS decoder's four
upsamplers, which the stage split put at 72% of a synthesis, and the
decoder stage went from 32.4 ms to 28.7 there.

Same sitting, alternated in pairs, the worse pair on each side:

| Text | Frames | Before | After | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `Tâi-lâm ū chiok chē hó-chia̍h--ê,` | 206 | 110.2 ms | 106.5 ms | 1.035x |
| `chhin-chhiūⁿ: Khah-sú môa-lî, ke-nn̄g-ko, ah-bah-mī` | 397 | 196.9 ms | 189.2 ms | 1.04x |
| `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó, Chhiah-khàm-lâu` | 513 | 252.8 ms | 243.5 ms | 1.04x |

And `xabe-tts-bench` on `mms-tts-nan`, three alternated pairs of twenty
runs, worst on each side: **46.25 ms to 42.64**, 56.4x realtime to 61.1x,
**1.085x**. The PyTorch comparison at the top of this file was not
re-run; the 1.24x there is against the engine as it was, and this is
1.085x on top of that engine, not a new ratio against the reference.

### The stride-one convolution as an implicit matmul: 1.48x on VITS, 1.03x on Tacotron2

`conv1d` was a thread per output tile fetching a weight from global memory
for every four multiply-adds, and on the VITS decoder's residual blocks -
256 channels, kernels of 3, 7 and 11 over 1300 positions - it ran at
about 1 TFLOP/s. A stride-one convolution is a `[out_ch, in_ch * k]`
by `[in_ch * k, out_t]` product whose right operand is gathered from the
input rather than materialised, and `conv1d_tiled` computes it that way
on the f32 cores: a block owns 32 channels by 128, 64 or 32 positions,
each thread 4 by 4, and sixteen (channel, tap) pairs of both operands
are staged through shared memory a trip, so every staged value feeds
sixteen multiply-adds. The wrapper takes the widest tile that still
gives the card a block an SM slot and keeps the direct kernel under 32
positions, where a tile would be mostly padding. The sums are the direct
kernel's to the bit - bias first, pairs ascending, each an `fmaf` - and
both synthesisers' WAVs were compared byte for byte against the previous
commit's engine: identical.

The kernel was written by a peer session (ttsxabe-15) that ended before
committing it; the measurements here are this session's. Same sitting,
GPU 2, alternated in pairs, the worse pair on each side:

| Model | Line | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: |
| `mms-tts-nan`, `xabe-tts-bench` default text, 3 x 20 runs | | 42.46 ms | 28.64 ms | **1.48x** |
| Tacotron2 | `Tâi-lâm ū chiok chē hó-chia̍h--ê,` (206 steps) | 106.1 ms | 103.0 ms | 1.03x |
| Tacotron2 | `chhin-chhiūⁿ: Khah-sú môa-lî, ke-nn̄g-ko, ah-bah-mī` (397) | 187.4 ms | 182.2 ms | 1.03x |
| Tacotron2 | `Tō͘-kui ē-tàng khì An-pêng Kó͘-pó, Chhiah-khàm-lâu` (513) | 240.1 ms | 232.7 ms | 1.03x |

On Tacotron2 the stages that moved are the ones that are convolutions:
the encoder 3.2 ms to 2.1, the location attention 8.1 to 5.9 and the
postnet 2.2 to 0.74 on the first line; the WaveGlow coupling, which is
a dilated convolution through `im2col` and the tiled `gemm`, is
untouched. The VITS synthesis is 28.6 ms for 2.6 s of audio, about 91x
realtime by the earlier run's audio length; the PyTorch ratio at the top
of this file was again not re-taken.

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

This section used to name two levers. **The first has been spent, and it
returned slightly more than it was costed at**: the 34 ms the encoder was
predicted to be spending moving an attention score matrix it never needed to
materialise came back as 38 ms, because `flash_attn` also took the query's head
split and the context's merge with it. Tuning that kernel's tile returned 6 ms
more. The encoder is 119 ms and the transcription 211.

Three things account for what is left, in the order they are worth taking.

**The matmul runs at 22-25 TFLOP/s against an instruction ceiling of 102.3
measured on this card** - the 99 this line used to name was near enough as a
number, but the reasoning attached to it elsewhere was not; see "Why f32
accumulation is not caution" in `docs/KERNELS.md`. `bench-gemm` on this card, at the
encoder's own shapes: 22.5 TFLOP/s for a 1500x1280x1280 projection, 21.9 for
the feed-forward up, 24.8 for the down. The projections and feed-forwards are
1.89 TFLOP, which is about 85 ms of the encoder's 125 at that rate. `ldmatrix`
and a double-buffered global-to-shared pipeline are the standard route to
40-50% of peak on this architecture; at 40% the same work is 47 ms. This is
the largest single lever left in the engine and it is **not** an ASR-only
change: the same `gemm` runs the Whisper encoder and both Llama stages'
prefill, which is upside and risk in the same sentence - the llama.cpp
comparison is settled at level-or-ahead and a regression there would cost more
than the ASR gains.

**The fused attention runs at 14.5 TFLOP/s**, which is 25 ms of the encoder,
and getting it there is worth writing down because the first diagnosis was
wrong. The kernel does about 117 us of tensor-core work a layer and took 967;
the gap was read as the four block-wide barriers a key tile costs, each
amortised over half as many `m16n8k8` issues at `hd` 64 as at 128. **A 64-key
trip fixes exactly that and changed the encoder by nothing** - 124.9 ms against
124.3, inside the noise. *That result was itself confounded, and is resolved
under "the encoder's fused attention" above: a 64-key trip pays once the score
tile is out of shared memory. What follows is the reasoning as it stood.* What the barrier account left out is that a block
stages every key and value it walks past, so all of K and V is re-staged once
per query block: 47 trips through 0.8 MB is 724 MB a layer, and a wider key
tile moves the same bytes in fewer trips. **The query tile divides it**, and 64
rows a block took the kernel to 791 us and the encoder to 119. Only about a
third of the halved traffic came back as time, so L2 was already absorbing much
of the re-read - which is also why the remaining ceiling is not obviously worth
another attempt. 128 rows is not reachable: its tile wants 52 KB where sm_75
gives a block 48 KB of static shared memory, and it fails in `ptxas` at module
load rather than running slowly. A `static_assert` says so at the tile now.

**The decode loop is 65 ms for six tokens.** Ten milliseconds a token for a
1.5 B decoder is about a third of what this card's bandwidth allows: the
weights a step actually reads are 734 M parameters at f16, the cross-attention
K and V caches are 491 MB because they are held at f32, and the tied output
head is 133 MB - about 2.1 GB, or 3.1 ms at 672 GB/s. Holding the cross
caches at f16 is the cheap half of that and worth about 2 ms of the six-token
loop; the rest is `gemv` efficiency.

None of these has been costed to the point of promising that they close the
gap to 144 ms, and this section still does not name a factor it has not
measured.

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

### The flow was on the host: 3.4 s to 1.1 s, bit-identical

That preliminary figure stood until the stage was timed on its own, which
`examples/stages.rs` now does - each of the four stages synchronised before
its clock stops, so the split is honest and the sum is slightly pessimistic
against `Cosy::synthesize`. One utterance of 17 text ids, 119 speech tokens,
238 mel frames, 4.76 s of audio, on card 0 with nothing else on the box:

| stage | before | after |
| --- | ---: | ---: |
| speech LLM, 119 tokens | 549 ms (4.6 ms a token) | 552 ms |
| flow, 10 Euler steps of a batch of two | **2809 ms** | **448 ms** |
| F0 and excitation | 7 ms | 8 ms |
| vocoder | 70 ms | 82 ms |
| utterance | 3436 ms | 1091 ms |

Two pairs alternated in one sitting; the table is the better pair of the old
binary against the worse of the new, and the other pair reads 3686 against
1094. **3.1x on the utterance, 6.3x on the flow**, and the mel is
**bit for bit** what the previous build produced on the same seed -
`examples/probe_est.rs` writes one estimator evaluation's every tap on
seeded inputs from either build, and the eleven agree exactly. 1.4x realtime
to 4.4x.

What the flow was doing: the DiT's adaptive normalisation was computed as a
`layer_norm` on the card, **downloaded**, modulated by `(1 + scale)` and
`shift` in a host loop, and uploaded again; the gated residual was the same
in reverse, with the residual stream downloaded and re-uploaded around it.
Twice a block, 22 blocks, two batch rows, ten solver steps: 3238
device-to-host copies and 3174 the other way an utterance, and the card busy
40% of the flow's time, a third of that busy time being the copies
themselves. Two kernels replace it - `layer_norm_mod`, the normalisation
with the modulation read straight out of the block's timestep projection,
and `gate_add`, the gated residual in place - and the timestep's SiLU is
taken once an evaluation rather than once a block. The card is busy 94% of
the flow's time now, and 277 of its 438 busy milliseconds are the tiled
`gemm`. `docs/KERNELS.md` has the two kernels and the one thing about them
that was not obvious.

Why bit-identical was worth insisting on: the first version of the two
kernels let the compiler contract the affine into an `fmaf`, one rounding
where the host loop had two. That is a one-ulp difference on the block's
input, and one ulp is enough to flip the f16 rounding the tiled matmul
applies to its operands, which is a 5e-4 relative jump on the elements it
flips. Through 22 blocks that read as 1e-4 relative after the first block
and 5e-4 after the last, and through ten solver steps as a mel that
correlated with the previous build's at 0.99996 and differed by 0.45 at
worst - inside what the flow's own oracle test permits, and impossible to
tell from a bug without the oracle to hand. With the multiply and the add
rounded separately the estimator is the same bits, and that identity is a
sharper test than any tolerance.

### The speech LLM: 4.6 to 2.6 ms a token, and a query group of seven

With the flow on the card the speech LLM was half the utterance: 119
decode steps at 4.6 ms, streaming 1.98 GB of f32 weight a token at the
card's bandwidth through an attention chain of eleven launches a layer
that rebuilt the whole key-value cache from f32 for every token. It runs
the way the chat model does now. The weights are held at f16 - the
prompt's tiled matmul staged them as f16 already, so the prompt's
arithmetic did not change at all and its logits are bit for bit the f32
build's; the decode mat-vec's did, from an exact f32 weight to an f16 one.
The cache is f16 in the attention layout with a doubling capacity, written
from `rope_cache_f16` and read by the fused decode attention; q, k and v
are one stacked mat-vec and gate and up one batched product with
`silu_mul_pair`. 346 launches a token became 274.

Two things this model needed that the Llama stages did not. Its query
group is seven - 14 heads over 2 - and the decode attention carried a
compile-time maximum of four, so the group is a template parameter now,
with the existing entries instantiated at four and unchanged, and a new
entry at eight. And with two key-value heads the kernel's grid is
`chunks * 2` blocks: at a context of 150 and a chunk of 64 that is six
blocks on a 72-SM card, 23 µs a layer, so the wide-group entry has a
32-wide chunk as well and takes it below 1024 of context - 14.8 µs a layer.

Same sitting, alternated pairs, the flow-only build against this one:

| | before | after |
| --- | ---: | ---: |
| speech LLM, a token | 4.59-4.61 ms | 2.63-2.65 ms |
| speech LLM, the utterance | 546-549 ms (119 tokens) | 316-318 ms (120 tokens) |
| flow | 446-450 ms | 444-447 ms |
| utterance | 1070-1077 ms | 838-846 ms |

**1.74x on the step**, 1.27x on the utterance; 4.4x realtime to 5.7x, and
3.44 s to 0.84 s across the two rounds. The token count moved by one
because the sampler read logits that differ at the third decimal, which
is the approximation's whole visible effect.

What the f16 decode is measured against, in the absence of the capture
`tests/speech_llm.rs` wants: `examples/probe_llm.rs` writes the
teacher-forced log-probabilities of every position from either build, over
the sampled tokens of the f32 one, through both paths of `forward`. The
prompt path agrees exactly. The decode path agrees on the argmax at all
120 positions, with a worst log-probability difference of 1.0e-2 and a
worst row correlation of 1.000000 - and the f32 build's own two paths
differ from each other by 1.25e-2 on the same positions, so the
approximation is inside the arithmetic noise the model already carried.
That is a build-to-build bound and not the oracle's; when the capture is
made, the oracle test is the one to trust.

The step is 80% card-busy at 2.13 ms of kernels, 1.46 of them the
mat-vecs at 0.99 GB a token, which is the memory system. What is left on
the CPU side is 274 launches, and on the card side a weight stream that
only a narrower weight would shorten.

### CosyVoice3's residency: 3 469 MiB to 2 157, and the number that did not move

The speech LLM's weights were halved and `xabe-vram` read **3 469 MiB**,
against the 3 266 in the table above from before - more, not less. The
same command on the commit before the halving read 3 469 too. The reason
is the memory pool, which is told to keep its pages: the loader was
uploading each f32 tensor and converting it on the card, so for a moment
both widths were resident, and the pool kept that peak as its own and
`nvidia-smi` reported it. The halving had happened; the measurement could
not see it.

Converting on the host (`Gpu::upload_f16`, round-to-nearest-even, the same
bits the card's conversion produces) removes the peak, and three more
tables go to f16 on the way: the text embedding, 151 936 rows of which a
prompt reads a few dozen, gathered by a new `embed_scaled_f16`; the speech
embedding and the speech head; and the flow's block linears - q, k, v, o
and the two feed-forwards of 22 blocks, plus the input and output
projections - which the tiled matmul stages as f16 regardless, so the
estimator is **bit for bit** what it was (`examples/probe_est.rs`, all
eleven taps). The single-row linears of the flow stay f32, because the
mat-vec reads a weight at its own width and the identity would not
survive them.

| | before | after |
| --- | ---: | ---: |
| `xabe-vram`, CosyVoice3 alone | 3 469 MiB | **2 157 MiB** |
| `xabe-vram`, CosyVoice3 after the four other stages | 3 266 MiB | **1 958 MiB** |
| speech LLM, a token | 2.63-2.65 ms | 2.61 ms |
| flow, 262 frames | 443 ms (240 frames) | 424 ms |

What the f16 embeddings cost: the teacher-forced log-probabilities
against the f32 build move from a worst difference of 1.0e-2 to 1.8e-2,
the argmax still agreeing at all 120 positions and the worst row
correlation 0.999999. The speech embedding was tried at f32 as well, since
every decoded token enters through it, and the figure came out 2.6e-2 -
no closer - so it is held like the rest. These are single-run figures,
not a paired sitting, and the flow's 4% is inside what a sitting could
move; the residency is the result here.

### The flow's two rows as one pass, and its attention fused: 409 ms to 276

An utterance's flow is ten Euler steps, each an estimator call over a
batch of two - the conditioned row and the unconditioned one that
classifier-free guidance extrapolates between - and the estimator is a
22-block DiT over about 300 positions. The trace of one block, per row,
was seven products of `[300, 1024]` against a `[1024, 1024]` or wider
weight at 72 to 88 µs each, and the unfused attention chain - two head
splits, a transposed one, a scores product, a scale, a softmax, a value
product and a merge - at about 150 µs. Both were the card underfilled:
at 300 rows a `[1024, 1024]` projection is 8 by 3 tiles with a split of
two, 48 blocks on 72 SMs, and the attention's scores product was a
batch of sixteen `[300, 64] x [64, 300]` products with a `gemm_reduce`
after it.

Three changes, all in `xabe-cosy` and none to a kernel:

- **Both rows go through the blocks as one `[600, 1024]` activation.**
  Every block kernel but the rope and the attention is row-wise - the
  adaptive normalisations, the gated residual adds, the feed-forward -
  so a product over 600 rows is the same arithmetic as two over 300 with
  the card twice as full. The input embedding stays per row, because its
  positional convolution pads on the left and a `[1024, 600]` would let
  row 1's first frames see row 0's last.
- **The query, key and value projections are one product** over a
  `[3072, 1024]` weight stacked at load, 120 blocks where three launches
  of 48 each left two thirds of the card idle. The rope is partial - the
  first 64 of 1024, one head of sixteen - and counts positions from the
  first row it is handed, so row 0 of the query is rotated where it lies
  and the other three `[300, 1024]` blocks are copied out first; a
  `rope_gptj` with an offset would remove three copies a block and is
  not written yet.
- **The attention is `flash_attn`**, the Whisper encoder's fused kernel
  at a head width of 64, one launch per row with no mask, in place of the
  eight-launch chain. Its arithmetic is the chain's - f32 accumulation
  from f16-rounded operands, which is what the tiled matmul stages
  anyway.

The estimator and whole-mel oracle tests pass unchanged (the mel
correlates above 0.9999 with the reference's, worst bin under 0.5).
The stages example on GPU 2, the same 131-token utterance, the 4894faa
build against this one alternated in two pairs of four runs, the first
run of each process set aside as its warm-up and the worst of the rest
on each side:

| Stage | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| flow, 262 frames | 409 ms | 276 ms | 1.48x |

Two-thirds of the flow's launches are still the products, now at 80 to
160 blocks each; the next cut is an offset rope and the positional
convolution, whose `grouped_conv1d` is 688 µs a call at 1.8 TFLOP/s
and runs four times an evaluation.

This change was written by the earlier process of this session, which
was still running after the conversation was resumed; it was tested,
measured and committed by the later one.

### The speech LLM's sampler was a tenth of the step

A decode step of the speech LLM was 273 launches and 2.87 ms on the
trace, 2.31 of them the card busy, and the largest single gap was on the
host: 270 µs between the logits coming down and the next token's
embedding going up. The sampler was sorting the whole 6761-way head to
take the first 25 of it - measured alone at 198 µs a token - because
upstream's `ras_sampling` reads the nucleus off a full sort. A partial
selection under the same comparator, with the prefix then sorted, is
exactly the full sort's prefix, 46 µs, and the draw is bit for bit what
it was: the token sequence of the utterance below is identical between
the two builds, hash and all. The full sort remains on the rejection
path, which redraws over the whole distribution and is rare. Two
launches a token went with it: the logits row was copied out before it
was downloaded, and the residual copy was zeroed before it was written.

Same sitting as the flow's table above, the stages example's warm
runs and, for the step itself, a harness that runs only `speech_tokens`
nine times a process, two alternated pairs, worst median on each side:

| Measure | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| speech LLM step, harness median | 2.61 ms/token | 2.49 ms/token | 1.05x |
| speech LLM stage, 131 tokens | 342 ms | 324 ms | 1.06x |
| utterance, 5.24 s of audio, all stages | 832 ms | 675 ms | 1.23x |

The utterance row carries the flow's round and the tiled convolution's
as well as this one; the step row is this change alone.

What is left in the step is the card: eleven launches a layer at one
token, four of them the two closing projections' residual adds and
normalisations, and a decode attention at 18 µs over 131 positions that
is mostly its own floor. The chat model folds those four into the
projections' tails with `gemv_norm`, which exists only for packed
weights; the f16 twin is the next round.

### The speech LLM's closing projections carry their norms: 2.60 to 2.32 ms a token

After the sampler the step was the card's: eleven launches a layer at one
token, four of them the two residual adds and the two `rms_norm`s that
follow the attention output and the MLP down projections, each under four
microseconds and each mostly its own floor. The chat model folds those
into the projections' tails with `gemv_norm`, which reads a packed weight
and an int8 activation; the speech LLM holds f16 weights and quantizes
nothing, so it needed the f16 twin - `gemv_norm_f16`, the same tail on
`gemv`'s f16 column product, in `docs/KERNELS.md`. Seven launches a layer
now, the down projection's tail making the next layer's first normalised
row and the last one the final norm's.

GPU 2, the nine-run harness, two alternated pairs, the c783811 build
against this one, worst median each side, and the utterance's token
sequence identical between them:

| Measure | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| speech LLM step, harness median | 2.60 ms/token | 2.32 ms/token | 1.12x |

Against the 2.61 the sampler's round started from, the step is 1.12x
over the two rounds together. What is left: the decode attention at
about 18 µs a layer over 131 positions, which is a fixed cost the chunk
width does not reach, and the weight stream itself at 1.2 ms a token.

### The flow's positional convolution as a tiled matmul: 275 ms to 250

With both rows batched the flow was 263 ms of kernels in a 273 ms span,
and the profile's largest entry that was not a matmul was the input
embedding's grouped convolution: 16 groups of 64 channels, a kernel of
31, 886 µs a call at 1.8 TFLOP/s and four calls an evaluation, 32 ms an
utterance. The kernel was a thread per output element walking 1984
weights. It is now `conv1d_tiled`'s implicit-matmul body a group a
`blockIdx.z` - the same staging, the same `fmaf` chain in the same
order - at 327 µs a call, and the flow's mel is byte for byte what it was.

The stages example on GPU 2, the same utterance, the cca2d71 build
against this one alternated in two pairs of four runs, the first run of
each set aside, worst of the rest on each side:

| Measure | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| flow, 262 frames | 275 ms | 251 ms | 1.10x |
| utterance, 5.24 s of audio | 656 ms | 633 ms | 1.04x |

The utterance is 8.3x realtime now. The flow's remaining entries are the
products - the stacked q, k and v at 282 µs, the feed-forward pair at
about 150 each, the per-row output projection at 69 - the fused
attention at 47 µs a row, and then small things: the five copies a block
that an offset rope would remove, the gated residual adds that the
matmul's epilogue could carry, the modulation projections read at f32.

### The tiled convolution's second round: 1.15x on VITS

After the tiled convolution the VITS decoder was 23.4 ms of a 28.8 ms
synthesis, and a harness timing the wrapper at the decoder's own shapes
put every one of them at 4.3 to 5.3 TFLOP/s against a 16.3 TFLOP/s f32
card. `ptxas -v` gave the kernel 64 registers and 10.5 KB of shared
memory, four blocks an SM, so the 128-channel k=3 shape is exactly one
wave, and its trips cost about four times their compute floor.

Two changes, both bit-identical (both synthesisers' WAVs byte for byte
the previous engine's). Each trip's operands are now fetched into
registers a trip ahead, so the global round trip hides under the
arithmetic: the decoder 23.4 ms to 21.0. And a second tile gives each
thread eight channels by four positions, halving the shared-memory
loads per multiply-add, for the shapes with 64 channels to fill it.
The harness, `conv1d` at the decoder's shapes on GPU 2, best of three
twenty-launch runs:

| Shape | Before | After |
| --- | ---: | ---: |
| 128 ch, 9216 positions, k=3 | 175 µs, 5.2 TFLOP/s | 119 µs, 7.6 |
| 128 ch, 9216, k=11, dilation 5 | 630 µs, 5.3 | 418 µs, 8.0 |
| 64 ch, 21376, k=3 | 112 µs, 4.7 | 92 µs, 5.7 |
| 64 ch, 21376, k=11, dilation 5 | 390 µs, 4.9 | 309 µs, 6.2 |
| 256 ch, 1344, k=3 | 116 µs, 4.5 | 115 µs, 4.6 |
| 32 ch, 42752, k=3 | 61 µs, 4.3 | 61 µs, 4.3 |

A trip of 32 pairs instead of 16 was measured at every shape and lost
by 2 to 8%; the syncs were not the cost. The 256-channel stage is
parallelism-limited - 84 blocks in the widest tile that reaches a block
an SM - and wants the contraction split rather than a tile; the
32-channel one has no channels to widen into.

`xabe-tts-bench` on `mms-tts-nan`, GPU 2, three alternated pairs of
twenty runs against the eb94f9c bench, worst on each side: **28.72 ms to
24.88**, **1.15x**. Tacotron2 on the same three lines is level - its
convolutions are the 32-channel location filter and a 512-channel
postnet over 200 positions, neither of which takes the wide tile.

### The text encoder's projections as the tiled kernel: 1.06x on VITS

With the decoder at 21 ms the profile of a VITS synthesis had a second
kernel worth reading: `linear`, the thread-per-output matmul the text
encoder runs for q, k, v and the output projection in each of its six
layers, 75 µs a launch at 70 rows of 192 - 1.8 ms of a 24.9 ms run for
2.6 MFLOP a call. A thread there walked its weight row while the
thirty-one lanes beside it walked theirs, so every load touched
thirty-two lines of the weight; at thirty rows of 512 to 1024, which is
what Tacotron2's encoder asks of it once a direction, the same kernel took
394 µs for 31 MFLOP.

The tiled convolution's body already is a matmul, gathering its right
operand as it stages it, so `linear_tiled` is that template with a flag:
the input read a row at a time along its channels and stored transposed
the way the weights are, `k` of one, the output row-major. The sum is
the old kernel's to the bit - bias, inputs ascending, `fmaf` - and both
synthesisers' WAVs are byte for byte the eb94f9c engine's on a long text
and a three-syllable one. The harness, `Gpu::linear` on GPU 2, best of
three fifty-launch runs, against the b1188a8 tree:

| Shape | What it is | Before | After |
| --- | --- | ---: | ---: |
| 70 x 192 -> 192 | text encoder q, k, v, o | 78.8 µs | 14.6 |
| 26 x 192 -> 192 | the same at a short clause | 33.6 | 14.7 |
| 1 x 192 -> 192 | one symbol | 26.5 | 14.3 |
| 30 x 512 -> 1024 | Tacotron2 encoder LSTM gates, a direction | 394.0 | 33.1 |
| 50 x 512 -> 1024 | the same at a longer line | 429.0 | 34.0 |
| 30 x 512 -> 128 | Tacotron2 attention memory | 108.4 | 32.7 |

The tile wins at one row, where it is 31/32 padding, because the old
kernel's cost was the weight read and not the arithmetic; so the wrapper
has no row count at which it keeps the old kernel, and the old kernel is
deleted rather than left unreachable. At 70 rows the tile is 14.6 µs for
2.6 MFLOP, which is a launch and twelve trips of sixteen pairs in
sequence, not a rate; the text encoder's projections are latency now.

`xabe-tts-bench` on `mms-tts-nan`, GPU 2, three alternated pairs of
twenty runs against the b1188a8 bench, worst on each side: **24.76 ms
to 23.30**, **1.06x**. Tacotron2's encoder on the same three lines is
2.19, 2.96 and 3.06 ms to 1.45, 1.99 and 2.05, and its synthesis is level
inside the pairs' spread - the encoder was 2% of it.

### The residual layer's seams folded into the tile: 1.12x on VITS

After the text encoder's round a VITS synthesis was 23.9 ms in 858
launches, and the ones that were not convolutions were worth reading
together: 103 residual adds, 77 leaky ReLUs, 87 copies and 208 memsets,
3.3 ms between them, most of it in the decoder's thirty-six residual
layers. Each ran copy, leaky ReLU, convolution, leaky ReLU, convolution,
add - the copy because the activation was in place and the input was
also the residual - and at the 32-channel stage every one of those
passes is 5.5 MB read and written.

The tiled convolution now takes both seams. With a slope it applies
`x < 0 ? x * slope : x` to each input value as it stages it, which is
`act_leaky_relu`'s formula on the same values; with a residual its store
writes `acc + res`, which is `add_inplace`'s sum in `add_inplace`'s
order. So a layer is two launches, the input is read once and never
written, and the numbers are the same: the test holds the fused launch
bit for bit to the separate kernels on the card, and both synthesisers'
WAVs are byte for byte the previous engine's on a long text and a
three-syllable one.

`xabe-tts-bench` on `mms-tts-nan`, GPU 2, three alternated pairs of
twenty runs against the 56b9f97 bench, worst on each side: **23.51 ms
to 20.90**, **1.12x**. The run is 654 launches from 858 and 19.3 ms of
card time from 21.7; the convolutions themselves cost 0.9 ms more,
because staging a value now compares it, and the passes that went away
were worth 3.3. Tacotron2 does not run this layer and is level.

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

## Isolate the constraint by removing everything else, not by trying alternatives

Six variants of the packed mat-vec were measured and all were worse, and the
conclusion drawn was that the block layout was the cap. It was not. The
measurement that settled it was not another variant: it was a kernel that read
the same bytes in the same pattern with **the arithmetic deleted**, which ran at
the streaming roof and proved the layout was never in the way.

The general lesson: **a run of failed alternatives tells you about those
alternatives and nothing else.** To find what a kernel is bound by, build the
one that does only the suspected part and see how fast it goes. That is one
experiment against six, and unlike them it produces a number with a known
meaning.

## A token is not a matmul, and after a point it is mostly not

At the roof, the chat model's mat-vecs are 8.7 ms of a 12.4 ms token. The other
3.6 ms was small kernels and a CPU that could not issue 1154 launches and 1126
allocations fast enough to keep the card fed - and closing it was worth more
than everything the kernel work had bought.

Most of it was not optimisation at all but bookkeeping that should never have
been there: a KV cache that reallocated and copied itself every layer of every
token, rearranged into attention's layout every step and thrown away; a one-float
argument allocated and zeroed on every call. Nothing profiles as "wrong" - each
kernel is individually fast - and the whole is 40% overhead.

The general lesson: **profile the token, not the kernel.** A kernel summary
sorted by time will show the matmul at the top for as long as you look at it,
and will never show you the gap between the kernels or the work that did not
need doing.

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

### The encoder's fused attention read an f16 cache and did not get faster

`flash_attn_64` was 25.2 ms of the ASR's 115 ms encoder and was doing 5.76
GFLOP a layer, which is 7.3 TFLOP/s - a third of what the tiled `gemm` beside
it manages. The arithmetic said traffic: the encoder's window is 1500
positions in 24 query tiles, so every block re-stages the whole key and value
cache and the kernel reads 361 MB a layer, which at 0.787 ms is 459 GB/s and
looks like the card's ceiling.

So the cache was stored at f16 instead. This is not an approximation and the
test asserted so: the kernel rounds the cache to f16 on its way into shared
memory anyway, `to_f16` rounds the same round-to-nearest-even way, and a
differential test compared the two paths at **exact equality of every output
bit** rather than at a tolerance. It passed, and the ASR oracle tests passed.

The kernel measured 25.0 ms against 25.2, and the encoder 116.4 against 115.2 -
worse, by the 1.5 ms the two conversion passes cost. **The re-reads were never
reaching memory.** One head's keys and values are 768 KB, 144 blocks are
resident at two per SM, and the blocks that share a head are adjacent in the
grid - so the working set is about 4.6 MB against a 6 MB L2, and the twenty-four
re-reads were already L2 hits. Halving a number that was already free bought
exactly nothing.

This is the measurement that redirected the work to shared memory rather than
global, and the round that fixed the kernel is above.

### Nor is the fused attention short of occupancy, taken on its own

The obvious next suspect, after bandwidth: at `QT` 64 the tile takes 28.8 KB of
shared memory, which is two blocks an SM and 50% of the threads this card will
hold. `QT` 32 takes 16.9 KB, which is three - so the kernel was instantiated at
`<64, 32, 32>` with `__launch_bounds__(256, 3)` to let the register allocator
know. The encoder measured **121.1 ms against 115.2**. Worse, and by more than
the earlier `QT` sweep found when it picked 64 at two blocks an SM.

Buying residency with a smaller query tile costs load ratio - `QT` 32 puts the
warp grid at two query groups by four column groups, so a warp owns one key
fragment where `QT` 64 gives it two, and the kernel goes from 1.75 shared words
a product to 2.5. That is what this row measures, and it is why the pair of
rejections here reads as "neither" rather than "not residency": **residency
turned out to dominate, and this experiment could not show it because it paid
for residency in the only currency that mattered more.**

What both rejections did establish is that the constraint was the instruction
mix, and the round above acts on that. Both experiments themselves were
reverted whole; neither entry point is carried unused, and the f16 cache is not
in the shipped kernel.

### The decode `gemv` does not want sixteen-byte weight loads either

"Sixteen bytes a lane" is the finding the packed mat-vec was rebuilt around,
and the f16 path - which is the one the ASR decoder runs, and the only stage
that runs it at scale - still read four. Widening it to a `uint4` of weight
against two `float4` of activation, guarded on the row dividing into whole
16-byte loads, left the ASR decode loop at 78.7 ms against 77.5. Reverted.

The reason is in "Sixteen bytes a lane" itself and was written down before this
was tried: a wide weight load only pays with an activation narrow enough to
keep up, which there meant int8. Here the activation is f32 and stays f32 -
narrowing it to f16 is available and worth 5% on the tiled path, which is not
enough to change the answer. The six attempts already recorded below were all
on the packed path at the chat model's shapes; this is the same conclusion
reached again on the other path and the other model's shapes.

### The tiled `gemm` cannot buy a third resident block

The move that worked for the fused attention - shrink the footprint, get a
third block an SM, take the latency hiding - does not transfer. `gemm` ships
with `__launch_bounds__(GEMM_WARPS * 32)` and no minimum, and ptxas already
chooses two blocks' worth of registers: pinning 2 explicitly measures 110.5 ms
against the unpinned 110.9, inside the noise. Pinning 3 measures **281.7 ms**
and pinning 4 **747.8**. The accumulator alone is 64 floats a thread and
capping registers at the 85 a third block needs spills it to local memory.

Shared memory is not the constraint here - the two staged tiles are 20 KB and
three would fit in the SM's 64 - which is what makes this different from the
attention kernel, where shared was binding and register pressure was not.

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

### Six attempts at the decode gemv, all measured worse

Decode reads every weight once a token, so its `gemv` is the whole of what a
clause costs once prefill is fixed. It runs at 372 GB/s. A kernel that reads the
same buffer and does nothing with it reaches 587, so the gap is real and it is
1.58x. None of these closed any of it, measured standalone at n=14336, k=4096
against the shipped kernel's 88.1 us:

- **The activation staged in shared memory**, once per block instead of once per
  warp - 122.7 us. Two `__syncthreads()` per super-block cost more than the
  traffic they save; the warps stop covering each other's latency.
- **Two output columns per warp**, so the activation registers serve twice the
  weights - 91.6 us. Halves the blocks and the parallelism with them.
- **The 16-byte block header as one `uint4`** instead of eight byte loads, which
  every lane issues identically - 108.8 us. The byte loads broadcast out of L1
  and cost less than the register shuffling to take them apart again.
- **The activation dropped entirely**, as a diagnostic rather than a candidate -
  80.0 us. That is the whole of what the activation loads cost: 9%. They are not
  the limiter, which is why the first two could not have worked.

Two more, added when llama.cpp was measured and turned out to be 1.66x ahead:

- **An int8-quantised activation with `__dp4a`**, which is where llama.cpp's
  mat-vec gets its integer throughput - 101.6 us against the float path's 106.0
  in the same harness, 4%, for 3.7e-3 relative error. **This entry drew the
  wrong conclusion from that number and the conclusion has been overturned.** It
  said `dp4a` was not what makes their kernel fast and that the arithmetic was
  never the limit. The second half is true and the first is not: `dp4a` is
  worth 4% on four-byte loads and 1.5x on sixteen-byte ones, because what it
  buys is not throughput but a *narrow enough activation to keep up with a wide
  weight load*. Neither half works alone. See "Sixteen bytes a lane" above,
  which is the shipped kernel.
- **Four super-blocks of loads issued before any is consumed**, aimed straight
  at the memory-level parallelism their source names - 394.8 us, 4.5x *slower*.
  Four headers and eight `float4` in flight is far past what 64 registers hold,
  and it spills to local memory. This one stands: what the shipped kernel does
  instead is widen each load rather than issue more of them.

This entry used to end by saying what was left was the shape of the kernel, and
that closing the gap meant one output row per thread block with a shared-memory
reduction. That shape was implemented and measured - it is the block-per-column
variant above, 99-105 us against 88.7 - and the gap was closed by something
else entirely.

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

### An integer tiled matmul, built and measured 12x *less* accurate

llama.cpp multiplies a packed weight as integers against an int8 activation and
never forms the dequantized value. This engine's tiled matmul dequantizes to
f16 first. That difference was the leading theory for why the chat model
disagreed with llama-server, and for why its prefill was then 3.5x behind - int8
tensor cores run at twice the f16 rate on this card, and
`m8n8k16.s32.s8.s8.s32` is one of only two mma shapes that assemble on sm_75.

So it was built: a 64x64 tile over a 32-element k-tile, which is exactly one
`Q4_K` sub-block, with the block minimum applied as a rank-1 correction from
the activation's row sums rather than per element.

It went from 193 to 556 tok/s over three fixes, and the profile lessons are
worth more than the kernel:

- Staging the weight with 64 of the 256 threads, each decoding 32 quants in
  sequence, was 193 tok/s. Eight quants a thread across all of them was 361 -
  the header comes out four times a column instead of once, and that is the
  cheaper end of the trade.
- Recomputing the activation's row sums from global memory on every trip
  through the contraction was the rest of it. They come out of `dp4a` against a
  word of ones on the registers the staging already holds: 361 to 556.
- Hoisting the rescale's shared loads out of the inner loop and folding the
  minimum into one `fmaf` bought accuracy rather than speed.

At 556 tok/s it matched the f16 kernel on the chat model and lost 10% on the
translator. And then the accuracy measurement, once it was labelled correctly:

| | integer | f16-staged |
| --- | ---: | ---: |
| Q4_K, 140x1024x512 | 0.0752% | **0.0064%** |
| Q6_K | 0.4811% | **0.0307%** |

**The integer kernel is twelve to fifteen times less accurate**, and the reason
is the operand nobody was looking at: an int8 activation carries 8 bits where
f16 carries 11, and a weight that reaches the multiply exactly does not make
that back. Teacher-forced agreement with llama-server was unchanged at 10 of
105 either way. Removed.

Two things survive it. The first is a correction: an earlier version of this
section reasoned that llama.cpp's exact-integer weights must be *more* accurate
than f16 staging. They are not - llama.cpp's prefill is the coarser of the two,
and this engine tracks the reference better by not copying it. The second is
that a measurement read off an un-labelled number is worth nothing: the
comparison above was run three times with the new kernel switched off by an
environment variable, and reported the f16 path's accuracy as the integer
path's each time.

### Attention in exact f32, which fixed the wrong thing

The batched path stages the two attention matmuls to f16 like everything else;
the one-token-at-a-time path runs them on the scalar mat-vec, which is exact.
That was the last structural difference between the path that agrees with
llama-server at 1 of 105 decisions and the path that agrees at 10, so it was
built: a batched f32 kernel, one thread an output element, no staging.

It worked, at what it actually does. This engine's own two paths went from
forking on **5 of 179 argmaxes to 2** - so rounding the scores and the context
is a real error source and now has a number.

It moved the disagreement with llama-server by nothing at all. Ten before, ten
after, the same ten. And it costs 23% of a prefill and 11% of a decode - the
decode half is pure loss, because at one token attention already runs on the
exact mat-vec and this only replaces it with something slower.

That is the third arithmetic intervention on this problem to change the
disagreement list by zero, after the integer matmul above and pre-rounding the
activation to the int8 grid. All three produced byte-identical lists. What that
rules out is arithmetic in the batched path; what it leaves is recorded in
`docs/TESTING.md`, and it is not yet an answer.

### A normalisation block sized to divide the row exactly

The fused normalise-and-quantise kernel runs one block a row and reads four
floats a thread. At the chat model's 4096 that is 1024 threads and exactly one
iteration; at the translator's 5120 it is two, the second a quarter full.

Sizing the block to divide the row instead - 256 threads and five full
iterations - made the translator *slower*, 55.5 to 55.1 tok/s. An earlier
version of the same kernel at 256 threads with scalar loads lost to the unfused
pair outright.

The general lesson: **at one block a row, the block is all the parallelism there
is, and starving it costs more than any amount of ragged tail.** The waste is
visible and the latency is not, which is what makes this the tempting direction.

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
