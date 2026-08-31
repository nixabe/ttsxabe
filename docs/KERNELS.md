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
| discrete Fourier transform, any length | mel frontend | `xabe_dsp::Fft` | (cpu only) | `xabe-dsp` fft |
| mel filter bank and spectrogram | ASR frontend | `xabe_audio::mel_power` | (cpu only) | `xabe-whisper` frontend |
| tiled matmul, f16 operands | ASR everywhere | `xabe_dsp::linear` | `gemm` | `xabe-cuda` kernels |
| matmul for a handful of rows | ASR decode | `xabe_dsp::linear` | `gemv` | `xabe-cuda` kernels |
| convolution as a matrix | ASR encoder stem | `xabe_dsp::conv1d_strided` | `im2col` + `gemm` | `xabe-cuda` kernels |
| head split, merge and transpose | ASR attention | (index formula, in the test) | `split_heads`, `split_heads_t`, `merge_heads` | `xabe-cuda` kernels |
| causal mask | ASR decoder self-attention | (index formula, in the test) | `causal_mask` | `xabe-cuda` kernels |
| round to f16 | ASR weights and KV cache | `half::f16::from_f32` | `pack_f16` | `xabe-cuda` kernels |
| RMS norm | translator, every layer twice | `xabe_dsp::rms_norm` | `rms_norm` | `xabe-cuda` kernels |
| SiLU, and SiLU-gated multiply | translator MLP | `xabe_dsp::silu`, `silu_mul` | `silu_mul` | `xabe-cuda` kernels |
| rotary position embedding | translator attention | `xabe_dsp::rope` | `rope` | `xabe-cuda` kernels |
| scaled rotary embedding | chat-model attention | `xabe_dsp::rope_scaled` | `rope` | `xabe-cuda` kernels |
| key-value head expansion | chat-model attention | — | `repeat_kv` | `xabe-cuda` kernels |
| int8 quantization of an activation | both Llama stages | `xabe_dsp::quantize_q8` | `quantize_q8` | `xabe-cuda` quant |
| tiled matmul, packed weight and int8 activation | both Llama stages, prefill | `xabe_dsp::linear` on the same approximation | `gemm_i8_q4k`, `gemm_i8_q6k` | `xabe-cuda` quant |
| split-contraction reduction | both Llama stages | (ordered sum, in the test) | `gemm_reduce` | `xabe-cuda` kernels |
| KV cache scatter | both Llama stages | (index formula, in the test) | `cache_append`, `cache_append_t` | `xabe-cuda` kernels |
| fused causal attention | both Llama stages, prefill | (scalar softmax-attention, in the test) | `flash_attn` | `xabe-cuda` kernels |

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

The last five have no `xabe-dsp` twin, and deliberately not. Three are pure
index permutations and one is a comparison, so their reference *is* the index
formula: written beside the assertion it can be read against the kernel, where
exported as a library function nothing calls it would be the same formula
further away. `pack_f16` has a host twin instead - `half::f16::from_f32` - and
the test asserts the two produce identical bits rather than close values.

The DFT is general-radix rather than radix-2, which is not gold-plating:
Whisper wants `n_fft = 400`, and 400 is `2^4 * 5^2`. The recursive mixed-radix
form is barely longer, takes any length, and degrades to a correct O(n^2) sum
on a prime rather than refusing.

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

### Either operand may already be f16

`Operand::F16` hands the kernel a tensor that is already rounded. On the tiled
path this changes *nothing* about the arithmetic and the tests assert exactly
that - bit-identical results, not close ones. `gemm_pack` rounds an F32 operand
to f16 round-to-nearest-even on the way into shared memory on every trip, and
two halves stored contiguously little-endian are the same 32 bits, so an f16
operand is staged with a plain word load and no conversion at all.

What it changes is traffic and residency: Whisper's weights are 3 GB rather
than 6.12, and the ASR's decode loop went from 86 ms to 57 on the strength of
it. Storing them as F32 was buying nothing.

On the *scalar* path it is a real precision decision - `gemv` accumulates an
F32 operand exactly, and rounding one costs about 3.5e-4 relative over a
64-long contraction, measured. That is why `Operand` is a type the caller
chooses rather than something the upload decides.

### And a weight may stay packed

`Operand::Q` hands the kernel the checkpoint's own block-quantized bytes and
unpacks them *inside* the matmul. It is the third storage for the same operand
and the only one that changes what fits on a card.

The distinction it removes is the one `docs/MODEL.md` used to end on. Reading a
quantized GGUF was already possible - `xabe-gguf` decodes nine block formats -
but it decoded them to f32 at load, so a 4-bit checkpoint bought disk and load
bandwidth and *nothing else*: the weights landed at full width and occupied
what f16 occupies. Unpacking per use instead makes the resident copy the packed
one.

The trade is ALU for bandwidth, which is the right way round here because both
kernels are bandwidth-bound. A `Q4_K` element is 4.5 bits against f16's 16, so
weight traffic falls by 3.6x while the arithmetic per element grows by about a
dozen integer ops that overlap with the loads.

**Which elements a lane owns follows the packing, not the contraction.** This is
the whole of the difference between a packed matmul at 86 GB/s and one at 372,
and it is easy to get wrong because the natural choice looks free: give each
lane a contiguous run of the contraction, the way every other kernel here does.
A K-quant byte does not hold contiguous elements. A `Q4_K` byte holds two 32
apart; a `Q6_K` `ql` byte holds two 64 apart and a `qh` byte four 32 apart. So a
lane taking eight adjacent elements loads a byte, uses a nibble or two bits of
it, and drops the rest - which the neighbouring lane then loads again. The read
is coalesced and looks healthy, and the warp still moves two to four times the
bytes it needs.

The fix is to let a lane own whole bytes and take whichever elements those bytes
happen to encode, which for `Q4_K` is four adjacent elements and the four 32
along, and for `Q6_K` two adjacent columns across all four 2-bit fields. It
costs extra scale-group decoding, because the elements of one byte can straddle
sub-blocks, and that is much cheaper than the duplicate fetches.

**Then how *wide* each lane's load is, which is a second question and was worth
more.** Owning whole bytes fetches each byte once; it does not say how many at a
time. Four bytes a lane reaches 373 GB/s and sixteen reaches 578 on a card whose
streaming roof for the same buffer is 587 - and the sixteen-byte number was
measured with the arithmetic deleted, which is what proved the layout had stopped
being the constraint. `docs/BENCHMARKS.md` has the sequence; the short version is
that six attempts to beat the four-byte kernel by rearranging it all failed, and
the one that worked changed nothing about the arrangement.

Sixteen bytes of `Q4_K` is 32 elements, and that is the catch: 32 f32
activations is 128 bytes of scattered reads, which costs more than the wide load
wins. So the activation goes to int8 (below), the dot product becomes four
`__dp4a` against nibble masks that are already four int8 weights in a word, and
the block's minimum comes out as one more `__dp4a` against a word of ones.

`Q6_K` needs one thing more. Its blocks are 210 bytes, so consecutive blocks in
a file sit at every alignment in turn and a 16-byte load is legal on none of
them. `Gpu::upload_quant` re-strides them to 224 - the next multiple of 16 -
which costs 6.7% of the bytes of a `Q6_K` tensor and is **the only place in this
engine where what sits in VRAM is not byte-for-byte what sits in the file**. It
is a stride and not a re-encoding: every block is still the file's own 210 bytes,
at a wider pitch. `Quant::device_stride` is the one function that knows.

Anything added here - a new format, a different grouping - should start from both
questions in order: how many times does the warp read each packed byte, and how
many bytes does one lane ask for at once.

**A weight may be packed; an activation is quantized, which is not the same
thing.** A weight arrives from the checkpoint already in blocks, and
`CudaError::QuantizedActivation` still refuses one as the *left* operand of a
matmul, because nothing produces one there.

What is new is that the activation is quantized at runtime, and only to feed the
wide load above. `Gpu::quantize_activation` writes int8 codes in groups of 32
with one f32 scale a group, codes and scales in one allocation. Three things
about it are deliberate:

- **The scale is `max|a| / 127`,** symmetric, so zero maps to zero and there is
  no zero point to carry a second term for. An all-zero group gets scale zero
  and quantizes exactly rather than dividing by it.
- **It is measured, not assumed.** 0.69% of the chat model's logit span and
  0.42% of the translator's, against the same weights at f16.
  `xabe_dsp::quantize_q8` is the CPU twin and the differential test compares at
  *exact equality* - the thing worth checking is that the two implementations
  approximate identically, and a tolerance there would hide exactly the
  group-boundary or rounding-mode disagreement it exists to catch.
- **It is no longer the engine's *one* approximation.** It was, while only the
  mat-vec read int8. `gemm_i8` reads the same codes, so prefill quantizes its
  activations too - see "The integer matmul" below. The two are the same
  approximation in two kernels, and the figures above cover both.
- **The caller takes it, not the matmul.** A transformer layer feeds one normed
  activation to three projections and another to two, so quantizing per
  projection is four fifths of a launch and an allocation wasted.
  `Operand::F32Q` carries an activation with its twin; `gemm_batched` still
  takes one for itself when handed a plain `Operand::F32`, and checks a
  caller-supplied twin's shape against `m` and `k`, because a mismatched one
  would index another tensor's codes in bounds and return numbers.

**A row must start on a block boundary.** `q_at` finds an element's block by
dividing, so a contraction that is not a whole number of blocks would read into
the previous row's last block - in bounds, and wrong.
`CudaError::RaggedBlock` refuses that shape. GGUF guarantees the
fastest-varying dimension is a whole number of blocks and this checks it
anyway, because "guaranteed by the format" is exactly the assumption worth
failing loudly on.

**The layouts are transcribed twice, so they are pinned to each other.**
`q_elem` in `kernels.rs` is a second transcription of the same block formats
`xabe_gguf::dequant` already carries, and a second transcription is a second
chance to permute a block - which produces a plausible tensor rather than an
error. So `xabe-cuda`'s `tests/quant.rs` compares against that decoder, which
is itself checked against `gguf-py` at exact equality, and does it *element for
element* through the exact f32 path rather than only through a dot product,
because a permutation inside a block is invisible to any check on magnitudes.

Two things the tests found that are worth keeping:

- **A negative scale times a zero quantum is `-0.0`,** and the warp reduction
  adds it to `0.0` and gets `+0.0`. The bit patterns differ and the numbers do
  not, so the element comparison is `==` and not `to_bits()`. Every other value
  is reproduced exactly.
- **A cancelling dot product needs a tolerance on the terms, not the sum.** A
  `Q5_0` row of 512 terms of magnitude 0.3 summed to -3.7e-4, so an ordinary
  reordering difference of 1.1e-5 was 3% of the answer and nothing was wrong.
  The bound used is `k * eps * sum|terms|`, which is still far too tight to
  hide a permuted block: permuting one moves the result by the size of the
  terms, not the size of the rounding.

The rope permutation survives packing for a reason worth naming.
`xabe_llama::gguf::unpermute_rope` never looks *inside* a row - it moves `cols`
contiguous elements at a time - so it is a permutation of whole rows, and
`unpermute_rope_bytes` applies the same shuffle to byte ranges. Without that,
`attn_q` and `attn_k` would have to be unpacked, permuted and repacked, which
would need a quantizer this workspace does not have.

### Any contraction length, and why that took three tries

`k` is unrestricted for F32 operands. It was twice wrongly restricted:

1. "A multiple of 8", on the theory that `m16n8k8` steps the contraction in
   eights. Wrong about the kernel: the staging loop zero-extends a short trip
   and the instruction accumulates the zeros.
2. "Even", which was right about the `float2` staging load - it sits at offset
   `row * k + kk` with `kk` even, so an odd `k` misaligns every row after the
   first - but the fix was to stage an odd `k` scalar, not to refuse it.

Both mattered. Attention contracts over 1500 encoder positions, which is even
and is not a multiple of 8, and over the 1, 2, 3, ... tokens emitted so far,
half of which are odd. An `Operand::F16` still needs an even `k`, because two
halves genuinely share a word - and every contraction in a transformer is even.

The general lesson: **a constraint inherited from an instruction is a claim
about the kernel, and the kernel is where it has to be checked.**

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

Re-swept after the staging rework, on the same encoder shapes and on a 128-row
prefill of both Llama stages, 128x128x32 still wins everything - the closest
were 128x128x64 (chat prefill 1306 against 1352) and 64x128x64 (1307). The
derivation above holds and the sweep did not need to move.

One warning about re-running it: the launch geometry lives in `device.rs` and
the tile in this file, and for one afternoon they disagreed. `n.div_ceil(128)`
was written next to a tile that had become 16x64, so the grid covered a
fraction of the output and every small tile "measured" three times faster by
not computing most of the answer. `kernels::define` now reads `GEMM_MT`,
`GEMM_NT` and `GEMM_WARPS` out of the CUDA source at compile time, so the two
cannot disagree again. The ASR oracle caught it; `xabe-llm-bench` checks no
numbers and did not.

### Splitting the contraction

At prefill the whole `m` dimension is one tile: 128 prompt tokens at
`GEMM_MT = 128` is one row of blocks, so a 1024-wide projection is eight blocks
on 72 SMs. The tile cannot fix that. Shrinking `GEMM_MT` does make more blocks,
but each weight is then dequantized once per block instead of once, which is
what the sweep above is measuring when it rejects small tiles.

`ksplit_for` splits `k` instead. Each slice takes a whole number of `GEMM_KC`
trips, so no slice boundary falls inside a staged tile and the staging loop is
untouched; `gemm_reduce` sums the slices afterwards. Every weight is still read
exactly once, by whichever slice owns its part of the contraction - the split
buys blocks without buying redundancy.

The reduction sums in index order rather than by `atomicAdd`. An atomic
reduction is faster and would pass every tolerance-based test in the workspace,
while making the matmul return slightly different numbers from one run to the
next - and every differential threshold in this workspace would then depend on
how the blocks happened to be scheduled.
`a_split_contraction_is_bit_identical_from_run_to_run` asserts it.

`SM_TARGET = 144` is two blocks an SM on 72 SMs, which is what the register and
shared footprints allow. Raising it measured worse (288 gives chat prefill 1224
against 1350), because the extra slices are short and a slice reads the whole
tile footprint however little of `k` it covers.

The rule has a second regime for the opposite shape: a launch already *over*
one wave but with a straggler tail - the translator's wide projections are 160
or 216 blocks against 144 slots, so most of the card idles while the last few
blocks finish. Splitting there is not about making blocks, it is about
levelling waves, so the rule computes the idle fraction of the last wave and
splits only when it exceeds 0.3, taking the largest factor in 2..4 that keeps
2048 elements of contraction a slice - a deeper floor than the fill regime's
512, because these slices come from projections that were never short of
work. Both constants were fitted to a per-shape sweep on this card, not
derived.

### WHY NOT

- **Do not time a GPU kernel by downloading its result.** The first version of
  `bench-gemm` called `download` to force the queue to drain, and for the wider
  shapes the PCIe copy was *most* of the measurement: 1500x5120 floats is 31 MB,
  about 5 ms at 6 GB/s, against a kernel that runs in under one. Every number in
  the first sweep was the bus, which is why the tile appeared not to matter -
  all seven configurations "measured" 2.9 to 3.5 TFLOP/s. `synchronize`.
- **KC=64 is slower than KC=32**, at every tile shape tried. More contraction
  per staging trip is fewer barriers and also a larger shared footprint; the
  second wins here. Still true after the staging rework: 1306 against 1352.
- **Do not hide the staging behind the mma by prefetching.** Both standard
  forms were built and both lost. Register prefetch took the kernel from 117
  registers to 158 and residency from two blocks an SM to one: 687 tok/s
  against 940. Shared double buffering needs 40960 bytes a block against the
  SM's 64 KB, so also one block: 1178 against 1355, and the encoder shapes fell
  from 19.4 to 11.8 TFLOP/s. sm_75 has no `cp.async`, so a global-to-shared
  copy has to occupy registers or a second buffer, and the second resident
  block was already providing the overlap that either scheme pays for.
  `docs/BENCHMARKS.md` has the numbers for every variant tried.
- **Do not widen `GEMM_NPW` to improve the shared-load to mma ratio.** The mma
  loop issues 18 shared loads per 16 mma, which looks like the binding
  constraint and is not: four warps at 128x128 gives NPW=4 and measures 819
  against 940, and eight warps at 128x256 gives 828. The accumulator array
  grows with NPW and comes out of the same register budget as everything else.

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

## The integer matmul, `gemm_i8`

Two entry points, `gemm_i8_q4k` and `gemm_i8_q6k`, over one templated body. It
replaces the f16 tiled `gemm` wherever the weight is a K-quant and the shape is
past `GEMV_MAX_M` - which is prefill on both Llama stages, and nothing else.

### Why there is a second matmul at all

The f16 kernel was measured at 86% of the card's `m16n8k8.f32.f16.f16.f32` peak.
Its remaining 14% is not enough to catch llama.cpp, so the only way up is the
other reachable shape: `m8n8k16.s32.s8.s8.s32`, which runs at four times the
rate. That is the whole reason, and it costs an approximation - both operands
quantized rather than one rounded. `docs/BENCHMARKS.md` has what the
approximation is worth, including the comparison against llama.cpp's own
integer path on the same file.

### The trip is 64 elements, and that number is not a tuning constant

Two Q4_K sub-blocks. Every part of the kernel's shape follows from it:

- A Q4_K sub-block is 32 elements with one `(d*sc, dmin*mn)` pair, and
  `xabe_dsp::quantize_q8` quantizes activations in groups of 32. So a
  *sub-block* is the span over which every scale is constant, and it is the span
  the integer accumulator may run before anything is converted.
- Q4_K stores the low nibble of a byte at element `j` and the high nibble at
  `j + 32`. A trip of 32 reads sixteen bytes and uses half of each. A trip of 64
  uses both, which halves what the kernel reads from global memory for four
  fifths of the weights in a Q4_K_M file. Worth 1993 to 2292 tok/s.
- 64 elements is also 32 bytes of int8 activation per row, which is one thread's
  worth for a 128-row tile across 256 threads. At 32 the activation staging ran
  on half the block and measured 15% of the kernel.

Q6_K gets the same trip because its *device* layout is built for it.
`Gpu::upload_quant` re-packs every Q6_K block on the way to the card: the low
nibbles are paired 32 elements apart - exactly Q4_K's shape, elements `j` and
`j + 32` in one byte - and the 2-bit high fields are packed one 16-element run
to a word, element `e` at bits `8*(e%4) + 2*(e/4)`. A staged run is then one
aligned sixteen-byte read plus eight bytes of highs, every fetched byte used,
where the file's own grouping - low halves 64 apart, high fields four to a
byte across 128 elements - cost 48 bytes a step and used half of each. The
same 224 bytes as the padded stride, so it costs no VRAM; every device-side
reader decodes this layout and only this layout, and nothing downloads a block
back. It is the one place the engine's copy of a checkpoint is not
byte-for-byte the file's.

Q6_K still converts twice a sub-block where Q4_K converts once - its scale
changes every sixteen elements - but the two conversions fold: the sub-scales
are 8-bit integers, so `sc0*dot0 + sc1*dot1` runs on the integer pipe and one
exact `I2F` replaces two quarter-rate ones. The folded sum stays under 2^24,
which is what keeps the conversion exact.

### `ldmatrix` is a 16-bit instruction and this is an 8-bit matmul

They fit exactly. An 8x8 tile of `b16` is eight rows of sixteen bytes, and after
the load lane `l` holds bytes `4*(l&3) .. +3` of row `l>>2` - which is the
`m8n8k16` fragment layout for both operands, to the byte. So one
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` fetches four int8 fragments, and the
alternative is 32 scalar shared loads a trip.

That constrains the shared stride twice over. It must be a multiple of four
words, so every 16-byte fragment row is 16-byte aligned, which `ldmatrix`
requires. And eight consecutive rows must cover all 32 banks, which needs
`STRIDE mod 32` to be an odd multiple of four: at `WORDS = 16` the choice is 20,
where `20r mod 32` runs 0, 20, 8, 28, 16, 4, 24, 12 and repeats.

### The minimum is a rank-one correction

Q4_K is `w = ds*q - dm`, so

    sum_k a_k w_k  =  ds * sum_k(a_k q_k)  -  dm * sum_k(a_k)

The first term is what the integer `mma` computes. The second needs only the
sum of the activation's codes over the sub-block, which depends on the row and
not on the column - so it is one `dp4a` against a word of ones while staging,
and one multiply-add per output. Q6_K is `d * sc * (q - 32)` and has no minimum
at all; only the `-32` folds into the code.

**`sc` does not fold into the code, and trying was a bug caught before it ran.**
Q6_K's scale reaches 127 and `q - 32` reaches 32, and the product does not fit
in the int8 the tensor core reads. It is applied after the `mma` instead - but
in *int32*, where `sc * dot` does fit, which is the scale fold above: the two
sub-scales multiply their dots on the integer pipe and one conversion covers
both.

### A square-ish warp grid

A warp's shared traffic is `(MS + NPW) * KS` words a trip while `MS * NPW` is
fixed by the `mma` count, so the sum wants the two factors close. Eight warps as
one column strip made that 18 words; as two by four it is 12. As four by two it
is 12 again and measured much worse, because that shape gives each warp eight
columns and four rows, and the column scales are the ones that have to be
re-read.

### The weight header is read once a super-block

A staging thread reads the same weight row on every trip, so it crosses a
super-block only once in four - and everything it needs from the header is
sixteen bytes, at the front for Q4_K and at byte 192 for Q6_K. Caching those in
registers, keyed on the super-block index, was worth 1657 to 1869 tok/s. It is
the same trick `gemm` uses.

The Q6_K scale is one of sixteen cached bytes chosen by a loop variable, and it
is extracted with selects rather than a subscript: indexing a register array by
a runtime value puts the whole array in local memory, which is the thing the
cache exists to avoid.

### Blocks are ordered by row tile, not column tile

`blockIdx.x` is the row tile. The blocks that share a weight tile are the ones
that differ in `m`, so making them consecutive puts them on the machine together
and lets L2 serve the weight to all but the first. Worth about 1%, and free.

### A batch may share its left operand

`Batch::a == 0` with a count above one means every matrix of the batch
multiplies the *same* activation. The attention projections are issued that way
- q, k and v against one normalised input - and it changes two things in the
packed paths, both of which were bugs before they were features.

Both `gemv` and `gemm_i8` address the int8 codes densely as `[batch, m, k]`,
derived from a row count rather than from `Batch::a`, because `quantize_q8`
writes them that way. A shared activation is quantized *once*, so that row
count has to be zero rather than `m` - otherwise the second matrix of the group
reads past the end of the codes and the third reads further. `gemv` was the one
that found this: the tiled kernel was fixed first, and a two-token prompt runs
on the mat-vec, so the chat model's block outputs diverged at layer 7 - the
first layer whose `attn_v` is Q4_K and therefore fuses with `attn_k`.

The other thing is the quantiser itself: `q_rows` is `m` and not `count * m`,
which is the same numbers two or three times over.

### The split-k partial is not zeroed

Every slice assigns every element of the tile it owns - a slice with no
contraction left assigns the zero it started with - so the memset was one pass
over `ksplit * m * n` floats that nothing ever read. Both tiled kernels have
this property and the test `every_output_element_is_written_exactly_once` is
what it rests on.

## Fused causal attention, `flash_attn`

Scores, mask, softmax and the value product for a whole prompt, in one kernel,
with nothing materialised. Both Llama stages take it for any multi-token pass;
a single decode step keeps the unfused chain, whose score row is one `gemv`
and has nothing to fuse.

### What it replaces, and why that was worth a kernel

The unfused chain wrote the score matrix, read it back to softmax it, wrote
the probabilities, and read them again for the value product -
`heads * tq * tk` floats three times over - plus a head split before and a
merge after, and the chat model's `repeat_kv` expanding the grouped cache
four-for-one on top. At 512 tokens on the 13 B that chain measured about 27 ms
of a 320 ms prefill, and the score tensor exists only so the softmax can find
its row maximum. The running-maximum trick removes that need: fold each tile's
maximum into a running one, rescale the output accumulator by
`exp(m_old - m_new)`, and no tile's scores outlive the iteration that made
them.

### Shape

One block owns 32 query rows of one head and walks the keys 32 at a time.
Scores by `m16n8k8` into f32, probabilities rounded to f16 on their way into
the value product, one more `m16n8k8` against the values, the output
accumulator in registers throughout. The loop's upper bound is the last key
its rows may see, so the upper triangle is never computed at all - the fusion
gets the triangle skip that would have been a special case in the tiled gemm
for free.

The layouts are the caches' own. K is `[kv_head][pos][hd]`, which is the
`[n][k]` shape the score product's B fragment wants; V is `[kv_head][hd][cap]`,
which is the same shape for the value product; the queries are read straight
out of the projection buffer and the merged context written straight back in
`[tq, heads * hd]`. So `split_heads`, `merge_heads` and `repeat_kv` all
disappear from the prompt path rather than getting faster. A grouped-query
model maps `head / (heads / kv_heads)` and reads the one cached copy.

`hd` is fixed at 128 - both Llama stages' - and the wrapper refuses anything
else by construction rather than by tolerance: another width would index
another head's values, in bounds, and return plausible context.

### The arithmetic is the chain's, deliberately

Operands round to f16 where the tiled `gemm` rounded them, scores accumulate
in f32, `__expf` because that is what `softmax_causal` uses, probabilities
round to f16 exactly where the chain rounded them - on their way into the
value product - and the normaliser sums unrounded f32. The differential test
compares against a scalar reference with the same roundings, on a *peaked*
softmax: near-uniform scores would let a permuted position hide inside the
tolerance, and a permuted position is precisely what the test exists to catch.

## The three kernels the translator added

They are small and none of them needed a design decision, with one exception
that cost an afternoon.

`rope` first used `__sincosf`, the hardware approximation. It matched at low
positions and drifted to 6e-4 by position 4095, because `__sincosf` performs
**no argument reduction** - at 4095 radians the argument is thousands of times
larger than the range the instruction is accurate over. Switching to `sincosf`,
`powf` and `expf` fixed it, and the residual 2.3e-4 that remains is not a bug
to chase: `inv_freq` differs from the reference by one ulp, and one ulp times
4095 radians *is* 2e-4. The tolerance says so out loud -
`1e-5 + first as f32 * f32::EPSILON` - rather than being a round number picked
to make the test pass.

The scalar twin computes `inv_freq` in **f32**, deliberately, even though f64
would be more accurate. The reference computes it in f32. A twin that is more
accurate than the thing it is checking is not a twin.


## Two kernels the chat model needed, and what each is guarding against

**`rope_scaled` is `rope` with a per-pair divisor**, which is Llama-3.1's
frequency scaling. It is the same kernel — the divisor arrives as a pointer plus
a flag, so the unscaled path costs one branch — and the flag is there rather
than a null pointer because every launch argument has to point at something
real, so the no-scaling case passes a one-element dummy the kernel is told never
to read.

The reason it is load-bearing is that Breeze2's `rope_freqs.weight` is **not all
ones**: 1.0 for the first 29 pairs, a ramp through six, then 8.0 for the rest.
A defaulted divisor gives a model fluent for one sentence and drifting after it,
which no shape check catches and no short test notices.

**`repeat_kv` expanded 8 key-value heads to 32 query heads.** It is a gather and
nothing more, and the chat and translator paths no longer call it: the grouped
heads are the *batch dimension* of a matmul that was already batched. Eight
products of four query rows each, reading one copy of each key head four times,
instead of thirty-two products against a key tensor expanded to match. The query
rows line up for free, because a head split lays them out `[head][t][d]` and the
`group * n` rows one key head serves are contiguous — and for a single decode
step they are contiguous already, which is why the split is skipped there too.
The kernel stays for the ASR's cross-attention.

That is one of five layout kernels the attention no longer runs per layer per
token. The other four went the same way, by **storing the cache in the layout
attention reads** rather than the one the projection produces: keys
`[kv_heads, capacity, head_dim]`, values `[kv_heads, head_dim, capacity]`.
`cache_append` scatters a step straight into either, at the one moment the data
is small. What it replaces is a head split for the keys, a transposed split for
the values, and two `repeat_kv` expansions, every one of them building a tensor
that was thrown away before the next token.

The value side is why `Batch` has a `w_row`. A head's values are
`[head_dim, capacity]` and attention contracts over the `tk` positions that
exist, so a row of the operand is a *capacity* apart, not a `tk`. Without a row
stride the cache would have to be exactly as long as the context, which is what
made the old one reallocate and copy itself every step.

The cache is held at the **narrow** width, not expanded. Storing the expansion
would quadruple it for no information — the four query heads in a group read the
same key-value head — which at 8k tokens across 32 layers is 6 GB against 1.5.
It also has **capacity distinct from length**, doubling on growth. Growing it by
exactly the tokens added meant an allocation, a zeroing and a full copy of the
whole cache for every layer of every token: quadratic in the context, and 128
allocations and 16 MB of copying a token at 64 of it.

**Three kernels fold work into a pass something else was making anyway.**
`rms_norm` takes an optional residual to add and an optional int8 twin to emit;
`silu_mul` takes the twin too; `softmax_causal` does the scale, the causal mask
and the softmax in one. None of these is a new algorithm and all of them are
worth more than they look: at a single decode row the row is 16 KB and the
kernel is one block, so what costs is the launch and the latency, not the work.
`rms_norm` also reads four floats a thread rather than one, for the same reason
the mat-vec does.

The shapes these fusions need are the reason they carry constructor checks: a
scale group is 32 columns and a warp must own a whole number of them, so
`rms_norm_q` refuses a width that is not a multiple of 32 and `silu_mul_q` one
that is not a multiple of the block. The general `rms_norm` keeps a scalar path
for ragged widths, because narrowing a public kernel's contract to fit an
optimisation is the wrong trade.

The trap next to it has no kernel of its own: **`rope` on `k` runs over
`kv_heads`, not `heads`.** The kernel walks the tensor as `heads * head_dim` per
position, so passing the query count reads 4096 floats out of a 1024-wide row
and rotates four positions' keys as if they were one position's heads. The
buffer is the right length in total, so nothing checks it and nothing crashes.

## The eleven kernels CosyVoice3 added

Five are activations, and each is in the checkpoint for a reason that is not
interchangeable with another's:

| kernel | what it is | where |
| --- | --- | --- |
| `act_snake` | `x + sin²(αx)/α`, with a **learned per-channel α** | every residual block of the vocoder |
| `act_elu` | `x` above zero, `exp(x) - 1` below | between the F0 predictor's five convolutions |
| `act_mish` | `x · tanh(softplus(x))` | the flow's convolutional positional embedding |
| `act_gelu_tanh` | GELU's **tanh approximation**, not the erf form | the DiT's feed-forward |
| `act_silu` | `x · σ(x)` | the DiT's timestep MLP |

`act_gelu_tanh` is a different function from `gelu`, not a faster one: they
agree to about 1e-3, and callers ask for the one their checkpoint was fitted
against. Snake is the only one with parameters, and α is per channel — 72 of
them across the vocoder — so it is a pointer argument rather than a constant.

Six are structural, and three of them exist because a shape that looks familiar
is not:

**`upsample_nearest` and `strided_conv1d`, because HiFT's `ups` is not a
transposed convolution.** It is a nearest-neighbour upsample followed by a
causal convolution. `repeat_kv` looks like it would do the upsample — it is a
gather along a contiguous axis — and it does not: it repeats *heads*, and this
repeats *positions* inside a channel.

**`grouped_conv1d`, because the flow's positional embedding is grouped.** 1024
channels in 16 groups, which is neither a dense convolution nor the depthwise
one `depthwise_conv1d` already had. Written rather than expressed as 16 dense
convolutions, because the launch overhead would have dominated a kernel this
small.

**`rope_gptj`, because the DiT's rotary embedding is partial and interleaved.**
Only the first 64 of 1024 dimensions are rotated, the pairs are `(2j, 2j+1)`
rather than `(j, j + d/2)`, and it is applied **before** the heads are split —
so one head of sixteen carries position and the other fifteen do not. Three
things that each look like a bug and are all deliberate, which is why this is
its own kernel rather than a flag on `rope`.

**`stft_dft` and `istft_ola`, because the vocoder's output head is an inverse
transform.** `conv_post` emits 18 channels, which is magnitude and phase over
the 9 bins of a 16-point transform with hop 4. Small enough that a direct DFT
beats a radix-2 plan, and the differential test is against `xabe_dsp::istft`
like every other kernel here.
