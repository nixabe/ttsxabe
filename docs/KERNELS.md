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
| conv1d, stride one from 32 positions | decoder resblocks, Tacotron2's encoder, postnet and location conv, CosyVoice's look-ahead | `xabe_dsp::conv1d` | `conv1d_tiled` (three tile widths) | `xabe-cuda` conv1d |
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
| fused attention | both Llama stages, prefill; the Whisper encoder | (scalar softmax-attention, in the test) | `flash_attn`, `flash_attn_64` | `xabe-cuda` kernels |
| single-row decode attention, with the context's int8 twin | both Llama stages, decode; the Whisper decoder, both attentions | (scalar softmax-attention, in the test); `quantize_q8` for the twin | `attn_decode_h128` at three chunk widths, `attn_decode_h64`, `attn_decode_f64` | `xabe-cuda` kernels |
| packed embedding gather | both Llama stages | `xabe_gguf::dequantize_blocks` | `embed_q` | `xabe-cuda` quant |
| mat-vec with a placed, activated epilogue | ASR decode | the mat-vec, `cache_append` and `gelu` in turn | `gemv` with `OutLayout` | `xabe-cuda` kernels |
| rotate-and-cache at one position | both Llama stages, decode | `rope_scaled` twice and `cache_append_f16` twice, in the test | `rope_cache_f16` | `xabe-cuda` kernels |
| mat-vec with the residual add and the next normalisation in its tail | both Llama stages, decode | the mat-vec, then `xabe_dsp::rms_norm` and `quantize_q8` | `gemv_norm` | `xabe-cuda` quant |
| the same, for an f16 weight and no twin | CosyVoice3 speech LLM, decode | the mat-vec, then `xabe_dsp::rms_norm` | `gemv_norm_f16` | `xabe-cuda` quant |
| f16 mat-vec with the residual add and the next layer normalisation in its tail | ASR decode | the mat-vec, then `xabe_dsp::layer_norm_add` | `gemv_ln` | `xabe-cuda` kernels |
| stacked q/k/v mat-vec, placed into the caches | ASR decode | `gemv_into` three times, in the test | `gemv_qkv_f16` | `xabe-cuda` kernels |
| packed mat-vec over several int8 rows, one weight stream | the translator, batched decode | `gemv` a row at a time, in the test | `gemv_q_rows2`, `gemv_q_rows3`, `gemv_q_rows4` | `xabe-cuda` quant |

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
  its differential test is really checking. The gather is a thread per eight
  consecutive windows of one channel at one phase, because samples a stride
  apart share every tap: a thread per sample read each weight once per sample
  and ran at 0.6 TFLOP/s on WaveGlow's upsample, and this reads it once per
  eight, bit for bit the same sums - 4.6 ms to 0.97, and 1.085x on a VITS
  synthesis whose decoder is four of these.
- **`conv1d_tiled` is the same convolution as an implicit matmul.** From 32
  positions a stride-one convolution is computed as a `[out_ch, in_ch * k]`
  by `[in_ch * k, out_t]` product on the f32 cores, the right operand
  gathered from the input as it is staged: a block of 32 channels by 128,
  64 or 32 positions, sixteen (channel, tap) pairs a trip through shared
  memory, sixteen multiply-adds per staged value. The sums are `conv1d`'s
  to the bit - bias first, pairs ascending, each an `fmaf` - so the same
  differential test covers both, at eight shapes chosen to be ragged on
  every axis, and both synthesisers' audio was byte-identical across the
  change. 1.48x on a VITS synthesis and 1.03x on Tacotron2; see
  `docs/BENCHMARKS.md`.
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
| `taco_energies`, `taco_context` | Tacotron2's location attention, one frame; the context written into the three buffers that read it | a CPU chain of the seven kernels they replace | two launches where there were seven and a transpose |
| `layer_norm_mod`, `gate_add` | the DiT's adaptive normalisation and gated residual | `xabe_dsp::layer_norm` on `1 + scale` and `shift`; the host loop, exactly | the flow's residual stream never leaves the card |
| `embed_scaled_f16` | the speech LLM's tables at f16 | `embed_scaled` on the table rounded on the host, exactly | a 544 MB table read a few rows at a time |
| `relu_mask`, `taco_emit` | the Tacotron2 decode loop's bookkeeping | the copies and elementwise ops they replace | each one launch where there were two or three |
| `gated_cond_rows` | WaveGlow's gate with its conditioning added on the way in | `add_strided` then `gated_activation_rows`, exactly | one read of the activation where there were a read, a write and a read |

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

### Why f32 accumulation is not caution, and on this card is not a cost either

fp16 *operands* are safe; fp16 *accumulation* is not. `llmxabe` records the
measurement: on IID-random data fp16 accumulation looks safe at every depth with
26-30x headroom, and it then broke an adversarial differential test by 209x,
with constant-input error growing monotonically 3.2e-2 at 8K to 7.3e-1 at 131K.
Rescale cadence does not help. So `m16n8k8.f32...f32` is the shape used, and the
operand rounding it does cost is measured rather than assumed: 6.5e-5 of full
scale on a k=1280 contraction.

**What that refusal costs on this card is nothing, and this file used to say
otherwise.** Measured back to back out of registers, this Quadro RTX 8000 runs:

| instruction | rate |
| --- | ---: |
| `mma.m16n8k8.row.col.f32.f16.f16.f32` | 102.3 TFLOP/s |
| `mma.m16n8k8.row.col.f16.f16.f16.f16` | 103.0 TFLOP/s |
| `mma.m8n8k16.row.col.s32.s8.s8.s32` | 203 TOPS |

Flat across one, two, four and eight independent accumulator chains a thread,
so it is a throughput ceiling and not a latency artefact, and 78-87% of the
card's 130.5 TFLOP/s and 261 TOPS - which is what a register-resident
microbenchmark should reach.

**f32 accumulation costs 0.7%.** The half-rate FP32 accumulate that Turing is
known for is a GeForce restriction; this is a Quadro and does not have it. Two
numbers recorded here were derived from assuming it did, and both are wrong:
65.3 TFLOP/s for the f32-accumulate shape - exactly half of 130.5 - and "four
times that" for the integer shape, which is twice.

Two conclusions have to be withdrawn with them. Adopting fp16 accumulation
would buy under 1% here, so it is not a lever anyone should reach for, whatever
they think of the accuracy; and "the f16 kernel is at 86% of its own ceiling so
no amount of work on the staging reaches llama.cpp" was computed against 65.3 -
against 102.3 the same measurement puts the `mma` at about 55% of the kernel's
time, which is a very different amount of room.

**Two caveats, and they matter.** The rate above is a *register-resident* one:
it says what the instruction issues at, not what any kernel that also has to
feed itself from memory can reach. And the tiled `gemm` is not close to it for
reasons that are not this instruction - it measures 22.4 TFLOP/s at the ASR's
projection shape, which is 22% of the ceiling but 78% of what the 128x128
tile's arithmetic intensity allows against this card's 672 GB/s. Rounding the
activations to f16, which halves the larger of the two operand streams, was
worth 5%, so it is not simply the memory system either. Where the rest goes is
the subject of the next section, and it is now established.

### Where the tiled `gemm`'s time goes: the register file

This used to say the question could not be settled, because `ncu` cannot be run
on this machine - `ERR_NVGPUCTRPERM`, the account lacks GPU performance-counter
permission. **`ptxas -v` settles it without counters**, and the answer is an
occupancy limit that no amount of work on the staging can lift.

The kernel compiles to **exactly 128 registers a thread, no spill**. That is
not a coincidental number: the register file is 65536 a SM, the block is 256
threads, and two blocks a SM is `65536 / (2 * 256) = 128`. The kernel sits on
the boundary. **64 of those 128 are accumulators** - `GEMM_MSTEPS` 8 by
`GEMM_NPW` 2 by 4 - and a 128x128 tile spread over 256 threads cannot give any
of them up, because 64 outputs a thread is what the tile *is*.

So half the register file is the answer sheet, and the trip has to be staged
inside the other half.

That is why the obvious fix does not work. The loop is `stage; sync; mma; sync`,
with no overlap: every thread waits for the slowest global load of the trip and
only then computes, and the staging is most of the time - with it deleted
(wrong results, timing only) the same mma loop ran a prefill in 41 ms against
107 with it. The textbook answer on an architecture with no `cp.async` is to
software-pipeline through registers - issue trip `kc + 1`'s global loads before
trip `kc`'s mma, so the latency is paid under arithmetic. It was written and
measured, at the encoder's three shapes, medians of 20 on one Quadro RTX 8000:

| | q/k/v/o | mlp up | mlp down | registers | blocks/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x128, as shipped | **22.5** | **22.0** | **25.1** | 128 | 2 |
| 128x128, pipelined | 11.7 | 12.9 | 12.7 | 184 | 1 |
| 128x128, pipelined, `__launch_bounds__(256, 2)` | 15.3 | 13.6 | 16.9 | 128 + 108 B spill | 2 |
| 128x64, pipelined | 16.7 | 18.6 | 18.1 | 122 | 2 |
| 64x128, pipelined | 12.6 | 14.0 | 13.8 | 128 | 2 |
| 64x64, pipelined | 13.4 | 14.8 | 14.4 | 80 | 3 |

TFLOP/s, higher is better. **Every arrangement of it loses**, and the two ways
it loses are the same fact seen twice.

Buffering one trip costs 56 registers, so at the shipped tile the kernel goes to
184 and the SM holds *one* block instead of two. Occupancy is the only
latency-hiding sm_75 has, so halving it to buy a pipeline that hides latency is
a trade that cannot come out ahead - and it does not, at exactly half the
throughput. Forcing two blocks back with `__launch_bounds__` does not rescue it
either: ptxas then spills 108 bytes a thread and lands between the two.

Shrinking the tile does make room - 128x64 buffers a trip in 122 registers and
keeps two blocks - but a smaller tile reads more memory for the same arithmetic,
and that costs more than the pipeline returns. 128x64 gives up a third of the
128x128 tile's flops per byte and comes back at 18.6 against 22.0.

The generalisation is worth stating plainly, because it is a property of the
architecture rather than of this code: sm_75 has **4 registers a SM per output
of a 128x128 tile**, and at the two blocks that latency-hiding needs, the
accumulators alone are half of them. There is no room in the other half for a
staged trip. A deep pipeline here needs `cp.async`, which stages global to
shared without a register in between and arrived with sm_80.

That is the honest end of this line of work. The remaining distance to cuBLAS is
not a missing trick in the staging loop; it is an architecture that this kernel
shape has run out of room on.

## The integer matmul, `gemm_i8`

Two entry points, `gemm_i8_q4k` and `gemm_i8_q6k`, over one templated body. It
replaces the f16 tiled `gemm` wherever the weight is a K-quant and the shape is
past `GEMV_MAX_M` - which is prefill on both Llama stages, and nothing else.

### Why there is a second matmul at all

The f16 kernel was measured at 86% of what was then believed to be the card's
`m16n8k8.f32.f16.f16.f32` peak, leaving 14% that would not have been enough to
catch llama.cpp. **That baseline was wrong** - see "Why f32 accumulation is not
caution" above; the instruction runs at 102.3 TFLOP/s, not 65.3, and the same
`mma` time is about 55% of the kernel rather than 86%. The integer shape is
twice the f16 rate, not four times.

Neither correction undoes this kernel. `gemm_i8` still measures what
`docs/BENCHMARKS.md` records, twice the arithmetic rate is still the largest
single lever available on this card, and llama.cpp still takes the integer path.
What changes is the claim that the f16 kernel had nothing left: it had more than
was recorded, and how much of it is reachable is not established. It costs an
approximation - both operands quantized rather than one rounded. `docs/BENCHMARKS.md` has what the
approximation is worth, including the comparison against llama.cpp's own
integer path on the same file.

### Two row tiles, because a prefill computes rows it does not have

A block owns `MT` rows of the activation and computes all of them whether `m`
reaches that far or not - the staging skips the missing rows, the `mma` does
not. At `MT = 128` and a twenty-four-token clause that is five sixths of the
arithmetic spent on nothing, and it is nearly all of what a short prefill
costs: `m = 24` and `m = 128` do the identical `mma` work and measured 69.8 ms
against 90.5, so the part that scales with real rows is small and the padded
part is about 65 of the 70.

So the tile is a template parameter, with a second pair of entry points at 64.
**Sixty-four is a floor and not a choice.** The activation fragment load is an
`ldmatrix .x4`, which takes four row groups of eight at once, so a warp's share
`MR` cannot be under 32 rows; two warps down the tile makes 64 the narrowest
`MT` this warp grid admits. A `static_assert` on `MS % 4 == 0` says so at
compile time rather than letting a narrower tile silently drop its `m4` loop.

The narrow entry costs 121-124 registers against the wide one's 128, and 19456
bytes of shared against 25600 - so it keeps the same two blocks a SM, and the
whole of the gain is arithmetic not done. `docs/BENCHMARKS.md` has what it is
worth: 27% off a twenty-four-token prefill, and 5% off the first clause of a
turn, which is what first audio waits on.

The choice is `m <= 64`, made where the kernel name is picked. Below that the
wide tile would be computing a majority of nothing; above it the narrow one
would double the block count for the same work.

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

## A transpose is a transpose, `split_heads_t`

`[t, heads*head_dim]` to `[heads, head_dim, t]` reads like a head permutation
and is not one. Element `(ti, c)` goes to `(c, ti)` for `c = h * head_dim + j`,
and `h` and `j` never appear apart in that expression - the head structure
cancels, and what is left is the transpose of a `[t, d]` matrix. Seeing that is
the whole fix, because a transpose has a well-known kernel and a permutation
invites an element-at-a-time scatter.

An element a thread reads coalesced and writes scattered: consecutive lanes
land `t` floats apart, so one warp store is 32 sectors where a coalesced one is
four. It measured **141 GB/s** on a card that copies at about 500, which at
encoder width is 3.5 ms across 32 layers for a pass that carries no arithmetic.

Staging a 32x32 tile in shared makes both halves coalesced and took it to
**0.042 ms** from 0.109, a shape-for-shape 2.6x. The tile row is **33** words
rather than 32, and that padding is the trick rather than a detail: the
write-back reads a *column* of the tile, and at a stride of 32 a column is one
bank and the read is 32-way conflicted. At 33 it walks all 32 banks exactly
once.

The f16 twin is the same kernel with a narrower store. Both now allocate their
output with `uninit`: every element is written, so the zeroing pass was
establishing nothing, and at encoder width it was 7.7 MB of it.

## Narrowing where the value is already in a register

`layer_norm_add_f16` and `act_gelu_f16` are the f32 kernels with `f32_to_f16`
on the store, and they exist because of an asymmetry that is easy to get
backwards.

The tiled matmul stages its left operand as f16 whatever width it arrives at.
So handing it f16 changes no arithmetic at all - `f32_to_f16` is the same
round-to-nearest-even `gemm_pack` applies during staging - and halves the
stream it re-reads once per column tile, which is ten times at encoder width
and forty at the MLP's. Measured at about **5% of each matmul**.

The asymmetry is that a rounding *pass* of its own costs more than one matmul
saves: reading 7.7 MB and writing 3.8 to save 5% of a 0.26 ms call is a loss,
and it was correctly rejected on that basis once. Fused into the kernel that
already has the value in a register it is free, and the same 5% is a win. So
the rule is not "narrow activations" but **narrow them where something is
already touching them**, or where the tensor is read many times:

- inside the encoder, fused into the normalisation and the GELU that produce
  the activation, because each result feeds one to four projections;
- in front of the cross-attention cache, as a pass of its own, because there
  the encoder output is read by *sixty-four* projections and one conversion
  serves all of them.

Below `GEMV_MAX_M` none of this applies - a mat-vec has no re-read to halve -
so the decoder takes the f32 path and the choice is made on the row count.

`h` stays f32 throughout. It is the residual stream, it is added to rather than
multiplied by, and narrowing it would be a real approximation rather than a
free one.

## `gemv_rows`, where the weight is not the weight

`gemv` gives output row `r` its own block through `blockIdx.y`. That is right
whenever the "weight" is a checkpoint tensor: `m` is one at decode, the weight
is the entire traffic, and there is nothing to share.

Attention is the case where it is wrong. The weight is the KV cache, and the
`m` rows are the query heads of one grouped-query group - heads that exist
precisely so that they can share a key head. Four blocks each fetching the same
cache is four times the traffic for arithmetic that was meant to be free, and
measured on this card it cost 2.4x rather than the 1.0x sharing would give:
L2 does not absorb it behind a 4.92 GB weight stream. `docs/BENCHMARKS.md` has
the sweep and the L2 argument that turned out not to apply.

So `gemv_rows` loads `wv[i]` once and spends it on every row:

```cuda
for (int i = lane; i < k; i += 32) {
    const float wi = wv[i];
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_MAX; ++r) {
        if (r < m) { acc[r] += af[(size_t)r * k + i] * wi; }
    }
}
```

`m` is a runtime count, so the row loop is unrolled against the compile-time
bound and predicated on it - an unrolled body is what makes the single load
pay, because the four products then issue back to back off one register.
`GEMV_ROWS_MAX` is 4 and `GEMV_MAX_M` must not exceed it; a `debug_assert` at
the launch says so.

**f32 or f16 on the right, and no packed path.** A packed weight is a
checkpoint tensor and has no case here. The KV cache is the only *unpacked*
operand that ever appears on the right of a mat-vec in these models, which is
why this kernel is narrow enough to be worth having rather than a second copy
of `gemv`. It is f16 now, so the loop has both widths - and the f16 one carries
the odd tail, below.

Two things it deliberately is not:

- **Not a fused decode attention.** Scores, softmax and the value product stay
  three kernels. Fusing them would need a split over the key range and a
  combining pass, because eight key heads is eight blocks on 72 SMs; the three
  kernels already have thousands.
- **Not a change to the arithmetic.** Lane `l` sums the same elements of `k` in
  the same order it would have, and the reduction after is the same tree. The
  test asks for **bit equality** against the same rows computed one at a time -
  a tolerance there would be hiding the only thing worth checking - and gets it
  on five shapes, including the strided `w_row` layout the value cache uses.

## The f16 KV cache, and the odd contraction it brings

Both Llama stages hold their caches at f16. That is five kernels with twins -
the two appends, the growth re-stride, both mat-vecs and the fused prefill
attention - and one thing that had never come up before.

**An f16 *weight* is never an odd length. An f16 *cache* is, half the time.**
Every f16 path here addresses 32-bit words of two elements, and
`RaggedContraction` refuses an odd `k` because a weight is a checkpoint tensor
whose contraction is a model dimension and always even. A value cache is not a
weight: the context product contracts over however many positions have been
decoded, which is 1, then 2, then 3. So the two mat-vecs read the last element
as a lone half - `wv[kh]`'s low half against `af[k - 1]`, by one lane, after
the pair loop - and only they are let through the refusal. The tiled kernel
keeps it, because it stages whole words throughout and an odd `k` has no layout
in it at all.

Two invariants make the addressing work, and both are refused rather than
assumed:

- **An even capacity.** The value layout puts a head's row a capacity apart, so
  an odd capacity lands every other row's pairs across a word boundary. That
  reads *in bounds* and returns the wrong two numbers - fluent text off a cache
  that is quietly wrong - so `OddCacheCapacity` rejects it where the buffer is
  made. Capacities double from 256 and are never odd; the check costs nothing
  and says what it depends on.
- **An even row stride.** `w_row` used to be refused on anything but f32.
  An f16 value cache is exactly that case, so the f16 paths honour it halved.

**`flash_attn` reading an f16 cache is simpler, not harder.** Both of its
staging loops already round what they read to f16 on the way into shared
memory, so an f16 source removes a conversion rather than adding one: the key
loop becomes a word copy. It is a template argument (`KVH`) and not a flag,
because those loops run once a key tile.

What this is worth, and where it is not, is in docs/BENCHMARKS.md: 4 GiB on the
translator and 512 MiB on the chat model, 6.6% of the translator's decode at a
1024 context, and **nothing on the chat model's** - whose score product
contracts over one head width and is latency bound rather than bandwidth bound
at four rows.

## Fused attention, `flash_attn`

Scores, mask, softmax and the value product for a whole prompt - or a whole
encoder window - in one kernel, with nothing materialised. Both Llama stages
take it for any multi-token pass and the Whisper encoder takes it for every
layer; a single decode step takes `attn_decode` below, which is organised
around a different constraint.

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

One block owns QT query rows of one head and walks the keys KT at a time.
Scores by `m16n8k8` into f32, probabilities rounded to f16 on their way into
the value product, one more `m16n8k8` against the values, the output
accumulator in registers throughout. When the pass is causal the loop's upper
bound is the last key its rows may see, so the upper triangle is never
computed at all - the fusion gets the triangle skip that would have been a
special case in the tiled gemm for free.

**Causality is a flag, not a shape.** The encoder attends over its whole
1500-position window, and all that costs is a per-row limit of the key count
instead of the row's own position. It is worth saying which direction is the
dangerous one: the causal loop bound stops at the last tile a row can see, so
a masking mistake there truncates the row and shows up, while with the mask
open the kernel reads every tile either way and a wrong per-row limit would
only shift *which* keys are summed - still a full softmax, still plausible
context. The differential test drives both modes for that reason.

The layouts are the caches' own. K is `[kv_head][pos][hd]`, which is the
`[n][k]` shape the score product's B fragment wants; V is `[kv_head][hd][cap]`,
which is the same shape for the value product; the queries are read straight
out of the projection buffer and the merged context written straight back in
`[tq, heads * hd]`. So `split_heads`, `merge_heads` and `repeat_kv` all
disappear from the prompt path rather than getting faster. A grouped-query
model maps `head / (heads / kv_heads)` and reads the one cached copy.

### The tile: traffic first, then `mma` per load

QT, KT and HD are all template parameters and the warp grid is derived from
them, so a new shape is an argument rather than an edit. HD is forced by the
model. The other two have each been the binding constraint in turn, and the
order matters because the second one was invisible until the first was fixed.

**QT paid first, and the reason was traffic.** A block stages every key and
value it walks past, so the whole of K and V is re-staged once per query block.
At the encoder's 1500 positions and QT 32 that is 47 trips through 0.8 MB -
724 MB a layer. QT 64 halved the re-staging and took the kernel from 967 us a
layer to 791. A Llama prefill keeps 32, whose causal loop stops at the block's
own diagonal and never walks a long key axis to begin with.

**Traffic then stopped being the answer**, and two measurements say so rather
than one. Only about a third of the halved re-staging came back as time; and
holding the whole cache at f16, which halves those same bytes again and is
*bit-for-bit* the same arithmetic because the kernel rounded them on the way
into shared memory anyway, changed nothing at all. One head's keys and values
are 768 KB and 144 blocks are resident, so the re-reads were already inside a
6 MB L2. `docs/BENCHMARKS.md` has both.

**What was left is shared loads per `mma`, and that is what KT buys.** A warp
issuing `MT * NF` products for `MT` `a` fragments and `NF` `b` fragments spends
`(2*MT + NF) / (MT*NF)` shared words a product - `ldmatrix.x2` for the row
fragment, `.x1` for the column. The tiled `gemm` next door has always been
shaped for this: `GEMM_MSTEPS` 8 by `GEMM_NPW` 2 is 1.125 words a product, and
it runs at three times this kernel's rate on the same tensor cores. The
encoder's fused attention sat at 1.75.

At `hd` 64 and QT 64 the warp grid is four query groups by two column groups,
so a warp's column fragment count is `KT / 16`. KT 32 leaves it at 2; **KT 64
puts it at 4 and takes the kernel to 1.5 words a product**, and halves the trip
count and its barriers on the way. Measured on the encoder, three rounds
alternated against the previous build: 117.5 ms to 112.5.

**KT 64 had been measured before and rejected, and that measurement was
confounded.** With the score tile still in shared memory a KT 64 block wanted
45.8 KB, which is one resident block on an SM's 64 KB and half the threads -
just enough to cancel what the wider tile gains, which is why it read as
"changed the encoder by nothing". Removing the score tile is what made the
knob work. The general lesson is in `docs/BENCHMARKS.md`: **a parameter that
measures flat may be paying for itself in a currency you are not looking at.**

Residency is the currency here, and it dominated everything in a thirteen-shape
sweep: every shape measured at one resident block came in at 114 ms or worse
and every one at two or three came in under 116, regardless of its load ratio.
The `static_assert` on the computed tile size says which limit and by how much -
QT 128 wants more than sm_75 gives a block in *static* shared memory and
otherwise fails in `ptxas` at module load with `CUDA_ERROR_INVALID_PTX` and no
reason attached - and three more check that the derived warp counts divide
exactly, because a warp grid that does not cover its tile drops work silently.

### The scores never leave the registers the `mma` wrote them to

This is what freed the shared memory KT 64 needed, and it is a saving twice
over. The scores used to go to a `[QT][KT]` f32 tile in shared memory, be read
back to find each row's maximum, and be read again to exponentiate - three
passes over 9 KB a trip, and a third of the block's shared budget, which is to
say a third of its residency.

None of that is necessary, because **what the warps have to tell each other is
not the scores but the reduction over them**. The `m16n8k8` accumulator hands a
lane two rows of its fragment and two adjacent columns, and the four lanes
sharing a `g` hold the rest of those two rows - so an xor butterfly over those
four lanes folds a warp's own columns without touching memory, and only CG
partials a row, one per warp column, reach shared memory at all. A `[QT][CG]`
array replaces a `[QT][KT]` one: at the encoder's shape, 512 bytes for 9216.

The masking moves with them. A lane knows its own row and its own two columns,
so the per-row key limit is applied to the accumulator directly rather than to
a shared tile - which is a different index than the old pass used, and is why
the differential tests now drive the narrow instantiation causally even though
nothing in the engine asks it to.

The probability layout falls out rather than being arranged: a lane's two
columns are adjacent and the first is even, so the pair it holds is exactly one
packed word of the `ps` tile the value product's `a` fragment reads.

### The head width

`hd` is a template parameter and the kernel is instantiated at the two widths
this engine has: 128 for both Llama stages, 64 for every Whisper size - Whisper
holds the head width fixed and varies the count, so large-v2's 1280 over 20
heads and tiny's 384 over 6 are the same 64. The width is the only thing the
fragment layout depends on: a warp owns `hd / 32` of the output's n8 column
fragments, four at 128 and two at 64, and the shared-memory strides follow.
The query and key tiles are not width-independent and are chosen per
instantiation - 64 rows and 64 keys a trip for the encoder, 32 and 32 for a
Llama prefill - but the eight warps and the derivation of the grid from them
are the same at both. A width the kernel is *not* instantiated at is refused by
construction rather than by tolerance: it would index another head's values, in
bounds, and return plausible context. `Gpu::supports_flash` is the predicate
callers use to pick a path, so choosing the unfused chain does not mean
allocating an output buffer and throwing it away.

### The arithmetic is the chain's, deliberately

Operands round to f16 where the tiled `gemm` rounded them, scores accumulate
in f32, `__expf` because that is what `softmax_causal` and `softmax_rows` both
use, probabilities round to f16 exactly where the chain rounded them - on their
way into the value product - and the normaliser sums unrounded f32.

**One place it is deliberately not the chain's arithmetic**, and it is worth
naming rather than discovering later. `softmax_rows` scales the probabilities
by `1/l` before rounding them to f16; an online softmax cannot, because `l` is
not known until the last tile, so it rounds `exp(s - m)` instead and divides
the f32 accumulator at the end. Both are one f16 step on a positive number, so
the relative error is the same size - and on a 1500-key encoder row the online
form is the better conditioned of the two, because `exp(s - m)` sits near 1
where `p / l` sits near 1/1500. The captured ASR oracle, layer by layer over
32 encoder layers, is what says that is harmless here.

The differential test compares against a scalar reference with the same
roundings, on a *peaked* softmax: near-uniform scores would let a permuted
position hide inside the tolerance, and a permuted position is precisely what
the test exists to catch.

**The reference accumulates at f64, and that is not pedantry.** On peaked data
some softmax rows are degenerate - one position takes essentially all the mass
- which makes the value product a difference of terms up to 45x its own
result. At that conditioning the kernel's blocked summation order and a
sequential f32 reference's disagree by 2%, every digit of it the reference's
summation order rather than the kernel's indexing; a reference has no business
being the less accurate of the two. The tolerance is sized by the *spread* of
the values the row averages rather than by the output's own magnitude, because
a convex combination's sensitivity to a rounding-sized score wobble is a
fraction of that spread and does not go to zero where the averaging cancels.
A permuted position substitutes one averaged value for another - a whole
spread, and many multiples of the tolerance - so the margin is still the thing
the test measures. It is reported as the worst error's fraction of its own
tolerance, so drift shows up before any single point crosses.

## Decode attention, `attn_decode`

The single-row twin of `flash_attn`: scores, softmax and the value product
for **one** query position, in one launch, off the caches in place. Both Llama
stages take it for every decoded token and the Whisper decoder takes it twice a
layer, for its self-attention over an f32 cache and its cross-attention over
the packed f16 encoder cache. `flash_attn` keeps the prompt.

### Why a second kernel and not the first one at `tq = 1`

`flash_attn` is built around `m16n8k8`, which wants sixteen query rows to fill
an instruction; at one row it would compute fifteen sixteenths of nothing.
And the single-row chain it replaces was not slow for the reason a prefill's
chain was. A prefill materialised a score matrix that was real traffic. A decode
step's score row is a few kilobytes - what it paid was **three launches a
layer** and a value product whose one-row shape ran the cache at 200 to
400 GB/s, on a card that streams at 585. So the single-row kernel is organised
around traffic and launch count, not around the tensor cores.

### Shape

Grid `(chunks, kv_heads)`, a block per 64 keys of one head, `HD` threads to
a block. Nothing is staged. A lane reads 16 bytes of a key row - eight halves
at `HD` 128 - and the sixteen lanes that share a key reduce their partial dot
products with four shuffles; a thread owns one value row and reads its
chunk's 64 positions straight out of the transposed cache, whose rows are
contiguous along exactly that axis. Every load a thread makes is issued
before anything waits on one: eight key loads, then eight value loads that
go out *before* the softmax they do not depend on, so the block's critical
path is three round trips to memory - the keys, the values, and the merge.

**That is the kernel's third shape, and the first two were measured slower
than the chain they replaced.** The first staged the keys and then the values
through a padded shared tile, a row of loads and a barrier at a time, which
made a block a chain of a dozen dependent round trips; the second kept the
staging and fixed the merge. Neither moved the chat model's token, and the
microbenchmark (`bench-attn`, thirty-two layers of one step, medians of
twenty) said why: at a 128-token context the fused kernel took 0.51 ms where
the three launches took 0.40. What a single-query kernel is short of is
neither bandwidth nor launches but the length of its critical path, and no
amount of coalescing shortens a chain of barriers.

**The chunks are merged by the last block to finish**, so it stays one launch.
Each block writes its running maximum, its sum and its unnormalised context to
a scratch buffer, bumps the head's counter behind a fence, and the block that
sees the count reach `chunks - 1` merges every chunk: one pass loading every
chunk's maximum and sum, a warp a group reducing them out of shared memory,
one pass of independent `ld.global.cg` loads for the context. The merge's
first version reduced each group in turn through block-wide barriers -
twelve dependent round trips for a grouped-query head - and that alone was a
third of the kernel at a short context. `DecodeScratch` owns the buffer and
the counters; it is held by the caller's cache rather than by `Gpu`, because
two sequences decoding by turns must not share a counter, and the Whisper
decoder keeps one for each of its two attentions for the same reason. At one
chunk the block writes the output directly.

**The grouped-query rows share every read.** The four query heads of a chat
model group are four score accumulators against one key load and four
context accumulators against one value load, which is what `gemv_rows` bought
by being a separate kernel and this gets for free.

Measured against the chain at every decoded shape, `bench-attn`, same card:

| shape, x32 layers | chain | fused |
| --- | ---: | ---: |
| chat 8 B, 128 ctx | 0.40 ms | 0.42 |
| chat 8 B, 512 ctx | 0.64 | **0.45** |
| chat 8 B, 1024 ctx | 0.91 | **0.60** |
| chat 8 B, 2048 ctx | 1.51 | **1.17** |
| translator 13 B, 128 ctx | 0.51 | **0.44** |
| translator 13 B, 1024 ctx | 1.62 | 1.61 |
| Whisper self, 40 of 448 | 0.47 | **0.18** |
| Whisper cross, 1500 | 0.91 | **0.76** |

Level at the short end and ahead everywhere the cache is worth reading
faster; what a token gains from it is in `docs/BENCHMARKS.md`.

### The arithmetic is the chain's

Scores are f32 sums of f16-or-f32 cache elements against f32 queries, the
softmax is `__expf` as in `softmax_causal`, the context is an f32 sum. The
association differs and nothing else. `scale_q` puts the scale on the query
before the product, which is where Whisper puts it - and where the
`scale_inplace` launch this also replaces put it - or on the scores after,
where Llama does. Same algebra, not the same rounding, and each model keeps
its own. The differential test runs every shape the engine decodes at, at
context lengths of one chunk, an exact chunk, a chunk and one key, several
chunks and an odd tail - the odd tail being the case where a packed value row's
last word holds a position the softmax must have zeroed - and runs each twice
through one scratch, because a counter left dirty merges too early or never.

### The query group is a template parameter

The kernel holds one query row a group in registers and in shared memory -
`qr[G][EPL]`, `acc[G]`, `o[G]`, `qs[G * HD]`, `sc[G * CH]` - so the widest
group it can serve is fixed when it is compiled. That was `AD_GMAX`, four,
for every entry, which covers the Llama stages (groups of 4 and 1) and the
Whisper decoder (1). CosyVoice3's speech LLM has 14 heads over 2, a group
of seven, and rather than widen every entry - which would cost the h128
entries registers they use - `G` is the fourth template parameter, the
existing entries pass `AD_GMAX` and are unchanged, and `attn_decode_h64_g8`
is instantiated at eight. Its scratch stride follows the entry's `G`, which
the wrapper reads off the name. With two key-value heads the grid is only
`chunks * 2` blocks, so the same entry exists at a 32-wide chunk too,
`attn_decode_h64_g8_c32`, taken below 1024 of context; `docs/BENCHMARKS.md`
has what each was worth.

### The chunk width is chosen by the context

The kernel is a template on the chunk width as well as the head width, and
`Gpu::attn_decode_f16` picks 32, 64 or 128 keys a block by the context it is
given. No single width wins, and `bench-attn` (thirty-two layers of one
step, ms) says where each does:

| shape | context | 32 keys | 64 keys | 128 keys |
| --- | ---: | ---: | ---: | ---: |
| chat, 32 heads over 8 | 128 | **0.337** | 0.425 | 0.456 |
| chat | 256 | **0.370** | 0.416 | 0.589 |
| chat | 512 | 0.505 | **0.444** | 0.596 |
| chat | 1024 | 0.778 | **0.597** | 0.632 |
| chat | 2048 | 1.155 | 1.170 | **1.065** |
| translator, 40 over 40 | 128 | **0.297** | 0.346 | 0.432 |
| translator | 1024 | **1.568** | 1.599 | 1.902 |
| whisper cross, 20 of 64 | 1500 | 0.794 | **0.756** | 0.821 |

The shape of that table is the kernel's critical path. A narrow chunk means
more blocks and less work in each, which wins while the context is short and
the merge is over a handful of partials; a wide chunk means fewer partials
for the last block to merge, which wins once that merge - a pass over every
chunk's context, alone, at the end - is what the launch waits on. A model
without query groups has less work a block at every width, so it stays on
the narrow chunk throughout. The rule in the launcher is those three lines:
32 below 256 positions or when `heads == kv_heads`, 128 from 2048 up, 64
between, and only at head width 128 - the 64-wide instantiations stay at 64,
which is where the Whisper decoder measured best. What that is worth end to
end is about 1% of a chat step at each end of the context range and nothing
in the middle; `docs/BENCHMARKS.md` has it.

The first sweep of this was run with a softmax that assumed 64 keys - two
scores a lane, hard-coded - and the 32- and 128-key rows were wrong in the
kernel's output while being about right in its timing. The differential test
caught it at one key, where the second score a lane read the next group's
row. The softmax now takes `CH / 32` scores a lane and the test drives every
width the launcher can choose.

### The context's twin

`attn_decode_f16_q` writes the context's int8 twin in the same pass. The `HD`
threads of a block own `HD` consecutive elements of one head's row, so a warp
is exactly one scale group of 32 and the group maximum is five shuffles over
a value the thread has just computed - `quantize_q8`'s arithmetic, on a row
that never leaves the register it was produced in. That is the launch the
output projection used to spend quantising the context on its way in, and
the test holds the twin at exact equality with `quantize_q8` run over the
same output, because a maximum associates exactly and there is no rounding to
allow for.

## The mat-vec's epilogue, `gemv_into`

A decode step's projections used to be followed by launches that did nothing
but move or finish what the projection had just written: `cache_append`
scattering a new key row into the head-major cache, `cache_append_t` doing
the same for the values, `gelu` over the MLP's inner activation. At one row
each of those is a few kilobytes and a launch, and the launch is the whole
cost. `gemv` now takes an epilogue: an activation flag, and a placement
`o_off + col * o_cs + (col / o_hd) * o_hs` that puts column `col` where the
append would have put it - a key cache when `o_hs` is a head's stride less
one position, a transposed value cache when `o_cs` is the capacity. The
defaults are the plain store and every other caller passes them.

`Gpu::gemv_into` is the entry that sets them, from an `OutLayout` it checks
against the destination's length before launching: the failure it guards is a
scatter into the next head's positions, in bounds and wrong. The arithmetic
is the mat-vec's to the bit - same kernel, same reduction, and `act_gelu`'s
expression character for character - so the test demands equality with the
two-launch form rather than closeness. The Whisper decoder takes it for its
self-attention keys and values and its `fc1`, which with the fused attention
above is eight launches a layer that used to be sixteen.

## Layer normalisation, on shuffles

`layer_norm` and `layer_norm_add` reduced through a shared-memory tree with a
barrier at every level - sixteen barriers for the two reductions - and read
the row from memory three times. At one decoded row of 1280 that block is all
the parallelism there is, and the kernel measured 11 us beyond the launch
floor for 10 KB of traffic: it was the length of its dependency chain. Both
now reduce through warp shuffles with one barrier each, load four floats at a
time, and keep the row in registers between the passes when it fits - up to
8192 columns at 256 threads, which is every row in the engine; past that the
later passes re-read it. The two-pass form is unchanged, the mean and then
the variance about it, so what differs from the tree is the association of
the sum and nothing else, and the oracle tests hold layer by layer.

## The cross-attention cache, batched over the layers

Building the ASR's cross-attention cache is sixty-four projections of the
same 1500x1280 encoder output, one per decoder layer per half. Each is 120
blocks of the tiled `gemm`, which on 144 resident slots is one wave whether
the launch has 120 blocks or 144 - so sixty-four launches pay sixty-four
waves for fifty-three and a third waves of work. The key weights of every
layer are one `[32, d, d]` allocation now, the values another, and each half
is one batched product over a shared activation: 3840 blocks, 27 waves. The
biases go with the split into head order, since a batched product carries one
bias for the whole batch and each layer has its own; the add is the same f32
add the matmul's epilogue would have made, so the cache holds the same bits.
15.3 ms to 13.5 on this card, which is the arithmetic's 16% within noise.

## Packed embeddings, `embed_q`

The one packed tensor a matmul never sees. `docs/MILESTONES.md` recorded that
the embedding table was still widened to f32 at load - 2.0 GB at the chat
model's 128 256-row vocabulary, 1.1 GB at the translator's - because a gather
is not a matmul and had no kernel that read blocks. It has one now: a block a
row, each thread unpacking its elements with `q_elem`, the general per-element
decoder the matmuls could not afford to call. A decode step gathers one row of
a few thousand elements, so the per-element header decode costs nothing that
can be measured and buys every block format at once. The numbers are the same
as before to the bit: the blocks decode to exactly the f32 that used to be
uploaded, and the test says so against the container crate's own decoder.

## One launch after the projections, `rope_cache_f16`

At one decoded position, what stands between the attention projections and
the attention is the query rotated in place, the key rotated and stored, and
the value stored. That was four launches a layer - `rope` twice,
`cache_append_f16` and its transpose - each moving a few kilobytes, so each
costing what a launch costs and nothing else. `rope_cache_f16` is the four in
one grid: the first `heads * hd / 2` threads rotate query pairs, the next
`kv_heads * hd / 2` rotate key pairs and write them to the cache as f16, the
last `kv_heads * hd` convert and scatter the value. The arithmetic is the four
kernels' character for character - the same `powf` and `sincosf`, the same
pairing of `j` with `j + half`, the same `f32_to_f16` - and the test holds
the caches and the query at exact equality with the chain, with both caches
pre-filled so a write at the wrong position is a difference and not a
coincidence.

The rotated key is never written back to the projection buffer, because at
one row nothing reads it from there: the attention reads the cache. The
three inputs are named as (buffer, offset) pairs into the projections' output
buffers rather than as three slices, because the translator issues q, k and v
as one batched product and they are then three offsets into one allocation -
which a `&mut` and a `&` could not both name.

## The mat-vec with the normalisation in its tail, `gemv_norm`

The two projections that close a sub-layer - the attention output and the
MLP's down - are each followed at one row by a normalisation that reads the
row they wrote plus the residual stream and writes the row the next
projections read plus its int8 twin. Two more launches a layer, under four
microseconds each and each mostly its own floor. `gemv_norm` is the packed
K-quant mat-vec with that tail in it: every block adds its columns into `h`
and publishes the sum of their squares; the last block to arrive sums the
partials and normalises the whole row.

Two things about that tail were measured rather than assumed. The reduction
is deterministic: partials go to a slot a block, the last block reads them
in a fixed order and reduces through a fixed tree, so the scale does not
depend on which block arrived last - an atomic float sum would have made
every decode step's arithmetic a function of scheduling. And the tail is on
the launch's critical path, because the last block runs it alone: the first
version walked the row a scalar at a time behind `ld.global.cg` loads and
cost ten microseconds a launch, more than the kernel it replaced. It now
loads four columns a thread with every load issued before any is used, which
is `rms_norm`'s mapping, and the residual element each warp adds to is
fetched before the contraction so its round trip hides under the weight
stream. What is left is about three microseconds a launch over the plain
mat-vec, which is the tail's own latency and the launch it removes; the
kernel time of a chat step is level with the chain's and the launch count is
what fell.

The column product is `gemv`'s and lands in `h` bit for bit as `gemv` then an
add would; the test holds `h` at exact equality against exactly that, and
the normalised row and its twin against the CPU twin at the tolerance
`rms_norm` is held to, with a code allowed to differ by one where a value
sits an ulp from a rounding boundary. Only the shape the Llama stages decode
with is covered - a `Q4_K` or `Q6_K` weight, one quantized activation row, a
contraction that is whole super-blocks - and the wrapper refuses the rest by
name rather than routing it to the chain, so a caller that asked for one
launch and did not get it hears why.

## The same two folds for the Whisper decoder, `gemv_ln` and `gemv_qkv_f16`

The Whisper decoder at one row had the same shape of waste as the Llama
stages after their round and one more of its own. Each of its three
sub-layers - self-attention, cross-attention, the MLP - closes with a
projection, a residual add and a layer normalisation, and each of the three
normalisations was a launch reading five kilobytes. And the self-attention
opened with three mat-vecs over one row - queries to a buffer, keys and
values placed into the caches by `gemv`'s epilogue - each reading 3.3 MB of
weight at a shape where the launch is a third of the time.

`gemv_ln` is `gemv_norm` for an f16 weight with a bias and a *layer*
normalisation. The structure is the same - every block adds its columns into
`h`, the last block to arrive normalises the row - but the tail is not: a
layer norm needs the mean and the variance about it, and the one-pass sum of
squares an RMS norm can afford cancels catastrophically when a row's mean is
large against its spread, which a residual stream's is. So the last block
takes the two passes `layer_norm_impl` takes, mean then variance, over the
settled row held in registers between them - four columns a thread, `GN_NL`
of those, which caps the row at 4096 and the wrapper refuses past it. No
partials are published at all: the row is 1280 floats and the last block
reads it from the L2 in one trip. The block sums are a fixed tree, so the
result does not depend on which block arrived last. `h` is held at exact
equality against `gemv` then `layer_norm_add`, the normalised row within
`1e-5` of the CPU twin, with and without a bias, and the odd contraction,
the ragged row, the too-wide row and the short operand are each refused.

`gemv_qkv_f16` is one launch over the three weights stacked `[3 d, d]`, each
column placed where the attention reads it: the first third into the query
row, the second into the head-major key cache at `pos`, the third into the
transposed value cache at `pos` - `OutLayout::KeyCache` and `ValueCache`'s
arithmetic, in the kernel rather than the wrapper. Each third keeps its own
bias pointer, because the key projection has no bias in any Whisper
checkpoint and a zero added is not always the bits of nothing added. The
stack is a layout, not a copy: the three weights were three allocations and
are one, and a prefix of several rows projects each third from its row
offset through `gemm_batched_from`. The test holds the query row and both
caches, prefilled, at exact equality against `gemv_into` three times.

Both kernels share `dot_f16_row`, which is `gemv`'s f16 loop for an f32
activation lifted out character for character, so that "bit for bit against
the chain" is a property of the code rather than of a test that happened to
pass. Thirteen launches a layer became eight. `docs/BENCHMARKS.md` has what
that is worth under "The decoder's round".

`gemv_norm_f16` is the third of the family: `gemv_norm`'s tail - the
partials a slot a block, the fixed tree, the RMS scale, the row four
columns a thread - on `dot_f16_row`'s column product, with an optional
bias and no int8 twin. It is what CosyVoice3's speech LLM decodes with:
every projection there is held at f16 and no activation is quantized, so
neither `gemv_norm` nor `gemv_ln` fit, and its two closing projections
were each followed by an add and an `rms_norm`, eleven launches a layer
at one token where seven would do. `h` is held at exact equality against
`gemv_batched` then `add_inplace` and the row within `1e-5` of the CPU
twin at the model's two closing shapes, with and without a bias; the odd
contraction, the row that is not four wide and the short bias are
refused. The speech LLM's token sequence on the benchmark utterance is
identical with and without it, and `docs/BENCHMARKS.md` has the step.

## Several rows, one weight stream: `gemv_q_rows`

A decode step at one row is the weight stream and nothing else - on the 13 B
translator, 8 GB a token at close to the card's bandwidth - and `gemv` puts
a second row at `blockIdx.y`, which streams the weight a second time. That
is the right shape for a single conversation and the wrong one for what the
translator actually receives: a reply chunked into clauses, the second
waiting before the first is half done. `gemv_q_rows` is the packed K-quant
mat-vec over up to `GEMV_Q_ROWS` int8 rows at once, each weight byte fetched
once and spent on every row. `gemm_batched` routes any packed product of two
to four rows to it, so the translator's batched step is the same call sites
as its single step with a different row count.

The per-row arithmetic is `q4k_wide` and `q6k_wide` character for character
- the same words, the same `dp4a` order, the same two-term combine, the same
warp reduction - only with the sixteen weight bytes, the header and the
scales loaded once above a loop over the rows. So a row of this kernel's
output is bit for bit a row of `gemv`'s, and the test holds it there at two,
three and four rows, both block formats, a contraction that is not a whole
number of warp trips, a batch that shares one activation, and a bias. The
row count is a template parameter - `R` accumulators a lane - so the
four-row instantiation is not paying for rows the two-row call does not
have, and the wrapper picks the instantiation by `m`.

What the rows share is the stream; what they do not share is attention,
which each takes over its own cache. `attn_decode` therefore grew a query
offset and a destination row - the query read from an element of the
batched projection buffer, the context and its int8 twin written into one
row of a `[rows, hidden]` buffer and one shared twin - so several sequences
can each attend into the operand the shared output projection then
multiplies. The twin's scales are one a group of 32, so a row must start on
a group, which every head geometry here does. Same kernel, same arithmetic;
the test holds the row against the single-row call bit for bit and the
neighbouring rows untouched. `docs/BENCHMARKS.md` has what the batched step
costs against the single one.

## The DiT's modulation on the card, `layer_norm_mod` and `gate_add`

A DiT block's normalisation is `LayerNorm` with no affine followed by
`* (1 + scale) + shift`, where `scale` and `shift` are two segments of the
six-chunk vector the block projects from the timestep; its residuals are
`h += gate * x` with `gate` a third segment. CosyVoice3's flow was computing
the normalisation on the card and the rest on the host, and
`docs/BENCHMARKS.md` has what that cost. `layer_norm_mod` is
`layer_norm_impl` handed `mods + scale_off` as its weight and
`mods + shift_off` as its bias, with a `wadd` of one applied to the weight -
the plain kernels pass zero, and adding zero is exact, so they are unchanged
bit for bit. The two offsets must be multiples of four because the affine is
read as `float4`, and the wrapper refuses one that is not. `gate_add` is a
flat elementwise kernel with the gate indexed by column.

The thing that was not obvious: both kernels round the multiply and the add
**separately**, with `__fmul_rn` and `__fadd_rn`, rather than letting the
compiler contract them into an `fmaf`. The host loops they replace did two
roundings, and the tiled matmul that reads the result stages it as f16, so
a one-ulp difference in f32 becomes a 5e-4 jump whenever it crosses an f16
rounding boundary. With the roundings matched the flow's estimator is bit
for bit what it was, which is the test; with them contracted it was a 0.45
difference in the mel that no tolerance could tell from a bug.

## The Tacotron2 decode loop in sixteen launches

The decoder produces one 80-channel mel frame a step through two LSTMs, a
location-sensitive attention and a projection, and the profile of that
loop was 39 operations a frame with the card busy under half the time. The
rule from the Llama decode rounds applies unchanged: at one row everything
here is under ten microseconds and mostly its own floor, so the unit that
matters is the launch, and a seam that puts two bodies under one grid is
free.

Two of the kernels are bookkeeping and say what they are: `relu_mask`
is the prenet's ReLU and its dropout mask in one pass, reading the mask
from a table drawn once for the whole line rather than uploaded every
frame; `taco_emit` is the end of a frame - the projection's row into the
output at the frame's index and its gate logit into a per-line buffer, so
the stop can be read every eighth frame instead of every frame. Each is
held against the copies it replaces at exact equality. Two more that the
round before this one added are gone: `concat2` built a cell's input
from two vectors, and it is not launched because the buffers are laid
out so that nothing is concatenated - each LSTM's hidden state sits at
the head of the buffer the next projection reads, the prenet's last
layer writes into the head of the attention cell's input, and
`taco_context` writes the context after all three, at their offsets, as
it computes it. `attn_weights_update` had been folded into
`taco_context` a round earlier and was still in the inventory.

The frame allocates nothing. `conv1d_into` and `gemv_into` write into
buffers the loop owns, the attention takes a caller-owned scratch for its
energies, and every temporary is a field of one state. That was done for
a CUDA-graph replay of the frame, which then measured level with issuing
it - the reason is the card's launch front end, not the CPU, and
`docs/BENCHMARKS.md` has the trace - so the graph is not in the tree and
the allocation-free frame is what remains of it, at sixteen launches
from twenty-two.

The location attention is the fold worth describing. It was a transpose,
a `linear` with the query as its bias, an `add_inplace`, a `tanh`, a second
`linear` down to one score a position, a `softmax_rows`, a `gemm` for the
context and the weights update - nine launches and five allocations. It is
two now. `taco_energies` is a block per encoder position and a thread per
attention unit: the energy is `linear`'s arithmetic - bias first, `fmaf`
over the location features in order - with the query as the bias, the
processed memory added and `tanhf` applied, reading the location features
as the convolution left them so the transpose is never launched, and the
score is the energies against `v` through a shared tree. `taco_context` is
one block: `softmax_rows`'s max, `__expf` and sum over the scores in shared
memory, then every thread owns channels of the context and sums the
alignment down its column of the memory, and the first `t` threads do the
weights update. The test holds both against the chain written out on the
CPU at 1e-4 on the context and the weights.

What the fold exposed is in `docs/BENCHMARKS.md`: with the launches gone,
two thirds of the card's time was four mat-vecs streaming the decoder
LSTMs' f32 weights at the card's bandwidth, and those weights turned out
to be f16-exact in the published file. `gemv` reads f16 weights against an
f32 activation already, so that was a binding change and not a kernel.

## Measured and rejected: sixteen-byte loads on the f16 mat-vec

The packed mat-vec's finding - a lane loading sixteen bytes reaches 578 GB/s
where four bytes reach 440 - was tried on the f16 weight path, where a lane
still loads one word a trip, because the Whisper decoder streams 1.47 GB of
f16 weights a token in 6.9 ms and that is 35% of the card. `bench-gemv`
(`cargo run --release -p xabe-cuda --bin bench-gemv`, medians of 200 launches
each followed by a synchronise, so every row carries the same round-trip
floor):

| shape | word loads | `uint4`, two in flight | `uint4`, four in flight |
| --- | ---: | ---: | ---: |
| 1280 x 1280 | 15.9 us | 16.0 | 16.6 |
| 1280 x 5120 | 33.0 | 34.1 | 34.9 |
| 5120 x 1280 | 33.3 | 34.6 | 34.8 |
| 4096 x 4096 | 66.9 | 68.9 | 68.4 |
| 4096 x 128 256 | 1 741.9 | 1 741.5 | 1 741.5 |

Level or a shade slower on every row, and the last row is the reason: at a
gigabyte the word loop already reaches 603 GB/s, so the body was never
load-width bound the way the packed one was - a packed lane has header
decoding between its loads, an f16 lane has two conversions. What the small
rows are short of is the per-grid floor, about three microseconds of ramp
and tail that a 3.3 MB launch cannot amortise, and no change inside the body
touches that. The wide code was removed; the bench stays, because the next
person will want to try the same thing.

## An accumulating product, `gemm_batched_into`

Not a new kernel: `gemm`, `gemv` and `gemv_rows` given three more things
to do at the store - write from row `out_first` of a buffer the caller
owns, read product `b`'s bias from `bias_stride * b`, and add to what the
buffer holds rather than store over it. `gemm_batched` and everything
above it are the same call with the three at zero and a fresh buffer.

What it is for is WaveGlow's coupling network. Each layer projects one
gated activation twice, residual and skip, and adds each into a running
sum; the checkpoint stores the two projections as one `[512, 256]`, so
with a shared left operand, a per-batch bias and an accumulating store
the two products and the two adds are one launch of 212 blocks writing
into the sums, where two launches of 106 each left a third of the card's
slots idle and the adds moved 20 MB a layer. The integer kernels have no
such epilogue and a packed weight is refused; the one caller never packs.

The store is written as two passes, and that is measured rather than
tidy. A loop that loaded, added and stored each element in turn ran at
twice the plain kernel on `k = 256`: a store to `out` may alias the next
load from it, so the compiler serialised the sixty-four loads a thread
behind the stores. Folding the running values into the accumulators in a
pass that stores nothing, then storing, lets the loads pipeline, and the
accumulating product costs the plain one plus the read of its output.
The order is `out + (acc + bias)`, so the result is exactly the plain
product added to what was there, which is what the test holds it to on
all three kernels, both weight widths, and the split contraction.

`gated_cond_rows` is the other half of the same round: WaveNet's gate
over `x + cond`, with `cond` a column block of the wide conditioning
product, reading the conditioning once where `add_strided` read and wrote
the whole activation to leave the sum for the gate to read again. Exactly
the two kernels it replaces.

## A weight read from a row, `gemm_batched_from`

Not a kernel: the same kernels, handed the weight from row `w_first` of its
allocation. What it is for is stacking. The chat model's q, k and v have one
width and, in most layers, one block format, so they are held as one
`[4096 + 1024 + 1024, 4096]` allocation - `[q; k]` and `[v]` where `attn_v`
is the `Q6_K` odd one out - and a decode step projects the whole stack in one
launch of 768 blocks where k and v used to be two launches of 128, each at
about 430 GB/s against the 585 a launch this card fills reaches. A prompt
cannot take that product, because its rows have to come out `[n, out]` a
projection and a `[n, 6144]` output is the wrong shape for everything
downstream; so a prompt runs the members one at a time, each from its first
row. The offset counts rows, which is whole blocks for a packed weight and
whole elements otherwise, and the wrapper checks that the rows asked for are
there. The test holds a product from an offset at exact equality with the
product of the rows copied out from there, on the mat-vec, the f16 tile and
the integer tile.

## `silu_mul_pair`, and why a launch is the unit that matters

At one decoded row, a 13 B step runs about fifteen kernels a layer that read
almost nothing - two normalisations, three for attention, the gate, RoPE twice,
the cache append twice - and together they are a seventh of the step. They do
not cost what they read. They cost what a launch costs, and on this card that
is about 3.3 us of queueing and a few microseconds of ramp and drain.

Two measurements say so, and both were taken before anything was written:

- Widening the row-reduction block from 256 threads to 1024, which is four
  times the warps on the one SM a single-row normalisation can use, moved the
  translator's decode by **nothing at all**: 16.10 ms a token before and after.
  It is not the reduction.
- Sweeping the mat-vec's column count from 4096 to 13824 gives a **smooth**
  climb from 442 to 545 GB/s with no sawtooth at the wave boundaries. It is not
  wave quantisation either - it is a fixed cost being amortised over more work.

So the lever is fewer launches, and the MLP is where one was free. `gate` and
`up` have the same shape and, in every checkpoint here, the same block format,
so they are one batched product whose output is `[2, rows, inter]` - which is
exactly the pair the SiLU gate needs. `silu_mul_pair` reads both halves of that
one buffer, `x[i] = silu(x[i]) * x[i + n]`, and writes the result over the first
half, which is where the down projection reads its left operand from anyway.

One buffer rather than two pointers because they alias one allocation and one
`&mut` is the only shape Rust will hand over. The arithmetic is `silu_mul`'s
exactly - same expression, same order, same group quantiser - and the three
reference translations come back character for character.

`GMlp::Split` is the fallback, and it is not hypothetical: `Q4_K_M` gives
`attn_v` and `ffn_down` `Q6_K` while their neighbours get `Q4_K`, which is why
the attention side of this checkpoint groups q with k and leaves v out.

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
