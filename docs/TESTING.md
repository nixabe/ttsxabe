# Testing and numerics

## The risk this exists for

VITS degrades gracefully into confident nonsense. A transposed convolution read
`[out, in, k]` instead of `[in, out, k]`, a flow coupling with its halves
swapped, an off-by-one in duration expansion — none of these crash, and none
produce silence. They produce fluent speech that is wrong.

If you do not speak Taigi you cannot hear it. If you do, you will hear it only
sometimes. Listening is not a test.

## The structure

Two layers, same as `llmxabe`:

- **Inline `#[cfg(test)] mod tests`** at the bottom of a source file, for pure
  logic — shape arithmetic, offset maths, vocabulary handling.
- **`tests/` integration files**, one per concern, for anything numeric. These
  compare against a captured oracle or against the CPU reference.

## Every kernel has a reference and a differential test

A CUDA kernel is compared against its `xabe-dsp` scalar implementation on the
same inputs. A `xabe-dsp` implementation is compared against activations
captured from PyTorch (see [ORACLE.md](ORACLE.md)).

**A kernel without a passing differential test is not done, regardless of how
fast it runs.**

## Report all the metrics

Per tensor, report **max absolute error** and **cosine similarity**, both. They
fail differently and the pair tells you where to look:

| max-abs | cosine | usually means |
| --- | --- | --- |
| bad | good | scale error — a missing normalisation, a wrong constant |
| good | bad | layout error — transposed weights, wrong stride |
| bad | bad | the algorithm is wrong |
| good | good | pass |

A single scalar summary hides exactly the distinction you need.

## Tolerances

Named presets, each carrying why it is what it is. Do not invent a threshold at
the call site to make a test pass — if a kernel needs a looser bound than its
neighbours, that is a finding, and it goes in the preset's doc comment.

f32 accumulation order differs between a scalar loop and a warp reduction, so
bitwise equality is not the standard anywhere. The standard is that the error is
too small to be a bug and stays that way as inputs scale.

## The stochastic parts are seeded

The duration predictor and the prior both sample noise. A differential test that
draws its own randomness compares nothing. Both the reference and the
implementation under test take an explicit seed, and the oracle capture records
the noise tensors it used so the comparison is against fixed draws.

Anything that cannot be made deterministic is tested on its *distribution* —
mean, variance, output length — never on a single sample.

## Skip loudly

A test needing the checkpoint, a GPU, or a golden file detects its absence,
prints `SKIP:` with the reason, and returns:

```rust
let Some(path) = find_model() else {
    eprintln!("SKIP: mms-tts-nan checkpoint not found; set XABE_TTS_MODEL");
    return;
};
```

A skipped test is not a passing test. Silence about why is how a whole suite
quietly stops covering anything.

## Running

```sh
cargo test --workspace --release
```

Release, always. The reference kernels run the full forward pass thousands of
times; a debug build turns a test run into a coffee break, which is why
`profile.dev.package."*"` is set to `opt-level = 2` for dependencies too.

### Choosing a card

```sh
XABE_TEST_DEVICE=2 cargo test --workspace --release
```

`XABE_TEST_DEVICE` and not `XABE_TTS_DEVICE`. The second is the engine's
`--tts-device` env twin, so exporting it to steer a test run also reaches into
`xabe-engine`'s flag tests, which then assert their defaults against whichever
card someone happened to pick. That cost eight failing tests once, all of them
looking like a broken flag parser.

Check `nvidia-smi` before choosing - this host is shared, and the tests will
happily allocate on a card that is already running somebody's training job.

### The environment variables a test may look for

Every one of these is optional: absent, the test that wants it prints `SKIP:`
and says which variable to set.

| variable | what it points at | default |
| --- | --- | --- |
| `XABE_TEST_DEVICE` | CUDA ordinal for tests that need a card | `0` |
| `XABE_TTS_MODEL` | the VITS checkpoint | `models/tts/mms-tts-nan` |
| `XABE_LLM_GGUF` | the Breeze2 chat GGUF | `models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf` |
| `XABE_TRANSLATOR_GGUF` | the translator as a GGUF | `models/taigi-translator-13b-f16.gguf` |
| `XABE_QUANT_DIR` | a directory of quantized copies | none; those tests skip. `models` is where they live |
| `XABE_CHAT_DEVICE` | the card to load the 8 B chat model onto | none; that test skips |
| `XABE_QUANT_FILE` | which file in `XABE_QUANT_DIR` the packed test reads | `breeze-Q4_K_M.gguf` |
| `XABE_TACO_DEVICE` | the card to load Tacotron2 + WaveGlow onto | none; those tests skip |
| `XABE_CHAT_MODEL` | the chat GGUF the `llama_server` oracle runs | none; that test skips |

`XABE_QUANT_DIR` is read as given, so an absolute path is the safe form: cargo
runs a test binary from the workspace root but a relative `models` is easy to
get wrong from a subdirectory, and the failure is a `SKIP:` rather than an
error.

`XABE_CHAT_DEVICE` has no default either, and for a different reason: this box
has three cards and two of them are running somebody's pipeline. `run.sh` says
to check `nvidia-smi` before taking one, and a test that silently lands on a
busy card is exactly what that is warning about. Every other GPU test defaults
to `0` because it needs a few hundred megabytes; this one needs sixteen
gigabytes, which is enough to evict a neighbour.

It is also why the whole chat comparison is **one test with sections** rather
than several. `cargo test` runs a file's tests on separate threads, and four
tests each loading their own copy of the weights is 64 GB onto a 48 GB card.
Requiring `--test-threads=1` instead would have been an invisible condition
that fails as an out-of-memory error rather than as a message.

`XABE_QUANT_DIR` has no default on purpose. The files are several gigabytes
each, they are derived rather than downloaded, and checking a multi-gigabyte
artefact into a fixed path that a test then silently depends on is how a suite
becomes unrunnable on a second machine. One command reproduces any of them:

```sh
llama-quantize models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
    $XABE_QUANT_DIR/breeze-Q4_K_M.gguf Q4_K_M 8
```

### The packed matmul is tested at two distances

`Operand::Q` lets a quantized weight stay packed in VRAM, and being wrong about
a block layout produces a *permuted* tensor rather than an error - a model that
loads, runs, and speaks fluent nonsense. So it is checked twice, at different
distances from the bytes, and neither check subsumes the other.

A third check sits between them and covers what the other two cannot. The wide
mat-vec is a different addressing scheme from the element-for-element path, and
the int8 activation makes exact comparison impossible - so
`the_wide_kquant_matvec_agrees_with_the_f32_product` runs both formats at a `k`
that takes the fast path and bounds the disagreement at 1% of the output span,
which is what quantizing the activation costs and nothing more. The bound is
loose on purpose and it is not what catches an addressing mistake: a lane
reading the wrong sixteen bytes does not land within a percent of the right
answer. It caught one during development - an element offset of `n * 64` where
the layout wanted `n * 128` - by being wrong by 74%.

The quantiser itself is compared to `xabe_dsp::quantize_q8` at **exact
equality**, on both codes and scales, including an all-zero group and one whose
maximum is a power of two so ties are reachable. The thing worth checking is
that the two implementations approximate identically; a tolerance there would
hide the group-boundary or rounding-mode disagreement the test exists to find.

Two kernels read those codes now rather than one - the packed mat-vec and
`gemm_i8`, the tiled matmul on the integer tensor cores - so the tiled path has
a reference of its own. `quantized_gemm_matches_the_cpu_dequantizer` builds the
*same* approximation on the host, quantizing the activation a row at a time
with `quantize_q8` and leaving the weights exact, and compares at 1e-5 relative
to the sum of the term magnitudes. Comparing the integer path against unrounded
operands instead would need a tolerance wide enough to hide a permuted block,
which is the failure this whole file exists to catch.

The fused attention kernel's test,
`fused_attention_matches_the_unfused_arithmetic`, carries the same lesson from
the other direction: it is about test *data*, not tolerances. Its scalar
reference rounds where the unfused chain rounds - scores and probabilities
through f16 - and its first version drove random queries whose scores all
landed within a few percent of each other, so every softmax row was nearly
uniform. Nearly-uniform attention is the one distribution that cannot catch a
fetched-from-the-wrong-position bug: attend everywhere equally and it barely
matters which value came back. The queries are scaled until the softmax is
peaked, the causal boundary is exercised, and the heads share KV heads
grouped-query style - and the tolerance is tight *because* the data is strong.
A test that passes on weak data is the same defect as a tolerance wide enough
to hide a permutation.

**Close in**, `xabe-cuda`'s `tests/quant.rs` compares against
`xabe_gguf::dequantize_blocks` - the decoder already checked against `gguf-py`
at exact equality. It extracts weights *element for element* through a one-hot
activation on the exact f32 path, so the comparison is equality rather than a
tolerance and a permutation inside a block cannot hide behind a dot product. It
also runs the whole product on both kernels, and pins the two size tables that
`xabe-cuda` duplicates because it may not depend on `xabe-gguf`.

`a_batch_over_one_activation_matches_the_same_products_apart` covers the shape
the attention projections are issued in: several matrices against one shared
left operand, in one product. It compares **bit for bit** against the same
matrices run separately, because the two do the same arithmetic in the same
order - the only things that can differ are the weight stride, which would read
the neighbouring matrix, and the activation's row stride, which both packed
paths derive from a row count and which has to stop advancing when the operand
is shared. `gemv` was the kernel that got that second one wrong, and this is
the test that would have said so first.

Alongside it in `tests/kernels.rs`,
`rope_at_an_offset_rotates_that_block_and_only_that_block` and
`cache_append_reads_its_own_block_of_a_batched_projection` pin the other half
of that change - the offsets that let those two kernels read one product's
block out of a shared output rather than copying it first. Each checks the
block against its scalar twin *and* checks that the bytes on either side did
not move, which is the half an in-place kernel can get wrong silently.

**Further out**, `xabe-chat`'s `tests/packed.rs` loads the same quantized file
twice - `Packing::Packed` and `Packing::F16` - and compares logits. That is the
only check on the *wiring*: that the ggml type maps to the right layout, that
the rope permutation reaches the packed bytes as well as the f16 ones, and that
the packed operand gets to every projection rather than most of them. It needs
`XABE_CHAT_DEVICE` with about 21 GB free, because the f16 half of the
comparison is the unpacked 16 GB.

The two paths are close rather than identical, and the reason is now the same
on both sides: **both** the mat-vec and the tiled matmul quantize the
*activation* to int8. The mat-vec does it to feed its wide loads; the tiled
matmul does it because `gemm_i8` multiplies on the integer tensor cores. So
this is a bound on what int8 activations cost, and nothing here is
bit-identical any more.

| | prompt | worst logit difference | logit span | |
| --- | --- | --- | --- | ---: |
| `xabe-translate` 13 B `Q4_K_M` | 30 tokens | 0.120685 | 28.657 | 0.42% |
| `xabe-chat` 8 B `Q4_K_M` | 14 tokens | 0.174942 | 25.323 | 0.69% |

An earlier version of this table read 0.000000 for the translator, and the
paragraph under it explained that its prompt is past `GEMV_MAX_M`, so every
projection takes the tiled kernel, "no activation is quantized, and the two
paths are bit-identical". The first half is still true and the second half is
not: since `gemm_i8`, being past `GEMV_MAX_M` is what puts a projection *on*
the integer path rather than what keeps it off one. The chat figure moved from
0.167 to 0.175 for the same reason - its prefill joined its decode.

That number is a bound on the arithmetic, not on the output. What it costs in
*tokens* is measured separately and is zero: greedy decoding against the
`llama-server` capture picks the same token at every position it picked before,
and the disagreement list below is byte-identical across the change.

### The chat model's disagreement was the reference, not the engine

This section recorded ten disagreements of 105 teacher-forced decisions against
llama-server, with margins up to 2.86 nats, as a real and unexplained defect.
It is explained. **The capture was taken from llama-server running the
*quantized* checkpoint**, and llama.cpp's quantized matmul multiplies the packed
weight against an int8 activation - a coarser arithmetic than this engine's
tiled path uses. The capture recorded that error as the target.

Re-captured from the same llama-server running the **f16** build of the same
model, and compared against this engine reading the *quantized* one:

| reference | decisions | replies |
| --- | ---: | ---: |
| llama-server on f16 | **1 of 125**, margin 0.056 | **7 of 8** identical |
| llama-server on Q4_K | 10 of 105, margin 2.86 | 3 of 7 identical |

The forked replies under the second row are, word for word, the replies
llama-server produces from the f16 build. This engine reading a 4-bit
checkpoint reproduces the full-precision model; llama.cpp reading the same 4-bit
checkpoint does not.

`.golden/chat/llama_server.json` is therefore captured from the f16 build, and
`tools/oracle/capture_chat_server.py` says so at the top with the numbers.

### Reading a consistency number against the wrong model

`a_batched_prefill_matches_the_same_tokens_one_at_a_time` reported 0.16% of the
logit span and one argmax fork in 179, and then 2.92% and seven. That looked
like the projection grouping had cost accuracy. It had not: the first number
was measured on the **f16** GGUF and the second on the **Q4_K** one. Run on the
f16 build after the change, it is 0.18% and zero forks.

The quantized number is larger for a reason that is not a defect. This test
compares a batched prefill against the same tokens fed one at a time, which on
a quantized model is the tiled integer matmul against the packed mat-vec - two
different kernels quantizing the activation in different groupings. On an f16
model both paths do the same arithmetic and the comparison is tight.

Two lessons, and the second is the one that keeps costing time. A test whose
tolerance is wide enough to pass on both checkpoints will not tell you which
one you ran. And **the model under test belongs in the number**: the same
oversight, in a stronger form, is the section below.

### The same mistake a second time, and what now prevents it

`tests/layer_taps.rs` compares this engine's per-block sums against
`llama-eval-callback` reading the same file. It failed at 0.36 of the layer
magnitude at block 1, and stayed flat there across every block after it - an
offset that enters at one layer and rides the residual stream, which is exactly
the shape of a wiring fault.

It was not one. `.golden/chat/layers.json` had been captured from the **Q4_K**
build and the test was pointed at the **f16** one. Two different models. Run
against the file the capture came from, the same comparison is 0.0986 of the
layer magnitude against a bound of 0.25.

Nothing in the capture said which file it came from, so nothing could refuse
this. Now something does: `capture_chat_layers.py` records the model's name and
byte count, and the test checks both before comparing anything, failing with
both names rather than with a number. A capture that predates the field is
refused as well - an oracle whose provenance is unknown is invalid state, and
the house rule is to reject invalid state rather than compute with it.

This is the second time a capture of the quantized build has been mistaken for
a defect in the engine, and both times the shape of the divergence was
convincing. **When a per-layer comparison shows a constant offset from an early
block, check what the capture was taken from before reading the kernel.**

Three arithmetic interventions were made before the reference was suspected, and
each moved the disagreement list by exactly zero: an integer tiled matmul
multiplying the packed blocks the way llama.cpp does, an activation pre-rounded
to the int8 grid, and both attention matmuls in exact f32. All three produced
byte-identical lists. `docs/BENCHMARKS.md` has them under WHY NOT. **A run of
interventions that change nothing is evidence about the target, not about the
thing being changed** - and that is the lesson, because it took three of them.

### What the int8 activation costs, in decisions

The engine has two multiplies for a packed weight and they are not equally
faithful. `tests/stepwise.rs` runs the same 125 decisions a token at a time, so
every projection lands on the mat-vec, which quantizes the activation to int8 to
feed its wide loads:

| weights | activation | agreement with the f16 reference |
| --- | --- | ---: |
| `Q4_K` | f16, tiled | 124 of 125 |
| f16 | f32, mat-vec | 124 of 125 |
| `Q4_K` | **int8, mat-vec** | **114 of 125** |

Quantizing the weights costs almost nothing. Quantizing the activation costs
about a tenth of greedy decisions, several at margins the reference won by nine
to twelve nats. That is what decode buys its 1.67x with, it is the same trade
llama.cpp makes, and the number is here so that it is not assumed to be small.

`tests/consistency.rs` bounds the rest: the batched prefill against the same
tokens one at a time - both this engine, no oracle - fork on 5 of 179 argmaxes.
A greedy comparison of an 8 B model is measuring chaos alongside correctness.

### Where that disagreement lives, and where it does not

This is the analysis that was done while the **10 of 105 teacher-forced
decisions** were still believed to be the engine's defect, kept because its
placement of the disagreement is what eventually pointed at the capture. They
were not a wiring bug and they were not spread across the engine. Four
measurements place them:

| | disagreements |
| --- | ---: |
| ours vs llama.cpp, f16 checkpoint, batched | 1 of 125, margin 0.056 |
| ours vs llama.cpp, Q4_K, one token at a time | 1 of 105, margin 0.19 |
| ours vs llama.cpp, Q4_K, batched | 10 of 105, margin 2.86 |
| llama.cpp CPU vs its own CUDA, Q4_K | none, to the printed precision |

The first says our arithmetic tracks llama.cpp's to a coin flip when both do
the same thing. The last says the reference is deterministic. The middle two
say the difference is **which of our kernels multiplied**, and nothing else:
`tests/stepwise.rs` runs the identical comparison one position at a time so
every projection lands on the mat-vec instead of the tiled matmul, and nine of
the ten disagreements go away.

What is left in the tiled path is not the packed weight. Replacing that matmul
outright - an integer kernel multiplying the checkpoint's blocks against an int8
activation, which is llama.cpp's own arithmetic - moved none of the ten, and
measured *less* accurate than the f16 staging it replaced (`docs/BENCHMARKS.md`
has the numbers and the kernel is gone). Feeding the f16 kernel an activation
pre-rounded to the int8 grid moved none of them either.

Attention was the next suspect and it is not the answer either. Computing both
attention matmuls in exact f32 in the batched path takes this engine's own two
paths from forking on 5 of 179 argmaxes to 2 - so rounding them is a real error
source - and leaves the disagreement with llama-server at exactly ten, the same
ten. It costs 23% of a prefill and was not kept; `docs/BENCHMARKS.md` has it.

So three arithmetic interventions have now moved this by zero: an integer
matmul, an activation pre-rounded to the int8 grid, and exact attention. Each
produced a byte-identical disagreement list. **The cause is not the precision of
the batched path**, and what is left is the small set of things that path does
and the one-token path does not: the head split and merge, the causal mask, and
writing the whole KV cache in one call rather than a position at a time. None of
those has been ruled out.

`tests/consistency.rs` bounds how much of this is anyone's fault. It runs the
batched prefill against the same tokens fed one at a time - both this engine,
no oracle - and they fork on **5 of 179 argmaxes**. A greedy comparison between
two implementations of an 8 B model at f16 is measuring chaos as much as
correctness, and the disagreement counts above should be read with that in mind.

### The order the sections run in, which is not the order they were written

`tests/llama_server.rs` is one test with numbered sections, and the teacher-forced
decisions used to be section 1 and to assert immediately. That meant a
*diagnostic* - one that drives every position through the tiled matmul in a
single pass, which is not how the model is ever used - failed the test before
section 2 could report whether the engine still says the same sentences.

The assertion is now held to the end. The replies, the streaming, the
cancellation and the sampler checks all report first, and the teacher-forced
verdict is raised after them. Nothing about the check changed; what changed is
that a reader sees the product result before the diagnostic that hid it.

The synthetic quantization corpus under `.golden/gguf/quants` is different: it
is kilobytes, and it covers all ten block formats where the real file covers
one. It is **not committed** — `.gitignore` excludes `.golden/` wholesale, so
it is regenerated by `tools/oracle/capture_quants.py` like every other capture.
This paragraph used to claim it was committed, which was wrong and stayed wrong
until CI made every oracle test skip on a clean checkout and the claim became
visible.

### Two tests that must not share a process

`xabe-translate`'s oracle and its GGUF twin each load 26.5 GB onto the card, so
they cannot both be resident on a 48 GB one. They live in **separate test
binaries** for that reason - cargo runs test targets one after another, so the
first process has exited and freed the card before the second starts. Within a
binary, one `OnceLock<Mutex<Translator>>` keeps cargo's parallel test threads
from loading three copies and reporting an out-of-memory that reads like a
broken loader.

### A consistency test that stopped 57 tokens short of the bug

`tests/consistency.rs` asks the right question - does prefilling `n` positions
compute what decoding them one at a time computes - and answered it correctly
for months while a decode past 256 tokens produced noise.

The cache is allocated at a floor of 256 positions and doubled from there. The
consistency prompt is 199 tokens. So the test never grew a cache, and growth was
where the bug was: `cap` is a **stride** in both cache layouts, keys being
`[kv_heads, cap, head_dim]` and values `[kv_heads, head_dim, cap]`, and the
growth copied the live prefix flat. Head 0 begins at zero in both the old
allocation and the new one, so it survived; every other head landed inside its
own earlier positions.

What that looks like from outside is the part worth remembering. The model
answered the first sentence correctly - off the one head that had not moved -
and then degenerated into fluent nonsense in the wrong language. No crash, no
shape error, no out-of-bounds read: the buffer is the right length and every
index is inside it. It reads as the checkpoint being bad.

Measured against a control that never grew, at a 251-token prompt: 65 of 120
positions over 4% of the logit span, in an unbroken run from exactly 256 to the
end, worst 80.1%. After the fix, nothing over 4%, worst 2.1%, and the largest
differences scattered across positions 202, 205, 293, 259 and 221 - no
clustering at the boundary, which is what the tiled-prefill-against-mat-vec
rounding floor looks like.

Two things now stand where nothing did. `Gpu::cache_grow` re-strides rather than
copies, and is checked in `xabe-cuda`'s kernel tests by an invariant that needs
no golden: appending at a small capacity and then growing must equal appending
at the large capacity to begin with, for both layouts and with `kv_heads > 1`,
because at one head the bug is invisible. And `xabe-chat/tests/cache_growth.rs`
prefills 200 tokens and then steps 120, which crosses the boundary the way
generation does.

The general lesson is about *coverage of the sizes a test runs at*, not about
caches. A differential test is only as good as the code paths its inputs reach,
and a capacity that doubles from a floor means the interesting path opens at one
specific length. Any test whose input sits below a threshold is not testing what
happens above it, however exactly it agrees.

## CosyVoice: two things that cannot be bit-exact, and how each is bounded

Most of this workspace is tested against a captured oracle to within float32
rounding. Two parts of CosyVoice3 cannot be, and the reason is a property of the
reference rather than a limitation here. Saying so is the point: a loose bound
with no reason attached is a bound that will be loosened again.

**The vocoder's dither is not in the checkpoint.** `SineGen2` and
`SourceModuleHnNSF` each call `torch.rand` in `__init__` and keep the result as
a plain attribute, so upstream redraws it on every construction and does not
reproduce across load orderings either. `tests/vocoder.rs` and
`tests/source.rs` load the captured buffers and hold correlation above 0.9999;
`tests/pipeline.rs` lets the engine draw its own and holds the **energy
envelope** instead.

**The excitation's phase is a cumulative sum**, so any difference in the
predicted F0 accumulates into a phase offset over the utterance. Measured, the
chained pipeline correlates at **-0.001** sample for sample and at 0.968 on a
10 ms envelope, at a gain of 1.008. The envelope is what a wrong stage moves; a
phase shift leaves it alone.

## What CI proves, and what it cannot

`.github/workflows/ci.yml` runs on a GitHub-hosted runner, which has no CUDA
device and no checkpoints, and `.gitignore` keeps `models/` and `.golden/` out
of the repository. So **every numerical test skips there**.

That did not work the first time it was tried, and the reason is worth keeping.
Every test skipped correctly on an absent *GPU* — that guard was uniform and
careful. But seven test targets across five crates **panicked** on an absent
*model or capture*, which is a different condition and which the paragraph
above sanctions skipping on. Twenty-six call sites said
`panic!("models/... is missing")` where the policy this document already stated
says to skip. They were written on a machine that has the models, where the
distinction never comes up.

The sharper form of the rule, and the one worth copying, is in
`xabe-asr`'s `checkpoint()`: **absent is a skip, present-but-broken is a
failure.** A tree that was never populated has nothing to test; a tree where
`models/asr/` exists but the shard index inside it does not is a setup mistake,
and passing over that in silence is how a green suite comes to mean nothing.
The other crates currently do the simpler thing and skip on absence alone.

That is a trap this document already warns about in general, so the workflow
does not get an exemption: it counts the skips and writes them into the run
summary, because a run where everything skipped otherwise looks identical to a
run where everything passed.

| | proved by CI | proved only locally |
| --- | --- | --- |
| compiles with no CUDA toolkit | ✅ | |
| `cargo fmt --all --check` | ✅ | |
| `clippy -D warnings` | ✅ | |
| rustdoc clean at `-D warnings` | ✅ | |
| shape arithmetic, tokenizers, containers, wire formats | ✅ | |
| every kernel against its scalar twin | | ✅ |
| every model against its captured oracle | | ✅ |
| end to end, audio in and audio out | | ✅ |

The second column is `cargo test --workspace --release` on a box with the
checkpoints, and it stays a human step. A green tick on a pull request means
the code is well-formed, not that the models are right.

## `XABE_COSY_DEVICE` has no default

The CosyVoice tests skip unless it is set, and it is not defaulted to 0 for the
same reason the ASR's are not: two of this box's three cards are running
somebody else's pipeline, and these models are not small. `nvidia-smi` first.

```sh
XABE_COSY_DEVICE=2 cargo test --release -p xabe-cosy
```

## `XABE_TACO_DEVICE`, and what Tacotron2 can be tested against

Same rule, same reason: `nvidia-smi` first.

```sh
XABE_TACO_DEVICE=2 cargo test --release -p xabe-taco
```

The text tests need no card and always run. Of the rest, **only the encoder is
compared against the reference**, and that is not a gap in the tests but a
property of the model: `Prenet.forward` passes `training=True` to `F.dropout`
unconditionally, so every decoder step multiplies by a fresh random mask, and
WaveGlow then starts from Gaussian noise. Two correct implementations do not
agree sample for sample, and neither do two runs of one.

The encoder has no dropout at inference and is therefore deterministic, which
also puts the three riskiest things in the crate inside it: the batch-norm
folding, the LSTM gate order, and the order the two LSTM directions are
concatenated in. Each of those produces bounded, plausible output when wrong.

```sh
python tools/oracle/capture_tacotron2.py \
    --src /path/to/taiwanese_tonal_tlpa_tacotron2/tacotron2 \
    --text "gua2 si7 tai5-uan5-lang5" --out .golden/tacotron2/nan
```

Measured at **max-abs 1.22e-6 and cosine 1.000000000** on that capture. The
test's bound is 1e-5, an order of magnitude above the observation rather than
the 2e-4 the arithmetic would permit: a tolerance far looser than the observed
error is a test that passes through the bug it exists to catch.

What the decoder and vocoder get instead is arithmetic with no tolerance at
all - the waveform is a whole number of 256-sample frames, the peak is at full
scale, the variance is not zero - plus a seed-reproducibility test, which is
what turns the stochasticity into a property rather than an excuse.
