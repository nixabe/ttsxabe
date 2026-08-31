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
maximum is a power of two so ties are reachable. It is the one approximation in
the engine, and the thing worth checking is that the two implementations
approximate identically; a tolerance there would hide the group-boundary or
rounding-mode disagreement the test exists to find.

**Close in**, `xabe-cuda`'s `tests/quant.rs` compares against
`xabe_gguf::dequantize_blocks` - the decoder already checked against `gguf-py`
at exact equality. It extracts weights *element for element* through a one-hot
activation on the exact f32 path, so the comparison is equality rather than a
tolerance and a permutation inside a block cannot hide behind a dot product. It
also runs the whole product on both kernels, and pins the two size tables that
`xabe-cuda` duplicates because it may not depend on `xabe-gguf`.

**Further out**, `xabe-chat`'s `tests/packed.rs` loads the same quantized file
twice - `Packing::Packed` and `Packing::F16` - and compares logits. That is the
only check on the *wiring*: that the ggml type maps to the right layout, that
the rope permutation reaches the packed bytes as well as the f16 ones, and that
the packed operand gets to every projection rather than most of them. It needs
`XABE_CHAT_DEVICE` with about 21 GB free, because the f16 half of the
comparison is the unpacked 16 GB.

The two paths are close rather than identical, and the asymmetry has a reason:
on the tiled kernel both stage the same f16 bits, while on the mat-vec the
packed path quantizes the *activation* to int8 to feed its wide loads. Decode
runs on the mat-vec, so that is where the difference shows, and it is now the
larger of the two effects by two orders of magnitude.

Measured, and the two crates land on opposite sides of exactly that:

| | prompt | worst logit difference | logit span |
| --- | --- | --- | --- |
| `xabe-translate` 13 B `Q4_K_M` | 30 tokens | **0.000000** | 28.655 |
| `xabe-chat` 8 B `Q4_K_M` | 14 tokens | 0.167409 | 25.323 |

The translator's prompt is past `GEMV_MAX_M`, so every projection takes the
tiled kernel, no activation is quantized, and the two paths are bit-identical -
its translation is character-identical as well. The chat prompt is not, so its
projections run on the mat-vec, and 0.167 against a span of 25.3 is **0.66%**,
which is what int8 activations cost. It was 0.004566 before that change.

That number is a bound on the arithmetic, not on the output. What it costs in
*tokens* is measured separately and is zero: greedy decoding against the
`llama-server` capture picks the same token at every position it picked before,
and the disagreement list below is byte-identical across the change.

### Where that disagreement lives, and where it does not

The five above are the ones with a wide margin; the full count is **10 of 105
teacher-forced decisions**. They are not a wiring bug and they are not spread
across the engine. Four measurements place them:

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

### The chat model disagrees with llama-server in five places, and did before

`xabe-chat`'s `tests/llama_server.rs` compares greedy replies against a capture
from `llama-server` running the same GGUF. On the current capture it **fails**,
at five positions across seven prompts, with llama-server's margin between 0.60
and 2.86 logits - too wide to be rounding.

This is not caused by anything in the packed or int8 work. It was confirmed by
capturing the oracle, running the test, stashing every change, running it again
on the unmodified tree, and comparing: the same five positions, the same
margins, to the last digit. It reproduces identically after the wide mat-vec,
the int8 activation, the head-major cache and every fusion.

So it is a real, pre-existing divergence and it is unexplained. It is recorded
here rather than fixed because finding it needs per-layer taps against an oracle
this model does not have - it exists as a GGUF and nothing else, which
`tools/oracle/capture_chat_server.py` says at the top. What the test *is* good
for meanwhile is exactly what it was used for here: as a fixed point that any
change to attention, caching or precision must reproduce byte for byte.

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
