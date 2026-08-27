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
| `XABE_LLM_GGUF` | the Breeze2 chat GGUF | `models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf` |
| `XABE_TRANSLATOR_GGUF` | the translator as a GGUF | `models/llm/taigi-translator-13b-f16.gguf` |
| `XABE_QUANT_DIR` | a directory of quantized Breeze2 copies | none; those tests skip |
| `XABE_CHAT_DEVICE` | the card to load the 8 B chat model onto | none; that test skips |

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
llama-quantize models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
    $XABE_QUANT_DIR/breeze-Q4_K_M.gguf Q4_K_M 8
```

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
of the repository. So **every numerical test skips there**, and the guards that
make that graceful — each test checks for its device and its capture and
prints `SKIP:` with what is missing — are what let the workspace run green on
a machine that can prove none of it.

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
