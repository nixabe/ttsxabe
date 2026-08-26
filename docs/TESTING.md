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
