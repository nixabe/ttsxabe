# Development

## Getting a working tree

```sh
git clone <repo> && cd ttsxabe
cargo test --workspace --release
```

That should be green without a GPU and without the checkpoint. Tests needing
either skip and say so.

To run the tests that read real weights:

```sh
huggingface-cli download facebook/mms-tts-nan --local-dir models/tts/mms-tts-nan
cargo test --workspace --release
```

The second VITS checkpoint is one more download and needs no conversion, since
`xabe-pt` reads the `.pth` as published:

```sh
huggingface-cli download neurlang/coqui-vits-suisiann-minnan-hokkien \
    --local-dir models/tts/coqui-vits-suisiann
```

Its oracle needs Python 3.10 and a torch of its own, which do not coexist with
the rest of the tooling here — [ORACLE.md](ORACLE.md) has the two commands, and
`.venv-coqui/` is gitignored. Nothing but regenerating `.golden/coqui-base` and
`.golden/coqui-tailo` needs it.

`models/` is gitignored and is where every model in the pipeline lives, so
populating it is the whole of provisioning a machine. Tests look there first,
fall back to `~/.cache/huggingface/hub/`, and take an environment variable over
both — [TESTING.md](TESTING.md) lists all of them.

Nothing in `models/` is required. Every test that reads real weights detects
their absence, prints `SKIP:` with the variable to set, and returns, so a clone
with an empty `models/` still runs green. The one directory with no default at
all is `XABE_QUANT_DIR`: those files are multi-gigabyte `llama-quantize`
outputs, derived rather than downloaded, and one command reproduces any of
them.

## The loop

1. Pick the next unticked row in [MILESTONES.md](MILESTONES.md).
2. If it is numeric, capture the oracle stage first ([ORACLE.md](ORACLE.md)).
   The expected values exist before the implementation does.
3. Write the `xabe-dsp` reference. Scalar, obvious, readable against the PyTorch
   source.
4. Write the differential test. Watch it fail for the right reason.
5. Make it pass.
6. Only then consider making it fast, and only with a measurement.

## Before committing

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```

All three, every time. Clippy is `-D warnings`, not advisory.

## Debugging a numeric failure

The differential harness reports max-abs and cosine per tensor. Read the pair
before reading the code:

- good cosine, bad max-abs → scale: a missing normalisation, a wrong constant
- bad cosine, good max-abs → layout: transposed weights, wrong stride
- both bad → the algorithm is wrong

Then bisect by stage. Every intermediate is captured, so you can find the first
tensor that disagrees rather than staring at a waveform.

If everything matches but the audio is wrong, the bug is upstream of the model:
tokenisation, or the orthography of the input text. See the POJ note in
[MODEL.md](MODEL.md), and — on the Coqui checkpoint — remember that its input is
IPA rather than any romanisation, so Han text in gives silence out by design.

One failure shape is worth recognising on sight. If a **late** stage matches and
an **early** one does not, suspect the capture rather than the code: an
arithmetic error large enough to wreck the text encoder cannot leave the
waveform correct, but a transposed comparison looks exactly like that. The two
references disagree about whether the encoder carries `[B, C, T]` or `[B, T, C]`
— [ORACLE.md](ORACLE.md) has the round this cost.

## Adding a dependency

Justify it in the commit body. The dependency list in
[TOOLCHAIN.md](TOOLCHAIN.md) is short on purpose, and an ML framework would
undo the reason this project exists.
