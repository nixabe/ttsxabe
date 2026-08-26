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
huggingface-cli download facebook/mms-tts-nan model.safetensors --local-dir /tmp/mms
export XABE_TTS_MODEL=/tmp/mms/model.safetensors
cargo test --workspace --release
```

They are also found automatically in `~/.cache/huggingface/hub/`.

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
[MODEL.md](MODEL.md).

## Adding a dependency

Justify it in the commit body. The dependency list in
[TOOLCHAIN.md](TOOLCHAIN.md) is short on purpose, and an ML framework would
undo the reason this project exists.
