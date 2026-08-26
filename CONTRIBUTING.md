# Contributing

[AGENTS.md](AGENTS.md) covers the same ground with more prescription, for
automated contributors. This is the human version.

## Setup

```sh
rustup toolchain install nightly      # rust-toolchain.toml pins it
cargo test --workspace --release
```

You need the checkpoint for the tests that read it:

```sh
huggingface-cli download facebook/mms-tts-nan --local-dir models/tts/mms-tts-nan
```

`models/` is gitignored and holds every model the pipeline uses. Tests look
there first, fall back to the HuggingFace cache, and take `XABE_TTS_MODEL` over
both.

A GPU is not required to build or to run most tests. Kernel tests that need one
skip when it is absent, and say so.

## Where to start

Read in this order: [docs/MODEL.md](docs/MODEL.md) for what VITS actually
computes, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the crates split
it up, then [docs/TESTING.md](docs/TESTING.md) before you write a kernel.

The work queue is [docs/MILESTONES.md](docs/MILESTONES.md).

## Design rules that are not up for negotiation

These are in [AGENTS.md](AGENTS.md) with their reasoning. In short:

1. Constructors reject invalid state; a wrong-geometry checkpoint fails at load,
   naming the tensor.
2. Mapped bytes are never cast to `f32` without proven alignment.
3. Every numeric kernel has a CPU reference and a differential test against it.
4. Expected values are captured from the reference implementation, never
   hand-transcribed.

## Correctness standard

VITS fails quietly. A transposed convolution read in the wrong order, an
off-by-one in duration expansion, a flow coupling with its halves swapped — each
produces fluent, confident, wrong speech. If you do not speak Taigi you will not
hear it, and if you do speak Taigi you will hear it only sometimes.

So: a kernel is done when its differential test passes, not when the output
sounds like a person. Report max-abs *and* cosine similarity; a good cosine with
a bad max-abs is a scale bug, and the reverse is usually a layout bug.

## Porting from upstream

The reference is 🤗 Transformers' `VitsModel`
(`transformers/models/vits/modeling_vits.py`) and, behind it, the original VITS
paper. Port *algorithms*, not code. When you do, cite the file and function in
the commit message so the next person can diff against the same thing you did.

Where this implementation deliberately differs from the reference, say so in a
comment at the point of difference, with the reason.

## Commits

Conventional Commits, scoped to the crate:

```
perf(xabe-dsp): fold the resblock bias into the preceding convolution
fix(xabe-vits): read conv weights as [out, in, k], not [in, out, k]
test(xabe-dsp): gate the CUDA decoder against the CPU reference
```

Subjects are lowercase and describe the mechanism. Numbers and method go in the
**body**, not the subject and not the docs — `docs/BENCHMARKS.md` records the
current state only, and the story of how it got there lives in `git log`.

Negative results deserve commits too. A measured rejection saves the next person
the same week.

## Things that must never be committed

- Model weights. `*.safetensors` is gitignored; keep it that way.
- Captured goldens. `.golden/` is regenerable — see [docs/ORACLE.md](docs/ORACLE.md).
- Generated audio.
- A performance claim that has not been measured on this hardware.

## Console output

`println!` is forbidden outside tests. Use `tracing`. Tool output that a user is
meant to read is `info!`, because INFO is the level that appears by default.

## Reporting results

Say what you measured, on what, how many times, and what varied. "Faster" is not
a result. Wall-clock on one run of one utterance is not a result either — this
card thermally drifts, and the difference between a real 5% and drift is several
alternating pairs.
