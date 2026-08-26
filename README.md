# ttsxabe

A from-scratch Rust implementation of VITS for Taiwanese Hokkien (Taigi) speech
synthesis, reading `facebook/mms-tts-nan` weights directly.

No ML framework, no bindings. The container reader, the weight schema and the
kernels all live in this repository and are verified against the PyTorch
reference.

## Why this exists

It is the only stage of the Taigi voice pipeline still running unoptimised
PyTorch. Everything upstream is already hand-tuned CUDA, and rewriting it would
buy nothing — measured, on one voice turn of that pipeline:

| stage | engine | time |
| --- | --- | --- |
| ASR | whisper.cpp, CUDA | ~0 ms once prefetched during the pause window |
| LLM → first clause | llama.cpp, CUDA | ~120 ms |
| **TTS** | **PyTorch** | **~1.4 s** |
| orchestration | Python | <100 ms |

TTS is roughly 85% of what is left. That is the number this project exists to
move. Porting Whisper or the LLM here is out of scope until this synthesiser is
finished and measured.

## Target hardware and model

3× Quadro RTX 8000 — Turing, sm_75, 48 GB, 672 GB/s each. Turing has fp16
tensor cores, no bf16 and no fp8, which rules out several otherwise obvious
optimisations and is why `docs/OPTIMIZATION.md` exists.

The model is `facebook/mms-tts-nan`: VITS, 36.3 M parameters in 762 F32
tensors, 16 kHz output, Tâi-lô romanisation in (48-symbol vocabulary), waveform
out. `docs/MODEL.md` has the geometry.

## Status

**Container reading works. Nothing synthesises yet.**

| crate | state |
| --- | --- |
| `xabe-st` | safetensors container, validated addressing — 11 tests |
| `xabe-vits` | config and weight schema — not started |
| `xabe-dsp` | CPU reference kernels + differential harness — not started |
| `xabe-tts` | forward pass, synthesis API, CLI — not started |

There is no performance claim to make yet, and `docs/BENCHMARKS.md` says so.

## Building

```sh
cargo build --workspace
cargo test --workspace --release
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Tests that need the checkpoint find it in the HuggingFace cache, or take
`XABE_TTS_MODEL=/path/to/model.safetensors`. Without it they print `SKIP:` and
the reason — a skipped test is not a passing test.

## Scope

This synthesises Taigi from Tâi-lô romanisation. It does not do grapheme-to-
phoneme conversion, does not translate, and does not know what Han characters
are. Producing Tâi-lô from Mandarin is the job of the pipeline upstream.

There is no KV cache and no request scheduler here; an utterance is a single
forward pass with no state carried between calls. If batching arrives it will be
because a measurement asked for it.

## Documentation

| | |
| --- | --- |
| [AGENTS.md](AGENTS.md) | the binding rule set |
| [CONTRIBUTING.md](CONTRIBUTING.md) | the same ground, for humans |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | crate boundaries and data flow |
| [docs/MODEL.md](docs/MODEL.md) | VITS geometry and tensor inventory |
| [docs/KERNELS.md](docs/KERNELS.md) | kernel inventory and status |
| [docs/TESTING.md](docs/TESTING.md) | differential testing and tolerances |
| [docs/ORACLE.md](docs/ORACLE.md) | how the PyTorch goldens are captured |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | current standing, WHY, WHY NOT |
| [docs/MILESTONES.md](docs/MILESTONES.md) | what is done and what is next |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | working on this |
| [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) | versions and why they are pinned |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | the performance model for sm_75 |
| [docs/API.md](docs/API.md) | library surface |
| [docs/CLI.md](docs/CLI.md) | command surface |
| [docs/DOCKER.md](docs/DOCKER.md) | container build |

## Licence

MIT OR Apache-2.0.

The **model weights are not**. `facebook/mms-tts-nan` is CC-BY-NC 4.0 —
non-commercial. This repository contains no weights; it reads a file you supply.
