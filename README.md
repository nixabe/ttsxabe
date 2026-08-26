# ttsxabe

A from-scratch Rust engine for a Taiwanese Hokkien (Taigi) voice assistant.

No ML framework, no bindings. The container readers, the weight schemas and the
kernels all live in this repository and are verified against captured
references.

One binary, `xabe-engine`, runs every stage of the pipeline except the chat
LLM, which stays in llama.cpp by decision. Which stages *this* process runs is
decided by flags, and each stage is satisfied either locally
(`--<stage>-model`) or by another process over HTTP (`--<stage>-url`) — so the
same binary is a monolith, a single-stage worker, or anything between.

Finished: the synthesiser. In progress: the rest. `docs/MILESTONES.md` has the
phases and `docs/CLI.md` the flag surface.

## Why this exists

It began as the synthesiser alone, which was the only stage of the Taigi voice
pipeline still running unoptimised PyTorch. Everything upstream was already
hand-tuned CUDA — measured, on one voice turn of that pipeline:

| stage | engine | time |
| --- | --- | --- |
| ASR | whisper.cpp, CUDA | ~0 ms once prefetched during the pause window |
| LLM → first clause | llama.cpp, CUDA | ~120 ms |
| **TTS** | **PyTorch** | **~1.4 s** |
| orchestration | Python | <100 ms |

TTS was roughly 85% of what was left, and that is the number this project was
built to move. It has been moved.

What widened the scope afterwards was not speed but the seams. Making that one
stage fast left a working system of **7 processes, 2 Python environments, 2
languages and 6 ports**, held together by a shell script — and the interesting
problems that remain are in the orchestration as much as the arithmetic. So the
engine is absorbing the rest of it. The chat LLM stays in llama.cpp
permanently; rewriting that would buy nothing.

## Target hardware and model

3× Quadro RTX 8000 — Turing, sm_75, 48 GB, 672 GB/s each. Turing has fp16
tensor cores, no bf16 and no fp8, which rules out several otherwise obvious
optimisations and is why `docs/OPTIMIZATION.md` exists.

The model is `facebook/mms-tts-nan`: VITS, 36.3 M parameters in 762 F32
tensors, 662 of them read at inference, 16 kHz output, waveform out. Its
48-symbol vocabulary is **Pe̍h-ōe-jī, not Tâi-lô** - it has `c` and U+0358 and
no `ts` - which is not what the model card says and is the single most
consequential thing to get right about it. `docs/MODEL.md` has the geometry and
the evidence.

## Status

**The synthesiser is complete**: text in, waveform out, on CPU and on CUDA,
matching the PyTorch reference stage by stage. The ASR, the VAD, the translator
loader and the serving layer are not built yet; their flags are, and each fails
naming the phase that builds it.

| crate | state |
| --- | --- |
| `xabe-st` | safetensors container, validated addressing |
| `xabe-golden` | reads the captured PyTorch oracle, verifies its checksums |
| `xabe-vits` | config, weight schema for all 662 inference tensors, tokenizer |
| `xabe-dsp` | scalar reference kernels |
| `xabe-cuda` | 22 CUDA kernels, each diffed against its scalar twin |
| `xabe-tts` | VITS forward pass on both devices, synthesis API, benchmark |
| `xabe-audio` | WAV reading and writing, sample handling |
| `xabe-engine` | the binary: flags, stage wiring, orchestration |

Correctness, against tensors captured from 🤗 `VitsModel`:

| stage | max absolute error |
| --- | --- |
| tokenizer | exact, 21/21 cases |
| text encoder | 3.8e-6 |
| duration predictor | 1.0e-5 |
| flow, reversed | 1.9e-6 |
| decoder | 1.7e-6 |
| **end to end, CPU** | **8.3e-6** |
| **end to end, CUDA** | **1.2e-5** |

And, because numerical agreement does not prove a file is *speech*: synthesising
`lí hó, kin-á-ji̍t thinn-khì chin hó.` and transcribing the result with
Breeze-ASR-26 - a model with no part in producing it - returns
你好 今天天氣很好, which is what the input means.

Speed, one Quadro RTX 8000, medians of 20 runs alternated with the baseline:

| | 2.6 s of audio | x realtime |
| --- | --- | --- |
| PyTorch, CUDA | 65.6 ms | 43.2 |
| **`xabe-tts`, CUDA** | **48.4 ms** | **53.9** |
| `xabe-tts`, CPU (scalar reference) | ~120 s | 0.02 |

**1.24x faster than PyTorch** per second of audio. `docs/BENCHMARKS.md` has the
stage breakdown, the computed FLOP ceiling, and the things that did not work.

## Using it

```sh
xabe-engine --tts-model models/tts/mms-tts-nan --tts-device 0 \
            --text "lí hó, kin-á-ji̍t thinn-khì chin hó." --out hello.wav
```

Every model lives under `models/`, which is gitignored: one tree to populate,
nothing tracked. `docs/CLI.md` has the whole flag surface, including the
topologies that split stages across processes.

Input must be NFC-normalised POJ. Anything outside the 48 symbols is deleted
silently - that is the reference's behaviour, and `docs/MODEL.md` explains why
it matters more than it sounds.

## Building

```sh
cargo build --workspace
cargo test --workspace --release
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Tests that need the checkpoint look in `models/tts/mms-tts-nan` first, fall
back to the HuggingFace cache, and take `XABE_TTS_MODEL` over both. Differential tests also need a
capture — `python tools/oracle/capture.py --out .golden/base --seed 0 --text
"..."`, see [docs/ORACLE.md](docs/ORACLE.md). Without either they print `SKIP:`
and the reason.

**A skipped test is not a passing test**, and this repository learned that the
hard way: twelve CUDA kernel tests once reported green while every one of them
had skipped, because the harness treated a kernel compile error as an absent
GPU. They now skip only when there is genuinely no device and fail on anything
else. Run with `--nocapture` if you want to see which skipped.

## Scope

The synthesiser speaks Taigi from Pe̍h-ōe-jī romanisation. It does not do
grapheme-to-phoneme conversion and does not know what Han characters are - text
made only of them tokenises to nothing and is refused rather than returned as
silence. Producing POJ from Mandarin is the translator's job, and the
translator is phase 5.

The chat model is out of scope permanently and is reachable only as
`--llm-url`. There is no `--llm-model`.

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
