# ttsxabe

[![CI](https://github.com/nixabe/ttsxabe/actions/workflows/ci.yml/badge.svg)](https://github.com/nixabe/ttsxabe/actions/workflows/ci.yml)

A from-scratch Rust engine for a Taiwanese Hokkien (Taigi) voice assistant.

> **NOTICE**: This project was created for experiment and was fully driven by
> **AI Agent** in 24 hours. It is NOT fully optimized, tested on multiple
> devices production-ready. Issues and undercover bugs are expected.

No ML framework, no bindings. The container readers, the weight schemas and the
kernels all live in this repository and are verified against captured
references.

One binary, `xabe-engine`, runs every stage of the pipeline. Which stages
*this* process runs is decided by flags, and each stage is satisfied either
locally (`--<stage>-model`) or by another process over HTTP (`--<stage>-url`)
— so the same binary is a monolith, a single-stage worker, or anything
between.

Finished: the synthesiser, the serving layer, voice activity detection, speech
recognition, Mandarin-to-Taigi translation, the chat model, and CosyVoice3.
Remaining inside CosyVoice: deriving a **new** voice still runs two ONNX models
once, through `tools/make_cosyvoice_voice.py`. `docs/MILESTONES.md` has the
phases and `docs/CLI.md` the flag surface.

The chat model was originally excluded by decision — llama.cpp already ran it
well, and a second decode loop was not obviously worth writing. That is
retracted: `--llm-model` reads a GGUF and runs it here, verified against
llama-server by teacher forcing at 124 of 125 decisions.

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

**The synthesiser and the serving layer are complete.** Text in, waveform out,
on CPU and on CUDA, matching the PyTorch reference stage by stage; and one
binary that can be the gateway, a single-stage worker, or the whole assistant.
A running engine holds a full voice turn today — speech in, Taigi reply, speech
out — with the ASR and the chat model delegated over HTTP.

**Voice activity detection is complete**: Silero from scratch, agreeing with
whisper.cpp on every segment of an eight-clip corpus, and refusing all four of
the noise cases that used to make the ASR invent sentences.

**Speech recognition is complete and correct, and slower than what it
replaces.** Whisper large-v2 from scratch — 1,259 tensors, a general-radix mel
frontend, a byte-level BPE, a tensor-core matmul, encoder and decoder matching
a captured oracle layer by layer, and greedy decoding reproducing 🤗
`WhisperForConditionalGeneration`'s transcripts token for token. Measured
against `whisper-server` on the same card with the same model and no VAD on
either side, it is **0.55x** — 264 ms against 144 on a 2.7-second clip, with
identical transcripts. That was a stated milestone and it is not met;
`docs/BENCHMARKS.md` computes what closing the gap would take rather than
restating the target.

**CosyVoice3 runs in the engine.** Three networks from scratch — a Qwen2 0.5 B
speech language model, a 22-layer diffusion transformer with its Euler solver,
and a causal HiFi-GAN with an inverse-STFT head and a harmonic source — plus a
Qwen2 byte-level BPE and eleven new CUDA kernels. Against a capture from
CosyVoice3 itself: the language model agrees at 143 of 143 positions on forced
log-probabilities, the flow reaches the reference mel at correlation 0.999970,
and the vocoder reproduces the reference waveform at 1.000000 with a worst
sample of 1e-5. `--tts-engine cosyvoice=<dir>` opens it in-process; deriving a
**new** voice still runs two ONNX models once, through a `tools/` script.

The translator is a Llama-2 13 B, all 363 tensors bound and shape-checked
before a byte is read, BF16 converted to f16 at load because Turing has no
bf16, and a hand-written SentencePiece tokenizer. Its output matches
`llama-server`'s on seven of eight fixed prompts at `temperature = 0`; on the
eighth, the float32 🤗 oracle agrees with *this* engine, so llama-server is the
one that diverges. `docs/ORACLE.md` says why that can happen at all.

| crate | state |
| --- | --- |
| `xabe-st` | safetensors container, validated addressing |
| `xabe-golden` | reads the captured PyTorch oracle, verifies its checksums |
| `xabe-vits` | config, weight schema for all 662 inference tensors, tokenizer |
| `xabe-dsp` | scalar reference kernels |
| `xabe-cuda` | 47 CUDA kernels, each diffed against its scalar twin |
| `xabe-tts` | VITS forward pass on both devices, synthesis API, benchmark |
| `xabe-cosy` | CosyVoice3: speech LM, flow, vocoder, Qwen2 BPE, voice bundles |
| `xabe-audio` | WAV reading and writing, sample handling |
| `xabe-serve` | HTTP, WebSocket, the web page, the conversation |
| `xabe-vad` | Silero voice activity detection, 15 tensors, from scratch |
| `xabe-whisper` | Whisper geometry, 1,259 tensors, byte-level BPE, mel frontend |
| `xabe-asr` | the Whisper forward pass and greedy decoding, CUDA only |
| `xabe-gguf` | GGUF container, mmap, metadata, nine block formats unpacked |
| `xabe-llama` | Llama geometry from `config.json` or a GGUF, SentencePiece |
| `xabe-translate` | the Llama-2 forward pass and the `[TRANS]` template, CUDA only |
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

Or as the whole assistant, with a web page at the address given:

```sh
xabe-engine --serve 127.0.0.1:8000 --direct-taigi \
            --tts-model models/tts/mms-tts-nan --tts-device 1 \
            --asr-url http://127.0.0.1:8080 \
            --llm-model models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf
```

Two synthesisers in one process, and the page chooses between them:

```sh
xabe-engine --serve 127.0.0.1:8000 \
            --tts-model  models/tts/mms-tts-nan       --tts-device 2 \
            --tts-engine cosyvoice=models/tts/cosyvoice3-0.5b \
            --tts-script cosyvoice=HAN
```

mms reads romanisation and CosyVoice reads Han, which is what `--tts-script`
settles per engine rather than per process.

Each stage is satisfied either locally or by another process, and nothing
downstream can tell which — so the same binary is a monolith, a worker, or
anything between. Every model lives under `models/`, which is gitignored: one
tree to populate, nothing tracked. `docs/CLI.md` has the whole flag surface and
the endpoints `--serve` publishes.

Input must be NFC-normalised POJ. Anything outside the 48 symbols is deleted
silently - that is the reference's behaviour, and `docs/MODEL.md` explains why
it matters more than it sounds.

## What CI checks

The badge means the workspace compiles with **no CUDA toolkit at all**, is
formatted, and is clean under `clippy` and `rustdoc` at `-D warnings`. It does
not mean any model is correct: a GitHub runner has no GPU and no checkpoints,
and `.gitignore` keeps `models/` and `.golden/` out of the repository, so every
numerical test skips there. The workflow counts those skips into the run
summary rather than letting a green tick stand for something it did not check.

The gate that does check the models is one command on a machine that has them:

```sh
XABE_COSY_DEVICE=<a free card> cargo test --workspace --release
```

`docs/TESTING.md` has the table of which column proves what.

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
translator is a separate stage with its own model and its own flag.

The chat model runs either way: `--llm-model` loads the GGUF and runs it here,
`--llm-url` delegates it to a llama-server. GGUF only, and no `cpu` device —
see `docs/CLI.md`. Quantized checkpoints load too: `Q4_0`, `Q4_1`, `Q5_0`,
`Q5_1`, `Q8_0` and `Q2_K` through `Q6_K` are unpacked on read. That is a smaller
file, not a smaller model — the weights land at full width either way.

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

Apache-2.0. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for what this was
written against — every model here is a from-scratch implementation verified
against a captured oracle, not a translation of someone's source, but the
references are a real debt either way.

**The model weights are not covered by it, and two of them are
non-commercial.** This repository contains no weights: it reads files you
supply, and their terms are between you and whoever published them.

| checkpoint | licence, as its own model card states |
| --- | --- |
| `facebook/mms-tts-nan` | CC-BY-NC 4.0 — **non-commercial** |
| `Taigi-Llama-2-Translator-13B` | CC-BY-NC-SA 4.0 — **non-commercial, share-alike** |
| `Breeze-ASR-26` | Apache-2.0 |
| `Fun-CosyVoice3-0.5B` | Apache-2.0 |

The chat model and the VAD carry their own terms too; check the card before you
ship anything. The non-commercial pair is the part that surprises people, which
is why it is a table rather than a sentence.
