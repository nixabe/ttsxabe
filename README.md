# ttsxabe

[![CI](https://github.com/nixabe/ttsxabe/actions/workflows/ci.yml/badge.svg)](https://github.com/nixabe/ttsxabe/actions/workflows/ci.yml)

A from-scratch Rust engine for a Taiwanese Hokkien (Taigi) voice assistant.

> **NOTICE**: This project was created for experiment and was fully driven by
> **AI Agent**. It is NOT fully optimized, tested on multiple devices or
> production-ready. Issues and undercover bugs are expected.

No ML framework, no bindings. The container readers, the weight schemas and the
kernels all live in this repository and are verified against captured
references.

One binary, `xabe-engine`, runs every stage of the pipeline. Which stages
*this* process runs is decided by flags, and each stage is satisfied either
locally (`--<stage>-model`) or by another process over HTTP (`--<stage>-url`)
— so the same binary is a monolith, a single-stage worker, or anything
between.

Finished: the synthesiser, the serving layer, voice activity detection, speech
recognition, Mandarin-to-Taigi translation, the chat model, CosyVoice3, and a
third synthesiser, Tacotron2 + WaveGlow, from
[taiwanese_tonal_tlpa_tacotron2](https://github.com/yfliao/taiwanese_tonal_tlpa_tacotron2).
A fourth reuses the first: `xabe-tts` also reads
[coqui-vits-suisiann-minnan-hokkien](https://huggingface.co/neurlang/coqui-vits-suisiann-minnan-hokkien),
which is the same VITS from a different trainer at 22.05 kHz, with no stage of
the forward pass changed — a new container, a new naming scheme and an IPA
vocabulary were all it needed. It reads IPA rather than romanisation, and
`xabe-taigi` transliterates the pipeline's POJ into it; see `docs/MODEL.md`.
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
engine is absorbing the rest of it.

This paragraph used to end by saying the chat LLM stayed in llama.cpp
permanently because rewriting it would buy nothing. Both halves are retracted.
It is written here, and both models are now **level with or ahead of
llama.cpp on every measured row** on this card. The chat model leads on all
three of its rows — 2447 against 2259 tokens per second of prefill at 128
tokens, 2928 against 2513 at 512, 100.9 against 95.3 decoding — and the
translator leads on decode, is level at 512-token prefill, and sits at 0.94x
of a 128-token llama.cpp median that swings 20% between its own runs. Prefill
was 3.5x behind once, then 1.6x, then 0.92x; the last stretch was the integer
tensor cores, a fused attention kernel, and a memory-pool attribute.
`docs/BENCHMARKS.md` has every one of those numbers with llama.cpp's own
repeat spread beside them, and does not round any of them kindly — the
translator's 128-token cell is recorded as inside llama.cpp's noise, not as
won, and the decode leads are noted to have appeared partly because
llama.cpp's own column moved between sittings.

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
out — with every stage either in-process or delegated over HTTP, and nothing
downstream able to tell which.

**Voice activity detection is complete**: Silero from scratch, agreeing with
whisper.cpp on every segment of an eight-clip corpus, and refusing all four of
the noise cases that used to make the ASR invent sentences.

**Speech recognition is complete and correct, and ahead of what it replaces
on every clip measured - by 2% at three seconds of speech and by 9% to 21%
from five seconds up.** Whisper large-v2 from
scratch — 1,259 tensors, a general-radix mel frontend, a byte-level BPE, a
tensor-core matmul, encoder and decoder matching a captured oracle layer by
layer, and greedy decoding reproducing 🤗 `WhisperForConditionalGeneration`'s
transcripts token for token.

| clip | `xabe-asr` | `whisper-server` | ratio |
| --- | --- | --- | --- |
| 2.93 s | 185.9 ms | 189.4 ms | **1.02x** |
| 4.98 s | 220.8 ms | 239.8 ms | **1.09x** |
| 7.28 s | 243.5 ms | 266.6 ms | **1.09x** |
| 9.95 s | 291.8 ms | 353.8 ms | **1.21x** |

Both halves of every row were measured in one sitting, against a
`whisper-server` built from this repository's own `whisper.cpp` checkout with
`GGML_CUDA=ON`, reading the same checkpoint converted by that tree's own
script, both sides strictly single-pass greedy and neither using VAD. An
earlier version of this section quoted 0.55x and then 0.68x; those were two
sittings arithmetically combined, and they are superseded rather than
corrected downward.

**It is recorded as a milestone won, by the margin the numbers give and no
more**: the 2.93 s clip is 3.5 ms ahead with both engines' twenty-round
spreads not overlapping, after two sittings at 1.3 ms behind. The encoder is
still about 104 ms against 83, and `docs/BENCHMARKS.md` argues that this
kernel shape has run out of room on sm_75 rather than that a trick is
missing; everything that moved came from the decoder, which had been set
aside as level and was spending twenty-two launches a layer on one token -
thirteen after one round, eight after the second.

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
| `xabe-vits` | config, weight schema and tokenizer, in both published dialects |
| `xabe-dsp` | scalar reference kernels |
| `xabe-cuda` | 75 CUDA kernels, each diffed against its scalar twin |
| `xabe-tts` | VITS forward pass on both devices, synthesis API, benchmark |
| `xabe-cosy` | CosyVoice3: speech LM, flow, vocoder, Qwen2 BPE, voice bundles |
| `xabe-taco` | Tacotron2 + WaveGlow, POJ to Tâi-lô, converted weights |
| `xabe-audio` | WAV reading and writing, sample handling |
| `xabe-serve` | HTTP, WebSocket, the web page, the conversation |
| `xabe-vad` | Silero voice activity detection, 15 tensors, from scratch |
| `xabe-whisper` | Whisper geometry, 1,259 tensors, byte-level BPE, mel frontend |
| `xabe-asr` | the Whisper forward pass and greedy decoding, CUDA only |
| `xabe-gguf` | GGUF container, mmap, metadata, nine block formats unpacked |
| `xabe-pt` | torch `.pth` container: zip, a state-dict pickle, validated addressing |
| `xabe-taigi` | POJ, Tâi-lô and IPA, and the conversions three checkpoints need |
| `xabe-llama` | Llama geometry from `config.json` or a GGUF, SentencePiece |
| `xabe-translate` | the Llama-2 forward pass and the `[TRANS]` template, CUDA only |
| `xabe-chat` | the chat model's forward pass, sampling and stop handling |
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
            --llm-model models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf
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
`Q5_1`, `Q8_0` and `Q2_K` through `Q6_K`. That used to be a smaller file and not
a smaller model, because the weights were unpacked on read and landed at full
width. They now stay **packed on the card** and are unpacked inside the matmul,
so a quantized model costs about what its file costs — which is what lets every
stage share one card. That used to be a residency claim and not a speed one;
it is both now, because the packed mat-vec stopped wasting its loads: the same
file widened to f16 reads 2.6x more bytes a token and decodes slower. The
embedding table is still widened at load, and only the matmul reads packed
blocks. See `docs/KERNELS.md` and `docs/BENCHMARKS.md`.

There is no request scheduler here and no reuse between calls: a reply is
prefilled from scratch each turn out of `--history-turns`, and an utterance is
one forward pass that shares nothing with the next. There *are* KV caches
**within** a call — the ASR's decoder keeps one, and both Llama stages hold
theirs at f16, growing by doubling from 256 positions — which is a buffer, not
a scheduler. If batching arrives it will be because a measurement asked for
it.

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
| `taiwanese_tonal_tlpa_tacotron2` | BSD-3-Clause, from NVIDIA's Tacotron2 |
| `coqui-vits-suisiann-minnan-hokkien` | CC-BY-SA 4.0 — **share-alike** |

The chat model and the VAD carry their own terms too; check the card before you
ship anything. The non-commercial pair is the part that surprises people, which
is why it is a table rather than a sentence.
