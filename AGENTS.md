# AGENTS.md

Operating instructions for AI agents working in this repository.

## What this project is

`ttsxabe` is a from-scratch Rust engine for a Taiwanese Hokkien voice
assistant, targeting 3× Quadro RTX 8000 (sm_75). No ML framework, no bindings:
it reads the published checkpoints directly and does the arithmetic itself.

It began as a synthesiser alone — a reimplementation of VITS as shipped in
`facebook/mms-tts-nan` (36.3 M parameters, 16 kHz, Tâi-lô input) — because TTS
was the one stage of the pipeline still running unoptimised PyTorch, at ~1.4 s
against the ASR's 0 ms and the LLM's ~120 ms to first clause. That is finished
and measured: 1.24× faster than PyTorch, stage-by-stage against a captured
oracle.

**The scope has since widened, by decision.** The engine is becoming every
stage of the pipeline except the chat LLM, which stays in llama.cpp: ASR
(Whisper large-v2 fine-tune), voice activity detection (Silero), the
Mandarin-to-Taigi translator's loader, turn-taking and the web front end, all
in one binary with per-stage flags. `docs/MILESTONES.md` has the phases.

An earlier version of this file said porting Whisper or the LLM here was
explicitly out of scope. That was the right rule while the synthesiser was
unfinished; it is retracted now that it is finished and measured. The chat LLM
remains out of scope permanently — llama.cpp is not a rewrite that buys
anything.

## Current standing

The synthesiser is finished: end-to-end CUDA agreement with the captured oracle
at 1.2e-5, and 1.24× faster than the PyTorch reference on interleaved medians.
That number and every other comparison belong in `docs/BENCHMARKS.md` and
nowhere else.

Nothing else has been measured, because nothing else has been built.

Do not write comments, commit messages, or documentation asserting a speedup
that has not been measured on this hardware.

## Non-negotiable design rules

1. **Reject invalid state in the constructor, not at use.** A checkpoint of the
   wrong geometry must fail while loading, naming the tensor. VITS fails
   *quietly* when weights are misread — it keeps producing audio, just wrong
   audio — so a shape that is never checked is a bug that ships.

2. **Never cast mapped bytes to `f32` without proving alignment.** safetensors
   does not guarantee a 4-byte-aligned data segment; every real producer pads
   the header, and nothing forces them to. `xabe-st` validates and refuses.
   This was found by a test, not by reading the spec.

3. **Every numeric kernel ships with a CPU reference and a differential test.**
   Thresholds are per-tensor max-abs and cosine similarity against the reference,
   not "the audio sounds fine". A kernel without a passing differential test is
   not done, regardless of how fast it runs.

4. **The reference implementation is the oracle, and it is captured, not
   described.** Intermediate activations come from the PyTorch model as binary
   goldens under `.golden/`. Do not hand-transcribe expected values.

## Correctness before speed

Every model here drifts silently, and VITS is only the clearest case. A wrong flow coupling, a transposed convolution kernel
read in the wrong order, an off-by-one in the duration expansion — all of these
produce *plausible speech* that is subtly wrong, and no listening test on a
language you do not speak will catch it. The differential harness is the only
thing standing between this project and confident nonsense.

## Crate map

Dependencies point one way. If a crate needs to know something about a crate
below it, the abstraction is wrong — fix the boundary, do not add the edge.

| Crate | Owns | Depends on |
| --- | --- | --- |
| `xabe-st` | safetensors container parsing, mmap, tensor addressing | — |
| `xabe-audio` | WAV containers, sample handling, framing | — |
| `xabe-vits` | model config, weight schema, shape validation | `xabe-st` |
| `xabe-dsp` | CPU reference kernels + differential compare harness | `xabe-vits` |
| `xabe-cuda` | CUDA kernels and the device handle | — |
| `xabe-tts` | the VITS forward pass and its API | all of the above |
| `xabe-serve` | HTTP, WebSocket, the page, the conversation | model internals |
| `xabe-engine` | flags, stage wiring, orchestration, the binary | all |

| `xabe-vad` | Silero geometry, weights and forward pass | audio capture |

Crates that the plan adds and that do not exist yet: `xabe-whisper` (phase 4),
`xabe-llama` (phase 5a). Their flags exist already and fail with the phase they
are waiting on.

## House style

- **No `anyhow`.** One `error.rs` per crate, `thiserror` enum, every variant
  doc-commented with what it prevents.
- `X::new(..) -> Result<Self, XError>` that rejects bad input. No builders, no
  traits, no generics unless forced.
- Every module opens with a `//!` header: why it exists, what it refuses to do,
  where to start reading.
- `tracing` only. `println!` is forbidden outside tests.
- `clap` derive, flat `Args`, doc comments as `--help`.
- Tests run in release: `cargo test --workspace --release`.
- A test that needs the checkpoint and cannot find it prints `SKIP:` and why.
- Conventional Commits scoped to the crate. Measurements go in the commit body.
