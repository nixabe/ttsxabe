# AGENTS.md

Operating instructions for AI agents working in this repository.

## What this project is

`ttsxabe` is a from-scratch Taiwanese Hokkien speech synthesiser: a Rust
reimplementation of VITS as shipped in `facebook/mms-tts-nan` (36.3 M
parameters, 16 kHz, Tâi-lô input), targeting 3× Quadro RTX 8000 (sm_75).

It exists because TTS is the one stage of the Taigi voice pipeline still running
unoptimised PyTorch. Everything upstream — Whisper and the LLM — is already
hand-tuned CUDA in `whisper.cpp` and `llama.cpp`, where a rewrite buys nothing.
Measured on that pipeline: the ASR contributed 0 ms once prefetched, the LLM
~120 ms to first clause, and TTS ~1.4 s. **This project targets the 1.4 s.**

Porting Whisper or the LLM here is explicitly out of scope until this
synthesiser is finished and measured.

## Current standing

There is no standing yet. The reference PyTorch implementation is the only thing
that has produced audio. When this engine does, the comparison belongs in
`docs/BENCHMARKS.md` and nowhere else.

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

VITS drifts silently. A wrong flow coupling, a transposed convolution kernel
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
| `xabe-vits` | model config, weight schema, shape validation | `xabe-st` |
| `xabe-dsp` | CPU reference kernels + differential compare harness | `xabe-vits` |
| `xabe-tts` | forward pass, synthesis API, CLI | all |

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
