# Container

`Dockerfile`, `docker-compose.yml` and `.env` at the repository root. Both
shapes below were built and run on the three-card box before this file said
they worked.

## The constraint that shapes it

There is no `build.rs`, no `cc` crate, and no nvcc. CUDA is reached at runtime
through `cudarc`'s dynamic loading, and kernels are compiled by NVRTC from
strings. So:

- The **builder** stage needs only a Rust toolchain. No CUDA toolkit.
- The **runtime** stage needs the CUDA *runtime* image, not `devel` — NVRTC and
  the driver API are enough.
- CI can build and run the whole test suite on a machine with no GPU. Tests that
  need one skip loudly.

That held up. The runtime image carries `libnvrtc.so.12` and
`libnvrtc-builtins.so.12.4`, `libcuda.so.1` arrives from the host driver
through the NVIDIA container runtime, and a container compiled all 61 kernels
and synthesised audio on the first try.

## The one thing the sketch got wrong: glibc

This file used to open the Dockerfile with `FROM rust:1-slim AS build` over
`FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`. **That pairing does not run.** A
dynamically linked binary needs the glibc it was built against or newer;
`rust:1-slim` is Debian bookworm at 2.36 and the CUDA runtime image is Ubuntu
22.04 at 2.35, so the build succeeds and the container dies at exec with a
missing `GLIBC_2.36`.

The builder is therefore `ubuntu:22.04` — the runtime's own distribution, with
rustup installed bare so `rust-toolchain.toml` stays the single source of truth
for the channel. Nothing CUDA is installed there, so the original constraint is
intact; only the base image changed.

## What must not go in the image

Model weights. `facebook/mms-tts-nan` is CC-BY-NC 4.0 and the translator is
CC-BY-NC-SA 4.0; this repository redistributes neither, and `models/` is
gitignored for the same reason. The tree is mounted at run time.

`.dockerignore` enforces it from the other side, and it is an allow-list rather
than a deny-list on purpose: `models/` is 43 GB and `target/` another 8, and an
exclude-list that forgets one of them does not fail — it ships the context to
the daemon on every build.

The repository layout was chosen to make that mount trivial — `models/` here
has the same shape as `/models` in the container, so the per-stage variables
point at the same relative paths either way:

```
models/asr/breeze-asr-26/       models/tts/mms-tts-nan/
models/vad/                     models/tts/cosyvoice3-0.5b/
models/*.gguf
```

**GGUFs sit at the top of `models/`, not in a subdirectory.** They are single
files rather than checkpoint directories, and the two stages that read them -
`--llm-model` and `--translator-model` - take a path to the file, so a folder
per container bought nothing but a level of nesting.

The three safetensors stages have no GGUF path at all: `xabe-tts`, `xabe-asr`
and `xabe-vad` do not depend on `xabe-gguf` and never will, because their
checkpoints are not published in that container. So the split above is not a
convention to be tidied - it is which crates can read what.

`--translator-model` accepts either container, so a deployment that wants one
copy of those weights can keep the `.gguf` and drop
`models/translator/taigi-llama2-13b/` entirely. The chat model has no such
choice: it is GGUF-only, because its vocabulary lives inside the file.

## Environment

Every CLI flag has an `env` twin (see [CLI.md](CLI.md)), so the container is
configured without rewriting argv:

```sh
docker run --gpus all \
  -e XABE_TTS_MODEL=/models/tts/mms-tts-nan \
  -e XABE_TTS_DEVICE=0 \
  -e RUST_LOG=info \
  -v /srv/models:/models:ro \
  ttsxabe --text "lí hó." --out -
```

A stage the container is not meant to run is configured by *leaving its
variables unset*: absent means off, so one image serves as the ASR worker, the
TTS worker or the whole pipeline depending only on which variables are present.

**Absent, not empty.** `clap` reads an empty `XABE_TRANSLATOR_MODEL` as the
path `""` and fails inside a weight schema rather than as "no translator", so
`docker-compose.yml` omits the optional names instead of defaulting them to the
empty string. Two more sharp edges of the same kind:

- `XABE_DIRECT_TAIGI` is a `bool` in its env form and takes `true` or `false`.
  Not `1` — that is `invalid value '1' for '--direct-taigi'`.
- `XABE_SERVE` must bind `0.0.0.0`, not the `127.0.0.1` of the host-side
  examples in [CLI.md](CLI.md). A server on loopback inside a container is
  reachable from nothing, its own healthcheck included.

The entrypoint is the `xabe-engine` binary itself and `command:` is the
argument list, so `docker run ... --help` prints the real flag surface rather
than a shell's.

**The system prompt is one of these variables**, which is the point of it
having an inline form: `XABE_SYSTEM_PROMPT` carries the whole prompt, and
`XABE_PROMPT_FILE` points at a file for anything long enough that a variable is
the wrong shape. They are alternatives; both is refused. [CLI.md](CLI.md) has
the rules - the important ones being that a given prompt replaces the built-in
whole, and must be written in whatever script the configured synthesiser reads.

`prompts/` is bind-mounted read-only at `/prompts` for that second form. It is
**gitignored**, because a system prompt names a character and whose character
that is differs per deployment, so the repository ships the engine's built-ins
and nothing else. The mount is unconditional and an empty directory is fine:
with `XABE_PROMPT_FILE` unset the engine uses a built-in and never looks. Both
lines ship commented in `.env` for that reason - a committed default pointing
into a gitignored directory would leave a fresh clone unable to start.

Uncomment `XABE_PROMPT_FILE` and `XABE_BOT` together. A file is read literally,
with no `{person}`/`{bot}` substitution, so the name written into it has to be
the name the transcript uses; the stop strings are derived from `XABE_BOT`, and
a mismatch lets the model write the user's next turn itself.

## The two shapes

`.env` sets `COMPOSE_PROFILES=mono`, so the bare command is the monolith and
`--profile split` replaces that value rather than adding to it. Both services
publish the same port, which is why each shape is profiled: a service with no
`profiles` key is enabled unconditionally, and the two would collide.

```sh
docker compose up -d --build          # the whole pipeline, one process
docker compose --profile split up -d  # ASR and TTS as workers, gateway in front
```

Measured on one Quadro RTX 8000, checkpoints warm in the page cache: the image
is 2.35 GB, the monolith reports healthy about 70 s after `up`, and it holds
18.7 GB — the chat model and the translator at `Q4_K_M`, the ASR at f16, the
synthesiser rounding to nothing beside them. `POST /tts` of a one-clause line
came back with 2.67 s of audio in 130 ms.

`start_period` is 600 s and that is not padding. Loading is tens of gigabytes
before the first NVRTC compile, and without it a container that is still
mapping its checkpoints reads as a crash loop rather than as `starting`.

### Why the monolith carries the translator

`--direct-taigi` is the faster path — [BENCHMARKS.md](BENCHMARKS.md) measures a
voice turn at 3.8 s against 1.6 — and it is *not* the default here, because of
what the synthesiser reads. `mms` reads POJ; direct Taigi has the chat model
answer in Han; the engine refuses the pair by name:

```
engine `mms` reads `POJ` and there is no translator to produce it;
give --translator-model, or use an engine that reads HAN
```

So the default is the full pipeline, translator included, which is also the
layout [CLI.md](CLI.md) says the packed checkpoints leave room for on one card.
To take the faster path instead, drop `XABE_TRANSLATOR_MODEL`, set
`XABE_DIRECT_TAIGI=true`, and give it an engine that reads Han:

```yaml
      XABE_TTS_ENGINES: cosyvoice=/models/tts/cosyvoice3-0.5b
      XABE_TTS_SCRIPTS: cosyvoice=HAN
      XABE_TTS_DEFAULT: cosyvoice
```

No `XABE_TTS_MODEL` in that set, and it does not need one: `--tts-engine`
stands alone and is the TTS stage by itself. Run as written it reports
`stages: ["asr","llm","tts"]` with `engines: ["cosyvoice"]` and no translator,
comes up in 35 s against the full pipeline's 70, and speaks a clause of Han in
3.6 s at 24 kHz. That last figure is one call, not a benchmark - the
CosyVoice numbers that mean anything are in [BENCHMARKS.md](BENCHMARKS.md),
which is also where its caveats are.

On a second card, `XABE_TRANSLATOR_DEVICE=1` is the split that matters: the
chat model and the translator decode at the same time, and
[BENCHMARKS.md](BENCHMARKS.md) measures first audio at 2659 ms with both on one
card against 2000 ms with the translator moved.

### The split profile no longer needs a workaround

This section used to describe one, and it was a lie: the `tts` worker set
`XABE_TTS_SCRIPTS: mms=HAN` because serving a *local* engine whose script was
not HAN was refused unless the same process held a translator. That check
assumed the process holding the synthesiser also holds the reply path, and a
split worker is exactly the case where it does not — the gateway owns the
translator, translates the reply, and POSTs POJ to a worker whose only job is
to speak what it is handed.

It is gated on the chat model now. The script is read by `script_for`, reached
only from the converse path, which needs an LLM; a process without one is never
asked what its engines read. So a synthesiser-only worker comes up on
`XABE_TTS_MODEL` alone, and [CLI.md](CLI.md)'s split example — `xabe-engine
--serve 127.0.0.1:8100 --tts-model models/tts/mms-tts-nan --tts-device 1` —
runs as written. Give that worker an LLM and the check applies again, which is
the point: that is the configuration where a Han reply into `mms` would
synthesise near-silence.

## What is not here

The image ships `xabe-engine` and nothing else. The four benchmark binaries in
the same crate need a card *and* a checkpoint before they can say anything, so
they belong on the host that has both rather than in a deployment image.

There is no llama.cpp service. The chat model runs in-process from a GGUF here,
which is what `XABE_LLM_MODEL` does; a deployment that would rather keep it in
`llama-server` sets `XABE_LLM_URL` at that process and leaves the model
variable unset.
