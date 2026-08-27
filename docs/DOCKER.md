# Container

**Not written yet.** Recorded so the constraints are not rediscovered later.

## The constraint that shapes it

There is no `build.rs`, no `cc` crate, and no nvcc. CUDA is reached at runtime
through `cudarc`'s dynamic loading, and kernels are compiled by NVRTC from
strings. So:

- The **builder** stage needs only a Rust toolchain. No CUDA toolkit.
- The **runtime** stage needs the CUDA *runtime* image, not `devel` — NVRTC and
  the driver API are enough.
- CI can build and run the whole test suite on a machine with no GPU. Tests that
  need one skip loudly.

```dockerfile
FROM rust:1-slim AS build
# ... cargo build --release --workspace

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# ... copy the binary
```

## What must not go in the image

Model weights. `facebook/mms-tts-nan` is CC-BY-NC 4.0 and the translator is
CC-BY-NC-SA 4.0; this repository redistributes neither, and `models/` is
gitignored for the same reason. The tree is mounted at run time.

The repository layout was chosen to make that mount trivial — `models/` here
has the same shape as `/models` in the container, so the per-stage variables
point at the same relative paths either way:

```
models/asr/breeze-asr-26/       models/tts/mms-tts-nan/
models/vad/                     models/translator/taigi-llama2-13b/
models/llm/                     models/tts/cosyvoice3-0.5b/
```

`models/llm/` holds GGUFs, and both models in it are now readable by this
engine directly. `--llm-model` takes the chat GGUF - which is also where a
delegated `--llm-url` llama-server would read from - and `--translator-model`
accepts a `.gguf` from the same directory, since the translator takes either
container. So a deployment that only wants one copy of those weights can drop
the safetensors directory and point at the GGUFs, and one that wants no second
runtime at all can drop llama-server with it.

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

The entrypoint is the `xabe-engine` binary itself and `command:` is the
argument list, so `docker run ... --help` prints the real flag surface rather
than a shell's.
