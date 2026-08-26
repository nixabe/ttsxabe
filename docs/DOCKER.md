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

Model weights. `facebook/mms-tts-nan` is CC-BY-NC 4.0 and this repository does
not redistribute it. The checkpoint is mounted or downloaded at run time, and
`XABE_TTS_MODEL` points at it.

## Environment

Every CLI flag has an `env` twin (see [CLI.md](CLI.md)), so the container is
configured without rewriting argv:

```sh
docker run --gpus all \
  -e XABE_TTS_MODEL=/models/mms-tts-nan.safetensors \
  -e RUST_LOG=info \
  -v /srv/models:/models:ro \
  ttsxabe --text "lí hó." --out -
```
