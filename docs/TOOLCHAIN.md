# Toolchain

## Rust

`rust-toolchain.toml` pins nightly with `rustfmt`, `clippy`, `rust-src`.
Verified on `rustc 1.99.0-nightly (d453bdd8f 2026-08-14)`.

Nightly is not for the sake of it. Edition 2024 with `rust-version = "1.88"` and
let-chains (`if let Some(x) = &y && !x.is_file()`) are used freely, and pinning
the channel means a toolchain update is a commit rather than a surprise.

## CUDA

`cudarc` with `fallback-dynamic-loading`, kernels as a `const &str` compiled at
runtime via NVRTC. All of `xabe-cuda` is one translation unit.

The consequence is worth stating early, because it shapes the build: **there is
no `build.rs`, no `cc` crate, no nvcc, and no CUDA feature flag.** The workspace
compiles and its non-GPU tests run on a machine with no CUDA toolkit and no GPU.
GPU-ness is a runtime skip, not a compile-time `cfg`.

Target: sm_75 (Turing). Driver 595.84, CUDA 12.4 on the development host.

## Dependencies

Deliberately few. Current set, all workspace-pinned:

| crate | why |
| --- | --- |
| `memmap2` | mapping the checkpoint without reading 139 MB into a Vec |
| `serde`, `serde_json` | the safetensors JSON header |
| `thiserror` | error enums; there is no `anyhow` in this workspace |
| `tracing`, `tracing-subscriber` | all output; `println!` is forbidden outside tests |
| `clap` | CLI, derive + `env` |
| `half`, `rustc-hash` | reserved for the kernel work |

No ML framework. No `candle`, `tch`, `ort`, `ndarray`, or bindings to
whisper.cpp or llama.cpp. If one appears, it needs an argument in a commit body.

## Build profiles

`release` is `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `debug = 1` —
symbols kept for profiling.

`profile.dev.package."*"` is `opt-level = 2`, because the reference kernels run
the full forward pass thousands of times in the differential harness and a plain
debug build makes that unusable.
