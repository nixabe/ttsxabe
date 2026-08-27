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
| `half` | rounding weights to f16 once, at load — see `docs/KERNELS.md` |
| `rustc-hash` | the tokenizer's vocabulary and merge tables |
| `sha2` | verifying a capture's checksums on read |
| `regex` | GPT-2's pre-tokenization pattern, and the hallucination phrase list |
| `axum`, `tokio`, `async-stream`, `base64` | the serving surface |
| `reqwest` | the client half of `--<stage>-url` |

The serving set is taken as a whole from `llmxabe/crates/xabe-server`, which
already runs it against this card, rather than being re-argued here. `tokio`'s
features are hand-picked rather than `full`: the engine drives GPU work on its
own OS threads and needs the executor for sockets, not for compute. `reqwest`
has default features *off* — every URL the engine dials is another process on
this host over plain HTTP, so a TLS stack would be weight with no user.

`regex` is used for one thing that genuinely needs it: `\p{L}` and `\p{N}` as
the reference defines them. Rust's `char::is_alphabetic` is the Alphabetic
property, which is not `\p{L}` — it also admits `Nl` and `Other_Alphabetic`,
and the difference falls exactly on the combining marks POJ uses. Guessing
there would have been a tokenizer that is subtly wrong on Taigi and right on
everything used to test it.

`xabe-gguf` takes no dependency either, and that is worth a sentence because
the obvious alternative exists. The container was adapted from
`llmxabe/crates/xabe-gguf`, the same author's LLM engine, which has been
reading GGUF on this hardware for a while: the bounds-checked cursor, the value
model and the parse order came from there. The accessors were reshaped to
mirror `xabe-st`'s `tensor`/`tensor_f16`, so a crate above cannot tell the two
containers apart, and the nine block formats were written fresh.

Those dequantizers are transcribed from `gguf-py/gguf/quants.py` in the local
llama.cpp checkout - the same code that writes the files - and checked against
it rather than reasoned about. The reference expresses each format as a
reshape-and-shift over whole blocks; walking one element at a time means
deriving the element *ordering* by hand, and that is where every plausible
mistake lives. `tools/oracle/capture_quants.py` captures packed bytes and the
f32 the reference unpacks them to, and the test asserts exact equality on all
ten. No quantizer is written, and none is needed: this side only ever unpacks.

No ML framework. No `candle`, `tch`, `ort`, `ndarray`, or bindings to
whisper.cpp or llama.cpp. If one appears, it needs an argument in a commit body.
No `sentencepiece` or `tokenizers` crate either: all three tokenizers are
written by hand and tested against captured outputs. That decision reaches
further than it looks for the translator, whose `tokenizer.model` is a
SentencePiece `ModelProto` — a protobuf. Rather than take `prost` or `protobuf`
and a build step to read one file, `xabe-llama` carries a ~65-line wire reader
that walks the fields it needs and skips the rest by wire type. A protobuf
crate would have been correct and would have been more code, more build, and a
generated-source directory, for a format this workspace reads exactly once.

## Every dependency is permissive, and that was checked rather than assumed

This workspace is Apache-2.0, so a copyleft dependency anywhere in the tree
would be a distribution problem rather than a preference. Audited across all
166 transitive packages with `cargo metadata`:

| licence | packages |
| --- | --- |
| `MIT OR Apache-2.0` and its spellings | 98 |
| `MIT` | 39 |
| `Unicode-3.0` | 18 |
| BSD-2/3-Clause, ISC, Unlicense, BSL-1.0, LLVM-exception | 11 |

Nothing copyleft. The single package that mentions the LGPL offers it as one
arm of `MIT OR Apache-2.0 OR LGPL-2.1-or-later`, so the permissive arm is the
one taken. Worth re-running when a dependency is added, which is rare here by
policy anyway:

```sh
cargo metadata --format-version 1 --all-features |
    jq -r '.packages[] | .license' | sort | uniq -c | sort -rn
```

The reason this is a section and not a footnote: the argument a new dependency
needs in its commit body is about weight and maintenance, and it is easy to
make that argument well and still miss the licence. Both have to hold.

## Build profiles

`release` is `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `debug = 1` —
symbols kept for profiling.

`profile.dev.package."*"` is `opt-level = 2`, because the reference kernels run
the full forward pass thousands of times in the differential harness and a plain
debug build makes that unusable.
