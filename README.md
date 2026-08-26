# ttsxabe

A from-scratch Rust implementation of VITS for Taiwanese Hokkien speech
synthesis, reading `facebook/mms-tts-nan` weights directly.

No ML framework, no bindings — the container reader, the weight schema, and the
kernels are all in this repository, verified against the PyTorch reference.

Status: **container reading works.** Nothing synthesises yet.

```sh
cargo test --workspace --release
```

See [AGENTS.md](AGENTS.md) for the rules this codebase is built under.
