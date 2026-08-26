# Architecture

## The shape

One process, one utterance at a time, no state carried between calls.

```
   text ──► xabe-tts ──► waveform
              │
              ├── xabe-vits    config, weight schema, shape validation
              ├── xabe-dsp     scalar reference kernels
              ├── xabe-cuda    CUDA kernels, tested against xabe-dsp
              ├── xabe-golden  reads the captured PyTorch oracle
              └── xabe-st      safetensors container, mmap, addressing
```

## Why no cache and no scheduler

`llmxabe` has both because an LLM server multiplexes long-lived sequences with a
shared prefix. A synthesiser does not: an utterance arrives whole, is one forward
pass, and shares nothing with the next. Adding a KV cache here would be
machinery for a reuse that does not exist.

Batching is a different question, and an open one. It would help if the caller
sends several clauses at once — which the pipeline upstream does, one per
sentence. That is a measurement to make after the single-utterance path is
correct, not a design decision to take now.

## Crate boundaries

Dependencies point one way, and the direction is the point: a crate must be
usable, testable and wrong-input-rejecting without knowing anything about the
crates above it.

| Crate | Owns | Refuses |
| --- | --- | --- |
| `xabe-st` | byte addressing inside a safetensors file | any idea what a tensor means |
| `xabe-vits` | model geometry, tensor names, shape contracts | doing arithmetic |
| `xabe-dsp` | scalar f32 reference kernels | being fast |
| `xabe-cuda` | CUDA kernels and the device handle | knowing what a VITS is |
| `xabe-golden` | reading captures, comparing tensors | producing them |
| `xabe-tts` | the forward pass, the public API, the CLI | container details |

`xabe-cuda` takes flat slices and dimensions, exactly as `xabe-dsp` does, and
knows nothing about the model. That is what lets its tests be a plain
kernel-against-kernel diff rather than a model test in disguise.

If `xabe-dsp` needs to know a file offset, the abstraction is wrong. Fix the
boundary; do not add the edge.

## `xabe-dsp` is deliberately slow

Its kernels are scalar, obvious, and written to be read against the reference
implementation line by line. They are the oracle that CUDA kernels are tested
against, so they optimise for being *evidently correct*, not for throughput. A
clever reference is a reference you cannot trust.

## Reading order

`docs/MODEL.md` for what is being computed, this file for where it lives, then
`docs/TESTING.md` before writing anything numeric.
