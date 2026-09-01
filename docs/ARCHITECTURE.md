# Architecture

## The shape

One binary, `xabe-engine`, for every stage of the Taigi voice pipeline.

```
   speech ──► VAD ──► ASR ──► [LLM] ──► [translator] ──► TTS ──► speech
                                 ▲
                                 └── always another process, over HTTP
```

Which of those stages *this* process runs is decided entirely by flags, and
each one is satisfied two ways:

```
   --<stage>-model PATH   run it here
   --<stage>-url   URL    delegate it to another process
```

Nothing downstream of `stage.rs` can tell which was used. That symmetry is what
makes one binary serve as a monolith, as a single-stage worker, or as anything
between - the six-process topology the Python pipeline runs today is one
configuration of the same flags, not a different program.

```
   xabe-engine
      │
      ├── xabe-serve    HTTP, WebSocket, the page, turn-taking policy
      ├── xabe-vad      Silero geometry, weights and forward pass
      ├── xabe-asr      the Whisper forward pass, CUDA only
      ├── xabe-whisper  Whisper geometry, weight schema, BPE, mel
      ├── xabe-translate the Llama-2 forward pass, CUDA only, either container
      ├── xabe-chat     the Llama-3 forward pass, CUDA only, grouped-query
      ├── xabe-llama    Llama geometry from either container, two tokenizers
      ├── xabe-tts      the VITS forward pass and synthesis API
      ├── xabe-cosy     CosyVoice3: speech LM, flow, vocoder, Qwen2 BPE, voices
      ├── xabe-audio    WAV, mel spectrogram, PCM framing
      ├── xabe-vits     config, weight schema, shape validation
      ├── xabe-dsp      scalar reference kernels
      ├── xabe-cuda     CUDA kernels, tested against xabe-dsp
      ├── xabe-golden   reads the captured oracle
      ├── xabe-st       safetensors container, mmap, addressing
      └── xabe-gguf     GGUF container, mmap, metadata, addressing
```

Every crate in that tree exists and every stage flag reaches a built stage.
That was not always true - the flag surface was deliberately finished ahead of
the stages behind it, so the topology could be settled first, and
`--translator-model` used to parse, validate and then fail naming the phase it
was waiting on. Nothing does that now; the only refusal left is `--vad-url`,
which is a permanent limit rather than a pending one.

**Two containers, one contract.** `xabe-st` reads safetensors and `xabe-gguf`
reads GGUF, and they are siblings rather than one wrapping the other: same
accessors, same errors, no shared trait. A model crate above them takes either.
The translator does exactly that - `Translator::open` accepts a 🤗 directory or
a `.gguf` file - and the awkward part is not the dispatch but that a GGUF Llama
stores `attn_q` and `attn_k` row-permuted. That is undone at load in
`xabe-llama`, so one rope kernel serves both and nothing downstream learns
which container it came from.

The ASR is split the way the TTS is: `xabe-whisper` says what the tensors are
and refuses to do arithmetic, `xabe-asr` runs them. The difference is that
`xabe-asr` has no CPU twin - see below. The translator repeats the split
exactly, `xabe-llama` to `xabe-translate`, and for the same reason.

## Why the ASR and the translator have no CPU path

Every other model in this engine runs both ways: a scalar version in
`xabe-dsp` that is written to be read against the reference, and a CUDA version
checked against it. The ASR does not, and the reason is arithmetic rather than
taste. One 30-second window is about 2.2 TFLOP through Whisper's encoder alone,
and the scalar kernels run at something under 2 GFLOP/s - twenty minutes an
utterance. That is not a slow option; it is a fictional one, and shipping it
would be shipping something nobody can use.

The translator is worse, and for a second reason on top of the first. Its
weights are 26.5 GB at f16 and a scalar path would have to hold them as f32,
which is 53 GB — more than this machine's cards have and more than most hosts
would want resident. So both stages carry `Kind::has_cpu() == false` and both
`--asr-device cpu` and `--translator-device cpu` are refused at preflight by
name, the mirror of the rule that refuses `--vad-device 0`. The kernels these
stages are built from still have their scalar twins and their differential
tests; it is only the assembly of them that is checked against the captured
oracle directly instead.

## How a model reaches the serving layer

`xabe-serve` owns HTTP and refuses to know what a model is. `xabe-tts` owns the
model and refuses to know what a socket is. They meet in `xabe-engine`, and the
join is a **channel**: a synthesiser thread reads `SynthesisJob`s and writes WAV
chunks back, and neither side learns anything about the other. A trait would
have worked and would have been the obvious move; the channel is narrower, and
the house style asks for no traits unless forced.

That thread is an OS thread, not an executor task. A forward pass is a blocking
GPU-bound 48 ms that would otherwise stall every socket the runtime is polling.
There is exactly one, because the model is one utterance at a time by design and
a second thread would only queue on the same device.

## Why the CLI left `xabe-tts`

`xabe-tts` used to own `main.rs`, because the synthesiser was the whole program.
It is now one stage of five, so the binary moved up into `xabe-engine` and the
crate went back to being a library. `wav.rs` moved the other way, down into
`xabe-audio`: the TTS writes audio, the ASR and VAD read it, and a WAV writer
living inside the synthesiser cannot be reached by the others without pointing
a dependency edge the wrong way.

## Why no cache and no scheduler

`llmxabe` has both because an LLM server multiplexes long-lived sequences with a
shared prefix. A synthesiser does not: an utterance arrives whole, is one forward
pass, and shares nothing with the next. Adding a KV cache here would be
machinery for a reuse that does not exist.

Batching is a different question, and an open one. It would help if the caller
sends several clauses at once — which the pipeline upstream does, one per
sentence. That is a measurement to make after the single-utterance path is
correct, not a design decision to take now.

The ASR changes this only slightly. An utterance is still whole and still shares
nothing with the next; what it adds is a decoder KV cache *within* one
utterance, which is a buffer, not a scheduler.

The two Llama stages add the same thing again and no more. Each holds a KV
cache across the tokens of **one** reply - f16, doubling its capacity from 256
positions so a long reply pays for the growth a logarithmic number of times -
and drops it at the end. A new turn re-prefills from `--history-turns` rather
than reusing the last one's prefix, which is exactly the reuse a server would
schedule around and this does not. So the heading still holds: three stages
have a cache, none of them has a scheduler, and nothing here shares a prefix
between calls.

## Crate boundaries

Dependencies point one way, and the direction is the point: a crate must be
usable, testable and wrong-input-rejecting without knowing anything about the
crates above it.

| Crate | Owns | Refuses |
| --- | --- | --- |
| `xabe-st` | byte addressing inside a safetensors file | any idea what a tensor means |
| `xabe-gguf` | byte addressing inside a GGUF file | any idea what a tensor means |
| `xabe-vits` | model geometry, tensor names, shape contracts | doing arithmetic |
| `xabe-dsp` | scalar f32 reference kernels | being fast |
| `xabe-cuda` | CUDA kernels and the device handle | knowing what a VITS is |
| `xabe-golden` | reading captures, comparing tensors | producing them |
| `xabe-audio` | WAV containers, sample handling | knowing which model consumes it |
| `xabe-serve` | HTTP, WebSocket, the page, the conversation | model internals |
| `xabe-vad` | Silero geometry, weights and forward pass | audio capture |
| `xabe-whisper` | Whisper geometry, weight schema, BPE, the mel frontend | doing model arithmetic |
| `xabe-asr` | the Whisper forward pass and greedy decoding | running anywhere but a card |
| `xabe-llama` | Llama geometry from either container, weight schema, SentencePiece and byte-level BPE | doing model arithmetic |
| `xabe-translate` | the Llama-2 forward pass and the `[TRANS]` template | running anywhere but a card |
| `xabe-chat` | the Llama-3 forward pass, the sampler and streaming completion | the prompt format, which is `xabe-serve`'s |
| `xabe-tts` | the VITS forward pass and its API | serving, or any other stage |
| `xabe-engine` | flags, stage wiring, orchestration | container and kernel details |

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
