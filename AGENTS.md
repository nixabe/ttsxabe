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

**The scope has since widened, by decision.** The engine is every stage of the
pipeline except the chat LLM, which stays in llama.cpp: ASR (Whisper large-v2
fine-tune), voice activity detection (Silero), Mandarin-to-Taigi translation
(Llama-2 13 B), turn-taking and the web front end, all in one binary with
per-stage flags. `docs/MILESTONES.md` has the phases.

The translator was planned as a loader only, because `DIRECT_TAIGI=1` takes it
out of the reply path. The loader proved the geometry, the forward pass was
built on the kernels that were already there, and it matches both references —
so the plan's "optional" is spent rather than pending.

An earlier version of this file said porting Whisper or the LLM here was
explicitly out of scope. That was the right rule while the synthesiser was
unfinished; it is retracted now that it is finished and measured. The chat LLM
remains out of scope for *inference* — llama.cpp is not a rewrite that buys
anything — but its **weights are now readable here**, which is a smaller claim
and a real one: `xabe-gguf` reads the GGUF container - nine block formats
included, checked against `gguf-py` at exact equality - and `xabe-llama` binds
all 292 tensors of the 8 B Breeze2 against its own metadata. Nothing runs them.

This file used to end that paragraph by saying unpacking a quantized file is
not running quantized - the weights landed at full width, so it bought disk and
load bandwidth and not memory - and that packed-in-VRAM inference would mean
teaching every matmul the block layouts, a kernel project rather than a loader
change. **That is now done, and the description of what it would take was
accurate.** `Operand::Q` hands the matmul the checkpoint's own blocks and
unpacks them inside the kernel, so a quantized model occupies its file's size
on the card rather than its unpacked size. `docs/KERNELS.md` has the design and
the two findings that cost time; `docs/BENCHMARKS.md` has what it measures.

That paragraph used to end by saying quantizing at *runtime* was not here at
all. **It is now**, and it is no longer in one place. It began in one: the
packed mat-vec reads sixteen bytes of weight a lane, that is 32 elements, and 32
f32 activations cost more to fetch than the wide load saves. So an activation is
quantized to int8 in groups of 32 on its way into that kernel -
`Gpu::quantize_activation`, with a CPU twin in `xabe-dsp` and a differential
test at exact equality.

**The tiled matmul now reads the same codes.** `gemm_i8` multiplies on the
integer tensor cores, which is the only way past the f16 kernel: that measured
86% of what was then believed to be its ceiling while llama.cpp was a third
ahead. That baseline has since been measured and was wrong - `docs/KERNELS.md`
has the correction - so there was more room in it than recorded, though the
integer path is still twice the arithmetic rate and still the right call. The engine therefore has two deliberate approximations
rather than one, and they are the same approximation in two kernels. Together
they cost 0.69% of the chat model's logit span and 0.42% of the translator's,
and they moved the agreement with llama-server by nothing at all - 1 of 125
teacher-forced decisions before and after. Measured against the same weights
computed at f16, this engine's integer matmul sits closer to exact than
llama.cpp's own does. `docs/BENCHMARKS.md` has every one of those numbers and
`docs/KERNELS.md` has the design.

The limits that remain are narrower. Only the **matmul** reads packed blocks -
the embedding table is still gathered at f32, because a gather is not a matmul
and it has its own kernel. And a *weight* still may not arrive packed as the
left operand of a matmul; that refusal stands.

## Current standing

Every stage is finished: the synthesiser, the serving layer, voice activity
detection, speech recognition, Mandarin-to-Taigi translation, the chat model,
CosyVoice3, and a third synthesiser in Tacotron2 + WaveGlow. This paragraph
said "CosyVoice is scoped and not started" long after phase 6 had closed; the
one thing still outside the engine is deriving a **new** CosyVoice voice, which
runs two ONNX models once through `tools/make_cosyvoice_voice.py`.
`docs/MILESTONES.md` has the state of every item and is the file to read before
starting work rather than this paragraph.

Three standings are worth knowing here because they are easy to assume wrongly.
The synthesiser is 1.24x faster than the PyTorch reference on interleaved
medians. The ASR is **0.94x to 1.00x** against `whisper-server` from three to
seven seconds of speech and **1.08x** at ten — level from about five seconds up
and short only on the briefest clip, and still recorded as a miss because the
milestone asks for the short end too. The two engines have opposite cost
structures: the encoder is a fixed 30-second window for both and ours is about
19 ms slower at it, so every transcription starts that far behind, while the
decode is about 1.8 ms a token cheaper here and pays it off at roughly seven
tokens.

That comparison used to be the one in the repository whose two halves were not
measured in the same sitting; it is not any more. `whisper.cpp` is built here
with CUDA against the same checkpoint, converted by its own script, both sides
are strictly single-pass greedy, and the two are alternated in pairs in one
run. What is left of the gap is the **encoder** and nothing else: 102 ms
against `whisper.cpp`'s 83, and about 73 of those 102 ms are a tiled `gemm`
running at 22.4 TFLOP/s.

Do not read that as "the encoder is the gemm". It was read that way for three
rounds and it was wrong by half: `whisper.cpp` does the encoder's 2256 GFLOP in
83.3 ms, which is 27.1 TFLOP/s *across its whole encoder* and barely above what
this engine's matmul reaches on its own — so cuBLAS was never running away with
it, and half the gap was the kernels either side. Timing every one of them
found a transpose written as a scatter at 141 GB/s and four projections a layer
reading f32 into a matmul that stages f16 regardless. `docs/BENCHMARKS.md` has
that round; the point to carry is that the profile was worth taking and the
assumption was not. **What that is short of is not the arithmetic.** The instruction is
measured at 102.3 TFLOP/s on this card and f32 accumulation costs 0.7% of it,
so the f16-accumulation trade this file used to name as the only way past buys
essentially nothing here. It is not the memory system either, since rounding the
activations to f16 was worth 5%. **It is the register file**, and that is
measured rather than inferred: `ptxas -v` puts the kernel at exactly 128
registers a thread with no spill, which is exactly the budget for the two
resident blocks that are this architecture's only latency hiding, and 64 of the
128 are accumulators a 128x128 tile over 256 threads cannot give up.
Software-pipelining the staging — the standard fix, and the one thing left to
try — needs 184 registers and was measured at half the throughput; five further
tile-and-occupancy arrangements of it were also measured and all lost. So the
remaining gap is an architecture this kernel shape has run out of room on, not
a missing trick: a deep pipeline here wants `cp.async`, which arrived with
sm_80. `docs/KERNELS.md` has the table and `ncu` was never needed.

And the Llama stages are **level with or ahead of llama.cpp on every measured
row**: the chat model ahead on prefill at both
prompt lengths (2447 against 2259 at 128 tokens, 2928 against 2513 at 512) and
on decode (100.9 against 95.3), the translator ahead on decode (61.4 against
60.0), level on 512-token prefill (1636 against 1647), and at 0.94x on
128-token prefill against a llama.cpp median that swings 20% between its own
runs — recorded as inside its noise, not as a win.

The translator's *latency* is a separate question from its throughput and has
its own section in `docs/BENCHMARKS.md`. What is worth knowing here: a
translation is 99.1% `forward_last` — the logits download, the repeat penalty,
the CPU argmax and the stop-string check are 0.8% between them, so the decode
loop is not where to look. The step itself streams 8.0 GB a token at 579 GB/s
against 602 for the same weights at f16, so unpacking costs 4% and the weight
stream is close to done. What is left is a seventh of the step in fifteen small
kernels a layer that cost what a launch costs, and a prefill that used to
compute 128 rows for a twenty-four-token prompt.

Prefill was 0.29x once,
then 0.75x, and has now been worked on four times. Three cautions: the
llama.cpp column does not hold still across sittings, so the only comparison
trusted is both tools alternated in one sitting; every engine figure is a
nine-round median, because five-round runs read about 3% high off the boost
clock; and the decode lead appeared because llama.cpp's own decode came in
lower that sitting — the engine's decode did not move, which was the
constraint the prefill work was under.
Both stages hold their KV cache at **f16**, which is worth 4.0 GiB on the
translator and 512 MiB on the chat model, and 6.6% of the translator's decode
at a 1024-token context but nothing at all of the chat model's - the reason is
grouped-query attention and it is in `docs/BENCHMARKS.md`.

A fourth caution has since been added: every decode figure in that table is
taken at a 128-token prompt, and decode is **context-sensitive** - it costs
0.42 ms more per 1024 tokens of context, because the KV cache is re-read in
full for every token. A conversation carries a system prompt and a history, so
the row a listener actually waits on is the 1024- or 2048-token one.
Those numbers and every other comparison belong in `docs/BENCHMARKS.md` and
nowhere else.

One correctness note that is easy to get backwards. The chat model was recorded
as disagreeing with llama-server at 10 of 105 teacher-forced decisions. **The
capture was the problem, not the engine**: it had been taken from llama-server
running the *quantized* checkpoint, whose matmul multiplies packed weights
against an int8 activation and is coarser than this engine's. Against the same
server running the f16 build, this engine reading the *quantized* file agrees at
1 of 125 decisions and 7 of 8 replies. Capture the chat oracle from the f16
GGUF; `tools/oracle/capture_chat_server.py` says so and `docs/TESTING.md` has
the numbers and the three arithmetic changes that were made before the reference
was suspected.

Do not write comments, commit messages, or documentation asserting a speedup
that has not been measured on this hardware.

One stage reads **converted** weights rather than the published checkpoint, and
it is the only one: WaveGlow ships as a pickled `nn.Module` object graph in the
pre-1.6 torch format, which cannot be parsed without PyTorch and the model's own
class definitions. `tools/convert_tacotron2.py` does that once, offline. The
claim at the top of this file holds everywhere except `xabe-taco`, and that
exception is a property of how NVIDIA saved the file in 2019.

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
| `xabe-st` | safetensors container parsing, mmap, tensor addressing, sharding | — |
| `xabe-gguf` | GGUF container parsing, mmap, metadata, block-format unpacking | — |
| `xabe-dsp` | CPU reference kernels + differential compare harness | — |
| `xabe-golden` | reading captures and comparing tensors | — |
| `xabe-audio` | WAV containers, sample handling, framing, mel | `xabe-dsp` |
| `xabe-cuda` | CUDA kernels and the device handle | `xabe-dsp` |
| `xabe-vits` | VITS config, weight schema, shape validation | `xabe-st`, `xabe-golden` |
| `xabe-whisper` | Whisper geometry, weight schema, BPE, the mel frontend | `xabe-st`, `xabe-dsp`, `xabe-audio` |
| `xabe-llama` | Llama geometry, weight schema, SentencePiece | `xabe-st`, `xabe-gguf` |
| `xabe-vad` | Silero geometry, weights and forward pass | `xabe-st`, `xabe-dsp`, `xabe-audio` |
| `xabe-tts` | the VITS forward pass and its API | `xabe-vits`, `xabe-cuda`, `xabe-dsp`, `xabe-st`, `xabe-golden` |
| `xabe-asr` | the Whisper forward pass and greedy decoding | `xabe-whisper`, `xabe-cuda`, `xabe-dsp`, `xabe-st`, `xabe-audio` |
| `xabe-translate` | the Llama-2 forward pass and the `[TRANS]` template | `xabe-llama`, `xabe-cuda`, `xabe-st` |
| `xabe-chat` | the chat model's forward pass, sampling, stop strings | `xabe-llama`, `xabe-cuda`, `xabe-gguf` |
| `xabe-cosy` | CosyVoice3: geometry and forward pass | `xabe-cuda`, `xabe-dsp`, `xabe-st` |
| `xabe-taco` | Tacotron2 + WaveGlow: geometry, forward pass, POJ to Tâi-lô | `xabe-cuda`, `xabe-st` |
| `xabe-serve` | HTTP, WebSocket, the page, the conversation | `xabe-audio` |
| `xabe-engine` | flags, stage wiring, orchestration, the binary | every stage crate |

The pattern to keep: each model is **two** crates, one that says what the
tensors are and refuses to do arithmetic, and one that runs them. `xabe-vits` to
`xabe-tts`, `xabe-whisper` to `xabe-asr`, `xabe-llama` to both `xabe-translate`
and `xabe-chat` - one geometry crate serving two forward passes, which is the
pattern working rather than an exception to it. A geometry crate that grows a
matmul has broken the rule.

`xabe-cosy` and `xabe-taco` are one crate each, and that is a deviation rather
than a second pattern. Neither model's geometry is read by anything but its own
forward pass - there is no third consumer the split would serve - so the
boundary is drawn between modules instead: `config` and `weights` know the
shapes and never touch an activation, the rest does arithmetic and never parses
a file. Split them if a second consumer ever appears.

`xabe-asr` and `xabe-translate` are CUDA-only and have no scalar twin of the
whole model - see `docs/ARCHITECTURE.md` for why, which is arithmetic rather
than taste. Their individual kernels still have twins and differential tests.

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
