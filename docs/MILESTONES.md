# Milestones

Numbered by what becomes *true*, not by how much work it was.

| # | State | Done |
| --- | --- | --- |
| 1 | The checkpoint can be opened, addressed and validated | ✅ |
| 2 | Model geometry and weight schema are typed and shape-checked | ✅ |
| 3 | The PyTorch oracle is captured and its format is read by tests | ✅ |
| 4 | Text → symbol ids matches the reference tokenizer exactly | ✅ |
| 5 | The text encoder matches the oracle within tolerance | ✅ |
| 6 | The stochastic duration predictor matches on fixed noise | ✅ |
| 7 | The flow, reversed, matches the oracle | ✅ |
| 8 | The HiFi-GAN decoder matches the oracle | ✅ |
| 9 | End-to-end synthesis on CPU matches the reference waveform | ✅ |
| 10 | The CLI synthesises a WAV from Tâi-lô/POJ on the command line | ✅ |
| 11 | CUDA kernels match the CPU reference, per kernel | ✅ |
| 12 | End-to-end CUDA synthesis is faster than PyTorch, measured | ✅ |

Milestone 9 is the one that matters. Everything before it is scaffolding, and
everything after it is optimisation — which cannot start until there is a
correct implementation to be faster *than*.

That set is finished. What follows widens the engine from a synthesiser to the
whole pipeline except the chat LLM.

## Folding in the rest of the pipeline

Each phase is independently useful and independently shippable, and the order
is chosen so the cheapest proof comes first: phase 3 is a 15-tensor model that
can be verified frame by frame in about a day, so if the approach is going to
fail it fails there for a tenth of the cost.

The discipline does not change: **capture the oracle, diff per stage, then
optimise** — `docs/ORACLE.md`, `docs/TESTING.md`.

### Phase 1 — consolidation and the engine skeleton

| # | State | Done |
| --- | --- | --- |
| 1 | Every model lives under one gitignored `models/`, and the pipeline still runs | ✅ |
| 2 | `xabe-engine` owns the flag surface; `xabe-audio` is split out; TTS runs as `--tts-model` | ✅ |

### Phase 2 — the serving surface

| # | State | Done |
| --- | --- | --- |
| 3 | `xabe-serve` speaks both halves: `--<stage>-url` as a client, `--serve` as a server | ✅ |
| 4 | The gateway is a behaviour-for-behaviour port of `gateway.py`, driving the Python services | ✅ |
| 5 | The turn-taking policy is server-side and tested; the browser executes it | ✅ |

Milestone 5 is narrower than it was first written, and deliberately. The plan
said turn-taking would move server-side and the browser would keep "only
capture, VAD and playback" — but the VAD *is* the turn detector, so that
sentence asked for two things at once. What moved is the **policy**: the
constants and the state machine now live in `xabe-serve::turntaking`, are unit
tested against synthetic energy traces, and reach the page through
`GET /api/config`, so tuning is a restart rather than an edit to an HTML file.
What did not move is the frame-by-frame execution, because sending every
4096-sample frame over the socket would cost more than it saves. `Endpointer`
takes one scalar per frame rather than audio, so when the engine owns a VAD of
its own (phase 3) the same state machine runs server-side over Silero's
probabilities with no change.

### Phase 3 — voice activity detection

| # | State | Done |
| --- | --- | --- |
| 6 | Silero is converted to safetensors; 15 tensors typed and shape-checked | ✅ |
| 7 | Per-frame probabilities match `whisper_vad_probs()` on a clip corpus | ✅ |
| 8 | The hysteresis segmenter matches, including whisper.cpp's own additions | ✅ |

Measured on the corpus: every segment matches on every clip, all four
hallucination triggers peak below 0.05 against a threshold of 0.6 and agree
with the reference to four decimals, and all 926 frames land on the same side
of both thresholds. Raw probabilities differ by at most 6.8e-3, which has an
explanation rather than a tolerance — see `docs/MODEL.md`.

Two items from later phases came forward because this one needed them:
`xabe-st` reads F16 and BF16 (part of item 18, since the Silero checkpoint
turned out to be F16), and `xabe-dsp` gained a strided convolution.

### Phase 4 — speech to text

| # | State | Done |
| --- | --- | --- |
| 9 | `xabe-st` reads sharded checkpoints via `model.safetensors.index.json` | |
| 10 | A real tiled GEMM in `xabe-cuda` | |
| 11 | The mel frontend matches `WhisperFeatureExtractor` | |
| 12 | Whisper geometry and weight schema, 1259 tensors, shape-checked at bind | |
| 13 | Byte-level BPE matches the reference tokenizer exactly | |
| 14 | The encoder matches a captured oracle, per layer | |
| 15 | The decoder, with KV cache and cross-attention, matches per layer | |
| 16 | Greedy decoding reproduces transcripts on held-out Taigi audio | |
| 17 | CUDA ASR is faster than whisper-server, measured and interleaved | |

### Phase 5a — the translator loader, built regardless

| # | State | Done |
| --- | --- | --- |
| 18 | `xabe-st` reads F16 and BF16, converting BF16 → F16 with a tested range check | |
| 19 | All 363 tensors of the 13 B bind and shape-check, with a parameter-count test | |
| 20 | SentencePiece matches the reference tokenizer on a captured corpus | |

### Phase 5b — translator inference, optional

`DIRECT_TAIGI=1` is the pipeline's default and takes the translator out of the
reply path entirely — measured 3.8 s → 1.6 s — so this is deferred behind a
decision to be taken once phase 5a proves the geometry is understood.

| # | State | Done |
| --- | --- | --- |
| 21 | The Llama-2 forward pass matches a captured oracle, per layer | |
| 22 | Translations match llama-server's at `temperature = 0` | |

### Phase 6 — CosyVoice

| # | State | Done |
| --- | --- | --- |
| 23 | Scoped separately: 4 sub-models across 3 formats, larger than the ASR port | |

## What the numbering does not cover

Batching and streaming synthesis are still deliberately absent. They are
answers to questions this project has not asked yet.

Serving *was* on that list, on the grounds that the pipeline upstream already
had an HTTP surface that worked. That is retracted in phase 2: the engine is
becoming the pipeline, so the gateway is not a surface being duplicated but one
being replaced. See `docs/CLI.md` for the full reasoning.
