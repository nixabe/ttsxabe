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
| 9 | `xabe-st` reads sharded checkpoints via `model.safetensors.index.json` | ✅ |
| 10 | A real tiled GEMM in `xabe-cuda` | ✅ |
| 11 | The mel frontend matches `WhisperFeatureExtractor` | ✅ |
| 12 | Whisper geometry and weight schema, 1259 tensors, shape-checked at bind | ✅ |
| 13 | Byte-level BPE matches the reference tokenizer exactly | ✅ |
| 14 | The encoder matches a captured oracle, per layer | ✅ |
| 15 | The decoder, with KV cache and cross-attention, matches per layer | ✅ |
| 16 | Greedy decoding reproduces transcripts on held-out Taigi audio | ✅ |
| 17 | CUDA ASR is faster than whisper-server, measured and interleaved | ❌ |

**Item 17 is not met.** Measured, alternated in pairs against a
`whisper-server` started without `--vad` so both do the same job: 264 ms
against 144 on a 2.67 s clip, 296 against 177 on a 3.9 s one - 0.55x to 0.60x.
Both produce identical transcripts. `docs/BENCHMARKS.md` computes what closing
the gap would take: flash attention for the 34 ms the encoder spends moving an
attention score matrix it never needed to materialise, and an `ldmatrix`
double-buffered matmul for the 79 ms of projections currently running at 24% of
this card's peak. Together those put the encoder near 100 ms and the whole
transcription near 190 - still short of 144, and what the third lever would be
has not been established. Recorded as a miss rather than restated as a
different target.

The filter bank is computed rather than shipped, and matches the capture *bit
for bit*: both sides evaluate the same closed form in f64 and round once, with
no reduction for an ordering to disagree about. That removes a runtime asset
that could go missing, go stale, or silently be the htk variant.

Three findings from item 13, none of which a shape check or a round trip would
have caught: `<|endoftext|>` is special but lives in `vocab.json` rather than
`added_tokens.json`; 50362 is spelled `<|nocaptions|>` here, and asking the
reference for `<|nospeech|>` returns the *unknown* id, which is also
end-of-text; and `decode_with_timestamps` is broken in transformers 5.15.1, so
it is asserted directly rather than captured.

### Phase 5a — the translator loader, built regardless

| # | State | Done |
| --- | --- | --- |
| 18 | `xabe-st` reads F16 and BF16, converting BF16 → F16 with a tested range check | ✅ |
| 19 | All 363 tensors of the 13 B bind and shape-check, with a parameter-count test | ✅ |
| 20 | SentencePiece matches the reference tokenizer on a captured corpus | ✅ |

Item 18 was half done before this phase started: phase 3 needed F16 because
the Silero checkpoint turned out to be one. What this phase added is
`tensor_f16`, which takes F32, F16 or BF16 and always returns f16 — F16 copied
bit for bit, the other two rounded to nearest even. The range check is not a
clamp: an input that would round to an infinity is **refused** by name and
index, because the failure mode being defended against is a model that loads
and then speaks nonsense. Underflow to a subnormal or to zero is counted and
logged instead of refused, since that is what the arithmetic does anyway.

Nothing was read to bind the 363 tensors. `LlamaWeights` holds a `Bound { name,
shape, dtype }` per tensor and checks each shape against the config before any
bytes move, so a geometry mistake fails at bind rather than as a wrong number
40 layers later. The parameter count comes out at 13,261,870,080 against the
checkpoint's own inventory.

Two findings from item 20. The tokenizer is a hand-written protobuf reader —
`tokenizer.model` is a SentencePiece `ModelProto` and the workspace takes no
protobuf dependency — and it reads only the two fields it needs, skipping the
rest by wire type. And `config.json` says `vocab_size` 56024 while the
tokenizer holds 56020 pieces: the four extra rows are real, allocated, trained
embedding rows that no piece maps to. The loader binds the 56024 and the
tokenizer refuses to emit an id above 56019, which is the only combination
that is both faithful to the checkpoint and safe to sample from.

### Phase 5b — translator inference, optional

`DIRECT_TAIGI=1` is the pipeline's default and takes the translator out of the
reply path entirely — measured 3.8 s → 1.6 s. It was built anyway, because
phase 5a proved the geometry and the remaining work was the forward pass this
workspace's kernels already covered.

| # | State | Done |
| --- | --- | --- |
| 21 | The Llama-2 forward pass matches a captured oracle, per layer | ✅ |
| 22 | Translations match llama-server's at `temperature = 0` | ✅ |

Item 22 is met with one disagreement, and the disagreement was chased rather
than tolerated. Seven of eight prompts are character-identical to
llama-server; the eighth, `你食飽未 [HAN]`, differs by a trailing `？`. The
🤗 oracle was then captured for that exact prompt at f32 and answers `你食飽未`
— agreeing with this engine, so llama-server is the one that diverges.
`docs/ORACLE.md` records why that is possible at all: llama-server reuses a KV
prefix across requests, so it is not request-independent even at temperature 0.

Getting to 7/8 needed llama.cpp's `penalize_nl = false`, which is a default and
not a flag anyone sets: the newline is exempt from the repetition penalty.
Without that exemption four of the eight prompts grew a trailing `。`. Worst
per-layer error across the corpus is 7.9e-4 in the final norm against a 3.0e-3
gate.

### Phase 6 — CosyVoice

| # | State | Done |
| --- | --- | --- |
| 23 | Scoped, against the checkpoint on disk rather than the plan's estimate | ✅ |
| 24 | `tools/convert_cosyvoice.py` converts all three `.pt` files; every tensor typed and shape-checked at bind | ✅ |
| 25 | The speech LM's forward pass, GQA 7:1 and q/k/v biases, matches at **143 of 143** positions | ✅ |
| 26 | The Qwen2 BPE matches the reference on both strings; `ras_sampling` transcribed with its traps named | ✅ |
| 27 | The DiT estimator and the Euler/CFG solver reach the reference mel at correlation **0.999970** | ✅ |
| 28 | Snake, iSTFT and the NSF source in `xabe-dsp`/`xabe-cuda`; the waveform matches at **1.000000** | ✅ |
| 29 | End to end in-process: `--tts-engine cosyvoice=<dir>`, Han text in, 24 kHz audio out | ✅ |
| 30 | The speech tokenizer and CAMPPlus ported from ONNX, so a new voice needs no Python | ⬜ |

The plan said "4 sub-models across 3 formats, larger than the ASR port". Only
the middle claim survives reading the files. It is **five** sub-models, and it
is *not* larger than the ASR port in parameters — 1.24 B against Whisper's
1.54 B. It is larger in every other way that matters, which is why the estimate
was right for the wrong reason and worth correcting anyway.

| sub-model | file | format | params | on the per-request path |
| --- | --- | --- | --- | --- |
| text-to-speech-token LM | `llm.pt` | pickle | 642 M | yes |
| flow matching, DiT estimator | `flow.pt` | pickle | 332 M | yes |
| HiFT vocoder | `hift.pt` | pickle | 20.8 M | yes |
| speech tokenizer | `speech_tokenizer_v3.onnx` | ONNX | 242 M | **no** |
| speaker embedding, CAMPPlus | `campplus.onnx` | ONNX | 6.9 M | **no** |

All three `.pt` files are float32 throughout. `flow.decoder.estimator.fp32.onnx`
is a redundant export of `flow.pt`'s estimator and is not a sixth model.
`CosyVoice-BlankEN/model.safetensors` is not a sixth model either: `llm.pt`
carries its own fine-tuned copy of every one of those weights, so that directory
is needed for the **tokenizer alone** — `vocab.json` and `merges.txt`, a Qwen2
byte-level BPE over 151,936 pieces.

#### The two ONNX models are not on the critical path, and that is the lever

The live daemon calls `add_zero_shot_spk` **once at startup** and caches the
result, because both ONNX models fall back to CPU here — onnxruntime's CUDA
provider wants `libcudnn.so.8`, which this host does not have. So per request
they do not run at all. What they produce is two tensors for a fixed reference
clip: a speech-token sequence and a 192-dimensional speaker embedding.

That means the ONNX half can be **captured rather than ported** for a first
working version — the same reference wav, the same two tensors, written once to
a file — and porting them becomes a later item about supporting a *new* voice
rather than a prerequisite for any audio at all. It also inverts the usual
argument: a Rust port of these two would be faster than what runs today, since
what runs today is CPU onnxruntime, but it would be faster at something that
happens once.

#### What each remaining piece actually is

**The LM** is a Qwen2-0.5B fine-tune: 24 layers, hidden 896, intermediate 4864,
14 query heads against **2 key/value heads**, `rope_theta` 1e6, RMS-norm eps
1e-6, and biases on q/k/v. `xabe-llama` currently *refuses* GQA by name, so this
is the item that makes that refusal into an implementation. On top of the
backbone sit `speech_embedding` and `llm_decoder`, both 6761x896 — 6561 speech
tokens plus 200 task and control tokens.

Sampling is **not greedy**. It is `ras_sampling` — repetition-aware, top-p 0.8,
top-k 25, window 10, `tau_r` 0.1 — which makes this the first stochastic decode
in the workspace, and the first place an oracle comparison needs a pinned RNG on
both sides rather than a fixed answer.

**The flow** is a conditional flow-matching decoder around a DiT: dim 1024,
depth 22, 16 heads of 64, feed-forward 2048, AdaLN-Zero (the 6144x1024
modulation projection), rotary embeddings, and a grouped convolutional
positional embedding. It runs an Euler solver on a cosine schedule with
classifier-free guidance at 0.7, which is why the ONNX export takes a batch of
**2** — conditional and unconditional in one pass. Every solver step is a full
22-layer forward, so the step count multiplies the cost of the largest thing
here.

**The vocoder** is a causal HiFi-GAN with three novelties this workspace has not
built. Its activations are **Snake**, not leaky ReLU — `x + sin²(αx)/α` with a
learned per-channel α, 72 of them. Its output head is an **iSTFT**: `conv_post`
emits 18 channels, which is magnitude and phase over the 9 bins of a 16-point
transform, so `xabe_dsp::Fft` needs an inverse it does not have. And it is
driven by an **NSF harmonic source** — F0 predicted by a small conv net, then 8
harmonics — which is a second signal path merged into the upsampler, not a
decoration.

164 of its 328 tensors are `parametrizations.weight.original0/1` pairs: **weight
norm**, unfused, exactly the trap that `xabe-vits`'s parameter-count test caught
once already. It is the one thing in this phase this project has been bitten by
before.

#### The container problem, and the precedent for it

`.pt` is a pickle in a ZIP. `xabe-st` reads safetensors and nothing else, and
teaching a Rust workspace to execute pickle opcodes to read three weight files
would be a genuinely bad trade. The precedent is phase 3: Silero was
legacy-ggml and was converted by a `tools/` script into safetensors, once,
outside the engine. Three more conversions is the same move, and it is also
where the dtype is decided — per sub-model, not once. The LM and the DiT want
f16 for the same reason everything else on this card does. The vocoder probably
does not: `docs/BENCHMARKS.md` already records fp16 being rejected upstream in
VITS's flow and decoder, and a HiFi-GAN with an iSTFT head is the same kind of
arithmetic. 20.8 M parameters is not worth the risk of finding out by ear.

#### What 24-29 came to, and the four things that decided them

Items 24-29 are done and 30 is not. Item 30 is a different kind of work —
deriving a network from an ONNX graph rather than from reference source — and
should be re-scoped when it is reached rather than estimated now.

Two of the guesses above were wrong, and are corrected here rather than quietly
left standing:

- **`xabe-llama` did not grow GQA.** The speech LM lives in `xabe-cosy` with its
  own forward pass. It shares no weights, no tokenizer and no rope convention
  with the translator, and folding a 6761-way speech head into a crate that
  exists to read Llama geometry would have bought a shared file and nothing
  else.
- **`ras_sampling` is not reproduced under a pinned RNG,** because it cannot be:
  upstream draws with `torch.multinomial`. Measured on the capture, its sampled
  run agrees with its own greedy argmax at **21 of 143** positions — so two
  correct implementations produce visibly different tokens, and comparing them
  would prove nothing in either direction. The LM is compared by **forced
  log-probabilities** instead: 143 observations rather than one, and it keeps
  measuring past the first divergence.

Four findings cost real time, and each is invisible in a shape check.

| | |
| --- | --- |
| The vocoder's final `F.leaky_relu` | takes torch's **0.01** default where every other slope in this checkpoint is 0.1. Every stage stayed exact, `conv_post` moved to 0.962, and the waveform came out with **ten times** its energy once the magnitudes were exponentiated. |
| `copy_into` has no source offset | so both prompt markers were written as `sos`. 113 of 143 positions still agreed, which reads exactly like a rounding problem. `copy_range` first, then copy. |
| A tap saved as a transposed **view** | `np.save` records it faithfully as `fortran_order: True`, so the header's shape and the bytes disagree and a reader that trusts the shape gets a permutation: identical rms, correlation 0.26. The capture now forces C order on every tap. |
| The excitation's phase is a cumulative sum | so a 1e-4 relative difference in the predicted F0 has grown into a fraction of a cycle by the end of six seconds. The waveform is then the same speech with its carrier slid along — and correlates at **zero** sample for sample. |

#### What is measured, stage by stage

Every figure is against `.golden/cosyvoice`, captured from CosyVoice3-0.5B
itself on the same reference clip.

| stage | measurement | result |
| --- | --- | --- |
| tokenizer | ids, instruct and utterance | identical |
| speech LM | argmax agreement on forced log-probabilities | 143 / 143 |
| flow | the whole thing, against the reference mel | correlation 0.999970 |
| F0 | per frame | correlation 1.000000, worst 0.0016 Hz |
| excitation | given the reference's own dither | correlation 0.999999 |
| vocoder | waveform, given the reference excitation | correlation 1.000000, worst sample 1e-5 |
| the acoustic half chained | energy envelope, its own dither and its own F0 | correlation 0.968, gain 1.008 |

#### Two things are deliberately not bit-exact, and could not be

- **The vocoder's dither.** `SineGen2` and `SourceModuleHnNSF` each call
  `torch.rand` in `__init__` and keep the result as a plain attribute — not a
  parameter, not a registered buffer — so none of it reaches `hift.pt`.
  Upstream redraws it on every construction and **does not reproduce across
  load orderings either**. The engine draws its own from a named seed and is
  reproducible on its own terms; on the reference mel that alone still leaves
  the waveform at correlation 0.996.
- **The estimator's tail.** The residual stream through twenty-two blocks
  carries a handful of activations near 280 against an rms of 8.7, and float32
  error concentrates on exactly those: worst element 0.35 against an rms of
  6.14, at correlation 0.999951.

#### The speaker is a file, not an ONNX runtime

`tools/make_cosyvoice_voice.py` runs both ONNX models **once per voice** and
writes four tensors plus the diffusion's starting noise. That last one is not a
property of the speaker at all — `CausalConditionalCFM.__init__` seeds the
global RNG to zero and draws it, so it is the same for every voice and every
utterance — and it is load-bearing, so it rides in the bundle because that is
the file the engine already opens. See `crates/xabe-cosy/src/voice.rs`.

## Outside the numbering: the chat model's loader

Not a plan phase. The plan said the chat LLM stays in llama.cpp and there is no
`--llm-model`; that is retracted **by half**, deliberately and with the halves
kept apart.

| | State | Done |
| --- | --- | --- |
| — | `xabe-gguf` reads the GGUF container: v3, three widths and nine block formats | ✅ |
| — | All 292 tensors of `Llama-Breeze2-8B-Instruct-text-only.f16.gguf` bind | ✅ |
| — | `rope_scaled` and `repeat_kv` kernels, for Llama-3's rope and its grouped-query heads | ✅ |
| — | The byte-level BPE, matching llama.cpp id-for-id on 60 captured cases | ✅ |
| — | The grouped-query forward pass, sampler and streaming completion | ✅ |
| — | `--llm-model`, wired through the engine like every other stage | ✅ |

So the retraction is complete: `--llm-url` still delegates to llama-server, and
`--llm-model` now runs the weights here. What changed the decision was not the
forward pass getting easier — it was that delegating the last stage kept a
second runtime, a second copy of the weights and a second GPU allocation alive
for it.

### What the measurement actually says

124 of 125 token decisions identical to llama-server on the same GGUF. That
number is teacher-forced, and the shape of the test matters more than the
number: comparing free-running *replies* is a poor measurement, because one
token going the other way forks the rest of the sentence and eight replies is
eight observations however long they are. Feeding the reference reply back and
asking what this engine would have chosen at every position is 125 observations,
keeps measuring past the first divergence, and stops a fork from hiding
everything behind it.

The one disagreement is at the tightest margin in the corpus — llama-server's
own `n_probs` records it winning by 0.056 nats, against a corpus that runs up to
3.0 — so it is two f16 implementations with different reduction orders, which no
version of this engine would not have. Recording the reference's margins is what
makes that a fact rather than an argument.

**The gap worth naming**: this model has *one* reference, not two. Every other
model here has a float32 🤗 oracle with per-layer taps beside its product
comparison; there is no 🤗 checkpoint for this one on this machine at all. So
agreeing with llama-server proves the replacement is a replacement, and does not
prove either of them computes the reference arithmetic. The loader was worth building on its own
terms for the same reason phase 5a was: it is the half that proves the geometry
is understood, and the half that costs nothing to keep if the arithmetic never
follows.

Three findings, each of which would have been a silent wrong answer rather than
an error:

**GGUF stores shapes transposed.** Dimensions are fastest-varying first, so
`blk.0.attn_k.weight` is `[4096, 1024]` on disk and `[1024, 4096]` row-major.
Binding against the stored order would agree for every square projection and
transpose only `k`, `v` and the feed-forward — a model correct in most places,
which is harder to diagnose than one wrong everywhere.

**The refusal of grouped-query attention was in the wrong crate.** Breeze2 is 32
query heads over 8 key-value heads, and `LlamaConfig::check` refused that
outright, which made a model that binds perfectly cleanly unreadable. A shape is
a fact about the file; whether a forward pass maps several query heads onto one
is a fact about that engine. So `check` now accepts it and
`refuse_grouped_query` is what an engine without the head mapping calls at open.
`xabe-translate` calls it, so nothing regressed.

**`rope_freqs.weight` has no safetensors counterpart**, and `rope_theta` is
500000 rather than 10000. Neither has a shape check that would catch it: a
defaulted rope base gives a model fluent for one sentence and drifting after,
and a skipped tensor gives a schema that binds 291 of 292 and calls it done.

Quantized formats came later and are worth their own line, because the crate
first refused them by name. `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` and the
K-quants `Q2_K` through `Q6_K` all decode now, checked against `gguf-py`'s own
dequantization — the code that wrote the files — at **exact** equality on all
ten. `IQ*`, `TQ*` and `Q8_K` are still refused; the last because it is an
intermediate used while quantizing and never appears in a stored tensor.

The limit to hold in mind is that unpacking is not running quantized: weights
land at full width, so a 4-bit 13 B is a 7 GB file and still 26.5 GB of f16 on
the card. That buys disk and load bandwidth, not memory. See `docs/MODEL.md`.

### The tokenizer, and why it is not `xabe-whisper`'s

The GPT-2 byte-level BPE the Breeze2 file carries — 128,256 tokens and 280,147
merges, all inside the GGUF — is now `xabe_llama::Bpe`. It looks like the one
`xabe-whisper` already has, and reusing it would have been wrong: the GGUF
declares `tokenizer.ggml.pre = "llama-bpe"`, a **different pre-tokenizer** than
GPT-2's. Llama-3 splits digit runs at three, matches contractions
case-insensitively, and lets a newline run take a whole alternative of its own.

The reference is **llama.cpp's `test-tokenizer-0` reading the same GGUF**, not
🤗 — the chat model exists on this machine as a GGUF and nothing else, and
llama-server is the thing being replaced, so matching it exactly is the property
that matters. 60 cases, id-for-id, chosen for where a reimplementation diverges
rather than for being representative text.

Two things fell out of it that a shape check could never have found:

**Digit grouping.** `1234567890` is four tokens, not one. A tokenizer that
borrowed GPT-2's pattern agrees on every sentence of prose and disagrees on
every number — which is the failure mode this whole capture discipline exists
for, since the model still produces fluent text from slightly-wrong ids.

**`\s+(?!\S)` applies to one alternative, not to whitespace generally.** Rust's
`regex` has no lookahead, so the rule is reconstructed explicitly: a run of *k*
spaces before a word is *k-1* spaces plus a word owning the last. Written
without the distinction it also stole a newline from `\s*[\r\n]+`, so a blank
line tokenized as two newlines instead of one. Both readings are real tokens and
both round-trip through `decode`, so nothing but the comparison against
llama.cpp caught it.

## Outside the numbering: the translator reads either container

`Translator::open` takes a 🤗 directory or a `.gguf` file. The 13 B translator
was already on this disk twice — 25 GB of safetensors for this engine and 25 GB
of GGUF for `llama-server` — so this is one model, two containers, and now one
reader.

Checked before wiring, because "same weights" was an assumption and not a fact:

| | |
| --- | --- |
| geometry | identical on every field |
| tensors / parameters | 363 and 13,261,870,080, both ways |
| weights | bit-identical, **except `attn_q` and `attn_k`** |
| tokenizer | identical on all 56,020 pieces; the GGUF adds 4 named padding rows |

The exception is the whole finding. llama.cpp bakes its interleaved rope
convention into the query and key projections by permuting their rows, so those
two tensors differ from the checkpoint in about 98% of their elements while
everything else matches exactly. `attn_v` is untouched, since values are not
rotated — which is the asymmetry that identifies the cause. Left alone it gives
a model that passes every shape check and speaks fluent nonsense.
`xabe_llama::gguf::unpermute_rope` undoes it at load and the tests assert the
round trip in **both** directions, bit for bit. See `docs/MODEL.md`.

The bf16-to-f16 conversion turned out to be a non-issue and is worth recording
as one: bf16 carries 7 mantissa bits against f16's 10, so the mantissa widens
and nothing is rounded at all. Only the exponent range can overflow, and that is
already refused by name.

## What the numbering does not cover

Batching and streaming synthesis are still deliberately absent. They are
answers to questions this project has not asked yet.

Streaming has a caveat now that phase 6 is scoped, and it is worth writing down
before someone reaches item 29 and is surprised. The Python CosyVoice backend
*does* stream — `inference_instruct2(stream=True)` yields audio as the flow
solver produces it, and `gateway.py` plays it as it arrives. Items 24-30 above
describe a non-streaming port: text in, one waveform out. That is the right
first target, because a chunked flow decoder that is subtly wrong sounds like a
chunked flow decoder that is subtly right. But it is a *reduction* against the
service being replaced, not parity with it, and closing it is its own item that
has not been written.

Serving *was* on that list, on the grounds that the pipeline upstream already
had an HTTP surface that worked. That is retracted in phase 2: the engine is
becoming the pipeline, so the gateway is not a surface being duplicated but one
being replaced. See `docs/CLI.md` for the full reasoning.
