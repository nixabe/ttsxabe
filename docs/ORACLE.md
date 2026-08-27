# The PyTorch oracle

## 1. Provenance

The reference is 🤗 Transformers' `VitsModel`
(`transformers/models/vits/modeling_vits.py`), running `facebook/mms-tts-nan`
on CPU in float32. It is the definition of correct for this project. Where this
implementation and the reference disagree, the reference is right until proven
otherwise in writing.

Version matters: record the `transformers` version in the capture manifest. The
VITS implementation has changed shape before.

## 2. Why captured, not described

Expected values are **captured as binary**, never hand-transcribed into a test.
Transcribed constants are a copy of what someone believed the model does, and
they rot silently. A capture is a copy of what it actually did.

This also makes the failure legible: when a differential test fails you can
bisect the *stage*, because every intermediate was captured, not just the
waveform.

## 3. What is captured

One record per stage of the inference path, for a fixed input and a fixed seed:

| tensor | shape | why it is a checkpoint |
| --- | --- | --- |
| `input_ids` | `[1, T]` | tokenisation is part of the contract |
| `embed` | `[1, T, 192]` | isolates the embedding table read |
| `enc_out` | `[1, T, 192]` | the 6-layer text encoder, incl. relative attention |
| `m_p`, `logs_p` | `[1, 192, T]` | the projection split |
| `log_duration` | `[1, 1, T]` | stochastic duration predictor |
| `noise_dur`, `noise_prior` | — | the exact draws used, so sampling is reproducible |
| `attn` | `[1, F, T]` | the expansion from symbols to frames |
| `z_p` | `[1, 192, F]` | prior after length regulation |
| `z` | `[1, 192, F]` | flow reversed — the most bug-prone stage |
| `waveform` | `[1, S]` | the whole thing |

Capturing `z_p` and `z` separately is deliberate: the flow is four coupling
blocks and a swapped half is invisible in the waveform but obvious here.

Three more are captured than the table asks for, because they cost nothing and
each turns a whole-stage failure into a located one:

| tensor | shape | why |
| --- | --- | --- |
| `embed_raw` | `[1, T, 192]` | the table lookup *before* the `sqrt(192)` scaling, so a missing scale factor is distinguishable from a wrong embedding row |
| `enc_layer_0..5` | `[1, T, 192]` | per-layer encoder output; turns "the text encoder is wrong" into "layer 3 is wrong" |
| `waveform_raw` | `[1, 1, S]` | the decoder's output before the channel squeeze |

`attn` is captured squeezed to `[1, F, T]`, which is the form the two
expansion matmuls actually use.

## 4. Format

A directory `.golden/<name>/` holding `manifest.json` and one `.bin` per tensor:
raw little-endian f32, C order, shape in the manifest. Same convention as
`xabe-st` reads, so the test harness needs no second parser.

`.golden/` is gitignored. It is regenerable and it is large.

## 5. Regenerating

```sh
python tools/oracle/capture.py --out .golden/base --seed 0 \
    --text "lí hó, kin-á-ji̍t thinn-khì chin hó."
```

Run it in an environment that has `torch` and `transformers`; the `taigi` conda
env upstream of this project already does. The capture is bit-reproducible -
two runs at the same seed produce identical SHA-256 for all 20 tensors - which
is the reason `torch.set_num_threads(1)` is in the script and not a stray
leftover. Float32 reduction order is not thread-invariant, and without it the
last bits of every tensor move between runs.

Reading it back is `xabe-golden`:

```rust
let g = Golden::open_default().expect("capture present");
let c = g.compare("enc_out", &computed, 1e-5, 1e-5)?;
assert!(c.passed(), "{c}");
```

Checksums are verified on every read. A truncated `.bin` has no header to
disagree with and would otherwise read as a plausible shorter tensor, failing
much later somewhere unrelated.

The seed is part of the record, not a default. A capture whose seed is unknown
is not an oracle.

## 6. Fidelity limits

- **CPU float32 only.** The reference on GPU produces slightly different results;
  comparing against a GPU capture would fold PyTorch's kernel choices into our
  definition of correct.
- **The tokenizer is captured, not reimplemented from the vocab file.** It
  lowercases and drops out-of-vocabulary characters, and those rules are worth
  pinning rather than inferring.
- **Sampling is pinned by capturing the noise, not the seed alone.** Two RNG
  implementations agreeing on a seed is not something to assume across
  languages.

## 7. Things found that the docs do not say

- The vocabulary is **POJ**, not Tâi-lô — `c` and U+0358 are present, `ⁿ` is
  not. See [MODEL.md](MODEL.md); this was found by reading the vocabulary, and
  confirmed by round-tripping synthesised audio back through an ASR model.
- The model card's example text (`"some example text in the Chinese, Min Nan
  language"`) is a generic placeholder that produces nothing meaningful. It is
  not an indication of the expected input format.
- `posterior_encoder`, a fifth of the checkpoint, is never read at inference.
- **The reverse duration path skips one of its own flows.** `self.flows` holds
  an elementwise-affine block followed by four convolutional flows, but the
  reverse pass runs `flows = list(reversed(self.flows))` and then
  `flows[:-2] + [flows[-1]]`, which drops `flows[1]` - the *first* convolutional
  flow. The reference comments this only as `# remove a useless vflow`. So the
  reverse order is conv 4, conv 3, conv 2, affine: four blocks, not five, and
  the one that is skipped is not the last one. Reimplementing the list as
  written rather than as reversed produces durations that are wrong but not
  obviously so - the audio still sounds like speech, at the wrong pace.
- **The noise is drawn in a fixed order.** `torch.randn` is called exactly twice
  on the inference path: the duration predictor's `[1, 2, T]` latents first,
  the prior's `[1, 192, F]` second. The capture relies on that order, and the
  second draw's shape depends on the *first* draw's outcome, so the two cannot
  be reordered even in principle.

---

# The VAD oracle

The reference for voice activity detection is **whisper.cpp's**, not Python
silero-vad, and that is a deliberate exception to the rule that the upstream
author's implementation is the oracle.

Every threshold the pipeline runs with — `vad_start`, the segmenter's 0.6, the
turn-taking constants in `xabe-serve::turntaking` — was tuned against
whisper.cpp's probabilities. whisper.cpp differs from upstream on purpose: it
parses `n_context` and then ignores it, substituting a reflective pad. Matching
Python instead would produce a more faithful Silero and invalidate every number
the rest of the system is built on.

## Capturing

```sh
WHISPER=~/whisper.cpp tools/oracle/capture_vad.sh
```

It builds `tools/oracle/vad_capture.cpp` against a built whisper.cpp checkout,
generates the corpus, synthesises two speech clips with the engine itself at a
fixed seed, and writes per-clip `probs.bin` and `segments.bin` under
`.golden/vad/`, with a `manifest.json` recording the whisper.cpp commit and the
segment parameters.

## The corpus

Eight clips, all deterministic (`tools/oracle/vad_corpus.py`, seed 20250827).
Four of them are the pipeline's known hallucination triggers, and they are the
reason the corpus exists:

| clip | what the ASR did with it |
| --- | --- |
| `silence` | digital silence transcribed as `我…` |
| `hiss` | faint broadband noise as `我現在在醫院` |
| `room` | low-frequency room tone as `(我會陪你一起走)` |
| `click` | a single transient opening a turn on its own |

The rest exercise the segmenter rather than the detector: `bursts` has two tone
spans with a gap the 200 ms merge has to decide about, `click_then_tone` puts a
transient immediately before speech so the onset rule and the padding interact,
and the two `speech` clips are real synthesis.

## The conversion

The VAD ships as legacy ggml, not safetensors — 864 KB with the magic `ggml` and
a hand-rolled header. `tools/vad/ggml_to_safetensors.py` converts it once, so
`xabe-st` keeps its single job rather than learning a second container format
for one 15-tensor model. The ggml header's geometry is written into
`__metadata__`, which is what lets the weight schema check the shapes it binds
against what the original file declared.

# The ASR oracle

The reference is 🤗 `WhisperForConditionalGeneration` on CPU in float32 with
one thread, **not** whisper.cpp — the opposite choice from the VAD, and for a
reason that reverses cleanly. The VAD exists to reproduce a *decision* the rest
of the pipeline was tuned against, so whisper.cpp's own divergence from upstream
Silero is the thing to match. The ASR exists to produce *text*, and here
whisper.cpp is the one that diverges: its tokenizer is a greedy longest-match
over a `std::regex` in which `[[:alpha:]]` is not `\p{L}`, so it already
disagrees with the reference on Han input, which is all this engine ever
transcribes. `whisper-server`'s transcripts stay a cross-check.

## Capturing

```sh
python tools/oracle/capture_asr.py --model models/asr/breeze-asr-26 \
    --wav .golden/vad/clips/speech.wav --out .golden/asr/speech
python tools/oracle/capture_tokenizer.py --model models/asr/breeze-asr-26 \
    --out .golden/asr/tokenizer.json
```

The first writes the mel filter bank, the input features, the samples, the
first four encoder and decoder block outputs, both final normalisations, the
logits and a greedy transcript, all as raw little-endian f32 with a
`manifest.json`. The per-layer taps are the point: "the encoder is wrong" is
not a fact anyone can act on, and "layer 7 is wrong" is.

`torch.set_num_threads(1)`, because f32 reduction order is not thread-invariant
and the last bits of every tensor move without it.

## The corpus reuses the VAD's clips

Deliberately. `silence` transcribes as `我…` — the hallucination the VAD exists
to prevent — so the same clip is evidence in two places at once: that the VAD
gates it, and that the ASR reproduces the reference's mistake exactly rather
than a different one.

## What the reference gets wrong

Two things are asserted directly rather than captured, because capturing them
would enshrine a defect. Both are recorded here so that a future capture does
not quietly reintroduce them.

**`decode_with_timestamps` is broken in transformers 5.15.1.** It computes
`timestamp_begin = self.all_special_ids[-1] + 1`, and `all_special_ids` on this
checkpoint holds exactly one entry — `<|endoftext|>`, 50257 — so it renders
`<|startoftranscript|>` as `<|0.00|>` and every control token after it as a
timestamp. The engine uses the real boundary, the id of `<|0.00|>`.

**`<|nospeech|>` is not in this checkpoint.** 50362 is spelled
`<|nocaptions|>`, OpenAI's original name. Asking the reference for
`<|nospeech|>` does not fail: it returns the *unknown* id, 50257, which is also
end-of-text. A caller that trusts the answer stops the decode on the wrong
token, and it looks like a model that gives up early.

## The filter bank is computed, not captured

`WhisperFeatureExtractor`'s mel bank is the one thing here that could have been
stored beside the model and is not. It is a closed form in four numbers, and
this engine's version matches the capture *bit for bit* — both sides evaluate
it in f64 and round once, with no reduction anywhere for an ordering to
disagree about. The capture is kept as the test. Shipping it as an asset would
have added a file that can go missing, go stale, or silently be the `htk`
variant instead of `slaney`.

---

# The translator oracle

The translator is the only stage with **two** references, because there are two
different questions to answer and one reference each.

| reference | what it says |
| --- | --- |
| 🤗 `LlamaForCausalLM`, CPU, float32, one thread | what the arithmetic *should* be |
| `llama-server` on the f16 GGUF of the same weights | what the pipeline runs *today* |

The first is the oracle in this document's sense: per-layer taps, exact
comparison, a gate on the numbers. The second is the thing being replaced, so
agreeing with it is the actual product claim. Having both is what let the one
disagreement be resolved instead of tolerated.

## Capturing

```sh
python tools/oracle/capture_llama.py --model models/translator/taigi-llama2-13b \
    --out .golden/translator/trans --src 今天天氣很好 --tgt POJ
python tools/oracle/capture_llama_tokenizer.py \
    --model models/translator/taigi-llama2-13b --out .golden/translator/tokenizer.json
python tools/oracle/capture_llama_server.py --url http://127.0.0.1:8081 \
    --out .golden/translator/llama_server.json
```

`capture_llama.py` writes the prompt ids, the embedding, the first four decoder
block outputs, the final norm, the logits and a greedy continuation, with a
`manifest.json`. Three prompts are captured — one per target script — because a
POJ answer and a Han answer take different paths through the vocabulary.

`torch.set_num_threads(1)`, for the same reason as the ASR: f32 reduction order
is not thread-invariant.

`capture_llama_server.py` sends `gateway.py`'s request body unchanged —
template, `temperature: 0`, `repeat_penalty: 1.1`, `n_predict: 256`, stops
`["[/", "\n["]`. A comparison against a differently configured server is not a
comparison.

## `llama-server` is not request-independent

This is the finding worth carrying. Seven of eight prompts matched and the
eighth, `你食飽未` to Han, did not: llama-server answers `你食飽未？` and this
engine answers `你食飽未`. Capturing the float32 🤗 oracle for that exact prompt
settled it — 🤗 answers `你食飽未`, agreeing with this engine.

So llama-server is the one that diverges, and the mechanism is not mysterious:
it **reuses a KV prefix across requests**. The same prompt at temperature 0 can
therefore produce different output depending on what was asked before it, which
means a captured llama-server transcript is a record of one server's history and
not a function of its input. That is exactly why it is the cross-check and 🤗 is
the oracle — the same relationship `whisper-server` has with 🤗 Whisper, arrived
at from the opposite direction.

## Two identical error values that were not a bug

Two different prompts reported the same maximum per-layer error to every digit,
which looked like a tap wired to the wrong tensor. It was not: the maximum lands
in the `<s>` row, and `<s>` is the same input in both prompts. A number that
repeats is worth checking and is not automatically wrong.

## A capture that was deleted rather than kept

`decode_with_timestamps` is not the only reference behaviour that turned out to
be a bug. The rule that came out of both: when the reference is wrong, do not
capture it. Assert the intended behaviour directly, in the test, with the reason
attached — a capture is a record of what the reference does, and enshrining a
defect in one makes it permanent and invisible.

---

# The quantization oracle

The reference is upstream llama.cpp's `gguf-py/gguf/quants.py` — literally the
code that writes the files being read — and the comparison is **exact**. There
is no tolerance to negotiate: this workspace never quantizes, it only unpacks,
and an unpacking either reproduces the reference bit for bit or has an indexing
bug.

## Capturing

```sh
python tools/oracle/capture_quants.py --out .golden/gguf/quants
```

Ten formats: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` and `Q2_K` through `Q6_K`.
For each it writes the packed bytes and the f32 the reference unpacks them to,
plus a `manifest.json` with the block geometry.

## Why the corpus is random bytes, not a quantizer's output

Two reasons, and the second is the better one.

Python `gguf` implements `quantize` for the five legacy formats only — the
K-quants exist in C alone — so a round-trip corpus would have covered half the
table and left the harder half untested.

And a quantizer only ever emits the well-conditioned subset of the encoding
space. Random encodings reach every nibble, every packed six-bit scale and
every high-bit mask. The failure this is guarding against is an element
*ordering* mistake, which produces a permutation of the right tensor — same
values, same histogram, no correlation — and a permutation is exactly the kind
of bug that survives a well-behaved corpus.

The one thing not left to chance is the f16 scale fields. Random bits there are
NaN about 3% of the time, which says nothing about a layout. Those offsets are
listed per format and filled with finite values spanning several orders of
magnitude, so no two blocks share a scale and a per-block indexing error has
nowhere to hide. For the five formats Python can quantize, a real round trip is
captured as well, so the corpus holds both the whole space and the part a
quantizer reaches.

## And one real file

`crates/xabe-gguf/tests/quantized_model.rs` opens an actual
`llama-quantize` output beside its f16 original. It is skipped unless
`XABE_QUANT_DIR` points at one, because the files are gigabytes and are
reproducible in a single command:

```sh
llama-quantize models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
    $XABE_QUANT_DIR/breeze-Q4_K_M.gguf Q4_K_M 8
```

What it asserts is **correlation**, not mean error, and that is the point: a
lossy-but-correct unpacking correlates above 0.99 with the weights it encoded,
while a permuted one correlates around zero despite having identical values and
an identical histogram. Mean absolute error would pass a permutation; this does
not.

It also checks that a `_M` mix carries more than one format in one file, which
is what `Q4_K_M` means — llama-quantize picks per tensor role. A reader that
assumed one type per file would open it and mis-size most of it.
