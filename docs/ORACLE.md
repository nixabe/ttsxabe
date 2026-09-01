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
llama-quantize models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
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

# The chat tokenizer oracle

The one oracle in this document that is **not** PyTorch, and the one place that
choice is not a compromise.

`Llama-Breeze2-8B-Instruct-text-only.f16.gguf` exists on this machine as a GGUF
and nothing else. Its vocabulary — 128,256 tokens and 280,147 merges — lives
*inside* that file as two metadata arrays, with no `tokenizer.json` beside it
for `AutoTokenizer` to read. So the reference is llama.cpp's `test-tokenizer-0`,
reading the same bytes:

```sh
python tools/oracle/capture_chat_tokenizer.py \
    --model models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
    --out .golden/chat/tokenizer.json
```

`llama-server` is what serves this model today, which makes its tokenizer the
thing being replaced rather than a stand-in for it. Agreeing with it exactly is
the product claim, not an approximation of one.

Note it is slow — about ninety seconds for sixty cases — because
`test-tokenizer-0` loads the model's vocabulary from a 16 GB file once per
invocation and it takes one input file at a time. That is fine for a capture
that runs when the corpus changes; it is the reason the corpus is captured
rather than consulted live.

## `parse_special` is captured off

`test-tokenizer-0` tokenizes with `parse_special = false`, so `<|eot_id|>` in
the input is seven ordinary tokens rather than one. The corpus is captured
against that reading, and it is the one that has to be exact: every character of
user text takes it.

The other reading is a table lookup over ids the capture also records, and
`Bpe::encode` takes it as an argument rather than picking a default, because
either default is wrong half the time. A chat template needs specials parsed; a
user who types `<|eot_id|>` into the box must not be able to end the model's
turn from inside the prompt.

## What the corpus is for

Not representative text. Sixty inputs chosen for where a byte-level BPE with the
`llama-bpe` pre-tokenizer diverges from the GPT-2 one it resembles — and the two
that actually caught something are worth naming, because both produce *real
tokens that decode back to the input*, so nothing short of an id-for-id
comparison would have found either.

**Digit runs split at three.** `1234567890` is four tokens. GPT-2 keeps it as
one. A tokenizer that borrowed the GPT-2 pattern passes every sentence of prose
in the corpus and fails every number in it.

**`\s+(?!\S)` belongs to one alternative.** Rust's `regex` has no lookahead, so
the rule is reconstructed by hand: a run of *k* spaces before a word is *k-1*
spaces plus a word owning the last. Written as a rule about whitespace
*generally*, it also stole a character from `\s*[\r\n]+` — so a blank line
tokenized as two newlines instead of the single token Llama-3 has for it, and
`\r\n` split in half.

The rest of the corpus is there to make a total failure visible before the
awkward cases: Han (which is what this model writes), POJ with combining marks
that have no precomposed form, emoji and a zero-width space through the byte
alphabet, contractions in both cases, punctuation runs, and the pipeline's own
system prompt so at least one input has realistic length.

# The chat model's oracle

The one model here with **one** reference instead of two, and the reason is not
a shortcut. Every other model in this workspace is compared against a float32 🤗
oracle with per-layer taps *and* against the service it replaces. There is no 🤗
checkpoint for `Llama-Breeze2-8B-Instruct-text-only` on this machine at all — it
exists as a GGUF and nothing else — so llama-server running that same GGUF is
both what the pipeline uses today and the only thing to compare against.

That makes this a *product* comparison rather than a numerical one. Agreeing
with llama-server proves the replacement is a replacement; it does not prove
either of them computes the reference arithmetic. Per-layer taps are what would,
and they need an oracle this model has none of. The gap is named here rather
than papered over.

```sh
python tools/oracle/capture_chat_server.py --url http://127.0.0.1:8082 \
    --out .golden/chat/llama_server.json
```

## Comparing replies is the weak test; comparing decisions is the strong one

Free-running text is what the product does and it is a poor measurement. One
token going the other way forks the rest of the sentence, so a single near-tie
reads as a whole reply disagreeing — and eight replies is eight observations no
matter how long they are.

So the main check is **teacher forcing**: llama-server's reply is fed back in
and every position is asked what this engine would have chosen there. That is
125 decisions across the same eight prompts instead of eight, it keeps measuring
past the first divergence, and a fork stops hiding everything behind it.
Measured: **124 of 125 identical**.

## The capture records how close each decision was

`n_probs = 2`, so the golden carries the margin by which llama-server's chosen
token beat the runner-up at every step. That turns "our reply differs" from a
verdict into a question the file already answers: a disagreement at a step it
won by three nats is an arithmetic bug, and one at a step it won by five
hundredths is two f16 implementations with different reduction orders. The
single disagreement is at 0.056 nats, against a corpus whose margins run to 3.0.

Recording the reference's own margins is what lets the test separate those
populations without inventing a tolerance.

## Two traps, both of which produce plausible output

**A generated sequence is not the canonical segmentation of its own text.**
Re-encoding the reply gives a different, equally valid cut — measured, 17 pieces
against the 15 that were generated. A per-position comparison built on
re-encoding is comparing two differently-cut sequences and calling the mismatch
an error. So the capture stores `return_tokens`, and the ids that were actually
produced are what is compared.

**llama-server is not request-independent.** It reuses a KV prefix across
requests, so the same prompt at temperature 0 can produce different text
depending on what was asked before it — and these prompts share a 170-token
common prefix, so the effect has every chance to bite. Each case is sent twice
and the pair must agree before it is recorded.

## Everything is captured greedy

`gateway.py` samples at 0.3, and a sampled reply is not comparable: two correct
implementations drawing from the same distribution give different text. The
capture pins `temperature: 0` with the repetition penalty off, which makes the
reply a function of the prompt alone. The sampler is tested separately, against
the distribution rather than against a draw.

# The CosyVoice3 oracle

`tools/oracle/capture_cosyvoice.py` runs `Fun-CosyVoice3-0.5B` on one reference
clip and one sentence, and writes 43 tensors: the frontend's outputs, the speech
LM's forced log-probabilities, every boundary inside the flow, the vocoder's
per-stage taps, and the waveform.

## Every tap is forced to C order, and that is not paperwork

A tap that is a transposed **view** — `h.transpose(1, 2)`, say — keeps its
original memory, and `np.save` records that faithfully as
`fortran_order: True`. The shape in the header then says one thing and the bytes
say another, and a reader that trusts the shape gets a *permutation* of the
right values.

That cost an afternoon. The estimator's largest input came back at identical rms
and correlation 0.26, which reads as a layout bug in the engine rather than in
the capture — and transposing the tensor on the way in changed nothing at all,
because a transpose of a Fortran-order view is the same bytes again. The capture
now calls `np.ascontiguousarray` on everything, and the readers refuse
`fortran_order: True` by name.

## The speech LM is captured as log-probabilities, because tokens prove nothing

`ras_sampling` draws with `torch.multinomial`. Measured on this capture,
upstream's sampled run agrees with its own greedy argmax at **21 of 143**
positions — so two correct implementations produce visibly different token
sequences, and a token-by-token comparison would be measuring the RNG.

So the captured tokens are fed back in and the *log-probabilities* are compared
at every position. Deterministic, 143 observations rather than one, and it keeps
measuring past the first divergence.

## `frontend_instruct2` deletes the LLM's audio prompt

Asserted in the capture rather than trusted: `llm_prompt_speech_token` is
removed, so in instruct mode the language model sees **only text** and the
speaker is carried entirely by the flow. Wiring the prompt tokens into the LLM
because the zero-shot path does is a mistake that produces fluent speech in the
wrong voice.

## Three buffers that are not in the checkpoint, and are captured anyway

`SineGen2` and `SourceModuleHnNSF` each call `torch.rand` in `__init__` and keep
the result as a plain attribute. None of it reaches `hift.pt`, and upstream
redraws it on every construction — so **upstream does not reproduce across
load orderings either**.

They are captured so the vocoder can be compared against upstream at all. The
engine does not ship them: it draws its own from a named seed, which is
reproducible on its own terms. See `crates/xabe-cosy/src/source.rs`.

One of the three turns out not to matter, and it is worth saying which:
`rand_ini` is added to phase row **0 only**, and the very next operation
decimates by 480 sampling at 239.5 — so row 0 is never read.

## The speaker bundle is a separate tool

`tools/make_cosyvoice_voice.py` runs `campplus.onnx` and
`speech_tokenizer_v3.onnx` **once per voice** and writes four tensors plus the
diffusion's starting noise. It is not part of the capture because it is not a
comparison: it produces an engine *input*, for any clip, rather than a reference
answer for one.

`tools/dump_cosyvoice_tokens.py` is the third: the tokenizer's 281 special
tokens are a literal list in CosyVoice's *source*, not in
`CosyVoice-BlankEN`, and their ids fall out of that list's order.
`<|endofprompt|>` is 151646 only because `<|im_start|>` and `<|im_end|>` were
already in `added_tokens_decoder`. Read out once, written down, rather than
re-derived by hand.

# The Coqui VITS oracle

A second oracle for the same architecture, because it is a different
*checkpoint* and the loader between the file and the arithmetic is new. The
forward pass it checks is the one the first oracle already validated; what it is
really testing is whether 738 tensors under different names, half of them
weight-normalised, land where the first checkpoint's did.

## Provenance

| | |
| --- | --- |
| reference | Coqui `TTS.tts.models.vits.Vits`, `coqui-tts-pygoruut` 0.27.4 |
| checkpoint | `neurlang/coqui-vits-suisiann-minnan-hokkien`, `best_model.pth` |
| device | CPU, float32, one thread |
| capture | `tools/oracle/capture_coqui.py` |
| default output | `.golden/coqui-base` |

The reference needs Python 3.10 and does not coexist with the rest of the
tooling here, so it lives in its own environment:

```bash
/usr/bin/python3.10 -m venv .venv-coqui
.venv-coqui/bin/pip install coqui-tts-pygoruut==0.27.4 "transformers>=4.47,<4.50"
```

The `transformers` pin is not optional. `coqui-tts` imports XTTS at package
import time, which imports `isin_mps_friendly` from `transformers.pytorch_utils`,
which does not exist before 4.45 - so a default resolve installs a version that
cannot be imported at all.

## Capturing

```bash
.venv-coqui/bin/python tools/oracle/capture_coqui.py \
    --out .golden/coqui-base --seed 0 --text "你好！我是蔡贏。我的人在台北。"
```

Same format as the first oracle - raw little-endian tensors, C order, a
`manifest.json` with shapes, dtypes and SHA-256 - so `xabe-golden` reads both
without a second parser. Same hooks in spirit, on a different module tree, and
the same `TorchFunctionMode` for the two random draws and the alignment matrix.

## The manifest records the phonemes, and that is load-bearing

This checkpoint's text front end is `pygoruut`, and it is not in this engine
(`docs/MODEL.md` says why). The capture therefore records **both** the text it
was given and the IPA that text became, and every differential test reads
`Manifest::input()` - the phonemes - rather than `text`.

A test that read `text` would not fail loudly. It would tokenise Han characters
to nothing, get an empty symbol sequence, and compare an empty utterance against
a real one. `Manifest::input()` exists so that choosing correctly is the default
rather than a thing to remember.

## The two references disagree about tensor layout

Coqui's text encoder carries its activations as `[B, C, T]`; 🤗's carries them
as `[B, T, C]`, because its port transposes the projection's output back before
splitting it. That applies to `embed`, every `enc_layer_*`, `enc_out`, `m_p` and
`logs_p`. After the duration expansion both are `[B, C, T]`, so `z_p`, `z` and
the waveform need nothing.

`capture_coqui.py` transposes on the way out rather than leaving it to the
tests, so one capture format means one convention per stage name and a test
never has to ask which dialect it opened.

This cost a round, and how it presented is the useful part: the **waveform
matched end to end** while `enc_out` disagreed at 18,595 of 18,624 values with a
maximum absolute error of 6.0. An arithmetic bug that large does not leave the
output correct. A transposed comparison does exactly that, and looks like this
every time.

## The phonemiser has to be stopped, not dropped

`pygoruut` starts the goruut binary with the parent's stdout inherited. Leaving
it running holds the pipe open for anything that captures the script's output,
so `PHONEMES=$(... phonemize_pygoruut.py ...)` waits forever on the daemon
rather than on the script. Both `capture_coqui.py` and
`tools/phonemize_pygoruut.py` terminate it explicitly; `__del__` at interpreter
shutdown is not reliable enough to depend on.

## What agreement looks like

Maximum absolute difference against the capture, on the CPU path, at
`atol=1e-4`, `rtol=1e-3`:

| stage | max abs |
| --- | --- |
| `input_ids` | exact |
| `embed` | 0 |
| `enc_layer_0` .. `enc_layer_5` | 2.9e-6 .. 9.5e-6 |
| `enc_out` | 3.3e-6 |
| `m_p` | 2.9e-6 |
| `logs_p` | 3.7e-7 |
| `waveform` (57,088 samples) | 5.8e-5 |

The CUDA path is held to `atol=2e-3`, `rtol=2e-2` for the reason
`gpu_end_to_end.rs` gives, and reaches 6.2e-5 on the waveform.

# The Tâi-lô to IPA oracle

The odd one out: there is **no reference implementation of this conversion**.
`xabe-taigi` turns romanisation into the IPA the Coqui checkpoint reads, and
goruut — the only other thing that produces that IPA — starts from Han instead.
There is nothing to run side by side.

So the oracle is built sideways, out of two things that already exist.

## Provenance

| | |
| --- | --- |
| corpus | `ceciliayl/SuiSiann_raw_tone`, `SuiSiann.csv` — 3,467 rows |
| phonemiser | goruut `MinnanHokkien2`, via `coqui-tts-pygoruut` 0.27.4 |
| inventory | `neurlang/goruut` `dicts/minnan/hokkien2/language.json`, 4,608 entries |
| capture | `tools/oracle/capture_tailo_ipa.py` |
| default output | `.golden/coqui-tailo/correspondence.json` |

SuiSiann is the corpus this checkpoint was trained on, and its metadata carries
both columns: the Han text of every recorded sentence **and** its Tâi-lô
transcription. Phonemise the Han with goruut and the two halves line up
syllable by syllable, so each sentence yields a set of (Tâi-lô, IPA) pairs.
Aggregated over the corpus that is a correspondence table nobody wrote down —
1,516 distinct syllables over 28,489 tokens.

```bash
.venv-coqui/bin/python tools/oracle/capture_tailo_ipa.py --out .golden/coqui-tailo
```

Only sentences whose two halves have the same syllable count are used: 2,152 of
3,467. A mismatch means goruut merged or split something, and guessing an
alignment would invent correspondences rather than record them.

## The two halves disagree, and that is the point

goruut has to guess which reading a Han character takes; the transcription is
what the speaker said. They differ on about a quarter of tokens — 我 as `ŋɔ˥˧`
where the corpus says `ɡua˥˧`, 人 as `dzin˨˦` where it says `laŋ˨˦`. The
transcription is right every time, because the audio is the ground truth.

So `tests/correspondence.rs` asserts a **floor on agreement**, not equality.
Exact equality would be a test that this crate reproduces goruut's mistakes,
which is the opposite of what is wanted. Measured when it was written:

| | |
| --- | --- |
| matches goruut's commonest reading | 71.5% of syllables |
| attested among goruut's readings | 80.2% |
| token-weighted | 71.3% |

A real regression in the table moves those by tens of points. A different
reading moves them by fractions.

## The sharp half is structural, not statistical

An agreement rate cannot catch a table that invents a phoneme — a wrong initial
would just look like another reading disagreement. So the capture also records
goruut's **inventory** from the dictionary: every initial, rime, tone letter and
syllable body it can write. The test then requires that

- every initial `xabe-taigi` can produce is one goruut writes — **18 of 18**,
- every tone letter is one goruut writes — **7 of 7**, exactly the set,
- at least 95% of syllable bodies are attested — **97.9%**, 1,476 of 1,508.

The residue of that last one is not error. goruut's dictionary has 4,608
entries and the language has more syllables than that has words, so `ɡun`,
`tʰue`, `dzik` and about thirty others are perfectly ordinary Taiwanese that
simply never came up, plus `russia` and `putin`, which are in the corpus and are
not syllables at all.

**This is what caught the two real bugs.** `ainn` was parsing as `ai` plus a
stray `n`, losing the nasalisation while keeping a plausible shape; and `thng`,
`hng`, `tshng` — syllables whose rime is a syllabic `ng` after an initial — were
not parsing at all and were being dropped. Neither moved the agreement rate
much. Both produced a body goruut has no word for, and the inventory check named
them.

## One quirk left in, deliberately

goruut writes the rime `ai` + `-h` as `aih`, with a literal `h`, where every
other checked rime gets `ʔ`. It does that in exactly two dictionary entries —
`aih` and `haih` — and `aiʔ` appears nowhere in the file.

`xabe-taigi` converts `-h` to `ʔ` uniformly and does not reproduce it. Two
entries is not a rule, and inferring one from them would put an `h` into
`saih`, `uaih` and every other `ai`-plus-stop syllable that goruut has never
spelled and has no opinion about. It costs 13 tokens of 28,489.
