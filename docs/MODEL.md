# The model

`facebook/mms-tts-nan` — VITS, trained by Meta's Massively Multilingual Speech
project on Min Nan. 36,286,512 parameters in 762 F32 tensors, 139 MB, 16 kHz
output. Licence CC-BY-NC 4.0: non-commercial, and not redistributed here.

## Geometry

| | |
| --- | --- |
| hidden size | 192 |
| text encoder layers | 6, 2 heads, FFN 768, kernel 3 |
| relative attention window | 4 |
| vocabulary | 48 symbols |
| flow | 4 coupling blocks, 4 WaveNet layers each, kernel 5, dilation 1 |
| duration predictor | stochastic, 4 flows, kernel 3 |
| decoder upsample rates | [8, 8, 2, 2] — 256× total, giving 16 kHz |
| decoder upsample kernels | [16, 16, 4, 4] |
| decoder channels | 512 → 256 → 128 → 64 → 32 → 1 |
| resblock kernels | [3, 7, 11], dilations [[1,3,5], [1,3,5], [1,3,5]] |
| noise scale | 0.667 (prior), 0.8 (duration) |

## Where the parameters are

| component | tensors | params | share | needed at inference |
| --- | --- | --- | --- | --- |
| `decoder` | 155 | 14.33 M | 39.5% | yes |
| `posterior_encoder` | 100 | 7.24 M | 19.9% | **no** |
| `flow` | 112 | 7.10 M | 19.6% | yes |
| `text_encoder` | 111 | 6.30 M | 17.4% | yes |
| `duration_predictor` | 284 | 1.32 M | 3.6% | yes |

`posterior_encoder` encodes ground-truth spectrograms during training and has no
role in synthesis. One fifth of the checkpoint is never read.

## The inference path

```
Tâi-lô/POJ text
  → embed_tokens                     [48, 192]
  → text_encoder                      6 × (relative self-attention, conv FFN)
  → project                          [384, 192, 1]  → m_p, logs_p
  → stochastic duration predictor    → per-symbol durations
  → length regulation                → expand to frame count
  → flow, reversed                   4 coupling blocks → z
  → decoder (HiFi-GAN)               conv_pre → 4 upsamples with resblocks → conv_post
  → waveform                          16 kHz mono
```

## The vocabulary is POJ, not Tâi-lô

This cost a real bug in the pipeline upstream, so it is written down here.

The 48 symbols are lowercase Latin, tone diacritics, `-`, `'`, `|` and space.
Two of them settle the orthography:

- **`c` is present.** POJ writes `ch`/`chh`; Tâi-lô writes `ts`/`tsh` and has no
  use for `c`.
- **U+0358 COMBINING DOT ABOVE RIGHT is present.** That is POJ's `o͘`. Tâi-lô
  writes `oo` and has no dotted-o.

The one POJ symbol missing is the nasal **`ⁿ` (U+207F)**, which must be written
`nn`.

Absent: `d f q r v w x y z`, all digits, and all punctuation except `-` and `'`.
The tokenizer silently drops anything out of vocabulary, so `,` and `.` cost
nothing but also convey nothing — phrasing has to come from the text itself.

Measured, on a sentence whose two spellings differ only in `chin` vs `tsin`,
synthesised and round-tripped through Breeze-ASR-26:

| input | ASR read-back |
| --- | --- |
| `... thinn-khì chin hó.` (POJ) | 你好 今天天氣很好 |
| `... thinn-khì tsin hó.` (Tâi-lô) | B號今天登記正好 |

So: **feed it POJ, replace `ⁿ` with `nn`, change nothing else.**

## Tokenisation

The whole tokeniser is: lower-case, drop everything outside the 48-symbol
vocabulary, intersperse token 0 between every pair of symbols and at both ends.
No subword model, no phonemiser, no romaniser. Its difficulty is entirely in
what it does *silently*.

- **Input must be NFC.** The vocabulary holds precomposed `í` (U+00ED) but not
  combining acute (U+0301), so decomposed input loses every tone mark: `lí`
  becomes `li`. Tones are lexical in this language, so that is a different
  word, not a blemish - and nothing reports it. Meanwhile U+030D (entering
  tone) and U+0358 (the o-dot) have no precomposed form and *are* in the
  vocabulary, so NFC is exactly right: it composes the vowel diacritics and
  leaves those two combining.
- **Punctuation is deleted, not honoured.** A comma is not a pause. It ceases
  to exist, along with the phrasing it implied.
- **`<unk>` is unreachable.** It is an added token, so the normaliser preserves
  the literal five characters and the filter then keeps `unk` and drops the
  angle brackets. No input can emit the unknown id - which is just as well,
  since it is 48, one past the end of a 48-row embedding table.
- **Empty input produces no symbols, not a lone blank.** The natural reading of
  "intersperse a blank" would give `[0]`; the reference gives `[]`.

## The decoder's last leaky ReLU has a different slope

Every leaky ReLU in the HiFi-GAN decoder takes `config.leaky_relu_slope`, which
is 0.1 - except the last, immediately before `conv_post`, where the reference
writes `nn.functional.leaky_relu(hidden_states)` with no slope argument and so
gets PyTorch's default of **0.01**.

Deliberate or not upstream, it is what produced these weights, so it is what
correct means. Using 0.1 there changes only the negative half of one
activation: audible as a slight roughness, invisible to every shape check.

## `torch.flip` on the channel axis

Both the duration predictor and the prior flow call `torch.flip(x, [1])`
between coupling blocks. It reverses the **whole** channel axis: for
`[1, 192, T]` the output channel order is 191, 190, ... 0.

It is tempting to read it as "swap the two halves", because that is what a
coupling flow conceptually wants and because it is *exactly right* for the
duration predictor - which has two channels, where reversing and swapping are
the same operation. At the flow's 192 channels they are not, and the mistake
costs 30,134 of 30,144 values while breaking no shape.

## Weight layout notes

Convolutions are stored `[out_channels, in_channels, kernel]`. The decoder's
upsamplers are *transposed* convolutions and store `[in, out, kernel]` —
`decoder.upsampler.0.weight` is `[512, 256, 16]`, going 512 channels in to 256
out. Reading that pair in the wrong order produces audio, which is precisely the
failure mode `docs/TESTING.md` exists to catch.

Bias, in a transposed convolution, is still per *output* channel — so the bias
length does not match the weight's leading dimension the way it does everywhere
else. That asymmetry is a separate loader path in `xabe-vits::weights`, not a
flag on the ordinary one.

Weight normalisation is applied inconsistently across the checkpoint, and this
is the detail most likely to waste an afternoon. PyTorch's `weight_norm`
reparameterises a kernel as `g · v / ‖v‖`, storing `weight_g` and `weight_v`
instead of `weight`. In this checkpoint **only the flow's WaveNet layers are
stored unfused**:

| module | storage |
| --- | --- |
| `flow.flows.*.wavenet.*` | `weight_g` + `weight_v` |
| `decoder.upsampler.*` | plain `weight` |
| `decoder.resblocks.*` | plain `weight` |
| `decoder.conv_pre`, `conv_post` | plain `weight` |
| text encoder, duration predictor | plain `weight` |

So the flow needs the norm computed at load time and the decoder does not.
Assuming either rule holds everywhere fails loudly — `MissingTensor:
decoder.upsampler.0.weight_v` in one direction, a silently wrong kernel in the
other. `conv_post` additionally has **no bias**; the `Conv` type carries
`Option<&[f32]>` for that one tensor alone.

---

# Silero VAD, v5.1.2

Fifteen tensors, 884 KB, and the only thing standing between the pipeline and
an assistant that answers sentences the ASR invented. It is not optional: on
digital silence Breeze-ASR-26 produced `我…`, on faint hiss `我現在在醫院`, on
room noise `(我會陪你一起走)`.

## Geometry

```
frame[512]  (32 ms at 16 kHz, one probability out)
  reflect-pad 64 each end                       → [640]
  conv1d, 258 kernels of 256, hop 128           → [258, 4]
  sqrt(re² + im²) over the two halves of 258    → [129, 4]
  conv 129→128, k3 s1 p1, ReLU                  → [128, 4]
  conv 128→64,  k3 s2 p1, ReLU                  → [64, 2]
  conv 64→64,   k3 s2 p1, ReLU                  → [64, 1]
  conv 64→128,  k3 s1 p1, ReLU                  → [128, 1]
  LSTM cell, hidden 128, gates i f g o          → [128]
  ReLU, dot with 128 weights, + bias, sigmoid   → one probability
```

The STFT is a convolution: the basis is 129 cosines followed by 129 sines,
each already multiplied by the analysis window, so a plain convolution gives
the real and imaginary parts and the magnitude is one `sqrt` per bin.

**The LSTM state carries across frames.** That is what makes this a detector
rather than a classifier — the probability for one 32 ms frame depends on
everything before it — and it is why `Vad::reset` is required between
independent clips rather than merely tidy.

**Gate order is i, f, g, o.** PyTorch's stacking, and not the only convention in
use. Getting it wrong produces a detector that runs, converges to
plausible-looking probabilities, and is wrong everywhere.

## The checkpoint is F16, and that matters

The convolutions and the STFT basis are stored half precision; only the biases
are F32. This was not in the plan — it was found by converting the file — and it
is why `xabe-st` grew F16 and BF16 support ahead of phase 5a.

It also explains the whole of the disagreement with the reference. whisper.cpp
runs ggml's F16 kernels, which round the **activations** to half precision at
the input of every convolution. This implementation widens the weights once at
load and keeps activations in f32. Rounding this implementation's conv inputs
through f16 as an experiment drops the worst disagreement across the corpus
from **6.8e-3 to 1.8e-4**, a 37× reduction, which locates the difference
entirely in that one choice.

F32 is kept. It is the more accurate of the two, it is what upstream Silero
computes in, and the experiment shows the reference's extra error is ggml's
storage format rather than anything about the model. What is asserted instead is
the property the pipeline depends on: **every one of the 926 frames in the
corpus lands on the same side of both thresholds as the reference**, and every
segment on every clip matches.

## Two divergences from upstream Silero, both whisper.cpp's

1. **`n_context` is parsed and ignored.** Upstream prepends 64 *real* samples of
   the previous frame; whisper.cpp substitutes a symmetric reflective pad. This
   follows whisper.cpp, because whisper.cpp produced every threshold in the
   pipeline and matching Python instead would invalidate the tuning.
2. **The segmenter has two rules upstream does not**: adjacent segments closer
   than 200 ms are merged, and a second minimum-duration sweep runs after that
   merge. Both are in `xabe-vad::segments`, marked where they appear.

## Segmenter constants

| | | |
| --- | --- | --- |
| `neg_threshold` | `max(threshold - 0.15, 0.01)` | hysteresis; one threshold chops a turn up at every unvoiced consonant |
| `min_silence_at_max_speech` | 98 ms | where a segment *could* be split if it overruns — a different question from whether it has ended |
| merge gap | 200 ms | whisper.cpp's; without it one sentence arrives as several ASR requests |
| second min-duration sweep | — | whisper.cpp's; the first sweep can pass a short segment the merge then fails to absorb |

# Whisper, `Breeze-ASR-26`

A large-v2 fine-tune for Taiwanese Mandarin and Taigi. 1,259 tensors,
1,543,304,960 parameters, F32 throughout — which is why it came before the
translator: no dtype work is on its critical path.

## Geometry

| | |
| --- | --- |
| `d_model` | 1280 |
| encoder / decoder layers | 32 / 32 |
| attention heads | 20, head dim 64 |
| feed-forward dim | 5120 |
| mel bins | 80 |
| encoder positions | 1500, from 3000 mel frames at stride 2 |
| decoder positions | 448, learned, no extrapolation |
| vocabulary | 51,865 |

The encoder's window is fixed at 30 seconds. A 2.7-second clip and a
29-second one cost exactly the same encoder pass, which is the single most
important thing to know about the ASR's timing.

## The frontend

400-point transform, hop 160, periodic Hann, centred with a reflective pad of
200, 80 slaney-normalised mel filters up to 8 kHz, then `log10`, a floor at
`max - 8`, and `(x + 4) / 4`.

**The normalisation is global, and it is load-bearing.** The floor is taken
over the *entire* 30-second window, so the value at second one depends on the
loudest frame anywhere in the file — including in the silence that was padded
on. That is why the frontend takes a whole window rather than a stream: any
chunked design has to answer this, and there is no good answer.

The filter bank is computed rather than shipped. `slaney`, not `htk`: they
disagree by a few percent across the whole band, which moves every filter edge
and produces a spectrogram that looks entirely reasonable and transcribes
badly.

The window is *periodic* Hann — `torch.hann_window` divides by `n`, and the
symmetric variant every textbook writes divides by `n - 1`. One sample of
taper, audible in the top bins.

## The attention scale goes on the query

Whisper multiplies the query by `head_dim ** -0.5` immediately after `q_proj`
and uses a softmax scale of 1. Algebraically that is the same as scaling the
scores, and it is not the same rounding. The reference says so itself, in a
comment: *"Scaling is susceptible to floating point arithmetics' imprecisions
which can lead to different results (this is dependent from model to model,
e.g. whisper is one such case)."*

whisper.cpp does something different again — the encoder leaves Q and K
unscaled and puts `1/sqrt(64)` inside the softmax, while the decoder multiplies
*both* by `64 ** -0.25`. Three spellings of one identity. Copy whichever
belongs to the reference you are matching.

## Cross-attention reads the encoder output raw

`encoder_attn_layer_norm` belongs to the decoder's own residual stream and
normalises the queries only. Normalising the keys and values with it as well is
an easy symmetry to assume, and is not what the reference does.

## 1,607 special tokens, and one that is not among them

`vocab.json` holds 50,258 entries against a vocabulary of 51,865. This
checkpoint ships `added_tokens.json`, so the 1,607 are read rather than derived
— whisper.cpp derives them, because its container has nowhere to put them, with
`num_languages = n_vocab - 51765 - 1`. Get that expression wrong and every
language and timestamp id is off by one: a transcript in the wrong language,
with nothing to indicate why. The arithmetic is checked once, in the tests,
against the file rather than against itself.

`<|endoftext|>` is *not* in `added_tokens.json`. It predates the multilingual
tokens and lives in `vocab.json`, declared special only by
`special_tokens_map.json`. A loader that reads only the former gets 1,607 of
1,608 specials right and leaves the one the decoder stops on looking like
ordinary text.

## Decoding: what is deliberately absent

The live pipeline runs `-nt`, greedy, at a fixed `language=zh`, on VAD-gated
utterances of a few seconds. That makes a large part of the reference dead
weight here. Each omission is a decision, listed so that re-adding one is also
a decision:

| absent | why |
| --- | --- |
| beam search, `best_of` | greedy at a fixed language; nothing samples |
| the temperature ladder and its retries | the fallbacks exist for long-form transcription |
| `no_speech_prob` gating | the VAD in front of the ASR is the gate, and a better one |
| DTW and token timestamps | `-nt`; every downstream stage takes plain text |
| the grammar engine | no grammar in this pipeline |
| `whisper_full_parallel` | one utterance at a time by design |
| chunking of long audio | the VAD gates to a single window |

`suppress_tokens` and `begin_suppress_tokens` are *not* absent. Skipping them
does not fail loudly — it produces a transcript that begins with a space, or an
empty one, on some utterances and not others.

## There is no CPU implementation

2.2 TFLOP for one encoder pass against scalar kernels that run at something
under 2 GFLOP/s. `--asr-device cpu` is refused at preflight rather than
accepted and then taking twenty minutes. See `docs/ARCHITECTURE.md`.

---

# Llama-2, `Taigi-Llama-2-Translator-13B`

A Llama-2 13 B fine-tune that translates between Mandarin, Han-script Taigi and
POJ. 363 tensors, 13,261,870,080 parameters, BF16 on disk.

## Geometry

| | |
| --- | --- |
| hidden size | 5120 |
| layers | 40 |
| attention heads | 40, head dim 128 |
| key/value heads | 40 — **no** grouped-query attention |
| intermediate size | 13824 |
| RMS norm epsilon | 1e-5 |
| RoPE theta | 10000 |
| vocabulary | 56,024 in the config, 56,020 in the tokenizer |
| tied embeddings | no — `lm_head` is its own 5120×56024 tensor |

The config is checked rather than trusted. `LlamaConfig::check` refuses an
architecture that is not `LlamaForCausalLM`, a hidden size that is not divisible
by the head count, and — for now — any checkpoint with fewer key/value heads
than query heads, because this engine has never run a GQA model and a silently
wrong head mapping is exactly the failure this project keeps trying to make
impossible.

## BF16 on a card with no bf16

Turing has fp16 tensor cores and no bf16 at all, so the weights are converted
once at load. F32 would be 53 GB, which is more than any card here has; f16 is
26.5 GB and fits. The conversion is `xabe-st`'s `tensor_f16`, and its behaviour
at the edges is a decision, not an accident: a value that would round to an
infinity is **refused** by tensor name and element index, while an underflow to
a subnormal or to zero is counted and logged. Saturating quietly is how you get
a model that loads and then speaks nonsense.

## Four vocabulary rows no token maps to

`config.json` says 56024 and `tokenizer.model` holds 56020 pieces. Both are
right: the extra four rows are real, allocated, trained parameters that the
tokenizer has no piece for. So the loader binds 56024 rows — anything else
fails the parameter count — and the tokenizer refuses to emit an id above
56019. That combination is faithful to the checkpoint *and* safe to sample from;
truncating the embedding would break the count, and letting the tokenizer reach
the extra rows would produce ids nothing can decode.

## RoPE pairs halves, not neighbours

🤗's implementation rotates element `i` against element `i + head_dim/2`. The
original paper, and most from-scratch implementations, pair `2i` with `2i+1`.
The two are a permutation apart and both "work" in the sense of training a
model, but a checkpoint is trained under one of them and reading it under the
other produces fluent nonsense rather than an error. This is the halves
convention because that is what the checkpoint was trained with.

## Where the attention scale goes, again

Whisper scales the *query* before the product and uses a softmax scale of 1.
Llama scales the *scores* after it. The algebra is identical and the rounding is
not, so each convention is copied where it belongs rather than unified. Whisper
is documented as being sensitive to the ordering; nothing says Llama is, and the
point of matching an oracle is that you do not have to find out.

## The newline is exempt from the repetition penalty

llama.cpp has a `penalize_nl` parameter that defaults to **false**, which means
almost nobody sets it and almost nobody knows it is there. With the newline
penalised, four of eight fixed prompts grew a trailing `。`; with it exempt,
seven of eight are character-identical to llama-server. The penalty itself is
llama.cpp's asymmetric form — divide a positive logit, multiply a negative one —
over the last 64 tokens at 1.1.

## The prompt template is exact

```
[TRANS]
{source}
[/TRANS]
[{target}]
```

with BOS prepended, stopping at `[/` or `\n[`. `target` is `POJ`, `HAN` or `HL`.
The template comes from the model card and the whitespace is load-bearing: this
is a fine-tune, not an instruction model, and it has no fallback behaviour for a
prompt shaped differently.

## The same checkpoint, in two containers

The 13 B translator is on this machine twice: as the 🤗 safetensors directory it
was published as, and as the f16 GGUF `llama-server` runs. `Translator::open`
reads either — a `.gguf` extension picks the GGUF reader, anything else is
treated as a checkpoint directory.

They are the same 363 tensors and the same 13,261,870,080 parameters, and the
weights are **bit-identical** — with two exceptions, one boring and one not.

The boring one: the safetensors is bf16 and is rounded to f16 on the way in,
while the GGUF is already f16. That costs nothing, because bf16 has 7 mantissa
bits against f16's 10, so the mantissa *widens* and no rounding happens at all.
Only the exponent range can bite, and `tensor_f16` refuses an overflow rather
than saturating.

### llama.cpp permutes the query and key projections

This is the one that matters, and it is invisible to every shape check.

Both conventions compute the same rotation and disagree about which two
elements of a head form a rotating pair. 🤗 pairs element `i` with
`i + head_dim/2` — the **halves** convention, which is what `xabe_dsp::rope`
implements. ggml pairs `2i` with `2i+1` — **interleaved**. Rather than carry
two rope kernels, llama.cpp's converter bakes the difference into the weights,
permuting the *rows* of `attn_q` and `attn_k` on the way into a GGUF.

`attn_v` is untouched, because no rotation is applied to values. That asymmetry
is the fingerprint. Measured on `taigi-translator-13b-f16.gguf`:

| tensor | differing elements, before un-permuting |
| --- | --- |
| `blk.0.attn_q.weight` | 25,779,757 of 26,214,400 |
| `blk.0.attn_k.weight` | 25,777,603 of 26,214,400 |
| `blk.0.attn_v.weight` | **0** |
| `blk.0.ffn_down.weight` | 0 |
| `token_embd.weight`, `output.weight` | 0 |

`xabe_llama::gguf::unpermute_rope` takes both to zero. So reading a GGUF Llama
without undoing this gives a model whose values, norms, feed-forward and
embedding are all exactly right and whose `q` and `k` are shuffled within every
head — shapes all correct, output fluent and wrong. It is the precise failure
this project's whole oracle discipline exists to catch, and no amount of shape
checking would have found it.

The permutation is undone at load, so one rope kernel serves both containers
and nothing downstream learns which one it came from.

### The GGUF names the four padding rows

`tokenizer.model` holds 56,020 pieces and the GGUF holds 56,024. The four extra
are spelled `[PAD56020]` through `[PAD56023]`: llama.cpp names the embedding's
padding rows so that its vocabulary and its tensor agree, where SentencePiece
simply does not mention them. That is independent confirmation of the four
unused rows recorded above, from a tool that had to solve the same problem.
Both readings are right about their own file, and neither can emit one — they
are `Unused`.

## There is no CPU implementation

`--translator-device cpu` is refused at preflight. See
`docs/ARCHITECTURE.md` — 40 layers of 13 B at under 2 GFLOP/s is not a slow
option, and the f32 weights would not fit anyway.

---

# Llama-3, `Llama-Breeze2-8B-Instruct-text-only`

The chat model. Loaded but not run: `xabe-gguf` reads the container and
`xabe-llama` binds all 292 tensors, and nothing in this workspace does
arithmetic with them. Serving it is still llama.cpp's job.

## Geometry

| | |
| --- | --- |
| hidden size | 4096 |
| layers | 32 |
| attention heads | 32, head dim 128 |
| key/value heads | **8** — grouped-query, four query heads per kv head |
| intermediate size | 14336 |
| RMS norm epsilon | 1e-5 |
| RoPE theta | **500000** |
| context length | 131072 |
| vocabulary | 128,256 |
| tied embeddings | no — `output.weight` is its own tensor |
| parameters | 8,030,261,312 |
| storage | 226 tensors f16, 66 f32 |

## Three things that differ from the Llama-2 translator

**Grouped-query attention.** 32 query heads share 8 key-value heads, so `k` and
`v` are `[1024, 4096]` where `q` and `o` are `[4096, 4096]`. This is what made
`xabe-llama` stop refusing grouped-query outright. The refusal moved rather than
disappearing: a *shape* is a fact about the file and every grouped-query file
binds fine, so `LlamaConfig::check` accepts it and
`LlamaConfig::refuse_grouped_query` is what an engine without the head mapping
calls at open. `xabe-translate` calls it. The old arrangement made an 8 B model
that binds cleanly unreadable for no reason a shape could justify.

**RoPE theta is 500000, not 10000.** Llama-3 stretched the base by fifty times
to reach a 128k context. A loader that defaulted this would produce a model
fluent for one sentence and drifting after that — which is exactly the class of
bug that has no shape check to catch it.

**`rope_freqs.weight` has no safetensors counterpart.** 64 f32, one per rotating
pair of a 128-wide head: Llama-3.1's per-frequency rope scaling. It is bound
rather than skipped, because a schema that binds 291 of 292 tensors is not a
proof that the geometry is understood, and the one left over is precisely the
one nothing else would have told you about.

## Quantized GGUFs load, and that is not the same as running quantized

`xabe-gguf` decodes nine block formats — `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
`Q8_0`, and the K-quants `Q2_K` through `Q6_K` — unpacking each to f32 on read.
So a `Q4_K_M` checkpoint opens and binds exactly like an f16 one, and nothing
above the container knows the difference.

**What that buys is disk and load bandwidth, not memory.** The weights are
unpacked on the way in, so a 4-bit 13 B is a 7 GB file and still 26.5 GB of f16
once it is on the card. Running *quantized* — keeping blocks packed in VRAM and
unpacking inside the matmul — is a different piece of work: every kernel would
have to learn every block layout, and on this card the tensor-core path that
makes it worthwhile is the int8 one, not f16. `llmxabe` has that path for Q8_0;
this workspace does not, and adding it is not a loader change.

Two limits remain, both by decision. The `IQ*` and `TQ*` families are refused —
importance-weighted and ternary formats this project has no file in. `Q8_K` is
refused too, for a different reason: it is an intermediate used *while*
quantizing and never appears in a stored tensor, so a file containing one is
malformed rather than unsupported.

### The dequantizers are transcribed, then checked

Each format is read off `gguf-py/gguf/quants.py` — the same code that writes
these files. The reference expresses a format as a reshape-and-shift over whole
blocks; unpacking one element at a time means deriving the element *ordering*
by hand, and that is the only hard part.

Q4_0 shows the shape of the trap. The reference does

```
qs.reshape((n, -1, 1, 16)) >> [0, 4]  ->  (n, 1, 2, 16)  ->  (n, 32)
```

so a block's first sixteen values are the **low** nibbles of bytes 0..16 and
its last sixteen are the **high** nibbles of the same bytes. The intuitive
reading — low then high of byte 0, low then high of byte 1 — is wrong, and
produces a tensor that is a *permutation* of the right one: same values, same
histogram, no correlation. Every format here has a trap of that shape, which is
why none of them is reasoned about. `tools/oracle/capture_quants.py` captures
packed bytes and the f32 the reference unpacks them to, and the test asserts
**exact** equality on all ten formats.

The corpus is pseudo-random encodings rather than a quantizer's output, for two
reasons. Python `gguf` can only quantize the five legacy formats — the
K-quants exist in C alone — so a round-trip corpus would have covered half the
table. And random encodings reach every nibble, every packed six-bit scale and
every high-bit mask, where a quantizer only ever emits the well-conditioned
subset. A nibble-order mistake survives the second and dies against the first.

## GGUF stores shapes transposed

The single easiest thing to get wrong. GGUF writes dimensions fastest-varying
first, so `blk.0.attn_k.weight` is `[4096, 1024]` on disk and `[1024, 4096]` as
a row-major matrix. `TensorInfo::dims` is what the file said and
`TensorInfo::shape()` is the row-major reading; the binding compares against
`shape()`.

Getting this wrong has the worst possible failure shape, which is why it has a
test of its own: every square projection agrees under either reading, so `q`,
`o` and both norms bind correctly and only `k`, `v` and the feed-forward come
out transposed. A model that is right in most places is harder to diagnose than
one that is wrong everywhere.

## The tokenizer is GPT-2 byte-level BPE

`tokenizer.ggml.model = gpt2`, `pre = llama-bpe`, 128,256 tokens and 280,147
merges, all carried inside the GGUF rather than in files beside it. So the
closer precedent here is `xabe-whisper`'s byte-level BPE, not `xabe-llama`'s
SentencePiece — the translator and the chat model share an architecture family
and not a tokenizer. It is **not written yet**: the metadata is read and the
arrays are reachable, and nothing turns them into a tokenizer.
