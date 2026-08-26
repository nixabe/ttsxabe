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
