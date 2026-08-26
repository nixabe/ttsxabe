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
