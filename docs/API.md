# Library API

**Nothing here is stable, and most of it does not exist yet.** This records the
intended surface so the crates are built toward it.

## `xabe-st` — implemented

```rust
use xabe_st::{StFile, StError};

let f = StFile::open("model.safetensors")?;

f.len();                                    // tensor count
f.tensors();                                // (&str, &TensorInfo), sorted
f.info("decoder.conv_post.weight");         // Option<&TensorInfo>
f.tensor("decoder.conv_post.weight")?;      // &[f32], borrowed from the mapping
f.tensor_shaped("decoder.conv_post.weight", &[1, 32, 7])?;
```

`tensor_shaped` is the one weight loading should use: it turns a
wrong-geometry checkpoint into a named error at load time instead of wrong audio
at synthesis time.

Every accessor borrows from the mapping. Opening a 139 MB checkpoint is one
`mmap` and no allocation.

## `xabe-vits` — planned

```rust
let cfg = VitsConfig::from_json(path)?;     // rejects unsupported geometry
let weights = VitsWeights::load(&file, &cfg)?;   // every tensor shape-checked
```

`load` reads only the inference subset; `posterior_encoder` is skipped.

## `xabe-tts`

```rust
let model = Synthesizer::open(checkpoint_dir)?;          // CPU reference
let model = GpuModel::open(checkpoint_dir, ordinal)?;    // CUDA
let audio: Vec<f32> = model.synthesize("lí hó, kin-á-ji̍t thinn-khì chin hó.", seed)?;
model.config().sampling_rate;   // 16_000
```

Writing the result is `xabe_audio::write_wav(&mut w, &audio, rate)`, not a
method on the audio. The samples are a plain `Vec<f32>`, and a container that
knew how to serialise itself would be the synthesiser owning a WAV writer — the
dependency edge that moving `wav.rs` into `xabe-audio` removed.

Input is POJ with `ⁿ` written `nn` — see [MODEL.md](MODEL.md). The synthesiser
does not romanise, translate, or read Han characters.

`Seed` is explicit rather than defaulted: the duration predictor and the prior
both sample, so a caller that wants reproducible output must be able to say so,
and a caller that does not must be made to notice.

## Errors

One `thiserror` enum per crate, in that crate's `error.rs`. No `anyhow`, and no
`Box<dyn Error>` crossing a crate boundary. Every variant is documented with the
condition it prevents.
