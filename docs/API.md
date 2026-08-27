# Library API

**Nothing here is stable.** It used to say "and most of it does not exist yet",
which was true when the workspace was one model; every crate below is now
implemented, and this is a record of the surface rather than a target to build
toward. The one thing that has not changed is that none of it is a promise.

## `xabe-st`

```rust
use xabe_st::{StFile, StError};

let f = StFile::open("model.safetensors")?;

f.len();                                    // tensor count
f.tensors();                                // (&str, &TensorInfo), sorted
f.info("decoder.conv_post.weight");         // Option<&TensorInfo>
f.tensor("decoder.conv_post.weight")?;      // &[f32], borrowed from the mapping
f.tensor_shaped("decoder.conv_post.weight", &[1, 32, 7])?;
f.tensor_f16("model.layers.0.self_attn.q_proj.weight")?;   // Vec<u16>
```

`tensor_f16` takes F32, F16 or BF16 and always returns f16: the source copied
bit for bit when it is already f16, rounded to nearest even otherwise, and
**refused by name and index** when a value would round to an infinity.

A checkpoint in shards is `StSet`, which opens a directory, reads
`model.safetensors.index.json` when there is one, falls back to the single-file
name when there is not, and offers the same accessors across the whole set.

`tensor_shaped` is the one weight loading should use: it turns a
wrong-geometry checkpoint into a named error at load time instead of wrong audio
at synthesis time.

Every accessor borrows from the mapping. Opening a 139 MB checkpoint is one
`mmap` and no allocation.

## `xabe-gguf`

```rust
use xabe_gguf::GgufFile;

let f = GgufFile::open("models/llm/Llama-Breeze2-8B...f16.gguf")?;

f.len();                                  // tensor count
f.tensors();                              // &[TensorInfo], in directory order
f.info("blk.0.attn_k.weight");            // Option<&TensorInfo>
f.get_u32("llama.block_count");           // metadata, typed
f.get_strings("tokenizer.ggml.tokens");   // the vocabulary lives in the file
f.tensor_bytes("output_norm.weight")?;    // borrowed from the mapping
f.tensor_f32("output_norm.weight")?;      // Vec<f32>, widened if needed
f.tensor_f16("blk.0.attn_q.weight")?;     // Vec<u16>, rounded if needed
```

The same contract as `xabe-st` over a different container, with one thing that
has no safetensors equivalent: `TensorInfo::dims` is what the file stores,
fastest-varying first, and `TensorInfo::shape()` is the row-major reading. Bind
against `shape()`. Binding against `dims` agrees for every square matrix and
silently transposes the rest.

## The geometry crates

`xabe-vits`, `xabe-whisper` and `xabe-llama` all have the same shape: a config
that refuses geometry it cannot run, and weights that bind every tensor by name
and check every shape. `xabe-vad` is the same idea with no config to read — 15
tensors and a fixed architecture — so it appears below with its forward pass.

```rust
let cfg = VitsConfig::from_json_path(path)?;     // rejects unsupported geometry
let weights = VitsWeights::load(&file, &cfg)?;   // every tensor shape-checked

let cfg = WhisperConfig::from_dir(dir)?;
let weights = WhisperWeights::load(&set, &cfg)?; // 1,259 tensors

let cfg = LlamaConfig::from_dir(dir)?;           // refuses ragged heads
let weights = LlamaWeights::load(&set, &cfg)?;   // 363 tensors, no bytes read

let cfg = LlamaConfig::from_gguf(&gguf)?;        // the same geometry, from metadata
let weights = LlamaWeights::from_gguf(&gguf, &cfg)?;  // 292 tensors, no bytes read
cfg.refuse_grouped_query()?;                     // what an engine calls, not the schema
```

`VitsWeights::load` reads only the inference subset; `posterior_encoder` is
skipped. `LlamaWeights::load` reads *nothing* — it binds names, shapes and
dtypes, and the 26.5 GB moves only when a stage asks for it.

Both tokenizers are constructed from the checkpoint directory and answer
`encode` / `decode`:

```rust
let tok = xabe_whisper::Tokenizer::from_dir(dir)?;   // byte-level BPE
let tok = xabe_llama::Tokenizer::from_dir(dir)?;     // SentencePiece
```

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

## `xabe-vad`

```rust
let mut vad = xabe_vad::open("models/vad/silero-v5.1.2.safetensors")?;
let probs = vad.probabilities(&samples);          // one scalar per 512 samples
let spans = xabe_vad::segments(&probs, SegmentParams::default());
```

`probabilities` takes `&mut self` because the LSTM carries state across frames,
which is the whole reason the detector is a struct and not a function.
`segments` takes probabilities rather than audio, so the same hysteresis runs
over a browser's VAD or over this one without knowing the difference.

## `xabe-asr` and `xabe-translate` — CUDA only

Both open a checkpoint directory onto a device ordinal and refuse the CPU.

```rust
let asr = AsrModel::open(dir, ordinal)?;
let text = asr.transcribe(&samples, "zh")?;       // 16 kHz mono f32

let tr = Translator::open(dir, ordinal)?;
let taigi = tr.translate("今天天氣很好", "POJ",
                         256, Translator::REPEAT_PENALTY)?;
```

`transcribe` and `translate` are the whole surface anyone should need. Beneath
them, `encode` / `decode` / `generate` are public because the oracle tests
compare per layer, and `encode_tapped` / `decode_tapped` exist for exactly that
— they are not a streaming API and should not be mistaken for one.

## Errors

One `thiserror` enum per crate, in that crate's `error.rs`. No `anyhow`, and no
`Box<dyn Error>` crossing a crate boundary. Every variant is documented with the
condition it prevents.
