# Command surface

`xabe-tts` builds a binary, and a second one, `xabe-tts-bench`, for timing.

## `xabe-tts`

```sh
xabe-tts --model model.safetensors --config config.json \
         --text "lí hó, kin-á-ji̍t thinn-khì chin hó." \
         --out hello.wav
```

| flag | env | default | |
| --- | --- | --- | --- |
| `--model` | `XABE_TTS_MODEL` | — | safetensors checkpoint |
| `--config` | `XABE_TTS_CONFIG` | next to the model | `config.json` |
| `--text` | — | — | POJ input; `-` reads stdin |
| `--out` | — | — | output WAV; `-` writes stdout |
| `--seed` | `XABE_TTS_SEED` | 0 | duration and prior sampling |
| `--noise-scale` | | 0.667 | prior temperature |
| `--noise-scale-duration` | | 0.8 | duration temperature |
| `--speaking-rate` | | 1.0 | duration multiplier |
| `--device` | `XABE_TTS_DEVICE` | `0` | `cpu`, or a CUDA device ordinal |
| `--log-level` | `RUST_LOG` | `info` | `info`, `debug`, `trace` |

`--seed` defaults to a fixed value rather than to entropy. Reproducible by
default is the right posture for something whose output is hard to check.

## What it costs

| device | 2.6 s of audio |
| --- | --- |
| `--device 0` | ~48 ms, 54x realtime |
| `--device cpu` | ~120 s, 0.02x realtime |

The CPU path is the scalar reference and is not meant to be used: it exists to
be read and to be correct. See [BENCHMARKS.md](BENCHMARKS.md) for the
comparison against PyTorch and where the GPU time goes.

## Verifying the output

Numerical agreement with the oracle says the arithmetic is right; it does not
say the file is speech. The check that does is an ASR round trip - synthesise,
then transcribe with a model that was never involved in producing it:

```sh
xabe-tts --model .../model.safetensors \
         --text "lí hó, kin-á-ji̍t thinn-khì chin hó." --out hello.wav
curl -s -F file=@hello.wav -F language=zh http://127.0.0.1:8080/inference
# {"text":"你好 今天天氣很好\n"}
```

That is the meaning of the POJ input, recovered from the audio by
Breeze-ASR-26. It is the only test here that can tell correct audio from
plausible audio, which matters when the language is one you cannot judge by ear.

## Conventions borrowed from `llmxabe`

- Every flag has an `env` twin, so a container needs no argv rewriting.
- Doc comments on the `Args` struct *are* the `--help` text; there is no second
  copy to drift.
- Three log levels only. Tool output a user is meant to read is `info!`, because
  INFO is the level that appears by default. INFO/DEBUG/TRACE go to stdout,
  WARN/ERROR to stderr, so `--out -` stays pipeable.
- `main() -> ExitCode` with a numbered preflight: each startup stage reports its
  own failure and returns, rather than unwinding a `Result` chain that loses
  which stage broke.

## Not planned

No `serve` subcommand. The pipeline this feeds already has an HTTP surface, and
duplicating it here would mean owning a second one. If that changes it will be
because a measurement asked for it.
