# Command surface

**Planned. `xabe-tts` has no binary yet.** This is the shape it is being built
toward, recorded so the flags are designed once rather than accreted.

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
| `--device` | `XABE_TTS_DEVICE` | `auto` | `cpu`, `cuda:N`, `auto` |
| `--log-level` | `RUST_LOG` | `info` | `info`, `debug`, `trace` |

`--seed` defaults to a fixed value rather than to entropy. Reproducible by
default is the right posture for something whose output is hard to check.

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
