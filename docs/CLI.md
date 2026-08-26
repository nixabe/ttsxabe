# Command surface

The workspace builds two binaries: `xabe-engine`, the engine itself, and
`xabe-tts-bench`, for timing the synthesiser.

## `xabe-engine`

Every stage is satisfied one of two ways, and the symmetry is the whole design:

| | |
| --- | --- |
| `--<stage>-model PATH` | run the stage **in this process** |
| `--<stage>-url URL` | delegate the stage to **another process** over HTTP |

The two are alternatives per stage, and either satisfies that stage's
dependency. Nothing downstream can tell which was used, so the same binary is a
monolith, a single-stage worker, or anything between. The six-port topology the
Python pipeline runs today is one configuration of these flags.

Stages are `asr`, `vad`, `tts` and `translator`. `llm` is **URL-only** — it
stays in llama.cpp by decision, so there is no `--llm-model`.

```sh
# everything in one process
xabe-engine --serve 127.0.0.1:8000 \
            --asr-model    models/asr/breeze-asr-26   --asr-device 0 \
            --vad-model    models/vad/silero-v5.1.2.safetensors \
            --tts-model    models/tts/mms-tts-nan     --tts-device 1 \
            --llm-url      http://127.0.0.1:8082

# split across processes and GPUs, as run.sh does today
xabe-engine --serve 127.0.0.1:8080 --asr-model models/asr/breeze-asr-26 \
            --vad-model models/vad/silero-v5.1.2.safetensors --asr-device 0
xabe-engine --serve 127.0.0.1:8100 --tts-model models/tts/mms-tts-nan --tts-device 1
xabe-engine --serve 127.0.0.1:8000 --asr-url http://127.0.0.1:8080 \
            --tts-url http://127.0.0.1:8100 --llm-url http://127.0.0.1:8082

# one stage, one shot, no server
xabe-engine --tts-model models/tts/mms-tts-nan --text "lí hó" --out hello.wav
xabe-engine --asr-model models/asr/breeze-asr-26 --in clip.wav
xabe-engine --vad-model models/vad/silero.safetensors --in clip.wav   # segments
```

### Flags

| flag | env | default | |
| --- | --- | --- | --- |
| `--serve` | `XABE_SERVE` | — | listen address; without it the run is one-shot |
| `--asr-model` | `XABE_ASR_MODEL` | — | speech-to-text checkpoint directory |
| `--asr-url` | `XABE_ASR_URL` | — | delegate speech-to-text |
| `--asr-device` | `XABE_ASR_DEVICE` | `0` | a CUDA device ordinal; **not** `cpu` |
| `--vad-model` | `XABE_VAD_MODEL` | — | voice-activity checkpoint |
| `--vad-url` | `XABE_VAD_URL` | — | delegate voice-activity detection |
| `--vad-device` | `XABE_VAD_DEVICE` | `cpu` | `cpu` only; see below |
| `--tts-model` | `XABE_TTS_MODEL` | — | directory, or the safetensors file itself |
| `--tts-url` | `XABE_TTS_URL` | — | delegate text-to-speech |
| `--tts-device` | `XABE_TTS_DEVICE` | `0` | `cpu`, or a CUDA device ordinal |
| `--translator-model` | `XABE_TRANSLATOR_MODEL` | — | Mandarin-to-Taigi checkpoint |
| `--translator-url` | `XABE_TRANSLATOR_URL` | — | delegate translation |
| `--translator-device` | `XABE_TRANSLATOR_DEVICE` | `0` | `cpu`, or a CUDA device ordinal |
| `--llm-url` | `XABE_LLM_URL` | — | chat model; there is no `--llm-model` |
| `--direct-taigi` | `XABE_DIRECT_TAIGI` | off | chat model answers in Taigi Han itself |
| `--in` | — | — | one-shot input WAV; `-` reads stdin |
| `--text` | — | — | one-shot input text; `-` reads stdin |
| `--out` | — | — | one-shot output; `-` writes stdout |
| `--tts-engine` | `XABE_TTS_ENGINES` | — | extra engines as `name=url`, repeatable |
| `--tts-default` | `XABE_TTS_DEFAULT` | `mms` | which engine the page selects |
| `--translator-target` | `XABE_TRANSLATOR_TARGET` | `POJ` | `POJ`, `HAN` or `HL` |
| `--asr-lang` | `XABE_ASR_LANG` | `zh` | never `en`; see below |
| `--person` / `--bot` | `XABE_PERSON` / `XABE_BOT` | 使用者 / 小助理 | names in the transcript |
| `--temperature` | `XABE_TEMPERATURE` | 0.3 | reply sampling |
| `--max-tokens` | `XABE_MAX_TOKENS` | 160 | reply length cap |
| `--history-turns` | `XABE_HISTORY_TURNS` | 6 | turns kept in the prompt |
| `--min-chunk` | `XABE_MIN_CHUNK` | 6 | characters before a later chunk is spoken |
| `--first-chunk` | `XABE_FIRST_CHUNK` | 4 | characters before the *first* chunk is spoken |
| `--prompt-file` | `XABE_PROMPT_FILE` | — | replaces the built-in system prompt |
| `--config` | `XABE_TTS_CONFIG` | next to the model | TTS `config.json` |
| `--seed` | `XABE_SEED` | 0 | duration and prior sampling |
| `--noise-scale` | | 0.667 | prior temperature |
| `--noise-scale-duration` | | 0.8 | duration temperature |
| `--speaking-rate` | | 1.0 | duration multiplier |
| `--log-level` | `RUST_LOG` | `info` | `info`, `debug`, `trace` |

`--seed` defaults to a fixed value rather than to entropy. Reproducible by
default is the right posture for something whose output is hard to check.

`--tts-model` accepts either the model directory or the `model.safetensors`
inside it. Both spellings are in use — the consolidated tree names directories,
while the older flag and every test named the file — and the directory is
recoverable from the file, so refusing one of them would break working commands
to no purpose.

`--<stage>-device` exists per stage because the pipeline deliberately spreads
stages across cards: putting the ASR and the TTS on one GPU makes the next
turn's prefill queue behind the last turn's synthesis.

`--vad-device` takes only `cpu`, and defaults to it. The VAD is 15 tensors and
about a millisecond of work per clip, so the transfer would cost more than the
arithmetic. `--vad-device 0` is refused by name rather than accepted and
silently ignored, which is what the flag surface did first — and what made the
startup log announce `device=cuda:0` for a stage running on the CPU.

### What the preflight refuses

Every rejection names the flag that caused it and happens *before* any
checkpoint is opened, so a mistyped topology costs a millisecond rather than
six gigabytes of reads. The order is: resolve the stages, decide what the run
does, refuse the stages that are not built yet, then work.

| | |
| --- | --- |
| `--tts-model` and `--tts-url` together | alternatives; give one |
| `--asr-device` with `--asr-url` | a device applies only to a local stage |
| `--asr-device` with no ASR stage at all | the same typo, one flag earlier |
| `--vad-*` with `--serve` and no ASR | served, a VAD is only ever a gate |
| `--vad-device 0` | the VAD has no CUDA implementation |
| `--asr-device cpu` | the ASR has no CPU implementation |
| no stage at all | names the four flag pairs that would give one |
| stages but no `--serve`, `--in` or `--text` | a serve command with `--serve` forgotten |
| `--in` and `--text` together | alternatives; give one |
| `--serve` with `--in` or `--text` | a server is not a one-shot run |
| `--text` with no TTS stage | names `--tts-model` and `--tts-url` |
| `--text` with no `--out` | nowhere to put the WAV |
| `--in` with neither ASR nor VAD | nothing here reads audio |

`--vad-model --in clip.wav` is *accepted*: over a file the VAD is a tool that
prints segments, and only when served is it a gate that needs something to gate.

### Stages that are not built yet

The flag surface is complete ahead of the stages behind it. `--translator-model`
parses and validates, then fails with the phase that builds it. That ordering
is deliberate: the topology is the part worth settling first, and a flag that
parses and then silently does nothing is worse than one that says what it is
waiting on.

Two stages refuse a device rather than accepting it and doing something else.
The VAD has no CUDA implementation and is not going to: 15 tensors and a
millisecond of work, where the transfer would cost more than the arithmetic.
The ASR has no CPU implementation for the mirror-image reason: one 30-second
window is 2.2 TFLOP through the encoder, which the scalar kernels would take
twenty minutes to do. Both would otherwise be a flag that is accepted and then
quietly means something else - which is exactly how the startup log once came
to announce `device=cuda:0` for a stage running on the CPU.

| stage | status |
| --- | --- |
| `--tts-model` | works |
| `--serve` | works |
| `--asr-url`, `--tts-url`, `--translator-url`, `--llm-url` | work |
| `--vad-model` | works, CPU only |
| `--vad-url` | never: the VAD is a millisecond of CPU, so a round trip would cost more than the work |
| `--asr-model` | works, CUDA only |
| `--translator-model` | phase 5a |

## What `--serve` publishes

The endpoints are the ones the Python services already publish, not new ones.
That is what makes the migration incremental rather than a flag day: an engine
process is a drop-in for `whisper-server`, for the TTS daemon, or for the
gateway, so each stage can be swapped in behind the existing system and A/B'd
against the service it replaces, one at a time.

| route | replaces | published when |
| --- | --- | --- |
| `GET /health` | all three | always |
| `POST /inference` | `whisper-server` | there is an ASR stage |
| `POST /tts`, `POST /tts_stream` | `taigi_tts_daemon.py` | there is a TTS stage |
| `GET /`, `GET /api/config`, `WS /ws` | `gateway.py` | ASR **and** LLM **and** TTS |

A process publishes only the routes for stages it owns. Asking a TTS worker for
`/inference` is a 503 naming the flag that would give it one. The page is
published only by a process that can actually hold a conversation — a TTS worker
answering `GET /` with a chat UI that cannot hear anything would be a worse
failure than a 404.

`GET /api/config` also carries the turn-taking constants, so the page does not
keep its own copy of numbers that were tuned against real speech. They are
defined and unit-tested in `xabe-serve::turntaking`; see `docs/MODEL.md` for
what each one is a fix for.

## Turn-taking

Every constant is a fix for an observed failure, not a round number:

| | | |
| --- | --- | --- |
| `vad_start` | 0.035 | 0.012 sat at room-noise level, so the mic fired on silence and the ASR hallucinated |
| `vad_stop` | 0.018 | hysteresis; one threshold chops a turn up at every unvoiced consonant |
| `onset_frames` | 3 | a click or a door cannot open a turn |
| `silence_ms` | 700 | a pause only *arms* the end of turn |
| `grace_ms` | 900 | ...and this much more silence commits it |
| `voiced_ms` | 250 | enough loud audio to be worth transcribing at all |
| `min_ms` / `max_ms` | 500 / 20000 | shortest turn worth sending; backstop for a stuck mic |

Committing on `silence_ms` alone cut people off at natural mid-sentence pauses,
sending half a question. Arming and then committing means finishing a turn costs
1600 ms of trailing silence, while a thinking pause inside one costs nothing.

## What it costs

| device | 2.6 s of audio |
| --- | --- |
| `--tts-device 0` | ~48 ms, 54x realtime |
| `--tts-device cpu` | ~120 s, 0.02x realtime |

The CPU path is the scalar reference and is not meant to be used: it exists to
be read and to be correct. See [BENCHMARKS.md](BENCHMARKS.md) for the
comparison against PyTorch and where the GPU time goes.

## Verifying the output

Numerical agreement with the oracle says the arithmetic is right; it does not
say the file is speech. The check that does is an ASR round trip - synthesise,
then transcribe with a model that was never involved in producing it:

```sh
xabe-engine --tts-model models/tts/mms-tts-nan \
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

## Retracted: "no serve subcommand"

This file used to say there would be no `serve` subcommand, on the grounds that
the pipeline already had an HTTP surface and duplicating it would mean owning a
second one.

That reasoning was sound and the premise changed. The engine now *is* the
pipeline: with ASR, VAD, TTS and turn-taking folded in, the Python gateway is
not a surface being duplicated but one being replaced, and the alternative to
`--serve` is keeping a seventh process alive to do nothing but route between
stages that live in this binary. `--serve` is also what makes the migration
incremental — an engine process that speaks the existing wire protocols can be
swapped in behind the existing gateway one stage at a time and A/B'd against
the service it replaces.

The flag it takes is an address, not a subcommand, which keeps the flat `Args`
struct the house style asks for.
