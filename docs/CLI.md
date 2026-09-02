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

Stages are `asr`, `vad`, `tts`, `translator` and `llm`.

**`llm` was URL-only, and is not any more.** The plan said the chat model stays
in llama.cpp by decision and there is no `--llm-model`; that is now retracted in
full, in two steps that are worth keeping apart because the first shipped long
before the second.

The loader came first: `xabe-gguf` reads the GGUF container and `xabe-llama`
binds all 292 tensors of `Llama-Breeze2-8B-Instruct-text-only.f16.gguf` against
its own metadata. That is the half that proves the geometry is understood and
the half that costs nothing to keep if the arithmetic never follows.

The arithmetic followed. `--llm-model` runs the weights here, and it is the same
symmetry as every other stage — `--llm-url` still delegates to llama-server and
nothing downstream can tell which was used. What changed the decision was not
that the forward pass got easier but that delegating the last stage kept a
second runtime, a second copy of the weights and a second GPU allocation alive
for it. Measured against the llama-server it replaces: 124 of 125 token
decisions identical, the one disagreement at the tightest margin in the corpus.
See `docs/ORACLE.md`.

Two limits are real and are not going away soon:

- **GGUF only.** Every other stage takes a 🤗 directory; this one does not,
  because this model is published as a GGUF and its vocabulary lives inside the
  file. A directory would load 16 GB of weights and then have nothing to
  tokenize with, so it is refused at open.
- **`--llm-device cpu` is refused.** 8 B is 16 GFLOP a token against scalar
  kernels that manage under 2 GFLOP/s. Not a slow option, a fictional one.

A **quantized** GGUF is accepted here and by `--translator-model`, and since
`Operand::Q` it costs what the file costs rather than what the file unpacks to.
That is the difference between one stage per card and the whole pipeline on
one; `docs/BENCHMARKS.md` has the residency table and `docs/KERNELS.md` the
kernel. Nothing about the flag changes - the same path takes an f16 or a
`Q4_K_M` file and the engine reads the format out of the container.

```sh
# everything in one process
xabe-engine --serve 127.0.0.1:8000 \
            --asr-model    models/asr/breeze-asr-26   --asr-device 0 \
            --vad-model    models/vad/silero-v5.1.2.safetensors \
            --tts-model    models/tts/mms-tts-nan     --tts-device 1 \
            --llm-model    models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
            --llm-device   0

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
| `--translator-model` | `XABE_TRANSLATOR_MODEL` | — | Mandarin-to-Taigi checkpoint: a 🤗 directory or a `.gguf` file |
| `--translator-url` | `XABE_TRANSLATOR_URL` | — | delegate translation |
| `--translator-device` | `XABE_TRANSLATOR_DEVICE` | `0` | `cpu`, or a CUDA device ordinal |
| `--translate-ahead` | `XABE_TRANSLATE_AHEAD` | by device | `0` translates clauses in step with the synthesiser; `1` translates a turn's first clause alone and every later one as it arrives, decoded together. Defaults to `1` when the translator and the synthesiser are on different cards; see docs/BENCHMARKS.md |
| `--llm-model` | `XABE_LLM_MODEL` | — | chat model, as a `.gguf`; GGUF only, see above |
| `--llm-url` | `XABE_LLM_URL` | — | delegate the chat model to a llama-server |
| `--llm-device` | `XABE_LLM_DEVICE` | `0` | a CUDA device ordinal; `cpu` is refused |
| `--direct-taigi` | `XABE_DIRECT_TAIGI` | off | chat model answers in Taigi Han itself |
| `--in` | — | — | one-shot input WAV; `-` reads stdin |
| `--text` | — | — | one-shot input text; `-` reads stdin |
| `--out` | — | — | one-shot output; `-` writes stdout |
| `--tts-engine` | `XABE_TTS_ENGINES` | — | engines as `name=url` or `name=path`, repeatable; on its own it *is* the TTS stage |
| `--cosy-voice` | `XABE_COSY_VOICE` | `<checkpoint>/voices/taigi-ref.safetensors` | speaker bundle for a local CosyVoice |
| `--cosy-instruct` | `XABE_COSY_INSTRUCT` | a Taigi instruction | what a local CosyVoice is told; must end on `<|endofprompt|>` |
| `--taco-sigma` | `XABE_TACO_SIGMA` | `tacotron2.json`'s `0.666` | WaveGlow's noise scale for a local Tacotron2 |
| `--tts-default` | `XABE_TTS_DEFAULT` | `mms` | which engine the page selects |
| `--translator-target` | `XABE_TRANSLATOR_TARGET` | `POJ` | `POJ`, `HAN` or `HL` |
| `--asr-lang` | `XABE_ASR_LANG` | `zh` | never `en`; see below |
| `--person` / `--bot` | `XABE_PERSON` / `XABE_BOT` | 使用者 / 小助理 | names in the transcript |
| `--temperature` | `XABE_TEMPERATURE` | 0.3 | reply sampling |
| `--max-tokens` | `XABE_MAX_TOKENS` | 160 | reply length cap |
| `--history-turns` | `XABE_HISTORY_TURNS` | 6 | turns kept in the prompt |
| `--min-chunk` | `XABE_MIN_CHUNK` | 6 | characters before a later chunk is spoken |
| `--first-chunk` | `XABE_FIRST_CHUNK` | 4 | characters before the *first* chunk is spoken |
| `--system-prompt` | `XABE_SYSTEM_PROMPT` | — | replaces the built-in system prompt, inline |
| `--prompt-file` | `XABE_PROMPT_FILE` | — | the same, read from a file |
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

### `--tts-engine` takes a URL or a directory

A value beginning `http://` or `https://` is another process. Anything else is
a directory, opened **in this one**, and which model it holds is read off the
filenames:

| It holds | It is opened as |
| --- | --- |
| `llm.safetensors`, `flow.safetensors`, `hift.safetensors` | CosyVoice3 |
| `tacotron2.safetensors`, `waveglow.safetensors`, `tacotron2.json` | Tacotron2 + WaveGlow |
| `best_model.pth`, `config.json` | Coqui VITS |
| anything else | 🤗 VITS |

All of a set, not one of it — a half-converted directory would otherwise be
opened as the model it is half of and fail deep inside a weight schema instead
of here, where the path is still in hand.

`--tts-model` sniffs the same way, so it takes any of them. Which one it is, is
a property of the directory rather than of a second flag.

### A Coqui VITS engine reads IPA, and the engine transliterates for it

The last two rows are both VITS and run the same forward pass, but they do not
eat the same thing: `mms-tts-nan` was trained on POJ and
`neurlang/coqui-vits-suisiann-minnan-hokkien` on IPA. The engine converts, with
`xabe-taigi`, on both the one-shot path and the conversation path — so a Coqui
engine wants `POJ`, the same as mms:

```sh
xabe-engine --serve 127.0.0.1:8000 \
            --tts-model  models/tts/mms-tts-nan            --tts-device 2 \
            --tts-engine suisiann=models/tts/coqui-vits-suisiann \
            --tts-script suisiann=POJ
```

and the one-shot path takes romanisation directly:

```sh
xabe-engine --tts-model models/tts/coqui-vits-suisiann --tts-device 0 \
            --text "Lí hó, guá sī Tâi-oân-lâng." --out hello.wav
```

That is a transliteration and not a pronunciation guess: the translator has
already decided how every word is read, so the conversion is a spelling table.
`--text` also accepts IPA and passes it through unchanged, since the two are
trivially distinguishable — Chao tone letters never occur in romanisation.

Handing it **Han** is still not silently wrong: every character is outside both
the symbol table and the transliterator, so it refuses with "contains no
symbols this model can speak". `tools/phonemize_pygoruut.py` is what converts
Han, and `docs/MODEL.md` says why that stayed a tool rather than becoming a
crate.

It also stands alone. `--tts-model` fills one unnamed slot and `--tts-engine`
fills named ones, but both are a synthesiser running in this process, so a run
given only the latter has a TTS stage and `--tts-device` says which card its
local engines land on. That is how one synthesiser is served without loading
the rest:

```sh
# tacotron2 and nothing else
xabe-engine --serve 127.0.0.1:8100 \
            --tts-engine  taco=models/tts/tacotron2-nan --tts-device 1 \
            --tts-default taco
```

Only `--tts-model` and `--tts-url` fill the slot the one-shot `--text` path
speaks with, because that path has no engine to select by name.

```sh
# all three synthesisers in one process, on card 2
xabe-engine --serve 127.0.0.1:8000 \
            --tts-model  models/tts/mms-tts-nan       --tts-device 2 \
            --tts-engine cosyvoice=models/tts/cosyvoice3-0.5b \
            --tts-engine tacotron2=models/tts/tacotron2-nan \
            --tts-script cosyvoice=HAN \
            --tts-script tacotron2=POJ
```

### The system prompt

Two built-ins, and which one a run gets follows who is expected to produce
Taigi rather than any flag of its own. With `--direct-taigi` the chat model
writes Taigi Han itself and the built-in asks for exactly that; without it the
model writes Mandarin and the translator converts, so the built-in asks for
Mandarin. Both are short, both name `--person` and `--bot`, and both say the
reply will be read aloud - at most two sentences, no Markdown, no parentheses.

`--system-prompt` gives one inline and `--prompt-file` reads one from a file.
They are the same setting with a level of indirection, they are **alternatives
rather than layers**, and giving both is refused:

```sh
xabe-engine --serve 0.0.0.0:8000 --llm-model models/breeze2-8b-Q4_K_M.gguf \
            --system-prompt "用台語漢字回答，逐句八到十二字。"

XABE_PROMPT_FILE=prompts/system-taigi.txt xabe-engine --serve 0.0.0.0:8000 ...
```

`prompts/` is gitignored. A system prompt is deployment content rather than
engine behaviour - it names a character, and whose character that is differs
per deployment - so the repository ships the built-ins and nothing else, and
the directory is yours to fill. `docker-compose.yml` mounts it read-only at
`/prompts` whether or not anything is in it.

A given prompt **replaces** the built-in whole rather than being prepended to
it. That is the decision worth knowing: a system prompt is one instruction to
one model, and two of them stacked is the shape that produces a model
following neither - which reads as a bad model rather than as two prompts.

Three consequences follow, and each is a thing to get wrong once:

- **It is taken literally.** No `{person}`/`{bot}` substitution. The built-ins
  interpolate those because they are the engine's own text; a prompt from
  outside is not the engine's to rewrite, and one containing a brace would
  otherwise change meaning depending on where it was written. Write the names
  in directly - and write the *same* name `--bot` gives the transcript, since
  the stop strings are derived from that and a mismatch lets the model write
  the user's next turn itself.
- **`--direct-taigi` still places the translator.** Giving a prompt takes over
  the text and nothing else, so the two flags are not the same switch. A
  prompt handed in here has to write in whatever script the configured
  synthesiser reads - `mms` reads POJ, CosyVoice reads Han - and the engine
  cannot check that for you the way it checks the built-in pairing.
- **An empty prompt is refused**, from either source. Trimming to nothing
  leaves the completion opening on a blank line with no instruction at all;
  the model still answers, just as though it had never been told who it is.
  An empty `--prompt-file` used to do that silently.

Which prompt a process ended up with is logged at startup beside the stages,
because a served process may have been configured from six environment
variables and this is the setting most likely to be wrong in a way that looks
like the model misbehaving:

```
INFO system prompt source="--system-prompt" chars=23
```

## Which card each stage goes on

Every `--*-device` defaults to `0`, which puts the whole pipeline on one card.
That fits - the packed checkpoints leave room - but it is not the fastest
layout, because the chat model and the translator are **both decoding at the
same time**: the reply is chunked as it streams, so clause one is being
translated while clause two is still being written.

Measured on a three-clause turn, moving only the translator off the chat
model's card:

| | one card | translator on its own |
| --- | ---: | ---: |
| first audio | 2659 ms | **2000 ms** |
| first clause, translate | 1618 ms | 1154 ms |
| first clause, synthesise | 440 ms | 210 ms |

The later clauses barely move, because by then the chat model has finished and
there was nothing to contend with. It is the *first* clause - the one a listener
is actually waiting through - that is paying for the overlap.

```sh
# the chat model and everything cheap on card 0, the translator alone on card 1
xabe-engine --serve 127.0.0.1:8000 \
            --llm-model        models/breeze2-8b-Q4_K_M.gguf           --llm-device 0 \
            --translator-model models/taigi-translator-13b-Q4_K_M.gguf --translator-device 1 \
            --asr-model        models/asr/breeze-asr-26                --asr-device 0 \
            --tts-engine       tacotron2=models/tts/tacotron2-nan \
            --tts-script       tacotron2=POJ --tts-default tacotron2
```

On one card, put the translator there anyway and expect the first clause to
cost what the table's left column says. On two, this is the split that matters;
a third card has nothing left to move onto it that is worth the VRAM.

Nothing else needs setting: the engine compares the translator's device with the
synthesiser's and overlaps the two stages only when they differ, because sharing
a card the overlap costs first audio rather than buying it. The `serving` line
prints `translate_ahead=1` or `0` so it is visible which was chosen.

**A synthesiser that reads romanisation needs the translator.** Tacotron2 and
mms read Tai-lo or POJ and their alphabets contain no Han, so a Han reply
synthesises as near-silence rather than as an error. `--direct-taigi` answers in
Taigi *Han* and removes the translator in the same move, so it pairs only with
an engine on `--tts-script <name>=HAN` - CosyVoice. The combination is refused at
startup rather than discovered as a quiet turn.

CosyVoice reads **Han**, so pair it with `--tts-script <name>=HAN`; mms and
Tacotron2 read romanisation and get POJ from `--translator-target`. Tacotron2
was trained on Tâi-lô with the tone as a trailing digit rather than on POJ with
diacritics, and `xabe-taco` transliterates between them — that is a fact about
the checkpoint, so it is not a flag.

Each engine answers at its own sample rate: mms at 16 kHz, Tacotron2 at
22.05 kHz, CosyVoice at its own.

`xabe-taco-bench` times the Tacotron2 path and prints where the time goes:

```sh
xabe-taco-bench --model models/tts/tacotron2-nan --device 0 --rounds 9
```

`xabe-llm-bench` does the same for the two Llama stages, separating prefill from
decode - they are bound by different things, and decode is what a listener waits
through. `--packing f16` widens a quantized checkpoint at load, which is how the
packed path's remaining headroom was measured:

```sh
xabe-llm-bench --model models/breeze2-8b-Q4_K_M.gguf --kind chat --device 0
xabe-llm-bench --model models/taigi-translator-13b-Q4_K_M.gguf --kind translate --device 0
```

A local engine registered through `--tts-engine` shares `--tts-device` with
`--tts-model`, and takes it from there when there is no `--tts-model` to share
it with. CosyVoice is CUDA only — there is no scalar path for a 642
M-parameter decode plus a 22-layer diffusion transformer, and offering one
would give a configuration that starts and then never answers. So is
Tacotron2: WaveGlow is 87.9 M parameters of dilated convolution run at the
sample rate.

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
does, refuse a stage asked to run somewhere it cannot, then work.

There used to be a fourth thing in that list — refusing stages the plan had not
built yet, each error naming the phase to wait for. Every stage is built now, so
the only refusal left is `--vad-url`, and it says the VAD runs in process rather
than naming a phase. A permanent limit and a pending one should not read the
same; someone told to wait for phase 3 would wait forever.

| | |
| --- | --- |
| `--tts-model` and `--tts-url` together | alternatives; give one |
| `--asr-device` with `--asr-url` | a device applies only to a local stage |
| `--asr-device` with no ASR stage at all | the same typo, one flag earlier |
| `--vad-*` with `--serve` and no ASR | served, a VAD is only ever a gate |
| `--vad-device 0` | the VAD has no CUDA implementation |
| `--asr-device cpu` | the ASR has no CPU implementation |
| no stage at all | names the flag pairs that would give one |
| `--tts-device` with no `--tts-model` and no `--tts-engine` | nothing local to place |
| stages but no `--serve`, `--in` or `--text` | a serve command with `--serve` forgotten |
| `--in` and `--text` together | alternatives; give one |
| `--system-prompt` and `--prompt-file` together | alternatives; give one |
| either of those two, empty | a prompt that says nothing is not a prompt |
| `--serve` with `--in` or `--text` | a server is not a one-shot run |
| `--text` with no TTS stage | names `--tts-model` and `--tts-url` |
| `--text` with no `--out` | nowhere to put the WAV |
| `--in` with neither ASR nor VAD | nothing here reads audio |

`--vad-model --in clip.wav` is *accepted*: over a file the VAD is a tool that
prints segments, and only when served is it a gate that needs something to gate.

### Stages that refuse a device

Every stage in the table below is built. What is left of the older wording here
is the part that still applies: three stages refuse a device rather than
accepting it and doing something else.

The VAD has no CUDA implementation and is not going to: 15 tensors and a
millisecond of work, where the transfer would cost more than the arithmetic.
The ASR has no CPU implementation for the mirror-image reason: one 30-second
window is 2.2 TFLOP through the encoder, which the scalar kernels would take
twenty minutes to do. The translator refuses the CPU for that reason and a
second one - a scalar path would hold its 26.5 GB of f16 weights as 53 GB of
f32, which no card here has. All three would otherwise be a flag that is
accepted and then quietly means something else - which is exactly how the
startup log once came to announce `device=cuda:0` for a stage running on the
CPU.

| stage | status |
| --- | --- |
| `--tts-model` | works |
| `--serve` | works |
| `--asr-url`, `--tts-url`, `--translator-url`, `--llm-url` | work |
| `--vad-model` | works, CPU only |
| `--vad-url` | never: the VAD is a millisecond of CPU, so a round trip would cost more than the work |
| `--asr-model` | works, CUDA only |
| `--translator-model` | works, CUDA only, safetensors or GGUF |

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
