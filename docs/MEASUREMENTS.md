# Benchmark: xabe vs llama.cpp, whisper.cpp and PyTorch

## Setup

| | |
| --- | --- |
| GPU | 1x Quadro RTX 8000 (sm_75, 48 GB), driver 595.91.07 |
| xabe | `813d72b`, release build |
| llama.cpp | `2145525`, CUDA 12.8, `llama-bench -ngl 99 -r 9` |
| whisper.cpp | `d09f61a`, CUDA 12.8, `whisper-server -nf -bo 1 -bs -1 -l zh` (greedy, single pass) |
| PyTorch | 2.5.1+cu121; Tacotron2 from `yfliao/taiwanese_tonal_tlpa_tacotron2` `d5f98c0`; VITS via transformers 5.17.0 |
| Method | Same card, same files, one sitting; reference and xabe alternated round by round; medians |

Speedup = reference time / xabe time (or xabe tok/s / reference tok/s). Above 1.00x means xabe is faster.

## Summary

| Stage | Model | Reference | Metric | Speedup |
| --- | --- | --- | --- | ---: |
| Chat LLM | Breeze2 8B Q4_K_M | llama.cpp | prefill, 16–8192 tokens | 1.05x – 1.32x |
| Chat LLM | Breeze2 8B Q4_K_M | llama.cpp | decode, depth 16–8192 | 1.03x – 1.10x |
| Translator | Taigi 13B Q4_K_M | llama.cpp | prefill, 16–3968 tokens | 1.02x – 1.26x |
| Translator | Taigi 13B Q4_K_M | llama.cpp | decode, depth 16–3968 | 1.02x – 1.09x |
| ASR | Breeze-ASR-26 (Whisper large-v2) | whisper.cpp | end-to-end, 2.9–10.3 s clips | 1.48x – 1.62x |
| TTS | Tacotron2 + WaveGlow | PyTorch fp16 (reference config) | time per audio second | 2.55x – 2.86x |
| TTS | Tacotron2 + WaveGlow | PyTorch fp32 | time per audio second | 2.39x – 3.24x |
| TTS | VITS mms-tts-nan | PyTorch fp32 (transformers) | time per audio second | **3.27x** |

## LLM stages vs llama.cpp

Prefill = `pp<N>` vs `xabe-llm-bench --prompt N`. Decode = 64 tokens at the given context depth (`tg64 @ d<N>` vs 64 tokens after an N-token prompt). TTFT = prefill of the prompt + one decode step, from the measured rates. Each cell is the median of 3 alternated rounds of 9 repetitions.

### Chat: Breeze2 8B Q4_K_M

| Prompt tokens | Prefill llama.cpp | Prefill xabe | Speedup | TTFT llama.cpp | TTFT xabe | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 779 tok/s | 1030 tok/s | **1.32x** | 30.6 ms | 24.7 ms | **1.24x** |
| 24 | 1243 tok/s | 1416 tok/s | **1.14x** | 29.4 ms | 26.1 ms | **1.12x** |
| 32 | 1471 tok/s | 1764 tok/s | **1.20x** | 31.8 ms | 27.3 ms | **1.16x** |
| 64 | 2021 tok/s | 2112 tok/s | **1.05x** | 41.7 ms | 39.5 ms | **1.06x** |
| 128 | 2347 tok/s | 2558 tok/s | **1.09x** | 64.6 ms | 59.3 ms | **1.09x** |
| 512 | 2726 tok/s | 3102 tok/s | **1.14x** | 198.0 ms | 174.5 ms | **1.14x** |
| 1024 | 2706 tok/s | 3109 tok/s | **1.15x** | 388.7 ms | 338.9 ms | **1.15x** |
| 2048 | 2668 tok/s | 3047 tok/s | **1.14x** | 778.1 ms | 682.0 ms | **1.14x** |
| 4096 | 2569 tok/s | 2976 tok/s | **1.16x** | 1605.3 ms | 1386.6 ms | **1.16x** |
| 8192 | 2395 tok/s | 2799 tok/s | **1.17x** | 3431.9 ms | 2938.4 ms | **1.17x** |

| Context depth | Decode llama.cpp | Decode xabe | Speedup | ms/token llama.cpp | ms/token xabe |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 99.2 tok/s | 109.1 tok/s | **1.10x** | 10.09 | 9.17 |
| 24 | 99.3 tok/s | 108.9 tok/s | **1.10x** | 10.07 | 9.18 |
| 32 | 99.3 tok/s | 108.8 tok/s | **1.10x** | 10.07 | 9.19 |
| 64 | 99.3 tok/s | 108.7 tok/s | **1.09x** | 10.07 | 9.20 |
| 128 | 99.2 tok/s | 108.5 tok/s | **1.09x** | 10.08 | 9.22 |
| 512 | 98.0 tok/s | 106.2 tok/s | **1.08x** | 10.20 | 9.42 |
| 1024 | 96.7 tok/s | 104.6 tok/s | **1.08x** | 10.35 | 9.56 |
| 2048 | 94.4 tok/s | 101.1 tok/s | **1.07x** | 10.60 | 9.89 |
| 4096 | 90.8 tok/s | 96.1 tok/s | **1.06x** | 11.01 | 10.41 |
| 8192 | 83.9 tok/s | 86.5 tok/s | **1.03x** | 11.92 | 11.56 |

### Translator: Taigi 13B Q4_K_M (Llama-2)

| Prompt tokens | Prefill llama.cpp | Prefill xabe | Speedup | TTFT llama.cpp | TTFT xabe | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 501 tok/s | 633 tok/s | **1.26x** | 48.4 ms | 40.4 ms | **1.20x** |
| 24 | 728 tok/s | 869 tok/s | **1.19x** | 49.4 ms | 42.8 ms | **1.16x** |
| 32 | 864 tok/s | 1057 tok/s | **1.22x** | 53.5 ms | 45.5 ms | **1.18x** |
| 64 | 1198 tok/s | 1218 tok/s | **1.02x** | 69.9 ms | 67.8 ms | **1.03x** |
| 128 | 1332 tok/s | 1468 tok/s | **1.10x** | 112.6 ms | 102.5 ms | **1.10x** |
| 512 | 1520 tok/s | 1672 tok/s | **1.10x** | 354.0 ms | 322.4 ms | **1.10x** |
| 1024 | 1496 tok/s | 1675 tok/s | **1.12x** | 702.3 ms | 628.3 ms | **1.12x** |
| 2048 | 1450 tok/s | 1697 tok/s | **1.17x** | 1432.0 ms | 1225.3 ms | **1.17x** |
| 3968 | 1363 tok/s | 1631 tok/s | **1.20x** | 2932.2 ms | 2454.1 ms | **1.19x** |

| Context depth | Decode llama.cpp | Decode xabe | Speedup | ms/token llama.cpp | ms/token xabe |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 60.6 tok/s | 66.0 tok/s | **1.09x** | 16.51 | 15.15 |
| 24 | 60.7 tok/s | 65.9 tok/s | **1.09x** | 16.48 | 15.17 |
| 32 | 60.6 tok/s | 65.8 tok/s | **1.09x** | 16.49 | 15.20 |
| 64 | 60.6 tok/s | 65.5 tok/s | **1.08x** | 16.49 | 15.27 |
| 128 | 60.7 tok/s | 65.0 tok/s | **1.07x** | 16.48 | 15.38 |
| 512 | 58.2 tok/s | 62.1 tok/s | **1.07x** | 17.18 | 16.10 |
| 1024 | 56.0 tok/s | 58.9 tok/s | **1.05x** | 17.84 | 16.98 |
| 2048 | 51.7 tok/s | 54.1 tok/s | **1.05x** | 19.34 | 18.48 |
| 3968 | 46.2 tok/s | 47.1 tok/s | **1.02x** | 21.65 | 21.23 |

## ASR vs whisper.cpp

Breeze-ASR-26 (Whisper large-v2 fine-tune). xabe reads the published safetensors; whisper.cpp runs its own f16 GGML conversion of the same checkpoint. The two were alternated in each of 20 timed rounds after 3 warm-up rounds.

### End to end

| Clip | Audio | Tokens | whisper.cpp | xabe | Speedup | RTF whisper.cpp | RTF xabe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a | 2.93 s | 10 | 191.1 ms | 129.4 ms | **1.48x** | 0.0652 | 0.0442 |
| b | 4.98 s | 16 | 241.6 ms | 159.2 ms | **1.52x** | 0.0485 | 0.0320 |
| c | 8.22 s | 22 | 294.6 ms | 189.3 ms | **1.56x** | 0.0358 | 0.0230 |
| d | 10.34 s | 28 | 352.9 ms | 218.5 ms | **1.62x** | 0.0341 | 0.0211 |

### Phases

whisper.cpp phases are from `whisper-bench` (warm, median of 3 runs). xabe phases are from `xabe-asr-bench --stages` (synchronised, median of 10 runs, then the median across the four clips).

| Phase | whisper.cpp | xabe | Speedup |
| --- | ---: | ---: | ---: |
| Encoder (30 s window) | 83.3 ms | 58.0 ms | **1.44x** |
| Decode, per token | 7.37 ms (136 tok/s) | 5.00 ms (200 tok/s) | **1.47x** |

xabe per clip:

| Clip | Mel (CPU) | Encoder | Cross-attn KV | Prompt prefix | Decode loop | Time to first token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | 3.9 ms | 58.7 ms | 10.0 ms | 9.4 ms | 61.3 ms | 82.0 ms |
| b | 5.7 ms | 58.0 ms | 9.9 ms | 9.5 ms | 90.2 ms | 83.1 ms |
| c | 5.8 ms | 57.6 ms | 9.9 ms | 9.5 ms | 118.4 ms | 82.8 ms |
| d | 6.5 ms | 57.9 ms | 10.0 ms | 9.5 ms | 146.4 ms | 83.9 ms |

Transcripts matched whisper.cpp exactly on clips a–c. On clip d, xabe wrote `身體才會壯` where whisper.cpp wrote `身體才會強壯`.

## TTS: Tacotron2 + WaveGlow vs PyTorch

The reference is `yfliao/taiwanese_tonal_tlpa_tacotron2` run as the repo runs it: the Tacotron2 decoder loop, WaveGlow (weight norm removed), sigma 0.666. fp16 is the precision the repo's inference notebook uses. Synthesis is stochastic (the prenet dropout stays on at inference), so utterance lengths differ between runs. Comparisons are therefore per second of audio produced, or per decoder frame. Medians over 3 alternated rounds of 9 runs.

### End to end (time per second of audio)

| Text | PyTorch fp32 | PyTorch fp16 | xabe | vs fp32 | vs fp16 | RTF xabe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `li2 ho2` | 132.4 ms | 155.8 ms | 55.5 ms | **2.39x** | **2.81x** | 0.0555 |
| `gua2 si7 tai5-uan5-lang5` | 130.9 ms | 124.2 ms | 43.4 ms | **3.02x** | **2.86x** | 0.0434 |
| two-clause line (~5.5 s) | 129.8 ms | 102.0 ms | 40.0 ms | **3.24x** | **2.55x** | 0.0400 |

### Phases (two-clause line)

| Phase | PyTorch fp32 | PyTorch fp16 | xabe | vs fp32 | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Encoder | 2.62 ms | 2.80 ms | 2.07 ms | **1.27x** | **1.35x** |
| Decoder, per frame | 0.939 ms | 0.925 ms | 0.144 ms | **6.51x** | **6.41x** |
| Postnet | 3.44 ms | 5.25 ms | 1.14 ms | **3.02x** | **4.61x** |
| WaveGlow, per audio second | 49.8 ms | 19.3 ms | 30.6 ms | **1.62x** | **0.63x** |
| Denoiser, per audio second | 1.0 ms | 0.9 ms | not run | – | – |

The decoder's per-frame cost is consistent across all three texts: 0.14–0.15 ms for xabe against 0.93–0.95 ms for PyTorch. WaveGlow is the one phase where the reference wins: its fp16 WaveGlow beats xabe's f32-activation WaveGlow per audio second.

## TTS: VITS (mms-tts-nan) vs PyTorch

`facebook/mms-tts-nan` through transformers `VitsModel`, fp32. Input: `lí hó, kin-á-ji̍t thinn-khì chin hó.` 20 timed calls after 5 warm-up calls, 3 alternated rounds.

| | PyTorch | xabe | Speedup |
| --- | ---: | ---: | ---: |
| Median per call | 67.3 ms (2.59 s audio) | 21.3 ms (2.61 s audio) | **3.16x** |
| Per audio second | 26.7 ms | 8.2 ms | **3.27x** |
| Realtime factor | 37.5x | 122.5x | |

xabe phases (synchronised):

| Text encoder | Duration predictor | Prior | Flow | Decoder (HiFi-GAN) | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.61 ms | 0.98 ms | 0.06 ms | 1.35 ms | 17.30 ms | 21.36 ms |

## Peak VRAM

Peak `nvidia-smi` memory while each tool runs a matching workload on an otherwise idle card. The LLM rows are a 512-token prompt with 64 decoded tokens, and the ASR rows use clip d.

| Stage | Reference | Reference VRAM | xabe VRAM | Ratio (ref / xabe) |
| --- | --- | ---: | ---: | ---: |
| Chat 8B | llama.cpp | 4936 MiB | 5196 MiB | 0.95x |
| Translator 13B | llama.cpp | 8238 MiB | 8972 MiB | 0.92x |
| ASR | whisper.cpp | 3686 MiB | 4012 MiB | 0.92x |
| Tacotron2 + WaveGlow | PyTorch fp32 | 1260 MiB | 908 MiB | 1.39x |
| Tacotron2 + WaveGlow | PyTorch fp16 | 1052 MiB | 908 MiB | 1.16x |
| VITS | PyTorch fp32 | 406 MiB | 332 MiB | 1.22x |
