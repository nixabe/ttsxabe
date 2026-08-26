#!/usr/bin/env bash
# Capture whisper.cpp's VAD output over the clip corpus.
#
#   WHISPER=~/whisper.cpp tools/oracle/capture_vad.sh
#
# Builds the corpus, adds real speech synthesised by the engine itself at a
# fixed seed, then runs the reference VAD over every clip and writes the
# probabilities and segments as binary under .golden/vad/.
#
# Needs a built whisper.cpp checkout, because the reference is whisper.cpp's
# VAD rather than Python silero-vad - see tools/oracle/vad_capture.cpp for why.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

WHISPER=${WHISPER:-$HOME/whisper.cpp}
PYTHON=${PYTHON:-python3}
VAD_MODEL=${VAD_MODEL:-models/vad/ggml-silero-v5.1.2.bin}
OUT=${OUT:-.golden/vad}
CLIPS="$OUT/clips"

[ -f "$WHISPER/build/bin/libwhisper.so" ] || {
  echo "no libwhisper.so under $WHISPER/build/bin - build whisper.cpp first"; exit 1; }
[ -f "$VAD_MODEL" ] || { echo "no VAD model at $VAD_MODEL"; exit 1; }

echo "building the capture tool"
BIN=$(mktemp -d)/vad_capture
g++ -O2 -std=c++17 -I "$WHISPER/include" -I "$WHISPER/ggml/include" \
    tools/oracle/vad_capture.cpp -L "$WHISPER/build/bin" -lwhisper -o "$BIN"

echo "building the corpus"
"$PYTHON" tools/oracle/vad_corpus.py "$CLIPS"

# Real speech, from this engine, at a fixed seed. Synthetic tones exercise the
# segmenter; only speech exercises the detector on what it was trained for.
if [ -x target/release/xabe-engine ] && [ -d models/tts/mms-tts-nan ]; then
  echo "synthesising speech clips"
  target/release/xabe-engine --tts-model models/tts/mms-tts-nan --tts-device cpu \
    --text "Lí hó, kin-á-ji̍t thinn-khì chin hó." --out "$CLIPS/speech.wav" \
    --log-level warn
  target/release/xabe-engine --tts-model models/tts/mms-tts-nan --tts-device cpu \
    --text "Góa beh khì chhī-tiûⁿ bé mih-kiāⁿ. Lí beh khì bô?" \
    --out "$CLIPS/speech_two.wav" --log-level warn
else
  echo "no engine binary or no TTS checkpoint; skipping the speech clips"
fi

echo "capturing"
for wav in "$CLIPS"/*.wav; do
  name=$(basename "$wav" .wav)
  mkdir -p "$OUT/$name"
  LD_LIBRARY_PATH="$WHISPER/build/bin" "$BIN" "$VAD_MODEL" "$wav" "$OUT/$name" \
    2>/dev/null | sed 's/^/  /'
done

cat > "$OUT/manifest.json" <<JSON
{
  "reference": "whisper.cpp whisper_vad_detect_speech + whisper_vad_segments_from_probs",
  "whisper_cpp": "$(git -C "$WHISPER" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "vad_model": "$(basename "$VAD_MODEL")",
  "device": "cpu",
  "threads": 1,
  "corpus_seed": 20250827,
  "segment_params": {
    "threshold": 0.6,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 200,
    "max_speech_duration_s": null,
    "speech_pad_ms": 30
  }
}
JSON

echo
echo "wrote $OUT ($(ls -d "$OUT"/*/ | wc -l) captures)"
