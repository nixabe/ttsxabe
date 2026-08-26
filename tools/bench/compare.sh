#!/usr/bin/env bash
# Alternates the two implementations in pairs, per docs/BENCHMARKS.md.
#
# This card thermally drifts, so running all of A and then all of B measures
# the drift as much as the difference. Each round is one PyTorch block and one
# Rust block, and the medians are taken across rounds.
set -euo pipefail

MODEL=${XABE_TTS_MODEL:?set XABE_TTS_MODEL to the safetensors checkpoint}
TEXT=${TEXT:-"lí hó, kin-á-ji̍t thinn-khì chin hó."}
ROUNDS=${ROUNDS:-3}
RUNS=${RUNS:-20}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

for r in $(seq 1 "$ROUNDS"); do
    echo "=== round $r/$ROUNDS ==="
    echo "--- pytorch ---"
    CUDA_VISIBLE_DEVICES=${XABE_TTS_DEVICE:-0} "$PY" "$HERE/pytorch_baseline.py" \
        --text "$TEXT" --device cuda:0 --runs "$RUNS" 2>/dev/null \
        | grep -E "samples|median|realtime"
    echo "--- xabe-tts ---"
    "$ROOT/target/release/xabe-tts-bench" --model "$MODEL" \
        --device "${XABE_TTS_DEVICE:-0}" --text "$TEXT" --runs "$RUNS" 2>/dev/null \
        | grep -E "samples|median|realtime"
done
