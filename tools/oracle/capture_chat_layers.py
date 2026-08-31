#!/usr/bin/env python3
"""Capture `llama-eval-callback`'s per-layer sums for the chat model, as JSON.

    python tools/oracle/capture_chat_layers.py \
        --bin ~/llama.cpp/build/bin/llama-eval-callback \
        --model models/breeze2-8b-Q4_K_M.gguf \
        --out .golden/chat/layers.json

`tests/llama_server.rs` says the chat model has no oracle with per-layer taps,
because there is no HuggingFace checkpoint for it on this machine - it exists as
a GGUF and nothing else. That was true of the *product* comparison and not of
the model: llama.cpp's own `eval-callback` prints every node of the graph with a
scalar sum, which is a per-layer tap on the same file.

It is a weaker tap than a captured tensor - a sum over 4096 elements can hide a
permutation, and a sum that cancels can exaggerate a small error into a large
percentage. It is far stronger than comparing replies, which is what it
replaces: it says *which layer* a divergence enters at, and a reply cannot.

Run with `-ngl 0`. The CPU and CUDA backends were measured to agree to the
printed precision at every layer, so the choice is about reproducibility rather
than accuracy, and the CPU one does not depend on which card is free.

The capture records **which file it came from**, and `tests/layer_taps.rs`
refuses one taken from a different model. That is not bookkeeping. A capture of
the Q4_K build compared against the engine reading the f16 build of the same
model diverges by 0.36 of the layer magnitude at block 1 and stays there - the
signature of a real wiring fault, produced by nothing worse than pointing the
test at the wrong file. The same class of mistake already cost this project a
day once; `docs/TESTING.md` has that one.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# One node: the name, then somewhere below it a line "sum = <float>".
NODE = re.compile(r"common_debug_cb_eval:\s+(\S+)\s+=")
SUM = re.compile(r"sum = (-?[\d.]+)")


def identity(model: str) -> dict:
    """What the capture was taken from, in terms a test can check.

    Name and byte count rather than a hash: it has to be cheap for the test to
    compute on a 16 GB file, and two GGUFs of the same model at different
    quantizations differ in both.
    """
    return {"name": os.path.basename(model), "bytes": os.path.getsize(model)}


def capture(binary: str, model: str, prompt: str, ngl: int) -> dict:
    """Runs the reference once and returns {node: [sum, ...]} in graph order."""
    out = subprocess.run(
        [binary, "-m", model, "-p", prompt, "-n", "1", "-ngl", str(ngl)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    text = out.stdout + out.stderr
    if "common_debug_cb_eval" not in text:
        sys.exit(f"no nodes in the output; the binary printed:\n{text[:2000]}")

    nodes: dict[str, list[float]] = {}
    for block in re.split(r"common_debug_cb_eval:\s+", text)[1:]:
        name = NODE.match("common_debug_cb_eval:  " + block)
        name = re.match(r"(\S+)\s+=", block)
        got = SUM.search(block)
        if name and got:
            nodes.setdefault(name.group(1), []).append(float(got.group(1)))
    tokens = re.search(r"number of input tokens = (\d+)", text)
    return {
        "prompt": prompt,
        "model": identity(model),
        "tokens": int(tokens.group(1)) if tokens else None,
        # The last occurrence of a name is the node in its final form: llama.cpp
        # reuses a name across a reshape and a rope, and the last is the one a
        # block output is built from.
        "nodes": {k: v[-1] for k, v in nodes.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to llama-eval-callback")
    ap.add_argument("--model", required=True, help="the GGUF to read")
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    got = capture(a.bin, a.model, a.prompt, a.ngl)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(got, f, ensure_ascii=False, indent=1)
    layers = sum(1 for k in got["nodes"] if k.startswith("l_out-"))
    print(
        f"wrote {a.out}: {got['tokens']} tokens, {layers} block outputs, "
        f"from {got['model']['name']}"
    )


if __name__ == "__main__":
    main()
