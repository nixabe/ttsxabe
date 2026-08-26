#!/usr/bin/env python3
"""Capture `llama-server`'s translations, as JSON.

    python tools/oracle/capture_llama_server.py --url http://127.0.0.1:8081 \
        --out .golden/translator/llama_server.json

The second reference for the translator, and a different kind of one. 🤗 on CPU
in float32 says what the arithmetic should be; `llama-server` running the f16
GGUF of the same weights is what the pipeline actually uses today, so it says
what the *replacement* has to reproduce. Agreeing with both is the claim; the
two disagreeing with each other would be worth knowing about on its own.

The request body is `gateway.py`'s, unchanged - template, temperature,
`repeat_penalty` and stop strings - because a comparison against a differently
configured server is not a comparison.
"""

import argparse
import json
import urllib.request

CORPUS = [
    ("今天天氣很好", "POJ"),
    ("今天天氣很好", "HAN"),
    ("我要去市場買東西", "POJ"),
    ("我要去市場買東西", "HAN"),
    ("你食飽未", "POJ"),
    ("你食飽未", "HAN"),
    ("這是一个測試", "POJ"),
    ("明天我欲去學校", "HAN"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cases = []
    for src, tgt in CORPUS:
        body = {
            "prompt": f"[TRANS]\n{src}\n[/TRANS]\n[{tgt}]\n",
            "temperature": 0.0,
            "repeat_penalty": 1.1,
            "n_predict": 256,
            "stop": ["[/", "\n["],
        }
        req = urllib.request.Request(
            a.url.rstrip("/") + "/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)["content"].strip()
        print(f"  {src} [{tgt}] -> {out}", flush=True)
        cases.append({"src": src, "tgt": tgt, "text": out})

    with open(a.out, "w") as f:
        json.dump({"url": a.url, "cases": cases}, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {a.out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
