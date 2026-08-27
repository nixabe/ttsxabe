#!/usr/bin/env python3
"""Capture `gguf-py`'s dequantization for every block format this crate reads.

    python tools/oracle/capture_quants.py --out .golden/gguf/quants

The reference is upstream llama.cpp's `gguf-py/gguf/quants.py` - the same code
that writes the files this workspace reads. For each format it writes the
packed bytes and the f32 the reference unpacks them to, so the Rust side's job
is exact rather than approximate: it never quantizes, only reproduces this
dequantization, which is a property with no tolerance to argue about.

The packed bytes are **pseudo-random rather than the output of a quantizer**,
and that is deliberate twice over. Python `gguf` can only quantize the five
legacy formats - the K-quants exist in C alone - so a round-trip corpus would
have covered half the table. And random encodings exercise every nibble, every
packed six-bit scale and every high-bit mask, where a quantizer only ever emits
the well-conditioned subset. A nibble-order mistake survives the second and
dies against the first.

The one thing not left to chance is the f16 scale fields: random bits there
would be NaN about 3% of the time, which says nothing about a layout and makes
the comparison awkward for no benefit. Those positions are listed per format
and filled with finite values; everything else is noise.

For the five formats Python can quantize, a real round trip is captured too,
under `<NAME>.q.*` - so the corpus has both the whole encoding space and the
part a quantizer actually reaches.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))

import numpy as np
from gguf import quants
from gguf.constants import GGMLQuantizationType as T

# Byte offsets of the f16 scale/minimum fields within one block.
F16_FIELDS = {
    "Q4_0": [0],
    "Q4_1": [0, 2],
    "Q5_0": [0],
    "Q5_1": [0, 2],
    "Q8_0": [0],
    "Q2_K": [80, 82],
    "Q3_K": [108],
    "Q4_K": [0, 2],
    "Q5_K": [0, 2],
    "Q6_K": [208],
}

QUANTIZABLE = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"}


def packed(name, qtype, rows, blocks_per_row, seed):
    """Random block bytes with finite scales."""
    block, tsize = quants.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    n_blocks = rows * blocks_per_row
    raw = rng.integers(0, 256, size=(n_blocks, tsize), dtype=np.uint8)
    # Scales spanning several orders of magnitude, both signs, so no block
    # shares another's scale and a per-block indexing error cannot hide.
    for off in F16_FIELDS[name]:
        mag = np.exp(rng.uniform(-5.0, 2.0, n_blocks)).astype(np.float16)
        sign = rng.choice([-1, 1], n_blocks).astype(np.float16)
        raw[:, off:off + 2] = (mag * sign).view(np.uint8).reshape(n_blocks, 2)
    return raw.reshape(rows, blocks_per_row * tsize)


def real_values(n, seed):
    rng = np.random.default_rng(seed)
    mag = np.exp(rng.uniform(-6.0, 3.0, n))
    sign = rng.choice([-1.0, 1.0], n)
    x = (mag * sign).astype(np.float32)
    x[rng.choice(n, n // 20, replace=False)] = 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cases = []
    for i, name in enumerate(F16_FIELDS):
        qtype = getattr(T, name)
        block, tsize = quants.GGML_QUANT_SIZES[qtype]

        raw = packed(name, qtype, a.rows, a.blocks, seed=1000 + i)
        back = quants.dequantize(raw, qtype).astype(np.float32)
        assert np.isfinite(back).all(), f"{name} produced a non-finite value"
        raw.tofile(os.path.join(a.out, f"{name}.bin"))
        back.tofile(os.path.join(a.out, f"{name}.f32"))

        case = {
            "name": name,
            "id": int(qtype.value),
            "block_size": int(block),
            "type_size": int(tsize),
            "rows": int(a.rows),
            "cols": int(a.blocks * block),
            "elements": int(back.size),
            "packed_bytes": int(raw.nbytes),
            "round_trip": False,
        }

        if name in QUANTIZABLE:
            x = real_values(a.rows * a.blocks * block, seed=2000 + i)
            x = x.reshape(a.rows, a.blocks * block)
            q = quants.quantize(x, qtype)
            d = quants.dequantize(q, qtype).astype(np.float32)
            q.tofile(os.path.join(a.out, f"{name}.q.bin"))
            d.tofile(os.path.join(a.out, f"{name}.q.f32"))
            case["round_trip"] = True

        cases.append(case)
        print(f"  {name:6} id={qtype.value:3} block={block:4} bytes/block={tsize:4} "
              f"packed={raw.nbytes:6} round_trip={case['round_trip']}")

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"cases": cases}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
