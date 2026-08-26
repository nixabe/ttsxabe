#!/usr/bin/env python3
"""Capture the Llama-2 oracle, stage by stage.

    python tools/oracle/capture_llama.py --model models/translator/taigi-llama2-13b \
        --out .golden/translator/trans

The reference is 🤗 `LlamaForCausalLM` on CPU in float32, which is 53 GB of
RAM and a few seconds a prompt. That is affordable exactly once, for a handful
of prompts, and it is the only way to get a reference the arithmetic can be
diffed against rather than merely compared to.

Everything is captured as raw little-endian f32, per-layer as well as final, so
a failure is located rather than merely detected.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

TEMPLATE = "[TRANS]\n{src}\n[/TRANS]\n[{tgt}]\n"


def save(out_dir, name, tensor):
    a = np.ascontiguousarray(
        tensor.detach().cpu().float().numpy().astype("<f4")
        if torch.is_tensor(tensor)
        else np.asarray(tensor, dtype="<f4")
    )
    a.tofile(os.path.join(out_dir, f"{name}.bin"))
    print(f"  {name:<24} {list(a.shape)}", flush=True)
    return list(a.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="今天天氣很好")
    ap.add_argument("--tgt", default="POJ")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=32)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # f32 reduction order is not thread-invariant, and the last bits of every
    # tensor move if this is not pinned.
    torch.set_num_threads(1)

    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()

    prompt = TEMPLATE.format(src=a.src, tgt=a.tgt)
    # `add_special_tokens` puts the BOS on, which the template's `{BOS}` means.
    ids = tok.encode(prompt, add_special_tokens=True)
    with open(os.path.join(a.out, "input_ids.json"), "w") as f:
        json.dump(ids, f)
    print(f"  input_ids                {ids}", flush=True)

    taps = {}

    def tap(name):
        # A layer hook hands back a tuple; a whole stack hands back a
        # ModelOutput. Both carry the hidden states first and neither by the
        # same accessor, and getting it wrong captures a struct rather than a
        # tensor - which is at least loud.
        def hook(_m, _i, output):
            if hasattr(output, "last_hidden_state"):
                t = output.last_hidden_state
            elif isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            taps[name] = t.detach()
        return hook

    inner = model.model
    handles = [inner.register_forward_hook(tap("final_norm"))]
    for i in range(min(a.layers, len(inner.layers))):
        handles.append(inner.layers[i].register_forward_hook(tap(f"layer_{i}")))

    shapes = {}
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    for h in handles:
        h.remove()

    for name, t in sorted(taps.items()):
        shapes[name] = save(a.out, name, t)
    shapes["logits"] = save(a.out, "logits", out.logits)

    with torch.no_grad():
        generated = model.generate(
            input_ids=torch.tensor([ids], dtype=torch.long),
            max_new_tokens=a.max_new,
            num_beams=1,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=[tok.eos_token_id, tok.pad_token_id],
        )
    new = generated[0].tolist()[len(ids):]
    text = tok.decode(new, skip_special_tokens=True)
    print(f"  generated                {text!r}", flush=True)

    manifest = {
        "model": os.path.basename(os.path.abspath(a.model)),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "src": a.src,
        "tgt": a.tgt,
        "prompt": prompt,
        "input_ids": ids,
        "generated_ids": new,
        "generated": text,
        "shapes": shapes,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
