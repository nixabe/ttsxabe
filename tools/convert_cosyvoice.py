#!/usr/bin/env python3
"""Convert CosyVoice3's three `.pt` checkpoints to safetensors.

    python tools/convert_cosyvoice.py \
        --model-dir models/tts/cosyvoice3-0.5b --out models/tts/cosyvoice3-0.5b

Writes `llm.safetensors`, `flow.safetensors` and `hift.safetensors` beside the
originals. Nothing is dropped, reshaped or fused: a converter that "helpfully"
rearranges is a converter whose output cannot be diffed against the source, and
this workspace reads safetensors because that is the one container `xabe-st`
knows - not because the layout wanted changing.

# Two renames, and why they are not "rearranging"

**Weight norm has two spellings in PyTorch.** The old `weight_g`/`weight_v` pair
and the new `parametrizations.weight.original0`/`original1` are the same two
tensors - a magnitude of shape `[out, 1, 1]` and a direction of shape
`[out, in, k]`, recombined as `g * v / ||v||`. `hift.pt` was saved by a torch
new enough to use the second spelling; `xabe-vits` already reads the first, and
teaching it a second name for the same thing would be two code paths that agree
until one is edited. So the names are normalised here, where the difference is
visibly a PyTorch version and not a model property.

**`lm_head.weight` is dropped.** Qwen2 0.5B ties it to the embedding - same
storage, verified by pointer, not merely equal - and CosyVoice never calls it:
its head is `llm_decoder`, which projects 896 to 6761 speech tokens rather than
to 151936 text ones. Writing it would add 544 MB of a tensor that is both a
duplicate and dead. The tie is recorded in the file's metadata so the reader
knows it was a decision.

**`llm.` is stripped from the Qwen2 backbone.** The checkpoint nests it as
`llm.model.model.layers.N...` because CosyVoice wraps a 🤗 `Qwen2Model` inside a
`Qwen2Encoder` inside a `CosyVoice3LM`. Three levels of wrapper are a fact about
their class hierarchy, not about the weights.

Both renames are printed and counted, so what happened is in the log rather
than inferred from the output.

# Why safetensors and not GGUF

The engine reads GGUF for the two Llama checkpoints because they *ship* as
GGUF. These ship as pickles, and converting a pickle to GGUF would mean writing
a GGUF *writer* and inventing metadata keys for a non-llama architecture. The
safetensors path already exists, already handles sharding, and already carries
`xabe-vits`'s weight-norm convention.
"""

import argparse
import collections
import json
import pathlib

import torch
from safetensors.torch import save_file

# `original0` is the magnitude and `original1` the direction. Checked against
# the shapes rather than assumed: the magnitude is `[out, 1, 1]` and the
# direction is `[out, in, k]`, so getting them backwards is visible.
WEIGHT_NORM = {
    "parametrizations.weight.original0": "weight_g",
    "parametrizations.weight.original1": "weight_v",
}


# Tied to `model.embed_tokens.weight` and unused: CosyVoice's output head is
# `llm_decoder`, to 6761 speech tokens, not `lm_head` to 151936 text ones.
DROP = {"llm.model.lm_head.weight"}


def rename(key: str, strip_llm: bool) -> str:
    for old, new in WEIGHT_NORM.items():
        if key.endswith("." + old):
            return key[: -len(old) - 1] + "." + new
    if strip_llm and key.startswith("llm.model.model."):
        return key[len("llm.model.") :]
    if strip_llm and key.startswith("llm.model.lm_head."):
        return key[len("llm.model.") :]
    return key


def convert(src: pathlib.Path, dst: pathlib.Path, strip_llm: bool) -> None:
    sd = torch.load(src, map_location="cpu", weights_only=True)

    out, renamed, dropped, shapes = {}, 0, 0, collections.Counter()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            print(f"  skipping non-tensor {k!r} ({type(v).__name__})")
            continue
        if k in DROP:
            print(f"  dropping {k} ({tuple(v.shape)}): tied to the embedding and unused")
            dropped += 1
            continue
        nk = rename(k, strip_llm)
        if nk != k:
            renamed += 1
        if nk in out:
            raise SystemExit(f"{src.name}: {nk} appears twice after renaming")
        # `.contiguous()` because safetensors stores a flat buffer and a
        # non-contiguous view would be written in the wrong order - silently,
        # since the shape is still right.
        out[nk] = v.contiguous()
        shapes[str(v.dtype)] += v.numel()

    # Costs nothing and catches the one mistake this script could plausibly
    # make: losing a tensor while renaming rather than while dropping.
    assert len(out) + dropped == sum(1 for v in sd.values() if torch.is_tensor(v))

    meta = {"format": "pt", "source": src.name}
    if dropped:
        meta["dropped"] = "lm_head.weight: tied to embed_tokens and unused"
    save_file(out, str(dst), metadata=meta)
    total = sum(shapes.values())
    print(
        f"  {src.name} -> {dst.name}: {len(out)} tensors, {total:,} params, "
        f"{renamed} renamed, {dropped} dropped, {dict(shapes)}"
    )


def verify(pt: pathlib.Path, st: pathlib.Path, strip_llm: bool) -> None:
    """Reads both back and requires every tensor to be bit-identical.

    The conversion is a copy, so anything short of equality is a bug - and a
    dtype or contiguity mistake produces a file that loads fine and holds the
    wrong numbers, which is the failure this is here to catch.
    """
    from safetensors.torch import load_file

    a = torch.load(pt, map_location="cpu", weights_only=True)
    b = load_file(str(st))
    bad = 0
    for k, v in a.items():
        if not torch.is_tensor(v) or k in DROP:
            continue
        nk = rename(k, strip_llm)
        if nk not in b:
            print(f"    MISSING {nk}")
            bad += 1
        elif not torch.equal(v, b[nk]):
            print(f"    DIFFERS {nk}")
            bad += 1
    if bad:
        raise SystemExit(f"{pt.name}: {bad} tensors did not survive the round trip")
    print(f"  {st.name}: {len(b)} tensors verified bit-identical")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()

    src_dir = pathlib.Path(a.model_dir)
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, strip in [("llm", True), ("flow", False), ("hift", False)]:
        src = src_dir / f"{name}.pt"
        if not src.is_file():
            raise SystemExit(f"{src} is missing")
        dst = out_dir / f"{name}.safetensors"
        convert(src, dst, strip)
        if not a.skip_verify:
            verify(src, dst, strip)
        manifest[name] = dst.name

    (out_dir / "cosyvoice.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nwrote {out_dir}/cosyvoice.json")


if __name__ == "__main__":
    main()
