"""Capture the PyTorch reference's intermediates as binary golden files.

The reference is 🤗 Transformers' ``VitsModel`` on CPU in float32. This script
records what it *actually computed*, tensor by tensor, so the Rust
implementation can be diffed against it stage by stage rather than only at the
waveform. See ``docs/ORACLE.md`` for why capture beats transcription.

Nothing here reimplements the model. Module-level tensors come from forward
hooks; the two random draws and the duration-expansion matrix come from a
``TorchFunctionMode`` that observes the calls as they happen. That distinction
matters: a recomputed ``attn`` would be a copy of what we *believe* the
reference does, and believing wrongly is the failure this file exists to
prevent.

Usage::

    python tools/oracle/capture.py --out .golden/base --seed 0 \
        --text "li ho, kin-a-jit thinn-khi chin ho."
"""

import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import torch
from transformers import VitsModel, VitsTokenizer

# Tensors are written raw, C order, little-endian, with shape and dtype in the
# manifest - the same convention `xabe-st` already reads, so the Rust side needs
# no second parser.
DTYPES = {torch.float32: "f32", torch.int64: "i64", torch.int32: "i32"}


class Capture:
    """Collects named tensors and writes them out as a golden directory."""

    def __init__(self):
        self.tensors = {}

    def add(self, name, tensor):
        if name in self.tensors:
            raise RuntimeError(f"{name} captured twice; the hook fired more than once")
        self.tensors[name] = tensor.detach().cpu().contiguous()

    def write(self, out, meta):
        out.mkdir(parents=True, exist_ok=True)
        entries = {}
        for name, t in sorted(self.tensors.items()):
            dtype = DTYPES.get(t.dtype)
            if dtype is None:
                raise RuntimeError(f"{name} has unsupported dtype {t.dtype}")
            path = out / f"{name}.bin"
            arr = t.numpy()
            if arr.dtype.byteorder == ">":
                arr = arr.byteswap().view(arr.dtype.newbyteorder("<"))
            path.write_bytes(arr.tobytes(order="C"))
            entries[name] = {
                "file": path.name,
                "shape": list(t.shape),
                "dtype": dtype,
                "bytes": path.stat().st_size,
                # A checksum makes a truncated or half-written capture loud
                # instead of merely wrong.
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        meta["tensors"] = entries
        (out / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return entries


class Observer(torch.overrides.TorchFunctionMode):
    """Records the random draws and the duration-expansion matrix in flight.

    ``torch.randn`` is called exactly twice on the inference path - once for the
    stochastic duration predictor's latents, once for the prior - and the order
    is stable, so positional naming is safe. Pinning the *draws* rather than the
    seed is deliberate: two RNG implementations agreeing on a seed is not
    something to assume across languages.

    ``attn`` is never a module's input or output, so a hook cannot see it. It is
    recognised instead by what it is: a binary [1, F, T] matrix multiplied
    against something whose last dimension is the flow size. The text encoder's
    attention matmuls are not binary, so they do not collide.
    """

    def __init__(self, capture, flow_size):
        super().__init__()
        self.capture = capture
        self.flow_size = flow_size
        self.draws = 0
        self.attn_seen = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))

        name = getattr(func, "__name__", "")
        if name in ("randn", "randn_like"):
            self.draws += 1
            label = {1: "noise_dur", 2: "noise_prior"}.get(self.draws)
            if label is not None:
                self.capture.add(label, out)
        elif name == "matmul" and not self.attn_seen and len(args) >= 2:
            a, b = args[0], args[1]
            if (
                torch.is_tensor(a)
                and torch.is_tensor(b)
                and a.dim() == 3
                and b.dim() == 3
                and b.shape[-1] == self.flow_size
                and torch.all((a == 0) | (a == 1))
            ):
                self.capture.add("attn", a)
                self.attn_seen = True

        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="facebook/mms-tts-nan")
    ap.add_argument("--text", required=True)
    # The seed is required, not defaulted: a capture whose seed is unknown is
    # not an oracle.
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    # Threading changes float32 reduction order, which moves the last bits of
    # every tensor here. One thread makes the capture reproducible.
    torch.set_num_threads(1)

    tokenizer = VitsTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg = model.config

    inputs = tokenizer(text=args.text, return_tensors="pt")
    cap = Capture()
    cap.add("input_ids", inputs["input_ids"])

    handles = []

    def on(module, name, arg=None):
        """Captures a module's output, or - when `arg` is given - one input.

        Inputs arrive as keywords throughout this model, so a positional-only
        pre-hook sees an empty tuple; `with_kwargs=True` is not optional here.
        """
        if arg is None:

            def hook(_m, _i, _kw, out):
                cap.add(name, first_tensor(out))

            h = module.register_forward_hook(hook, with_kwargs=True)
        else:

            def hook(_m, inp, kw):
                cap.add(name, kw[arg] if arg in kw else inp[0])

            h = module.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(h)

    def first_tensor(out):
        """Unwraps the tensor from a tuple- or dataclass-valued forward."""
        if torch.is_tensor(out):
            return out
        return out[0]

    def on_fields(module, fields):
        """Captures several named fields of one module's output.

        The text encoder returns a dataclass carrying three tensors; taking them
        from the real forward pass rather than re-running the encoder keeps the
        capture a record of one execution instead of two.
        """

        def hook(_m, _i, _kw, out):
            for name, key in fields.items():
                cap.add(name, getattr(out, key))

        handles.append(module.register_forward_hook(hook, with_kwargs=True))

    te = model.text_encoder
    on_fields(te, {"m_p": "prior_means", "logs_p": "prior_log_variances"})
    on(te.embed_tokens, "embed_raw")
    on(te.encoder, "embed", arg="hidden_states")  # post sqrt(hidden_size) scaling
    on(te.encoder, "enc_out")
    on(model.duration_predictor, "log_duration")
    on(model.flow, "z_p", arg="inputs")
    on(model.flow, "z")
    on(model.decoder, "waveform_raw")

    # Per-layer encoder outputs are not required by any milestone, but they turn
    # "the text encoder is wrong" into "layer 3 is wrong" for free.
    for i, layer in enumerate(te.encoder.layers):
        on(layer, f"enc_layer_{i}")

    torch.manual_seed(args.seed)
    with Observer(cap, cfg.flow_size):
        out = model(**inputs)

    for h in handles:
        h.remove()

    cap.tensors["waveform"] = out.waveform.detach().cpu().contiguous()

    missing = {"noise_dur", "noise_prior", "attn"} - set(cap.tensors)
    if missing:
        # Silently missing a stage would make the oracle quietly weaker, so make
        # it fatal rather than a warning.
        raise RuntimeError(f"observer never saw: {sorted(missing)}")

    meta = {
        "model": args.model,
        "text": args.text,
        "seed": args.seed,
        "sampling_rate": cfg.sampling_rate,
        "noise_scale": model.noise_scale,
        "noise_scale_duration": model.noise_scale_duration,
        "speaking_rate": model.speaking_rate,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
    }
    entries = cap.write(args.out, meta)

    for name, e in entries.items():
        print(f"{name:16s} {str(e['shape']):24s} {e['dtype']:4s} {e['bytes']:>10,d} B")
    print(f"\nwrote {len(entries)} tensors to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
