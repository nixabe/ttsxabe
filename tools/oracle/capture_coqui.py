"""Capture the Coqui reference's intermediates as binary golden files.

The reference is Coqui TTS' own ``Vits`` on CPU in float32, reading
``neurlang/coqui-vits-suisiann-minnan-hokkien`` exactly as the model card says
to. It is the same architecture ``capture.py`` records for ``mms-tts-nan`` and
the same capture format, so ``xabe-golden`` reads both without changes - what
differs is the module tree the hooks attach to and the tokenizer in front of it.

Nothing here reimplements the model. Module-level tensors come from forward
hooks; the two random draws and the duration-expansion matrix come from a
``TorchFunctionMode`` that observes the calls as they happen.

**The phonemes are captured too, and that matters.** This checkpoint is trained
on IPA produced by ``pygoruut``, which is a Go binary with its own language
data. The Rust engine takes phonemes and does not produce them, so the string
this script phonemised is written into the manifest and the differential tests
feed exactly that back in. Without it the two sides would be speaking different
sentences and every stage would disagree for a reason that is not arithmetic.

Usage::

    .venv-coqui/bin/python tools/oracle/capture_coqui.py \\
        --out .golden/coqui --seed 0 --text "你好！我是蔡贏。"
"""

import argparse
import hashlib
import json
import pathlib
import sys

import torch

from TTS.config import load_config
from TTS.tts.models.vits import Vits

# Same convention as `capture.py`: raw, C order, little-endian, with shape and
# dtype in the manifest.
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
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        meta["tensors"] = entries
        (out / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return entries


class Observer(torch.overrides.TorchFunctionMode):
    """Records the random draws and the duration-expansion matrix in flight.

    ``torch.randn`` is called exactly twice on the inference path - once inside
    the stochastic duration predictor, once for the prior - and the order is
    stable, so positional naming is safe. Pinning the *draws* rather than the
    seed is deliberate: two RNG implementations agreeing on a seed is not
    something to assume across languages.

    ``attn`` is never a module's input or output, so a hook cannot see it. It is
    recognised instead by what it is: a binary matrix multiplied against
    something whose last dimension is the flow size.
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


def phonemize(model, text):
    """Runs the reference's own front end: clean, then phonemise.

    This is the half of the pipeline the Rust engine does not have. Splitting it
    out here - rather than letting ``text_to_ids`` do both invisibly - is what
    lets the manifest record the exact string the model was given.
    """
    tok = model.tokenizer
    cleaned = tok.text_cleaner(text) if tok.text_cleaner is not None else text
    if not tok.use_phonemes:
        return cleaned
    return tok.phonemizer.phonemize(cleaned, separator="", language=tok.phonemizer.language)


def stop_phonemizer(phonemizer):
    """Terminates the goruut subprocess, if this phonemiser started one."""
    inner = getattr(phonemizer, "pygoruut", None)
    process = getattr(inner, "process", None)
    if process is None:
        return
    process.terminate()
    process.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("models/tts/coqui-vits-suisiann"),
        help="directory holding best_model.pth and config.json",
    )
    ap.add_argument("--text", required=True)
    # The seed is required, not defaulted: a capture whose seed is unknown is
    # not an oracle.
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    # Threading changes float32 reduction order, which moves the last bits of
    # every tensor here. One thread makes the capture reproducible.
    torch.set_num_threads(1)

    config = load_config(str(args.model / "config.json"))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(args.model / "best_model.pth"), eval=True)
    model.cpu()

    phonemes = phonemize(model, args.text)
    # Stopped as soon as it has done its one job. `pygoruut` starts the goruut
    # binary with this process's stdout inherited, so leaving it running holds
    # the pipe open for anything that captures this script's output - and
    # `__del__` at interpreter shutdown is not reliable enough to prevent that.
    stop_phonemizer(model.tokenizer.phonemizer)
    ids = model.tokenizer.encode(phonemes)
    if model.tokenizer.add_blank:
        ids = model.tokenizer.intersperse_blank_char(ids, True)
    input_ids = torch.LongTensor(ids).unsqueeze(0)

    cap = Capture()
    cap.add("input_ids", input_ids)

    handles = []

    def on(module, name, index=None, seq_major=False):
        """Captures a module's output, or - when `index` is given - one input.

        ``seq_major`` transposes a ``[B, C, T]`` tensor to ``[B, T, C]``. The two
        references disagree about which way round the text encoder carries its
        activations - 🤗 works in ``[B, T, C]`` and Coqui in ``[B, C, T]`` - and
        `xabe-golden` has one convention per stage name, not two. Transposing
        here rather than in the test keeps the capture directly comparable to
        the 🤗 one, which is the whole point of sharing the format.

        Only the encoder's own activations need it. ``m_p``, ``logs_p``, ``z``
        and the waveform are ``[B, C, T]`` on both sides already, because they
        are produced by convolutions rather than by attention.
        """
        if index is None:

            def hook(_m, _i, _kw, out):
                cap.add(name, maybe_transpose(first_tensor(out), seq_major))

            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        else:

            def hook(_m, inp, _kw):
                cap.add(name, maybe_transpose(inp[index], seq_major))

            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def maybe_transpose(t, seq_major):
        return t.transpose(1, 2) if seq_major else t

    def first_tensor(out):
        """Unwraps the tensor from a tuple-valued forward."""
        if torch.is_tensor(out):
            return out
        return out[0]

    te = model.text_encoder

    # The text encoder returns `(x, m, logs, x_mask)`; taking all three from the
    # real forward pass keeps the capture a record of one execution rather than
    # of two.
    def on_text_encoder(_m, _i, _kw, out):
        # All three come out `[B, C, T]` here and `[B, T, C]` in the 🤗 capture,
        # whose text encoder transposes the projection's output back before
        # splitting it. Note this is the *unexpanded* prior: after the duration
        # expansion both references carry it as `[B, C, T]`, which is why `z_p`
        # and `z` below need no transpose.
        cap.add("enc_out", out[0].transpose(1, 2))
        cap.add("m_p", out[1].transpose(1, 2))
        cap.add("logs_p", out[2].transpose(1, 2))

    handles.append(te.register_forward_hook(on_text_encoder, with_kwargs=True))

    on(te.emb, "embed_raw")
    # The transformer's first input is the embedding after the sqrt(hidden)
    # scaling, the transpose and the mask.
    on(te.encoder, "embed", index=0, seq_major=True)
    on(model.duration_predictor, "log_duration")
    on(model.flow, "z_p", index=0)
    on(model.flow, "z")
    on(model.waveform_decoder, "waveform_raw")

    # Per-layer outputs are not required by any milestone, but they turn "the
    # text encoder is wrong" into "layer 3 is wrong" for free. The layer's
    # output is whatever its second norm produced.
    for i, norm in enumerate(te.encoder.norm_layers_2):
        on(norm, f"enc_layer_{i}", seq_major=True)

    torch.manual_seed(args.seed)
    with Observer(cap, model.args.hidden_channels):
        out = model.inference(input_ids)

    for h in handles:
        h.remove()

    cap.tensors["waveform"] = out["model_outputs"].detach().cpu().contiguous()
    cap.tensors["durations"] = out["durations"].detach().cpu().contiguous()

    missing = {"noise_dur", "noise_prior", "attn"} - set(cap.tensors)
    if missing:
        raise RuntimeError(f"observer never saw: {sorted(missing)}")

    meta = {
        "model": str(args.model),
        "dialect": "coqui",
        "text": args.text,
        # What the engine is actually fed. The Rust side reads this, not `text`.
        "phonemes": phonemes,
        "phonemizer": config.phonemizer,
        "phoneme_language": config.phoneme_language,
        "seed": args.seed,
        "sampling_rate": config.audio["sample_rate"],
        "noise_scale": model.inference_noise_scale,
        "noise_scale_duration": model.inference_noise_scale_dp,
        "length_scale": model.length_scale,
        # The reference multiplies durations by `length_scale`; this workspace
        # divides by `speaking_rate`. Both are written so the manifest reads the
        # same way for either dialect.
        "speaking_rate": 1.0 / model.length_scale,
        "coqui_tts": __import__("TTS").__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
    }
    entries = cap.write(args.out, meta)

    for name, e in entries.items():
        print(f"{name:16s} {str(e['shape']):24s} {e['dtype']:4s} {e['bytes']:>10,d} B")
    print(f"\nphonemes: {phonemes}")
    print(f"wrote {len(entries)} tensors to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
