#!/usr/bin/env python3
"""Convert the Taiwanese Tacotron2 + WaveGlow checkpoints to safetensors.

The checkpoints, and the class definitions `--glow-src` points at, come from
yfliao/taiwanese_tonal_tlpa_tacotron2 (BSD-3-Clause, itself derived from
NVIDIA's Tacotron2 and WaveGlow). See NOTICE.

    python tools/convert_tacotron2.py \
        --tacotron /path/to/tacotron2/model/checkpoint_100000 \
        --waveglow /path/to/tacotron2/model/waveglow/waveglow_main.pt \
        --glow-src /path/to/tacotron2/waveglow \
        --out      models/tts/tacotron2-nan

Writes `tacotron2.safetensors`, `waveglow.safetensors` and `tacotron2.json`
into `--out`. Nothing is renamed, fused or reshaped: a converter that
"helpfully" rearranges is a converter whose output cannot be diffed against the
source. Every tensor written is round-tripped and required to be bit-identical.

One thing *is* dropped, and it is the only decision in the file. Each of the
eight `BatchNorm1d` layers carries a `num_batches_tracked` scalar, which
PyTorch uses to form a cumulative moving average while training and never reads
at inference - it is not even consulted to normalise, since `running_mean` and
`running_var` already hold the result. It is also `int64`, and `xabe-st`
refuses a file holding a dtype it cannot map to `f32` rather than silently
skipping tensors. Keeping eight dead integers would mean loosening that.

# Why this checkpoint needs a converter at all, and the other ones do not

`xabe-st` reads safetensors and `xabe-gguf` reads GGUF, and every other stage in
this workspace ships as one of those. This pair ships as neither.

The Tacotron2 half is only mildly awkward: a modern torch zip archive holding a
plain `state_dict`, which `weights_only=True` reads without executing anything.

**The WaveGlow half cannot be read without PyTorch, and that is not a choice.**
It is the legacy pre-1.6 torch format (`protocol_version: 1001`), and its
`['model']` entry is a *pickled `nn.Module` object graph* rather than a tensor
dict - it will not unpickle at all without the real `glow.WaveGlow`, `glow.WN`
and `glow.Invertible1x1Conv` class definitions. That is what `--glow-src` is
for, and it is why loading it requires `weights_only=False`.

`weights_only=False` executes the pickle. Point this at a checkpoint you trust
and nothing else; the reason it is unavoidable here is stated above rather than
waved at.

So this engine's weights are *converted*, not read directly, unlike every other
stage. That is a real dent in the claim `AGENTS.md` opens with and it belongs
written down rather than quietly true.

# The symbol table, and why it is emitted here

`tacotron2/text/symbols.py` builds a 71-entry alphabet - pad, `-`, `!,.:;? `,
A-Za-z, 0-9 - and that is the whole input vocabulary: tonal TLPA, where the tone
is a trailing ASCII digit. The same file also *defines* 20,950 Han characters,
an ARPAbet set and a Taiwanese initial/final set, and then exports none of them.

That matters more than it looks. `text_to_sequence` drops every symbol outside
the table without a word, so Han fed to this model tokenises to the empty
sequence and it synthesises near-silence rather than failing - the exact
silent-wrong-audio class `AGENTS.md` was written against. The table is written
into `tacotron2.json` so the Rust tokeniser reads it instead of reconstructing
it, and it is cross-checked against `embedding.weight.shape[0]`: a symbol table
that disagrees with the embedding is a bug that ships as plausible speech.

# Shapes are checked, not copied

Both state dicts are validated against the geometry in `tacotron2.json` before
anything is written, naming the tensor that disagreed. The WaveGlow per-flow
channel schedule (8,8,8,8,6,6,6,6,4,4,4,4 as early outputs are split off) is
derived from `n_flows`/`n_group`/`n_early_every`/`n_early_size` and checked
against `convinv.k` and `WN.k.start`, so a checkpoint trained with a different
schedule fails here rather than in a kernel.
"""

import argparse
import json
import pathlib
import sys
import warnings

import torch
from safetensors.torch import load_file, save_file

# Written while training, unread at inference, and `int64` - see the module
# docstring. Matched by suffix because every BatchNorm has one.
DROP_SUFFIX = ".num_batches_tracked"

# Mirrors tacotron2/text/symbols.py's export line exactly. The commented-out
# `_chinese`, `_arpabet` and `_iniFin` sets in that file are not exported and
# are not part of the model's vocabulary.
PAD = "_"
SPECIAL = "-"
PUNCTUATION = "!,.:;? "
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
SYMBOLS = [PAD] + list(SPECIAL) + list(PUNCTUATION) + list(LETTERS) + list(DIGITS)


def config() -> dict:
    """The geometry both halves are validated against and the Rust side reads.

    Values come from `tacotron2/hparams.py` and `waveglow/config.json`, except
    `max_decoder_steps`: the file says 2000 and `han2tts.py` raises it to 3000
    before every synthesis, so 3000 is what inference actually uses.
    """
    return {
        "symbols": SYMBOLS,
        "n_symbols": len(SYMBOLS),
        # `basic_cleaners` is lowercase + collapse whitespace, and
        # `_should_keep_symbol` drops anything unknown *and* the pad, so id 0
        # is never emitted by the tokeniser - only by the length padding.
        "tokenizer": {
            "cleaner": "basic_cleaners",
            "lowercase": True,
            "collapse_whitespace": True,
            "drop_unknown": True,
            "pad_id": 0,
            "pad_never_emitted": True,
        },
        "audio": {
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "max_wav_value": 32768.0,
        },
        "encoder": {
            "symbols_embedding_dim": 512,
            "encoder_embedding_dim": 512,
            "encoder_kernel_size": 5,
            "encoder_n_convolutions": 3,
            "lstm_hidden": 256,
        },
        "decoder": {
            "n_frames_per_step": 1,
            "prenet_dim": 256,
            "attention_rnn_dim": 1024,
            "decoder_rnn_dim": 1024,
            "attention_dim": 128,
            "attention_location_n_filters": 32,
            "attention_location_kernel_size": 31,
            "gate_threshold": 0.5,
            "max_decoder_steps": 3000,
        },
        "postnet": {
            "postnet_embedding_dim": 512,
            "postnet_kernel_size": 5,
            "postnet_n_convolutions": 5,
        },
        "waveglow": {
            "n_flows": 12,
            "n_group": 8,
            "n_early_every": 4,
            "n_early_size": 2,
            "n_layers": 8,
            "n_channels": 256,
            "kernel_size": 3,
            "sigma": 0.666,
            "denoiser_strength": 0.01,
        },
    }


def flow_channels(cfg: dict) -> list:
    """Channels surviving into each flow, as early outputs are split off.

    Reproduces glow.py's loop: every `n_early_every` flows after the first,
    `n_early_size` channels leave the stack. For this checkpoint that is
    8,8,8,8,6,6,6,6,4,4,4,4 - verified against `convinv.k.conv.weight`.
    """
    w = cfg["waveglow"]
    remaining = w["n_group"]
    out = []
    for k in range(w["n_flows"]):
        if k % w["n_early_every"] == 0 and k > 0:
            remaining -= w["n_early_size"]
        out.append(remaining)
    return out


def expect(sd: dict, key: str, shape: tuple, where: str) -> None:
    """Rejects a geometry mismatch by name, before anything is written."""
    if key not in sd:
        raise SystemExit(f"{where}: {key} is missing")
    got = tuple(sd[key].shape)
    if got != shape:
        raise SystemExit(f"{where}: {key} is {got}, expected {shape}")


def check_tacotron(sd: dict, cfg: dict) -> None:
    e, d, p = cfg["encoder"], cfg["decoder"], cfg["postnet"]
    emb, enc = e["symbols_embedding_dim"], e["encoder_embedding_dim"]
    w = "tacotron2"

    expect(sd, "embedding.weight", (cfg["n_symbols"], emb), w)

    for i in range(e["encoder_n_convolutions"]):
        expect(sd, f"encoder.convolutions.{i}.0.conv.weight",
               (enc, enc, e["encoder_kernel_size"]), w)
        expect(sd, f"encoder.convolutions.{i}.1.weight", (enc,), w)

    h = e["lstm_hidden"]
    for suffix in ("", "_reverse"):
        expect(sd, f"encoder.lstm.weight_ih_l0{suffix}", (4 * h, enc), w)
        expect(sd, f"encoder.lstm.weight_hh_l0{suffix}", (4 * h, h), w)

    expect(sd, "decoder.prenet.layers.0.linear_layer.weight",
           (d["prenet_dim"], cfg["audio"]["n_mel_channels"] * d["n_frames_per_step"]), w)
    expect(sd, "decoder.prenet.layers.1.linear_layer.weight",
           (d["prenet_dim"], d["prenet_dim"]), w)

    # The two LSTMCells are the whole autoregressive loop. Their input widths
    # are what pins the wiring: prenet+context, then attention state+context.
    expect(sd, "decoder.attention_rnn.weight_ih",
           (4 * d["attention_rnn_dim"], d["prenet_dim"] + enc), w)
    expect(sd, "decoder.attention_rnn.weight_hh",
           (4 * d["attention_rnn_dim"], d["attention_rnn_dim"]), w)
    expect(sd, "decoder.decoder_rnn.weight_ih",
           (4 * d["decoder_rnn_dim"], d["attention_rnn_dim"] + enc), w)
    expect(sd, "decoder.decoder_rnn.weight_hh",
           (4 * d["decoder_rnn_dim"], d["decoder_rnn_dim"]), w)

    a = d["attention_dim"]
    expect(sd, "decoder.attention_layer.query_layer.linear_layer.weight",
           (a, d["attention_rnn_dim"]), w)
    expect(sd, "decoder.attention_layer.memory_layer.linear_layer.weight", (a, enc), w)
    expect(sd, "decoder.attention_layer.v.linear_layer.weight", (1, a), w)
    # Two channels in: the cumulative and the previous attention weights.
    expect(sd, "decoder.attention_layer.location_layer.location_conv.conv.weight",
           (d["attention_location_n_filters"], 2, d["attention_location_kernel_size"]), w)
    expect(sd, "decoder.attention_layer.location_layer.location_dense.linear_layer.weight",
           (a, d["attention_location_n_filters"]), w)

    mel = cfg["audio"]["n_mel_channels"]
    expect(sd, "decoder.linear_projection.linear_layer.weight",
           (mel * d["n_frames_per_step"], d["decoder_rnn_dim"] + enc), w)
    expect(sd, "decoder.gate_layer.linear_layer.weight",
           (d["n_frames_per_step"], d["decoder_rnn_dim"] + enc), w)

    pe, pk, pn = p["postnet_embedding_dim"], p["postnet_kernel_size"], p["postnet_n_convolutions"]
    for i in range(pn):
        cin = mel if i == 0 else pe
        cout = mel if i == pn - 1 else pe
        expect(sd, f"postnet.convolutions.{i}.0.conv.weight", (cout, cin, pk), w)


def check_waveglow(sd: dict, cfg: dict) -> None:
    a, g = cfg["audio"], cfg["waveglow"]
    mel, n_ch, w = a["n_mel_channels"], g["n_channels"], "waveglow"

    # ConvTranspose1d(80, 80, filter_length, stride=hop_length): mel to samples.
    expect(sd, "upsample.weight", (mel, mel, a["filter_length"]), w)

    for k, remaining in enumerate(flow_channels(cfg)):
        half = remaining // 2
        expect(sd, f"convinv.{k}.conv.weight", (remaining, remaining, 1), w)
        expect(sd, f"WN.{k}.start.weight_v", (n_ch, half, 1), w)
        # The end layer alone carries no weight norm, so it is `weight`.
        expect(sd, f"WN.{k}.end.weight", (2 * half, n_ch, 1), w)
        expect(sd, f"WN.{k}.cond_layer.weight_v",
               (2 * n_ch * g["n_layers"], mel * g["n_group"], 1), w)
        for i in range(g["n_layers"]):
            expect(sd, f"WN.{k}.in_layers.{i}.weight_v",
                   (2 * n_ch, n_ch, g["kernel_size"]), w)
            last = i == g["n_layers"] - 1
            expect(sd, f"WN.{k}.res_skip_layers.{i}.weight_v",
                   (n_ch if last else 2 * n_ch, n_ch, 1), w)


def write(sd: dict, dst: pathlib.Path, source: str) -> dict:
    out, dtypes, dropped = {}, {}, 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            raise SystemExit(f"{source}: {k!r} is a {type(v).__name__}, not a tensor")
        if k.endswith(DROP_SUFFIX):
            dropped += 1
            continue
        # `.contiguous()` because safetensors stores a flat buffer, and a
        # non-contiguous view is written in the wrong order - silently, since
        # the shape is still right.
        out[k] = v.contiguous()
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + v.numel()
    meta = {"format": "pt", "source": source}
    if dropped:
        meta["dropped"] = f"{dropped} x {DROP_SUFFIX}: written while training, unread at inference"
    save_file(out, str(dst), metadata=meta)
    total = sum(dtypes.values())
    print(f"  {source} -> {dst.name}: {len(out)} tensors, {total:,} params, "
          f"{total * 4 / 1e6:.1f} MB f32, {dropped} dropped, {dtypes}")
    return out


def verify(original: dict, dst: pathlib.Path) -> None:
    """Requires every tensor to survive bit-identically.

    The conversion is a copy, so anything short of equality is a bug - and a
    dtype or contiguity mistake produces a file that loads fine and holds the
    wrong numbers, which is the failure this is here to catch.
    """
    back = load_file(str(dst))
    bad = [k for k, v in original.items()
           if k not in back or not torch.equal(v, back[k])]
    if bad:
        for k in bad[:10]:
            print(f"    {'MISSING' if k not in back else 'DIFFERS'} {k}")
        raise SystemExit(f"{dst.name}: {len(bad)} tensors did not survive the round trip")
    print(f"  {dst.name}: {len(back)} tensors verified bit-identical")


def load_tacotron(path: pathlib.Path) -> dict:
    """A modern zip archive holding a plain `state_dict`; no pickle executed."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in ck:
        raise SystemExit(f"{path.name}: no 'state_dict' key (found {list(ck)})")
    print(f"  {path.name}: iteration {ck.get('iteration')}, "
          f"{len(ck['state_dict'])} tensors (optimizer state not carried over)")
    return ck["state_dict"]


def load_waveglow(path: pathlib.Path, glow_src: pathlib.Path) -> dict:
    """Unpickles the `nn.Module` object graph. See the module docstring."""
    if not (glow_src / "glow.py").is_file():
        raise SystemExit(f"--glow-src {glow_src} has no glow.py; "
                         "the checkpoint cannot be unpickled without it")
    sys.path.insert(0, str(glow_src))
    # The saved class source differs from today's torch. That is expected for a
    # pre-1.6 checkpoint and says nothing about the weights.
    warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model" not in obj:
        raise SystemExit(f"{path.name}: expected a dict with a 'model' key")
    sd = obj["model"].state_dict()
    print(f"  {path.name}: unpickled {type(obj['model']).__name__}, {len(sd)} tensors")
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Tacotron2 + WaveGlow checkpoints to safetensors.")
    ap.add_argument("--tacotron", required=True, type=pathlib.Path,
                    help="Tacotron2 checkpoint (a torch zip archive with a state_dict)")
    ap.add_argument("--waveglow", required=True, type=pathlib.Path,
                    help="WaveGlow checkpoint (legacy torch format, a pickled nn.Module)")
    ap.add_argument("--glow-src", required=True, type=pathlib.Path,
                    help="Directory holding glow.py, needed to unpickle --waveglow")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--skip-verify", action="store_true")
    a = ap.parse_args()

    for p in (a.tacotron, a.waveglow):
        if not p.is_file():
            raise SystemExit(f"{p} is missing")

    cfg = config()
    a.out.mkdir(parents=True, exist_ok=True)

    print("reading:")
    taco = load_tacotron(a.tacotron)
    glow = load_waveglow(a.waveglow, a.glow_src)

    print("\nchecking geometry:")
    check_tacotron(taco, cfg)
    check_waveglow(glow, cfg)
    print(f"  tacotron2: {len(taco)} tensors match the declared geometry")
    print(f"  waveglow : {len(glow)} tensors match, "
          f"flow channels {flow_channels(cfg)}")

    print("\nwriting:")
    w1 = write(taco, a.out / "tacotron2.safetensors", a.tacotron.name)
    w2 = write(glow, a.out / "waveglow.safetensors", a.waveglow.name)

    if not a.skip_verify:
        print("\nverifying:")
        verify(w1, a.out / "tacotron2.safetensors")
        verify(w2, a.out / "waveglow.safetensors")

    (a.out / "tacotron2.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"\nwrote {a.out}/tacotron2.json ({cfg['n_symbols']} symbols)")


if __name__ == "__main__":
    main()
