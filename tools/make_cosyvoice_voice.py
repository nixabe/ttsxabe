#!/usr/bin/env python3
"""Derives a speaker bundle from one reference clip, once.

# Why the engine does not do this itself

Everything here comes from two ONNX models - `campplus.onnx` for the speaker
embedding and `speech_tokenizer_v3.onnx` for the prompt's speech tokens - plus
a mel frontend that belongs to neither of them. Porting all three would be a
second engine's worth of work for something that runs **once per voice**, never
per utterance, and whose output is four small tensors.

So they are computed here and written down. The engine reads the bundle and
never sees an ONNX runtime.

# What is in it, and what each one does

- `flow_prompt_speech_token` - the reference clip as speech tokens. Prepended
  to the generated tokens so the flow starts from the speaker's own voice.
- `prompt_speech_feat` - the reference clip's mel, `[frames, 80]`. The flow's
  condition for the stretch of timeline the prompt covers, and its length is
  where the generated mel begins.
- `flow_embedding` / `llm_embedding` - the campplus x-vector, `[192]`. The same
  numbers under both names, kept apart because upstream does and because a
  future checkpoint could reasonably diverge them.
- `cfm_noise` - the diffusion's starting noise, `[80, 15000]`. Not a property
  of the speaker at all: `CausalConditionalCFM.__init__` seeds the global RNG
  to 0 and draws it, so it is the same for every voice and every utterance and
  is **load-bearing** - a different draw is a different mel. It ships in the
  bundle because that is the file the engine already opens.

`sine_waves`, the vocoder's dither, is deliberately *not* here. It is 300
seconds of `torch.rand` taken from the global RNG at construction, so upstream
does not reproduce it across load orderings either; the engine draws its own.
See `crates/xabe-cosy/src/source.rs`.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np
import torch


def write_safetensors(path: pathlib.Path, tensors: dict) -> None:
    """A minimal float32 safetensors writer.

    Written here rather than pulled in as a dependency: the format is a JSON
    header and a blob, this writes one dtype, and `xabe-st` is the reader it
    has to satisfy.
    """
    import json

    header, blob, at = {}, bytearray(), 0
    for name in sorted(tensors):
        a = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32))
        header[name] = {
            "dtype": "F32",
            "shape": list(a.shape),
            "data_offsets": [at, at + a.nbytes],
        }
        blob += a.tobytes()
        at += a.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(raw)) % 8
    path.write_bytes(struct.pack("<Q", len(raw) + pad) + raw + b" " * pad + bytes(blob))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/tts/cosyvoice3-0.5b")
    ap.add_argument("--speaker", required=True, help="reference wav, any rate")
    ap.add_argument("--out", required=True, help="where to write the bundle")
    a = ap.parse_args()

    from cosyvoice.cli.cosyvoice import AutoModel

    # The same seed the capture pins, for the same reason: the buffers drawn
    # during construction come from the global RNG.
    torch.manual_seed(1986)
    m = AutoModel(model_dir=a.model_dir, fp16=False)
    fe = m.frontend

    feat, _ = fe._extract_speech_feat(a.speaker)
    token, _ = fe._extract_speech_token(a.speaker)
    # `frontend_zero_shot`'s own trim: at 24 kHz the flow wants exactly two mel
    # frames per speech token, and the two extractors do not agree on length to
    # the frame. Dropping the remainder here is what upstream does, and doing
    # it anywhere later would leave a bundle whose two halves disagree.
    n = min(int(feat.shape[1] / 2), token.shape[1])
    feat, token = feat[:, : 2 * n], token[:, :n]
    embedding = fe._extract_spk_embedding(a.speaker)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(
        out,
        {
            "flow_prompt_speech_token": token[0].float().cpu().numpy(),
            "prompt_speech_feat": feat[0].float().cpu().numpy(),
            "flow_embedding": embedding[0].float().cpu().numpy(),
            "llm_embedding": embedding[0].float().cpu().numpy(),
            "cfm_noise": m.model.flow.decoder.rand_noise[0].float().cpu().numpy(),
        },
    )
    print(
        f"wrote {out}: {token.shape[1]} prompt tokens, {feat.shape[1]} mel frames, "
        f"{embedding.shape[1]}-wide embedding"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
