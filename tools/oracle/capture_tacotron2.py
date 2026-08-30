#!/usr/bin/env python3
"""Capture the Tacotron2 encoder from the reference, for the differential test.

    python tools/oracle/capture_tacotron2.py \
        --src /path/to/taiwanese_tonal_tlpa_tacotron2/tacotron2 \
        --text "gua2 si7 tai5-uan5-lang5" \
        --out .golden/tacotron2/nan

# Why only the encoder

The rest of this model cannot be compared tensor for tensor against anything,
including a second run of itself. `Prenet.forward` passes `training=True` to
`F.dropout` unconditionally, so every decoder step multiplies by a fresh random
mask; WaveGlow then starts from Gaussian noise. Both are the model rather than
an artefact - the prenet noise is what stops the decoder learning to copy its
own last frame - and a comparison would have to replay the reference's draws,
which means capturing them, which means patching the reference.

The encoder has no such thing. Its dropout is conditioned on training mode and
is therefore absent at inference, so embedding, three convolutions with their
batch norms, and one bidirectional LSTM are a deterministic function of the
token ids. That is also where the engine's risk is concentrated: it is the only
recurrence in the workspace, and the only batch norm, and getting either the
gate order or the direction-concatenation order wrong produces speech that is
merely wrong rather than absent.

Captured on CPU in float32 with one thread, like every other capture here:
float32 reduction order is not thread-invariant, and a GPU capture would fold
PyTorch's kernel choices into the definition of correct.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=pathlib.Path,
                    help="the reference tacotron2 package directory")
    ap.add_argument("--checkpoint", type=pathlib.Path,
                    help="defaults to <src>/model/checkpoint_100000")
    ap.add_argument("--text", required=True, help="Tâi-lô with numeric tones")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    src = a.src.resolve()
    sys.path.insert(0, str(src))
    from hparams import create_hparams          # noqa: E402
    from model import Tacotron2                 # noqa: E402
    from text import text_to_sequence           # noqa: E402

    ck = a.checkpoint or (src / "model" / "checkpoint_100000")
    hp = create_hparams()
    hp.fp16_run = False

    model = Tacotron2(hp)
    model.load_state_dict(
        torch.load(ck, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()

    ids = text_to_sequence(a.text, ["basic_cleaners"])
    # han2tts.py's own guard: the encoder convolutions are five wide.
    while len(ids) < hp.encoder_kernel_size:
        ids.append(0)
    seq = torch.tensor(ids, dtype=torch.int64)[None, :]

    embedded = model.embedding(seq).transpose(1, 2)
    memory = model.encoder.inference(embedded)   # [1, tokens, 512]
    memory = memory[0].contiguous().numpy().astype(np.float32)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "encoder.bin").write_bytes(memory.tobytes())
    (a.out / "encoder.json").write_text(json.dumps({
        "text": a.text,
        "ids": ids,
        "shape": list(memory.shape),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "checkpoint": ck.name,
    }, indent=1) + "\n")

    print(f"tokens {len(ids)}  memory {memory.shape}  "
          f"absmax {np.abs(memory).max():.6f}")
    print(f"wrote {a.out}/encoder.bin and encoder.json")


if __name__ == "__main__":
    main()
