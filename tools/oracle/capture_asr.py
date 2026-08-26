#!/usr/bin/env python3
"""Capture the Whisper oracle, stage by stage.

    python tools/oracle/capture_asr.py --model models/asr/breeze-asr-26 \
        --wav .golden/vad/clips/speech.wav --out .golden/asr/speech

The reference is 🤗 `WhisperForConditionalGeneration` on CPU in float32, not
whisper.cpp. Same discipline as the VITS work, and it sidesteps a real
divergence: whisper.cpp's tokenizer is a greedy longest-match over a
`std::regex` where `[[:alpha:]]` is not `\\p{L}`, so it already disagrees with
the reference on Han input. whisper-server's transcripts stay a cross-check,
not the definition of correct.

Everything is captured as raw little-endian f32, because a constant transcribed
into a test is a copy of what someone believed the model does. Per-layer
outputs are captured as well as the final ones, so a failure is located rather
than merely detected - "the encoder is wrong" becomes "layer 7 is wrong".
"""

import argparse
import json
import os
import wave

import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


def save(out_dir, name, tensor):
    a = np.ascontiguousarray(
        tensor.detach().cpu().numpy().astype("<f4")
        if torch.is_tensor(tensor)
        else np.asarray(tensor, dtype="<f4")
    )
    a.tofile(os.path.join(out_dir, f"{name}.bin"))
    print(f"  {name:<28} {list(a.shape)}")
    return list(a.shape)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16-bit mono"
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4,
                    help="how many encoder/decoder layers to capture individually")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # float32 reduction order is not thread-invariant, and the last bits of
    # every tensor move if this is not pinned.
    torch.set_num_threads(1)
    torch.manual_seed(0)

    samples, rate = read_wav(a.wav)
    assert rate == 16000, f"{a.wav} is {rate} Hz"

    fe = WhisperFeatureExtractor.from_pretrained(a.model)
    tok = WhisperTokenizer.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.float32
    ).eval()

    shapes = {}

    # The filter bank is not in the safetensors, so it is captured here and
    # stored beside the model - the same capture discipline as everything else.
    shapes["mel_filters"] = save(a.out, "mel_filters", np.asarray(fe.mel_filters))

    feats = fe(samples, sampling_rate=16000, return_tensors="pt")
    features = feats["input_features"].to(torch.float32)
    shapes["input_features"] = save(a.out, "input_features", features)
    shapes["samples"] = save(a.out, "samples", samples)

    # Decoder input: the forced prefix this checkpoint generates with, plus a
    # few real tokens so cross-attention and the causal mask are both exercised.
    forced = [model.config.decoder_start_token_id] + [
        t for _, t in (model.config.forced_decoder_ids or [])
    ]
    extra = tok.encode("今天天氣", add_special_tokens=False)[:4]
    ids = forced + extra
    with open(os.path.join(a.out, "decoder_ids.json"), "w") as f:
        json.dump(ids, f)
    print(f"  decoder_ids                  {ids}")

    taps = {}

    def tap(name):
        # A layer hook hands back a tuple; a whole encoder or decoder hands back
        # a ModelOutput. Both carry the hidden states first, but not by the same
        # accessor, and getting it wrong here captures a struct rather than a
        # tensor - which is at least loud.
        def hook(_m, _i, output):
            if hasattr(output, "last_hidden_state"):
                taps[name] = output.last_hidden_state
            elif isinstance(output, tuple):
                taps[name] = output[0]
            else:
                taps[name] = output
        return hook

    enc = model.model.encoder
    dec = model.model.decoder
    handles = [enc.register_forward_hook(tap("encoder_out"))]
    for i in range(min(a.layers, len(enc.layers))):
        handles.append(enc.layers[i].register_forward_hook(tap(f"encoder_layer_{i}")))
    for i in range(min(a.layers, len(dec.layers))):
        handles.append(dec.layers[i].register_forward_hook(tap(f"decoder_layer_{i}")))
    handles.append(dec.register_forward_hook(tap("decoder_out")))

    with torch.no_grad():
        out = model(
            input_features=features,
            decoder_input_ids=torch.tensor([ids], dtype=torch.long),
            use_cache=False,
        )

    for h in handles:
        h.remove()

    for name, t in sorted(taps.items()):
        shapes[name] = save(a.out, name, t)
    shapes["logits"] = save(a.out, "logits", out.logits)

    # A greedy transcript, as the end-to-end check.
    with torch.no_grad():
        generated = model.generate(
            input_features=features,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False,
            language="zh",
            task="transcribe",
        )
    text = tok.decode(generated[0], skip_special_tokens=True)
    print(f"  transcript                   {text!r}")

    manifest = {
        "model": os.path.basename(os.path.abspath(a.model)),
        "wav": os.path.basename(a.wav),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 1,
        "decoder_ids": ids,
        "generated_ids": generated[0].tolist(),
        "transcript": text,
        "shapes": shapes,
        "config": {
            k: getattr(model.config, k)
            for k in (
                "d_model", "encoder_layers", "decoder_layers",
                "encoder_attention_heads", "decoder_attention_heads",
                "encoder_ffn_dim", "decoder_ffn_dim", "num_mel_bins",
                "vocab_size", "max_source_positions", "max_target_positions",
                "decoder_start_token_id", "eos_token_id", "pad_token_id",
            )
        },
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
