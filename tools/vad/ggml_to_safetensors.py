#!/usr/bin/env python3
"""Convert whisper.cpp's Silero VAD checkpoint to safetensors.

    python tools/vad/ggml_to_safetensors.py \
        models/vad/ggml-silero-v5.1.2.bin models/vad/silero-v5.1.2.safetensors

The VAD ships as legacy ggml, not safetensors: 864 KB with the magic `ggml`
and a hand-rolled header. Teaching the workspace a second container format for
one 15-tensor model would be a lot of parser for very little model, so it is
converted once here instead and `xabe-st` keeps its single job.

The format, read out of `whisper_vad_init_with_params_no_state` in
whisper.cpp:

    u32   magic 0x67676d6c
    u32   len, then `len` bytes of model type
    i32   major, minor, patch
    i32   n_window, n_context
    i32   n_encoder_layers
          per layer: i32 in_channels, out_channels, kernel_size
    i32   lstm_input_size, lstm_hidden_size, final_conv_in, final_conv_out
    then, until EOF:
    i32   n_dims, name_length, dtype
    i32   ne[n_dims]        -- ggml order: ne[0] varies fastest
          name_length bytes of name
          the data

`ne` is reversed on the way out, so the safetensors shape reads the way the
rest of the workspace expects: outermost dimension first.

Everything the header declares is written into the safetensors `__metadata__`
rather than dropped, because the geometry has to be checked against the tensors
somewhere and a schema that reads its own dimensions from the file cannot
disagree with it.
"""

import json
import struct
import sys

MAGIC = 0x67676D6C
# ggml type ids. The VAD checkpoint is entirely F32; F16 is listed so an
# unexpected file fails by name rather than by producing nonsense.
DTYPES = {0: ("F32", 4), 1: ("F16", 2)}


class Reader:
    def __init__(self, data):
        self.d = data
        self.at = 0

    def take(self, n):
        if self.at + n > len(self.d):
            raise SystemExit(
                f"file ends after {len(self.d)} bytes; "
                f"needed {n} more at offset {self.at}"
            )
        out = self.d[self.at:self.at + n]
        self.at += n
        return out

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def eof(self):
        return self.at >= len(self.d)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <in.bin> <out.safetensors>")
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        r = Reader(f.read())

    magic = r.u32()
    if magic != MAGIC:
        raise SystemExit(f"{src}: magic is {magic:#x}, expected {MAGIC:#x}")

    model_type = r.take(r.u32()).decode("utf-8")
    version = f"{r.i32()}.{r.i32()}.{r.i32()}"
    n_window = r.i32()
    n_context = r.i32()

    n_layers = r.i32()
    layers = [
        {"in": r.i32(), "out": r.i32(), "kernel": r.i32()}
        for _ in range(n_layers)
    ]
    lstm_input_size = r.i32()
    lstm_hidden_size = r.i32()
    final_conv_in = r.i32()
    final_conv_out = r.i32()

    print(f"type    {model_type}  version {version}")
    print(f"window  {n_window} samples   context {n_context}")
    for i, l in enumerate(layers):
        print(f"encoder {i}: {l['in']:>3} -> {l['out']:>3}  k={l['kernel']}")
    print(f"lstm    {lstm_input_size} -> {lstm_hidden_size}")
    print(f"final   {final_conv_in} -> {final_conv_out}")

    tensors = {}
    order = []
    while not r.eof():
        n_dims = r.i32()
        name_len = r.i32()
        dtype = r.i32()
        if not 0 <= n_dims <= 4:
            raise SystemExit(f"n_dims {n_dims} at offset {r.at - 12}")
        if dtype not in DTYPES:
            raise SystemExit(f"unsupported ggml dtype {dtype}")
        ne = [r.i32() for _ in range(n_dims)]
        name = r.take(name_len).decode("utf-8")

        name_st, width = DTYPES[dtype]
        count = 1
        for d in ne:
            count *= d
        data = r.take(count * width)

        # ggml stores ne[0] as the fastest-varying dimension; safetensors and
        # the rest of this workspace put the outermost dimension first.
        shape = list(reversed(ne))
        tensors[name] = {"dtype": name_st, "shape": shape, "data": data}
        order.append(name)
        print(f"  {name:<28} {name_st} {shape}")

    expected = n_layers * 2 + 4 + 2 + 1
    if len(tensors) != expected:
        raise SystemExit(
            f"read {len(tensors)} tensors, expected {expected} "
            f"({n_layers} encoder layers x2, 4 lstm, 2 final, 1 stft)"
        )

    header = {
        "__metadata__": {
            "model_type": model_type,
            "version": version,
            "n_window": str(n_window),
            "n_context": str(n_context),
            "lstm_input_size": str(lstm_input_size),
            "lstm_hidden_size": str(lstm_hidden_size),
            "final_conv_in": str(final_conv_in),
            "final_conv_out": str(final_conv_out),
            "encoder_layers": json.dumps(layers, separators=(",", ":")),
            "converted_from": src.split("/")[-1],
        }
    }
    offset = 0
    for name in order:
        t = tensors[name]
        header[name] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, offset + len(t["data"])],
        }
        offset += len(t["data"])

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors does not require an aligned data segment and nothing forces a
    # producer to pad, but xabe-st refuses an unaligned f32 segment rather than
    # casting bytes it cannot prove are aligned - so pad the header to 8.
    pad = (-len(blob)) % 8
    blob += b" " * pad

    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in order:
            f.write(tensors[name]["data"])

    print(f"\nwrote {dst}: {len(tensors)} tensors, {offset} bytes of weights")


if __name__ == "__main__":
    main()
