"""Times the PyTorch reference on the same card, for milestone 12.

The comparison milestone 12 asks for is only meaningful if both sides are
measured the same way, so this mirrors the Rust benchmark exactly: same text,
same device, warm-up runs discarded, median of the timed runs reported, and the
model load excluded from all of them.

CUDA is asynchronous, so every timed region ends with `torch.cuda.synchronize()`.
Without it this measures how fast PyTorch can enqueue work, which on a model
this small is most of what there is to measure.
"""

import argparse
import statistics
import time

import torch
from transformers import VitsModel, VitsTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="facebook/mms-tts-nan")
    ap.add_argument("--text", default="lí hó, kin-á-ji̍t thinn-khì chin hó.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    # `torch.backends.cudnn.benchmark = True` is the obvious thing to enable
    # here and it is left OFF deliberately: measured, it makes this model 13x
    # slower. Autotuning caches per input shape, and every utterance has a
    # different frame count because the durations are sampled - so it re-tunes
    # on every call and never reuses anything. See WHY NOT in
    # docs/BENCHMARKS.md.
    #
    # TF32 is an Ampere feature and this card is Turing, so there is nothing to
    # enable there either. Default settings are PyTorch's best settings for this
    # workload.
    tok = VitsTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model, dtype=torch.float32)
    model = model.to(args.device).eval()
    inputs = tok(text=args.text, return_tensors="pt").to(args.device)

    for _ in range(args.warmup):
        model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = model(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    samples = out.waveform.shape[-1]
    rate = model.config.sampling_rate
    median = statistics.median(times)
    print(f"device        {args.device}")
    print(f"text          {args.text!r}")
    print(f"samples       {samples} ({samples / rate:.2f} s at {rate} Hz)")
    print(f"runs          {args.runs} after {args.warmup} warm-up")
    print(f"median        {median:.2f} ms")
    print(f"min / max     {min(times):.2f} / {max(times):.2f} ms")
    print(f"realtime x    {samples / rate * 1000.0 / median:.1f}")


if __name__ == "__main__":
    main()
