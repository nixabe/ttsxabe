#!/usr/bin/env python3
"""Build the VAD clip corpus, deterministically.

    python tools/oracle/vad_corpus.py .golden/vad/clips

Everything is generated from a fixed seed so the corpus can be rebuilt byte for
byte on another machine, which is what lets the captured probabilities be
compared at all. Real speech is not generated here - `capture_vad.sh`
synthesises it with the engine itself, at a fixed seed, and drops it alongside.

The first four clips are the pipeline's four known hallucination triggers. Each
one is a case where Whisper invented a sentence out of nothing and the assistant
answered it:

    silence   digital silence, transcribed as "我…"
    hiss      faint broadband noise, as "我現在在醫院"
    room      low-frequency room tone, as "(我會陪你一起走)"
    click     a single transient, which used to open a turn on its own

The VAD is what stops all four, so a VAD that disagrees with the reference on
these is a VAD that lets the pipeline hallucinate again.
"""

import os
import struct
import sys
import wave

import numpy as np

RATE = 16_000
SEED = 20250827


def write(path, x):
    """Writes float samples in [-1, 1] as 16-bit mono."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    print(f"  {os.path.basename(path):<24} {len(x) / RATE:5.2f} s")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else ".golden/vad/clips"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    p = lambda name: os.path.join(out, name)

    n = RATE * 4

    # 1. Digital silence: exactly zero, which is not a thing a microphone ever
    #    produces and which the ASR is least equipped to refuse.
    write(p("silence.wav"), np.zeros(n, dtype=np.float32))

    # 2. Faint broadband hiss, below any sane speech threshold.
    write(p("hiss.wav"), (rng.standard_normal(n) * 0.002).astype(np.float32))

    # 3. Room tone: low-frequency rumble plus mains hum plus a little hiss.
    #    This is the one that sat *above* the old VAD_START of 0.012.
    t = np.arange(n) / RATE
    room = (
        0.010 * np.sin(2 * np.pi * 50 * t)
        + 0.006 * np.sin(2 * np.pi * 120 * t)
        + 0.004 * rng.standard_normal(n)
    )
    write(p("room.wav"), room.astype(np.float32))

    # 4. A single transient in silence - a door, a chair, a knock on the desk.
    click = np.zeros(n, dtype=np.float32)
    at = RATE  # one second in
    click[at:at + 120] = (
        0.9 * np.exp(-np.arange(120) / 25.0) * np.sin(2 * np.pi * 900 * np.arange(120) / RATE)
    )
    write(p("click.wav"), click)

    # 5. A tone burst with a gap, which exercises the segmenter rather than the
    #    detector: two spans, close enough that the 200 ms merge has to decide.
    burst = np.zeros(RATE * 5, dtype=np.float32)
    tone = lambda k: 0.3 * np.sin(2 * np.pi * 220 * np.arange(k) / RATE)
    burst[RATE // 2:RATE // 2 + RATE] = tone(RATE)
    burst[RATE * 2:RATE * 2 + RATE] = tone(RATE)          # 1000 ms gap: separate
    burst[RATE * 7 // 2:RATE * 7 // 2 + RATE // 2] = tone(RATE // 2)
    write(p("bursts.wav"), burst)

    # 6. A transient immediately followed by a tone, so the onset rule and the
    #    padding interact: the click must not open the turn, the tone must.
    mixed = np.zeros(RATE * 3, dtype=np.float32)
    mixed[RATE // 2:RATE // 2 + 120] = click[at:at + 120]
    mixed[RATE:RATE + RATE] = tone(RATE)
    write(p("click_then_tone.wav"), mixed)

    print(f"\n{6} clips in {out}, seed {SEED}")


if __name__ == "__main__":
    main()
