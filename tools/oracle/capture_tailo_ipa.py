"""Capture what goruut writes for Taiwanese, as a Tâi-lô to IPA correspondence.

`xabe-taigi` converts romanisation to the IPA the Coqui SuiSiann checkpoint was
trained on. There is no reference implementation of *that* conversion to diff
against - goruut goes from Han, not from romanisation - so the oracle is built
sideways, out of two things that already exist:

  * **SuiSiann's own metadata**, which pairs every recorded sentence's Han text
    with its Tâi-lô transcription. That is the corpus the checkpoint was trained
    on, so it is the right distribution as well as the right language.
  * **goruut's reading of the Han**, which is what the training text was
    phonemised with.

Line the two up syllable by syllable and each sentence yields a set of
(Tâi-lô syllable, IPA syllable) pairs. Aggregated over the corpus that is a
correspondence table nobody wrote down, and it is what `tests/correspondence.rs`
holds the Rust table to.

**The two do not agree everywhere, and that is expected rather than a defect.**
goruut has to guess which reading a Han character takes and it often guesses
differently from the transcriber - 我 as `ŋɔ` where the corpus says `ɡua`, 人 as
`dzin` where it says `laŋ`. Those are *reading* disagreements; the romanisation
is right, because it is what the speaker said. So the test asserts on how often
the spelling agrees where the reading does, not on exact equality.

The dictionary is also captured, for the inventory: every initial, rime and tone
letter goruut can write. A table that produces an initial goruut has never
written is wrong in a way no agreement rate would show.

Usage::

    .venv-coqui/bin/python tools/oracle/capture_tailo_ipa.py --out .golden/coqui-tailo
"""

import argparse
import collections
import json
import pathlib
import re
import sys
import unicodedata
import urllib.request

DICT_URL = (
    "https://raw.githubusercontent.com/neurlang/goruut/master/"
    "dicts/minnan/hokkien2/language.json"
)
CORPUS = ("ceciliayl/SuiSiann_raw_tone", "SuiSiann.csv")

# goruut writes ASCII `g`; Coqui's wrapper translates it to U+0261 before the
# tokenizer sees it, because the vocabulary contains only the latter.
SCRIPT_G = str.maketrans("g", "ɡ")

TONE_MARKS = {"́": 2, "̀": 3, "̂": 5, "̌": 6, "̄": 7, "̍": 8}
TONE_LETTERS = "˥˦˧˨˩"


def tailo_syllables(text):
    """Splits a Tâi-lô transcription into syllables with numeric tones.

    The same rule `xabe-taigi` applies: an unmarked syllable is tone 1, except
    that a stop-final one is tone 4.
    """
    text = unicodedata.normalize("NFD", text).lower()
    out, base, tone = [], "", None

    def flush():
        nonlocal base, tone
        if base:
            t = tone or (4 if base[-1] in "ptkh" else 1)
            out.append(f"{base}{t}")
        base, tone = "", None

    for c in text:
        if c in TONE_MARKS:
            tone = tone or TONE_MARKS[c]
        elif c == "͘":
            # POJ's dot; Tâi-lô spells the same vowel `oo`.
            base += "o" if base.endswith("o") else ""
        elif c.isalpha() or c.isdigit():
            base += c
        else:
            flush()
    flush()
    return out


def ipa_syllables(text):
    """Splits goruut's output into syllables.

    Every syllable ends in a run of Chao tone letters, and nothing else in the
    output contains one, so the split needs no knowledge of the phonology.
    """
    return re.findall(rf"[^\s{TONE_LETTERS}]*?[a-zãõĩũẽɔʔʰŋɡ]+[{TONE_LETTERS}]+", text)


def inventory(language):
    """Every initial, rime and tone letter goruut's dictionary can write."""
    initials = ["tsʰ", "ts", "pʰ", "tʰ", "kʰ", "dz",
                "p", "b", "m", "t", "n", "l", "s", "k", "ɡ", "ŋ", "h"]
    bodies, tones = set(), set()
    for readings in language["Map"].values():
        for reading in readings:
            for m in re.finditer(rf"([^{TONE_LETTERS}_]+)([{TONE_LETTERS}]*)", reading.translate(SCRIPT_G)):
                if m.group(1):
                    bodies.add(m.group(1))
                if m.group(2):
                    tones.add(m.group(2))

    seen_initials, rimes = set(), set()
    for body in bodies:
        if not re.fullmatch(r"[a-zãõĩũẽɔʔʰŋɡ]+", body):
            continue
        for i in initials:
            if body.startswith(i) and len(body) > len(i):
                seen_initials.add(i)
                rimes.add(body[len(i):])
                break
        else:
            seen_initials.add("")
            rimes.add(body)
    return {
        "initials": sorted(seen_initials),
        "rimes": sorted(rimes),
        "tones": sorted(tones),
        "bodies": sorted(bodies),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--aligned",
        type=pathlib.Path,
        help="a cached [{han, lo, ipa}] run, to skip calling goruut again",
    )
    args = ap.parse_args()

    language = json.loads(urllib.request.urlopen(DICT_URL).read().decode())

    if args.aligned:
        rows = json.loads(args.aligned.read_text())
    else:
        rows = phonemise_corpus()

    pairs = collections.Counter()
    aligned = 0
    for r in rows:
        lo, ipa = tailo_syllables(r["lo"]), ipa_syllables(r["ipa"])
        # Only sentences whose two halves have the same syllable count can be
        # lined up. A mismatch means goruut merged or split something, and
        # guessing an alignment would invent correspondences.
        if lo and len(lo) == len(ipa):
            aligned += 1
            for a, b in zip(lo, ipa):
                pairs[(a, b)] += 1

    by_syllable = collections.defaultdict(collections.Counter)
    for (a, b), n in pairs.items():
        by_syllable[a][b] += n

    meta = {
        "source": f"{CORPUS[0]}/{CORPUS[1]} phonemised by goruut MinnanHokkien2",
        "dictionary": DICT_URL,
        "sentences": len(rows),
        "aligned": aligned,
        "tokens": sum(pairs.values()),
        "syllables": len(by_syllable),
        "correspondence": {a: v.most_common() for a, v in sorted(by_syllable.items())},
        "inventory": inventory(language),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "correspondence.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")

    print(f"sentences {len(rows)}, aligned 1:1 {aligned}, syllable tokens {meta['tokens']}")
    print(f"distinct Tâi-lô syllables {meta['syllables']}")
    print(f"inventory: {len(meta['inventory']['initials'])} initials, "
          f"{len(meta['inventory']['rimes'])} rimes, {len(meta['inventory']['tones'])} tones")
    print(f"wrote {path}")


def phonemise_corpus():
    """Downloads SuiSiann and runs goruut over every Han sentence."""
    import csv
    from huggingface_hub import hf_hub_download
    from pygoruut.pygoruut import Pygoruut

    path = hf_hub_download(CORPUS[0], CORPUS[1], repo_type="dataset")
    rows = list(csv.DictReader(open(path)))

    goruut = Pygoruut(writeable_bin_dir="")
    try:
        out = []
        for i, r in enumerate(rows):
            ipa = str(
                goruut.phonemize(
                    language="MinnanHokkien2", sentence=r["漢字"], separator="", is_punct=True
                )
            ).translate(SCRIPT_G)
            out.append({"han": r["漢字"], "lo": r["羅馬字"], "ipa": ipa})
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
        return out
    finally:
        # Started with this process's stdout inherited, so leaving it running
        # holds the pipe open for anything capturing our output.
        goruut.process.terminate()
        goruut.process.wait()


if __name__ == "__main__":
    sys.exit(main())
