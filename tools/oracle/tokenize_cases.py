"""Capture the reference tokenizer's output for a battery of awkward inputs.

Milestone 4 asks for an *exact* match, not a close one, so the useful test
corpus is not a sentence - it is the set of inputs where a reimplementation
would plausibly diverge. Casing, combining marks that have no precomposed form,
characters outside the 48-symbol vocabulary, the literal spelling of the unknown
token, and the blank interspersion at the edges are all here for that reason.

Output is ``.golden/tokenizer/cases.json``: a list of ``{text, ids}`` pairs plus
the tokenizer's configuration, so the Rust side can assert against what the
reference actually produced rather than against a reading of its source.
"""

import argparse
import json
import pathlib
import unicodedata

from transformers import VitsTokenizer

# Each entry is a specific way to be wrong, not a specific thing to say.
CASES = [
    # The sentence the waveform capture uses, so the two goldens agree.
    "lí hó, kin-á-ji̍t thinn-khì chin hó.",
    # Casing. The vocabulary is lower-case only, so every upper-case character
    # must survive as its lower-case self rather than being dropped.
    "LÍ HÓ",
    "Lí Hó",
    # Punctuation outside the vocabulary is deleted, not replaced. A comma is
    # not a pause here; it simply ceases to exist.
    "li, ho. li! ho?",
    # Leading and trailing space is stripped even though space is *in* the
    # vocabulary, because the reference calls .strip() after filtering.
    "   li ho   ",
    # Interior runs of space are not collapsed.
    "li    ho",
    # The two combining marks with no precomposed form: U+030D (the entering
    # tone) and U+0358 (the o-dot). These must arrive decomposed.
    "ji̍t",
    "o͘",
    "kok͘ si̍t",
    # The same string in NFC and NFD. NFD moves the acute off the vowel onto
    # U+0301, which is *not* in the vocabulary - so the tone silently vanishes.
    # This is the single most damaging way to get the tokenizer wrong.
    unicodedata.normalize("NFC", "lí"),
    unicodedata.normalize("NFD", "lí"),
    # The literal spelling of the unknown token. The reference matches it as an
    # added token and then filters it character by character, so it does not
    # survive as a token.
    "<unk>",
    "li <unk> ho",
    # Han characters: entirely outside the vocabulary.
    "你好",
    "li 你好 ho",
    # Degenerate inputs.
    "",
    " ",
    "a",
    "...",
    # Every symbol in the vocabulary, so no single id mapping goes unchecked.
    "abcdefghijklmnopqrstuvwxyz '-",
    "âêîôûáéíóúàèìòùāēīōūńǹḿ",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--model", default="facebook/mms-tts-nan")
    args = ap.parse_args()

    tok = VitsTokenizer.from_pretrained(args.model)

    cases = []
    for text in CASES:
        ids = tok(text=text, return_tensors=None)["input_ids"]
        cases.append(
            {
                "text": text,
                # Also record what normalisation produced, so a Rust failure
                # says which of the two stages diverged.
                "normalized": tok.prepare_for_tokenization(text)[0],
                "ids": list(map(int, ids)),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cases.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "transformers": __import__("transformers").__version__,
                "add_blank": tok.add_blank,
                "normalize": tok.normalize,
                "phonemize": tok.phonemize,
                "is_uroman": tok.is_uroman,
                "language": tok.language,
                "vocab_size": tok.vocab_size,
                "unk_token_id": tok.unk_token_id,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    for c in cases:
        print(f"{c['text']!r:44s} -> {len(c['ids']):3d} ids")
    print(f"\nwrote {len(cases)} cases to {args.out / 'cases.json'}")


if __name__ == "__main__":
    main()
