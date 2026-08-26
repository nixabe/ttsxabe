#!/usr/bin/env python3
"""Capture 🤗 `WhisperTokenizer` on a corpus, as JSON.

    python tools/oracle/capture_tokenizer.py --model models/asr/breeze-asr-26 \
        --out .golden/asr/tokenizer.json

The corpus is chosen for the ways a byte-level BPE goes wrong rather than for
being representative text: runs of spaces (the negative lookahead in the
pre-tokenizer), Han (three bytes per character, routinely split across two
tokens), POJ with its combining diacritics, digits next to letters, and the
`<|...|>` special tokens that must survive a round trip intact.

`decode_with_timestamps` is deliberately *not* captured. In transformers
5.15.1 it computes `timestamp_begin = self.all_special_ids[-1] + 1`, and
`all_special_ids` on this checkpoint has exactly one entry - `<|endoftext|>`,
50257 - so it treats everything from `<|startoftranscript|>` up as a
timestamp and renders it as `<|0.00|>`. Capturing that would enshrine a bug.
The engine's own behaviour for that one function is asserted directly, in
`tests/tokenizer.rs`, with this note attached.
"""

import argparse
import json

from transformers import WhisperTokenizer

CORPUS = [
    "",
    " ",
    "   ",
    "hello",
    " hello",
    "hello world",
    "hello  world",
    "hello   world",
    "hello world ",
    "\n\nhello\tworld\n",
    "It's a test, isn't it? I'd say we've won.",
    "abc123 456def 7.89",
    "你好",
    "你好，今天天氣很好。",
    "我要去市場買東西 你要去嗎",
    "毋過真濟人食飽矣",
    "lí hó, góa sī Tâi-oân lâng",
    "chiaⁿ-goe̍h",
    "Góa beh khì chhī-tiûⁿ bé mi̍h-kiāⁿ",
    "mixed 中文 and English 123",
    "emoji 🎧 and 🇹🇼 flags",
    "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>你好",
    "<|0.00|>hello<|2.50|>",
    "a<|zh|>b",
    "<|endoftext|>",
    "hello<|endoftext|>",
    "<|notatoken|>",
    "<|",
    "trailing <|",
    "……「引號」——",
    "ＡＢＣ全形",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = WhisperTokenizer.from_pretrained(a.model)

    cases = []
    for text in CORPUS:
        ids = tok.encode(text, add_special_tokens=False)
        cases.append({
            "text": text,
            "ids": ids,
            # `clean_up_tokenization_spaces` is destructive for BPE - it strips
            # spaces before punctuation - so it is off here and in the engine.
            "decoded": tok.decode(ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False),
            "decoded_skip": tok.decode(ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False),
        })

    out = {
        "model": a.model.rstrip("/").rsplit("/", 1)[-1],
        "transformers": __import__("transformers").__version__,
        "vocab_size": len(tok),
        "specials": {
            "startoftranscript": tok.convert_tokens_to_ids("<|startoftranscript|>"),
            "zh": tok.convert_tokens_to_ids("<|zh|>"),
            "en": tok.convert_tokens_to_ids("<|en|>"),
            "transcribe": tok.convert_tokens_to_ids("<|transcribe|>"),
            "translate": tok.convert_tokens_to_ids("<|translate|>"),
            "notimestamps": tok.convert_tokens_to_ids("<|notimestamps|>"),
            # This checkpoint carries OpenAI's original spelling. Asking for
            # `<|nospeech|>` does not fail - it returns the unknown id, 50257,
            # which is also end-of-text, so a caller that trusts it stops the
            # decode on the wrong token.
            "nocaptions": tok.convert_tokens_to_ids("<|nocaptions|>"),
            "nospeech_or_unk": tok.convert_tokens_to_ids("<|nospeech|>"),
            "endoftext": tok.convert_tokens_to_ids("<|endoftext|>"),
            "timestamp_begin": tok.convert_tokens_to_ids("<|0.00|>"),
        },
        "cases": cases,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: {len(cases)} cases, vocab {out['vocab_size']}")


if __name__ == "__main__":
    main()
