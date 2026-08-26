#!/usr/bin/env python3
"""Capture 🤗 `LlamaTokenizer` on a corpus, as JSON.

    python tools/oracle/capture_llama_tokenizer.py \
        --model models/translator/taigi-llama2-13b \
        --out .golden/translator/tokenizer.json

The corpus is chosen for the ways a SentencePiece BPE goes wrong rather than
for being representative text: the dummy prefix and what happens to leading and
repeated whitespace, Han and Taigi Han that the extended vocabulary was trained
for, POJ with combining diacritics, characters that fall through to byte
fallback, and the `[TRANS]` template the translator is actually prompted with.
"""

import argparse
import json

from transformers import AutoTokenizer

CORPUS = [
    "",
    " ",
    "  ",
    "   spaced",
    "Hello world",
    "Hello world ",
    "a\nb",
    "tabs\there",
    "It's a test, isn't it?",
    "abc123 456def 7.89",
    "你好",
    "你好，今天天氣很好。",
    "我要去市場買東西 你要去嗎",
    "毋過真濟人食飽矣",
    "食飽未？",
    "lí hó, góa sī Tâi-oân lâng",
    "chiaⁿ-goe̍h",
    "Góa beh khì chhī-tiûⁿ bé mi̍h-kiāⁿ",
    "mixed 中文 and English 123",
    "🎧",
    "emoji 🎧 and 🇹🇼 flags",
    "ＡＢＣ全形",
    "……「引號」——",
    "[TRANS]今天天氣很好[/TRANS]",
    "<unk>",
    "<s>hello</s>",
    "<pad>",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)

    cases = []
    for text in CORPUS:
        ids = tok.encode(text, add_special_tokens=False)
        cases.append({
            "text": text,
            "ids": ids,
            "pieces": tok.convert_ids_to_tokens(ids),
            "decoded": tok.decode(ids, skip_special_tokens=False),
            "decoded_skip": tok.decode(ids, skip_special_tokens=True),
        })

    out = {
        "model": a.model.rstrip("/").rsplit("/", 1)[-1],
        "transformers": __import__("transformers").__version__,
        # `len(tok)` is the tokenizer's, which is *not* config.json's
        # `vocab_size`: the embedding is padded and the last rows are unused.
        "tokenizer_size": len(tok),
        "added": tok.get_added_vocab(),
        "bos": tok.bos_token_id,
        "eos": tok.eos_token_id,
        "unk": tok.unk_token_id,
        "cases": cases,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {a.out}: {len(cases)} cases, {out['tokenizer_size']} pieces")


if __name__ == "__main__":
    main()
