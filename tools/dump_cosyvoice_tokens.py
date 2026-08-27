#!/usr/bin/env python3
"""Writes the CosyVoice3 tokenizer's added tokens beside the checkpoint.

# Why this exists

`CosyVoice-BlankEN` ships `vocab.json` and `merges.txt`, which are the learned
half of the tokenizer, and nothing at all for the special half. The special
tokens are a *literal list in CosyVoice's source* - `<|endofprompt|>`, the
paralinguistic markers, and some three hundred pinyin finals - handed to
`add_special_tokens` at construction. Their ids are therefore decided by the
order of that list and by how much of it 🤗 finds already present, which is not
something to re-derive by hand: `<|endofprompt|>` is 151646 only because
`<|im_start|>` and `<|im_end|>` were already in `added_tokens_decoder`.

So the ids are read out of the reference tokenizer once and written down, the
same discipline `docs/ORACLE.md` uses for every other captured constant. The
engine then reads a file rather than reimplementing a list.
"""

import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="models/tts/cosyvoice3-0.5b")
    a = ap.parse_args()

    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    d = pathlib.Path(a.model_dir) / "CosyVoice-BlankEN"
    t = get_qwen_tokenizer(str(d), skip_special_tokens=True, version="cosyvoice3")
    added = t.tokenizer.get_added_vocab()

    out = d / "added_tokens.json"
    out.write_text(json.dumps(added, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    lo, hi = min(added.values()), max(added.values())
    print(f"wrote {out}: {len(added)} tokens, ids {lo}..{hi}")
    print(f"  <|endofprompt|> = {added['<|endofprompt|>']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
