#!/usr/bin/env python3
"""Capture llama.cpp's own tokenization of the chat model, as JSON.

    python tools/oracle/capture_chat_tokenizer.py \
        --model models/Llama-Breeze2-8B-Instruct-text-only.f16.gguf \
        --out .golden/chat/tokenizer.json

# Why llama.cpp and not 🤗

The other tokenizer captures in this directory drive `transformers`, because
the checkpoint they describe ships as a 🤗 directory and `AutoTokenizer` is the
definition of correct for it. This one cannot: the chat model exists on this
machine as a GGUF and nothing else, and its vocabulary lives *inside* that
file - 128,256 tokens and 280,147 merges as metadata arrays, with no
`tokenizer.json` beside it to read.

That is not a downgrade. llama-server is what serves this model today, so its
tokenizer is not an approximation of the reference, it *is* the thing being
replaced, and matching it exactly is the property that matters. The reference
binary is `test-tokenizer-0`, which reads a vocabulary out of a GGUF and writes
one id per line.

# What it does not parse

`test-tokenizer-0` tokenizes with `parse_special = false`, so `<|eot_id|>` in
the input is nine ordinary tokens rather than one. The Rust side takes that as
an argument, and this corpus is captured against the `false` reading - which is
the one that has to be exact, since it is the path every character of user text
takes. The `true` reading is a table lookup over ids this file also records.

# The corpus

Chosen for where a byte-level BPE with the `llama-bpe` pre-tokenizer diverges
from the GPT-2 one it resembles, not for being representative text:

- **Digit grouping.** Llama-3 splits digit runs at three; GPT-2 does not. Any
  reimplementation that borrowed the GPT-2 pattern agrees on prose and
  disagrees on every number in the corpus.
- **Whitespace runs.** The reference pattern ends `\\s+(?!\\S)|\\s+`, so *k*
  spaces before a word become *k-1* spaces plus a word owning the last. Rust's
  `regex` has no lookahead, so this is the case that proves the explicit
  reconstruction is equivalent.
- **Case-insensitive contractions.** `IT'S` splits where GPT-2 keeps it whole.
- **Han and Taigi Han**, which is what this model actually emits, at three and
  four bytes per character through the byte alphabet.
- **POJ with combining diacritics**, where a codepoint has no precomposed form
  and the pre-tokenizer has to keep the mark with its letter.
- **Newline runs**, which `\\s*[\\r\\n]+` handles separately from spaces.
"""

import argparse
import json
import pathlib
import subprocess
import tempfile

CORPUS = [
    # Nothing at all, and the whitespace-only inputs either side of it.
    "",
    " ",
    "  ",
    "\n",
    "\n\n",
    "   ",
    # Plain prose, so a total failure is visible before the awkward cases.
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    # Whitespace runs against `\s+(?!\S)`. One space, two, three, and a run
    # that ends the string with nothing after it to hand a space to.
    "a b",
    "a  b",
    "a   b",
    "a    b   c",
    "trailing   ",
    "\tindented",
    "  leading",
    # Newlines, which take a different alternative than spaces do.
    "line\nline",
    "para\n\npara",
    "windows\r\nline",
    "punct.\n\n\nmore",
    # Digit grouping at three. Every length from one to ten, so the boundary
    # is crossed in both directions.
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "1234567890",
    "3.14159",
    "Room 101",
    "1,000,000",
    "2026-08-27",
    "v0.22.0",
    # Contractions, in both cases, which `(?i:...)` is there for.
    "It's fine",
    "IT'S FINE",
    "don't",
    "DON'T",
    "we've they're I'm you'll he'd",
    # Punctuation runs, and punctuation that swallows the newlines after it.
    "!!!",
    "...",
    "?!?!",
    "end.\n",
    "end...\n\n",
    "(parens) [brackets] {braces}",
    # Han. This is what the model writes, so it is the case that matters most.
    "你好",
    "毋過真濟人食飽",
    "我今仔日欲去台北。",
    "請問你食飽未？",
    "台語真好聽，我真愛學。",
    # POJ with combining marks. `chia̍h` has no precomposed form for a-with-
    # vertical-line-above, so it is two codepoints and four bytes.
    "lí hó",
    "chia̍h pá--bōe?",
    "goá sī Tâi-oân-lâng",
    "kin-á-ji̍t thinn-khì chin hó",
    # Mixed scripts in one pre-token's neighbourhood.
    "台北 101 大樓",
    "Taigi 台語 POJ",
    # Bytes that are not text: the byte alphabet's job.
    "emoji 🎉 here",
    "​zero-width",
    "café vs café",
    # The chat template's own scaffolding, read as ordinary text - which is
    # what `parse_special = false` means and what the reference captured.
    "<|begin_of_text|>",
    "<|start_header_id|>user<|end_header_id|>",
    "<|eot_id|>",
    "a <| bare opener",
    # The system prompt this pipeline actually sends, so the corpus contains at
    # least one input of realistic length and shape.
    "你是一个台語助手。請用台語漢字回答，簡短自然。",
]

# Read back out of the file rather than transcribed, so a checkpoint with
# different ids is recorded correctly instead of asserted wrongly.
SPECIALS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
]


def tokenize(binary: str, model: str, text: str, work: pathlib.Path) -> list[int]:
    """One case through `test-tokenizer-0`, which works on files."""
    src = work / "case.txt"
    # Written as bytes: `write_text` would re-encode a lone `\r` on some
    # platforms, and the point of several cases is the exact byte sequence.
    src.write_bytes(text.encode("utf-8"))
    out = src.with_suffix(".txt.tokcpp")
    out.unlink(missing_ok=True)

    r = subprocess.run(
        [binary, model, str(src)], capture_output=True, text=True, check=False
    )
    if not out.is_file():
        raise SystemExit(
            f"{binary} wrote no token file for {text!r}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    return [int(line) for line in out.read_text().split() if line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the chat GGUF")
    ap.add_argument(
        "--binary",
        default=str(pathlib.Path.home() / "llama.cpp/build/bin/test-tokenizer-0"),
        help="llama.cpp's test-tokenizer-0",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    import sys

    sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(args.model)

    def field(key):
        f = reader.fields.get(key)
        return f

    tokens = field("tokenizer.ggml.tokens")
    vocab = [str(bytes(tokens.parts[i]), "utf-8") for i in tokens.data]
    spellings = {t: i for i, t in enumerate(vocab)}

    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td)
        cases = []
        for text in CORPUS:
            ids = tokenize(args.binary, args.model, text, work)
            cases.append({"text": text, "ids": ids})
            print(f"  {len(ids):4d}  {text[:56]!r}")

    payload = {
        "model": str(pathlib.Path(args.model).name),
        "reference": "llama.cpp test-tokenizer-0, parse_special=false",
        "vocab_size": len(vocab),
        "specials": {s: spellings[s] for s in SPECIALS if s in spellings},
        "cases": cases,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
