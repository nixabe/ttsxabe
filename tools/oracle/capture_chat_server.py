#!/usr/bin/env python3
"""Capture `llama-server`'s replies for the chat model, as JSON.

    python tools/oracle/capture_chat_server.py --url http://127.0.0.1:8082 \
        --out .golden/chat/llama_server.json

The chat model has **one** reference, not two. The translator has a 🤗 float32
oracle beside its llama-server one; this model does not, because there is no 🤗
checkpoint for it on this machine - it exists as a GGUF and nothing else. So
llama-server running that GGUF is both what the pipeline uses today and the
only thing to compare against.

That makes this a *product* comparison rather than a numerical one, and it is
weaker in a way worth being explicit about: agreeing with llama-server proves
the replacement is a replacement, and does not prove either of them computes
the reference arithmetic. Per-layer taps are the thing that would, and they
need an oracle this model has none of.

# Everything is captured at temperature 0

`gateway.py` runs at 0.3, and a sampled reply is not comparable: two correct
implementations drawing from the same distribution give different text. So the
capture pins `temperature: 0` and `repeat_penalty: 1.0`, which makes the reply
a function of the prompt alone, and the Rust side is run with
`Sampling::greedy` against it.

The sampler itself is then tested separately, against the distribution rather
than against a draw - see `crates/xabe-chat/src/sample.rs`.

# And llama-server is not request-independent

The finding from the translator capture applies here unchanged and matters
more, because these prompts share a long common prefix. llama-server **reuses a
KV prefix across requests**, so the same prompt at temperature 0 can produce
different text depending on what was asked before it. Each case is therefore
sent twice and the pair is required to agree before it is recorded; a case
that disagrees with itself is dropped with a note rather than captured as a
fact about the model.
"""

import argparse
import json
import urllib.request

# The system prompt and few-shot turns `gateway.py` sends under
# `--direct-taigi`, verbatim. A comparison against a differently prompted
# server is not a comparison, and these three examples are what make the model
# write Taigi rather than Mandarin in Han characters.
PERSON, BOT = "使用者", "小助理"
SYSTEM = (
    f"以下是 {PERSON} 佮台語助理 {BOT} 咧講話的紀錄。\n"
    f"{BOT} 干焦用台語漢字回答，袂使用華語。\n"
    f"{BOT} 的回答愛真短，上濟兩句，因為會唸出聲。\n"
    "無註解、無括號、無 Markdown。\n\n"
    f"{PERSON}: 你好\n{BOT}: 你好！有啥物代誌我會使鬥相共？\n"
    f"{PERSON}: 台北今仔日天氣按怎？\n{BOT}: 台北今仔日好天，溫度差不多二十五度。\n"
    f"{PERSON}: 你食飽未？\n{BOT}: 食飽矣，多謝關心。"
)

CORPUS = [
    "你好",
    "你食飽未？",
    "台北今仔日天氣按怎？",
    "你會曉講台語無？",
    "請你共我講一个笑話。",
    "我今仔日真忝。",
    "台灣上好食的物件是啥物？",
    "你叫啥物名？",
]


def prompt_for(user_text: str) -> str:
    """`gateway.py`'s `build_prompt`, with an empty history."""
    return "\n".join([SYSTEM, "", f"{PERSON}: {user_text}", f"{BOT}:"])


def complete(url: str, prompt: str, n_predict: int, n_probs: int = 0) -> dict:
    body = {
        "prompt": prompt,
        "temperature": 0.0,
        # Off, not `gateway.py`'s 1.1: with a penalty the reply depends on the
        # order tokens happened to appear in, which is one more thing that has
        # to match exactly before the text can. The penalty is tested on its
        # own in the sampler.
        "repeat_penalty": 1.0,
        "n_predict": n_predict,
        "stop": [f"{PERSON}:", f"{BOT}:", "\n\n"],
        # The generated ids, not just the text. A generated sequence is **not**
        # the canonical BPE segmentation of its own text - re-encoding the
        # reply gives a different, equally valid cut - so a per-position
        # comparison has to use the ids that were actually produced. Measured:
        # 17 pieces on re-encoding against the 15 that were generated.
        "return_tokens": True,
    }
    if n_probs:
        body["n_probs"] = n_probs
    req = urllib.request.Request(
        url.rstrip("/") + "/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def tokenize(url: str, text: str) -> list[int]:
    """The prompt's ids, from the server that will consume them.

    Belt and braces: `Bpe` already matches llama.cpp id-for-id on a captured
    corpus, so this should agree by construction. Recording it anyway means a
    per-position comparison rests on the reference's own numbering rather than
    on that agreement continuing to hold.
    """
    req = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["tokens"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8082")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cases, unstable = [], []
    for user in CORPUS:
        p = prompt_for(user)
        # `n_probs = 2` on the recorded run, so the capture carries **how close
        # the decision was** at every step and not only which way it went.
        #
        # That turns "our reply differs" from a verdict into a question with an
        # answer already in the file. Two f16 implementations with different
        # reduction orders will disagree at a step where the top two candidates
        # are a few hundredths of a nat apart, and will not disagree anywhere
        # else. Recording the margin is what lets the Rust side tell those two
        # cases apart without inventing a tolerance of its own.
        first = complete(a.url, p, a.n_predict, n_probs=2)
        again = complete(a.url, p, a.n_predict)
        if first["content"].strip() != again["content"].strip():
            # The prefix-reuse effect, caught rather than recorded. See the
            # module docstring.
            print(f"  UNSTABLE {user!r}", flush=True)
            unstable.append(
                {"user": user, "first": first["content"], "again": again["content"]}
            )
            continue

        steps = []
        for step in first.get("completion_probabilities", []):
            top = step.get("top_logprobs") or step.get("probs") or []
            lp = [t.get("logprob") for t in top[:2] if t.get("logprob") is not None]
            steps.append(
                {
                    "token": step.get("content", step.get("token", "")),
                    # How much the chosen token won by. Zero would be an exact
                    # tie; the interesting range is the first tenth of a nat.
                    "margin": (lp[0] - lp[1]) if len(lp) == 2 else None,
                }
            )

        # `tokens` runs past the text: generation continues into the stop
        # string, which the content is then cut before. `steps` is one per
        # token of the *reply*, so it is what sets the length.
        reply_tokens = first.get("tokens", [])[: len(steps)]

        text = first["content"].strip()
        print(f"  {user} -> {text}", flush=True)
        cases.append(
            {
                "user": user,
                "prompt": p,
                "prompt_tokens": tokenize(a.url, p),
                "text": text,
                # Unstripped: the leading space after `小助理:` is a token.
                "raw": first["content"],
                "tokens": reply_tokens,
                "steps": steps,
            }
        )

    payload = {
        "url": a.url,
        "reference": "llama-server /completion, temperature 0, no repeat penalty",
        "person": PERSON,
        "bot": BOT,
        "system": SYSTEM,
        "n_predict": a.n_predict,
        "stops": [f"{PERSON}:", f"{BOT}:", "\n\n"],
        "cases": cases,
        "unstable": unstable,
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {a.out}: {len(cases)} stable, {len(unstable)} dropped")


if __name__ == "__main__":
    main()
