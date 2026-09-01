"""Turn text into the IPA phonemes the Coqui SuiSiann checkpoint speaks.

This is the half of that model's pipeline the Rust engine does not have, and
deliberately. The phonemiser is ``pygoruut``: a Go binary, downloaded on first
use, carrying a 140 KB Han-to-IPA dictionary for ``MinnanHokkien2`` and a
learned fallback for anything not in it. Reimplementing the dictionary would be
easy and reimplementing the fallback would not, and a half-port is worse than
none - an out-of-dictionary word would come out *mispronounced* rather than
missing, which is the failure this workspace is built to refuse. See
``docs/MODEL.md``.

So the front end stays here, where it is the reference's own code rather than a
reimplementation of it, and the engine takes what it produces::

    .venv-coqui/bin/python tools/phonemize_pygoruut.py --text "你好！我是蔡贏。"
    li˥˧ho˥˧ɡua˥˧si˧˧ĩã˨˦

    xabe --tts-model models/tts/coqui-vits-suisiann \\
         --text "$(.venv-coqui/bin/python tools/phonemize_pygoruut.py --text '...')" \\
         --out hello.wav

The cleaner runs first, exactly as ``TTSTokenizer.text_to_ids`` runs it, so what
comes out of here is what the model was trained on and not an approximation of
it. Characters the dictionary does not know are passed through unchanged - the
Han character stays a Han character - and the engine's tokenizer then drops
them, which is also what the reference does.

**The goruut process is stopped explicitly before this exits**, and that is not
tidiness. ``pygoruut`` starts the binary with the parent's stdout inherited, so
a caller writing ``$(... phonemize_pygoruut.py ...)`` waits on the *daemon* to
close the pipe rather than on this script - which is to say, forever. Relying on
``__del__`` at interpreter shutdown is not enough to prevent that.
"""

import argparse
import pathlib
import sys

from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("models/tts/coqui-vits-suisiann"),
        help="directory holding config.json",
    )
    ap.add_argument("--text", help="text to phonemise; omit to read stdin")
    args = ap.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()

    config = load_config(str(args.model / "config.json"))
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    if not tokenizer.use_phonemes:
        # Nothing to do, and saying so beats printing the input and letting the
        # caller believe it was converted.
        raise SystemExit("this checkpoint was trained on graphemes, not phonemes")

    cleaned = tokenizer.text_cleaner(text) if tokenizer.text_cleaner else text
    try:
        phonemes = tokenizer.phonemizer.phonemize(
            cleaned, separator="", language=tokenizer.phonemizer.language
        )
    finally:
        stop(tokenizer.phonemizer)
    print(phonemes)


def stop(phonemizer):
    """Terminates the goruut subprocess, if this phonemiser started one."""
    inner = getattr(phonemizer, "pygoruut", None)
    process = getattr(inner, "process", None)
    if process is None:
        return
    process.terminate()
    process.wait()


if __name__ == "__main__":
    sys.exit(main())
