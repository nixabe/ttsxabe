//! Streaming completion, shaped like llama-server's `/completion`.
//!
//! # Why this surface and not a chat-template one
//!
//! The GGUF carries Llama-3's chat template, and the pipeline does not use it.
//! `gateway.py` sends `POST /completion` with a **plain-text transcript** -
//! a system prompt, three few-shot turns, then `使用者: …\n小助理:` - and stops
//! on the speaker labels. `xabe-serve`'s `config.rs` is a behaviour-for-
//! behaviour port of that and already builds the string.
//!
//! So the seam is the same one llama-server offers: a prompt in, tokens out.
//! Putting a chat template in here would give this crate a second opinion
//! about prompt format, and the two would drift.
//!
//! That the few-shot turns are load-bearing is not a style preference. Breeze2
//! writes real Taigi - 毋過, 真濟, 食飽 - only when shown examples; without them
//! it writes Mandarin transliterated into Han, which the TTS then pronounces.
//! See `docs/MODEL.md`.
//!
//! # Stop strings are matched on text, not on tokens
//!
//! `小助理:` is not one token, and which tokens spell it depends on what
//! precedes it. Matching ids would miss every spelling but the one that was
//! looked up. So the decoded text is what is scanned, and the emitted prefix
//! is trimmed at the match - which also means a stop string can be *partially*
//! present at the end of a chunk and must not be emitted yet.

use crate::{ChatError, ChatModel, Rng, Sampling, sample};

/// What a completion produced, and why it ended.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The reply text, with any stop string removed.
    pub text: String,
    /// How many tokens were generated.
    pub tokens: usize,
    /// Why generation ended.
    pub stop: Stop,
}

/// Why a completion ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// An end-of-turn token.
    Eos,
    /// One of the caller's stop strings appeared in the text.
    Text(String),
    /// `max_tokens` was reached.
    Limit,
    /// The caller's callback asked to stop.
    Cancelled,
}

impl ChatModel {
    /// Completes `prompt`, calling `on_token` with each new piece of text.
    ///
    /// `on_token` returns whether to keep going, which is how the gateway
    /// cancels a reply when the user starts talking over it - the same
    /// `asr_cancel` path `gateway.py` has. Returning `false` stops the loop at
    /// the next token rather than at the next chunk, so a cancelled reply
    /// stops costing GPU time immediately.
    ///
    /// The text handed to `on_token` is **safe to emit**: anything that might
    /// still turn out to be the start of a stop string is held back until it
    /// either completes or is ruled out.
    pub fn complete(
        &self,
        prompt: &str,
        s: &Sampling,
        stops: &[String],
        on_token: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion, ChatError> {
        s.check()?;
        // `parse_special = false`: the prompt is user-influenced text, and a
        // user who types `<|eot_id|>` into the box must not be able to end the
        // model's turn from inside it. The transcript format has no specials
        // of its own, so nothing is lost.
        let mut ids = vec![self.tokenizer().bos()];
        ids.extend(self.tokenizer().encode(prompt, false));
        if ids.len() == 1 {
            return Err(ChatError::NothingToAnswer);
        }

        let mut cache = self.cache();
        let mut rng = Rng::new(s.seed);
        let mut produced: Vec<u32> = Vec::new();
        let mut text = String::new();
        // How much of `text` has been handed to the caller.
        let mut emitted = 0usize;
        let mut pending = ids.clone();
        let mut stop = Stop::Limit;

        while produced.len() < s.max_tokens {
            let logits = self.forward(&pending, &mut cache)?;
            // Only the last position's row predicts the next token; the rest
            // exist so the cache is filled.
            let n = pending.len();
            let vocab = self.config().vocab_size;
            let mut row =
                self.gpu()
                    .download(&self.gpu().copy_range(&logits, (n - 1) * vocab, vocab)?)?;

            let back = produced.len().min(s.repeat_last_n);
            let id = sample(&mut row, &produced[produced.len() - back..], s, &mut rng);

            if self.is_eos(id) {
                stop = Stop::Eos;
                break;
            }
            produced.push(id);
            pending = vec![id];

            // Decoded from the whole run each time rather than per token: a
            // multi-byte character spans several tokens, and decoding one at a
            // time would emit a replacement character at every boundary.
            //
            // Decoding the whole run is not enough on its own, because the run
            // itself can end mid-character - one Han character is often two or
            // three tokens. Emitting the replacement character that a lossy
            // decode produces there is worse than holding it back: the next
            // token turns it into a real character of a *different* byte
            // length, and the offset already handed to the caller no longer
            // means what it did.
            let bytes = self.tokenizer().decode_bytes(&produced, true);
            let stable = stable_prefix(&bytes);
            text = String::from_utf8_lossy(&bytes[..stable]).into_owned();

            if let Some((at, which)) = first_stop(&text, stops) {
                text.truncate(at);
                stop = Stop::Text(which);
                break;
            }

            // Hold back anything that could still become a stop string.
            let safe = text.len() - dangling(&text, stops);
            if safe > emitted
                && let Some(chunk) = text.get(emitted..safe)
                && !chunk.is_empty()
            {
                if !on_token(chunk) {
                    stop = Stop::Cancelled;
                    break;
                }
                emitted = safe;
            }
        }

        // A run that ends mid-character has no more tokens coming, so the
        // replacement character is now the final answer rather than something
        // that will change. Skipped when a stop string already truncated the
        // text, which would otherwise put the tail back.
        if !matches!(stop, Stop::Text(_)) {
            text = self.tokenizer().decode(&produced, true);
        }

        // Whatever was held back and turned out to be ordinary text.
        //
        // Not after a cancellation. The caller returned `false` to say it does
        // not want any more of this reply, and handing it the held-back tail
        // anyway would call it once after it said stop - which is exactly the
        // thing a cancelling caller is not expecting to have to handle.
        if stop != Stop::Cancelled
            && let Some(rest) = text.get(emitted..)
            && !rest.is_empty()
        {
            on_token(rest);
        }
        Ok(Completion {
            text,
            tokens: produced.len(),
            stop,
        })
    }

    /// Whether `id` ends the turn.
    ///
    /// Llama-3 has two: `<|end_of_text|>` ends the *document* and `<|eot_id|>`
    /// ends the *turn*, and an instruction-tuned checkpoint emits the second.
    /// Stopping only on the one the metadata calls `eos_token_id` would run to
    /// the token limit on every reply.
    fn is_eos(&self, id: u32) -> bool {
        let t = self.tokenizer();
        id == t.eos()
            || [("<|eot_id|>"), ("<|end_of_text|>"), ("<|eom_id|>")]
                .iter()
                .any(|s| t.special(s) == Some(id))
    }
}

/// How much of `bytes` will still mean the same once more tokens arrive.
///
/// Everything but a trailing, incomplete character - which is the only part a
/// later token can change. Bytes that can *never* be valid UTF-8 are stepped
/// over rather than held: a lossy decode renders each such run as one
/// replacement character and that will not change either, so holding them back
/// would stall the stream on output the model is never going to fix.
fn stable_prefix(bytes: &[u8]) -> usize {
    let mut at = 0;
    loop {
        match std::str::from_utf8(&bytes[at..]) {
            Ok(_) => return bytes.len(),
            Err(e) => match e.error_len() {
                None => return at + e.valid_up_to(),
                Some(bad) => at += e.valid_up_to() + bad,
            },
        }
    }
}

/// The earliest stop string in `text`, and where it starts.
fn first_stop(text: &str, stops: &[String]) -> Option<(usize, String)> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()).map(|at| (at, s.clone())))
        .min_by_key(|&(at, _)| at)
}

/// How many trailing bytes of `text` could still be the start of a stop string.
///
/// Without this, a reply ending `小助` would be emitted, and then the next
/// token would complete `小助理:` - by which point the browser has already
/// been handed the prefix and spoken it. The cost is that the last few bytes
/// of a reply arrive one token late, which nobody can hear.
fn dangling(text: &str, stops: &[String]) -> usize {
    let mut most = 0;
    for s in stops.iter().filter(|s| !s.is_empty()) {
        // The longest proper prefix of `s` that is a suffix of `text`.
        let max = s.len().min(text.len()).saturating_sub(1);
        for take in (1..=max).rev() {
            if !s.is_char_boundary(take) {
                continue;
            }
            if text.ends_with(&s[..take]) {
                most = most.max(take);
                break;
            }
        }
    }
    most
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_incomplete_character_is_not_yet_stable() {
        // The bug this is here for: 你 is three bytes, and a byte-level BPE
        // reaches them over more than one token. Emitting the first one or two
        // gives a replacement character that the next token replaces with a
        // real one of a different length, so every offset after it is wrong.
        let full = "好你".as_bytes();
        assert_eq!(stable_prefix(full), full.len());
        assert_eq!(stable_prefix(&full[..4]), 3, "one byte into 你");
        assert_eq!(stable_prefix(&full[..5]), 3, "two bytes into 你");
        assert_eq!(stable_prefix(b""), 0);
        assert_eq!(stable_prefix(b"ascii"), 5);
    }

    #[test]
    fn bytes_that_can_never_be_valid_do_not_stall_the_stream() {
        // A lone continuation byte is not a truncated character - nothing that
        // follows makes it valid. Holding it back would stop the stream
        // permanently on output the model is never going to complete, so it is
        // stepped over and the text after it still flows.
        assert_eq!(stable_prefix(b"\x80"), 1);
        assert_eq!(stable_prefix(b"ok\xffmore"), 7);
        // ...and a genuine truncation *after* invalid bytes is still held.
        let mixed = b"\xff\xe4\xbd";
        assert_eq!(stable_prefix(mixed), 1);
    }

    #[test]
    fn a_partial_stop_string_is_held_back() {
        let stops = vec!["小助理:".to_string(), "\n\n".to_string()];
        // `小` is three bytes and is the first character of a stop string.
        assert_eq!(dangling("你好小", &stops), "小".len());
        assert_eq!(dangling("你好小助", &stops), "小助".len());
        // Nothing to hold: the text ends in a character no stop starts with.
        assert_eq!(dangling("你好", &stops), 0);
        // A single newline could still become the two-newline stop.
        assert_eq!(dangling("done\n", &stops), 1);
    }

    #[test]
    fn a_complete_stop_string_is_found_at_its_start() {
        let stops = vec!["小助理:".to_string(), "使用者:".to_string()];
        let (at, which) = first_stop("好的\n小助理: 又來了", &stops).expect("found");
        assert_eq!(&"好的\n小助理: 又來了"[..at], "好的\n");
        assert_eq!(which, "小助理:");
        assert!(first_stop("沒有", &stops).is_none());
    }

    #[test]
    fn the_earliest_stop_wins_when_several_match() {
        // Order in the list must not decide it; position must.
        let stops = vec!["b".to_string(), "a".to_string()];
        assert_eq!(first_stop("xxaxxb", &stops).expect("found").0, 2);
    }

    #[test]
    fn dangling_never_splits_a_character() {
        // A prefix that ends mid-codepoint is not a prefix anything can match,
        // and slicing at one would panic rather than return a wrong answer.
        let stops = vec!["台語".to_string()];
        for text in ["台", "x台", "", "台語"] {
            let d = dangling(text, &stops);
            assert!(text.is_char_boundary(text.len() - d), "{text:?} -> {d}");
        }
    }
}
