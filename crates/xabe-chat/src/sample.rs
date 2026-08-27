//! Token selection, in llama.cpp's order.
//!
//! # Why the order is the specification
//!
//! `gateway.py` sends `temperature: 0.3, top_p: 0.9, repeat_penalty: 1.1` to
//! `llama-server`, so those three are what this has to reproduce - and the
//! *sequence* they are applied in is not a detail. Penalising after truncating
//! is not the same distribution as penalising before it, because the penalty
//! can push a token out of the nucleus that top-p had already admitted.
//!
//! llama.cpp's default chain is penalties, then temperature, then top-k, then
//! top-p, then the draw. This follows it, minus top-k, which the pipeline
//! leaves at its disabled default.
//!
//! # Why not just take the argmax
//!
//! Because `temp` is 0.3 and not 0. The replies this model gives at greedy are
//! noticeably more repetitive across a conversation, which is what the
//! repetition penalty is also there for - and neither knob was chosen here,
//! they were chosen by whoever tuned the running pipeline. Reproducing them is
//! the product claim; improving on them is a separate decision.

use crate::ChatError;

/// The sampler's knobs, defaulting to what the pipeline runs.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    /// Flattens or sharpens the distribution. Zero means take the argmax.
    pub temperature: f32,
    /// Nucleus mass to keep. 1.0 keeps everything.
    pub top_p: f32,
    /// Divides the logit of a recently seen token. 1.0 turns it off.
    pub repeat_penalty: f32,
    /// How far back the penalty looks.
    pub repeat_last_n: usize,
    /// The most tokens to produce before stopping regardless.
    pub max_tokens: usize,
    /// The PRNG seed.
    ///
    /// Fixed rather than drawn from the clock, which is the same decision
    /// `xabe-tts` took for its noise: a reply that cannot be reproduced cannot
    /// be diffed against a reference, and "it sounded different that time" is
    /// not a bug report anyone can act on.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        // `gateway.py`'s values, not llama.cpp's. The two differ - llama.cpp
        // defaults `temperature` to 0.8 - and the pipeline's are the ones this
        // engine has to match.
        Self {
            temperature: 0.3,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            max_tokens: 160,
            seed: 0x5EED_5EED,
        }
    }
}

impl Sampling {
    /// Greedy decoding, for comparing against a reference that does not draw.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            max_tokens,
            ..Self::default()
        }
    }

    /// Refuses a combination that has no meaning, at construction.
    ///
    /// A negative temperature would invert the distribution and a top-p above
    /// one would silently keep everything - both are more likely a units
    /// mistake than an intent, and both produce output rather than an error.
    pub fn check(&self) -> Result<(), ChatError> {
        let bad = |what, got, range| Err(ChatError::BadSampler { what, got, range });
        if !(self.temperature >= 0.0 && self.temperature.is_finite()) {
            return bad("temperature", self.temperature, "[0, inf)");
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return bad("top_p", self.top_p, "(0, 1]");
        }
        if !(self.repeat_penalty > 0.0 && self.repeat_penalty.is_finite()) {
            return bad("repeat_penalty", self.repeat_penalty, "(0, inf)");
        }
        Ok(())
    }
}

/// llama.cpp's `xorshift`-family generator, so a seed means the same thing.
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator.
    ///
    /// Zero is remapped: a state of zero is a fixed point of the recurrence
    /// and would return zero forever, which reads as "the sampler is broken"
    /// rather than "the seed was".
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A uniform draw in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        // 24 bits, which is exactly f32's mantissa: taking more would be
        // rounded away and taking fewer would leave visible steps.
        (self.next_u64() >> 40) as f32 / f32::from(1u16 << 12) / f32::from(1u16 << 12)
    }
}

/// Picks one token from a row of logits.
///
/// `recent` is the tail of the sequence the repetition penalty looks at,
/// caller-trimmed to `repeat_last_n`.
pub fn sample(logits: &mut [f32], recent: &[u32], s: &Sampling, rng: &mut Rng) -> u32 {
    // 1. Penalties, before anything narrows the field. llama.cpp divides a
    //    positive logit and *multiplies* a negative one, so the penalty always
    //    moves a token toward less likely rather than flipping its sign - the
    //    naive `logit / penalty` makes an already-unlikely token *more* likely.
    if (s.repeat_penalty - 1.0).abs() > f32::EPSILON {
        for &t in recent {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 {
                    *l / s.repeat_penalty
                } else {
                    *l * s.repeat_penalty
                };
            }
        }
    }

    // 2. Temperature. Zero is the argmax, taken directly rather than as a
    //    limit - dividing by zero would give infinities that softmax turns
    //    into NaN.
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    for l in logits.iter_mut() {
        *l /= s.temperature;
    }

    // 3. Softmax, in the numerically stable order.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, (l - max).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|&(_, p)| p).sum();
    for p in &mut probs {
        p.1 /= sum;
    }

    // 4. Top-p, over the sorted tail. `sort_unstable_by` with a total order on
    //    the probability alone would leave ties in an arbitrary order and make
    //    the draw depend on the sort's internals; breaking ties by id keeps a
    //    seed meaning one thing.
    probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if s.top_p < 1.0 {
        let mut acc = 0.0;
        let mut keep = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            acc += p;
            if acc >= s.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep.max(1));
    }

    // 5. The draw, over the renormalised remainder.
    let total: f32 = probs.iter().map(|&(_, p)| p).sum();
    let mut r = rng.unit() * total;
    for &(id, p) in &probs {
        r -= p;
        if r <= 0.0 {
            return id;
        }
    }
    // Floating-point accumulation can leave `r` a hair above zero after the
    // whole sum; the last token is the right answer then, not a failure.
    probs.last().map_or(0, |&(id, _)| id)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(i, _)| i as u32)
}
