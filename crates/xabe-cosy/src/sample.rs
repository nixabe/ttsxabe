//! `ras_sampling`: nucleus with a repetition escape hatch.
//!
//! # Why this is not the chat model's sampler
//!
//! Two differences, and the second one is the whole point of the algorithm.
//!
//! **The nucleus is bounded twice.** Upstream stops adding candidates when
//! *either* the cumulative mass reaches `top_p` **or** the count reaches
//! `top_k` - and it checks both *before* adding, so the mass can finish below
//! `top_p`. A nucleus that adds the token which crosses the threshold is a
//! different distribution, and at `top_p = 0.8` over a 6761-way head the
//! difference is not small.
//!
//! **A token that has been repeating is thrown away and redrawn.** If the
//! drawn token already occupies `win_size * tau_r` of the last `win_size`
//! outputs - one of the last ten, at the shipped settings - its score is set
//! to negative infinity and the draw is redone over the *whole* distribution
//! rather than over the nucleus. That is what stops a speech model getting
//! stuck on one code and emitting a held buzz, and it is why this cannot be
//! expressed as a repetition penalty: the penalty is not a nudge, it is a
//! rejection with a different fallback.
//!
//! # This cannot reproduce upstream's draws
//!
//! Upstream calls `torch.multinomial`, so its output is a function of
//! PyTorch's RNG as much as of the weights. Matching that bit for bit is not a
//! reasonable thing to attempt. So the *distribution* is what is tested -
//! against the captured log-probabilities, position by position - and the draw
//! is reproducible from this crate's own seed rather than from theirs. See
//! `docs/ORACLE.md`.

use crate::RasConfig;

/// The `xorshift` generator the rest of this workspace uses, so a seed means
/// the same thing here as it does in `xabe-chat`.
pub struct Rng(u64);

impl Rng {
    /// Seeds the generator.
    ///
    /// Zero is remapped: it is a fixed point of the recurrence and would
    /// return zero forever, which reads as a broken sampler rather than a
    /// badly chosen seed.
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
        // 24 bits, which is exactly f32's mantissa.
        (self.next_u64() >> 40) as f32 / f32::from(1u16 << 12) / f32::from(1u16 << 12)
    }
}

/// Draws one index from `probs` by cumulative mass.
fn draw(probs: &[(u32, f32)], rng: &mut Rng) -> u32 {
    let total: f32 = probs.iter().map(|&(_, p)| p).sum();
    let mut r = rng.unit() * total;
    for &(id, p) in probs {
        r -= p;
        if r <= 0.0 {
            return id;
        }
    }
    // Accumulation can leave `r` a hair above zero after the whole sum; the
    // last candidate is the right answer then, not a failure.
    probs.last().map_or(0, |&(id, _)| id)
}

/// Softmax over log-probabilities, returned sorted by descending probability.
///
/// Ties break by id so that a seed means one thing. Left to the sort's own
/// order, two runs with the same seed could draw differently.
fn sorted_probs(logits: &[f32]) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, (l - max).exp()))
        .collect();
    let sum: f32 = p.iter().map(|&(_, v)| v).sum();
    for x in &mut p {
        x.1 /= sum;
    }
    p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    p
}

/// The nucleus, bounded by mass *and* count, both checked before adding.
pub fn nucleus(sorted: &[(u32, f32)], top_p: f32, top_k: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(top_k);
    let mut mass = 0.0f32;
    for &c in sorted {
        // Upstream's loop condition, kept in this order on purpose: the token
        // that would carry the mass past `top_p` is **excluded**, so the
        // nucleus usually holds slightly less than `top_p`. Including it is
        // the natural way to write this and is a different distribution.
        if mass < top_p && out.len() < top_k {
            mass += c.1;
            out.push(c);
        } else {
            break;
        }
    }
    out
}

/// Picks one speech token.
///
/// `recent` is the tail of what has already been produced; only the last
/// `win_size` of it is looked at.
pub fn ras_sample(logits: &[f32], recent: &[u32], cfg: &RasConfig, rng: &mut Rng) -> u32 {
    let sorted = sorted_probs(logits);
    let top = draw(&nucleus(&sorted, cfg.top_p, cfg.top_k), rng);

    let window = &recent[recent.len().saturating_sub(cfg.win_size)..];
    let repeats = window.iter().filter(|&&t| t == top).count();

    // `>=`, and against a float threshold that is not generally an integer:
    // at the shipped `win_size = 10, tau_r = 0.1` the bar is 1.0, so a single
    // repeat inside the window is already enough to reject. Rounding this to
    // an integer count would be a different rule at other settings.
    if (repeats as f32) < cfg.win_size as f32 * cfg.tau_r {
        return top;
    }

    // Rejected. The redraw is over the **whole** distribution, not the
    // nucleus - upstream's `random_sampling` ignores `top_p` and `top_k`
    // entirely, which is what lets it escape a mode the nucleus is stuck in.
    let rest: Vec<(u32, f32)> = sorted.into_iter().filter(|&(id, _)| id != top).collect();
    draw(&rest, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distribution with a known shape, so the nucleus can be reasoned about.
    fn logits(probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn the_nucleus_excludes_the_token_that_would_cross_the_threshold() {
        // 0.5, 0.3, 0.2 with top_p = 0.8. Upstream checks the mass *before*
        // adding, so it takes 0.5 (mass 0 < 0.8) and 0.3 (mass 0.5 < 0.8),
        // then stops because 0.8 is not < 0.8. The third is excluded even
        // though the first two only just reach the threshold.
        let s = sorted_probs(&logits(&[0.5, 0.3, 0.2]));
        let n = nucleus(&s, 0.8, 25);
        assert_eq!(n.len(), 2, "{n:?}");
        assert_eq!(n[0].0, 0);
        assert_eq!(n[1].0, 1);
    }

    #[test]
    fn top_k_bounds_the_nucleus_even_when_the_mass_has_not_been_reached() {
        // Twenty equal candidates and top_k = 3: the mass after three is 0.15,
        // far below top_p, and the count is what stops it.
        let s = sorted_probs(&logits(&[0.05; 20]));
        assert_eq!(nucleus(&s, 0.8, 3).len(), 3);
    }

    #[test]
    fn a_repeating_token_is_rejected_and_something_else_is_drawn() {
        // One candidate has essentially all the mass, so the nucleus draw is
        // deterministic - which makes the rejection observable. With the token
        // already in the window, the result must be a different one.
        let l = logits(&[0.999, 0.0005, 0.0005]);
        let cfg = RasConfig::default();
        let mut rng = Rng::new(1);
        assert_eq!(ras_sample(&l, &[], &cfg, &mut rng), 0, "nothing to repeat");

        let mut rng = Rng::new(1);
        let got = ras_sample(&l, &[0], &cfg, &mut rng);
        assert_ne!(got, 0, "one repeat in the window should already reject");
    }

    #[test]
    fn the_window_is_the_last_few_and_not_the_whole_history() {
        // The same token, far enough back that it has left the window.
        let l = logits(&[0.999, 0.0005, 0.0005]);
        let cfg = RasConfig::default();
        let mut recent = vec![0u32];
        recent.extend(std::iter::repeat_n(1u32, cfg.win_size));
        let mut rng = Rng::new(1);
        assert_eq!(ras_sample(&l, &recent, &cfg, &mut rng), 0);
    }

    #[test]
    fn a_draw_is_reproducible_from_its_seed() {
        let l = logits(&[0.4, 0.3, 0.2, 0.1]);
        let cfg = RasConfig::default();
        let run = || {
            let mut rng = Rng::new(cfg.seed);
            (0..32)
                .map(|_| ras_sample(&l, &[], &cfg, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
