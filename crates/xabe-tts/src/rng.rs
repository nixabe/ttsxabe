//! A small deterministic normal generator.
//!
//! Used when the caller wants reproducible synthesis from a seed rather than
//! supplying its own noise. It deliberately does **not** try to reproduce
//! PyTorch's draws: two RNG implementations agreeing on a seed across languages
//! is not something to assume, and the differential tests do not rely on it -
//! they feed the reference's captured noise in directly.
//!
//! So the contract is only this: the same seed gives the same audio, on any
//! machine, forever.

/// xoshiro256++, seeded through SplitMix64.
///
/// Chosen for being short enough to read in one sitting and having no state
/// worth getting subtly wrong. Nothing here is cryptographic and nothing needs
/// to be.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seeds the generator. Every seed is valid, including zero.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    /// The next raw 64 bits.
    fn next_u64(&mut self) -> u64 {
        let r = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// A uniform in `(0, 1]`.
    ///
    /// Open at zero on purpose: Box-Muller takes a logarithm of it.
    fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 1.0) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `n` standard normal samples, by Box-Muller.
    pub fn normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = self.uniform();
            let u2 = self.uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < n {
                out.push((r * theta.sin()) as f32);
            }
        }
        out
    }
}
