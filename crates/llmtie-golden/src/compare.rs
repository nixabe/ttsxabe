//! Comparing a computed stage against the captured one.
//!
//! A boolean pass/fail is almost useless when a stage disagrees: every kernel
//! in this model is a few lines of arithmetic, and the *shape* of the
//! disagreement is what identifies which line. A transposed axis produces a
//! huge error at nearly every index; an off-by-one in a convolution's padding
//! produces a huge error at exactly the edges; an accumulation-order difference
//! produces a tiny error everywhere. [`Comparison`] carries enough to tell
//! those apart without re-running anything.

use std::fmt;

/// The result of diffing one computed stage against its captured counterpart.
#[derive(Debug, Clone)]
pub struct Comparison {
    /// The stage that was compared.
    pub name: String,
    /// How many values were compared.
    pub count: usize,
    /// Largest absolute difference over all values.
    pub max_abs: f32,
    /// Largest relative difference, taken against the oracle's magnitude.
    pub max_rel: f32,
    /// Mean absolute difference. Small mean with large max means a few bad
    /// indices; large mean means the whole tensor is wrong.
    pub mean_abs: f32,
    /// Index of the largest absolute difference.
    pub worst_index: usize,
    /// The oracle's value at [`Self::worst_index`].
    pub worst_expected: f32,
    /// The computed value at [`Self::worst_index`].
    pub worst_actual: f32,
    /// How many computed values are NaN or infinite. Non-zero here means the
    /// arithmetic broke down, not that it drifted.
    pub non_finite: usize,
    /// How many values exceed the tolerance the comparison was judged at.
    pub over_tolerance: usize,
}

impl Comparison {
    /// Diffs `actual` against `expected`, judging each value at
    /// `atol + rtol * |expected|`.
    ///
    /// The mixed absolute/relative tolerance is the same rule NumPy's
    /// `allclose` uses, and it is the right one here: the waveform lives in
    /// [-1, 1] where absolute error is meaningful, while `logs_p` and the
    /// duration logits span several orders of magnitude where it is not.
    pub fn new(name: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> Self {
        let mut c = Self {
            name: name.to_string(),
            count: expected.len(),
            max_abs: 0.0,
            max_rel: 0.0,
            mean_abs: 0.0,
            worst_index: 0,
            worst_expected: 0.0,
            worst_actual: 0.0,
            non_finite: 0,
            over_tolerance: 0,
        };

        let mut sum = 0.0f64;
        for (i, (&e, &a)) in expected.iter().zip(actual).enumerate() {
            if !a.is_finite() {
                c.non_finite += 1;
            }
            let d = (e - a).abs();
            sum += f64::from(d);
            if d > c.max_abs {
                c.max_abs = d;
                c.worst_index = i;
                c.worst_expected = e;
                c.worst_actual = a;
            }
            if e != 0.0 {
                c.max_rel = c.max_rel.max(d / e.abs());
            }
            if d > atol + rtol * e.abs() {
                c.over_tolerance += 1;
            }
        }
        if c.count > 0 {
            c.mean_abs = (sum / c.count as f64) as f32;
        }
        c
    }

    /// Whether every value was within the tolerance this comparison was built
    /// with, and none of them was NaN or infinite.
    pub fn passed(&self) -> bool {
        self.over_tolerance == 0 && self.non_finite == 0
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}/{} values out of tolerance, max abs {:.3e}, max rel {:.3e}, \
             mean abs {:.3e}; worst at [{}] oracle {:.6} vs computed {:.6}",
            self.name,
            self.over_tolerance,
            self.count,
            self.max_abs,
            self.max_rel,
            self.mean_abs,
            self.worst_index,
            self.worst_expected,
            self.worst_actual,
        )?;
        if self.non_finite > 0 {
            write!(f, "; {} non-finite values", self.non_finite)?;
        }
        Ok(())
    }
}
