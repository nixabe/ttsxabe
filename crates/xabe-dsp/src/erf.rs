//! The error function, and the GELU that needs it.
//!
//! PyTorch's default GELU is the *exact* one - `0.5 x (1 + erf(x / sqrt 2))` -
//! not the tanh approximation. The two differ by up to 4.7e-4, worst around
//! `|x| = 2.7`, which is an order of magnitude above the tolerance the rest of
//! this project runs at. So the approximation is not an option, and Rust's
//! standard library has no `erf`.
//!
//! What is here is W. J. Cody's rational Chebyshev approximation - the same
//! algorithm most libm implementations use - in three ranges, evaluated in
//! `f64`. It is accurate to near double precision, which leaves the GELU
//! limited by `f32` rounding rather than by the approximation.

/// `1 / sqrt(pi)`.
const INV_SQRT_PI: f64 = 5.641_895_835_477_563e-1;

/// `1 / sqrt(2)`.
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[rustfmt::skip]
const A: [f64; 5] = [
    3.161_123_743_870_565_6e0, 1.138_641_541_510_501_6e2, 3.774_852_376_853_02e2,
    3.209_377_589_138_469_5e3, 1.857_777_061_846_031_5e-1,
];

#[rustfmt::skip]
const B: [f64; 4] = [
    2.360_129_095_234_412_1e1, 2.440_246_379_344_441_7e2,
    1.282_616_526_077_372_3e3, 2.844_236_833_439_171e3,
];

#[rustfmt::skip]
const C: [f64; 9] = [
    5.641_884_969_886_701e-1, 8.883_149_794_388_376e0, 6.611_919_063_714_163e1,
    2.986_351_381_974_001e2, 8.819_522_212_417_69e2, 1.712_047_612_634_070_6e3,
    2.051_078_377_826_071_5e3, 1.230_339_354_797_997_2e3, 2.153_115_354_744_038_5e-8,
];

#[rustfmt::skip]
const D: [f64; 8] = [
    1.574_492_611_070_983_5e1, 1.176_939_508_913_125e2, 5.371_811_018_620_099e2,
    1.621_389_574_566_690_2e3, 3.290_799_235_733_459_6e3, 4.362_619_090_143_247e3,
    3.439_367_674_143_721_6e3, 1.230_339_354_803_749_4e3,
];

#[rustfmt::skip]
const P: [f64; 6] = [
    3.053_266_349_612_323_4e-1, 3.603_448_999_498_044_4e-1, 1.257_817_261_112_292_5e-1,
    1.608_378_514_874_228e-2, 6.587_491_615_298_378e-4, 1.631_538_713_730_209_8e-2,
];

#[rustfmt::skip]
const Q: [f64; 5] = [
    2.568_520_192_289_822, 1.872_952_849_923_460_5e0, 5.279_051_029_514_284e-1,
    6.051_834_131_244_132e-2, 2.335_204_976_268_691_8e-3,
];

/// The complementary error function, for non-negative `y`.
fn erfc_pos(y: f64) -> f64 {
    if y <= 0.46875 {
        return 1.0 - erf_small(y);
    }
    if y <= 4.0 {
        let mut num = C[8];
        let mut den = 1.0;
        for i in 0..8 {
            num = num * y + C[i];
            den = den * y + D[i];
        }
        return (-y * y).exp() * num / den;
    }
    let z = 1.0 / (y * y);
    let mut num = P[5];
    let mut den = 1.0;
    for i in 0..5 {
        num = num * z + P[i];
        den = den * z + Q[i];
    }
    let r = z * num / den;
    (-y * y).exp() * (INV_SQRT_PI - r) / y
}

/// The error function on the central range, where the rational form is direct.
fn erf_small(x: f64) -> f64 {
    let z = x * x;
    let num = (((A[4] * z + A[0]) * z + A[1]) * z + A[2]) * z + A[3];
    let den = (((z + B[0]) * z + B[1]) * z + B[2]) * z + B[3];
    x * num / den
}

/// The error function.
pub fn erf(x: f64) -> f64 {
    let y = x.abs();
    let v = if y <= 0.46875 {
        erf_small(y)
    } else {
        1.0 - erfc_pos(y)
    };
    if x < 0.0 { -v } else { v }
}

/// Gaussian error linear unit, in place.
///
/// The exact form, matching `torch.nn.functional.gelu` with its default
/// `approximate="none"`. The tanh approximation differs by up to 4.7e-4, worst
/// around `|x| = 2.7`, which this project's tolerances would reject.
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let d = f64::from(*v);
        *v = (0.5 * d * (1.0 + erf(d * INV_SQRT_2))) as f32;
    }
}
