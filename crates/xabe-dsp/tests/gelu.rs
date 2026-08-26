//! GELU against values captured from PyTorch.
//!
//! Rust has no `erf`, so this is the one kernel where the arithmetic is an
//! approximation of the reference rather than a rearrangement of it. The pairs
//! below came from `torch.nn.functional.gelu` in float64, and they bracket
//! every branch of Cody's rational approximation - the `|x| <= 0.46875`
//! polynomial, the middle range, and the `|x| > 4` asymptotic form - on both
//! signs.

use xabe_dsp::gelu;

/// `(input, torch.nn.functional.gelu(input))`, float64.
#[rustfmt::skip]
const CASES: [(f64, f64); 24] = [
    (-8.0, -4.884_981_308_350_689e-15),
    (-5.0, -1.433_257_859_340_120_2e-06),
    (-3.0, -4.049_694_094_890_31e-3),
    (-2.0, -4.550_026_389_635_842e-2),
    (-1.5, -1.002_108_019_032_870_4e-01),
    (-1.0, -1.586_552_539_314_570_2e-01),
    (-0.7, -1.693_745_565_561_510_8e-01),
    (-0.46875, -1.498_238_303_986_488_8e-01),
    (-0.4, -1.378_313_033_558_703_2e-01),
    (-0.1, -4.601_721_627_229_71e-2),
    (-1e-08, -4.999_999_960_105_772e-9),
    (0.0, 0.0),
    (1e-08, 5.000_000_039_894_228e-9),
    (0.1, 5.398_278_372_770_290_4e-02),
    (0.4, 2.621_686_966_441_297e-1),
    (0.46875, 3.189_261_696_013_511e-1),
    (0.7, 5.306_254_434_438_489e-1),
    (1.0, 8.413_447_460_685_43e-1),
    (1.5, 1.399_789_198_096_713),
    (2.0, 1.954_499_736_103_641_6e+00),
    (3.0, 2.995_950_305_905_11),
    (4.0, 3.999_873_315_032_667_5e+00),
    (4.5, 4.499_984_710_470_938_5e+00),
    (8.0, 7.999_999_999_999_995),
];

#[test]
fn gelu_matches_torch_at_every_branch_boundary() {
    for (x, want) in CASES {
        let mut v = [x as f32];
        gelu(&mut v);
        let got = f64::from(v[0]);
        // f32 carries about 7 significant digits, so the comparison is
        // relative with an absolute floor for the values near zero.
        let tol = 1e-6 * want.abs() + 1e-9;
        assert!(
            (got - want).abs() <= tol,
            "gelu({x}) = {got}, torch says {want}",
        );
    }
}

#[test]
fn gelu_is_not_the_tanh_approximation() {
    // The tanh approximation is what a reimplementation reaches for when the
    // standard library has no erf. Measured against torch over [-6, 6], the two
    // differ by up to 4.7e-4, worst at |x| = 2.699 - an order of magnitude
    // above this project's tolerances. Asserting the gap exists, at the point
    // where it is widest, stops a future "simplify this" from being silent.
    let x = -2.699f32;
    let mut v = [x];
    gelu(&mut v);
    let tanh_approx = 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    assert!(
        (v[0] - tanh_approx).abs() > 4e-4,
        "expected the two forms to differ by ~4.7e-4 at the worst point, got {}",
        (v[0] - tanh_approx).abs(),
    );
}
