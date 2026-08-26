//! The transform against the direct sum it is an optimisation of.
//!
//! The direct sum in `dft` is O(n^2) and accumulates in f64, so it is both the
//! definition and the more accurate of the two. Every case here is a length
//! with a different factorisation, because the failure this catches is a
//! twiddle index that happens to be right for radix 2 and wrong for radix 5.

use xabe_dsp::{Fft, dft};

/// Deterministic pseudo-random input; a fixed signal would let a wrong
/// transform agree by symmetry.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// Largest absolute disagreement, scaled by the largest magnitude present -
/// an FFT's outputs span orders of magnitude, so a bare absolute bound would
/// be dominated by whichever bin happened to be loud.
fn max_rel(a: &[[f32; 2]], b: &[[f32; 2]]) -> f32 {
    let scale = a
        .iter()
        .flat_map(|c| [c[0].abs(), c[1].abs()])
        .fold(0.0f32, f32::max)
        .max(1e-12);
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| [(x[0] - y[0]).abs(), (x[1] - y[1]).abs()])
        .fold(0.0f32, f32::max)
        / scale
}

#[test]
fn matches_the_direct_sum_at_every_factorisation() {
    // 400 is Whisper's; 512 is pure radix 2; 200 and 60 mix; 27 is 3^3; 25 is
    // 5^2 with no radix 2 at all; 31 is prime and takes the O(n^2) path.
    for (i, &n) in [400usize, 512, 200, 60, 27, 25, 31, 1, 2]
        .iter()
        .enumerate()
    {
        let mut x = noise(2 * n, i as u64 + 1);
        let want = dft(&x);
        Fft::new(n).forward(&mut x);
        let got: Vec<[f32; 2]> = x.as_chunks::<2>().0.to_vec();
        let e = max_rel(&want, &got);
        assert!(e < 1e-5, "n={n}: {e:e} relative to the largest bin");
    }
}

#[test]
fn a_real_transform_is_the_first_half_of_the_complex_one() {
    let n = 400;
    let real = noise(n, 7);
    let mut interleaved: Vec<f32> = real.iter().flat_map(|&v| [v, 0.0]).collect();
    Fft::new(n).forward(&mut interleaved);

    let mut bins = vec![0.0f32; 2 * (n / 2 + 1)];
    Fft::new(n).forward_real(&real, &mut bins);

    assert_eq!(&bins[..], &interleaved[..bins.len()]);
}

#[test]
fn a_pure_tone_lands_in_one_bin() {
    // The sanity check the differential test cannot give: that bin `k` means
    // `k` cycles per window, and not `k` off by one or mirrored.
    let n = 64;
    let k = 5usize;
    let x: Vec<f32> = (0..n)
        .flat_map(|j| {
            let a = 2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
            [a.cos(), 0.0]
        })
        .collect();
    let mut y = x.clone();
    Fft::new(n).forward(&mut y);
    let mag: Vec<f32> = y
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| c[0].hypot(c[1]))
        .collect();
    // A real cosine splits its energy between k and n-k, each n/2.
    for (i, &m) in mag.iter().enumerate() {
        let want = if i == k || i == n - k {
            n as f32 / 2.0
        } else {
            0.0
        };
        assert!((m - want).abs() < 1e-3, "bin {i}: {m} wanted {want}");
    }
}
