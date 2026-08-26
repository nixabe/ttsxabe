//! The discrete Fourier transform, at any length.

/// A planned transform of one length.
///
/// The plan is a twiddle table and nothing else; the algorithm is chosen per
/// recursion level from the factorisation of `n`. That generality is not
/// decoration - Whisper's frontend wants `n = 400`, which is `2^4 * 5^2` and
/// so is out of reach of the radix-2 transform every FFT tutorial writes.
///
/// A prime `n` degenerates to an O(n^2) direct sum, correctly. Nothing in this
/// workspace asks for one, and refusing it here would trade a real answer for
/// an error on a case that never arrives.
#[derive(Debug, Clone)]
pub struct Fft {
    n: usize,
    /// `W_n^k = exp(-2*pi*i*k/n)` for `k` in `0..n`, interleaved re/im.
    ///
    /// Computed in f64 and rounded once. The alternative - stepping a complex
    /// rotor in f32 - accumulates a phase drift that grows with `n` and shows
    /// up as a spectral tilt rather than as noise, which is the kind of error
    /// that survives a plausibility check.
    tw: Vec<[f32; 2]>,
}

/// Smallest prime factor of `n`, which is the radix this level splits by.
fn smallest_factor(n: usize) -> usize {
    let mut f = 2;
    while f * f <= n {
        if n.is_multiple_of(f) {
            return f;
        }
        f += 1;
    }
    n
}

/// Recursive decimation in time, general radix.
///
/// Reads `src[0], src[stride], src[2*stride], ...` - `n` of them - and writes
/// `n` outputs to `dst`. `buf` is scratch of the same length, disjoint from
/// `dst`; each recursion swaps their roles, which is what keeps the whole
/// transform to two buffers and no allocation.
///
/// `step` is `tw.len() / n`, so a sub-transform of length `m` indexes the same
/// full-length table with `step * r`.
fn fft_into(
    src: &[[f32; 2]],
    stride: usize,
    n: usize,
    tw: &[[f32; 2]],
    step: usize,
    dst: &mut [[f32; 2]],
    buf: &mut [[f32; 2]],
) {
    if n == 1 {
        dst[0] = src[0];
        return;
    }
    let r = smallest_factor(n);
    let m = n / r;

    for j in 0..r {
        let (child_dst, child_buf) = (&mut buf[j * m..(j + 1) * m], &mut dst[j * m..(j + 1) * m]);
        fft_into(
            &src[j * stride..],
            stride * r,
            m,
            tw,
            step * r,
            child_dst,
            child_buf,
        );
    }

    // X[q*m + k] = sum_j W_n^{j*(q*m+k)} * Y_j[k], where Y_j is the transform
    // of the subsequence that starts at j and steps by r. For r = 2 this is
    // the butterfly everyone recognises; the loop is the same identity without
    // assuming which radix arrived.
    for k in 0..m {
        for q in 0..r {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for j in 0..r {
                let y = buf[j * m + k];
                let w = tw[step * ((j * (q * m + k)) % n)];
                re += y[0] * w[0] - y[1] * w[1];
                im += y[0] * w[1] + y[1] * w[0];
            }
            dst[q * m + k] = [re, im];
        }
    }
}

impl Fft {
    /// Plans a transform of length `n`.
    ///
    /// # Panics
    ///
    /// If `n` is zero.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "an FFT of length zero has no meaning");
        let tw = (0..n)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        Self { n, tw }
    }

    /// The planned length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the planned length is zero, which [`Fft::new`] refuses.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward transform of `n` interleaved re/im pairs.
    ///
    /// # Panics
    ///
    /// If `x` is not `2 * n` long.
    pub fn forward(&self, x: &mut [f32]) {
        assert_eq!(x.len(), 2 * self.n, "forward wants 2n interleaved floats");
        let src: Vec<[f32; 2]> = x.as_chunks::<2>().0.to_vec();
        let mut dst = vec![[0.0f32; 2]; self.n];
        let mut buf = vec![[0.0f32; 2]; self.n];
        fft_into(&src, 1, self.n, &self.tw, 1, &mut dst, &mut buf);
        for (out, c) in x.as_chunks_mut::<2>().0.iter_mut().zip(&dst) {
            *out = *c;
        }
    }

    /// Forward transform of real input, keeping the `n/2 + 1` unique bins.
    ///
    /// The full complex transform is run and half of it discarded. A packed
    /// real transform would halve the work, and would also be a second piece
    /// of index arithmetic to get right for no gain this workspace can
    /// measure - the mel frontend is a rounding error beside the model it
    /// feeds.
    ///
    /// # Panics
    ///
    /// If `x` is not `n` long or `out` is not `2 * (n/2 + 1)` long.
    pub fn forward_real(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.n, "forward_real wants n real samples");
        let bins = self.n / 2 + 1;
        assert_eq!(out.len(), 2 * bins, "forward_real writes n/2+1 bins");
        let src: Vec<[f32; 2]> = x.iter().map(|&v| [v, 0.0]).collect();
        let mut dst = vec![[0.0f32; 2]; self.n];
        let mut buf = vec![[0.0f32; 2]; self.n];
        fft_into(&src, 1, self.n, &self.tw, 1, &mut dst, &mut buf);
        for (o, c) in out.as_chunks_mut::<2>().0.iter_mut().zip(&dst[..bins]) {
            *o = *c;
        }
    }
}

/// The direct sum, which is the definition the transform is tested against.
///
/// O(n^2) and deliberately so: it is written to be read against the formula,
/// not run on real input.
pub fn dft(x: &[f32]) -> Vec<[f32; 2]> {
    let n = x.len() / 2;
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, c) in x.as_chunks::<2>().0.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * j % n) as f64 / n as f64;
                let (wr, wi) = (a.cos(), a.sin());
                re += c[0] as f64 * wr - c[1] as f64 * wi;
                im += c[0] as f64 * wi + c[1] as f64 * wr;
            }
            [re as f32, im as f32]
        })
        .collect()
}
