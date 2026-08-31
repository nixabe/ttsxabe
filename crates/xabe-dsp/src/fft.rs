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
    //
    // The exponent is reduced by carrying it rather than by `%`, which is a
    // hardware division on a length that is not a compile-time constant and
    // was most of this transform's cost - the mel frontend spent 10 ms of 16
    // in here on a three-second clip. Two facts make the carry exact:
    //
    // - `j*q*m mod n` is `((j*q) mod r) * m`, because `m * r == n`. That is
    //   the base, and it is below `n`.
    // - `j*k` never reaches `n`: `j <= r-1` and `k <= m-1`, so their product
    //   is at most `n - r - m + 1`.
    //
    // So the running index is below `2n` and one conditional subtraction
    // reduces it. Accumulating into `dst` rather than into a scalar is what
    // lets `j` be the inner-most loop, which is what makes the carry possible.
    for q in 0..r {
        let out = &mut dst[q * m..(q + 1) * m];
        // `j == 0` has twiddle 1 for every `k`, so it is a copy - which is
        // also what leaves the accumulator initialised without a zeroing pass.
        out.copy_from_slice(&buf[..m]);
        for j in 1..r {
            let y = &buf[j * m..(j + 1) * m];
            let mut idx = (j * q % r) * m;
            for k in 0..m {
                let w = tw[step * idx];
                out[k][0] += y[k][0] * w[0] - y[k][1] * w[1];
                out[k][1] += y[k][0] * w[1] + y[k][1] * w[0];
                idx += j;
                if idx >= n {
                    idx -= n;
                }
            }
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
    /// of index arithmetic to get right for a gain this workspace has not
    /// needed yet.
    ///
    /// This allocates three buffers per call. A caller running a transform per
    /// frame of a spectrogram wants [`Fft::forward_real_with`] instead.
    ///
    /// # Panics
    ///
    /// If `x` is not `n` long or `out` is not `2 * (n/2 + 1)` long.
    pub fn forward_real(&self, x: &[f32], out: &mut [f32]) {
        self.forward_real_with(x, out, &mut self.scratch());
    }

    /// Working buffers sized for this plan, to be reused across transforms.
    pub fn scratch(&self) -> Scratch {
        Scratch {
            src: vec![[0.0; 2]; self.n],
            dst: vec![[0.0; 2]; self.n],
            buf: vec![[0.0; 2]; self.n],
        }
    }

    /// [`Fft::forward_real`], reusing buffers the caller owns.
    ///
    /// The transform needs three arrays of `n` complex numbers and cannot keep
    /// them itself without either interior mutability - which would cost
    /// `Sync`, and this plan is shared - or a `&mut self` that would stop two
    /// threads transforming different frames against one plan. So the
    /// scratch is a value the caller holds. A spectrogram is a transform per
    /// frame and there are three thousand frames in Whisper's window; the
    /// allocation was the larger half of the frontend's cost.
    ///
    /// # Panics
    ///
    /// If `x` is not `n` long, `out` is not `2 * (n/2 + 1)` long, or the
    /// scratch was made by a plan of a different length.
    pub fn forward_real_with(&self, x: &[f32], out: &mut [f32], s: &mut Scratch) {
        assert_eq!(x.len(), self.n, "forward_real wants n real samples");
        let bins = self.n / 2 + 1;
        assert_eq!(out.len(), 2 * bins, "forward_real writes n/2+1 bins");
        assert_eq!(s.src.len(), self.n, "the scratch is for a different length");
        for (c, &v) in s.src.iter_mut().zip(x) {
            *c = [v, 0.0];
        }
        fft_into(&s.src, 1, self.n, &self.tw, 1, &mut s.dst, &mut s.buf);
        for (o, c) in out.as_chunks_mut::<2>().0.iter_mut().zip(&s.dst[..bins]) {
            *o = *c;
        }
    }
}

/// Working buffers for one [`Fft`] plan, made by [`Fft::scratch`].
///
/// One per thread, not one per call: it exists so that a spectrogram does not
/// allocate three arrays per frame. It carries the plan's length and is
/// refused by a plan of any other.
pub struct Scratch {
    src: Vec<[f32; 2]>,
    dst: Vec<[f32; 2]>,
    buf: Vec<[f32; 2]>,
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

/// A centred real STFT, computed as a direct DFT.
///
/// The scalar twin of `xabe_cuda::Gpu::stft`. Returns `(real, imag, frames)`,
/// each spectrum laid out `[bins, frames]`.
///
/// Direct rather than a butterfly because the caller's `n_fft` is 16, and the
/// reflect padding reproduces torch's `center=True` default: frame `f` is
/// *centred* on sample `f * hop` rather than starting there. A version that
/// starts there shifts every frame by half a window, which sounds like a delay
/// and measures like noise.
pub fn stft(x: &[f32], window: &[f32], n_fft: usize, hop: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let n = x.len();
    let frames = n / hop + 1;
    let bins = n_fft / 2 + 1;
    let half = n_fft / 2;
    let (mut re, mut im) = (vec![0.0; bins * frames], vec![0.0; bins * frames]);

    for bin in 0..bins {
        for f in 0..frames {
            let (mut sr, mut si) = (0.0f32, 0.0f32);
            for (j, &w) in window.iter().enumerate().take(n_fft) {
                let mut at = (f * hop + j) as isize - half as isize;
                if at < 0 {
                    at = -at;
                }
                if at >= n as isize {
                    at = 2 * (n as isize - 1) - at;
                }
                let at = at.clamp(0, n as isize - 1) as usize;
                let v = x[at] * w;
                let ang = -std::f32::consts::TAU * bin as f32 * j as f32 / n_fft as f32;
                sr += v * ang.cos();
                si += v * ang.sin();
            }
            re[bin * frames + f] = sr;
            im[bin * frames + f] = si;
        }
    }
    (re, im, frames)
}

/// The inverse, as overlap-add with torch's window-envelope division.
///
/// The scalar twin of `xabe_cuda::Gpu::istft`.
pub fn istft(
    re: &[f32],
    im: &[f32],
    window: &[f32],
    frames: usize,
    n_fft: usize,
    hop: usize,
) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let half = n_fft / 2;
    let out_n = (frames - 1) * hop;
    let mut out = vec![0.0; out_n];

    for (s, o) in out.iter_mut().enumerate() {
        let p = s + half;
        let (mut acc, mut env) = (0.0f32, 0.0f32);
        let first = (p + 1).saturating_sub(n_fft).div_ceil(hop);
        for f in first..=(p / hop).min(frames.saturating_sub(1)) {
            let j = p - f * hop;
            if j >= n_fft {
                continue;
            }
            let mut v = 0.0f32;
            for b in 0..bins {
                let ang = std::f32::consts::TAU * b as f32 * j as f32 / n_fft as f32;
                let term = re[b * frames + f] * ang.cos() - im[b * frames + f] * ang.sin();
                // Bin 0 and, for even `n_fft`, Nyquist stand for themselves;
                // every other bin stands for a conjugate pair and counts
                // twice. Treating them alike is a constant offset on the
                // waveform.
                let edge = b == 0 || (n_fft.is_multiple_of(2) && b == bins - 1);
                v += if edge { term } else { 2.0 * term };
            }
            v /= n_fft as f32;
            let w = window[j];
            acc += v * w;
            env += w * w;
        }
        *o = if env > 1e-11 { acc / env } else { 0.0 };
    }
    out
}

/// A periodic Hann window, which is what `get_window("hann", n, fftbins=True)`
/// produces - the `fftbins` variant, dividing by `n` and not by `n - 1`.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect()
}
