//! Milestone 11: every CUDA kernel against its `xabe-dsp` scalar twin.
//!
//! Per kernel, on the same inputs, before anything is assembled from them. A
//! GPU pipeline that is wrong somewhere is nearly impossible to bisect after
//! the fact and trivially bisected before it exists, so this file comes first.
//!
//! The inputs are pseudo-random rather than captured: these are kernels, not
//! stages, and what matters is that they agree on arbitrary numbers at the
//! awkward shapes - odd lengths, even kernels, dilation, a bias and no bias.
//! The captured tensors are for the stage tests, where the values mean
//! something.
//!
//! Skips when there is no device. `Gpu::open` reports that as an ordinary
//! error rather than failing to link, so this file compiles and runs on a
//! machine with no CUDA at all.

use xabe_cuda::{Batch, Gpu, Operand};

/// Absolute floor, for values near zero.
const ATOL: f32 = 1e-5;

/// Relative tolerance, which is what actually judges these.
///
/// The GPU fuses multiply-add where the scalar twin does not, so the last bits
/// differ - in the GPU's favour, since FMA rounds once instead of twice. A
/// purely absolute tolerance is the wrong rule here and was the wrong rule at
/// first: these kernels accumulate hundreds of terms, so a dot product of
/// magnitude 4e4 is out by 1e-2 while being correct to within one ulp.
const RTOL: f32 = 1e-5;

/// A deterministic spread of values in roughly [-2, 2].
///
/// Not a real RNG: a kernel test wants *some* numbers with sign changes and no
/// pattern that could hide an index error, and wants the same ones every run.
fn seq(n: usize, salt: u64) -> Vec<f32> {
    let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) * 4.0 - 2.0
        })
        .collect()
}

/// Largest absolute difference, with a name for the failure message.
/// The same, with the tolerance named at the call site.
///
/// The llama kernels use CUDA's fast intrinsics - `__expf`, `__sincosf`,
/// `__powf` - which are about 2^-21 accurate against CPU twins that are not.
/// Folding that into the shared constants would loosen every other test to
/// suit three.
fn assert_close_to(name: &str, want: &[f32], got: &[f32], tol: f32) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let (worst, at) = want
        .iter()
        .zip(got)
        .enumerate()
        .map(|(i, (a, b))| ((a - b).abs(), i))
        .fold((0.0f32, 0), |acc, e| if e.0 > acc.0 { e } else { acc });
    assert!(
        worst / scale < tol,
        "{name}: {:e} of full scale at [{at}], cpu {} vs gpu {}",
        worst / scale,
        want[at],
        got[at],
    );
}

fn assert_close(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let mut worst = 0.0f32;
    let mut at = 0;
    let mut over = 0;
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        let d = (a - b).abs();
        if d > ATOL + RTOL * a.abs() {
            over += 1;
        }
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        over == 0,
        "{name}: {over}/{} values out of tolerance; worst {worst:.3e} at [{at}], \
         cpu {} vs gpu {}",
        want.len(),
        want[at],
        got[at],
    );
}

/// Opens the device, skipping only when there genuinely is not one.
///
/// Skipping on *any* error is a trap, and this file fell into it once: a
/// kernel that failed to compile was reported as an absent GPU, and twelve
/// tests passed without running. A missing device is an environment fact; a
/// compile failure is a defect, and it has to fail.
fn gpu() -> Option<Gpu> {
    match Gpu::open(ordinal()) {
        Ok(g) => Some(g),
        Err(xabe_cuda::CudaError::NoDevice(why)) => {
            eprintln!("SKIP: no CUDA device ({why})");
            None
        }
        Err(e) => panic!("the device is present but unusable: {e}"),
    }
}

/// Which device to use. Check `nvidia-smi` first - do not clobber other
/// people's jobs.
///
/// `XABE_TEST_DEVICE` and not `XABE_TTS_DEVICE`: the latter is the engine's
/// `--tts-device` env twin, and setting it to steer a test run also reaches
/// into `xabe-engine`'s flag tests, which then assert against the card someone
/// happened to pick. That cost eight failing tests once.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

#[test]
fn conv1d_matches_at_every_awkward_shape() {
    let Some(g) = gpu() else { return };

    // Odd lengths, an even kernel, dilation, and both with and without a bias.
    // The even kernel is the one that matters: its padding is asymmetric.
    for &(in_ch, t, out_ch, k, dilation, with_bias) in &[
        (3usize, 17usize, 5usize, 3usize, 1usize, true),
        (3, 17, 5, 3, 1, false),
        (8, 31, 8, 5, 1, true),
        (4, 23, 6, 4, 1, true),
        (4, 23, 6, 3, 3, true),
        (4, 23, 6, 3, 9, true),
        (16, 64, 32, 7, 1, true),
    ] {
        let x = seq(in_ch * t, 1);
        let w = seq(out_ch * in_ch * k, 2);
        let b = seq(out_ch, 3);
        let bias = with_bias.then_some(&b[..]);
        let (pl, pr) = xabe_dsp::same_padding(if dilation == 1 {
            k
        } else {
            k * dilation - dilation + 1
        });

        let want = xabe_dsp::conv1d(&x, in_ch, t, &w, bias, out_ch, k, pl, pr, dilation);

        let dx = g.upload(&x).unwrap();
        let dw = g.upload(&w).unwrap();
        let db = g.upload(&b).unwrap();
        let (out, _) = g
            .conv1d(
                &dx,
                &dw,
                with_bias.then_some(&db),
                in_ch,
                t,
                out_ch,
                k,
                pl,
                pr,
                dilation,
            )
            .unwrap();
        assert_close(
            &format!("conv1d {in_ch}x{t} -> {out_ch}, k{k} d{dilation} bias={with_bias}"),
            &want,
            &g.download(&out).unwrap(),
        );
    }
}

#[test]
fn depthwise_conv1d_matches() {
    let Some(g) = gpu() else { return };
    for &(ch, t, k, dilation) in &[
        (8usize, 31usize, 3usize, 1usize),
        (16, 47, 3, 3),
        (16, 47, 3, 9),
    ] {
        let x = seq(ch * t, 4);
        let w = seq(ch * k, 5);
        let b = seq(ch, 6);
        let pad = (k * dilation - dilation) / 2;

        let want = xabe_dsp::depthwise_conv1d(&x, ch, t, &w, Some(&b), k, pad, pad, dilation);
        let dx = g.upload(&x).unwrap();
        let dw = g.upload(&w).unwrap();
        let db = g.upload(&b).unwrap();
        let (out, _) = g
            .depthwise_conv1d(&dx, &dw, Some(&db), ch, t, k, pad, pad, dilation)
            .unwrap();
        assert_close(
            &format!("depthwise {ch}x{t} k{k} d{dilation}"),
            &want,
            &g.download(&out).unwrap(),
        );
    }
}

#[test]
fn transposed_conv1d_matches_the_scatter_form() {
    let Some(g) = gpu() else { return };

    // The CPU twin scatters and this one gathers, so they are not the same
    // algorithm written twice - they are an inverse pair, and the inversion is
    // where the off-by-ones live. These are the decoder's real shapes.
    for &(in_ch, t, out_ch, k, stride) in &[
        (8usize, 13usize, 4usize, 16usize, 8usize),
        (8, 13, 4, 4, 2),
        (6, 9, 3, 5, 3),
        (4, 7, 2, 3, 1),
    ] {
        let x = seq(in_ch * t, 7);
        let w = seq(in_ch * out_ch * k, 8);
        let b = seq(out_ch, 9);
        let pad = (k - stride) / 2;

        let want = xabe_dsp::transposed_conv1d(&x, in_ch, t, &w, Some(&b), out_ch, k, stride, pad);
        let dx = g.upload(&x).unwrap();
        let dw = g.upload(&w).unwrap();
        let db = g.upload(&b).unwrap();
        let (out, out_t) = g
            .transposed_conv1d(&dx, &dw, Some(&db), in_ch, t, out_ch, k, stride, pad)
            .unwrap();
        assert_eq!(out_t * out_ch, want.len());
        assert_close(
            &format!("transposed {in_ch}x{t} -> {out_ch}, k{k} s{stride}"),
            &want,
            &g.download(&out).unwrap(),
        );
    }
}

#[test]
fn linear_matches() {
    let Some(g) = gpu() else { return };
    let (rows, in_c, out_c) = (37usize, 64usize, 48usize);
    let x = seq(rows * in_c, 10);
    let w = seq(out_c * in_c, 11);
    let b = seq(out_c, 12);

    let want = xabe_dsp::linear(&x, rows, in_c, &w, Some(&b), out_c);
    let dx = g.upload(&x).unwrap();
    let dw = g.upload(&w).unwrap();
    let db = g.upload(&b).unwrap();
    let out = g.linear(&dx, &dw, Some(&db), rows, in_c, out_c).unwrap();
    assert_close("linear", &want, &g.download(&out).unwrap());
}

#[test]
fn layer_norm_matches() {
    let Some(g) = gpu() else { return };
    // Column counts above and below the block size, and one that is not a
    // multiple of it - the reduction strides by blockDim and the tail is where
    // a shared-memory reduction goes wrong.
    for &(rows, cols) in &[(11usize, 192usize), (5, 700), (3, 257), (2, 1)] {
        let x = seq(rows * cols, 13);
        let w = seq(cols, 14);
        let b = seq(cols, 15);

        let want = xabe_dsp::layer_norm(&x, rows, cols, &w, &b, 1e-5);
        let dx = g.upload(&x).unwrap();
        let dw = g.upload(&w).unwrap();
        let db = g.upload(&b).unwrap();
        let out = g.layer_norm(&dx, rows, cols, &dw, &db, 1e-5).unwrap();
        assert_close(
            &format!("layer_norm {rows}x{cols}"),
            &want,
            &g.download(&out).unwrap(),
        );
    }
}

/// The fused residual-and-normalise, on both of the things it writes.
///
/// It updates the residual stream in place *and* returns the normalisation of
/// it, and the in-place half is the one that is easy to get wrong quietly: a
/// kernel that normalised the sum correctly but left `h` unsummed would pass
/// any test that only looked at the return, and the next sub-layer would add
/// to a stale stream. Both are checked, at the same column counts the
/// unfused kernel is checked at.
#[test]
fn layer_norm_add_matches_the_sum_and_the_normalisation() {
    let Some(g) = gpu() else { return };
    for &(rows, cols) in &[(11usize, 192usize), (5, 700), (3, 257), (2, 1)] {
        let h = seq(rows * cols, 41);
        let res = seq(rows * cols, 42);
        let w = seq(cols, 43);
        let b = seq(cols, 44);

        let mut want_h = h.clone();
        let want = xabe_dsp::layer_norm_add(&mut want_h, &res, rows, cols, &w, &b, 1e-5);

        let mut dh = g.upload(&h).unwrap();
        let dres = g.upload(&res).unwrap();
        let dw = g.upload(&w).unwrap();
        let db = g.upload(&b).unwrap();
        let out = g
            .layer_norm_add(&mut dh, &dres, rows, cols, &dw, &db, 1e-5)
            .unwrap();
        assert_close(
            &format!("layer_norm_add {rows}x{cols}"),
            &want,
            &g.download(&out).unwrap(),
        );
        assert_close(
            &format!("layer_norm_add residual {rows}x{cols}"),
            &want_h,
            &g.download(&dh).unwrap(),
        );
    }
}

/// The packed head splits are the f32 ones with `to_f16` applied, exactly.
///
/// They exist to save a pass, not to round differently, so the claim is
/// equality of every bit against the two-kernel chain they replace - anything
/// looser would mean the cross-attention cache is a second approximation
/// rather than the same one, and it is read by every decode step.
#[test]
fn the_packed_head_splits_are_the_f32_ones_converted() {
    let Some(g) = gpu() else { return };
    for &(t, heads, hd) in &[(7usize, 3usize, 4usize), (1500, 20, 64), (1, 2, 8)] {
        let x = seq(t * heads * hd, 61);
        let dx = g.upload(&x).unwrap();
        for (plain, packed) in [
            (
                g.split_heads(&dx, t, heads, hd).unwrap(),
                g.split_heads_f16(&dx, t, heads, hd).unwrap(),
            ),
            (
                g.split_heads_t(&dx, t, heads, hd).unwrap(),
                g.split_heads_t_f16(&dx, t, heads, hd).unwrap(),
            ),
        ] {
            let want = g
                .download_u16(&g.to_f16(&plain, t * heads * hd).unwrap())
                .unwrap();
            let got = g.download_u16(&packed).unwrap();
            assert_eq!(want, got, "split {t}x{heads}x{hd} disagrees packed");
        }
    }
}

#[test]
fn softmax_matches() {
    let Some(g) = gpu() else { return };
    for &(rows, cols) in &[(9usize, 69usize), (4, 513), (3, 1)] {
        let x = seq(rows * cols, 16);
        let mut want = x.clone();
        xabe_dsp::softmax_rows(&mut want, rows, cols);

        let mut dx = g.upload(&x).unwrap();
        g.softmax_rows(&mut dx, rows, cols).unwrap();
        assert_close(
            &format!("softmax {rows}x{cols}"),
            &want,
            &g.download(&dx).unwrap(),
        );
    }
}

#[test]
fn the_activations_match() {
    let Some(g) = gpu() else { return };
    let n = 1000;
    let x = seq(n, 17);

    let mut want = x.clone();
    xabe_dsp::relu(&mut want);
    let mut d = g.upload(&x).unwrap();
    g.relu(&mut d, n).unwrap();
    assert_close("relu", &want, &g.download(&d).unwrap());

    let mut want = x.clone();
    xabe_dsp::leaky_relu(&mut want, 0.1);
    let mut d = g.upload(&x).unwrap();
    g.leaky_relu(&mut d, n, 0.1).unwrap();
    assert_close("leaky_relu", &want, &g.download(&d).unwrap());

    let want: Vec<f32> = x.iter().map(|v| v.tanh()).collect();
    let mut d = g.upload(&x).unwrap();
    g.tanh(&mut d, n).unwrap();
    assert_close("tanh", &want, &g.download(&d).unwrap());

    // The device has an IEEE-accurate erff and the CPU has Cody's rational
    // approximation, so this compares two different implementations of the
    // same function rather than one implementation twice. Agreeing to 1e-5 is
    // evidence about both.
    let mut want = x.clone();
    xabe_dsp::gelu(&mut want);
    let mut d = g.upload(&x).unwrap();
    g.gelu(&mut d, n).unwrap();
    assert_close("gelu", &want, &g.download(&d).unwrap());
}

#[test]
fn gated_activation_matches() {
    let Some(g) = gpu() else { return };
    let (ch, t) = (32usize, 45usize);
    let x = seq(2 * ch * t, 18);
    let want = xabe_dsp::gated_activation(&x, ch, t);
    let d = g.upload(&x).unwrap();
    let out = g.gated_activation(&d, ch, t).unwrap();
    assert_close("gated_activation", &want, &g.download(&out).unwrap());
}

#[test]
fn the_layout_kernels_match() {
    let Some(g) = gpu() else { return };
    let (rows, cols) = (23usize, 17usize);
    let x = seq(rows * cols, 19);

    let want = xabe_dsp::transpose(&x, rows, cols);
    let d = g.upload(&x).unwrap();
    let out = g.transpose(&d, rows, cols).unwrap();
    assert_close("transpose", &want, &g.download(&out).unwrap());

    // Reversal, not a half-swap. An odd channel count makes the two impossible
    // to confuse.
    let want = xabe_dsp::flip_channels(&x, rows, cols);
    let out = g.flip_channels(&d, rows, cols).unwrap();
    assert_close("flip_channels", &want, &g.download(&out).unwrap());
}

#[test]
fn fuse_weight_norm_matches() {
    let Some(g) = gpu() else { return };
    for &(out_ch, in_ch, k) in &[(8usize, 16usize, 5usize), (64, 192, 1), (3, 700, 1)] {
        let v = seq(out_ch * in_ch * k, 20);
        let gg = seq(out_ch, 21);
        let want = xabe_dsp::fuse_weight_norm(&v, &gg, out_ch, in_ch, k);

        let dv = g.upload(&v).unwrap();
        let dg = g.upload(&gg).unwrap();
        let out = g.fuse_weight_norm(&dv, &dg, out_ch, in_ch, k).unwrap();
        assert_close(
            &format!("fuse_weight_norm {out_ch}x{in_ch}x{k}"),
            &want,
            &g.download(&out).unwrap(),
        );
    }
}

#[test]
fn attention_matches() {
    let Some(g) = gpu() else { return };
    let (t, embed, heads, window) = (29usize, 64usize, 2usize, 4usize);
    let head_dim = embed / heads;
    let span = 2 * window + 1;

    let x = seq(t * embed, 22);
    let qw = seq(embed * embed, 23);
    let kw = seq(embed * embed, 24);
    let vw = seq(embed * embed, 25);
    let ow = seq(embed * embed, 26);
    let qb = seq(embed, 27);
    let kb = seq(embed, 28);
    let vb = seq(embed, 29);
    let ob = seq(embed, 30);
    let erk = seq(span * head_dim, 31);
    let erv = seq(span * head_dim, 32);

    let want = xabe_dsp::self_attention(
        &x,
        t,
        embed,
        heads,
        window,
        &qw,
        Some(&qb),
        &kw,
        Some(&kb),
        &vw,
        Some(&vb),
        &ow,
        Some(&ob),
        &erk,
        &erv,
    );

    // The GPU path splits what the CPU does in one function into four kernels,
    // so this is a composition test as much as a kernel test - which is the
    // point, since it is the composition the pipeline will use.
    let dx = g.upload(&x).unwrap();
    let up = |v: &[f32]| g.upload(v).unwrap();
    let (dqw, dkw, dvw, dow) = (up(&qw), up(&kw), up(&vw), up(&ow));
    let (dqb, dkb, dvb, dob) = (up(&qb), up(&kb), up(&vb), up(&ob));
    let (derk, derv) = (up(&erk), up(&erv));

    let q = g.linear(&dx, &dqw, Some(&dqb), t, embed, embed).unwrap();
    let k = g.linear(&dx, &dkw, Some(&dkb), t, embed, embed).unwrap();
    let v = g.linear(&dx, &dvw, Some(&dvb), t, embed, embed).unwrap();
    let mut scores = g
        .attention_scores(&q, &k, &derk, t, embed, heads, window)
        .unwrap();
    g.softmax_rows(&mut scores, heads * t, t).unwrap();
    let ctx = g
        .attention_context(&scores, &v, &derv, t, embed, heads, window)
        .unwrap();
    let out = g.linear(&ctx, &dow, Some(&dob), t, embed, embed).unwrap();

    assert_close("self_attention", &want, &g.download(&out).unwrap());
}

#[test]
fn the_elementwise_kernels_match() {
    let Some(g) = gpu() else { return };
    let n = 777;
    let a = seq(n, 33);
    let b = seq(n, 34);

    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let mut d = g.upload(&a).unwrap();
    let db = g.upload(&b).unwrap();
    g.add_inplace(&mut d, &db, n).unwrap();
    assert_close("add_inplace", &want, &g.download(&d).unwrap());

    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    let mut d = g.upload(&a).unwrap();
    g.mul_inplace(&mut d, &db, n).unwrap();
    assert_close("mul_inplace", &want, &g.download(&d).unwrap());

    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let mut d = g.upload(&a).unwrap();
    g.sub_inplace(&mut d, &db, n).unwrap();
    assert_close("sub_inplace", &want, &g.download(&d).unwrap());

    // `add_strided` takes a column block of a wide matrix. The offset is
    // deliberately not zero and the stride not a multiple of the width, so a
    // kernel that quietly treated it as contiguous would disagree here.
    let (rows, cols, stride, off) = (37usize, 11usize, 40usize, 17usize);
    let narrow = seq(rows * cols, 35);
    let wide = seq(rows * stride, 36);
    let want: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, j) = (i / cols, i % cols);
            narrow[i] + wide[r * stride + off + j]
        })
        .collect();
    let mut d = g.upload(&narrow).unwrap();
    let dw = g.upload(&wide).unwrap();
    g.add_strided(&mut d, &dw, cols, stride, off, rows).unwrap();
    assert_close("add_strided", &want, &g.download(&d).unwrap());

    let want: Vec<f32> = a.iter().map(|x| x / 3.0).collect();
    let mut d = g.upload(&a).unwrap();
    g.scale_inplace(&mut d, n, 1.0 / 3.0).unwrap();
    assert_close("scale_inplace", &want, &g.download(&d).unwrap());
}

// ---------------------------------------------------------------------- gemm

/// Rounds a slice through f16, so the reference sees what the kernel sees.
///
/// This is the whole trick that makes a tensor-core kernel testable. `gemm`
/// rounds both operands to f16 before multiplying; a reference fed the original
/// f32 values is measuring that rounding, not the kernel, and no tolerance can
/// separate the two. Feeding the reference the same rounded values leaves only
/// accumulation order, which is what a differential test is supposed to judge.
fn through_f16(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| f32::from(half::f16::from_f32(*v)))
        .collect()
}

/// Gate for `gemm` against a reference fed the same f16-rounded operands.
///
/// Both sides now compute the same products; only the summation order differs,
/// and the accumulator is f32 on both. Measured worst across the shapes below
/// is 1.2e-6 relative.
const GEMM_RTOL: f32 = 1e-4;

/// Absolute floor, for outputs near zero where relative error is meaningless.
const GEMM_ATOL: f32 = 1e-4;

fn assert_close_gemm(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length");
    let mut worst_rel = 0.0f32;
    let mut at = 0;
    let mut over = 0;
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        let d = (a - b).abs();
        if d > GEMM_ATOL + GEMM_RTOL * a.abs() {
            over += 1;
        }
        let rel = d / a.abs().max(1e-3);
        if rel > worst_rel {
            worst_rel = rel;
            at = i;
        }
    }
    assert!(
        over == 0,
        "{name}: {over}/{} values out of tolerance; worst relative {worst_rel:.3e} \
         at [{at}], cpu {} vs gpu {}",
        want.len(),
        want[at],
        got[at],
    );
}

/// The scalar product `gemm` should agree with, fed the operands it will
/// actually see.
///
/// Which reference depends on which kernel the shape dispatches to: at or
/// below `GEMV_MAX_M` rows the path is exact f32, above it the operands are
/// rounded to f16 first. Getting this wrong measures the rounding rather than
/// the kernel, which is a mistake this file has already made once.
fn reference_gemm(a: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let (a, w) = (&a[..m * k], &w[..n * k]);
    if m <= xabe_cuda::GEMV_MAX_M {
        xabe_dsp::linear(a, m, k, w, None, n)
    } else {
        xabe_dsp::linear(&through_f16(a), m, k, &through_f16(w), None, n)
    }
}

#[test]
fn gemm_matches_the_scalar_linear_at_every_awkward_shape() {
    let Some(g) = gpu() else { return };

    // The shapes that matter, plus the ones that break tiling. 64x32 is the
    // block tile, so anything not a multiple of it exercises the predication:
    // a partial m tile, a partial n tile, and both at once.
    let shapes: &[(usize, usize, usize)] = &[
        (16, 8, 8),         // one instruction's worth, and the dispatch boundary
        (17, 8, 8),         // one row past it, so both kernels are exercised
        (1, 1280, 1280),    // a single token: the decode shape
        (64, 32, 32),       // exactly one block tile
        (65, 32, 32),       // one row into the next m tile
        (64, 32, 33),       // one column into the next n tile
        (63, 40, 31),       // partial in both, and k not a multiple of 32
        (7, 1280, 51),      // nothing lines up with anything
        (1500, 1280, 1280), // the encoder's self-attention projections
    ];

    for &(m, k, n) in shapes {
        let name = format!("gemm {m}x{k}x{n}");
        let a = seq(m * k, 1);
        let w = seq(n * k, 2);
        let bias = seq(n, 3);

        // Which reference depends on which kernel `gemm` will dispatch to, and
        // that is a property of the shape. At or below GEMV_MAX_M rows it takes
        // the exact scalar path, so the reference must be exact too; above it
        // the operands are rounded to f16 and the reference has to see the same
        // values or it is measuring the rounding rather than the kernel.
        let (a_ref, w_ref) = if m <= xabe_cuda::GEMV_MAX_M {
            (a.clone(), w.clone())
        } else {
            (through_f16(&a), through_f16(&w))
        };
        let want = xabe_dsp::linear(&a_ref, m, k, &w_ref, Some(&bias), n);
        let got = g
            .download(
                &g.gemm(
                    &g.upload(&a).expect("upload a"),
                    &g.upload(&w).expect("upload w"),
                    Some(&g.upload(&bias).expect("upload bias")),
                    m,
                    k,
                    n,
                )
                .expect("gemm"),
            )
            .expect("download");
        assert_close_gemm(&name, &want, &got);
    }
}

#[test]
fn gemm_without_a_bias_adds_nothing() {
    let Some(g) = gpu() else { return };
    let (m, k, n) = (64, 64, 32);
    let a = seq(m * k, 11);
    let w = seq(n * k, 12);

    let want = xabe_dsp::linear(&through_f16(&a), m, k, &through_f16(&w), None, n);
    let got = g
        .download(
            &g.gemm(
                &g.upload(&a).expect("upload"),
                &g.upload(&w).expect("upload"),
                None,
                m,
                k,
                n,
            )
            .expect("gemm"),
        )
        .expect("download");
    assert_close_gemm("gemm no bias", &want, &got);
}

#[test]
fn gemm_accepts_every_contraction_length() {
    let Some(g) = gpu() else { return };
    // This test twice asserted a refusal, and both were wrong about the
    // kernel. "A multiple of 8" was wrong about the instruction: the staging
    // loop zero-extends a short trip and `m16n8k8` accumulates the zeros.
    // "Even" was right about the `float2` staging - the load is at offset
    // `row * k + kk` with `kk` even, so an odd `k` misaligns every row after
    // the first - but the fix was to stage an odd `k` scalar, not to refuse
    // it. Both mattered: attention contracts over 1500 encoder positions,
    // which is even and not a multiple of 8, and over the 1, 2, 3, ... tokens
    // emitted so far, half of which are odd.
    for k in [1usize, 2, 3, 6, 7, 30, 1500] {
        for m in [4usize, 100] {
            let n = 10;
            let (a, w) = (seq(m * k, 1), seq(n * k, 2));
            let want = reference_gemm(&a, m, k, &w, n);
            let got = g
                .download(
                    &g.gemm(
                        &g.upload(&a).expect("upload"),
                        &g.upload(&w).expect("upload"),
                        None,
                        m,
                        k,
                        n,
                    )
                    .expect("gemm"),
                )
                .expect("download");
            assert_close_gemm(&format!("gemm k={k} m={m}"), &want, &got);
        }
    }
}

#[test]
fn a_batched_product_is_the_same_as_running_each_one_alone() {
    let Some(g) = gpu() else { return };
    // Attention is the only caller, and it batches over heads with three
    // different strides. A batch that reads the same operand every time would
    // pass a shape check and produce twenty copies of head zero.
    let (batch, m, k, n) = (5usize, 40usize, 32usize, 24usize);
    let a = seq(batch * m * k, 3);
    let w = seq(batch * n * k, 4);
    let got = g
        .download(
            &g.gemm_batched(
                Operand::F32(&g.upload(&a).expect("upload")),
                Operand::F32(&g.upload(&w).expect("upload")),
                None,
                Batch {
                    count: batch,
                    a: m * k,
                    w: n * k,
                    out: m * n,
                    w_row: 0,
                },
                m,
                k,
                n,
            )
            .expect("gemm_batched"),
        )
        .expect("download");
    for b in 0..batch {
        let want = reference_gemm(&a[b * m * k..], m, k, &w[b * n * k..], n);
        assert_close_gemm(&format!("batch {b}"), &want, &got[b * m * n..][..m * n]);
    }
}

#[test]
fn every_output_element_is_written_exactly_once() {
    let Some(g) = gpu() else { return };
    // A tiling bug that writes an element twice is invisible when the second
    // write has the same value. Ones against ones makes every output equal the
    // contraction length, so a double-add reads 2k and a skipped tile reads 0 -
    // both obvious, where a random-input test would show neither.
    let (m, k, n) = (100, 64, 70);
    let a = vec![1.0f32; m * k];
    let w = vec![1.0f32; n * k];
    let got = g
        .download(
            &g.gemm(
                &g.upload(&a).expect("upload"),
                &g.upload(&w).expect("upload"),
                None,
                m,
                k,
                n,
            )
            .expect("gemm"),
        )
        .expect("download");

    assert_eq!(got.len(), m * n);
    for (i, v) in got.iter().enumerate() {
        assert!(
            (v - k as f32).abs() < 1e-3,
            "output[{i}] is {v}, not {k}: the tiling covers it {} times",
            v / k as f32,
        );
    }
}

#[test]
fn f16_operands_cost_six_parts_in_a_hundred_thousand_of_full_scale() {
    let Some(g) = gpu() else { return };

    // The test above proves the kernel computes what it set out to compute.
    // This one measures what that choice costs against exact f32, because a
    // caller deciding between `gemm` and `linear` needs the number rather than
    // a promise - and because a regression that silently widened it would
    // otherwise pass.
    //
    // The error is proportional to the magnitude of the *terms*, not of the
    // result: each product carries about 2^-11 of relative error before any
    // summation, and a k-term sum accumulates them as a random walk. So a
    // heavily cancelled output - a small number that is the difference of large
    // ones - has a large relative error and a perfectly ordinary absolute one.
    // Gating on relative error alone fails there, and it failed there first.
    let (m, k, n) = (256, 1280, 256);
    let a = seq(m * k, 21);
    let w = seq(n * k, 22);

    let exact = xabe_dsp::linear(&a, m, k, &w, None, n);
    let got = g
        .download(
            &g.gemm(
                &g.upload(&a).expect("upload"),
                &g.upload(&w).expect("upload"),
                None,
                m,
                k,
                n,
            )
            .expect("gemm"),
        )
        .expect("download");

    let scale = exact.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let worst = exact
        .iter()
        .zip(&got)
        .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));

    // Measured: worst absolute 4.1e-1 against a peak output of 6.4e3, so
    // 6.5e-5 of full scale on a k=1280 contraction. The gate is 2e-3, about
    // thirty times the measurement - loose enough not to flap on a different
    // card, tight enough that losing another factor of ten would fail.
    let relative_to_scale = worst / scale;
    eprintln!(
        "f16 operands cost {worst:.3e} absolute, {relative_to_scale:.3e} of a \
         full scale of {scale:.3e}, at k={k}",
    );
    assert!(
        relative_to_scale < 2e-3,
        "f16 operand rounding costs {relative_to_scale:.3e} of full scale",
    );
}

// ------------------------------------------------------------------ whisper
//
// These four have no `xabe-dsp` twin, and deliberately not: three are pure
// index permutations and the fourth is a comparison, so their "reference" is
// the index formula itself. Written out beside the assertion it can be read
// against the kernel; exported as a library function nothing else calls it
// would only be the same formula, further away.

#[test]
fn im2col_then_gemm_is_a_convolution() {
    let Some(g) = gpu() else { return };
    // Whisper's stem, at both its strides. The reference is the scalar
    // convolution the rest of the workspace uses, which takes its input
    // channel-major - so the test transposes, and a layout mistake in either
    // direction shows up as a diff rather than as a plausible spectrogram.
    // The last three are WaveGlow's coupling network, whose dilation doubles
    // per layer: padding tracks it so the length is unchanged, and a dilation
    // read as a stride would still produce a tensor of the right shape.
    for &(in_ch, out_ch, t, k, stride, pad, dil) in &[
        (8usize, 12usize, 20usize, 3usize, 1usize, 1usize, 1usize),
        (12, 12, 20, 3, 2, 1, 1),
        (16, 32, 40, 3, 1, 1, 1),
        (16, 32, 40, 3, 1, 2, 2),
        (16, 32, 40, 3, 1, 4, 4),
    ] {
        let x_tc = seq(t * in_ch, 21); // [t, in_ch], the transformer layout
        let w = seq(out_ch * in_ch * k, 22); // [out_ch, in_ch, k]

        // `im2col` only gathers, so rounding the input before the convolution
        // is the same as rounding the gathered matrix - which is what the
        // tiled matmul will do to it. Below GEMV_MAX_M rows the path is exact
        // instead, and the reference has to follow; see `reference_gemm`.
        let out_t = (t + 2 * pad - (dil * (k - 1) + 1)) / stride + 1;
        let (x_ref, w_ref) = if out_t <= xabe_cuda::GEMV_MAX_M {
            (x_tc.clone(), w.clone())
        } else {
            (through_f16(&x_tc), through_f16(&w))
        };
        let x_ct = xabe_dsp::transpose(&x_ref, t, in_ch); // [in_ch, t]
        let want_ct = xabe_dsp::conv1d_strided(
            &x_ct, in_ch, t, &w_ref, None, out_ch, k, stride, pad, pad, dil,
        );
        assert_eq!(want_ct.len(), out_ch * out_t, "out_t");
        let want = xabe_dsp::transpose(&want_ct, out_ch, out_t); // [out_t, out_ch]

        let (col, got_t) = g
            .im2col(
                &g.upload(&x_tc).expect("upload"),
                t,
                in_ch,
                k,
                stride,
                pad,
                dil,
            )
            .expect("im2col");
        assert_eq!(got_t, out_t, "out_t from the kernel");
        let got = g
            .download(
                &g.gemm(
                    &col,
                    &g.upload(&w).expect("upload"),
                    None,
                    out_t,
                    in_ch * k,
                    out_ch,
                )
                .expect("gemm"),
            )
            .expect("download");
        assert_close_gemm(&format!("conv stride {stride} dilation {dil}"), &want, &got);
    }
}

#[test]
fn the_head_permutations_are_each_other() {
    let Some(g) = gpu() else { return };
    let (t, heads, hd) = (7usize, 4usize, 6usize);
    // Below one 32x32 staging tile in both axes, so this pins the ragged
    // corner; `the_transpose_covers_every_tile_of_a_ragged_shape` pins the
    // case where the tiling actually tiles.
    let x = seq(t * heads * hd, 31);
    let dev = g.upload(&x).expect("upload");

    let split = g
        .download(&g.split_heads(&dev, t, heads, hd).expect("split"))
        .expect("download");
    for ti in 0..t {
        for h in 0..heads {
            for j in 0..hd {
                assert_eq!(
                    split[(h * t + ti) * hd + j],
                    x[ti * heads * hd + h * hd + j]
                );
            }
        }
    }

    let split_t = g
        .download(&g.split_heads_t(&dev, t, heads, hd).expect("split_t"))
        .expect("download");
    for ti in 0..t {
        for h in 0..heads {
            for j in 0..hd {
                assert_eq!(
                    split_t[(h * hd + j) * t + ti],
                    x[ti * heads * hd + h * hd + j]
                );
            }
        }
    }

    // And `merge_heads` undoes `split_heads`, which is the property the
    // forward pass actually relies on.
    let round = g
        .download(
            &g.merge_heads(&g.upload(&split).expect("upload"), t, heads, hd)
                .expect("merge"),
        )
        .expect("download");
    assert_eq!(round, x);
}

/// The transposing split over a shape that is several staging tiles wide and a
/// whole number of them in neither axis.
///
/// `split_heads_t` is a tiled transpose, and the failure it can have that the
/// tiny case above cannot see is a tile-indexing one: a shape inside a single
/// tile exercises the bounds checks but never a second tile, and a shape that
/// divides evenly never exercises the checks at all. 1500 by 1280 is the
/// encoder's own, and 1500 is 46 tiles and a remainder of 28.
#[test]
fn the_transpose_covers_every_tile_of_a_ragged_shape() {
    let Some(g) = gpu() else { return };
    let (t, heads, hd) = (1500usize, 20usize, 64usize);
    let d = heads * hd;
    let x = seq(t * d, 17);
    let dev = g.upload(&x).expect("upload");

    let got = g
        .download(&g.split_heads_t(&dev, t, heads, hd).expect("split_t"))
        .expect("download");
    assert_eq!(got.len(), t * d);
    for ti in (0..t).step_by(7) {
        for c in (0..d).step_by(3) {
            assert_eq!(got[c * t + ti], x[ti * d + c], "row {ti} column {c}");
        }
    }
    // Every element, not only the sampled ones: a tile the kernel skipped
    // entirely would leave allocator leftovers here, and the stride pattern
    // above could step right over it.
    let mut want = vec![0.0f32; t * d];
    for ti in 0..t {
        for c in 0..d {
            want[c * t + ti] = x[ti * d + c];
        }
    }
    assert!(got == want, "the transpose is not the transpose");
}

#[test]
fn the_causal_mask_hides_the_future_and_nothing_else() {
    let Some(g) = gpu() else { return };
    let (batch, tq, tk, offset) = (2usize, 3usize, 5usize, 2usize);
    let scores = vec![1.0f32; batch * tq * tk];
    let mut dev = g.upload(&scores).expect("upload");
    g.causal_mask(&mut dev, batch, tq, tk, offset)
        .expect("causal_mask");
    let got = g.download(&dev).expect("download");
    for b in 0..batch {
        for q in 0..tq {
            for j in 0..tk {
                let v = got[(b * tq + q) * tk + j];
                // With `offset` keys already cached, query `q` really sits at
                // position `q + offset` and may see every key up to it.
                if j > q + offset {
                    assert!(v.is_infinite() && v < 0.0, "b{b} q{q} j{j} = {v}");
                } else {
                    assert_eq!(v, 1.0, "b{b} q{q} j{j}");
                }
            }
        }
    }
}

#[test]
fn an_f16_operand_is_what_the_tiled_kernel_would_have_made_of_an_f32_one() {
    let Some(g) = gpu() else { return };
    // The claim this rests on: `gemm_pack` rounds an F32 operand to f16
    // round-to-nearest-even on the way into shared memory, and
    // `half::f16::from_f32` does the same rounding on the host, so the two
    // paths stage the same 32 bits. If that is true the results are
    // *bit-identical*, not merely close - which is a far stronger check than a
    // tolerance, and it is the reason storing a weight as f16 is a bandwidth
    // decision rather than a precision one.
    let (m, k, n) = (100usize, 64usize, 70usize);
    let a = seq(m * k, 41);
    let w = seq(n * k, 42);
    let bias = seq(n, 43);
    let (ad, wd) = (g.upload(&a).expect("upload"), g.upload(&w).expect("upload"));
    let bd = g.upload(&bias).expect("upload");
    let (ah, wh) = (
        g.upload_f16(&a).expect("upload"),
        g.upload_f16(&w).expect("upload"),
    );

    let f32_f32 = g
        .download(&g.gemm(&ad, &wd, Some(&bd), m, k, n).expect("gemm"))
        .expect("download");
    for (name, a_op, w_op) in [
        ("f32 x f16", Operand::F32(&ad), Operand::F16(&wh)),
        ("f16 x f32", Operand::F16(&ah), Operand::F32(&wd)),
        ("f16 x f16", Operand::F16(&ah), Operand::F16(&wh)),
    ] {
        let got = g
            .download(
                &g.gemm_batched(a_op, w_op, Some(&bd), Batch::single(m * n), m, k, n)
                    .expect("gemm_batched"),
            )
            .expect("download");
        assert_eq!(got, f32_f32, "{name} disagreed with the F32 staging");
    }

    // The scalar path is a different claim: it accumulates an F32 operand
    // exactly, so packing one there really does change the arithmetic. Close,
    // not identical, and worth saying out loud rather than discovering.
    let rows = xabe_cuda::GEMV_MAX_M;
    let a = seq(rows * k, 44);
    let (ad, ah) = (
        g.upload(&a).expect("upload"),
        g.upload_f16(&a).expect("upload"),
    );
    let exact = g
        .download(&g.gemm(&ad, &wd, None, rows, k, n).expect("gemm"))
        .expect("download");
    let packed = g
        .download(
            &g.gemm_batched(
                Operand::F16(&ah),
                Operand::F16(&wh),
                None,
                Batch::single(rows * n),
                rows,
                k,
                n,
            )
            .expect("gemm_batched"),
        )
        .expect("download");
    assert_ne!(
        exact, packed,
        "the scalar path is supposed to be exact in F32"
    );
    // How far apart, measured rather than asserted loosely: 3.5e-4 relative
    // over a 64-long contraction. That is the cost of rounding an operand the
    // scalar path would otherwise have accumulated exactly, and it is why
    // `Operand` is a type the caller chooses rather than something the upload
    // decides. `GEMM_RTOL` is not the right bound here - it is calibrated for
    // two f16-staged paths agreeing with each other.
    let worst = exact
        .iter()
        .zip(&packed)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "gemv packed: {worst:e} relative");
}

#[test]
fn packing_on_the_device_and_on_the_host_agree() {
    let Some(g) = gpu() else { return };
    // `to_f16` uses `cvt.rn.f16.f32`; `upload_f16` uses `half::f16::from_f32`.
    // Both are round-to-nearest-even, and a tensor rounded on either side has
    // to be the same bits or the test above proves nothing.
    let x = seq(1000, 51);
    let host = g.upload_f16(&x).expect("upload");
    let device = g
        .to_f16(&g.upload(&x).expect("upload"), x.len())
        .expect("to_f16");
    let read = |s: &xabe_cuda::CudaSlice<u16>| -> Vec<u16> { g.download_u16(s).expect("download") };
    assert_eq!(read(&host), read(&device));
}

#[test]
fn an_odd_contraction_with_a_packed_operand_is_refused() {
    let Some(g) = gpu() else { return };
    // The F32 path takes any length. This one cannot: two halves to a word
    // puts the boundary inside one. Every contraction in a transformer is
    // even, so this is a check rather than a limitation - but a silent wrong
    // answer here would look like a model that had been trained badly.
    let (m, k, n) = (4usize, 7usize, 8usize);
    let e = g
        .gemm_batched(
            Operand::F16(&g.upload_f16(&seq(m * k, 61)).expect("upload")),
            Operand::F32(&g.upload(&seq(n * k, 62)).expect("upload")),
            None,
            Batch::single(m * n),
            m,
            k,
            n,
        )
        .unwrap_err();
    assert!(e.to_string().contains("odd"), "{e}");
}

// --------------------------------------------------------------------- llama

#[test]
fn rms_norm_matches() {
    let Some(g) = gpu() else { return };
    // Awkward widths on purpose: 5120 is the model's, and the others are
    // shapes the block reduction has to predicate rather than fill.
    for &(rows, dim) in &[(4usize, 5120usize), (1, 128), (7, 33), (3, 1)] {
        let x = seq(rows * dim, 71);
        let w = seq(dim, 72);
        let want = xabe_dsp::rms_norm(&x, rows, dim, &w, 1e-5);
        let got = g
            .download(
                &g.rms_norm(
                    &g.upload(&x).expect("upload"),
                    rows,
                    dim,
                    &g.upload(&w).expect("upload"),
                    1e-5,
                )
                .expect("rms_norm"),
            )
            .expect("download");
        // The reduction orders differ - the CPU twin sums left to right and
        // the kernel is a tree - so this is a tolerance rather than equality,
        // and it is the reduction that sets it.
        assert_close_to(&format!("rms_norm {rows}x{dim}"), &want, &got, 1e-5);
    }
}

#[test]
fn silu_mul_matches() {
    let Some(g) = gpu() else { return };
    let n = 1000;
    let (a, b) = (seq(n, 73), seq(n, 74));
    let mut want = a.clone();
    xabe_dsp::silu_mul(&mut want, &b);

    let mut dev = g.upload(&a).expect("upload");
    g.silu_mul(&mut dev, &g.upload(&b).expect("upload"), n)
        .expect("silu_mul");
    // `__expf` is the fast intrinsic, so this is two implementations of the
    // same function rather than one checked against itself.
    assert_close_to(
        "silu_mul",
        &want,
        &g.download(&dev).expect("download"),
        1e-5,
    );
}

#[test]
fn rope_matches_at_every_offset() {
    let Some(g) = gpu() else { return };
    // The offset is the part that is a no-op at position zero and wrong
    // everywhere else, so it is the part worth sweeping.
    for &first in &[0usize, 1, 17, 4095] {
        let (t, heads, hd) = (5usize, 3usize, 128usize);
        let x = seq(t * heads * hd, 75);
        let mut want = x.clone();
        xabe_dsp::rope(&mut want, t, heads, hd, 10_000.0, first);

        let mut dev = g.upload(&x).expect("upload");
        g.rope(&mut dev, 0, t, heads, hd, 10_000.0, first)
            .expect("rope");
        // The tolerance has to grow with the position, and the reason is not
        // sloppiness on either side. Both compute `angle = pos * inv_freq` in
        // f32, as the reference does; the two `powf` implementations differ in
        // the last bit of `inv_freq`, which is a relative 6e-8, and the angle
        // multiplies that by `pos`. At 4095 radians a 1-ulp frequency
        // disagreement *is* 2.4e-4 of phase, and no amount of care in either
        // implementation removes it.
        //
        // The same arithmetic says something about the model: RoPE in f32 is
        // intrinsically imprecise at long positions, in 🤗 as much as here.
        // Translation prompts are twenty tokens, so it never arrives - but it
        // would, at four thousand.
        assert_close_to(
            &format!("rope first={first}"),
            &want,
            &g.download(&dev).expect("download"),
            1e-5 + first as f32 * f32::EPSILON,
        );
    }
}

#[test]
fn scaled_rope_matches_the_twin_with_llama3_factors() {
    let Some(g) = gpu() else { return };
    // Breeze2's real shape of divisor: ones for the low pairs, a short ramp,
    // then a flat 8.0. A uniform factor would not catch a kernel that indexed
    // the divisor by the wrong axis, because every entry would be the same.
    let hd = 128usize;
    let half = hd / 2;
    let div: Vec<f32> = (0..half)
        .map(|i| match i {
            0..=28 => 1.0,
            29 => 1.207_484,
            30 => 1.553_415,
            31 => 2.026_313,
            32 => 2.694_53,
            33 => 3.684_253,
            34 => 5.257_327,
            _ => 8.0,
        })
        .collect();

    for &first in &[0usize, 1, 4095] {
        let (t, heads) = (4usize, 3usize);
        let x = seq(t * heads * hd, 91);
        let mut want = x.clone();
        xabe_dsp::rope_scaled(&mut want, t, heads, hd, 500_000.0, first, Some(&div));

        let mut dev = g.upload(&x).expect("upload");
        let d = g.upload(&div).expect("upload div");
        g.rope_scaled(&mut dev, 0, Some(&d), t, heads, hd, 500_000.0, first)
            .expect("rope_scaled");
        let got = g.download(&dev).expect("download");

        // The same tolerance the unscaled test uses and for the same reason:
        // both sides compute the angle in f32, and a 1-ulp disagreement in
        // `inv_freq` becomes 2.4e-4 of phase at position 4095.
        let tol = 1e-5 + first as f32 * f32::EPSILON;
        assert_close_to(&format!("rope_scaled first={first}"), &want, &got, tol);
    }
}

#[test]
fn scaled_rope_with_no_divisor_is_the_unscaled_one() {
    // The flag path, which is what every Llama-2 call takes. If `None` ever
    // started reading the dummy buffer this would drift immediately.
    let Some(g) = gpu() else { return };
    let (t, heads, hd) = (3usize, 2usize, 64usize);
    let x = seq(t * heads * hd, 12);

    let mut a = g.upload(&x).expect("upload");
    g.rope(&mut a, 0, t, heads, hd, 10_000.0, 7).expect("rope");
    let mut b = g.upload(&x).expect("upload");
    g.rope_scaled(&mut b, 0, None, t, heads, hd, 10_000.0, 7)
        .expect("rope_scaled");

    assert_eq!(
        g.download(&a).expect("a"),
        g.download(&b).expect("b"),
        "the unscaled path must be bit-identical, not merely close"
    );
}

#[test]
fn repeat_kv_widens_a_grouped_cache() {
    let Some(g) = gpu() else { return };
    // Breeze2's ratio: 32 query heads over 8 key-value heads.
    let (heads, kv_heads, t, hd) = (8usize, 2usize, 3usize, 4usize);
    let group = heads / kv_heads;
    let src = seq(kv_heads * t * hd, 33);

    let dev = g.upload(&src).expect("upload");
    let out = g.repeat_kv(&dev, heads, kv_heads, t, hd).expect("repeat");
    let got = g.download(&out).expect("download");

    assert_eq!(got.len(), heads * t * hd);
    // Head h must be a byte-for-byte copy of kv head h / group. Anything else
    // is a model whose queries attend to the wrong keys - which produces
    // fluent output, so it has to be checked exactly rather than by norm.
    for h in 0..heads {
        let per = t * hd;
        let want = &src[(h / group) * per..(h / group + 1) * per];
        let have = &got[h * per..(h + 1) * per];
        assert_eq!(have, want, "head {h} should copy kv head {}", h / group);
    }
}

#[test]
fn snake_matches_its_scalar_twin() {
    let Some(g) = gpu() else { return };
    // Shapes chosen so the per-channel alpha lookup is exercised: a channel
    // count that does not divide the block size, and a `t` that does not
    // either, so `i / t` has to be right rather than incidentally right.
    for (ch, t) in [(1, 1), (3, 7), (256, 129), (64, 1024), (17, 5)] {
        let x = seq(ch * t, 11 + ch as u64);
        // Alphas around one, never zero: the guard is tested separately.
        let alpha: Vec<f32> = seq(ch, 97).iter().map(|v| v.abs() + 0.25).collect();

        let mut want = x.clone();
        xabe_dsp::snake(&mut want, &alpha, ch, t);

        let mut d = g.upload(&x).expect("upload");
        g.snake(&mut d, &g.upload(&alpha).expect("alpha"), ch, t)
            .expect("snake");
        assert_close_to(
            &format!("snake {ch}x{t}"),
            &want,
            &g.download(&d).expect("download"),
            2e-5,
        );
    }
}

#[test]
fn snake_with_a_vanishing_alpha_does_what_the_reference_does() {
    // Upstream guards the *divisor*, so an alpha at zero gives `x + 1e9*sin^2`
    // rather than a NaN or a passthrough. That is a strange thing to want and
    // it is what the reference does; the point of pinning it is that a
    // "sensible" guard on the result would diverge only on a checkpoint whose
    // alpha had trained to zero, which is not a case anyone would think to
    // test later.
    let Some(g) = gpu() else { return };
    let (ch, t) = (2, 8);
    let x = seq(ch * t, 3);
    let alpha = vec![0.0, 1.0];

    let mut want = x.clone();
    xabe_dsp::snake(&mut want, &alpha, ch, t);
    let mut d = g.upload(&x).expect("upload");
    g.snake(&mut d, &g.upload(&alpha).expect("alpha"), ch, t)
        .expect("snake");
    let got = g.download(&d).expect("download");

    assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    // Relative, because the values are around 1e9 and an absolute tolerance
    // would be meaningless there.
    for (a, b) in want.iter().zip(&got) {
        let scale = a.abs().max(1.0);
        assert!(
            (a - b).abs() / scale < 1e-4,
            "want {a}, got {b} (relative {})",
            (a - b).abs() / scale
        );
    }
}

#[test]
fn the_stft_pair_matches_its_scalar_twin() {
    let Some(g) = gpu() else { return };
    let (n_fft, hop) = (16, 4);
    let window = xabe_dsp::hann_periodic(n_fft);
    let gw = g.upload(&window).expect("window");

    for n in [64usize, 128, 1024, 4096] {
        let x = seq(n, 5 + n as u64);
        let (wr, wi, frames) = xabe_dsp::stft(&x, &window, n_fft, hop);

        let (gr, gi, gframes) = g
            .stft(&g.upload(&x).expect("x"), &gw, n, n_fft, hop)
            .expect("stft");
        assert_eq!(gframes, frames, "frame count at n={n}");
        assert_close_to(
            &format!("stft re n={n}"),
            &wr,
            &g.download(&gr).expect("re"),
            3e-4,
        );
        assert_close_to(
            &format!("stft im n={n}"),
            &wi,
            &g.download(&gi).expect("im"),
            3e-4,
        );

        let want = xabe_dsp::istft(&wr, &wi, &window, frames, n_fft, hop);
        let got = g.istft(&gr, &gi, &gw, frames, n_fft, hop).expect("istft");
        assert_close_to(
            &format!("istft n={n}"),
            &want,
            &g.download(&got).expect("wave"),
            3e-4,
        );
    }
}

#[test]
fn the_stft_round_trip_reconstructs_the_signal_it_was_given() {
    // The property that says the pair is a *transform* and not just two
    // kernels that agree with two other kernels. A Hann window at hop
    // `n_fft / 4` satisfies the constant-overlap-add condition, so a forward
    // transform followed by an inverse returns the input - away from the
    // edges, where the envelope is genuinely smaller and the reference does
    // not reconstruct either.
    let Some(g) = gpu() else { return };
    let (n_fft, hop) = (16, 4);
    let window = xabe_dsp::hann_periodic(n_fft);
    let gw = g.upload(&window).expect("window");

    let n = 2048;
    let x: Vec<f32> = (0..n)
        .map(|i| (0.017 * i as f32).sin() * 0.6 + (0.31 * i as f32).sin() * 0.2)
        .collect();

    let (re, im, frames) = g
        .stft(&g.upload(&x).expect("x"), &gw, n, n_fft, hop)
        .expect("stft");
    let back = g
        .download(&g.istft(&re, &im, &gw, frames, n_fft, hop).expect("istft"))
        .expect("download");

    assert_eq!(back.len(), n, "a round trip should return its own length");
    let edge = n_fft;
    let worst = x[edge..n - edge]
        .iter()
        .zip(&back[edge..n - edge])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "round trip differs by {worst}");
}

#[test]
fn gated_activation_rows_is_the_transpose_of_gated_activation() {
    let Some(g) = gpu() else { return };
    let (ch, t) = (24usize, 37usize);
    // Compared against the twin rather than a fresh reference, because that is
    // the actual requirement: the two must be the same function seen through a
    // transpose, and a sign or an index error in either shows up as a diff.
    let x_ct = seq(2 * ch * t, 76);
    let want_ct = xabe_dsp::gated_activation(&x_ct, ch, t);

    let x_tc = xabe_dsp::transpose(&x_ct, 2 * ch, t); // [t, 2 * ch]
    let got_tc = g
        .download(
            &g.gated_activation_rows(&g.upload(&x_tc).unwrap(), ch, t)
                .unwrap(),
        )
        .unwrap();
    let got_ct = xabe_dsp::transpose(&got_tc, t, ch);
    assert_close("gated_activation_rows", &want_ct, &got_ct);
}

#[test]
fn lstm_gates_matches() {
    let Some(g) = gpu() else { return };
    let hidden = 137usize;
    let (gi, gh) = (seq(4 * hidden, 71), seq(4 * hidden, 72));
    // A non-zero starting cell, because a zero one hides the forget gate
    // entirely - the one term that only shows up when there is state to keep.
    let c0 = seq(hidden, 73);

    let (mut c, mut h) = (c0.clone(), vec![0.0f32; hidden]);
    xabe_dsp::lstm_gates(&gi, &gh, &mut c, &mut h, hidden);

    let (dgi, dgh) = (g.upload(&gi).unwrap(), g.upload(&gh).unwrap());
    let (mut dc, mut dh) = (g.upload(&c0).unwrap(), g.zeros(hidden).unwrap());
    g.lstm_gates(&dgi, &dgh, &mut dc, &mut dh, hidden).unwrap();

    // `__expf` in the sigmoid, so the same loosened tolerance the llama
    // kernels use rather than the shared one.
    assert_close_to("lstm_gates cell", &c, &g.download(&dc).unwrap(), 1e-4);
    assert_close_to("lstm_gates hidden", &h, &g.download(&dh).unwrap(), 1e-4);
}

#[test]
fn coupling_inverse_matches() {
    let Some(g) = gpu() else { return };
    let (half, t) = (4usize, 61usize);
    let x = seq(half * t, 74);
    // Scaled down: `st` holds a *log* scale, and seq spans [-2, 2], so an
    // unscaled second half would divide by e^2 and swamp the shift.
    let st: Vec<f32> = seq(2 * half * t, 75).iter().map(|v| v * 0.5).collect();

    let mut want = vec![0.0f32; half * t];
    xabe_dsp::coupling_inverse(&x, &st, &mut want, half, t);

    let (dx, dst) = (g.upload(&x).unwrap(), g.upload(&st).unwrap());
    let out = g.coupling_inverse(&dx, &dst, half, t).unwrap();
    assert_close("coupling_inverse", &want, &g.download(&out).unwrap());
}

#[test]
fn the_split_contraction_agrees_with_the_whole_one() {
    let Some(g) = gpu() else { return };

    // Shapes chosen so `ksplit_for` returns more than one: a narrow output on a
    // long contraction is exactly the prefill projection that left the card
    // idle, and it is the only shape the split path ever runs on. A shape it
    // declines is included so a heuristic that stopped splitting anything would
    // not pass this test by accident.
    let shapes: &[(usize, usize, usize)] = &[
        (128, 4096, 1024),  // the grouped-attention k/v projection: 8 blocks
        (128, 4096, 4096),  // the query and output projections
        (128, 14336, 4096), // the mlp down projection, the longest contraction
        (256, 2048, 128),   // two m tiles, one n tile
    ];
    let mut split_seen = 0;

    for &(m, k, n) in shapes {
        let ks = xabe_cuda::ksplit_for(m, k, n, 1);
        if ks > 1 {
            split_seen += 1;
        }
        let (a, w) = (seq(m * k, 5), seq(n * k, 6));
        let want = reference_gemm(&a, m, k, &w, n);
        let got = g
            .download(
                &g.gemm(
                    &g.upload(&a).expect("upload"),
                    &g.upload(&w).expect("upload"),
                    None,
                    m,
                    k,
                    n,
                )
                .expect("gemm"),
            )
            .expect("download");
        assert_close_gemm(&format!("split {m}x{k}x{n} ksplit={ks}"), &want, &got);
    }

    // The point of the test is the split path, so a run where nothing split is
    // a pass that proved nothing.
    assert!(
        split_seen >= 3,
        "no shape here reached the split path: ksplit_for never returned > 1"
    );
}

#[test]
fn a_split_contraction_is_bit_identical_from_run_to_run() {
    let Some(g) = gpu() else { return };

    // Summing the slices in index order rather than by `atomicAdd` is what
    // makes this hold. It is asserted rather than assumed because an atomic
    // reduction is the obvious way to write `gemm_reduce`, it is faster, and
    // it would pass every tolerance-based test in this file while making the
    // differential thresholds elsewhere depend on block scheduling.
    let (m, k, n) = (128usize, 4096usize, 1024usize);
    assert!(
        xabe_cuda::ksplit_for(m, k, n, 1) > 1,
        "shape does not split"
    );
    let (a, w) = (seq(m * k, 7), seq(n * k, 8));
    let (da, dw) = (g.upload(&a).expect("upload"), g.upload(&w).expect("upload"));

    let first = g
        .download(&g.gemm(&da, &dw, None, m, k, n).expect("gemm"))
        .expect("download");
    for round in 1..4 {
        let again = g
            .download(&g.gemm(&da, &dw, None, m, k, n).expect("gemm"))
            .expect("download");
        assert_eq!(first, again, "round {round} differs from the first");
    }
}

/// One block of a batched projection, rotated where it sits.
///
/// The attention projections are issued as one product, so `q` and `k` are two
/// contiguous blocks of one allocation and `rope` is handed the whole thing
/// plus an offset. What this guards is that the offset moves the *read* and
/// nothing else: the rotation of a block at offset `o` must be bit-identical to
/// the rotation of the same numbers at offset zero, and the bytes on either
/// side of it must not move at all.
#[test]
fn rope_at_an_offset_rotates_that_block_and_only_that_block() {
    let Some(g) = gpu() else { return };
    let (t, heads, hd) = (5usize, 3usize, 64usize);
    let span = t * heads * hd;
    // A whole buffer of three blocks, rotating the middle one.
    let all = seq(3 * span, 77);
    let off = span;

    let mut dev = g.upload(&all).expect("upload");
    g.rope(&mut dev, off, t, heads, hd, 10_000.0, 11)
        .expect("rope at an offset");
    let got = g.download(&dev).expect("download");

    // The block itself against the scalar twin, at the tolerance the other
    // rope tests use - both sides compute the angle in f32 and `sincosf`
    // disagrees in the last bits.
    let mut want = all[off..off + span].to_vec();
    xabe_dsp::rope(&mut want, t, heads, hd, 10_000.0, 11);
    assert_close_to("rope at an offset", &want, &got[off..off + span], 1e-5);

    // Everything outside the block, exactly: an offset that moved the write
    // would show here and nowhere else.
    assert_eq!(&got[..off], &all[..off], "wrote before the block");
    assert_eq!(
        &got[off + span..],
        &all[off + span..],
        "wrote after the block",
    );

    // And the same numbers alone, to show the offset is not doing arithmetic.
    let mut alone = g.upload(&all[off..off + span]).expect("upload");
    g.rope(&mut alone, 0, t, heads, hd, 10_000.0, 11)
        .expect("rope");
    assert_eq!(
        g.download(&alone).expect("download"),
        got[off..off + span],
        "a block rotated in place differs from the same block rotated alone",
    );
}

/// The same for the cache, which reads a block rather than writing one.
#[test]
fn cache_append_reads_its_own_block_of_a_batched_projection() {
    let Some(g) = gpu() else { return };
    let (n, kv_heads, hd, cap, past) = (4usize, 2usize, 8usize, 16usize, 3usize);
    let span = n * kv_heads * hd;
    let all = seq(3 * span, 41);
    let off = 2 * span;

    for transposed in [false, true] {
        let src = g.upload(&all).expect("upload");
        let mut fused = g.zeros(cap * kv_heads * hd).expect("cache");
        g.cache_append(
            &src, off, &mut fused, n, kv_heads, hd, cap, past, transposed,
        )
        .expect("append at an offset");

        let lone = g.upload(&all[off..off + span]).expect("upload");
        let mut apart = g.zeros(cap * kv_heads * hd).expect("cache");
        g.cache_append(&lone, 0, &mut apart, n, kv_heads, hd, cap, past, transposed)
            .expect("append");

        assert_eq!(
            g.download(&fused).expect("fused"),
            g.download(&apart).expect("apart"),
            "transposed={transposed}: the offset changed more than the source",
        );
    }
}

/// Growing a cache leaves every head reading what it read before.
///
/// The bug this is here for: `cap` is the stride between heads in both cache
/// layouts, so a growth that copies the live prefix flat lands head 0 correctly
/// and buries the rest inside their own earlier positions. Nothing downstream
/// notices - the buffer is the right length and every read is in bounds - so
/// the check has to be here, and it has to be per head rather than on the
/// buffer as a whole.
///
/// Expressed as an invariant rather than against a golden: appending `live`
/// tokens at the small capacity and then growing must equal appending the same
/// tokens at the large capacity to begin with.
#[test]
fn growing_a_cache_puts_every_head_where_the_larger_capacity_wants_it() {
    let Some(g) = gpu() else { return };
    // `kv_heads > 1` is the whole point: head 0 is correct either way.
    let (live, kv_heads, hd, small, large) = (5usize, 3usize, 4usize, 8usize, 16usize);
    let src = seq(live * kv_heads * hd, 7);

    for transposed in [false, true] {
        // Appended at the small capacity, then grown.
        let up = g.upload(&src).expect("upload");
        let mut grew = g.zeros(small * kv_heads * hd).expect("cache");
        g.cache_append(&up, 0, &mut grew, live, kv_heads, hd, small, 0, transposed)
            .expect("append at the small capacity");
        let mut moved = g.zeros(large * kv_heads * hd).expect("cache");
        g.cache_grow(
            &grew, &mut moved, kv_heads, hd, small, large, live, transposed,
        )
        .expect("grow");

        // Appended at the large capacity from the start.
        let mut direct = g.zeros(large * kv_heads * hd).expect("cache");
        g.cache_append(
            &up,
            0,
            &mut direct,
            live,
            kv_heads,
            hd,
            large,
            0,
            transposed,
        )
        .expect("append at the large capacity");

        assert_eq!(
            g.download(&moved).expect("moved"),
            g.download(&direct).expect("direct"),
            "transposed={transposed}: growth did not re-stride the heads",
        );
    }
}

/// Growth refuses the two ways it can be asked for the impossible.
#[test]
fn a_growth_that_shrinks_or_overruns_is_refused() {
    let Some(g) = gpu() else { return };
    let (kv_heads, hd, small, large) = (2usize, 4usize, 8usize, 16usize);
    let src = g.zeros(small * kv_heads * hd).expect("cache");
    let mut dst = g.zeros(large * kv_heads * hd).expect("cache");

    assert!(
        g.cache_grow(&src, &mut dst, kv_heads, hd, small, large, small + 1, false)
            .is_err(),
        "more live tokens than the source held, and it did not say so",
    );
    let mut smaller = g.zeros(small * kv_heads * hd).expect("cache");
    assert!(
        g.cache_grow(&src, &mut smaller, kv_heads, hd, large, small, 4, false)
            .is_err(),
        "a growth into a smaller capacity, and it did not say so",
    );
}

/// A read that starts inside the buffer and ends past it is refused.
#[test]
fn an_offset_past_the_end_of_the_source_is_refused() {
    let Some(g) = gpu() else { return };
    let (t, heads, hd) = (2usize, 2usize, 8usize);
    let span = t * heads * hd;
    let mut x = g.zeros(span).expect("alloc");
    assert!(
        g.rope(&mut x, 1, t, heads, hd, 10_000.0, 0).is_err(),
        "rope read one element past the end and did not say so",
    );
    let src = g.zeros(span).expect("alloc");
    let mut dst = g.zeros(8 * heads * hd).expect("alloc");
    assert!(
        g.cache_append(&src, 1, &mut dst, t, heads, hd, 8, 0, false)
            .is_err(),
        "cache_append read past the end and did not say so",
    );
}

/// The fused attention against a scalar reference of the same arithmetic.
///
/// The reference applies the roundings where the kernel does: operands to f16
/// going into both products, scores accumulated in f32, probabilities rounded
/// to f16 on their way into the value product, the normaliser summed at f32.
/// The kernel's running-maximum rescaling is algebraically the same softmax,
/// so what remains inside the tolerance is rounding order, and what a wrong
/// index would produce - another head's or another position's context - is a
/// full-scale disagreement, not a rounding one.
///
/// Driven at both head widths the kernel is instantiated at and in both
/// masking modes by the two tests below. The width is what the fragment
/// layout depends on - a warp owns `hd / 32` output column fragments - so a
/// width that is only ever exercised through the model would have its
/// indexing checked by nothing.
fn fused_attention_case(heads: usize, kv_heads: usize, hd: usize, causal: bool) {
    let Some(g) = gpu() else { return };

    // Odd on purpose: a query count that is not a whole tile, and a cache with
    // more capacity than positions.
    let (tq, past, cap) = (70usize, 33usize, 160usize);
    let tk = past + tq;
    let scale = (hd as f32).powf(-0.5);

    // The queries are scaled up so the softmax is *peaked*: near-uniform
    // scores would let a permuted position hide inside the tolerance, and
    // permutations are exactly what this test exists to catch.
    let q: Vec<f32> = seq(tq * heads * hd, 71).iter().map(|v| v * 100.0).collect();
    let kvals = seq(kv_heads * tk * hd, 72);
    let vvals = seq(kv_heads * tk * hd, 73);
    // The caches' own layouts: K `[kv_head][pos][hd]`, V `[kv_head][hd][cap]`,
    // with the unused capacity zeroed as the real cache's is.
    let mut kc = vec![0.0f32; kv_heads * cap * hd];
    let mut vc = vec![0.0f32; kv_heads * hd * cap];
    for h in 0..kv_heads {
        for p in 0..tk {
            for d in 0..hd {
                kc[(h * cap + p) * hd + d] = kvals[(h * tk + p) * hd + d];
                vc[(h * hd + d) * cap + p] = vvals[(h * tk + p) * hd + d];
            }
        }
    }

    let dq = g.upload(&q).expect("upload q");
    let dk = g.upload(&kc).expect("upload k");
    let dv = g.upload(&vc).expect("upload v");
    let out = g
        .flash_attn(
            &dq, &dk, &dv, tq, past, heads, kv_heads, hd, cap, scale, causal,
        )
        .expect("flash_attn");
    let got = g.download(&out).expect("download");

    // Operands are rounded to f16 because the kernel rounds them; the
    // *accumulation* is f64 because the kernel's blocked order and a scalar
    // loop's sequential one are both approximations of it, and a reference
    // has no business being the less accurate of the two. This is not
    // pedantry: on this data some softmax rows are degenerate - one position
    // takes essentially all the mass - which makes the value product a
    // difference of terms up to 45x its own result. At that conditioning a
    // sequential f32 reference disagrees with the kernel by 2%, and every
    // digit of it is the reference's summation order rather than the
    // kernel's indexing.
    let h16 = |x: f32| f64::from(f32::from(half::f16::from_f32(x)));
    // The probabilities, which the reference now carries at f64, get the same
    // one f16 step the kernel gives them on their way into the value product.
    let h16p = |x: f64| f64::from(f32::from(half::f16::from_f32(x as f32)));
    let dq_row = heads * hd;
    let mut worst = 0.0f64;
    for h in 0..heads {
        let kh = h / (heads / kv_heads);
        for r in 0..tq {
            // Causal: this row sees up to its own position. Otherwise it
            // sees every key, which is the encoder's case. `past + tq == tk`
            // here, so a causal row's own position is always in range.
            let last = if causal { past + r } else { tk - 1 };
            let mut s = vec![f64::NEG_INFINITY; tk];
            for (p, sp) in s.iter_mut().enumerate().take(last + 1) {
                let mut acc = 0.0f64;
                for d in 0..hd {
                    acc += h16(q[r * dq_row + h * hd + d]) * h16(kvals[(kh * tk + p) * hd + d]);
                }
                *sp = acc * f64::from(scale);
            }
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s
                .iter()
                .map(|&x| if x.is_finite() { (x - m).exp() } else { 0.0 })
                .collect();
            let l: f64 = e.iter().sum();
            for d in 0..hd {
                let mut acc = 0.0f64;
                // The spread of the values this row averages. The output is a
                // convex combination of them, so a rounding-sized wobble in
                // the scores moves it by a fraction of this - never by a
                // fraction of the output's own magnitude, which is what a
                // purely relative tolerance would assume and which goes to
                // zero exactly where the averaging cancels.
                let mut spread = 0.0f64;
                for (p, &ep) in e.iter().enumerate() {
                    acc += h16p(ep) * h16(vvals[(kh * tk + p) * hd + d]);
                }
                let want = acc / l;
                for (p, &ep) in e.iter().enumerate() {
                    spread += (ep / l) * (h16(vvals[(kh * tk + p) * hd + d]) - want).abs();
                }
                let have = f64::from(got[r * dq_row + h * hd + d]);
                let tol = 2e-4 + 1e-2 * want.abs() + 1e-2 * spread;
                let err = (want - have).abs();
                worst = worst.max(err / tol);
                assert!(
                    err <= tol,
                    "head {h} row {r} dim {d}: {have} wanted {want} \
                     (err {err}, tol {tol}, spread {spread})",
                );
            }
        }
    }
    // A permuted position or head substitutes one of the averaged values for
    // another, which is a whole `spread` and so many multiples of the
    // tolerance. Reporting the worst as a fraction of its own tolerance is
    // what keeps that margin visible: if this ever creeps towards 1 the
    // kernel has drifted, whether or not any single point has crossed.
    assert!(
        worst < 0.5,
        "worst error is {worst} of its tolerance, not comfortably inside it",
    );
}

/// Both Llama stages' shape: 128-wide heads, grouped, causal.
#[test]
fn fused_attention_matches_the_unfused_arithmetic() {
    fused_attention_case(4, 2, 128, true);
}

/// The Whisper encoder's shape: 64-wide heads, ungrouped, and attending over
/// the whole window rather than a triangle.
///
/// Non-causal is the mode that can go wrong quietly. The causal loop bound
/// stops at the last tile a row can see, so a masking mistake there truncates
/// the row and shows up; with the mask open the kernel reads every tile
/// either way, and a wrong per-row limit would only shift *which* keys are
/// summed - still a full softmax, still plausible context.
#[test]
fn fused_attention_attends_the_whole_window_when_it_is_not_causal() {
    fused_attention_case(4, 4, 64, false);
}

/// A 64-wide head that is grouped, and a 128-wide one that is not.
///
/// The width and the grouping are independent, and the two tests above happen
/// to pair each width with one grouping. This crosses them so that neither
/// instantiation's `head / (heads / kv_heads)` is only ever exercised at a
/// ratio of one.
#[test]
fn fused_attention_crosses_head_width_with_grouping() {
    fused_attention_case(4, 2, 64, false);
    fused_attention_case(4, 4, 128, true);
}

/// The 64-wide instantiation masked, which production never asks it for.
///
/// Only the encoder takes this width and it attends over the whole window, so
/// nothing in the engine reaches the causal branch here. The branch exists
/// anyway - the width is a template argument and the flag is a runtime one -
/// and the masking is per lane against its own `mma` accumulator columns, so
/// this width and that flag exercise a different index than the 128-wide
/// causal case does. An unreachable path that is wrong is a trap for whoever
/// makes it reachable.
#[test]
fn the_narrow_instantiation_masks_as_well_as_the_wide_one() {
    fused_attention_case(4, 4, 64, true);
    fused_attention_case(4, 2, 64, true);
}

/// A head width the kernel is not instantiated at is refused, not indexed.
#[test]
fn an_uninstantiated_head_width_is_refused() {
    let Some(g) = gpu() else { return };
    assert!(
        !g.supports_flash(96, 4, 4),
        "96-wide heads are not instantiated and supports_flash said they were",
    );
    let q = g.zeros(4 * 4 * 96).expect("alloc");
    assert!(
        g.flash_attn(&q, &q, &q, 4, 0, 4, 4, 96, 4, 1.0, false)
            .is_err(),
        "a 96-wide head was attended rather than refused",
    );
    assert!(
        !g.supports_flash(128, 4, 3),
        "heads are not a multiple of kv_heads and supports_flash said they were",
    );
}
