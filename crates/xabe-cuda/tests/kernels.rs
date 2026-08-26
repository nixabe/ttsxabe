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

use xabe_cuda::Gpu;

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

/// Which device to use. GPU 2 on this host runs somebody else's job.
fn ordinal() -> usize {
    std::env::var("XABE_TTS_DEVICE")
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

    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let mut d = g.upload(&a).unwrap();
    g.sub_inplace(&mut d, &db, n).unwrap();
    assert_close("sub_inplace", &want, &g.download(&d).unwrap());

    let want: Vec<f32> = a.iter().map(|x| x / 3.0).collect();
    let mut d = g.upload(&a).unwrap();
    g.scale_inplace(&mut d, n, 1.0 / 3.0).unwrap();
    assert_close("scale_inplace", &want, &g.download(&d).unwrap());
}
