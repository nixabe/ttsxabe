//! The packed-weight matmul, against the container crate's own decoder.
//!
//! `xabe-gguf` decodes the ten block formats to f32 at load, and is checked
//! against `gguf-py` - the code that writes these files - at exact equality.
//! That decoder is the reference here, because `q_elem` in `kernels.rs` is a
//! *second* transcription of the same layouts and a second transcription is a
//! second chance to permute a block. A permuted block is the failure mode that
//! matters: it produces a plausible tensor rather than an error, and a model
//! built on one speaks fluent nonsense.
//!
//! Three properties, in increasing distance from the bytes:
//!
//! 1. the two size tables agree, which needs no device;
//! 2. every element unpacks to the same number on both sides, extracted one at
//!    a time through the exact f32 path so nothing is hidden by accumulation;
//! 3. the whole product agrees, on both kernels, which is what the model runs.
//!
//! Skips when there is no device, like every other GPU test here.

use xabe_cuda::{Batch, GEMV_MAX_M, Gpu, NormScratch, Operand, Quant};
use xabe_gguf::GgmlType;

/// Relative tolerance for the accumulated products.
///
/// The same rule as `tests/kernels.rs`: the GPU fuses multiply-add where the
/// scalar twin does not, so a dot product of a few hundred terms differs in
/// the last bits and an absolute tolerance is the wrong judge.
const RTOL: f32 = 1e-5;

/// Absolute floor, for values near zero.
const ATOL: f32 = 1e-5;

/// Every format, paired with its spelling in the container crate.
///
/// Both halves are listed rather than mapped, because the point of the pairing
/// is that nothing derives one from the other.
const FORMATS: &[(Quant, GgmlType)] = &[
    (Quant::Q4_0, GgmlType::Q4_0),
    (Quant::Q4_1, GgmlType::Q4_1),
    (Quant::Q5_0, GgmlType::Q5_0),
    (Quant::Q5_1, GgmlType::Q5_1),
    (Quant::Q8_0, GgmlType::Q8_0),
    (Quant::Q2K, GgmlType::Q2K),
    (Quant::Q3K, GgmlType::Q3K),
    (Quant::Q4K, GgmlType::Q4K),
    (Quant::Q5K, GgmlType::Q5K),
    (Quant::Q6K, GgmlType::Q6K),
];

/// Where each format keeps its f16 scales, as byte offsets into a block.
///
/// Random bytes are a legal block everywhere else - every quantized field is a
/// bit pattern with no invalid values - but a random f16 is an infinity or a
/// NaN about one time in 256, and one of those poisons a whole row. So the
/// scales are written rather than drawn, and everything else is noise.
fn scale_offsets(q: Quant) -> &'static [usize] {
    match q {
        Quant::Q4_0 | Quant::Q5_0 | Quant::Q8_0 => &[0],
        Quant::Q4_1 | Quant::Q5_1 | Quant::Q4K | Quant::Q5K => &[0, 2],
        Quant::Q2K => &[80, 82],
        Quant::Q3K => &[108],
        Quant::Q6K => &[208],
    }
}

/// `count` blocks of pseudo-random, well-formed data.
fn blocks(q: Quant, count: usize, salt: u64) -> Vec<u8> {
    let ts = q.type_size();
    let mut s = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let mut raw = vec![0u8; count * ts];
    for b in raw.iter_mut() {
        *b = (next() >> 33) as u8;
    }
    for i in 0..count {
        for &off in scale_offsets(q) {
            // Small and positive. Q6_K multiplies this by a signed byte and a
            // 6-bit quantum, so a large scale there would run the products up
            // into thousands and measure the tolerance rather than the layout.
            let d = 0.001 + ((next() >> 40) as f32 / 16_777_216.0) * 0.02;
            let bits = half::f16::from_f32(d).to_bits().to_le_bytes();
            raw[i * ts + off] = bits[0];
            raw[i * ts + off + 1] = bits[1];
        }
    }
    raw
}

/// A deterministic spread of values in roughly [-2, 2].
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

fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The two size tables, which are duplicated across a crate boundary.
///
/// `xabe-cuda` cannot depend on `xabe-gguf` - the crate map has it depending
/// on `xabe-dsp` alone - so it carries its own copy of the block and type
/// sizes. A copy that drifts would put every block boundary in the wrong place
/// and produce garbage rather than an error, so the copies are pinned here.
#[test]
fn quant_sizes_match_the_container_crate() {
    for &(q, g) in FORMATS {
        assert_eq!(q.id() as u32, g as u32, "{q:?}: ggml type id");
        assert_eq!(
            q.block_size() as u64,
            g.block_size(),
            "{q:?}: elements per block",
        );
        assert_eq!(
            q.type_size() as u64,
            g.type_size(),
            "{q:?}: bytes per block"
        );
        // The sizing helper the model loaders use to cut a tensor out of the
        // mapping, checked against the same arithmetic done the long way.
        assert_eq!(q.bytes(4096), 4096 / q.block_size() * q.type_size());
    }
}

/// Every element, one at a time, through the exact f32 path.
///
/// The activation is one-hot, so each output *is* one weight rather than a sum
/// containing it: `1.0 * w` is exact and the zeros added around it change
/// nothing, so this compares bit patterns and not tolerances. That is the test
/// that catches an ordering mistake, because a permutation inside a block is
/// invisible to any check on magnitudes.
///
/// `GEMV_MAX_M` rows at a time - taken from the constant rather than written
/// out, because the whole point of the constant being public is that a test
/// asserting the scalar path's exactness has to sit on the scalar side of it -
/// so it stays on the scalar
/// kernel. One row per launch would work and take sixteen times as long.
#[test]
fn every_block_format_unpacks_element_for_element() {
    let Some(g) = gpu() else { return };

    // 512 is a whole number of blocks in both families - sixteen 32-element
    // blocks or two 256-element superblocks - so one contraction covers every
    // format and every format crosses a block boundary.
    let (k, n) = (512usize, 6usize);

    for &(q, gt) in FORMATS {
        let raw = blocks(q, n * k / q.block_size(), 7);
        let want = xabe_gguf::dequantize_blocks(gt, &raw, n * k).expect("the CPU decoder");
        let dw = g.upload_quant(q, &raw).unwrap();

        for j0 in (0..k).step_by(16) {
            let rows = GEMV_MAX_M.min(k - j0);
            let mut a = vec![0.0f32; rows * k];
            for (r, chunk) in a.chunks_mut(k).enumerate() {
                chunk[j0 + r] = 1.0;
            }
            let da = g.upload(&a).unwrap();
            let out = g
                .gemm_batched(
                    Operand::F32(&da),
                    Operand::Q { data: &dw, ty: q },
                    None,
                    Batch::single(rows * n),
                    rows,
                    k,
                    n,
                )
                .expect("the quantized gemv");
            let got = g.download(&out).unwrap();

            for r in 0..rows {
                for col in 0..n {
                    let w = want[col * k + j0 + r];
                    let e = got[r * n + col];
                    // Equality and not `to_bits()`, for one reason found by
                    // this test: a negative scale times a zero quantum is
                    // `-0.0` on the CPU, and the warp reduction adds it to
                    // `0.0` and gets `+0.0`. The bit patterns differ, the
                    // numbers do not, and IEEE says they are equal.
                    //
                    // The two K-quants are the exception and it is one rounding
                    // wide. They take the packed mat-vec's int8 path, so the
                    // one-hot arrives as the code 127 with a scale of 1/127 and
                    // the product is `w * (127/127)` - exact in real arithmetic
                    // and one ulp off in binary. A permuted block moves a value
                    // by the size of the value, so this still separates the two
                    // things it exists to separate.
                    let ok = match q {
                        Quant::Q4K | Quant::Q6K => (w - e).abs() <= 1e-6 * w.abs(),
                        _ => w == e,
                    };
                    assert!(ok, "{q:?}: element [{col}][{}] is {e}, wanted {w}", j0 + r,);
                }
            }
        }
    }
}

/// The whole product on the scalar kernel, which is exact f32.
#[test]
fn quantized_gemv_matches_the_cpu_dequantizer() {
    let Some(g) = gpu() else { return };
    let (m, k, n) = (3usize, 512usize, 24usize);
    assert!(m <= xabe_cuda::GEMV_MAX_M, "this shape must take gemv");

    for &(q, gt) in FORMATS {
        let raw = blocks(q, n * k / q.block_size(), 11);
        let w = xabe_gguf::dequantize_blocks(gt, &raw, n * k).expect("the CPU decoder");
        let a = seq(m * k, 12);
        let bias = seq(n, 13);
        let want = xabe_dsp::linear(&a, m, k, &w, Some(&bias), n);

        let da = g.upload(&a).unwrap();
        let dw = g.upload_quant(q, &raw).unwrap();
        let db = g.upload(&bias).unwrap();
        let out = g
            .gemm_batched(
                Operand::F32(&da),
                Operand::Q { data: &dw, ty: q },
                Some(&db),
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .expect("the quantized gemv");
        // Judged against the size of the terms, for the reason the tiled test
        // below spells out - and this one found the same thing a second time.
        // The specialised K-quant path sums eight contiguous elements at a time
        // where the generic one strides by a warp, and on a Q6_K row whose 512
        // terms cancel from 2.9e4 down to 15.8 that reordering moves the last
        // four digits. Measured against an f64 sum of the same terms, it moves
        // them the *right* way: the kernel lands 2.2e-4 from exact where this
        // f32 reference lands 1.2e-3.
        //
        // Nothing is hidden by this. A permuted block changes a result by the
        // size of the terms, and `every_block_format_unpacks_element_for_element`
        // compares bit patterns through a one-hot activation, where no sum
        // happens at all.
        let got = g.download(&out).unwrap();
        let name = format!("gemv {q:?}");
        assert_eq!(want.len(), got.len(), "{name}: length");
        for row in 0..m {
            for col in 0..n {
                let i = row * n + col;
                let scale: f32 = (0..k)
                    .map(|j| (a[row * k + j] * w[col * k + j]).abs())
                    .sum();
                // The two K-quants quantize the activation to int8 on the way
                // in, which is a far larger and entirely deliberate error than
                // any reordering - see `Gpu::quantize_activation`. Their bound
                // is that approximation's size against the terms, and it is
                // still two orders of magnitude below what permuting a block
                // would move.
                let rtol = match q {
                    Quant::Q4K | Quant::Q6K => 1e-3,
                    _ => RTOL,
                };
                let tol = ATOL + rtol * scale;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "{name}: element {i} is {}, wanted {} (tolerance {tol})",
                    got[i],
                    want[i],
                );
            }
        }
    }
}

/// The activation as the int8 tiled kernel sees it: quantised in groups of
/// `GROUP` and read back.
///
/// This is `xabe_dsp::quantize_q8` applied a row at a time, which is what the
/// kernel does - `GEMM_I8_KC` is `GROUP`, so a trip is a group and no lane's
/// maximum crosses one. The rounding is the same to the bit; only the order of
/// the sum that follows differs.
fn as_int8(v: &[f32], k: usize) -> Vec<f32> {
    v.chunks(k)
        .flat_map(|row| {
            let (codes, scales) = xabe_dsp::quantize_q8(row);
            let out: Vec<f32> = codes
                .iter()
                .enumerate()
                .map(|(j, &c)| f32::from(c) * scales[j / xabe_dsp::GROUP])
                .collect();
            out
        })
        .collect()
}

/// The whole product on the tiled kernel, against the operands it truly reads.
///
/// Two dispatches hide behind one call and the reference has to follow them,
/// because a tolerance wide enough to cover both would be wide enough to hide
/// a permuted block:
///
///   * the two K-quants take `gemm_i8`, which hands the tensor core the
///     checkpoint's own codes and quantises the *activation* to int8. Exact
///     weights, approximate activation.
///   * everything else takes `gemm`, which rounds both operands to f16.
///
/// Comparing either against unrounded operands would measure the staging
/// rather than the unpacking, which is not what this test is for.
#[test]
fn quantized_gemm_matches_the_cpu_dequantizer() {
    let Some(g) = gpu() else { return };
    // Past GEMV_MAX_M so it dispatches to the tensor cores, and past one block
    // tile in n so the predication runs too.
    let (m, k, n) = (140usize, 512usize, 130usize);
    assert!(m > xabe_cuda::GEMV_MAX_M, "this shape must take gemm");

    for &(q, gt) in FORMATS {
        let raw = blocks(q, n * k / q.block_size(), 21);
        let w = xabe_gguf::dequantize_blocks(gt, &raw, n * k).expect("the CPU decoder");
        let a = seq(m * k, 22);

        let half_of = |v: &[f32]| -> Vec<f32> {
            v.iter()
                .map(|&x| f32::from(half::f16::from_f32(x)))
                .collect()
        };
        let int8 = matches!(q, Quant::Q4K | Quant::Q6K);
        let (ah, wh) = match int8 {
            true => (as_int8(&a, k), w.clone()),
            false => (half_of(&a), half_of(&w)),
        };
        let want = xabe_dsp::linear(&ah, m, k, &wh, None, n);

        let da = g.upload(&a).unwrap();
        let dw = g.upload_quant(q, &raw).unwrap();
        let out = g
            .gemm_batched(
                Operand::F32(&da),
                Operand::Q { data: &dw, ty: q },
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .expect("the quantized gemm");
        let got = g.download(&out).unwrap();

        // Judged against the size of the *terms*, not the size of the sum.
        //
        // A tolerance relative to the result is the wrong rule for a dot
        // product that cancels, and this test found out how wrong: a Q5_0 row
        // of 512 terms of magnitude 0.3 summed to -3.7e-4, so a perfectly
        // ordinary reordering error of 1.1e-5 was 3% of the answer. The
        // backward-error bound for f32 summation is `k * eps * sum|terms|`,
        // which is what this is - and it is still nowhere near loose enough to
        // hide a permuted block, since permuting one changes the result by the
        // size of the terms rather than by the size of the rounding.
        let name = format!("gemm {q:?}");
        assert_eq!(want.len(), got.len(), "{name}: length");
        for row in 0..m {
            for col in 0..n {
                let i = row * n + col;
                let scale: f32 = (0..k)
                    .map(|j| (ah[row * k + j] * wh[col * k + j]).abs())
                    .sum();
                let tol = ATOL + RTOL * scale;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "{name}: element {i} is {}, wanted {} (tolerance {tol})",
                    got[i],
                    want[i],
                );
            }
        }
    }
}

/// The two shapes the packed path refuses, and why each is a refusal.
#[test]
fn the_packed_path_refuses_what_it_cannot_address() {
    let Some(g) = gpu() else { return };

    // A contraction that is not a whole number of blocks. `q_at` divides to
    // find the block, so this would read into the previous row's last block:
    // in bounds, and wrong.
    let raw = blocks(Quant::Q4K, 4, 31);
    let dw = g.upload_quant(Quant::Q4K, &raw).unwrap();
    let da = g.upload(&seq(300, 32)).unwrap();
    let err = g
        .gemm_batched(
            Operand::F32(&da),
            Operand::Q {
                data: &dw,
                ty: Quant::Q4K,
            },
            None,
            Batch::single(4),
            1,
            300,
            4,
        )
        .expect_err("300 is not a multiple of 256");
    assert!(
        matches!(
            err,
            xabe_cuda::CudaError::RaggedBlock { k: 300, block: 256 }
        ),
        "wrong error: {err}",
    );

    // A quantized left operand. Activations are produced at f32 by the
    // previous kernel; there is nothing to unpack and no path that reads one.
    let dwf = g.upload(&seq(256 * 4, 33)).unwrap();
    let err = g
        .gemm_batched(
            Operand::Q {
                data: &dw,
                ty: Quant::Q4K,
            },
            Operand::F32(&dwf),
            None,
            Batch::single(4),
            1,
            256,
            4,
        )
        .expect_err("an activation may not be quantized");
    assert!(
        matches!(err, xabe_cuda::CudaError::QuantizedActivation),
        "wrong error: {err}",
    );
}

/// The runtime quantiser, against its CPU twin, code for code.
///
/// Exact equality on both halves rather than a tolerance. This is the one
/// approximation the engine introduces deliberately, and the thing worth
/// checking is that the two implementations approximate *identically* - a
/// tolerance here would hide exactly the disagreement that matters, which is a
/// group boundary in the wrong place or a rounding mode that differs at .5.
#[test]
fn the_runtime_quantiser_matches_its_cpu_twin() {
    let Some(g) = gpu() else { return };

    // Awkward on purpose: 3 rows so the row/group division is exercised, and a
    // k that is several groups but not a power-of-two count of them.
    let (rows, k) = (3usize, 320usize);
    let mut x = seq(rows * k, 91);
    // One group entirely zero, which must give scale zero and codes zero
    // rather than a division by zero.
    x[64..96].fill(0.0);
    // One group whose maximum is a power of two, so the scale is exact and
    // ties are reachable.
    for (j, v) in x[96..128].iter_mut().enumerate() {
        *v = (j as f32 - 16.0) / 16.0;
    }

    let (want_c, want_s) = xabe_dsp::quantize_q8(&x);
    let dx = g.upload(&x).unwrap();
    let (got_c, got_s) = g.quantize_q8_for_test(&dx, k, rows).unwrap();

    assert_eq!(got_c, want_c, "codes");
    assert_eq!(got_s, want_s, "scales");
    assert_eq!(want_s[2], 0.0, "the zero group did not get a zero scale");
}

/// The wide K-quant mat-vecs, against the same product taken at f32.
///
/// Two things at once, and only one of them is a tolerance. The addressing is
/// checked by the *shape* of the disagreement: a lane that reads the wrong
/// sixteen bytes does not land within a few parts in a thousand of the right
/// answer, it lands somewhere else entirely. So the bound below is what int8
/// activation costs and nothing more - measured at 3.7e-3 relative on this
/// card, allowed 1e-2 here.
///
/// `k` is a multiple of 1024 because that is the fast path's condition: four
/// super-blocks to a warp. At 512 the kernel takes the older per-byte path,
/// which `every_block_format_unpacks_element_for_element` already covers.
#[test]
fn the_wide_kquant_matvec_agrees_with_the_f32_product() {
    let Some(g) = gpu() else { return };

    let (k, n) = (1024usize, 24usize);
    for &(q, gt) in &[(Quant::Q4K, GgmlType::Q4K), (Quant::Q6K, GgmlType::Q6K)] {
        let raw = blocks(q, n * k / 256, 5);
        let wf = xabe_gguf::dequantize_blocks(gt, &raw, n * k).expect("the CPU decoder");

        let dq = g.upload_quant(q, &raw).unwrap();
        let df = g.upload(&wf).unwrap();

        for rows in [1usize, 5] {
            let x = seq(rows * k, 17);
            let dx = g.upload(&x).unwrap();
            let want = g
                .gemm_batched(
                    Operand::F32(&dx),
                    Operand::F32(&df),
                    None,
                    Batch::single(rows * n),
                    rows,
                    k,
                    n,
                )
                .unwrap();
            let got = g
                .gemm_batched(
                    Operand::F32(&dx),
                    Operand::Q { data: &dq, ty: q },
                    None,
                    Batch::single(rows * n),
                    rows,
                    k,
                    n,
                )
                .unwrap();
            let (want, got) = (g.download(&want).unwrap(), g.download(&got).unwrap());
            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-2 * scale,
                    "{q:?} rows={rows} element {i}: {a} against {b}",
                );
            }
        }
    }
}

/// What the tiled matmul's f16 staging costs at the scale a prefill runs it.
///
/// Not a rejection and not a gate - a number, printed, because "the operands
/// are rounded to f16" is a description and the question that matters is how
/// far that moves a projection of the width these models use.
#[test]
fn the_tiled_matmul_reports_its_own_staging_error() {
    let Some(g) = gpu() else { return };
    let (m, k, n) = (64usize, 4096usize, 512usize);
    // Centred, which `seq` is not - it spans [-2, 6]. That matters more than it
    // looks: an operand with a mean makes the dot product a large positive
    // number that no rounding threatens, and reports an error a hundred times
    // smaller than the same kernel makes on an activation, which cancels.
    let a: Vec<f32> = seq(m * k, 11).iter().map(|v| v - 2.0).collect();
    let w: Vec<f32> = seq(n * k, 12).iter().map(|v| v - 2.0).collect();
    let da = g.upload(&a).unwrap();
    let dw = g.upload(&w).unwrap();
    let got = g
        .download(
            &g.gemm_batched(
                Operand::F32(&da),
                Operand::F32(&dw),
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .unwrap(),
        )
        .unwrap();

    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for r in 0..m {
        for c in 0..n {
            let want: f64 = (0..k)
                .map(|i| f64::from(a[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            worst = worst.max((want - f64::from(got[r * n + c])).abs());
            scale = scale.max(want.abs());
        }
    }
    println!(
        "gemm {m}x{k}x{n}: worst {worst:.5} against a span of {scale:.5} ({:.3}%)",
        100.0 * worst / scale
    );
    // Loose: this exists to report, and a regression of an order of magnitude
    // is what it would catch.
    assert!(worst < 0.05 * scale, "staging error {worst} of {scale}");
}

/// Several matrices against one activation, as one product.
///
/// The attention projections are issued this way: `Batch::a == 0` says every
/// matrix of the batch multiplies the *same* left operand, which is what turns
/// three launches of forty blocks into one launch of a hundred and twenty. Two
/// things could go wrong silently and this pins both - the weight stride, which
/// would read the neighbouring matrix, and the activation's own row stride,
/// which the packed paths derive from a row count rather than from `Batch::a`
/// and which has to stop advancing when the operand is shared.
///
/// Bit-for-bit, not a tolerance: the batched product and the separate ones do
/// the same arithmetic in the same order on the same bytes. Anything else is a
/// mistake rather than a rounding difference.
#[test]
fn a_batch_over_one_activation_matches_the_same_products_apart() {
    let Some(g) = gpu() else { return };
    let (k, n, count) = (512usize, 128usize, 3usize);

    // Both dispatches: `m` under GEMV_MAX_M takes the mat-vec, past it the
    // tiled integer kernel, and the shared activation has to hold on both.
    for &m in &[1usize, xabe_cuda::GEMV_MAX_M, 140] {
        for &(q, _) in FORMATS {
            let raw = blocks(q, count * n * k / q.block_size(), 31);
            let a = seq(m * k, 32);
            let da = g.upload(&a).unwrap();
            let dw = g.upload_quant(q, &raw).unwrap();

            let together = g
                .gemm_batched(
                    Operand::F32(&da),
                    Operand::Q { data: &dw, ty: q },
                    None,
                    Batch {
                        count,
                        a: 0,
                        w: n * k,
                        out: m * n,
                        w_row: 0,
                    },
                    m,
                    k,
                    n,
                )
                .expect("the batched product");
            let got = g.download(&together).unwrap();

            let per = q.block_size();
            for c in 0..count {
                let bytes = raw.len() / count;
                let one = g.upload_quant(q, &raw[c * bytes..(c + 1) * bytes]).unwrap();
                let apart = g
                    .gemm_batched(
                        Operand::F32(&da),
                        Operand::Q { data: &one, ty: q },
                        None,
                        Batch::single(m * n),
                        m,
                        k,
                        n,
                    )
                    .expect("one product");
                assert_eq!(
                    g.download(&apart).unwrap(),
                    got[c * m * n..(c + 1) * m * n],
                    "{q:?} m={m} block={per}: element {c} of the batch differs \
                     from the same matrix on its own",
                );
            }
        }
    }
}

/// `merge_heads_q` quantizes exactly as `quantize_q8` would have, after the
/// same merge.
///
/// Exact equality, same as the runtime quantiser's own test and for the same
/// reason: the fused twin exists to *replace* a separate quantise pass over
/// the merged context, so the thing worth checking is that the two produce
/// identical codes and scales - a tolerance would hide precisely the
/// group-boundary disagreement a wrong index would cause.
#[test]
fn the_merged_context_twin_matches_a_separate_quantise() {
    let Some(g) = gpu() else { return };

    // Odd-shaped on purpose: three tokens, and a head count times head_dim
    // that is several groups of 32 without being a power of two.
    let (t, heads, hd) = (3usize, 6usize, 64usize);
    let x = seq(t * heads * hd, 61);
    let dx = g.upload(&x).unwrap();

    let merged = g.merge_heads(&dx, t, heads, hd).unwrap();
    let (want_c, want_s) = g.quantize_q8_for_test(&merged, heads * hd, t).unwrap();

    let (fused, q8) = g.merge_heads_q(&dx, t, heads, hd).unwrap();
    let (got_c, got_s) = g.q8_parts_for_test(&q8).unwrap();

    assert_eq!(
        g.download(&fused).unwrap(),
        g.download(&merged).unwrap(),
        "the fused merge changed the f32 output"
    );
    assert_eq!(got_c, want_c, "codes");
    assert_eq!(got_s, want_s, "scales");
}

/// The packed embedding gather returns the rows the CPU decoder returns, in
/// every format, including the two whose device layout is not the file's.
#[test]
fn the_packed_embedding_gather_matches_the_cpu_dequantizer() {
    let Some(g) = gpu() else { return };
    let (vocab, ch) = (9usize, 512usize);
    for &(q, gt) in FORMATS {
        let raw = blocks(q, vocab * ch / q.block_size(), 11);
        let want = xabe_gguf::dequantize_blocks(gt, &raw, vocab * ch).expect("the CPU decoder");
        let table = g.upload_quant(q, &raw).unwrap();
        let ids: Vec<i64> = vec![8, 0, 3, 3, 8];
        let dids = g.upload_i64(&ids).unwrap();
        let got = g
            .download(
                &g.embed_packed(&table, q, &dids, ids.len(), ch, 1.0)
                    .unwrap(),
            )
            .unwrap();
        for (r, &id) in ids.iter().enumerate() {
            let w = &want[id as usize * ch..(id as usize + 1) * ch];
            let e = &got[r * ch..(r + 1) * ch];
            assert!(
                w.iter().zip(e).all(|(a, b)| a == b),
                "{q:?}: row {id} gathered wrong"
            );
        }
        // A scale rides along, as the models that scale their embeddings want.
        let scaled = g
            .download(&g.embed_packed(&table, q, &dids, 1, ch, 0.5).unwrap())
            .unwrap();
        assert!(
            scaled
                .iter()
                .zip(&want[8 * ch..9 * ch])
                .all(|(a, b)| *a == b * 0.5),
            "{q:?}: the scale was not applied"
        );
    }
    // A row that is not a whole number of blocks is refused, not decoded.
    let raw = blocks(Quant::Q4K, 4, 12);
    let table = g.upload_quant(Quant::Q4K, &raw).unwrap();
    let dids = g.upload_i64(&[0]).unwrap();
    assert!(
        g.embed_packed(&table, Quant::Q4K, &dids, 1, 300, 1.0)
            .is_err()
    );
}

/// The norm-fused mat-vec is the mat-vec, the add and the normalisation it
/// replaces: `h` bit for bit, the row and its twin to the CPU twin's tolerance.
#[test]
fn the_norm_fused_matvec_is_the_chain_it_replaces() {
    let Some(g) = gpu() else { return };
    let (k, n, eps) = (1024usize, 96usize, 1e-5f32);
    for q in [Quant::Q4K, Quant::Q6K] {
        let raw = blocks(q, n * k / q.block_size(), 31);
        let a = seq(k, 32);
        let h0 = seq(n, 33);
        let nw: Vec<f32> = seq(n, 34).iter().map(|v| 1.0 + 0.25 * v).collect();

        let da = g.upload(&a).unwrap();
        let aq = g.quantize_activation(&da, 1, k).unwrap();
        let dw = g.upload_quant(q, &raw).unwrap();
        let dnw = g.upload(&nw).unwrap();

        // The chain: the same packed mat-vec on the same twin, then the add
        // and the normalisation.
        let out = g
            .gemm_batched(
                Operand::F32Q { data: &da, q8: &aq },
                Operand::Q { data: &dw, ty: q },
                None,
                Batch::single(n),
                1,
                k,
                n,
            )
            .expect("the chain's mat-vec");
        let out = g.download(&out).unwrap();
        let h_want: Vec<f32> = h0.iter().zip(&out).map(|(h, o)| h + o).collect();
        let x_want = xabe_dsp::rms_norm(&h_want, 1, n, &nw, eps);
        let (codes_want, scales_want) = xabe_dsp::quantize_q8(&x_want);

        let mut h = g.upload(&h0).unwrap();
        let mut scratch = NormScratch::new();
        // Twice through one scratch: the counter must come back to zero.
        for pass in 0..2 {
            let mut h_pass = g.upload(&h0).unwrap();
            let (x, xq) = g
                .gemv_norm(
                    Operand::Q { data: &dw, ty: q },
                    &aq,
                    k,
                    n,
                    &mut h_pass,
                    &dnw,
                    eps,
                    &mut scratch,
                )
                .expect("gemv_norm");
            let h_got = g.download(&h_pass).unwrap();
            assert_eq!(
                h_want, h_got,
                "{q:?} pass {pass}: the residual stream differs"
            );
            let x_got = g.download(&x).unwrap();
            for i in 0..n {
                assert!(
                    (x_want[i] - x_got[i]).abs() <= 1e-5 + 1e-5 * x_want[i].abs(),
                    "{q:?} pass {pass}: x[{i}] is {} wanted {}",
                    x_got[i],
                    x_want[i]
                );
            }
            let (codes_got, scales_got) = g.q8_parts_for_test(&xq).unwrap();
            for gi in 0..n / 32 {
                assert!(
                    (scales_want[gi] - scales_got[gi]).abs() <= 1e-6 + 1e-5 * scales_want[gi],
                    "{q:?} pass {pass}: scale {gi} is {} wanted {}",
                    scales_got[gi],
                    scales_want[gi]
                );
            }
            for i in 0..n {
                // A value an ulp from a rounding boundary may round the other
                // way; a code two apart is a different number.
                assert!(
                    (codes_want[i] as i32 - codes_got[i] as i32).abs() <= 1,
                    "{q:?} pass {pass}: code {i} is {} wanted {}",
                    codes_got[i],
                    codes_want[i]
                );
            }
            h = h_pass;
        }
        drop(h);

        // Refusals: a ragged `n`, and a twin of the wrong length.
        let mut hh = g.upload(&h0).unwrap();
        assert!(
            g.gemv_norm(
                Operand::Q { data: &dw, ty: q },
                &aq,
                k,
                n - 1,
                &mut hh,
                &dnw,
                eps,
                &mut scratch
            )
            .is_err()
        );
        let short = g
            .quantize_activation(&g.upload(&seq(512, 35)).unwrap(), 1, 512)
            .unwrap();
        assert!(
            g.gemv_norm(
                Operand::Q { data: &dw, ty: q },
                &short,
                k,
                n,
                &mut hh,
                &dnw,
                eps,
                &mut scratch
            )
            .is_err()
        );
    }
}

/// A product read from row `w_first` of a weight is the product of the rows
/// from there, bit for bit, on every path the offset can take.
///
/// One row goes through the mat-vec, many through the tiled kernels - the
/// integer one for the K-quant, the f16 one otherwise - and the f16 weight
/// is offset in elements where the packed one is offset in whole blocks. The
/// refusal is the offset that is not whole blocks.
#[test]
fn a_product_from_an_offset_row_is_the_product_of_the_rows_from_there() {
    let Some(g) = gpu() else { return };
    let (k, n_all, first, n) = (1024usize, 96usize, 40usize, 24usize);
    for m in [1usize, 64] {
        let a = seq(m * k, 71);
        let da = g.upload(&a).unwrap();
        // Packed.
        let raw = blocks(Quant::Q4K, n_all * k / 256, 72);
        let dw = g.upload_quant(Quant::Q4K, &raw).unwrap();
        let bpr = k / 256 * Quant::Q4K.type_size();
        let part = g
            .upload_quant(Quant::Q4K, &raw[first * bpr..(first + n) * bpr])
            .unwrap();
        let want = g
            .gemm_batched(
                Operand::F32(&da),
                Operand::Q {
                    data: &part,
                    ty: Quant::Q4K,
                },
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .unwrap();
        let got = g
            .gemm_batched_from(
                Operand::F32(&da),
                Operand::Q {
                    data: &dw,
                    ty: Quant::Q4K,
                },
                first,
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .unwrap();
        assert_eq!(
            g.download(&want).unwrap(),
            g.download(&got).unwrap(),
            "packed, m {m}"
        );
        // f16.
        let wf = seq(n_all * k, 73);
        let dh = g.upload_f16(&wf).unwrap();
        let ph = g.upload_f16(&wf[first * k..(first + n) * k]).unwrap();
        let want = g
            .gemm_batched(
                Operand::F32(&da),
                Operand::F16(&ph),
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .unwrap();
        let got = g
            .gemm_batched_from(
                Operand::F32(&da),
                Operand::F16(&dh),
                first,
                None,
                Batch::single(m * n),
                m,
                k,
                n,
            )
            .unwrap();
        assert_eq!(
            g.download(&want).unwrap(),
            g.download(&got).unwrap(),
            "f16, m {m}"
        );
        // An offset past the rows there are.
        assert!(
            g.gemm_batched_from(
                Operand::F32(&da),
                Operand::Q {
                    data: &dw,
                    ty: Quant::Q4K
                },
                n_all - n + 1,
                None,
                Batch::single(m * n),
                m,
                k,
                n
            )
            .is_err()
        );
    }
}

/// `gemv_q_rows` - several int8 rows against one packed weight stream - is
/// `gemv` a row at a time, bit for bit: two, three and four rows, both block
/// formats, a contraction that is not a whole number of warp trips, a batch
/// that shares its activation, and a bias.
#[test]
fn several_rows_against_a_packed_weight_are_the_rows_one_at_a_time() {
    let Some(g) = gpu() else { return };
    // Five super-blocks a row: the fourth trip of a warp has one lane's
    // worth, so the `b + sub < nb` guard is exercised.
    let (k, n) = (1280usize, 96usize);
    for q in [Quant::Q4K, Quant::Q6K] {
        let raw = blocks(q, n * k / q.block_size(), 71);
        let dw = g.upload_quant(q, &raw).unwrap();
        let bias = g.upload(&seq(n, 72)).unwrap();
        for m in 2..=4usize {
            for (count, shared) in [(1usize, false), (3, true)] {
                let a = seq(if shared { m * k } else { count * m * k }, 73 + m as u64);
                let da = g.upload(&a).unwrap();
                let sa = if shared { 0 } else { m * k };
                // The chain: one row at a time, the same weight.
                let mut want = Vec::new();
                for z in 0..count {
                    for r in 0..m {
                        let row = if shared { r * k } else { (z * m + r) * k };
                        let one = g.upload(&a[row..row + k]).unwrap();
                        let out = g
                            .gemm_batched(
                                Operand::F32(&one),
                                Operand::Q { data: &dw, ty: q },
                                Some(&bias),
                                Batch::single(n),
                                1,
                                k,
                                n,
                            )
                            .expect("one row");
                        want.extend(g.download(&out).unwrap());
                    }
                }
                let out = g
                    .gemm_batched(
                        Operand::F32(&da),
                        Operand::Q { data: &dw, ty: q },
                        Some(&bias),
                        Batch {
                            count,
                            a: sa,
                            w: 0,
                            out: m * n,
                            w_row: 0,
                        },
                        m,
                        k,
                        n,
                    )
                    .expect("several rows");
                let got = g.download(&out).unwrap();
                assert_eq!(want, got, "{q:?} m {m} count {count} shared {shared}");
            }
        }
    }
}
