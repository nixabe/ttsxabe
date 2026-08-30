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

use xabe_cuda::{Batch, Gpu, Operand, Quant};
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
/// Sixteen rows at a time, which is `GEMV_MAX_M`, so it stays on the scalar
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
        let dw = g.upload_u8(&raw).unwrap();

        for j0 in (0..k).step_by(16) {
            let rows = 16.min(k - j0);
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
                    // `==` and not `to_bits()`, for one reason found by this
                    // test: a negative scale times a zero quantum is `-0.0` on
                    // the CPU, and the warp reduction adds it to `0.0` and gets
                    // `+0.0`. The bit patterns differ, the numbers do not, and
                    // IEEE says they are equal. Every other value here is
                    // reproduced exactly, so this stays an equality rather
                    // than becoming a tolerance.
                    assert!(
                        w == e,
                        "{q:?}: element [{col}][{}] is {e}, wanted {w}",
                        j0 + r,
                    );
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
        let dw = g.upload_u8(&raw).unwrap();
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

/// The whole product on the tiled kernel, which rounds its operands to f16.
///
/// The reference sees the rounded weights for the same reason
/// `gemm_matches_the_scalar_linear_at_every_awkward_shape` does: comparing
/// against the unrounded ones would measure the staging rather than the
/// unpacking, and would need a tolerance loose enough to hide a real mistake.
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
        let (ah, wh) = (half_of(&a), half_of(&w));
        let want = xabe_dsp::linear(&ah, m, k, &wh, None, n);

        let da = g.upload(&a).unwrap();
        let dw = g.upload_u8(&raw).unwrap();
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
    let dw = g.upload_u8(&raw).unwrap();
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
