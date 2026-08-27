//! A real llama.cpp-quantized checkpoint, opened and compared to its f16 twin.
//!
//! The `dequant` tests prove the unpacking matches `gguf-py` on synthetic
//! encodings. This proves the same code reaches a file `llama-quantize`
//! actually wrote: the directory sizes correctly at the block arithmetic, the
//! offsets land, and the values come out near the weights they were quantized
//! from.
//!
//! "Near" is the whole point and is why this is a separate file. Quantization
//! is lossy on purpose, so there is nothing to assert exactly - what is
//! asserted is that the error is the size quantization explains and not the
//! size a layout bug would produce. A transposed nibble order gives a tensor
//! that is a *permutation* of the right one: same values, same histogram,
//! utterly wrong correlation. So correlation is what is checked, not the mean
//! absolute error, which a permutation would pass.

use std::path::PathBuf;

fn f16_model() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/llm/Llama-Breeze2-8B-Instruct-text-only.f16.gguf");
    p.is_file().then_some(p)
}

/// Quantized copies live outside the repository: they are several gigabytes
/// each and are reproducible with one `llama-quantize` invocation.
fn quantized(kind: &str) -> Option<PathBuf> {
    let dir = std::env::var("XABE_QUANT_DIR").ok()?;
    let p = PathBuf::from(dir).join(format!("breeze-{kind}.gguf"));
    p.is_file().then_some(p)
}

/// Pearson correlation, which is what separates "lossy" from "shuffled".
fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

fn check(kind: &str, min_corr: f64) {
    let Some(f16_path) = f16_model() else {
        println!("SKIP: the f16 Breeze2 is missing");
        return;
    };
    let Some(q_path) = quantized(kind) else {
        println!(
            "SKIP: set XABE_QUANT_DIR to a directory holding breeze-{kind}.gguf \
             (llama-quantize <f16> <out> {kind})"
        );
        return;
    };

    let a = xabe_gguf::GgufFile::open(&f16_path).expect("open f16");
    // A half-written file fails here rather than later, with
    // `TensorOutOfBounds` naming the tensor that ran past the end - which is
    // exactly what happened the first time this ran against a `llama-quantize`
    // that had not finished. The message is the useful part; the point of the
    // hint is that racing the writer looks like a corrupt model otherwise.
    let b = xabe_gguf::GgufFile::open(&q_path).unwrap_or_else(|e| {
        panic!(
            "{kind}: {e}\n(is llama-quantize still writing {}?)",
            q_path.display()
        )
    });

    // Same model, so the same directory - only the widths change.
    assert_eq!(a.len(), b.len(), "{kind}: tensor count");
    assert_eq!(
        a.get_u32("llama.block_count"),
        b.get_u32("llama.block_count"),
        "{kind}: geometry"
    );

    let mut compared = 0;
    let mut worst = 1.0f64;
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.16.attn_v.weight",
        "blk.31.ffn_gate.weight",
    ] {
        let (ia, ib) = (a.info(name).expect(name), b.info(name).expect(name));
        assert_eq!(ia.dims, ib.dims, "{kind}: {name} dims");
        assert_eq!(ia.n_elements, ib.n_elements, "{kind}: {name} element count");

        let wide = a.tensor_f32(name).expect("f16 side");
        let got = b.tensor_f32(name).expect("quantized side");
        assert_eq!(got.len(), wide.len(), "{kind}: {name} length");
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{kind}: {name} has a non-finite value"
        );

        let c = correlation(&wide, &got);
        worst = worst.min(c);
        // A correct unpacking of a lossy encoding correlates very highly with
        // what it encoded. A permuted one correlates around zero, which is
        // what makes this the right statistic rather than a mean error.
        assert!(
            c > min_corr,
            "{kind}: {name} correlates {c:.5} with the f16, wanted > {min_corr}"
        );
        compared += 1;
    }
    println!("  {kind:8} {compared} tensors, worst correlation {worst:.5}");
    assert_eq!(compared, 4);
}

#[test]
fn a_q8_0_checkpoint_reads_back_close_to_its_f16_original() {
    // Eight bits and a per-32 scale: essentially lossless at this precision.
    check("Q8_0", 0.9999);
}

#[test]
fn a_q6_k_checkpoint_reads_back_close_to_its_f16_original() {
    check("Q6_K", 0.999);
}

#[test]
fn a_q4_k_m_checkpoint_reads_back_close_to_its_f16_original() {
    // Four bits, six-bit scales and minimums per sub-block of 32. Lossy
    // enough to be visible and nowhere near lossy enough to decorrelate.
    check("Q4_K_M", 0.99);
}

#[test]
fn a_mixed_precision_checkpoint_holds_more_than_one_format() {
    // What `Q4_K_M` actually means: llama-quantize picks a different format
    // per tensor role, so one file carries several. A reader that assumed a
    // file had a single type would open this one and mis-size most of it.
    let Some(q_path) = quantized("Q4_K_M") else {
        println!("SKIP: set XABE_QUANT_DIR");
        return;
    };
    let f = xabe_gguf::GgufFile::open(&q_path).expect("open");
    let mut kinds: Vec<String> = f
        .tensors()
        .iter()
        .map(|t| format!("{:?}", t.ggml_type))
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    println!("  Q4_K_M carries: {kinds:?}");
    assert!(
        kinds.len() > 1,
        "a _M mix should hold several formats, found {kinds:?}"
    );
}
