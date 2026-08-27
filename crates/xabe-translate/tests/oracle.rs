//! The forward pass against 🤗 `LlamaForCausalLM`, per layer, and its output
//! against the reference's greedy translation.
//!
//! The captures come from `tools/oracle/capture_llama.py`, which runs the
//! reference on CPU in float32 with one thread - 53 GB of RAM and a few
//! seconds a prompt, affordable exactly once.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use xabe_translate::Translator;

/// How many layers the capture tapped individually.
const TAPS: usize = 4;

/// The gate every stage is held to, as a fraction of the tensor's own scale.
///
/// The weights are f16 on this side and f32 on the reference's, which is not a
/// rounding choice but the only width this card can hold the model in. Forty
/// layers of residual stream compound it. Measured worst is printed by every
/// run; the gate is set above it, and it is not the test that matters - the
/// argmax and the translation are.
///
/// Measured: 2.8e-5 at layer 0 rising to 7.9e-4 at the final normalisation,
/// forty layers later. The gate is set at four times that.
///
/// The two prompts report the *same* worst value at the same index for every
/// layer, which looks like a bug and is not: both begin with `<s>` and the
/// largest disagreement lands in that first token's row, which is identical
/// input either way. The logits differ, because there the maximum lands
/// elsewhere.
const GATE: f32 = 3e-3;

#[derive(Debug, Deserialize)]
struct Manifest {
    src: String,
    tgt: String,
    input_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    generated: String,
}

/// The checkpoint, or `None` if it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/translator/taigi-llama2-13b");
    p.join("model.safetensors.index.json")
        .is_file()
        .then_some(p)
}

/// Every captured prompt directory.
fn captures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/translator");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    out.sort();
    out
}

/// Which device to use. See `docs/TESTING.md`; check `nvidia-smi` first.
fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The model, loaded once for the whole test binary.
///
/// Not once per test. Every test here needs the same 27 GB of weights, cargo
/// runs the tests in one binary on several threads, and three concurrent loads
/// is 81 GB against a 48 GB card - which fails as `CUDA_ERROR_OUT_OF_MEMORY`
/// from whichever test lost the race, and reads like a broken loader rather
/// than a test-harness problem. Sharing it also serialises the device, which
/// is what a single-model engine wants anyway.
static MODEL: OnceLock<Option<Mutex<Translator>>> = OnceLock::new();

/// Borrows the shared model, skipping only when there genuinely is no device.
fn model(dir: &Path) -> Option<MutexGuard<'static, Translator>> {
    let slot = MODEL.get_or_init(|| match Translator::open(dir, ordinal()) {
        Ok(m) => Some(Mutex::new(m)),
        Err(xabe_translate::TranslateError::Cuda(xabe_cuda::CudaError::NoDevice(why))) => {
            eprintln!("SKIP: no CUDA device ({why})");
            None
        }
        Err(e) => panic!("the checkpoint is present but unusable: {e}"),
    });
    // A poisoned lock means another test panicked while holding the model.
    // Its failure is the one worth reading, so this one says so and stops.
    slot.as_ref()
        .map(|m| m.lock().expect("another test panicked holding the model"))
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn manifest(dir: &Path) -> Manifest {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).expect("manifest"))
        .expect("parse manifest")
}

/// Largest absolute disagreement scaled by the reference's own scale.
fn worst(want: &[f32], got: &[f32]) -> (f32, usize, f32) {
    assert_eq!(want.len(), got.len(), "length mismatch");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let (e, at) = want
        .iter()
        .zip(got)
        .enumerate()
        .map(|(i, (a, b))| ((a - b).abs(), i))
        .fold((0.0f32, 0), |acc, x| if x.0 > acc.0 { x } else { acc });
    (e / scale, at, scale)
}

#[test]
fn the_forward_pass_matches_the_oracle_layer_by_layer() {
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/translator/taigi-llama2-13b is missing");
        return;
    };
    let Some(m) = model(&dir) else { return };
    let clips = captures();
    assert!(!clips.is_empty(), "no captures in .golden/translator");

    let mut report: Vec<(String, f32)> = Vec::new();
    for dir in clips {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let man = manifest(&dir);
        // The reference's own ids, so a tokenizer disagreement cannot leak
        // into a forward-pass measurement. The tokenizer has its own test.
        assert_eq!(
            man.input_ids,
            m.prompt_ids(&man.src, &man.tgt),
            "{name}: the prompt tokenizes differently",
        );

        let mut cache = m.cache();
        let (logits, taps) = m
            .forward_tapped(&man.input_ids, &mut cache, TAPS)
            .expect("forward");

        for (i, got) in taps.iter().take(TAPS).enumerate() {
            let want = read_f32(&dir.join(format!("layer_{i}.bin")));
            let (e, at, scale) = worst(&want, got);
            println!("  {name} layer {i:<2} {e:.3e}  at {at:>7}, scale {scale:.3}");
            report.push((format!("{name} layer {i}"), e));
        }
        let want = read_f32(&dir.join("final_norm.bin"));
        let (e, at, scale) = worst(&want, taps.last().expect("final norm tap"));
        println!("  {name} final norm {e:.3e}  at {at:>7}, scale {scale:.3}");
        report.push((format!("{name} final norm"), e));

        let want = read_f32(&dir.join("logits.bin"));
        let got = m.gpu().download(&logits).expect("download");
        let (e, at, scale) = worst(&want, &got);
        println!("  {name} logits     {e:.3e}  at {at:>7}, scale {scale:.3}");
        report.push((format!("{name} logits"), e));

        // What matters about the logits is which token wins.
        let vocab = m.config().vocab_size;
        let arg = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
                .expect("non-empty")
                .0
        };
        for t in 0..man.input_ids.len() {
            assert_eq!(
                arg(&want[t * vocab..][..vocab]),
                arg(&got[t * vocab..][..vocab]),
                "{name}: a different token wins at position {t}",
            );
        }
    }

    let (name, e) = report
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).expect("no NaN"))
        .expect("nothing was compared");
    println!("  worst: {name} at {e:.3e}, gate {GATE:.3e}");
    assert!(e < GATE, "{name}: {e:e} of full scale");
}

#[test]
fn greedy_decoding_reproduces_the_reference_translations() {
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/translator/taigi-llama2-13b is missing");
        return;
    };
    let Some(m) = model(&dir) else { return };
    let clips = captures();
    assert!(!clips.is_empty(), "no captures in .golden/translator");

    for dir in clips {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let man = manifest(&dir);
        // Penalty off: the capture is pure greedy, and comparing two different
        // computations proves nothing about either.
        let got = m.translate(&man.src, &man.tgt, 32, 1.0).expect("translate");
        // The capture keeps the model's closing tag; `translate` cuts at it,
        // which is what the model card's own example does.
        let want = man
            .generated
            .split_once("[/")
            .map_or(man.generated.as_str(), |(h, _)| h)
            .trim();
        println!("  {name:<6} {:?} -> {got:?}", man.src);
        assert_eq!(got, want, "{name}");
        assert!(
            !man.generated_ids.is_empty(),
            "{name}: nothing was captured"
        );
    }
}

/// One translation `llama-server` produced.
#[derive(Debug, Deserialize)]
struct ServerCase {
    src: String,
    tgt: String,
    text: String,
}

/// The whole `llama-server` capture.
#[derive(Debug, Deserialize)]
struct ServerCapture {
    cases: Vec<ServerCase>,
}

#[test]
fn translations_match_the_llama_server_the_pipeline_runs_today() {
    // The second reference, and a different kind of one. 🤗 in float32 says
    // what the arithmetic should be; `llama-server` running the f16 GGUF of
    // the same weights is what the pipeline actually uses, so it says what the
    // replacement has to reproduce. This is milestone 22 as it was written.
    //
    // The capture's order is part of the fixture. `llama-server` reuses a KV
    // prefix across requests, so the same prompt can answer differently
    // depending on what preceded it - observed once, on 你食飽未, which came
    // back as `Lí ū chia̍h-pá--bōe?` after one predecessor and
    // `Lí chia̍h-pá--bōe?` after another. Two runs of the fixed corpus agree
    // with each other; a shuffled one need not.
    let Some(dir) = checkpoint() else {
        println!("SKIP: models/translator/taigi-llama2-13b is missing");
        return;
    };
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.golden/translator/llama_server.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        println!("SKIP: run tools/oracle/capture_llama_server.py first");
        return;
    };
    let cap: ServerCapture = serde_json::from_str(&text).expect("parse");
    let Some(m) = model(&dir) else { return };

    let mut agreed = 0;
    let mut trailing = Vec::new();
    for c in &cap.cases {
        // The pipeline's own settings, so this compares two implementations
        // rather than two configurations.
        let got = m
            .translate(
                &c.src,
                &c.tgt,
                256,
                xabe_translate::Translator::REPEAT_PENALTY,
            )
            .expect("translate");
        println!(
            "  {} [{}]\n    ours   {got:?}\n    theirs {:?}",
            c.src, c.tgt, c.text
        );
        if got == c.text {
            agreed += 1;
            continue;
        }
        // Where they differ it must be a single trailing character, and it
        // must be `llama-server` that added it. On 你食飽未 [HAN] it answers
        // 你食飽未？ where this engine answers 你食飽未 - and the float32 🤗
        // capture in `.golden/translator/pa` answers 你食飽未 too, so the
        // reference is on this side. One token at a near-tie, decided
        // differently by two f16 implementations with different accumulation
        // orders, is the expected shape of a disagreement here; anything
        // larger is not.
        assert!(
            c.text.starts_with(&got) && c.text.len() - got.len() <= 4,
            "{} [{}]: ours {got:?} against theirs {:?} is more than a trailing token",
            c.src,
            c.tgt,
            c.text,
        );
        trailing.push(format!("{} [{}]", c.src, c.tgt));
    }
    println!(
        "  {agreed}/{} identical; {} differ by a trailing character: {trailing:?}",
        cap.cases.len(),
        trailing.len(),
    );
    // A regression to catch: if this starts drifting on half the corpus, the
    // difference is no longer a rounding tie.
    assert!(
        trailing.len() * 4 <= cap.cases.len(),
        "{} of {} disagree, which is too many to be ties",
        trailing.len(),
        cap.cases.len(),
    );
}
