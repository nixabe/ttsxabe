//! The same checkpoint, read from the GGUF instead of the safetensors.
//!
//! A separate test binary on purpose. Each translator is 26.5 GB on the
//! device, so a safetensors instance and a GGUF one cannot coexist on a 48 GB
//! card; cargo runs test *targets* one after another, so putting these here
//! means the other binary's process has exited and freed the card first.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use xabe_translate::Translator;

fn workspace(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn gguf() -> Option<PathBuf> {
    let p = match std::env::var("XABE_TRANSLATOR_GGUF") {
        Ok(v) => PathBuf::from(v),
        Err(_) => workspace("models/llm/taigi-translator-13b-f16.gguf"),
    };
    p.is_file().then_some(p)
}

fn safetensors() -> Option<PathBuf> {
    let p = workspace("models/translator/taigi-llama2-13b");
    p.is_dir().then_some(p)
}

fn ordinal() -> usize {
    std::env::var("XABE_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// One instance for the whole binary: three concurrent 26.5 GB loads is an
/// out-of-memory that reads like a broken loader.
static MODEL: OnceLock<Option<Mutex<Translator>>> = OnceLock::new();

fn model(path: &Path) -> Option<MutexGuard<'static, Translator>> {
    let slot = MODEL.get_or_init(|| match Translator::open(path, ordinal()) {
        Ok(m) => Some(Mutex::new(m)),
        Err(e) => {
            println!("SKIP: the GGUF translator did not open: {e}");
            None
        }
    });
    slot.as_ref().map(|m| m.lock().expect("not poisoned"))
}

macro_rules! gguf_or_skip {
    () => {
        match gguf() {
            Some(p) => p,
            None => {
                println!("SKIP: models/llm/taigi-translator-13b-f16.gguf is missing");
                return;
            }
        }
    };
}

#[test]
fn the_two_containers_describe_the_same_model() {
    // Metadata only - no weights are read, so this is cheap and runs even
    // where the device does not.
    let Some(g) = gguf() else {
        println!("SKIP: the GGUF is missing");
        return;
    };
    let Some(dir) = safetensors() else {
        println!("SKIP: the safetensors checkpoint is missing");
        return;
    };
    let f = xabe_gguf::GgufFile::open(&g).expect("open the gguf");
    let a = xabe_llama::LlamaConfig::from_gguf(&f).expect("gguf geometry");
    let b = xabe_llama::LlamaConfig::from_dir(&dir).expect("safetensors geometry");

    assert_eq!(a.hidden_size, b.hidden_size);
    assert_eq!(a.intermediate_size, b.intermediate_size);
    assert_eq!(a.num_hidden_layers, b.num_hidden_layers);
    assert_eq!(a.num_attention_heads, b.num_attention_heads);
    assert_eq!(a.num_key_value_heads, b.num_key_value_heads);
    assert_eq!(a.vocab_size, b.vocab_size);
    assert_eq!(a.rope_theta, b.rope_theta);
    assert_eq!(a.rms_norm_eps, b.rms_norm_eps);

    let w = xabe_llama::LlamaWeights::from_gguf(&f, &a).expect("bind");
    assert_eq!(w.tensor_count(), 363);
    assert_eq!(w.parameter_count(), 13_261_870_080);
}

#[test]
fn undoing_the_rope_permutation_reproduces_the_safetensors_bits_exactly() {
    // The finding this whole path turns on, asserted rather than described.
    //
    // llama.cpp bakes its interleaved rope convention into `attn_q` and
    // `attn_k` by permuting their rows. Left alone, those two tensors differ
    // from the checkpoint in about 98% of their elements while every other
    // tensor is bit-identical - a model whose shapes all check out and whose
    // output is fluent and wrong.
    let Some(g) = gguf() else {
        println!("SKIP: the GGUF is missing");
        return;
    };
    let Some(dir) = safetensors() else {
        println!("SKIP: the safetensors checkpoint is missing");
        return;
    };
    let f = xabe_gguf::GgufFile::open(&g).expect("open");
    let st = xabe_st::StSet::open(&dir).expect("open");
    let cfg = xabe_llama::LlamaConfig::from_dir(&dir).expect("geometry");

    for (hf, gg, heads) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
            cfg.num_attention_heads,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "blk.0.attn_k.weight",
            cfg.num_key_value_heads,
        ),
    ] {
        let a = st.tensor_f16(hf).expect("safetensors");
        let b = f.tensor_f16(gg).expect("gguf");
        let info = st.info(hf).expect("bound");
        let (rows, cols) = (info.shape[0], info.shape[1]);

        let raw = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert!(
            raw > a.len() / 2,
            "{gg} should differ wildly before un-permuting, got {raw} of {}",
            a.len()
        );

        let fixed = xabe_llama::gguf::unpermute_rope(&b, rows, cols, heads);
        assert_eq!(fixed, a, "{gg} un-permuted must be the checkpoint's bits");

        // And the other direction, which is the stronger statement: the
        // checkpoint permuted forward *is* the GGUF.
        let forward = xabe_llama::gguf::permute_rope(&a, rows, cols, heads);
        assert_eq!(forward, b, "{hf} permuted must be the GGUF's bits");
    }
}

#[test]
fn the_value_projection_is_not_permuted() {
    // The asymmetry that identifies the cause. No rotation is applied to
    // values, so `attn_v` is untouched by the converter - and un-permuting it
    // by mistake would break a model that was correct.
    let Some(g) = gguf() else {
        println!("SKIP: the GGUF is missing");
        return;
    };
    let Some(dir) = safetensors() else {
        println!("SKIP: the safetensors checkpoint is missing");
        return;
    };
    let f = xabe_gguf::GgufFile::open(&g).expect("open");
    let st = xabe_st::StSet::open(&dir).expect("open");

    let a = st
        .tensor_f16("model.layers.0.self_attn.v_proj.weight")
        .expect("safetensors");
    let b = f.tensor_f16("blk.0.attn_v.weight").expect("gguf");
    assert_eq!(a, b, "attn_v must be bit-identical without any permutation");

    assert!(xabe_llama::gguf::is_rope_permuted("blk.0.attn_q.weight"));
    assert!(xabe_llama::gguf::is_rope_permuted("blk.0.attn_k.weight"));
    assert!(!xabe_llama::gguf::is_rope_permuted("blk.0.attn_v.weight"));
    assert!(!xabe_llama::gguf::is_rope_permuted(
        "blk.0.attn_output.weight"
    ));
}

#[test]
fn the_gguf_tokenizer_agrees_with_tokenizer_model() {
    let Some(g) = gguf() else {
        println!("SKIP: the GGUF is missing");
        return;
    };
    let Some(dir) = safetensors() else {
        println!("SKIP: the safetensors checkpoint is missing");
        return;
    };
    let f = xabe_gguf::GgufFile::open(&g).expect("open");
    let a = xabe_llama::Tokenizer::from_gguf(&f).expect("gguf tokenizer");
    let b = xabe_llama::Tokenizer::from_dir(&dir).expect("sentencepiece");

    // The GGUF names the embedding's four padding rows; SentencePiece does
    // not mention them. Both are right about their own file.
    assert_eq!(b.len(), 56_020);
    assert_eq!(a.len(), 56_024);

    // Every spelling matches, and that is the claim that matters for the
    // vocabulary. Scores are checked separately below, because the two
    // writers record "never merge this" differently.
    for id in 0..b.len() as u32 {
        let (x, y) = (a.piece(id).expect("gguf"), b.piece(id).expect("spm"));
        assert_eq!(x.text, y.text, "piece {id}");
    }
    for id in b.len() as u32..a.len() as u32 {
        let p = a.piece(id).expect("gguf");
        assert!(
            p.text.starts_with("[PAD"),
            "the extra pieces are the padding rows, got {:?}",
            p.text
        );
    }

    // Scores agree on every piece that can actually take part in a merge.
    //
    // Exactly four disagree, and they are the four that cannot: `<unk>`,
    // `<s>`, `</s>` and `<pad>`. llama.cpp writes -1000 so a control token
    // can never win a merge; SentencePiece leaves them at 0 and relies on the
    // piece kind to exclude them. Both engines exclude them by kind, so the
    // number is inert - and asserting equality on it, as this test first did,
    // was testing a writer's bookkeeping rather than the vocabulary.
    let mut disagreed = Vec::new();
    for id in 0..b.len() as u32 {
        let (x, y) = (a.piece(id).expect("gguf"), b.piece(id).expect("spm"));
        if x.score != y.score {
            assert_eq!(x.score, -1000.0, "piece {id} {:?}", y.text);
            assert_eq!(y.score, 0.0, "piece {id} {:?}", y.text);
            disagreed.push(y.text.clone());
        }
    }
    disagreed.sort();
    assert_eq!(
        disagreed,
        vec!["</s>", "<pad>", "<s>", "<unk>"],
        "only the non-mergeable pieces may disagree on score"
    );

    // Two pieces are also filed under different kinds, and both still end up
    // special. `<pad>` is the interesting one: this reader takes it as a
    // NORMAL piece that `special_tokens_map.json` promotes, llama.cpp marks
    // it CONTROL outright. Same destination, different route - so what is
    // asserted is the destination.
    for name in ["<unk>", "<s>", "</s>", "<pad>"] {
        assert!(a.special(name).is_some(), "{name} special in the gguf");
        assert!(b.special(name).is_some(), "{name} special in the spm");
        assert_eq!(a.special(name), b.special(name), "{name} id");
    }

    assert_eq!(a.bos(), b.bos());
    assert_eq!(a.eos(), b.eos());

    // The behavioural claim, which is the one the engine depends on.
    for s in [
        "今天天氣很好",
        "你食飽未",
        "我要去市場買東西",
        "hello",
        "[TRANS]\n今天天氣很好\n[/TRANS]\n[POJ]\n",
        "明仔載我欲去學校",
    ] {
        assert_eq!(a.encode(s), b.encode(s), "encoding {s:?}");
    }
}

#[test]
fn translations_from_the_gguf_match_the_captured_ones() {
    // The end of the argument. Same prompts, same settings, weights reached
    // through the other container.
    let path = gguf_or_skip!();
    let Some(m) = model(&path) else { return };

    let mut checked = 0;
    for (src, tgt, want) in [
        ("今天天氣很好", "POJ", "kin-á-ji̍t thiⁿ-khì chin hó"),
        ("我要去市場買東西", "HAN", "我欲來去菜市仔買物件。"),
        ("你食飽未", "HAN", "你食飽未"),
    ] {
        let got = m
            .translate(src, tgt, 256, Translator::REPEAT_PENALTY)
            .expect("translate");
        println!("  {src} [{tgt}] -> {got:?}");
        assert_eq!(got, want, "{src} [{tgt}] from the GGUF");
        checked += 1;
    }
    assert_eq!(checked, 3);
}
