//! Compares the same checkpoint read as safetensors and as GGUF.
//!
//! Two questions, both of which decide whether the containers are
//! interchangeable: do the weights round to the same f16 bits, and does the
//! GGUF's embedded tokenizer hold the same pieces as `tokenizer.model`.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: DIR GGUF");
    let gguf = std::env::args().nth(2).expect("usage: DIR GGUF");
    let dir = std::path::Path::new(&dir);

    let g = xabe_gguf::GgufFile::open(&gguf).expect("open gguf");
    let gcfg = xabe_llama::LlamaConfig::from_gguf(&g).expect("gguf geometry");
    let scfg = xabe_llama::LlamaConfig::from_dir(dir).expect("st geometry");
    let st = xabe_st::StSet::open(dir).expect("open st");

    println!("geometry:");
    for (n, a, b) in [
        ("hidden", scfg.hidden_size, gcfg.hidden_size),
        ("layers", scfg.num_hidden_layers, gcfg.num_hidden_layers),
        ("heads", scfg.num_attention_heads, gcfg.num_attention_heads),
        (
            "kv_heads",
            scfg.num_key_value_heads,
            gcfg.num_key_value_heads,
        ),
        ("vocab", scfg.vocab_size, gcfg.vocab_size),
        ("ffn", scfg.intermediate_size, gcfg.intermediate_size),
    ] {
        println!(
            "  {n:9} st={a:<8} gguf={b:<8} {}",
            if a == b { "same" } else { "DIFFER" }
        );
    }
    println!(
        "  theta     st={} gguf={}",
        scfg.rope_theta, gcfg.rope_theta
    );
    println!(
        "  eps       st={} gguf={}",
        scfg.rms_norm_eps, gcfg.rms_norm_eps
    );

    // The pairs to compare: HF name, GGUF name.
    let pairs = [
        ("model.norm.weight", "output_norm.weight"),
        (
            "model.layers.0.input_layernorm.weight",
            "blk.0.attn_norm.weight",
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            "blk.0.ffn_down.weight",
        ),
        (
            "model.layers.39.self_attn.o_proj.weight",
            "blk.39.attn_output.weight",
        ),
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("lm_head.weight", "output.weight"),
    ];
    println!("\nweights, safetensors rounded to f16 against the GGUF's stored f16:");
    for (sn, gn) in pairs {
        let a = match st.tensor_f16(sn) {
            Ok(v) => v,
            Err(e) => {
                println!("  {sn}: {e}");
                continue;
            }
        };
        let b = match g.tensor_f16(gn) {
            Ok(v) => v,
            Err(e) => {
                println!("  {gn}: {e}");
                continue;
            }
        };
        if a.len() != b.len() {
            println!("  {sn:42} LENGTH {} vs {}", a.len(), b.len());
            continue;
        }
        let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                (half::f16::from_bits(*x).to_f32() - half::f16::from_bits(*y).to_f32()).abs()
            })
            .fold(0.0f32, f32::max);
        println!(
            "  {sn:42} n={:<11} differing={diff:<8} worst={worst:.3e} {}",
            a.len(),
            if diff == 0 {
                "BIT-IDENTICAL"
            } else {
                "differs"
            }
        );
    }

    println!("\ntokenizer:");
    let tok = xabe_llama::Tokenizer::from_dir(dir).expect("sentencepiece");
    println!("  tokenizer.model pieces {}", tok.len());
    match g.get_strings("tokenizer.ggml.tokens") {
        Some(toks) => {
            println!("  gguf tokens            {}", toks.len());
            println!(
                "  gguf model             {:?}",
                g.get_str("tokenizer.ggml.model")
            );
            let mut differ = 0usize;
            let mut first: Vec<String> = Vec::new();
            for (id, t) in toks.iter().enumerate() {
                match tok.piece(id as u32) {
                    Some(p) if &p.text == t => {}
                    Some(p) => {
                        differ += 1;
                        if first.len() < 5 {
                            first.push(format!("{id}: st={:?} gguf={:?}", p.text, t));
                        }
                    }
                    None => {
                        differ += 1;
                        if first.len() < 5 {
                            first.push(format!("{id}: st=<none> gguf={t:?}"));
                        }
                    }
                }
            }
            println!("  differing pieces       {differ}");
            for f in first {
                println!("    {f}");
            }
        }
        None => println!("  the GGUF carries no tokenizer.ggml.tokens"),
    }
}
