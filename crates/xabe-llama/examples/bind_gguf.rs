//! Binds a GGUF and prints what the schema made of it.
fn main() {
    let path = std::env::args().nth(1).expect("usage: bind_gguf PATH");
    let f = xabe_gguf::GgufFile::open(&path).expect("open");
    let cfg = xabe_llama::LlamaConfig::from_gguf(&f).expect("geometry");
    let w = xabe_llama::LlamaWeights::from_gguf(&f, &cfg).expect("bind");
    println!(
        "hidden {} layers {} heads {}/{} kv_dim {} vocab {} theta {}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.kv_dim(),
        cfg.vocab_size,
        cfg.rope_theta
    );
    println!(
        "file tensors {} bound {} params {}",
        f.len(),
        w.tensor_count(),
        w.parameter_count()
    );
    println!(
        "grouped-query refused? {:?}",
        cfg.refuse_grouped_query().err().map(|e| e.to_string())
    );
}
