//! Is a GGUF's attn_q/attn_k the HF tensor with its rows permuted?
//!
//! llama.cpp's converter reshapes each head's rows from HF's "halves" layout
//! (i paired with i+head_dim/2) into ggml's interleaved one (2i with 2i+1).
//! If that is what the difference is, applying the same permutation to the
//! safetensors tensor must reproduce the GGUF's bits exactly.
fn permute(w: &[u16], rows: usize, cols: usize, heads: usize) -> Vec<u16> {
    let hd = rows / heads;
    let half = hd / 2;
    let mut out = vec![0u16; w.len()];
    for h in 0..heads {
        for i in 0..half {
            for k in 0..2 {
                // HF row (k*half + i) within the head becomes ggml row (2i + k).
                let src = h * hd + k * half + i;
                let dst = h * hd + 2 * i + k;
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let gguf = std::env::args().nth(2).unwrap();
    let dir = std::path::Path::new(&dir);
    let g = xabe_gguf::GgufFile::open(&gguf).unwrap();
    let cfg = xabe_llama::LlamaConfig::from_dir(dir).unwrap();
    let st = xabe_st::StSet::open(dir).unwrap();

    for (sn, gn, heads) in [
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
        (
            "model.layers.7.self_attn.q_proj.weight",
            "blk.7.attn_q.weight",
            cfg.num_attention_heads,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "blk.0.attn_v.weight",
            cfg.num_key_value_heads,
        ),
    ] {
        let a = st.tensor_f16(sn).unwrap();
        let b = g.tensor_f16(gn).unwrap();
        let info = st.info(sn).unwrap();
        let (rows, cols) = (info.shape[0], info.shape[1]);
        let raw_diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        let p = permute(&a, rows, cols, heads);
        let perm_diff = p.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!(
            "{sn:42} rows={rows} cols={cols} heads={heads}\n    as-is differing={raw_diff:<10} permuted differing={perm_diff:<10} {}",
            if perm_diff == 0 {
                "PERMUTATION EXPLAINS IT"
            } else if raw_diff == 0 {
                "identical already"
            } else {
                "still differs"
            }
        );
    }
}
