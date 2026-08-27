//! Prints what a GGUF file declares. `cargo run -p xabe-gguf --example probe -- PATH`
fn main() {
    let path = std::env::args().nth(1).expect("usage: probe PATH");
    let f = xabe_gguf::GgufFile::open(&path).expect("open");
    println!(
        "version {} alignment {} tensors {}",
        f.version(),
        f.alignment(),
        f.len()
    );
    let mut keys: Vec<_> = f.keys().collect();
    keys.sort_unstable();
    for k in keys {
        let v = format!("{:?}", f.get(k).unwrap());
        let cut = (0..=90.min(v.len()))
            .rev()
            .find(|&i| v.is_char_boundary(i))
            .unwrap_or(0);
        println!("  {k} = {}", &v[..cut]);
    }
    let mut total = 0u64;
    for t in f.tensors() {
        total += t.n_elements;
    }
    println!("parameters {total}");
    for t in f.tensors().iter().take(4) {
        println!(
            "  {:32} dims {:?} shape {:?} {:?}",
            t.name,
            t.dims,
            t.shape(),
            t.ggml_type
        );
    }
}
