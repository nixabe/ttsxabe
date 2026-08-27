//! Prints a tensor's values. `cargo run --example dump_tensor -- PATH NAME [N]`
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let f = xabe_gguf::GgufFile::open(&a[1]).expect("open");
    let v = f.tensor_f32(&a[2]).expect("read");
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(v.len());
    println!("{} len={} first {}:", a[2], v.len(), n.min(v.len()));
    for (i, x) in v.iter().take(n).enumerate() {
        print!("{i}:{x:.6} ");
        if i % 8 == 7 {
            println!();
        }
    }
    println!();
    let all_one = v.iter().all(|&x| (x - 1.0).abs() < 1e-6);
    println!(
        "all ones? {all_one}   min={:.6} max={:.6}",
        v.iter().cloned().fold(f32::INFINITY, f32::min),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}
