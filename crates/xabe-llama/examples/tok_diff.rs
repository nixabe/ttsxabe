//! Where do the two tokenizer sources disagree, and does it change encoding?
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let gguf = std::env::args().nth(2).unwrap();
    let dir = std::path::Path::new(&dir);
    let f = xabe_gguf::GgufFile::open(&gguf).unwrap();
    let a = xabe_llama::Tokenizer::from_gguf(&f).unwrap();
    let b = xabe_llama::Tokenizer::from_dir(dir).unwrap();

    let mut by_kind: std::collections::BTreeMap<String, usize> = Default::default();
    let mut text_diff = 0usize;
    let mut kind_diff = 0usize;
    let mut samples = Vec::new();
    for id in 0..b.len() as u32 {
        let (x, y) = (a.piece(id).unwrap(), b.piece(id).unwrap());
        if x.text != y.text {
            text_diff += 1;
        }
        if x.kind != y.kind {
            kind_diff += 1;
            if samples.len() < 6 {
                samples.push(format!(
                    "{id} {:?}: gguf={:?} spm={:?}",
                    y.text, x.kind, y.kind
                ));
            }
        }
        if x.score != y.score {
            *by_kind.entry(format!("{:?}", y.kind)).or_default() += 1;
            if by_kind.values().sum::<usize>() <= 6 {
                samples.push(format!(
                    "{id} {:?} score gguf={} spm={} kind={:?}",
                    y.text, x.score, y.score, y.kind
                ));
            }
        }
    }
    println!("text differing: {text_diff}");
    println!("kind differing: {kind_diff}");
    println!("score differing, by kind: {by_kind:?}");
    for s in &samples {
        println!("  {s}");
    }

    let cases = [
        "今天天氣很好",
        "你食飽未",
        "我要去市場買東西",
        "hello",
        "[TRANS]\n今天天氣很好\n[/TRANS]\n[POJ]\n",
        "明仔載我欲去學校",
    ];
    let mut enc_diff = 0;
    for c in cases {
        let (x, y) = (a.encode(c), b.encode(c));
        if x != y {
            enc_diff += 1;
            println!("  ENCODE DIFFERS {c:?}\n    gguf={x:?}\n    spm ={y:?}");
        }
    }
    println!("encode differing: {enc_diff} of {}", cases.len());
    println!(
        "bos gguf={} spm={}  eos gguf={} spm={}",
        a.bos(),
        b.bos(),
        a.eos(),
        b.eos()
    );
}
