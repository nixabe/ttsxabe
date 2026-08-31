//! Times the two Llama stages: prefill and decode, separately.
//!
//! `docs/BENCHMARKS.md` says how to measure. The split matters more here than
//! anywhere else in the pipeline, because the two halves are bound by different
//! things and only one of them scales with the reply:
//!
//! - **Prefill** runs the whole prompt through in one pass. It is a matmul with
//!   as many rows as there are tokens, so it reaches the tensor cores and is
//!   compute bound.
//! - **Decode** runs one token at a time and must read every weight in the
//!   model to produce it. It is a `gemv` per projection and is bound by memory
//!   bandwidth, not arithmetic - the ceiling is the checkpoint's size divided
//!   by what the card can stream.
//!
//! A reply of N tokens costs one prefill and N decodes, so decode is what the
//! listener waits through. Reported as tokens per second, which is what
//! `llama-server` reports and so the one number that can be compared.

use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

/// Time prefill and decode for a chat or translator checkpoint.
#[derive(Debug, Parser)]
#[command(name = "xabe-llm-bench", version, about)]
struct Args {
    /// The GGUF to load.
    #[arg(long)]
    model: PathBuf,

    /// `chat` or `translate`.
    #[arg(long, default_value = "chat")]
    kind: String,

    /// CUDA device ordinal. Check `nvidia-smi` first.
    #[arg(long, default_value_t = 0)]
    device: usize,

    /// How many prompt tokens to prefill.
    #[arg(long, default_value_t = 128)]
    prompt: usize,

    /// How many tokens to decode.
    #[arg(long, default_value_t = 64)]
    decode: usize,

    /// Timed rounds.
    #[arg(long, default_value_t = 5)]
    rounds: usize,

    /// How a quantized checkpoint is held: `packed` or `f16`.
    ///
    /// The same weights either way. `packed` hands the matmul the file's own
    /// blocks and unpacks them inside the kernel; `f16` widens them at load and
    /// reads 2.6x more bytes per token. Comparing the two on one file is what
    /// separates the cost of the unpacking from the cost of the traffic.
    #[arg(long, default_value = "packed")]
    packing: String,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    v[v.len() / 2]
}

/// Bytes the decode loop must stream per token, from the file on disk.
///
/// The floor on decode is this divided by achievable bandwidth: every weight is
/// read once to produce one token, and no reuse is available at one row.
fn weight_bytes(path: &PathBuf) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64)
        .unwrap_or(0.0)
}

fn report(name: &str, prompt: usize, decode: usize, pre: f64, dec: f64, bytes: f64) {
    println!(
        "  prefill {prompt:>4} tok  {pre:8.1} ms  {:9.1} tok/s",
        prompt as f64 * 1e3 / pre
    );
    println!(
        "  decode  {decode:>4} tok  {dec:8.1} ms  {:9.1} tok/s  ({:.2} ms/tok)",
        decode as f64 * 1e3 / dec,
        dec / decode as f64
    );
    let per_tok_s = dec / decode as f64 / 1e3;
    println!(
        "  {name}: {:.0} GB/s effective against {:.2} GB of weights",
        bytes / per_tok_s / 1e9,
        bytes / 1e9
    );
}

fn main() -> ExitCode {
    let args = Args::parse();
    let bytes = weight_bytes(&args.model);
    // Token ids that exist in both vocabularies and are not special.
    let ids: Vec<u32> = (0..args.prompt).map(|i| 1000 + (i as u32 % 500)).collect();

    match args.kind.as_str() {
        "chat" => {
            let pack = match args.packing.as_str() {
                "f16" => xabe_chat::Packing::F16,
                _ => xabe_chat::Packing::Packed,
            };
            let m = match xabe_chat::ChatModel::open_with(&args.model, args.device, pack) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("open: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let (mut pres, mut decs) = (Vec::new(), Vec::new());
            for r in 0..args.rounds + 1 {
                let mut cache = m.cache();
                let t0 = Instant::now();
                if m.forward_last(&ids, &mut cache).is_err() {
                    eprintln!("prefill failed");
                    return ExitCode::FAILURE;
                }
                m.gpu().synchronize().ok();
                let pre = t0.elapsed().as_secs_f64() * 1e3;

                let t0 = Instant::now();
                for i in 0..args.decode {
                    if m.forward_last(&[1500 + i as u32 % 100], &mut cache)
                        .is_err()
                    {
                        eprintln!("decode failed");
                        return ExitCode::FAILURE;
                    }
                }
                m.gpu().synchronize().ok();
                let dec = t0.elapsed().as_secs_f64() * 1e3;
                // The first round is warm-up.
                if r > 0 {
                    pres.push(pre);
                    decs.push(dec);
                }
            }
            println!("chat {} [{}]", args.model.display(), args.packing);
            report(
                "chat",
                args.prompt,
                args.decode,
                median(pres),
                median(decs),
                bytes,
            );
        }
        "translate" => {
            let pack = match args.packing.as_str() {
                "f16" => xabe_translate::Packing::F16,
                _ => xabe_translate::Packing::Packed,
            };
            let m = match xabe_translate::Translator::open_with(&args.model, args.device, pack) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("open: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let (mut pres, mut decs) = (Vec::new(), Vec::new());
            for r in 0..args.rounds + 1 {
                let mut cache = m.cache();
                let t0 = Instant::now();
                if m.forward_last(&ids, &mut cache).is_err() {
                    eprintln!("prefill failed");
                    return ExitCode::FAILURE;
                }
                m.gpu().synchronize().ok();
                let pre = t0.elapsed().as_secs_f64() * 1e3;

                let t0 = Instant::now();
                for i in 0..args.decode {
                    if m.forward_last(&[1500 + i as u32 % 100], &mut cache)
                        .is_err()
                    {
                        eprintln!("decode failed");
                        return ExitCode::FAILURE;
                    }
                }
                m.gpu().synchronize().ok();
                let dec = t0.elapsed().as_secs_f64() * 1e3;
                if r > 0 {
                    pres.push(pre);
                    decs.push(dec);
                }
            }
            println!("translate {} [{}]", args.model.display(), args.packing);
            report(
                "translate",
                args.prompt,
                args.decode,
                median(pres),
                median(decs),
                bytes,
            );
        }
        other => {
            eprintln!("--kind must be chat or translate, not {other}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
