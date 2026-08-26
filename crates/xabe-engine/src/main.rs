//! The `xabe-engine` binary.
//!
//! Everything is in the library beside this file; `main` exists only to parse
//! the arguments, install logging, and turn an error into an exit code. See
//! `lib.rs` for what the engine is.

use clap::Parser;
use std::process::ExitCode;
use xabe_engine::{Args, run};

fn main() -> ExitCode {
    let args = Args::parse();

    // INFO/DEBUG/TRACE to stdout, WARN/ERROR to stderr, so `--out -` stays
    // pipeable even while the tool is talking.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&args.log_level)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stdout)
        .without_time()
        .with_target(false)
        .init();

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}
