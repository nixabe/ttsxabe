//! The Taigi voice engine.
//!
//! One binary for every stage of the pipeline except the chat LLM, which stays
//! in llama.cpp by decision. Which stages *this* process runs is decided
//! entirely by flags: `--<stage>-model` loads it here, `--<stage>-url`
//! delegates it to another process, and the two are indistinguishable to
//! everything downstream.
//!
//! ```sh
//! # everything in one process
//! xabe-engine --serve 127.0.0.1:8000 \
//!             --asr-model models/asr/breeze-asr-26 \
//!             --tts-model models/tts/mms-tts-nan --tts-device 1 \
//!             --llm-url http://127.0.0.1:8082
//!
//! # one stage, one shot, no server
//! xabe-engine --tts-model models/tts/mms-tts-nan --text "lí hó" --out hello.wav
//! ```
//!
//! Preflight runs in a fixed order and every failure names the flag that caused
//! it: resolve the stages, decide what the run does, and only then load
//! anything. That ordering is the point - a flag combination that cannot mean
//! anything costs a millisecond to reject rather than six gigabytes of weights.
//!
//! See `docs/CLI.md` for the flag design and `stage.rs` for the symmetry.
//!
//! The crate is a library with a thin binary on top so that the flag
//! surface can be tested the way it is used - parsed from an argument
//! vector - rather than by spawning a process and reading its stderr.

pub mod action;
pub mod args;
pub mod error;
pub mod serve;
pub mod stage;
pub mod tts;

pub use action::{Action, ActionError};
pub use args::Args;
pub use error::EngineError;
pub use stage::{Device, Kind, Requested, Stage, StageError, Stages};

/// Runs the engine from an already-parsed argument set.
///
/// The preflight order is the design point: resolve the stages, decide what the
/// run does, refuse the stages that are not built yet, and only then load
/// anything. A flag combination that cannot mean anything costs a millisecond
/// to reject rather than six gigabytes of weights.
pub fn run(args: &Args) -> Result<(), EngineError> {
    // 1: which stages this process owns.
    let stages = args.stages()?;

    // 2: what it has been asked to do with them.
    let action = Action::resolve(
        &stages,
        args.serve.as_deref(),
        args.input.as_ref(),
        args.text.as_deref(),
        args.out.as_ref(),
    )?;
    announce(&stages);

    // 3: a stage placed somewhere it cannot run. Reported before any work
    //    starts, so an impossible topology fails at once rather than after the
    //    stages that are possible have already run.
    misplaced(&stages)?;

    // 4: the work.
    match action {
        Action::Serve { addr } => serve::serve(args, &stages, &addr),
        Action::Speak { text, out } => match &stages.tts {
            Stage::Local { path, device } => tts::speak(args, path, *device, &text, &out),
            Stage::Remote { url } => tts::speak_remote(url, &text, &out),
            Stage::Off => unreachable!("Action::Speak is only produced with a TTS stage"),
        },
        Action::Transcribe { input } => match &stages.asr {
            Stage::Remote { url } => tts::transcribe_remote(args, url, &input),
            Stage::Local { path, device } => tts::transcribe(args, path, *device, &input),
            Stage::Off => unreachable!("Action::Transcribe needs an ASR stage"),
        },
        Action::Segment { input } => match &stages.vad {
            Stage::Local { path, .. } => tts::segment(path, &input),
            Stage::Remote { .. } => Err(EngineError::LocalOnly { stage: Kind::Vad }),
            Stage::Off => unreachable!("Action::Segment needs a VAD stage"),
        },
    }
}

/// Logs what this process turned out to be, which is not obvious from a command
/// line that may have come from six environment variables.
fn announce(stages: &Stages) {
    for (kind, stage) in stages.summary() {
        match stage {
            Stage::Local { path, device } => {
                tracing::info!(stage = %kind, %device, path = %path.display(), "local");
            }
            Stage::Remote { url } => tracing::info!(stage = %kind, %url, "delegated"),
            Stage::Off => {}
        }
    }
    if let Some(url) = &stages.llm {
        tracing::info!(stage = "llm", %url, "delegated");
    }
    if stages.full_chain() {
        tracing::info!("this process can answer a whole voice turn");
    }
}

/// Refuses a stage asked to run somewhere it cannot.
///
/// Every stage is satisfied either locally or over HTTP, and the symmetry is
/// the point of the flag surface - but the VAD is the one exception, and it is
/// permanent rather than pending. Fifteen tensors and a millisecond of CPU do
/// not survive a round trip, so `--vad-url` is refused by name instead of
/// being accepted and then costing more than the work it delegates.
fn misplaced(stages: &Stages) -> Result<(), EngineError> {
    for (kind, stage) in stages.summary() {
        match (kind, stage) {
            (Kind::Vad, Stage::Remote { .. }) => {
                return Err(EngineError::LocalOnly { stage: kind });
            }
            _ => continue,
        }
    }
    Ok(())
}
