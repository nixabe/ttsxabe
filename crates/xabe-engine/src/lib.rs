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

    // 3: the stages that are configured but not yet built. Reported before any
    //    work starts, so a half-built configuration fails at once rather than
    //    after the one stage that does exist has already run.
    unbuilt(&stages)?;

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
            Stage::Remote { .. } => Err(EngineError::NotImplemented {
                stage: Kind::Vad,
                phase: "3",
            }),
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

/// Refuses stages whose implementation is still ahead in the plan.
///
/// The flag surface is complete before the stages behind it are, so that the
/// topology can be designed and tested first. A flag that parses and then
/// silently does nothing would be worse than one that says which phase it is
/// waiting on.
fn unbuilt(stages: &Stages) -> Result<(), EngineError> {
    for (kind, stage) in stages.summary() {
        match stage {
            // A delegated stage needs no implementation here beyond an HTTP
            // client, and that exists - except for the VAD, whose wire protocol
            // arrives with the model that defines it.
            Stage::Remote { .. } => match kind {
                Kind::Vad => {
                    return Err(EngineError::NotImplemented {
                        stage: kind,
                        phase: "3",
                    });
                }
                _ => continue,
            },
            Stage::Local { .. } => match kind {
                Kind::Tts => {}
                Kind::Vad => {}
                Kind::Asr => {}
                Kind::Translator => {}
            },
            Stage::Off => {}
        }
    }
    Ok(())
}
