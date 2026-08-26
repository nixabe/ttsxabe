//! What the engine refuses, and what it has not built yet.

use crate::stage::Kind;
use thiserror::Error;

/// A run that could not proceed, named at the point it stopped.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The stage flags do not describe a valid process.
    #[error(transparent)]
    Stage(#[from] crate::stage::StageError),

    /// The flags describe no work.
    #[error(transparent)]
    Action(#[from] crate::action::ActionError),

    /// A stage was configured that this build cannot run yet.
    ///
    /// This exists so the flag surface can be complete and validated before
    /// the stages behind it are written. A flag that parses and then does
    /// nothing is worse than one that says which milestone it is waiting on.
    #[error("the {stage} stage is not implemented yet (plan phase {phase}); use --{stage}-url")]
    NotImplemented {
        /// Which stage was asked for.
        stage: Kind,
        /// The plan phase that builds it.
        phase: &'static str,
    },

    /// The serving layer failed.
    #[error(transparent)]
    Serve(#[from] xabe_serve::ServeError),

    /// A `--tts-engine` argument was not `name=url`.
    #[error("--tts-engine wants NAME=URL, got `{0}`")]
    BadEngine(String),

    /// An audio file could not be read.
    #[error(transparent)]
    Audio(#[from] xabe_audio::AudioError),

    /// The TTS could not load or could not speak.
    #[error("text-to-speech: {0}")]
    Tts(#[from] xabe_tts::SynthesisError),

    /// Reading or writing a file failed.
    #[error("{what} {path}: {source}")]
    Io {
        /// What was being attempted.
        what: &'static str,
        /// The path involved.
        path: String,
        /// The underlying failure.
        source: std::io::Error,
    },
}
