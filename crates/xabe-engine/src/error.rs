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

    /// The voice-activity checkpoint could not be loaded.
    #[error(transparent)]
    Vad(#[from] xabe_vad::VadError),

    /// Audio was handed to a stage at a rate it cannot process.
    #[error("{path} is {rate} Hz; this stage needs {want} Hz")]
    WrongRate {
        /// The file involved.
        path: String,
        /// What it is.
        rate: u32,
        /// What was needed.
        want: u32,
    },

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
    /// A clip at a rate this engine's own ASR does not resample from.
    ///
    /// A resampler good enough for an ASR is a real piece of work, and one
    /// that is not good enough is a transcript that is quietly worse. So the
    /// rate is a requirement rather than something silently fixed.
    #[error("the clip is {found} Hz; this stage wants {wanted}")]
    SampleRate {
        /// What the file declares.
        found: u32,
        /// What the stage needs.
        wanted: u32,
    },

    /// The ASR stage failed.
    #[error(transparent)]
    Asr(#[from] xabe_asr::AsrError),
}
