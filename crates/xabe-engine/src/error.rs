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

    /// A stage was asked to run over HTTP that only ever runs in process.
    ///
    /// Not a milestone waiting to be reached: the VAD is fifteen tensors and a
    /// millisecond of CPU, so a round trip would cost more than the work it
    /// delegates. Naming it as unbuilt would invite someone to wait for it.
    #[error("--{stage}-url: the {stage} stage only runs in process; give --{stage}-model")]
    LocalOnly {
        /// Which stage was asked for.
        stage: Kind,
    },

    /// A `--tts-script` argument was not `name=script`.
    #[error("--tts-script wants NAME=SCRIPT, got `{0}`")]
    BadScript(String),

    /// A `--tts-script` named an engine this process does not have.
    ///
    /// Refused rather than ignored: a script set on a misspelled engine is a
    /// silent turn later, since a synthesiser handed the wrong script says
    /// nothing at all rather than failing.
    #[error("--tts-script names `{name}`, which is not a registered engine; have: {known}")]
    UnknownEngine {
        /// What was asked for.
        name: String,
        /// What exists.
        known: String,
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

    /// The translator stage failed.
    #[error(transparent)]
    Translate(#[from] xabe_translate::TranslateError),

    /// The chat model could not be loaded or run.
    #[error(transparent)]
    Chat(#[from] xabe_chat::ChatError),
}
