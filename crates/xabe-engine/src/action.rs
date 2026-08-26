//! What a run actually does, decided before anything expensive is loaded.
//!
//! There are only three shapes: serve HTTP, run one stage over one input, or
//! fail because the flags do not describe either. Deciding which *first* is
//! what lets the engine report "--text needs a TTS stage" in a millisecond
//! rather than after six gigabytes of weights have been read.
//!
//! This module refuses to do the work; it only names it. Start at
//! [`Action::resolve`].

use crate::stage::Stages;
use std::path::PathBuf;

/// The one thing this run will do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Serve HTTP on an address, exposing whatever stages this process owns.
    Serve {
        /// The listen address, as given.
        addr: String,
    },
    /// Synthesise text to a WAV.
    Speak {
        /// The text, or `-` for stdin.
        text: String,
        /// Where the WAV goes, or `-` for stdout.
        out: PathBuf,
    },
    /// Transcribe audio to text on stdout.
    Transcribe {
        /// The input WAV, or `-` for stdin.
        input: PathBuf,
    },
    /// Print speech segments found in audio.
    Segment {
        /// The input WAV, or `-` for stdin.
        input: PathBuf,
    },
}

/// A run whose flags do not describe anything to do.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ActionError {
    /// Stages were configured but nothing was asked of them.
    ///
    /// The common shape of this is a serve command with `--serve` forgotten,
    /// which would otherwise load every model and exit successfully.
    #[error("nothing to do: add --serve to listen, or --in/--text for a one-shot run")]
    Nothing,

    /// Both one-shot inputs at once.
    #[error("--in and --text are alternatives; give one, not both")]
    BothInputs,

    /// A one-shot input alongside a server.
    #[error("--serve runs a server; --in and --text are for one-shot runs")]
    ServeWithInput,

    /// Text to speak, with no TTS.
    #[error("--text needs a TTS stage: add --tts-model or --tts-url")]
    TextWithoutTts,

    /// Audio to read, with nothing that reads audio.
    #[error("--in needs an ASR or VAD stage: add --asr-model/--asr-url or --vad-model/--vad-url")]
    InputWithoutReader,

    /// A synthesis with nowhere to put the result.
    #[error("--text needs --out to say where the WAV goes (use - for stdout)")]
    SpeakWithoutOut,
}

impl Action {
    /// Decides what this run does, from the resolved stages and the one-shot flags.
    pub fn resolve(
        stages: &Stages,
        serve: Option<&str>,
        input: Option<&PathBuf>,
        text: Option<&str>,
        out: Option<&PathBuf>,
    ) -> Result<Action, ActionError> {
        if input.is_some() && text.is_some() {
            return Err(ActionError::BothInputs);
        }
        if serve.is_some() && (input.is_some() || text.is_some()) {
            return Err(ActionError::ServeWithInput);
        }
        if let Some(addr) = serve {
            return Ok(Action::Serve {
                addr: addr.to_string(),
            });
        }
        if let Some(text) = text {
            if !stages.tts.is_on() {
                return Err(ActionError::TextWithoutTts);
            }
            let out = out.ok_or(ActionError::SpeakWithoutOut)?;
            return Ok(Action::Speak {
                text: text.to_string(),
                out: out.clone(),
            });
        }
        if let Some(input) = input {
            // ASR wins when both are on, because that is the pipeline's use:
            // the VAD is a gate in front of it, not a separate answer.
            if stages.asr.is_on() {
                return Ok(Action::Transcribe {
                    input: input.clone(),
                });
            }
            if stages.vad.is_on() {
                return Ok(Action::Segment {
                    input: input.clone(),
                });
            }
            return Err(ActionError::InputWithoutReader);
        }
        Err(ActionError::Nothing)
    }
}
