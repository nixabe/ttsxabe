//! The message shapes on both sides of this crate.
//!
//! Two protocols meet here and neither was invented by this engine:
//!
//! - **Downstream**, the browser's WebSocket protocol, which the shipped page
//!   already speaks.
//! - **Upstream**, the HTTP surfaces of `whisper-server`, `llama-server` and
//!   the Python TTS daemon, which the engine must be a drop-in for.
//!
//! Being wire-compatible with both is what makes the migration incremental:
//! each stage can be swapped in behind the existing gateway and A/B'd against
//! the service it replaces, one at a time.
//!
//! This module refuses to do anything but describe shapes.

use serde::{Deserialize, Serialize};

// ------------------------------------------------------------ browser -> here

/// What the page sends.
///
/// `#[serde(other)]` on [`ClientMessage::Unknown`] is load-bearing. The Python
/// this replaces indexed `msg["type"]` bare, so one malformed frame - or one
/// frame from a newer page - tore down the whole connection and the user saw
/// 連線中斷. An unknown frame is now a message the server declines, not a
/// reason to drop the conversation.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Transcribe this audio now, during the VAD grace window.
    AsrPrefetch {
        /// Client-assigned id, echoed back by `audio` or `asr_cancel`.
        id: u64,
        /// Base64 little-endian 16-bit mono PCM.
        pcm: String,
        /// Sample rate of that PCM.
        #[serde(default = "default_rate")]
        rate: u32,
    },
    /// The speaker resumed; drop the prefetch.
    AsrCancel {
        /// The id given to `asr_prefetch`.
        id: u64,
    },
    /// A finished turn.
    Audio {
        /// The prefetch to collect, if one was started.
        id: Option<u64>,
        /// Base64 little-endian 16-bit mono PCM.
        pcm: String,
        /// Sample rate of that PCM.
        #[serde(default = "default_rate")]
        rate: u32,
        /// Which TTS engine to answer with.
        engine: Option<String>,
    },
    /// A typed turn.
    Text {
        /// What was typed.
        content: String,
        /// Which TTS engine to answer with.
        engine: Option<String>,
    },
    /// Forget the conversation.
    Reset,
    /// Anything else, so one bad frame does not end the connection.
    #[serde(other)]
    Unknown,
}

fn default_rate() -> u32 {
    16_000
}

// ------------------------------------------------------------ here -> browser

/// What the page receives.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// The user's turn, as transcribed.
    Asr {
        /// The cleaned transcript.
        text: String,
        /// How long transcription took, in milliseconds.
        ms: u64,
    },
    /// A transcript that the hallucination guard rejected.
    Ignored {
        /// What was heard, so the user can see why it was dropped.
        raw: String,
    },
    /// One piece of the reply, as the model writes it.
    Token {
        /// The piece.
        text: String,
    },
    /// One synthesised chunk.
    Audio {
        /// Playback order. Monotonic for the life of the connection.
        seq: u64,
        /// Base64 WAV.
        wav: String,
        /// The Taigi Han the synthesiser was given, if it was translated.
        taigi: String,
        /// The romanisation, when the engine speaks POJ.
        roman: String,
        /// Milliseconds from the start of the turn to the first audio.
        first_audio_ms: Option<u64>,
    },
    /// The turn is over.
    Done {
        /// The whole reply.
        #[serde(skip_serializing_if = "Option::is_none")]
        reply: Option<String>,
        /// Milliseconds for the whole turn.
        #[serde(skip_serializing_if = "Option::is_none")]
        total_ms: Option<u64>,
        /// Milliseconds to first audio.
        #[serde(skip_serializing_if = "Option::is_none")]
        first_audio_ms: Option<u64>,
        /// Why the turn ended without a reply.
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Something failed, without ending the conversation.
    Error {
        /// What failed.
        message: String,
    },
    /// The conversation was cleared.
    ResetOk,
}

// ------------------------------------------------------------------- upstream

/// `whisper-server`'s `POST /inference` response.
#[derive(Debug, Deserialize, Serialize)]
pub struct Transcription {
    /// The transcript, with whatever whitespace the server produced.
    pub text: String,
}

/// One line of the TTS daemon's `POST /tts_stream` NDJSON.
#[derive(Debug, Deserialize, Serialize)]
pub struct TtsChunk {
    /// Order within this utterance. Restarts at 1 per request.
    pub seq: u64,
    /// Base64 WAV, self-contained so the browser decodes it like any other.
    pub wav: String,
    /// The Taigi Han that was synthesised, if a translator ran.
    #[serde(default)]
    pub taigi: String,
    /// The romanisation, when the engine speaks POJ.
    #[serde(default)]
    pub roman: String,
}

/// The request body both TTS endpoints take.
#[derive(Debug, Serialize, Deserialize)]
pub struct TtsRequest {
    /// The text to speak.
    pub text: String,
    /// Which registered engine should speak it.
    ///
    /// Absent means the configured default. The WebSocket path has carried an
    /// engine per utterance since the page grew a selector; these two
    /// endpoints did not, so a request that asked for CosyVoice was answered
    /// by whatever the default was - audibly a different voice, with nothing
    /// in the response to say so.
    #[serde(default)]
    pub engine: Option<String>,
}

/// One `data:` line of `llama-server`'s streamed `POST /completion`.
#[derive(Debug, Deserialize)]
pub struct CompletionChunk {
    /// The new text, which may be empty.
    #[serde(default)]
    pub content: String,
    /// Whether this is the last chunk.
    #[serde(default)]
    pub stop: bool,
}

/// A non-streamed `POST /completion` response.
#[derive(Debug, Deserialize)]
pub struct Completion {
    /// The whole generated text.
    #[serde(default)]
    pub content: String,
}

/// What `GET /api/config` tells the page.
#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    /// TTS engines this gateway can reach.
    pub engines: Vec<String>,
    /// Which one the page should select on load.
    pub default: String,
    /// The turn-taking constants, so the page does not carry its own copy.
    pub turn: crate::turntaking::TurnPolicy,
}
