//! What one server process owns.
//!
//! The stages reachable from here are a *set*, not a fixed chain: a process
//! configured with only a TTS is a TTS worker, one with everything is the whole
//! assistant, and the code below is the same either way. Whether a stage is
//! local or delegated is settled here and nowhere else.
//!
//! Local synthesis arrives as a channel rather than as a model. That is
//! deliberate - this crate owns HTTP and refuses to know what a VITS is, and a
//! channel is the narrowest boundary that carries "text in, audio out" without
//! a trait. `xabe-engine` is what puts a GPU on the other end of it.

use crate::client::Upstream;
use crate::config::GatewayConfig;
use crate::wire::TtsChunk;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::mpsc;

/// One request to synthesise, and where its audio should go.
#[derive(Debug)]
pub struct SynthesisJob {
    /// The text to speak.
    pub text: String,
    /// Chunks, as they are produced.
    pub reply: mpsc::Sender<TtsChunk>,
}

/// The sending half of a local synthesiser's work queue.
pub type LocalTts = mpsc::Sender<SynthesisJob>;

/// A span of speech within a clip, in samples.
///
/// Defined here rather than reused from `xabe-vad` for the same reason the TTS
/// arrives as a channel: this crate owns HTTP and refuses to know which model
/// produced the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSpan {
    /// First sample.
    pub start: usize,
    /// One past the last sample.
    pub end: usize,
}

/// One request to find speech, and where the answer should go.
#[derive(Debug)]
pub struct VadJob {
    /// Mono 16 kHz samples.
    pub samples: Vec<f32>,
    /// The spans found, or an empty vector if there is no speech.
    pub reply: tokio::sync::oneshot::Sender<Vec<SpeechSpan>>,
}

/// The sending half of a local detector's work queue.
pub type LocalVad = mpsc::Sender<VadJob>;

/// One request to transcribe, and where the text should go.
///
/// The request carries a WAV rather than samples, so that a local stage and a
/// delegated one are handed exactly the same thing. The alternative - samples
/// for one, a container for the other - is two code paths that agree until the
/// day one of them is given a clip the other would have rejected.
#[derive(Debug)]
pub struct TranscribeJob {
    /// A 16 kHz mono WAV.
    pub wav: Vec<u8>,
    /// The language to force, as a Whisper language code.
    pub language: String,
    /// The transcript, or a message describing why there is none.
    pub reply: tokio::sync::oneshot::Sender<Result<String, String>>,
}

/// The sending half of a local transcriber's work queue.
pub type LocalAsr = mpsc::Sender<TranscribeJob>;

/// Where speech-to-text lives.
#[derive(Clone)]
pub enum AsrBackend {
    /// In this process, behind a work queue.
    Local(LocalAsr),
    /// In another process.
    Remote(Upstream),
}

impl std::fmt::Debug for AsrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrBackend::Local(_) => f.write_str("local"),
            AsrBackend::Remote(u) => write!(f, "remote({})", u.base()),
        }
    }
}

impl AsrBackend {
    /// Transcribes a WAV, wherever the model happens to be.
    pub async fn transcribe(
        &self,
        wav: Vec<u8>,
        language: &str,
    ) -> Result<String, crate::ServeError> {
        match self {
            AsrBackend::Remote(u) => u.transcribe(wav, language).await,
            AsrBackend::Local(tx) => {
                let (reply, done) = tokio::sync::oneshot::channel();
                tx.send(TranscribeJob {
                    wav,
                    language: language.to_string(),
                    reply,
                })
                .await
                .map_err(|_| crate::ServeError::Upstream {
                    stage: "asr",
                    message: "the local transcriber stopped".into(),
                })?;
                done.await
                    .map_err(|_| crate::ServeError::Upstream {
                        stage: "asr",
                        message: "the local transcriber dropped the job".into(),
                    })?
                    .map_err(|message| crate::ServeError::Upstream {
                        stage: "asr",
                        message,
                    })
            }
        }
    }
}

/// Where a TTS engine lives.
#[derive(Clone)]
pub enum TtsBackend {
    /// In this process, behind a work queue.
    Local(LocalTts),
    /// In another process.
    Remote(Upstream),
}

impl std::fmt::Debug for TtsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtsBackend::Local(_) => f.write_str("local"),
            TtsBackend::Remote(u) => write!(f, "remote({})", u.base()),
        }
    }
}

/// Everything a request handler can reach.
#[derive(Clone)]
pub struct AppState(pub Arc<Inner>);

/// The owned half of [`AppState`].
pub struct Inner {
    /// Prompt, thresholds and engine names.
    pub config: GatewayConfig,
    /// Speech to text, if this process can reach one.
    pub asr: Option<AsrBackend>,
    /// Voice activity detection, if this process runs one.
    ///
    /// Always local: the VAD is 15 tensors and a millisecond of CPU, so
    /// delegating it over HTTP would cost more than running it.
    pub vad: Option<LocalVad>,
    /// The chat model, which is always another process.
    pub llm: Option<Upstream>,
    /// Mandarin to Taigi, if configured.
    pub translator: Option<Upstream>,
    /// Which target the translator is asked for: `POJ`, `HAN` or `HL`.
    pub translator_target: String,
    /// Synthesisers by engine name.
    pub tts: BTreeMap<String, TtsBackend>,
    /// The page this server serves, held in memory.
    pub page: &'static str,
}

impl std::ops::Deref for AppState {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.0
    }
}

impl Inner {
    /// Picks the engine the browser asked for, falling back to the default.
    ///
    /// A page that asks for an engine this process does not have gets the
    /// default rather than an error: the alternative is a silent turn, and the
    /// user cannot act on "engine cosyvoice not configured" anyway.
    pub fn tts_for(&self, requested: Option<&str>) -> Option<&TtsBackend> {
        requested
            .and_then(|name| self.tts.get(name))
            .or_else(|| self.tts.get(&self.config.tts_default))
            .or_else(|| self.tts.values().next())
    }

    /// Whether this process can answer a whole voice turn.
    pub fn can_converse(&self) -> bool {
        self.asr.is_some() && self.llm.is_some() && !self.tts.is_empty()
    }

    /// Finds speech, or returns the whole clip when there is no detector.
    ///
    /// Returning the whole clip rather than nothing is what keeps the VAD
    /// optional: a process without one behaves exactly as it did before the
    /// stage existed.
    pub async fn speech_in(&self, samples: Vec<f32>) -> Vec<SpeechSpan> {
        let whole = vec![SpeechSpan {
            start: 0,
            end: samples.len(),
        }];
        let Some(vad) = &self.vad else {
            return whole;
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vad.send(VadJob { samples, reply: tx }).await.is_err() {
            tracing::warn!("the vad worker is gone; treating the clip as speech");
            return whole;
        }
        match rx.await {
            Ok(spans) => spans,
            Err(_) => {
                tracing::warn!("the vad worker dropped a job; treating the clip as speech");
                whole
            }
        }
    }
}
