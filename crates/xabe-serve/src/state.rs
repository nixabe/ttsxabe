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
    pub asr: Option<Upstream>,
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
}
