//! The `--<stage>-url` half of the symmetry: stages that live elsewhere.
//!
//! Each client speaks the protocol of the service it replaces, not a protocol
//! of this engine's invention. That is what lets a topology be mixed - an
//! engine process delegating its ASR to `whisper-server` today and to another
//! engine process tomorrow, with nothing above this module changing.
//!
//! Streaming responses are delivered over a channel rather than returned,
//! because the caller has to do something with each piece *while* the next is
//! arriving. See `turn.rs` for why that matters: synthesising inline stops the
//! caller consuming the LLM's stream for the two or three seconds a chunk takes,
//! which freezes the reply mid-sentence.
//!
//! Nothing here retries. A stage that is down should say so on the turn it is
//! needed, not silently double every latency.

use crate::error::ServeError;
use crate::wire::{Completion, CompletionChunk, Transcription, TtsChunk, TtsRequest};
use serde_json::json;
use std::time::Duration;
use tokio::sync::mpsc;

/// How long a single upstream request may take.
///
/// Generous, because a 13 B translator prefill on a busy card genuinely takes
/// tens of seconds, and a timeout that fires mid-turn looks to the user exactly
/// like the assistant ignoring them.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// A base URL with a shared connection pool.
#[derive(Debug, Clone)]
pub struct Upstream {
    base: String,
    http: reqwest::Client,
}

impl Upstream {
    /// Builds a client for a base URL, trimming any trailing slash.
    pub fn new(base: &str) -> Result<Upstream, ServeError> {
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            // Keeping connections alive matters more than it looks: the TTS is
            // called several times per turn, once per clause.
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| ServeError::Client(e.to_string()))?;
        Ok(Upstream {
            base: base.trim_end_matches('/').to_string(),
            http,
        })
    }

    /// The base URL, for logging.
    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// `GET /health`, used to say which stages are reachable at startup.
    pub async fn health(&self) -> bool {
        matches!(
            self.http
                .get(self.url("/health"))
                .timeout(Duration::from_secs(3))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        )
    }

    // ------------------------------------------------------------------- ASR

    /// `POST /inference`, matching `whisper-server`'s multipart form.
    pub async fn transcribe(&self, wav: Vec<u8>, language: &str) -> Result<String, ServeError> {
        let part = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| ServeError::Client(e.to_string()))?;
        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("temperature", "0.0")
            .text("response_format", "json")
            .text("language", language.to_string());

        let resp = self
            .http
            .post(self.url("/inference"))
            .multipart(form)
            .send()
            .await
            .map_err(|e| ServeError::Upstream {
                stage: "asr",
                message: e.to_string(),
            })?;
        let resp = expect_ok("asr", resp).await?;

        // whisper-server answers JSON when asked, but a build configured
        // otherwise answers text/plain, and the Python accepted both. Keeping
        // that means the engine is a drop-in for either.
        let body = resp.text().await.map_err(|e| ServeError::Upstream {
            stage: "asr",
            message: e.to_string(),
        })?;
        match serde_json::from_str::<Transcription>(&body) {
            Ok(t) => Ok(t.text.trim().to_string()),
            Err(_) => Ok(body.trim().to_string()),
        }
    }

    // ------------------------------------------------------------------- LLM

    /// Streams `POST /completion`, sending each piece as it arrives.
    ///
    /// Returns when the stream ends. The receiver closing is not an error: it
    /// means the browser went away, and the right response is to stop asking
    /// the model for more of a reply nobody will hear.
    pub async fn stream_completion(
        &self,
        body: serde_json::Value,
        out: mpsc::Sender<String>,
    ) -> Result<(), ServeError> {
        let resp = self
            .http
            .post(self.url("/completion"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ServeError::Upstream {
                stage: "llm",
                message: e.to_string(),
            })?;
        let mut resp = expect_ok("llm", resp).await?;

        let mut buf = String::new();
        while let Some(bytes) = resp.chunk().await.map_err(|e| ServeError::Upstream {
            stage: "llm",
            message: e.to_string(),
        })? {
            buf.push_str(&String::from_utf8_lossy(&bytes));
            // A chunk boundary can fall anywhere, including mid-line and
            // mid-character. Only whole lines are parsed; the remainder stays
            // in the buffer for the next chunk.
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                let line = line.trim();
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let Ok(chunk) = serde_json::from_str::<CompletionChunk>(payload.trim()) else {
                    continue;
                };
                if !chunk.content.is_empty() && out.send(chunk.content).await.is_err() {
                    return Ok(());
                }
                if chunk.stop {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// One non-streamed `POST /completion`, for the translator.
    pub async fn completion(&self, body: serde_json::Value) -> Result<String, ServeError> {
        let resp = self
            .http
            .post(self.url("/completion"))
            .json(&body)
            .send()
            .await
            .map_err(|e| ServeError::Upstream {
                stage: "translator",
                message: e.to_string(),
            })?;
        let resp = expect_ok("translator", resp).await?;
        let c: Completion = resp.json().await.map_err(|e| ServeError::Upstream {
            stage: "translator",
            message: e.to_string(),
        })?;
        Ok(c.content.trim().to_string())
    }

    // ------------------------------------------------------------------- TTS

    /// Streams `POST /tts_stream`, sending each NDJSON chunk as it arrives.
    pub async fn stream_tts(
        &self,
        text: &str,
        out: mpsc::Sender<TtsChunk>,
    ) -> Result<(), ServeError> {
        let resp = self
            .http
            .post(self.url("/tts_stream"))
            .json(&TtsRequest {
                text: text.to_string(),
                // The upstream is a process dedicated to one engine; naming
                // ours would be naming a registration it does not have.
                engine: None,
            })
            .send()
            .await
            .map_err(|e| ServeError::Upstream {
                stage: "tts",
                message: e.to_string(),
            })?;
        let mut resp = expect_ok("tts", resp).await?;

        let mut buf = String::new();
        while let Some(bytes) = resp.chunk().await.map_err(|e| ServeError::Upstream {
            stage: "tts",
            message: e.to_string(),
        })? {
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<TtsChunk>(line) {
                    Ok(chunk) => {
                        if out.send(chunk).await.is_err() {
                            return Ok(());
                        }
                    }
                    Err(e) => tracing::warn!(%e, "tts stream produced a line that is not a chunk"),
                }
            }
        }
        Ok(())
    }
}

/// Turns a non-2xx response into an error carrying the body.
///
/// The body is the part worth having: llama-server explains a context overflow
/// there, and a bare "500" would send the reader to the wrong service's log.
async fn expect_ok(
    stage: &'static str,
    resp: reqwest::Response,
) -> Result<reqwest::Response, ServeError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    Err(ServeError::Upstream {
        stage,
        message: format!(
            "HTTP {status}: {}",
            body.chars().take(300).collect::<String>()
        ),
    })
}

/// Builds the translator's `[TRANS]` prompt.
///
/// The template, the zero temperature and the stop sequences are all the
/// checkpoint's own convention. `temperature` is 0 because this is a
/// transliteration, not a generation: the same Mandarin must always produce the
/// same Taigi, or the synthesised audio changes between identical turns.
pub fn translate_body(text: &str, target: &str) -> serde_json::Value {
    json!({
        "prompt": format!("[TRANS]\n{text}\n[/TRANS]\n[{target}]\n"),
        "temperature": 0.0,
        "repeat_penalty": 1.1,
        "n_predict": 256,
        "stop": ["[/", "\n["],
    })
}
