//! The serving layer: both halves of the engine's stage symmetry.
//!
//! `--<stage>-url` is [`client`], `--serve` is [`server`], and neither knows
//! which of the two the other end is. That is what lets the current
//! seven-process topology and a single monolith be the same code.
//!
//! What this crate refuses: model internals. Local synthesis reaches it as a
//! channel of [`SynthesisJob`]s, not as a model, so it can be wired to a GPU by
//! `xabe-engine` without this crate learning what a VITS is.
//!
//! Start at [`server::router`] for what is published, [`turn::run`] for the
//! conversation, and [`text`] for the behaviour that was tuned against real
//! speech and must not drift.

pub mod client;
pub mod config;
pub mod error;
pub mod server;
pub mod state;
pub mod text;
pub mod turn;
pub mod turntaking;
pub mod wire;

pub use client::{Upstream, translate_body};
pub use config::{GatewayConfig, LOCAL_ENGINE, Role, direct_taigi_prompt, mandarin_prompt};
pub use error::ServeError;
pub use state::{AppState, Inner, LocalTts, SynthesisJob, TtsBackend};
pub use text::{Chunker, clean, normalize_for_mms, sanitize_asr, split_poj, split_sentences};
pub use turntaking::{Decision, Endpointer, TurnPolicy};
pub use wire::{ClientMessage, PageConfig, ServerMessage, TtsChunk, TtsRequest};

/// The page this server serves, compiled in.
///
/// Held as a `&'static str` rather than read from disk so a deployed binary has
/// no runtime dependency on a checkout, and so the page and the protocol it
/// speaks cannot drift out of the same commit.
pub const PAGE: &str = include_str!("../static/index.html");

/// Binds and serves until the process is asked to stop.
pub async fn serve(addr: &str, state: AppState) -> Result<(), ServeError> {
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| ServeError::Bind {
            addr: addr.to_string(),
            source,
        })?;
    let local = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| addr.to_string());
    tracing::info!(addr = %local, converses = state.can_converse(), "serving");

    axum::serve(listener, server::router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("stopping");
        })
        .await
        .map_err(ServeError::Serve)
}
