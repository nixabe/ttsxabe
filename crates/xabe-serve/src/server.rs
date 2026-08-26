//! The `--serve` half of the symmetry: stages this process offers to others.
//!
//! The endpoints are the ones the Python services already publish, not new
//! ones. That is what makes the migration incremental rather than a flag day:
//! an engine process is a drop-in for `whisper-server`, for the TTS daemon, or
//! for the gateway, so each stage can be swapped in behind the existing system
//! and A/B'd against the service it replaces, one at a time.
//!
//! | route | replaces |
//! | --- | --- |
//! | `POST /inference` | `whisper-server` |
//! | `POST /tts` and `POST /tts_stream` | `taigi_tts_daemon.py` |
//! | `GET /`, `GET /api/config`, `WS /ws` | `gateway.py` |
//! | `GET /health` | all three |
//!
//! A process only publishes the routes for stages it owns. Asking a TTS worker
//! for `/inference` is a 503 naming the flag that would give it one, rather
//! than a 404 that looks like a typo.

use crate::state::{AppState, SynthesisJob, TtsBackend};
use crate::wire::{PageConfig, Transcription, TtsChunk, TtsRequest};
use axum::Router;
use axum::body::Body;
use axum::extract::{Multipart, State, WebSocketUpgrade};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use tokio::sync::mpsc;

/// Builds the router for whatever stages this process owns.
pub fn router(state: AppState) -> Router {
    let mut app = Router::new().route("/health", get(health));

    if state.asr.is_some() {
        app = app.route("/inference", post(inference));
    }
    if !state.tts.is_empty() {
        app = app
            .route("/tts", post(tts))
            .route("/tts_stream", post(tts_stream));
    }
    // The page is only worth serving if this process can actually hold a
    // conversation. A TTS worker that answered `GET /` with a chat UI that
    // cannot hear anything would be a worse failure than a 404.
    if state.can_converse() {
        app = app
            .route("/", get(index))
            .route("/api/config", get(config))
            .route("/ws", any(websocket));
    }

    app.with_state(state)
}

async fn health(State(state): State<AppState>) -> Response {
    let stages: Vec<&str> = [
        state.asr.is_some().then_some("asr"),
        state.vad.is_some().then_some("vad"),
        state.llm.is_some().then_some("llm"),
        state.translator.is_some().then_some("translator"),
        (!state.tts.is_empty()).then_some("tts"),
    ]
    .into_iter()
    .flatten()
    .collect();

    axum::Json(serde_json::json!({
        "status": "ok",
        "stages": stages,
        "engines": state.tts.keys().collect::<Vec<_>>(),
        "converses": state.can_converse(),
    }))
    .into_response()
}

async fn index(State(state): State<AppState>) -> Html<&'static str> {
    Html(state.page)
}

async fn config(State(state): State<AppState>) -> axum::Json<PageConfig> {
    axum::Json(PageConfig {
        engines: state.tts.keys().cloned().collect(),
        default: state
            .tts
            .contains_key(&state.config.tts_default)
            .then(|| state.config.tts_default.clone())
            .or_else(|| state.tts.keys().next().cloned())
            .unwrap_or_default(),
        turn: state.config.turn,
    })
}

async fn websocket(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| crate::turn::run(state, socket))
}

/// `POST /inference`, matching `whisper-server`'s multipart form.
async fn inference(State(state): State<AppState>, mut form: Multipart) -> Response {
    let Some(asr) = state.asr.clone() else {
        return no_stage("asr", "--asr-model or --asr-url");
    };

    let mut wav: Option<Vec<u8>> = None;
    let mut language = state.config.asr_lang.clone();
    while let Ok(Some(field)) = form.next_field().await {
        match field.name().unwrap_or_default() {
            "file" => wav = field.bytes().await.ok().map(|b| b.to_vec()),
            "language" => {
                if let Ok(v) = field.text().await
                    && !v.is_empty()
                {
                    language = v;
                }
            }
            // `temperature` and `response_format` are accepted and ignored: the
            // pipeline sends them, and rejecting a field a caller sends is a
            // worse compatibility break than not acting on it.
            _ => {}
        }
    }

    let Some(wav) = wav else {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "no `file` part in the form"})),
        )
            .into_response();
    };

    match asr.transcribe(wav, &language).await {
        Ok(text) => axum::Json(Transcription { text }).into_response(),
        Err(e) => upstream_error(e),
    }
}

/// `POST /tts`, returning one WAV for the whole utterance.
async fn tts(State(state): State<AppState>, axum::Json(req): axum::Json<TtsRequest>) -> Response {
    let Some(backend) = state.tts_for(None).cloned() else {
        return no_stage("tts", "--tts-model or --tts-url");
    };

    let mut chunks = Vec::new();
    if let Err(e) = collect_chunks(&backend, &req.text, &mut chunks).await {
        return upstream_error(e);
    }

    // Concatenating WAVs is not concatenating files: each chunk carries its own
    // 44-byte header, so the samples have to be unwrapped and rewrapped once.
    let mut samples: Vec<f32> = Vec::new();
    let mut rate = 16_000;
    for chunk in &chunks {
        let Ok(bytes) = B64.decode(&chunk.wav) else {
            continue;
        };
        if let Ok(wav) = xabe_audio::parse_wav(&bytes) {
            rate = wav.sample_rate;
            samples.extend_from_slice(&wav.samples);
        }
    }
    if samples.is_empty() {
        // 204 rather than 200-with-silence, which is what the Python did and
        // what the pipeline's callers already handle.
        return (StatusCode::NO_CONTENT, [("X-Info", "{}")]).into_response();
    }

    let info = serde_json::json!({
        "taigi": chunks.first().map(|c| c.taigi.clone()).unwrap_or_default(),
        "chunks": chunks.len(),
    })
    .to_string();
    (
        [
            (header::CONTENT_TYPE, "audio/wav".to_string()),
            // Headers are latin-1, so the JSON stays \uXXXX-escaped and Han
            // survives the trip.
            (
                header::HeaderName::from_static("x-info"),
                info.chars().take(900).collect(),
            ),
        ],
        xabe_audio::wav_bytes(&samples, rate),
    )
        .into_response()
}

/// `POST /tts_stream`, NDJSON of chunks as they are produced.
async fn tts_stream(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<TtsRequest>,
) -> Response {
    let Some(backend) = state.tts_for(None).cloned() else {
        return no_stage("tts", "--tts-model or --tts-url");
    };

    let (tx, mut rx) = mpsc::channel::<TtsChunk>(4);
    tokio::spawn(async move {
        if let Err(e) = stream_chunks(&backend, &req.text, tx).await {
            tracing::warn!(%e, "tts stream failed");
        }
    });

    let body = async_stream::stream! {
        let mut seq = 0u64;
        while let Some(mut chunk) = rx.recv().await {
            seq += 1;
            // Renumbered here rather than trusted from upstream: a chunk that
            // passed through a second engine carries that engine's numbering.
            chunk.seq = seq;
            match serde_json::to_string(&chunk) {
                Ok(mut line) => {
                    line.push('\n');
                    yield Ok::<_, std::io::Error>(line.into_bytes());
                }
                Err(e) => tracing::warn!(%e, "could not serialise a tts chunk"),
            }
        }
    };

    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(body),
    )
        .into_response()
}

/// Runs a synthesis and collects every chunk.
async fn collect_chunks(
    backend: &TtsBackend,
    text: &str,
    out: &mut Vec<TtsChunk>,
) -> Result<(), crate::error::ServeError> {
    let (tx, mut rx) = mpsc::channel::<TtsChunk>(4);
    let backend = backend.clone();
    let text = text.to_string();
    let task = tokio::spawn(async move { stream_chunks(&backend, &text, tx).await });
    while let Some(chunk) = rx.recv().await {
        out.push(chunk);
    }
    match task.await {
        Ok(r) => r,
        Err(e) => Err(crate::error::ServeError::Upstream {
            stage: "tts",
            message: e.to_string(),
        }),
    }
}

/// Sends a synthesis to whichever backend owns it.
async fn stream_chunks(
    backend: &TtsBackend,
    text: &str,
    out: mpsc::Sender<TtsChunk>,
) -> Result<(), crate::error::ServeError> {
    match backend {
        TtsBackend::Remote(up) => up.stream_tts(text, out).await,
        TtsBackend::Local(jobs) => jobs
            .send(SynthesisJob {
                text: text.to_string(),
                reply: out,
            })
            .await
            .map_err(|_| crate::error::ServeError::NoStage("tts")),
    }
}

/// 503 naming the flag that would provide the stage.
fn no_stage(stage: &'static str, flags: &'static str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::Json(serde_json::json!({
            "error": format!("this process has no {stage} stage; start it with {flags}"),
        })),
    )
        .into_response()
}

/// 502 with the upstream's own words.
fn upstream_error(e: crate::error::ServeError) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        axum::Json(serde_json::json!({"error": e.to_string()})),
    )
        .into_response()
}
