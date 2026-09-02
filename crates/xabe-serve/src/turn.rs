//! One browser conversation.
//!
//! This is a behaviour-for-behaviour port of the Python gateway, and the
//! behaviours are the point. Three of them cost real latency or real failures
//! to discover, and none is recoverable from reading the code:
//!
//! **Synthesis runs off the LLM stream.** A chunk takes two or three seconds.
//! Synthesising inline stops the reply being consumed for that long, which
//! freezes the text mid-sentence and can stall the upstream connection. So
//! chunks go into a queue and one worker drains it, which also keeps audio
//! strictly ordered.
//!
//! **Transcription starts when the pause *arms*, not when the turn commits.**
//! The browser arms end-of-turn after `SILENCE_MS` and commits `GRACE_MS`
//! later; transcribing in that gap hides the ASR latency entirely when the
//! speaker does not resume, which is the common case. If they do resume, the
//! prefetch is cancelled.
//!
//! **The first chunk breaks at a clause, later ones at a sentence.** A Taigi
//! reply is often one long sentence, so waiting for 。 means waiting for all of
//! it. Measured 4.1 s → 2.7 s to first audio. See [`crate::text::Chunker`].
//!
//! The socket has exactly one owner: everything that writes to the browser goes
//! through the loop in [`run`], fed by channels. Splitting it would let a
//! synthesis error and a token race for the same frame.

use crate::config::Role;
use crate::error::ServeError;
use crate::state::{AppState, AsrBackend, SynthesisJob, TranslatorBackend, TtsBackend};
use crate::text::{Chunker, sanitize_asr};
use crate::wire::{ClientMessage, ServerMessage, TtsChunk};
use axum::extract::ws::{Message, WebSocket};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;

/// How many synthesised chunks may be buffered before the worker waits.
///
/// Small on purpose. A large buffer would let synthesis run far ahead of
/// playback, which wastes GPU on a reply the user is about to interrupt.
const AUDIO_BUFFER: usize = 4;

/// Runs one WebSocket conversation to its end.
pub async fn run(state: AppState, mut socket: WebSocket) {
    let mut history: Vec<(Role, String)> = Vec::new();
    let mut prefetch: HashMap<u64, JoinHandle<Result<String, ServeError>>> = HashMap::new();
    // Monotonic for the life of the connection, not per turn.
    //
    // The Python restarted `seq` at 1 for every turn while the browser sorted
    // one global queue, so a new turn's first chunk could sort ahead of the
    // previous turn's last and play out of order. Fixing it here rather than in
    // the page also fixes every other client.
    let mut seq: u64 = 0;

    while let Some(Ok(frame)) = socket.recv().await {
        let Message::Text(raw) = frame else {
            // Binary, ping and close frames are not part of this protocol.
            // axum answers pings itself; a close ends the loop below.
            if matches!(frame, Message::Close(_)) {
                break;
            }
            continue;
        };

        let msg: ClientMessage = match serde_json::from_str(&raw) {
            Ok(m) => m,
            Err(e) => {
                // One malformed frame is not a reason to end the conversation.
                // The Python indexed `msg["type"]` bare and tore down the whole
                // handler, which the user saw as 連線中斷.
                let _ = send(
                    &mut socket,
                    ServerMessage::Error {
                        message: format!("could not read that frame: {e}"),
                    },
                )
                .await;
                continue;
            }
        };

        let engine = engine_of(&msg);
        let user_text = match handle(&state, &mut socket, &mut prefetch, msg).await {
            Turn::Continue => continue,
            Turn::Say(text) => text,
        };

        if user_text.is_empty() {
            let _ = send(
                &mut socket,
                ServerMessage::Done {
                    reply: None,
                    total_ms: None,
                    first_audio_ms: None,
                    reason: Some("empty".into()),
                },
            )
            .await;
            continue;
        }

        if let Err(e) = reply(
            &state,
            &mut socket,
            &mut history,
            &mut seq,
            &user_text,
            engine,
        )
        .await
        {
            let _ = send(
                &mut socket,
                ServerMessage::Error {
                    message: e.to_string(),
                },
            )
            .await;
        }
    }

    // A conversation that ends with transcriptions in flight would leave them
    // running against a GPU nobody is waiting on.
    for (_, task) in prefetch.drain() {
        task.abort();
    }
}

/// What a frame did.
enum Turn {
    /// Nothing further; read the next frame.
    Continue,
    /// The user's turn, ready to answer.
    Say(String),
}

/// Which TTS engine a frame asked for.
fn engine_of(msg: &ClientMessage) -> Option<String> {
    match msg {
        ClientMessage::Audio { engine, .. } | ClientMessage::Text { engine, .. } => engine.clone(),
        _ => None,
    }
}

/// Handles one frame, up to the point where a reply is owed.
async fn handle(
    state: &AppState,
    socket: &mut WebSocket,
    prefetch: &mut HashMap<u64, JoinHandle<Result<String, ServeError>>>,
    msg: ClientMessage,
) -> Turn {
    match msg {
        ClientMessage::AsrPrefetch { id, pcm, rate } => {
            let Some(asr) = state.asr.clone() else {
                // No ASR here. Not an error worth showing: the turn that
                // follows will say so, once, rather than on every pause.
                return Turn::Continue;
            };
            let lang = state.config.asr_lang.clone();
            let state = state.clone();
            prefetch.insert(
                id,
                tokio::spawn(
                    async move { gate_then_transcribe(&state, &asr, &pcm, rate, &lang).await },
                ),
            );
            Turn::Continue
        }

        ClientMessage::AsrCancel { id } => {
            if let Some(task) = prefetch.remove(&id) {
                task.abort();
            }
            Turn::Continue
        }

        ClientMessage::Audio { id, pcm, rate, .. } => {
            let t0 = Instant::now();
            let raw = match collect(state, prefetch, id, &pcm, rate).await {
                Ok(text) => text,
                Err(e) => {
                    let _ = send(
                        socket,
                        ServerMessage::Error {
                            message: e.to_string(),
                        },
                    )
                    .await;
                    return Turn::Continue;
                }
            };
            let text = sanitize_asr(&raw);
            if !raw.is_empty() && text.is_empty() {
                let _ = send(socket, ServerMessage::Ignored { raw }).await;
                return Turn::Continue;
            }
            let _ = send(
                socket,
                ServerMessage::Asr {
                    text: text.clone(),
                    ms: t0.elapsed().as_millis() as u64,
                },
            )
            .await;
            Turn::Say(text)
        }

        ClientMessage::Text { content, .. } => {
            let text = content.trim().to_string();
            let _ = send(
                socket,
                ServerMessage::Asr {
                    text: text.clone(),
                    ms: 0,
                },
            )
            .await;
            Turn::Say(text)
        }

        ClientMessage::Reset => {
            let _ = send(socket, ServerMessage::ResetOk).await;
            Turn::Continue
        }

        ClientMessage::Unknown => {
            let _ = send(
                socket,
                ServerMessage::Error {
                    message: "unrecognised frame type".into(),
                },
            )
            .await;
            Turn::Continue
        }
    }
}

/// Collects a prefetched transcript, or transcribes now.
async fn collect(
    state: &AppState,
    prefetch: &mut HashMap<u64, JoinHandle<Result<String, ServeError>>>,
    id: Option<u64>,
    pcm: &str,
    rate: u32,
) -> Result<String, ServeError> {
    if let Some(task) = id.and_then(|id| prefetch.remove(&id)) {
        // Usually already finished, which is the whole point of prefetching.
        return match task.await {
            Ok(result) => result,
            Err(e) if e.is_cancelled() => Ok(String::new()),
            Err(e) => Err(ServeError::Upstream {
                stage: "asr",
                message: format!("prefetch task failed: {e}"),
            }),
        };
    }
    let asr = state.asr.clone().ok_or(ServeError::NoStage("asr"))?;
    gate_then_transcribe(state, &asr, pcm, rate, &state.config.asr_lang).await
}

/// Runs the VAD, then transcribes only if there was speech.
///
/// This is the first of the pipeline's three hallucination layers, and the
/// cheapest: a clip with no speech never reaches the ASR at all, which both
/// prevents the invented sentence and saves the round trip. The clip is also
/// trimmed to the speech it found - leading and trailing silence is exactly
/// what Whisper turns into 謝謝觀看.
///
/// With no VAD stage configured this is a plain transcription, so adding or
/// removing the stage changes latency and hallucination rate, never the shape
/// of the answer.
async fn gate_then_transcribe(
    state: &AppState,
    asr: &AsrBackend,
    pcm_b64: &str,
    rate: u32,
    lang: &str,
) -> Result<String, ServeError> {
    let pcm = B64
        .decode(pcm_b64)
        .map_err(|e| ServeError::BadPcm(e.to_string()))?;
    let wav = xabe_audio::wav_from_pcm16(&pcm, rate);

    if state.vad.is_none() {
        return asr.transcribe(wav.bytes, lang).await;
    }

    let audio = xabe_audio::parse_wav(&wav.bytes).map_err(|e| ServeError::BadPcm(e.to_string()))?;
    let spans = state.speech_in(audio.samples.clone()).await;
    if spans.is_empty() {
        tracing::debug!(ms = wav.millis(), "vad found no speech; not transcribing");
        return Ok(String::new());
    }

    // One span from the first speech to the last, rather than several requests.
    // The gaps between segments are part of the utterance's rhythm, and the ASR
    // reads a pause better than it reads a splice.
    let start = spans.iter().map(|s| s.start).min().unwrap_or(0);
    let end = spans
        .iter()
        .map(|s| s.end)
        .max()
        .unwrap_or(audio.samples.len())
        .min(audio.samples.len());
    let trimmed = &audio.samples[start..end];
    tracing::debug!(
        spans = spans.len(),
        kept_ms = (end - start) * 1000 / rate.max(1) as usize,
        of_ms = wav.millis(),
        "vad gated the clip",
    );
    asr.transcribe(xabe_audio::wav_bytes(trimmed, rate), lang)
        .await
}

/// Streams a reply, synthesising it clause by clause.
async fn reply(
    state: &AppState,
    socket: &mut WebSocket,
    history: &mut Vec<(Role, String)>,
    seq: &mut u64,
    user_text: &str,
    engine: Option<String>,
) -> Result<(), ServeError> {
    let llm = state.llm.clone().ok_or(ServeError::NoStage("llm"))?;
    let backend = state
        .tts_for(engine.as_deref())
        .ok_or(ServeError::NoStage("tts"))?
        .clone();

    let t1 = Instant::now();
    let prompt = state.config.build_prompt(history, user_text);

    let (piece_tx, mut piece_rx) = mpsc::channel::<String>(64);
    let (jobs_tx, jobs_rx) = mpsc::channel::<String>(8);
    let (audio_tx, mut audio_rx) = mpsc::channel::<TtsChunk>(AUDIO_BUFFER);

    let config = state.config.clone();
    let llm_task = tokio::spawn(async move { llm.stream(prompt, &config, piece_tx).await });
    let worker = tokio::spawn(synthesis_worker(
        backend,
        state.translator.clone(),
        // The script follows the engine, not the process. Resolved from the
        // engine the frame asked for, so mms and CosyVoice can be served by
        // one translator and still each get what they read.
        state.script_for(engine.as_deref()).to_string(),
        state.config.translate_ahead,
        jobs_rx,
        audio_tx,
    ));

    let mut chunker = Chunker::new(state.config.first_chunk, state.config.min_chunk);
    let mut reply_text = String::new();
    let mut first_audio: Option<u64> = None;
    let mut jobs = Some(jobs_tx);

    // Both streams are consumed in the same task so the socket keeps one owner.
    // Dropping `jobs` when the reply ends is what tells the worker to finish.
    loop {
        tokio::select! {
            piece = piece_rx.recv() => match piece {
                Some(piece) => {
                    reply_text.push_str(&piece);
                    send(socket, ServerMessage::Token { text: piece.clone() }).await?;
                    if let Some(chunk) = chunker.push(&piece)
                        && let Some(tx) = &jobs
                        && tx.send(chunk).await.is_err() {
                            jobs = None;
                        }
                }
                None => {
                    if let Some(tail) = chunker.finish()
                        && let Some(tx) = &jobs {
                            let _ = tx.send(tail).await;
                        }
                    // Dropping the sender is what tells the worker there is
                    // nothing more coming; without it the drain below would
                    // wait on a channel that never closes.
                    drop(jobs.take());
                    break;
                }
            },
            // `None` here means the worker died: it cannot finish while
            // `jobs` is still held. Keep serving the text half either way.
            chunk = audio_rx.recv() => if let Some(chunk) = chunk {
                *seq += 1;
                if first_audio.is_none() {
                    first_audio = Some(t1.elapsed().as_millis() as u64);
                }
                send(socket, ServerMessage::Audio {
                    seq: *seq,
                    wav: chunk.wav,
                    taigi: chunk.taigi,
                    roman: chunk.roman,
                    first_audio_ms: first_audio,
                })
                .await?;
            },
        }
    }

    // The reply is written; drain whatever audio is still being produced.
    while let Some(chunk) = audio_rx.recv().await {
        *seq += 1;
        if first_audio.is_none() {
            first_audio = Some(t1.elapsed().as_millis() as u64);
        }
        send(
            socket,
            ServerMessage::Audio {
                seq: *seq,
                wav: chunk.wav,
                taigi: chunk.taigi,
                roman: chunk.roman,
                first_audio_ms: first_audio,
            },
        )
        .await?;
    }

    if let Ok(Err(e)) = llm_task.await {
        return Err(e);
    }
    if let Ok(Err(e)) = worker.await {
        send(
            socket,
            ServerMessage::Error {
                message: e.to_string(),
            },
        )
        .await?;
    }

    let reply_text = reply_text.trim().to_string();
    history.push((Role::User, user_text.to_string()));
    history.push((Role::Bot, reply_text.clone()));

    send(
        socket,
        ServerMessage::Done {
            reply: Some(reply_text),
            total_ms: Some(t1.elapsed().as_millis() as u64),
            first_audio_ms: first_audio,
            reason: None,
        },
    )
    .await
}

/// A chunk that has been translated and is waiting to be spoken.
struct Ready {
    /// What the synthesiser reads.
    text: String,
    /// The Taigi to show beside the audio, empty when nothing translated it.
    taigi: String,
    /// Characters of source, for the timing line.
    chars: usize,
    /// How long the translation took.
    translate_ms: u64,
    /// Held until this chunk has been *spoken*, not merely received.
    ///
    /// The channel alone cannot express `ahead` of zero: its smallest capacity
    /// is one, and one buffered item is already one chunk of overlap. The
    /// permit is released at the end of the loop below, so `ahead + 1` permits
    /// means one chunk in the synthesiser and `ahead` translated in front of
    /// it.
    _permit: OwnedSemaphorePermit,
}

/// Clauses a turn may have in translation at once when `ahead` is not zero.
/// A reply is a handful of clauses; this is a bound on a queue, not a policy.
const MAX_AHEAD: usize = 64;

/// Drains the chunk queue, translating chunks ahead of the one being spoken
/// when `ahead` allows it.
///
/// Two stages rather than one, because translation and synthesis are different
/// models: chunk N+1 can be translated while chunk N is still becoming a
/// waveform, and translation is four times the cost of synthesis so the
/// overlap hides all of the latter after the first chunk. With `ahead` not
/// zero, every chunk after a turn's first is handed to the translator as it
/// is cut, and a local translator decodes them together over one weight
/// stream; the first is translated alone, because it is what the listener is
/// waiting through and a clause decoding beside it slows its every step.
///
/// `ahead` of zero runs the stages one after the other - see
/// [`crate::config::GatewayConfig::translate_ahead`] for the measurements
/// either way.
///
/// Synthesis stays a single ordered consumer either way. Audio has to reach the
/// browser in the order it will be played, and a second synthesiser would
/// finish a short later clause before a long earlier one.
async fn synthesis_worker(
    backend: TtsBackend,
    translator: Option<TranslatorBackend>,
    target: String,
    ahead: usize,
    mut jobs: mpsc::Receiver<String>,
    audio: mpsc::Sender<TtsChunk>,
) -> Result<(), ServeError> {
    // `ahead` of zero keeps the stages in step: one permit, taken before a
    // clause is translated and held until it has been spoken. Anything else
    // hands every clause to the translator the moment it arrives - the local
    // translator decodes them together over one weight stream, and a remote
    // one takes them as it likes - so the permits are as good as unbounded
    // and only the order is kept, by a queue of one slot a clause.
    let in_flight = Arc::new(Semaphore::new(if ahead == 0 { 1 } else { MAX_AHEAD }));
    let (ready_tx, mut ready) = mpsc::channel::<Ready>(ahead + 1);
    let (slot_tx, mut slots) = mpsc::channel::<tokio::sync::oneshot::Receiver<Ready>>(MAX_AHEAD);
    // Translations finish in whatever order they finish; the slots hand them
    // on in the order the clauses were cut.
    let ordering = tokio::spawn(async move {
        while let Some(slot) = slots.recv().await {
            let Ok(ready) = slot.await else {
                continue;
            };
            if ready_tx.send(ready).await.is_err() {
                break;
            }
        }
    });
    let permits = in_flight.clone();
    let translating = tokio::spawn(async move {
        // The first clause of a turn is translated on its own, and every
        // later one is handed over the moment it arrives. A clause decoding
        // beside the first slows the first's every step, and the first is
        // what the listener is waiting through; the later ones share a
        // stream among themselves and finish about when the longest of them
        // would have alone. Measured on a three-clause turn in
        // docs/BENCHMARKS.md, which is where the policy comes from.
        let mut first: Option<tokio::sync::oneshot::Receiver<()>> = None;
        let mut clauses = 0usize;
        while let Some(chunk) = jobs.recv().await {
            if clauses == 1
                && let Some(gate) = first.take()
            {
                let _ = gate.await;
            }
            // Taken before the work, not before the send, so that with `ahead`
            // at zero nothing is translated until the previous clause has
            // finished being spoken.
            let Ok(permit) = permits.clone().acquire_owned().await else {
                break;
            };
            let (done_tx, done_rx) = tokio::sync::oneshot::channel();
            if slot_tx.send(done_rx).await.is_err() {
                break;
            }
            let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
            if clauses == 0 {
                first = Some(gate_rx);
            }
            clauses += 1;
            let translator = translator.clone();
            let target = target.clone();
            tokio::spawn(async move {
                // Timed per chunk because the two stages are bound by
                // different things and only measurement says which one a
                // listener is waiting on. Reported once per chunk at info,
                // by the stage below.
                let queued = Instant::now();
                let chars = chunk.chars().count();
                // A translator in front of the synthesiser is optional. With
                // --direct-taigi the chat model has already answered in Taigi
                // Han, and this hop is skipped entirely - measured 3.8 s ->
                // 1.6 s.
                let (text, taigi) = match &translator {
                    None => (chunk, String::new()),
                    Some(t) => match t.translate(&chunk, &target).await {
                        Ok(out) if !out.is_empty() => (out.clone(), out),
                        Ok(_) => (chunk, String::new()),
                        Err(e) => {
                            // Speaking the untranslated Mandarin is wrong but
                            // audible; silence is wrong and looks like a
                            // crash.
                            tracing::warn!(%e, "translator failed, speaking the source text");
                            (chunk, String::new())
                        }
                    },
                };
                let _ = gate_tx.send(());
                let _ = done_tx.send(Ready {
                    text,
                    taigi,
                    chars,
                    translate_ms: queued.elapsed().as_millis() as u64,
                    _permit: permit,
                });
            });
        }
    });

    let result = speak_in_order(&backend, &audio, &mut ready).await;
    // The receiver is dropped by now either way, so the stages above have
    // been told to stop; this only waits for them to notice.
    drop(ready);
    let _ = translating.await;
    let _ = ordering.await;
    result
}

/// Synthesises translated chunks one at a time, in order.
async fn speak_in_order(
    backend: &TtsBackend,
    audio: &mpsc::Sender<TtsChunk>,
    ready: &mut mpsc::Receiver<Ready>,
) -> Result<(), ServeError> {
    while let Some(Ready {
        text,
        taigi,
        chars,
        translate_ms,
        // Dropped at the end of this iteration, which is what lets the stage
        // above start the next translation.
        _permit,
    }) = ready.recv().await
    {
        let translated = Instant::now();
        match backend {
            TtsBackend::Remote(up) => {
                let (tx, mut rx) = mpsc::channel::<TtsChunk>(AUDIO_BUFFER);
                let up = up.clone();
                let text2 = text.clone();
                let stream = tokio::spawn(async move { up.stream_tts(&text2, tx).await });
                while let Some(mut part) = rx.recv().await {
                    if part.taigi.is_empty() {
                        part.taigi = taigi.clone();
                    }
                    if audio.send(part).await.is_err() {
                        return Ok(());
                    }
                }
                if let Ok(Err(e)) = stream.await {
                    return Err(e);
                }
            }
            TtsBackend::Local(tx) => {
                let (reply_tx, mut rx) = mpsc::channel::<TtsChunk>(AUDIO_BUFFER);
                if tx
                    .send(SynthesisJob {
                        text: text.clone(),
                        reply: reply_tx,
                    })
                    .await
                    .is_err()
                {
                    return Err(ServeError::NoStage("tts"));
                }
                while let Some(mut part) = rx.recv().await {
                    if part.taigi.is_empty() {
                        part.taigi = taigi.clone();
                    }
                    if audio.send(part).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }
        tracing::info!(
            chars,
            translate_ms,
            synth_ms = translated.elapsed().as_millis() as u64,
            "chunk spoken"
        );
    }
    Ok(())
}

/// Sends one frame, turning a closed socket into an error the caller can stop on.
async fn send(socket: &mut WebSocket, msg: ServerMessage) -> Result<(), ServeError> {
    let text = serde_json::to_string(&msg).map_err(|e| ServeError::Client(e.to_string()))?;
    socket
        .send(Message::Text(text.into()))
        .await
        .map_err(|e| ServeError::Client(format!("websocket closed: {e}")))
}
