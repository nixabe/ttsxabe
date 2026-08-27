//! The flag surface, which is also the `--help` text.
//!
//! The doc comments on [`Args`] *are* what `--help` prints; there is no second
//! copy to drift. See `docs/CLI.md` for the conventions and `stage.rs` for the
//! symmetry the `--<stage>-model` / `--<stage>-url` pairs implement.
//!
//! Every flag has an environment twin, so a container can be configured
//! without rewriting its command line.

use crate::stage::{Requested, StageError, Stages};
use clap::Parser;
use std::path::PathBuf;

/// The Taigi voice engine: ASR, VAD, translation, TTS and the web front end.
///
/// Each stage is satisfied either by a local checkpoint (`--<stage>-model`) or
/// by another process over HTTP (`--<stage>-url`). Give the stages you want and
/// this process becomes exactly that: a monolith, a single-stage worker, or
/// anything between. The chat LLM stays in llama.cpp and is URL-only.
#[derive(Debug, Parser)]
#[command(name = "xabe-engine", version, about, long_about = None)]
pub struct Args {
    /// Serve HTTP on this address. Without it the run is one-shot.
    #[arg(long, env = "XABE_SERVE", value_name = "ADDR")]
    pub serve: Option<String>,

    /// Speech-to-text checkpoint directory.
    #[arg(long, env = "XABE_ASR_MODEL", value_name = "PATH")]
    pub asr_model: Option<PathBuf>,
    /// Delegate speech-to-text to a process at this base URL.
    #[arg(long, env = "XABE_ASR_URL", value_name = "URL")]
    pub asr_url: Option<String>,
    /// Where to run the ASR: cpu, or a CUDA device ordinal.
    #[arg(long, env = "XABE_ASR_DEVICE", value_name = "DEV")]
    pub asr_device: Option<String>,

    /// Voice-activity checkpoint. Gates the ASR; alone with --in it prints segments.
    #[arg(long, env = "XABE_VAD_MODEL", value_name = "PATH")]
    pub vad_model: Option<PathBuf>,
    /// Delegate voice-activity detection to a process at this base URL.
    #[arg(long, env = "XABE_VAD_URL", value_name = "URL")]
    pub vad_url: Option<String>,
    /// Where to run the VAD: cpu, or a CUDA device ordinal.
    #[arg(long, env = "XABE_VAD_DEVICE", value_name = "DEV")]
    pub vad_device: Option<String>,

    /// Text-to-speech checkpoint directory, or the safetensors file itself.
    #[arg(long, env = "XABE_TTS_MODEL", value_name = "PATH")]
    pub tts_model: Option<PathBuf>,
    /// Delegate text-to-speech to a process at this base URL.
    #[arg(long, env = "XABE_TTS_URL", value_name = "URL")]
    pub tts_url: Option<String>,
    /// Where to run the TTS: cpu, or a CUDA device ordinal.
    ///
    /// The CPU path is the scalar reference and is roughly 45x slower than real
    /// time; it exists to be read and to be correct, not to be used.
    #[arg(long, env = "XABE_TTS_DEVICE", value_name = "DEV")]
    pub tts_device: Option<String>,

    /// Mandarin-to-Taigi checkpoint directory.
    #[arg(long, env = "XABE_TRANSLATOR_MODEL", value_name = "PATH")]
    pub translator_model: Option<PathBuf>,
    /// Delegate translation to a llama-server at this base URL.
    #[arg(long, env = "XABE_TRANSLATOR_URL", value_name = "URL")]
    pub translator_url: Option<String>,
    /// Where to run the translator: cpu, or a CUDA device ordinal.
    #[arg(long, env = "XABE_TRANSLATOR_DEVICE", value_name = "DEV")]
    pub translator_device: Option<String>,

    /// Chat model, as a llama-server base URL. There is no --llm-model.
    #[arg(long, env = "XABE_LLM_URL", value_name = "URL")]
    pub llm_url: Option<String>,

    /// Have the chat model answer in Taigi Han itself, skipping the translator.
    ///
    /// Measured 3.8 s -> 1.6 s on a voice turn, and it frees the translator's
    /// VRAM. This is what the shipped pipeline runs.
    #[arg(long, env = "XABE_DIRECT_TAIGI")]
    pub direct_taigi: bool,

    /// One-shot input audio. Use - to read stdin.
    #[arg(long = "in", value_name = "WAV")]
    pub input: Option<PathBuf>,

    /// One-shot input text. Use - to read stdin.
    #[arg(long, value_name = "TEXT")]
    pub text: Option<String>,

    /// One-shot output. Use - to write stdout.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// TTS model config. Defaults to config.json beside the checkpoint.
    #[arg(long, env = "XABE_TTS_CONFIG", value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Seed for the duration and prior draws.
    #[arg(long, env = "XABE_SEED", default_value_t = 0)]
    pub seed: u64,

    /// Prior temperature. Higher is more varied.
    #[arg(long)]
    pub noise_scale: Option<f32>,

    /// Duration temperature. Higher varies the rhythm more.
    #[arg(long)]
    pub noise_scale_duration: Option<f32>,

    /// Speaking rate. Higher is faster.
    #[arg(long)]
    pub speaking_rate: Option<f32>,

    /// Extra TTS engines the page can choose, as `name=url`. Repeatable.
    ///
    /// `--tts-model` and `--tts-url` register one engine; this registers the
    /// others, which is how the page offers mms and cosyvoice side by side.
    #[arg(
        long = "tts-engine",
        env = "XABE_TTS_ENGINES",
        value_name = "NAME=URL",
        value_delimiter = ','
    )]
    pub tts_engines: Vec<String>,

    /// Which engine the page selects on load.
    #[arg(long, env = "XABE_TTS_DEFAULT", value_name = "NAME")]
    pub tts_default: Option<String>,

    /// What the translator is asked to produce: POJ, HAN or HL.
    ///
    /// mms consumes romanisation, so it needs POJ; CosyVoice reads Han. This
    /// is the default for every engine; `--tts-script` overrides it per engine.
    #[arg(long, env = "XABE_TRANSLATOR_TARGET", default_value = "POJ")]
    pub translator_target: String,

    /// The script one engine wants, as `name=POJ|HAN|HL`. Repeatable.
    ///
    /// Two synthesisers in one process do not have to read the same script,
    /// and the two this pipeline runs do not: mms consumes romanisation and
    /// CosyVoice reads Han. A single `--translator-target` can only be right
    /// for one of them, and being wrong is not an error anywhere - mms handed
    /// Han tokenises to nothing and simply says nothing at all.
    #[arg(
        long = "tts-script",
        env = "XABE_TTS_SCRIPTS",
        value_name = "NAME=SCRIPT",
        value_delimiter = ','
    )]
    pub tts_scripts: Vec<String>,

    /// Language given to the ASR. Never `en`.
    ///
    /// Breeze-ASR-26 transcribes Taigi speech *into* Mandarin Han; asking it
    /// for English gets a translation instead of a transcript.
    #[arg(long, env = "XABE_ASR_LANG", default_value = "zh")]
    pub asr_lang: String,

    /// What the user is called in the chat transcript.
    #[arg(long, env = "XABE_PERSON", default_value = "使用者")]
    pub person: String,

    /// What the assistant is called in the chat transcript.
    #[arg(long, env = "XABE_BOT", default_value = "小助理")]
    pub bot: String,

    /// Sampling temperature for the reply.
    #[arg(long, env = "XABE_TEMPERATURE", default_value_t = 0.3)]
    pub temperature: f32,

    /// Maximum reply length, in tokens.
    #[arg(long, env = "XABE_MAX_TOKENS", default_value_t = 160)]
    pub max_tokens: u32,

    /// How many previous turns stay in the prompt.
    #[arg(long, env = "XABE_HISTORY_TURNS", default_value_t = 6)]
    pub history_turns: usize,

    /// Minimum characters before a later chunk is synthesised.
    #[arg(long, env = "XABE_MIN_CHUNK", default_value_t = 6)]
    pub min_chunk: usize,

    /// Minimum characters before the first chunk is synthesised.
    ///
    /// Lower than --min-chunk on purpose: getting the voice started is worth
    /// more than getting the first clause exactly right.
    #[arg(long, env = "XABE_FIRST_CHUNK", default_value_t = 4)]
    pub first_chunk: usize,

    /// Read the system prompt from a file instead of using the built-in one.
    #[arg(long, env = "XABE_PROMPT_FILE", value_name = "PATH")]
    pub prompt_file: Option<PathBuf>,

    /// Log verbosity: info, debug or trace.
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub log_level: String,
}

impl Args {
    /// Resolves the stage flags into [`Stages`].
    pub fn stages(&self) -> Result<Stages, StageError> {
        Stages::resolve(
            &Requested {
                model: self.asr_model.clone(),
                url: self.asr_url.clone(),
                device: self.asr_device.clone(),
            },
            &Requested {
                model: self.vad_model.clone(),
                url: self.vad_url.clone(),
                device: self.vad_device.clone(),
            },
            &Requested {
                model: self.tts_model.clone(),
                url: self.tts_url.clone(),
                device: self.tts_device.clone(),
            },
            &Requested {
                model: self.translator_model.clone(),
                url: self.translator_url.clone(),
                device: self.translator_device.clone(),
            },
            self.llm_url.clone(),
            self.serve.is_some(),
        )
    }
}
