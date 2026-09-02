//! The four stages as one call: text in, 24 kHz audio out.
//!
//! ```text
//! text --[Tokenizer]--> ids --[SpeechLlm]--> speech tokens (25 Hz)
//!      --[Flow]--> mel (80 x N, 50 Hz) --[F0 + oscillators]--> excitation
//!      --[Vocoder]--> waveform (24 kHz)
//! ```
//!
//! # The instruct string is configuration, not an argument
//!
//! `frontend_instruct2` deletes the LLM's audio prompt, so in instruct mode
//! the language model sees **only text**: the instruct string, then the
//! utterance. The speaker is carried entirely by the flow. That makes the
//! instruct part of what the model *is* for a given deployment - change it and
//! the same text produces different speech tokens - so it is pinned at
//! construction beside the voice rather than passed per utterance.
//!
//! It must contain `<|endofprompt|>`; [`SpeechLlm::prompt`] refuses a prompt
//! without it, and that refusal is the one worth keeping loudest, because a
//! model that never saw the marker does not fail, it produces confident
//! nonsense.
//!
//! # Three devices' worth of state on one card
//!
//! The LLM, the flow and the vocoder each open their own [`xabe_cuda::Gpu`]. They pass
//! host vectors between them, which is a few megabytes per utterance against a
//! 642 M-parameter decode - and it is what lets each stage be opened, tested
//! and benchmarked on its own.

use crate::{
    CosyError, Dither, F0Predictor, Flow, FlowConfig, HiftConfig, LlmConfig, RasConfig, Rng,
    SourceConfig, SpeechLlm, Tokenizer, Vocoder, Voice, excitation, ras_sample,
};
use std::path::Path;
use xabe_st::StFile;

/// How the decode loop is bounded, as ratios against the text's length.
///
/// Upstream's `min_token_text_ratio` and `max_token_text_ratio`. The minimum is
/// enforced by masking the end token rather than by ignoring it, which is a
/// different thing from a length penalty: below the floor the model is not
/// *allowed* to stop, and the mask is lifted the moment it is reached.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Speech tokens per text token, below which the end token is masked.
    pub min_ratio: usize,
    /// Speech tokens per text token, at which generation is refused.
    pub max_ratio: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min_ratio: 2,
            max_ratio: 20,
        }
    }
}

/// A whole CosyVoice3, resident on one card.
pub struct Cosy {
    tokenizer: Tokenizer,
    llm: SpeechLlm,
    flow: Flow,
    vocoder: Vocoder,
    f0: F0Predictor,
    /// `m_source.l_linear`, which mixes the nine harmonics down to one signal.
    source_w: Vec<f32>,
    source_b: f32,
    voice: Voice,
    instruct: String,
    instruct_ids: Vec<u32>,
    ras: RasConfig,
    bounds: Bounds,
}

impl Cosy {
    /// Opens every stage from a converted checkpoint directory.
    ///
    /// `voice` is a bundle from `tools/make_cosyvoice_voice.py`; `instruct` is
    /// the instruction the language model is given, and has to end on
    /// `<|endofprompt|>`.
    pub fn open(
        dir: &Path,
        voice: &Path,
        instruct: &str,
        ordinal: usize,
    ) -> Result<Self, CosyError> {
        let tokenizer = Tokenizer::from_dir(&dir.join("CosyVoice-BlankEN"))?;
        let instruct_ids = tokenizer.encode(instruct);
        // Checked here as well as inside `prompt`, because here it can name the
        // configuration that is wrong rather than the tensor that is empty.
        if !instruct_ids.contains(&LlmConfig::ENDOFPROMPT) {
            return Err(CosyError::NoEndOfPrompt(LlmConfig::ENDOFPROMPT));
        }

        let llm = SpeechLlm::open(&dir.join("llm.safetensors"), ordinal)?;
        let flow = Flow::open(&dir.join("flow.safetensors"), ordinal)?;
        let vocoder = Vocoder::open(&dir.join("hift.safetensors"), ordinal)?;

        // The excitation's own weights live in `hift.safetensors` beside the
        // vocoder's, and are read from the same file a second time rather than
        // threaded out of `Vocoder`, which does not use them.
        let hift = StFile::open(dir.join("hift.safetensors"))?;
        let f0 = F0Predictor::bind(&hift, vocoder.gpu())?;
        let source_w = hift
            .tensor_shaped("m_source.l_linear.weight", &[1, crate::HARMONICS])?
            .to_vec();
        let source_b = hift.tensor_shaped("m_source.l_linear.bias", &[1])?[0];

        let voice = Voice::open(voice, flow.config().mel_dim)?;
        tracing::info!(
            prompt_tokens = voice.prompt_token.len(),
            instruct_tokens = instruct_ids.len(),
            "cosyvoice3 ready"
        );

        Ok(Self {
            tokenizer,
            llm,
            flow,
            vocoder,
            f0,
            source_w,
            source_b,
            voice,
            instruct: instruct.to_string(),
            instruct_ids,
            ras: RasConfig::default(),
            bounds: Bounds::default(),
        })
    }

    /// The instruction this engine was opened with.
    pub fn instruct(&self) -> &str {
        &self.instruct
    }

    /// The sampler's settings, so a caller can pin a different seed.
    pub fn sampling_mut(&mut self) -> &mut RasConfig {
        &mut self.ras
    }

    /// The output rate, which is the vocoder's.
    pub fn sample_rate(&self) -> usize {
        SourceConfig::default().sample_rate
    }

    /// Text to a waveform at [`Cosy::sample_rate`].
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, CosyError> {
        let ids = self.tokenizer.encode(text);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.speech_tokens(&ids)?;
        self.vocode(&tokens)
    }

    /// The language model half: text ids to speech tokens.
    ///
    /// The prompt is `[sos] embed(instruct ++ text) [task_id]` with the speech
    /// slot empty, and then the model writes tokens one at a time until it
    /// emits one of the three stop ids.
    pub fn speech_tokens(&self, text: &[u32]) -> Result<Vec<u32>, CosyError> {
        let cfg = *self.llm.config();
        let vocab = cfg.speech_vocab_size;

        let mut all = self.instruct_ids.clone();
        all.extend_from_slice(text);
        let prompt = self.llm.prompt(&all, text.len())?;

        let mut cache = self.llm.cache();
        let mut logits = self.llm.forward(&prompt.h, prompt.len, &mut cache)?;
        let mut at = prompt.len;

        // The bounds are against the *utterance*, not against the instruct,
        // which is why `text` here is the utterance's ids alone.
        let (min_len, max_len) = (
            text.len() * self.bounds.min_ratio,
            text.len() * self.bounds.max_ratio,
        );
        let mut rng = Rng::new(self.ras.seed);
        let mut out: Vec<u32> = Vec::new();

        for i in 0..max_len {
            // The last row of the logits. After the prompt there is only
            // ever one, and it is downloaded where it lies rather than
            // through a copy - two launches a token for nothing.
            let mut row = if at == 1 {
                self.llm.gpu().download(&logits)?
            } else {
                let row = self
                    .llm
                    .gpu()
                    .copy_range(&logits, (at - 1) * vocab, vocab)?;
                self.llm.gpu().download(&row)?
            };

            // Below the floor the end token is *masked*, not penalised: the
            // model is not allowed to stop, and the mask lifts the moment the
            // floor is reached. Only the end token - the two ids above it are
            // never sampled either way and upstream leaves them alone.
            if i < min_len {
                row[cfg.speech_token_size] = f32::NEG_INFINITY;
            }

            let id = ras_sample(&row, &out, &self.ras, &mut rng);
            if id as usize >= cfg.speech_token_size {
                return Ok(out);
            }
            out.push(id);

            let e = self.llm.speech_step(id)?;
            logits = self.llm.forward(&e, 1, &mut cache)?;
            at = 1;
        }

        // Reaching the ceiling is not a long utterance, it is a model that has
        // stopped ending sentences - twenty speech tokens per text token is
        // roughly eight hundred milliseconds of audio per character. Refusing
        // is cheaper than emitting a minute of babble.
        Err(CosyError::RanAway {
            got: out.len(),
            text: text.len(),
            max: self.bounds.max_ratio,
        })
    }

    /// The acoustic half: speech tokens to a waveform.
    pub fn vocode(&self, tokens: &[u32]) -> Result<Vec<f32>, CosyError> {
        let v = &self.voice;
        let frames_needed =
            (v.prompt_token.len() + tokens.len()) * self.flow.config().token_mel_ratio;
        if frames_needed > v.max_frames() {
            return Err(CosyError::Geometry {
                what: "the utterance is longer than the bundle's diffusion noise",
                got: frames_needed,
                want: v.max_frames(),
            });
        }

        let (mel, frames) = self.flow.mel(
            &v.prompt_token,
            tokens,
            &v.prompt_feat,
            &v.embedding,
            &v.cfm_noise,
        )?;

        let gpu = self.vocoder.gpu();
        let gmel = gpu.upload(&mel)?;
        let f0 = self.f0.predict(gpu, &gmel, frames)?;

        let samples = frames * self.vocoder.config().hop();
        // Seeded off the sampler's seed, so one knob makes a whole utterance
        // reproducible rather than two that have to be kept in step.
        let dither = Dither::seeded(samples, self.ras.seed ^ 0x5F1E_C0DE);
        let source = excitation(
            &f0,
            &SourceConfig::default(),
            &dither,
            &self.source_w,
            self.source_b,
        )?;
        let gsrc = gpu.upload(&source)?;
        self.vocoder.decode(&gmel, frames, &gsrc, samples)
    }

    /// The geometries the stages were opened with, for a caller that logs them.
    pub fn geometry(&self) -> (&LlmConfig, &FlowConfig, &HiftConfig) {
        (self.llm.config(), self.flow.config(), self.vocoder.config())
    }
}
