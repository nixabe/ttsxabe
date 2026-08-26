//! The whole forward pass, end to end.
//!
//! # Why this is staged rather than one call
//!
//! The prior needs `flow_size * frames` normal samples, and the frame count is
//! not known until the durations have been predicted - which itself needs
//! noise. So a single `synthesize(text, noise)` cannot state its own contract:
//! the caller would have to know the output length before asking for it.
//!
//! The stages below make that ordering explicit. [`Synthesizer::prepare`] is
//! deterministic; [`Synthesizer::durations`] takes the first draw;
//! [`Prosody::prior_noise_len`] then says exactly how much more noise is
//! needed, and [`Synthesizer::render`] takes it. [`Synthesizer::synthesize`]
//! wraps all three for callers who just want audio from a seed.
//!
//! It is also what makes the differential tests possible: they feed the
//! reference's captured draws in at exactly the two points it drew them.

use crate::{
    EncoderOutput, decoder, duration_predictor, expand_prior, flow_reverse, rng::Rng, text_encoder,
};
use std::path::Path;
use xabe_st::StFile;
use xabe_vits::{Tokenizer, VitsConfig, VitsWeights};

/// A loaded model, ready to synthesise.
#[derive(Debug)]
pub struct Synthesizer {
    file: StFile,
    cfg: VitsConfig,
    tok: Tokenizer,
}

/// The deterministic half of the pipeline: tokenisation and the text encoder.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// Symbol ids.
    pub ids: Vec<i64>,
    /// The text encoder's outputs.
    pub encoded: EncoderOutput,
}

impl Prepared {
    /// How many normal samples [`Synthesizer::durations`] needs.
    pub fn duration_noise_len(&self) -> usize {
        2 * self.ids.len()
    }
}

/// Predicted timing, and what it implies for the second draw.
#[derive(Debug, Clone)]
pub struct Prosody {
    /// One log duration per symbol.
    pub log_duration: Vec<f32>,
    /// Frames the durations expand to.
    pub frames: usize,
}

impl Prosody {
    /// How many normal samples [`Synthesizer::render`] needs.
    pub fn prior_noise_len(&self, cfg: &VitsConfig) -> usize {
        cfg.flow_size * self.frames
    }
}

impl Synthesizer {
    /// Loads a model directory: `model.safetensors`, `config.json`,
    /// `vocab.json` and `tokenizer_config.json`.
    pub fn open(dir: &Path) -> Result<Self, SynthesisError> {
        Self::open_files(
            &dir.join("model.safetensors"),
            &dir.join("config.json"),
            dir,
        )
    }

    /// Loads a model from individually addressed files.
    ///
    /// The CLI needs this because it lets `--config` point somewhere other than
    /// beside the checkpoint; the tokenizer's two JSON files are still taken
    /// from one directory, since they are meaningless apart.
    pub fn open_files(
        model: &Path,
        config: &Path,
        tokenizer_dir: &Path,
    ) -> Result<Self, SynthesisError> {
        let cfg = VitsConfig::from_json_path(config)?;
        let tok = Tokenizer::load(tokenizer_dir)?;
        let file = StFile::open(model)?;
        // Bound once here purely to fail early: a checkpoint that does not
        // match its config should be rejected at open, not on the first
        // synthesis.
        VitsWeights::load(&file, &cfg)?;
        tracing::info!(model = %model.display(), "loaded model");
        Ok(Self { file, cfg, tok })
    }

    /// The model's geometry.
    pub fn config(&self) -> &VitsConfig {
        &self.cfg
    }

    /// The model's geometry, for the handful of sampling parameters the CLI
    /// exposes as overrides.
    ///
    /// Only `noise_scale`, `noise_scale_duration` and `speaking_rate` are
    /// meaningful to change - they are temperatures and a rate, not geometry.
    /// Changing anything else here would contradict the checkpoint.
    pub fn config_mut(&mut self) -> &mut VitsConfig {
        &mut self.cfg
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// Tokenises and runs the text encoder. Deterministic.
    pub fn prepare(&self, text: &str) -> Result<Prepared, SynthesisError> {
        let ids = self.tok.encode(text);
        if ids.is_empty() {
            return Err(SynthesisError::NoSymbols {
                text: text.to_string(),
            });
        }
        let w = self.weights()?;
        let encoded = text_encoder(&ids, &w.text_encoder, &self.cfg);
        Ok(Prepared { ids, encoded })
    }

    /// Predicts timing from the first noise draw.
    pub fn durations(&self, prepared: &Prepared, noise: &[f32]) -> Result<Prosody, SynthesisError> {
        let want = prepared.duration_noise_len();
        if noise.len() != want {
            return Err(SynthesisError::NoiseLength {
                stage: "duration",
                want,
                got: noise.len(),
            });
        }
        let w = self.weights()?;
        let log_duration = duration_predictor(
            &prepared.encoded.hidden,
            noise,
            &w.duration_predictor,
            &self.cfg,
        );
        let frames = log_duration
            .iter()
            .map(|v| (v.exp() / self.cfg.speaking_rate).ceil().max(0.0) as usize)
            .sum::<usize>()
            .max(1);
        Ok(Prosody {
            log_duration,
            frames,
        })
    }

    /// Expands the prior, inverts the flow, and decodes to a waveform.
    pub fn render(
        &self,
        prepared: &Prepared,
        prosody: &Prosody,
        noise: &[f32],
    ) -> Result<Vec<f32>, SynthesisError> {
        let want = prosody.prior_noise_len(&self.cfg);
        if noise.len() != want {
            return Err(SynthesisError::NoiseLength {
                stage: "prior",
                want,
                got: noise.len(),
            });
        }
        let w = self.weights()?;
        let prior = expand_prior(
            &prepared.encoded.m_p,
            &prepared.encoded.logs_p,
            &prosody.log_duration,
            noise,
            &self.cfg,
        );
        let z = flow_reverse(&prior.z_p, &w.flow, &self.cfg);
        Ok(decoder(&z, &w.decoder, &self.cfg))
    }

    /// Synthesises audio from text, drawing its own noise.
    ///
    /// The same seed gives the same audio on any machine. It does *not* give
    /// the same audio as PyTorch at that seed - see [`crate::rng`].
    pub fn synthesize(&self, text: &str, seed: u64) -> Result<Vec<f32>, SynthesisError> {
        let mut rng = Rng::new(seed);
        let prepared = self.prepare(text)?;
        let noise = rng.normals(prepared.duration_noise_len());
        let prosody = self.durations(&prepared, &noise)?;
        let noise = rng.normals(prosody.prior_noise_len(&self.cfg));
        let audio = self.render(&prepared, &prosody, &noise)?;
        tracing::info!(
            symbols = prepared.ids.len(),
            frames = prosody.frames,
            samples = audio.len(),
            "synthesised",
        );
        Ok(audio)
    }

    /// Re-binds the weight schema.
    ///
    /// Binding is 662 lookups into an already-parsed header and copies nothing,
    /// so it is done per call rather than stored - which would make this a
    /// self-referential struct for no measurable gain against a decoder that
    /// takes seconds.
    fn weights(&self) -> Result<VitsWeights<'_>, SynthesisError> {
        Ok(VitsWeights::load(&self.file, &self.cfg)?)
    }
}

/// Synthesis could not be performed.
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    /// The checkpoint could not be opened or addressed.
    #[error(transparent)]
    Container(#[from] xabe_st::StError),

    /// The config could not be read or is not supported.
    #[error(transparent)]
    Config(#[from] xabe_vits::ConfigError),

    /// The tokenizer could not be loaded.
    #[error(transparent)]
    Tokenizer(#[from] xabe_vits::TokenizerError),

    /// The GPU path failed, or there is no device.
    #[error(transparent)]
    Cuda(#[from] xabe_cuda::CudaError),

    /// The checkpoint does not match its config.
    #[error(transparent)]
    Weights(#[from] xabe_vits::WeightError),

    /// The text produced no symbols. Every character was outside the
    /// vocabulary - Han text and punctuation alone both do this - and
    /// synthesising would give a zero-length waveform rather than an error.
    #[error("{text:?} contains no symbols this model can speak")]
    NoSymbols {
        /// The text as given.
        text: String,
    },

    /// A noise buffer was the wrong length for the stage it was given to.
    #[error("{stage} noise must be {want} samples, got {got}")]
    NoiseLength {
        /// Which draw was wrong.
        stage: &'static str,
        /// The length required.
        want: usize,
        /// The length supplied.
        got: usize,
    },
}
