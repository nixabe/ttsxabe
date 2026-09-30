//! VITS geometry, read from the checkpoint's `config.json` and validated.
//!
//! # What is (and isn't) here
//!
//! Numbers and the relationships between them. [`VitsConfig::from_json_str`]
//! rejects a geometry this implementation cannot express — a hidden size that does not
//! divide into heads, upsample rates and kernels of different lengths — so that
//! every consumer downstream can index without checking.
//!
//! Nothing here reads weights or does arithmetic.
//!
//! Start at [`VitsConfig::from_json_path`].

use crate::error::ConfigError;

/// The fields of `config.json` this implementation reads.
///
/// Deserialised permissively — the file carries training-time settings that are
/// irrelevant at inference — then validated by [`VitsConfig::from_json_str`],
/// which both constructors go through.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VitsConfig {
    /// Model width. 192 for mms-tts-nan.
    pub hidden_size: usize,
    /// Text encoder depth.
    pub num_hidden_layers: usize,
    /// Attention heads in the text encoder.
    pub num_attention_heads: usize,
    /// Inner width of the text encoder's convolutional feed-forward.
    pub ffn_dim: usize,
    /// Kernel size of both feed-forward convolutions.
    pub ffn_kernel_size: usize,
    /// Symbol count.
    pub vocab_size: usize,
    /// Output sample rate in Hz.
    pub sampling_rate: u32,
    /// Channel count carried by the normalising flow.
    pub flow_size: usize,
    /// Number of coupling blocks in the flow.
    pub prior_encoder_num_flows: usize,
    /// WaveNet layers inside each coupling block.
    pub prior_encoder_num_wavenet_layers: usize,
    /// WaveNet convolution kernel size.
    pub wavenet_kernel_size: usize,
    /// WaveNet dilation base.
    pub wavenet_dilation_rate: usize,
    /// One upsample factor per decoder stage. Their product is the number of
    /// waveform samples produced per input frame.
    pub upsample_rates: Vec<usize>,
    /// Transposed-convolution kernel per decoder stage.
    pub upsample_kernel_sizes: Vec<usize>,
    /// Channels entering the first decoder stage.
    pub upsample_initial_channel: usize,
    /// Kernel size of each multi-receptive-field resblock.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilations within each resblock, one list per kernel size.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Depthwise-separable stack depth in the duration predictor.
    pub depth_separable_channels: usize,
    /// Duration predictor convolution kernel size.
    pub duration_predictor_kernel_size: usize,
    /// Coupling blocks in the stochastic duration predictor.
    pub duration_predictor_num_flows: usize,
    /// Layers in each depthwise-separable convolution stack.
    pub depth_separable_num_layers: usize,
    /// Bins in each rational-quadratic spline.
    pub duration_predictor_flow_bins: usize,
    /// Beyond +/- this, the spline is the identity.
    pub duration_predictor_tail_bound: f32,
    /// Half-width of the relative attention window.
    pub window_size: usize,
    /// Epsilon added inside every layer norm's square root.
    ///
    /// Defaulted rather than required because older configs omit it, and the
    /// reference's own default is the value every published checkpoint uses.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Activation in the text encoder's feed-forward.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Temperature applied to the prior when sampling.
    pub noise_scale: f32,
    /// Temperature applied to the duration predictor when sampling.
    pub noise_scale_duration: f32,
    /// Multiplier on predicted durations. Larger is slower.
    pub speaking_rate: f32,
    /// Negative slope of the decoder's leaky ReLU.
    pub leaky_relu_slope: f32,
    /// Whether durations are sampled rather than regressed. Must be true.
    pub use_stochastic_duration_prediction: bool,
}

impl VitsConfig {
    /// Reads and validates a `config.json`.
    pub fn from_json_path(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_str(&text)
    }

    /// Parses and validates a `config.json` already in memory.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: Self = serde_json::from_str(text)?;
        raw.validate()?;
        Ok(raw)
    }

    /// Rejects a geometry this implementation cannot express.
    ///
    /// Called by both constructors. Everything downstream indexes on the
    /// invariants established here, so this is the only place they are checked.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        // A different activation would change every output while breaking no
        // shape, so it is checked rather than assumed.
        if self.hidden_act != "relu" {
            return Err(ConfigError::UnsupportedActivation {
                act: self.hidden_act.clone(),
            });
        }

        for (field, v) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("ffn_dim", self.ffn_dim),
            ("vocab_size", self.vocab_size),
            ("flow_size", self.flow_size),
            ("upsample_initial_channel", self.upsample_initial_channel),
        ] {
            if v == 0 {
                return Err(ConfigError::Zero { field });
            }
        }

        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(ConfigError::HeadSplit {
                hidden: self.hidden_size,
                heads: self.num_attention_heads,
            });
        }

        if self.upsample_rates.len() != self.upsample_kernel_sizes.len() {
            return Err(ConfigError::UpsampleMismatch {
                rates: self.upsample_rates.len(),
                kernels: self.upsample_kernel_sizes.len(),
            });
        }

        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(ConfigError::ResblockMismatch {
                kernels: self.resblock_kernel_sizes.len(),
                dilations: self.resblock_dilation_sizes.len(),
            });
        }

        // Each decoder stage halves the channel count. If it reaches zero the
        // final convolution has nothing to read.
        if self.upsample_initial_channel >> self.upsample_rates.len() == 0 {
            return Err(ConfigError::ChannelUnderflow {
                channels: self.upsample_initial_channel,
                stages: self.upsample_rates.len(),
            });
        }

        if !self.use_stochastic_duration_prediction {
            return Err(ConfigError::DeterministicDuration);
        }

        Ok(())
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of decoder upsample stages.
    pub fn num_upsample_stages(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Resblocks per upsample stage.
    pub fn resblocks_per_stage(&self) -> usize {
        self.resblock_kernel_sizes.len()
    }

    /// Total resblocks in the decoder, flattened as the checkpoint stores them.
    pub fn num_resblocks(&self) -> usize {
        self.num_upsample_stages() * self.resblocks_per_stage()
    }

    /// Channels entering decoder stage `stage`.
    ///
    /// The decoder halves its width at every stage: 512 → 256 → 128 → 64 → 32.
    pub fn upsample_in_channels(&self, stage: usize) -> usize {
        self.upsample_initial_channel >> stage
    }

    /// Channels leaving decoder stage `stage`, which is also the width the
    /// resblocks attached to that stage operate at.
    pub fn upsample_out_channels(&self, stage: usize) -> usize {
        self.upsample_initial_channel >> (stage + 1)
    }

    /// Waveform samples produced per input frame — the product of the rates.
    pub fn hop_length(&self) -> usize {
        self.upsample_rates.iter().product()
    }

    /// Width of the relative position embedding table: one entry either side of
    /// centre, plus centre.
    pub fn rel_window(&self) -> usize {
        2 * self.window_size + 1
    }

    /// Half of [`Self::flow_size`], the width each coupling block splits into.
    pub fn flow_half(&self) -> usize {
        self.flow_size / 2
    }
}

/// The reference's layer-norm epsilon.
fn default_layer_norm_eps() -> f32 {
    1e-5
}

/// The activation every published MMS-TTS checkpoint uses.
fn default_hidden_act() -> String {
    "relu".to_string()
}
