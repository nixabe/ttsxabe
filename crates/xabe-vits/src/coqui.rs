//! The Coqui dialect of the same model.
//!
//! # Why a second reader for one architecture
//!
//! `neurlang/coqui-vits-suisiann-minnan-hokkien` is VITS - the same text
//! encoder, the same stochastic duration predictor, the same mean-only flow and
//! the same HiFi-GAN. Nothing in `xabe-tts` changes to run it. What changes is
//! everything around the arithmetic:
//!
//! - **The container.** A trainer's `.pth` rather than a safetensors export, so
//!   it is read through `xabe-pt` instead of `xabe-st`.
//! - **The names.** `text_encoder.encoder.attn_layers.0.conv_q` here is
//!   `text_encoder.encoder.layers.0.attention.q_proj` there. 🤗 renamed every
//!   tensor when it ported the model.
//! - **The weight norm.** The 🤗 export fused the decoder's parameterisation
//!   away; this checkpoint keeps it, on the upsamplers and on every resblock
//!   convolution. See [`MaybeWn`].
//! - **The vocabulary.** 137 IPA phonemes rather than 48 letters of Pe̍h-ōe-jī,
//!   with the blank at id 3 rather than id 0.
//! - **The geometry file.** A Coqui `config.json` is a training run's whole
//!   configuration; the fields this model's shapes come from are a dozen of its
//!   two hundred, and several of the numbers are not in it at all because the
//!   reference hardcodes them.
//!
//! # What is (and isn't) here
//!
//! Shapes, names and the vocabulary. No arithmetic - not the weight-norm
//! fusion, which is a division and belongs to `xabe-dsp`, and not the
//! phonemisation, which is not arithmetic at all and is discussed in
//! [`CoquiTokenizer`].
//!
//! Start at [`CoquiConfig::from_json_path`], then [`VitsWeights::load_coqui`].

use rustc_hash::FxHashMap;
use std::path::Path;

use xabe_pt::PtFile;

use crate::config::VitsConfig;
use crate::error::{ConfigError, TokenizerError, WeightError};
use crate::weights::{
    Conv, DdsConv, Decoder, DurationFlow, DurationPredictor, EncoderLayer, FlowBlock, MaybeWn,
    Norm, ResBlock, TextEncoder, VitsWeights, WaveNetLayer, WnConv,
};

/// Half-width of the relative attention window.
///
/// `TextEncoder` passes `rel_attn_window_size=4` to the transformer as a
/// literal, so it is not in `config.json` and cannot be read from one. The
/// checkpoint's `emb_rel_k` is `[1, 9, 96]`, and binding checks that.
const REL_ATTN_WINDOW: usize = 4;

/// Coupling blocks in the prior flow.
///
/// `ResidualCouplingBlocks` defaults `num_flows` to 4 and VITS never overrides
/// it, so - like the window above - it is a property of the reference's code
/// rather than of its config.
const PRIOR_FLOWS: usize = 4;

/// Coupling blocks in the stochastic duration predictor, likewise a literal.
const DURATION_FLOWS: usize = 4;

/// Kernel size of the duration predictor's convolutions, passed as `3`.
const DURATION_KERNEL: usize = 3;

/// Depth of every dilated depthwise-separable stack, passed as `num_layers=3`.
const DDS_LAYERS: usize = 3;

/// Bins in each rational-quadratic spline: `ConvFlow`'s `num_bins` default.
///
/// The checkpoint's `proj` emits `3 * bins - 1 = 29` channels, which is what
/// binding checks.
const SPLINE_BINS: usize = 10;

/// Beyond +/- this the spline is the identity: `ConvFlow`'s `tail_bound`.
const SPLINE_TAIL_BOUND: f32 = 5.0;

/// The decoder's leaky ReLU slope, `LRELU_SLOPE` in `hifigan_generator.py`.
const LRELU_SLOPE: f32 = 0.1;

/// `LayerNorm2`'s epsilon, which is the norm this model's encoder uses.
const LAYER_NORM_EPS: f32 = 1e-5;

/// The `audio` block of a Coqui config. Only the sample rate reaches inference.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CoquiAudio {
    /// Output sample rate in Hz. 22050 for this checkpoint, against
    /// `mms-tts-nan`'s 16000.
    pub sample_rate: u32,
}

/// The `characters` block: how the symbol table is built.
///
/// The four special tokens are multi-character strings - `<PAD>`, `<BLNK>` -
/// and so can never be produced by looking a phoneme up. They occupy ids all
/// the same, which is why the blank is 3 rather than 0.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CoquiCharacters {
    /// The padding token, first in the vocabulary.
    pub pad: String,
    /// End-of-sequence. Present in the table, unused at inference.
    pub eos: String,
    /// Beginning-of-sequence. Present in the table, unused at inference.
    pub bos: String,
    /// The token interspersed between symbols when `add_blank` is set.
    pub blank: String,
    /// The symbol alphabet, one code point per symbol.
    pub characters: String,
    /// Punctuation, appended after the alphabet rather than mixed into it.
    pub punctuations: String,
    /// Whether duplicates were removed before the table was built.
    #[serde(default)]
    pub is_unique: bool,
    /// Whether the alphabet was sorted before the table was built.
    #[serde(default)]
    pub is_sorted: bool,
}

impl CoquiCharacters {
    /// Rebuilds the reference's symbol table, in the reference's order.
    ///
    /// `BaseCharacters._create_vocab` prepends the four special tokens in the
    /// order blank, bos, eos, pad - each pushing the others down - and appends
    /// the punctuation last. Getting that order wrong shifts every id by a
    /// constant, which produces fluent speech saying something else.
    pub fn vocab(&self) -> Result<Vec<String>, ConfigError> {
        let mut alphabet: Vec<char> = self.characters.chars().collect();
        if self.is_unique {
            if !self.is_sorted {
                // The reference does `list(set(...))`, whose order Python does
                // not define. With a sort after it the result is determined;
                // without one it is not, and guessing would be a coin flip on
                // every id.
                return Err(ConfigError::UnorderedVocab);
            }
            alphabet.sort_unstable();
            alphabet.dedup();
        } else if self.is_sorted {
            alphabet.sort_unstable();
        }

        // Written in the order the reference prepends them, then reversed, so
        // this reads the same way the Python does.
        let mut vocab: Vec<String> = Vec::with_capacity(alphabet.len() + 16);
        for special in [&self.pad, &self.eos, &self.bos, &self.blank] {
            if !special.is_empty() {
                vocab.push(special.clone());
            }
        }
        vocab.extend(alphabet.iter().map(|c| c.to_string()));
        vocab.extend(self.punctuations.chars().map(|c| c.to_string()));
        Ok(vocab)
    }
}

/// The `model_args` block: everything about the model's shape that is written
/// down rather than hardcoded.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CoquiModelArgs {
    /// Symbol count, which must agree with the table
    /// [`CoquiCharacters::vocab`] builds.
    pub num_chars: usize,
    /// Model width.
    pub hidden_channels: usize,
    /// Inner width of the text encoder's convolutional feed-forward.
    pub hidden_channels_ffn_text_encoder: usize,
    /// Attention heads in the text encoder.
    pub num_heads_text_encoder: usize,
    /// Text encoder depth.
    pub num_layers_text_encoder: usize,
    /// Kernel size of both feed-forward convolutions.
    pub kernel_size_text_encoder: usize,
    /// WaveNet convolution kernel size inside the prior flow.
    pub kernel_size_flow: usize,
    /// WaveNet dilation base inside the prior flow.
    pub dilation_rate_flow: usize,
    /// WaveNet layers inside each coupling block.
    pub num_layers_flow: usize,
    /// Which HiFi-GAN residual block the decoder was built from.
    pub resblock_type_decoder: String,
    /// Dilations within each resblock, one list per kernel size.
    pub resblock_dilation_sizes_decoder: Vec<Vec<usize>>,
    /// Kernel size of each multi-receptive-field resblock.
    pub resblock_kernel_sizes_decoder: Vec<usize>,
    /// Transposed-convolution kernel per decoder stage.
    pub upsample_kernel_sizes_decoder: Vec<usize>,
    /// Channels entering the first decoder stage.
    pub upsample_initial_channel_decoder: usize,
    /// One upsample factor per decoder stage.
    pub upsample_rates_decoder: Vec<usize>,
    /// Whether durations are sampled rather than regressed.
    pub use_sdp: bool,
    /// Temperature applied to the prior when sampling.
    pub inference_noise_scale: f32,
    /// Multiplier on predicted durations. Larger is slower.
    pub length_scale: f32,
    /// Temperature applied to the duration predictor when sampling.
    pub inference_noise_scale_dp: f32,
    /// Whether a speaker embedding conditions the model.
    #[serde(default)]
    pub use_speaker_embedding: bool,
    /// Whether an external d-vector conditions the model.
    #[serde(default)]
    pub use_d_vector_file: bool,
    /// Whether a language embedding conditions the model.
    #[serde(default)]
    pub use_language_embedding: bool,
    /// Speakers the checkpoint was trained on.
    #[serde(default)]
    pub num_speakers: usize,
    /// A rate the encoder ran at when it differs from the vocoder's, which
    /// turns on an interpolation of `z` this implementation does not have.
    #[serde(default)]
    pub encoder_sample_rate: Option<u32>,
}

/// A Coqui training run's configuration, read for the dozen fields that
/// describe the model rather than the run.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CoquiConfig {
    /// Which model the run trained. Must be `vits`.
    pub model: String,
    /// Sample rate and framing.
    pub audio: CoquiAudio,
    /// The model's shape.
    pub model_args: CoquiModelArgs,
    /// The symbol table's definition.
    pub characters: CoquiCharacters,
    /// Whether a blank is interspersed between symbols.
    #[serde(default)]
    pub add_blank: bool,
    /// Whether the text was phonemised before tokenisation. See
    /// [`CoquiTokenizer`] for why this matters more than any other field here.
    #[serde(default)]
    pub use_phonemes: bool,
    /// Which phonemiser produced the training transcripts, if any.
    #[serde(default)]
    pub phonemizer: Option<String>,
    /// Which language that phonemiser was asked for.
    #[serde(default)]
    pub phoneme_language: Option<String>,
}

impl CoquiConfig {
    /// Reads a Coqui `config.json`.
    pub fn from_json_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_str(&text)
    }

    /// Parses a Coqui `config.json` already in memory.
    ///
    /// The file carries `Infinity` for two of its training limits, which is
    /// what Python's `json` module writes for `float("inf")` and is not JSON.
    /// Neither field is read here, but the parser still has to get past them,
    /// so they are rewritten before it sees them.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let raw: Self = serde_json::from_str(&repair_non_json_numbers(text))?;
        raw.validate()?;
        Ok(raw)
    }

    /// Rejects a run this implementation cannot reproduce.
    fn validate(&self) -> Result<(), ConfigError> {
        if self.model != "vits" {
            return Err(ConfigError::UnsupportedModel {
                model: self.model.clone(),
            });
        }
        let a = &self.model_args;
        // ResBlock2 is a different graph - two convolutions per dilation rather
        // than four - and would bind against a shape check that happens to pass.
        if a.resblock_type_decoder != "1" {
            return Err(ConfigError::UnsupportedResblock {
                kind: a.resblock_type_decoder.clone(),
            });
        }
        // Every one of these adds a conditioning tensor the forward pass here
        // does not carry, and each would change the output while breaking no
        // shape.
        for (field, on) in [
            ("use_speaker_embedding", a.use_speaker_embedding),
            ("use_d_vector_file", a.use_d_vector_file),
            ("use_language_embedding", a.use_language_embedding),
            ("num_speakers", a.num_speakers > 0),
            ("encoder_sample_rate", a.encoder_sample_rate.is_some()),
        ] {
            if on {
                return Err(ConfigError::UnsupportedConditioning { field });
            }
        }

        let vocab = self.characters.vocab()?;
        if vocab.len() != a.num_chars {
            return Err(ConfigError::VocabMismatch {
                declared: a.num_chars,
                built: vocab.len(),
            });
        }
        Ok(())
    }

    /// Converts to the geometry the forward pass reads.
    ///
    /// Several of the numbers below are not in the file: the reference passes
    /// them as literals where it builds the modules, and they are the constants
    /// at the top of this module. Every one of them is checked against the
    /// checkpoint's own shapes by [`VitsWeights::load_coqui`].
    pub fn to_vits(&self) -> Result<VitsConfig, ConfigError> {
        let a = &self.model_args;
        let cfg = VitsConfig {
            hidden_size: a.hidden_channels,
            num_hidden_layers: a.num_layers_text_encoder,
            num_attention_heads: a.num_heads_text_encoder,
            ffn_dim: a.hidden_channels_ffn_text_encoder,
            ffn_kernel_size: a.kernel_size_text_encoder,
            vocab_size: a.num_chars,
            sampling_rate: self.audio.sample_rate,
            // The prior flow runs at the model's width, not at the posterior
            // encoder's `out_channels`, which is a spectrogram and has no part
            // in synthesis.
            flow_size: a.hidden_channels,
            prior_encoder_num_flows: PRIOR_FLOWS,
            prior_encoder_num_wavenet_layers: a.num_layers_flow,
            wavenet_kernel_size: a.kernel_size_flow,
            wavenet_dilation_rate: a.dilation_rate_flow,
            upsample_rates: a.upsample_rates_decoder.clone(),
            upsample_kernel_sizes: a.upsample_kernel_sizes_decoder.clone(),
            upsample_initial_channel: a.upsample_initial_channel_decoder,
            resblock_kernel_sizes: a.resblock_kernel_sizes_decoder.clone(),
            resblock_dilation_sizes: a.resblock_dilation_sizes_decoder.clone(),
            // `xabe-vits` counts the depthwise-separable stack as "channels
            // plus one"; the reference counts it as three layers. Same stack.
            depth_separable_channels: DDS_LAYERS - 1,
            depth_separable_num_layers: DDS_LAYERS,
            duration_predictor_kernel_size: DURATION_KERNEL,
            duration_predictor_num_flows: DURATION_FLOWS,
            duration_predictor_flow_bins: SPLINE_BINS,
            duration_predictor_tail_bound: SPLINE_TAIL_BOUND,
            window_size: REL_ATTN_WINDOW,
            layer_norm_eps: LAYER_NORM_EPS,
            hidden_act: "relu".to_string(),
            noise_scale: a.inference_noise_scale,
            noise_scale_duration: a.inference_noise_scale_dp,
            // The reference multiplies durations by `length_scale`; this
            // implementation divides by `speaking_rate`. They are reciprocals,
            // and both mean "larger is slower" only after the inversion.
            speaking_rate: 1.0 / a.length_scale,
            leaky_relu_slope: LRELU_SLOPE,
            use_stochastic_duration_prediction: a.use_sdp,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Rewrites the non-finite literals a Coqui config writes so JSON can be read.
///
/// `max_audio_len` and `max_text_len` come out as `Infinity`, which is what
/// Python's `json` module emits for `float("inf")` and which no parser that
/// follows the specification will accept. Both are training limits with no
/// bearing on inference, so they become `null` rather than a number: no field
/// here reads them, and a null cannot be mistaken for a real bound.
///
/// Only a value position is rewritten - the match includes the colon - so a
/// string that happens to contain the word is left alone.
fn repair_non_json_numbers(text: &str) -> String {
    let mut out = text.to_string();
    for literal in ["-Infinity", "Infinity", "NaN"] {
        out = out
            .replace(&format!(": {literal}"), ": null")
            .replace(&format!(":{literal}"), ":null");
    }
    out
}

/// Phoneme string to symbol ids, matching Coqui's `TTSTokenizer`.
///
/// # This tokenizer does not phonemise, and the model needs phonemes
///
/// The reference's pipeline is *clean the text, phonemise it, then look each
/// phoneme up*. The middle step is `pygoruut`, a Go binary whose language data
/// is a Han-to-IPA dictionary plus a learned fallback for words not in it, and
/// it is **not** ported here. Half-porting it would be worse than not having
/// it: an out-of-dictionary word would come out mispronounced rather than
/// missing, which is exactly the failure this workspace is built to refuse.
///
/// Two things produce phonemes instead, and which one applies depends on what
/// the caller has:
///
/// - **From romanisation**, which is what this pipeline produces:
///   `xabe_taigi::poj_to_ipa`. That is a spelling table and not a guess - the
///   translator upstream has already decided how each word is read, so nothing
///   is left to choose. This is the path the engine takes.
/// - **From Han**, which nothing here needs but a person with a corpus might:
///   `tools/phonemize_pygoruut.py`, which runs the reference's own front end.
///
/// # Dropping is silent, as it is on the other path
///
/// A character outside the table is discarded with no unknown id and no change
/// in length. The reference prints a warning the first time; here it is traced.
/// Feeding Han text straight in produces an empty sequence rather than an
/// error, which is why [`CoquiTokenizer::encode`]'s caller checks for one.
#[derive(Debug)]
pub struct CoquiTokenizer {
    /// Every vocabulary entry that is a single code point. The four specials
    /// are not, so they are unreachable through [`Self::encode`].
    chars: FxHashMap<char, i64>,
    /// The id interspersed between symbols. **3 for this checkpoint**, because
    /// the four special tokens come first - not 0, which is the padding.
    blank: i64,
    /// Whether to intersperse at all.
    add_blank: bool,
    /// Size of the symbol table, which the embedding is exactly this wide.
    vocab_size: usize,
}

impl CoquiTokenizer {
    /// Builds the tokenizer from a config's `characters` block.
    pub fn new(cfg: &CoquiConfig) -> Result<Self, TokenizerError> {
        let vocab = cfg
            .characters
            .vocab()
            .map_err(|source| TokenizerError::Vocabulary {
                source: Box::new(source),
            })?;

        let mut chars = FxHashMap::default();
        for (id, token) in vocab.iter().enumerate() {
            let mut cs = token.chars();
            if let (Some(c), None) = (cs.next(), cs.next()) {
                chars.insert(c, id as i64);
            }
        }

        let blank = vocab
            .iter()
            .position(|t| *t == cfg.characters.blank)
            .ok_or_else(|| TokenizerError::NoSuchBlank {
                token: cfg.characters.blank.clone(),
            })? as i64;

        tracing::debug!(
            vocab = vocab.len(),
            symbols = chars.len(),
            blank,
            add_blank = cfg.add_blank,
            "loaded Coqui tokenizer",
        );
        Ok(Self {
            chars,
            blank,
            add_blank: cfg.add_blank,
            vocab_size: vocab.len(),
        })
    }

    /// Size of the symbol table.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// The id interspersed between symbols.
    pub fn blank(&self) -> i64 {
        self.blank
    }

    /// Encodes a phoneme string into symbol ids.
    ///
    /// The input is IPA as the reference's phonemiser writes it, with no
    /// separator between phonemes. Anything the table does not hold is dropped.
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        let mut kept = Vec::with_capacity(phonemes.len());
        let mut dropped = 0usize;
        for c in phonemes.chars() {
            match self.chars.get(&c) {
                Some(id) => kept.push(*id),
                None => dropped += 1,
            }
        }
        if dropped > 0 {
            tracing::debug!(
                dropped,
                "characters outside the symbol table were discarded"
            );
        }

        if !self.add_blank {
            return kept;
        }
        // `intersperse_blank_char` builds `2n + 1` entries only when `n > 0`;
        // an empty input stays empty rather than becoming a lone blank.
        if kept.is_empty() {
            return kept;
        }
        let mut ids = Vec::with_capacity(kept.len() * 2 + 1);
        for id in kept {
            ids.push(self.blank);
            ids.push(id);
        }
        ids.push(self.blank);
        ids
    }
}

/// Fetches a convolution with an explicit shape and a bias.
///
/// Every projection in this checkpoint is a `Conv1d`, including the four in
/// each attention block that 🤗 stores as a `Linear` - so the trailing `1` is
/// present here where the other reader has a two-dimensional shape. The bytes
/// are the same either way; the shape check is not, which is the point.
fn conv<'a>(
    f: &'a PtFile,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<Conv<'a>, WeightError> {
    Ok(Conv {
        weight: f.tensor_shaped(&format!("{prefix}.weight"), &[out_ch, in_ch, k])?,
        bias: Some(f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?),
        out_ch,
        in_ch,
        k,
    })
}

/// Fetches a layer norm, whose parameters this dialect calls `gamma` and
/// `beta` rather than `weight` and `bias`.
///
/// The class is `LayerNorm2`, which transposes and calls
/// `F.layer_norm(x, (channels,), gamma, beta, 1e-5)` - the same arithmetic 🤗's
/// `nn.LayerNorm` does. Note that the *other* `LayerNorm` in the same reference
/// file normalises over the channel axis by hand with `eps=1e-4` and stores its
/// parameters at `[1, C, 1]`; the text encoder does not use it, and the shape
/// check below is what tells them apart.
fn norm<'a>(f: &'a PtFile, prefix: &str, ch: usize) -> Result<Norm<'a>, WeightError> {
    Ok(Norm {
        weight: f.tensor_shaped(&format!("{prefix}.gamma"), &[ch])?,
        bias: f.tensor_shaped(&format!("{prefix}.beta"), &[ch])?,
    })
}

/// The names this checkpoint's weight norm is stored under.
///
/// Torch has spelled it two ways. `weight_norm` up to 2.0 registered `weight_g`
/// and `weight_v` directly; the `parametrizations` rewrite moved them to
/// `parametrizations.weight.original0` and `original1`. This checkpoint uses the
/// second, and both are tried so that an older Coqui save still binds.
fn wn_names(f: &PtFile, prefix: &str) -> (String, String) {
    let modern = format!("{prefix}.parametrizations.weight.original0");
    if f.info(&modern).is_some() {
        return (
            modern,
            format!("{prefix}.parametrizations.weight.original1"),
        );
    }
    (format!("{prefix}.weight_g"), format!("{prefix}.weight_v"))
}

/// Fetches a weight-normalised convolution, unfused.
fn wn_conv<'a>(
    f: &'a PtFile,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<WnConv<'a>, WeightError> {
    let (g, v) = wn_names(f, prefix);
    Ok(WnConv {
        weight_v: f.tensor_shaped(&v, &[out_ch, in_ch, k])?,
        weight_g: f.tensor_shaped(&g, &[out_ch, 1, 1])?,
        bias: f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?,
        out_ch,
        in_ch,
        k,
    })
}

/// Fetches a weight-normalised *transposed* convolution, unfused.
///
/// Separate from [`wn_conv`] for the reason [`conv_transposed`] is separate in
/// the other dialect's loader, plus one more that only matters here. A
/// transposed convolution stores `[in, out, k]`, and weight norm normalises over
/// every axis but the first - so the magnitude has one entry per **input**
/// channel, while the bias still has one per output channel. Binding this
/// through [`wn_conv`] would ask for a `[512, 1, 1]` magnitude under the name of
/// a 256-channel one and fail; getting it wrong the other way, by fusing 512
/// rows against 256 norms, would not fail at all.
///
/// [`conv_transposed`]: crate::weights
fn wn_conv_transposed<'a>(
    f: &'a PtFile,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    k: usize,
) -> Result<WnConv<'a>, WeightError> {
    let (g, v) = wn_names(f, prefix);
    Ok(WnConv {
        weight_v: f.tensor_shaped(&v, &[in_ch, out_ch, k])?,
        weight_g: f.tensor_shaped(&g, &[in_ch, 1, 1])?,
        bias: f.tensor_shaped(&format!("{prefix}.bias"), &[out_ch])?,
        out_ch,
        in_ch,
        k,
    })
}

/// Fetches a dilated depthwise-separable stack of `DDS_LAYERS` levels.
fn dds<'a>(f: &'a PtFile, prefix: &str, ch: usize, k: usize) -> Result<DdsConv<'a>, WeightError> {
    let mut dilated = Vec::with_capacity(DDS_LAYERS);
    let mut pointwise = Vec::with_capacity(DDS_LAYERS);
    let mut norms_1 = Vec::with_capacity(DDS_LAYERS);
    let mut norms_2 = Vec::with_capacity(DDS_LAYERS);
    for i in 0..DDS_LAYERS {
        // Depthwise: `groups=channels`, so the stored input width is 1.
        dilated.push(conv(f, &format!("{prefix}.convs_sep.{i}"), ch, 1, k)?);
        pointwise.push(conv(f, &format!("{prefix}.convs_1x1.{i}"), ch, ch, 1)?);
        norms_1.push(norm(f, &format!("{prefix}.norms_1.{i}"), ch)?);
        norms_2.push(norm(f, &format!("{prefix}.norms_2.{i}"), ch)?);
    }
    Ok(DdsConv {
        dilated,
        pointwise,
        norms_1,
        norms_2,
    })
}

/// Fetches one branch of the duration predictor's flow.
///
/// Index 0 is the elementwise affine; the rest are spline couplings. The `Flip`
/// between them holds no parameters, so the checkpoint stores `num_flows + 1`
/// entries for `num_flows` couplings - the same layout the other reader finds,
/// under different names: `translation` here against `translate` there.
fn duration_flows<'a>(
    f: &'a PtFile,
    prefix: &str,
    cfg: &VitsConfig,
) -> Result<Vec<DurationFlow<'a>>, WeightError> {
    let mut flows = Vec::with_capacity(cfg.duration_predictor_num_flows + 1);
    flows.push(DurationFlow::Affine {
        log_scale: f.tensor_shaped(&format!("{prefix}.0.log_scale"), &[2, 1])?,
        translate: f.tensor_shaped(&format!("{prefix}.0.translation"), &[2, 1])?,
    });

    let hidden = cfg.hidden_size;
    let k = cfg.duration_predictor_kernel_size;
    // Half of the two channels, times three parameters per bin less one shared
    // derivative: `half_channels * (num_bins * 3 - 1)`.
    let params = 3 * cfg.duration_predictor_flow_bins - 1;
    for i in 1..=cfg.duration_predictor_num_flows {
        let p = format!("{prefix}.{i}");
        flows.push(DurationFlow::Spline {
            conv_pre: conv(f, &format!("{p}.pre"), hidden, 1, 1)?,
            conv_dds: dds(f, &format!("{p}.convs"), hidden, k)?,
            conv_proj: conv(f, &format!("{p}.proj"), params, hidden, 1)?,
        });
    }
    Ok(flows)
}

impl<'a> VitsWeights<'a> {
    /// Binds every inference tensor of a Coqui checkpoint.
    ///
    /// Same contract as [`VitsWeights::load`]: returns on the first
    /// disagreement, naming the tensor, and copies nothing. The result is the
    /// identical structure, so every stage in `xabe-tts` runs against it
    /// unchanged.
    ///
    /// `posterior_encoder` and `disc` are not read. The first encodes
    /// ground-truth spectrograms during training and the second is the
    /// discriminator; between them they are 211 of the checkpoint's 949
    /// tensors and neither has a role in synthesis.
    pub fn load_coqui(f: &'a PtFile, cfg: &VitsConfig) -> Result<Self, WeightError> {
        let hidden = cfg.hidden_size;
        let flow_half = cfg.flow_half();

        // ---- text encoder ------------------------------------------------
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let attn = format!("text_encoder.encoder.attn_layers.{i}");
            let rel = [1, cfg.rel_window(), cfg.head_dim()];
            layers.push(EncoderLayer {
                q: conv(f, &format!("{attn}.conv_q"), hidden, hidden, 1)?,
                k: conv(f, &format!("{attn}.conv_k"), hidden, hidden, 1)?,
                v: conv(f, &format!("{attn}.conv_v"), hidden, hidden, 1)?,
                out: conv(f, &format!("{attn}.conv_o"), hidden, hidden, 1)?,
                emb_rel_k: f.tensor_shaped(&format!("{attn}.emb_rel_k"), &rel)?,
                emb_rel_v: f.tensor_shaped(&format!("{attn}.emb_rel_v"), &rel)?,
                // The norms are separate module lists here rather than fields
                // of the layer, so their prefixes do not nest under `attn`.
                norm: norm(
                    f,
                    &format!("text_encoder.encoder.norm_layers_1.{i}"),
                    hidden,
                )?,
                ffn_1: conv(
                    f,
                    &format!("text_encoder.encoder.ffn_layers.{i}.conv_1"),
                    cfg.ffn_dim,
                    hidden,
                    cfg.ffn_kernel_size,
                )?,
                ffn_2: conv(
                    f,
                    &format!("text_encoder.encoder.ffn_layers.{i}.conv_2"),
                    hidden,
                    cfg.ffn_dim,
                    cfg.ffn_kernel_size,
                )?,
                final_norm: norm(
                    f,
                    &format!("text_encoder.encoder.norm_layers_2.{i}"),
                    hidden,
                )?,
            });
        }
        let text_encoder = TextEncoder {
            embed: f.tensor_shaped("text_encoder.emb.weight", &[cfg.vocab_size, hidden])?,
            layers,
            project: conv(f, "text_encoder.proj", 2 * cfg.flow_size, hidden, 1)?,
        };

        // ---- duration predictor ------------------------------------------
        let dk = cfg.duration_predictor_kernel_size;
        let duration_predictor = DurationPredictor {
            conv_pre: conv(f, "duration_predictor.pre", hidden, hidden, 1)?,
            conv_dds: dds(f, "duration_predictor.convs", hidden, dk)?,
            conv_proj: conv(f, "duration_predictor.proj", hidden, hidden, 1)?,
            flows: duration_flows(f, "duration_predictor.flows", cfg)?,
            post_conv_pre: conv(f, "duration_predictor.post_pre", hidden, 1, 1)?,
            post_conv_dds: dds(f, "duration_predictor.post_convs", hidden, dk)?,
            post_conv_proj: conv(f, "duration_predictor.post_proj", hidden, hidden, 1)?,
            post_flows: duration_flows(f, "duration_predictor.post_flows", cfg)?,
        };

        // ---- flow ---------------------------------------------------------
        let mut flow = Vec::with_capacity(cfg.prior_encoder_num_flows);
        for i in 0..cfg.prior_encoder_num_flows {
            let p = format!("flow.flows.{i}");
            let n = cfg.prior_encoder_num_wavenet_layers;
            let mut wavenet = Vec::with_capacity(n);
            for j in 0..n {
                wavenet.push(WaveNetLayer {
                    in_layer: wn_conv(
                        f,
                        &format!("{p}.enc.in_layers.{j}"),
                        2 * hidden,
                        hidden,
                        cfg.wavenet_kernel_size,
                    )?,
                    // The last layer projects to residual only; every earlier
                    // one emits residual and skip, hence the doubled width.
                    res_skip: wn_conv(
                        f,
                        &format!("{p}.enc.res_skip_layers.{j}"),
                        if j + 1 < n { 2 * hidden } else { hidden },
                        hidden,
                        1,
                    )?,
                });
            }
            flow.push(FlowBlock {
                conv_pre: conv(f, &format!("{p}.pre"), hidden, flow_half, 1)?,
                wavenet,
                // Mean-only coupling: `post` emits one half-width mean and no
                // log-scale, which is why this is `flow_half` and not twice it.
                conv_post: conv(f, &format!("{p}.post"), flow_half, hidden, 1)?,
            });
        }

        // ---- decoder -------------------------------------------------------
        //
        // Every convolution here is weight-normalised except the first and the
        // last, which VITS builds with `conv_pre_weight_norm=False` and
        // `conv_post_weight_norm=False`. The 🤗 export fused all of them, which
        // is the one place the two dialects differ by more than a name.
        let stages = cfg.num_upsample_stages();
        let mut upsampler = Vec::with_capacity(stages);
        for s in 0..stages {
            upsampler.push(MaybeWn::Normalised(wn_conv_transposed(
                f,
                &format!("waveform_decoder.ups.{s}"),
                cfg.upsample_in_channels(s),
                cfg.upsample_out_channels(s),
                cfg.upsample_kernel_sizes[s],
            )?));
        }

        let mut resblocks = Vec::with_capacity(cfg.num_resblocks());
        for s in 0..stages {
            let ch = cfg.upsample_out_channels(s);
            for (r, &k) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let idx = s * cfg.resblocks_per_stage() + r;
                let n = cfg.resblock_dilation_sizes[r].len();
                let mut convs1 = Vec::with_capacity(n);
                let mut convs2 = Vec::with_capacity(n);
                for c in 0..n {
                    convs1.push(MaybeWn::Normalised(wn_conv(
                        f,
                        &format!("waveform_decoder.resblocks.{idx}.convs1.{c}"),
                        ch,
                        ch,
                        k,
                    )?));
                    convs2.push(MaybeWn::Normalised(wn_conv(
                        f,
                        &format!("waveform_decoder.resblocks.{idx}.convs2.{c}"),
                        ch,
                        ch,
                        k,
                    )?));
                }
                resblocks.push(ResBlock { convs1, convs2 });
            }
        }

        let last_ch = cfg.upsample_out_channels(stages - 1);
        let decoder = Decoder {
            conv_pre: conv(
                f,
                "waveform_decoder.conv_pre",
                cfg.upsample_initial_channel,
                cfg.flow_size,
                7,
            )?,
            upsampler,
            resblocks,
            // `conv_post_bias=False`, so this is the one convolution in the
            // checkpoint with no bias - as it is in the other dialect.
            conv_post: Conv {
                weight: f.tensor_shaped("waveform_decoder.conv_post.weight", &[1, last_ch, 7])?,
                bias: None,
                out_ch: 1,
                in_ch: last_ch,
                k: 7,
            },
        };

        let bound = Self {
            text_encoder,
            duration_predictor,
            flow,
            decoder,
        };
        tracing::debug!(
            layers = cfg.num_hidden_layers,
            flows = cfg.prior_encoder_num_flows,
            parameters = bound.total_elements(),
            "bound Coqui VITS weights",
        );
        Ok(bound)
    }
}
