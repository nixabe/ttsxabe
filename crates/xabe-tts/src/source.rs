//! Which checkpoint a model was read from, and which symbol table goes with it.
//!
//! # Why this exists
//!
//! Two published VITS checkpoints speak Taiwanese Hokkien and they are the same
//! architecture: `facebook/mms-tts-nan`, a 🤗 safetensors export at 16 kHz over
//! 48 letters of Pe̍h-ōe-jī, and `neurlang/coqui-vits-suisiann-minnan-hokkien`,
//! a Coqui trainer's `.pth` at 22.05 kHz over 137 IPA phonemes. Every stage
//! from the text encoder to the vocoder is identical arithmetic on identically
//! shaped tensors, so there is one forward pass and not two.
//!
//! What differs is the container the tensors arrived in and the table the
//! symbols came from. Both are decided once, at load, and this module is where
//! that decision is kept - so no stage below has to ask which model it is
//! running.
//!
//! Neither enum is a plug-in point. Adding a third would mean a third
//! published checkpoint of this architecture, and the compiler would name every
//! place that has to learn about it.

use std::path::Path;

use xabe_pt::PtFile;
use xabe_st::StFile;
use xabe_vits::{CoquiConfig, CoquiTokenizer, Tokenizer, VitsConfig, VitsWeights, WeightError};

use crate::synthesize::SynthesisError;

/// A checkpoint, still mapped, in whichever container it shipped in.
#[derive(Debug)]
pub enum Source {
    /// A 🤗 safetensors export, read by `xabe-st`.
    Huggingface(StFile),
    /// A Coqui trainer's torch save, read by `xabe-pt`.
    Coqui(PtFile),
}

impl Source {
    /// Binds the inference tensors, shape-checked against `cfg`.
    ///
    /// Binding is a few hundred lookups into an already-parsed directory and
    /// copies nothing, so it is done per call rather than stored - which would
    /// make the owner a self-referential struct for no measurable gain against
    /// a decoder that takes seconds.
    pub fn weights(&self, cfg: &VitsConfig) -> Result<VitsWeights<'_>, WeightError> {
        match self {
            Self::Huggingface(f) => VitsWeights::load(f, cfg),
            Self::Coqui(f) => VitsWeights::load_coqui(f, cfg),
        }
    }
}

/// The symbol table a model's embedding is indexed by.
#[derive(Debug)]
pub enum Symbols {
    /// 🤗 `VitsTokenizer`: lower-case, drop, intersperse a blank at id 0. The
    /// input is romanised text.
    Huggingface(Tokenizer),
    /// Coqui `TTSTokenizer`: drop, intersperse a blank at id 3. The input is
    /// **IPA phonemes**, not romanisation. `xabe-taigi` transliterates the
    /// pipeline's POJ into them; this crate takes what it is given.
    Coqui(CoquiTokenizer),
}

impl Symbols {
    /// Encodes an input into symbol ids.
    ///
    /// What "input" means differs between the two, and the difference is not
    /// cosmetic: the 🤗 path takes Pe̍h-ōe-jī and the Coqui path takes IPA.
    /// Handing either the other's input produces a short sequence or an empty
    /// one rather than an error, so the conversion belongs to the caller and is
    /// `xabe_taigi::poj_to_ipa`. This crate deliberately does not do it
    /// itself: the differential tests feed captured phonemes straight in, and a
    /// transliteration hidden inside `encode` would silently rewrite them.
    pub fn encode(&self, text: &str) -> Vec<i64> {
        match self {
            Self::Huggingface(t) => t.encode(text),
            Self::Coqui(t) => t.encode(text),
        }
    }

    /// Size of the symbol table, which the embedding is exactly this wide.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Huggingface(t) => t.vocab_size(),
            Self::Coqui(t) => t.vocab_size(),
        }
    }

    /// Which dialect this is, for traces and for `--help` text.
    pub fn dialect(&self) -> &'static str {
        match self {
            Self::Huggingface(_) => "huggingface",
            Self::Coqui(_) => "coqui",
        }
    }
}

/// Opens a Coqui model directory: `best_model.pth` beside its `config.json`.
///
/// Returns the geometry, the symbol table and the mapped checkpoint, all
/// validated against each other. Both callers - the CPU [`crate::Synthesizer`]
/// and the CUDA [`crate::GpuModel`] - go through here so that a mismatch is
/// rejected in one place.
pub fn open_coqui(dir: &Path) -> Result<(VitsConfig, Symbols, Source), SynthesisError> {
    let raw = CoquiConfig::from_json_path(dir.join("config.json"))?;
    let cfg = raw.to_vits()?;
    let tok = CoquiTokenizer::new(&raw)?;
    let file = PtFile::open_section(dir.join("best_model.pth"), COQUI_SECTION)?;

    // Bound once here purely to fail early: a checkpoint that does not match
    // its config should be rejected at open, not on the first synthesis.
    VitsWeights::load_coqui(&file, &cfg)?;

    if !raw.use_phonemes {
        // Not fatal - the model still runs - but the input this crate expects
        // is whatever the training transcripts were, and if that was graphemes
        // then feeding it phonemes is the mistake rather than the reverse.
        tracing::warn!("this checkpoint was trained on graphemes, not phonemes");
    }
    tracing::info!(
        model = %dir.join("best_model.pth").display(),
        sample_rate = cfg.sampling_rate,
        symbols = cfg.vocab_size,
        phonemizer = raw.phonemizer.as_deref().unwrap_or("none"),
        language = raw.phoneme_language.as_deref().unwrap_or("none"),
        "loaded Coqui model",
    );
    Ok((cfg, Symbols::Coqui(tok), Source::Coqui(file)))
}

/// The key a Coqui trainer stores the weights under, beside the optimiser.
const COQUI_SECTION: &str = "model";
