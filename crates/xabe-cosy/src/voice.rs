//! A speaker, as four tensors and a noise buffer.
//!
//! # Why a voice is a file and not a wav
//!
//! Everything CosyVoice needs from a reference clip comes out of two ONNX
//! models and a mel frontend: an x-vector from `campplus.onnx`, the clip's own
//! speech tokens from `speech_tokenizer_v3.onnx`, and its mel. All three run
//! **once per voice**, never per utterance, and porting them would be a second
//! engine's worth of work to produce four small tensors.
//!
//! So `tools/make_cosyvoice_voice.py` derives them and writes a bundle, and
//! this reads it. The engine never sees an ONNX runtime, and adding a voice is
//! a one-line script run rather than a code change.
//!
//! # `cfm_noise` is in here and is not a property of the speaker
//!
//! `CausalConditionalCFM.__init__` seeds the global RNG to zero and draws
//! `randn([1, 80, 15000])`, so every voice and every utterance starts its
//! diffusion from the same numbers. It is load-bearing - a different draw is a
//! different mel, not a differently-rounded one - and it rides in the bundle
//! because that is the file the engine already opens.

use crate::CosyError;
use xabe_st::StFile;

/// One speaker, ready for the flow.
#[derive(Debug)]
pub struct Voice {
    /// The reference clip as speech tokens, prepended to what the LLM writes.
    pub prompt_token: Vec<u32>,
    /// Its mel, `[frames, 80]`, position-major.
    pub prompt_feat: Vec<f32>,
    /// How many frames that is.
    pub prompt_frames: usize,
    /// The campplus x-vector, `[192]`.
    pub embedding: Vec<f32>,
    /// The diffusion's starting noise, `[80, cap]`, channel-major.
    pub cfm_noise: Vec<f32>,
    /// How many mel frames of noise there are, which caps an utterance.
    pub noise_frames: usize,
}

impl Voice {
    /// Reads a bundle written by `tools/make_cosyvoice_voice.py`.
    pub fn open(path: &std::path::Path, mel_dim: usize) -> Result<Self, CosyError> {
        let f = StFile::open(path)?;
        let shape = |name: &str| -> Result<Vec<usize>, CosyError> {
            f.info(name)
                .map(|i| i.shape.clone())
                .ok_or_else(|| CosyError::MissingTensor(name.into()))
        };

        // The tokens were written through float32, like everything else the
        // capture and the tools produce, so they are rounded rather than cast:
        // 2715.9999 is token 2716.
        let prompt_token: Vec<u32> = f
            .tensor("flow_prompt_speech_token")?
            .iter()
            .map(|v| v.round() as u32)
            .collect();

        let feat_shape = shape("prompt_speech_feat")?;
        let frames = match feat_shape.as_slice() {
            [frames, bands] if *bands == mel_dim => *frames,
            other => {
                return Err(CosyError::Shape {
                    name: "prompt_speech_feat".into(),
                    found: other.to_vec(),
                    want: vec![0, mel_dim],
                });
            }
        };

        let noise_shape = shape("cfm_noise")?;
        let noise_frames = match noise_shape.as_slice() {
            [bands, n] if *bands == mel_dim => *n,
            other => {
                return Err(CosyError::Shape {
                    name: "cfm_noise".into(),
                    found: other.to_vec(),
                    want: vec![mel_dim, 0],
                });
            }
        };

        // Two mel frames per speech token, and the bundle's own two halves have
        // to agree on it - a bundle whose mel is a frame longer than its tokens
        // justify produces a flow whose condition is offset by half a token for
        // the whole utterance, which sounds like a slur rather than an error.
        if frames != prompt_token.len() * 2 {
            return Err(CosyError::Geometry {
                what: "the bundle's prompt mel and prompt tokens disagree",
                got: frames,
                want: prompt_token.len() * 2,
            });
        }

        Ok(Self {
            prompt_feat: f.tensor("prompt_speech_feat")?.to_vec(),
            prompt_frames: frames,
            embedding: f.tensor("flow_embedding")?.to_vec(),
            cfm_noise: f.tensor("cfm_noise")?.to_vec(),
            noise_frames,
            prompt_token,
        })
    }

    /// The longest utterance this bundle's noise can cover, in mel frames.
    ///
    /// Named rather than left to fail inside the solver: the noise is 300
    /// seconds at 50 Hz, so this is 15,000 frames and nothing a sentence will
    /// reach - but a caller feeding it a paragraph deserves the number.
    pub fn max_frames(&self) -> usize {
        self.noise_frames
    }
}
