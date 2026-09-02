//! Opening a checkpoint pair and speaking with it.
//!
//! Start here. [`Taco::open`] binds both files and refuses a geometry it cannot
//! run; [`Taco::synthesize`] takes a line of POJ or Tâi-lô and returns samples
//! at [`Taco::sample_rate`].

use crate::clock::{Clock, Timings};
use crate::model::Rng;
use crate::text::{Tokenizer, poj_to_tlpa, with_gate_cue};
use crate::weights::{Glow, Taco2};
use crate::{Config, TacoError, model, vocoder};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use xabe_cuda::Gpu;
use xabe_st::StFile;

/// The three files a converted checkpoint directory holds.
pub const FILES: [&str; 3] = [
    "tacotron2.safetensors",
    "waveglow.safetensors",
    "tacotron2.json",
];

/// Tacotron2 and WaveGlow, on one device.
pub struct Taco {
    gpu: Gpu,
    cfg: Config,
    taco: Taco2,
    glow: Glow,
    tok: Tokenizer,
    sigma: f32,
    /// Bumped per utterance so repeated calls are not identical, while a given
    /// starting seed still reproduces a whole session.
    seed: AtomicU64,
}

impl Taco {
    /// Binds both checkpoints on CUDA device `ordinal`.
    ///
    /// CUDA only, and deliberately: WaveGlow is 87.9 M parameters of dilated
    /// convolution run at the sample rate, so a scalar path would be a
    /// configuration that starts and then never answers.
    pub fn open(
        dir: &Path,
        ordinal: usize,
        sigma: Option<f32>,
        seed: u64,
    ) -> Result<Self, TacoError> {
        for f in FILES {
            let p = dir.join(f);
            if !p.is_file() {
                return Err(TacoError::Missing {
                    what: match f {
                        "tacotron2.safetensors" => "the Tacotron2 weights",
                        "waveglow.safetensors" => "the WaveGlow weights",
                        _ => "the geometry",
                    },
                    path: p.display().to_string(),
                });
            }
        }

        let cfg = Config::open(&dir.join("tacotron2.json"))?;
        let gpu = Gpu::open(ordinal)?;

        let tf = StFile::open(dir.join("tacotron2.safetensors"))?;
        let taco = Taco2::open(&tf, &gpu, &cfg)?;
        let gf = StFile::open(dir.join("waveglow.safetensors"))?;
        let glow = Glow::open(&gf, &gpu, &cfg)?;

        let tok = Tokenizer::new(&cfg.symbols);
        let sigma = sigma.unwrap_or(cfg.sigma);
        tracing::info!(
            symbols = cfg.symbols.len(),
            rate = cfg.sampling_rate,
            flows = cfg.n_flows,
            sigma,
            "tacotron2 ready"
        );
        Ok(Self {
            gpu,
            cfg,
            taco,
            glow,
            tok,
            sigma,
            seed: AtomicU64::new(seed),
        })
    }

    /// 22050 Hz for the published checkpoint.
    pub fn sample_rate(&self) -> usize {
        self.cfg.sampling_rate
    }

    /// The geometry both halves were bound against.
    pub fn geometry(&self) -> &Config {
        &self.cfg
    }

    /// The symbol ids a line becomes, or `None` if none of it is speakable.
    ///
    /// The encoder convolutions are five wide and cannot see a shorter sequence
    /// at all, so a short line is padded - the reference's own fix, and pad is
    /// id 0.
    fn tokens(&self, converted: &str) -> Option<Vec<i64>> {
        let (mut ids, dropped) = self.tok.encode(converted);
        if dropped > 0 {
            tracing::debug!(dropped, text = %converted, "symbols outside the alphabet");
        }
        if ids.is_empty() {
            return None;
        }
        while ids.len() < self.cfg.encoder_kernel {
            ids.push(0);
        }
        Some(ids)
    }

    /// The encoder's output for a line: `[tokens, 512]`, and the token count.
    ///
    /// Public because it is the half of this model that is deterministic, and
    /// therefore the half a differential test can pin against the reference.
    /// The decoder and the vocoder are both stochastic by design and can only
    /// be compared by replaying their draws.
    pub fn encoder(&self, text: &str) -> Result<(Vec<f32>, usize), TacoError> {
        let converted = poj_to_tlpa(text);
        let Some(ids) = self.tokens(&converted) else {
            return Ok((Vec::new(), 0));
        };
        let memory = model::encode(&self.gpu, &self.taco, &self.cfg, &ids)?;
        Ok((self.gpu.download(&memory)?, ids.len()))
    }

    /// Speaks one line.
    ///
    /// POJ is transliterated to the Tâi-lô-with-digits the model reads; input
    /// that is already numeric passes through. An empty result is returned for
    /// text that tokenises to nothing, rather than a sequence of pad symbols
    /// that would synthesise a quarter second of nothing.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, TacoError> {
        Ok(self.run(text, Clock::off())?.0)
    }

    /// The same, with per-stage wall time and the decoder's step count.
    ///
    /// Timing synchronises after every stage, so these numbers are a breakdown
    /// of a run that is not quite the run [`Taco::synthesize`] does. Use them to
    /// find where the time is, and the total from an untimed call to say how
    /// much there is.
    pub fn synthesize_timed(&self, text: &str) -> Result<(Vec<f32>, Timings, usize), TacoError> {
        let (audio, marks, steps) = self.run(text, Clock::on())?;
        Ok((audio, marks, steps))
    }

    fn run(&self, text: &str, mut clock: Clock) -> Result<(Vec<f32>, Timings, usize), TacoError> {
        // The cue goes on *after* the conversion, because the conversion is
        // what folds a full-width mark into one the cue recognises - the other
        // order appends a stop to a line that already ended in one and speaks
        // `?.`. And it goes on here rather than in `encoder`, which is pinned
        // against a captured oracle and must transform text exactly as the
        // reference does. See `text::with_gate_cue`.
        let tlpa = poj_to_tlpa(text);
        let converted = with_gate_cue(&tlpa);
        let Some(ids) = self.tokens(&converted) else {
            if !clock.enabled() {
                tracing::warn!(text = %text, "nothing in this clause is in the alphabet");
            }
            return Ok((Vec::new(), Vec::new(), 0));
        };

        let mut rng = Rng::new(
            self.seed
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1),
        );
        let mel = model::synthesize(
            &self.gpu,
            &self.taco,
            &self.cfg,
            &ids,
            &mut rng,
            &mut clock,
            model::GATE_LOOKAHEAD,
        )?;
        if !mel.stopped {
            tracing::warn!(frames = mel.frames, text = %converted, "no stop token");
        }

        let audio = vocoder::infer(
            &self.gpu, &self.glow, &self.cfg, &mel.data, mel.frames, self.sigma, &mut rng,
            &mut clock,
        )?;

        // The reference's peak normalisation, in floating point rather than to
        // int16 full scale. The floor keeps a near-silent clause from being
        // amplified into hiss.
        let peak = audio.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(0.01);
        let steps = clock.steps;
        Ok((
            audio.into_iter().map(|v| v / peak).collect(),
            clock.into_marks(),
            steps,
        ))
    }
}

#[cfg(test)]
mod tests {
    //! The batched gate read against the frame-by-frame one, on the real
    //! checkpoint: the same seed, the same line, the mel frames bit for bit
    //! and the same stop. Needs `models/tts/tacotron2-nan` and
    //! `XABE_TEST_DEVICE`; prints `SKIP` without them.

    use super::*;
    use crate::model::{self, Rng};
    use std::path::PathBuf;

    fn mel(taco: &Taco, text: &str, seed: u64, lookahead: usize) -> (Vec<f32>, usize, bool) {
        mel_with(taco, &taco.taco, text, seed, lookahead)
    }

    fn mel_with(
        taco: &Taco,
        weights: &Taco2,
        text: &str,
        seed: u64,
        lookahead: usize,
    ) -> (Vec<f32>, usize, bool) {
        let ids = taco
            .tokens(&with_gate_cue(&poj_to_tlpa(text)))
            .expect("the line is in the alphabet");
        let mut rng = Rng::new(seed);
        let mut clock = Clock::off();
        let m = model::synthesize(
            &taco.gpu, weights, &taco.cfg, &ids, &mut rng, &mut clock, lookahead,
        )
        .expect("a synthesis");
        let data = taco.gpu.download(&m.data).expect("the mel");
        (
            data[..m.frames * taco.cfg.n_mel].to_vec(),
            m.frames,
            m.stopped,
        )
    }

    fn checkpoint() -> Option<(PathBuf, usize)> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tts/tacotron2-nan");
        let Some(dev) = std::env::var("XABE_TEST_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
        else {
            println!("SKIP: set XABE_TEST_DEVICE");
            return None;
        };
        if !dir.is_dir() {
            println!("SKIP: no {}", dir.display());
            return None;
        }
        Some((dir, dev))
    }

    /// The decoder with its LSTM weights at f16 against the same decoder at
    /// f32: the same seed, the same masks, the same line. The weights are
    /// f16-exact in the published file - see `Lstm` - so what this measures
    /// is the half-width mat-vec's accumulation order over a recurrence of
    /// hundreds of steps: 5.7e-6 and 6.0e-6 on mels of span 10, on the two
    /// lines below. The bound is an order of magnitude above that.
    #[test]
    fn the_half_width_decoder_says_what_the_full_width_one_says() {
        let Some((dir, dev)) = checkpoint() else {
            return;
        };
        let taco = Taco::open(&dir, dev, None, 7).expect("the checkpoint");
        let tf = StFile::open(dir.join("tacotron2.safetensors")).expect("the weights");
        let full = Taco2::open_with(&tf, &taco.gpu, &taco.cfg, false).expect("the f32 decoder");
        for text in ["Tâi-lâm ū chiok chē hó-chia̍h--ê,", "Lí hó."] {
            let (a, fa, sa) = mel_with(&taco, &full, text, 12345, model::GATE_LOOKAHEAD);
            let (b, fb, sb) = mel(&taco, text, 12345, model::GATE_LOOKAHEAD);
            assert_eq!((fa, sa), (fb, sb), "{text}: the stop");
            let (mut worst, mut span) = (0.0f32, 0.0f32);
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
                span = span.max(x.abs());
                dot += f64::from(*x) * f64::from(*y);
                na += f64::from(*x).powi(2);
                nb += f64::from(*y).powi(2);
            }
            let cosine = dot / (na.sqrt() * nb.sqrt());
            println!(
                "{text}: {fa} frames, max-abs {worst:.3e} of span {span:.2}, cosine {cosine:.7}"
            );
            assert!(
                worst < 1e-4,
                "{text}: max-abs {worst:.3e} against span {span:.2}"
            );
            assert!(
                cosine > 0.999_999,
                "{text}: cosine {cosine:.7} - a different mel"
            );
        }
    }

    #[test]
    fn the_batched_gate_read_produces_the_frames_the_frame_by_frame_one_did() {
        let Some((dir, dev)) = checkpoint() else {
            return;
        };
        let taco = Taco::open(&dir, dev, None, 7).expect("the checkpoint");
        for text in ["Tâi-lâm ū chiok chē hó-chia̍h--ê,", "Lí hó."] {
            let (a, fa, sa) = mel(&taco, text, 12345, 1);
            let (b, fb, sb) = mel(&taco, text, 12345, model::GATE_LOOKAHEAD);
            assert_eq!((fa, sa), (fb, sb), "{text}: the stop");
            assert_eq!(a, b, "{text}: the frames");
            println!("{text}: {fa} frames, identical");
        }
    }
}
