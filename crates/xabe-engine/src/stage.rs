//! Which stages this process owns, and how it was told to satisfy them.
//!
//! The whole design is one piece of symmetry: every stage can be satisfied
//! either by loading a model *here* or by delegating to another process over
//! HTTP, and nothing downstream of this module can tell which. That is what
//! makes one binary serve as a monolith, as a single-stage worker, or as
//! anything in between - the six-process topology the Python pipeline runs
//! today is just one configuration of the same flags.
//!
//! This module refuses to know how either half works. It resolves flags into
//! [`Stages`] and rejects combinations that cannot mean anything, by name, at
//! startup. It does not open a file or a socket.
//!
//! Start at [`Stages::resolve`]; the rules it enforces are each a named
//! variant of [`StageError`].

use std::fmt;
use std::path::PathBuf;

/// The stages the engine knows about.
///
/// `Llm` was once absent from this list on purpose - the plan said the chat
/// model stays in llama.cpp and there is no `--llm-model`. It is here now, and
/// it is the same symmetry as every other stage rather than a special case:
/// `--llm-url` still delegates to llama-server, `--llm-model` runs the weights
/// here, and nothing downstream can tell which. `docs/MILESTONES.md` carries
/// the reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Speech to text.
    Asr,
    /// Voice activity detection, which gates the ASR.
    Vad,
    /// Text to speech.
    Tts,
    /// Mandarin to Taigi.
    Translator,
    /// The chat model that writes the reply.
    Llm,
}

impl Kind {
    /// Whether this stage has a CUDA implementation at all.
    ///
    /// The VAD does not, and is not going to: it is 15 tensors and a
    /// millisecond of CPU per clip, so the transfer would cost more than the
    /// arithmetic. Saying so is better than accepting `--vad-device 0` and then
    /// running on the CPU anyway, which is what the flag surface did first and
    /// what made the startup log claim `device=cuda:0` for a CPU stage.
    pub fn has_cuda(self) -> bool {
        !matches!(self, Kind::Vad)
    }

    /// Whether this stage has a CPU implementation at all.
    ///
    /// The ASR, the translator and the chat model do not. One 30-second
    /// window is about 2.2 TFLOP through Whisper's encoder alone, the 13 B
    /// translator is 26 GFLOP a token and the 8 B chat model 16, against
    /// scalar kernels that run at something under 2 GFLOP/s. None is a slow
    /// option; all three are fictional ones. The
    /// mirror of [`Kind::has_cuda`], and for the mirror-image reason:
    /// accepting `--asr-device cpu` and then taking twenty minutes is worse
    /// than saying so at preflight.
    pub fn has_cpu(self) -> bool {
        !matches!(self, Kind::Asr | Kind::Translator | Kind::Llm)
    }

    /// Where this stage runs when no device is named.
    pub fn default_device(self) -> Device {
        if self.has_cuda() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }

    /// The flag prefix, so an error can quote the flag the user actually typed.
    pub fn flag(self) -> &'static str {
        match self {
            Kind::Asr => "asr",
            Kind::Vad => "vad",
            Kind::Tts => "tts",
            Kind::Translator => "translator",
            Kind::Llm => "llm",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.flag())
    }
}

/// Where a device ordinal came from, or that the CPU was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// The scalar reference path.
    Cpu,
    /// A CUDA device ordinal.
    Cuda(usize),
}

impl Device {
    /// Parses `cpu` or a device ordinal.
    pub fn parse(s: &str) -> Option<Device> {
        if s == "cpu" {
            return Some(Device::Cpu);
        }
        s.parse().ok().map(Device::Cuda)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(o) => write!(f, "cuda:{o}"),
        }
    }
}

/// How one stage is satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// Not configured. The engine does not do this.
    Off,
    /// Run it here, from a checkpoint on this machine.
    Local {
        /// Directory holding the checkpoint, or the checkpoint itself.
        path: PathBuf,
        /// Where to run it.
        device: Device,
    },
    /// Delegate it to another process.
    Remote {
        /// Base URL of a process serving this stage.
        url: String,
    },
}

impl Stage {
    /// Whether the stage is configured at all.
    pub fn is_on(&self) -> bool {
        !matches!(self, Stage::Off)
    }

    /// Where the stage runs, when it runs here.
    ///
    /// `None` for a stage that is off or delegated - not `Device::Cpu`, which
    /// would silently put a model on the wrong side of the PCIe bus.
    pub fn device(&self) -> Option<Device> {
        match self {
            Stage::Local { device, .. } => Some(*device),
            Stage::Off | Stage::Remote { .. } => None,
        }
    }
}

/// Every stage, resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stages {
    /// Speech to text.
    pub asr: Stage,
    /// Voice activity detection.
    pub vad: Stage,
    /// Text to speech.
    pub tts: Stage,
    /// Mandarin to Taigi.
    pub translator: Stage,
    /// The chat model that writes the reply.
    pub llm: Stage,
}

/// One stage's two flags, before resolution.
#[derive(Debug, Clone, Default)]
pub struct Requested {
    /// `--<stage>-model`.
    pub model: Option<PathBuf>,
    /// `--<stage>-url`.
    pub url: Option<String>,
    /// `--<stage>-device`.
    pub device: Option<String>,
}

/// A combination of flags that cannot mean anything.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StageError {
    /// Both halves of a stage's symmetry were given.
    ///
    /// Picking one silently would make a typo in a URL look like a working
    /// local model, or the reverse, and the two have very different latencies.
    #[error("--{0}-model and --{0}-url are alternatives; give one, not both")]
    Both(Kind),

    /// A device was named for a stage that is not running here.
    ///
    /// Almost always a delegated stage that was meant to be local: the device
    /// flag is the half that got typed and the model flag is the half that
    /// did not.
    #[error("--{0}-device applies only to --{0}-model, and this run has --{0}-url")]
    DeviceWithoutModel(Kind),

    /// A device string that is neither `cpu` nor an ordinal.
    #[error("--{0}-device must be `cpu` or a CUDA device ordinal, got `{1}`")]
    BadDevice(Kind, String),

    /// A CUDA device was asked for by a stage that only runs on the CPU.
    #[error("--{0}-device must be `cpu`; the {0} stage has no CUDA implementation")]
    CpuOnly(Kind),

    /// The CPU was asked for by a stage that only runs on a card.
    #[error("--{0}-device must be a CUDA device ordinal; the {0} stage has no CPU implementation")]
    GpuOnly(Kind),

    /// Voice activity detection with nothing to gate and nothing to read.
    ///
    /// VAD alone is meaningful over `--in`, where it prints segments. Served,
    /// it is only ever a gate in front of an ASR, so a VAD-only server is a
    /// flag that was meant for a different process.
    #[error("--vad-* with --serve needs an ASR stage to gate: add --asr-model or --asr-url")]
    VadWithoutAsr,

    /// No stage at all.
    #[error(
        "no stage configured; give at least one of \
         --asr-model/--asr-url, --vad-model/--vad-url, \
         --tts-model/--tts-url, --translator-model/--translator-url, \
         --llm-model/--llm-url"
    )]
    Nothing,
}

impl Stages {
    /// Resolves the flag pairs, rejecting combinations that cannot mean anything.
    ///
    /// `serving` changes exactly one rule - see [`StageError::VadWithoutAsr`] -
    /// because a VAD is a gate when served and a tool when run over a file.
    pub fn resolve(
        asr: &Requested,
        vad: &Requested,
        tts: &Requested,
        translator: &Requested,
        llm: &Requested,
        serving: bool,
    ) -> Result<Stages, StageError> {
        let stages = Stages {
            asr: one(Kind::Asr, asr)?,
            vad: one(Kind::Vad, vad)?,
            tts: one(Kind::Tts, tts)?,
            translator: one(Kind::Translator, translator)?,
            llm: one(Kind::Llm, llm)?,
        };

        if !stages.any_on() {
            return Err(StageError::Nothing);
        }
        if serving && stages.vad.is_on() && !stages.asr.is_on() {
            return Err(StageError::VadWithoutAsr);
        }
        Ok(stages)
    }

    /// Whether any stage at all is configured.
    pub fn any_on(&self) -> bool {
        self.asr.is_on()
            || self.vad.is_on()
            || self.tts.is_on()
            || self.translator.is_on()
            || self.llm.is_on()
    }

    /// Whether this process can answer a whole voice turn by itself.
    ///
    /// The web UI needs speech in, a reply, and speech out. The translator is
    /// not in the list: `--direct-taigi` takes it out of the reply path, and
    /// that is the default the measured pipeline runs.
    pub fn full_chain(&self) -> bool {
        self.asr.is_on() && self.tts.is_on() && self.llm.is_on()
    }

    /// Every configured stage, for logging what this process turned out to be.
    pub fn summary(&self) -> Vec<(Kind, &Stage)> {
        [
            (Kind::Asr, &self.asr),
            (Kind::Vad, &self.vad),
            (Kind::Tts, &self.tts),
            (Kind::Translator, &self.translator),
            (Kind::Llm, &self.llm),
        ]
        .into_iter()
        .filter(|(_, s)| s.is_on())
        .collect()
    }
}

/// Resolves one stage's flag pair.
fn one(kind: Kind, r: &Requested) -> Result<Stage, StageError> {
    match (&r.model, &r.url) {
        (Some(_), Some(_)) => Err(StageError::Both(kind)),
        (None, Some(url)) => {
            if r.device.is_some() {
                return Err(StageError::DeviceWithoutModel(kind));
            }
            Ok(Stage::Remote { url: url.clone() })
        }
        (Some(path), None) => {
            let device = match r.device.as_deref() {
                None => kind.default_device(),
                Some(s) => {
                    Device::parse(s).ok_or_else(|| StageError::BadDevice(kind, s.to_string()))?
                }
            };
            if device != Device::Cpu && !kind.has_cuda() {
                return Err(StageError::CpuOnly(kind));
            }
            if device == Device::Cpu && !kind.has_cpu() {
                return Err(StageError::GpuOnly(kind));
            }
            Ok(Stage::Local {
                path: path.clone(),
                device,
            })
        }
        (None, None) => {
            if r.device.is_some() {
                return Err(StageError::DeviceWithoutModel(kind));
            }
            Ok(Stage::Off)
        }
    }
}
