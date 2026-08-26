//! Audio containers and sample handling, with no opinion about models.
//!
//! Everything here takes flat slices and a sample rate. It does not know
//! whether the samples came from a microphone, a decoder or a file on disk,
//! and it does not know which model is about to consume them - that knowledge
//! belongs one layer up, in the crate that owns the stage.
//!
//! It exists because the engine now has more than one stage that touches
//! audio. The TTS writes it, the ASR and the VAD read it, and the server moves
//! it between them; a WAV writer that lives inside the synthesiser cannot be
//! reached by any of the others without pointing a dependency edge the wrong
//! way. See `docs/ARCHITECTURE.md` for the one-way rule this is obeying.
//!
//! Start at [`wav`].

mod error;
mod wav;

pub use error::AudioError;
pub use wav::{Wav, parse_wav, read_wav, wav_bytes, write_wav};
