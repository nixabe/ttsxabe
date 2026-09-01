//! The orthographies of Taiwanese Hokkien, and the conversions between them.
//!
//! # Why this crate exists
//!
//! Three checkpoints in this workspace read this language and no two of them
//! read it the same way:
//!
//! | model | reads |
//! | --- | --- |
//! | `mms-tts-nan` | Pe̍h-ōe-jī, tones as diacritics |
//! | Tacotron2 | Tâi-lô, tones as trailing digits |
//! | Coqui VITS | IPA, tones as Chao letters |
//!
//! The translator upstream emits exactly one of those - POJ - so something has
//! to convert. That something used to be a private function inside `xabe-taco`,
//! which was right while one model needed it. Two do now, and a crate below
//! both is the boundary; an edge from `xabe-tts` to `xabe-taco` would not be.
//!
//! # What is (and isn't) here
//!
//! Spelling. This crate knows that POJ's `chh` is Tâi-lô's `tsh` and that
//! Tâi-lô's `tsh` is IPA's `tsʰ`; it has no idea what a phoneme sounds like,
//! what a checkpoint is, or what any of it will be used for. No dependencies
//! but `tracing`, and no errors: every function converts what it recognises and
//! reports what it did not.
//!
//! # It is not a grapheme-to-phoneme converter
//!
//! Nothing here reads Han characters. Going from 你好 to a pronunciation needs a
//! dictionary and a decision about which reading a character takes, and that
//! decision is not spelling - see `docs/MODEL.md`. Romanisation has already
//! made it, which is the whole reason this crate can be a table.
//!
//! Start at [`poj_to_tailo`], then [`tailo_to_ipa`].

mod ipa;
mod poj;

pub use ipa::{Phonemes, poj_to_ipa, tailo_to_ipa};
pub use poj::poj_to_tailo;
