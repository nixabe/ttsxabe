//! When a turn starts and when it ends.
//!
//! This is the policy that decides, from a stream of frame energies, whether
//! somebody is speaking and whether they have finished. It was tuned against
//! real speech over many turns, and every constant below is a fix for a
//! specific observed failure rather than a round number.
//!
//! It lives in Rust rather than in the page for two reasons. The numbers are
//! now in one place and reach every client through `GET /api/config`, so tuning
//! is a restart rather than an edit to an HTML file. And the state machine is
//! testable: [`Endpointer`] can be run over a synthetic or recorded energy
//! trace and asserted on, which the JavaScript version never could be.
//!
//! The frame-by-frame *execution* is still in the browser, because sending
//! every 4096-sample frame over the socket would cost more than it saves. When
//! the engine owns a VAD of its own (plan phase 3) the same [`Endpointer`] runs
//! server-side over Silero's probabilities instead of over energy, which is why
//! it takes a scalar per frame rather than audio.

use serde::{Deserialize, Serialize};

/// The turn-taking constants, as sent to the page.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TurnPolicy {
    /// Sample rate the client should send.
    pub rate: u32,
    /// Frame energy above which speech may be starting.
    pub vad_start: f32,
    /// Frame energy above which speech is still going.
    pub vad_stop: f32,
    /// Silence that arms the end of a turn.
    pub silence_ms: u64,
    /// Further silence that commits it.
    pub grace_ms: u64,
    /// Shortest utterance worth sending.
    pub min_ms: u64,
    /// Longest turn before it is cut.
    pub max_ms: u64,
    /// Consecutive loud frames needed to open a turn.
    pub onset_frames: u32,
    /// Loud audio needed inside a turn before it is worth transcribing.
    pub voiced_ms: u64,
}

impl Default for TurnPolicy {
    fn default() -> Self {
        TurnPolicy {
            rate: 16_000,
            // 0.035, not the 0.012 this started at. That sat at room-noise
            // level, so the microphone fired on silence and Whisper
            // hallucinated a sentence out of it - the failure the whole
            // three-layer hallucination defence exists to prevent.
            vad_start: 0.035,
            // Lower than vad_start on purpose: hysteresis. A single threshold
            // chops a turn into pieces at every unvoiced consonant.
            vad_stop: 0.018,
            // A pause only *arms* the end of turn. Committing on silence_ms
            // alone cut people off at natural mid-sentence pauses, sending half
            // a question: "台北的天氣…嗯…今天如何".
            silence_ms: 700,
            // ...and grace_ms more silence commits it. So finishing a turn
            // costs 1600 ms of trailing silence, while a thinking pause inside
            // one costs nothing at all.
            grace_ms: 900,
            min_ms: 500,
            // A cap, not a target. Without it a stuck-open microphone sends a
            // buffer that grows until the tab dies.
            max_ms: 20_000,
            // Three frames, so a click or a door cannot open a turn.
            onset_frames: 3,
            voiced_ms: 250,
        }
    }
}

/// What the endpointer decided on one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Nothing is happening.
    Idle,
    /// A turn just opened.
    Opened,
    /// A turn is in progress.
    Listening,
    /// A pause has armed the end of turn; start transcribing now.
    Armed,
    /// Speech resumed inside the grace window; cancel any transcription.
    Resumed,
    /// The turn is over and should be sent.
    Committed {
        /// Whether enough loud audio accumulated to be worth sending.
        voiced: bool,
        /// Whether the turn ended because it hit `max_ms` rather than a pause.
        truncated: bool,
    },
}

/// The turn state machine, driven one frame at a time.
#[derive(Debug)]
pub struct Endpointer {
    policy: TurnPolicy,
    open: bool,
    onset: u32,
    elapsed_ms: u64,
    since_voice_ms: u64,
    armed_ms: Option<u64>,
    voiced_ms: u64,
    turn_ms: u64,
}

impl Endpointer {
    /// A detector that has heard nothing.
    pub fn new(policy: TurnPolicy) -> Endpointer {
        Endpointer {
            policy,
            open: false,
            onset: 0,
            elapsed_ms: 0,
            since_voice_ms: 0,
            armed_ms: None,
            voiced_ms: 0,
            turn_ms: 0,
        }
    }

    /// The policy this was built with.
    pub fn policy(&self) -> TurnPolicy {
        self.policy
    }

    /// How much loud audio the current turn has accumulated.
    pub fn voiced_ms(&self) -> u64 {
        self.voiced_ms
    }

    /// How long the current turn has been open.
    pub fn turn_ms(&self) -> u64 {
        self.turn_ms
    }

    /// Feeds one frame's energy and its duration.
    pub fn push(&mut self, energy: f32, frame_ms: u64) -> Decision {
        self.elapsed_ms += frame_ms;

        if !self.open {
            if energy > self.policy.vad_start {
                self.onset += 1;
                if self.onset >= self.policy.onset_frames {
                    self.open = true;
                    self.onset = 0;
                    self.since_voice_ms = 0;
                    self.armed_ms = None;
                    self.voiced_ms = 0;
                    // The frames that proved speech was starting are part of
                    // the turn, not a preamble to it. Counting them is what
                    // stops the first syllable being clipped.
                    self.turn_ms = frame_ms * u64::from(self.policy.onset_frames);
                    return Decision::Opened;
                }
            } else {
                self.onset = 0;
            }
            return Decision::Idle;
        }

        self.turn_ms += frame_ms;
        let mut resumed = false;
        if energy > self.policy.vad_stop {
            self.since_voice_ms = 0;
            self.voiced_ms += frame_ms;
            if self.armed_ms.take().is_some() {
                resumed = true;
            }
        } else {
            self.since_voice_ms += frame_ms;
        }

        if self.turn_ms >= self.policy.max_ms {
            return self.commit(true);
        }
        if let Some(armed) = self.armed_ms {
            if self.elapsed_ms.saturating_sub(armed) > self.policy.grace_ms {
                return self.commit(false);
            }
        } else if self.since_voice_ms > self.policy.silence_ms {
            self.armed_ms = Some(self.elapsed_ms);
            return Decision::Armed;
        }

        if resumed {
            return Decision::Resumed;
        }
        Decision::Listening
    }

    /// Ends the turn, as a released push-to-talk button does.
    ///
    /// `voiced` is reported as true regardless of how much loud audio was
    /// heard: a held button is an explicit statement that this is a turn, and
    /// applying the energy test to it discarded every push-to-talk utterance.
    pub fn release(&mut self) -> Decision {
        let long_enough = self.turn_ms >= self.policy.min_ms;
        self.reset();
        Decision::Committed {
            voiced: long_enough,
            truncated: false,
        }
    }

    fn commit(&mut self, truncated: bool) -> Decision {
        let voiced = self.voiced_ms >= self.policy.voiced_ms && self.turn_ms >= self.policy.min_ms;
        self.reset();
        Decision::Committed { voiced, truncated }
    }

    fn reset(&mut self) {
        self.open = false;
        self.onset = 0;
        self.armed_ms = None;
        self.since_voice_ms = 0;
        self.voiced_ms = 0;
        self.turn_ms = 0;
    }
}
