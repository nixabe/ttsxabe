//! One card, two stages: who runs when the translator and the synthesiser
//! share a device.
//!
//! Two GPU jobs on one set of SMs do not run in half the time; they run in the
//! same total time and delay whichever finishes first. On a spoken turn that
//! is the synthesis of the clause the listener is waiting for, and measured:
//! a clause's synthesis went from 380 ms to 780 with the translator decoding
//! the next clauses beside it - see docs/BENCHMARKS.md, "Several clauses, one
//! weight stream". So on a shared card synthesis has the card while it runs,
//! and the translator - which decodes several clauses a step and loses nothing
//! by waiting a few hundred milliseconds between steps - steps only while it
//! is free.
//!
//! The chat model and the translator share a card the same way, measured in
//! `docs/MEASUREMENTS.md`: while a reply streams the translator waits, so the
//! reply decodes at about the rate it has alone rather than at half of it -
//! except that a turn's first clause waits only a bounded while, because the
//! listener is waiting through it in silence. See `serve::spawn_translator`.
//!
//! This refuses to be a lock: neither the synthesiser nor the chat model ever
//! waits for the translator - the chat model lets go of the card whenever it
//! is itself blocked handing a piece on - so there is nothing to deadlock,
//! and a translator step that has already started is not interrupted. Start reading at [`SharedCard::hold`].

use std::sync::{Arc, Condvar, Mutex};

/// A device two stages share, and whether a synthesis is running on it.
pub struct SharedCard {
    /// Syntheses in flight: several engines may share the card.
    busy: Mutex<usize>,
    freed: Condvar,
}

impl SharedCard {
    /// A card nobody is synthesising on.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            busy: Mutex::new(0),
            freed: Condvar::new(),
        })
    }

    /// Marks a synthesis running until the guard is dropped.
    pub fn hold(self: &Arc<Self>) -> Held {
        *self
            .busy
            .lock()
            .expect("the card's count is never poisoned") += 1;
        Held(self.clone())
    }

    /// Returns once no synthesis is running, or at `deadline`, whichever is
    /// first.
    pub fn wait_free_until(&self, deadline: std::time::Instant) {
        let mut busy = self
            .busy
            .lock()
            .expect("the card's count is never poisoned");
        while *busy > 0 {
            let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
                return;
            };
            busy = self
                .freed
                .wait_timeout(busy, left)
                .expect("the card's count is never poisoned")
                .0;
        }
    }

    /// Returns once no synthesis is running. The translator calls this before
    /// every step and every prompt.
    pub fn wait_free(&self) {
        let mut busy = self
            .busy
            .lock()
            .expect("the card's count is never poisoned");
        while *busy > 0 {
            busy = self
                .freed
                .wait(busy)
                .expect("the card's count is never poisoned");
        }
    }
}

/// A synthesis in flight; dropping it frees the card.
pub struct Held(Arc<SharedCard>);

impl Drop for Held {
    fn drop(&mut self) {
        let mut busy = self
            .0
            .busy
            .lock()
            .expect("the card's count is never poisoned");
        *busy -= 1;
        if *busy == 0 {
            self.0.freed.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_bounded_wait_returns_at_its_deadline_while_the_card_is_held() {
        let card = SharedCard::new();
        let held = card.hold();
        let start = Instant::now();
        card.wait_free_until(start + Duration::from_millis(50));
        let waited = start.elapsed();
        assert!(waited >= Duration::from_millis(50), "{waited:?}");
        assert!(waited < Duration::from_secs(2), "{waited:?}");
        drop(held);
    }

    #[test]
    fn a_bounded_wait_returns_when_the_card_is_freed() {
        let card = SharedCard::new();
        let held = card.hold();
        let freer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(held);
        });
        let start = Instant::now();
        card.wait_free_until(start + Duration::from_secs(10));
        assert!(start.elapsed() < Duration::from_secs(5));
        freer.join().unwrap();
        // And at once on a free card, however far off the deadline.
        let start = Instant::now();
        card.wait_free_until(start + Duration::from_secs(10));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
