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
//! This refuses to be a lock: the synthesiser never waits for the translator,
//! so there is nothing to deadlock, and a translator step that has already
//! started is not interrupted. Start reading at [`SharedCard::hold`].

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
