//! Stage timing, off by default.
//!
//! Off by default because measuring costs correctness of the measurement: a
//! CUDA launch is asynchronous, so timing one means synchronising after it, and
//! a pipeline synchronised at every stage is not the pipeline that runs. So the
//! production path constructs [`Clock::off`], which takes no clock readings and
//! issues no syncs, and only the bench binary asks for the other one.
//!
//! Repeated names accumulate rather than append. The decode loop runs the same
//! six stages several hundred times and a list of two thousand entries is not a
//! measurement of anything.

use crate::TacoError;
use std::time::Instant;
use xabe_cuda::Gpu;

/// Stage name to milliseconds, in first-seen order.
pub type Timings = Vec<(&'static str, f64)>;

/// Accumulates per-stage wall time, or does nothing.
pub(crate) struct Clock {
    on: bool,
    marks: Timings,
    /// How many times the loop stages were entered, for per-step figures.
    pub(crate) steps: usize,
}

impl Clock {
    /// A clock that reads nothing and synchronises nothing.
    pub(crate) fn off() -> Self {
        Self {
            on: false,
            marks: Vec::new(),
            steps: 0,
        }
    }

    /// A clock that synchronises after every stage it is asked to close.
    pub(crate) fn on() -> Self {
        Self {
            on: true,
            marks: Vec::new(),
            steps: 0,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.on
    }

    /// Opens an interval, or returns `None` when timing is off.
    pub(crate) fn start(&self) -> Option<Instant> {
        self.on.then(Instant::now)
    }

    /// Closes an interval: synchronise, then add to `name`'s running total.
    pub(crate) fn stop(
        &mut self,
        gpu: &Gpu,
        name: &'static str,
        at: Option<Instant>,
    ) -> Result<(), TacoError> {
        let Some(at) = at else { return Ok(()) };
        gpu.synchronize()?;
        let ms = at.elapsed().as_secs_f64() * 1e3;
        match self.marks.iter_mut().find(|(n, _)| *n == name) {
            Some((_, total)) => *total += ms,
            None => self.marks.push((name, ms)),
        }
        Ok(())
    }

    /// The accumulated totals.
    pub(crate) fn into_marks(self) -> Timings {
        self.marks
    }
}
