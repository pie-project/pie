//! Settlement state: who is airborne ([`Airborne`]), which seat they took
//! ([`Arms`]), and where a finished command buffer reports ([`Done`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Two monotone counters: `issued` (steps committed with a completion
/// handler) and `settled` (handlers that have run). "step n finished" is
/// `settled > n`.
#[derive(Debug, Clone, Default)]
pub struct Airborne {
    counts: Arc<Counts>,
}

#[derive(Debug, Default)]
struct Counts {
    issued: AtomicU64,
    settled: AtomicU64,
}

impl Airborne {
    /// A fresh pair of counters.
    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
    }

    /// Register a step. Answers the sequence number it took.
    pub fn enter(&self) -> u64 {
        self.counts.issued.fetch_add(1, Ordering::AcqRel)
    }

    /// A completion handler ran. Called from Metal's completion thread, so
    /// just one `fetch_add`.
    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    /// Undo an [`Airborne::enter`] whose commit never happened. Bumps
    /// `settled` rather than winding `issued` back.
    pub fn abandon(&self) {
        self.leave();
    }

    /// How many issued steps have not reported.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .issued
            .load(Ordering::Acquire)
            .saturating_sub(self.counts.settled.load(Ordering::Acquire))
    }

    /// Has the step stamped `seq` reported?
    #[must_use]
    pub fn settled_past(&self, seq: u64) -> bool {
        self.counts.settled.load(Ordering::Acquire) > seq
    }
}

/// Where an asynchronous step publishes that it is done: a `StepDone` to
/// correlate on and the sink to call with it.
pub struct Done {
    /// Which step of which frame this is.
    pub at: engine::StepDone,
    /// Where to say so.
    pub sink: engine::CompletionSink,
}

impl std::fmt::Debug for Done {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Done").field("at", &self.at).finish()
    }
}

/// The A/B seat ring: how many duplicated seat sets there are, and which of
/// them a step may write into. Depth 1 is the degenerate eager ring (every
/// step takes seat zero).
///
/// A seat is taken at `settle`, not `prepare`: a step staged and then
/// refused never reaches the device, so its seat must go straight back to
/// the free set.
#[derive(Debug, Clone)]
pub struct Arms {
    taken: Vec<bool>,
}

impl Arms {
    /// A ring of `depth` seats, `depth >= 1`, all free.
    #[must_use]
    pub fn of(depth: usize) -> Arms {
        Arms {
            taken: vec![false; depth.max(1)],
        }
    }

    /// How many seat sets there are.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.taken.len()
    }

    /// The lowest seat no committed step owns, or `None` when every one of
    /// them is in flight.
    #[must_use]
    pub fn free(&self) -> Option<usize> {
        self.taken.iter().position(|held| !held)
    }

    /// This seat now belongs to a committed step.
    pub fn take(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = true;
        }
    }

    /// The step that held this seat has been harvested.
    pub fn give(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = false;
        }
    }

    /// How many seats a committed step owns.
    #[must_use]
    pub fn held(&self) -> usize {
        self.taken.iter().filter(|held| **held).count()
    }
}
