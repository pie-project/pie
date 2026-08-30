//! **The settlement plane**: who is airborne, which seat they took, and where
//! a finished command buffer reports.
//!
//! Three small things the asynchronous fire path could not do without, kept
//! out of [`serve`](crate::serve) because they are STATE rather than call
//! order — which is the whole of what that file's header promises.
//!
//! # [`Airborne`] — the run-ahead, counted
//!
//! One monotone counter for steps ISSUED and one for steps whose completion
//! handler has RUN. Their difference is how far ahead of the device this
//! shell is. F1 answered "is the device still reading this?" with *every fire
//! ends synchronized*; this answers it with arithmetic, which is the same
//! answer at depth 1 and a true one above it.
//!
//! The counters are `Arc`-shared because the second one is bumped on **Metal's
//! own completion thread**, where a lock would be a hazard — a `fetch_add` is
//! the whole vocabulary that thread gets, exactly as it is on the CUDA plane's
//! host-function thread.
//!
//! # [`Done`] — where a step publishes that it finished
//!
//! The engine's half of the completion seam: a `StepDone` to correlate on and
//! the runtime's sink to call with it. Both are the CALLER's — `api.rs` mints
//! the ids and the runtime installs the sink — because the shell has no
//! opinion about who is waiting.
//!
//! # [`Arms`] — the A/B seats, claimed and released in order
//!
//! Every host-mutable seat the GPU reads is duplicated per in-flight step (the
//! resident inputs) and so is every host-read seat the GPU writes (the readout
//! rows). Which copy a step took is one small integer, and this is the ring
//! that hands them out.
//!
//! **IT IS A CURSOR AND NOT A FREE SET, AND THAT IS A CLAIM ABOUT THE
//! ORDER.** Command buffers on one `MTLCommandQueue` retire in the order they
//! were committed, and this shell settles them in that same order — oldest
//! first, always — so the arm that comes free next is always the one after the
//! arm that came free last. A free-word compare-exchange (the CUDA staging
//! ring's shape, which it needs because its release runs on the driver's
//! callback thread) would be machinery for a choice that has one answer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// **How far ahead of the device this shell is**, as two monotone counters.
///
/// `issued` counts steps that have been committed with a completion handler;
/// `settled` counts the handlers that have run. Both only ever go up, and
/// steps complete in the order they were committed because they ride ONE
/// command queue — which is what makes "step `n` has finished" the plain
/// comparison `settled > n` rather than a set membership test.
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

    /// A completion handler ran. **Called from Metal's completion thread**,
    /// which is why it is one `fetch_add` and nothing else.
    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    /// Undo an [`Airborne::enter`] whose commit never happened.
    ///
    /// Bumps the settled side rather than winding the issued side back,
    /// because the issued side is what stamps are compared against and a
    /// number that went backwards would make a finished step look airborne
    /// forever.
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

/// **Where an asynchronous step publishes that it is done.**
///
/// A `StepDone` to correlate on and the sink to call with it. Identical to
/// the CUDA sibling's type on purpose: one seam, one shape, whichever shell
/// is behind it.
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

/// **The A/B seat ring**: how many duplicated seat sets there are, and which
/// of them a step may write into.
///
/// `depth` is [`Runahead::frames_in_flight`](engine::runahead::Runahead), and
/// it is the number for the reason article 1 states it: one step executing
/// while the next is already committed behind it needs two of everything the
/// host writes and the device reads. A depth of one is the degenerate ring —
/// every step takes seat zero — which is the eager shell, kept reachable
/// because it is the golden model a divergence at depth two is bisected
/// against.
///
/// **A SEAT IS TAKEN AT `settle` AND NOT AT `prepare`, WHICH IS WHY THIS IS A
/// SET AND NOT A CURSOR.** A step that is staged and then refused — an
/// admission that did not fit, a dispatch this plane has no arm for — never
/// reaches the device, so its seat was never at risk and must go straight back
/// to the free set. A round-robin cursor cannot express that: it would advance
/// past the abandoned seat and hand the NEXT step a seat a running command
/// buffer still owns. So `free` answers which seat a step may stage into,
/// `take` is what filing the flight does, and `give` is the harvest's.
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
    ///
    /// Deterministic on purpose: a step stages into the seat this answers and
    /// files it one call later, and the two must agree without anything
    /// between them having to remember which.
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
