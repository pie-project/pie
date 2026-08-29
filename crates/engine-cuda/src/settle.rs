//! **The settlement plane**: who is airborne, and where the callbacks ride.
//!
//! Two small things the F2b fire path could not do without, kept out of
//! [`serve`](crate::serve) because they are state rather than call order.
//!
//! # [`Airborne`] — the run-ahead, counted
//!
//! One monotone counter for steps ISSUED a settlement and one for steps whose
//! settlement callback has RUN. Their difference is how far ahead of the
//! device this shell is; their absolute values are what the graph cache reads
//! to answer the only question eviction and the prebind ever needed: *is the
//! exec I am about to overwrite one the device may still be running?* F1
//! answered it with "every fire ends synchronized". F2b answers it with
//! arithmetic, which is the same answer at depth 1 and a true one above it.
//!
//! The counters are `Arc`-shared because the second one is bumped on the CUDA
//! driver's host-function thread, where a lock is a hazard and a CUDA call is
//! forbidden — a `fetch_add` is the whole vocabulary that thread gets.
//!
//! # [`Settlement`] — the events, pooled
//!
//! One `cudaEvent_t` per in-flight step, created at load (article 9: the fire
//! path allocates nothing) and recycled by the same one-word free set the
//! staging ring uses, for the same reason: the release happens on the callback
//! thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::graph::Event;
use crate::error::Result;
use crate::inputs::Free;

/// **How far ahead of the device this shell is**, as two monotone counters.
///
/// `issued` counts steps that have had a settlement registered; `settled`
/// counts the callbacks that have run. Both only ever go up, and steps
/// complete in the order they were issued because they ride ONE compute
/// stream — which is what makes "step `n` has finished" the plain comparison
/// `settled > n` rather than a set membership test.
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
    /// **The stamp of something no fire has launched.** Settled by
    /// definition — see [`Airborne::settled_past`].
    pub const NEVER: u64 = u64::MAX;

    /// A fresh pair of counters.
    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
    }

    /// The sequence number the NEXT step to be issued will get, read without
    /// consuming it.
    ///
    /// `enqueue` stamps this onto whatever it launches, `settle` consumes it,
    /// and the two are one host thread apart with nothing between them — so a
    /// stamp is never stale. A step that FAILS between the two leaves the
    /// number unconsumed, which makes everything it stamped look in-flight
    /// until the next step settles: conservative in the safe direction, which
    /// is the direction an eviction wants to be wrong in.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.counts.issued.load(Ordering::Acquire)
    }

    /// Register a step. Answers the sequence number it took.
    pub fn enter(&self) -> u64 {
        self.counts.issued.fetch_add(1, Ordering::AcqRel)
    }

    /// A settlement callback ran. **Called from the driver's callback
    /// thread**, which is why it is one `fetch_add` and nothing else.
    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    /// Undo an [`Airborne::enter`] whose settlement never got registered.
    ///
    /// Bumps the settled side rather than winding the issued side back,
    /// because the issued side is what stamps are compared against and a
    /// number that went backwards would make a finished step look airborne
    /// forever.
    pub fn abandon(&self) {
        self.leave();
    }

    /// How many issued steps have not settled.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .issued
            .load(Ordering::Acquire)
            .saturating_sub(self.counts.settled.load(Ordering::Acquire))
    }

    /// **Has the step stamped `seq` finished?**
    ///
    /// The one question `record::Graphs` asks before it overwrites or destroys
    /// a `cudaGraphExec_t`, and the arithmetic that replaced "every fire ends
    /// synchronized". [`Airborne::NEVER`] answers `true`: an exec no fire has
    /// launched cannot be one the device is running, and a freshly
    /// instantiated ping-pong twin is exactly that case.
    #[must_use]
    pub fn settled_past(&self, seq: u64) -> bool {
        seq == Airborne::NEVER || self.counts.settled.load(Ordering::Acquire) > seq
    }
}

/// **The event pool the notify stream waits on.**
///
/// One event per in-flight step — `Runahead::staging_depth()` of them, the
/// same number and for the same reason as the staging slots, since a step
/// holds exactly one of each between `settle` and its callback.
#[derive(Debug)]
pub struct Settlement {
    events: Vec<Event>,
    free: Arc<Free>,
}

impl Settlement {
    /// Create `depth` events, once, at load.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`](crate::Fault::Runtimeless) or
    /// [`Fault::Device`](crate::Fault::Device).
    pub fn open(depth: usize) -> Result<Settlement> {
        let mut events = Vec::with_capacity(depth);
        for _ in 0..depth {
            events.push(Event::new()?);
        }
        Ok(Settlement {
            events,
            free: Free::of(depth),
        })
    }

    /// Take an event, or say the pool is empty.
    ///
    /// Cannot fail against a caller keeping to its stated depth: the staging
    /// claim in `prepare` already bounded the in-flight steps by the same
    /// number, so a step that got a slot has an event waiting for it.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) naming the pool.
    pub fn claim(&self) -> Result<u32> {
        self.free.take().ok_or(crate::error::Fault::Ceiling {
            what: "settlement events (one per in-flight step)",
            need: self.events.len() as u64 + 1,
            have: self.events.len() as u64,
        })
    }

    /// The event at `at`.
    #[must_use]
    pub fn event(&self, at: u32) -> &Event {
        &self.events[at as usize]
    }

    /// The free set, cloneable into a settlement callback.
    #[must_use]
    pub fn recycler(&self) -> Arc<Free> {
        Arc::clone(&self.free)
    }
}
