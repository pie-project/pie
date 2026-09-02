//! `Airborne` (run-ahead step counters) and `Settlement` (pooled CUDA
//! events), kept out of `serve` because they hold state, not call order.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::graph::Event;
use crate::error::Result;
use crate::inputs::Free;

/// Two monotone counters: issued steps and settled callbacks. `settled > n`
/// means step `n` finished — this comparison works because steps complete
/// in issue order (one compute stream).
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
    /// Sequence stamp for "no fire has launched" — always counts as settled.
    pub const NEVER: u64 = u64::MAX;

    /// A fresh pair of counters.
    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
    }

    /// Sequence number the next issued step will get, read without
    /// consuming it. A step that fails after this leaves the number
    /// unconsumed, so everything it stamped looks in-flight until the next
    /// step settles — safe direction for an eviction to be wrong in.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.counts.issued.load(Ordering::Acquire)
    }

    /// Register a step. Answers the sequence number it took.
    pub fn enter(&self) -> u64 {
        self.counts.issued.fetch_add(1, Ordering::AcqRel)
    }

    /// A settlement callback ran. Called from the driver's callback
    /// thread, hence a bare `fetch_add`.
    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    /// Undoes an [`Airborne::enter`] whose settlement never registered.
    /// Bumps `settled` rather than un-issuing: the issued counter must
    /// never go backwards.
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

    /// Has the step stamped `seq` finished? [`Airborne::NEVER`] always
    /// answers `true`.
    #[must_use]
    pub fn settled_past(&self, seq: u64) -> bool {
        seq == Airborne::NEVER || self.counts.settled.load(Ordering::Acquire) > seq
    }
}

/// Event pool the notify stream waits on: one event per in-flight step
/// (`Runahead::staging_depth()`).
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

    /// Takes an event, or says the pool is empty. Cannot fail if the
    /// caller keeps to the depth staging already bounded.
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
