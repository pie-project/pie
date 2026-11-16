//! What an engine counts while it serves, read by the runtime's model
//! status: the fires served some way other than a captured body the load
//! prepared ([`Tally`]).
//!
//! The engine keeps a [`Live`] and counts into it from wherever the event
//! happens; the runtime holds a clone from load on
//! ([`crate::Engine::live`]) and reads it without asking the engine's lane.
//!
//! Nothing else is said here. Which rungs the engine armed, what it holds
//! back of its range and what the driver holds beside its span are load
//! facts ([`crate::chunk::ChunkFacts::eager_rungs`],
//! [`crate::chunk::ChunkFacts::held_bytes`],
//! [`crate::chunk::ChunkFacts::outside_bytes`]; what is held back settles
//! with the residency the load starts with, [`crate::Engine::weight_residency`]):
//! every capture ends before the load returns, and nothing arms while the
//! engine serves.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// See the module docs. Clones share one record.
#[derive(Debug, Clone)]
pub struct Live(Arc<Record>);

#[derive(Debug)]
struct Record {
    /// Counts of [`Tally`] events since load, in its order.
    tallies: [AtomicU64; Tally::ALL.len()],
}

/// What an engine counts while it serves, each a per-fire event: a way a
/// serving fire was served other than its load prepared, or not at all.
/// Never silent; read by the runtime's model status. A slower arm an
/// engine chooses at load is no tally: it is a load fact, stated once in
/// the load's summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tally {
    /// A program, shape or pipeline built or loaded (from a disk cache)
    /// while serving: a fire waited on it, its shape not prepared at load.
    Compiled,
    /// A legitimate slower path taken for a fire (a guest stage the host
    /// interprets).
    Fallback,
    /// A serving fire that ran with no captured body or program where one
    /// is required: a bug under the rule that every serving fire at an
    /// armed rung replays one, kept apart so it is never mistaken for a
    /// compile. Never a fire at a rung the engine stated eager: that is
    /// [`Tally::Eager`], and no bug.
    Unrecorded,
    /// A frame or registration the engine refused.
    Refused,
    /// A fire at a rung the engine stated eager at load
    /// ([`crate::chunk::ChunkFacts::eager_rungs`]): it ran uncaptured, as
    /// stated.
    Eager,
    /// A pipeline, module or program built after ready: one a serving
    /// fire, a registration or a bind reached that the load did not build,
    /// whatever the backend (a kernel pipeline, a guest step's pipeline, a
    /// guest region's module). The rule is that every one serving can
    /// reach is built before ready, so this reads 0.
    BuiltAfterReady,
    /// A body recorded after ready: a serving fire whose frame no recording
    /// the load made replays, so it walked and kept one. The rule is that
    /// every body serving fires is recorded before ready, so this reads 0.
    RecordedAfterReady,
}

impl Tally {
    pub const ALL: [Tally; 7] = [
        Tally::Compiled,
        Tally::Fallback,
        Tally::Unrecorded,
        Tally::Refused,
        Tally::Eager,
        Tally::BuiltAfterReady,
        Tally::RecordedAfterReady,
    ];

    /// Its name in the model status.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tally::Compiled => "compiled",
            Tally::Fallback => "fallback",
            Tally::Unrecorded => "unrecorded",
            Tally::Refused => "refused",
            Tally::Eager => "eager",
            Tally::BuiltAfterReady => "built_after_ready",
            Tally::RecordedAfterReady => "recorded_after_ready",
        }
    }
}

impl Live {
    /// Nothing counted yet: what a load starts from.
    #[must_use]
    pub fn armed() -> Live {
        Live(Arc::new(Record {
            tallies: Default::default(),
        }))
    }

    /// `n` more `tally` events.
    pub fn tally(&self, tally: Tally, n: u64) {
        self.0.tallies[tally as usize].fetch_add(n, Ordering::AcqRel);
    }

    /// The `tally` events since load.
    #[must_use]
    pub fn tallied(&self, tally: Tally) -> u64 {
        self.0.tallies[tally as usize].load(Ordering::Acquire)
    }
}

impl Default for Live {
    fn default() -> Self {
        Live::armed()
    }
}

#[cfg(test)]
mod tests {
    use super::{Live, Tally};

    #[test]
    fn a_tally_counts_on_every_clone() {
        let live = Live::armed();
        let read = live.clone();
        live.tally(Tally::Compiled, 2);
        live.tally(Tally::Refused, 1);
        assert_eq!(read.tallied(Tally::Compiled), 2);
        assert_eq!(read.tallied(Tally::Fallback), 0);
        assert_eq!(read.tallied(Tally::Unrecorded), 0);
        assert_eq!(read.tallied(Tally::Refused), 1);
        assert_eq!(read.tallied(Tally::Eager), 0);
        live.tally(Tally::Eager, 3);
        assert_eq!(read.tallied(Tally::Eager), 3);
        assert_eq!(
            read.tallied(Tally::Unrecorded),
            0,
            "an eager fire is no bug"
        );
    }
}
