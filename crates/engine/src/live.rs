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
    /// A guest program's code built at its registration after ready, the
    /// load's prewarm not having built it: a program the runtime's
    /// standard samplers do not register, built before its first fire. No
    /// fire waits on it, so it breaks no rule; read beside
    /// [`Tally::BuiltInFire`], and never gated on.
    BuiltAtRegister,
    /// The wall time [`Tally::BuiltAtRegister`]'s builds took, in
    /// milliseconds, where the backend times them: a build an OS cache
    /// serves (tens of ms) reads apart from a cold compile (seconds).
    BuiltAtRegisterMs,
    /// A pipeline, module or program built after ready on a fire's path:
    /// one a serving fire or a bind reached that the load did not build,
    /// whatever the backend (a kernel pipeline, a JIT instantiation, a
    /// guest step's pipeline). The rule is that every one a fire can reach
    /// is built before ready, so this reads 0; what gates or warns reads
    /// this one. A backend that does not yet tell a registration's builds
    /// apart counts them here too.
    BuiltInFire,
    /// What a dispatch binds (a bind group, a descriptor set) built after
    /// ready on a fire's path: one a backend that builds them per set of
    /// views made for a fire the load's bind walk did not reach. The rule
    /// is that the load builds every one a fire can bind, so this reads 0.
    BoundInFire,
    /// A call that waited for every frame in flight to land before the host
    /// touched device memory (a tier move through the host): dispatch fell
    /// to depth one for it. Counted while such a move is not yet on the
    /// queue, so a backend's remaining drains are seen.
    Drained,
    /// A submit-bound probe fired after the device idled (the smallest
    /// matmul re-timing the clock where the platform states a bound on one
    /// submission): not a model fire, and recorded at load where the
    /// backend keeps recordings, so never [`Tally::Unrecorded`].
    Probed,
    /// A program the load compiled and parked in its disk cache (the host
    /// held no more in memory), loaded on the first fire of its shape: a
    /// load from disk, never a build, so neither [`Tally::Compiled`] nor
    /// [`Tally::BuiltInFire`]; counted so the stall it costs is seen.
    LoadedInFire,
}

impl Tally {
    pub const ALL: [Tally; 12] = [
        Tally::Compiled,
        Tally::Fallback,
        Tally::Unrecorded,
        Tally::Refused,
        Tally::Eager,
        Tally::BuiltAtRegister,
        Tally::BuiltAtRegisterMs,
        Tally::BuiltInFire,
        Tally::BoundInFire,
        Tally::Drained,
        Tally::Probed,
        Tally::LoadedInFire,
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
            Tally::BuiltAtRegister => "built_at_register",
            Tally::BuiltAtRegisterMs => "built_at_register_ms",
            Tally::BuiltInFire => "built_in_fire",
            Tally::BoundInFire => "bound_in_fire",
            Tally::Drained => "drained",
            Tally::Probed => "probed",
            Tally::LoadedInFire => "loaded_in_fire",
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
        live.tally(Tally::BuiltAtRegister, 2);
        assert_eq!(read.tallied(Tally::BuiltAtRegister), 2);
        assert_eq!(
            read.tallied(Tally::BuiltInFire),
            0,
            "a registration's build is no fire's"
        );
    }
}
