//! What an engine learns of itself after load, read by the runtime while it
//! serves: the rungs whose graph bodies are captured (REDESIGN 2.5) and the
//! device bytes the driver holds beside the engine's own (`outside`, 2.2).
//!
//! The engine keeps a [`Live`] and advances it from wherever it learns the
//! facts (a rung armed in [`crate::Engine::arm_step`], the lane time the
//! runtime gives it; the first body instantiated); the
//! runtime holds a clone from load on ([`crate::Engine::live`]) and reads
//! it without asking the engine's lane.
//!
//! The contract on rungs: every serving fire replays a captured body, so a
//! fire is never wider than the widest rung armed. An engine arms its rungs
//! smallest first, so what is armed is always a prefix of the ladder and
//! one number says it. The runtime serves at the rungs armed meanwhile, and
//! a raise to a rung still being captured waits on [`Live::arm`].

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

/// See the module docs. Clones share one record.
#[derive(Debug, Clone)]
pub struct Live(Arc<Record>);

#[derive(Debug)]
struct Record {
    /// The widest rung armed, in tokens; `u32::MAX` once every rung is.
    armed: AtomicU32,
    /// Bytes the driver allocated beside the engine's span since load.
    outside: AtomicU64,
    /// Bytes the engine holds outside the pool's cache rows for fires and
    /// bodies (arming's scratch among them), lowered after the last step.
    held: AtomicU64,
    /// Counts of [`Tally`] events since load, in its order.
    tallies: [AtomicU64; Tally::ALL.len()],
    /// The sites [`Tally::Unbuilt`] was met at since load, a bit each
    /// ([`Unbuilt`]).
    unbuilt_at: AtomicU8,
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
    /// is required: a bug under the rule that every serving fire replays
    /// one, kept apart so it is never mistaken for a compile.
    Unrecorded,
    /// A frame or registration the engine refused.
    Refused,
    /// A bind or a fire met a pipeline, module or stage that should have
    /// been built before it: an engine bug under the rule that a model's
    /// kernels are built at load and a guest program's at its registration
    /// ([`crate::Engine::register_program`]). It is refused by name, never
    /// built on the spot, so it is no [`Tally::Compiled`]. One count for
    /// both sites, told by [`Live::unbuilt`], which keeps which was met
    /// ([`Unbuilt`]).
    Unbuilt,
}

/// Where a [`Tally::Unbuilt`] was met: whose build missed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unbuilt {
    /// A kernel of the model, built at load.
    Kernel,
    /// A guest program, built at its registration.
    Program,
}

impl Unbuilt {
    pub const ALL: [Unbuilt; 2] = [Unbuilt::Kernel, Unbuilt::Program];

    /// Its name in the model status.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Unbuilt::Kernel => "kernel",
            Unbuilt::Program => "program",
        }
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

impl Tally {
    pub const ALL: [Tally; 5] = [
        Tally::Compiled,
        Tally::Fallback,
        Tally::Unrecorded,
        Tally::Refused,
        Tally::Unbuilt,
    ];

    /// Its name in the model status.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tally::Compiled => "compiled",
            Tally::Fallback => "fallback",
            Tally::Unrecorded => "unrecorded",
            Tally::Refused => "refused",
            Tally::Unbuilt => "unbuilt",
        }
    }
}

impl Live {
    /// Every rung armed and nothing measured outside: what an engine that
    /// captures every body before load returns states.
    #[must_use]
    pub fn armed() -> Live {
        Live::with(u32::MAX)
    }

    /// Nothing armed past `rung` yet (the smallest, armed at load): an
    /// engine that arms the rest in the background starts here.
    #[must_use]
    pub fn arming(rung: u32) -> Live {
        Live::with(rung)
    }

    fn with(armed: u32) -> Live {
        Live(Arc::new(Record {
            armed: AtomicU32::new(armed),
            outside: AtomicU64::new(0),
            held: AtomicU64::new(0),
            tallies: Default::default(),
            unbuilt_at: AtomicU8::new(0),
        }))
    }

    /// The bodies of `rung` (in tokens) are captured, and every narrower
    /// rung's were before; `u32::MAX` once the last is.
    pub fn arm(&self, rung: u32) {
        self.0.armed.fetch_max(rung, Ordering::AcqRel);
    }

    /// The widest rung armed, in tokens; `u32::MAX` once every rung is.
    #[must_use]
    pub fn armed_through(&self) -> u32 {
        self.0.armed.load(Ordering::Acquire)
    }

    /// The driver holds `bytes` beside the engine's span now (graph body
    /// instantiation, modules loaded on first use, library workspaces).
    pub fn set_outside(&self, bytes: u64) {
        self.0.outside.store(bytes, Ordering::Release);
    }

    /// What the driver holds beside the engine's span, as last stated.
    #[must_use]
    pub fn outside(&self) -> u64 {
        self.0.outside.load(Ordering::Acquire)
    }

    /// The engine holds `bytes` outside the pool's cache rows for its fires
    /// and bodies now, inside the range it stated at load: the runtime's
    /// pool takes no more than the range less this.
    pub fn set_held(&self, bytes: u64) {
        self.0.held.store(bytes, Ordering::Release);
    }

    /// What the engine holds outside the cache rows, as last stated.
    #[must_use]
    pub fn held(&self) -> u64 {
        self.0.held.load(Ordering::Acquire)
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

    /// One more [`Tally::Unbuilt`], met at `site`. The engine refuses the
    /// bind or the fire by the name of what was unbuilt and logs that
    /// name once; the record keeps the count and the sites.
    pub fn unbuilt(&self, site: Unbuilt) {
        self.0.unbuilt_at.fetch_or(site.bit(), Ordering::AcqRel);
        self.tally(Tally::Unbuilt, 1);
    }

    /// Whether a [`Tally::Unbuilt`] was met at `site` since load.
    #[must_use]
    pub fn unbuilt_at(&self, site: Unbuilt) -> bool {
        self.0.unbuilt_at.load(Ordering::Acquire) & site.bit() != 0
    }
}

impl Default for Live {
    fn default() -> Self {
        Live::armed()
    }
}

/// The pool slots one [`crate::Engine::arm_step`] writes while it captures:
/// its synthetic fires' kv pages (slots of the kv list shape) and record
/// slots (the record shape). Stated once at load; zero for an engine whose
/// arming writes no pool slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArmingRoom {
    pub pages: u32,
    pub records: u32,
}

/// The slots the runtime lends one [`crate::Engine::arm_step`], as slot
/// refs: [`ArmingRoom::pages`] of the kv list shape and
/// [`ArmingRoom::records`] of the record shape. Free slots no sequence
/// holds; the engine may write them, and what it leaves in them is garbage
/// the runtime never reads. They go back to free when the call returns.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArmingLease {
    pub pages: Vec<u32>,
    pub records: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::{Live, Tally, Unbuilt};

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
    }

    /// One counter for both sites: the count is every unbuilt thing met,
    /// and the record says which sites met one.
    #[test]
    fn an_unbuilt_is_one_count_that_keeps_its_sites() {
        let live = Live::armed();
        let read = live.clone();
        assert_eq!(Tally::Unbuilt.name(), "unbuilt");
        assert!(Tally::ALL.contains(&Tally::Unbuilt));
        assert_eq!(read.tallied(Tally::Unbuilt), 0);
        assert!(Unbuilt::ALL.iter().all(|&site| !read.unbuilt_at(site)));
        live.unbuilt(Unbuilt::Program);
        assert_eq!(read.tallied(Tally::Unbuilt), 1);
        assert!(read.unbuilt_at(Unbuilt::Program));
        assert!(!read.unbuilt_at(Unbuilt::Kernel));
        live.unbuilt(Unbuilt::Kernel);
        live.unbuilt(Unbuilt::Kernel);
        assert_eq!(read.tallied(Tally::Unbuilt), 3);
        assert!(read.unbuilt_at(Unbuilt::Kernel));
        assert!(read.unbuilt_at(Unbuilt::Program), "a later site hides none");
        assert_eq!(read.tallied(Tally::Refused), 0, "its own count");
    }

    #[test]
    fn arming_is_a_prefix_that_only_widens() {
        let live = Live::arming(256);
        let read = live.clone();
        assert_eq!(read.armed_through(), 256);
        live.arm(1024);
        live.arm(512);
        assert_eq!(read.armed_through(), 1024, "a clone reads the same record");
        live.arm(u32::MAX);
        assert_eq!(read.armed_through(), u32::MAX);
        assert_eq!(Live::armed().armed_through(), u32::MAX);
        live.set_outside(845 << 20);
        assert_eq!(read.outside(), 845 << 20);
        live.set_held(64 << 20);
        assert_eq!(read.held(), 64 << 20);
    }
}
