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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// See the module docs. Clones share one record.
#[derive(Debug, Clone)]
pub struct Live(Arc<Record>);

#[derive(Debug)]
struct Record {
    /// The widest rung armed, in tokens; `u32::MAX` once every rung is.
    armed: AtomicU32,
    /// Bytes the driver allocated beside the engine's span since load.
    outside: AtomicU64,
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
    use super::Live;

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
    }
}
