//! What this crate built for a product the load did not prepare, counted on
//! the thread that asked: a JIT instantiation compiled or read from the disk
//! cache and loaded. The engine reads [`counts`] on either side of a serving
//! fire and counts what the fire built; a fire that replays a captured body
//! runs none of this crate's host code and moves nothing.
//!
//! Per thread, because an engine fires its lane on one thread and a process
//! may serve several devices at once: a count read across a fire is that
//! fire's alone (kernels-rocm's `tally` is the same).

use std::cell::Cell;

/// The counts on this thread since it began. See the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Instantiations resolved: compiled, or read from the disk cache, and
    /// loaded.
    pub built: u64,
}

thread_local! {
    static COUNTS: Cell<Counts> = const { Cell::new(Counts { built: 0 }) };
}

/// This thread's counts.
#[must_use]
pub fn counts() -> Counts {
    COUNTS.with(Cell::get)
}

/// An instantiation resolved on this thread.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn built() {
    COUNTS.with(|counts| {
        let mut now = counts.get();
        now.built += 1;
        counts.set(now);
    });
}
