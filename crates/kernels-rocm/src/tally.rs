//! What this crate did for a product that the load did not prepare, counted
//! on the thread that asked: a code object built or loaded (a JIT unit, a
//! hipBLASLt plan or library), one refused, and a faster route that
//! declined a product a slower one took. The engine reads [`counts`] on
//! either side of a serving fire and says how the fire was served; a fire
//! that replays a captured body runs none of this crate's host code and
//! moves nothing.
//!
//! Per thread, because an engine fires its lane on one thread and a
//! process may serve several devices at once: a count read across a fire
//! is that fire's alone.

use std::cell::Cell;

/// The counts on this thread since it began. See the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Code objects built or loaded: a JIT unit compiled or read from the
    /// disk cache and loaded, a hipBLASLt plan made for a new shape, a
    /// grouped-GEMM library opened.
    pub built: u64,
    /// Code objects refused: a unit that would not compile, or whose code
    /// object asks the runtime for what a captured body cannot have.
    pub refused: u64,
    /// Products a faster route declined and a slower route took.
    pub declined: u64,
}

thread_local! {
    static COUNTS: Cell<Counts> = const { Cell::new(Counts { built: 0, refused: 0, declined: 0 }) };
}

/// This thread's counts.
#[must_use]
pub fn counts() -> Counts {
    COUNTS.with(Cell::get)
}

fn bump(field: impl FnOnce(&mut Counts) -> &mut u64) {
    COUNTS.with(|counts| {
        let mut now = counts.get();
        *field(&mut now) += 1;
        counts.set(now);
    });
}

/// A code object built or loaded on this thread.
/// Only the HIP backend builds code objects.
#[cfg(feature = "rocm")]
pub(crate) fn built() {
    bump(|c| &mut c.built);
}

/// A code object refused on this thread.
/// Only the HIP backend builds code objects.
#[cfg(feature = "rocm")]
pub(crate) fn refused() {
    bump(|c| &mut c.refused);
}

/// A product a faster route declined and a slower one took, on this thread:
/// this crate's routes, and the engine's own on this device (a select that
/// reads its experts across the bus).
pub fn declined() {
    bump(|c| &mut c.declined);
}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;

    #[test]
    fn a_count_moves_on_its_own_thread_only() {
        let before = counts();
        declined();
        built();
        let there = std::thread::spawn(|| {
            refused();
            counts()
        })
        .join()
        .expect("joined");
        let after = counts();
        assert_eq!(after.declined, before.declined + 1);
        assert_eq!(after.built, before.built + 1);
        assert_eq!(after.refused, before.refused);
        assert_eq!(there, Counts { built: 0, refused: 1, declined: 0 });
    }
}
