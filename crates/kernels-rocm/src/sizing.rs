//! The size-only walk: a thread that walks a fire's ops with sizing on
//! launches nothing and copies nothing, and every scratch slab a launcher
//! asks for is sized as `jit::device` would grow it, against what the
//! arenas already hold, without allocating. The engine walks the
//! compositions it is about to arm this way before it fits the card, so
//! the room it holds for arming is what arming will take.
//!
//! The switch is the walking thread's alone: another engine of the same
//! process loading on its own thread launches as ever. A read of device
//! data back to the host has no answer in a walk, so it refuses.
use core::cell::Cell;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread walks sizing only.
#[must_use]
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// What a walk would have allocated past what the slab arenas held.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Every allocation the walk's slabs would have made, outgrown ones
    /// included: an outgrown slab is retired, not freed, until its arena
    /// drops.
    pub bytes: usize,
    /// By slab name, largest first.
    pub slabs: Vec<(&'static str, usize)>,
}

/// Starts a walk on this thread.
pub fn begin() {
    crate::jit::device::sizing_reset();
    ACTIVE.with(|active| active.set(true));
}

/// Ends this thread's walk: what it would have allocated.
#[must_use]
pub fn end() -> Footprint {
    ACTIVE.with(|active| active.set(false));
    crate::jit::device::sizing_take()
}

/// The address a slab or an unbacked span names in a walk: never
/// dereferenced, as nothing launches and no copy touching it runs, but
/// non-null and aligned like a real allocation so a launcher's arithmetic
/// on it runs as it would.
pub const STAND_IN: usize = 0x7e00_0000_0000;

/// The stand-in span: every address a walk hands out lies in it.
const STAND_IN_SPAN: usize = 1 << 42;

/// Whether `at` is an address a walk handed out, not real memory.
#[must_use]
pub fn stands_in(at: usize) -> bool {
    (STAND_IN..STAND_IN + STAND_IN_SPAN).contains(&at)
}
