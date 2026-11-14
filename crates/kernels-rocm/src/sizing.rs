//! The size-only walk: a thread that walks a fire's ops with sizing on
//! launches nothing and copies nothing, and every scratch slab a launcher
//! asks for is counted at the most asked of it, without allocating. A
//! caller walks the fires it will run this way and lays the slabs the walk
//! asked for ([`crate::jit::Slabs::place`]) before it runs any: no slab is
//! allocated by a take.
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

/// The slabs a walk asked for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// By slab name, largest first: the most asked of each key, summed.
    pub slabs: Vec<(&'static str, usize)>,
    /// Every slab the walk asked for, by its key.
    pub keys: Vec<Sized>,
}

impl Footprint {
    /// The footprint of `keys`, in key order.
    #[must_use]
    pub fn of(mut keys: Vec<Sized>) -> Footprint {
        keys.sort_by(|a, b| a.key.cmp(&b.key));
        let mut by_name: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();
        for slab in &keys {
            *by_name.entry(slab.key.name).or_insert(0) += slab.asked;
        }
        let mut slabs: Vec<(&'static str, usize)> = by_name.into_iter().collect();
        slabs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        Footprint { slabs, keys }
    }

    /// Each key and the bytes to lay it at, as [`crate::jit::Slabs::place`]
    /// takes them.
    #[must_use]
    pub fn placed(&self) -> Vec<(crate::jit::SlabKey, usize)> {
        self.keys.iter().map(|slab| (slab.key, slab.asked)).collect()
    }
}

/// One slab of a walk: the most the walk asked of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sized {
    pub key: crate::jit::SlabKey,
    pub asked: usize,
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
