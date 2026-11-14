#![allow(unsafe_code)]
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::error::{Fault, Result};
use crate::store::Pools;

use super::held::Cells;
use super::launch::{ChannelShape, Cursor};

pub const MAX_ATTACHMENTS: u32 = 8;

#[derive(Debug)]
pub struct SharedRing {
    shape: ChannelShape,
    /// Its cells, in `Channel` slots (a buffer of its own only for a ring
    /// larger than a chunk). Behind a lock only so evacuation can rebind a
    /// ring every attachment shares.
    cells: RwLock<Cells>,
    counters: Counters,
}

#[derive(Debug, Default)]
struct Counters {
    head: AtomicU64,
    tail: AtomicU64,
    attachments: AtomicU32,
    seeded: AtomicU32,
}

// SAFETY: everything below `cells` is an atomic; `cells` holds
// `device::Buffer`s, whose `Send`/`Sync` rest on `MTLBuffer` being
// thread-safe for retain/release, `contents`, and encoder binding.
unsafe impl Send for SharedRing {}
// SAFETY: as above.
unsafe impl Sync for SharedRing {}

impl SharedRing {
    /// A ring of `shape` in `slots`, the `Channel` slot refs the runtime
    /// took for it.
    pub fn open(shape: ChannelShape, pools: &Pools, slots: &[u32]) -> Result<SharedRing> {
        Ok(SharedRing {
            shape,
            cells: RwLock::new(shape.cells(pools, slots)?),
            counters: Counters::default(),
        })
    }

    #[must_use]
    pub const fn shape(&self) -> ChannelShape {
        self.shape
    }

    pub fn with_cells<T>(&self, with: impl FnOnce(&Cells) -> Result<T>) -> Result<T> {
        let cells = self
            .cells
            .read()
            .map_err(|_| Fault::program("program::shared", "a ring's lock was poisoned"))?;
        with(&cells)
    }

    pub fn write_cell(&self, index: u64, bytes: &[u8]) -> Result<()> {
        self.cells
            .write()
            .map_err(|_| Fault::program("program::shared", "a ring's lock was poisoned"))?
            .write(index, bytes)
    }

    /// Moves a pooled ring to `slots`.
    pub fn retarget(&self, pools: &Pools, slots: &[u32]) -> Result<()> {
        self.cells
            .write()
            .map_err(|_| Fault::program("program::shared", "a ring's lock was poisoned"))?
            .retarget(pools, slots)
    }

    /// Points a pooled ring at the chunks its slots sit in now.
    pub fn rebind(&self, pools: &Pools) -> Result<()> {
        self.cells
            .write()
            .map_err(|_| Fault::program("program::shared", "a ring's lock was poisoned"))?
            .rebind(pools)
    }

    #[must_use]
    pub fn cursor(&self) -> Cursor {
        self.counters.cursor()
    }

    pub fn bump_head(&self) {
        self.counters.bump_head();
    }

    pub fn bump_tail(&self) {
        self.counters.bump_tail();
    }

    pub fn attach(&self) -> Result<u32> {
        self.counters.attach()
    }

    pub fn detach(&self) {
        self.counters.detach();
    }

    #[must_use]
    pub fn attachments(&self) -> u32 {
        self.counters.attachments.load(Ordering::Acquire)
    }

    pub fn claim_seeding(&self) -> bool {
        self.counters.claim_seeding()
    }
}

impl Counters {
    fn cursor(&self) -> Cursor {
        Cursor {
            head: self.head.load(Ordering::Acquire),
            tail: self.tail.load(Ordering::Acquire),
        }
    }

    fn bump_head(&self) {
        self.head.fetch_add(1, Ordering::Release);
    }

    fn bump_tail(&self) {
        self.tail.fetch_add(1, Ordering::Release);
    }

    fn attach(&self) -> Result<u32> {
        let taken = self.attachments.fetch_add(1, Ordering::AcqRel) + 1;
        if taken > MAX_ATTACHMENTS {
            self.attachments.fetch_sub(1, Ordering::AcqRel);
            return Err(Fault::program(
                "program::shared",
                format!(
                    "this channel already has {MAX_ATTACHMENTS} instances bound to it \
                     and a {taken}th asked to bind: a shared ring is ordered by the \
                     pipeline FIFO its attachments fire in, and that bound is \
                     {MAX_ATTACHMENTS} — past it there is no ordering \
                     argument, so there is no ring"
                ),
            ));
        }
        Ok(taken)
    }

    fn detach(&self) {
        let _ = self
            .attachments
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(1))
            });
    }

    fn claim_seeding(&self) -> bool {
        self.seeded
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{Counters, MAX_ATTACHMENTS};

    fn ring() -> Counters {
        Counters::default()
    }

    #[test]
    fn a_shared_ring_seats_eight_attachments_and_refuses_the_ninth() {
        let ring = ring();
        for seat in 1..=MAX_ATTACHMENTS {
            assert_eq!(ring.attach().expect("a seat inside the bound"), seat);
        }
        let ninth = ring.attach();
        assert!(ninth.is_err(), "the ninth attachment is refused: {ninth:?}");
        let why = format!("{}", ninth.expect_err("just checked"));
        assert!(why.contains("8"), "the refusal names the bound: {why}");
        ring.detach();
        assert_eq!(
            ring.attach().expect("the seat that was just given back"),
            MAX_ATTACHMENTS
        );
    }
}
