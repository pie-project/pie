//! A device-only ring that belongs to its channel, not to one instance: a
//! private ring shared by up to 8 attachments, ordered by the pipeline FIFO.
//! Carries the device slab and the two cursors as atomics; visibility across
//! the shared ring is fenced at `serve::fence_instances`, not here, since
//! cursors advance at [`Session::commit`] on harvest.
//!
//! [`Session::commit`]: super::session::Session

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor};

/// How many instances may share one ring. A bound, not a budget: attachments
/// fire in pipeline order, so an unbounded ring has no stated order. Matches
/// `program::endpoint::MAX_ATTACHMENTS` in the CUDA sibling.
pub const MAX_ATTACHMENTS: u32 = 8;

/// One device-only channel's ring: the slab every attachment addresses and
/// the counters they all read.
#[derive(Debug)]
pub struct SharedRing {
    /// What the registration declared; every attachment's own declaration is
    /// held against this at bind.
    shape: ChannelShape,
    /// The one slab, `capacity + 1` cells at the shape's stride. Handed out
    /// by clone, which is a retain: every attachment shares the same
    /// `MTLBuffer` at the same offsets.
    slab: Buffer,
    /// Everything about this ring that is a number rather than a reservation.
    counters: Counters,
}

/// The ring's bookkeeping: two cursors, the seats, and the seeding claim.
/// Split out from the slab so it can be tested without a device.
#[derive(Debug, Default)]
struct Counters {
    /// The committed front — the cell a take reads.
    head: AtomicU64,
    /// The pending back — the cell a put writes.
    tail: AtomicU64,
    /// How many instances hold a seat, [`MAX_ATTACHMENTS`] at most.
    attachments: AtomicU32,
    /// Whether a bind has already planted this ring's seeds. A shared ring
    /// is seeded once; the first bind claims the right and the rest lose
    /// the race ([`SharedRing::claim_seeding`]).
    seeded: AtomicU32,
}

// SAFETY: everything below `slab` is an atomic; `slab` is a
// `device::Buffer`, whose `Send`/`Sync` rest on `MTLBuffer` being
// thread-safe for retain/release, `contents`, and encoder binding.
unsafe impl Send for SharedRing {}
// SAFETY: as above.
unsafe impl Sync for SharedRing {}

impl SharedRing {
    /// Cut one channel's ring: `capacity + 1` cells, both cursors at zero.
    /// The spare cell distinguishes "full" from "empty" with two monotone
    /// cursors.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a shape whose ring will not fit a `u64`,
    /// [`Fault::Device`] when the device declined the reservation, and
    /// [`Fault::Deviceless`] off Apple.
    ///
    /// [`Rings::allocate`]: super::launch::Rings::allocate
    pub fn open(device: &Context, shape: ChannelShape) -> Result<SharedRing> {
        let cells = u64::from(shape.capacity) + 1;
        let bytes = cells
            .checked_mul(shape.cell_stride() as u64)
            .ok_or_else(|| Fault::program("program::shared", "a ring past what a u64 counts"))?;
        Ok(SharedRing {
            shape,
            slab: Buffer::zeroed(device, bytes.max(1))?,
            counters: Counters::default(),
        })
    }

    /// The geometry this ring was cut at.
    #[must_use]
    pub const fn shape(&self) -> ChannelShape {
        self.shape
    }

    /// A retain of the one slab, for an attachment's own [`Rings`].
    ///
    /// [`Rings`]: super::launch::Rings
    #[must_use]
    pub fn slab(&self) -> Buffer {
        self.slab.clone()
    }

    /// Where this ring stands right now, as both attachments see it.
    #[must_use]
    pub fn cursor(&self) -> Cursor {
        self.counters.cursor()
    }

    /// Advance the committed front by one — a take that committed.
    pub fn bump_head(&self) {
        self.counters.bump_head();
    }

    /// Advance the pending back by one — a put that committed.
    pub fn bump_tail(&self) {
        self.counters.bump_tail();
    }

    /// Take one of this ring's [`MAX_ATTACHMENTS`] seats.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] past the design's bound. A refusal here is a bind
    /// that does not happen, so nothing has to be undone.
    pub fn attach(&self) -> Result<u32> {
        self.counters.attach()
    }

    /// Give one seat back, when an instance that held one is closed.
    pub fn detach(&self) {
        self.counters.detach();
    }

    /// How many seats are taken.
    #[must_use]
    pub fn attachments(&self) -> u32 {
        self.counters.attachments.load(Ordering::Acquire)
    }

    /// Claim the right to plant this ring's seeds: `true` for the first
    /// caller, `false` after.
    pub fn claim_seeding(&self) -> bool {
        self.counters.claim_seeding()
    }
}

impl Counters {
    fn cursor(&self) -> Cursor {
        // `Acquire` paired with `Release` in `bump_*`: a reader that sees
        // the advanced counter sees the preceding cell write.
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
    

    /// The ring's numbers with no reservation behind them.
    fn ring() -> Counters {
        Counters::default()
    }

    /// Eight seats is a bound; the ninth is refused.
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
        // The refusal leaves the count where it was.
        ring.detach();
        assert_eq!(
            ring.attach().expect("the seat that was just given back"),
            MAX_ATTACHMENTS
        );
    }

}
