//! **A device-only ring that belongs to its CHANNEL, not to one instance**
//! (design §5: *"draft→verify chaining is free: a device-only private ring
//! shared by ≤8 attachments, ordered by the pipeline FIFO"*).
//!
//! # What was broken here
//!
//! [`Rings::allocate`](super::launch::Rings::allocate) cuts one slab per
//! channel per INSTANCE and [`Session`](super::session::Session) keeps that
//! channel's two cursors in its own `Vec<Cursor>`. That is exactly right for a
//! ring one pass owns — a loop-carried decode accumulator, a `hook_fold_acc` —
//! and silently wrong for the shape design §5 names by hand: a ring two passes
//! SHARE, one putting and the other taking. Two sessions cut two slabs and
//! kept two cursors, so the prefill epilogue's put landed in one copy and the
//! decode pass's `embed` read the other, forever empty. The bench guest died
//! at its first decode frame, every request, with
//!
//! ```text
//! channel 0 is empty and its program takes from it (needs a cell, holds 0 of 1)
//! ```
//!
//! — a true sentence about a ring nobody had written. `engine-cuda` fixed the
//! same defect in `27de300fa` by giving the ring to the channel; this is that
//! ring, in this plane's own materials.
//!
//! # What one carries, and what it deliberately does not
//!
//! Two things, because on this plane a ring IS those two things:
//!
//! * **the device slab.** One `MTLBuffer` of `capacity + 1` cells, handed to
//!   every attachment as a CLONE — which on this plane is a retain of the same
//!   allocation, so `Rings` keeps its one-buffer-per-channel shape and every
//!   session's `write_cell`/`read_cell`/`slab` path works unchanged against
//!   the same bytes.
//! * **the two cursors**, as atomics. The CUDA sibling pins two host words
//!   because its device kernels read them; nothing on this plane reads a
//!   cursor from the device at all — readiness and commit are host arithmetic
//!   (`session`'s header) — so what a shared cursor has to be is a number two
//!   SESSIONS can agree on, and an `AtomicU64` is that.
//!
//! What it does NOT carry is a host mirror, and that is the role's own
//! statement rather than an omission: a [`HostRole::None`] channel has no
//! guest end, its cells never cross, and `api::register_channel` answers a
//! [`RegisteredChannel`](engine::channel::RegisteredChannel) with `mirror:
//! None` so the runtime keeps owning the host rings of the channels that do.
//!
//! # Where the fence is, and why it is not here
//!
//! A shared ring on this plane needs one thing its CUDA twin does not: the
//! consumer resolves its descriptor ports on the HOST, at `serve::stage`,
//! before anything is encoded — so when the decode pass reads this ring's
//! committed cell, the producer's put may still be inside a command buffer
//! that has not landed. That dependency is threaded at the boundary this
//! shell already has (`serve::fence_instances`, widened by
//! [`Plane::cohort`](super::Plane::cohort)) rather than by a second one here:
//! the cursors below advance at [`Session::commit`], which runs at the
//! harvest, so a cell is visible through this ring exactly when the flight
//! that wrote it has landed. See `serve::fence_instances`' own note.
//!
//! [`HostRole::None`]: eta_ir::container::HostRole::None
//! [`Session::commit`]: super::session::Session

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor};

/// **How many instances may share one ring** — design §5's number, verbatim
/// ("a device-only private ring shared by ≤8 attachments").
///
/// A bound rather than a budget: the ordering argument for a shared ring is
/// that its attachments fire in pipeline order, and a ring with an unbounded
/// number of attachments is a ring whose order nobody has stated. Eight is
/// what the design says, so eight is what [`SharedRing::attach`] refuses past
/// — by name, because silently serving a ninth would be serving a shape the
/// design has no ordering argument for. The same number the CUDA sibling
/// refuses at (`program::endpoint::MAX_ATTACHMENTS`).
pub const MAX_ATTACHMENTS: u32 = 8;

/// One device-only channel's ring: the slab every attachment addresses and
/// the counters they all read.
#[derive(Debug)]
pub struct SharedRing {
    /// What the registration declared. Every attachment's own declaration is
    /// held against this at bind — a ring cut for one cell width and
    /// addressed at another is a wrong token and never a fault.
    shape: ChannelShape,
    /// **The one slab, `capacity + 1` cells at the shape's stride.** Handed
    /// out by clone, which is a retain: every attachment's `Rings` holds the
    /// same `MTLBuffer` at the same offsets.
    slab: Buffer,
    /// Everything about this ring that is a NUMBER rather than a reservation.
    counters: Counters,
}

/// The ring's bookkeeping: two cursors, the seats, and the seeding claim.
///
/// **SPLIT OUT FROM THE SLAB SO IT CAN BE TESTED WITHOUT A DEVICE.** A
/// [`Buffer`] cannot be built off a bound [`Context`], and none of the
/// arithmetic below touches one — the seats are a bound, the seeding claim is
/// a race, and the cursors are counts. The CUDA sibling makes the same split
/// by building an `Endpoint` around a null pinned mapping; this plane has no
/// null reservation to build around, so the numbers are their own struct.
#[derive(Debug, Default)]
struct Counters {
    /// The committed front — the cell a take reads.
    head: AtomicU64,
    /// The pending back — the cell a put writes.
    tail: AtomicU64,
    /// How many instances hold a seat, [`MAX_ATTACHMENTS`] at most.
    attachments: AtomicU32,
    /// **Has a bind already planted this ring's seeds?**
    ///
    /// A seed is a cell the ring starts life holding, and a shared ring starts
    /// life once. Two instances binding a seeded shared channel each arrive
    /// carrying the same seed bytes — the runtime hands every attachment the
    /// declaration — and planting them twice would leave the ring holding the
    /// seed twice and its tail two on. So the first bind claims the right to
    /// seed and the rest are told they lost the race
    /// ([`SharedRing::claim_seeding`]).
    seeded: AtomicU32,
}

// SAFETY: everything below the `slab` is an atomic, and the slab is a
// `device::Buffer` — whose own `Send` rests on `MTLBuffer` being documented
// thread-safe for retain/release, `contents` and encoder binding, which are
// the only operations any holder of this ring performs on it. `Sync` is the
// same argument one step further: the buffer is never mutated as an object
// after `open`, and the bytes inside it are ordered by the flight boundary
// the module header names, not by a `&mut`.
unsafe impl Send for SharedRing {}
// SAFETY: as above.
unsafe impl Sync for SharedRing {}

impl SharedRing {
    /// Cut one channel's ring: `capacity + 1` cells, both cursors at zero.
    ///
    /// The spare cell is what makes "full" distinguishable from "empty" with
    /// two monotone cursors — the same rule [`Rings::allocate`] cuts a
    /// per-instance ring by, and stated once in
    /// [`ChannelShape::cell_stride`] rather than twice here.
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

    /// **Take one of this ring's [`MAX_ATTACHMENTS`] seats**, or say who is
    /// already in them.
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

    /// **Claim the right to plant this ring's seeds**, `true` for the first
    /// caller and `false` for every one after it.
    ///
    /// See [`Counters::seeded`](Counters#structfield.seeded): a shared ring
    /// starts life once, and every attachment arrives carrying the same
    /// declaration.
    pub fn claim_seeding(&self) -> bool {
        self.counters.claim_seeding()
    }
}

impl Counters {
    fn cursor(&self) -> Cursor {
        // `Acquire` on both, paired with the `Release` in `bump_*`: a reader
        // that sees the advanced counter sees the cell write that preceded
        // it. On this plane the cell write is usually a KERNEL's, ordered by
        // the flight boundary the module header names rather than by this
        // pair — what these orderings carry across on their own is the HOST's
        // write, which is the seeding put.
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
                     pipeline FIFO its attachments fire in, and design §5 states that \
                     bound at {MAX_ATTACHMENTS} — past it there is no ordering \
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
    use crate::program::launch::Cursor;

    /// The ring's numbers with no reservation behind them — see [`Counters`]
    /// for why the split exists.
    fn ring() -> Counters {
        Counters::default()
    }

    /// **THE EIGHT SEATS ARE A BOUND AND THE NINTH IS A REFUSAL** (design §5:
    /// "a device-only private ring shared by <=8 attachments").
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
        // **AND THE REFUSAL LEAVES THE COUNT WHERE IT WAS**, so a caller that
        // is refused and retries after closing something is not permanently
        // one seat over.
        ring.detach();
        assert_eq!(
            ring.attach().expect("the seat that was just given back"),
            MAX_ATTACHMENTS
        );
    }

    /// A shared ring is seeded ONCE, however many attachments carry the same
    /// declaration to it.
    #[test]
    fn only_the_first_attachment_plants_a_shared_rings_seeds() {
        let ring = ring();
        assert!(ring.claim_seeding(), "the first bind plants the seed");
        assert!(!ring.claim_seeding(), "and the second finds it already there");
        assert!(!ring.claim_seeding());
    }

    /// The cursors are the RING's, so a bump by one attachment is a read by
    /// the next — which is the whole of what "the ring belongs to the
    /// channel" buys the host arithmetic.
    #[test]
    fn a_bump_by_one_attachment_is_what_the_next_one_reads() {
        let ring = ring();
        assert_eq!(ring.cursor(), Cursor { head: 0, tail: 0 });
        ring.bump_tail();
        assert_eq!(
            ring.cursor(),
            Cursor { head: 0, tail: 1 },
            "the putter's commit is the taker's depth"
        );
        ring.bump_head();
        assert_eq!(ring.cursor(), Cursor { head: 1, tail: 1 });
    }
}
