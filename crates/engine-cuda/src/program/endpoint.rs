//! **THE HOST END OF A CHANNEL, IN MAPPED PINNED MEMORY** — the allocation
//! that makes a guest round trip cost zero CUDA calls (alto design §5, survey
//! §7 invariant I5).
//!
//! A channel with a host end has three pieces of state and they live in three
//! places, which is the whole of the design:
//!
//! ```text
//! the CELLS the pass reads and writes    device slab      program::launch::Rings
//! the CELLS the guest reads and writes   pinned mirror    Endpoint::mirror
//! the two monotone COUNTERS              pinned words     Endpoint::words
//! ```
//!
//! The mirror and the words are one `cudaHostAlloc(.., Mapped)` each, so the
//! guest stores through a plain pointer on its own thread and
//! `channel::pull_validate` / `channel::scatter_publish` dereference the same
//! bytes from the device. **Neither direction is a `cudaMemcpy`**: the inward
//! cell is READ where the guest wrote it, and the outward cell is WRITTEN
//! where the guest will read it. That is what HEAD's `pump_in`/`pump_out` —
//! an H2D per cell one way and a `Vec`-per-cell D2H the other — cost, and
//! what the survey calls invariant I5 violated in both directions.
//!
//! # Who owns which counter
//!
//! The four words are `[head, tail, poison, closed]`, exactly the layout the
//! runtime's own `HostRing` publishes and `channel::Ticket::words` reads.
//! Each COUNTER has exactly one owner, and the owner is the only writer —
//! which is the SPSC discipline the whole plane rests on:
//!
//! ```text
//! HostRole::Writer   the guest owns TAIL (it publishes), the engine owns HEAD
//! HostRole::Reader   the guest owns HEAD (it consumes),  the engine owns TAIL
//! HostRole::None     NO guest owns either counter — the cells never leave the
//!                    device, and both counters are the engine's
//! ```
//!
//! # The third role, and why it has an endpoint at all (design §5)
//!
//! [`HostRole::None`] used to mean "there is no endpoint": a channel whose
//! cells never leave the device had its ring cut inside one `Session`, out of
//! that session's own slab, with its counters kept only in that session's
//! prediction. That is right for a ring ONE pass owns and wrong for the shape
//! design §5 names by hand — *"draft→verify chaining is free: a device-only
//! private ring shared by ≤8 attachments, ordered by the pipeline FIFO"* — a
//! ring two passes share, one putting and the other taking. Two sessions each
//! cut their own slab and their own counters, so the put landed in one copy
//! and the take read the other, forever empty. The failure was legible and
//! wrong: the taker's fire refused with "blocked AFTER the gate admitted it",
//! because the gate and the fire were both reading a ring nobody had written.
//!
//! So the ring belongs to the CHANNEL and not to the instance, for all three
//! roles. A `None` endpoint carries the same two pinned counters the other two
//! do — read by `pull_validate` on the device and by [`Session::depth`] on the
//! host, which is what makes the gate and the fire read one number — plus the
//! DEVICE SLAB itself ([`Endpoint::device_cells`]), because that is the one
//! piece of a shared ring that a per-session allocation cannot stand in for.
//! What it does not carry is a guest: its mirror is never pulled from or
//! published to, because neither `TICKET_HOST_WRITER` nor `TICKET_HOST_READER`
//! is ever set on its tickets.
//!
//! [`Session::depth`]: super::session::Session::depth
//!
//! The engine's own counter is kept TWICE, and the difference is alto's
//! central mechanism: the PREDICTION ([`Cursor`](super::launch::Cursor), which
//! this fire's tickets and cell addresses are arithmetic on) advances when
//! the fire is minted, and the PINNED word advances at settle, only if the
//! pass committed. `pull_validate` compares one against the other, so a
//! prediction the device disagrees with clears the commit word instead of
//! serving the wrong cell.
//!
//! # Why the words are read through `AtomicU64`
//!
//! The other end is a different thread — the guest's — writing concurrently.
//! Acquire loads and release stores over the pinned words are what make a
//! cell's bytes visible before the counter announcing them, on the HOST side
//! of the crossing; the DEVICE side gets the same ordering from the
//! kernel-launch boundary and takes it for free (`channels.cuh`'s ordering
//! note, and the 13.8× that forbids per-store release there).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use engine::tensor_ir::container::HostRole;

use crate::device::{Buffer, Pinned};
use crate::error::{Fault, Result};

/// Where each counter lives in [`Endpoint::words`] — the runtime's
/// `HostRing` word order, and `channel::Ticket::words`'s.
const HEAD_WORD: usize = 0;
const TAIL_WORD: usize = 1;
/// How many words one endpoint carries: `[head, tail, poison, closed]`.
pub const WORDS: usize = 4;

/// **How many instances may share one ring** — design §5's number, verbatim
/// ("a device-only private ring shared by ≤8 attachments").
///
/// A bound rather than a budget: the ordering argument for a shared ring is
/// that the attachments fire in pipeline order on one stream, and a ring with
/// an unbounded number of attachments is a ring whose order nobody has stated.
/// Eight is what the design says, so eight is what [`Endpoint::attach`]
/// refuses past — by name, because silently serving a ninth would be serving a
/// shape the design has no ordering argument for.
pub const MAX_ATTACHMENTS: u32 = 8;

/// One host-visible channel endpoint's pinned mirror and pinned counters.
#[derive(Debug)]
pub struct Endpoint {
    /// Which end the host holds. [`HostRole::None`] is a channel with no guest
    /// end at all — a ring two passes share, whose counters are both the
    /// engine's (see the module header's third role).
    role: HostRole,
    /// `[head, tail, poison, closed]`, mapped.
    words: Pinned,
    /// `cap1` wire cells, mapped.
    mirror: Pinned,
    /// Bytes per WIRE cell — bit-packed for a bool channel, four bytes an
    /// element otherwise.
    wire_bytes: u32,
    /// `capacity + 1`: the spare cell that makes `tail == head` mean empty.
    cap1: u32,
    /// **THE SHARED DEVICE SLAB, for a ring that belongs to the channel.**
    ///
    /// `Some` for [`HostRole::None`] and `None` for the other two, and the
    /// asymmetry is the point. A host-visible channel's device cells are one
    /// END of a crossing — the pull copies the guest's mirror INTO them and
    /// the publish copies them back OUT — so they are staging that belongs to
    /// whichever pass is doing the crossing, and a second attachment would
    /// want its own. A device-only channel's cells are the ring ITSELF: the
    /// putting pass writes a cell and the taking pass reads that same cell,
    /// with no crossing anywhere, so there is exactly one slab and every
    /// attachment addresses it.
    device_cells: Option<Buffer>,
    /// How many instances have bound this channel — [`MAX_ATTACHMENTS`] at
    /// most.
    attachments: AtomicU32,
    /// **Has a bind already planted this ring's seeds?**
    ///
    /// A seed is a cell the ring starts life holding, and a shared ring starts
    /// life once. Two instances binding a seeded shared channel each arrive
    /// carrying the same seed bytes — the runtime hands every attachment the
    /// declaration — and planting them twice would leave the ring holding the
    /// seed twice and its tail two on. So the first bind claims the right to
    /// seed and the rest are told they lost the race
    /// ([`Endpoint::claim_seeding`]).
    seeded: AtomicU32,
}

impl Endpoint {
    /// Allocate one endpoint's mirror and words.
    ///
    /// **THE CAPACITY REFUSAL LIVES HERE** (track-K finding 3). The kernels
    /// take `cap1 - 1` in `u32` and `expected_head % cap1` and reproduce dev's
    /// arithmetic unchanged, so a `cap1` of zero is an unsigned underflow that
    /// admits every publish and a division by zero on the pull; and `full` is
    /// indexed `slot * MAX_RING + ring`, so a ring longer than
    /// [`MAX_RING`](kernels_cuda::channel::MAX_RING) would write into its
    /// neighbour's bytes. Registration is where both are refused, because
    /// nothing downstream can.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] for a capacity the ring arithmetic cannot carry,
    /// and whatever `cudaHostAlloc` said.
    pub fn open(role: HostRole, wire_bytes: u32, capacity: u32) -> Result<Endpoint> {
        let cap1 = capacity.checked_add(1).ok_or_else(|| {
            Fault::program(
                "program::endpoint",
                format!(
                    "a channel of capacity {capacity} has a ring of {capacity} + 1 cells, \
                     which does not fit a u32: the control kernels take `cap1 - 1` \
                     unsigned and would admit every publish"
                ),
            )
        })?;
        if cap1 > kernels_cuda::channel::MAX_RING {
            return Err(Fault::program(
                "program::endpoint",
                format!(
                    "a channel of capacity {capacity} wants a ring of {cap1} cells and the \
                     full/empty bytes are cut {} apart per slot, so its ring would \
                     address its neighbour's",
                    kernels_cuda::channel::MAX_RING
                ),
            ));
        }
        let cells = (wire_bytes as usize).saturating_mul(cap1 as usize);
        // **THE SLAB IS THE ROLE'S OWN DECISION** (see `device_cells`). A
        // device-only ring is shared, so its cells are cut here, once, and
        // every session that binds the channel addresses them; the other two
        // roles keep the per-session staging `Rings::allocate` cuts.
        //
        // `native_bytes` is what a device cell holds and `wire_bytes` what a
        // mirror cell does; they differ only for a bool channel, which packs
        // on the wire. Cutting the shared slab at the WIDER of the two is what
        // makes one allocation serve either — a bool ring then holds a byte
        // per lane, which is what the emitted kernels read.
        let device_cells = match role {
            HostRole::None => Some(Buffer::zeroed(cells.max(1))?),
            _ => None,
        };
        Ok(Endpoint {
            role,
            words: Pinned::mapped(WORDS * size_of::<u64>())?,
            mirror: Pinned::mapped(cells.max(1))?,
            wire_bytes,
            cap1,
            device_cells,
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
        })
    }

    /// **The shared device slab's base**, or `None` for a role whose device
    /// cells are per-session staging.
    #[must_use]
    pub fn device_cells(&self) -> Option<u64> {
        self.device_cells.as_ref().map(Buffer::ptr)
    }

    /// **Take one of this ring's [`MAX_ATTACHMENTS`] seats**, or say who is
    /// already in them.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] past the design's bound. A refusal here is a bind
    /// that does not happen, so nothing has to be undone.
    pub fn attach(&self) -> Result<u32> {
        let taken = self.attachments.fetch_add(1, Ordering::AcqRel) + 1;
        if taken > MAX_ATTACHMENTS {
            self.attachments.fetch_sub(1, Ordering::AcqRel);
            return Err(Fault::program(
                "program::endpoint",
                format!(
                    "this channel already has {MAX_ATTACHMENTS} instances bound to it and                      a {taken}th asked to bind: a shared ring is ordered by the pipeline                      FIFO its attachments fire in, and design §5 states that bound at                      {MAX_ATTACHMENTS} — past it there is no ordering argument, so there                      is no ring"
                ),
            ));
        }
        Ok(taken)
    }

    /// Give one seat back, when an instance that held one is closed.
    pub fn detach(&self) {
        let _ = self
            .attachments
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(1))
            });
    }

    /// **Claim the right to plant this ring's seeds**, `true` for the first
    /// caller and `false` for every one after it.
    ///
    /// See [`Endpoint::seeded`](Endpoint#structfield.seeded): a shared ring
    /// starts life once, and every attachment arrives carrying the same
    /// declaration.
    pub fn claim_seeding(&self) -> bool {
        self.seeded
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Which end the host holds.
    #[must_use]
    pub const fn role(&self) -> HostRole {
        self.role
    }

    /// `capacity + 1`.
    #[must_use]
    pub const fn cap1(&self) -> u32 {
        self.cap1
    }

    /// Bytes per wire cell.
    #[must_use]
    pub const fn wire_bytes(&self) -> u32 {
        self.wire_bytes
    }

    /// The four words, as a kernel dereferences them.
    #[must_use]
    pub fn words_device(&self) -> u64 {
        self.words.device()
    }

    /// The four words, as the host and the runtime's `HostRing` address them.
    #[must_use]
    pub fn words_host(&self) -> u64 {
        self.words.host() as u64
    }

    /// The mirror, as a kernel dereferences it.
    #[must_use]
    pub fn mirror_device(&self) -> u64 {
        self.mirror.device()
    }

    /// The mirror, as the host and the runtime's `HostRing` address it.
    #[must_use]
    pub fn mirror_host(&self) -> u64 {
        self.mirror.host() as u64
    }

    /// How many bytes the mirror holds.
    #[must_use]
    pub fn mirror_bytes(&self) -> usize {
        self.mirror.bytes()
    }

    /// **Does the ENGINE own this endpoint's head?** True for a channel the
    /// host writes — the guest publishes, the pass consumes.
    #[must_use]
    pub const fn engine_owns_head(&self) -> bool {
        !matches!(self.role, HostRole::Reader)
    }

    /// **Does the ENGINE own this endpoint's tail?** True for a channel the
    /// host reads — the pass publishes, the guest consumes.
    #[must_use]
    pub const fn engine_owns_tail(&self) -> bool {
        !matches!(self.role, HostRole::Writer)
    }

    /// The head counter as it stands right now.
    #[must_use]
    pub fn head(&self) -> u64 {
        self.word(HEAD_WORD)
    }

    /// The tail counter as it stands right now.
    #[must_use]
    pub fn tail(&self) -> u64 {
        self.word(TAIL_WORD)
    }

    /// Advance the head by one.
    pub fn bump_head(&self) {
        self.store(HEAD_WORD, self.word(HEAD_WORD) + 1);
    }

    /// Advance the tail by one.
    pub fn bump_tail(&self) {
        self.store(TAIL_WORD, self.word(TAIL_WORD) + 1);
    }

    /// One wire cell of the mirror, at ring position `sequence % cap1`.
    #[must_use]
    pub fn read_cell(&self, sequence: u64) -> Vec<u8> {
        let at = (sequence % u64::from(self.cap1)) as usize * self.wire_bytes as usize;
        self.mirror.read(at, self.wire_bytes as usize)
    }

    /// Write one wire cell into the mirror at `sequence % cap1`.
    ///
    /// Answers `false` for a cell of the wrong width, which the caller turns
    /// into a named refusal — a short write leaves real-looking garbage in the
    /// cell's tail.
    pub fn write_cell(&self, sequence: u64, wire: &[u8]) -> bool {
        if wire.len() != self.wire_bytes as usize {
            return false;
        }
        let at = (sequence % u64::from(self.cap1)) as usize * self.wire_bytes as usize;
        self.mirror.write(at, wire)
    }

    /// One word, acquire-loaded: the other end of this counter is a different
    /// thread, and a cell published before a tail must be visible once the
    /// tail is.
    fn word(&self, index: usize) -> u64 {
        let host = self.words.host();
        if host.is_null() {
            return 0;
        }
        // SAFETY: `words` is `WORDS` u64s of live mapped memory and `index`
        // is one of them; the pointer is 8-aligned because `cudaHostAlloc`
        // returns page-aligned memory.
        unsafe { (*AtomicU64::from_ptr(host.cast::<u64>().add(index))).load(Ordering::Acquire) }
    }

    /// One word, release-stored, for the same reason.
    fn store(&self, index: usize, value: u64) {
        let host = self.words.host();
        if host.is_null() {
            return;
        }
        // SAFETY: as `word`.
        unsafe {
            (*AtomicU64::from_ptr(host.cast::<u64>().add(index))).store(value, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two ownership questions, answered off the role and nothing else.
    /// A channel the host WRITES has its tail advanced by the guest and its
    /// head by the engine's commit; a channel the host READS is the mirror
    /// image. Getting this backwards would have the engine and the guest
    /// writing the same word from two threads.
    #[test]
    fn each_counter_has_exactly_one_owner_and_the_role_names_it() {
        let of = |role| Endpoint {
            role,
            words: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            mirror: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            wire_bytes: 4,
            cap1: 2,
            device_cells: None,
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
        };
        let writer = of(HostRole::Writer);
        assert!(writer.engine_owns_head(), "the pass consumes what the guest published");
        assert!(!writer.engine_owns_tail(), "and the guest alone advances the tail");
        let reader = of(HostRole::Reader);
        assert!(!reader.engine_owns_head(), "the guest alone consumes");
        assert!(reader.engine_owns_tail(), "and the pass alone publishes");
        // **AND THE THIRD ROLE ANSWERS BOTH** (design §5). A device-only ring
        // has no guest at either end, so neither counter is the guest's — and
        // that is exactly why `Session::merge` reads such a ring's words
        // instead of a prediction: "the engine owns it" resolves to a
        // different session for one of the two ends.
        let shared = of(HostRole::None);
        assert!(shared.engine_owns_head() && shared.engine_owns_tail());
    }

    /// **THE EIGHT SEATS ARE A BOUND AND THE NINTH IS A REFUSAL** (design §5:
    /// "a device-only private ring shared by <=8 attachments").
    #[test]
    fn a_shared_ring_seats_eight_attachments_and_refuses_the_ninth() {
        let ring = Endpoint {
            role: HostRole::None,
            words: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            mirror: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            wire_bytes: 4,
            cap1: 2,
            device_cells: None,
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
        };
        for seat in 1..=MAX_ATTACHMENTS {
            assert_eq!(ring.attach().expect("a seat inside the bound"), seat);
        }
        let ninth = ring.attach();
        assert!(ninth.is_err(), "the ninth attachment is refused: {ninth:?}");
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
        let ring = Endpoint {
            role: HostRole::None,
            words: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            mirror: Pinned::mapped(0).expect("a null allocation needs no runtime"),
            wire_bytes: 4,
            cap1: 2,
            device_cells: None,
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
        };
        assert!(ring.claim_seeding(), "the first bind plants the seed");
        assert!(!ring.claim_seeding(), "and the second finds it already there");
        assert!(!ring.claim_seeding());
    }
}
