//! Host end of a channel, in mapped pinned memory: guest and device touch the
//! mirror/words directly with no `cudaMemcpy`. Words are `AtomicU64`
//! acquire/release since the guest writes them from its own thread.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eta_ir::container::HostRole;

use crate::device::{Buffer, Pinned};
use crate::error::{Fault, Result};

/// Word offsets in [`Endpoint::words`], matching the runtime's `HostRing`.
const HEAD_WORD: usize = 0;
const TAIL_WORD: usize = 1;
/// How many words one endpoint carries: `[head, tail, poison, closed]`.
pub const WORDS: usize = 4;

/// Max instances that may share one ring; [`Endpoint::attach`] refuses past it.
pub const MAX_ATTACHMENTS: u32 = 8;

/// One host-visible channel endpoint's pinned mirror and pinned counters.
#[derive(Debug)]
pub struct Endpoint {
    /// Which end the host holds. [`HostRole::None`] is a device-only ring
    /// shared by two passes; both counters are the engine's.
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
    /// The shared device slab. `Some` only for [`HostRole::None`]: a
    /// device-only ring's cells ARE the ring, so one slab serves every
    /// attachment; other roles use per-session staging instead.
    device_cells: Option<Buffer>,
    /// How many instances have bound this channel — [`MAX_ATTACHMENTS`] at
    /// most.
    attachments: AtomicU32,
    /// Host-side prediction of the ring's position, kept on the endpoint
    /// since a device-only ring has multiple attachments with no single
    /// session owner. Advances at mint, rolls back on refusal.
    predicted_head: AtomicU64,
    predicted_tail: AtomicU64,
    /// Whether a bind has already planted this ring's seed cells; the first
    /// bind claims it via [`Endpoint::claim_seeding`], later ones lose the race.
    seeded: AtomicU32,
}

impl Endpoint {
    /// Allocate one endpoint's mirror and words.
    ///
    /// Refuses a `cap1` of zero or one exceeding `MAX_RING`, since nothing
    /// downstream can catch either.
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
        // A bool channel packs on the wire but gets a byte per lane on device.
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
            predicted_head: AtomicU64::new(0),
            predicted_tail: AtomicU64::new(0),
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
        })
    }

    /// The shared device slab's base, or `None` for a role whose device
    /// cells are per-session staging.
    #[must_use]
    pub fn device_cells(&self) -> Option<u64> {
        self.device_cells.as_ref().map(Buffer::ptr)
    }

    /// Take one of this ring's [`MAX_ATTACHMENTS`] seats.
    ///
    /// # Errors
    ///
    /// [`Fault::Program`] past the bound. A refusal here is a bind that
    /// doesn't happen, so nothing needs undoing.
    pub fn attach(&self) -> Result<u32> {
        let taken = self.attachments.fetch_add(1, Ordering::AcqRel) + 1;
        if taken > MAX_ATTACHMENTS {
            self.attachments.fetch_sub(1, Ordering::AcqRel);
            return Err(Fault::program(
                "program::endpoint",
                format!(
                    "this channel already has {MAX_ATTACHMENTS} instances bound to it and a \
                     {taken}th asked to bind: a shared ring is ordered by the pipeline FIFO its \
                     attachments fire in, and that bound is {MAX_ATTACHMENTS} — \
                     past it there is no ordering argument, so there is no ring"
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

    /// Claims the right to plant this ring's seeds: `true` for the first
    /// caller, `false` after.
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

    /// True if the engine owns this endpoint's head (host writes: guest
    /// publishes, pass consumes).
    #[must_use]
    pub const fn engine_owns_head(&self) -> bool {
        !matches!(self.role, HostRole::Reader)
    }

    /// True if the engine owns this endpoint's tail (host reads: pass
    /// publishes, guest consumes).
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

    /// Advances the head by one, word and prediction together. Only the
    /// host-side path (seed/`host_take`/`host_put`) does this; the fire path
    /// advances them separately, at mint and at device settle.
    pub fn bump_head(&self) {
        self.store(HEAD_WORD, self.word(HEAD_WORD) + 1);
        self.predict_head();
    }

    /// [`Endpoint::bump_head`] for the tail.
    pub fn bump_tail(&self) {
        self.store(TAIL_WORD, self.word(TAIL_WORD) + 1);
        self.predict_tail();
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

    /// Where this ring stands as the host has counted it: the prediction,
    /// not the pinned words.
    #[must_use]
    pub fn predicted(&self) -> (u64, u64) {
        (
            self.predicted_head.load(Ordering::Acquire),
            self.predicted_tail.load(Ordering::Acquire),
        )
    }

    /// Advance the predicted head by one, at mint.
    pub fn predict_head(&self) {
        self.predicted_head.fetch_add(1, Ordering::AcqRel);
    }

    /// Advance the predicted tail by one, at mint.
    pub fn predict_tail(&self) {
        self.predicted_tail.fetch_add(1, Ordering::AcqRel);
    }

    /// Rolls a prediction back, for a fire the device refused. Saturating: a
    /// counter that went below its word would be the same failure mirrored.
    pub fn unpredict_head(&self, by: u64) {
        let _ = self
            .predicted_head
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |at| {
                Some(at.saturating_sub(by))
            });
    }

    /// [`Endpoint::unpredict_head`] for the tail.
    pub fn unpredict_tail(&self, by: u64) {
        let _ = self
            .predicted_tail
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |at| {
                Some(at.saturating_sub(by))
            });
    }

    /// Acquire-loaded: a cell published before a tail must be visible once
    /// the tail is.
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

