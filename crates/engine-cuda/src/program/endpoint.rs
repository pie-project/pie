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
//! HostRole::None     there is no endpoint at all — the cells never leave the
//!                    device and no counter is pinned
//! ```
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

use std::sync::atomic::{AtomicU64, Ordering};

use engine::tensor_ir::container::HostRole;

use crate::device::Pinned;
use crate::error::{Fault, Result};

/// Where each counter lives in [`Endpoint::words`] — the runtime's
/// `HostRing` word order, and `channel::Ticket::words`'s.
const HEAD_WORD: usize = 0;
const TAIL_WORD: usize = 1;
/// How many words one endpoint carries: `[head, tail, poison, closed]`.
pub const WORDS: usize = 4;

/// One host-visible channel endpoint's pinned mirror and pinned counters.
#[derive(Debug)]
pub struct Endpoint {
    /// Which end the host holds. Never [`HostRole::None`] — a channel with no
    /// host end has no endpoint at all.
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
        Ok(Endpoint {
            role,
            words: Pinned::mapped(WORDS * size_of::<u64>())?,
            mirror: Pinned::mapped(cells.max(1))?,
            wire_bytes,
            cap1,
        })
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
        };
        let writer = of(HostRole::Writer);
        assert!(writer.engine_owns_head(), "the pass consumes what the guest published");
        assert!(!writer.engine_owns_tail(), "and the guest alone advances the tail");
        let reader = of(HostRole::Reader);
        assert!(!reader.engine_owns_head(), "the guest alone consumes");
        assert!(reader.engine_owns_tail(), "and the pass alone publishes");
    }
}
