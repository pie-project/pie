//! The device-backed channel ring: the GPU's copy of a channel's cells and
//! words.
//!
//! The interpreter's [`ChannelState`](crate::channel::ChannelState) keeps a
//! ring in host memory — a `Vec` of cells and four atomics — because a CPU
//! pass is its only reader. The launch path's rings are different in exactly
//! one way: the *kernels* read the cells and advance the words, so both must
//! live in memory the GPU can address. On unified memory that is a
//! shared-storage buffer, host-visible and device-visible at once, and this
//! type is that pair of buffers plus the layout facts a fire derives handles
//! from.
//!
//! ## Why this closes the standalone-buffer hole
//!
//! The C++ allocates ring storage through `create_standalone_buffer`, which
//! hands back a `SlotHandle` **with no owner**. `release_standalone_buffer`
//! exists only because of that, and the shell's own comment records the
//! consequence of forgetting it: `resize_pool` leaked every previous K/V
//! buffer, retained and resident, forever. A [`Ring`] owns its two buffers —
//! creation adds them to the residency set, `Drop` removes them, and there
//! is no release call to forget because there is nothing to call it on. Both
//! buffers are [`Allocation`]s, so that `Drop` is the type's rather than
//! hand-written here -- this module used to own the primitive that could not
//! own, and now owns neither it nor the symmetry it was missing.
//! `.wiki/driver/progress-metal.md`'s four `missing` standalone-buffer entries are this hole; the
//! channel-ring use of it lands here, as an owning type rather than as the
//! primitive that could not own.
//!
//! ## The words are read atomically, and the C++'s were not
//!
//! A commit kernel advances `head`/`tail` from the GPU — as plain `device
//! ulong` stores; what orders them against the host is the command buffer's
//! completion fence, which the encoder waits on before anything reads back.
//! The C++ then reads the words through plain `uint64_t` loads on the mapped
//! pointer, so nothing in the *host's* program text says the bytes are
//! shared. Here the host side of every word is an [`AtomicU64`] view of the
//! same bytes: the fence still provides the ordering, and the atomic makes
//! the host's access defined rather than a race that UMA happens to forgive.
//!
//! ## One authority per ring
//!
//! A channel is either interpreted (a `ChannelState`) or device-run (a
//! `Ring`) — never both at once. The two share the readiness check through
//! [`crate::channel::Words`], which is a snapshot either can
//! produce, not a common base class.

use core::sync::atomic::{AtomicU64, Ordering};

use tensor_ir::DType;

use crate::Result;
use crate::channel::{Words, wire_cell_bytes};
use crate::device::allocation::Allocation;
use crate::device::context::Context;
use crate::device::handle::Handle;
use crate::layout::region::Region;

/// Index of the head word in the words buffer.
const HEAD: usize = 0;
/// Index of the tail word.
const TAIL: usize = 1;
/// Index of the poison word.
const POISON: usize = 2;
/// Index of the closed word.
const CLOSED: usize = 3;
/// The words buffer: four `u64`s.
const WORDS_BYTES: u64 = 32;

/// A channel ring in shared GPU memory: `capacity + 1` wire cells and the
/// four ring words, each in its own buffer, resident for the ring's life.
pub struct Ring {
    /// The element type of a cell, after `Act` folds to `f32`.
    dtype: DType,
    /// Lanes per cell.
    numel: usize,
    /// Live cells the ring holds.
    capacity: usize,
    /// Bytes of one wire cell.
    cell_bytes: usize,
    /// The physical slot count, one more than [`capacity`](Self::capacity):
    /// the pending slot a put writes before commit advances the ring.
    cap1: usize,
    /// The cell array.
    cells: Allocation,
    /// The four ring words.
    words: Allocation,
}

impl Ring {
    /// Allocate a zeroed, resident ring for `capacity` live cells of `numel`
    /// lanes.
    ///
    /// Mirrors [`ChannelState::host`](crate::channel::ChannelState::host)
    /// exactly — `dtype` folded through the wire encoding, lane count and
    /// capacity floored at one — because a ring that the interpreter would
    /// size one way and the device another is two rings wearing one channel
    /// id.
    ///
    /// # Errors
    ///
    /// [`Error::Create`](crate::Error::Create) when the device declines an
    /// allocation.
    pub fn new(context: &Context, dtype: DType, numel: usize, capacity: usize) -> Result<Self> {
        let numel = numel.max(1);
        let capacity = capacity.max(1);
        let cell_bytes = wire_cell_bytes(dtype, numel);
        let cap1 = capacity + 1;

        let cells = Allocation::new(context, (cell_bytes * cap1) as u64, "channel ring cells")?;
        let words = Allocation::new(context, WORDS_BYTES, "channel ring words")?;
        // A fresh Metal buffer's contents are not promised to be zero, and a
        // ring whose words start as garbage is a ring that is already
        // poisoned, closed, and mid-sequence.
        // SAFETY: the buffers were just created; no GPU work names them.
        unsafe {
            cells.zero(0, cells.len())?;
            words.zero(0, words.len())?;
        }

        Ok(Self {
            dtype,
            numel,
            capacity,
            cell_bytes,
            cap1,
            cells,
            words,
        })
    }

    /// The element type of a cell.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Lanes per cell.
    #[must_use]
    pub const fn numel(&self) -> usize {
        self.numel
    }

    /// Live cells the ring holds.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes of one wire cell.
    #[must_use]
    pub const fn cell_bytes(&self) -> usize {
        self.cell_bytes
    }

    /// The whole cell array, for binding.
    #[must_use]
    pub const fn cells(&self) -> &Handle {
        self.cells.handle()
    }

    /// The four ring words, for binding.
    #[must_use]
    pub const fn words(&self) -> &Handle {
        self.words.handle()
    }

    /// The word at `index`, as the host's atomic view of the shared bytes.
    fn word(&self, index: usize) -> &AtomicU64 {
        // SAFETY: the words buffer is 32 zero-initialised, 8-aligned bytes
        // owned by this ring, so indexes 0..4 are in bounds and the pointer
        // is valid for the atomic view for as long as `&self` lives. The GPU
        // writes these words as plain device stores ordered by the command
        // buffer's completion fence; the atomic view is what makes the host
        // side of that access defined.
        unsafe { AtomicU64::from_ptr(self.words.contents().cast::<u64>().add(index).as_ptr()) }
    }

    /// The consume sequence.
    #[must_use]
    pub fn head(&self) -> u64 {
        self.word(HEAD).load(Ordering::Acquire)
    }

    /// The publish sequence.
    #[must_use]
    pub fn tail(&self) -> u64 {
        self.word(TAIL).load(Ordering::Acquire)
    }

    /// Non-zero when a producer faulted the ring.
    #[must_use]
    pub fn poison(&self) -> u64 {
        self.word(POISON).load(Ordering::Acquire)
    }

    /// Non-zero when no further cell will ever arrive.
    #[must_use]
    pub fn closed(&self) -> u64 {
        self.word(CLOSED).load(Ordering::Acquire)
    }

    /// The four words at once, for [`check_words`](crate::channel::check_words).
    #[must_use]
    pub fn snapshot(&self) -> Words {
        Words {
            head: self.head(),
            tail: self.tail(),
            poison: self.poison(),
            closed: self.closed(),
        }
    }

    /// The cell a take at sequence `head` reads: slot `head % (capacity + 1)`.
    ///
    /// The C++ writes `(head % state.cap1) * state.cell_bytes` inline at each
    /// site; the modulo lives here once.
    ///
    /// # Errors
    ///
    /// Never in practice — the slot arithmetic is bounded by construction —
    /// but the slice is checked rather than trusted.
    pub fn committed_cell(&self, head: u64) -> Result<Handle> {
        self.cell(head)
    }

    /// The cell a put at sequence `tail` writes, before commit publishes it.
    ///
    /// # Errors
    ///
    /// As [`committed_cell`](Self::committed_cell).
    pub fn pending_cell(&self, tail: u64) -> Result<Handle> {
        self.cell(tail)
    }

    fn cell(&self, sequence: u64) -> Result<Handle> {
        let slot = sequence % self.cap1 as u64;
        self.cells
            .slice(slot * self.cell_bytes as u64, self.cell_bytes as u64)
    }
}

impl std::fmt::Debug for Ring {
    /// The shape and where the ring stands, not the cells.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ring")
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("capacity", &self.capacity)
            .field("head", &self.head())
            .field("tail", &self.tail())
            .finish_non_exhaustive()
    }
}
