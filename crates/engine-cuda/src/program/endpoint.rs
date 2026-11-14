#![allow(unsafe_code)]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eta_ir::container::HostRole;

use crate::device::Pinned;
use crate::error::{Fault, Result};

const HEAD_WORD: usize = 0;
const TAIL_WORD: usize = 1;
pub const WORDS: usize = 4;

pub const MAX_ATTACHMENTS: u32 = 8;

#[derive(Debug)]
pub struct Endpoint {
    role: HostRole,
    words: Pinned,
    mirror: Pinned,
    wire_bytes: u32,
    cap1: u32,
    /// Where the cells of a ring only the device touches lie in the pool
    /// now (0: the ring's cells are each instance's own), and the
    /// `Channel` slots they lie in.
    cells: AtomicU64,
    slots: std::sync::Mutex<Vec<u32>>,
    attachments: AtomicU32,
    predicted_head: AtomicU64,
    predicted_tail: AtomicU64,
    seeded: AtomicU32,
    /// The pool this endpoint's memory returns to.
    pooled: Option<pool::PoolKey>,
}

impl Endpoint {
    /// The bytes of the cells a ring of `capacity` cells of `wire_bytes`
    /// keeps on the device.
    #[must_use]
    pub fn cell_span(wire_bytes: u32, capacity: u32) -> u64 {
        (u64::from(wire_bytes) * (u64::from(capacity) + 1)).max(1)
    }

    /// Opens an endpoint. A ring only the device touches (`HostRole::None`)
    /// has its cells at `cells`, the address of the `Channel` slots `slots`,
    /// zeroed on `stream` ahead of the fires that read them.
    pub fn open(
        role: HostRole,
        wire_bytes: u32,
        capacity: u32,
        cells: Option<(u64, Vec<u32>)>,
        stream: *mut core::ffi::c_void,
    ) -> Result<Endpoint> {
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
        if cap1 > kernels_cuda::runtime::channel::MAX_RING {
            return Err(Fault::program(
                "program::endpoint",
                format!(
                    "a channel of capacity {capacity} wants a ring of {cap1} cells and the \
                     full/empty bytes are cut {} apart per slot, so its ring would \
                     address its neighbour's",
                    kernels_cuda::runtime::channel::MAX_RING
                ),
            ));
        }
        let span = usize::try_from(Endpoint::cell_span(wire_bytes, capacity)).unwrap_or(usize::MAX);
        let (at, slots) = match (role, cells) {
            (HostRole::None, Some((at, slots))) => {
                crate::device::zero_span_on(stream, at, span)?;
                (at, slots)
            }
            (HostRole::None, None) => {
                return Err(Fault::program(
                    "program::endpoint",
                    "a registered ring only the device touches names no `Channel` slots: \
                     its cells lie in the pool",
                ));
            }
            _ => (0, Vec::new()),
        };
        let pooled = pool::key(matches!(role, HostRole::None), span);
        let held = pooled.and_then(|key| pool::take(key, Some(stream)));
        let (words, mirror) = match held {
            Some((words, mirror, _)) => (words, mirror),
            None => (
                Pinned::mapped(WORDS * size_of::<u64>())?,
                Pinned::mapped(span)?,
            ),
        };
        Ok(Endpoint {
            role,
            words,
            mirror,
            wire_bytes,
            cap1,
            cells: AtomicU64::new(at),
            slots: std::sync::Mutex::new(slots),
            predicted_head: AtomicU64::new(0),
            predicted_tail: AtomicU64::new(0),
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
            pooled,
        })
    }

    #[must_use]
    pub fn device_cells(&self) -> Option<u64> {
        Some(self.cells.load(Ordering::Acquire)).filter(|&at| at != 0)
    }

    /// The `Channel` slots the cells lie in; empty where the ring's cells
    /// are each instance's own.
    #[must_use]
    pub fn slots(&self) -> Vec<u32> {
        self.slots.lock().expect("slots lock poisoned").clone()
    }

    /// The cells lie in `slots` now, at `at`, where the runtime copied
    /// them.
    pub fn rebind(&self, at: u64, slots: Vec<u32>) {
        *self.slots.lock().expect("slots lock poisoned") = slots;
        self.rebase(at);
    }

    /// The cells lie at `at` now: their slots' chunk moved.
    pub fn rebase(&self, at: u64) {
        self.cells.store(at, Ordering::Release);
    }

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

    pub fn detach(&self) {
        let _ = self
            .attachments
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(1))
            });
    }

    pub fn claim_seeding(&self) -> bool {
        self.seeded
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    #[must_use]
    pub const fn role(&self) -> HostRole {
        self.role
    }

    #[must_use]
    pub const fn cap1(&self) -> u32 {
        self.cap1
    }

    #[must_use]
    pub const fn wire_bytes(&self) -> u32 {
        self.wire_bytes
    }

    #[must_use]
    pub fn words_device(&self) -> u64 {
        self.words.device()
    }

    #[must_use]
    pub fn words_host(&self) -> u64 {
        self.words.host() as u64
    }

    #[must_use]
    pub fn mirror_device(&self) -> u64 {
        self.mirror.device()
    }

    #[must_use]
    pub fn mirror_host(&self) -> u64 {
        self.mirror.host() as u64
    }

    #[must_use]
    pub fn mirror_bytes(&self) -> usize {
        self.mirror.bytes()
    }

    #[must_use]
    pub const fn engine_owns_head(&self) -> bool {
        !matches!(self.role, HostRole::Reader)
    }

    #[must_use]
    pub const fn engine_owns_tail(&self) -> bool {
        !matches!(self.role, HostRole::Writer)
    }

    #[must_use]
    pub fn head(&self) -> u64 {
        self.word(HEAD_WORD)
    }

    #[must_use]
    pub fn tail(&self) -> u64 {
        self.word(TAIL_WORD)
    }

    pub fn bump_head(&self) {
        self.store(HEAD_WORD, self.word(HEAD_WORD) + 1);
        self.predict_head();
    }

    pub fn bump_tail(&self) {
        self.store(TAIL_WORD, self.word(TAIL_WORD) + 1);
        self.predict_tail();
    }

    #[must_use]
    pub fn read_cell(&self, sequence: u64) -> Vec<u8> {
        let at = (sequence % u64::from(self.cap1)) as usize * self.wire_bytes as usize;
        self.mirror.read(at, self.wire_bytes as usize)
    }

    pub fn write_cell(&self, sequence: u64, wire: &[u8]) -> bool {
        if wire.len() != self.wire_bytes as usize {
            return false;
        }
        let at = (sequence % u64::from(self.cap1)) as usize * self.wire_bytes as usize;
        self.mirror.write(at, wire)
    }

    #[must_use]
    pub fn predicted(&self) -> (u64, u64) {
        (
            self.predicted_head.load(Ordering::Acquire),
            self.predicted_tail.load(Ordering::Acquire),
        )
    }

    pub fn predict_head(&self) {
        self.predicted_head.fetch_add(1, Ordering::AcqRel);
    }

    pub fn predict_tail(&self) {
        self.predicted_tail.fetch_add(1, Ordering::AcqRel);
    }

    pub fn unpredict_head(&self, by: u64) {
        let _ = self
            .predicted_head
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |at| {
                Some(at.saturating_sub(by))
            });
    }

    pub fn unpredict_tail(&self, by: u64) {
        let _ = self
            .predicted_tail
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |at| {
                Some(at.saturating_sub(by))
            });
    }

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

/// A closed channel's memory goes back to the pool for the next of its
/// shape rather than to the driver (`device::held::pool`).
impl Drop for Endpoint {
    fn drop(&mut self) {
        let Some(key) = self.pooled else {
            return;
        };
        let words = core::mem::replace(&mut self.words, Pinned::empty());
        let mirror = core::mem::replace(&mut self.mirror, Pinned::empty());
        pool::give(key, (words, mirror, None));
    }
}

use crate::device::held::pool;
