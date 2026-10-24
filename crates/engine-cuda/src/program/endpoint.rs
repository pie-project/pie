#![allow(unsafe_code)]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use eta_ir::container::HostRole;

use crate::device::{Buffer, Pinned};
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
    device_cells: Option<Buffer>,
    attachments: AtomicU32,
    predicted_head: AtomicU64,
    predicted_tail: AtomicU64,
    seeded: AtomicU32,
    /// The pool this endpoint's memory returns to.
    pooled: Option<pool::PoolKey>,
}

impl Endpoint {
    /// Opens an endpoint, its device cells zeroed on `stream` ahead of the fires
    /// that read them.
    pub fn open(
        role: HostRole,
        wire_bytes: u32,
        capacity: u32,
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
        let cells = (wire_bytes as usize).saturating_mul(cap1 as usize);
        let pooled = pool::key(matches!(role, HostRole::None), cells.max(1));
        let held = pooled.and_then(pool::take);
        let (words, mirror, device_cells) = match held {
            Some(held) => held,
            None => (
                Pinned::mapped(WORDS * size_of::<u64>())?,
                Pinned::mapped(cells.max(1))?,
                match role {
                    HostRole::None => Some(Buffer::zeroed_on(stream, cells.max(1))?),
                    _ => None,
                },
            ),
        };
        Ok(Endpoint {
            role,
            words,
            mirror,
            wire_bytes,
            cap1,
            device_cells,
            predicted_head: AtomicU64::new(0),
            predicted_tail: AtomicU64::new(0),
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
            pooled,
        })
    }

    #[must_use]
    pub fn device_cells(&self) -> Option<u64> {
        self.device_cells.as_ref().map(Buffer::ptr)
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
        let cells = self.device_cells.take();
        pool::give(key, (words, mirror, cells));
    }
}

use crate::device::held::pool;
