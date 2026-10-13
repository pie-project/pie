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
    /// The pool key this endpoint's memory returns under.
    pooled: Option<PoolKey>,
    attachments: AtomicU32,
    predicted_head: AtomicU64,
    predicted_tail: AtomicU64,
    seeded: AtomicU32,
}

impl Endpoint {
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
        if cap1 > kernels_rocm::channel::MAX_RING {
            return Err(Fault::program(
                "program::endpoint",
                format!(
                    "a channel of capacity {capacity} wants a ring of {cap1} cells and the \
                     full/empty bytes are cut {} apart per slot, so its ring would \
                     address its neighbour's",
                    kernels_rocm::channel::MAX_RING
                ),
            ));
        }
        let cells = (wire_bytes as usize).saturating_mul(cap1 as usize);
        let key = pool::key(matches!(role, HostRole::None), cells.max(1));
        let (words, mirror, device_cells) = match key.and_then(pool::take) {
            Some(held) => held,
            None => (
                Pinned::mapped(WORDS * size_of::<u64>())?,
                Pinned::mapped(cells.max(1))?,
                match role {
                    HostRole::None => Some(Buffer::zeroed(cells.max(1))?),
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
            pooled: key,
            predicted_head: AtomicU64::new(0),
            predicted_tail: AtomicU64::new(0),
            attachments: AtomicU32::new(0),
            seeded: AtomicU32::new(0),
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

/// Endpoint memory kept for the next channel of its shape. Freeing pinned
/// or device memory on ROCm waits for the whole device, so a channel closed
/// mid-serve stalled every frame in flight, and pinning fresh pages costs
/// about a millisecond a channel; a burst of requests opens and closes
/// thousands. A returned endpoint is reused only once an event recorded on
/// the device's compute stream at its return has landed, so no frame that
/// could still read it is airborne; its memory is zeroed again on the way
/// out, as fresh memory would be.
pub mod pool {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicPtr, Ordering};

    use crate::device::{Buffer, Pinned};

    pub type Held = (Pinned, Pinned, Option<Buffer>);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PoolKey {
        device: i32,
        device_only: bool,
        cells: usize,
        /// An instance's commit slot rather than a channel's endpoint.
        commit: bool,
    }

    /// Endpoints of one shape kept at most.
    const PER_KEY: usize = 4096;

    /// Cells past this many bytes are freed rather than kept.
    const LARGEST: usize = 64 << 20;

    const ORDINALS: usize = 64;

    static STREAMS: [AtomicPtr<core::ffi::c_void>; ORDINALS] =
        [const { AtomicPtr::new(core::ptr::null_mut()) }; ORDINALS];

    /// The stream a returned endpoint's event is recorded on: the compute
    /// stream every fire of `ordinal` launches on.
    pub fn adopt_stream(ordinal: i32, stream: *mut core::ffi::c_void) {
        if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
            slot.store(stream, Ordering::Release);
        }
    }

    struct Entry {
        held: Held,
        event: kernels_rocm::hip::rt::Event,
    }

    // SAFETY: entries are touched only under the pool's mutex; the event and
    // the memory are process-wide HIP objects.
    unsafe impl Send for Entry {}

    static POOL: Mutex<Option<HashMap<PoolKey, Vec<Entry>>>> = Mutex::new(None);

    fn current_device() -> Option<i32> {
        let mut device = 0i32;
        // SAFETY: writes one int.
        let code = unsafe { kernels_rocm::hip::rt::hipGetDevice(&raw mut device) };
        (code == 0).then_some(device)
    }

    fn stream_of(device: i32) -> Option<*mut core::ffi::c_void> {
        let stream = STREAMS
            .get(usize::try_from(device).ok()?)?
            .load(Ordering::Acquire);
        (!stream.is_null()).then_some(stream)
    }

    /// The key an endpoint of this shape pools under, if any: none on a
    /// device whose compute stream was never adopted, or for huge cells.
    pub fn key(device_only: bool, cells: usize) -> Option<PoolKey> {
        let device = current_device()?;
        stream_of(device)?;
        (cells <= LARGEST).then_some(PoolKey {
            device,
            device_only,
            cells,
            commit: false,
        })
    }

    /// The key an instance's `bytes`-wide commit slot pools under.
    pub fn commit_key(bytes: usize) -> Option<PoolKey> {
        let device = current_device()?;
        stream_of(device)?;
        Some(PoolKey {
            device,
            device_only: false,
            cells: bytes,
            commit: true,
        })
    }

    /// A kept endpoint of `key`'s shape whose last reader has landed, its
    /// memory zeroed (the device cells on the compute stream, ahead of any
    /// fire that reads them).
    pub fn take(key: PoolKey) -> Option<Held> {
        use kernels_rocm::hip::rt;
        let stream = stream_of(key.device)?;
        let mut pool = POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let list = pool.as_mut()?.get_mut(&key)?;
        // SAFETY: each event is the entry's own, recorded at its return.
        let at = list
            .iter()
            .position(|entry| unsafe { rt::hipEventQuery(entry.event) } == 0)?;
        let entry = list.swap_remove(at);
        drop(pool);
        // SAFETY: the event has landed and is destroyed once.
        unsafe { rt::hipEventDestroy(entry.event) };
        let (words, mirror, cells) = entry.held;
        words.zero();
        mirror.zero();
        if let Some(cells) = &cells {
            // SAFETY: the buffer is the entry's, no airborne fire reads it,
            // and the memset is ordered before every later fire.
            let code = unsafe {
                rt::hipMemsetAsync(
                    cells.ptr() as *mut core::ffi::c_void,
                    0,
                    cells.bytes(),
                    stream.cast(),
                )
            };
            if code != 0 {
                return None;
            }
        }
        Some((words, mirror, cells))
    }

    /// Keeps `held` for `key`'s shape behind an event on the compute
    /// stream; frees it (by dropping) when the pool is full or the event
    /// will not record.
    pub fn give(key: PoolKey, held: Held) {
        use kernels_rocm::hip::rt;
        let Some(stream) = stream_of(key.device) else {
            return;
        };
        let mut event: rt::Event = core::ptr::null_mut();
        // SAFETY: writes and records a fresh event on a live stream.
        let recorded = unsafe {
            rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming) == 0
                && rt::hipEventRecord(event, stream.cast()) == 0
        };
        if !recorded {
            return;
        }
        let mut pool = POOL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let list = pool
            .get_or_insert_with(HashMap::new)
            .entry(key)
            .or_default();
        if list.len() >= PER_KEY {
            drop(pool);
            // SAFETY: the event is this call's own.
            unsafe { rt::hipEventDestroy(event) };
            return;
        }
        list.push(Entry { held, event });
    }
}

use pool::PoolKey;
