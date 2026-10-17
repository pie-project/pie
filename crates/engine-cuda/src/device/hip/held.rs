//! What HIP makes dear that CUDA does not: `hipFree` and freeing pinned
//! memory wait for the whole device, and a pageable host copy past HIP's
//! own staging waits for its stream. Device buffers and endpoint memory are
//! kept behind an event on the compute stream and handed out again once it
//! lands; small host copies go through a pinned ring.

use crate::device::{Buffer, Pinned};
use crate::error::Result;

/// `ctx::check` over a code HIP answered.
fn check(call: &'static str, code: kernels_cuda::hip::rt::HipError) -> Result<()> {
    crate::device::ctx::check(call, super::rt::cudaError::of(code))
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

    use super::{Buffer, Pinned};

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

    /// The live contexts' compute streams, by ordinal, in adoption order: a
    /// stream is recorded on only while this is read-held, and retired (and
    /// destroyed) only while it is write-held, so no event is recorded on a
    /// destroyed stream, whichever context of the process goes first.
    static LIVE: std::sync::RwLock<Vec<(i32, usize)>> = std::sync::RwLock::new(Vec::new());

    /// The stream a returned endpoint's event is recorded on: the compute
    /// stream every fire of `ordinal` launches on (the latest live
    /// context's).
    pub fn adopt_stream(ordinal: i32, stream: *mut core::ffi::c_void) {
        let mut live = LIVE
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.push((ordinal, stream as usize));
        if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
            slot.store(stream, Ordering::Release);
        }
    }

    /// Takes `stream` out of `ordinal`'s adoption (another live context's
    /// stream, if any, takes its place) and runs `destroy` before any
    /// recorder can see the stream again.
    pub fn retire_stream(ordinal: i32, stream: *mut core::ffi::c_void, destroy: impl FnOnce()) {
        let mut live = LIVE
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(at) = live
            .iter()
            .position(|&entry| entry == (ordinal, stream as usize))
        {
            live.remove(at);
        }
        let next = live
            .iter()
            .rev()
            .find(|entry| entry.0 == ordinal)
            .map_or(core::ptr::null_mut(), |entry| {
                entry.1 as *mut core::ffi::c_void
            });
        if let Some(slot) = usize::try_from(ordinal).ok().and_then(|at| STREAMS.get(at)) {
            slot.store(next, Ordering::Release);
        }
        destroy();
    }

    struct Entry {
        held: Held,
        event: kernels_cuda::hip::rt::Event,
    }

    // SAFETY: entries are touched only under the pool's mutex; the event and
    // the memory are process-wide HIP objects.
    unsafe impl Send for Entry {}

    static POOL: Mutex<Option<HashMap<PoolKey, Vec<Entry>>>> = Mutex::new(None);

    pub(crate) fn current_device() -> Option<i32> {
        let mut device = 0i32;
        // SAFETY: writes one int.
        let code = unsafe { kernels_cuda::hip::rt::hipGetDevice(&raw mut device) };
        (code == 0).then_some(device)
    }

    fn stream_of(device: i32) -> Option<*mut core::ffi::c_void> {
        let stream = STREAMS
            .get(usize::try_from(device).ok()?)?
            .load(Ordering::Acquire);
        (!stream.is_null()).then_some(stream)
    }

    /// Whether `device` has an adopted compute stream.
    pub(crate) fn adopted(device: i32) -> bool {
        stream_of(device).is_some()
    }

    /// `use_stream` on `device`'s adopted compute stream, kept live for the
    /// call; none if it has none.
    pub(crate) fn with_stream<R>(
        device: i32,
        use_stream: impl FnOnce(*mut core::ffi::c_void) -> R,
    ) -> Option<R> {
        let _live = LIVE
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stream_of(device).map(use_stream)
    }

    /// The key an endpoint of this shape pools under, if any: none on a
    /// device whose compute stream was never adopted, or for huge cells.
    pub fn key(device_only: bool, cells: usize) -> Option<PoolKey> {
        let device = current_device()?;
        if !adopted(device) {
            return None;
        }
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
        if !adopted(device) {
            return None;
        }
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
        use kernels_cuda::hip::rt;
        if !adopted(key.device) {
            return None;
        }
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
            let code = with_stream(key.device, |stream| unsafe {
                rt::hipMemsetAsync(
                    cells.ptr() as *mut core::ffi::c_void,
                    0,
                    cells.bytes(),
                    stream.cast(),
                )
            });
            if code != Some(0) {
                return None;
            }
        }
        Some((words, mirror, cells))
    }

    /// Keeps `held` for `key`'s shape behind an event on the compute
    /// stream; frees it (by dropping) when the pool is full or the event
    /// will not record.
    pub fn give(key: PoolKey, held: Held) {
        use kernels_cuda::hip::rt;
        let mut event: rt::Event = core::ptr::null_mut();
        // SAFETY: writes and records a fresh event on a live stream.
        let recorded = with_stream(key.device, |stream| unsafe {
            rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming) == 0
                && rt::hipEventRecord(event, stream.cast()) == 0
        });
        if recorded != Some(true) {
            if !event.is_null() {
                // SAFETY: the event is this call's own.
                unsafe { rt::hipEventDestroy(event) };
            }
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

/// The largest copy [`Buffer::stage`] bounces through the pinned ring.
const RING_SLOT_BYTES: usize = 1 << 20;

/// Slots in a device's ring; a slot is written again only once the copy
/// that last read it has landed, which with this many is long past.
const RING_SLOTS: usize = 32;

struct StageRing {
    slots: Vec<(Pinned, kernels_cuda::hip::rt::Event, bool)>,
    next: usize,
}

// SAFETY: the ring is only touched under its mutex; the events and pinned
// spans it owns are process-wide HIP objects.
unsafe impl Send for StageRing {}

/// A host-to-device copy out of pageable memory past HIP's own small-copy
/// staging waits for the stream to drain before it returns, so a fire that
/// stages a large table would wait out the frame still in flight. Copied
/// into a pinned slot first, the copy is asynchronous for real. Returns
/// `false` (and copies nothing) for a copy the ring does not take: one too
/// large for a slot, or one into a stream being captured, whose recorded
/// copy must not read a slot the ring will reuse.
pub(crate) fn staged_through_ring(
    dst: u64,
    bytes: &[u8],
    stream: *mut core::ffi::c_void,
) -> Result<bool> {
    use kernels_cuda::hip::rt;
    use std::collections::HashMap;
    use std::sync::Mutex;

    if bytes.len() > RING_SLOT_BYTES {
        return Ok(false);
    }
    let mut capturing: core::ffi::c_int = 0;
    // SAFETY: writes one int for a live stream.
    unsafe {
        check(
            "hipStreamIsCapturing",
            rt::hipStreamIsCapturing(stream.cast(), &raw mut capturing),
        )?
    };
    if capturing != 0 {
        return Ok(false);
    }
    let mut device: core::ffi::c_int = 0;
    // SAFETY: writes one int.
    unsafe { check("hipGetDevice", rt::hipGetDevice(&raw mut device))? };

    static RINGS: Mutex<Option<HashMap<core::ffi::c_int, StageRing>>> = Mutex::new(None);
    let mut rings = RINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rings = rings.get_or_insert_with(HashMap::new);
    let ring = match rings.entry(device) {
        std::collections::hash_map::Entry::Occupied(held) => held.into_mut(),
        std::collections::hash_map::Entry::Vacant(free) => {
            let mut slots = Vec::with_capacity(RING_SLOTS);
            for _ in 0..RING_SLOTS {
                let pinned = Pinned::mapped_uninit(RING_SLOT_BYTES)?;
                let mut event: rt::Event = core::ptr::null_mut();
                // SAFETY: writes the event.
                unsafe {
                    check(
                        "hipEventCreateWithFlags",
                        rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming),
                    )?;
                }
                slots.push((pinned, event, false));
            }
            free.insert(StageRing { slots, next: 0 })
        }
    };
    let at = ring.next;
    ring.next = (ring.next + 1) % RING_SLOTS;
    let (pinned, event, used) = &mut ring.slots[at];
    // SAFETY: the event is this slot's; waiting on it is what frees the slot.
    if *used {
        unsafe { check("hipEventSynchronize", rt::hipEventSynchronize(*event))? };
    }
    // SAFETY: the slot holds RING_SLOT_BYTES ≥ bytes.len(), and the copy that
    // last read it has landed; the new copy reads it until the event records.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), pinned.host(), bytes.len());
        check(
            "hipMemcpyAsync",
            rt::hipMemcpyAsync(
                dst as *mut core::ffi::c_void,
                pinned.host().cast_const().cast(),
                bytes.len(),
                rt::hipMemcpyHostToDevice,
                stream.cast(),
            ),
        )?;
        check("hipEventRecord", rt::hipEventRecord(*event, stream.cast()))?;
    }
    *used = true;
    Ok(true)
}

/// Device memory a dropped [`Buffer`] leaves, kept for the next of its size.
/// `hipFree` waits for the whole device, so a buffer dropped while frames
/// are airborne (a lane table outgrown, a channel closed) would stall the
/// engine thread for the frame in flight; kept behind an event on the
/// compute stream instead, it is handed out again once that event lands.
pub(crate) mod recycle {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use kernels_cuda::hip::rt;

    use super::check;
    use crate::error::Result;

    /// The bytes kept across every size at most; an allocation the device
    /// refuses frees them all first, so this bounds only what an idle load
    /// sits on.
    const BUDGET: usize = 8 << 30;

    struct Kept {
        ptr: u64,
        event: rt::Event,
    }

    // SAFETY: entries are touched only under the mutex; the memory and the
    // event are process-wide HIP objects.
    unsafe impl Send for Kept {}

    /// Kept buffers by (device, bytes).
    type Shelf = Option<HashMap<(i32, usize), Vec<Kept>>>;

    static KEPT: Mutex<Shelf> = Mutex::new(None);
    static HELD: AtomicUsize = AtomicUsize::new(0);

    fn kept() -> std::sync::MutexGuard<'static, Shelf> {
        KEPT.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `bytes` of device memory: a kept buffer whose last reader has
    /// landed, else a fresh one; on a refusal everything kept is freed and
    /// the allocation is tried once more.
    pub(crate) fn alloc(bytes: usize) -> Result<u64> {
        if let Some(device) = super::pool::current_device() {
            let mut kept = kept();
            if let Some(list) = kept.as_mut().and_then(|map| map.get_mut(&(device, bytes)))
                // SAFETY: each event is its entry's own, recorded at its release.
                && let Some(at) = list
                    .iter()
                    .position(|entry| unsafe { rt::hipEventQuery(entry.event) } == 0)
            {
                let entry = list.swap_remove(at);
                drop(kept);
                HELD.fetch_sub(bytes, Ordering::Relaxed);
                // SAFETY: the event has landed and is destroyed once.
                unsafe { rt::hipEventDestroy(entry.event) };
                return Ok(entry.ptr);
            }
        }
        let fresh = |bytes: usize| {
            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local the call fills.
            let code = unsafe { rt::hipMalloc(&raw mut base, bytes) };
            check("hipMalloc", code).map(|()| base as u64)
        };
        match fresh(bytes) {
            Ok(ptr) => Ok(ptr),
            Err(_) if drain() => {
                fresh(bytes).map_err(|fault| crate::device::alloc::out_of_room(fault, bytes))
            }
            Err(fault) => Err(crate::device::alloc::out_of_room(fault, bytes)),
        }
    }

    /// Keeps `ptr` for its size behind an event on its device's compute
    /// stream, or frees it when the budget is spent or no stream was
    /// adopted.
    pub(crate) fn release(ptr: u64, bytes: usize) {
        if HELD.load(Ordering::Relaxed) + bytes <= BUDGET
            && let Some(device) = super::pool::current_device()
        {
            let mut event: rt::Event = core::ptr::null_mut();
            // SAFETY: creates and records a fresh event on a stream kept
            // live for the call.
            let recorded = super::pool::with_stream(device, |stream| unsafe {
                rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming) == 0
                    && rt::hipEventRecord(event, stream.cast()) == 0
            }) == Some(true);
            if recorded {
                HELD.fetch_add(bytes, Ordering::Relaxed);
                kept()
                    .get_or_insert_with(HashMap::new)
                    .entry((device, bytes))
                    .or_default()
                    .push(Kept { ptr, event });
                return;
            }
            if !event.is_null() {
                // SAFETY: the event is this call's own.
                unsafe { rt::hipEventDestroy(event) };
            }
        }
        // SAFETY: the pointer came from `alloc` and is freed exactly once.
        let _ = unsafe { rt::hipFree(ptr as *mut core::ffi::c_void) };
    }

    /// Frees every kept buffer; whether there was any.
    fn drain() -> bool {
        let Some(map) = kept().take() else {
            return false;
        };
        let mut any = false;
        for entry in map.into_values().flatten() {
            any = true;
            // SAFETY: each pointer and event is its entry's own; `hipFree`
            // waits for the device, so no reader is left.
            unsafe {
                rt::hipEventDestroy(entry.event);
                let _ = rt::hipFree(entry.ptr as *mut core::ffi::c_void);
            }
        }
        HELD.store(0, Ordering::Relaxed);
        any
    }
}
