//! Endpoint memory kept for the next channel of its shape, on CUDA.
//!
//! A channel's endpoint pins two host spans and, device-only, a device one:
//! a `cudaHostAlloc` and a `cudaMalloc` cost a millisecond or two each, and a
//! burst of requests opens a few channels a request, so 256 arrivals spent
//! ~50 ms of host time with the card idle before their prefill, and again
//! before their first decode. A returned endpoint is reused only once an
//! event recorded on each of the device's compute streams at its return has
//! landed, so no frame that could still read it is airborne; its memory is zeroed
//! again on the way out, as fresh memory would be. The same shape as ROCm's
//! pool (`device::hip::held::pool`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::device::rt;
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

/// The live contexts' compute streams, by ordinal, in adoption order: a
/// stream is recorded on only while this is read-held, and retired (and
/// destroyed) only while it is write-held.
static LIVE: std::sync::RwLock<Vec<(i32, usize)>> = std::sync::RwLock::new(Vec::new());

/// The stream a returned endpoint's event is recorded on: the compute
/// stream every fire of `ordinal` launches on (the latest live context's).
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
/// stream, if any, takes its place) and runs `destroy` before any recorder
/// can see the stream again.
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
    fence: Fence,
}

// SAFETY: entries are touched only under the pool's mutex; the events and
// the memory are process-wide CUDA objects.
unsafe impl Send for Entry {}

/// An event on each live compute stream of a device at a span's return: no
/// context's fire that could still read it is airborne once all landed.
pub(crate) struct Fence(Vec<rt::cudaEvent_t>);

// SAFETY: the events are process-wide CUDA objects, queried and destroyed
// by whoever holds the fence.
unsafe impl Send for Fence {}

impl Fence {
    /// The fence of `device`'s live streams now; none with no stream live
    /// or an event that will not record.
    pub(crate) fn raise(device: i32) -> Option<Fence> {
        let live = LIVE
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut events = Vec::new();
        for &(ordinal, stream) in live.iter().filter(|entry| entry.0 == device) {
            let _ = ordinal;
            let mut event: rt::cudaEvent_t = core::ptr::null_mut();
            // SAFETY: creates and records a fresh event on a stream the read
            // guard keeps live.
            let recorded = unsafe {
                rt::cudaEventCreateWithFlags(&raw mut event, 2) == rt::cudaError::cudaSuccess
                    && rt::cudaEventRecord(event, stream as rt::cudaStream_t)
                        == rt::cudaError::cudaSuccess
            };
            if !event.is_null() {
                events.push(event);
            }
            if !recorded {
                drop(Fence(events));
                return None;
            }
        }
        (!events.is_empty()).then_some(Fence(events))
    }

    /// An empty fence: a span never handed out.
    pub(crate) const fn open() -> Fence {
        Fence(Vec::new())
    }

    pub(crate) fn landed(&self) -> bool {
        // SAFETY: each event is the fence's own.
        self.0
            .iter()
            .all(|&event| unsafe { rt::cudaEventQuery(event) } == rt::cudaError::cudaSuccess)
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        for &event in &self.0 {
            // SAFETY: the event is the fence's own and destroyed once.
            unsafe { rt::cudaEventDestroy(event) };
        }
    }
}

/// Zeroes `bytes` at `ptr` ahead of every later fire on `device`: on its
/// compute stream when it has one live, else on the legacy stream, which
/// orders against every context's.
pub(crate) fn zero_before_fires(device: i32, ptr: u64, bytes: usize) -> bool {
    let live = LIVE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut streams = live.iter().filter(|entry| entry.0 == device);
    let code = match (streams.next(), streams.next()) {
        // SAFETY: the span is the caller's and nothing airborne reads it; the
        // stream is kept live by the guard.
        (Some(&(_, stream)), None) => unsafe {
            rt::cudaMemsetAsync(ptr as *mut core::ffi::c_void, 0, bytes, stream as rt::cudaStream_t)
        },
        // SAFETY: as above.
        _ => unsafe { rt::cudaMemset(ptr as *mut core::ffi::c_void, 0, bytes) },
    };
    code == rt::cudaError::cudaSuccess
}

static POOL: Mutex<Option<HashMap<PoolKey, Vec<Entry>>>> = Mutex::new(None);

fn current_device() -> Option<i32> {
    let mut device = 0i32;
    // SAFETY: writes one int.
    let code = unsafe { rt::cudaGetDevice(&raw mut device) };
    (code == rt::cudaError::cudaSuccess).then_some(device)
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

/// The key an endpoint of this shape pools under, if any: none on a device
/// whose compute stream was never adopted, or for huge cells.
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
/// memory zeroed (the device cells on the compute stream, ahead of any fire
/// that reads them).
pub fn take(key: PoolKey) -> Option<Held> {
    stream_of(key.device)?;
    let mut pool = POOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let list = pool.as_mut()?.get_mut(&key)?;
    let at = list.iter().position(|entry| entry.fence.landed())?;
    let entry = list.swap_remove(at);
    drop(pool);
    let (words, mirror, cells) = entry.held;
    words.zero();
    mirror.zero();
    if let Some(cells) = &cells
        && !zero_before_fires(key.device, cells.ptr(), cells.bytes())
    {
        return None;
    }
    Some((words, mirror, cells))
}

/// Keeps `held` for `key`'s shape behind a fence on the device's compute
/// streams; frees it (by dropping) when the pool is full or the fence will
/// not record.
pub fn give(key: PoolKey, held: Held) {
    let Some(fence) = Fence::raise(key.device) else {
        return;
    };
    let mut pool = POOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let list = pool
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_default();
    if list.len() >= PER_KEY {
        return;
    }
    list.push(Entry { held, fence });
}
