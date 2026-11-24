//! What HIP makes dear that CUDA does not: a pageable host copy past HIP's
//! own staging waits for its stream, so small host copies go through a
//! pinned ring. (Device buffers and endpoint memory are kept on both:
//! `device::held`.)
#![allow(unsafe_code)]

use crate::device::Pinned;
use crate::error::Result;

/// `ctx::check` over a code HIP answered.
fn check(call: &'static str, code: kernels_cuda::hip::rt::HipError) -> Result<()> {
    crate::device::ctx::check(call, super::rt::cudaError::of(code))
}

/// The largest pageable copy HIP stages itself without waiting for the
/// stream.
const HIP_STAGED_BYTES: usize = 4 << 10;

/// The largest copy [`Buffer::stage`] bounces through the pinned ring.
const RING_SLOT_BYTES: usize = 1 << 20;

/// Slots in a device's ring; a slot is written again only once the copy
/// that last read it has landed, which with this many is long past.
const RING_SLOTS: usize = 32;

/// A slot: its pinned span, the event recorded after the copy that last
/// read it, and the stream that event was recorded on (0: never recorded).
/// The event belongs to that stream: once the stream is destroyed, a query
/// of it reads freed state (on ROCm, a lock no thread holds, so the query
/// waits forever), so a stream's slots are forgotten before it goes
/// ([`forget_stream`]).
struct StageRing {
    slots: Vec<(Pinned, kernels_cuda::hip::rt::Event, usize)>,
    next: usize,
}

/// Every device's ring, by ordinal.
static RINGS: std::sync::Mutex<Option<std::collections::HashMap<core::ffi::c_int, StageRing>>> =
    std::sync::Mutex::new(None);

/// Forgets what `stream` recorded in the rings, before it is destroyed:
/// each slot whose event it recorded waits for that event (the stream still
/// live) and takes a fresh, unrecorded one, so no later stage queries an
/// event of a destroyed stream.
pub(crate) fn forget_stream(stream: usize) {
    use kernels_cuda::hip::rt;
    // A poisoned ring's slots are not to be trusted: their events stay as
    // they are rather than read (a teardown never panics).
    let Ok(mut rings) = RINGS.lock() else {
        return;
    };
    let Some(rings) = rings.as_mut() else {
        return;
    };
    for ring in rings.values_mut() {
        for (_, event, recorded_on) in &mut ring.slots {
            if *recorded_on != stream {
                continue;
            }
            // SAFETY: the slot's own event, recorded on `stream`, which is
            // live until its caller destroys it after this returns.
            unsafe {
                let _ = rt::hipEventSynchronize(*event);
                let _ = rt::hipEventDestroy(*event);
            }
            let mut fresh: rt::Event = core::ptr::null_mut();
            // SAFETY: writes the new event.
            let made = unsafe { rt::hipEventCreateWithFlags(&raw mut fresh, rt::hipEventDisableTiming) };
            // A slot whose event could not be made again is left with none:
            // the next stage that reaches it takes a fresh slot ahead of it.
            *event = if made == 0 { fresh } else { core::ptr::null_mut() };
            *recorded_on = 0;
        }
    }
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

    if bytes.len() <= HIP_STAGED_BYTES || bytes.len() > RING_SLOT_BYTES {
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

    let mut rings = RINGS.lock().expect("rings lock poisoned");
    let rings = rings.get_or_insert_with(HashMap::new);
    let ring = match rings.entry(device) {
        std::collections::hash_map::Entry::Occupied(held) => held.into_mut(),
        std::collections::hash_map::Entry::Vacant(free) => {
            let slots = (0..RING_SLOTS)
                .map(|_| ring_slot())
                .collect::<Result<_>>()?;
            free.insert(StageRing { slots, next: 0 })
        }
    };
    let at = ring.next;
    // A slot whose copy is still queued behind airborne work is not waited
    // out: a fresh one goes in ahead of it, so the ring holds as many slots
    // as copies are in flight and the oldest stays next.
    let (_, event, recorded_on) = &ring.slots[at];
    // SAFETY: queries the slot's own event, recorded on a live stream (a
    // stream's slots are forgotten before it is destroyed).
    if event.is_null() || (*recorded_on != 0 && unsafe { rt::hipEventQuery(*event) } != 0) {
        ring.slots.insert(at, ring_slot()?);
    }
    ring.next = (at + 1) % ring.slots.len();
    let (pinned, event, recorded_on) = &mut ring.slots[at];
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
    *recorded_on = stream as usize;
    Ok(true)
}

/// A pinned stage slot and the event its copy records.
fn ring_slot() -> Result<(Pinned, kernels_cuda::hip::rt::Event, usize)> {
    use kernels_cuda::hip::rt;
    let pinned = Pinned::mapped_uninit(RING_SLOT_BYTES)?;
    let mut event: rt::Event = core::ptr::null_mut();
    // SAFETY: writes the event.
    unsafe {
        check(
            "hipEventCreateWithFlags",
            rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming),
        )?;
    }
    Ok((pinned, event, 0))
}
