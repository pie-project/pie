#![allow(unsafe_code)]
use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::hip::sys;
use crate::jit::Fault;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

#[derive(Default)]
struct Arena {
    slabs: HashMap<(&'static str, u32, usize), Slab>,
    retired: Vec<*mut c_void>,
}

// SAFETY: the only pointers here are device allocations this map owns
// outright. `hipMalloc`/`hipFree` are thread-safe and the map is behind a
// mutex; nothing else holds a slab except a launch that has already been
// enqueued.
unsafe impl Send for Arena {}

fn arenas() -> &'static Mutex<HashMap<u32, Arena>> {
    static ARENAS: OnceLock<Mutex<HashMap<u32, Arena>>> = OnceLock::new();
    ARENAS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn locked() -> std::sync::MutexGuard<'static, HashMap<u32, Arena>> {
    arenas().lock().expect("lock poisoned")
}

pub(crate) fn attach(arena: u32, stream: *mut c_void) {
    let _ = (arena, stream);
}

pub(crate) fn release(arena: u32) {
    let Ok(api) = sys::api() else { return };
    let mut arenas = locked();
    let Some(held) = arenas.remove(&arena) else {
        return;
    };
    for (_, slab) in held.slabs {
        if !slab.ptr.is_null() {
            // SAFETY: an address this map allocated with `hipMalloc` and
            // handed to nobody who outlives the arena.
            let _ = unsafe { (api.hip.hipFree)(slab.ptr) };
        }
    }
    for ptr in held.retired {
        if !ptr.is_null() {
            // SAFETY: as above — an address this map allocated and never
            // handed to anything that outlives the arena.
            let _ = unsafe { (api.hip.hipFree)(ptr) };
        }
    }
}

/// The stream's `hipStreamCaptureStatus`, or `None` when HIP would not say.
pub(crate) fn capture_status(stream: *mut c_void) -> Option<i32> {
    let api = sys::api().ok()?;
    let mut status = sys::STREAM_CAPTURE_STATUS_NONE;
    // SAFETY: `stream` is the live stream of the caller's `Ctx` (`Ctx::on`);
    // `status` a live local.
    if unsafe { (api.hip.hipStreamIsCapturing)(stream, &raw mut status) } != sys::HIP_SUCCESS {
        // SAFETY: `hipGetLastError` takes nothing; it clears the error the
        // failed query left.
        let _ = unsafe { (api.hip.hipGetLastError)() };
        return None;
    }
    Some(status)
}

fn scope_of(stream: *mut c_void, region: u32) -> usize {
    if region == super::ctx::NO_REGION {
        stream.addr()
    } else {
        0
    }
}

/// `owner` is the stream a stream-scoped slab belongs to: the launching
/// stream, or the one whose order it launches in.
pub(crate) fn take(
    arena: u32,
    stream: *mut c_void,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    take_as(arena, stream, owner, name, region, bytes, false)
}

/// [`take`], a slab grown zeroed: for counters a launch leaves as it found
/// them (each fire's last block puts its tickets back to zero).
pub(crate) fn take_zeroed(
    arena: u32,
    stream: *mut c_void,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    take_as(arena, stream, owner, name, region, bytes, true)
}

fn take_as(
    arena: u32,
    stream: *mut c_void,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
    zero: bool,
) -> Result<*mut c_void, Fault> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    let mut arenas = locked();
    let held = arenas.entry(arena).or_default();
    let scope = scope_of(owner, region);
    if let Some(slab) = held.slabs.get(&(name, region, scope))
        && slab.bytes >= bytes
    {
        return Ok(slab.ptr);
    }
    if capture_status(stream).is_some_and(|s| s != sys::STREAM_CAPTURE_STATUS_NONE) {
        return Err(Fault::Unwarmed {
            name,
            have: held
                .slabs
                .get(&(name, region, scope))
                .map_or(0, |slab| slab.bytes),
            need: bytes,
        });
    }
    grow(held, name, region, scope, bytes, zero)?;
    Ok(held.slabs[&(name, region, scope)].ptr)
}

#[must_use]
pub fn census() -> Vec<(&'static str, usize)> {
    let arenas = locked();
    let mut by_name: HashMap<&'static str, usize> = HashMap::new();
    for arena in arenas.values() {
        for ((name, _, _), slab) in &arena.slabs {
            *by_name.entry(name).or_insert(0) += slab.bytes;
        }
    }
    let mut out: Vec<(&'static str, usize)> = by_name.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    out
}

const SLAB_GRAIN: usize = 8 << 20;

fn grow(
    arena: &mut Arena,
    name: &'static str,
    region: u32,
    scope: usize,
    bytes: usize,
    zero: bool,
) -> Result<(), Fault> {
    let (old_ptr, old_bytes) = arena
        .slabs
        .get(&(name, region, scope))
        .map_or((core::ptr::null_mut(), 0), |slab| (slab.ptr, slab.bytes));
    if old_bytes >= bytes {
        return Ok(());
    }
    let want = bytes.div_ceil(SLAB_GRAIN).saturating_mul(SLAB_GRAIN);
    let api = sys::api().map_err(|_| Fault::Device {
        call: "hipMalloc",
        code: -1,
    })?;
    let mut fresh: *mut c_void = core::ptr::null_mut();

    // SAFETY: a live local out-parameter and a byte count this caller checked
    // is non-zero.
    let code = unsafe { (api.hip.hipMalloc)(&raw mut fresh, want) };
    if code == sys::ERROR_OUT_OF_MEMORY {
        // SAFETY: `hipGetLastError` takes nothing; it clears the out-of-memory
        // error so the next call does not see it.
        let _ = unsafe { (api.hip.hipGetLastError)() };
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let _ = unsafe { (api.hip.hipMemGetInfo)(&raw mut free, &raw mut total) };
        return Err(Fault::Exhausted {
            name,
            have: old_bytes,
            need: want,
            free,
        });
    }
    if code != sys::HIP_SUCCESS || fresh.is_null() {
        return Err(Fault::Device {
            call: "hipMalloc",
            code,
        });
    }
    if zero {
        // SAFETY: the slab just allocated, `want` bytes long.
        let code = unsafe { (api.hip.hipMemset)(fresh, 0, want) };
        if code != sys::HIP_SUCCESS {
            return Err(Fault::Device {
                call: "hipMemset",
                code,
            });
        }
    }
    if !old_ptr.is_null() {
        arena.retired.push(old_ptr);
    }
    arena.slabs.insert(
        (name, region, scope),
        Slab {
            ptr: fresh,
            bytes: want,
        },
    );
    Ok(())
}

fn current() -> Option<i32> {
    let api = sys::api().ok()?;
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a live local out-parameter.
    (unsafe { (api.hip.hipGetDevice)(&raw mut ordinal) } == sys::HIP_SUCCESS).then_some(ordinal)
}

/// A device fact read once per device: a process may drive several GPUs,
/// and the first one asked does not answer for the others.
fn per_device(
    cache: &'static OnceLock<Mutex<HashMap<i32, Option<u32>>>>,
    read: impl FnOnce(i32) -> Option<u32>,
) -> Option<u32> {
    let ordinal = current()?;
    let mut map = cache
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    *map.entry(ordinal).or_insert_with(|| read(ordinal))
}

#[must_use]
pub(crate) fn multiprocessors() -> Option<u32> {
    static COUNT: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&COUNT, |o| attribute(sys::ATTR_MULTIPROCESSOR_COUNT, o))
}

/// The gfx major: 9 on CDNA (gfx90a, gfx942, gfx950), 11/12 on RDNA.
#[must_use]
pub(crate) fn compute_capability_major() -> Option<u32> {
    static MAJOR: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&MAJOR, |o| attribute(sys::ATTR_COMPUTE_CAPABILITY_MAJOR, o))
}

#[must_use]
pub(crate) fn max_shared_memory_per_sm() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(sys::ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, o)
    })
}

#[must_use]
pub(crate) fn max_shared_memory_per_block_optin() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(sys::ATTR_SHARED_MEM_PER_BLOCK_OPTIN, o)
    })
}

/// The device's `gcnArchName` with its target features stripped: `gfx942`
/// for `gfx942:sramecc+:xnack-`.
pub(crate) fn arch_name(ordinal: i32) -> Option<String> {
    let api = sys::api().ok()?;
    let mut prop = vec![0u8; sys::DEVICE_PROP_BYTES];
    // SAFETY: `prop` is a zeroed buffer the size of `hipDeviceProp_tR0600`.
    let code = unsafe { (api.hip.hipGetDevicePropertiesR0600)(prop.as_mut_ptr().cast(), ordinal) };
    if code != sys::HIP_SUCCESS {
        return None;
    }
    let field = &prop[sys::DEVICE_PROP_GCN_ARCH_NAME..sys::DEVICE_PROP_GCN_ARCH_NAME + 256];
    let full = std::ffi::CStr::from_bytes_until_nul(field)
        .ok()?
        .to_str()
        .ok()?;
    let base = full.split(':').next()?;
    base.starts_with("gfx").then(|| base.to_owned())
}

fn attribute(which: i32, ordinal: i32) -> Option<u32> {
    let api = sys::api().ok()?;
    let mut value: i32 = 0;

    // SAFETY: `value` is a live local out-parameter; HIP range-checks `which`
    // and `ordinal`.
    let code = unsafe { (api.hip.hipDeviceGetAttribute)(&raw mut value, which, ordinal) };
    (code == sys::HIP_SUCCESS && value > 0).then(|| value.unsigned_abs())
}
