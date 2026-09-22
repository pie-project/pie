use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;

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
// outright. `cudaMalloc`/`cudaFree` are thread-safe and the map is behind a
// mutex; nothing else holds a slab except a launch that has already been
// enqueued.
unsafe impl Send for Arena {}

fn arenas() -> &'static Mutex<HashMap<u32, Arena>> {
    static ARENAS: OnceLock<Mutex<HashMap<u32, Arena>>> = OnceLock::new();
    ARENAS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn locked() -> std::sync::MutexGuard<'static, HashMap<u32, Arena>> {
    arenas()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn attach(arena: u32, stream: *mut c_void) {
    let _ = (arena, stream);
}

pub(crate) fn release(arena: u32) {
    let mut arenas = locked();
    let Some(held) = arenas.remove(&arena) else {
        return;
    };
    for (_, slab) in held.slabs {
        if !slab.ptr.is_null() {
            // SAFETY: an address this map allocated with `cudaMalloc` and
            // handed to nobody who outlives the arena.
            let _ = unsafe { rt::cudaFree(slab.ptr) };
        }
    }
    for ptr in held.retired {
        if !ptr.is_null() {
            // SAFETY: as above — an address this map allocated and never
            // handed to anything that outlives the arena.
            let _ = unsafe { rt::cudaFree(ptr) };
        }
    }
}

pub(crate) fn capture_status(stream: *mut c_void) -> Option<rt::cudaStreamCaptureStatus> {
    let mut status = rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    if unsafe { rt::cudaStreamIsCapturing(stream.cast(), &raw mut status) }
        != rt::cudaError::cudaSuccess
    {
        let _ = unsafe { rt::cudaGetLastError() };
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

pub(crate) fn take(
    arena: u32,
    stream: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    let mut arenas = locked();
    let held = arenas.entry(arena).or_default();
    let scope = scope_of(stream, region);
    if let Some(slab) = held.slabs.get(&(name, region, scope))
        && slab.bytes >= bytes
    {
        return Ok(slab.ptr);
    }
    if capture_status(stream)
        .is_some_and(|s| s != rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
    {
        return Err(Fault::Unwarmed {
            name,
            have: held
                .slabs
                .get(&(name, region, scope))
                .map_or(0, |slab| slab.bytes),
            need: bytes,
        });
    }
    grow(held, name, region, scope, bytes)?;
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
) -> Result<(), Fault> {
    let (old_ptr, old_bytes) = arena
        .slabs
        .get(&(name, region, scope))
        .map_or((core::ptr::null_mut(), 0), |slab| (slab.ptr, slab.bytes));
    if old_bytes >= bytes {
        return Ok(());
    }
    let want = bytes.div_ceil(SLAB_GRAIN).saturating_mul(SLAB_GRAIN);
    let mut fresh: *mut c_void = core::ptr::null_mut();

    // SAFETY: a live local out-parameter and a byte count this caller checked
    // is non-zero.
    let code = unsafe { rt::cudaMalloc(&raw mut fresh, want) };
    if code != rt::cudaError::cudaSuccess || fresh.is_null() {
        return Err(Fault::Device {
            call: "cudaMalloc",
            code: code as i32,
        });
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

#[must_use]
pub(crate) fn multiprocessors() -> Option<u32> {
    static COUNT: OnceLock<Option<u32>> = OnceLock::new();
    *COUNT.get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount))
}

#[must_use]
pub(crate) fn compute_capability_major() -> Option<u32> {
    static MAJOR: OnceLock<Option<u32>> = OnceLock::new();
    *MAJOR.get_or_init(|| {
        use cudarc::driver::sys as dr;

        let mut ordinal: i32 = 0;

        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return None;
        }
        cudarc::driver::result::init().ok()?;
        let mut device: dr::CUdevice = 0;

        if unsafe { dr::cuDeviceGet(&raw mut device, ordinal) } != dr::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut major: i32 = 0;

        let code = unsafe {
            dr::cuDeviceGetAttribute(
                &raw mut major,
                dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        };
        (code == dr::CUresult::CUDA_SUCCESS && major > 0).then(|| major.unsigned_abs())
    })
}

#[allow(dead_code)]
#[must_use]
pub(crate) fn max_shared_memory_per_sm() -> Option<u32> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    *BYTES
        .get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor))
}

#[allow(dead_code)]
#[must_use]
pub(crate) fn max_shared_memory_per_block_optin() -> Option<u32> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    *BYTES.get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlockOptin))
}

pub(crate) fn properties(ordinal: i32) -> Option<rt::cudaDeviceProp> {
    let mut prop: rt::cudaDeviceProp = unsafe { core::mem::zeroed() };
    let code = unsafe { rt::cudaGetDeviceProperties(&raw mut prop, ordinal) };
    (code == rt::cudaError::cudaSuccess).then_some(prop)
}

fn attribute(which: rt::cudaDeviceAttr) -> Option<u32> {
    let mut ordinal: i32 = 0;

    if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
        return None;
    }
    let mut value: i32 = 0;

    let code = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
    (code == rt::cudaError::cudaSuccess && value > 0).then(|| value.unsigned_abs())
}
