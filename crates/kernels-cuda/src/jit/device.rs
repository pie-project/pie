#![allow(unsafe_code)]
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
    /// Slabs laid in memory the engine holds ([`place`]): never grown,
    /// never freed here.
    placed: HashMap<(&'static str, u32, usize), Slab>,
    /// The engine laid every slab its fires take ([`seal`]): a take past
    /// them is a fire it did not walk, and nothing grows.
    sealed: bool,
}

use super::ctx::{PLACED_ALIGN, SlabKey};

/// [`super::Slabs::place`].
pub(crate) fn place(base: *mut c_void, slabs: &[(SlabKey, usize)]) {
    let mut arenas = locked();
    let mut at = 0usize;
    for (key, bytes) in slabs {
        let bytes = bytes.next_multiple_of(PLACED_ALIGN);
        arenas.entry(key.arena).or_default().placed.insert(
            (key.name, key.region, key.scope),
            Slab {
                // SAFETY: inside the span the caller holds, by `placed_bytes`.
                ptr: unsafe { base.byte_add(at) },
                bytes,
            },
        );
        at += bytes;
    }
}

/// [`super::Slabs::seal`].
pub(crate) fn seal(arena: u32) {
    locked().entry(arena).or_default().sealed = true;
}

/// [`super::Slabs::unplaced_bytes`].
pub(crate) fn unplaced_bytes() -> usize {
    UNPLACED.load(core::sync::atomic::Ordering::Relaxed)
}

static UNPLACED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

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
    arenas().lock().expect("lock poisoned")
}

/// A size-only walk's slabs (`crate::sizing`): each key's size as the walk
/// would have grown it, and every allocation it would have made.
#[derive(Default)]
struct Sized {
    slabs: HashMap<(u32, &'static str, u32, usize), usize>,
    grown: usize,
    by_name: HashMap<&'static str, usize>,
    /// By key: the most asked, and what growing to it allocated.
    asked: HashMap<(u32, &'static str, u32, usize), (usize, usize)>,
}

thread_local! {
    static SIZED: core::cell::RefCell<Sized> = core::cell::RefCell::new(Sized::default());
}

pub(crate) fn sizing_reset() {
    SIZED.with(|sized| *sized.borrow_mut() = Sized::default());
}

pub(crate) fn sizing_take() -> crate::sizing::Footprint {
    SIZED.with(|sized| {
        let sized = core::mem::take(&mut *sized.borrow_mut());
        let mut slabs: Vec<(&'static str, usize)> = sized.by_name.into_iter().collect();
        slabs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let mut keys: Vec<crate::sizing::Sized> = sized
            .asked
            .into_iter()
            .map(|((arena, name, region, scope), (asked, grown))| crate::sizing::Sized {
                key: SlabKey {
                    arena,
                    name,
                    region,
                    scope,
                },
                asked,
                grown,
            })
            .collect();
        keys.sort_by(|a, b| a.key.cmp(&b.key));
        crate::sizing::Footprint {
            bytes: sized.grown,
            slabs,
            keys,
        }
    })
}

/// [`take`] in a size-only walk: the slab `grow` would leave, counted, and
/// a stand-in address for it.
fn take_sized(
    arena: u32,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> *mut c_void {
    let scope = scope_of(owner, region);
    let held = locked()
        .get(&arena)
        .and_then(|held| held.slabs.get(&(name, region, scope)))
        .map_or(0, |slab| slab.bytes);
    SIZED.with(|sized| {
        let mut sized = sized.borrow_mut();
        let key = (arena, name, region, scope);
        let have = sized.slabs.get(&key).copied().unwrap_or(held);
        let asked = sized.asked.entry(key).or_insert((0, 0));
        asked.0 = asked.0.max(bytes);
        if bytes > have {
            let grain = if bytes < SLAB_GRAIN {
                SMALL_SLAB_GRAIN
            } else {
                SLAB_GRAIN
            };
            let want = bytes.div_ceil(grain).saturating_mul(grain);
            sized.slabs.insert(key, want);
            if let Some(asked) = sized.asked.get_mut(&key) {
                asked.1 = asked.1.saturating_add(want);
            }
            sized.grown = sized.grown.saturating_add(want);
            *sized.by_name.entry(name).or_insert(0) += want;
        }
    });
    crate::sizing::STAND_IN as *mut c_void
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
    // SAFETY: `stream` is a `Ctx`'s stream, live by `Ctx::on`'s contract;
    // `status` is a local out-pointer.
    if unsafe { rt::cudaStreamIsCapturing(stream.cast(), &raw mut status) }
        != rt::cudaError::cudaSuccess
    {
        // SAFETY: takes no arguments; it only clears the sticky error the
        // failed query left on this thread.
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
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    if crate::sizing::active() {
        let _ = stream;
        return Ok(take_sized(arena, owner, name, region, bytes));
    }
    let mut arenas = locked();
    let held = arenas.entry(arena).or_default();
    let scope = scope_of(owner, region);
    if let Some(slab) = held.placed.get(&(name, region, scope))
        && slab.bytes >= bytes
    {
        return Ok(slab.ptr);
    }
    if let Some(slab) = held.slabs.get(&(name, region, scope))
        && slab.bytes >= bytes
    {
        return Ok(slab.ptr);
    }
    // An arena whose slabs are laid grows none: every fire it serves was
    // walked, so a take past what is laid is a fire the walk did not see.
    if held.sealed {
        UNPLACED.fetch_add(bytes, core::sync::atomic::Ordering::Relaxed);
        return Err(Fault::Unwarmed {
            name,
            have: held
                .placed
                .get(&(name, region, scope))
                .map_or(0, |slab| slab.bytes),
            need: bytes,
        });
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

/// [`take`], a slab zeroed on `stream` whenever it is fresh: for counts its
/// kernels leave at zero between launches, so a slab is cleared once, not a
/// launch. (A capture never grows a slab, so the clear is never captured.)
pub(crate) fn take_zeroed(
    arena: u32,
    stream: *mut c_void,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    let key = (name, region, scope_of(owner, region));
    let (before, placed) = {
        let arenas = locked();
        let held = arenas.get(&arena);
        (
            held.and_then(|held| held.slabs.get(&key)).map(|slab| slab.ptr),
            held.and_then(|held| held.placed.get(&key)).map(|slab| slab.ptr),
        )
    };
    let ptr = take(arena, stream, owner, name, region, bytes)?;
    // A placed slab was zeroed where it was laid.
    if ptr.is_null() || before == Some(ptr) || placed == Some(ptr) || crate::sizing::active() {
        return Ok(ptr);
    }
    let size = locked()
        .get(&arena)
        .and_then(|held| held.slabs.get(&key))
        .map_or(bytes, |slab| slab.bytes);
    // SAFETY: the slab `take` just allocated, `size` bytes, on the stream
    // that launches what reads it.
    let code = unsafe { rt::cudaMemsetAsync(ptr, 0, size, stream.cast()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Fault::Device {
            call: "cudaMemsetAsync",
            code: code as i32,
        });
    }
    Ok(ptr)
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
const SMALL_SLAB_GRAIN: usize = 256 << 10;

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
    // A slab under the grain takes its own size to the small grain: one a
    // region (Gemma-4-26B's 30 MoE route orders of 0.1 MiB each) held 8 MiB
    // apiece, 240 MiB the cache rows never saw.
    let grain = if bytes < SLAB_GRAIN {
        SMALL_SLAB_GRAIN
    } else {
        SLAB_GRAIN
    };
    let want = bytes.div_ceil(grain).saturating_mul(grain);
    let mut fresh: *mut c_void = core::ptr::null_mut();

    // SAFETY: a live local out-parameter and a byte count this caller checked
    // is non-zero.
    let code = unsafe { rt::cudaMalloc(&raw mut fresh, want) };
    if code == rt::cudaError::cudaErrorMemoryAllocation {
        // SAFETY: takes no arguments; it only clears the failed allocation's
        // error so the next call does not report it.
        let _ = unsafe { rt::cudaGetLastError() };
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let _ = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        return Err(Fault::Exhausted {
            name,
            have: old_bytes,
            need: want,
            free,
        });
    }
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

fn current() -> Option<i32> {
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a local out-pointer.
    (unsafe { rt::cudaGetDevice(&raw mut ordinal) } == rt::cudaError::cudaSuccess)
        .then_some(ordinal)
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
    per_device(&COUNT, |o| {
        attribute(rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount, o)
    })
}

#[must_use]
pub(crate) fn compute_capability_major() -> Option<u32> {
    static MAJOR: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&MAJOR, |ordinal| {
        use cudarc::driver::sys as dr;
        cudarc::driver::result::init().ok()?;
        let mut device: dr::CUdevice = 0;

        // SAFETY: the driver was initialised just above and `device` is a
        // local out-pointer.
        if unsafe { dr::cuDeviceGet(&raw mut device, ordinal) } != dr::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut major: i32 = 0;

        // SAFETY: `major` is a local out-pointer and `device` is the handle
        // `cuDeviceGet` just returned.
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

#[must_use]
pub(crate) fn max_shared_memory_per_sm() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(
            rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor,
            o,
        )
    })
}

#[must_use]
pub(crate) fn max_shared_memory_per_block_optin() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(
            rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlockOptin,
            o,
        )
    })
}

pub(crate) fn properties(ordinal: i32) -> Option<rt::cudaDeviceProp> {
    // SAFETY: `cudaDeviceProp` is a C struct of integers, arrays and a
    // UUID, for which all-zero bytes are a valid value.
    let mut prop: rt::cudaDeviceProp = unsafe { core::mem::zeroed() };
    // SAFETY: `prop` is a local out-pointer of the struct's own type; a bad
    // `ordinal` comes back as an error code.
    let code = unsafe { rt::cudaGetDeviceProperties(&raw mut prop, ordinal) };
    (code == rt::cudaError::cudaSuccess).then_some(prop)
}

fn attribute(which: rt::cudaDeviceAttr, ordinal: i32) -> Option<u32> {
    let mut value: i32 = 0;

    // SAFETY: `value` is a local out-pointer; a bad attribute or ordinal
    // comes back as an error code.
    let code = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
    (code == rt::cudaError::cudaSuccess && value > 0).then(|| value.unsigned_abs())
}
