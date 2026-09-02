//! The device probe and the scratch slabs. Every probe runs once and is
//! cached; slabs grow and never shrink, since an entry may not allocate per
//! fire (graph capture forbids it). A slab that would have to grow under a
//! capturing stream is refused, not grown.
//!
//! A grown slab retires its predecessor rather than freeing it: a
//! `cudaGraphExec_t` recorded at fire N holds the staging pointer the entry
//! took then, so freeing the old block on growth would leave earlier-recorded
//! graphs replaying against freed memory. The retired block stays alive,
//! unreferenced by this map, freed only with the arena. Growth is geometric
//! (a slab at least doubles), so a name grows a bounded number of times and
//! everything retired for it sums to less than what it currently holds.
//!
//! A slab is keyed by `(arena, name, region, scope)`. The arena is the
//! shell's own CUDA context (minted by [`Slabs::open`](crate::Slabs::open),
//! released with it), so two shells in one process never share a slab. The
//! region is the template region the walk was inside when the entry asked:
//! two regions are disjoint by construction (two arms of one fork group run
//! at the same instant), regardless of which stream the walk put them on.
//! For a caller in no region ([`super::ctx::NO_REGION`]), the stream is the
//! separation instead — see [`scope_of`].
//!
//! The shell warms a load by firing it eagerly on one stream, then records
//! the same regions across side streams; keying by region (not stream) means
//! the eager warm pass and the capture of the same region resolve to the
//! same block regardless of which stream either fires on, so [`take`] never
//! spuriously answers [`Fault::Unwarmed`] for a properly warmed load.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;

use crate::jit::Fault;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

/// One arena: one slab per `(name, region, scope)`. See module docs for what
/// region and scope key against.
#[derive(Default)]
struct Arena {
    slabs: HashMap<(&'static str, u32, usize), Slab>,
    /// Superseded allocations, still live: a slab that grew handed its old
    /// block here instead of to `cudaFree`, since a graph recorded before
    /// the growth still launches against that address. Freed only with the arena.
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

/// Tell `arena` that it fires on `stream` too. A no-op now that a slab is
/// keyed by region rather than stream; kept as a seam for the day a
/// per-stream fact is needed again.
pub(crate) fn attach(arena: u32, stream: *mut c_void) {
    let _ = (arena, stream);
}

/// Free every slab this arena holds. Called only at the shell's teardown; a
/// slab is only ever read by a launch enqueued on one of this arena's
/// streams, so a synchronized, dropping context has nothing left in flight.
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
    // The blocks growth superseded free the same way: only a launch on a
    // stream this arena's owner has synchronized ever read one.
    for ptr in held.retired {
        if !ptr.is_null() {
            // SAFETY: as above — an address this map allocated and never
            // handed to anything that outlives the arena.
            let _ = unsafe { rt::cudaFree(ptr) };
        }
    }
}

/// Whether `stream` is mid-capture; `None` when the runtime will not say
/// (the pending error is cleared). The one `cudaStreamIsCapturing` query
/// the plane makes — the dense autotuner's guards and [`take`]'s growth
/// refusal both read it.
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

/// What separates two askers that are not in a region: the stream. Every
/// caller outside a walk keys [`NO_REGION`](super::ctx::NO_REGION), so
/// without this they would share one block despite running at once (ad-hoc
/// contexts — a transform executor, a bench, test threads — share one arena
/// on `Slabs::PROCESS`). Inside a walk the region separates and the stream
/// is not asked; outside one the stream separates. Neither subsumes the
/// other.
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
    // Growth allocates (fresh block, old one retired — see `grow` below),
    // and an allocation under capture is illegal host work, so an un-warmed
    // slab is a refusal, not a corruption.
    if capture_status(stream)
        .is_some_and(|s| s != rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
    {
        return Err(Fault::Unwarmed {
            name,
            have: held.slabs.get(&(name, region, scope)).map_or(0, |slab| slab.bytes),
            need: bytes,
        });
    }
    // One block, no broadcast: the eager pass that warms this name warms the
    // one block the capture pass will bake, since both walks ask for the
    // same region's slab.
    grow(held, name, region, scope, bytes)?;
    Ok(held.slabs[&(name, region, scope)].ptr)
}

/// Size one `(name, region, scope)` slab to at least `bytes`, allocating a
/// fresh block and retiring the old one when short. Never called under
/// capture — [`take`] refuses there first. Growth is geometric, and a failed
/// `cudaMalloc` leaves the old slab intact rather than freeing it first.
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
    let want = bytes.max(old_bytes.saturating_mul(2));
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

/// Unfired until the attn wave: its plan builders size shared-memory tiles
/// from these two probes.
#[allow(dead_code)]
#[must_use]
pub(crate) fn max_shared_memory_per_sm() -> Option<u32> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    *BYTES
        .get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor))
}

/// Unfired until the attn wave, as above.
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
