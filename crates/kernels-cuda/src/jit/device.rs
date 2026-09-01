//! The device probe and the scratch slabs. Every probe runs once and is
//! cached; the slabs grow and never shrink, because an entry may not
//! allocate per fire (graph capture forbids it). A slab that would have to
//! grow under a capturing stream is refused, not grown — the contract lives
//! on [`Ctx::scratch`](crate::jit::Ctx::scratch).
//!
//! # A GROWN SLAB RETIRES ITS PREDECESSOR; IT DOES NOT FREE IT
//!
//! **ADDRESSES ARE BAKE-TIME** (alto article 7), and a scratch address is
//! baked exactly like a pool address: a `cudaGraphExec_t` recorded at fire N
//! holds the staging pointer the entry took THEN, and a folded bucket replays
//! that exec for the rest of the load. Growth used to be `cudaFree` +
//! `cudaMalloc`, which made every graph recorded before the growth hold a
//! freed block — a `cudaErrorIllegalAddress` on the first replay after a fire
//! brought one more row than any fire before it. (Measured: the fold-hint gate
//! died in `ssm_gdn_prep_v_gates` writing `g_log_out` into
//! `attn.ssm_gdn_chunk_gates` at the address a five-row fire grew and a
//! six-row fire freed.)
//!
//! `Ctx::scratch`'s "warm at full fire shape before capturing" is what keeps a
//! capture IN PROGRESS from growing, and it is all it can keep: the shapes a
//! serving load brings are not bounded by the ones it has already brought, so
//! a later, larger fire will grow a name whatever the warm pass did. So the
//! superseded block is RETIRED — kept alive, unreferenced by this map, freed
//! with the arena — and a graph recorded against it keeps reading and writing
//! a block that is still its own and still big enough for the shape it was
//! recorded at. Growth is geometric (a slab at least doubles), so a name
//! grows a bounded number of times and everything retired for it sums to less
//! than what it currently holds: the ceiling is twice the live scratch, once,
//! rather than an unbounded ladder.
//!
//! # A slab is per `(arena, name, stream)`, and each of the three is a bug
//! that was measured
//!
//! It was keyed by NAME alone, process-wide, and that key was wrong twice.
//!
//! **The ARENA is the shell.** Two `Shell`s in one process fired into one
//! another's staging planes and both computed: build log 18 measured a
//! continuation of `"PPP is目前是. \{ a a \)"` where the same load alone says
//! `" Paris"`, and the whole tree worked around it by admitting one shell per
//! process. An arena is minted by [`Slabs::open`](crate::Slabs::open), one per
//! CUDA context, and [`released`](crate::Slabs::release) with it, so two
//! shells now share nothing and neither leaks the other's slabs at teardown.
//!
//! **The STREAM is P6.** Two arms of one fork group run at the same instant
//! by construction, so two regions staging through one slab stage over each
//! other — which is why `engine_cuda::EXCLUSIVE` named eleven entries and
//! ordered every linear-attention split in qwen and kimi back into a line
//! (build log 24). Per stream, they are disjoint by the same argument that
//! makes two streams worth having.
//!
//! # Growth is BROADCAST across the arena's streams, and that is the warming
//! contract
//!
//! The shell warms a load by firing it EAGERLY — one stream, program order —
//! and only then records the same regions across the side streams. A slab
//! sized on the eager pass's stream would therefore be missing on every
//! stream the capture actually uses, and [`take`] would answer
//! [`Fault::Unwarmed`] for a load that did exactly what the contract asked.
//!
//! So a name that grows on one of an arena's streams grows on all of them, at
//! the same instant and to the same size. The eager warm pass is then the
//! warm pass for the capture too, unchanged, and the cost is the honest one:
//! an arena holds `streams × bytes` of a name it uses on one stream. Side
//! streams are counted in ones and twos (`DeviceProfile::side_streams`), and
//! the alternative — allocating on the stream that asks — is a load that
//! refuses.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;

use crate::jit::Fault;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

/// One arena: the streams a context fires on, and one slab per
/// `(name, stream)`.
///
/// The streams are addresses rather than pointers because that is all a key
/// needs, and because a `usize` is `Send` without an unsafe promise about a
/// handle this map never dereferences.
#[derive(Default)]
struct Arena {
    /// Every stream this arena's growth must cover. Registered at
    /// [`attach`], which the shell calls when it opens a stream — before its
    /// first fire, which is what makes the broadcast above complete.
    streams: Vec<usize>,
    slabs: HashMap<(&'static str, usize), Slab>,
    /// **Superseded allocations, still live.** A slab that grew handed its
    /// old block here instead of to `cudaFree`, because a graph recorded
    /// before the growth still launches against that address (the module
    /// comment argues it). Freed with the arena, and with nothing else.
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

/// Tell `arena` that it fires on `stream` too, so a slab grown anywhere in it
/// is grown here as well.
///
/// Idempotent, and deliberately not retroactive: a stream attached AFTER a
/// name has already grown gets its slab on that name's next growth, which for
/// the shell is never a question because [`Context::open_lanes`] runs at load
/// and every fire is after it.
pub(crate) fn attach(arena: u32, stream: *mut c_void) {
    let mut arenas = locked();
    let held = arenas.entry(arena).or_default();
    let at = stream.addr();
    if !held.streams.contains(&at) {
        held.streams.push(at);
    }
}

/// Free every slab this arena holds and forget its streams.
///
/// **THE SHELL'S TEARDOWN IS THE ONLY CALLER**, and it is what makes a second
/// shell in one process cost what the first one did rather than twice it. A
/// slab is only ever read by a launch that was enqueued on one of this
/// arena's streams, so a context that has synchronized and is dropping them
/// has nothing left in flight.
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
    // The blocks that growth superseded. The same argument frees them: the
    // only thing that ever read one is a launch on a stream this arena's
    // owner has synchronized and is dropping.
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

pub(crate) fn take(
    arena: u32,
    stream: *mut c_void,
    name: &'static str,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    let mut arenas = locked();
    let held = arenas.entry(arena).or_default();
    let here = stream.addr();
    if let Some(slab) = held.slabs.get(&(name, here))
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
            have: held.slabs.get(&(name, here)).map_or(0, |slab| slab.bytes),
            need: bytes,
        });
    }
    // Every stream at once, so that the eager pass that warms this name warms
    // it for the capture pass that will run the same region elsewhere.
    let mut on = held.streams.clone();
    if !on.contains(&here) {
        on.push(here);
    }
    for at in on {
        grow(held, name, at, bytes)?;
    }
    Ok(held.slabs[&(name, here)].ptr)
}

/// Size one `(name, stream)` slab to at least `bytes`, allocating a fresh
/// block and RETIRING the old one when it is short. Never called under
/// capture — [`take`] refuses there first.
///
/// Three properties this shape has and the `cudaFree`-then-`cudaMalloc` one
/// did not:
///
/// * **The old address stays valid** for the graphs that baked it (the module
///   comment).
/// * **Growth is geometric**, so a name is reallocated a bounded number of
///   times and what it retires sums to less than what it holds.
/// * **A failed `cudaMalloc` leaves the slab it had.** The old order freed
///   first, so an out-of-memory growth left a null slab behind a `Fault` the
///   caller could survive.
fn grow(arena: &mut Arena, name: &'static str, stream: usize, bytes: usize) -> Result<(), Fault> {
    let (old_ptr, old_bytes) = arena
        .slabs
        .get(&(name, stream))
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
        (name, stream),
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
    #[cfg(feature = "cuda-12")]
    let code = unsafe { rt::cudaGetDeviceProperties_v2(&raw mut prop, ordinal) };
    #[cfg(feature = "cuda-13")]
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
