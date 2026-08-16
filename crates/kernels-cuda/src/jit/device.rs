//! The device facilities a body needs that are not a launch.
//!
//! Scratch memory, the four attributes a geometry is computed from, and the
//! host-to-device copy a plan upload is. Every one of them is a property of
//! the DEVICE rather than of a call, which is why the memos below are
//! process-wide statics and why [`Ctx`](crate::jit::Ctx) holds none of them
//! as a field.
//!
//! `pub(crate)` rather than private to `jit`, and the exception is worth
//! naming. A routine BODY reaches all of this through a `Ctx` method, which
//! is the surface the safety argument is written against. But FA2's device
//! facts are read by `attn::fa2`'s planners, which run before any fire and
//! have no `Ctx` to be given — a plan is built for a whole batch and a `Ctx`
//! is per call. Those three call `attribute`'s memos directly.
//!
//! `scratch.rs` until the FA2 host half descended into this crate. The file
//! held the allocator and, beside it, two attribute memos that were already
//! not scratch; the descent brought two more attributes and a `cudaMemcpyAsync`
//! and the name stopped describing any of it.
//!
//! # Why the attributes are memoised and the scratch is not
//!
//! An attribute cannot change for the life of a process, so the first answer
//! is the only answer and a `OnceLock` is exactly right. A scratch buffer
//! CAN change — it grows — so it is a `Mutex<HashMap>` that reallocates, and
//! the thing that must not change is the ADDRESS a launch was already handed,
//! which is why growth never frees anything a pending launch could be using.

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;
use kernels::routine::Refusal;

/// One named device scratch buffer: grow-only, process-lifetime.
struct Slab {
    /// The device address, or null before the first allocation.
    ptr: *mut c_void,
    /// How many bytes it holds.
    bytes: usize,
}

// SAFETY: the address is a device pointer, which is not tied to a thread; it
// is only ever read under the map's mutex.
unsafe impl Send for Slab {}

/// Every scratch buffer this process has taken, by the name that asked.
///
/// Keyed by name so that two routines wanting scratch do not hand each other
/// the same bytes. Grow-only and never freed: a buffer is reused for the rest
/// of the process, and freeing one while a launch that was handed it is still
/// in flight on some stream is exactly the failure this shape avoids.
fn slabs() -> &'static Mutex<HashMap<&'static str, Slab>> {
    static SLABS: OnceLock<Mutex<HashMap<&'static str, Slab>>> = OnceLock::new();
    SLABS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A device buffer of at least `bytes`, kept under `name`.
///
/// # Errors
///
/// [`Refusal::Device`] if the allocation fails. The C++ this replaces ignored
/// `cudaMalloc`'s return code and launched over a null pointer, which reads as
/// a token id out of unwritten memory rather than as a failure.
pub fn take(name: &'static str, bytes: usize) -> Result<*mut c_void, Refusal> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    let mut slabs = slabs().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let slab = slabs.entry(name).or_insert(Slab { ptr: core::ptr::null_mut(), bytes: 0 });
    if slab.bytes >= bytes {
        return Ok(slab.ptr);
    }
    if !slab.ptr.is_null() {
        // SAFETY: the address came from `cudaMalloc` below, and no launch that
        // was handed it can still be pending -- see this module's note on why
        // the buffer is never freed except to grow it.
        let _ = unsafe { rt::cudaFree(slab.ptr) };
        slab.ptr = core::ptr::null_mut();
        slab.bytes = 0;
    }
    let mut fresh: *mut c_void = core::ptr::null_mut();
    // SAFETY: `fresh` is a live, writable out-parameter.
    let code = unsafe { rt::cudaMalloc(&raw mut fresh, bytes) };
    if code != rt::cudaError::cudaSuccess || fresh.is_null() {
        return Err(Refusal::Device { why: "the device scratch could not be allocated" });
    }
    slab.ptr = fresh;
    slab.bytes = bytes;
    Ok(fresh)
}

/// How many multiprocessors this device has.
///
/// # Errors
///
/// [`Refusal::Device`] if the driver will not say. There is no default worth
/// guessing: for the fused LM-head GEMV the number is the grid AND the operand
/// the kernel strides the vocabulary by.
pub fn multiprocessors() -> Result<u32, Refusal> {
    static COUNT: OnceLock<Option<u32>> = OnceLock::new();
    (*COUNT.get_or_init(|| {
        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter.
        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut count: i32 = 0;
        // SAFETY: `count` is a live, writable out-parameter.
        let code = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut count,
                rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                ordinal,
            )
        };
        (code == rt::cudaError::cudaSuccess && count > 0).then(|| count.unsigned_abs())
    }))
    .ok_or(Refusal::Device { why: "the device would not say how many multiprocessors it has" })
}

/// This device's compute-capability major number, or `None` if the driver
/// will not say.
///
/// `None` rather than a refusal: every caller so far picks a tuning constant
/// from it and has a defensible answer for an unknown device, which is not
/// true of [`multiprocessors`] — that one IS a kernel argument.
pub fn compute_capability_major() -> Option<u32> {
    static MAJOR: OnceLock<Option<u32>> = OnceLock::new();
    *MAJOR.get_or_init(|| {
        use cudarc::driver::sys as dr;

        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter.
        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return None;
        }
        cudarc::driver::result::init().ok()?;
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a live, writable handle slot.
        if unsafe { dr::cuDeviceGet(&raw mut device, ordinal) } != dr::CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut major: i32 = 0;
        // SAFETY: `major` is a live out-parameter and `device` came from
        // `cuDeviceGet`.
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

/// How many bytes of shared memory one multiprocessor has.
///
/// `cudaDevAttrMaxSharedMemoryPerMultiprocessor`. FA2's occupancy arithmetic
/// divides by it: how many CTAs of a given shared-storage size fit on an SM is
/// the whole of what decides whether a decode plan may skip the split.
///
/// # Errors
///
/// [`Refusal::Device`] if the driver will not say. No default is guessed --
/// the number is a divisor in a geometry, and a wrong one silently produces a
/// grid that either leaves the device idle or will not launch.
pub fn max_shared_memory_per_sm() -> Result<u32, Refusal> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    (*BYTES.get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor)))
        .ok_or(Refusal::Device { why: "the device would not say its shared memory per SM" })
}

/// The largest dynamic shared-memory allocation a block may OPT IN to.
///
/// `cudaDevAttrMaxSharedMemoryPerBlockOptin`, which is deliberately not
/// `cudaDevAttrMaxSharedMemoryPerBlock`: the latter is the 48 KiB a kernel
/// gets without asking, and every FA2 prefill tile above the smallest asks.
/// Reading the wrong one of the two caps the lattice at a third of the device
/// and reports it as an unsupported tile.
///
/// # Errors
///
/// [`Refusal::Device`] if the driver will not say.
pub fn max_shared_memory_per_block_optin() -> Result<u32, Refusal> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    (*BYTES.get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlockOptin)))
        .ok_or(Refusal::Device { why: "the device would not say its opt-in shared memory cap" })
}

/// One device's whole `cudaDeviceProp`, or `None` if the driver will not say.
///
/// # Why this is a function at all, and why it is HERE
///
/// CUDA 12 renamed the entry point to `_v2` when `cudaDeviceProp` grew; CUDA
/// 13 dropped the suffix again and kept the wider struct. `cudarc` follows the
/// headers, so the SAME call is two different symbols depending on which
/// `cuda-1x` feature is on, and naming either one directly makes the crate
/// build under exactly one of them. That `#[cfg]` pair is the whole body.
///
/// It was written twice, and the second copy is what the quantised router's
/// descent found. `gemm::dense` had one (for the disk cache's `sm{major}
/// {minor}` header line) and `driver-cuda`'s `bind/quant_gemm.rs` had another,
/// byte-identical in its body and near-identical in its doc, because the two
/// callers were in different CRATES and neither could see the other. The
/// router moved into `gemm::quant`, the two copies landed in one directory,
/// and one of them had to go. It goes to the module whose subject is *"the
/// device facilities a body needs that are not a launch"*, rather than to
/// whichever of the two siblings happened to be older.
///
/// Not memoised, unlike the attribute queries above. Those answer one integer
/// each and are read per launch; this answers a kilobyte-wide struct and is
/// read twice per process — once when a cuBLASLt context probes for FP8
/// support, once when the dense tuner builds its cache header.
pub(crate) fn properties(ordinal: i32) -> Option<rt::cudaDeviceProp> {
    // SAFETY: `prop` is a live, writable `cudaDeviceProp` for the duration of
    // the call, which is the entry point's whole obligation. It is zeroed
    // first because the runtime fills only the fields its header knows and a
    // caller reading `name` off uninitialised bytes is the failure mode.
    let mut prop: rt::cudaDeviceProp = unsafe { core::mem::zeroed() };
    #[cfg(feature = "cuda-12")]
    // SAFETY: as above.
    let code = unsafe { rt::cudaGetDeviceProperties_v2(&raw mut prop, ordinal) };
    #[cfg(feature = "cuda-13")]
    // SAFETY: as above.
    let code = unsafe { rt::cudaGetDeviceProperties(&raw mut prop, ordinal) };
    (code == rt::cudaError::cudaSuccess).then_some(prop)
}

/// One `cudaDeviceGetAttribute` on the current device, or `None`.
///
/// Shared by the memos above rather than written out per attribute, because
/// what differs between them is one enum variant and what is identical is the
/// two failure modes -- no current device, and a driver that answers with a
/// non-positive number.
fn attribute(which: rt::cudaDeviceAttr) -> Option<u32> {
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a live, writable out-parameter.
    if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
        return None;
    }
    let mut value: i32 = 0;
    // SAFETY: `value` is a live, writable out-parameter.
    let code = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
    (code == rt::cudaError::cudaSuccess && value > 0).then(|| value.unsigned_abs())
}

/// Copy `src` from host memory to `dst` on the device, on `stream`.
///
/// The one host-to-device copy this crate makes, and it exists for one caller:
/// the FA2 plan upload. A plan is computed on the host into a staging buffer
/// and the kernels read it as an `int` arena, so the descent of that host half
/// brought the copy with it -- `driver-cuda`'s `pie::write_raw_span` was
/// added for exactly this call site and this is that function, one crate down.
///
/// **Asynchronous, and the caller owns what that means.** `src` is host memory
/// that must stay put until the copy retires. It is pageable rather than
/// pinned, and for pageable memory `cudaMemcpyAsync` stages through a driver
/// buffer before returning, so the borrow ends at return -- which is why this
/// takes a slice rather than demanding a pinned allocation. A pinned source
/// would be faster and would not have that property.
///
/// An empty `src` is a no-op rather than a refusal: a plan with no requests
/// has nothing to upload and is not an error.
///
/// # Errors
///
/// [`Refusal::Device`] if the copy fails.
///
/// # Safety
///
/// `dst` must address at least `src.len()` bytes of device memory, live until
/// the copy retires on `stream`, and `stream` must be a live CUDA stream.
pub unsafe fn upload(dst: *mut c_void, src: &[u8], stream: *mut c_void) -> Result<(), Refusal> {
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: the caller's obligation on `dst` and `stream`; `src` is a live
    // slice for the duration of the call, which is what a pageable source
    // needs since the driver stages it before returning.
    let code = unsafe {
        rt::cudaMemcpyAsync(
            dst,
            src.as_ptr().cast(),
            src.len(),
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.cast(),
        )
    };
    if code == rt::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Refusal::Device { why: "the host-to-device copy failed" })
    }
}
