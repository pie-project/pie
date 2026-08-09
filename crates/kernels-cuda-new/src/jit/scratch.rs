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
