//! The device probe and the process-global scratch slabs. Every probe runs
//! once and is cached; the slabs grow and never shrink, because an entry may
//! not allocate per fire (graph capture forbids it). A slab that would have
//! to grow under a capturing stream is refused, not grown — the contract
//! lives on [`Ctx::scratch`](crate::jit::Ctx::scratch).

use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;

use crate::jit::Fault;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for Slab {}

fn slabs() -> &'static Mutex<HashMap<&'static str, Slab>> {
    static SLABS: OnceLock<Mutex<HashMap<&'static str, Slab>>> = OnceLock::new();
    SLABS.get_or_init(|| Mutex::new(HashMap::new()))
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
    stream: *mut c_void,
    name: &'static str,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    let mut slabs = slabs()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let slab = slabs.entry(name).or_insert(Slab {
        ptr: core::ptr::null_mut(),
        bytes: 0,
    });
    if slab.bytes >= bytes {
        return Ok(slab.ptr);
    }
    // Growth is `cudaFree` + `cudaMalloc`; under capture that poisons the
    // graph, so an un-warmed slab is a refusal, not a corruption.
    if capture_status(stream)
        .is_some_and(|s| s != rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusNone)
    {
        return Err(Fault::Unwarmed {
            name,
            have: slab.bytes,
            need: bytes,
        });
    }
    if !slab.ptr.is_null() {
        let _ = unsafe { rt::cudaFree(slab.ptr) };
        slab.ptr = core::ptr::null_mut();
        slab.bytes = 0;
    }
    let mut fresh: *mut c_void = core::ptr::null_mut();

    let code = unsafe { rt::cudaMalloc(&raw mut fresh, bytes) };
    if code != rt::cudaError::cudaSuccess || fresh.is_null() {
        return Err(Fault::Device {
            call: "cudaMalloc",
            code: code as i32,
        });
    }
    slab.ptr = fresh;
    slab.bytes = bytes;
    Ok(fresh)
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
