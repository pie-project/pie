use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;
use kernels::plane::Refusal;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for Slab {}

fn slabs() -> &'static Mutex<HashMap<&'static str, Slab>> {
    static SLABS: OnceLock<Mutex<HashMap<&'static str, Slab>>> = OnceLock::new();
    SLABS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn take(name: &'static str, bytes: usize) -> Result<*mut c_void, Refusal> {
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
    if !slab.ptr.is_null() {
        let _ = unsafe { rt::cudaFree(slab.ptr) };
        slab.ptr = core::ptr::null_mut();
        slab.bytes = 0;
    }
    let mut fresh: *mut c_void = core::ptr::null_mut();

    let code = unsafe { rt::cudaMalloc(&raw mut fresh, bytes) };
    if code != rt::cudaError::cudaSuccess || fresh.is_null() {
        return Err(Refusal::Device {
            why: "the device scratch could not be allocated",
        });
    }
    slab.ptr = fresh;
    slab.bytes = bytes;
    Ok(fresh)
}

pub fn multiprocessors() -> Result<u32, Refusal> {
    static COUNT: OnceLock<Option<u32>> = OnceLock::new();
    (*COUNT.get_or_init(|| {
        let mut ordinal: i32 = 0;

        if unsafe { rt::cudaGetDevice(&raw mut ordinal) } != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut count: i32 = 0;

        let code = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut count,
                rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                ordinal,
            )
        };
        (code == rt::cudaError::cudaSuccess && count > 0).then(|| count.unsigned_abs())
    }))
    .ok_or(Refusal::Device {
        why: "the device would not say how many multiprocessors it has",
    })
}

pub fn compute_capability_major() -> Option<u32> {
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

pub fn max_shared_memory_per_sm() -> Result<u32, Refusal> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    (*BYTES
        .get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerMultiprocessor)))
    .ok_or(Refusal::Device {
        why: "the device would not say its shared memory per SM",
    })
}

pub fn max_shared_memory_per_block_optin() -> Result<u32, Refusal> {
    static BYTES: OnceLock<Option<u32>> = OnceLock::new();
    (*BYTES.get_or_init(|| attribute(rt::cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlockOptin)))
        .ok_or(Refusal::Device {
            why: "the device would not say its opt-in shared memory cap",
        })
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

pub unsafe fn upload(dst: *mut c_void, src: &[u8], stream: *mut c_void) -> Result<(), Refusal> {
    if src.is_empty() {
        return Ok(());
    }

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
        Err(Refusal::Device {
            why: "the host-to-device copy failed",
        })
    }
}
