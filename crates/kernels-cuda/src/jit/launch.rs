#![allow(unsafe_code)]
use core::ffi::c_void;

use cudarc::driver::sys as dr;

use crate::jit::{Fault, Launch};

const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;

/// # Safety
///
/// `function` is a live function handle in the thread's current context and
/// `stream` a live stream on that context (or null). `slots` holds one
/// pointer per kernel parameter, in order, each to at least that
/// parameter's size of readable bytes. Every device address among the
/// arguments is valid for whatever the kernel reads and writes there.
pub(crate) unsafe fn issue(
    function: dr::CUfunction,
    launch: Launch,
    slots: &mut [*mut c_void],
    stream: *mut c_void,
) -> Result<(), Fault> {
    if launch.smem > DEFAULT_DYNAMIC_SMEM {
        raise_dynamic_smem_cap(function, launch.smem)?;
    }

    // SAFETY: `CUlaunchAttribute` is a C struct of an enum id whose zero is
    // `CU_LAUNCH_ATTRIBUTE_IGNORE` and a plain-data union; all-zero bytes are
    // a valid value. Only the first `n` entries reach the driver.
    let mut attrs: [dr::CUlaunchAttribute; 3] = unsafe { core::mem::zeroed() };
    let mut n = 0usize;
    if launch.cooperative {
        attrs[n].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attrs[n].value.cooperative = 1;
        n += 1;
    }
    // Past the portable 8 a cluster is the function's to allow (B200 seats
    // 16 one-block SMs a cluster).
    if launch.cluster > 8 {
        // SAFETY: `function` is live by this fn's contract; the attribute
        // takes a plain flag.
        let code = unsafe {
            dr::cuFuncSetAttribute(
                function,
                dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(Fault::Device {
                call: "cuFuncSetAttribute",
                code: code as i32,
            });
        }
    }
    if launch.cluster > 1 {
        attrs[n].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attrs[n].value.clusterDim.x = launch.cluster;
        attrs[n].value.clusterDim.y = 1;
        attrs[n].value.clusterDim.z = 1;
        n += 1;
    }
    if launch.pdl {
        attrs[n].id =
            dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attrs[n].value.programmaticStreamSerializationAllowed = 1;
        n += 1;
    }

    let config = dr::CUlaunchConfig {
        gridDimX: launch.grid[0],
        gridDimY: launch.grid[1],
        gridDimZ: launch.grid[2],
        blockDimX: launch.block[0],
        blockDimY: launch.block[1],
        blockDimZ: launch.block[2],
        sharedMemBytes: launch.smem,
        hStream: stream.cast(),
        attrs: if n == 0 {
            std::ptr::null_mut()
        } else {
            attrs.as_mut_ptr()
        },
        numAttrs: n as core::ffi::c_uint,
    };

    // SAFETY: `config` and the `n` attributes it points at are locals that
    // outlive the call. `function`, `stream` and `slots` are as this fn's
    // contract states; `extra` is null, so the driver takes the arguments
    // from `slots` alone.
    let code = unsafe {
        dr::cuLaunchKernelEx(
            std::ptr::addr_of!(config),
            function,
            slots.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if code == dr::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call: "cuLaunchKernelEx",
            code: code as i32,
        })
    }
}

fn raise_dynamic_smem_cap(function: dr::CUfunction, bytes: u32) -> Result<(), Fault> {
    let mut device: dr::CUdevice = 0;

    // SAFETY: `device` is a local out-pointer.
    let code = unsafe { dr::cuCtxGetDevice(&raw mut device) };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Fault::Device {
            call: "cuCtxGetDevice",
            code: code as i32,
        });
    }
    let key = (device, function.addr());
    let mut granted = GRANTED.lock().expect("granted lock poisoned");
    if let Some((_, high_water)) = granted.iter().find(|(k, _)| *k == key)
        && bytes <= *high_water
    {
        return Ok(());
    }

    // SAFETY: `function` is the live handle `issue`'s caller vouched for;
    // the attribute takes a byte count, and one the device cannot seat
    // comes back as an error code.
    let code = unsafe {
        dr::cuFuncSetAttribute(
            function,
            dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(bytes).unwrap_or(i32::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Fault::Device {
            call: "cuFuncSetAttribute",
            code: code as i32,
        });
    }
    match granted.iter_mut().find(|(k, _)| *k == key) {
        Some((_, high_water)) => *high_water = bytes,
        None => granted.push((key, bytes)),
    }
    Ok(())
}

static GRANTED: std::sync::Mutex<Vec<((dr::CUdevice, usize), u32)>> =
    std::sync::Mutex::new(Vec::new());
