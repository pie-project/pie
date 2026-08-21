use core::ffi::c_void;

use cudarc::driver::sys as dr;

use crate::jit::Launch;
use crate::jit::Error;

const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;

pub unsafe fn issue(
    function: dr::CUfunction,
    launch: Launch,
    slots: &mut [*mut c_void],
    stream: *mut c_void,
) -> Result<(), Error> {
    if launch.smem > DEFAULT_DYNAMIC_SMEM {
        raise_dynamic_smem_cap(function, launch.smem)?;
    }

    let mut attrs: [dr::CUlaunchAttribute; 1] = unsafe { core::mem::zeroed() };
    let mut n = 0usize;
    if launch.cooperative {
        attrs[0].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
        attrs[0].value.cooperative = 1;
        n = 1;
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
        attrs: if n == 0 { std::ptr::null_mut() } else { attrs.as_mut_ptr() },
        numAttrs: n as core::ffi::c_uint,
    };

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
        Err(Error::Driver { what: "cuLaunchKernelEx", code: code as i32, why: format!("{code:?}") })
    }
}

fn raise_dynamic_smem_cap(function: dr::CUfunction, bytes: u32) -> Result<(), Error> {
    let mut device: dr::CUdevice = 0;

    let code = unsafe { dr::cuCtxGetDevice(&raw mut device) };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Error::Driver {
            what: "cuCtxGetDevice",
            code: code as i32,
            why: format!("{code:?}"),
        });
    }
    let key = (device, function.addr());
    let mut granted = GRANTED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, high_water)) = granted.iter().find(|(k, _)| *k == key)
        && bytes <= *high_water
    {
        return Ok(());
    }

    let code = unsafe {
        dr::cuFuncSetAttribute(
            function,
            dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(bytes).unwrap_or(i32::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return Err(Error::Driver {
            what: "cuFuncSetAttribute",
            code: code as i32,
            why: format!("{code:?}"),
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
