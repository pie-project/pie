#![allow(unsafe_code)]
use core::ffi::c_void;

use crate::hip::sys;
use crate::jit::{Fault, Launch};

/// AMD grants a kernel the whole of a CU's LDS without the per-function
/// opt-in CUDA needs above 48 KiB, so the only check is that the launch
/// asks for no more than the device has.
///
/// # Safety
///
/// `function` must be a kernel of a loaded module; `slots` must hold one
/// pointer per kernel parameter, each to a live value of that parameter's
/// type; `stream` must be a live stream on the current device.
pub(crate) unsafe fn issue(
    function: sys::Function,
    launch: Launch,
    slots: &mut [*mut c_void],
    stream: *mut c_void,
) -> Result<(), Fault> {
    let api = sys::api().map_err(|_| Fault::Device {
        call: "hipModuleLaunchKernel",
        code: -1,
    })?;
    if let Some(cap) = crate::jit::device::max_shared_memory_per_block_optin()
        && launch.smem > cap
    {
        return Err(Fault::Device {
            call: "hipModuleLaunchKernel",
            // hipErrorInvalidValue: the launch would be refused for its LDS.
            code: 1,
        });
    }

    let [gx, gy, gz] = launch.grid;
    let [bx, by, bz] = launch.block;
    let (call, code) = if launch.cooperative {
        // SAFETY: the caller guarantees `function`, `slots` and `stream` (this
        // fn's contract); HIP copies the arguments out of `slots` before
        // returning.
        ("hipModuleLaunchCooperativeKernel", unsafe {
            (api.hip.hipModuleLaunchCooperativeKernel)(
                function,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                launch.smem,
                stream,
                slots.as_mut_ptr(),
            )
        })
    } else {
        // SAFETY: the caller guarantees `function`, `slots` and `stream` (this
        // fn's contract); HIP copies the arguments out of `slots` before
        // returning, and `extra` is null.
        ("hipModuleLaunchKernel", unsafe {
            (api.hip.hipModuleLaunchKernel)(
                function,
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                launch.smem,
                stream,
                slots.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        })
    };
    if code == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device { call, code })
    }
}
