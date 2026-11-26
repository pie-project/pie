//! What this process holds of the device its thread is bound to, as the
//! driver counts it: NVML on CUDA, KFD on ROCm. The runtime reports a card's
//! free bytes only through a context, so a reading of the card already holds
//! the context's own; the driver says whose they are.
#![allow(unsafe_code)]

use crate::error::{Fault, Result};

/// What this process took of the bound device before a load of its own
/// resolved a ceiling: the context and what binding it puts down, which the
/// card counts held and no other process put there. `latch` keeps the
/// reading for the loads after the first, whose process holds an earlier
/// load's bytes too. Refused where the driver does not answer for this
/// process (no management library installed, another pid namespace).
pub fn before_loads(latch: bool) -> Result<u64> {
    static TAKEN: std::sync::Mutex<Vec<(i32, u64)>> = std::sync::Mutex::new(Vec::new());
    let device = bound()?;
    let mut taken = TAKEN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, bytes)) = taken.iter().find(|(of, _)| *of == device) {
        return Ok(bytes);
    }
    let bytes = held(device).ok_or_else(|| {
        Fault::program(
            "device::own",
            format!(
                "the driver does not say what this process holds of device {device} \
                 (NVML on CUDA, KFD on ROCm: the library is missing, or this process \
                 runs in another pid namespace than the driver's)"
            ),
        )
    })?;
    if latch {
        taken.push((device, bytes));
    }
    Ok(bytes)
}

/// What this process holds of the device its thread is bound to now, as the
/// driver counts it ([`held`]): its context, its driver's queues and module
/// images, every allocation; `None` where the driver does not say.
#[must_use]
pub fn used_now() -> Option<u64> {
    held(bound().ok()?)
}

/// The device this thread is bound to.
fn bound() -> Result<i32> {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

        let mut device: i32 = 0;
        // SAFETY: a live local; the call only writes it.
        let asked = unsafe { rt::cudaGetDevice(&raw mut device) };
        crate::device::ctx::check("cudaGetDevice", asked)?;
        Ok(device)
    }
    #[cfg(not(feature = "device"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// The bytes of `device` the management library counts as this process's.
#[cfg(all(feature = "cuda", unix))]
fn held(device: i32) -> Option<u64> {
    use crate::device::rt;
    use core::ffi::{c_char, c_int, c_uint, c_void};

    /// `nvmlProcessInfo_t`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Process {
        pid: c_uint,
        used: u64,
        gpu_instance: c_uint,
        compute_instance: c_uint,
    }
    type ByBus = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int;
    type Processes = unsafe extern "C" fn(*mut c_void, *mut c_uint, *mut Process) -> c_int;
    /// `NVML_VALUE_NOT_AVAILABLE`.
    const UNKNOWN: u64 = u64::MAX;

    static LIBRARY: std::sync::OnceLock<Option<(ByBus, Processes)>> = std::sync::OnceLock::new();
    let (by_bus, processes) = (*LIBRARY.get_or_init(|| {
        // SAFETY: the names are NUL-terminated; each symbol is null-checked
        // before it is read as the function the library documents under
        // that name, and the library stays loaded for the process's life.
        unsafe {
            let library = libc::dlopen(c"libnvidia-ml.so.1".as_ptr(), libc::RTLD_NOW);
            if library.is_null() {
                return None;
            }
            let symbol = |name: &core::ffi::CStr| {
                let at = libc::dlsym(library, name.as_ptr());
                (!at.is_null()).then_some(at)
            };
            let init = symbol(c"nvmlInit_v2")?;
            let by_bus = symbol(c"nvmlDeviceGetHandleByPciBusId_v2")?;
            let processes = symbol(c"nvmlDeviceGetComputeRunningProcesses_v3")?;
            let init: unsafe extern "C" fn() -> c_int = core::mem::transmute(init);
            (init() == 0).then(|| {
                (
                    core::mem::transmute::<*mut c_void, ByBus>(by_bus),
                    core::mem::transmute::<*mut c_void, Processes>(processes),
                )
            })
        }
    }))?;

    let mut bus = [0 as c_char; 32];
    let mut handle: *mut c_void = core::ptr::null_mut();
    let mut listed = [Process {
        pid: 0,
        used: 0,
        gpu_instance: 0,
        compute_instance: 0,
    }; 256];
    let mut count = listed.len() as c_uint;
    // SAFETY: `bus` is written within its length and NUL-terminated by the
    // runtime; `handle` and `count` are live locals; `listed` holds `count`
    // entries of the library's layout.
    unsafe {
        let asked = rt::cudaDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as c_int, device);
        if asked != rt::cudaError::cudaSuccess
            || by_bus(bus.as_ptr(), &raw mut handle) != 0
            || processes(handle, &raw mut count, listed.as_mut_ptr()) != 0
        {
            return None;
        }
    }
    let own = std::process::id();
    listed[..(count as usize).min(listed.len())]
        .iter()
        .find(|process| process.pid == own && process.used != UNKNOWN)
        .map(|process| process.used)
}

/// The bytes of the bound device KFD counts as this process's: every
/// allocation it holds there, the driver's own for its queues among them
/// ([`kernels_cuda::hip::kfd::process_vram`] reads the device the thread is
/// bound to, which [`bound`] named).
#[cfg(feature = "rocm")]
fn held(_device: i32) -> Option<u64> {
    kernels_cuda::hip::kfd::process_vram()
}

#[cfg(not(any(all(feature = "cuda", unix), feature = "rocm")))]
fn held(_device: i32) -> Option<u64> {
    None
}
