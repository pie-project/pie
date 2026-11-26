//! What this process holds of the device its thread is bound to, as the
//! driver counts it: NVML on CUDA, KFD on ROCm. The runtime reports a card's
//! free bytes only through a context, so a reading of the card already holds
//! the context's own; the driver says whose they are.
#![allow(unsafe_code)]

use crate::error::Result;

/// What this process took of the bound device before a load of its own
/// resolved a ceiling: the context and what binding it puts down, which the
/// card counts held and no other process put there. `latch` keeps the
/// reading for the loads after the first, whose process holds an earlier
/// load's bytes too. `None`, said by name, where the driver does not answer
/// for this process: the card's own figure is the load's bound, every
/// byte held on it counted another's.
pub fn before_loads(latch: bool) -> Result<Option<u64>> {
    static TAKEN: std::sync::Mutex<Vec<(i32, u64)>> = std::sync::Mutex::new(Vec::new());
    let device = bound()?;
    let mut taken = TAKEN.lock().expect("the taken devices poisoned");
    if let Some(&(_, bytes)) = taken.iter().find(|(of, _)| *of == device) {
        return Ok(Some(bytes));
    }
    let bytes = match held(device) {
        Ok(bytes) => bytes,
        Err(why) => {
            tracing::info!(
                "what this process holds of device {device} is unknown: {why}; the card's \
                 free figure (cudaMemGetInfo) bounds the load, every byte held on it counted \
                 another's"
            );
            return Ok(None);
        }
    };
    if latch {
        taken.push((device, bytes));
    }
    Ok(Some(bytes))
}

/// The reader that says what this process holds of a device: the
/// management library on CUDA, the kernel driver's process list on ROCm.
pub const READER: &str = if cfg!(feature = "rocm") {
    "KFD (/sys/class/kfd/kfd/proc)"
} else if cfg!(windows) {
    "NVML (nvml.dll)"
} else {
    "NVML (libnvidia-ml.so.1)"
};

/// Why the driver does not say what this process holds of a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unread {
    /// The reader is not there: the library is not installed or does not
    /// start, or the kernel driver exposes no process list.
    Missing(String),
    /// The reader answers, but this process's pid is not among those it
    /// lists: it runs in another pid namespace than the driver's (a
    /// container), whose pids the driver lists.
    NotListed(u32),
    /// The reader lists this process without its usage: under WDDM the
    /// driver keeps it, not the management library.
    NotAvailable,
    /// The reader was asked and failed.
    Failed(String),
    /// This build has no reader.
    NoReader,
}

impl std::fmt::Display for Unread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unread::Missing(why) => write!(f, "{READER} is missing: {why}"),
            Unread::NotListed(pid) => write!(
                f,
                "{READER} lists no process {pid}: this process runs in another pid \
                 namespace than the driver's (a container), whose pids it lists"
            ),
            Unread::NotAvailable => write!(
                f,
                "{READER} lists this process without its usage (NVML_VALUE_NOT_AVAILABLE, \
                 as under WDDM)"
            ),
            Unread::Failed(why) => write!(f, "{READER} was asked and failed: {why}"),
            Unread::NoReader => write!(f, "this build reads no driver's process usage"),
        }
    }
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
        Err(crate::error::Fault::Runtimeless)
    }
}

/// The bytes of `device` the management library counts as this process's.
#[cfg(all(feature = "cuda", any(unix, windows)))]
fn held(device: i32) -> std::result::Result<u64, Unread> {
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

    /// The library's symbol `name`, null where it has none.
    #[cfg(unix)]
    unsafe fn open() -> Option<impl Fn(&core::ffi::CStr) -> Option<*mut c_void>> {
        // SAFETY: the name is NUL-terminated; the library stays loaded for
        // the process's life.
        let library = unsafe { libc::dlopen(c"libnvidia-ml.so.1".as_ptr(), libc::RTLD_NOW) };
        (!library.is_null()).then_some({
            move |name: &core::ffi::CStr| {
                // SAFETY: a live handle and a NUL-terminated name.
                let at = unsafe { libc::dlsym(library, name.as_ptr()) };
                (!at.is_null()).then_some(at)
            }
        })
    }
    #[cfg(windows)]
    unsafe fn open() -> Option<impl Fn(&core::ffi::CStr) -> Option<*mut c_void>> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
        // SAFETY: the name is NUL-terminated; the library stays loaded for
        // the process's life (the driver puts it in System32).
        let library = unsafe { LoadLibraryA(c"nvml.dll".as_ptr().cast()) };
        (!library.is_null()).then(|| {
            move |name: &core::ffi::CStr| {
                // SAFETY: a live module and a NUL-terminated name.
                unsafe { GetProcAddress(library, name.as_ptr().cast()) }.map(|at| at as *mut c_void)
            }
        })
    }

    static LIBRARY: std::sync::OnceLock<std::result::Result<(ByBus, Processes), &'static str>> =
        std::sync::OnceLock::new();
    let library = *LIBRARY.get_or_init(|| {
        // SAFETY: each symbol is null-checked before it is read as the
        // function the library documents under that name.
        unsafe {
            let symbol = open().ok_or("the library does not load")?;
            let init = symbol(c"nvmlInit_v2").ok_or("it has no nvmlInit_v2")?;
            let by_bus = symbol(c"nvmlDeviceGetHandleByPciBusId_v2")
                .ok_or("it has no nvmlDeviceGetHandleByPciBusId_v2")?;
            let processes = symbol(c"nvmlDeviceGetComputeRunningProcesses_v3")
                .ok_or("it has no nvmlDeviceGetComputeRunningProcesses_v3")?;
            let init: unsafe extern "C" fn() -> c_int = core::mem::transmute(init);
            if init() != 0 {
                return Err("nvmlInit_v2 refused");
            }
            Ok((
                core::mem::transmute::<*mut c_void, ByBus>(by_bus),
                core::mem::transmute::<*mut c_void, Processes>(processes),
            ))
        }
    });
    let (by_bus, processes) = library.map_err(|why| Unread::Missing(why.to_string()))?;

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
        if asked != rt::cudaError::cudaSuccess {
            return Err(Unread::Failed(format!(
                "device {device}'s PCI bus id: {asked:?}"
            )));
        }
        let code = by_bus(bus.as_ptr(), &raw mut handle);
        if code != 0 {
            return Err(Unread::Failed(format!(
                "nvmlDeviceGetHandleByPciBusId_v2 answered {code}"
            )));
        }
        let code = processes(handle, &raw mut count, listed.as_mut_ptr());
        if code != 0 {
            return Err(Unread::Failed(format!(
                "nvmlDeviceGetComputeRunningProcesses_v3 answered {code}"
            )));
        }
    }
    let own = std::process::id();
    let process = listed[..(count as usize).min(listed.len())]
        .iter()
        .find(|process| process.pid == own)
        .ok_or(Unread::NotListed(own))?;
    if process.used == UNKNOWN {
        return Err(Unread::NotAvailable);
    }
    Ok(process.used)
}

/// The bytes of the bound device KFD counts as this process's: every
/// allocation it holds there, the driver's own for its queues among them
/// ([`kernels_cuda::hip::kfd::process_vram`] reads the device the thread is
/// bound to, which [`bound`] named).
#[cfg(feature = "rocm")]
fn held(_device: i32) -> std::result::Result<u64, Unread> {
    use kernels_cuda::hip::kfd;
    if let Some(bytes) = kfd::process_vram() {
        return Ok(bytes);
    }
    let processes = std::path::Path::new(kfd::PROCESSES);
    let own = std::process::id();
    Err(if !processes.is_dir() {
        Unread::Missing(format!("{} is not there", kfd::PROCESSES))
    } else if !processes.join(own.to_string()).is_dir() {
        Unread::NotListed(own)
    } else {
        Unread::Failed("this process's vram count for the bound device is not readable".to_string())
    })
}

#[cfg(not(any(all(feature = "cuda", any(unix, windows)), feature = "rocm")))]
fn held(_device: i32) -> std::result::Result<u64, Unread> {
    Err(Unread::NoReader)
}
