//! The bound device's free and total bytes, the figures a load sizes by.
//!
//! A discrete card's are the runtime's (`cudaMemGetInfo`). An integrated
//! device (`cudaDevAttrIntegrated`) on Windows is the firmware's carve-out of
//! system memory, and the runtime's figures add to it the system memory the
//! display driver lends it: memory the operating system, the host tier and
//! every other process draw on too, slower to the device, and charged to the
//! system's commit like every device byte. There a load's capacity is the
//! carve-out the device does not hold yet, as the driver counts it, within
//! what Windows can still commit: measured at the load, the lent memory not
//! counted. A budget the operator states (`[engine.memory] budget_device`)
//! is theirs to size the load by, lent memory and all: under one the
//! runtime's figures stand, the budget capping them.
#![allow(unsafe_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::Result;

/// The terms an integrated device's capacity is measured from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Integrated {
    /// The carve-out: the device's own memory, as the driver counts it.
    pub dedicated: u64,
    /// What the device holds of it now, every process's.
    pub held: u64,
    /// The bytes Windows can still commit: the device's draw on them, one
    /// for one.
    pub commit_available: u64,
}

impl Integrated {
    /// What a load may still take: the carve-out not held, within what
    /// Windows can commit.
    #[must_use]
    pub fn free(&self) -> u64 {
        self.dedicated
            .saturating_sub(self.held)
            .min(self.commit_available)
    }
}

/// Whether the operator stated a device budget: an integrated device's
/// figures are then the runtime's.
static STATED: AtomicBool = AtomicBool::new(false);

/// Records whether the load's allowance states a device budget.
pub fn stated(budget: Option<u64>) {
    STATED.store(budget.is_some(), Ordering::Relaxed);
}

/// What bounds the load's room on the bound device, as a refusal names it:
/// the operator's budget where one is stated, else an integrated device's
/// carve-out within Windows' commit, else the whole device.
#[must_use]
pub fn bound() -> &'static str {
    if STATED.load(Ordering::Relaxed) {
        "under `[engine.memory] budget_device`"
    } else if matches!(integrated(), Ok(Some(_))) {
        "within the integrated device's carve-out and Windows' commit (state \
         `[engine.memory] budget_device` to size by the system memory the driver lends it)"
    } else {
        "on the whole device"
    }
}

/// The bound device's free and total bytes now.
pub fn memory() -> Result<(u64, u64)> {
    #[cfg(feature = "device")]
    {
        #[cfg(all(windows, feature = "cuda"))]
        if !STATED.load(Ordering::Relaxed)
            && let Some(terms) = integrated()?
        {
            return Ok((terms.free(), terms.dedicated));
        }
        runtime()
    }
    #[cfg(not(feature = "device"))]
    {
        Err(crate::error::Fault::Runtimeless)
    }
}

/// The runtime's free and total bytes of the bound device.
#[cfg(feature = "device")]
fn runtime() -> Result<(u64, u64)> {
    use crate::device::rt;

    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    crate::device::ctx::check("cudaMemGetInfo", asked)?;
    Ok((free as u64, total as u64))
}

/// Whose memory device `ordinal`'s is: the host's where the runtime states
/// it integrated (`cudaDevAttrIntegrated`: the Spark, Jetson-class parts, an
/// APU under ROCm), else its own. Where it is the host's, the load states no
/// host tier ([`engine::DeviceMemory::host_tier`]).
pub fn whose_memory(ordinal: i32) -> Result<engine::DeviceMemory> {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

        let mut flag = 0i32;
        // SAFETY: a live out-parameter; the call only writes it.
        let asked = unsafe {
            rt::cudaDeviceGetAttribute(
                &raw mut flag,
                rt::cudaDeviceAttr::cudaDevAttrIntegrated,
                ordinal,
            )
        };
        crate::device::ctx::check("cudaDeviceGetAttribute(Integrated)", asked)?;
        Ok(integrated_memory(flag))
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = ordinal;
        Err(crate::error::Fault::Runtimeless)
    }
}

/// Whose memory a device's is, of its `cudaDevAttrIntegrated` flag.
#[must_use]
pub fn integrated_memory(flag: i32) -> engine::DeviceMemory {
    if flag == 0 {
        engine::DeviceMemory::Own
    } else {
        engine::DeviceMemory::Hosts
    }
}

/// The bound device's capacity terms where it is an integrated device on
/// Windows; `None` for any other device.
#[cfg(all(feature = "device", windows, feature = "cuda"))]
pub fn integrated() -> Result<Option<Integrated>> {
    let device = crate::device::ctx::current()?;
    if !whose_memory(device)?.unified() {
        return Ok(None);
    }
    let (dedicated, held) = nvml::memory(device)?;
    Ok(Some(Integrated {
        dedicated,
        held,
        commit_available: commit_available()?,
    }))
}

#[cfg(not(all(feature = "device", windows, feature = "cuda")))]
pub fn integrated() -> Result<Option<Integrated>> {
    Ok(None)
}

/// The bytes Windows can still commit (`ullAvailPageFile`).
#[cfg(all(feature = "device", windows, feature = "cuda"))]
fn commit_available() -> Result<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    // SAFETY: a live struct whose length field is set, as the call requires.
    if unsafe { GlobalMemoryStatusEx(&raw mut status) } == 0 {
        return Err(crate::error::Fault::Card(format!(
            "an integrated device's capacity cannot be measured: Windows did not report its \
             available commit ({})",
            std::io::Error::last_os_error()
        )));
    }
    Ok(status.ullAvailPageFile)
}

/// The driver's count of the device's own memory, through its management
/// library (`nvml.dll`, installed with the display driver).
#[cfg(all(feature = "device", windows, feature = "cuda"))]
mod nvml {
    use core::ffi::{c_char, c_int, c_void};

    use crate::error::{Fault, Result};

    /// `nvmlMemory_t`.
    #[repr(C)]
    struct Memory {
        total: u64,
        free: u64,
        used: u64,
    }

    type Init = unsafe extern "C" fn() -> c_int;
    type ByBus = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> c_int;
    type Info = unsafe extern "C" fn(*mut c_void, *mut Memory) -> c_int;

    fn refused(why: &str) -> Fault {
        Fault::Card(format!(
            "an integrated device's capacity cannot be measured: the driver's management \
             library {why}"
        ))
    }

    fn library() -> Result<(ByBus, Info)> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

        static LIBRARY: std::sync::OnceLock<Option<(ByBus, Info)>> = std::sync::OnceLock::new();
        LIBRARY
            .get_or_init(|| {
                // SAFETY: the names are NUL-terminated; each symbol is
                // null-checked before it is read as the function the library
                // documents under that name, and the library stays loaded for
                // the process's life.
                unsafe {
                    let library = LoadLibraryA(c"nvml.dll".as_ptr().cast());
                    if library.is_null() {
                        return None;
                    }
                    let init = GetProcAddress(library, c"nvmlInit_v2".as_ptr().cast())?;
                    let by_bus = GetProcAddress(
                        library,
                        c"nvmlDeviceGetHandleByPciBusId_v2".as_ptr().cast(),
                    )?;
                    let info = GetProcAddress(library, c"nvmlDeviceGetMemoryInfo".as_ptr().cast())?;
                    let init: Init = core::mem::transmute(init);
                    (init() == 0).then(|| {
                        (
                            core::mem::transmute::<_, ByBus>(by_bus),
                            core::mem::transmute::<_, Info>(info),
                        )
                    })
                }
            })
            .ok_or_else(|| refused("(nvml.dll) did not load or initialize"))
    }

    /// The device's own bytes and the bytes of them held, every process's.
    pub(super) fn memory(device: i32) -> Result<(u64, u64)> {
        use crate::device::rt;

        let (by_bus, info) = library()?;
        let mut bus = [0 as c_char; 32];
        let mut handle: *mut c_void = core::ptr::null_mut();
        let mut memory = Memory {
            total: 0,
            free: 0,
            used: 0,
        };
        // SAFETY: `bus` is written within its length and NUL-terminated by
        // the runtime; `handle` and `memory` are live locals of the
        // library's layout.
        unsafe {
            let asked = rt::cudaDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as c_int, device);
            crate::device::ctx::check("cudaDeviceGetPCIBusId", asked)?;
            if by_bus(bus.as_ptr(), &raw mut handle) != 0 {
                return Err(refused("does not know the device by its bus id"));
            }
            if info(handle, &raw mut memory) != 0 {
                return Err(refused("did not report the device's memory"));
            }
        }
        Ok((memory.total, memory.used))
    }
}

#[cfg(test)]
mod tests {
    use super::{Integrated, integrated_memory};

    const MIB: u64 = 1 << 20;

    /// The Spark (sm_121) and a Jetson state `cudaDevAttrIntegrated` = 1:
    /// no host tier there, and a stated one is refused by name. A discrete
    /// card (0) seats what is stated, or its default, as before.
    #[test]
    fn an_integrated_device_states_no_host_tier_and_refuses_a_stated_one() {
        let integrated = integrated_memory(1);
        assert_eq!(integrated, engine::DeviceMemory::Hosts);
        assert_eq!(integrated.host_tier(None), Ok(engine::HostTier::Absent));
        let refused = integrated.host_tier(Some(32 << 30)).unwrap_err();
        assert!(refused.contains("`[engine.memory] budget_host`"), "{refused}");
        let discrete = integrated_memory(0);
        assert_eq!(discrete, engine::DeviceMemory::Own);
        assert_eq!(discrete.host_tier(None), Ok(engine::HostTier::Seated(None)));
        assert_eq!(
            discrete.host_tier(Some(32 << 30)),
            Ok(engine::HostTier::Seated(Some(32 << 30)))
        );
    }

    #[test]
    fn an_integrated_device_takes_its_carve_out_less_what_is_held() {
        // An RTX Spark at load: 64000 MiB carved out, 953 held, 90304 MiB
        // of commit left.
        let terms = Integrated {
            dedicated: 64000 * MIB,
            held: 953 * MIB,
            commit_available: 90304 * MIB,
        };
        assert_eq!(terms.free(), 63047 * MIB);
    }

    #[test]
    fn windows_commit_bounds_an_integrated_device() {
        let terms = Integrated {
            dedicated: 64000 * MIB,
            held: 953 * MIB,
            commit_available: 20000 * MIB,
        };
        assert_eq!(
            terms.free(),
            20000 * MIB,
            "every device byte is charged to commit"
        );
    }

    #[test]
    fn a_device_held_past_its_carve_out_has_nothing_free() {
        let terms = Integrated {
            dedicated: 64000 * MIB,
            held: 70000 * MIB,
            commit_available: 90304 * MIB,
        };
        assert_eq!(terms.free(), 0);
    }
}
