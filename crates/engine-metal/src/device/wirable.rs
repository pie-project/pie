//! What the OS reports of its memory at the moment a load asks: what is
//! available (free, inactive, purgeable and speculative pages) and what is
//! wired. On unified memory the GPU's working set is wired from the same
//! pages everything else on the machine uses, and the GPU's wired limit is
//! the machine's, not a process's: what other processes hold wired counts
//! against the same limit Metal's recommended working set states.
#![allow(unsafe_code)]

/// The OS's memory now, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsMemory {
    /// Free, inactive, purgeable and speculative pages.
    pub available: u64,
    /// Wired pages, every process's and the kernel's.
    pub wired: u64,
}

/// The OS's memory now, `None` where it would not say.
#[cfg(target_vendor = "apple")]
#[must_use]
pub fn os_memory() -> Option<OsMemory> {
    use std::mem::{MaybeUninit, size_of};
    // SAFETY: no arguments; answers this task's host port.
    #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
    let host = unsafe { libc::mach_host_self() };
    let mut stats = MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = (size_of::<libc::vm_statistics64>() / size_of::<libc::integer_t>()) as u32;
    // SAFETY: `count` states `stats`' size in the kernel's units; the call
    // writes no more.
    let status = unsafe {
        #[allow(deprecated, reason = "the mach crate is not a dependency for one call")]
        libc::host_statistics64(host, libc::HOST_VM_INFO64, stats.as_mut_ptr().cast(), &raw mut count)
    };
    if status != 0 {
        return None;
    }
    // SAFETY: the call succeeded; counters it did not reach are zero.
    let stats = unsafe { stats.assume_init() };
    // SAFETY: no arguments.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    let available = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.purgeable_count)
        + u64::from(stats.speculative_count);
    Some(OsMemory {
        available: available.saturating_mul(page),
        wired: u64::from(stats.wire_count).saturating_mul(page),
    })
}

#[cfg(not(target_vendor = "apple"))]
#[must_use]
pub fn os_memory() -> Option<OsMemory> {
    None
}
