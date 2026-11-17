//! What the OS can wire for this process now: the memory it reports
//! available (free, inactive, purgeable and speculative pages) at the
//! moment a load asks. On unified memory the GPU's working set is wired
//! from the same pages everything else on the machine uses, so a load's
//! device capacity is the lesser of what Metal recommends and this.
#![allow(unsafe_code)]

/// The bytes the OS reports available now, `None` where it would not say.
#[cfg(target_vendor = "apple")]
#[must_use]
pub fn os_available() -> Option<u64> {
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
    let pages = u64::from(stats.free_count)
        + u64::from(stats.inactive_count)
        + u64::from(stats.purgeable_count)
        + u64::from(stats.speculative_count);
    Some(pages.saturating_mul(page))
}

#[cfg(not(target_vendor = "apple"))]
#[must_use]
pub fn os_available() -> Option<u64> {
    None
}
