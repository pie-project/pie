//! Generic-unix fallback platform module for unknown targets (FreeBSD,
//! NetBSD, etc.). Reuses the POSIX shmem helpers from [`super::posix`]
//! and degrades the cross-process park to a short sleep loop. Pie
//! doesn't actively support these targets — this exists so the build
//! doesn't break on tier-2 platforms. If you ship on one of them, add
//! a proper park primitive (e.g. FreeBSD's `_umtx_op`) in a dedicated
//! module.

#![cfg(all(
    unix,
    not(any(target_os = "linux", target_os = "android", target_os = "macos"))
))]

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub(super) use super::posix::{
    ClientMapping, ServerMapping, map_shmem_client, map_shmem_server, unmap_shmem_client,
    unmap_shmem_server,
};

/// Sleep-loop fallback — polls the atomic every 200 µs. ~kernel
/// timer slop of wake latency; CPU is essentially idle when parked.
pub(super) unsafe fn park(
    addr: *const AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
) -> bool {
    let deadline = timeout.map(|t| Instant::now() + t);
    let a = unsafe { &*addr };
    loop {
        if a.load(Ordering::Acquire) != expected {
            return true;
        }
        if let Some(d) = deadline
            && Instant::now() >= d
        {
            return false;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
}

/// No-op — the sleep loop in `park` picks up the value change on its
/// next iteration. Real platforms (linux/macos/windows) issue a kernel
/// wake here.
pub(super) unsafe fn wake_all(_addr: *const AtomicU32) {}
