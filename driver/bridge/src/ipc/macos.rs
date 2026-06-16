//! macOS platform module: cross-process park via `__ulock_wait` /
//! `__ulock_wake` with `UL_COMPARE_AND_WAIT_SHARED` (operation = 3).
//!
//! `__ulock_*` is private Apple SPI but it's been ABI-stable since
//! macOS 10.13 — abseil, folly, and libdispatch all rely on it for
//! cross-process synchronization. The shared variant (operation = 3)
//! is the one that crosses process boundaries; `UL_COMPARE_AND_WAIT`
//! (operation = 1, what libc++ uses) is process-local and would not
//! work for our shmem case.
//!
//! POSIX shmem region setup is re-exported from
//! [`super::posix`] — Linux and macOS share the same `shm_open` /
//! `mmap` syscalls.

#![cfg(target_os = "macos")]

use std::sync::atomic::AtomicU32;
use std::time::Duration;

pub(super) use super::posix::{
    ClientMapping, ServerMapping, map_shmem_client, map_shmem_server, unmap_shmem_client,
    unmap_shmem_server,
};

// From xnu/bsd/sys/ulock.h. Stable since macOS 10.13.
const UL_COMPARE_AND_WAIT_SHARED: u32 = 3;
const ULF_WAKE_ALL: u32 = 0x0000_0100;
const ETIMEDOUT: i32 = 60;

unsafe extern "C" {
    fn __ulock_wait(
        operation: u32,
        addr: *const ::core::ffi::c_void,
        value: u64,
        timeout_us: u32,
    ) -> i32;
    fn __ulock_wake(operation: u32, addr: *const ::core::ffi::c_void, value: u64) -> i32;
}

/// See [`super::park`] for semantics.
///
/// # Safety
/// `addr` must point to a valid `AtomicU32` inside shared memory.
pub(super) unsafe fn park(
    addr: *const AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
) -> bool {
    // `timeout_us == 0` in `__ulock_wait` means "no timeout".
    let us = match timeout {
        Some(d) => {
            let raw = d.as_micros();
            if raw == 0 {
                return false; // 0-duration → immediate timeout
            }
            raw.min(u32::MAX as u128) as u32
        }
        None => 0,
    };
    let r = unsafe {
        __ulock_wait(
            UL_COMPARE_AND_WAIT_SHARED,
            addr as *const _,
            expected as u64,
            us,
        )
    };
    if r >= 0 {
        return true;
    }
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    errno != ETIMEDOUT
}

/// See [`super::wake_all`] for semantics.
///
/// # Safety
/// `addr` must point to memory safe for the `__ulock_wake` syscall.
pub(super) unsafe fn wake_all(addr: *const AtomicU32) {
    unsafe {
        __ulock_wake(
            UL_COMPARE_AND_WAIT_SHARED | ULF_WAKE_ALL,
            addr as *const _,
            0,
        );
    }
}
