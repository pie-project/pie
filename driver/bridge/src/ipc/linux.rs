//! Linux/Android platform module: cross-process park via `futex(2)`
//! (no `FUTEX_PRIVATE_FLAG` — kernel keys by physical page so two
//! processes mmap'd to the same shmem region see each other's wake).
//!
//! POSIX shmem region setup is re-exported from
//! [`super::posix`] — Linux and macOS share the same `shm_open` /
//! `mmap` syscalls.

#![cfg(any(target_os = "linux", target_os = "android"))]

use std::sync::atomic::AtomicU32;
use std::time::Duration;

// Re-export the POSIX shmem helpers so the parent `ipc` module sees a
// uniform `platform::*` surface across all targets.
pub(super) use super::posix::{
    ClientMapping, ServerMapping, map_shmem_client, map_shmem_server, unmap_shmem_client,
    unmap_shmem_server,
};

/// Park the current thread on `*addr` until another process changes
/// the value and calls `wake_all`, the timeout fires, or the syscall
/// returns spuriously (signal / `EAGAIN`).
///
/// Returns `true` if woken (or spurious), `false` on timeout. Caller
/// MUST re-check the underlying condition on return.
///
/// # Safety
/// `addr` must point to a valid `AtomicU32` inside shared memory
/// mapped by every process that may call `wake_all` on the same
/// address.
pub(super) unsafe fn park(
    addr: *const AtomicU32,
    expected: u32,
    timeout: Option<Duration>,
) -> bool {
    let ts = timeout.map(|d| libc::timespec {
        tv_sec: d.as_secs() as libc::time_t,
        tv_nsec: d.subsec_nanos() as libc::c_long,
    });
    let ts_ptr: *const libc::timespec = match &ts {
        Some(t) => t as *const _,
        None => ::core::ptr::null(),
    };
    // FUTEX_WAIT (no PRIVATE_FLAG) → kernel keys waiters by physical
    // page, so cross-process wakes on the same shmem region are
    // delivered.
    let r = unsafe { libc::syscall(libc::SYS_futex, addr, libc::FUTEX_WAIT, expected, ts_ptr) };
    if r == 0 {
        return true;
    }
    // -1 + errno: ETIMEDOUT → timeout; EAGAIN (value mismatch),
    // EINTR (signal) → treat as spurious wake (caller re-checks).
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    errno != libc::ETIMEDOUT
}

/// Wake every thread parked on `addr` (across processes).
///
/// # Safety
/// `addr` must point to memory safe to dereference for the futex
/// syscall — typically an `AtomicU32` inside shmem.
pub(super) unsafe fn wake_all(addr: *const AtomicU32) {
    unsafe {
        libc::syscall(libc::SYS_futex, addr, libc::FUTEX_WAKE, i32::MAX);
    }
}
