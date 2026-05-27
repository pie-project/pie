//! POSIX shared-memory mapping (Linux + macOS + other unix).
//!
//! `ipc/linux.rs` and `ipc/macos.rs` both re-export the
//! `map_shmem_server` / `map_shmem_client` / `unmap_*` functions from
//! this module — the actual region-setup syscalls (`shm_open`,
//! `ftruncate`, `mmap`, `shm_unlink`) are identical between the two
//! platforms. The per-platform files keep only what's actually
//! different (the futex / `__ulock` park primitive).

#![cfg(unix)]

use std::ffi::CStr;

use anyhow::{Result, anyhow};

// `shm_open` lives in `librt` on Linux but in `libc` on macOS / BSDs.
// The link attr ensures the right thing on each.
#[cfg_attr(target_os = "linux", link(name = "rt"))]
unsafe extern "C" {
    fn shm_open(name: *const libc::c_char, oflag: libc::c_int, mode: libc::mode_t) -> libc::c_int;
}

/// Owned shmem region from the server's perspective. Held inside
/// `ShmemServerInner` and torn down on drop.
pub(super) struct ServerMapping {
    pub base: *mut u8,
    pub fd: libc::c_int,
    pub total_size: usize,
}

/// View of the shmem region from the client's perspective. The client
/// doesn't own the fd or the name (those belong to the server) — just
/// the mapping.
pub(super) struct ClientMapping {
    pub base: *mut u8,
    pub total_size: usize,
}

/// Create (or replace, via best-effort `shm_unlink` first) a shmem
/// region of `total_size` bytes named `name`. Returns the mmap'd base
/// pointer and the underlying fd; the caller writes the header and
/// holds the mapping in `ShmemServerInner` until drop.
pub(super) fn map_shmem_server(name: &CStr, total_size: usize) -> Result<ServerMapping> {
    // Replace any stale region with the same name (best-effort).
    unsafe { libc::shm_unlink(name.as_ptr()) };

    let fd = unsafe { shm_open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o600) };
    if fd < 0 {
        return Err(anyhow!(
            "shm_open({:?}) failed: {}",
            name,
            std::io::Error::last_os_error()
        ));
    }
    if unsafe { libc::ftruncate(fd, total_size as libc::off_t) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe {
            libc::close(fd);
            libc::shm_unlink(name.as_ptr());
        }
        return Err(anyhow!("ftruncate failed: {err}"));
    }
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            total_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        let err = std::io::Error::last_os_error();
        unsafe {
            libc::close(fd);
            libc::shm_unlink(name.as_ptr());
        }
        return Err(anyhow!("mmap failed: {err}"));
    }
    Ok(ServerMapping {
        base: base as *mut u8,
        fd,
        total_size,
    })
}

/// Attach to an existing shmem region created by some other process.
/// The size is discovered via `fstat`; the caller subsequently
/// validates magic / schema-version / hash against the region header.
pub(super) fn map_shmem_client(name: &CStr) -> Result<ClientMapping> {
    let fd = unsafe { shm_open(name.as_ptr(), libc::O_RDWR, 0o600) };
    if fd < 0 {
        return Err(anyhow!(
            "shm_open({:?}) for client failed: {}",
            name,
            std::io::Error::last_os_error()
        ));
    }
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(anyhow!("fstat shmem fd: {err}"));
    }
    let total_size = st.st_size as usize;
    if total_size == 0 {
        unsafe { libc::close(fd) };
        return Err(anyhow!("shmem region {name:?} reports size 0"));
    }
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            total_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    let close_err = unsafe { libc::close(fd) };
    debug_assert_eq!(close_err, 0, "close(shmem fd) failed");
    if base == libc::MAP_FAILED {
        return Err(anyhow!(
            "mmap shmem region {:?} failed: {}",
            name,
            std::io::Error::last_os_error()
        ));
    }
    Ok(ClientMapping {
        base: base as *mut u8,
        total_size,
    })
}

/// Drop a server mapping: unmap the region, close the fd, unlink the
/// name from `/dev/shm` so a future `map_shmem_server` with the same
/// name doesn't pick up our stale region.
///
/// # Safety
/// `mapping` must come from a successful `map_shmem_server` call and
/// must not have been freed already.
pub(super) unsafe fn unmap_shmem_server(mapping: &ServerMapping, name: &CStr) {
    unsafe {
        if !mapping.base.is_null() {
            libc::munmap(mapping.base as *mut libc::c_void, mapping.total_size);
        }
        if mapping.fd >= 0 {
            libc::close(mapping.fd);
        }
        libc::shm_unlink(name.as_ptr());
    }
}

/// Drop a client mapping (just unmaps; the region itself stays alive
/// for other clients and the server).
///
/// # Safety
/// `mapping` must come from a successful `map_shmem_client` call.
pub(super) unsafe fn unmap_shmem_client(mapping: &ClientMapping) {
    unsafe {
        if !mapping.base.is_null() {
            libc::munmap(mapping.base as *mut libc::c_void, mapping.total_size);
        }
    }
}
