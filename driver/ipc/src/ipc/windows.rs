//! Windows platform module: cross-process park via `WaitOnAddress` /
//! `WakeByAddressAll`, plus shmem regions backed by file-mapping
//! objects (`CreateFileMappingA` / `OpenFileMappingA` /
//! `MapViewOfFile`).
//!
//! Per MSDN: "WaitOnAddress can be used for inter-process
//! synchronization, provided that both processes have a shared view
//! of the same memory."

#![cfg(windows)]

use std::ffi::CStr;
use std::sync::atomic::AtomicU32;
use std::time::Duration;

use anyhow::{Result, anyhow};
use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingA, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
    OpenFileMappingA, PAGE_READWRITE, UnmapViewOfFile,
};
use windows_sys::Win32::System::Threading::{WaitOnAddress, WakeByAddressAll};

const INVALID_HANDLE_VALUE: *mut libc::c_void = !0usize as *mut libc::c_void;
const INFINITE: u32 = 0xFFFF_FFFF;

/// Owned shmem region from the server's perspective.
pub(super) struct ServerMapping {
    pub base: *mut u8,
    pub mapping: *mut libc::c_void,
    pub total_size: usize,
}

/// View of the shmem region from the client's perspective.
pub(super) struct ClientMapping {
    pub base: *mut u8,
    pub total_size: usize,
    pub mapping: *mut libc::c_void,
}

/// Create a file-mapping object backed by the system paging file and
/// map a view of it. `name` is a global object name (e.g. `pie_shmem_g0`).
pub(super) fn map_shmem_server(name: &CStr, total_size: usize) -> Result<ServerMapping> {
    let size_low = (total_size & 0xFFFF_FFFF) as u32;
    let size_high = ((total_size as u64) >> 32) as u32;
    let mapping = unsafe {
        CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            ::core::ptr::null_mut(),
            PAGE_READWRITE,
            size_high,
            size_low,
            name.as_ptr() as *const u8,
        )
    };
    if mapping.is_null() {
        return Err(anyhow!(
            "CreateFileMappingA({:?}) failed: {}",
            name,
            std::io::Error::last_os_error()
        ));
    }
    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size) };
    if view.Value.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { CloseHandle(mapping) };
        return Err(anyhow!("MapViewOfFile failed: {err}"));
    }
    Ok(ServerMapping {
        base: view.Value as *mut u8,
        mapping,
        total_size,
    })
}

/// Open an existing file-mapping object and map a view. The total
/// size is discovered from the server's view via `VirtualQuery` on
/// the mapped pointer — same approach as the previous monolithic
/// `ipc.rs` (see commit history).
pub(super) fn map_shmem_client(name: &CStr) -> Result<ClientMapping> {
    let mapping = unsafe { OpenFileMappingA(FILE_MAP_ALL_ACCESS, 0, name.as_ptr() as *const u8) };
    if mapping.is_null() {
        return Err(anyhow!(
            "OpenFileMappingA({:?}) failed: {}",
            name,
            std::io::Error::last_os_error()
        ));
    }
    // Map at offset 0 with size 0 → maps the full section.
    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0) };
    if view.Value.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { CloseHandle(mapping) };
        return Err(anyhow!("MapViewOfFile (client) failed: {err}"));
    }
    // Query the actual mapped size.
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe {
        VirtualQuery(
            view.Value,
            &mut info,
            std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if n == 0 {
        let err = std::io::Error::last_os_error();
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(mapping);
        }
        return Err(anyhow!("VirtualQuery failed: {err}"));
    }
    Ok(ClientMapping {
        base: view.Value as *mut u8,
        total_size: info.RegionSize as usize,
        mapping,
    })
}

/// Drop a server mapping.
///
/// # Safety
/// `mapping` must come from a successful `map_shmem_server` call.
pub(super) unsafe fn unmap_shmem_server(mapping: &ServerMapping, _name: &CStr) {
    unsafe {
        if !mapping.base.is_null() {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: mapping.base.cast(),
            });
        }
        if !mapping.mapping.is_null() {
            CloseHandle(mapping.mapping);
        }
    }
}

/// Drop a client mapping.
///
/// # Safety
/// `mapping` must come from a successful `map_shmem_client` call.
pub(super) unsafe fn unmap_shmem_client(mapping: &ClientMapping) {
    unsafe {
        if !mapping.base.is_null() {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: mapping.base.cast(),
            });
        }
        if !mapping.mapping.is_null() {
            CloseHandle(mapping.mapping);
        }
    }
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
    let dw = match timeout {
        // Saturate to (INFINITE - 1) since INFINITE means "no timeout".
        Some(d) => {
            let ms = d.as_millis();
            if ms >= (INFINITE - 1) as u128 {
                INFINITE - 1
            } else {
                ms as u32
            }
        }
        None => INFINITE,
    };
    let expected_ptr: *const u32 = &expected;
    // WaitOnAddress returns nonzero on success (woken), 0 on timeout.
    let r = unsafe { WaitOnAddress(addr as *const _, expected_ptr as *const _, 4, dw) };
    r != 0
}

/// See [`super::wake_all`] for semantics.
///
/// # Safety
/// `addr` must point to memory safe for the `WakeByAddressAll` call.
pub(super) unsafe fn wake_all(addr: *const AtomicU32) {
    unsafe { WakeByAddressAll(addr as *const _) };
}
