//! What the driver allocates for this process, as `VK_EXT_device_memory_report`
//! tells it: every device memory object it makes and frees, its own among
//! them (queues, descriptor pools, command buffers, shader code), heap by
//! heap. On a device without `VK_EXT_memory_budget` (the Adreno 840's
//! driver states none) it is the per-process reading of what the driver
//! holds past this process's reservations ([`super::ctx::Context::outside`]).
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use ash::vk;

/// Each live memory object's bytes and heap, by the driver's id.
static LIVE: Mutex<Option<HashMap<u64, (u64, u32)>>> = Mutex::new(None);

/// The driver's report of one allocation event: what is made is counted
/// live, what is freed leaves.
///
/// # Safety
///
/// Called by the driver with a valid callback-data pointer, on any thread.
pub(crate) unsafe extern "system" fn reported(
    data: *const vk::DeviceMemoryReportCallbackDataEXT<'_>,
    _user: *mut core::ffi::c_void,
) {
    // SAFETY: the driver passes a valid record for the call's duration.
    let Some(data) = (unsafe { data.as_ref() }) else {
        return;
    };
    let mut live = LIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let live = live.get_or_insert_with(HashMap::new);
    match data.ty {
        vk::DeviceMemoryReportEventTypeEXT::ALLOCATE | vk::DeviceMemoryReportEventTypeEXT::IMPORT => {
            live.insert(data.memory_object_id, (data.size, data.heap_index));
        }
        vk::DeviceMemoryReportEventTypeEXT::FREE | vk::DeviceMemoryReportEventTypeEXT::UNIMPORT => {
            live.remove(&data.memory_object_id);
        }
        _ => {}
    }
}

/// The bytes the driver holds live in every heap for this process: where
/// the device's heaps are all the host's memory (a phone's protected and
/// lazily-allocated heaps beside its device-local one), each counts.
#[must_use]
pub fn live_bytes_all() -> u64 {
    LIVE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .flatten()
        .map(|(_, (bytes, _))| *bytes)
        .sum()
}

/// The bytes the driver holds live in `heap` for this process, every
/// object it reported made there and not yet freed.
#[must_use]
pub fn live_bytes(heap: usize) -> u64 {
    LIVE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .flatten()
        .filter(|(_, (_, at))| *at as usize == heap)
        .map(|(_, (bytes, _))| *bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What is made counts until it is freed, heap by heap.
    #[test]
    fn the_report_counts_what_is_made_until_it_is_freed() {
        let event = |ty, id, size, heap| vk::DeviceMemoryReportCallbackDataEXT {
            ty,
            memory_object_id: id,
            size,
            heap_index: heap,
            ..Default::default()
        };
        let made = event(vk::DeviceMemoryReportEventTypeEXT::ALLOCATE, 901, 4096, 7);
        let other = event(vk::DeviceMemoryReportEventTypeEXT::ALLOCATE, 902, 100, 8);
        let freed = event(vk::DeviceMemoryReportEventTypeEXT::FREE, 901, 4096, 7);
        // SAFETY: each record is a live local.
        unsafe {
            reported(&made, core::ptr::null_mut());
            reported(&other, core::ptr::null_mut());
        }
        assert_eq!(live_bytes(7), 4096);
        assert_eq!(live_bytes(8), 100);
        // SAFETY: as above.
        unsafe { reported(&freed, core::ptr::null_mut()) };
        assert_eq!(live_bytes(7), 0);
    }
}
