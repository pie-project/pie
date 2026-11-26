//! What the driver allocates for this process, as `VK_EXT_device_memory_report`
//! tells it: every device memory object it makes and frees, its own among
//! them (queues, descriptor pools, command buffers, shader code), heap by
//! heap. On a device without `VK_EXT_memory_budget` (the Adreno 840's
//! driver states none) it is the per-process reading of what the driver
//! holds past this process's reservations ([`super::ctx::Context::outside`]).
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;

/// Each device's live memory objects, their bytes and heap by the driver's
/// id, under the key its creation gave the driver to report with
/// ([`new_key`]): one device's objects are never another's.
type Objects = HashMap<u64, (u64, u32)>;
static LIVE: Mutex<Option<HashMap<usize, Objects>>> = Mutex::new(None);

/// A key for a device about to be made: the driver hands it back with every
/// report of that device's ([`reported`]'s user pointer).
pub(crate) fn new_key() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The device of `key` is gone: its count goes with it.
pub(crate) fn forget(key: usize) {
    if let Some(live) = LIVE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_mut()
    {
        live.remove(&key);
    }
}

/// The driver's report of one allocation event of the device whose key is
/// `user` ([`new_key`]): what is made is counted live, what is freed
/// leaves.
///
/// # Safety
///
/// Called by the driver with a valid callback-data pointer, on any thread.
pub(crate) unsafe extern "system" fn reported(
    data: *const vk::DeviceMemoryReportCallbackDataEXT<'_>,
    user: *mut core::ffi::c_void,
) {
    // SAFETY: the driver passes a valid record for the call's duration.
    let Some(data) = (unsafe { data.as_ref() }) else {
        return;
    };
    let mut live = LIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let live = live
        .get_or_insert_with(HashMap::new)
        .entry(user as usize)
        .or_default();
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

/// The bytes the driver holds live in every heap for the device of `key`:
/// where the device's heaps are all the host's memory (a phone's protected
/// and lazily-allocated heaps beside its device-local one), each counts.
#[must_use]
pub fn live_bytes_all(key: usize) -> u64 {
    held(key, |_| true)
}

/// The bytes the driver holds live in `heap` for the device of `key`,
/// every object it reported made there and not yet freed.
#[must_use]
pub fn live_bytes(key: usize, heap: usize) -> u64 {
    held(key, |at| at as usize == heap)
}

fn held(key: usize, heap: impl Fn(u32) -> bool) -> u64 {
    LIVE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .and_then(|live| live.get(&key))
        .into_iter()
        .flatten()
        .filter(|(_, (_, at))| heap(*at))
        .map(|(_, (bytes, _))| *bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What is made counts until it is freed, heap by heap, each device's
    /// apart: a report under one device's key is never another's.
    #[test]
    fn the_report_counts_what_is_made_until_it_is_freed() {
        let (one, two) = (new_key(), new_key());
        let as_user = |key: usize| key as *mut core::ffi::c_void;
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
            reported(&made, as_user(one));
            reported(&other, as_user(one));
            reported(&made, as_user(two));
        }
        assert_eq!(live_bytes(one, 7), 4096);
        assert_eq!(live_bytes(one, 8), 100);
        assert_eq!(live_bytes_all(one), 4196);
        assert_eq!(live_bytes_all(two), 4096);
        // SAFETY: as above.
        unsafe { reported(&freed, as_user(one)) };
        assert_eq!(live_bytes(one, 7), 0);
        assert_eq!(live_bytes(two, 7), 4096);
        forget(two);
        assert_eq!(live_bytes_all(two), 0);
    }
}
