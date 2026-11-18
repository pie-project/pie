//! What the device states it has, read once at load through the adapter's
//! native handles: wgpu lays its pool whole at load, so it is never read
//! again while serving. `[engine.memory] budget_device` overrides it on
//! every backend; a backend that states none is refused by name unless it
//! is stated.

// The native handles are read through wgpu-hal's `as_hal`.
#![allow(unsafe_code)]

use std::fmt;

use engine_gpu::error::{Fault, Result};

/// Where a load's device memory was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `[engine.memory] budget_device`.
    Config,
    /// `VK_EXT_memory_budget`'s `heapBudget` of this device-local heap.
    VulkanBudget(usize),
    /// The size of this device-local heap, without `VK_EXT_memory_budget`.
    VulkanHeap(usize),
    /// `IDXGIAdapter3::QueryVideoMemoryInfo`'s local segment `Budget`.
    DxgiLocal,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Config => write!(f, "config budget_device"),
            Source::VulkanBudget(heap) => write!(f, "VK_EXT_memory_budget heap {heap}"),
            Source::VulkanHeap(heap) => write!(f, "Vulkan heap {heap} size"),
            Source::DxgiLocal => write!(f, "DXGI QueryVideoMemoryInfo local"),
        }
    }
}

/// The bytes a load's device has, and where they were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    pub bytes: u64,
    pub source: Source,
}

impl fmt::Display for Reading {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gib = self.bytes as f64 / f64::from(1u32 << 30);
        write!(f, "device memory: {gib:.1} GiB from {}", self.source)
    }
}

/// Of Vulkan's memory heaps, `(size, device-local)` each, the largest
/// device-local one, which device reservations take: its budget where
/// `VK_EXT_memory_budget` states `budgets`, else its size.
#[must_use]
pub fn vulkan_heaps(heaps: &[(u64, bool)], budgets: Option<&[u64]>) -> Option<Reading> {
    let (heap, size) = heaps
        .iter()
        .enumerate()
        .filter(|&(_, &(_, local))| local)
        .map(|(at, &(size, _))| (at, size))
        .max_by_key(|&(_, size)| size)?;
    Some(match budgets {
        Some(budgets) => Reading {
            bytes: budgets[heap],
            source: Source::VulkanBudget(heap),
        },
        None => Reading {
            bytes: size,
            source: Source::VulkanHeap(heap),
        },
    })
}

/// What a load on `adapter` takes: `budget_device` where stated, else the
/// adapter's own reading, refused by name where its backend states none.
pub fn at_load(adapter: &wgpu::Adapter, allowance: engine::Allowance) -> Result<Reading> {
    if let Some(bytes) = allowance.budget {
        return Ok(Reading {
            bytes,
            source: Source::Config,
        });
    }
    read(adapter).map_err(refused)
}

fn refused(why: String) -> Fault {
    Fault::Residency(format!(
        "wgpu: {why}; state `[engine.memory] budget_device` for what the load may take"
    ))
}

fn read(adapter: &wgpu::Adapter) -> std::result::Result<Reading, String> {
    let backend = adapter.get_info().backend;
    match backend {
        #[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
        wgpu::Backend::Vulkan => vulkan(adapter),
        #[cfg(windows)]
        wgpu::Backend::Dx12 => dx12(adapter),
        other => Err(format!(
            "the {} backend reports no memory size",
            other.to_str()
        )),
    }
}

#[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
fn vulkan(adapter: &wgpu::Adapter) -> std::result::Result<Reading, String> {
    use ash::vk;
    // SAFETY: the native adapter is only read, and the guard is dropped
    // before `adapter` is.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("the Vulkan adapter has no native handle")?;
    let shared = hal.shared_instance();
    let instance = shared.raw_instance();
    let physical = hal.raw_physical_device();
    let caps = hal.physical_device_capabilities();
    // The budget is chained onto vkGetPhysicalDeviceMemoryProperties2,
    // core from Vulkan 1.1.
    let budgeted = caps.supports_extension(ash::ext::memory_budget::NAME)
        && caps.properties().api_version >= vk::API_VERSION_1_1
        && shared.instance_api_version() >= vk::API_VERSION_1_1;
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let memory = if budgeted {
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: `physical` belongs to `instance`, both live while `hal`
        // does, the device supports the extension at Vulkan 1.1, and both
        // structs outlive the call.
        unsafe { instance.get_physical_device_memory_properties2(physical, &mut props) };
        props.memory_properties
    } else {
        // SAFETY: `physical` belongs to the live `instance`.
        unsafe { instance.get_physical_device_memory_properties(physical) }
    };
    let count = memory.memory_heap_count as usize;
    let heaps: Vec<(u64, bool)> = memory.memory_heaps[..count]
        .iter()
        .map(|h| (h.size, h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL)))
        .collect();
    vulkan_heaps(&heaps, budgeted.then_some(&budget.heap_budget[..count]))
        .ok_or_else(|| "the Vulkan adapter has no device-local heap".to_string())
}

#[cfg(windows)]
fn dx12(adapter: &wgpu::Adapter) -> std::result::Result<Reading, String> {
    use windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
    // SAFETY: the native adapter is only read, and the guard is dropped
    // before `adapter` is.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }
        .ok_or("the DX12 adapter has no native handle")?;
    let info = hal
        .raw_adapter()
        .query_video_memory_info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
        .map_err(|e| format!("DXGI QueryVideoMemoryInfo: {e}"))?;
    Ok(Reading {
        bytes: info.Budget,
        source: Source::DxgiLocal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn the_largest_device_local_heap_is_read_at_its_budget() {
        let heaps = [(46 * GIB, true), (64 * GIB, false), (GIB / 4, true)];
        let budgets = [GIB * 47 / 2, 60 * GIB, GIB / 8];
        let read = vulkan_heaps(&heaps, Some(&budgets)).unwrap();
        assert_eq!(read.bytes, GIB * 47 / 2);
        assert_eq!(
            read.to_string(),
            "device memory: 23.5 GiB from VK_EXT_memory_budget heap 0"
        );
    }

    #[test]
    fn without_the_budget_extension_the_heap_size_is_read() {
        let heaps = [(64 * GIB, false), (GIB / 4, true), (24 * GIB, true)];
        let read = vulkan_heaps(&heaps, None).unwrap();
        assert_eq!(read.bytes, 24 * GIB);
        assert_eq!(
            read.to_string(),
            "device memory: 24.0 GiB from Vulkan heap 2 size"
        );
    }

    #[test]
    fn no_device_local_heap_reads_nothing() {
        assert_eq!(vulkan_heaps(&[(64 * GIB, false)], None), None);
    }

    #[test]
    fn each_source_names_itself() {
        let line = |source| {
            Reading {
                bytes: 16 * GIB,
                source,
            }
            .to_string()
        };
        assert_eq!(
            line(Source::Config),
            "device memory: 16.0 GiB from config budget_device"
        );
        assert_eq!(
            line(Source::DxgiLocal),
            "device memory: 16.0 GiB from DXGI QueryVideoMemoryInfo local"
        );
    }

    #[test]
    fn a_refusal_names_budget_device() {
        let why = refused("the gl backend reports no memory size".to_string()).to_string();
        assert!(why.contains("reports no memory size"), "{why}");
        assert!(why.contains("[engine.memory] budget_device"), "{why}");
    }
}
