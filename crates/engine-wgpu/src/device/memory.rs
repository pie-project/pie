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
    /// The size of this device-local heap, without `VK_EXT_memory_budget`,
    /// on a device with memory of its own.
    VulkanHeap(usize),
    /// What the host's memory has available, less what lmkd keeps free
    /// past it and the floor ([`engine_gpu::host_memory`]), plus what this
    /// process already holds: a device that shares the host's memory,
    /// without `VK_EXT_memory_budget`, whose heap size is the host's RAM.
    HostAvailable,
    /// `IDXGIAdapter3::QueryVideoMemoryInfo`'s local segment `Budget`.
    DxgiLocal,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Source::Config => write!(f, "config budget_device"),
            Source::VulkanBudget(heap) => write!(f, "VK_EXT_memory_budget heap {heap}"),
            Source::VulkanHeap(heap) => write!(f, "Vulkan heap {heap} size"),
            Source::HostAvailable => write!(f, "the host's available memory (unified)"),
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

/// Whether a Vulkan device shares the host's memory: an integrated or CPU
/// device does, a discrete one does not (resizable BAR maps its heap, but
/// the memory is its own), and any other where the host maps its largest
/// device-local heap (`mapped`). The native engine's rule.
#[must_use]
pub fn shares_host(device: wgpu::DeviceType, mapped: bool) -> bool {
    match device {
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => true,
        wgpu::DeviceType::DiscreteGpu => false,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => mapped,
    }
}

/// Whose memory a device's is: the host's where it shares it
/// ([`shares_host`]) and on every Android device, as the native engine
/// states it. Metal states an adapter with unified memory, and DX12 one
/// with UMA, as integrated. Where it is the host's, the load states no host
/// tier.
#[must_use]
pub fn device_memory(device: wgpu::DeviceType, mapped: bool) -> engine::DeviceMemory {
    if cfg!(target_os = "android") || shares_host(device, mapped) {
        engine::DeviceMemory::Hosts
    } else {
        engine::DeviceMemory::Own
    }
}

/// Whose memory `adapter`'s is ([`device_memory`]), read at load: under
/// Vulkan, whether the host maps its largest device-local heap counts for
/// a device that states neither integrated nor discrete.
///
/// # Errors
///
/// A Vulkan adapter without its native handle, refused by name.
pub fn whose(adapter: &wgpu::Adapter) -> Result<engine::DeviceMemory> {
    let info = adapter.get_info();
    let mapped = match info.backend {
        #[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
        wgpu::Backend::Vulkan => vulkan_mapped(adapter).map_err(|why| {
            Fault::Residency(format!("wgpu: whose memory the device's is: {why}"))
        })?,
        _ => false,
    };
    Ok(device_memory(info.device_type, mapped))
}

#[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
fn vulkan_mapped(adapter: &wgpu::Adapter) -> std::result::Result<bool, String> {
    // SAFETY: the native adapter is only read, and the guard is dropped
    // before `adapter` is.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("the Vulkan adapter has no native handle")?;
    // SAFETY: the physical device belongs to the live instance.
    let memory = unsafe {
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device())
    };
    Ok(largest_mapped(&memory))
}

/// Whether the host maps the largest device-local heap of `memory`: a
/// memory type on it both `DEVICE_LOCAL` and `HOST_VISIBLE`.
#[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
fn largest_mapped(memory: &ash::vk::PhysicalDeviceMemoryProperties) -> bool {
    use ash::vk;
    let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
    let largest = (0..heaps.len())
        .filter(|&at| heaps[at].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .max_by_key(|&at| heaps[at].size);
    memory.memory_types[..memory.memory_type_count as usize]
        .iter()
        .any(|t| {
            Some(t.heap_index as usize) == largest
                && t.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
        })
}

/// Of Vulkan's memory heaps, `(size, device-local)` each, the largest
/// device-local one, which device reservations take: its budget where
/// `VK_EXT_memory_budget` states `budgets`; else, on a device that shares
/// the host's memory, whose heap is the host's RAM, what the host has
/// available (`host`); else its size.
///
/// # Errors
///
/// Names the reading that could not be taken: no device-local heap, or
/// the host's room.
pub fn vulkan_heaps(
    heaps: &[(u64, bool)],
    budgets: Option<&[u64]>,
    shares_host: bool,
    host: impl FnOnce() -> std::result::Result<u64, String>,
) -> std::result::Result<Reading, String> {
    let (heap, size) = heaps
        .iter()
        .enumerate()
        .filter(|&(_, &(_, local))| local)
        .map(|(at, &(size, _))| (at, size))
        .max_by_key(|&(_, size)| size)
        .ok_or("the Vulkan adapter has no device-local heap")?;
    Ok(match budgets {
        Some(budgets) => Reading {
            bytes: budgets[heap],
            source: Source::VulkanBudget(heap),
        },
        None if shares_host => Reading {
            bytes: host()?,
            source: Source::HostAvailable,
        },
        None => Reading {
            bytes: size,
            source: Source::VulkanHeap(heap),
        },
    })
}

/// What the host gives a load on a device that shares its memory: its room
/// now, plus what this process already holds, which that room counts as
/// taken.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn host_room() -> std::result::Result<u64, String> {
    let room = engine_gpu::host_memory::room().map_err(|fault| fault.to_string())?;
    let held = engine_gpu::books::process_reserved();
    tracing::info!(
        available = room.available,
        defended = room.defended,
        cgroup = ?room.cgroup,
        floor = engine_gpu::host_memory::SAFETY_FLOOR_BYTES,
        room = room.room,
        held,
        "the device shares the host's memory, so the load takes what it has available, \
         less what lmkd keeps free past it, capped at what \
         its cgroups leave, less the floor"
    );
    Ok(room.room.saturating_add(held))
}

#[cfg(all(
    not(any(target_os = "linux", target_os = "android")),
    not(any(target_arch = "wasm32", target_vendor = "apple"))
))]
fn host_room() -> std::result::Result<u64, String> {
    Err(
        "the Vulkan device shares the host's memory without VK_EXT_memory_budget, and \
         this host has no reader of its available memory"
            .to_string(),
    )
}

/// The most bytes one reservation takes: the least of the device's
/// `max_buffer_size`, `max_storage_buffer_binding_size` (a slab is bound
/// whole) and, where the backend states it, the most one allocation backs
/// (`allocation`: Vulkan's `maxMemoryAllocationSize`). A limit stated as
/// zero states nothing.
#[must_use]
pub fn reservation_bound(limits: &wgpu::Limits, allocation: Option<u64>) -> u64 {
    [Some(limits.max_storage_buffer_binding_size), allocation]
        .into_iter()
        .flatten()
        .filter(|&limit| limit > 0)
        .fold(limits.max_buffer_size, u64::min)
}

/// The most bytes one allocation of `adapter` backs, where its backend
/// states it: Vulkan's `maxMemoryAllocationSize` (core from 1.1).
#[must_use]
pub fn allocation_limit(adapter: &wgpu::Adapter) -> Option<u64> {
    match adapter.get_info().backend {
        #[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
        wgpu::Backend::Vulkan => vulkan_allocation(adapter),
        _ => None,
    }
}

#[cfg(not(any(target_arch = "wasm32", target_vendor = "apple")))]
fn vulkan_allocation(adapter: &wgpu::Adapter) -> Option<u64> {
    use ash::vk;
    // SAFETY: the native adapter is only read, and the guard is dropped
    // before `adapter` is.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let shared = hal.shared_instance();
    let caps = hal.physical_device_capabilities();
    if caps.properties().api_version < vk::API_VERSION_1_1
        || shared.instance_api_version() < vk::API_VERSION_1_1
    {
        return None;
    }
    let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance3);
    // SAFETY: the physical device belongs to the live instance, both at
    // Vulkan 1.1, and both structs outlive the call.
    unsafe {
        shared
            .raw_instance()
            .get_physical_device_properties2(hal.raw_physical_device(), &mut props);
    }
    Some(maintenance3.max_memory_allocation_size)
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
    let mapped = largest_mapped(&memory);
    vulkan_heaps(
        &heaps,
        budgeted.then_some(&budget.heap_budget[..count]),
        shares_host(adapter.get_info().device_type, mapped),
        host_room,
    )
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

    /// The Adreno 840 and the Spark under Vulkan, Apple under Metal: no host
    /// tier, a stated one refused by name. A discrete card is unchanged.
    #[test]
    fn a_device_whose_memory_is_the_hosts_states_no_host_tier() {
        use wgpu::DeviceType::{DiscreteGpu, IntegratedGpu};
        let unified = device_memory(IntegratedGpu, false);
        assert_eq!(unified, engine::DeviceMemory::Hosts);
        assert_eq!(unified.host_tier(None), Ok(engine::HostTier::Absent));
        let refused = unified.host_tier(Some(1 << 30)).unwrap_err();
        assert!(
            refused.contains("`[engine.memory] budget_host`"),
            "{refused}"
        );
        if !cfg!(target_os = "android") {
            let discrete = device_memory(DiscreteGpu, true);
            assert_eq!(discrete, engine::DeviceMemory::Own);
            assert_eq!(
                discrete.host_tier(Some(1 << 30)),
                Ok(engine::HostTier::Seated(Some(1 << 30)))
            );
            assert_eq!(discrete.host_tier(None), Ok(engine::HostTier::Seated(None)));
        }
    }

    #[test]
    fn a_refusal_names_budget_device() {
        let why = refused("the gl backend reports no memory size".to_string()).to_string();
        assert!(why.contains("reports no memory size"), "{why}");
        assert!(why.contains("[engine.memory] budget_device"), "{why}");
    }
}
