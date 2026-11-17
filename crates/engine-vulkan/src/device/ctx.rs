#![allow(unsafe_code)]
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::api::DeviceBoot;
use engine_gpu::error::{Fault, Result};

use super::alloc::{Buffer, Memory, Raw, Slab};

static RESERVATIONS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_reservation() {
    RESERVATIONS.fetch_add(1, Ordering::Relaxed);
}

#[must_use]
pub fn reservations() -> u64 {
    RESERVATIONS.load(Ordering::Relaxed)
}

fn entry() -> std::result::Result<ash::Entry, String> {
    static ENTRY: std::sync::OnceLock<std::result::Result<ash::Entry, String>> =
        std::sync::OnceLock::new();
    ENTRY
        // SAFETY: this loads the system Vulkan loader, whose initializers are
        // trusted as any linked system library's; the `OnceLock` loads it
        // once and keeps it loaded for the life of the process.
        .get_or_init(|| unsafe { ash::Entry::load() }.map_err(|e| e.to_string()))
        .clone()
}

fn instance_gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GATE.lock().expect("gate lock poisoned")
}

/// What it takes to see a layered driver (MoltenVK), which the loader hides
/// from an instance that does not ask for it.
fn portability(exts: &[vk::ExtensionProperties]) -> (Vec<*const c_char>, vk::InstanceCreateFlags) {
    let name = ash::khr::portability_enumeration::NAME;
    if has_extension(exts, name) {
        (
            vec![name.as_ptr()],
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
        )
    } else {
        (Vec::new(), vk::InstanceCreateFlags::empty())
    }
}

#[must_use]
pub fn present() -> bool {
    let Ok(entry) = entry() else {
        return false;
    };
    let _gate = instance_gate();
    let Ok((instance, _)) = shared_instance(&entry, false) else {
        return false;
    };
    // SAFETY: the shared instance lives for the process; its physical
    // devices come from it.
    unsafe {
        instance
            .enumerate_physical_devices()
            .map(|devices| {
                devices.iter().any(|&physical| {
                    instance
                        .get_physical_device_queue_family_properties(physical)
                        .iter()
                        .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
            })
            .unwrap_or(false)
    }
}

/// The process's instance (with the validation layer, or without), made
/// on first use and never destroyed: every device is made from it. A
/// driver may refuse instances past a count in one process however many
/// were destroyed (NVIDIA 595.91 refuses the 27th: "Unable to find a Vulkan
/// driver"), so loads made one after another (tests, reloads) share one.
/// The caller holds the instance gate.
fn shared_instance(
    entry: &ash::Entry,
    validated: bool,
) -> std::result::Result<(ash::Instance, bool), Fault> {
    static SHARED: Mutex<Vec<(bool, ash::Instance, bool)>> = Mutex::new(Vec::new());
    let mut shared = SHARED.lock().expect("shared instance lock poisoned");
    if let Some((_, instance, props2)) = shared.iter().find(|(v, ..)| *v == validated) {
        return Ok((instance.clone(), *props2));
    }
    let app = vk::ApplicationInfo::default()
        .application_name(c"pie")
        .api_version(vk::API_VERSION_1_3);
    let validation = c"VK_LAYER_KHRONOS_validation";
    let enabled_layers: Vec<*const c_char> = if validated {
        vec![validation.as_ptr()]
    } else {
        Vec::new()
    };
    let instance_exts =
        // SAFETY: a global query of the loaded entry, with no handles.
        unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
    let (mut enabled_exts, flags) = portability(&instance_exts);
    if validated {
        enabled_exts.push(ash::ext::debug_utils::NAME.as_ptr());
    }
    let props2 = has_extension(
        &instance_exts,
        ash::khr::get_physical_device_properties2::NAME,
    );
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .flags(flags)
        .enabled_layer_names(&enabled_layers)
        .enabled_extension_names(&enabled_exts);
    // SAFETY: `info` names only layers and extensions found above, its
    // borrowed names and `app` outlive the call, and the gate is held.
    let instance = unsafe { entry.create_instance(&info, None) }.map_err(|e| Fault::NoDevice {
        detail: format!("vkCreateInstance: {e}"),
    })?;
    if validated {
        let debug = ash::ext::debug_utils::Instance::new(entry, &instance);
        let create = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            )
            .pfn_user_callback(Some(on_validation));
        // SAFETY: `debug_utils` is enabled on `instance` whenever
        // `validated`, and `on_validation` matches the callback ABI and
        // reads only what the layer passes it. The messenger lives as long
        // as the instance, the process.
        let _ = unsafe { debug.create_debug_utils_messenger(&create, None) };
    }
    shared.push((validated, instance.clone(), props2));
    Ok((instance, props2))
}

pub(crate) const STAGING_BYTES: u64 = 256 << 20;

pub(crate) const DUMMY_BYTES: u64 = 256;

pub(crate) struct Transfer {
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) fence: vk::Fence,
    pub(crate) staging: Option<Slab>,
}

const SPARE_FENCES: usize = 32;

pub(crate) struct Core {
    _entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) queue: Mutex<vk::Queue>,

    pub(crate) pools: Mutex<std::collections::HashMap<std::thread::ThreadId, vk::CommandPool>>,

    spare_fences: Mutex<Vec<vk::Fence>>,

    /// Command buffers whose frame was dropped off the thread that owns
    /// their pool, which a pool's own thread frees: a pool is not shared.
    orphans: Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,

    pub(crate) family: u32,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    pub(crate) transfer: Mutex<Transfer>,
    pub(crate) books: engine_gpu::Books,
    pub(crate) name: String,
    pub(crate) subgroup_size: u32,
    /// The narrowest subgroup a pipeline runs at.
    pub(crate) subgroup_min: u32,
    /// `requiredSubgroupSize` for every pipeline, where the device can pin
    /// one: 32, which the coopmat and decode attention kernels assume.
    pub(crate) pinned: Option<u32>,
    pub(crate) cores: Option<u32>,
    pub(crate) device_local: u64,
    /// The largest device-local heap, which device reservations take.
    heap: usize,
    /// Where what is free of that heap is read from.
    pub(crate) source: BudgetSource,
    /// Whether the heap is the host's memory: an integrated device, or a
    /// device-local heap the host maps.
    pub(crate) unified: bool,
    /// The largest buffer a load makes ([`buffer_ceiling`]), and the limit
    /// that sets it; a test lowers it to lay a plane in pieces.
    max_buffer: AtomicU64,
    max_buffer_limit: Mutex<&'static str>,
    pub(crate) vendor_id: u32,
    pub(crate) discrete: bool,
    pub(crate) api_version: u32,
    pub(crate) features: Enabled,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Enabled {
    pub shader_int16: bool,
    pub shader_int64: bool,
    pub shader_float16: bool,
    pub storage_16bit: bool,
    pub storage_8bit: bool,
    pub memory_model: bool,
    pub subgroup_size_control: bool,
    pub memory_budget: bool,

    pub coopmat: bool,
}

/// Where a device's free memory is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// `VK_EXT_memory_budget`: the heap's budget for this process less its
    /// usage.
    Extension,
    /// The host's available memory (`MemAvailable` of `/proc/meminfo`): a
    /// unified-memory device without the extension, whose heap is the
    /// host's RAM, all of it, however much the host has handed out.
    OsAvailable,
    /// The device-local heap less every reservation of this process: a
    /// discrete device without the extension, blind to other processes.
    Heap,
}

impl std::fmt::Display for BudgetSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BudgetSource::Extension => "the memory_budget extension",
            BudgetSource::OsAvailable => "the OS's MemAvailable",
            BudgetSource::Heap => {
                "the device-local heap less this process's reservations (no memory_budget \
                 extension)"
            }
        })
    }
}

/// The host's available memory now: `MemAvailable` of `/proc/meminfo`, on
/// Android less the reserve its low-memory killer defends ([`lmk_reserve`]).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_available() -> std::result::Result<u64, String> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| format!("reading /proc/meminfo: {e}"))?;
    let available =
        mem_available(&text).ok_or_else(|| "/proc/meminfo states no MemAvailable".to_string())?;
    Ok(available.saturating_sub(lmk_reserve().map_or(0, |(bytes, _)| bytes)))
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn os_available() -> std::result::Result<u64, String> {
    Err("this host has no reader of its available memory (only Linux and Android's \
         /proc/meminfo is read)"
        .to_string())
}

/// What Android's low-memory killer keeps free before it kills a foreground
/// app (one at `oom_score_adj` 0, as a process serving in the foreground
/// is): the free pages of its levels whose kills reach adj 0, the most of
/// them, and where they were read. lmkd states its levels as the property
/// `sys.lmk.minfree_levels` (`pages:adj,...`); the kernel's own killer as
/// `/sys/module/lowmemorykiller/parameters/{minfree,adj}`. `None` off
/// Android, or where neither is readable.
#[cfg(target_os = "android")]
pub(crate) fn lmk_reserve() -> Option<(u64, &'static str)> {
    // SAFETY: `sysconf` reads a constant of the process.
    let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    if let Some(levels) = property(c"sys.lmk.minfree_levels")
        && let Some(pages) = foreground_minfree(&levels)
    {
        return Some((pages * page, "sys.lmk.minfree_levels"));
    }
    let dir = std::path::Path::new("/sys/module/lowmemorykiller/parameters");
    let minfree = std::fs::read_to_string(dir.join("minfree")).ok()?;
    let adj = std::fs::read_to_string(dir.join("adj")).ok()?;
    let levels: Vec<String> = minfree
        .trim()
        .split(',')
        .zip(adj.trim().split(','))
        .map(|(pages, adj)| format!("{pages}:{adj}"))
        .collect();
    foreground_minfree(&levels.join(","))
        .map(|pages| (pages * page, "/sys/module/lowmemorykiller/parameters/minfree"))
}

#[cfg(not(target_os = "android"))]
pub(crate) fn lmk_reserve() -> Option<(u64, &'static str)> {
    None
}

/// An Android system property's value, where it is set.
#[cfg(target_os = "android")]
fn property(name: &CStr) -> Option<String> {
    let mut value = [0 as c_char; 92];
    // SAFETY: `name` is NUL-terminated and `value` holds PROP_VALUE_MAX (92)
    // bytes, the most the call writes, NUL included.
    let len = unsafe { libc::__system_property_get(name.as_ptr(), value.as_mut_ptr()) };
    if len <= 0 {
        return None;
    }
    // SAFETY: the call wrote a NUL-terminated string into `value`.
    let text = unsafe { CStr::from_ptr(value.as_ptr()) };
    Some(text.to_string_lossy().into_owned())
}

/// The most free pages a `pages:adj,...` level list keeps before a kill
/// reaches `oom_score_adj` 0: its levels whose adj is 0 or below.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub(crate) fn foreground_minfree(levels: &str) -> Option<u64> {
    levels
        .split(',')
        .filter_map(|level| {
            let (pages, adj) = level.trim().split_once(':')?;
            let adj: i32 = adj.trim().parse().ok()?;
            (adj <= 0).then(|| pages.trim().parse::<u64>().ok()).flatten()
        })
        .max()
}

/// `MemAvailable`'s bytes in a `/proc/meminfo` text.
pub(crate) fn mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let mut words = line.split_whitespace();
    let value: u64 = words.next()?.parse().ok()?;
    match words.next() {
        Some("kB") => value.checked_mul(1024),
        None => Some(value),
        Some(_) => None,
    }
}

impl Core {
    /// The largest device-local heap's budget for this process and what
    /// this process holds of it, as the driver's memory budget reports them
    /// now; `None` without `VK_EXT_memory_budget`.
    fn budget(&self) -> Option<(u64, u64)> {
        if !self.features.memory_budget {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: `physical` belongs to `instance`, the budget struct is
        // chained only when the device enabled `VK_EXT_memory_budget`, and
        // both structs outlive the call.
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut props);
        }
        Some((budget.heap_budget[self.heap], budget.heap_usage[self.heap]))
    }

    /// The device-local bytes free to this process now, read from
    /// [`Core::source`]: the heap's budget less its usage; the host's
    /// available memory, no more than the heap less every reservation; or
    /// the heap less every reservation (blind to other processes).
    pub(crate) fn free(&self) -> u64 {
        let heap_left = || {
            self.device_local
                .saturating_sub(engine_gpu::books::process_reserved())
        };
        match self.source {
            BudgetSource::Extension => self
                .budget()
                .map_or_else(heap_left, |(budget, usage)| budget.saturating_sub(usage)),
            BudgetSource::OsAvailable => match os_available() {
                Ok(available) => available.min(heap_left()),
                Err(why) => {
                    // Read at load, so this is the host taking it away.
                    tracing::warn!(%why, "the host's available memory reads as none");
                    0
                }
            },
            BudgetSource::Heap => heap_left(),
        }
    }

    /// What the driver holds for this process in the heap beside every
    /// reservation of this process: its usage less them, where it reports it.
    pub(crate) fn outside(&self) -> u64 {
        self.budget().map_or(0, |(_, usage)| {
            usage.saturating_sub(engine_gpu::books::process_reserved())
        })
    }

    /// The largest buffer a load makes, and the device limit that sets it.
    pub(crate) fn max_buffer(&self) -> (u64, &'static str) {
        (
            self.max_buffer.load(Ordering::Relaxed),
            *self
                .max_buffer_limit
                .lock()
                .expect("max buffer lock poisoned"),
        )
    }

    pub(crate) fn fault(&self, call: &'static str, code: vk::Result) -> Fault {
        Fault::Vulkan {
            what: call,
            code: code.as_raw(),
        }
    }

    pub(crate) fn submit_once(
        &self,
        transfer: &Transfer,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        let d = &self.device;
        // SAFETY: a `&Transfer` exists only through the guard of
        // `self.transfer`, so the caller holds that lock and no one else
        // touches its command buffer or fence. The previous use waited on the
        // fence, so the buffer is not pending when it is begun again; `record`
        // only records into it, and the queue is used under its own lock.
        unsafe {
            d.begin_command_buffer(
                transfer.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| self.fault("vkBeginCommandBuffer", e))?;
            record(d, transfer.cmd);
            d.end_command_buffer(transfer.cmd)
                .map_err(|e| self.fault("vkEndCommandBuffer", e))?;
            d.reset_fences(&[transfer.fence])
                .map_err(|e| self.fault("vkResetFences", e))?;
            let bufs = [transfer.cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            {
                let queue = self.queue.lock().expect("queue lock poisoned");
                d.queue_submit(*queue, &submits, transfer.fence)
                    .map_err(|e| self.fault("vkQueueSubmit", e))?;
            }
            d.wait_for_fences(&[transfer.fence], true, FENCE_TIMEOUT_NS)
                .map_err(|e| self.fault("vkWaitForFences", e))?;
        }
        Ok(())
    }

    pub(crate) fn staging<'a>(self: &Arc<Self>, transfer: &'a mut Transfer) -> Result<&'a Slab> {
        if transfer.staging.is_none() {
            // No larger than the largest buffer the device makes.
            let bytes = STAGING_BYTES.min(self.max_buffer().0 & !3);
            transfer.staging = Some(Raw::new(
                self,
                bytes,
                Memory::Staging,
                engine_gpu::books::owners::STAGING,
            )?);
        }
        Ok(transfer.staging.as_ref().expect("just made"))
    }
}

const FENCE_TIMEOUT_NS: u64 = 120_000_000_000;

impl Core {
    fn lend_fence(&self) -> Result<vk::Fence> {
        if let Some(fence) = self
            .spare_fences
            .lock()
            .expect("spare fences lock poisoned")
            .pop()
        {
            return Ok(fence);
        }
        // SAFETY: `self.device` is live while `self` is; a default fence info
        // creates an unsignaled fence.
        unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(|e| self.fault("vkCreateFence", e))
    }

    fn return_fence(&self, fence: vk::Fence) {
        let mut spare = self
            .spare_fences
            .lock()
            .expect("spare fences lock poisoned");
        // SAFETY: the only caller, `Inner`'s drop, returns its fence after
        // waiting on it, so no queue submission still names it; nothing else
        // holds the handle once it is returned.
        if spare.len() >= SPARE_FENCES || unsafe { self.device.reset_fences(&[fence]) }.is_err() {
            drop(spare);
            // SAFETY: as above, the fence is ours and no longer in use.
            unsafe { self.device.destroy_fence(fence, None) };
            return;
        }
        spare.push(fence);
    }

    pub(crate) fn thread_pool(&self) -> Result<vk::CommandPool> {
        let mut pools = self.pools.lock().expect("pools lock poisoned");
        if let Some(&pool) = pools.get(&std::thread::current().id()) {
            let mut orphans = self.orphans.lock().expect("orphans lock poisoned");
            let mut freed = Vec::new();
            orphans.retain(|&(owner, cmd)| {
                let mine = owner == pool;
                if mine {
                    freed.push(cmd);
                }
                !mine
            });
            if !freed.is_empty() {
                // SAFETY: `pool` is this thread's own pool, so it is not used
                // concurrently, and each orphan was allocated from it by a
                // frame whose drop already waited out its submission.
                unsafe { self.device.free_command_buffers(pool, &freed) };
            }
            return Ok(pool);
        }
        // SAFETY: `self.device` is live and `family` is the queue family it
        // was created with.
        let pool = unsafe {
            self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| self.fault("vkCreateCommandPool", e))?;
        pools.insert(std::thread::current().id(), pool);
        Ok(pool)
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // SAFETY: this is the last `Arc<Core>`, so every slab, frame, cache and
        // pipeline that held one is gone; `device_wait_idle` retires any work
        // still queued. Children are destroyed before the device.
        unsafe {
            let _ = self.device.device_wait_idle();
            {
                let transfer = self.transfer.get_mut().expect("transfer lock poisoned");
                transfer.staging = None;
                self.device.destroy_fence(transfer.fence, None);
            }
            for fence in std::mem::take(
                self.spare_fences
                    .get_mut()
                    .expect("spare fences lock poisoned"),
            ) {
                self.device.destroy_fence(fence, None);
            }
            let pools = std::mem::take(self.pools.get_mut().expect("pools lock poisoned"));
            for pool in pools.into_values() {
                self.device.destroy_command_pool(pool, None);
            }
            // The instance is the process's (`shared_instance`).
            self.device.destroy_device(None);
        }
    }
}

unsafe extern "system" fn on_validation(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: `data` is non-null and the validation layer keeps it, and its
    // message string, valid for the duration of this callback.
    let message = unsafe { (*data).message_as_c_str() }
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("vulkan validation [{:#x}]: {message}", severity.as_raw());
    vk::FALSE
}

pub struct Context {
    pub(crate) core: Arc<Core>,
    /// The operator's cap on what the load takes, all of it together.
    budget: Option<u64>,
    pub(crate) dummy: Slab,
    pipeline_cache: Arc<PipelineCache>,
    device_index: u32,
}

/// The device's pipeline cache, which kernel and guest pipelines share. A
/// build marks it changed, and it reaches its file shortly after rather
/// than at shutdown: a phone kills an app's process more often than the app
/// shuts down.
pub struct PipelineCache {
    core: Arc<Core>,
    cache: vk::PipelineCache,
    path: Option<std::path::PathBuf>,
    changed: std::sync::atomic::AtomicBool,
    written: Mutex<std::time::Instant>,
}

/// How long a changed cache waits for more builds before it is written.
const PIPELINE_CACHE_SETTLE: std::time::Duration = std::time::Duration::from_secs(2);

impl PipelineCache {
    fn open(core: &Arc<Core>, path: Option<std::path::PathBuf>) -> Result<PipelineCache> {
        let initial = path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .unwrap_or_default();
        // SAFETY: `core.device` is live; any bytes are acceptable initial
        // data, as the driver validates the header and ignores a stale cache.
        let make = |bytes: &[u8]| unsafe {
            core.device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(bytes),
                None,
            )
        };
        let cache = make(&initial)
            .or_else(|_| make(&[]))
            .map_err(|e| core.fault("vkCreatePipelineCache", e))?;
        Ok(PipelineCache {
            core: Arc::clone(core),
            cache,
            path,
            changed: std::sync::atomic::AtomicBool::new(false),
            written: Mutex::new(std::time::Instant::now()),
        })
    }

    #[must_use]
    pub fn handle(&self) -> vk::PipelineCache {
        self.cache
    }

    #[must_use]
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }

    /// Marks the cache changed by a build.
    pub fn built(&self) {
        self.changed.store(true, Ordering::Release);
    }

    /// Writes a changed cache once builds have paused for a moment.
    pub fn settle(&self) {
        let due = self.changed.load(Ordering::Acquire)
            && self
                .written
                .lock()
                .expect("written lock poisoned")
                .elapsed()
                >= PIPELINE_CACHE_SETTLE;
        if due && let Err(fault) = self.persist() {
            tracing::warn!(%fault, "writing the pipeline cache");
        }
    }

    pub fn persist(&self) -> Result<()> {
        if !self.changed.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        *self.written.lock().expect("written lock poisoned") = std::time::Instant::now();
        let Some(path) = &self.path else {
            return Ok(());
        };
        // SAFETY: `self.cache` is a live cache of `self.core.device`; reading
        // its data may race pipeline builds that use it, which Vulkan allows,
        // as `vkGetPipelineCacheData` does not need it externally synchronized.
        let bytes = unsafe { self.core.device.get_pipeline_cache_data(self.cache) }
            .map_err(|e| self.core.fault("vkGetPipelineCacheData", e))?;
        let io = |call: &'static str| {
            move |e: std::io::Error| Fault::Device {
                call,
                why: e.to_string(),
            }
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(io("pipeline cache dir"))?;
        }
        // Whole or not at all: a reader never meets a half-written file.
        let partial = path.with_extension("partial");
        std::fs::write(&partial, bytes).map_err(io("pipeline cache write"))?;
        std::fs::rename(&partial, path).map_err(io("pipeline cache rename"))
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        let _ = self.persist();
        // SAFETY: this is the last `Arc<PipelineCache>`, so no build is using
        // the cache; pipelines made from it do not depend on it.
        unsafe { self.core.device.destroy_pipeline_cache(self.cache, None) };
    }
}

/// The staging buffer holds the `Core` that holds it; left in place, that
/// cycle leaks the device after every load until `vkCreateDevice` fails.
///
/// The queue is first drained: a shell drops its pipelines, descriptor
/// layouts and guest sessions after its context, and frames still in flight
/// (whose watchers outlive the shell) may use them.
impl Drop for Context {
    fn drop(&mut self) {
        {
            let queue = self.core.queue.lock().expect("queue lock poisoned");
            // SAFETY: the queue is this device's, and its lock is the external
            // sync `vkQueueWaitIdle` needs.
            let _ = unsafe { self.core.device.queue_wait_idle(*queue) };
        }
        self.core
            .transfer
            .lock()
            .expect("transfer lock poisoned")
            .staging = None;
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("name", &self.core.name)
            .field("budget", &self.budget)
            .finish()
    }
}

fn has_extension(exts: &[vk::ExtensionProperties], name: &CStr) -> bool {
    exts.iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|s| s == name))
}

/// The largest buffer a load makes: one a single storage descriptor binds
/// whole (`maxStorageBufferRange`), that one allocation backs
/// (`maxMemoryAllocationSize`, core 1.1) and that the device creates
/// (`maxBufferSize`, core 1.3; absent before it), with the name of the
/// limit that sets it. A limit reported as zero states nothing.
pub(crate) fn buffer_ceiling(
    storage_range: u32,
    allocation: u64,
    buffer: Option<u64>,
) -> (u64, &'static str) {
    [
        ("maxMemoryAllocationSize", Some(allocation)),
        ("maxBufferSize", buffer),
    ]
    .into_iter()
    .filter_map(|(name, limit)| limit.filter(|&l| l > 0).map(|l| (l, name)))
    .fold(
        (u64::from(storage_range), "maxStorageBufferRange"),
        |held, next| if next.0 < held.0 { next } else { held },
    )
}

impl Context {
    /// Binds the device `boot` names for a load that takes what it has,
    /// no more than `allowance.budget` where the operator states one.
    pub fn bind(boot: &DeviceBoot, allowance: engine::Allowance) -> Result<Context> {
        let budget = allowance.budget;
        let entry = entry().map_err(|e| Fault::NoDevice {
            detail: format!("no Vulkan loader: {e}"),
        })?;
        let _gate = instance_gate();
        // SAFETY: a global query of the loaded entry, with no handles.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let validation = c"VK_LAYER_KHRONOS_validation";
        let validated = boot.validation;
        if validated
            && !layers
                .iter()
                .any(|l| l.layer_name_as_c_str().is_ok_and(|s| s == validation))
        {
            return Err(Fault::NoDevice {
                detail: "`[vulkan] validation = true` asks for VK_LAYER_KHRONOS_validation, \
                         which the Vulkan loader does not find installed; install the \
                         layer or set `validation = false`"
                    .to_string(),
            });
        }
        let (instance, props2) = shared_instance(&entry, validated)?;
        let context = Self::finish(entry, instance, boot, budget, props2)?;
        if let Some(bytes) = boot.max_buffer {
            context.limit_buffers(bytes);
        }
        Ok(context)
    }

    #[allow(
        clippy::type_complexity,
        clippy::too_many_lines,
        clippy::result_large_err
    )]
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        boot: &DeviceBoot,
        budget: Option<u64>,
        _props2: bool,
    ) -> Result<Context> {
        macro_rules! bail {
            ($fault:expr) => {
                return Err($fault)
            };
        }
        // In this function every query takes a `physical` enumerated from
        // `instance`, which lives until the end of it, and every chained
        // output struct is a local that outlives its call.
        // SAFETY: `instance` is live.
        let physicals = match unsafe { instance.enumerate_physical_devices() } {
            Ok(p) => p,
            Err(e) => bail!(Fault::NoDevice {
                detail: format!("vkEnumeratePhysicalDevices: {e}"),
            }),
        };

        let mut candidates: Vec<(u32, usize, vk::PhysicalDevice, u32, String)> = Vec::new();
        for (ordinal, &physical) in physicals.iter().enumerate() {
            // SAFETY: `physical` was enumerated from the live `instance`.
            let props = unsafe { instance.get_physical_device_properties(physical) };
            let families =
                // SAFETY: as above.
                unsafe { instance.get_physical_device_queue_family_properties(physical) };
            let Some(family) = pick_family(&families) else {
                continue;
            };
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU => 4,
                _ => 3,
            };
            let name = props
                .device_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            candidates.push((rank, ordinal, physical, family, name));
        }
        candidates.sort_by_key(|c| (c.0, c.1));
        let Some(&(_, _, physical, family, ref name)) = candidates.get(boot.device_index as usize)
        else {
            let roster: Vec<String> = candidates.iter().map(|c| c.4.clone()).collect();
            bail!(Fault::NoDevice {
                detail: format!(
                    "no compute-capable Vulkan device at index {} (seen: {roster:?})",
                    boot.device_index
                ),
            });
        };
        let name = name.clone();
        // SAFETY: `physical` was enumerated from the live `instance`.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut sizes = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let sized = props.api_version >= vk::API_VERSION_1_3;
        let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Properties::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut maintenance3);
            if sized {
                p2 = p2.push_next(&mut sizes).push_next(&mut maintenance4);
            }
            // SAFETY: the chain holds the 1.1 subgroup and maintenance3
            // structs and, only on a 1.3 device, the size-control and
            // maintenance4 structs; all outlive the call.
            unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
        }
        let max_buffer = buffer_ceiling(
            props.limits.max_storage_buffer_range,
            maintenance3.max_memory_allocation_size,
            sized.then_some(maintenance4.max_buffer_size),
        );
        // The narrowest subgroup a pipeline may be compiled at: a driver
        // that varies the width (Intel: 8 to 32) reports its default as
        // `subgroupSize`, which a kernel folding 32 lanes cannot rely on.
        let subgroup_min = if sized && sizes.min_subgroup_size > 0 {
            sizes.min_subgroup_size.min(subgroup.subgroup_size.max(1))
        } else {
            subgroup.subgroup_size.max(1)
        };
        // SAFETY: `physical` was enumerated from the live `instance`.
        let device_exts =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let memory_budget = has_extension(&device_exts, ash::ext::memory_budget::NAME);
        let coopmat_ext = has_extension(&device_exts, ash::khr::cooperative_matrix::NAME);

        let coopmat_shape = coopmat_ext && {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
            // SAFETY: the device advertises `VK_KHR_cooperative_matrix`, so the
            // loaded query exists for `physical`, which `instance` owns.
            unsafe { cm.get_physical_device_cooperative_matrix_properties(physical) }
                .map(|props| {
                    props.iter().any(|p| {
                        p.m_size == 16
                            && p.n_size == 16
                            && p.k_size == 16
                            && p.a_type == vk::ComponentTypeKHR::FLOAT16
                            && p.b_type == vk::ComponentTypeKHR::FLOAT16
                            && p.c_type == vk::ComponentTypeKHR::FLOAT32
                            && p.result_type == vk::ComponentTypeKHR::FLOAT32
                            && p.scope == vk::ScopeKHR::SUBGROUP
                    })
                })
                .unwrap_or(false)
        };

        let cores = kernels_slang::cores::shader_cores(&device_exts, &mut |p2| {
            // SAFETY: `shader_cores` chains only structs of extensions it found
            // in `device_exts`, live for the call; `physical` is `instance`'s.
            unsafe { instance.get_physical_device_properties2(physical, p2) };
        });

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let api_13 = props.api_version >= vk::API_VERSION_1_3;
        {
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12);
            if api_13 {
                query = query.push_next(&mut f13);
            }
            if coopmat_ext {
                query = query.push_next(&mut fcm);
            }
            // SAFETY: the 1.3 and coopmat structs are chained only when the
            // device reports 1.3 or the extension; all outlive the call.
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
        }
        // SAFETY: `physical` was enumerated from the live `instance`.
        let core = unsafe { instance.get_physical_device_features(physical) };
        // Pin every pipeline to 32 lanes where the driver allows it (AMD
        // RDNA defaults to 64, Intel varies 8 to 32).
        let size_control = api_13 && f13.subgroup_size_control == vk::TRUE;
        let pinned = (size_control
            && sizes.min_subgroup_size <= 32
            && sizes.max_subgroup_size >= 32
            && sizes
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE))
        .then_some(32u32);
        let subgroup_min = pinned.unwrap_or(subgroup_min);
        let enabled = Enabled {
            shader_int16: core.shader_int16 == vk::TRUE,
            shader_int64: core.shader_int64 == vk::TRUE,
            shader_float16: f12.shader_float16 == vk::TRUE,
            storage_16bit: f11.storage_buffer16_bit_access == vk::TRUE,
            storage_8bit: f12.storage_buffer8_bit_access == vk::TRUE,
            memory_model: f12.vulkan_memory_model == vk::TRUE
                && f12.vulkan_memory_model_device_scope == vk::TRUE,
            subgroup_size_control: size_control,
            memory_budget,
            coopmat: coopmat_shape
                && fcm.cooperative_matrix == vk::TRUE
                && f12.shader_float16 == vk::TRUE
                && f12.vulkan_memory_model == vk::TRUE
                && (pinned == Some(32) || subgroup.subgroup_size == 32),
        };
        if !enabled.shader_int16 || !enabled.storage_16bit {
            bail!(Fault::NoDevice {
                detail: format!(
                    "{name} lacks shaderInt16/storageBuffer16BitAccess, which every bf16 kernel needs"
                ),
            });
        }
        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(enabled.storage_16bit)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(enabled.shader_float16)
            .shader_int8(f12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(enabled.storage_8bit)
            .uniform_and_storage_buffer8_bit_access(
                f12.uniform_and_storage_buffer8_bit_access == vk::TRUE,
            )
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut e13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(enabled.subgroup_size_control)
            .synchronization2(f13.synchronization2 == vk::TRUE);
        let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(enabled.coopmat);
        let mut enable = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(true)
                    .shader_int64(enabled.shader_int64)
                    .robust_buffer_access(core.robust_buffer_access == vk::TRUE),
            )
            .push_next(&mut e11)
            .push_next(&mut e12);
        if api_13 {
            enable = enable.push_next(&mut e13);
        }
        if enabled.coopmat {
            enable = enable.push_next(&mut ecm);
        }
        let mut exts: Vec<*const c_char> = Vec::new();
        if memory_budget {
            exts.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        if enabled.coopmat {
            exts.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        // A device that states it must have it named.
        if has_extension(&device_exts, ash::khr::portability_subset::NAME) {
            exts.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&exts)
            .push_next(&mut enable);
        // SAFETY: every feature enabled was reported supported above, every
        // extension named was found in `device_exts`, `family` is a family
        // of `physical`, and all borrowed create structs outlive the call.
        let device = match unsafe { instance.create_device(physical, &create, None) } {
            Ok(d) => d,
            Err(e) => bail!(Fault::NoDevice {
                detail: format!("vkCreateDevice on {name}: {e}"),
            }),
        };
        // SAFETY: queue 0 of `family` was requested in `create`.
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: `device` is live and `family` is the one it was made with.
        let pool = match unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: the device has no child objects yet and is not used
                // after this.
                unsafe { device.destroy_device(None) };
                bail!(Fault::Vulkan {
                    what: "vkCreateCommandPool",
                    code: e.as_raw(),
                });
            }
        };
        // SAFETY: `pool` is this new device's pool, used by no other thread
        // yet. On failure only the pool exists as a child, and it is destroyed
        // before the device, neither used again.
        let transfer = unsafe {
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map(|v| v[0]);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None);
            match (cmd, fence) {
                (Ok(cmd), Ok(fence)) => Transfer {
                    cmd,
                    fence,
                    staging: None,
                },
                (Err(e), _) | (_, Err(e)) => {
                    device.destroy_command_pool(pool, None);
                    device.destroy_device(None);
                    bail!(Fault::Vulkan {
                        what: "transfer scratch",
                        code: e.as_raw(),
                    });
                }
            }
        };
        // SAFETY: `physical` was enumerated from the live `instance`.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (heap, device_local) = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .enumerate()
            .filter(|(_, h)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|(at, h)| (at, h.size))
            .max_by_key(|&(_, size)| size)
            .unwrap_or((0, 0));
        let mapped_heap = memory.memory_types[..memory.memory_type_count as usize]
            .iter()
            .any(|t| {
                t.heap_index as usize == heap
                    && t.property_flags.contains(
                        vk::MemoryPropertyFlags::DEVICE_LOCAL
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    )
            });
        let unified = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU || mapped_heap;
        let source = if memory_budget {
            BudgetSource::Extension
        } else if unified {
            if let Err(why) = os_available() {
                // SAFETY: the device has its pool, command buffer and fence
                // as children, destroyed before it; none is used again.
                unsafe {
                    device.destroy_fence(transfer.fence, None);
                    device.destroy_command_pool(pool, None);
                    device.destroy_device(None);
                }
                bail!(Fault::NoDevice {
                    detail: format!(
                        "{name} shares the host's memory and has no VK_EXT_memory_budget, so \
                         what it has free is the host's available memory, and {why}"
                    ),
                });
            }
            BudgetSource::OsAvailable
        } else {
            BudgetSource::Heap
        };
        let said = match (source, lmk_reserve()) {
            (BudgetSource::OsAvailable, Some((reserve, from))) => format!(
                "{source} less the {} MiB Android's low-memory killer keeps free ({from})",
                reserve >> 20
            ),
            _ => source.to_string(),
        };
        match budget {
            Some(cap) => tracing::info!(
                device = %name,
                unified,
                "device memory: free memory from {said}, capped by the operator's --budget \
                 of {} MiB",
                cap >> 20
            ),
            None => tracing::info!(
                device = %name,
                unified,
                "device memory: free memory from {said}"
            ),
        }
        let core = Arc::new(Core {
            _entry: entry,
            instance,
            physical,
            device,
            queue: Mutex::new(queue),
            pools: Mutex::new(std::collections::HashMap::from([(
                std::thread::current().id(),
                pool,
            )])),
            spare_fences: Mutex::new(Vec::new()),
            orphans: Mutex::new(Vec::new()),
            family,
            memory,
            limits: props.limits,
            transfer: Mutex::new(transfer),
            books: engine_gpu::Books::new(),
            name: name.clone(),
            subgroup_size: pinned.unwrap_or(subgroup.subgroup_size.max(1)),
            subgroup_min,
            pinned,
            cores,
            device_local,
            heap,
            source,
            unified,
            max_buffer: AtomicU64::new(max_buffer.0),
            max_buffer_limit: Mutex::new(max_buffer.1),
            vendor_id: props.vendor_id,
            discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            api_version: props.api_version,
            features: enabled,
        });

        let dummy = Raw::new(
            &core,
            DUMMY_BYTES,
            Memory::Device,
            engine_gpu::books::owners::DUMMY,
        )?;
        let pipeline_cache = Arc::new(PipelineCache::open(&core, boot.pipeline_cache.clone())?);
        Ok(Context {
            core,
            budget,
            dummy,
            pipeline_cache,
            device_index: boot.device_index,
        })
    }

    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.core
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.core.name
    }

    /// What the device has for this load, measured now: every reservation,
    /// what the driver holds beside them and what is free, capped by the
    /// operator's budget.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        let core = &self.core;
        let measured = core.books.device() + core.outside() + core.free();
        self.budget.map_or(measured, |budget| measured.min(budget))
    }

    /// What the driver holds for this process beside every reservation.
    #[must_use]
    pub fn outside(&self) -> u64 {
        self.core.outside()
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.core.max_buffer().0
    }

    /// Lowers the largest buffer a load makes to `bytes`, as a device whose
    /// limits are that small states it: what a load lays from now on is
    /// laid under it, and an allocation past it refused.
    pub fn limit_buffers(&self, bytes: u64) {
        self.core.max_buffer.fetch_min(bytes, Ordering::Relaxed);
        *self
            .core
            .max_buffer_limit
            .lock()
            .expect("max buffer lock poisoned") = "max_buffer as the boot states it";
    }

    /// Whether the device's memory is the host's ([`Core::unified`]).
    #[must_use]
    pub fn unified_memory(&self) -> bool {
        self.core.unified
    }

    /// Where the device's free memory is read from.
    #[must_use]
    pub fn budget_source(&self) -> BudgetSource {
        self.core.source
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        self.core.cores.unwrap_or(crate::chip::UNREPORTED_CORES)
    }

    /// The alignment a guest table takes in its pool slot: 16 at least, and
    /// what a storage or uniform descriptor's offset asks.
    #[must_use]
    pub fn table_align(&self) -> u64 {
        let limits = &self.core.limits;
        limits
            .min_storage_buffer_offset_alignment
            .max(limits.min_uniform_buffer_offset_alignment)
            .max(engine_gpu::guest::held::TABLE_ALIGN)
    }

    #[must_use]
    pub fn subgroup_size(&self) -> u32 {
        self.core.subgroup_size
    }

    /// The narrowest subgroup any pipeline runs at.
    #[must_use]
    pub fn lanes(&self) -> u32 {
        self.core.subgroup_min
    }

    #[must_use]
    pub fn limits(&self) -> crate::tuning::Limits {
        crate::tuning::Limits {
            cores: self.cores(),
            lanes: self.lanes(),
            shared_bytes: self.core.limits.max_compute_shared_memory_size,
            invocations: self.core.limits.max_compute_work_group_invocations,
        }
    }

    #[must_use]
    pub fn api_version(&self) -> u32 {
        self.core.api_version
    }

    #[must_use]
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    #[must_use]
    pub fn pipeline_cache_path(&self) -> Option<&std::path::Path> {
        self.pipeline_cache.path()
    }

    #[must_use]
    pub fn pipeline_cache(&self) -> &PipelineCache {
        &self.pipeline_cache
    }

    #[must_use]
    pub fn enabled(&self) -> Enabled {
        self.core.features
    }

    #[must_use]
    pub fn chip(&self) -> crate::chip::Device {
        crate::chip::Device {
            name: self.core.name.clone(),
            vendor: profile::vendor::of_pci(self.core.vendor_id),
            subgroup: self.core.subgroup_min,
            cores: self.core.cores,
            coopmat: self.core.features.coopmat,
            discrete: self.core.discrete,
        }
    }

    /// What the physical device answers of sparse binding and of indexing
    /// an array of storage buffers, each under its Vulkan name; this device
    /// enables none of them. With them the sparse address space, and
    /// whether the queue family the engine fires on, or any family at all,
    /// binds sparse memory.
    #[must_use]
    pub fn stated(&self) -> Vec<(&'static str, String)> {
        let core = &self.core;
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let features = {
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
            // SAFETY: `physical` was enumerated from the live `instance`, and
            // the one chained struct is a local that outlives the call.
            unsafe {
                core.instance
                    .get_physical_device_features2(core.physical, &mut query);
            }
            query.features
        };
        // SAFETY: as above.
        let families = unsafe {
            core.instance
                .get_physical_device_queue_family_properties(core.physical)
        };
        let sparse = |family: &vk::QueueFamilyProperties| {
            family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
        };
        let granted = |feature: vk::Bool32| (feature == vk::TRUE).to_string();
        vec![
            ("sparseBinding", granted(features.sparse_binding)),
            (
                "sparseResidencyBuffer",
                granted(features.sparse_residency_buffer),
            ),
            (
                "sparseAddressSpaceSize",
                core.limits.sparse_address_space_size.to_string(),
            ),
            (
                "shaderStorageBufferArrayDynamicIndexing",
                granted(features.shader_storage_buffer_array_dynamic_indexing),
            ),
            (
                "shaderStorageBufferArrayNonUniformIndexing",
                granted(f12.shader_storage_buffer_array_non_uniform_indexing),
            ),
            (
                "the engine's queue family binds sparse memory",
                families
                    .get(core.family as usize)
                    .is_some_and(sparse)
                    .to_string(),
            ),
            (
                "a queue family binds sparse memory",
                families.iter().any(sparse).to_string(),
            ),
        ]
    }

    /// What the capacity has left now, past every reservation and the
    /// driver's own.
    #[must_use]
    pub fn left(&self) -> u64 {
        self.capacity()
            .saturating_sub(self.core.books.device())
            .saturating_sub(self.core.outside())
    }

    /// The device's books: every reservation under its owner.
    #[must_use]
    pub fn books(&self) -> &engine_gpu::Books {
        &self.core.books
    }

    pub fn bind_thread(&self) -> Result<()> {
        Ok(())
    }

    pub fn frame(&self) -> Result<Frame> {
        self.frame_with(true)
    }

    pub fn frame_kept(&self) -> Result<Frame> {
        self.frame_with(false)
    }

    fn frame_with(&self, one_time: bool) -> Result<Frame> {
        let core = &self.core;
        let d = &core.device;
        let pool = core.thread_pool()?;
        // SAFETY: `pool` is this thread's own pool, which no other thread
        // allocates from or frees into, so its external sync holds.
        let cmd = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| core.fault("vkAllocateCommandBuffers", e))?[0];
        let fence = core.lend_fence()?;
        // SAFETY: `cmd` was just allocated on this thread and is in the
        // initial state, so it may be begun; the barrier below records into it
        // while it is still only this function's.
        unsafe {
            let flags = if one_time {
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            } else {
                vk::CommandBufferUsageFlags::empty()
            };
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(flags))
        }
        .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
        // Frames run ahead of one another on the queue: each waits for every
        // write submitted before it.
        // SAFETY: `cmd` is recording and owned here; a global memory barrier
        // across all stages is valid on any queue.
        unsafe {
            let all = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[all],
                &[],
                &[],
            );
        }
        let timing = if crate::encode::profiling() && core.limits.timestamp_period > 0.0 {
            // SAFETY: `d` is live; a nonzero `timestampPeriod` means the
            // device supports timestamp queries.
            let pool = unsafe {
                d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(TIMED_DISPATCHES * 2),
                    None,
                )
            }
            .map_err(|e| core.fault("vkCreateQueryPool", e))?;
            // SAFETY: `cmd` is recording outside a render pass, and the range is
            // exactly the pool's `TIMED_DISPATCHES * 2` queries.
            unsafe { d.cmd_reset_query_pool(cmd, pool, 0, TIMED_DISPATCHES * 2) };
            Some(Timing {
                pool,
                names: Vec::new(),
                capacity: TIMED_DISPATCHES,
                harvested: false,
            })
        } else {
            None
        };
        Ok(Frame {
            inner: Arc::new(Inner {
                core: Arc::clone(core),
                cmd: std::sync::atomic::AtomicU64::new(vk::Handle::as_raw(cmd)),
                cuts: Mutex::new(Vec::new()),
                cmd_pool: pool,
                fence,
                pools: Mutex::new(Vec::new()),
                keep: Mutex::new(Vec::new()),
                submitted: std::sync::atomic::AtomicBool::new(false),
                timing: Mutex::new(timing),
                spare: Mutex::new(std::collections::HashMap::new()),
            }),
            open: true,
            kept: !one_time,
            hazards: std::cell::RefCell::new(Hazards::default()),
            retained: std::cell::RefCell::new(std::collections::HashSet::new()),
            dispatches: std::cell::Cell::new(0),
        })
    }
}

fn pick_family(families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    let dedicated = families.iter().position(|q| {
        q.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    });
    dedicated
        .or_else(|| {
            families
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        })
        .map(|i| i as u32)
}

pub(crate) struct Inner {
    pub(crate) core: Arc<Core>,
    /// The command buffer being recorded (a kept frame's last segment).
    cmd: std::sync::atomic::AtomicU64,
    /// A kept frame's segments before its last, in order: each ended where
    /// the walk needed the frame so far landed before the host went on (an
    /// expert stage); a replay submits each, waits, and lets its caller do
    /// that host work before the next.
    cuts: Mutex<Vec<vk::CommandBuffer>>,

    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) fence: vk::Fence,

    pub(crate) pools: Mutex<Vec<vk::DescriptorPool>>,

    pub(crate) keep: Mutex<Vec<Slab>>,
    submitted: std::sync::atomic::AtomicBool,

    pub(crate) timing: Mutex<Option<Timing>>,

    spare: Mutex<std::collections::HashMap<u64, Vec<vk::DescriptorSet>>>,
}

impl Inner {
    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        vk::Handle::from_raw(self.cmd.load(Ordering::Acquire))
    }
}

pub(crate) struct Timing {
    pub(crate) pool: vk::QueryPool,
    pub(crate) names: Vec<&'static str>,
    pub(crate) capacity: u32,
    pub(crate) harvested: bool,
}

pub(crate) const TIMED_DISPATCHES: u32 = 8192;

impl Drop for Inner {
    fn drop(&mut self) {
        let d = &self.core.device;
        // SAFETY: this is the last `Arc<Inner>`, so nothing else records into
        // `cmd` or allocates from its descriptor pools. A submitted frame is
        // waited on first, so the pools, queries and buffer are no longer in
        // use (a wait that times out after two minutes is taken as a lost
        // device). The command buffer is freed only on the thread whose pool
        // it came from, and otherwise left to that thread as an orphan.
        unsafe {
            if self.submitted.load(Ordering::Acquire) {
                let _ = d.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS);
            }
            for pool in self.pools.get_mut().expect("pools lock poisoned").drain(..) {
                d.destroy_descriptor_pool(pool, None);
            }
            if let Some(timing) = self.timing.get_mut().expect("timing lock poisoned").take() {
                d.destroy_query_pool(timing.pool, None);
            }
            let owned = self
                .core
                .pools
                .lock()
                .expect("pools lock poisoned")
                .get(&std::thread::current().id())
                == Some(&self.cmd_pool);
            let mut buffers = std::mem::take(self.cuts.get_mut().expect("cuts lock poisoned"));
            buffers.push(self.cmd());
            if owned {
                d.free_command_buffers(self.cmd_pool, &buffers);
            } else {
                let mut orphans = self.core.orphans.lock().expect("orphans lock poisoned");
                for cmd in buffers {
                    orphans.push((self.cmd_pool, cmd));
                }
            }
        }
        self.core.return_fence(self.fence);
    }
}

const SETS_PER_POOL: u32 = 1024;

const SETS_PER_BATCH: usize = 64;
const BUFFERS_PER_POOL: u32 = SETS_PER_POOL * 8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Span {
    buffer: u64,
    start: u64,
    end: u64,
}

impl Span {
    pub(crate) fn new(buffer: vk::Buffer, start: u64, len: u64) -> Span {
        use ash::vk::Handle;
        Span {
            buffer: buffer.as_raw(),
            start,
            end: start.saturating_add(len),
        }
    }

    fn overlaps(self, other: Span) -> bool {
        self.buffer == other.buffer && self.start < other.end && other.start < self.end
    }
}

const HAZARD_CAP: usize = 96;

#[derive(Default)]
pub(crate) struct Hazards {
    reads: Vec<Span>,
    writes: Vec<Span>,

    barriers: u64,
    seen: u64,
}

impl Hazards {
    fn owed(&self, reads: &[Span], writes: &[Span]) -> bool {
        if self.reads.len() + self.writes.len() > HAZARD_CAP {
            return true;
        }

        if reads
            .iter()
            .any(|r| self.writes.iter().any(|w| r.overlaps(*w)))
        {
            return true;
        }

        writes.iter().any(|w| {
            self.writes.iter().any(|p| w.overlaps(*p)) || self.reads.iter().any(|p| w.overlaps(*p))
        })
    }

    fn extend(&mut self, reads: &[Span], writes: &[Span]) {
        self.reads.extend_from_slice(reads);
        self.writes.extend_from_slice(writes);
        self.seen += 1;
    }

    fn cleared(&mut self) {
        self.reads.clear();
        self.writes.clear();
    }
}

pub struct Frame {
    pub(crate) inner: Arc<Inner>,
    open: bool,
    /// Opened by `frame_kept`: a flush cuts a segment rather than reusing
    /// the command buffer, so the whole frame replays.
    kept: bool,
    pub(crate) dispatches: std::cell::Cell<u64>,

    pub(crate) hazards: std::cell::RefCell<Hazards>,

    retained: std::cell::RefCell<std::collections::HashSet<usize>>,
}

impl Frame {
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.inner.core
    }

    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        self.inner.cmd()
    }

    pub(crate) fn barrier_owed(&self, reads: &[Span], writes: &[Span]) -> bool {
        let mut hazards = self.hazards.borrow_mut();
        let owed = hazards.owed(reads, writes);
        if owed {
            hazards.cleared();
            hazards.barriers += 1;
        }
        hazards.extend(reads, writes);
        owed
    }

    fn hazards_settled(&self) {
        self.hazards.borrow_mut().cleared();
    }

    pub(crate) fn retain(&self, slab: &Slab) {
        let key = Arc::as_ptr(slab) as usize;
        if !self.retained.borrow_mut().insert(key) {
            return;
        }
        self.inner
            .keep
            .lock()
            .expect("keep lock poisoned")
            .push(Arc::clone(slab));
    }

    pub(crate) fn descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        let core = &self.inner.core;
        let d = &core.device;
        let key = layout.as_raw();
        {
            let mut spare = self.inner.spare.lock().expect("spare lock poisoned");
            if let Some(set) = spare.get_mut(&key).and_then(Vec::pop) {
                return Ok(set);
            }
        }
        let layouts = [layout; SETS_PER_BATCH];
        let mut pools = self.inner.pools.lock().expect("pools lock poisoned");
        let mut got: Option<Vec<vk::DescriptorSet>> = None;
        if let Some(&pool) = pools.last() {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: the `pools` lock is held, which is the external sync
            // the descriptor pool needs; `layout` is a live layout of `d`.
            got = unsafe { d.allocate_descriptor_sets(&info) }.ok();
        }
        let mut sets = match got {
            Some(sets) => sets,
            None => {
                let sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(BUFFERS_PER_POOL)];
                // SAFETY: `d` is live and the create info is complete.
                let pool = unsafe {
                    d.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(SETS_PER_POOL)
                            .pool_sizes(&sizes),
                        None,
                    )
                }
                .map_err(|e| core.fault("vkCreateDescriptorPool", e))?;
                pools.push(pool);
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts);
                // SAFETY: as above, under the `pools` lock, from the fresh pool.
                unsafe { d.allocate_descriptor_sets(&info) }
                    .map_err(|e| core.fault("vkAllocateDescriptorSets", e))?
            }
        };
        drop(pools);
        let one = sets.pop().expect("a batch of one or more");
        if !sets.is_empty() {
            self.inner
                .spare
                .lock()
                .expect("spare lock poisoned")
                .entry(key)
                .or_default()
                .extend(sets);
        }
        Ok(one)
    }

    pub fn copy(
        &mut self,
        source: &Buffer,
        source_at: u64,
        into: &Buffer,
        into_at: u64,
        len: u64,
    ) -> Result<()> {
        self.copy_regions(source, into, &[(source_at, into_at, len)])
    }

    /// Copies each `(source_at, into_at, len)` region of `source` into
    /// `into` under one pair of barriers; a host-mapped `into` is made
    /// visible to the host once the frame's fence signals.
    pub fn copy_regions(
        &mut self,
        source: &Buffer,
        into: &Buffer,
        regions: &[(u64, u64, u64)],
    ) -> Result<()> {
        let regions: Vec<vk::BufferCopy> = regions
            .iter()
            .filter(|&&(_, _, len)| len > 0)
            .map(|&(source_at, into_at, len)| {
                source.span(source_at, len)?;
                into.span(into_at, len)?;
                Ok(vk::BufferCopy::default()
                    .src_offset(source_at)
                    .dst_offset(into_at)
                    .size(len))
            })
            .collect::<Result<_>>()?;
        if regions.is_empty() {
            return Ok(());
        }
        let (read_back, source, into) = (into.is_host(), source.slab(), into.slab());
        let d = &self.inner.core.device;
        let cmd = self.inner.cmd();
        // SAFETY: `&mut self` is the only recorder of this open frame's `cmd`.
        // Both slabs are live buffers of this device and every region was
        // checked to lie inside them; the frame retains both below until its
        // fence signals.
        unsafe {
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_copy_buffer(cmd, source.buffer, into.buffer, &regions);
            let (host_stage, host_read) = if read_back {
                (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ)
            } else {
                (vk::PipelineStageFlags::empty(), vk::AccessFlags::empty())
            };
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | host_read,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | host_stage,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        self.retain(source);
        self.retain(into);
        self.hazards_settled();
        Ok(())
    }

    pub fn fill_zero(&mut self, into: &Buffer, at: u64, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        into.span(at, len)?;
        let into = into.slab();
        let d = &self.inner.core.device;
        let cmd = self.inner.cmd();
        // SAFETY: `&mut self` is the only recorder of this open frame's `cmd`;
        // the slab is live and retained below, and `span` bounds the fill. The
        // rounding of `len` to 4 stays inside the slab (a multiple of 4 long)
        // only for a 4-aligned `at`, which `vkCmdFillBuffer` requires and
        // callers keep, but which is not checked here.
        unsafe {
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_fill_buffer(cmd, into.buffer, at, len.next_multiple_of(4), 0);
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        self.retain(into);
        self.hazards_settled();
        Ok(())
    }

    #[must_use]
    pub fn dispatches(&self) -> u64 {
        self.dispatches.get()
    }

    /// The segments a kept frame's flushes have cut so far.
    #[must_use]
    pub fn cuts(&self) -> usize {
        self.inner.cuts.lock().expect("cuts lock poisoned").len()
    }

    /// [`Frame::flush`], then the queue idle: every frame submitted before
    /// this one has landed too (an earlier fire still in flight among them),
    /// so the host may rewrite memory any of them reads.
    pub fn flush_settled(&self) -> Result<()> {
        self.flush()?;
        let core = &self.inner.core;
        let queue = core.queue.lock().expect("queue lock poisoned");
        // SAFETY: the queue is this device's, and its lock is the external
        // sync `vkQueueWaitIdle` needs.
        unsafe { core.device.queue_wait_idle(*queue) }.map_err(|e| core.fault("vkQueueWaitIdle", e))
    }

    pub fn flush(&self) -> Result<()> {
        let core = &self.inner.core;
        let d = &core.device;
        let cmd = self.inner.cmd();
        // In `flush`, `Frame` is not `Sync`, so `&self` here is the only
        // thread using this frame's `cmd` and fence, and the frame is open:
        // every method that ends it consumes the frame.
        // SAFETY: `cmd` is recording, so it may be ended.
        unsafe { d.end_command_buffer(cmd) }.map_err(|e| core.fault("vkEndCommandBuffer", e))?;
        {
            let bufs = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // SAFETY: `cmd` is executable, the fence is unsignaled (reset after
            // each wait) and the queue is held under its lock.
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
        }
        // SAFETY: the fence is this frame's own, just submitted.
        unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
            .map_err(|e| core.fault("vkWaitForFences", e))?;
        // SAFETY: the wait returned success, so no submission names the fence.
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        if self.kept {
            // The segment stays for replays; the frame records on in a new
            // buffer from the same (this thread's) pool.
            // SAFETY: `cmd_pool` is the pool this frame's buffers came from,
            // owned by this thread (frames record on the thread that opened
            // them).
            let next = unsafe {
                d.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.inner.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|e| core.fault("vkAllocateCommandBuffers", e))?[0];
            self.inner
                .cuts
                .lock()
                .expect("cuts lock poisoned")
                .push(cmd);
            self.inner
                .cmd
                .store(vk::Handle::as_raw(next), Ordering::Release);
            // SAFETY: `next` was just allocated, in the initial state.
            unsafe { d.begin_command_buffer(next, &vk::CommandBufferBeginInfo::default()) }
                .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
            self.hazards_settled();
            return Ok(());
        }
        // SAFETY: the work is complete, so `cmd` is not pending; its pool was
        // made with `RESET_COMMAND_BUFFER`.
        unsafe { d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| core.fault("vkResetCommandBuffer", e))?;
        // SAFETY: `cmd` was just reset to the initial state.
        unsafe {
            d.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
        self.hazards_settled();
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        self.open = false;
        let core = &self.inner.core;
        // SAFETY: `open` was set, so `cmd` is recording; `&mut self` is its
        // only user.
        unsafe { core.device.end_command_buffer(self.inner.cmd()) }
            .map_err(|e| core.fault("vkEndCommandBuffer", e))
    }

    fn submit(&mut self) -> Result<()> {
        self.end()?;
        let core = &self.inner.core;
        let bufs = [self.inner.cmd()];
        let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
        let queue = core.queue.lock().expect("queue lock poisoned");
        // SAFETY: `cmd` was just ended and never submitted, the fence is the
        // frame's own unsignaled one, and the queue is held under its lock.
        unsafe { core.device.queue_submit(*queue, &submits, self.inner.fence) }
            .map_err(|e| core.fault("vkQueueSubmit", e))?;
        self.inner.submitted.store(true, Ordering::Release);
        Ok(())
    }

    pub fn commit(mut self) -> Result<()> {
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()
    }

    pub fn commit_timed(mut self) -> Result<f64> {
        let started = std::time::Instant::now();
        self.submit()?;
        Pending {
            inner: Arc::clone(&self.inner),
        }
        .wait()?;
        Ok(started.elapsed().as_secs_f64())
    }

    pub fn into_kept(mut self) -> Result<Kept> {
        self.end()?;
        Ok(Kept {
            inner: Arc::clone(&self.inner),
        })
    }

    pub fn commit_async(mut self, on_done: Option<engine_gpu::backend::OnDone>) -> Result<Pending> {
        let at = crate::encode::profiling().then(std::time::Instant::now);
        self.submit()?;
        if let Some(at) = at {
            crate::encode::credit_submit(at.elapsed().as_nanos() as u64);
        }
        let pending = Pending {
            inner: Arc::clone(&self.inner),
        };
        if let Some(on_done) = on_done {
            let watched = Pending {
                inner: Arc::clone(&self.inner),
            };
            std::thread::spawn(move || {
                on_done(watched.wait().err().map(|fault| fault.to_string()));
            });
        }
        Ok(pending)
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        if self.open {
            self.open = false;
            // SAFETY: an open frame's `cmd` is recording and only this frame
            // records into it.
            unsafe {
                let _ = self.inner.core.device.end_command_buffer(self.inner.cmd());
            }
        }
    }
}

pub struct Kept {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept").finish()
    }
}

impl Kept {
    /// The segments before the last ([`Frame::cuts`] at `into_kept`).
    #[must_use]
    pub fn cuts(&self) -> usize {
        self.inner.cuts.lock().expect("cuts lock poisoned").len()
    }

    /// Submits segment `at` (of [`Kept::cuts`]) and waits for it: with it
    /// everything submitted before has landed (a fence's signal covers all
    /// earlier submissions), so the host may rewrite what the next segment
    /// reads. Segments replay in order, then [`Kept::replay`] the last.
    pub fn replay_cut(&self, at: usize) -> Result<()> {
        let core = &self.inner.core;
        let d = &core.device;
        let cmd = *self
            .inner
            .cuts
            .lock()
            .expect("cuts lock poisoned")
            .get(at)
            .ok_or_else(|| Fault::Unbound {
                what: format!("segment {at} of a kept frame"),
            })?;
        if self.inner.submitted.load(Ordering::Acquire) {
            // SAFETY: the fence is this kept frame's own.
            unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
                .map_err(|e| core.fault("vkWaitForFences", e))?;
        }
        // SAFETY: no pending submission names the fence (waited above, or
        // the last one was a segment, waited and reset below).
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        self.inner.submitted.store(false, Ordering::Release);
        {
            let bufs = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // SAFETY: `cmd` is executable (ended when the next segment
            // began) and not pending; the fence is unsignaled; the queue is
            // held under its lock.
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
        }
        // SAFETY: the fence is this frame's own, just submitted.
        unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
            .map_err(|e| core.fault("vkWaitForFences", e))?;
        // SAFETY: the wait returned, so no submission names the fence.
        unsafe { d.reset_fences(&[self.inner.fence]) }.map_err(|e| core.fault("vkResetFences", e))
    }

    pub fn replay(
        &self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        let core = &self.inner.core;
        let d = &core.device;
        // In `replay`, the kept `cmd` is executable, ended by `into_kept`; its
        // one caller keeps only frames opened by `frame_kept`, so it was
        // recorded without `ONE_TIME_SUBMIT` (`into_kept` does not check
        // this). It also relies on one replay at a
        // time per `Kept`: `Kept` is `Sync` and nothing here stops two threads
        // from submitting the same buffer and fence at once; the serve loop
        // replays from one thread only.
        if self.inner.submitted.load(Ordering::Acquire) {
            // SAFETY: the fence is this kept frame's own.
            unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
                .map_err(|e| core.fault("vkWaitForFences", e))?;
        }
        // SAFETY: any previous submission has signaled (waited above), so no
        // pending submission names the fence.
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        let at = crate::encode::profiling().then(std::time::Instant::now);
        {
            let bufs = [self.inner.cmd()];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // SAFETY: `cmd` is executable and no longer pending, the fence is
            // unsignaled, and the queue is held under its lock.
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
        }
        self.inner.submitted.store(true, Ordering::Release);
        if let Some(at) = at {
            crate::encode::credit_submit(at.elapsed().as_nanos() as u64);
        }
        // A frame recorded while profiling carries its timestamp queries
        // (the pool's reset and a write around each dispatch): each replay
        // writes them again, so each replay's wait reads them again while
        // profiling.
        if crate::encode::profiling()
            && let Some(timing) = self
                .inner
                .timing
                .lock()
                .expect("timing lock poisoned")
                .as_mut()
        {
            timing.harvested = false;
        }
        let pending = Pending {
            inner: Arc::clone(&self.inner),
        };
        if let Some(on_done) = on_done {
            let watched = Pending {
                inner: Arc::clone(&self.inner),
            };
            std::thread::spawn(move || {
                on_done(watched.wait().err().map(|fault| fault.to_string()));
            });
        }
        Ok(pending)
    }
}

pub struct Pending {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish()
    }
}

impl Pending {
    #[must_use]
    pub fn landed(&self) -> bool {
        // SAFETY: the fence lives as long as `inner`, which `self` holds;
        // querying a fence needs no external sync.
        unsafe { self.inner.core.device.get_fence_status(self.inner.fence) }.unwrap_or(true)
    }

    pub fn wait(&self) -> Result<()> {
        let core = &self.inner.core;
        let at = crate::encode::profiling().then(std::time::Instant::now);
        // SAFETY: the fence lives as long as `inner`, which `self` holds;
        // waiting needs no external sync, so concurrent waiters are fine.
        unsafe {
            core.device
                .wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS)
        }
        .map_err(|e| core.fault("vkWaitForFences", e))?;
        if let Some(at) = at {
            crate::encode::credit_fence(at.elapsed().as_nanos() as u64);
        }
        self.inner.harvest_timing();
        Ok(())
    }
}

impl Inner {
    fn harvest_timing(&self) {
        let mut slot = self.timing.lock().expect("timing lock poisoned");
        let Some(timing) = slot.as_mut() else {
            return;
        };
        if timing.harvested || timing.names.is_empty() {
            return;
        }
        timing.harvested = true;
        let n = timing.names.len() as u32;
        let mut ticks = vec![0u64; (n * 2) as usize];
        // SAFETY: the caller waited on the fence, so every written query is
        // available; `n * 2` queries fit the pool (`names` stops at
        // `capacity`), and `ticks` holds one u64 for each with `TYPE_64`.
        let ok = unsafe {
            self.core.device.get_query_pool_results(
                timing.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .is_ok();
        if !ok {
            return;
        }
        let period = f64::from(self.core.limits.timestamp_period);

        if let (Some(first), Some(last)) = (ticks.iter().min(), ticks.iter().max()) {
            crate::encode::credit_frame_span((last.saturating_sub(*first) as f64 * period) as u64);
            let _ = first;
        }
        for (at, name) in timing.names.iter().enumerate() {
            let span = ticks[at * 2 + 1].saturating_sub(ticks[at * 2]);
            let ns = (span as f64 * period) as u64;
            crate::encode::credit(name, ns);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_available_reads_kilobytes_as_bytes() {
        let text = "MemTotal:       11534336 kB\nMemFree:  1000 kB\nMemAvailable:    5242880 kB\n";
        assert_eq!(mem_available(text), Some(5_242_880 * 1024));
        assert_eq!(mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(mem_available("MemAvailable: 12 MB\n"), None);
    }

    /// lmkd's levels on a phone: a foreground app is killed under 18432
    /// free pages; the levels past adj 0 reach only background apps.
    #[test]
    fn the_lmk_reserve_is_the_level_that_reaches_the_foreground() {
        let levels = "18432:0,23040:100,27648:200,32256:250,55296:900,80640:950";
        assert_eq!(foreground_minfree(levels), Some(18432));
        assert_eq!(foreground_minfree("4096:-1000,8192:0,9000:100"), Some(8192));
        assert_eq!(foreground_minfree("23040:100"), None);
        assert_eq!(foreground_minfree(""), None);
    }

    #[test]
    fn the_buffer_ceiling_names_the_limit_that_sets_it() {
        // An Adreno 840: 1 GiB allocations under a 2 GiB - 1 storage range.
        assert_eq!(
            buffer_ceiling(u32::MAX >> 1, 1 << 30, Some(1 << 32)),
            (1 << 30, "maxMemoryAllocationSize")
        );
        assert_eq!(
            buffer_ceiling(1 << 27, 1 << 30, None),
            (1 << 27, "maxStorageBufferRange")
        );
        assert_eq!(
            buffer_ceiling(u32::MAX, 0, Some(1 << 31)),
            (1 << 31, "maxBufferSize")
        );
    }
}
