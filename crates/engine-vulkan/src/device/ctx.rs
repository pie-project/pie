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

    /// The descriptor pools frames bind through: those no frame holds,
    /// reset, and what is lent ([`Spares`]). A frame takes its pools here
    /// and gives them back once its fence passed; serving makes none
    /// ([`Context::seal_descriptor_pools`]).
    spares: Mutex<Spares>,
    /// Descriptor sets shared for the device's life ([`Frame::shared_set`],
    /// [`Frame::fire_set`]), from pools of their own.
    shared: Mutex<SharedSets>,
    /// Woken as pools come back.
    given: std::sync::Condvar,

    /// The one thread that waits on every watched submission's fence, in
    /// the queue's order, started with the device ([`Watch`]).
    watcher: Mutex<Option<std::sync::mpsc::Sender<Watch>>>,
    /// The submission that went to the queue last: the queue has emptied
    /// once it landed ([`Context::idle`]).
    last: Mutex<Last>,

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
    /// Where the device shares the host's memory and no budget says what
    /// it has, the most the load may hold in all, read when the device
    /// bound ([`host_ceiling`]).
    ceiling: Option<u64>,
    /// Whose memory the device's is ([`device_memory`]): where it is the
    /// host's, the load states no host tier.
    whose_memory: engine::DeviceMemory,
    /// The largest device-local heap, which device reservations take.
    heap: usize,
    pub(crate) vendor_id: u32,
    /// The device writes storage-buffer descriptors after a recording binds
    /// them (descriptorBindingStorageBufferUpdateAfterBind): a recording's
    /// pool segments are rebound as they are backed and given back.
    pub(crate) update_after_bind: bool,
    /// The pool's segment buffers and their stand-ins, by raw handle, each
    /// to its segment ([`Context::rebind_segments`]): a kept frame notes each
    /// descriptor that binds one, to write it when the segment changes.
    pub(crate) segments: Mutex<std::collections::HashMap<u64, u32>>,
    /// What each segment's descriptors bind now, by raw handle: a rebind
    /// writes only the segments whose buffer changed.
    pub(crate) segments_bound: Mutex<Vec<u64>>,
    /// Every kept frame that binds a pool segment, which a rebind writes.
    pub(crate) rebound: Mutex<Vec<std::sync::Weak<Inner>>>,
    /// The device's PCI id and its driver's version, as the physical
    /// device states them: with the vendor, what a stored measurement is
    /// keyed by ([`Context::measured_on`]).
    pub(crate) device_id: u32,
    pub(crate) driver_version: u32,
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

    /// The device-local bytes free to this process now: the heap's budget
    /// less its usage where the driver reports them, else the heap less
    /// every reservation (blind to other processes); where the device
    /// shares the host's memory, no more than the ceiling read at bind less
    /// every reservation.
    pub(crate) fn free(&self) -> u64 {
        let reserved = engine_gpu::books::process_reserved();
        let free = match self.budget() {
            Some((budget, usage)) => budget.saturating_sub(usage),
            None => self.device_local.saturating_sub(reserved),
        };
        self.ceiling
            .map_or(free, |ceiling| free.min(ceiling.saturating_sub(reserved)))
    }

    /// What the driver holds for this process in the heap beside every
    /// reservation of this process: its usage less them, where it reports it.
    pub(crate) fn outside(&self) -> u64 {
        self.budget().map_or(0, |(_, usage)| {
            usage.saturating_sub(engine_gpu::books::process_reserved())
        })
    }

    pub(crate) fn fault(&self, call: &'static str, code: vk::Result) -> Fault {
        if matches!(
            code,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY
                | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                | vk::Result::ERROR_OUT_OF_POOL_MEMORY
                | vk::Result::ERROR_FRAGMENTED_POOL
                | vk::Result::ERROR_FRAGMENTATION
        ) {
            SPENT.store(true, Ordering::Release);
        }
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
            transfer.staging = Some(Raw::new(
                self,
                STAGING_BYTES,
                Memory::Staging,
                engine_gpu::books::owners::STAGING,
            )?);
        }
        Ok(transfer.staging.as_ref().expect("just made"))
    }
}

const FENCE_TIMEOUT_NS: u64 = 120_000_000_000;

impl Core {
    /// The bits of a timestamp the queue's family writes: 0 where it writes
    /// none (or the device states no `timestampPeriod`).
    fn stamp_bits(&self) -> u32 {
        if self.limits.timestamp_period <= 0.0 {
            return 0;
        }
        // SAFETY: `physical` is the instance's live physical device.
        let families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical)
        };
        families
            .get(self.family as usize)
            .map_or(0, |family| family.timestamp_valid_bits)
    }

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

/// The descriptor pools: `free` reset and held by no frame, `lent` held by
/// frames, `kept` of those by kept frames (which replay, so hold theirs
/// until they drop); `sealed` from the start of serving, after which a
/// one-time frame (every frame a served step submits) takes only these and
/// a kept one (a recording, built and counted as one) makes its own, with
/// `stock` the pools the load laid.
/// Descriptor sets the device shares for its life: a kept frame's set by
/// its layout and what it binds, the fire block's set by its layout and
/// buffer.
#[derive(Debug, Default)]
pub(crate) struct SharedSets {
    sets: std::collections::HashMap<(u64, u64), vk::DescriptorSet>,
    pools: Vec<vk::DescriptorPool>,
}

/// Command buffers frames hold now, and frames alive (each holding a
/// fence): named on the arm walk's progress (a driver that caps them fails
/// past a count, not a size).
pub(crate) static LIVE_CMDS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
pub(crate) static LIVE_FRAMES: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// The driver ran out of what frames take (memory, pool room, command
/// buffers) since the last [`take_spent`]: set where the call failed, read by
/// the caller whatever its error became on the way up.
static SPENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the driver ran out of what frames take since the last call.
pub fn take_spent() -> bool {
    SPENT.swap(false, Ordering::AcqRel)
}

/// What the device holds for frames now: command buffers, frames, shared
/// descriptor sets, descriptor pools made and freed.
#[must_use]
pub fn held_counts() -> (i64, i64, u64, u64, u64) {
    (
        LIVE_CMDS.load(Ordering::Relaxed),
        LIVE_FRAMES.load(Ordering::Relaxed),
        SHARED_SETS.load(Ordering::Relaxed),
        POOLS_MADE.load(Ordering::Relaxed),
        POOLS_FREED.load(Ordering::Relaxed),
    )
}

impl Core {
    /// One primary command buffer from `pool` (this thread's), refused by
    /// name where the driver gives none, or a null one, back.
    pub(crate) fn alloc_cmd(&self, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
        // SAFETY: `pool` is the calling thread's own pool, which no other
        // thread allocates from or frees into, so its external sync holds.
        let got = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| self.fault("vkAllocateCommandBuffers", e))?;
        match got.first().copied() {
            Some(cmd) if cmd != vk::CommandBuffer::null() => {
                LIVE_CMDS.fetch_add(1, Ordering::Relaxed);
                Ok(cmd)
            }
            _ => Err({
                SPENT.store(true, Ordering::Release);
                Fault::Ceiling {
                what: "command buffers: the driver allocated a null one",
                need: u64::try_from(LIVE_CMDS.load(Ordering::Relaxed)).unwrap_or(0) + 1,
                have: u64::try_from(LIVE_CMDS.load(Ordering::Relaxed)).unwrap_or(0),
            }}),
        }
    }
}

impl Core {
    /// The set shared under `key` of `layout`, allocated where none is yet;
    /// whether it was.
    fn shared_set(
        &self,
        layout: vk::DescriptorSetLayout,
        key: u64,
    ) -> Result<(vk::DescriptorSet, bool)> {
        use ash::vk::Handle;
        let mut shared = self.shared.lock().expect("shared lock poisoned");
        let at = (layout.as_raw(), key);
        if let Some(&set) = shared.sets.get(&at) {
            return Ok((set, false));
        }
        let layouts = [layout];
        let mut got = None;
        if let Some(&pool) = shared.pools.last() {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: the `shared` lock is the pool's external sync; `layout`
            // is a live layout of this device.
            got = unsafe { self.device.allocate_descriptor_sets(&info) }.ok();
        }
        let set = match got {
            Some(sets) => sets[0],
            None => {
                let pool = self.new_descriptor_pool_of(SETS_PER_POOL)?;
                shared.pools.push(pool);
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts);
                // SAFETY: as above, from the fresh pool.
                unsafe { self.device.allocate_descriptor_sets(&info) }
                    .map_err(|e| self.fault("vkAllocateDescriptorSets", e))?[0]
            }
        };
        SHARED_SETS.fetch_add(1, Ordering::Relaxed);
        shared.sets.insert(at, set);
        Ok((set, true))
    }
}

#[derive(Debug, Default)]
pub(crate) struct Spares {
    free: Vec<vk::DescriptorPool>,
    lent: usize,
    kept: usize,
    sealed: bool,
    stock: usize,
}

impl Core {
    /// A reset descriptor pool for a frame that holds `mine` already (a
    /// `one_time` frame gives them back once its fence passed; a kept one
    /// when it drops). Before serving, a new one where none is free. While
    /// serving, a kept frame's own, and a one-time frame's a free one,
    /// waited for while a one-time frame other than this one holds pools
    /// (each lands, so gives them back), and refused by name where none can
    /// come back.
    fn descriptor_pool(&self, one_time: bool, mine: usize) -> Result<vk::DescriptorPool> {
        let mut spares = self.spares.lock().expect("spares lock poisoned");
        loop {
            // A kept frame (a recording) holds its pools for its life: pools
            // of its own, a few sets each, so a body holds about the sets its
            // dispatches bind and no stock of a thousand.
            let pool = if !one_time {
                Some(self.new_descriptor_pool_of(KEPT_SETS_PER_POOL)?)
            } else {
                match spares.free.pop() {
                    Some(pool) => Some(pool),
                    None if !spares.sealed => Some(self.new_descriptor_pool()?),
                    None => None,
                }
            };
            if let Some(pool) = pool {
                spares.lent += 1;
                if !one_time {
                    spares.kept += 1;
                }
                return Ok(pool);
            }
            let others = spares.lent - spares.kept - if one_time { mine } else { 0 };
            if others == 0 {
                return Err(Fault::Ceiling {
                    what: "descriptor pools while serving: no spare descriptor pool, and no frame \
                           in flight holds one to give back (the load stocked them)",
                    need: spares.lent as u64 + 1,
                    have: spares.stock as u64,
                });
            }
            spares = self.given.wait(spares).expect("spares lock poisoned");
        }
    }

    /// `pools`, reset, back to the spares from a frame (`one_time`, else a
    /// kept one).
    fn give_pools(&self, pools: Vec<vk::DescriptorPool>, one_time: bool) {
        if pools.is_empty() {
            return;
        }
        if !one_time {
            // A kept frame's own pools go with it.
            for &pool in &pools {
                // SAFETY: the kept frame holding the pool is gone, and a
                // submission naming its sets waited its fence before then.
                unsafe { self.device.destroy_descriptor_pool(pool, None) };
            }
            POOLS_FREED.fetch_add(pools.len() as u64, Ordering::Relaxed);
            let mut spares = self.spares.lock().expect("spares lock poisoned");
            spares.lent -= pools.len();
            spares.kept -= pools.len();
            return;
        }
        for &pool in &pools {
            // SAFETY: the frame that held `pool` landed (its fence passed) or
            // was never submitted, so no command uses its sets; a pool is
            // reset by its one holder.
            let _ = unsafe {
                self.device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            };
        }
        let mut spares = self.spares.lock().expect("spares lock poisoned");
        spares.lent -= pools.len();
        if !one_time {
            spares.kept -= pools.len();
        }
        spares.free.extend(pools);
        self.given.notify_all();
    }

    fn new_descriptor_pool(&self) -> Result<vk::DescriptorPool> {
        self.new_descriptor_pool_of(SETS_PER_POOL)
    }

    /// A descriptor pool of `sets` sets, each binding up to 8 storage
    /// buffers and a uniform on average.
    fn new_descriptor_pool_of(&self, sets: u32) -> Result<vk::DescriptorPool> {
        POOLS_MADE.fetch_add(1, Ordering::Relaxed);
        // A set binds its kernel's scalars as one uniform (the fire block's).
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(sets * 8),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(sets),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(sets),
        ];
        // A recording's pool segments are written after it binds them: its
        // sets are allocated from pools that allow it.
        let flags = if self.update_after_bind {
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
        } else {
            vk::DescriptorPoolCreateFlags::empty()
        };
        // SAFETY: `device` is live and the create info is complete.
        unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(flags)
                    .max_sets(sets)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| self.fault("vkCreateDescriptorPool", e))
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
            for pool in
                std::mem::take(&mut self.spares.get_mut().expect("spares lock poisoned").free)
            {
                self.device.destroy_descriptor_pool(pool, None);
            }
            for pool in std::mem::take(&mut self.shared.get_mut().expect("shared lock poisoned").pools) {
                self.device.destroy_descriptor_pool(pool, None);
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
    tracing::warn!(severity = severity.as_raw(), "vulkan validation: {message}");
    vk::FALSE
}

pub struct Context {
    pub(crate) core: Arc<Core>,
    /// The operator's cap on what the load takes, all of it together.
    budget: Option<u64>,
    max_buffer: u64,
    pub(crate) dummy: Slab,
    pipeline_cache: Arc<PipelineCache>,
    device_index: u32,
    /// The kernel a frame moves a fire's readout rows with
    /// ([`Frame::copy_rows`]), built with the device.
    copier: Arc<super::pipelines::Pipeline>,
    /// The dispatches the next frame makes indirect, by their place in its
    /// walk ([`Context::indirect_next`]).
    indirect_next: Mutex<Option<Arc<Vec<bool>>>>,
    /// The fire block the next frame opened lays in ([`Context::arm_next`]).
    arm_next: Mutex<Option<Arc<super::block::ArmBlock>>>,
    /// The score room the next frame's capturing lanes lay their rows in
    /// ([`Context::room_next`]): its buffer and byte.
    room_next: Mutex<Option<(vk::Buffer, u64)>>,
    /// Whether the next frame's rows are constants of its commands
    /// ([`Context::fixed_next`]).
    fixed_next: Mutex<Option<bool>>,
    /// The workgroups the device holds at once, as the load's rates step
    /// measured them ([`Context::resident`]).
    resident: std::sync::OnceLock<u32>,
}

/// The kernel [`Frame::copy_rows`] dispatches.
const COPIER: (&str, &str) = ("gather/rows.slang", "gather_rows_copy_regions_words");

/// The device's pipeline cache, which kernel and guest pipelines share. A
/// build marks it changed, and it reaches its file shortly after rather
/// than at shutdown (a phone kills an app's process more often than the app
/// shuts down), written by a thread of its own once builds have paused: no
/// submission waits on the driver's serialization of the cache or on the
/// file.
pub struct PipelineCache {
    core: Arc<Core>,
    cache: vk::PipelineCache,
    path: Option<std::path::PathBuf>,
    signal: Arc<CacheSignal>,
    writer: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// What a build tells the cache's writer: when the cache last changed, and
/// that it is to stop (writing what changed first).
#[derive(Default)]
struct CacheSignal {
    state: Mutex<CacheState>,
    wake: std::sync::Condvar,
}

#[derive(Default)]
struct CacheState {
    changed_at: Option<std::time::Instant>,
    stop: bool,
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
        let signal = Arc::new(CacheSignal::default());
        let writer = path.clone().map(|path| {
            let (core, signal) = (Arc::clone(core), Arc::clone(&signal));
            std::thread::Builder::new()
                .name("pipeline cache".into())
                .spawn(move || write_settled(&core, cache, &path, &signal))
                .expect("a pipeline cache writer thread")
        });
        Ok(PipelineCache {
            core: Arc::clone(core),
            cache,
            path,
            signal,
            writer: Mutex::new(writer),
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

    /// Marks the cache changed by a build: its writer writes it once builds
    /// have paused for [`PIPELINE_CACHE_SETTLE`].
    pub fn built(&self) {
        let mut state = self.signal.state.lock().expect("cache signal poisoned");
        state.changed_at = Some(std::time::Instant::now());
        self.signal.wake.notify_one();
    }
}

/// The cache's writer: waits for a change, then for builds to pause for
/// [`PIPELINE_CACHE_SETTLE`], and writes the cache to `path`; on stop, writes
/// what changed and returns.
fn write_settled(
    core: &Core,
    cache: vk::PipelineCache,
    path: &std::path::Path,
    signal: &CacheSignal,
) {
    let mut state = signal.state.lock().expect("cache signal poisoned");
    loop {
        match state.changed_at {
            None if state.stop => return,
            None => {
                state = signal.wake.wait(state).expect("cache signal poisoned");
            }
            Some(at) if !state.stop && at.elapsed() < PIPELINE_CACHE_SETTLE => {
                let left = PIPELINE_CACHE_SETTLE - at.elapsed();
                state = signal
                    .wake
                    .wait_timeout(state, left)
                    .expect("cache signal poisoned")
                    .0;
            }
            Some(_) => {
                state.changed_at = None;
                drop(state);
                if let Err(fault) = write_cache(core, cache, path) {
                    tracing::warn!(%fault, "writing the pipeline cache");
                }
                state = signal.state.lock().expect("cache signal poisoned");
            }
        }
    }
}

/// The cache's data, written whole to `path`.
fn write_cache(core: &Core, cache: vk::PipelineCache, path: &std::path::Path) -> Result<()> {
    // SAFETY: `cache` is a live cache of `core.device` (its owner joins this
    // thread before destroying it); reading its data may race pipeline
    // builds that use it, which Vulkan allows, as `vkGetPipelineCacheData`
    // does not need it externally synchronized.
    let bytes = unsafe { core.device.get_pipeline_cache_data(cache) }
        .map_err(|e| core.fault("vkGetPipelineCacheData", e))?;
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

impl Drop for PipelineCache {
    fn drop(&mut self) {
        // The writer writes what changed and returns before the cache goes.
        self.signal
            .state
            .lock()
            .expect("cache signal poisoned")
            .stop = true;
        self.signal.wake.notify_one();
        if let Some(writer) = self.writer.lock().expect("writer lock poisoned").take() {
            let _ = writer.join();
        }
        // SAFETY: this is the last `Arc<PipelineCache>` and its writer has
        // returned, so nothing uses the cache; pipelines made from it do not
        // depend on it.
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
/// (`maxBufferSize`, core 1.3; absent before it). A limit reported as zero
/// states nothing.
pub(crate) fn buffer_ceiling(storage_range: u32, allocation: u64, buffer: Option<u64>) -> u64 {
    [Some(allocation), buffer]
        .into_iter()
        .flatten()
        .filter(|&limit| limit > 0)
        .fold(u64::from(storage_range), u64::min)
}

/// The most a load may hold in all where the device shares the host's
/// memory and no memory budget says what it has: what the host's memory
/// has available less what lmkd keeps free past it and the floor
/// ([`engine_gpu::host_memory`]), plus what this process already holds,
/// which that reading counts as taken. Every Android device shares it; off
/// Android, an integrated or CPU device, or one whose largest device-local
/// heap the host maps, without `VK_EXT_memory_budget`. A discrete card
/// whose heap the host maps (resizable BAR) has its own memory, so it is
/// not one.
/// Whose memory a Vulkan device's is: the host's on every Android device,
/// and elsewhere on an integrated or CPU device, or any other whose largest
/// device-local heap the host maps (`mapped`: a heap both `DEVICE_LOCAL`
/// and `HOST_VISIBLE`). A discrete card whose heap the host maps (resizable
/// BAR) has memory of its own.
fn device_memory(device: vk::PhysicalDeviceType, mapped: bool) -> engine::DeviceMemory {
    let hosts = cfg!(target_os = "android")
        || match device {
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU => true,
            vk::PhysicalDeviceType::DISCRETE_GPU => false,
            _ => mapped,
        };
    if hosts {
        engine::DeviceMemory::Hosts
    } else {
        engine::DeviceMemory::Own
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn host_ceiling(name: &str, shares_host: bool, memory_budget: bool) -> Result<Option<u64>> {
    if !cfg!(target_os = "android") && (memory_budget || !shares_host) {
        return Ok(None);
    }
    let room = engine_gpu::host_memory::room()?;
    let held = engine_gpu::books::process_reserved();
    tracing::info!(
        device = name,
        available = room.available,
        defended = room.defended,
        cgroup = ?room.cgroup,
        floor = engine_gpu::host_memory::SAFETY_FLOOR_BYTES,
        room = room.room,
        held,
        "read at bind: the device shares the host's memory, so the load takes what it has \
         available, less what lmkd keeps free past it, capped at what \
         its cgroups leave, less the floor"
    );
    Ok(Some(room.room.saturating_add(held)))
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[allow(clippy::unnecessary_wraps)]
fn host_ceiling(_name: &str, _shares_host: bool, _memory_budget: bool) -> Result<Option<u64>> {
    Ok(None)
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
        let validated = boot.validation
            && layers
                .iter()
                .any(|l| l.layer_name_as_c_str().is_ok_and(|s| s == validation));
        let (instance, props2) = shared_instance(&entry, validated)?;
        Self::finish(entry, instance, boot, budget, props2)
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
        #[cfg(feature = "test-limits")]
        let max_buffer = max_buffer.min(boot.max_buffer.unwrap_or(u64::MAX));
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
        // Read before the device is made, so a refusal leaves nothing to free.
        let whose_memory;
        let ceiling = {
            // SAFETY: `physical` was enumerated from the live `instance`.
            let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
            let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
            let largest = heaps
                .iter()
                .enumerate()
                .filter(|(_, h)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .max_by_key(|&(_, h)| h.size)
                .map(|(at, _)| at);
            let mapped = memory.memory_types[..memory.memory_type_count as usize]
                .iter()
                .any(|t| {
                    Some(t.heap_index as usize) == largest
                        && t.property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE,
                        )
                });
            whose_memory = device_memory(props.device_type, mapped);
            host_ceiling(&name, whose_memory.unified(), memory_budget)?
        };
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
        // A recording's pool segment bindings are written as the segments
        // are backed and given back, while it stands: update-after-bind
        // storage buffers. A device without them lays its pool whole
        // (`Context::resizes_pool`). `PIE_VULKAN_WHOLE_POOL` takes that path
        // on one that has them, to test it.
        let update_after_bind = f12.descriptor_binding_storage_buffer_update_after_bind
            == vk::TRUE
            && std::env::var_os("PIE_VULKAN_WHOLE_POOL").is_none();
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
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE)
            .descriptor_binding_storage_buffer_update_after_bind(update_after_bind);
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
            spares: Mutex::new(Spares::default()),
            shared: Mutex::new(SharedSets::default()),
            given: std::sync::Condvar::new(),
            watcher: Mutex::new(None),
            last: Mutex::new(Last::None),
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
            ceiling,
            whose_memory,
            heap,
            vendor_id: props.vendor_id,
            update_after_bind,
            segments: Mutex::new(std::collections::HashMap::new()),
            segments_bound: Mutex::new(Vec::new()),
            rebound: Mutex::new(Vec::new()),
            device_id: props.device_id,
            driver_version: props.driver_version,
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
        let (file, entry) = COPIER;
        let unbuilt = |why: &str| Fault::Shader {
            file,
            entrypoint: entry,
            why: why.to_string(),
        };
        let key = kernels_slang::key(entry, kernels_slang::Tier::Baseline)
            .ok_or_else(|| unbuilt("kernels-slang names no such instantiation"))?;
        let code = kernels_slang::spirv(key)
            .ok_or_else(|| unbuilt("kernels-slang holds no SPIR-V module by that key"))?;
        let copier = Arc::new(super::pipelines::build(
            &core,
            pipeline_cache.handle(),
            file,
            key,
            code,
        )?);
        pipeline_cache.built();
        // The watcher starts with the device: no thread is spawned per
        // submission, nor after the load.
        let (watches, watched) = std::sync::mpsc::channel::<Watch>();
        std::thread::Builder::new()
            .name("vk-fence-watch".to_string())
            .spawn(move || {
                for watch in watched {
                    watch.run();
                }
            })
            .map_err(|e| Fault::NoDevice {
                detail: format!("the fence watcher did not start: {e}"),
            })?;
        *core.watcher.lock().expect("watcher lock poisoned") = Some(watches);
        Ok(Context {
            core,
            budget,
            max_buffer,
            dummy,
            pipeline_cache,
            device_index: boot.device_index,
            copier,
            indirect_next: Mutex::new(None),
            arm_next: Mutex::new(None),
            room_next: Mutex::new(None),
            fixed_next: Mutex::new(None),
            resident: std::sync::OnceLock::new(),
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

    /// Where the device shares the host's memory (a ceiling was read at
    /// bind), the reader of what the host can still give the process, the
    /// bind's reading taken afresh ([`host_ceiling`]).
    #[must_use]
    pub fn whose_memory(&self) -> engine::DeviceMemory {
        self.core.whose_memory
    }

    #[must_use]
    pub fn host_room(&self) -> Option<engine::HostRoom> {
        self.core.ceiling?;
        Some(engine_gpu::host_memory::room_reader())
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    /// What sizes work to the device, as its kernels count cores: the
    /// compute units the driver reports, or where it reports none the
    /// workgroups the device holds at once ([`Context::resident`]). `None`
    /// before the load has measured them.
    #[must_use]
    pub fn cores(&self) -> Option<u32> {
        self.core.cores.or_else(|| self.resident.get().copied())
    }

    /// The workgroups the device holds at once, as the load's rates step
    /// measured them on every device (`profile::fire::RESIDENT_WORKGROUPS`):
    /// what a guest step's grid strides over. `None` before the load has.
    #[must_use]
    pub fn resident(&self) -> Option<u32> {
        self.resident.get().copied()
    }

    /// Lays the workgroups the load's rates step measured
    /// ([`Context::resident`]); the first holds.
    pub(crate) fn hold_resident(&self, workgroups: u32) {
        let _ = self.resident.set(workgroups);
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
            // Before the rates step has measured a device the driver reports
            // no cores of, one: that step's own tuning, whose timed kernels
            // read neither the qmm's fill nor a decode's split per core (a
            // load's tuning states them, `crate::rates::at_load`).
            cores: self.cores().unwrap_or(1),
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

    /// The device as a measurement stored in the user's profile is keyed
    /// by (`engine_gpu::stored`): its vendor and device ids and its
    /// driver's version. Another driver measures again.
    #[must_use]
    pub fn measured_on(&self) -> String {
        format!(
            "vulkan {:04x}:{:04x} driver {:#x}",
            self.core.vendor_id, self.core.device_id, self.core.driver_version
        )
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

    /// Lays the slab every host write to device memory stages through
    /// (a guest bind's tables among them), where nothing has made it yet:
    /// a load lays it, so serving reserves none.
    pub fn lay_staging(&self) -> Result<()> {
        let mut transfer = self.core.transfer.lock().expect("transfer lock poisoned");
        self.core.staging(&mut transfer).map(|_| ())
    }

    /// States the pool's segments below the span's: for each, the buffer
    /// backing it (or `None`) and its stand-in. Each kept frame's descriptor
    /// of a segment is written to bind the segment's buffer, or its stand-in
    /// where it is not backed; a segment whose buffer stands as it was is
    /// not written. Called with every frame that reads them landed (a
    /// release drains first; a segment just backed is read by no fire yet),
    /// and before a released segment's buffer goes. The descriptors written.
    pub fn rebind_segments(&self, segments: &[(Option<&Buffer>, &Buffer)]) -> Result<u64> {
        use ash::vk::Handle;
        let bound: Vec<&Buffer> = segments
            .iter()
            .map(|(backed, stand_in)| backed.unwrap_or(stand_in))
            .collect();
        {
            let mut held = self.core.segments.lock().expect("segments lock poisoned");
            held.clear();
            for (at, (backed, stand_in)) in segments.iter().enumerate() {
                held.insert(stand_in.slab().buffer.as_raw(), at as u32);
                if let Some(backed) = backed {
                    held.insert(backed.slab().buffer.as_raw(), at as u32);
                }
            }
        }
        let now: Vec<u64> = bound.iter().map(|b| b.slab().buffer.as_raw()).collect();
        let changed: Vec<bool> = {
            let mut before = self
                .core
                .segments_bound
                .lock()
                .expect("segments lock poisoned");
            let changed = (0..now.len())
                .map(|at| before.get(at) != Some(&now[at]))
                .collect();
            *before = now;
            changed
        };
        if !changed.iter().any(|&c| c) {
            return Ok(0);
        }
        let kept: Vec<Arc<Inner>> = {
            let mut rebound = self.core.rebound.lock().expect("rebound lock poisoned");
            rebound.retain(|held| held.strong_count() > 0);
            rebound.iter().filter_map(std::sync::Weak::upgrade).collect()
        };
        let range = u64::from(self.core.limits.max_storage_buffer_range);
        let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut places: Vec<(vk::DescriptorSet, u32)> = Vec::new();
        for inner in &kept {
            for &(set, binding, segment, offset, bytes) in
                inner.rebinds.lock().expect("rebinds lock poisoned").iter()
            {
                if !changed.get(segment as usize).copied().unwrap_or(false) {
                    continue;
                }
                let buffer = bound.get(segment as usize).ok_or_else(|| Fault::Unbound {
                    what: format!("pool segment {segment}, past the {} stated", bound.len()),
                })?;
                // A whole binding (`u64::MAX` bytes) binds the buffer whole;
                // a view past a stand-in's few bytes binds the stand-in whole,
                // which no fire reads while the segment is not backed.
                let (offset, view) = if bytes != u64::MAX && offset + bytes <= buffer.bytes() {
                    (offset, bytes)
                } else {
                    (0, buffer.bytes())
                };
                infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer.slab().buffer)
                        .offset(offset)
                        .range(view.min(range)),
                );
                places.push((set, binding));
            }
        }
        let writes: Vec<vk::WriteDescriptorSet<'_>> = places
            .iter()
            .zip(&infos)
            .map(|(&(set, binding), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        if !writes.is_empty() {
            // SAFETY: every set named is a live kept frame's (its `Inner`
            // upgraded above), its binding laid update-after-bind (only those
            // are noted, `Frame::rebinds`), and no submission in flight
            // reads it (the caller's contract); each buffer is a live slab.
            unsafe { self.core.device.update_descriptor_sets(&writes, &[]) };
        }
        tracing::debug!(
            segments = ?changed,
            descriptors = writes.len(),
            recordings = kept.len(),
            "the recordings' descriptors of the pool segments that changed are rebound"
        );
        Ok(writes.len() as u64)
    }

    /// Whether a recording's pool segments are rebound as they change
    /// (update-after-bind storage buffers), else the pool is laid whole.
    #[must_use]
    pub fn rebinds_segments(&self) -> bool {
        self.core.update_after_bind
    }

    /// Serving starts: the spares are stocked to `frames` free pools (the
    /// frames serving holds at once, each one pool: the load's kept frames
    /// hold their own beside them), and from here no frame makes one
    /// ([`Core::descriptor_pool`]).
    pub fn seal_descriptor_pools(&self, frames: usize) -> Result<()> {
        let mut spares = self.core.spares.lock().expect("spares lock poisoned");
        while spares.free.len() < frames {
            let pool = self.core.new_descriptor_pool()?;
            spares.free.push(pool);
        }
        spares.stock = spares.free.len();
        spares.sealed = true;
        Ok(())
    }

    /// Names the descriptor sets and pools frames allocated so far (as
    /// every 10,000 sets do).
    pub fn name_descriptor_sets(&self) {
        count_sets(&self.core, 0, false);
    }

    /// The descriptor pools free, lent and kept now.
    #[must_use]
    pub fn descriptor_pools(&self) -> (usize, usize, usize) {
        let spares = self.core.spares.lock().expect("spares lock poisoned");
        (spares.free.len(), spares.lent, spares.kept)
    }

    pub fn frame(&self) -> Result<Frame> {
        self.frame_with(true)
    }

    /// The dispatches, by their place in the walk, the next frame opened
    /// makes indirect: those whose group counts differ between the shapes of
    /// one recording. Every other dispatch carries its counts.
    pub fn indirect_next(&self, at: Option<Arc<Vec<bool>>>) {
        *self.indirect_next.lock().expect("indirect lock poisoned") = at;
    }

    /// The offset alignments a fire block lays at: a uniform's and a
    /// storage view's.
    #[must_use]
    pub fn fire_alignments(&self) -> (u64, u64) {
        let limits = &self.core.limits;
        (
            limits.min_uniform_buffer_offset_alignment.max(4),
            limits.min_storage_buffer_offset_alignment.max(4),
        )
    }

    /// The bytes of scalars the readout copier takes.
    #[must_use]
    pub fn copier_scalars(&self) -> u32 {
        self.copier
            .fire
            .as_ref()
            .map_or(0, |fire| fire.bytes)
            .max(self.copier.push_bytes)
    }

    /// The fire block the next frame opened lays in: the arm's whose fire
    /// it walks. A frame opened with none lays its block in chunks of its
    /// own.
    pub fn arm_next(&self, arm: Option<Arc<super::block::ArmBlock>>) {
        *self.arm_next.lock().expect("arm lock poisoned") = arm;
    }

    /// The score room (`room`'s buffer at byte `at`) the next frame opened
    /// lays its capturing lanes' rows in: a recording binds it by place
    /// ([`Kept::bind_room`]).
    pub fn room_next(&self, room: Option<(&Buffer, u64)>) {
        *self.room_next.lock().expect("room lock poisoned") =
            room.map(|(buffer, at)| (buffer.slab().buffer, at));
    }

    /// Whether the next frame opened carries its rows as constants of its
    /// commands (a decode body, a fire no wider than a row tile) or reads
    /// them from the fire block (a body every row count of a rung shares).
    pub fn fixed_next(&self, fixed: bool) {
        *self.fixed_next.lock().expect("fixed lock poisoned") = Some(fixed);
    }

    pub fn frame_kept(&self) -> Result<Frame> {
        self.frame_with(false)
    }

    /// Whether every submission made on the device so far has landed, read
    /// now without waiting: the queue is empty. Frames run in queue order
    /// (each begins with a barrier on all work before it), so the queue is
    /// empty once the frame that submitted last has landed: dropped (a
    /// submitted frame's drop waits its fence), seen landed by a wait, or
    /// its fence signaled now.
    #[must_use]
    pub fn idle(&self) -> bool {
        let last = match &*self.core.last.lock().expect("last lock poisoned") {
            Last::None => return true,
            Last::Frame(frame) => frame.upgrade().map(Ok),
            Last::Replay(replay) => replay.upgrade().map(Err),
        };
        match last {
            None => true,
            Some(Ok(last)) => {
                if last.landed.load(Ordering::Acquire) >= last.seq.load(Ordering::Acquire) {
                    return true;
                }
                // SAFETY: the fence lives as long as `last`; querying it
                // needs no external sync.
                let signaled =
                    unsafe { self.core.device.get_fence_status(last.fence) }.unwrap_or(false);
                if signaled {
                    last.landed_now();
                }
                signaled
            }
            // A replay's lent fence goes back once it landed, its
            // submission's last holder gone with it.
            Some(Err(replay)) => {
                replay.landed.load(Ordering::Acquire)
                    // SAFETY: the lent fence is the submission's until it
                    // drops, which `replay` holds off.
                    || unsafe { self.core.device.get_fence_status(replay.fence) }.unwrap_or(false)
            }
        }
    }

    /// A frame whose device time [`Frame::commit_spanned`] reads back: a
    /// timestamp written at the top of the pipe before its work and one at
    /// the bottom after it, where the queue writes timestamps (its family's
    /// `timestampValidBits` nonzero, a nonzero `timestampPeriod`); a plain
    /// frame where it writes none.
    pub fn spanned(&self) -> Result<Frame> {
        self.spanned_with(true)
    }

    /// [`Context::spanned`] kept to replay ([`Frame::into_kept_spanned`],
    /// [`Kept::replay_spanned`]): its timestamps' reset and writes recorded
    /// with its work, so each replay writes them anew.
    pub fn spanned_kept(&self) -> Result<Frame> {
        self.spanned_with(false)
    }

    /// Whether the queue writes timestamps (its family's
    /// `timestampValidBits` nonzero, a nonzero `timestampPeriod`): a
    /// spanned frame then reads its device time back.
    #[must_use]
    pub fn writes_timestamps(&self) -> bool {
        self.core.stamp_bits() > 0
    }

    fn spanned_with(&self, one_time: bool) -> Result<Frame> {
        if self.core.stamp_bits() == 0 {
            return self.frame_with(one_time);
        }
        let stamps = self.stamps()?;
        self.spanned_into(&stamps, !one_time)
    }

    /// Two timestamps' query pool, laid now, which a spanned frame writes
    /// ([`Context::spanned_into`]): laid at load and written by one frame
    /// in flight at a time, so no serving frame lays one.
    pub fn stamps(&self) -> Result<Stamps> {
        let core = &self.core;
        if core.stamp_bits() == 0 {
            return Err(Fault::Unbound {
                what: format!(
                    "timestamps on {}, whose queue family writes none (timestampValidBits 0, or \
                     timestampPeriod 0)",
                    core.name
                ),
            });
        }
        // SAFETY: `core.device` is live; the queue writes timestamps
        // (checked above).
        let pool = unsafe {
            core.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|e| core.fault("vkCreateQueryPool", e))?;
        Ok(Stamps(Arc::new(StampPool {
            core: Arc::clone(core),
            pool,
        })))
    }

    /// A frame writing its two timestamps into `stamps` (laid before,
    /// [`Context::stamps`]): their reset and the opening one recorded first,
    /// the closing one as it is submitted or kept; kept to replay where
    /// `keep`, each replay writing them anew. One frame in flight writes a
    /// pool at a time: its device time is read ([`Pending::span_ns`])
    /// before the next is submitted.
    pub fn spanned_into(&self, stamps: &Stamps, keep: bool) -> Result<Frame> {
        let frame = self.frame_with(!keep)?;
        let d = &self.core.device;
        let pool = stamps.0.pool;
        let cmd = frame.inner.cmd();
        // SAFETY: `cmd` is recording outside a render pass and only this
        // frame records into it; the range is the pool's two queries.
        unsafe {
            d.cmd_reset_query_pool(cmd, pool, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, 0);
        }
        *frame.inner.span.lock().expect("span lock poisoned") = Some(stamps.clone());
        Ok(frame)
    }

    /// A frame a fire walks to fill an armed body ([`Frame::dry`]): what the
    /// walk lays in the fire block and what its commands would carry,
    /// hashed, with no command recorded and no descriptor set taken. It is
    /// dropped unsubmitted ([`Frame::into_fill`]).
    pub fn frame_dry(&self) -> Result<Frame> {
        let mut frame = self.frame_with(true)?;
        frame.dry = true;
        Ok(frame)
    }

    fn frame_with(&self, one_time: bool) -> Result<Frame> {
        let core = &self.core;
        let d = &core.device;
        let pool = core.thread_pool()?;
        // SAFETY: `pool` is this thread's own pool, which no other thread
        // allocates from or frees into, so its external sync holds.
        let cmd = core.alloc_cmd(pool)?;
        let fence = core.lend_fence()?;
        LIVE_FRAMES.fetch_add(1, Ordering::Relaxed);
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
                one_time,
                cmd: std::sync::atomic::AtomicU64::new(vk::Handle::as_raw(cmd)),
                cuts: Mutex::new(Vec::new()),
                heads: Mutex::new(Vec::new()),
                segments: Mutex::new(Vec::new()),
                held: Mutex::new(Vec::new()),
                segmented: std::sync::atomic::AtomicBool::new(false),
                cmd_pool: pool,
                fence,
                pools: Mutex::new(Vec::new()),
                keep: Mutex::new(Vec::new()),
                submitted: std::sync::atomic::AtomicBool::new(false),
                timing: Mutex::new(timing),
                span: Mutex::new(None),
                spare: Mutex::new(std::collections::HashMap::new()),
                seq: std::sync::atomic::AtomicU64::new(0),
                landed: std::sync::atomic::AtomicU64::new(0),
                waiters: std::sync::atomic::AtomicU32::new(0),
                most_waiters: std::sync::atomic::AtomicU32::new(0),
                rebinds: Mutex::new(Vec::new()),
                pinned: Mutex::new(None),
                room_binds: Mutex::new(Vec::new()),
                room_bound: Mutex::new(None),
                room_fixed: std::sync::atomic::AtomicBool::new(false),
                recording: std::sync::atomic::AtomicBool::new(true),
            }),
            open: true,
            room: self.room_next.lock().expect("room lock poisoned").take(),
            room_pasts: std::cell::RefCell::new(Vec::new()),
            dry: false,
            kept: !one_time,
            hazards: std::cell::RefCell::new(Hazards::default()),
            retained: std::cell::RefCell::new(std::collections::HashSet::new()),
            dispatches: std::cell::Cell::new(0),
            block: std::cell::RefCell::new(
                match self.arm_next.lock().expect("arm lock poisoned").take() {
                    Some(arm) => super::block::FireBlock::of_arm(arm),
                    None => super::block::FireBlock::own(),
                },
            ),
            copier: Arc::clone(&self.copier),
            fixed_rows: self
                .fixed_next
                .lock()
                .expect("fixed lock poisoned")
                .take()
                .unwrap_or(true),
            indirect: if one_time {
                None
            } else {
                self.indirect_next
                    .lock()
                    .expect("indirect lock poisoned")
                    .take()
            },
            groups: std::cell::RefCell::new(Vec::new()),
            copies: std::cell::RefCell::new(Vec::new()),
            baked: std::cell::RefCell::new(std::collections::hash_map::DefaultHasher::new()),
            trail: std::cell::RefCell::new(Vec::new()),
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
    /// Submitted once (not kept): its descriptor pools go back once its
    /// fence passed.
    one_time: bool,
    /// The command buffer being recorded (a kept frame's last segment).
    cmd: std::sync::atomic::AtomicU64,
    /// A kept frame's segments before its last, in order: each ended where
    /// the walk needed the frame so far landed before the host went on (an
    /// expert stage); a replay submits each, waits, and lets its caller do
    /// that host work before the next.
    cuts: Mutex<Vec<vk::CommandBuffer>>,
    /// Each head ([`Frame::head`]) recorded for the frame's submissions.
    heads: Mutex<Vec<vk::CommandBuffer>>,
    /// A split step's segments before its last, submitted unawaited
    /// ([`Frame::submit_segment`]): each held until the frame is dropped,
    /// whose last segment's fence lands after them (it begins with a barrier
    /// on every command before it on the queue).
    segments: Mutex<Vec<vk::CommandBuffer>>,
    /// The segments ended and not yet submitted: submitted with the frame,
    /// back to back ([`Frame::submit`]).
    held: Mutex<Vec<vk::CommandBuffer>>,
    /// A segment was submitted: a frame dropped unsubmitted waits for the
    /// queue before it frees them.
    segmented: std::sync::atomic::AtomicBool,

    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) fence: vk::Fence,

    pub(crate) pools: Mutex<Vec<vk::DescriptorPool>>,

    pub(crate) keep: Mutex<Vec<Slab>>,
    submitted: std::sync::atomic::AtomicBool,

    pub(crate) timing: Mutex<Option<Timing>>,

    /// A spanned frame's two timestamps, before its work and after it
    /// ([`Context::spanned`]).
    span: Mutex<Option<Stamps>>,

    spare: Mutex<std::collections::HashMap<u64, Vec<vk::DescriptorSet>>>,

    /// The fenced submissions of this frame so far, and the most of them
    /// seen landed ([`Context::idle`]).
    seq: std::sync::atomic::AtomicU64,
    landed: std::sync::atomic::AtomicU64,

    /// Threads blocked on `fence` now, and the most at once.
    waiters: std::sync::atomic::AtomicU32,
    most_waiters: std::sync::atomic::AtomicU32,

    /// A kept frame's descriptors that bind a pool segment: the set, the
    /// binding, the segment, and the view's offset and bytes in it
    /// ([`Context::rebind_segments`]).
    pub(crate) rebinds: Mutex<Vec<(vk::DescriptorSet, u32, u32, u64, u64)>>,
    /// A kept frame binds a pool segment where no rebind can write it (a
    /// binding its layout does not lay update-after-bind, or a copy's
    /// command): it stands only while the segments do, the name of what
    /// pinned it beside.
    pub(crate) pinned: Mutex<Option<String>>,
    /// A kept frame's descriptors that bind its score room: the set, the
    /// binding, the view's byte past the room's and its bytes
    /// ([`Kept::bind_room`]).
    pub(crate) room_binds: Mutex<Vec<(vk::DescriptorSet, u32, u64, u64)>>,
    /// The room its descriptors bind now: its buffer and byte.
    pub(crate) room_bound: Mutex<Option<(vk::Buffer, u64)>>,
    /// A command of it names the room directly (a copy of its rows): it
    /// replays only at the room it was recorded at.
    pub(crate) room_fixed: std::sync::atomic::AtomicBool,
    /// `cmd` is begun and not ended: the one state it may be ended in. Every
    /// end clears it before anything that can fail, and every begin sets it
    /// once it succeeded, so a frame dropped on an error between an end and
    /// the next begin (a failed submit, wait or allocation of a flush or a
    /// segment) leaves its buffer alone.
    pub(crate) recording: std::sync::atomic::AtomicBool,
}

impl Inner {
    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        vk::Handle::from_raw(self.cmd.load(Ordering::Acquire))
    }

    /// Ends `cmd`, which must be recording, and marks it ended first.
    fn end_cmd(&self) -> std::result::Result<(), vk::Result> {
        self.recording.store(false, Ordering::Release);
        // SAFETY: callers end only a buffer they began (`recording` was set),
        // from the frame that records into it.
        unsafe { self.core.device.end_command_buffer(self.cmd()) }
    }

    /// `cmd` was begun.
    fn begun(&self) {
        self.recording.store(true, Ordering::Release);
    }
}

/// Two timestamps' query pool a spanned frame writes
/// ([`Context::stamps`]), shared by the frames that write it, one in flight
/// at a time; destroyed with the last.
#[derive(Clone)]
pub struct Stamps(Arc<StampPool>);

impl std::fmt::Debug for Stamps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stamps").finish()
    }
}

struct StampPool {
    core: Arc<Core>,
    pool: vk::QueryPool,
}

impl Drop for StampPool {
    fn drop(&mut self) {
        // SAFETY: the last frame holding the pool is gone, and a submitted
        // frame waits its fence before it drops, so no submission names it.
        unsafe { self.core.device.destroy_query_pool(self.pool, None) };
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
            // A fence already signaled is not waited on: the frame landed,
            // its segments before it.
            if self.submitted.load(Ordering::Acquire) {
                if d.get_fence_status(self.fence) != Ok(true) {
                    let _ = d.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS);
                }
            } else if self.segmented.load(Ordering::Acquire) {
                let queue = self.core.queue.lock().expect("queue lock poisoned");
                let _ = d.queue_wait_idle(*queue);
            }
            // Its descriptor pools go back to the device's spares, reset:
            // the next frame takes them rather than making its own.
            self.give_pools();
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
            buffers.extend(std::mem::take(
                self.heads.get_mut().expect("heads lock poisoned"),
            ));
            buffers.extend(std::mem::take(
                self.segments.get_mut().expect("segments lock poisoned"),
            ));
            buffers.push(self.cmd());
            buffers.retain(|&cmd| cmd != vk::CommandBuffer::null());
            LIVE_CMDS.fetch_sub(buffers.len() as i64, Ordering::Relaxed);
            LIVE_FRAMES.fetch_sub(1, Ordering::Relaxed);
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

impl Inner {
    /// The device's time between a spanned frame's two timestamps, ns, once
    /// it landed; `None` for a frame that writes none.
    fn span_ns(&self) -> Result<Option<u64>> {
        let Some(pool) = self
            .span
            .lock()
            .expect("span lock poisoned")
            .as_ref()
            .map(|stamps| stamps.0.pool)
        else {
            return Ok(None);
        };
        let core = &self.core;
        let mut ticks = [0u64; 2];
        // SAFETY: the frame's fence was waited on, so both queries are
        // written; `ticks` holds one u64 for each with `TYPE_64`.
        unsafe {
            core.device.get_query_pool_results(
                pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|e| core.fault("vkGetQueryPoolResults", e))?;
        let bits = core.stamp_bits();
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let ticks = ticks[1].wrapping_sub(ticks[0]) & mask;
        Ok(Some(
            (ticks as f64 * f64::from(core.limits.timestamp_period)) as u64,
        ))
    }
}

const SETS_PER_POOL: u32 = 1024;

/// Descriptor sets allocated by frames, all and those kept frames took, and
/// descriptor pools made, over the process: named every 10,000 sets (a
/// driver that allocates host memory per set runs out of its allocator's
/// room for them before the host's).
static SETS_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static KEPT_SETS_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static POOLS_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Sets shared for the device's life ([`SharedSets`]).
static SHARED_SETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Kept frames' pools destroyed with them.
static POOLS_FREED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static TUPLES: std::sync::Mutex<Option<std::collections::HashSet<u64>>> =
    std::sync::Mutex::new(None);

/// Notes a kept dispatch's binding tuple (diagnostic).
pub(crate) fn note_tuple(key: u64) {
    TUPLES
        .lock()
        .expect("tuples lock poisoned")
        .get_or_insert_with(std::collections::HashSet::new)
        .insert(key);
}

fn count_sets(core: &Core, sets: u64, kept: bool) {
    if kept {
        KEPT_SETS_MADE.fetch_add(sets, Ordering::Relaxed);
    }
    let before = SETS_MADE.fetch_add(sets, Ordering::Relaxed);
    if sets == 0 || (before + sets) / 10_000 > before / 10_000 {
        let spares = core.spares.lock().expect("spares lock poisoned");
        tracing::info!(
            sets = before + sets,
            kept_sets = KEPT_SETS_MADE.load(Ordering::Relaxed),
            pools_made = POOLS_MADE.load(Ordering::Relaxed),
            pools_freed = POOLS_FREED.load(Ordering::Relaxed),
            shared_sets = SHARED_SETS.load(Ordering::Relaxed),
            pools_free = spares.free.len(),
            pools_lent = spares.lent,
            pools_kept = spares.kept,
            distinct_kept_tuples = TUPLES
                .lock()
                .expect("tuples lock poisoned")
                .as_ref()
                .map_or(0, std::collections::HashSet::len),
            "vulkan descriptor sets allocated so far"
        );
    }
}

/// A kept frame's pools: a body's sets a few pools of these.
const KEPT_SETS_PER_POOL: u32 = 64;

const SETS_PER_BATCH: usize = 64;

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

/// Whether frames keep what each command carried, rendered
/// (`PIE_VK_BAKE_TRAIL`): to tell a fill walk that matches no armed family
/// apart from the nearest one.
pub(crate) fn trailing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_VK_BAKE_TRAIL").is_some())
}

pub struct Frame {
    pub(crate) inner: Arc<Inner>,
    open: bool,
    /// The score room its capturing lanes lay their rows in: its buffer and
    /// byte ([`Context::room_next`]).
    pub(crate) room: Option<(vk::Buffer, u64)>,
    /// Each room binding's bytes past the room's byte, in the walk's order.
    room_pasts: std::cell::RefCell<Vec<u64>>,
    /// A fill walk's frame ([`Context::frame_dry`]): it lays what a fire
    /// writes in its block and hashes what its commands carry, and records
    /// no command; it is never submitted.
    pub(crate) dry: bool,
    /// The fire's rows are constants of its commands ([`Context::fixed_next`]).
    fixed_rows: bool,
    /// Opened by `frame_kept`: a flush cuts a segment rather than reusing
    /// the command buffer, so the whole frame replays.
    kept: bool,
    pub(crate) dispatches: std::cell::Cell<u64>,

    pub(crate) hazards: std::cell::RefCell<Hazards>,

    retained: std::cell::RefCell<std::collections::HashSet<usize>>,

    /// Where the walk lays each dispatch's scalars and group counts.
    block: std::cell::RefCell<super::block::FireBlock>,

    /// Everything the frame's commands carry, hashed: two walks whose
    /// commands hash alike differ only in what their fire blocks hold.
    baked: std::cell::RefCell<std::collections::hash_map::DefaultHasher>,

    /// What each command folded into `baked` carried, rendered, where
    /// `PIE_VK_BAKE_TRAIL` asks ([`trailing`]): what a fill walk that
    /// matches no armed family is told apart from the nearest by.
    trail: std::cell::RefCell<Vec<String>>,

    copier: Arc<super::pipelines::Pipeline>,

    /// The dispatches made indirect, by their place in the walk.
    indirect: Option<Arc<Vec<bool>>>,

    /// Each dispatch's group counts, in the walk's order.
    groups: std::cell::RefCell<Vec<[u32; 3]>>,

    /// Each readout copy's mark (its most rows: its regions are the copier's
    /// table), in order.
    copies: std::cell::RefCell<Vec<[u32; 3]>>,
}

impl Frame {
    /// A frame that records nothing ([`Context::frame_dry`]).
    pub fn is_dry(&self) -> bool {
        self.dry
    }
}

impl Frame {
    /// Opened to be kept and replayed (a recording).
    pub(crate) fn kept(&self) -> bool {
        self.kept
    }

    pub(crate) fn fixed_rows(&self) -> bool {
        self.fixed_rows
    }

    /// Takes the next dispatch's group counts, and answers whether it reads
    /// them from the fire block (indirect) rather than carrying them.
    pub(crate) fn dispatched(&self, groups: [u32; 3]) -> bool {
        let mut seen = self.groups.borrow_mut();
        let at = seen.len();
        seen.push(groups);
        // A kept frame reads every dispatch's group counts from the block:
        // what sizes a grid but the fire's rows (a lane's buffered rows, the
        // fold / buffer split of a prompt) is the fire's data, so one
        // recording serves every fire of its commands.
        self.kept
            || self
                .indirect
                .as_ref()
                .is_some_and(|set| set.get(at).copied().unwrap_or(false))
    }

    /// Lays `bytes` in the frame's fire block at an offset a multiple of
    /// `align`, the chunk retained with the frame, and answers where: the
    /// chunk's buffer, and its place among the chunks with the offset, which
    /// is the same in every walk laid alike.
    pub(crate) fn lay(&self, bytes: &[u8], align: u64) -> Result<(vk::Buffer, (usize, u64))> {
        let (slab, chunk, at) = self
            .block
            .borrow_mut()
            .lay(&self.inner.core, bytes, align)?;
        self.retain(&slab);
        Ok((slab.buffer, (chunk, at)))
    }

    /// The commands a submission of the frame runs first: the arm's staging
    /// slice copied into its fire block, then a barrier before every read
    /// of it (a uniform, an indirect dispatch's counts, a table). `None`
    /// where the frame laid nothing in an arm's block. The head is recorded
    /// once and kept with the frame.
    fn head(&self, one_time: bool) -> Result<Option<vk::CommandBuffer>> {
        let block = self.block.borrow();
        let (Some(arm), bytes) = (block.arm(), block.arm_bytes()) else {
            return Ok(None);
        };
        if bytes == 0 {
            return Ok(None);
        }
        let core = &self.inner.core;
        let d = &core.device;
        // A recording's head is its arm's, shared by every recording whose
        // image fits the bytes it copies (the next power of two): it lives
        // with the device's pools, not the frame. A frame run once has its
        // own, freed with it.
        let mut shared = None;
        let bytes = if one_time {
            bytes
        } else {
            let size = bytes.next_power_of_two().min(arm.bytes());
            let heads = arm.heads.lock().expect("heads lock poisoned");
            if let Some(&head) = heads.get(&size) {
                return Ok(Some(head));
            }
            shared = Some(size);
            size
        };
        let head = core.alloc_cmd(self.inner.cmd_pool)?;
        match shared {
            Some(size) => {
                arm.heads.lock().expect("heads lock poisoned").insert(size, head);
            }
            None => self
                .inner
                .heads
                .lock()
                .expect("heads lock poisoned")
                .push(head),
        }
        let flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        // SAFETY: `head` was just allocated and is recorded only here; both
        // slabs are the arm's, live as long as the arm the frame holds, and
        // `bytes` lies inside both (`FireBlock::lay` bounds it by the
        // block's size, which the staging slice shares). The barrier makes
        // the copy visible to every compute read and indirect fetch after.
        unsafe {
            d.begin_command_buffer(head, &vk::CommandBufferBeginInfo::default().flags(flags))
                .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
            let before = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::UNIFORM_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(
                head,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_copy_buffer(
                head,
                arm.staging.slab().buffer,
                arm.block.slab().buffer,
                &[vk::BufferCopy::default().size(bytes)],
            );
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::UNIFORM_READ
                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                );
            d.cmd_pipeline_barrier(
                head,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            d.end_command_buffer(head)
                .map_err(|e| core.fault("vkEndCommandBuffer", e))?;
        }
        Ok(Some(head))
    }

    /// Writes what the frame laid so far into its arm's staging slice and
    /// records the head that copies it: what a submission of the frame as it
    /// walks runs first. The arm's last fire has landed: the arm was taken.
    fn staged_head(&self) -> Result<Option<vk::CommandBuffer>> {
        if let Some(arm) = self.block.borrow().arm() {
            // A walk's image is unsealed: it is always written, and its head
            // always copies it.
            arm.stage(self.block.borrow().image())?;
        }
        self.head(true)
    }

    /// The pool segment (below the span's, which stands for good) `buffer`
    /// is, or stands in for ([`Context::rebind_segments`]).
    pub(crate) fn segment_of(&self, buffer: vk::Buffer) -> Option<u32> {
        use ash::vk::Handle;
        self.inner
            .core
            .segments
            .lock()
            .expect("segments lock poisoned")
            .get(&buffer.as_raw())
            .copied()
    }

    /// A kept frame whose command names a pool segment directly (`what`)
    /// stands only while the segments do: no rebind writes a command.
    fn pin_copied(&self, buffers: &[vk::Buffer], what: &str) {
        if self.kept && buffers.iter().any(|&buffer| self.segment_of(buffer).is_some()) {
            self.pin(what.to_string());
        }
    }

    /// Notes that the kept frame binds a pool segment where a rebind cannot
    /// write it, `what` naming where.
    pub(crate) fn pin(&self, what: String) {
        self.inner
            .pinned
            .lock()
            .expect("pinned lock poisoned")
            .get_or_insert(what);
    }

    /// Notes descriptor `binding` of `set`, a kept frame's, as binding pool
    /// segment `segment` at `offset` for `bytes` (`u64::MAX`: the buffer
    /// whole): a rebind writes it.
    pub(crate) fn rebinds(&self, set: vk::DescriptorSet, binding: u32, segment: u32, offset: u64, bytes: u64) {
        self.inner
            .rebinds
            .lock()
            .expect("rebinds lock poisoned")
            .push((set, binding, segment, offset, bytes));
    }

    /// Notes descriptor `binding` of `set`, a kept frame's, as binding its
    /// score room `past` bytes past the room's byte, for `bytes`.
    pub(crate) fn room_binds(&self, set: vk::DescriptorSet, binding: u32, past: u64, bytes: u64) {
        *self.inner.room_bound.lock().expect("room lock poisoned") = self.room;
        self.inner
            .room_binds
            .lock()
            .expect("room lock poisoned")
            .push((set, binding, past, bytes));
    }

    /// Notes the next room binding lies `past` bytes past the room's byte.
    pub(crate) fn room_past(&self, past: u64) {
        self.room_pasts.borrow_mut().push(past);
    }

    /// Each room binding's bytes past the room's byte, in order.
    #[must_use]
    pub fn room_pasts(&self) -> Vec<u64> {
        self.room_pasts.borrow().clone()
    }

    /// Whether `buffer` is the frame's score room's.
    pub(crate) fn is_room(&self, buffer: vk::Buffer) -> bool {
        self.room.is_some_and(|(room, _)| room == buffer)
    }

    /// A command copies `buffers`: one the frame's score room is makes a
    /// kept frame replay only at that room.
    fn fix_room(&self, buffers: &[vk::Buffer]) {
        if self.kept && buffers.iter().any(|&buffer| self.is_room(buffer)) {
            self.inner.room_fixed.store(true, Ordering::Release);
        }
    }

    /// What a dry walk laid ([`Context::frame_dry`]): the hash of what its
    /// commands would carry, each dispatch's group counts, each readout
    /// copy's regions hashed, the block's image, and the cuts it made.
    #[must_use]
    pub fn into_fill(mut self) -> Fill {
        use std::hash::Hasher;
        // A fill walk's buffer recorded nothing: it goes unended, as meant.
        self.open = false;
        Fill {
            baked: self.baked.borrow().finish(),
            trail: std::mem::take(&mut *self.trail.borrow_mut()),
            groups: std::mem::take(&mut *self.groups.borrow_mut()),
            copies: std::mem::take(&mut *self.copies.borrow_mut()),
            image: self.block.borrow_mut().take_image().seal(),
            cuts: self.cuts(),
            room_pasts: self.room_pasts.borrow().clone(),
        }
    }

    /// Folds `what`, a command the frame records, into its hash.
    pub(crate) fn bake(&self, what: &(impl std::hash::Hash + std::fmt::Debug)) {
        self.bake_named("", what);
    }

    /// [`Self::bake`], the trail naming the command `name` (a kernel's
    /// entry point): what reads a parting as more than a key.
    pub(crate) fn bake_named(&self, name: &str, what: &(impl std::hash::Hash + std::fmt::Debug)) {
        if tracing::enabled!(target: "vkbakeall", tracing::Level::TRACE) {
            tracing::trace!(target: "vkbakeall", what = ?what, "bake");
        }
        what.hash(&mut *self.baked.borrow_mut());
        if trailing() {
            self.trail.borrow_mut().push(format!("{what:?} {name}"));
        }
    }

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

    /// A kept frame's set for a dispatch whose layout and bindings hash to
    /// `key`: one set for every recording that binds the same, for the
    /// device's life; whether it is new (its bindings to write). Its
    /// bindings are written again by each frame that takes it, alike but for
    /// a pool segment's or a score room's, which a rebind or a replay writes
    /// in any case.
    pub(crate) fn shared_set(
        &self,
        layout: vk::DescriptorSetLayout,
        key: u64,
    ) -> Result<vk::DescriptorSet> {
        self.inner.core.shared_set(layout, key).map(|(set, _)| set)
    }

    /// [`Self::shared_set`], and whether it is new: a held one's bindings
    /// are the ones its key hashes (a pool segment's and a score room's
    /// written by a rebind or a replay), written when it was made.
    pub(crate) fn shared_set_fresh(
        &self,
        layout: vk::DescriptorSetLayout,
        key: u64,
    ) -> Result<(vk::DescriptorSet, bool)> {
        self.inner.core.shared_set(layout, key)
    }

    /// The fire block's set of `layout` over `buffer` (its uniform at the
    /// buffer's start, `range` bytes, a dispatch's place its dynamic
    /// offset): one for every dispatch of the kernel over that buffer.
    pub(crate) fn fire_set(
        &self,
        layout: vk::DescriptorSetLayout,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
    ) -> Result<vk::DescriptorSet> {
        use ash::vk::Handle;
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (buffer.as_raw(), range, binding).hash(&mut h);
        let core = &self.inner.core;
        let (set, fresh) = core.shared_set(layout, h.finish())?;
        if fresh {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .offset(0)
                .range(range);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(std::slice::from_ref(&info));
            // SAFETY: the set was just allocated and no command binds it yet;
            // `buffer` is the arm's fire block, live for the device.
            unsafe { core.device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(set)
    }

    /// The frame lays in an arm's fire block (live for the device), not
    /// chunks of its own.
    pub(crate) fn arms_block(&self) -> bool {
        self.block.borrow().arm().is_some()
    }

    /// [`Frame::fire_set`] for an arm-less frame, whose chunks go with it:
    /// a set of the frame's own.
    pub(crate) fn own_fire_set(
        &self,
        layout: vk::DescriptorSetLayout,
        binding: u32,
        buffer: vk::Buffer,
        range: u64,
    ) -> Result<vk::DescriptorSet> {
        let set = self.descriptor_set(layout)?;
        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(range);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(std::slice::from_ref(&info));
        // SAFETY: the set was handed to this dispatch alone; `buffer` is a
        // chunk the frame retains.
        unsafe { self.inner.core.device.update_descriptor_sets(&[write], &[]) };
        Ok(set)
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
        // A frame run once takes its sets a batch at a time (it gives them
        // back reset); a recording keeps every set it takes for its life, so
        // takes one at a time: a body holds the sets its dispatches bind.
        let batch = if self.kept { 1 } else { SETS_PER_BATCH };
        let layouts = vec![layout; batch];
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
                let pool = core.descriptor_pool(self.inner.one_time, pools.len())?;
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
        count_sets(core, batch as u64, self.kept);
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
        self.copy_regions_baked(source, into, regions, true)
    }

    /// [`Frame::copy_regions`]; `bake` false leaves the regions out of the
    /// frame's hash, for a caller that weighs them itself.
    fn copy_regions_baked(
        &mut self,
        source: &Buffer,
        into: &Buffer,
        regions: &[(u64, u64, u64)],
        bake: bool,
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
        if bake {
            use ash::vk::Handle;
            let spans: Vec<(u64, u64, u64)> = regions
                .iter()
                .map(|r| (r.src_offset, r.dst_offset, r.size))
                .collect();
            self.bake(&("copy", source.buffer.as_raw(), into.buffer.as_raw(), spans));
        }
        self.pin_copied(&[source.buffer, into.buffer], "a buffer copy");
        self.fix_room(&[source.buffer, into.buffer]);
        if self.dry {
            return Ok(());
        }
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

    /// Copies each `(source_at, into_at, len)` region of `source` into
    /// `into`, the regions and their count laid in the fire block under a
    /// table of `most` rows. Where the frame's shapes move other regions the
    /// copier kernel moves them from the table, under commands the same for
    /// every fire of at most `most` regions; elsewhere they are copied as
    /// carried commands.
    /// Regions the kernel cannot move (not whole words, past a binding's
    /// range, more than `most`) are copied as [`Frame::copy_regions`] does.
    pub fn copy_rows(
        &mut self,
        source: &Buffer,
        into: &Buffer,
        regions: &[(u64, u64, u64)],
        most: usize,
    ) -> Result<()> {
        use ash::vk::Handle;
        let core = Arc::clone(&self.inner.core);
        let limits = &core.limits;
        let range = u64::from(limits.max_storage_buffer_range);
        let (src, dst) = (source.slab(), into.slab());
        let (src_range, dst_range) = (src.size.min(range), dst.size.min(range));
        // The rows a fire reads out are its lanes': a recording moves them
        // with the copier, their regions its table (data), so one body serves
        // every fire of its commands whatever its lanes read. Rows copied out
        // of the fire's score room move with it too: the room is bound where
        // the fire's lies ([`Kept::bind_room`]).
        let room_src = self.is_room(src.buffer);
        let signature = [u32::MAX, u32::MAX, u32::try_from(most).unwrap_or(u32::MAX)];
        self.copies.borrow_mut().push(signature);
        let words = |n: u64| n.is_multiple_of(4);
        let movable = regions.len() <= most
            && most as u64 <= u64::from(limits.max_compute_work_group_count[1])
            && regions.iter().all(|&(from, to, len)| {
                words(from)
                    && words(to)
                    && words(len)
                    && from + len <= src_range
                    && to + len <= dst_range
                    && (from + len) / 4 <= u64::from(u32::MAX)
                    && (to + len) / 4 <= u64::from(u32::MAX)
            });
        if most == 0 || !movable {
            return self.copy_regions(source, into, regions);
        }
        let mut table = vec![0u8; most * 16];
        for (row, &(from, to, len)) in table.as_chunks_mut::<16>().0.iter_mut().zip(regions) {
            let (words, _) = row.as_chunks_mut::<4>();
            for (word, value) in words.iter_mut().zip([from / 4, to / 4, len / 4]) {
                *word = (value as u32).to_le_bytes();
            }
        }
        let longest = regions
            .iter()
            .map(|&(_, _, len)| len / 4)
            .max()
            .unwrap_or(0);
        let groups = [
            longest
                .div_ceil(u64::from(self.copier.local[0]))
                .min(u64::from(limits.max_compute_work_group_count[0])) as u32,
            regions.len() as u32,
            1,
        ];
        let group_words: Vec<u8> = groups.iter().flat_map(|g| g.to_le_bytes()).collect();
        let fire = self.copier.fire.clone().ok_or_else(|| Fault::Shader {
            file: COPIER.0,
            entrypoint: COPIER.1,
            why: "the copier reads no fire block".to_string(),
        })?;
        let count = (regions.len() as u32).to_le_bytes();
        let (table_buffer, table_place) =
            self.lay(&table, limits.min_storage_buffer_offset_alignment)?;
        let (params_buffer, params_place) =
            self.lay(&count, limits.min_uniform_buffer_offset_alignment)?;
        let (groups_buffer, groups_place) = self.lay(&group_words, 4)?;
        // The regions' own hash stands for them in the family's weighing: a
        // family whose shapes move the same regions copies them as carried
        // commands; one whose shapes differ moves them with the copier, from
        // the table.
        let indirect = self.dispatched(signature) || room_src || self.kept;
        // The count is read from the fire block alone: the push block
        // carries none of it.
        let pushed = vec![0u8; self.copier.push_bytes as usize];
        self.bake(&(
            "rows",
            if room_src { u64::MAX } else { src.buffer.as_raw() },
            dst.buffer.as_raw(),
            table_place,
            params_place,
            groups_place,
            most,
        ));
        if !indirect {
            return self.copy_regions_baked(source, into, regions, false);
        }
        // The copier reads its regions from the table: a room it copies from
        // is bound whole, by its buffer (which the frame's hash carries).
        self.pin_copied(&[src.buffer, dst.buffer], "a readout copier's binding");
        if self.dry {
            return Ok(());
        }
        // A recording's copier set is the device's for every recording that
        // copies alike on the arm (its table lies in the arm's block); a
        // score room's source by its tag, a replay writing it.
        let set = if self.kept {
            use ash::vk::Handle;
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            let source = if room_src && self.copier.rooms & 1 != 0 { 0 } else { src.buffer.as_raw() };
            (
                source,
                src_range,
                dst.buffer.as_raw(),
                dst_range,
                table_buffer.as_raw(),
                table_place.1,
                table.len(),
            )
                .hash(&mut h);
            self.shared_set(self.copier.set_layout, h.finish())?
        } else {
            self.descriptor_set(self.copier.set_layout)?
        };
        if room_src && self.kept && self.copier.rooms & 1 != 0 {
            self.room_binds(set, 0, 0, u64::MAX);
        } else if room_src && self.kept {
            self.inner.room_fixed.store(true, Ordering::Release);
        }
        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(src.buffer)
                .offset(0)
                .range(src_range),
            vk::DescriptorBufferInfo::default()
                .buffer(dst.buffer)
                .offset(0)
                .range(dst_range),
            vk::DescriptorBufferInfo::default()
                .buffer(table_buffer)
                .offset(table_place.1)
                .range(table.len() as u64),
        ];
        let writes: Vec<vk::WriteDescriptorSet<'_>> = infos
            .iter()
            .enumerate()
            .map(|(at, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(at as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // Its count at its place in the fire block, the set 1 offset.
        let fire_layout = self
            .copier
            .fire_layout
            .expect("the copier reads its count from the fire block");
        let fire_set = if self.arms_block() {
            self.fire_set(fire_layout, fire.binding, params_buffer, u64::from(fire.bytes))?
        } else {
            self.own_fire_set(fire_layout, fire.binding, params_buffer, u64::from(fire.bytes))?
        };
        let fire_at = u32::try_from(params_place.1).map_err(|_| Fault::Ceiling {
            what: "the fire block's place of a copy's count, as a dynamic offset",
            need: params_place.1,
            have: u64::from(u32::MAX),
        })?;
        let host = into.is_host();
        let d = &core.device;
        let cmd = self.inner.cmd();
        // SAFETY: `&mut self` is the only recorder of this open frame's `cmd`;
        // `set` was handed to this dispatch alone and no submitted command
        // uses it yet. Both slabs are live and retained below, every region
        // was checked to lie inside the ranges bound, and the table, the
        // count and the group counts lie in fire block chunks the frame
        // retains, at offsets the device's alignments allow.
        unsafe {
            d.update_descriptor_sets(&writes, &[]);
            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.copier.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.copier.layout,
                0,
                &[set, fire_set],
                &[fire_at],
            );
            if self.copier.push_bytes > 0 {
                d.cmd_push_constants(
                    cmd,
                    self.copier.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &pushed,
                );
            }
            d.cmd_dispatch_indirect(cmd, groups_buffer, groups_place.1);
            let (host_stage, host_read) = if host {
                (vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ)
            } else {
                (vk::PipelineStageFlags::empty(), vk::AccessFlags::empty())
            };
            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | host_read,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | host_stage,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        self.retain(src);
        self.retain(dst);
        self.hazards_settled();
        Ok(())
    }

    pub fn fill_zero(&mut self, into: &Buffer, at: u64, len: u64) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        into.span(at, len)?;
        let into = into.slab();
        self.bake(&("zero", ash::vk::Handle::as_raw(into.buffer), at, len));
        self.pin_copied(&[into.buffer], "a buffer fill");
        if self.dry {
            return Ok(());
        }
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

    /// [`engine_gpu::backend::Frame::submit_segment`]: what the frame holds
    /// ended as a segment of its own, recording on in a new buffer from the
    /// same (this thread's) pool that begins with a barrier on every command
    /// before it, as every frame begins. The segments are submitted with the
    /// frame, each its own submission, back to back ([`Frame::submit`]): the
    /// device runs them without waiting on the host's encoding between
    /// them. A kept frame's segment is a cut its replays wait on
    /// ([`Frame::flush`]).
    pub fn submit_segment(&self) -> Result<()> {
        if self.kept || self.dry {
            return self.flush();
        }
        let core = &self.inner.core;
        let d = &core.device;
        let cmd = self.inner.cmd();
        // The segment reads what the walk laid in the fire block so far: its
        // head writes it first (the bytes an earlier segment read again, as
        // they were).
        let head = self.staged_head()?;
        // SAFETY: as in `flush`: this frame's `cmd` is recording and only
        // `&self` records into it.
        self.inner.end_cmd().map_err(|e| core.fault("vkEndCommandBuffer", e))?;
        // The segment's head runs before it, submitted with it.
        if let Some(head) = head {
            self.inner.held.lock().expect("held lock poisoned").push(head);
        }
        self.inner
            .segments
            .lock()
            .expect("segments lock poisoned")
            .push(cmd);
        self.inner
            .held
            .lock()
            .expect("held lock poisoned")
            .push(cmd);
        // SAFETY: `cmd_pool` is this frame's (this thread's) pool.
        let next = core.alloc_cmd(self.inner.cmd_pool)?;
        self.inner
            .cmd
            .store(vk::Handle::as_raw(next), Ordering::Release);
        // SAFETY: `next` was just allocated, in the initial state; the
        // barrier records into it while only this frame holds it.
        unsafe {
            d.begin_command_buffer(
                next,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
            self.inner.begun();
            let all = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
            d.cmd_pipeline_barrier(
                next,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[all],
                &[],
                &[],
            );
        }
        self.hazards_settled();
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        self.bake(&"flush");
        if self.dry {
            // A cut the fill's body was recorded with: counted, nothing run.
            self.inner
                .cuts
                .lock()
                .expect("cuts lock poisoned")
                .push(vk::CommandBuffer::null());
            return Ok(());
        }
        let core = &self.inner.core;
        let d = &core.device;
        let cmd = self.inner.cmd();
        // In `flush`, `Frame` is not `Sync`, so `&self` here is the only
        // thread using this frame's `cmd` and fence, and the frame is open:
        // every method that ends it consumes the frame.
        let head = self.staged_head()?;
        // SAFETY: `cmd` is recording, so it may be ended.
        self.inner.end_cmd().map_err(|e| core.fault("vkEndCommandBuffer", e))?;
        {
            let bufs: Vec<vk::CommandBuffer> = head.into_iter().chain([cmd]).collect();
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // A split step's held segments go first, in order.
            let held = std::mem::take(&mut *self.inner.held.lock().expect("held lock poisoned"));
            for segment in held {
                let bufs = [segment];
                let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
                // SAFETY: `segment` was ended and never submitted; it has no
                // fence; the queue is held under its lock.
                unsafe { d.queue_submit(*queue, &submits, vk::Fence::null()) }
                    .map_err(|e| core.fault("vkQueueSubmit", e))?;
                self.inner.segmented.store(true, Ordering::Release);
            }
            // SAFETY: `cmd` is executable, the fence is unsignaled (reset after
            // each wait) and the queue is held under its lock.
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
            self.inner.submitted_now();
        }
        // SAFETY: the fence is this frame's own, just submitted.
        unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
            .map_err(|e| core.fault("vkWaitForFences", e))?;
        self.inner.landed_now();
        // SAFETY: the wait returned success, so no submission names the fence.
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        if self.kept {
            // The segment stays for replays; the frame records on in a new
            // buffer from the same (this thread's) pool.
            // SAFETY: `cmd_pool` is the pool this frame's buffers came from,
            // owned by this thread (frames record on the thread that opened
            // them).
            let next = core.alloc_cmd(self.inner.cmd_pool)?;
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
            self.inner.begun();
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
        self.inner.begun();
        self.hazards_settled();
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        self.open = false;
        if !self.inner.recording.load(Ordering::Acquire) {
            return Err(self.inner.core.fault("vkEndCommandBuffer", vk::Result::ERROR_UNKNOWN));
        }
        let core = &self.inner.core;
        self.inner.end_cmd().map_err(|e| core.fault("vkEndCommandBuffer", e))
    }

    fn submit(&mut self) -> Result<()> {
        let head = self.staged_head()?;
        self.end()?;
        let core = &self.inner.core;
        let bufs: Vec<vk::CommandBuffer> = head.into_iter().chain([self.inner.cmd()]).collect();
        let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
        let queue = core.queue.lock().expect("queue lock poisoned");
        // A split step's segments first, each its own submission, back to
        // back under the one hold of the queue.
        let held = std::mem::take(&mut *self.inner.held.lock().expect("held lock poisoned"));
        for segment in held {
            let bufs = [segment];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            // SAFETY: `segment` was ended and never submitted; it has no
            // fence; the queue is held under its lock.
            unsafe { core.device.queue_submit(*queue, &submits, vk::Fence::null()) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
            self.inner.segmented.store(true, Ordering::Release);
        }
        // SAFETY: `cmd` was just ended and never submitted, the fence is the
        // frame's own unsignaled one, and the queue is held under its lock.
        unsafe { core.device.queue_submit(*queue, &submits, self.inner.fence) }
            .map_err(|e| core.fault("vkQueueSubmit", e))?;
        self.inner.submitted_now();
        self.inner.submitted.store(true, Ordering::Release);
        Ok(())
    }

    pub fn commit(mut self) -> Result<()> {
        self.submit()?;
        Pending::watched(Pending::of_frame(&self.inner), None).wait()
    }

    pub fn commit_timed(mut self) -> Result<f64> {
        let started = std::time::Instant::now();
        self.submit()?;
        Pending::watched(Pending::of_frame(&self.inner), None).wait()?;
        Ok(started.elapsed().as_secs_f64())
    }

    /// Submits and waits: the device's time between the two timestamps a
    /// [`Context::spanned`] frame writes, ns; `None` for another frame.
    pub fn commit_spanned(mut self) -> Result<Option<u64>> {
        self.stamp_end();
        self.submit()?;
        Pending::watched(Pending::of_frame(&self.inner), None).wait()?;
        self.inner.span_ns()
    }

    /// A spanned frame's closing timestamp, written while it is open.
    fn stamp_end(&self) {
        let span = self
            .inner
            .span
            .lock()
            .expect("span lock poisoned")
            .as_ref()
            .map(|stamps| stamps.0.pool);
        if let Some(pool) = span
            && self.open
        {
            // SAFETY: the frame is open, so `cmd` is recording; query 1 of
            // the pool was reset when the frame opened.
            unsafe {
                self.inner.core.device.cmd_write_timestamp(
                    self.inner.cmd(),
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    1,
                );
            }
        }
    }

    /// A [`Context::spanned_kept`] frame, its closing timestamp written,
    /// kept to replay ([`Kept::replay_spanned`]).
    pub fn into_kept_spanned(self) -> Result<Kept> {
        self.into_kept()
    }

    /// Kept to replay; a spanned frame's closing timestamp written first,
    /// so each replay writes both anew.
    pub fn into_kept(mut self) -> Result<Kept> {
        use std::hash::Hasher;
        self.stamp_end();
        self.end()?;
        let head = self.head(false)?;
        // A rebind writes the pool segments it binds, as they change.
        if !self
            .inner
            .rebinds
            .lock()
            .expect("rebinds lock poisoned")
            .is_empty()
        {
            let mut rebound = self.inner.core.rebound.lock().expect("rebound lock poisoned");
            rebound.retain(|held| held.strong_count() > 0);
            rebound.push(Arc::downgrade(&self.inner));
        }
        Ok(Kept {
            head,
            copy: std::sync::atomic::AtomicBool::new(true),
            inner: Arc::clone(&self.inner),
            block: std::mem::take(&mut *self.block.borrow_mut()),
            baked: self.baked.borrow().finish(),
            trail: std::mem::take(&mut *self.trail.borrow_mut()),
            groups: std::mem::take(&mut *self.groups.borrow_mut()),
            copies: std::mem::take(&mut *self.copies.borrow_mut()),
            indirect: self.indirect.clone(),
            indirect_all: true,
        })
    }

    /// Submits; `on_done` runs once it lands. A spanned frame's closing
    /// timestamp is written first, so [`Pending::span_ns`] reads its device
    /// time once it landed.
    pub fn commit_async(mut self, on_done: Option<engine_gpu::backend::OnDone>) -> Result<Pending> {
        self.stamp_end();
        let at = crate::encode::profiling().then(std::time::Instant::now);
        self.submit()?;
        if let Some(at) = at {
            crate::encode::credit_submit(at.elapsed().as_nanos() as u64);
        }
        Ok(Pending::watched(Pending::of_frame(&self.inner), on_done))
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // A frame dropped open was never committed: an error unwound the
        // walk that recorded it, and its buffer is freed with it (a buffer
        // may be freed while recording). It is not ended: after an error the
        // buffer may be in no state to end (a driver that failed an
        // allocation inside it, or a flush that ended it and failed before
        // the next began), and ending it there faults in the driver.
        if std::mem::take(&mut self.open) && !self.dry {
            tracing::warn!(
                recording = self.inner.recording.load(Ordering::Acquire),
                "vulkan: an open frame dropped uncommitted; its buffer freed unended"
            );
        }
    }
}

/// What a fill walk laid ([`Frame::into_fill`]).
#[derive(Debug)]
pub struct Fill {
    pub baked: u64,
    /// What each command carried, rendered ([`trailing`]); empty elsewhere.
    pub trail: Vec<String>,
    pub groups: Vec<[u32; 3]>,
    pub copies: Vec<[u32; 3]>,
    pub image: super::block::Image,
    pub cuts: usize,
    pub room_pasts: Vec<u64>,
}

pub struct Kept {
    inner: Arc<Inner>,

    /// What a submission of it runs first ([`Frame::head`]), where the
    /// block does not hold its image already.
    head: Option<vk::CommandBuffer>,

    /// The next replay runs the head: its image reached the staging slice
    /// since the last head copied it.
    copy: std::sync::atomic::AtomicBool,

    block: super::block::FireBlock,

    baked: u64,

    trail: Vec<String>,

    groups: Vec<[u32; 3]>,

    copies: Vec<[u32; 3]>,

    indirect: Option<Arc<Vec<bool>>>,

    /// Every dispatch reads its group counts from the block (a body whose
    /// rows vary).
    indirect_all: bool,
}

impl std::fmt::Debug for Kept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kept").finish()
    }
}

impl Kept {
    /// Binds the score room `room` (its buffer at byte `at`) in every
    /// descriptor that binds its recording's room, where it moved: written
    /// with the arm staged, whose last fire has landed, so no submission in
    /// flight reads the set. Whether the recording replays at it: one whose
    /// commands name the room it was recorded at (a copy of its rows)
    /// replays only there.
    pub fn bind_room(&self, room: Option<(&Buffer, u64)>, pasts: &[u64]) -> Result<bool> {
        let mut binds = self.inner.room_binds.lock().expect("room lock poisoned");
        let mut bound = self.inner.room_bound.lock().expect("room lock poisoned");
        let now = room.map(|(buffer, at)| (buffer.slab().buffer, at));
        if binds.is_empty() {
            return Ok(true);
        }
        // The kernels' room bindings take the fire's places in the room (a
        // copier's whole-buffer binding none: its regions are its table).
        let views: Vec<usize> = (0..binds.len()).filter(|&at| binds[at].3 != u64::MAX).collect();
        if views.len() != pasts.len() {
            return Ok(false);
        }
        let moved = views.iter().zip(pasts).any(|(&at, &past)| binds[at].2 != past);
        if now == *bound && !moved {
            return Ok(true);
        }
        for (&at, &past) in views.iter().zip(pasts) {
            binds[at].2 = past;
        }
        if self.inner.room_fixed.load(Ordering::Acquire) {
            return Ok(false);
        }
        let Some((buffer, at)) = room else {
            return Ok(false);
        };
        // A copier binds its source buffer whole, which may hold another
        // plane's rows beside the room's: it follows a room within its
        // buffer, never into another.
        let whole = binds.iter().any(|&(_, _, _, bytes)| bytes == u64::MAX);
        if whole && bound.is_some_and(|(held, _)| held != buffer.slab().buffer) {
            return Ok(false);
        }
        let range = u64::from(self.inner.core.limits.max_storage_buffer_range);
        let infos: Vec<vk::DescriptorBufferInfo> = binds
            .iter()
            .map(|&(_, _, past, bytes)| {
                // `u64::MAX` bytes: the room's buffer whole (a copier's
                // source, its regions the fire's table).
                let (offset, bytes) = match bytes {
                    u64::MAX => (0, buffer.bytes()),
                    // To the buffer's end, as a view of the room binds.
                    0 => (at + past, buffer.bytes().saturating_sub(at + past)),
                    bytes => (at + past, bytes),
                };
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.slab().buffer)
                    .offset(offset)
                    .range(bytes.min(buffer.bytes().saturating_sub(offset)).min(range))
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = binds
            .iter()
            .zip(&infos)
            .map(|(&(set, binding, _, _), info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: every set is this kept frame's, each binding laid
        // update-after-bind (only those are noted), and no submission in
        // flight reads them: the caller holds the arm, whose last fire
        // landed. The room's buffer is live, held by its pair.
        unsafe { self.inner.core.device.update_descriptor_sets(&writes, &[]) };
        *bound = now;
        Ok(true)
    }

    /// What binds a pool segment in it where no rebind writes it, if any: it
    /// stands only while the segments do.
    #[must_use]
    pub fn pinned(&self) -> Option<String> {
        self.inner.pinned.lock().expect("pinned lock poisoned").clone()
    }

    /// What its commands carry, hashed ([`Frame::bake`]).
    #[must_use]
    pub fn trail(&self) -> &[String] {
        &self.trail
    }

    #[must_use]
    pub fn baked(&self) -> u64 {
        self.baked
    }

    /// What its walk laid in its fire block.
    #[must_use]
    pub fn image(&self) -> &super::block::Image {
        self.block.image()
    }

    /// Each dispatch's group counts in its walk, in order.
    #[must_use]
    pub fn groups(&self) -> &[[u32; 3]] {
        &self.groups
    }

    /// Each readout copy's mark (its most rows), in order.
    #[must_use]
    pub fn copies(&self) -> &[[u32; 3]] {
        &self.copies
    }

    /// Whether its commands carry their rows: no dispatch reads its group
    /// counts from the block.
    #[must_use]
    pub fn fixed(&self) -> bool {
        !self.indirect_all && self.indirect.as_ref().is_none_or(|set| !set.iter().any(|&v| v))
    }

    /// Whether dispatch `at` of its walk reads its group counts from the
    /// fire block.
    #[must_use]
    pub fn indirect(&self, at: usize) -> bool {
        self.indirect_all
            || self
                .indirect
                .as_ref()
                .is_some_and(|set| set.get(at).copied().unwrap_or(false))
    }

    /// Writes `image`, a walk's whose commands hash as this frame's, into
    /// its arm's staging slice, where its next submission's head copies it
    /// from. The caller holds the arm: its last fire has landed, so nothing
    /// in flight reads the slice and no wait is owed.
    pub fn stage(&self, image: &super::block::Image) -> Result<()> {
        match self.block.arm() {
            Some(arm) => {
                let wrote = arm.stage(image)?;
                // The block holds the image already where the slice did:
                // the next replay runs no head.
                self.copy.store(wrote, Ordering::Release);
                Ok(())
            }
            None => Err(Fault::Unbound {
                what: "an arm's fire block for a kept frame walked without one".to_string(),
            }),
        }
    }

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
            self.inner.landed_now();
        }
        // SAFETY: no pending submission names the fence (waited above, or
        // the last one was a segment, waited and reset below).
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        self.inner.submitted.store(false, Ordering::Release);
        {
            let head = self.head.filter(|_| at == 0);
            let bufs: Vec<vk::CommandBuffer> = head.into_iter().chain([cmd]).collect();
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // SAFETY: `cmd` (and the head) are executable (ended when the
            // next segment began) and not pending; the fence is unsignaled;
            // the queue is held under its lock.
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
            self.inner.submitted_now();
        }
        // SAFETY: the fence is this frame's own, just submitted.
        unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
            .map_err(|e| core.fault("vkWaitForFences", e))?;
        self.inner.landed_now();
        // SAFETY: the wait returned, so no submission names the fence.
        unsafe { d.reset_fences(&[self.inner.fence]) }.map_err(|e| core.fault("vkResetFences", e))
    }

    /// Replays a [`Frame::into_kept_spanned`] frame and waits: the device's
    /// time between its timestamps, ns (`None` where it writes none).
    pub fn replay_spanned(&self) -> Result<Option<u64>> {
        self.replay(None)?.wait()?;
        self.inner.span_ns()
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
        // this). Each replay takes a fence of its own, so it waits on no
        // earlier one: the queue orders it behind its arm's last fire, which
        // had landed before the arm was taken again.
        let fence = core.lend_fence()?;
        let at = crate::encode::profiling().then(std::time::Instant::now);
        {
            let head = self.head.filter(|_| self.copy.swap(true, Ordering::AcqRel));
            let bufs: Vec<vk::CommandBuffer> = head.into_iter().chain([self.inner.cmd()]).collect();
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core.queue.lock().expect("queue lock poisoned");
            // SAFETY: the head and `cmd` are executable and recorded without
            // `ONE_TIME_SUBMIT`, so they may be pending again; `fence` was
            // just lent unsignaled; the queue is held under its lock.
            if let Err(e) = unsafe { d.queue_submit(*queue, &submits, fence) } {
                drop(queue);
                core.return_fence(fence);
                if let Some(arm) = self.block.arm() {
                    arm.forget_staged();
                }
                return Err(core.fault("vkQueueSubmit", e));
            }
        }
        if let Some(at) = at {
            crate::encode::credit_submit(at.elapsed().as_nanos() as u64);
        }
        let submission = Arc::new(Submission {
            inner: Arc::clone(&self.inner),
            fence,
            lent: true,
            landed: std::sync::atomic::AtomicBool::new(false),
        });
        *core.last.lock().expect("last lock poisoned") = Last::Replay(Arc::downgrade(&submission));
        Ok(Pending::watched(
            submission,
            on_done.map(|f| -> engine_gpu::backend::OnDone { Box::new(move |refused| f(refused)) }),
        ))
    }
}

/// The submission that went to the queue last ([`Context::idle`]): a
/// frame's, on its own fence, or a replay's, on a lent one.
enum Last {
    None,
    Frame(std::sync::Weak<Inner>),
    Replay(std::sync::Weak<Submission>),
}

/// What a frame's one fence waiter tells every other: the fence's wait
/// returned (and with what), after the frame's `on_done` ran. A fence has
/// one thread blocked on it at most: a second `vkWaitForFences` on the same
/// fence wakes on a coarse timer rather than the signal (about 5.7 ms late
/// on an RTX PRO 4500, a ~10 ms step per short fire).
#[derive(Default)]
struct Landing {
    result: Mutex<Option<std::result::Result<(), vk::Result>>>,
    landed: std::sync::Condvar,
}

impl Landing {
    fn land(&self, result: std::result::Result<(), vk::Result>) {
        *self.result.lock().expect("landing lock poisoned") = Some(result);
        self.landed.notify_all();
    }

    fn wait(&self) -> std::result::Result<(), vk::Result> {
        let mut result = self.result.lock().expect("landing lock poisoned");
        loop {
            if let Some(result) = *result {
                return result;
            }
            result = self.landed.wait(result).expect("landing lock poisoned");
        }
    }
}

/// A submission's fence and the frame it ran: a replay's fence is lent from
/// the device's spares and handed back once the submission landed.
struct Submission {
    inner: Arc<Inner>,
    fence: vk::Fence,
    lent: bool,
    landed: std::sync::atomic::AtomicBool,
}

impl Submission {
    /// Notes the wait on the fence returned: a lent fence goes back unwaited,
    /// and a frame's own one is not waited on again when the frame drops.
    fn landed(&self, ok: bool) {
        self.landed.store(ok, Ordering::Release);
        if ok && !self.lent {
            // Its segments landed before its last (which began with a
            // barrier on every command ahead of it).
            self.inner.submitted.store(false, Ordering::Release);
            self.inner.segmented.store(false, Ordering::Release);
        }
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if !self.lent {
            return;
        }
        // A lent fence goes back once nothing waits on it: its wait has
        // returned (the watcher's, or the caller's), else it is waited on
        // here (a submission dropped unwaited, as on a refused step).
        if !self.landed.load(Ordering::Acquire) {
            let _ = self.inner.block_on_fence(self.fence);
        }
        self.inner.core.return_fence(self.fence);
    }
}

/// A watched submission: the watcher waits on its fence, runs `on_done`,
/// then releases every [`Pending::wait`] through `landing`.
struct Watch {
    submission: Arc<Submission>,
    on_done: engine_gpu::backend::OnDone,
    landing: Arc<Landing>,
}

impl Watch {
    fn run(self) {
        let submission = self.submission;
        let result = submission.inner.block_on_fence(submission.fence);
        submission.landed(result.is_ok());
        if result.is_ok() {
            submission.inner.landed_pools();
        }
        (self.on_done)(
            result
                .err()
                .map(|e| submission.inner.core.fault("vkWaitForFences", e).to_string()),
        );
        self.landing.land(result);
    }
}

/// A clone is the same submission: each of the sessions whose passes one
/// frame carries keeps one, and the frame's last drop waits on its fence.
#[derive(Clone)]
pub struct Pending {
    submission: Arc<Submission>,
    /// Set where the watcher is the fence's one waiter.
    landing: Option<Arc<Landing>>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish()
    }
}

impl Pending {
    /// A frame's one submission, on its own fence.
    fn of_frame(inner: &Arc<Inner>) -> Arc<Submission> {
        Arc::new(Submission {
            inner: Arc::clone(inner),
            fence: inner.fence,
            lent: false,
            landed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[must_use]
    pub fn landed(&self) -> bool {
        // SAFETY: the fence lives as long as the submission, which `self`
        // holds; querying a fence needs no external sync.
        unsafe {
            self.submission
                .inner
                .core
                .device
                .get_fence_status(self.submission.fence)
        }
        .unwrap_or(true)
    }

    /// A pending submission; where `on_done` is given, the device's watcher
    /// thread blocks on the fence, runs it, then releases every
    /// [`Pending::wait`], so the fence has the one waiter.
    fn watched(submission: Arc<Submission>, on_done: Option<engine_gpu::backend::OnDone>) -> Pending {
        let Some(on_done) = on_done else {
            return Pending {
                submission,
                landing: None,
            };
        };
        let landing = Arc::new(Landing::default());
        let watch = Watch {
            submission: Arc::clone(&submission),
            on_done,
            landing: Arc::clone(&landing),
        };
        let core = &submission.inner.core;
        let sent = core
            .watcher
            .lock()
            .expect("watcher lock poisoned")
            .as_ref()
            .map(|watcher| watcher.send(watch));
        if let Some(Err(std::sync::mpsc::SendError(watch))) = sent {
            // The watcher is gone only with the device: wait here instead.
            watch.run();
        }
        Pending {
            submission,
            landing: Some(landing),
        }
    }

    pub fn wait(&self) -> Result<()> {
        let inner = &self.submission.inner;
        let core = &inner.core;
        let at = crate::encode::profiling().then(std::time::Instant::now);
        match &self.landing {
            Some(landing) => landing.wait(),
            None => {
                let result = inner.block_on_fence(self.submission.fence);
                self.submission.landed(result.is_ok());
                result
            }
        }
        .map_err(|e| core.fault("vkWaitForFences", e))?;
        if let Some(at) = at {
            crate::encode::credit_fence(at.elapsed().as_nanos() as u64);
        }
        inner.landed_pools();
        inner.harvest_timing();
        Ok(())
    }

    /// The device's time between a spanned frame's two timestamps, ns, read
    /// once it landed (its fence waited: the read waits on nothing more);
    /// `None` for a frame that writes none.
    pub fn span_ns(&self) -> Result<Option<u64>> {
        self.submission.inner.span_ns()
    }
}

impl Inner {
    /// A one-time frame landed: its descriptor pools go back now, not when
    /// its last handle drops (a session holds a landed pass's until the
    /// next departs).
    fn landed_pools(&self) {
        if self.one_time {
            self.give_pools();
        }
    }

    /// Gives every descriptor pool the frame holds back to the spares, and
    /// the sets it kept of them: once only.
    fn give_pools(&self) {
        let pools: Vec<vk::DescriptorPool> =
            std::mem::take(&mut *self.pools.lock().expect("pools lock poisoned"));
        self.spare.lock().expect("spare lock poisoned").clear();
        self.core.give_pools(pools, self.one_time);
    }

    /// Blocks on `fence`, one of this frame's submissions'. Only its one
    /// waiter calls this: the watcher where it has one, else
    /// [`Pending::wait`].
    fn block_on_fence(&self, fence: vk::Fence) -> std::result::Result<(), vk::Result> {
        let waiting = self.waiters.fetch_add(1, Ordering::AcqRel) + 1;
        self.most_waiters.fetch_max(waiting, Ordering::AcqRel);
        // SAFETY: the fence lives as long as `self`; waiting needs no
        // external sync.
        let result = unsafe {
            self.core
                .device
                .wait_for_fences(&[fence], true, FENCE_TIMEOUT_NS)
        };
        self.waiters.fetch_sub(1, Ordering::AcqRel);
        if result.is_ok() {
            self.landed_now();
        }
        result
    }

    /// A fenced submission of this frame went to the queue: the device's
    /// last ([`Context::idle`]).
    fn submitted_now(self: &Arc<Self>) {
        self.seq.fetch_add(1, Ordering::AcqRel);
        *self.core.last.lock().expect("last lock poisoned") = Last::Frame(Arc::downgrade(self));
    }

    /// This frame's fence was seen signaled: every submission of it so far
    /// has landed.
    fn landed_now(&self) {
        self.landed
            .fetch_max(self.seq.load(Ordering::Acquire), Ordering::AcqRel);
    }

    fn harvest_timing(&self) {
        let mut slot = self.timing.lock().expect("timing lock poisoned");
        let Some(timing) = slot.as_mut() else {
            return;
        };
        if timing.harvested || timing.names.is_empty() {
            return;
        }
        // A kept frame's timestamps are written again on every replay.
        timing.harvested = self.one_time;
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
        crate::encode::dump_profile();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use ash::vk;

    use super::{Context, Ordering, device_memory};

    /// The phone (Adreno), an integrated GPU and the Spark share the host's
    /// memory: no host tier there, and a stated one is refused by name. A
    /// discrete card, its heap mapped or not, is unchanged.
    #[test]
    fn a_device_whose_memory_is_the_hosts_states_no_host_tier() {
        let integrated = device_memory(vk::PhysicalDeviceType::INTEGRATED_GPU, true);
        assert_eq!(integrated, engine::DeviceMemory::Hosts);
        assert_eq!(integrated.host_tier(None), Ok(engine::HostTier::Absent));
        let refused = integrated.host_tier(Some(1 << 30)).unwrap_err();
        assert!(refused.contains("`[engine.memory] budget_host`"), "{refused}");
        assert_eq!(
            device_memory(vk::PhysicalDeviceType::VIRTUAL_GPU, true),
            engine::DeviceMemory::Hosts
        );
        if !cfg!(target_os = "android") {
            for mapped in [false, true] {
                let discrete = device_memory(vk::PhysicalDeviceType::DISCRETE_GPU, mapped);
                assert_eq!(discrete, engine::DeviceMemory::Own);
                assert_eq!(
                    discrete.host_tier(Some(1 << 30)),
                    Ok(engine::HostTier::Seated(Some(1 << 30)))
                );
            }
            assert_eq!(
                device_memory(vk::PhysicalDeviceType::VIRTUAL_GPU, false),
                engine::DeviceMemory::Own
            );
        }
    }

    /// A storage-buffer set layout, as a kernel's.
    fn layout(context: &Context) -> ash::vk::DescriptorSetLayout {
        use ash::vk;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        // SAFETY: the device is live and the create info outlives the call.
        unsafe {
            context
                .core
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("a set layout")
        }
    }

    /// Serving makes no descriptor pool: with every stocked pool held by a
    /// frame in flight, one more frame waits for one to come back rather
    /// than making its own, and every pool is free again once they land.
    #[test]
    fn serving_with_every_pool_lent_waits_and_makes_none() {
        let bound = Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE);
        let Ok(context) = bound else {
            eprintln!("not asked: {:?}", bound.err());
            return;
        };
        let layout = layout(&context);
        context.seal_descriptor_pools(2).unwrap();
        let (stock, lent, _) = context.descriptor_pools();
        assert_eq!(lent, 0);
        // Every stocked pool lent, each to a frame on the queue.
        let busy: Vec<super::Frame> = (0..stock)
            .map(|_| {
                let frame = context.frame().unwrap();
                frame.descriptor_set(layout).unwrap();
                frame
            })
            .collect();
        assert_eq!(context.descriptor_pools(), (0, stock, 0));
        let context = Arc::new(context);
        let waiting = {
            let context = Arc::clone(&context);
            std::thread::spawn(move || {
                let frame = context.frame().unwrap();
                frame.descriptor_set(layout).unwrap();
                frame.commit().unwrap();
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!waiting.is_finished(), "a frame made a pool while serving");
        for frame in busy {
            frame.commit().unwrap();
        }
        waiting.join().unwrap();
        assert_eq!(
            context.descriptor_pools(),
            (stock, 0, 0),
            "a pool was made after the seal, or one never came back"
        );
        // SAFETY: no set of the layout is in use: every frame landed.
        unsafe {
            context
                .core
                .device
                .destroy_descriptor_set_layout(layout, None)
        };
    }

    /// Where no frame holds a pool to give back, a frame asking for one
    /// while serving is refused by name, not given a new pool.
    #[test]
    fn a_frame_finding_no_pool_and_none_lent_is_refused() {
        let bound = Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE);
        let Ok(context) = bound else {
            eprintln!("not asked: {:?}", bound.err());
            return;
        };
        let layout = layout(&context);
        context.seal_descriptor_pools(0).unwrap();
        let (stock, ..) = context.descriptor_pools();
        let held: Vec<super::Frame> = (0..stock)
            .map(|_| {
                let frame = context.frame_kept().unwrap();
                frame.descriptor_set(layout).unwrap();
                frame
            })
            .collect();
        let frame = context.frame().unwrap();
        let refused = frame.descriptor_set(layout).unwrap_err().to_string();
        assert!(refused.contains("no spare descriptor pool"), "{refused}");
        drop(frame);
        drop(held);
        // SAFETY: no set of the layout is in use.
        unsafe {
            context
                .core
                .device
                .destroy_descriptor_set_layout(layout, None)
        };
    }

    /// A build marks the cache changed and returns without writing it (no
    /// submission or build waits on the file); the cache's writer writes it,
    /// at the latest when the cache goes.
    #[test]
    fn a_built_cache_is_written_off_the_build_and_by_the_time_it_goes() {
        let dir = std::env::temp_dir().join(format!("pie-vk-cache-{}", std::process::id()));
        let path = dir.join("pipelines.bin");
        let _ = std::fs::remove_file(&path);
        let boot = crate::api::DeviceBoot {
            pipeline_cache: Some(path.clone()),
            ..crate::api::DeviceBoot::default()
        };
        let bound = Context::bind(&boot, engine::Allowance::WHOLE);
        let Ok(context) = bound else {
            eprintln!("not asked: {:?}", bound.err());
            return;
        };
        context.pipeline_cache().built();
        assert!(!path.exists(), "the build wrote the cache itself");
        drop(context);
        assert!(
            path.exists(),
            "the cache went without its writer writing it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A replayed frame's fence has one waiter, the watcher that runs its
    /// `on_done`; the caller's wait returns when that watcher releases it,
    /// not on a second `vkWaitForFences` (which wakes on a coarse timer).
    #[test]
    fn a_replay_blocks_one_thread_on_its_fence_and_releases_the_caller() {
        let bound = Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE);
        let Ok(context) = bound else {
            eprintln!("not asked: {:?}", bound.err());
            return;
        };
        let kept = context.frame_kept().unwrap().into_kept().unwrap();
        let mut lags = Vec::new();
        for _ in 0..32 {
            let ran: Arc<Mutex<Option<Instant>>> = Arc::default();
            let told = Arc::clone(&ran);
            let pending = kept
                .replay(Some(Box::new(move |refused| {
                    assert_eq!(refused, None);
                    *told.lock().unwrap() = Some(Instant::now());
                })))
                .unwrap();
            pending.wait().unwrap();
            let woke = Instant::now();
            let ran = ran
                .lock()
                .unwrap()
                .expect("on_done ran before the wait returned");
            lags.push(woke.duration_since(ran));
        }
        assert_eq!(kept.inner.most_waiters.load(Ordering::Acquire), 1);
        lags.sort();
        let median = lags[lags.len() / 2];
        let worst = *lags.last().unwrap();
        eprintln!("caller woke {median:?} (median), {worst:?} (worst) after the watcher's landing");
    }
}

