use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::api::DeviceBoot;
use crate::error::{Fault, Result};

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
        .get_or_init(|| unsafe { ash::Entry::load() }.map_err(|e| e.to_string()))
        .clone()
}

fn instance_gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GATE.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[must_use]
pub fn present() -> bool {
    let Ok(entry) = entry() else {
        return false;
    };
    let _gate = instance_gate();
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    let Ok(instance) = (unsafe { entry.create_instance(&info, None) }) else {
        return false;
    };
    let found = unsafe { instance.enumerate_physical_devices() }
        .map(|devices| {
            devices.iter().any(|&physical| {
                unsafe { instance.get_physical_device_queue_family_properties(physical) }
                    .iter()
                    .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
        })
        .unwrap_or(false);
    unsafe { instance.destroy_instance(None) };
    found
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

    pub(crate) family: u32,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    pub(crate) transfer: Mutex<Transfer>,
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub(crate) allocated: AtomicU64,
    pub(crate) name: String,
    pub(crate) subgroup_size: u32,
    pub(crate) cores: u32,
    pub(crate) device_local: u64,
    pub(crate) vendor_id: u32,
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
                let queue = self
                    .queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            transfer.staging = Some(Raw::new(self, STAGING_BYTES, Memory::Staging)?);
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
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
        {
            return Ok(fence);
        }
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if spare.len() >= SPARE_FENCES || unsafe { self.device.reset_fences(&[fence]) }.is_err() {
            drop(spare);
            unsafe { self.device.destroy_fence(fence, None) };
            return;
        }
        spare.push(fence);
    }

    pub(crate) fn thread_pool(&self) -> Result<vk::CommandPool> {
        let mut pools = self
            .pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(pool) = pools.get(&std::thread::current().id()) {
            return Ok(*pool);
        }
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
        unsafe {
            let _ = self.device.device_wait_idle();
            {
                let transfer = self
                    .transfer
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                transfer.staging = None;
                self.device.destroy_fence(transfer.fence, None);
            }
            for fence in std::mem::take(
                self.spare_fences
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            ) {
                self.device.destroy_fence(fence, None);
            }
            let pools = std::mem::take(
                self.pools
                    .get_mut()
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            for pool in pools.into_values() {
                self.device.destroy_command_pool(pool, None);
            }
            self.device.destroy_device(None);
            if let Some((debug, messenger)) = self.messenger.take() {
                debug.destroy_debug_utils_messenger(messenger, None);
            }
            let _gate = instance_gate();
            self.instance.destroy_instance(None);
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
    let message = unsafe { (*data).message_as_c_str() }
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("vulkan validation [{:#x}]: {message}", severity.as_raw());
    vk::FALSE
}

pub struct Context {
    pub(crate) core: Arc<Core>,
    working_set: u64,
    max_buffer: u64,
    pub(crate) dummy: Slab,
    pipeline_cache: Option<std::path::PathBuf>,
    device_index: u32,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("name", &self.core.name)
            .field("working_set", &self.working_set)
            .finish()
    }
}

fn has_extension(exts: &[vk::ExtensionProperties], name: &CStr) -> bool {
    exts.iter()
        .any(|e| e.extension_name_as_c_str().is_ok_and(|s| s == name))
}

impl Context {
    pub fn bind(boot: &DeviceBoot) -> Result<Context> {
        let entry = entry().map_err(|e| Fault::NoDevice {
            detail: format!("no Vulkan loader: {e}"),
        })?;
        let _gate = instance_gate();
        let app = vk::ApplicationInfo::default()
            .application_name(c"pie")
            .api_version(vk::API_VERSION_1_3);
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let validation = c"VK_LAYER_KHRONOS_validation";
        let validated = boot.validation
            && layers
                .iter()
                .any(|l| l.layer_name_as_c_str().is_ok_and(|s| s == validation));
        let enabled_layers: Vec<*const c_char> = if validated {
            vec![validation.as_ptr()]
        } else {
            Vec::new()
        };
        let mut enabled_exts: Vec<*const c_char> = Vec::new();
        if validated {
            enabled_exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let instance_exts =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let props2 = has_extension(
            &instance_exts,
            ash::khr::get_physical_device_properties2::NAME,
        );
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_exts);
        let instance =
            unsafe { entry.create_instance(&info, None) }.map_err(|e| Fault::NoDevice {
                detail: format!("vkCreateInstance: {e}"),
            })?;
        let messenger = if validated {
            let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
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
            unsafe { debug.create_debug_utils_messenger(&create, None) }
                .ok()
                .map(|m| (debug, m))
        } else {
            None
        };
        match Self::finish(entry, instance, messenger, boot, props2) {
            Ok(context) => {
                kernels_vulkan::tuning::describe(context.info());
                Ok(context)
            }
            Err((fault, Some((instance, messenger)))) => {
                unsafe {
                    if let Some((debug, m)) = messenger {
                        debug.destroy_debug_utils_messenger(m, None);
                    }
                    instance.destroy_instance(None);
                }
                Err(fault)
            }
            Err((fault, None)) => Err(fault),
        }
    }

    #[allow(
        clippy::type_complexity,
        clippy::too_many_lines,
        clippy::result_large_err
    )]
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
        boot: &DeviceBoot,
        _props2: bool,
    ) -> std::result::Result<Context, (Fault, Option<Undone>)> {
        macro_rules! bail {
            ($fault:expr) => {
                return Err(($fault, Some((instance, messenger))))
            };
        }
        let physicals = match unsafe { instance.enumerate_physical_devices() } {
            Ok(p) => p,
            Err(e) => bail!(Fault::NoDevice {
                detail: format!("vkEnumeratePhysicalDevices: {e}"),
            }),
        };

        let mut candidates: Vec<(u32, usize, vk::PhysicalDevice, u32, String)> = Vec::new();
        for (ordinal, &physical) in physicals.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(physical) };
            let families =
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
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
        }
        let device_exts =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let memory_budget = has_extension(&device_exts, ash::ext::memory_budget::NAME);
        let coopmat_ext = has_extension(&device_exts, ash::khr::cooperative_matrix::NAME);

        let coopmat_shape = coopmat_ext && {
            let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
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

        let cores = if has_extension(&device_exts, ash::nv::shader_sm_builtins::NAME) {
            let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
            unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
            sm.shader_sm_count
        } else if has_extension(&device_exts, ash::amd::shader_core_properties::NAME) {
            let mut cu = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut cu);
            unsafe { instance.get_physical_device_properties2(physical, &mut p2) };
            cu.shader_engine_count
                * cu.shader_arrays_per_engine_count
                * cu.compute_units_per_shader_array
        } else {
            0
        }
        .max(16);

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
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
        }
        let core = unsafe { instance.get_physical_device_features(physical) };
        let enabled = Enabled {
            shader_int16: core.shader_int16 == vk::TRUE,
            shader_int64: core.shader_int64 == vk::TRUE,
            shader_float16: f12.shader_float16 == vk::TRUE,
            storage_16bit: f11.storage_buffer16_bit_access == vk::TRUE,
            storage_8bit: f12.storage_buffer8_bit_access == vk::TRUE,
            memory_model: f12.vulkan_memory_model == vk::TRUE
                && f12.vulkan_memory_model_device_scope == vk::TRUE,
            subgroup_size_control: api_13 && f13.subgroup_size_control == vk::TRUE,
            memory_budget,
            coopmat: coopmat_shape
                && fcm.cooperative_matrix == vk::TRUE
                && f12.shader_float16 == vk::TRUE
                && f12.vulkan_memory_model == vk::TRUE
                && subgroup.subgroup_size == 32,
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
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&exts)
            .push_next(&mut enable);
        let device = match unsafe { instance.create_device(physical, &create, None) } {
            Ok(d) => d,
            Err(e) => bail!(Fault::NoDevice {
                detail: format!("vkCreateDevice on {name}: {e}"),
            }),
        };
        let queue = unsafe { device.get_device_queue(family, 0) };
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
                unsafe { device.destroy_device(None) };
                bail!(Fault::Vulkan {
                    what: "vkCreateCommandPool",
                    code: e.as_raw(),
                });
            }
        };
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
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_local: u64 = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
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
            family,
            memory,
            limits: props.limits,
            transfer: Mutex::new(transfer),
            messenger,
            allocated: AtomicU64::new(0),
            name: name.clone(),
            subgroup_size: subgroup.subgroup_size.max(1),
            cores,
            device_local,
            vendor_id: props.vendor_id,
            api_version: props.api_version,
            features: enabled,
        });

        let dummy = Raw::new(&core, DUMMY_BYTES, Memory::Device).map_err(|fault| (fault, None))?;
        let fraction = if boot.gpu_mem_utilization.is_finite()
            && boot.gpu_mem_utilization > 0.0
            && boot.gpu_mem_utilization <= 1.0
        {
            boot.gpu_mem_utilization
        } else {
            0.9
        };
        let working_set = (device_local as f64 * fraction) as u64;
        let max_buffer = u64::from(core.limits.max_storage_buffer_range).max(1 << 30);
        Ok(Context {
            core,
            working_set,
            max_buffer,
            dummy,
            pipeline_cache: boot.pipeline_cache.clone(),
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

    #[must_use]
    pub fn working_set(&self) -> u64 {
        self.working_set
    }

    #[must_use]
    pub fn max_buffer(&self) -> u64 {
        self.max_buffer
    }

    #[must_use]
    pub fn cores(&self) -> u32 {
        self.core.cores
    }

    #[must_use]
    pub fn subgroup_size(&self) -> u32 {
        self.core.subgroup_size
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
        self.pipeline_cache.as_deref()
    }

    #[must_use]
    pub fn enabled(&self) -> Enabled {
        self.core.features
    }

    #[must_use]
    pub fn info(&self) -> kernels_vulkan::DeviceInfo {
        kernels_vulkan::DeviceInfo {
            vendor: match self.core.vendor_id {
                0x10DE => kernels_vulkan::tuning::Vendor::Nvidia,
                0x1002 => kernels_vulkan::tuning::Vendor::Amd,
                0x8086 => kernels_vulkan::tuning::Vendor::Intel,
                0x106B => kernels_vulkan::tuning::Vendor::Apple,
                _ => kernels_vulkan::tuning::Vendor::Other,
            },
            subgroup_size: self.core.subgroup_size,
            max_workgroup_invocations: self.core.limits.max_compute_work_group_invocations,
            max_shared_bytes: self.core.limits.max_compute_shared_memory_size,
            coopmat: self.core.features.coopmat,
            multiprocessors: self.core.cores,
        }
    }

    #[must_use]
    pub fn used(&self) -> u64 {
        if self.core.features.memory_budget {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                self.core
                    .instance
                    .get_physical_device_memory_properties2(self.core.physical, &mut props);
            }
            let heaps = self.core.memory.memory_heap_count as usize;
            let local = (0..heaps)
                .filter(|&i| {
                    self.core.memory.memory_heaps[i]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                })
                .map(|i| budget.heap_usage[i])
                .max();
            if let Some(used) = local {
                return used;
            }
        }
        self.core.allocated.load(Ordering::Relaxed)
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
        unsafe {
            let flags = if one_time {
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            } else {
                vk::CommandBufferUsageFlags::empty()
            };
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(flags))
        }
        .map_err(|e| core.fault("vkBeginCommandBuffer", e))?;
        let timing = if crate::encode::profiling() && core.limits.timestamp_period > 0.0 {
            let pool = unsafe {
                d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(TIMED_DISPATCHES * 2),
                    None,
                )
            }
            .map_err(|e| core.fault("vkCreateQueryPool", e))?;
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
                cmd,
                cmd_pool: pool,
                fence,
                pools: Mutex::new(Vec::new()),
                keep: Mutex::new(Vec::new()),
                submitted: std::sync::atomic::AtomicBool::new(false),
                timing: Mutex::new(timing),
                spare: Mutex::new(std::collections::HashMap::new()),
            }),
            open: true,
            hazards: std::cell::RefCell::new(Hazards::default()),
            retained: std::cell::RefCell::new(std::collections::HashSet::new()),
            dispatches: std::cell::Cell::new(0),
        })
    }
}

type Undone = (
    ash::Instance,
    Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
);

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
    pub(crate) cmd: vk::CommandBuffer,

    pub(crate) cmd_pool: vk::CommandPool,
    pub(crate) fence: vk::Fence,

    pub(crate) pools: Mutex<Vec<vk::DescriptorPool>>,

    pub(crate) keep: Mutex<Vec<Slab>>,
    submitted: std::sync::atomic::AtomicBool,

    pub(crate) timing: Mutex<Option<Timing>>,

    spare: Mutex<std::collections::HashMap<u64, Vec<vk::DescriptorSet>>>,
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
        unsafe {
            if self.submitted.load(Ordering::Acquire) {
                let _ = d.wait_for_fences(&[self.fence], true, FENCE_TIMEOUT_NS);
            }
            for pool in self
                .pools
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .drain(..)
            {
                d.destroy_descriptor_pool(pool, None);
            }
            if let Some(timing) = self
                .timing
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                d.destroy_query_pool(timing.pool, None);
            }
            d.free_command_buffers(self.cmd_pool, &[self.cmd]);
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
    pub(crate) dispatches: std::cell::Cell<u64>,

    pub(crate) hazards: std::cell::RefCell<Hazards>,

    retained: std::cell::RefCell<std::collections::HashSet<usize>>,
}

impl Frame {
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.inner.core
    }

    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        self.inner.cmd
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

    #[must_use]
    pub fn sync_counts(&self) -> (u64, u64) {
        let hazards = self.hazards.borrow();
        (hazards.barriers, hazards.seen)
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
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
            let mut spare = self
                .inner
                .spare
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(set) = spare.get_mut(&key).and_then(Vec::pop) {
                return Ok(set);
            }
        }
        let layouts = [layout; SETS_PER_BATCH];
        let mut pools = self
            .inner
            .pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut got: Option<Vec<vk::DescriptorSet>> = None;
        if let Some(&pool) = pools.last() {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            got = unsafe { d.allocate_descriptor_sets(&info) }.ok();
        }
        let mut sets = match got {
            Some(sets) => sets,
            None => {
                let sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(BUFFERS_PER_POOL)];
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
                .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        if len == 0 {
            return Ok(());
        }
        source.span(source_at, len)?;
        into.span(into_at, len)?;
        let (source, into) = (source.slab(), into.slab());
        let d = &self.inner.core.device;
        let cmd = self.inner.cmd;
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
            d.cmd_copy_buffer(
                cmd,
                source.buffer,
                into.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(source_at)
                    .dst_offset(into_at)
                    .size(len)],
            );
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
        let cmd = self.inner.cmd;
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

    pub fn flush(&self) -> Result<()> {
        let core = &self.inner.core;
        let d = &core.device;
        let cmd = self.inner.cmd;
        unsafe { d.end_command_buffer(cmd) }.map_err(|e| core.fault("vkEndCommandBuffer", e))?;
        {
            let bufs = [cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
        }
        unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
            .map_err(|e| core.fault("vkWaitForFences", e))?;
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        unsafe { d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()) }
            .map_err(|e| core.fault("vkResetCommandBuffer", e))?;
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
        unsafe { core.device.end_command_buffer(self.inner.cmd) }
            .map_err(|e| core.fault("vkEndCommandBuffer", e))
    }

    fn submit(&mut self) -> Result<()> {
        self.end()?;
        let core = &self.inner.core;
        let bufs = [self.inner.cmd];
        let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
        let queue = core
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    pub fn commit_async(
        mut self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
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
            unsafe {
                let _ = self.inner.core.device.end_command_buffer(self.inner.cmd);
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
    pub fn replay(
        &self,
        on_done: Option<Box<dyn Fn(Option<String>) + Send + 'static>>,
    ) -> Result<Pending> {
        let core = &self.inner.core;
        let d = &core.device;
        if self.inner.submitted.load(Ordering::Acquire) {
            unsafe { d.wait_for_fences(&[self.inner.fence], true, FENCE_TIMEOUT_NS) }
                .map_err(|e| core.fault("vkWaitForFences", e))?;
        }
        unsafe { d.reset_fences(&[self.inner.fence]) }
            .map_err(|e| core.fault("vkResetFences", e))?;
        let at = crate::encode::profiling().then(std::time::Instant::now);
        {
            let bufs = [self.inner.cmd];
            let submits = [vk::SubmitInfo::default().command_buffers(&bufs)];
            let queue = core
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe { d.queue_submit(*queue, &submits, self.inner.fence) }
                .map_err(|e| core.fault("vkQueueSubmit", e))?;
        }
        self.inner.submitted.store(true, Ordering::Release);
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
        unsafe { self.inner.core.device.get_fence_status(self.inner.fence) }.unwrap_or(true)
    }

    pub fn wait(&self) -> Result<()> {
        let core = &self.inner.core;
        let at = crate::encode::profiling().then(std::time::Instant::now);
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
        let mut slot = self
            .timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(timing) = slot.as_mut() else {
            return;
        };
        if timing.harvested || timing.names.is_empty() {
            return;
        }
        timing.harvested = true;
        let n = timing.names.len() as u32;
        let mut ticks = vec![0u64; (n * 2) as usize];
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
