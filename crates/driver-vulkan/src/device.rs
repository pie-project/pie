//! The device half: an instance, a queue, and pipelines that actually fire.
//!
//! Only compiled under the `native` feature, and only because it needs a GPU to
//! be PRESENT — every line of it builds anywhere, which is why the feature
//! gates the tests rather than the compilation.
//!
//! # What this is for, beyond wrapping `ash`
//!
//! `kernels-vulkan`'s docs list what a shell that runs its modules has to do,
//! and each item on that list is there because something got it wrong:
//!
//! * A layout must cover every binding the MODULE decorates, not every operand
//!   the row names. 292 entrypoints name no operands at all, and a layout built
//!   from such a row segfaults inside `vkCreateComputePipelines` rather than
//!   returning an error. [`Pipelines`] builds from
//!   [`crate::spirv::Declared::bindings`] for exactly this reason.
//! * A push range may not be wider than the block the module declares, and it
//!   may not be narrower than what the shader reads.
//! * `robustBufferAccess` must be on. `quant/qmm_t.comp` accumulates over its
//!   whole tile and guards only the store, so at a ragged shape it deliberately
//!   fetches outside the matrix and needs those reads defined as zero.
//! * A descriptor's offset must be a multiple of
//!   `minStorageBufferOffsetAlignment`, which has no Metal counterpart worth
//!   the name and which an arena allocator meets only if it asks. See
//!   [`Bound`].
//! * A binding must be given at least the bytes the shader's block reads.
//!   `robustBufferAccess` makes the tail read as ZERO rather than fault, so a
//!   short parameter block is a plausible number and not an error. See
//!   [`crate::spirv::Declared::block_bytes`].
//! * A descriptor's range should be the operand's extent and not
//!   `VK_WHOLE_SIZE`. In an arena the two differ by every tensor allocated
//!   after it, and `robustBufferAccess` -- already required above -- makes the
//!   narrow one CONFINE an overrun rather than merely describe it.
//! * The grid is a count of WORKGROUPS. See [`crate::geometry`].
//!
//! # The validation layer
//!
//! Enabled when the loader can find it, and an error ends the process. That is
//! not politeness: without it this driver answers a malformed request by
//! crashing or hanging instead of failing, so a green test without the layer is
//! not evidence that a dispatch was legal. It is a soft dependency because a
//! build machine will not have one.

use crate::geometry::{self, Dims, Module, Rule, Ungeometric};
use crate::spirv::{self, Declared};
use ash::vk;
use kernels_vulkan::Capability;
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::path::PathBuf;

/// Why there is no device to run on.
///
/// A distinct type from [`Ungeometric`] because the two mean opposite things to
/// a caller: this one is the environment, and a machine with no GPU is the
/// normal state of a build host rather than a defect.
#[derive(Clone, Debug)]
pub struct Unavailable(pub String);

impl core::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Unavailable {}

/// Why a dispatch could not be made.
///
/// Compared by value because [`crate::binding`] carries one inside its own
/// refusal, and a test that asserts WHICH refusal came back is the only way
/// an alignment failure stays distinguishable from a length one.
#[derive(Clone, Debug, PartialEq)]
pub enum Failed {
    /// The launch shape could not be worked out.
    Geometry(Ungeometric),
    /// The module could not be read.
    Module(spirv::Malformed),
    /// The caller bound a different number of buffers than the module declares.
    ///
    /// A short set is the one that matters: Vulkan reads a descriptor the
    /// shader uses and the set never filled, which is undefined rather than
    /// empty.
    Bindings {
        /// What the module decorates, one past its highest.
        module: u32,
        /// What the caller bound.
        bound: usize,
    },
    /// The push block is a different size than the pipeline's range.
    ///
    /// Both directions are refused. Too wide overruns the range; too narrow
    /// leaves the shader reading push memory that was never written, which the
    /// layer does not always catch and which reads as whatever the last
    /// dispatch left.
    Push {
        /// The range the pipeline was built with.
        range: u32,
        /// The bytes the caller offered.
        given: usize,
    },
    /// A sub-range starts at an offset the device will not address from.
    ///
    /// `minStorageBufferOffsetAlignment` is a hardware granularity, not a
    /// preference: a descriptor written at an offset it does not divide is
    /// invalid, and the layer says so. Without the layer it is undefined and
    /// on this device it happens to read the right bytes, which is the worst
    /// of the outcomes because it makes the defect a property of the machine
    /// it was tested on.
    Unaligned {
        /// The offset asked for.
        offset: u64,
        /// What the device requires it to be a multiple of.
        alignment: u64,
    },
    /// A sub-range runs past the end of the buffer holding it.
    ///
    /// The one refusal here that Vulkan itself would also catch -- but only
    /// with a layer, and only sometimes: `VK_WHOLE_SIZE` is legal, so a range
    /// that overruns is a number the driver may clamp, may honour, or may
    /// fault on, and the shader reads whichever happened.
    Overrun {
        /// Where it starts.
        offset: u64,
        /// How much it asked for.
        len: u64,
        /// What the buffer holds.
        size: u64,
    },
    /// A binding was given fewer bytes than the block the shader reads.
    ///
    /// The defect with no symptom at all, and the reason this refusal exists
    /// rather than a comment saying to be careful. `robustBufferAccess` is on
    /// -- the tiled GEMM needs it -- so a read past the bound range returns
    /// ZERO. A parameter block one word short does not fault and does not
    /// return garbage: the missing scalar is zero, which for a pitch or a flag
    /// is an entirely plausible value, and nothing downstream can tell it from
    /// one that was meant.
    ///
    /// `driver-metal` was found packing exactly this defect into two blocks,
    /// and there it was LOUDER: Metal's params region is written per dispatch,
    /// so the shader read the next dispatch's scalars instead of zeros.
    Short {
        /// Which descriptor.
        binding: u32,
        /// What the module's block needs.
        needs: u32,
        /// What the caller bound.
        given: u64,
    },
    /// A Vulkan call failed.
    Vulkan(String),
}

impl core::fmt::Display for Failed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Geometry(e) => write!(f, "no launch geometry: {e}"),
            Self::Module(e) => write!(f, "the module is malformed: {e}"),
            Self::Bindings { module, bound } => write!(
                f,
                "the module decorates {module} bindings but {bound} buffers were bound"
            ),
            Self::Push { range, given } => write!(
                f,
                "the pipeline's push range is {range} bytes and {given} were pushed"
            ),
            Self::Unaligned { offset, alignment } => write!(
                f,
                "this device addresses a storage buffer every {alignment} bytes \
                 and the range starts at {offset}"
            ),
            Self::Overrun { offset, len, size } => write!(
                f,
                "a range of {len} bytes at {offset} runs past the {size} the \
                 buffer holds"
            ),
            Self::Short {
                binding,
                needs,
                given,
            } => write!(
                f,
                "binding {binding} reads a {needs}-byte block and was given \
                 {given} bytes, whose tail reads as zero"
            ),
            Self::Vulkan(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for Failed {}

impl From<Ungeometric> for Failed {
    fn from(e: Ungeometric) -> Self {
        Self::Geometry(e)
    }
}

impl From<spirv::Malformed> for Failed {
    fn from(e: spirv::Malformed) -> Self {
        Self::Module(e)
    }
}

/// End the process on a validation error, printing what the layer said.
///
/// # Safety
///
/// Called by the Vulkan loader with a `p_callback_data` valid for the duration
/// of the call. Nothing here outlives it.
unsafe extern "system" fn fail_on_validation_error(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kinds: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let said = unsafe { data.as_ref() }
        .and_then(|d| (!d.p_message.is_null()).then(|| unsafe { CStr::from_ptr(d.p_message) }))
        .map_or_else(
            || "<no message>".to_string(),
            |m| m.to_string_lossy().into_owned(),
        );
    eprintln!(
        "\nthe Vulkan validation layer reported an ERROR:\n\n{said}\n\nThis \
         driver treats one as fatal, because without the layer the same \
         mistakes are a crash inside the driver or a wrong answer rather than \
         a message.\n"
    );
    // Abort and not panic: this is called across an `extern \"system\"`
    // boundary, where an unwind is undefined.
    std::process::abort();
}

/// An open Vulkan device with a compute queue.
///
/// Owns everything it creates and destroys it in reverse order on drop, so a
/// caller cannot leak a pipeline by dropping the device first.
pub struct Device {
    entry: ash::Entry,
    /// Kept alive for the instance's lifetime: dropping the messenger early
    /// silences exactly the calls that are about to be made.
    messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    name: String,
    max_push: u32,
    /// `minStorageBufferOffsetAlignment`.
    ///
    /// The limit with no Metal counterpart worth the name. `setBuffer:offset:`
    /// takes a byte offset and Apple silicon asks 4 of it; Vulkan will not
    /// address a storage buffer from an offset this does not divide, and a
    /// driver that binds a sub-range of an arena is doing nothing else.
    min_storage_offset: u64,
    /// Is host memory and device memory the same memory?
    ///
    /// Read from `deviceType`, not from the heaps. A discrete card exposes a
    /// small heap that is both DEVICE_LOCAL and HOST_VISIBLE -- the resizable
    /// BAR -- and a driver that concluded "unified" from finding one would be
    /// wrong about every allocation that did not fit in it.
    unified: bool,
    validated: bool,
    /// The tiers this device can actually load, best first.
    ///
    /// Derived from what was ENABLED rather than from what was reported: a
    /// feature the device has and the driver did not turn on is a feature a
    /// module may not declare a capability for.
    tiers: Vec<Capability>,
}

impl Device {
    /// Open the first device with a compute queue.
    ///
    /// # Errors
    ///
    /// [`Unavailable`] when there is no loader, no device, or no compute queue.
    /// None of those is a defect — they are what a machine without a GPU looks
    /// like — so this returns rather than panics.
    pub fn open() -> Result<Self, Unavailable> {
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| Unavailable(format!("no loader: {e}")))?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"driver-vulkan")
            .api_version(vk::API_VERSION_1_3);

        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let validation = c"VK_LAYER_KHRONOS_validation";
        let validated = layers
            .iter()
            .any(|l| l.layer_name_as_c_str().is_ok_and(|s| s == validation));
        let enabled_layers: Vec<*const c_char> = if validated {
            vec![validation.as_ptr()]
        } else {
            Vec::new()
        };
        let enabled_exts: Vec<*const c_char> = if validated {
            vec![ash::ext::debug_utils::NAME.as_ptr()]
        } else {
            Vec::new()
        };

        // Beyond the default checks, which only read what the API was ASKED to
        // do. Synchronization validation tracks real hazards between
        // dispatches -- a missing barrier gives the right answer on this device
        // most of the time, so no comparison of numbers will ever find one.
        let wanted = [
            vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
        ];
        let mut features = vk::ValidationFeaturesEXT::default();
        if validated {
            features = features.enabled_validation_features(&wanted);
        }

        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_exts);
        if validated {
            info = info.push_next(&mut features);
        }
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|e| Unavailable(format!("no instance: {e}")))?;

        let messenger = validated.then(|| {
            let debug = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let create = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                )
                .pfn_user_callback(Some(fail_on_validation_error));
            unsafe { debug.create_debug_utils_messenger(&create, None) }
                .ok()
                .map(|m| (debug, m))
        });
        let messenger = messenger.flatten();

        // From here on a failure has to destroy the instance, so the body is a
        // closure and the cleanup happens once.
        match Self::finish(entry, instance, messenger, validated) {
            Ok(d) => Ok(d),
            Err((e, instance, messenger)) => {
                unsafe {
                    if let Some((debug, m)) = messenger {
                        debug.destroy_debug_utils_messenger(m, None);
                    }
                    instance.destroy_instance(None);
                }
                Err(e)
            }
        }
    }

    /// Pick a device and create it, handing the instance back if it cannot.
    #[allow(clippy::type_complexity)]
    // The `Err` is large because it CARRIES the instance and the messenger.
    // They were created by the caller, this function is the only thing that
    // could have consumed them, and on failure the caller is the only thing
    // left that can destroy them. Boxing to shrink the variant would put an
    // allocation between the loader and its own teardown; making the error
    // small by dropping them would leak an instance on every machine without a
    // compute queue.
    #[allow(clippy::result_large_err)]
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
        validated: bool,
    ) -> Result<
        Self,
        (
            Unavailable,
            ash::Instance,
            Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
        ),
    > {
        macro_rules! bail {
            ($($t:tt)*) => {
                return Err((Unavailable(format!($($t)*)), instance, messenger))
            };
        }

        let devices = match unsafe { instance.enumerate_physical_devices() } {
            Ok(d) => d,
            Err(e) => bail!("cannot enumerate devices: {e}"),
        };
        let Some(&physical) = devices.first() else {
            bail!("the loader found no physical device")
        };

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = props
            .device_name_as_c_str()
            .map_or_else(|_| "<unnamed>".to_string(), |s| s.to_string_lossy().into());

        let family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
        let Some(family) = family else {
            bail!("{name} has no compute queue")
        };
        let family = family as u32;

        // Ask what the device has, then enable the subset the shader tree
        // needs. Naming them one by one rather than handing back everything
        // reported is what makes this list documentation: these are the
        // non-core things every module here assumes.
        // Whether the matrix unit is even offered. Asked of the device rather
        // than assumed, and asked BEFORE the feature query, because pushing
        // `PhysicalDeviceCooperativeMatrixFeaturesKHR` into a chain on a
        // device without the extension is not a question it can answer.
        let extensions = match unsafe { instance.enumerate_device_extension_properties(physical) } {
            Ok(e) => e,
            Err(e) => bail!("cannot enumerate device extensions: {e}"),
        };
        let has_coopmat = extensions.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|s| s == ash::khr::cooperative_matrix::NAME)
        });

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        {
            // Its own scope, and fresh structs below. `push_next` leaves each
            // struct's `p_next` pointing at the previous one, so feeding the
            // same structs into a second chain closes a cycle and the loader
            // walks it until it overflows.
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12);
            if has_coopmat {
                query = query.push_next(&mut fcm);
            }
            unsafe { instance.get_physical_device_features2(physical, &mut query) };
        }
        let core = unsafe { instance.get_physical_device_features(physical) };

        if core.robust_buffer_access != vk::TRUE {
            // Refusing rather than continuing. `quant/qmm_t.comp` fetches
            // outside its matrix on purpose and needs those reads defined as
            // zero; without this the spec allows neighbouring memory, which is
            // a wrong answer and not a crash.
            bail!("{name} has no robustBufferAccess, which the tiled GEMM depends on");
        }

        let mut e11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(f11.storage_buffer16_bit_access == vk::TRUE)
            .uniform_and_storage_buffer16_bit_access(
                f11.uniform_and_storage_buffer16_bit_access == vk::TRUE,
            );
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(f12.shader_float16 == vk::TRUE)
            .shader_int8(f12.shader_int8 == vk::TRUE)
            .storage_buffer8_bit_access(f12.storage_buffer8_bit_access == vk::TRUE)
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut ecm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(fcm.cooperative_matrix == vk::TRUE);
        let mut enable = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(core.shader_int16 == vk::TRUE)
                    .robust_buffer_access(true),
            )
            .push_next(&mut e11)
            .push_next(&mut e12);

        // Every name `Capability::requires()` lists, checked together rather
        // than one at a time. The tier's A/B operands are `float16_t`, so a
        // matrix unit with no `shaderFloat16` behind it is not enough; and
        // `GL_KHR_cooperative_matrix` pulls in `GL_KHR_memory_scope_semantics`,
        // so all 146 modules in the tier declare `OpCapability
        // VulkanMemoryModel`, which a module may not do unless the feature is
        // on (`VUID-VkShaderModuleCreateInfo-pCode-08740`).
        //
        // `vulkanMemoryModelDeviceScope` is the subtle one and it is not
        // optional: turning the memory model on changes the rules for EVERY
        // module the device loads, and `moe/route.comp` -- a BASELINE module
        // that has nothing to do with matrices -- counts token histograms with
        // a device-scoped `atomicAdd`. Enabling the model without the scope
        // breaks it.
        let coopmat = has_coopmat
            && fcm.cooperative_matrix == vk::TRUE
            && f12.shader_float16 == vk::TRUE
            && f12.vulkan_memory_model == vk::TRUE
            && f12.vulkan_memory_model_device_scope == vk::TRUE;

        let mut tiers = vec![Capability::Baseline];
        if f12.shader_float16 == vk::TRUE {
            tiers.push(Capability::Fp16);
        }
        if coopmat {
            tiers.push(Capability::Coopmat);
            enable = enable.push_next(&mut ecm);
        }
        tiers.sort_unstable();
        tiers.reverse();

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // The EXTENSION as well as the feature. A feature struct pushed into
        // the create chain for an extension that was not enabled is ignored,
        // silently, and the tier would then load a module declaring a
        // capability that is not there.
        let device_exts: Vec<*const c_char> = if coopmat {
            vec![ash::khr::cooperative_matrix::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&device_exts)
            .push_next(&mut enable);
        let device = match unsafe { instance.create_device(physical, &create, None) } {
            Ok(d) => d,
            Err(e) => bail!("cannot create a device on {name}: {e}"),
        };
        let queue = unsafe { device.get_device_queue(family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                unsafe { device.destroy_device(None) };
                bail!("cannot create a command pool on {name}: {e}")
            }
        };

        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        Ok(Self {
            entry,
            messenger,
            instance,
            device,
            queue,
            pool,
            memory,
            name,
            max_push: props.limits.max_push_constants_size,
            min_storage_offset: props.limits.min_storage_buffer_offset_alignment,
            unified: matches!(
                props.device_type,
                vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
            ),
            validated,
            tiers,
        })
    }

    /// What the device calls itself.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Is a validation layer watching?
    ///
    /// Worth asking rather than assuming: a test that means to prove a dispatch
    /// is LEGAL proves nothing without one, and should say so instead of
    /// passing.
    #[must_use]
    pub fn validated(&self) -> bool {
        self.validated
    }

    /// The tiers this device can load, best first.
    ///
    /// What was ENABLED, not what was reported. The distinction matters
    /// because a module may not declare a capability whose feature is off, so
    /// a device that has `shaderFloat16` and a driver that did not turn it on
    /// is a device that cannot load the fp16 tier -- and answering from the
    /// report would say otherwise.
    ///
    /// [`Capability::Baseline`] is always first-or-last and always present:
    /// every entrypoint has a baseline module, so a device offering nothing
    /// optional still resolves all 480.
    #[must_use]
    pub fn tiers(&self) -> &[Capability] {
        &self.tiers
    }

    /// The best module for `entrypoint` this device can load, and its tier.
    ///
    /// Walks the tiers best-first and takes the first one `dir` actually has.
    /// Both halves are required: a device may support a tier this build did
    /// not compile, and a build may have a tier this device cannot load. The
    /// answer is `None` only when even the baseline module is missing, which
    /// means the directory is not one a `native` build wrote.
    #[must_use]
    pub fn module_for(
        &self,
        dir: &std::path::Path,
        entrypoint: &str,
    ) -> Option<(PathBuf, Capability)> {
        self.tiers.iter().find_map(|&tier| {
            let path = dir.join(tier.module(entrypoint));
            path.exists().then_some((path, tier))
        })
    }

    /// `maxPushConstantsSize`.
    #[must_use]
    pub fn max_push(&self) -> u32 {
        self.max_push
    }

    /// `minStorageBufferOffsetAlignment`: the granularity a sub-range may
    /// start at.
    ///
    /// Always a power of two and at least 1, per the specification, so
    /// [`Bound::at`] may mask rather than divide.
    #[must_use]
    pub fn min_storage_offset(&self) -> u64 {
        self.min_storage_offset
    }

    /// Is host memory and device memory the same memory?
    ///
    /// True on an integrated GPU and on a software implementation, false on a
    /// discrete card. See the field for why the heaps do not answer this.
    #[must_use]
    pub fn unified(&self) -> bool {
        self.unified
    }

    /// Does the device expose memory the host cannot see?
    ///
    /// The second signal for [`Self::unified`], and independent of it: that one
    /// reads `deviceType`, this one reads the heaps. A part where every
    /// device-local type is also host-visible has one pool of memory; a part
    /// with a device-local type the host cannot map has two. They must agree,
    /// and `tests/device.rs` is where that is held.
    #[must_use]
    pub fn device_only_memory(&self) -> bool {
        self.memory.memory_types[..self.memory.memory_type_count as usize]
            .iter()
            .any(|t| {
                t.property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !t
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
    }

    /// A host-visible storage buffer holding `bytes`.
    ///
    /// Host-visible and coherent throughout. This is a correctness shell: being
    /// able to read a result back without a staging copy is worth more here
    /// than the bandwidth a device-local heap would add.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the allocation fails or the device exposes no
    /// host-visible memory type.
    pub fn buffer(&self, bytes: &[u8]) -> Result<Buffer, Failed> {
        // At least four bytes: a zero-sized buffer cannot be created, and an
        // operand a variant never reads still needs a descriptor to point at.
        let size = bytes.len().max(4) as u64;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { self.device.create_buffer(&info, None) }
            .map_err(|e| Failed::Vulkan(format!("create buffer: {e}")))?;
        let need = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = (0..self.memory.memory_type_count).find(|i| {
            need.memory_type_bits & (1 << i) != 0
                && self.memory.memory_types[*i as usize]
                    .property_flags
                    .contains(want)
        });
        let Some(index) = index else {
            unsafe { self.device.destroy_buffer(handle, None) };
            return Err(Failed::Vulkan("no host-visible memory type".into()));
        };

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(need.size)
            .memory_type_index(index);
        let memory = match unsafe { self.device.allocate_memory(&alloc, None) } {
            Ok(m) => m,
            Err(e) => {
                unsafe { self.device.destroy_buffer(handle, None) };
                return Err(Failed::Vulkan(format!("allocate: {e}")));
            }
        };
        if let Err(e) = unsafe { self.device.bind_buffer_memory(handle, memory, 0) } {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(handle, None);
            }
            return Err(Failed::Vulkan(format!("bind: {e}")));
        }

        let buffer = Buffer {
            handle,
            memory,
            size,
            mapped: need.size,
        };
        if !bytes.is_empty() {
            self.write(&buffer, bytes)?;
        }
        Ok(buffer)
    }

    /// Overwrite a buffer's first `bytes.len()` bytes.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the mapping fails, or if `bytes` is longer than
    /// the buffer — which is refused rather than truncated, since a short write
    /// leaves the tail holding the previous fire's numbers and every kernel
    /// here reads its whole operand.
    pub fn write(&self, buffer: &Buffer, bytes: &[u8]) -> Result<(), Failed> {
        if bytes.len() as u64 > buffer.size {
            return Err(Failed::Vulkan(format!(
                "{} bytes into a {}-byte buffer",
                bytes.len(),
                buffer.size
            )));
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            self.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    /// Copy one range of a buffer over another range of the SAME buffer.
    ///
    /// # Why one buffer and not two
    ///
    /// The cache is one buffer per layer holding every page, so moving a
    /// conversation's history is a move within a buffer and never between two.
    /// A general two-buffer copy would be a second thing to test for the one
    /// caller that does not exist.
    ///
    /// # Why on the host
    ///
    /// Every buffer this driver allocates is host-visible and coherent -- see
    /// [`Device::buffer`] -- so a `memmove` through a mapping is a copy with
    /// no command buffer, no barrier and nothing in flight when it returns.
    /// `driver-metal`'s `copy_kv` says the same thing for the same reason: a
    /// completion the caller waits on would be waiting for nothing. A device
    /// with a device-local cache would want `vkCmdCopyBuffer` instead, which
    /// is why [`crate::facts`] reports whether the memory is unified.
    ///
    /// Overlapping ranges are allowed and move correctly: this is a `memmove`,
    /// not a `memcpy`. Stated rather than left to the reader because a page
    /// compaction moves pages DOWN, and every such move within one page-size
    /// stride overlaps.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if either range leaves the buffer, or if the
    /// mapping fails. A range that left the buffer would otherwise be a write
    /// past an allocation, which this card does not report.
    pub fn copy_within(
        &self,
        buffer: &Buffer,
        from: u64,
        to: u64,
        bytes: u64,
    ) -> Result<(), Failed> {
        let ends = |at: u64| at.checked_add(bytes).is_some_and(|e| e <= buffer.size);
        if !ends(from) || !ends(to) {
            return Err(Failed::Vulkan(format!(
                "{bytes} bytes from {from} to {to} in a {}-byte buffer",
                buffer.size
            )));
        }
        if bytes == 0 || from == to {
            return Ok(());
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            // `copy`, not `copy_nonoverlapping`: see above.
            std::ptr::copy(ptr.add(from as usize), ptr.add(to as usize), bytes as usize);
            self.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    /// Read a buffer's contents back.
    ///
    /// # Errors
    ///
    /// [`Failed::Vulkan`] if the mapping fails.
    pub fn read(&self, buffer: &Buffer) -> Result<Vec<u8>, Failed> {
        let mut out = vec![0u8; buffer.size as usize];
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, buffer.mapped, vk::MemoryMapFlags::empty())
                .map_err(|e| Failed::Vulkan(format!("map: {e}")))?
                .cast::<u8>();
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(buffer.memory);
        }
        Ok(out)
    }

    /// Destroy a buffer.
    ///
    /// Explicit rather than a `Drop` on [`Buffer`], because freeing needs the
    /// device and a handle that carried one would make every buffer as large as
    /// a reference and impossible to store beside the device that owns it.
    pub fn free(&self, buffer: Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.handle, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    /// Run one dispatch to completion and wait for it.
    ///
    /// Synchronous on purpose. This is the shell a correctness test drives, and
    /// a pipelined submission would make a wrong answer a race rather than a
    /// wrong answer.
    ///
    /// # Errors
    ///
    /// [`Failed::Bindings`] or [`Failed::Push`] when the call does not match
    /// what the module declares, [`Failed::Geometry`] when the rule cannot
    /// answer, and [`Failed::Vulkan`] for a failed call.
    pub fn run(
        &self,
        pipeline: &Pipeline,
        buffers: &[Bound<'_>],
        push: &[u8],
        groups: [u32; 3],
    ) -> Result<(), Failed> {
        // One per slot in the layout, less the module's HOLES.
        //
        // Two different things make a layout wider than the bindings a module
        // decorates, and they pull opposite ways:
        //
        // * a hole, where `glslc` dropped a binding in the MIDDLE of the set.
        //   `affine_qmv_routed` has seven slots and one hole, and a real
        //   lowering states exactly six operands for it. Demanding seven
        //   would mean demanding a buffer for a binding no shader reads and
        //   the plan does not name -- the caller would have to invent one.
        //
        // * a caller whose row lists MORE buffers than the module decorates,
        //   which happens for eleven entrypoints and is legal.
        //   `layer_scalar_mul_bfloat16` lists four against a module of three,
        //   and `Pipelines::get` widens the layout to four so the call is not
        //   refused for being right.
        //
        // Subtracting holes from the layout answers both: four for the
        // second, six for the first. Counting only the decorated bindings
        // answered the first and broke the second, which is how the two were
        // found to be different questions.
        let real = pipeline.bindings as usize - pipeline.declared.holes();
        if buffers.len() != real {
            return Err(Failed::Bindings {
                module: real as u32,
                bound: buffers.len(),
            });
        }
        // Both directions. A short push leaves the shader reading bytes nothing
        // wrote, which is the previous dispatch's block and reads as a
        // plausible number.
        if push.len() != pipeline.push as usize {
            return Err(Failed::Push {
                range: pipeline.push,
                given: push.len(),
            });
        }
        if groups.contains(&0) {
            // Legal Vulkan, and always a defect: it runs nothing, returns
            // success, and leaves the output holding whatever it was born with.
            return Err(Failed::Vulkan(format!(
                "a dispatch of {groups:?} workgroups would run nothing and report success"
            )));
        }
        // Only the bindings whose block has a fixed size, which is the 39
        // PARAMETER blocks. A tensor binding ends in a runtime array and its
        // extent is the call's to decide, so there is nothing here to check it
        // against and nothing is claimed.
        // Zipped against the decorated bindings rather than counted from zero:
        // `block_bytes` is indexed by BINDING NUMBER, and past a hole the
        // caller's nth buffer is not binding n.
        for (binding, bound) in slots(pipeline).zip(buffers) {
            let Some(Some(needs)) = pipeline.declared.block_bytes.get(binding) else {
                continue;
            };
            if bound.len < u64::from(*needs) {
                return Err(Failed::Short {
                    binding: binding as u32,
                    needs: *needs,
                    given: bound.len,
                });
            }
        }

        unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(pipeline.bindings.max(1))];
            let pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| Failed::Vulkan(format!("descriptor pool: {e}")))?;

            let answer = self.record(pipeline, buffers, push, groups, pool);
            self.device.destroy_descriptor_pool(pool, None);
            answer
        }
    }

    /// The body of [`Self::run`], with the descriptor pool already made.
    unsafe fn record(
        &self,
        pipeline: &Pipeline,
        buffers: &[Bound<'_>],
        push: &[u8],
        groups: [u32; 3],
        pool: vk::DescriptorPool,
    ) -> Result<(), Failed> {
        let device = &self.device;
        let layouts = [pipeline.set_layout];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| Failed::Vulkan(format!("descriptor set: {e}")))?;
        let set = sets[0];

        let infos: Vec<_> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer.handle)
                    .offset(b.offset)
                    // The range, never `WHOLE_SIZE`. `WHOLE_SIZE` means "to the
                    // end of the buffer", so a sub-range written that way binds
                    // its own start and everything after it, and a shader that
                    // runs one row too far reads the NEXT tensor instead of
                    // faulting. The extent is the half of an operand that makes
                    // the overrun visible, and discarding it here would discard
                    // it at the only point where the device could act on it.
                    .range(b.len)
            })
            .collect();
        // Only the bindings the module actually decorates.
        //
        // 165 of this tree's 665 modules leave a hole -- 358 of them in all --
        // because `glslc` drops the declaration of a buffer a variant never
        // reads, and `kv_append_paged` holes 10 and 11 on purpose to keep
        // Metal's ring-ABI slots. A hole is free on Metal, where an argument
        // index nothing is set at is one the shader does not read; the
        // question here was whether Vulkan agrees, since the SET still needs
        // a slot at every number up to the highest.
        //
        // It does, and the specification says so in the VUID this would
        // otherwise trip: descriptors "must be valid IF THEY ARE ACCESSED".
        // Measured under GPU-assisted validation rather than assumed --
        // dispatching with both holes of a 7-binding module unwritten
        // succeeds and the layer stays silent, while leaving a decorated one
        // unwritten reports VUID-vkCmdDispatch-None-08114 by name.
        //
        // Skipping them is not merely allowed, it is the only thing this
        // driver can do: a hole has no operand in the plan, so there is no
        // buffer to put there and inventing one would bind an unrelated
        // tensor to a slot on the theory that nothing reads it.
        let writes: Vec<_> = slots(pipeline)
            .zip(&infos)
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        if !writes.is_empty() {
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmds = unsafe { device.allocate_command_buffers(&buffers_info) }
            .map_err(|e| Failed::Vulkan(format!("command buffer: {e}")))?;
        let cmd = cmds[0];

        let result = (|| -> Result<(), Failed> {
            unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| Failed::Vulkan(format!("begin: {e}")))?;
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[set],
                    &[],
                );
                if !push.is_empty() {
                    device.cmd_push_constants(
                        cmd,
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        push,
                    );
                }
                device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| Failed::Vulkan(format!("end: {e}")))?;

                let fence = device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| Failed::Vulkan(format!("fence: {e}")))?;
                let cmd_bufs = [cmd];
                let submits = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
                let submitted = device
                    .queue_submit(self.queue, &submits, fence)
                    .map_err(|e| Failed::Vulkan(format!("submit: {e}")));
                // A generous timeout, not an infinite one: a wait with no
                // deadline on a hung device is a test run that never returns
                // and reports nothing.
                let waited = submitted.and_then(|()| {
                    device
                        .wait_for_fences(&[fence], true, 10_000_000_000)
                        .map_err(|e| Failed::Vulkan(format!("wait: {e}")))
                });
                device.destroy_fence(fence, None);
                waited
            }
        })();

        unsafe { device.free_command_buffers(self.pool, &[cmd]) };
        result
    }

    /// Record a run of dispatches into one command buffer and submit once.
    ///
    /// [`Self::run`] is one dispatch, one command buffer, one submit and one
    /// fence wait, which is right for a test and wrong for a fire: a real
    /// plan states thousands of rectangles -- six texts here state 6272 --
    /// and one round trip to the queue per rectangle is most
    /// of the time a small model spends.
    ///
    /// Between every pair a full compute-to-compute memory barrier. Vulkan
    /// gives NO ordering between dispatches in one command buffer -- they may
    /// overlap, and a plan's launches are chained through the arena, so the
    /// second one reading the first one's output is the normal case rather
    /// than the exception. This is the single thing that makes a recorded run
    /// mean what the plan says, and its absence is silent: the layer does not
    /// complain, every call returns success, and the numbers are whatever the
    /// scheduler happened to produce.
    ///
    /// Each dispatch is checked exactly as [`Self::run`] checks one, and the
    /// whole run is refused if any of them is -- nothing is submitted, so a
    /// caller never has to reason about a partially executed plan.
    ///
    /// # Errors
    ///
    /// [`Failed`], with the index of the dispatch that produced it.
    pub fn run_all(&self, run: &[Recorded<'_, '_>]) -> Result<(), (usize, Failed)> {
        for (at, one) in run.iter().enumerate() {
            self.check(one).map_err(|e| (at, e))?;
        }
        if run.is_empty() {
            return Ok(());
        }
        unsafe {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(run.iter().map(|r| r.pipeline.bindings).sum::<u32>().max(1))];
            let pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(run.len() as u32)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| (0, Failed::Vulkan(format!("descriptor pool: {e}"))))?;
            let answer = self.record_all(run, pool);
            self.device.destroy_descriptor_pool(pool, None);
            answer
        }
    }

    /// Everything [`Self::run`] refuses a dispatch for, without recording it.
    fn check(&self, one: &Recorded<'_, '_>) -> Result<(), Failed> {
        let pipeline = one.pipeline;
        let real = pipeline.bindings as usize - pipeline.declared.holes();
        if one.buffers.len() != real {
            return Err(Failed::Bindings {
                module: real as u32,
                bound: one.buffers.len(),
            });
        }
        if one.push.len() != pipeline.push as usize {
            return Err(Failed::Push {
                range: pipeline.push,
                given: one.push.len(),
            });
        }
        if one.groups.contains(&0) {
            return Err(Failed::Vulkan(format!(
                "a dispatch of {:?} workgroups would run nothing and report success",
                one.groups
            )));
        }
        for (binding, bound) in slots(pipeline).zip(one.buffers) {
            let Some(Some(needs)) = pipeline.declared.block_bytes.get(binding) else {
                continue;
            };
            if bound.len < u64::from(*needs) {
                return Err(Failed::Short {
                    binding: binding as u32,
                    needs: *needs,
                    given: bound.len,
                });
            }
        }
        Ok(())
    }

    unsafe fn record_all(
        &self,
        run: &[Recorded<'_, '_>],
        pool: vk::DescriptorPool,
    ) -> Result<(), (usize, Failed)> {
        let device = &self.device;
        // Every set allocated and written BEFORE any recording. A descriptor
        // set is read when the command executes, not when it is bound, so a
        // set rewritten between two `cmd_bind_descriptor_sets` in one command
        // buffer would give both dispatches the second write. One set each is
        // the only arrangement that means what it looks like.
        let mut sets = Vec::with_capacity(run.len());
        for (at, one) in run.iter().enumerate() {
            let layouts = [one.pipeline.set_layout];
            let allocated = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|e| (at, Failed::Vulkan(format!("descriptor set: {e}"))))?;
            let set = allocated[0];
            let infos: Vec<_> = one
                .buffers
                .iter()
                .map(|b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.buffer.handle)
                        .offset(b.offset)
                        .range(b.len)
                })
                .collect();
            let writes: Vec<_> = slots(one.pipeline)
                .zip(&infos)
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            if !writes.is_empty() {
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sets.push(set);
        }

        let cmds = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|e| (0, Failed::Vulkan(format!("command buffer: {e}"))))?;
        let cmd = cmds[0];

        let result = (|| -> Result<(), Failed> {
            unsafe {
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| Failed::Vulkan(format!("begin: {e}")))?;
                for (at, one) in run.iter().enumerate() {
                    if at > 0 {
                        // One global barrier rather than a buffer barrier per
                        // operand. A plan chains through ONE arena, so every
                        // buffer barrier this could state would name the same
                        // buffer, and a driver coalesces them into the same
                        // stall. Correct and coarse; the finer version is a
                        // measurement this crate has not made.
                        let barrier = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barrier,
                            &[],
                            &[],
                        );
                    }
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        one.pipeline.pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        one.pipeline.layout,
                        0,
                        &[sets[at]],
                        &[],
                    );
                    if !one.push.is_empty() {
                        device.cmd_push_constants(
                            cmd,
                            one.pipeline.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            one.push,
                        );
                    }
                    device.cmd_dispatch(cmd, one.groups[0], one.groups[1], one.groups[2]);
                }
                device
                    .end_command_buffer(cmd)
                    .map_err(|e| Failed::Vulkan(format!("end: {e}")))?;

                let fence = device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(|e| Failed::Vulkan(format!("fence: {e}")))?;
                let cmd_bufs = [cmd];
                let submits = [vk::SubmitInfo::default().command_buffers(&cmd_bufs)];
                let submitted = device
                    .queue_submit(self.queue, &submits, fence)
                    .map_err(|e| Failed::Vulkan(format!("submit: {e}")));
                let waited = submitted.and_then(|()| {
                    device
                        .wait_for_fences(&[fence], true, 10_000_000_000)
                        .map_err(|e| Failed::Vulkan(format!("wait: {e}")))
                });
                device.destroy_fence(fence, None);
                waited
            }
        })();

        unsafe { device.free_command_buffers(self.pool, &[cmd]) };
        result.map_err(|e| (0, e))
    }
}

/// One dispatch in a recorded run.
///
/// The same four things [`Device::run`] takes, named rather than positional
/// because a run states many of them and a swapped pair of arguments in a
/// list of hundreds is not something a reader would catch.
#[derive(Clone, Copy)]
pub struct Recorded<'a, 'b> {
    /// The compiled module and its layout.
    pub pipeline: &'a Pipeline,
    /// One range per binding the module reads, less its holes.
    pub buffers: &'a [Bound<'b>],
    /// The push block, empty if the module has none.
    pub push: &'a [u8],
    /// Workgroups in each dimension, none of them zero.
    pub groups: [u32; 3],
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Everything in flight must finish before anything it touched is
            // destroyed. A device that is dropped while a submission is live is
            // undefined, and the layer reports it as a use-after-free with no
            // obvious connection to the test that caused it.
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if let Some((debug, m)) = self.messenger.take() {
                debug.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
        // `entry` owns the loaded loader and is dropped after, by field order.
        let _ = &self.entry;
    }
}

/// A storage buffer and the memory behind it.
///
/// Plain data with no device reference, so a caller can keep a table of these
/// beside the [`Device`] that made them. Freed with [`Device::free`].
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    /// What the caller asked for, which is what [`Device::read`] returns.
    size: u64,
    /// What was actually allocated, which is what may be mapped. The driver
    /// rounds up, and mapping past this is invalid while mapping only `size`
    /// is not always allowed for a coherent range.
    mapped: u64,
}

impl Buffer {
    /// Bytes the caller asked for.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// A buffer of a stated size that names no device allocation.
    ///
    /// Binding produces offsets and lengths AGAINST a buffer and never
    /// dereferences its handle, so every question in [`crate::binding`] can be
    /// asked without a GPU -- and asking them without one is why the arena
    /// arithmetic is tested at all, since the machines that change a lowering
    /// are not the machines that have a Vulkan device.
    ///
    /// Passing one of these to [`Device::run`] would bind a null handle, so it
    /// is `#[doc(hidden)]`: `tests/arena.rs` needs it to put a real plan
    /// through the real binder on a machine with no GPU, and nothing else has
    /// a reason to reach for it.
    #[doc(hidden)]
    #[must_use]
    pub fn placeholder(size: u64) -> Self {
        Self {
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
            mapped: size,
        }
    }
}

/// What one descriptor addresses: a buffer, and which part of it.
///
/// # Why this is not `&Buffer`
///
/// A driver does not allocate a buffer per tensor. It allocates an arena and
/// hands out offsets into it, because a fire's activations are hundreds of
/// values whose lifetimes nest and whose sizes are known together --
/// `driver-metal`'s binder resolves an operand to `Slice { address, bytes }`
/// for exactly this reason.
///
/// Metal can take that literally: `setBuffer:offset:` moves the base by a byte
/// count and the shader addresses from there. **Vulkan cannot.** A descriptor
/// carries an offset the device must be able to address from, and
/// `minStorageBufferOffsetAlignment` says how coarsely. So the arena model
/// needs a type that carries the offset and can be refused, rather than a
/// reference that cannot.
///
/// The extent travels with the address for the reason the Metal binder gives:
/// an arena reused across fires can be smaller than the new one needs, and an
/// operand whose length lives in a neighbouring field is a bound two call
/// sites have to agree about.
#[derive(Clone, Copy, Debug)]
pub struct Bound<'a> {
    buffer: &'a Buffer,
    offset: u64,
    len: u64,
}

/// Two bounds are the same range when they name the same MEMORY, not the
/// same `&Buffer`.
///
/// Written out rather than derived because the derived version would compare
/// the reference, and a caller holding two borrows of one buffer would find
/// its two identical ranges unequal. What a test asks of a dispatch is where
/// it points, and the answer is the handle, the offset and the length.
impl PartialEq for Bound<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.buffer.handle == other.buffer.handle
            && self.offset == other.offset
            && self.len == other.len
    }
}

impl Eq for Bound<'_> {}

impl<'a> Bound<'a> {
    /// The whole buffer.
    ///
    /// Offset zero, which every alignment divides, so this cannot be refused
    /// and does not need the device to say so.
    #[must_use]
    pub fn whole(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            len: buffer.size,
        }
    }

    /// `len` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// [`Failed::Unaligned`] when the device cannot address from there, and
    /// [`Failed::Overrun`] when the range leaves the buffer.
    ///
    /// A zero-length range is [`Failed::Overrun`] with `len` zero rather than a
    /// variant of its own: it is illegal Vulkan, and it is also always the
    /// same defect -- a width computed from a shape that came out empty --
    /// so the numbers that produced it are the useful part of the message.
    pub fn at(device: &Device, buffer: &'a Buffer, offset: u64, len: u64) -> Result<Self, Failed> {
        Self::within(buffer, offset, len, device.min_storage_offset)
    }

    /// `len` bytes at `offset`, against a stated alignment.
    ///
    /// The same rule as [`Bound::at`] with the device's one number passed in
    /// rather than read out. It exists because binding a plan is arithmetic
    /// over offsets and extents, and the machines that CHANGE a plan -- the
    /// ones running `model-compiler`'s tests -- do not have a Vulkan device to
    /// ask. Splitting the number out is what lets `crate::binding` be tested
    /// where the numbers are produced instead of only where they are used.
    ///
    /// It is also how a check can be made against the SPECIFICATION's 256
    /// rather than the local card's 16, which is the difference between
    /// knowing a plan binds here and knowing it binds anywhere.
    ///
    /// # Errors
    ///
    /// As [`Bound::at`].
    pub fn within(
        buffer: &'a Buffer,
        offset: u64,
        len: u64,
        alignment: u64,
    ) -> Result<Self, Failed> {
        // `max(1)` because the specification's guarantee is a promise from the
        // driver, and dividing by a zero one would panic where refusing is the
        // whole job. Every offset is a multiple of 1, so a device that reports
        // nothing constrains nothing.
        //
        // UNWITNESSED, and unwitnessably so from here: deleting it changes no
        // test because the alignment comes from the device's own limits and
        // this card reports 16. Reaching it needs a driver that reports zero,
        // which is the case it exists for and not one a test can produce. Kept
        // rather than removed -- unlike the dead clamp in `geometry.rs`, this
        // one guards a division and its absence is a panic, not a wrong
        // answer.
        let alignment = alignment.max(1);
        if !offset.is_multiple_of(alignment) {
            return Err(Failed::Unaligned { offset, alignment });
        }
        // Checked rather than added: an offset near `u64::MAX` would wrap to a
        // small sum and pass a bound it is nowhere near.
        let end = offset.checked_add(len);
        if len == 0 || end.is_none_or(|e| e > buffer.size) {
            return Err(Failed::Overrun {
                offset,
                len,
                size: buffer.size,
            });
        }
        Ok(Self {
            buffer,
            offset,
            len,
        })
    }

    /// Where the range starts in its buffer.
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Bytes the range covers.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Is the range empty? Never, by construction -- both constructors refuse
    /// it -- but clippy asks for it beside `len` and a caller may prefer to ask.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer the range lives in.
    #[must_use]
    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }
}

impl<'a> From<&'a Buffer> for Bound<'a> {
    fn from(buffer: &'a Buffer) -> Self {
        Self::whole(buffer)
    }
}

/// A compute pipeline and the layout it was built with.
pub struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    /// Descriptors the layout has: the module's own count or the caller's,
    /// whichever is larger. See [`Pipelines::get`] for why neither alone.
    bindings: u32,
    /// Bytes the push range covers.
    push: u32,
    /// What the module said about itself, kept so a caller can build the grid
    /// without reading the file twice.
    declared: Declared,
}

impl Pipeline {
    /// What the module declared.
    #[must_use]
    pub fn declared(&self) -> &Declared {
        &self.declared
    }

    /// Descriptors this pipeline's set layout has.
    #[must_use]
    pub fn bindings(&self) -> u32 {
        self.bindings
    }

    /// Bytes of push constants this pipeline expects.
    #[must_use]
    pub fn push(&self) -> u32 {
        self.push
    }
}

/// Compiled pipelines, one per entrypoint, built on first use.
///
/// Separate from [`Device`] rather than a field on it, so that the borrow of a
/// pipeline does not borrow the device — a caller needs both at once on every
/// dispatch.
pub struct Pipelines {
    /// Keyed by entrypoint AND tier, because they are different modules.
    ///
    /// A key of the name alone is the bug this is written to avoid: the same
    /// entrypoint has up to three modules, they are compiled from different
    /// bodies with different extensions, and the first one built would then
    /// answer for all of them. That is not a slow path or a wrong tier -- it
    /// is a pipeline whose SPIR-V does not match the layout the caller sized
    /// from the tier it thinks it selected.
    built: HashMap<(String, Capability), Pipeline>,
}

impl Default for Pipelines {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipelines {
    /// How many pipelines this cache holds.
    ///
    /// Not `len`, because a cache is not a collection the caller iterates and
    /// `is_empty` would be the next thing a lint asked for. This exists so
    /// that "the second fire built nothing new" is a number rather than a
    /// claim about how long something took.
    #[must_use]
    pub fn built(&self) -> usize {
        self.built.len()
    }

    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            built: HashMap::new(),
        }
    }

    /// The pipeline for `entrypoint`, building it from `code` if new.
    ///
    /// The layout is built from what the MODULE declares, not from the row.
    /// This is the crate's central hazard: 292 of the 480 entrypoints name no
    /// operands, and a layout of no descriptors under a module that decorates
    /// several is a segfault inside `vkCreateComputePipelines` rather than an
    /// error return — so there is nothing to check after the fact.
    ///
    /// `push` is the range in bytes. Passing the row's `push_size` is right for
    /// a row that states its scalars; for one that does not,
    /// [`Device::max_push`] is the only safe answer, since any block the module
    /// declares fits inside it.
    ///
    /// # Errors
    ///
    /// [`Failed::Module`] if `code` is not a module this tree built, and
    /// [`Failed::Vulkan`] if a Vulkan call fails.
    ///
    /// `descriptors` is the count the CALLER will bind, and it is a floor
    /// under what the module declares rather than an alternative to it. Both
    /// numbers are needed because they disagree, and measurement says which
    /// way: over the 188 stated entrypoints 177 agree and 11 have a module
    /// declaring exactly one binding FEWER than the row does, because glslc
    /// drops the `OpDecorate Binding` of a buffer the shader never reads.
    /// `layer_scalar_mul_bfloat16` is one -- the row lists four buffers and
    /// the compiled module decorates three.
    ///
    /// Taking the module's number alone would then refuse those eleven at
    /// `run`, a legitimate call rejected. Taking the caller's alone is the
    /// SIGSEGV: a module reading `binding = 11` under a layout that stops at
    /// 10 does not return an error from `vkCreateComputePipelines`, it takes
    /// the process down. A descriptor declared and never read costs nothing,
    /// so the maximum is the only answer that is safe in both directions.
    pub fn get(
        &mut self,
        device: &Device,
        entrypoint: &str,
        code: &[u8],
        push: u32,
        descriptors: u32,
        tier: Capability,
    ) -> Result<&Pipeline, Failed> {
        let key = (entrypoint.to_string(), tier);
        if !self.built.contains_key(&key) {
            let built = Self::build(device, code, push, descriptors)?;
            self.built.insert(key.clone(), built);
        }
        Ok(&self.built[&key])
    }

    /// A pipeline already built, without the borrow that building takes.
    ///
    /// [`Self::get`] needs `&mut self` because it may build, which means a
    /// caller cannot hold a reference to one pipeline while asking for the
    /// next. A caller recording a whole plan needs exactly that -- one
    /// reference per launch, all alive at once -- so it builds every distinct
    /// module first and then asks for them.
    #[must_use]
    pub fn peek(&self, entrypoint: &str, tier: Capability) -> Option<&Pipeline> {
        self.built.get(&(entrypoint.to_string(), tier))
    }

    /// Build one pipeline.
    fn build(
        device: &Device,
        code: &[u8],
        push: u32,
        descriptors: u32,
    ) -> Result<Pipeline, Failed> {
        let words = spirv::words(code)?;
        let declared = spirv::declared(&words)?;
        let count = declared.bindings.max(descriptors);

        let bindings: Vec<_> = (0..count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let d = &device.device;
        unsafe {
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| Failed::Vulkan(format!("set layout: {e}")))?;

            let set_layouts = [set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push)];
            let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            if push > 0 {
                info = info.push_constant_ranges(&ranges);
            }

            let layout = match d.create_pipeline_layout(&info, None) {
                Ok(l) => l,
                Err(e) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(Failed::Vulkan(format!("pipeline layout: {e}")));
                }
            };

            let module = match d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_pipeline_layout(layout, None);
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(Failed::Vulkan(format!("shader module: {e}")));
                }
            };

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let made = d.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );
            // The module is only needed while the pipeline is being made.
            d.destroy_shader_module(module, None);

            match made {
                Ok(pipelines) => Ok(Pipeline {
                    pipeline: pipelines[0],
                    layout,
                    set_layout,
                    bindings: count,
                    push,
                    declared,
                }),
                Err((_, e)) => {
                    d.destroy_pipeline_layout(layout, None);
                    d.destroy_descriptor_set_layout(set_layout, None);
                    Err(Failed::Vulkan(format!("compute pipeline: {e}")))
                }
            }
        }
    }

    /// Destroy everything built so far.
    ///
    /// Explicit and not a `Drop`, for the same reason [`Device::free`] is: the
    /// device is what destroys these, and a cache that borrowed it could not be
    /// held beside it.
    pub fn clear(&mut self, device: &Device) {
        let d = &device.device;
        unsafe {
            let _ = d.device_wait_idle();
            for (_, p) in self.built.drain() {
                d.destroy_pipeline(p.pipeline, None);
                d.destroy_pipeline_layout(p.layout, None);
                d.destroy_descriptor_set_layout(p.set_layout, None);
            }
        }
    }
}

/// The workgroup count for a fire, from the rule and the module it will run.
///
/// The one place [`geometry`] and this module meet, and the reason the loaded
/// module is what answers: the divisor and the GEMM tile are the module's, and
/// asking it is what keeps them from being assumed.
///
/// # Errors
///
/// [`Failed::Geometry`] when the rule cannot answer for these dimensions.
pub fn groups_for(
    entrypoint: &str,
    rule: Rule,
    dims: Dims,
    pipeline: &Pipeline,
) -> Result<[u32; 3], Failed> {
    Ok(geometry::groups(
        rule,
        dims,
        Module::loaded(entrypoint, &pipeline.declared),
    )?)
}

/// The binding numbers a caller's buffers go to, in order.
///
/// Every slot of the layout except the module's holes. A slot at or past
/// `declared.bindings` is not a hole -- it is a descriptor the caller asked
/// the layout for, and skipping it would shift every buffer after it.
fn slots(pipeline: &Pipeline) -> impl Iterator<Item = usize> + '_ {
    (0..pipeline.bindings as usize)
        .filter(|i| pipeline.declared.used.get(*i).copied().unwrap_or(true))
}
