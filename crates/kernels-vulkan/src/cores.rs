//! The shader cores a Vulkan device reports: read the same way by the
//! Vulkan engine and by the wgpu engine on its Vulkan adapter. The engine
//! makes the two driver calls; this crate holds no unsafe code.

use ash::vk;

/// The cores a device reports through a vendor extension: NVIDIA's SMs
/// (`VK_NV_shader_sm_builtins`), AMD's compute units
/// (`VK_AMD_shader_core_properties`), Arm's shader cores
/// (`VK_ARM_shader_core_builtins`). `None` where no extension says, or one
/// says zero.
///
/// `exts` is the device's extension list; `properties2` fills a
/// `VkPhysicalDeviceProperties2` chain for the device
/// (`vkGetPhysicalDeviceProperties2`).
#[must_use]
pub fn shader_cores(
    exts: &[vk::ExtensionProperties],
    properties2: &mut dyn FnMut(&mut vk::PhysicalDeviceProperties2<'_>),
) -> Option<u32> {
    let has = |name: &std::ffi::CStr| {
        exts.iter()
            .any(|e| e.extension_name_as_c_str().is_ok_and(|s| s == name))
    };
    let cores = if has(ash::nv::shader_sm_builtins::NAME) {
        let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
        properties2(&mut vk::PhysicalDeviceProperties2::default().push_next(&mut sm));
        sm.shader_sm_count
    } else if has(ash::amd::shader_core_properties::NAME) {
        let mut cu = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        properties2(&mut vk::PhysicalDeviceProperties2::default().push_next(&mut cu));
        cu.shader_engine_count * cu.shader_arrays_per_engine_count * cu.compute_units_per_shader_array
    } else if has(ash::arm::shader_core_builtins::NAME) {
        let mut arm = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
        properties2(&mut vk::PhysicalDeviceProperties2::default().push_next(&mut arm));
        arm.shader_core_count
    } else {
        0
    };
    (cores > 0).then_some(cores)
}
