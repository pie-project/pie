//! Shared by the Vulkan engine and the wgpu engine's Vulkan adapter.

use ash::vk;

/// The SMs, compute units or shader cores a vendor extension reports.
/// `properties2` is `vkGetPhysicalDeviceProperties2` on the device.
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
        cu.shader_engine_count
            * cu.shader_arrays_per_engine_count
            * cu.compute_units_per_shader_array
    } else if has(ash::arm::shader_core_builtins::NAME) {
        let mut arm = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
        properties2(&mut vk::PhysicalDeviceProperties2::default().push_next(&mut arm));
        arm.shader_core_count
    } else {
        0
    };
    (cores > 0).then_some(cores)
}
