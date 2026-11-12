//! The shared guest forms ([`engine_gpu::guest`]) bound to Vulkan: a
//! stage's WGSL lowers to SPIR-V.

pub use engine_gpu::guest::{ENTRY, Refused, step_source};

pub type Compiled = engine_gpu::guest::Compiled<crate::device::Api>;
pub type Lowered = engine_gpu::guest::Lowered<Vec<u32>>;
pub type Forms = engine_gpu::guest::Forms<crate::device::Api>;

impl engine_gpu::guest::Lower for crate::device::Api {
    type Code = Vec<u32>;

    fn lower(stage: usize, source: String) -> Result<Vec<u32>, Refused> {
        lower(stage, &source, &stage_layout())
    }
}

/// How a pipeline layout binds a buffer, which the shader's declaration of
/// that binding must match: a mismatch is invalid usage a driver may run
/// anyway, slowly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Storage,
    Uniform,
}

/// The layout every guest stage's pipelines bind (group 0): storage
/// buffers, the config a uniform.
#[must_use]
pub fn stage_layout() -> Vec<(u32, Slot)> {
    use engine_gpu::guest::session::{BINDINGS, CFG_BINDING};
    (0..BINDINGS)
        .map(|at| {
            let slot = if at == CFG_BINDING {
                Slot::Uniform
            } else {
                Slot::Storage
            };
            (at, slot)
        })
        .collect()
}

/// The descriptor type a layout binds `slot` as.
#[cfg(feature = "vulkan")]
#[must_use]
pub fn descriptor_type(slot: Slot) -> ash::vk::DescriptorType {
    match slot {
        Slot::Storage => ash::vk::DescriptorType::STORAGE_BUFFER,
        Slot::Uniform => ash::vk::DescriptorType::UNIFORM_BUFFER,
    }
}

/// Lowers `source` to SPIR-V for a pipeline whose layout binds `layout`
/// (group 0); refused where a buffer the shader declares is not bound as it
/// declares it.
#[cfg(feature = "vulkan")]
pub fn lower(stage: usize, source: &str, layout: &[(u32, Slot)]) -> Result<Vec<u32>, Refused> {
    let refuse = |why: String| Refused::Lowering { stage, why };
    let module =
        naga::front::wgsl::parse_str(source).map_err(|e| refuse(e.emit_to_string(source)))?;
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        let declared = match global.space {
            naga::AddressSpace::Storage { .. } => Slot::Storage,
            naga::AddressSpace::Uniform => Slot::Uniform,
            _ => continue,
        };
        let bound = (binding.group == 0)
            .then(|| layout.iter().find(|(at, _)| *at == binding.binding))
            .flatten()
            .map(|&(_, slot)| slot);
        if bound != Some(declared) {
            return Err(refuse(format!(
                "binding {}.{} is declared {declared:?} and the pipeline layout binds {bound:?}",
                binding.group, binding.binding
            )));
        }
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| refuse(e.emit_to_string(source)))?;
    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .map_err(|e| refuse(e.to_string()))
}

#[cfg(not(feature = "vulkan"))]
pub fn lower(stage: usize, _source: &str, _layout: &[(u32, Slot)]) -> Result<Vec<u32>, Refused> {
    Err(Refused::Lowering {
        stage,
        why: "this build carries no Vulkan device".to_string(),
    })
}

#[cfg(feature = "vulkan")]
pub mod run;
#[cfg(feature = "vulkan")]
pub mod session;
#[cfg(feature = "vulkan")]
pub mod widen;

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use engine_gpu::guest::widen::{STORAGE_SOURCE, UNIFORM_SOURCE};

    #[test]
    fn a_binding_declared_unlike_its_layout_refuses_to_lower() {
        let storage = [(0, Slot::Storage), (1, Slot::Storage), (2, Slot::Storage)];
        assert!(lower(0, STORAGE_SOURCE, &storage).is_ok());
        let refused = lower(0, UNIFORM_SOURCE, &storage).expect_err("binding 2 is a uniform");
        assert!(refused.to_string().contains("binding 0.2"), "{refused}");
    }
}
