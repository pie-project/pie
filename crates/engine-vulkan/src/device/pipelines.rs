use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use kernels_vulkan::Fire;

use crate::error::{Fault, Result};

use super::ctx::{Context, Core};
use super::spirv;

pub struct Pipeline {
    core: Arc<Core>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,

    pub bindings: u32,

    pub used: Vec<bool>,

    pub writable: Vec<bool>,
    pub push_bytes: u32,
    pub local: [u32; 3],
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let d = &self.core.device;
        unsafe {
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[derive(Default)]
pub struct Pipelines {
    built: RefCell<HashMap<&'static str, Arc<Pipeline>>>,
    compiles: Cell<u64>,
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines")
            .field("pipelines", &self.built.borrow().len())
            .field("compiles", &self.compiles.get())
            .finish()
    }
}

impl Pipelines {
    #[must_use]
    pub fn new() -> Pipelines {
        Pipelines::default()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.compiles.get()
    }

    pub fn warm(&self, device: &Context, fire: Fire) -> Result<()> {
        self.get(device, fire).map(|_| ())
    }

    pub fn get(&self, device: &Context, fire: Fire) -> Result<Arc<Pipeline>> {
        let entrypoint = fire.entrypoint;
        if let Some(pipeline) = self.built.borrow().get(entrypoint) {
            return Ok(Arc::clone(pipeline));
        }
        let code = kernels_vulkan::sources::module_for(entrypoint, device.lanes()).ok_or(
            Fault::Shader {
                file: fire.file,
                entrypoint,
                why: "the kernel table holds no compiled module by that name (built without \
                  `kernels-vulkan/native`, or the variant is not instantiated)"
                    .to_string(),
            },
        )?;
        let cache = device.pipeline_cache();
        let pipeline = Arc::new(build(device.core(), cache.handle(), fire, code)?);
        cache.built();
        self.compiles.set(self.compiles.get() + 1);
        self.built
            .borrow_mut()
            .insert(entrypoint, Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

fn build(core: &Arc<Core>, cache: vk::PipelineCache, fire: Fire, code: &[u8]) -> Result<Pipeline> {
    let shader = |why: spirv::Malformed| Fault::Shader {
        file: fire.file,
        entrypoint: fire.entrypoint,
        why: why.to_string(),
    };
    let words = spirv::words(code).map_err(shader)?;
    let declared = spirv::declared(&words).map_err(shader)?;
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..declared.bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let d = &core.device;
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
        let set_layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(declared.push_bytes)];
        let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if declared.push_bytes > 0 {
            info = info.push_constant_ranges(&ranges);
        }
        let layout = match d.create_pipeline_layout(&info, None) {
            Ok(l) => l,
            Err(e) => {
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreatePipelineLayout", e));
            }
        };
        let module = match d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        {
            Ok(m) => m,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreateShaderModule", e));
            }
        };
        let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(core.pinned.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        if core.pinned.is_some() {
            stage = stage.push_next(&mut required);
        }
        let made = d.create_compute_pipelines(
            cache,
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        );
        d.destroy_shader_module(module, None);
        match made {
            Ok(pipelines) => Ok(Pipeline {
                core: Arc::clone(core),
                pipeline: pipelines[0],
                layout,
                set_layout,
                bindings: declared.bindings,
                used: declared.used,
                writable: declared.writable,
                push_bytes: declared.push_bytes,
                local: declared.local,
            }),
            Err((_, e)) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                Err(core.fault("vkCreateComputePipelines", e))
            }
        }
    }
}
