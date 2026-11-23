#![allow(unsafe_code)]
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine_gpu::error::{Fault, Result};
use ash::vk;

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
    /// The scalars the kernel reads from the fire block.
    pub fire: Option<spirv::Fire>,
    /// The bindings laid update-after-bind, a pool segment's each: bit `i`
    /// binding `i`. A recording binding a segment there has it rebound as
    /// the segment is backed or given back
    /// ([`super::Context::rebind_segments`]); 0 on a device that lays none.
    pub segments: u64,
    pub local: [u32; 3],
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        let d = &self.core.device;
        // SAFETY: the last `Arc<Pipeline>` is gone, so no frame records with
        // it any more; the shell's context drains the queue before its
        // pipelines drop, so no pending command uses it either.
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
    /// The kernels built, in the order their pipelines were.
    order: RefCell<Vec<&'static str>>,
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines")
            .field("pipelines", &self.built.borrow().len())
            .field("compiles", &self.order.borrow().len())
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
        self.order.borrow().len() as u64
    }

    /// The most bytes of scalars any built kernel takes.
    #[must_use]
    pub fn most_scalars(&self) -> u32 {
        self.built
            .borrow()
            .values()
            .map(|pipeline| {
                pipeline
                    .fire
                    .as_ref()
                    .map_or(0, |fire| fire.bytes)
                    .max(pipeline.push_bytes)
            })
            .max()
            .unwrap_or(0)
    }

    /// The kernels whose pipelines were built after the first `from`, in
    /// the order built.
    #[must_use]
    pub fn built_since(&self, from: u64) -> Vec<&'static str> {
        let order = self.order.borrow();
        order[(from as usize).min(order.len())..].to_vec()
    }

    /// The pipeline of kernels-slang's module `key` (an instantiation at a
    /// tier, `kernels_slang::key`), from `file`; built on first use.
    pub fn get(
        &self,
        device: &Context,
        file: &'static str,
        key: &'static str,
    ) -> Result<Arc<Pipeline>> {
        if let Some(pipeline) = self.built.borrow().get(key) {
            return Ok(Arc::clone(pipeline));
        }
        let code = kernels_slang::spirv(key).ok_or(Fault::Shader {
            file,
            entrypoint: key,
            why: "kernels-slang holds no SPIR-V module by that key (built without \
                  `kernels-slang/spirv`, or the instantiation is limited to WGSL)"
                .to_string(),
        })?;
        let cache = device.pipeline_cache();
        let pipeline = Arc::new(build(device.core(), cache.handle(), file, key, code)?);
        cache.built();
        self.order.borrow_mut().push(key);
        self.built.borrow_mut().insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// Holds kernel `key` of `file` to the storage buffers `device` binds a
    /// shader stage: its layout declares every binding of its module one.
    pub fn hold(device: &Context, file: &'static str, key: &'static str) -> Result<()> {
        let shader = |why: String| Fault::Shader {
            file,
            entrypoint: key,
            why,
        };
        let code = kernels_slang::spirv(key).ok_or_else(|| {
            shader(
                "kernels-slang holds no SPIR-V module by that key (built without \
                 `kernels-slang/spirv`, or the instantiation is limited to WGSL)"
                    .to_string(),
            )
        })?;
        let words = spirv::words(code).map_err(|why| shader(why.to_string()))?;
        let declared = spirv::declared(&words).map_err(|why| shader(why.to_string()))?;
        let limit = device
            .core()
            .limits
            .max_per_stage_descriptor_storage_buffers;
        // The scalars' uniform is not a storage buffer.
        let storage = declared.bindings - u32::from(declared.fire.is_some());
        if storage > limit {
            return Err(Fault::StageBuffers {
                kernel: key.to_string(),
                binds: storage,
                limit,
            });
        }
        Ok(())
    }
}

pub(crate) fn build(
    core: &Arc<Core>,
    cache: vk::PipelineCache,
    file: &'static str,
    key: &'static str,
    code: &[u8],
) -> Result<Pipeline> {
    let shader = |why: spirv::Malformed| Fault::Shader {
        file,
        entrypoint: key,
        why: why.to_string(),
    };
    let words = spirv::words(code).map_err(shader)?;
    let declared = spirv::declared(&words).map_err(shader)?;
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..declared.bindings)
        .map(|i| {
            let ty = if declared.fire.as_ref().is_some_and(|fire| fire.binding == i) {
                vk::DescriptorType::UNIFORM_BUFFER
            } else {
                vk::DescriptorType::STORAGE_BUFFER
            };
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // A pool segment's bindings are written after a recording binds them:
    // laid update-after-bind where the device lets them be, and only those
    // (a binding laid so is read slower).
    let segments = if core.update_after_bind {
        declared.segments
    } else {
        0
    };
    let flags: Vec<vk::DescriptorBindingFlags> = (0..declared.bindings)
        .map(|i| {
            if i < 64 && segments & (1 << i) != 0 {
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            } else {
                vk::DescriptorBindingFlags::empty()
            }
        })
        .collect();
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    if segments != 0 {
        layout_info = layout_info
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags);
    }
    let d = &core.device;
    // SAFETY: `d` is live; every create info and the arrays it borrows are
    // locals that outlive their call. `words` is SPIR-V that `spirv::words`
    // and `spirv::declared` parsed, whose bindings and push range the layout
    // copies; the subgroup size is chained only when the device can pin it.
    // Each failure path destroys exactly what was made before it.
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(&layout_info, None)
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
                fire: declared.fire,
                segments,
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
