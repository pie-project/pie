#![allow(unsafe_code)]
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use engine_gpu::error::{Fault, Result};
use ash::vk;

use super::ctx::{Context, Core};
use super::spirv;

/// A descriptor set layout of some count of storage buffers, which every
/// pipeline declaring that many shares: a set laid for one serves any of
/// them ([`Pipelines::set_layouts`]).
pub struct SetLayout {
    core: Arc<Core>,
    pub(crate) layout: vk::DescriptorSetLayout,
}

impl Drop for SetLayout {
    fn drop(&mut self) {
        // SAFETY: the last `Arc<SetLayout>` is gone with every pipeline
        // whose layout named it.
        unsafe {
            self.core
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

pub struct Pipeline {
    core: Arc<Core>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set: Arc<SetLayout>,

    pub bindings: u32,

    pub used: Vec<bool>,

    pub writable: Vec<bool>,
    pub push_bytes: u32,
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
        }
    }
}

#[derive(Default)]
pub struct Pipelines {
    built: RefCell<HashMap<&'static str, Arc<Pipeline>>>,
    /// The set layouts the pipelines share, by their bindings.
    sets: RefCell<HashMap<u32, Arc<SetLayout>>>,
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

    /// Every set layout the pipelines built so far share.
    #[must_use]
    pub fn set_layouts(&self) -> Vec<Arc<SetLayout>> {
        self.sets.borrow().values().cloned().collect()
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
        let pipeline = Arc::new(build(
            device.core(),
            cache.handle(),
            &self.sets,
            file,
            key,
            code,
        )?);
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
        if declared.bindings > limit {
            return Err(Fault::StageBuffers {
                kernel: key.to_string(),
                binds: declared.bindings,
                limit,
            });
        }
        Ok(())
    }
}

/// The set layout of `bindings` storage buffers from `sets`, made there
/// where no pipeline before declared as many.
fn set_layout(
    core: &Arc<Core>,
    sets: &RefCell<HashMap<u32, Arc<SetLayout>>>,
    bindings: u32,
) -> Result<Arc<SetLayout>> {
    if let Some(held) = sets.borrow().get(&bindings) {
        return Ok(Arc::clone(held));
    }
    let entries: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    // Each binding may be written after a recording binds the set: a
    // recording's pool segments are rebound as they come and go.
    let flags = vec![vk::DescriptorBindingFlags::UPDATE_AFTER_BIND; entries.len()];
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&entries)
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .push_next(&mut binding_flags);
    // SAFETY: `core.device` is live and the create info, `entries` and
    // `flags` outlive the call; the device enabled update-after-bind (a
    // load is refused without it).
    let layout = unsafe { core.device.create_descriptor_set_layout(&info, None) }
    .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
    let held = Arc::new(SetLayout {
        core: Arc::clone(core),
        layout,
    });
    sets.borrow_mut().insert(bindings, Arc::clone(&held));
    Ok(held)
}

fn build(
    core: &Arc<Core>,
    cache: vk::PipelineCache,
    sets: &RefCell<HashMap<u32, Arc<SetLayout>>>,
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
    let set = set_layout(core, sets, declared.bindings)?;
    let d = &core.device;
    // SAFETY: `d` is live; every create info and the arrays it borrows are
    // locals that outlive their call. `words` is SPIR-V that `spirv::words`
    // and `spirv::declared` parsed, whose bindings and push range the layout
    // copies; the subgroup size is chained only when the device can pin it.
    // Each failure path destroys exactly what was made before it.
    unsafe {
        let set_layouts = [set.layout];
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
            Err(e) => return Err(core.fault("vkCreatePipelineLayout", e)),
        };
        let module = match d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        {
            Ok(m) => m,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
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
                set,
                bindings: declared.bindings,
                used: declared.used,
                writable: declared.writable,
                push_bytes: declared.push_bytes,
                local: declared.local,
            }),
            Err((_, e)) => {
                d.destroy_pipeline_layout(layout, None);
                Err(core.fault("vkCreateComputePipelines", e))
            }
        }
    }
}
