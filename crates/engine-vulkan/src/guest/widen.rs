#![allow(unsafe_code)]
use std::sync::Arc;

use ash::vk;

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame};
use engine_gpu::guest::widen::{Cfg, STORAGE_SOURCE, WG, config_bytes, word};

use engine_gpu::error::Result;

/// What the widen's pipeline layout binds: source, destination, config.
const LAYOUT: [(u32, super::Slot); 3] = [
    (0, super::Slot::Storage),
    (1, super::Slot::Storage),
    (2, super::Slot::Storage),
];

/// The config's bytes, a whole number of words (`vkCmdUpdateBuffer` writes words).
const CFG_BYTES: u64 = size_of::<Cfg>() as u64;

pub struct Widen {
    core: Arc<Core>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// The one config cell every widen reads, written in the command stream
    /// ahead of its dispatch: a frame records as many widens as its passes
    /// read rows, and each runs with its own offsets.
    cfg: Buffer,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Widen")
    }
}

impl Widen {
    pub fn new(device: &Context) -> Result<Widen> {
        let words = super::lower(usize::MAX, STORAGE_SOURCE, &LAYOUT).map_err(|why| {
            engine_gpu::error::Fault::Program {
                at: "guest::widen",
                why: format!("the readout widen does not lower: {why}"),
            }
        })?;
        let core = device.core().clone();
        let cfg = Buffer::with(
            device,
            CFG_BYTES,
            Memory::Host,
            engine_gpu::books::owners::GUEST,
        )?;

        let bindings = LAYOUT.map(|(at, slot)| binding(at, super::descriptor_type(slot)));
        let d = &core.device;
        // SAFETY: `d` is live; each create info and what it borrows are locals
        // that outlive the call, and `words` is the SPIR-V the guest compiler
        // lowered from `STORAGE_SOURCE`, whose three storage bindings the layout
        // declares. The module is destroyed only after the pipeline is built.
        unsafe {
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
            let set_layouts = [set_layout];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(|e| core.fault("vkCreatePipelineLayout", e))?;
            let module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|e| core.fault("vkCreateShaderModule", e))?;
            let name = std::ffi::CString::new("main").expect("`main` holds no interior nul");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(module)
                        .name(&name),
                )
                .layout(layout);
            let built = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
            d.destroy_shader_module(module, None);
            let pipeline = built.map_err(|(_, e)| core.fault("vkCreateComputePipelines", e))?[0];

            Ok(Widen {
                core,
                set_layout,
                layout,
                pipeline,
                cfg,
            })
        }
    }

    pub fn encode(
        &self,
        frame: &Frame,
        src: &Buffer,
        src_at: u64,
        dst: &Buffer,
        dst_at: u64,
        lanes: u32,
    ) -> Result<()> {
        if !src_at.is_multiple_of(4) || !dst_at.is_multiple_of(4) {
            return Err(engine_gpu::error::Fault::Program {
                at: "guest::widen",
                why: format!(
                    "a widen reads and writes whole words, and was offered byte offsets \
                     {src_at} and {dst_at}"
                ),
            });
        }
        let cfg = Cfg {
            src_word: word(src_at)?,
            dst_word: word(dst_at)?,
            lanes,
            pad: 0,
        };

        let set = frame.descriptor_set(self.set_layout)?;
        frame.retain(self.cfg.slab());
        let d = &self.core.device;
        // SAFETY: `set` was handed to this widen alone by `descriptor_set` and
        // no command uses it yet; its writes name live buffers with ranges
        // inside them (the config cell is `CFG_BYTES` wide), and `infos`
        // outlives the update. `cmd` is the open frame's buffer, recorded only
        // on this thread (`Frame` is not `Sync`), and the pipeline and layout
        // are this widen's own.
        unsafe {
            let infos = [
                [buffer_info(src)],
                [buffer_info(dst)],
                [vk::DescriptorBufferInfo::default()
                    .buffer(self.cfg.slab().buffer)
                    .offset(0)
                    .range(CFG_BYTES)],
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..3)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i])
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);

            let cmd = frame.cmd();

            // The config cell is rewritten in the stream: once every widen
            // ahead of this one on the queue has read it, the update lands
            // this widen's offsets, and the barrier below orders the
            // dispatch after it.
            let read = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[read],
                &[],
                &[],
            );
            d.cmd_update_buffer(cmd, self.cfg.slab().buffer, 0, &config_bytes(cfg));

            // The readout may still be landing: a forward submitted ahead of
            // this frame on the queue writes it from its shaders.
            let before = vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            d.cmd_dispatch(cmd, lanes.div_ceil(WG).max(1), 1, 1);

            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

}

impl Drop for Widen {
    fn drop(&mut self) {
        // SAFETY: the program plane drops its instances before its widen, and
        // their sessions settle every frame a widen was recorded into, so no
        // pending command uses these objects.
        unsafe {
            let d = &self.core.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn binding(at: u32, kind: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(at)
        .descriptor_type(kind)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn buffer_info(buffer: &Buffer) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::default()
        .buffer(buffer.slab().buffer)
        .offset(0)
        .range(buffer.bytes())
}
