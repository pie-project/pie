use std::cell::Cell;
use std::sync::Arc;

use ash::vk;

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame};
use crate::error::Result;

const WG: u32 = 256;

const SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read>       src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

struct Cfg {
  src_word : u32,
  dst_word : u32,
  lanes    : u32,
  pad      : u32,
};
@group(0) @binding(2) var<storage, read> cfg : Cfg;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.lanes) { return; }
  let w = src[cfg.src_word + (i >> 1u)];
  var half : u32 = w & 0xffffu;
  if ((i & 1u) == 1u) { half = w >> 16u; }
  dst[cfg.dst_word + i] = half << 16u;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Cfg {
    src_word: u32,
    dst_word: u32,
    lanes: u32,
    pad: u32,
}

/// Config slots, one per widen still on the queue: a frame that runs ahead
/// keeps its own offsets while the next frame records.
const SLOTS: u64 = 16;

/// Apart by any device's storage-buffer offset alignment.
const SLOT_BYTES: u64 = 256;

pub struct Widen {
    core: Arc<Core>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cfg: Buffer,
    next: Cell<u64>,

    armed: Cell<bool>,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Widen")
    }
}

impl Widen {
    pub fn new(device: &Context) -> Result<Widen> {
        let words =
            super::lower(usize::MAX, SOURCE).map_err(|why| crate::error::Fault::Program {
                at: "guest::widen",
                why: format!("the readout widen does not lower: {why}"),
            })?;
        let core = device.core().clone();
        let cfg = Buffer::with(device, SLOTS * SLOT_BYTES, Memory::Host)?;

        let bindings = [
            binding(0, vk::DescriptorType::STORAGE_BUFFER),
            binding(1, vk::DescriptorType::STORAGE_BUFFER),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let d = &core.device;
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
                next: Cell::new(0),
                armed: Cell::new(false),
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
        if self.armed.get() {
            return Err(crate::error::Fault::Program {
                at: "guest::widen",
                why: "a second readout widen was recorded into one frame; there is one config \
                      slot and the first would run with the second's offsets"
                    .into(),
            });
        }
        if !src_at.is_multiple_of(4) || !dst_at.is_multiple_of(4) {
            return Err(crate::error::Fault::Program {
                at: "guest::widen",
                why: format!(
                    "a widen reads and writes whole words, and was offered byte offsets \
                     {src_at} and {dst_at}"
                ),
            });
        }
        let slot = self.next.get();
        self.next.set((slot + 1) % SLOTS);
        self.cfg.write_shared(
            slot * SLOT_BYTES,
            &config_bytes(Cfg {
                src_word: u32::try_from(src_at / 4).unwrap_or(u32::MAX),
                dst_word: u32::try_from(dst_at / 4).unwrap_or(u32::MAX),
                lanes,
                pad: 0,
            }),
        )?;
        self.armed.set(true);

        let set = frame.descriptor_set(self.set_layout)?;
        frame.retain(self.cfg.slab());
        let d = &self.core.device;
        unsafe {
            let infos = [
                [buffer_info(src)],
                [buffer_info(dst)],
                [vk::DescriptorBufferInfo::default()
                    .buffer(self.cfg.slab().buffer)
                    .offset(slot * SLOT_BYTES)
                    .range(size_of::<Cfg>() as u64)],
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

    pub fn disarm(&self) {
        self.armed.set(false);
    }
}

impl Drop for Widen {
    fn drop(&mut self) {
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

fn config_bytes(cfg: Cfg) -> [u8; size_of::<Cfg>()] {
    let mut out = [0u8; size_of::<Cfg>()];
    out[0..4].copy_from_slice(&cfg.src_word.to_le_bytes());
    out[4..8].copy_from_slice(&cfg.dst_word.to_le_bytes());
    out[8..12].copy_from_slice(&cfg.lanes.to_le_bytes());
    out[12..16].copy_from_slice(&cfg.pad.to_le_bytes());
    out
}
