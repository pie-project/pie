use std::cell::Cell;
use std::sync::Arc;

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame};
use crate::error::{Fault, Result};

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
@group(0) @binding(2) var<uniform> cfg : Cfg;

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

pub struct Widen {
    core: Arc<Core>,

    binds: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    cfg: wgpu::Buffer,

    armed: Cell<bool>,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Widen")
    }
}

impl Widen {
    pub fn new(device: &Context) -> Result<Widen> {
        let core = device.core().clone();
        let cfg = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pie widen cfg"),
            size: size_of::<Cfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        core.take_error("create_buffer")?;

        let storage = |at: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let binds = core
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pie widen"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pie widen"),
                bind_group_layouts: &[Some(&binds)],
                immediate_size: 0,
            });
        core.take_error("create_pipeline_layout")?;

        let scope = core.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pie widen"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SOURCE)),
            });
        let pipeline = core
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pie widen"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(Fault::Shader {
                file: "widen.wgsl",
                entrypoint: "main",
                why: format!("the readout widen was refused: {error}"),
            });
        }

        Ok(Widen {
            core,
            binds,
            pipeline,
            cfg,
            armed: Cell::new(false),
        })
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
            return Err(Fault::Program {
                at: "guest::widen",
                why: "a second readout widen was recorded into one frame; there is one config \
                      slot and the first would run with the second's offsets"
                    .into(),
            });
        }
        if !src_at.is_multiple_of(4) || !dst_at.is_multiple_of(4) {
            return Err(Fault::Program {
                at: "guest::widen",
                why: format!(
                    "a widen reads and writes whole words, and was offered byte offsets \
                     {src_at} and {dst_at}"
                ),
            });
        }
        self.core.queue.write_buffer(
            &self.cfg,
            0,
            &config_bytes(Cfg {
                src_word: u32::try_from(src_at / 4).unwrap_or(u32::MAX),
                dst_word: u32::try_from(dst_at / 4).unwrap_or(u32::MAX),
                lanes,
                pad: 0,
            }),
        );
        self.armed.set(true);

        let group = self
            .core
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pie widen"),
                layout: &self.binds,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: whole(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: whole(dst),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cfg.as_entire_binding(),
                    },
                ],
            });
        self.core.take_error("create_bind_group")?;

        frame.dispatch(
            "widen",
            &self.pipeline,
            &group,
            [lanes.div_ceil(WG).max(1), 1, 1],
        )
    }

    pub fn disarm(&self) {
        self.armed.set(false);
    }
}

fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &buffer.slab().buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(buffer.bytes()),
    })
}

fn config_bytes(cfg: Cfg) -> [u8; size_of::<Cfg>()] {
    let mut out = [0u8; size_of::<Cfg>()];
    out[0..4].copy_from_slice(&cfg.src_word.to_le_bytes());
    out[4..8].copy_from_slice(&cfg.dst_word.to_le_bytes());
    out[8..12].copy_from_slice(&cfg.lanes.to_le_bytes());
    out[12..16].copy_from_slice(&cfg.pad.to_le_bytes());
    out
}
