use std::sync::Arc;

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame};
use engine_gpu::guest::widen::{Cfg, UNIFORM_SOURCE, WG, config_bytes};

use engine_gpu::error::{Fault, Result};

pub struct Widen {
    core: Arc<Core>,

    binds: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl std::fmt::Debug for Widen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Widen")
    }
}

impl Widen {
    pub fn new(device: &Context) -> Result<Widen> {
        let core = device.core().clone();

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

        let pipeline = crate::guest::validated(&core, "create_compute_pipeline", || {
            let module = core
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("pie widen"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(UNIFORM_SOURCE)),
                });
            core.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pie widen"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
        .map_err(|error| Fault::Shader {
            file: "widen.wgsl",
            entrypoint: "main",
            why: format!("the readout widen was refused: {error}"),
        })?;

        Ok(Widen {
            core,
            binds,
            pipeline,
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
        if !src_at.is_multiple_of(4) || !dst_at.is_multiple_of(4) {
            return Err(Fault::Program {
                at: "guest::widen",
                why: format!(
                    "a widen reads and writes whole words, and was offered byte offsets \
                     {src_at} and {dst_at}"
                ),
            });
        }
        let bytes = config_bytes(Cfg {
            src_word: u32::try_from(src_at / 4).unwrap_or(u32::MAX),
            dst_word: u32::try_from(dst_at / 4).unwrap_or(u32::MAX),
            lanes,
            pad: 0,
        });
        let (cfg, cfg_at) = frame.uniform_slot(bytes.len() as u64);
        self.core.queue.write_buffer(&cfg, cfg_at, &bytes);

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
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cfg,
                            offset: cfg_at,
                            size: std::num::NonZeroU64::new(bytes.len() as u64),
                        }),
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
}

fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &buffer.slab().buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(buffer.bytes()),
    })
}
