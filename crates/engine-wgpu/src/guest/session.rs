use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, OpParams, ValueDesc, describe, layout};

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame};
use crate::error::{Fault, Result};

const DESC_WORDS: usize = 9;

const BINDINGS: u32 = 6;

const CFG_BINDING: u32 = 5;

const READ_ONLY: [bool; BINDINGS as usize] = [false, true, true, true, false, false];

const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

const GROUPS_PER_CORE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Cfg {
    value_count: u32,
    temporary: u32,
    op_count: u32,
    lane: u32,
}

const _: () = assert!(size_of::<Cfg>() == 16);

fn desc_words(d: &ValueDesc) -> [u32; DESC_WORDS] {
    let mut out = [0u32; DESC_WORDS];
    out[0] = d.len;
    out[1] = d.rows;
    out[2] = d.last;
    out[3] = d.rank;
    out[4] = d.dtype;
    for (at, &dim) in d.dims.iter().enumerate() {
        out[5 + at] = dim;
    }
    out
}

fn as_bytes<T: Copy>(items: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size_of_val(items));
    for item in items {
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts((item as *const T).cast::<u8>(), size_of::<T>())
        });
    }
    out
}

struct Stage {
    pipelines: Vec<wgpu::ComputePipeline>,

    group: wgpu::BindGroup,

    groups: u32,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,

    _descs: Buffer,
    _params: Buffer,
    _offs: Buffer,

    _cfg: wgpu::Buffer,

    heap: Buffer,

    landing: Buffer,
}

pub struct Session {
    status: Buffer,
    stages: Vec<Stage>,

    _binds: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field(
                "heap",
                &self.stages.iter().map(|s| s.heap.bytes()).sum::<u64>(),
            )
            .finish()
    }
}

impl Session {
    pub fn new(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: &[crate::guest::Lowered],
        extents: &Extents,
    ) -> Result<Session> {
        if plans.is_empty() {
            return Err(Fault::Program {
                at: "guest::session",
                why: "a guest package with no stages has nothing to run".into(),
            });
        }
        if code.len() != plans.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "{} stage plans but {} compiled forms; a session runs every stage or none",
                    plans.len(),
                    code.len()
                ),
            });
        }

        let mut status = Buffer::with(device, eta_exec::STATUS_BYTES as u64, Memory::Host)?;
        status.write(0, &[0u8; eta_exec::STATUS_BYTES])?;

        let core = device.core().clone();
        let binds = bind_layout(&core);
        let layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pie guest"),
                bind_group_layouts: &[Some(&binds)],
                immediate_size: 0,
            });
        core.take_error("create_pipeline_layout")?;

        let mut stages = Vec::with_capacity(plans.len());
        for (plan, lowered) in plans.iter().zip(code) {
            stages.push(build_stage(
                device, &core, &binds, &layout, &status, plan, lowered, extents,
            )?);
        }

        Ok(Session {
            status,
            stages,
            _binds: binds,
            _layout: layout,
        })
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub fn span(&self, stage: usize, value: u32) -> Option<(u64, u64)> {
        let stage = self.stages.get(stage)?;
        let at = value as usize;
        Some((
            *stage.offsets.get(at)?,
            stage.descriptors.get(at)?.device_bytes(),
        ))
    }

    #[must_use]
    pub fn heap(&self, stage: usize) -> Option<&Buffer> {
        self.stages.get(stage).map(|stage| &stage.heap)
    }

    pub fn stage_in(&mut self, stage: usize, value: u32, bytes: &[u8]) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        if bytes.len() as u64 != len {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "value {value} of stage {stage} holds {len} bytes, and a root of \
                     {} was offered; a short root leaves the last fire's tail under it",
                    bytes.len()
                ),
            });
        }
        let stage = &mut self.stages[stage];

        if len.is_multiple_of(4) {
            stage.heap.write(at, bytes)?;
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(len.next_multiple_of(4) as usize, 0);
            stage.heap.write(at, &padded)?;
        }
        Ok(())
    }

    pub fn flush(&mut self, frame: &mut Frame, stage: usize) -> Result<()> {
        let _ = frame;
        if stage >= self.stages.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!("stage {stage} is past this package's {}", self.stages.len()),
            });
        }
        Ok(())
    }

    pub fn copy_in(
        &mut self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        source: &Buffer,
        source_at: u64,
        len: u64,
    ) -> Result<()> {
        let (at, held) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        if len > held {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "value {value} of stage {stage} holds {held} bytes and a copy of {len} \
                     was asked for"
                ),
            });
        }
        let stage = &self.stages[stage];
        frame.copy(source, source_at, &stage.heap, at, len)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {at} is past this package's {}", self.stages.len()),
        })?;
        for pipeline in &stage.pipelines {
            frame.dispatch("guest", pipeline, &stage.group, [stage.groups, 1, 1])?;
        }
        Ok(())
    }

    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, value: u32) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let stage = &self.stages[stage];

        let len = len.next_multiple_of(4).min(stage.heap.bytes() - at);
        frame.copy(&stage.heap, at, &stage.landing, at, len)
    }

    pub fn taken(&self, stage: usize, value: u32) -> Result<Vec<u8>> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let mut out = vec![0u8; len as usize];
        self.stages[stage].landing.read(at, &mut out)?;
        Ok(out)
    }

    pub fn status(&self) -> Result<Option<u32>> {
        let mut raw = [0u8; eta_exec::STATUS_BYTES];
        self.status.read(0, &mut raw)?;
        let code = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok((code != 0).then_some(code))
    }

    pub fn clear_status(&mut self) -> Result<()> {
        self.status.write(0, &[0u8; eta_exec::STATUS_BYTES])
    }
}

fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &buffer.slab().buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(buffer.bytes()),
    })
}

fn bind_layout(core: &Core) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..BINDINGS)
        .map(|at| wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if at == CFG_BINDING {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: READ_ONLY[at as usize],
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    core.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pie guest"),
            entries: &entries,
        })
}

fn groups_for(descriptors: &[ValueDesc], device: &Context) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    let cap = device.cores().max(1).saturating_mul(GROUPS_PER_CORE);
    wanted.clamp(1, cap)
}

fn alias_reshapes(plan: &LaunchStagePlan, values: &mut [u64]) {
    for (result, source) in eta_compiler::codegen::wgsl_analysis::analyze_stage(plan).aliases {
        let Some(&at) = values.get(source as usize) else {
            continue;
        };
        if let Some(slot) = values.get_mut(result as usize) {
            *slot = at;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_stage(
    device: &Context,
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    pipeline_layout: &wgpu::PipelineLayout,
    status: &Buffer,
    plan: &LaunchStagePlan,
    lowered: &crate::guest::Lowered,
    extents: &Extents,
) -> Result<Stage> {
    let descriptors = plan
        .value_types
        .iter()
        .map(|value| {
            describe(value, extents).map_err(|why| Fault::Program {
                at: "guest::session",
                why: format!("a guest value's shape does not resolve against this fire: {why:?}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scratch = layout(&descriptors).map_err(|why| Fault::Program {
        at: "guest::session",
        why: format!("a guest pass's scratch does not fit: {why:?}"),
    })?;
    alias_reshapes(plan, &mut scratch.values);

    let desc_bytes: Vec<u8> = descriptors
        .iter()
        .flat_map(|d| as_bytes(&desc_words(d)))
        .collect();
    let offset_words: Vec<u32> = scratch
        .values
        .iter()
        .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
        .collect();

    let mut records = Vec::with_capacity(plan.ops.len());
    let mut result_base = 0u32;
    for op in &plan.ops {
        records.push(OpParams::of(
            op,
            result_base,
            eta_exec::OpRuntime::default(),
        ));
        result_base += u32::from(op.result_count);
    }
    let config = Cfg {
        value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
        temporary: u32::try_from(scratch.temporary / 4).unwrap_or(u32::MAX),
        op_count: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        lane: 0,
    };

    let mut descs = Buffer::zeroed(device, desc_bytes.len().max(DESC_WORDS * 4) as u64)?;
    descs.write(0, &desc_bytes)?;
    let mut offs = Buffer::zeroed(device, (offset_words.len().max(1) * 4) as u64)?;
    offs.write(0, &as_bytes(&offset_words))?;
    let mut params = Buffer::zeroed(
        device,
        (records.len().max(1) * size_of::<OpParams>()) as u64,
    )?;
    if !records.is_empty() {
        params.write(0, &as_bytes(&records))?;
    }

    let cfg = core.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pie guest cfg"),
        size: size_of::<Cfg>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    core.take_error("create_buffer")?;
    core.queue.write_buffer(&cfg, 0, &as_bytes(&[config]));

    let bytes = scratch.total.max(256);
    let heap = Buffer::zeroed(device, bytes)?;
    let landing = Buffer::with(device, bytes, Memory::Host)?;

    if lowered.entries.is_empty() {
        return Err(Fault::Program {
            at: "guest::session",
            why: "a stage emitted no dispatches, so nothing would run".into(),
        });
    }

    let scope = core.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = core
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pie guest"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&lowered.source)),
        });
    let pipelines: Vec<wgpu::ComputePipeline> = lowered
        .entries
        .iter()
        .map(|entry| {
            core.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pie guest"),
                    layout: Some(pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
        .collect();
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(Fault::Wgpu {
            what: "create_compute_pipeline",
            why: format!(
                "a guest stage's {} step(s) ({}) were refused: {error}",
                lowered.entries.len(),
                lowered.entries.join(", ")
            ),
        });
    }

    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pie guest"),
        layout: binds,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: whole(status),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: whole(&descs),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: whole(&params),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: whole(&offs),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: whole(&heap),
            },
            wgpu::BindGroupEntry {
                binding: CFG_BINDING,
                resource: cfg.as_entire_binding(),
            },
        ],
    });
    core.take_error("create_bind_group")?;

    Ok(Stage {
        pipelines,
        group,
        groups: if lowered.strides_the_grid {
            groups_for(&descriptors, device)
        } else {
            1
        },
        offsets: scratch.values,
        descriptors,
        _descs: descs,
        _params: params,
        _offs: offs,
        _cfg: cfg,
        heap,
        landing,
    })
}
