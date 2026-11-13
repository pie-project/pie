use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, ValueDesc};

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame};
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::INSTANCE;

use engine_gpu::guest::held::{self, consecutive};
use engine_gpu::guest::session::{BINDINGS, CFG_BINDING, StageTables, groups_for, stage_tables};

/// A span of a pool slot (or run) a guest holds.
pub type Held = held::Held<crate::device::Api>;

/// Where an instance's tables go: the `Instance` slots the runtime took
/// for them, and a table's alignment there.
pub struct Tables<'a> {
    pub pools: &'a Pools<crate::device::Api>,
    pub slots: &'a [u32],
    pub align: u64,
}

const READ_ONLY: [bool; BINDINGS as usize] = [false, true, true, true, false, false];

const STATUS_SPAN: u64 = eta_exec::STATUS_BYTES as u64;

const OUTBOX_FLOOR: u64 = 256;

const OUTBOX_KIND: Memory = Memory::Host;

pub struct Code {
    binds: wgpu::BindGroupLayout,
    _layout: wgpu::PipelineLayout,
    stages: Vec<Vec<wgpu::ComputePipeline>>,

    strides: Vec<bool>,
}

impl std::fmt::Debug for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Code")
            .field("stages", &self.stages.len())
            .field(
                "dispatches",
                &self.stages.iter().map(Vec::len).sum::<usize>(),
            )
            .finish()
    }
}

impl Code {
    pub fn new(device: &Context, code: &[engine_gpu::guest::Lowered<String>]) -> Result<Code> {
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
        let stages = code
            .iter()
            .map(|lowered| build_pipelines(&core, &layout, lowered))
            .collect::<Result<Vec<_>>>()?;
        Ok(Code {
            binds,
            _layout: layout,
            stages,
            strides: code
                .iter()
                .map(|lowered| lowered.strides_the_grid)
                .collect(),
        })
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }
}

struct Stage {
    group: wgpu::BindGroup,

    groups: u32,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,

    /// The stage's bind-time tables in the instance's `Instance` slots,
    /// in `TABLE_BINDINGS` order.
    tables: [Held; 4],

    heap: Buffer,

    outbox: Buffer,

    outbox_layout: Vec<(u32, u64, u64)>,

    collected: Vec<u8>,
}

pub struct Session {
    core: Arc<Core>,
    status: Buffer,
    stages: Vec<Stage>,
    /// The `Instance` slots the stages' tables lie in, in binding order,
    /// and whether they are one run.
    slots: Vec<u32>,
    run: bool,

    code: Arc<Code>,
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
        code: Arc<Code>,
        extents: &Extents,
        tables: &Tables<'_>,
    ) -> Result<Session> {
        if code.stages() != plans.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "{} stage plans but {} compiled forms; a session runs every stage or none",
                    plans.len(),
                    code.stages()
                ),
            });
        }

        let mut status = Buffer::with(
            device,
            eta_exec::STATUS_BYTES as u64,
            Memory::Host,
            engine_gpu::books::owners::GUEST,
        )?;
        status.write(0, &[0u8; eta_exec::STATUS_BYTES])?;

        let core = device.core().clone();
        let resolved = plans
            .iter()
            .map(|plan| {
                stage_tables(plan, extents).map_err(|why| Fault::Program {
                    at: "guest::session",
                    why,
                })
            })
            .collect::<Result<Vec<StageTables>>>()?;
        // Every stage's tables, in the instance's slots, written at once.
        let bytes: Vec<Vec<u8>> = resolved
            .iter()
            .flat_map(|stage| stage.tables.iter().cloned())
            .collect();
        let (placed, run) =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let mut stages = Vec::with_capacity(plans.len());
        for (at, resolved) in resolved.into_iter().enumerate() {
            let tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            stages.push(build_stage(
                device,
                &core,
                &code.binds,
                &status,
                resolved,
                tables,
                code.strides[at],
            )?);
        }

        Ok(Session {
            core,
            status,
            stages,
            slots: tables.slots.to_vec(),
            run,
            code,
        })
    }

    /// Points the stages' bind groups at the chunks their tables' slots
    /// sit in now, after an evacuation.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                table.rebind(pools)?;
            }
        }
        self.regroup()
    }

    /// Moves the tables to `slots` (`Instance` slot refs, as many as they
    /// lie in, in order), whose bytes the runtime copied there.
    pub fn retarget(&mut self, pools: &Pools<crate::device::Api>, slots: &[u32]) -> Result<()> {
        if slots.len() != self.slots.len() || (self.run && !consecutive(slots)) {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "tables in {} instance slot(s){} rebound to {slots:?}",
                    self.slots.len(),
                    if self.run { " of one run" } else { "" }
                ),
            });
        }
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                let at = self
                    .slots
                    .iter()
                    .position(|&slot| slot == table.slot())
                    .expect("a table lies in one of the session's slots");
                table.retarget(pools, slots[at])?;
            }
        }
        self.slots = slots.to_vec();
        self.regroup()
    }

    fn regroup(&mut self) -> Result<()> {
        for stage in &mut self.stages {
            stage.group = group(
                &self.core,
                &self.code.binds,
                &self.status,
                &stage.tables,
                &stage.heap,
            )?;
        }
        Ok(())
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

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {at} is past this package's {}", self.stages.len()),
        })?;
        for pipeline in &self.code.stages[at] {
            frame.dispatch("guest", pipeline, &stage.group, [stage.groups, 1, 1])?;
        }
        Ok(())
    }

    pub fn read_back(
        &mut self,
        device: &Context,
        frame: &mut Frame,
        stage: usize,
        values: &[u32],
    ) -> Result<()> {
        let mut layout = Vec::with_capacity(values.len());
        let mut cursor = STATUS_SPAN;
        for &value in values {
            let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} is not stage {stage}'s"),
            })?;
            let held = self.stages[stage].heap.bytes() - at;
            layout.push((value, at, cursor, len.next_multiple_of(4).min(held)));
            cursor += len.next_multiple_of(4).min(held);
        }
        let Session { status, stages, .. } = self;
        let seat = &mut stages[stage];
        if seat.outbox.bytes() < cursor {
            seat.outbox = Buffer::with(
                device,
                cursor,
                OUTBOX_KIND,
                engine_gpu::books::owners::GUEST,
            )?;
        }
        frame.copy(status, 0, &seat.outbox, 0, STATUS_SPAN)?;
        for &(_, at, into, len) in &layout {
            frame.copy(&seat.heap, at, &seat.outbox, into, len)?;
        }
        seat.outbox_layout = layout
            .into_iter()
            .map(|(value, _, into, len)| (value, into, len))
            .collect();
        seat.collected.clear();
        Ok(())
    }

    pub fn collect(&mut self, stage: usize) -> Result<()> {
        let stages = self.stages.len();
        let seat = self.stages.get_mut(stage).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {stage} is past this package's {stages}"),
        })?;
        let span = seat
            .outbox_layout
            .last()
            .map_or(STATUS_SPAN, |&(_, into, len)| into + len);
        let mut out = vec![0u8; span as usize];
        seat.outbox.read(0, &mut out)?;
        seat.collected = out;
        Ok(())
    }

    pub fn taken(&self, stage: usize, value: u32) -> Result<Vec<u8>> {
        let (_, held) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let seat = &self.stages[stage];
        let &(_, into, len) = seat
            .outbox_layout
            .iter()
            .find(|&&(seated, _, _)| seated == value)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} of stage {stage} was not read back"),
            })?;
        let (into, end) = (into as usize, (into + len.min(held)) as usize);
        seat.collected
            .get(into..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} of stage {stage} was not collected"),
            })
    }

    pub fn status(&self, stage: usize) -> Result<Option<u32>> {
        let raw = self
            .stages
            .get(stage)
            .and_then(|seat| seat.collected.get(..4))
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("stage {stage}'s status was not collected"),
            })?;
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

fn build_pipelines(
    core: &Arc<Core>,
    pipeline_layout: &wgpu::PipelineLayout,
    lowered: &engine_gpu::guest::Lowered<String>,
) -> Result<Vec<wgpu::ComputePipeline>> {
    if lowered.entries.is_empty() {
        return Ok(Vec::new());
    }
    let started = crate::device::host::Instant::now();
    let built = crate::guest::validated(core, "create_compute_pipeline", || {
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pie guest"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&lowered.code)),
            });
        lowered
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
            .collect()
    })
    .map_err(|error| Fault::Wgpu {
        what: "create_compute_pipeline",
        why: format!(
            "a guest stage's {} step(s) ({}) were refused: {error}",
            lowered.entries.len(),
            lowered.entries.join(", ")
        ),
    });
    tracing::info!(
        entries = lowered.entries.len(),
        ms = started.elapsed().as_millis() as u64,
        "guest stage pipelines built"
    );
    built
}

fn span(held: &Held) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: &held.buffer().slab().buffer,
        offset: held.offset(),
        size: std::num::NonZeroU64::new(held.bytes()),
    })
}

/// A stage's bind group: the status word, its tables where their slots lie
/// now (each at an offset the table layout aligned to the device's binding
/// alignment), its heap.
fn group(
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    status: &Buffer,
    tables: &[Held; 4],
    heap: &Buffer,
) -> Result<wgpu::BindGroup> {
    let [descs, params, offs, cfg] = tables;
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
                resource: span(descs),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: span(params),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: span(offs),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: whole(heap),
            },
            wgpu::BindGroupEntry {
                binding: CFG_BINDING,
                resource: span(cfg),
            },
        ],
    });
    core.take_error("create_bind_group")?;
    Ok(group)
}

fn build_stage(
    device: &Context,
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    status: &Buffer,
    resolved: StageTables,
    tables: [Held; 4],
    strides_the_grid: bool,
) -> Result<Stage> {
    let StageTables {
        descriptors,
        offsets,
        scratch,
        ..
    } = resolved;
    let heap = Buffer::zeroed(device, scratch, engine_gpu::books::owners::GUEST_STATE)?;
    let outbox = Buffer::with(
        device,
        OUTBOX_FLOOR,
        OUTBOX_KIND,
        engine_gpu::books::owners::GUEST,
    )?;
    let group = group(core, binds, status, &tables, &heap)?;

    Ok(Stage {
        group,
        groups: if strides_the_grid {
            groups_for(&descriptors, device.cores())
        } else {
            1
        },
        offsets,
        descriptors,
        tables,
        heap,
        outbox,
        outbox_layout: Vec::new(),
        collected: Vec::new(),
    })
}
