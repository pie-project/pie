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
use engine_gpu::guest::session::{
    BINDINGS, CFG_BINDING, STEP_BINDING, StageTables, groups_for, stage_tables, step_stride,
    step_uniforms,
};

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

/// Every guest pipeline: one a call of the closed set
/// ([`engine_gpu::guest::step_calls`]), each from a module of its own, all
/// on one layout. Built at load, before the engine serves, and shared by
/// every session: a program registered while serving dispatches these and
/// builds none. One a bind finds missing it builds then and counts
/// ([`Library::built_serving`]).
pub struct Library {
    core: Arc<Core>,
    binds: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    /// By call; why one is missing where it did not build.
    pipelines: std::sync::Mutex<Vec<std::result::Result<wgpu::ComputePipeline, String>>>,
    /// Pipelines built after the load, at a bind.
    serving: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for Library {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Library")
            .field("calls", &engine_gpu::guest::step_calls())
            .field("built", &self.built())
            .finish()
    }
}

impl Library {
    pub fn build(device: &Context) -> Result<Library> {
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
        let started = crate::device::host::Instant::now();
        let pipelines: Vec<_> = (0..engine_gpu::guest::step_calls())
            .map(|call| build_call(&core, &layout, call))
            .collect();
        let library = Library {
            core: core.clone(),
            binds,
            layout,
            pipelines: std::sync::Mutex::new(pipelines),
            serving: std::sync::atomic::AtomicU64::new(0),
        };
        for (call, why) in library
            .pipelines
            .lock()
            .expect("library lock poisoned")
            .iter()
            .enumerate()
            .filter_map(|(at, built)| built.as_ref().err().map(|why| (at, why)))
        {
            tracing::warn!(call, %why, "a guest step's pipeline did not build");
        }
        tracing::info!(
            calls = engine_gpu::guest::step_calls(),
            built = library.built(),
            ms = started.elapsed().as_millis() as u64,
            "every guest step's pipeline is built"
        );
        Ok(library)
    }

    /// The calls whose pipelines are built.
    #[must_use]
    pub fn built(&self) -> usize {
        let pipelines = self.pipelines.lock().expect("library lock poisoned");
        pipelines.iter().filter(|built| built.is_ok()).count()
    }

    /// The pipelines built after the load, each at a bind that found its
    /// call missing: none, where the load built every call.
    #[must_use]
    pub fn built_serving(&self) -> u64 {
        self.serving.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Takes call `call`'s pipeline out, so the next bind that dispatches
    /// it builds it: the counter's control. A session that holds it keeps
    /// its own handle.
    pub fn forget(&self, call: u32) {
        let mut pipelines = self.pipelines.lock().expect("library lock poisoned");
        if let Some(slot) = pipelines.get_mut(call as usize) {
            *slot = Err("forgotten".to_owned());
        }
    }

    /// Call `call`'s pipeline; one the load did not build is built now,
    /// counted, and named.
    fn pipeline(&self, call: u32) -> Result<wgpu::ComputePipeline> {
        let mut pipelines = self.pipelines.lock().expect("library lock poisoned");
        let count = pipelines.len();
        let slot = pipelines
            .get_mut(call as usize)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("guest step call {call} is past the {count} the library holds"),
            })?;
        if let Ok(pipeline) = slot {
            return Ok(pipeline.clone());
        }
        let built = build_call(&self.core, &self.layout, call as usize);
        self.serving
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        tracing::error!(
            call,
            "a bind built a guest step's pipeline the load did not"
        );
        *slot = built.clone();
        built.map_err(|why| Fault::Program {
            at: "guest::session",
            why: format!("guest step call {call} has no pipeline: {why}"),
        })
    }
}

struct Stage {
    group: wgpu::BindGroup,

    /// Its steps' pipelines, in dispatch order.
    steps: Vec<wgpu::ComputePipeline>,
    /// Its steps' uniforms, `stride` apart ([`STEP_BINDING`]).
    args: Buffer,
    stride: u64,

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

    library: Arc<Library>,
    /// A widen config slot for each output root of its widest stage
    /// ([`Session::widen_config`]): a stage's frame widens each root once.
    widens: Option<crate::device::ctx::Held>,
    widen_slots: u64,
    widen_stride: u64,
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
    /// The config slot of a stage frame's `n`th widen: one an output root.
    pub fn widen_config(&self, n: u64) -> Result<(wgpu::Buffer, u64)> {
        match &self.widens {
            Some(held) if n < self.widen_slots => Ok(((**held).clone(), n * self.widen_stride)),
            _ => Err(Fault::Program {
                at: "guest::widen",
                why: format!(
                    "a stage's widen {n} is past the {} its package's output roots laid",
                    self.widen_slots
                ),
            }),
        }
    }

    pub fn new(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: &[engine_gpu::guest::Lowered],
        args: &[StepArgs],
        library: Arc<Library>,
        extents: &Extents,
        tables: &Tables<'_>,
        widen_slots: u64,
    ) -> Result<Session> {
        engine_gpu::made::in_session(|| {
            Session::made(
                device,
                plans,
                code,
                args,
                library,
                extents,
                tables,
                widen_slots,
            )
        })
    }

    /// [`Session::new`]: what it makes is the session's
    /// ([`engine_gpu::made::in_session`]).
    #[allow(clippy::too_many_arguments)]
    fn made(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: &[engine_gpu::guest::Lowered],
        args: &[StepArgs],
        library: Arc<Library>,
        extents: &Extents,
        tables: &Tables<'_>,
        widen_slots: u64,
    ) -> Result<Session> {
        if code.len() != plans.len() || args.len() != plans.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "{} stage plans but {} compiled forms; a session runs every stage or none",
                    plans.len(),
                    code.len()
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
        for ((lowered, resolved), args) in code.iter().zip(resolved).zip(args) {
            let tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            stages.push(build_stage(
                device, &core, &library, &status, resolved, tables, lowered, args,
            )?);
        }

        let widen_stride = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
        let widens = (widen_slots > 0).then(|| {
            core.held(
                &wgpu::BufferDescriptor {
                    label: Some("pie widen configs"),
                    size: widen_slots * widen_stride,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
                engine_gpu::books::owners::GUEST,
            )
        });
        Ok(Session {
            core,
            status,
            stages,
            slots: tables.slots.to_vec(),
            run,
            library,
            widens,
            widen_slots,
            widen_stride,
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
                &self.library.binds,
                &self.status,
                &stage.tables,
                &stage.heap,
                &stage.args,
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
        for (step, pipeline) in stage.steps.iter().enumerate() {
            // The step's node, rung and selector, at its own offset.
            let offset = u32::try_from(step as u64 * stage.stride)
                .expect("a stage's step uniforms fit a dynamic offset");
            frame.dispatch_at(
                "guest",
                pipeline,
                &stage.group,
                &[offset],
                [stage.groups, 1, 1],
            )?;
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
    let step = wgpu::BindGroupLayoutEntry {
        binding: STEP_BINDING,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(u64::from(
                eta_compiler::codegen::wgsl::STEP_BYTES,
            )),
        },
        count: None,
    };
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
        .chain([step])
        .collect();
    core.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pie guest"),
            entries: &entries,
        })
}

/// Builds call `call`'s pipeline from its own module.
fn build_call(
    core: &Arc<Core>,
    pipeline_layout: &wgpu::PipelineLayout,
    call: usize,
) -> std::result::Result<wgpu::ComputePipeline, String> {
    let source =
        engine_gpu::guest::step_module(call).ok_or_else(|| format!("no step call {call}"))?;
    let source = <crate::device::Api as engine_gpu::guest::Lower>::lower(call, source)
        .map_err(|why| why.to_string())?;
    crate::guest::validated(core, "create_compute_pipeline", || {
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pie guest"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(&source)),
            });
        core.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pie guest"),
                layout: Some(pipeline_layout),
                module: &module,
                entry_point: Some(engine_gpu::guest::ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    })
    .map_err(|error| format!("guest step call {call} was refused: {error}"))
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
    args: &Buffer,
) -> Result<wgpu::BindGroup> {
    let [descs, params, offs, cfg] = tables;
    core.made.note("bind group", "guest stage");
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
            wgpu::BindGroupEntry {
                binding: STEP_BINDING,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &args.slab().buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(u64::from(
                        eta_compiler::codegen::wgsl::STEP_BYTES,
                    )),
                }),
            },
        ],
    });
    core.take_error("create_bind_group")?;
    Ok(group)
}

/// A stage's step uniforms ([`step_uniforms`]), `stride` apart: what a
/// program's steps fix, so every bind of the program shares one.
#[derive(Clone)]
pub struct StepArgs {
    buffer: Buffer,
    stride: u64,
}

impl StepArgs {
    pub fn of(device: &Context, lowered: &engine_gpu::guest::Lowered) -> Result<StepArgs> {
        let stride = step_stride(u64::from(
            device.core().limits.min_uniform_buffer_offset_alignment,
        ));
        let uniforms = step_uniforms(&lowered.steps, stride);
        let mut buffer = Buffer::zeroed(
            device,
            uniforms.len() as u64,
            engine_gpu::books::owners::GUEST,
        )?;
        buffer.write(0, &uniforms)?;
        Ok(StepArgs { buffer, stride })
    }
}

fn build_stage(
    device: &Context,
    core: &Arc<Core>,
    library: &Library,
    status: &Buffer,
    resolved: StageTables,
    tables: [Held; 4],
    lowered: &engine_gpu::guest::Lowered,
    args: &StepArgs,
) -> Result<Stage> {
    let StageTables {
        descriptors,
        offsets,
        scratch,
        ..
    } = resolved;
    let steps = lowered
        .steps
        .iter()
        .map(|step| library.pipeline(step.call))
        .collect::<Result<Vec<_>>>()?;
    let (args, stride) = (args.buffer.clone(), args.stride);
    let heap = Buffer::zeroed(device, scratch, engine_gpu::books::owners::GUEST_STATE)?;
    let outbox = Buffer::with(
        device,
        OUTBOX_FLOOR,
        OUTBOX_KIND,
        engine_gpu::books::owners::GUEST,
    )?;
    let group = group(core, &library.binds, status, &tables, &heap, &args)?;

    Ok(Stage {
        group,
        steps,
        args,
        stride,
        groups: if lowered.strides_the_grid {
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
