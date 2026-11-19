use std::sync::Arc;

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};

use eta_compiler::codegen::launch::{LaunchPackage, LaunchStagePlan};
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
use engine_gpu::guest::state::{GuestLayout, Side};

/// A span of a pool slot (or run) a guest holds.
pub type Held = held::Held<crate::device::Api>;

/// Where an instance's or its pair's state lies
/// ([`engine_gpu::guest::state::Placed`]).
pub type Pair = engine_gpu::guest::state::Placed<crate::device::Api>;

/// Where an instance's tables go: the `Instance` slots the runtime took
/// for them, and a table's alignment there.
pub struct Tables<'a> {
    pub pools: &'a Pools<crate::device::Api>,
    pub slots: &'a [u32],
    pub align: u64,
}

/// What the runtime declares read-only (the tables, storage) or a uniform
/// (the config), and the same binding declared read_write storage. An
/// instance's tables and config lie in its pool slots, in the segment buffer
/// its heap binds read_write, and a WebGPU usage scope takes one buffer one
/// way only: a stage's module declares them read_write, and its layout binds
/// every binding read_write storage.
const WIDENED: [(&str, &str); 5] = [
    (
        "@binding(1) var<storage, read>       descs",
        "@binding(1) var<storage, read_write> descs",
    ),
    (
        "@binding(2) var<storage, read>       params",
        "@binding(2) var<storage, read_write> params",
    ),
    (
        "@binding(3) var<storage, read>       offs",
        "@binding(3) var<storage, read_write> offs",
    ),
    (
        "@binding(5) var<uniform>             cfg",
        "@binding(5) var<storage, read_write> cfg",
    ),
    (
        "@binding(6) var<uniform> pie_step",
        "@binding(6) var<storage, read_write> pie_step",
    ),
];

/// `source` with every binding [`WIDENED`] names declared read_write;
/// refused by name where the runtime declares one otherwise.
fn widened(call: usize, source: String) -> std::result::Result<String, String> {
    WIDENED.iter().try_fold(source, |source, (read, written)| {
        if source.contains(read) {
            Ok(source.replace(read, written))
        } else {
            Err(format!(
                "guest step call {call}'s module declares no `{read}`, which this backend \
                 binds read_write storage beside the heap in the pool's segments"
            ))
        }
    })
}

const STATUS_SPAN: u64 = eta_exec::STATUS_BYTES as u64;

/// A status's fault code: none where it is unset, the code where it is a
/// fault, and refused where its state is one no guest runtime writes.
fn read_status(status: eta_exec::Status) -> std::result::Result<Option<u32>, String> {
    match status.state() {
        Some(eta_exec::State::Unset) => Ok(None),
        Some(eta_exec::State::Fault) => Ok(Some(status.fault)),
        _ => Err(format!(
            "status was overwritten: its words read {:#x} {:#x} {:#x} {:#x}, and no guest \
             runtime writes state {:#x}",
            status.state, status.fault, status.reserved0, status.reserved1, status.state
        )),
    }
}

/// What an instance's tables take in its `Instance` slots, in the order a
/// bind places them: each stage's four bind-time tables and its steps'
/// uniforms (`stride` apart), then the status word its stages share.
#[must_use]
pub fn instance_table_sizes(
    plans: &[LaunchStagePlan],
    code: &[engine_gpu::guest::Lowered],
    stride: u64,
) -> Vec<u64> {
    let mut sizes: Vec<u64> = plans
        .iter()
        .zip(code)
        .flat_map(|(plan, lowered)| {
            let mut stage = eta_exec::tables::table_sizes(plan).to_vec();
            stage.push(step_uniforms(&lowered.steps, stride).len() as u64);
            stage
        })
        .collect();
    sizes.push(STATUS_SPAN);
    sizes
}

/// What each stage of `package` reads back at its collect: its status, then
/// each value it puts, in whole words. A put is its channel's shape, fixed in
/// the container, so this is known when the program registers; a put into a
/// channel the package does not declare is refused.
pub fn stage_read_bytes(package: &LaunchPackage) -> std::result::Result<Vec<u64>, String> {
    package
        .stages
        .iter()
        .enumerate()
        .map(|(at, stage)| {
            stage.puts.iter().try_fold(STATUS_SPAN, |sum, put| {
                let channel = package
                    .channels
                    .iter()
                    .find(|channel| channel.id == put.channel)
                    .ok_or_else(|| {
                        format!(
                            "stage {at} puts value {} into channel {}, which the package \
                             declares no channel of",
                            put.value, put.channel
                        )
                    })?;
                let len = channel
                    .shape
                    .iter()
                    .map(|&dim| u64::from(dim))
                    .product::<u64>();
                let bytes = match channel.dtype {
                    eta_ir::container::ChanDType::Concrete(eta_ir::Dtype::Bool) => len,
                    eta_ir::container::ChanDType::Concrete(eta_ir::Dtype::Bf16) => len * 2,
                    _ => len * 4,
                };
                Ok(sum + bytes.max(4).next_multiple_of(4))
            })
        })
        .collect()
}

/// The values stage `at` of `package` puts that its collect copies from the
/// device, in the stage's own numbering: the read-back set, which a bind
/// checks against the stage's heap ([`values_in_heap`]) and a run reads back.
/// A put whose value the stage's plan lays nowhere on the device (its value
/// map names none) is held on the host and copied from no span. A stage the
/// package does not list, or a put of another stage's value, is refused.
pub fn device_puts(package: &LaunchPackage, at: usize) -> Result<Vec<u32>> {
    let stage = package.stages.get(at).ok_or_else(|| Fault::Program {
        at: "guest::session",
        why: format!(
            "stage {at} has a plan, and the package lists {} stage(s)",
            package.stages.len()
        ),
    })?;
    let mut puts = Vec::with_capacity(stage.puts.len());
    for put in &stage.puts {
        // A put with no local is held on the host: no span to copy, none
        // to check.
        if let Some(local) = engine_gpu::guest::run::local_of(package, at, put.value)? {
            puts.push(local);
        }
    }
    Ok(puts)
}

/// Refuses a stage whose values `reads` do not each lie, in whole words,
/// inside its heap of `heap_bytes`: a span past it would have the device
/// reach, or a collect read back, another instance's bytes. A bind checks
/// every value (the device reads their offsets unchecked) and the puts.
fn values_in_heap(
    reads: &[u32],
    offsets: &[u64],
    descriptors: &[ValueDesc],
    heap_bytes: u64,
) -> std::result::Result<(), String> {
    for &value in reads {
        let at = value as usize;
        let (Some(&offset), Some(desc)) = (offsets.get(at), descriptors.get(at)) else {
            return Err(format!("value {value} has no span in its stage"));
        };
        let end = offset.checked_add(desc.device_bytes().next_multiple_of(4));
        if end.is_none_or(|end| end > heap_bytes) {
            return Err(format!(
                "value {value} at byte {offset} of {} runs past its heap of \
                 {heap_bytes} bytes",
                desc.device_bytes()
            ));
        }
    }
    Ok(())
}

thread_local! {
    /// The readback buffer this thread collects its stages through, taken on
    /// its first collect.
    static OUTBOX: Cell<Option<u32>> = const { Cell::new(None) };
}

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
    /// One readback buffer for each thread that runs guest passes, made at
    /// load: a stage's collect copies its status and puts there and maps it
    /// on the thread that ran it (a mapped buffer takes no part in a
    /// submission, so the threads share none).
    outboxes: Vec<Buffer>,
    taken: AtomicU32,
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
    /// The guest pipelines, and `threads` readback buffers of
    /// `outbox_bytes` each.
    pub fn build(device: &Context, outbox_bytes: u64, threads: u32) -> Result<Library> {
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
        let outboxes = (0..threads)
            .map(|_| {
                Buffer::with(
                    device,
                    outbox_bytes,
                    Memory::Readback,
                    engine_gpu::books::owners::GUEST,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let library = Library {
            core: core.clone(),
            binds,
            layout,
            pipelines: std::sync::Mutex::new(pipelines),
            serving: std::sync::atomic::AtomicU64::new(0),
            outboxes,
            taken: AtomicU32::new(0),
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

    /// The bytes one stage's collect reads back at most: each thread's
    /// readback buffer.
    #[must_use]
    pub fn outbox_bytes(&self) -> u64 {
        self.outboxes.first().map_or(0, Buffer::bytes)
    }

    /// This thread's readback buffer, taken on its first collect.
    fn outbox(&self) -> Result<&Buffer> {
        let at = match OUTBOX.get() {
            Some(at) => at,
            None => {
                let at = self.taken.fetch_add(1, Ordering::Relaxed);
                OUTBOX.set(Some(at));
                at
            }
        };
        self.outboxes.get(at as usize).ok_or(Fault::Ceiling {
            what: "threads collecting a guest stage (the readback buffers made at load)",
            need: u64::from(at) + 1,
            have: self.outboxes.len() as u64,
        })
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
    /// Its steps' uniforms in the instance's `Instance` slots, `stride`
    /// apart ([`STEP_BINDING`]).
    args: Held,
    stride: u64,

    groups: u32,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,

    /// The stage's bind-time tables in the instance's `Instance` slots,
    /// in `TABLE_BINDINGS` order.
    tables: [Held; 4],

    /// Its heap ([`engine_gpu::guest::state`]): the side whose state it
    /// lies in, its first byte there and its bytes.
    heap: (Side, u64, u64),

    /// The readback buffer its last read-back went to (its thread's).
    outbox: Option<Buffer>,

    outbox_layout: Vec<(u32, u64, u64)>,

    collected: Vec<u8>,
}

pub struct Session {
    core: Arc<Core>,
    /// The status word the stages share, in the instance's `Instance` slots.
    status: Held,
    stages: Vec<Stage>,
    /// The `Instance` slots the stages' tables lie in, in binding order,
    /// and whether they are one run.
    slots: Vec<u32>,
    run: bool,
    /// The instance's state and its pair's ([`engine_gpu::guest::state`]),
    /// which its stages' heaps and the pair's score room lie in, their
    /// contents dead between fires.
    instance: Pair,
    pair: Pair,
    /// The score room's first byte in the pair's state and its bytes.
    scores: Option<(u64, u64)>,

    library: Arc<Library>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field(
                "heap",
                &self.stages.iter().map(|s| s.heap.2).sum::<u64>(),
            )
            .finish()
    }
}

impl Session {
    pub fn new(
        device: &Context,
        package: &LaunchPackage,
        code: &[engine_gpu::guest::Lowered],
        library: Arc<Library>,
        extents: &Extents,
        tables: &Tables<'_>,
        [instance, pair]: [Pair; 2],
        layout: &GuestLayout,
    ) -> Result<Session> {
        let plans = &package.plans;
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
        // Every stage's tables and step uniforms, then the status word, in
        // the instance's slots ([`instance_table_sizes`]), written at once.
        // The step uniforms lie in a pool segment beside storage, so they
        // bind as storage too, each at an offset storage's alignment allows.
        let stride = step_stride(device.table_align());
        let mut bytes: Vec<Vec<u8>> = resolved
            .iter()
            .zip(code)
            .flat_map(|(stage, lowered)| {
                let mut tables = stage.tables.to_vec();
                tables.push(step_uniforms(&lowered.steps, stride));
                tables
            })
            .collect();
        bytes.push(vec![0u8; eta_exec::STATUS_BYTES]);
        let (placed, run) =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let status = placed
            .next_back()
            .expect("the status word is the last table placed");
        if layout.stages.len() != plans.len()
            || resolved
                .iter()
                .zip(&layout.stages)
                .any(|(stage, &(_, _, bytes))| stage.scratch != bytes)
            || instance.bytes() < layout.instance
            || pair.bytes() < layout.pair
        {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "a guest's state of {} bytes an instance and {} its pair lays heaps {:?}, \
                     and these {} stage(s) take {:?}",
                    instance.bytes(),
                    pair.bytes(),
                    layout.stages,
                    plans.len(),
                    resolved.iter().map(|stage| stage.scratch).collect::<Vec<_>>()
                ),
            });
        }
        let places = [&instance, &pair];
        let mut stages = Vec::with_capacity(plans.len());
        for (at, ((lowered, resolved), &heap)) in
            code.iter().zip(resolved).zip(&layout.stages).enumerate()
        {
            let tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            let args = placed.next().expect("a stage's step uniforms");
            let reads = device_puts(package, at)?;
            let values: Vec<u32> =
                (0..u32::try_from(resolved.descriptors.len()).unwrap_or(u32::MAX)).collect();
            values_in_heap(&values, &resolved.offsets, &resolved.descriptors, heap.2)
                .and_then(|()| {
                    values_in_heap(&reads, &resolved.offsets, &resolved.descriptors, heap.2)
                })
                .map_err(|why| Fault::Program {
                    at: "guest::session",
                    why: format!("stage {at}: {why}"),
                })?;
            let span = heap_span(places, heap)?;
            stages.push(build_stage(
                device, &core, &library, &status, resolved, tables, lowered, args, stride, heap,
                span,
            )?);
        }

        Ok(Session {
            core,
            status,
            stages,
            slots: tables.slots.to_vec(),
            run,
            instance,
            pair,
            scores: layout.scores,
            library,
        })
    }

    /// The pair's score room, where its program reads captured scores.
    #[must_use]
    pub fn scores(&self) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        let (at, bytes) = self.scores?;
        let (buffer, base) = self.pair.span()?;
        Some(engine_gpu::serve::DeviceCell {
            buffer: buffer.clone(),
            at: base + at,
            bytes,
        })
    }

    /// Stage `stage`'s heap: the segment buffer it lies in, its first byte
    /// there and its bytes.
    fn heap_at(&self, stage: usize) -> Result<(&Buffer, u64, u64)> {
        let stage = self.stages.get(stage).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {stage} is past this package's {}", self.stages.len()),
        })?;
        heap_span([&self.instance, &self.pair], stage.heap)
    }

    /// Points the stages' bind groups at the chunks their tables' slots
    /// and the guest's state sit in now, after an evacuation.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                table.rebind(pools)?;
            }
            stage.args.rebind(pools)?;
        }
        self.status.rebind(pools)?;
        self.instance.rebind(pools)?;
        self.pair.rebind(pools)?;
        self.regroup()
    }

    /// Moves the tables to `slots` (`Instance` slot refs, as many as they
    /// lie in, in order), whose bytes the runtime copied there, and the
    /// instance's state and its pair's to `scratch` and `pair`, fresh room
    /// whose contents are dead.
    pub fn retarget(
        &mut self,
        pools: &Pools<crate::device::Api>,
        slots: &[u32],
        scratch: &engine::program::ScratchPlace,
        pair: &engine::program::ScratchPlace,
    ) -> Result<()> {
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
        let to = |table: &Held| {
            let at = self
                .slots
                .iter()
                .position(|&slot| slot == table.slot())
                .expect("a table lies in one of the session's slots");
            slots[at]
        };
        for stage in &mut self.stages {
            for table in &mut stage.tables {
                table.retarget(pools, to(table))?;
            }
            stage.args.retarget(pools, to(&stage.args))?;
        }
        let status = to(&self.status);
        self.status.retarget(pools, status)?;
        self.slots = slots.to_vec();
        self.instance.retarget(pools, scratch)?;
        self.pair.retarget(pools, pair)?;
        self.regroup()
    }

    fn regroup(&mut self) -> Result<()> {
        let places = [&self.instance, &self.pair];
        for stage in &mut self.stages {
            stage.group = group(
                &self.core,
                &self.library.binds,
                &self.status,
                &stage.tables,
                heap_span(places, stage.heap)?,
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

    /// Stage `stage`'s heap: the segment buffer it lies in and its first
    /// byte there.
    #[must_use]
    pub fn heap(&self, stage: usize) -> Option<(&Buffer, u64)> {
        self.heap_at(stage).ok().map(|(buffer, at, _)| (buffer, at))
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
        let (heap, base, _) = self.heap_at(stage)?;
        if len.is_multiple_of(4) {
            heap.write_shared(base + at, bytes)?;
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(len.next_multiple_of(4) as usize, 0);
            heap.write_shared(base + at, &padded)?;
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

    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, values: &[u32]) -> Result<()> {
        let mut layout = Vec::with_capacity(values.len());
        let mut cursor = STATUS_SPAN;
        for &value in values {
            let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} is not stage {stage}'s"),
            })?;
            // Inside its heap, as its bind checked ([`values_in_heap`]).
            let len = len.next_multiple_of(4);
            layout.push((value, at, cursor, len));
            cursor += len;
        }
        let (heap, base, _) = self.heap_at(stage)?;
        let heap = heap.clone();
        let outbox = self.library.outbox()?.clone();
        if outbox.bytes() < cursor {
            return Err(Fault::Ceiling {
                what: "bytes one guest stage reads back (a thread's readback buffer)",
                need: cursor,
                have: outbox.bytes(),
            });
        }
        let Session { status, stages, .. } = self;
        let seat = &mut stages[stage];
        frame.copy(status.buffer(), status.offset(), &outbox, 0, STATUS_SPAN)?;
        for &(_, at, into, len) in &layout {
            frame.copy(&heap, base + at, &outbox, into, len)?;
        }
        seat.outbox = Some(outbox);
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
        seat.outbox
            .as_ref()
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("stage {stage} was collected before it was read back"),
            })?
            .read(0, &mut out)?;
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

    /// Stage `stage`'s fault code, if it faulted: the runtimes' status
    /// layout ([`eta_exec::Status`]), a fault the state `Fault` with its code
    /// in the word after. A stage that ran clean leaves its status unset;
    /// any other state is a status something else overwrote, refused by name
    /// rather than read as a code.
    pub fn status(&self, stage: usize) -> Result<Option<u32>> {
        let status = self
            .stages
            .get(stage)
            .and_then(|seat| eta_exec::Status::read(&seat.collected))
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("stage {stage}'s status was not collected"),
            })?;
        read_status(status).map_err(|why| Fault::Program {
            at: "guest::session",
            why: format!("stage {stage}'s {why}"),
        })
    }

    pub fn clear_status(&mut self) -> Result<()> {
        self.status
            .buffer()
            .write_shared(self.status.offset(), &[0u8; eta_exec::STATUS_BYTES])
    }
}

fn bind_layout(core: &Core) -> wgpu::BindGroupLayout {
    let step = wgpu::BindGroupLayoutEntry {
        binding: STEP_BINDING,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            // Read_write storage, every one ([`WIDENED`]).
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
    let source = widened(call, source)?;
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
    status: &Held,
    tables: &[Held; 4],
    (heap, heap_at, heap_bytes): (&Buffer, u64, u64),
    args: &Held,
) -> Result<wgpu::BindGroup> {
    let [descs, params, offs, cfg] = tables;
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pie guest"),
        layout: binds,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: span(status),
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
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &heap.slab().buffer,
                    offset: heap_at,
                    size: std::num::NonZeroU64::new(heap_bytes),
                }),
            },
            wgpu::BindGroupEntry {
                binding: CFG_BINDING,
                resource: span(cfg),
            },
            wgpu::BindGroupEntry {
                binding: STEP_BINDING,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &args.buffer().slab().buffer,
                    offset: args.offset(),
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

/// Where a stage's heap (`heap`: its side, first byte and bytes) lies in
/// `places` (the instance's state, its pair's): the segment buffer, its
/// first byte there and its bytes.
fn heap_span(
    [instance, pair]: [&Pair; 2],
    (side, at, bytes): (Side, u64, u64),
) -> Result<(&Buffer, u64, u64)> {
    let place = match side {
        Side::Instance => instance,
        Side::Pair => pair,
    };
    let (buffer, base) = place.span().ok_or_else(|| Fault::Program {
        at: "guest::session",
        why: format!("a stage's heap on the {side:?}'s side, which keeps no state"),
    })?;
    Ok((buffer, base + at, bytes))
}

#[allow(clippy::too_many_arguments)]
fn build_stage(
    device: &Context,
    core: &Arc<Core>,
    library: &Library,
    status: &Held,
    resolved: StageTables,
    tables: [Held; 4],
    lowered: &engine_gpu::guest::Lowered,
    args: Held,
    stride: u64,
    heap: (Side, u64, u64),
    span: (&Buffer, u64, u64),
) -> Result<Stage> {
    let StageTables {
        descriptors,
        offsets,
        ..
    } = resolved;
    let steps = lowered
        .steps
        .iter()
        .map(|step| library.pipeline(step.call))
        .collect::<Result<Vec<_>>>()?;
    let group = group(core, &library.binds, status, &tables, span, &args)?;

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
        outbox: None,
        outbox_layout: Vec::new(),
        collected: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::{STATUS_SPAN, device_puts, stage_read_bytes, values_in_heap};
    use eta_compiler::codegen::launch::{LaunchChannel, LaunchPackage, LaunchPut, LaunchStage};
    use eta_ir::Dtype;
    use eta_ir::container::ChanDType;

    fn channel(id: u32, shape: &[u32], dtype: Dtype) -> LaunchChannel {
        LaunchChannel {
            id,
            shape: shape.to_vec(),
            dtype: ChanDType::Concrete(dtype),
            ..LaunchChannel::default()
        }
    }

    #[test]
    fn a_stage_reads_back_its_status_and_each_put_by_its_channel() {
        let package = LaunchPackage {
            channels: vec![
                channel(0, &[4], Dtype::F32),
                channel(1, &[151_936], Dtype::F32),
                channel(2, &[3], Dtype::Bool),
                channel(3, &[], Dtype::I32),
            ],
            stages: vec![
                LaunchStage {
                    puts: vec![
                        LaunchPut {
                            channel: 0,
                            value: 0,
                        },
                        LaunchPut {
                            channel: 2,
                            value: 1,
                        },
                        LaunchPut {
                            channel: 3,
                            value: 2,
                        },
                    ],
                    ..LaunchStage::default()
                },
                LaunchStage {
                    puts: vec![LaunchPut {
                        channel: 1,
                        value: 0,
                    }],
                    ..LaunchStage::default()
                },
                LaunchStage::default(),
            ],
            ..LaunchPackage::default()
        };
        // f32 x4, three bools in a word, a scalar; then a vocab row; then none.
        assert_eq!(
            stage_read_bytes(&package).unwrap(),
            [
                STATUS_SPAN + 16 + 4 + 4,
                STATUS_SPAN + 151_936 * 4,
                STATUS_SPAN
            ]
        );
    }

    #[test]
    fn a_put_into_a_channel_the_package_does_not_declare_is_refused() {
        let package = LaunchPackage {
            channels: vec![channel(0, &[4], Dtype::F32)],
            stages: vec![LaunchStage {
                puts: vec![LaunchPut {
                    channel: 7,
                    value: 0,
                }],
                ..LaunchStage::default()
            }],
            ..LaunchPackage::default()
        };
        let why = stage_read_bytes(&package).expect_err("refused");
        assert!(why.contains("channel 7"), "{why}");
    }

    fn f32s(len: u32) -> eta_exec::ValueDesc {
        eta_exec::ValueDesc {
            len,
            ..eta_exec::ValueDesc::default()
        }
    }

    #[test]
    fn a_put_inside_its_heap_is_read_back() {
        let descriptors = [f32s(3), f32s(1)];
        let offsets = [0, 12];
        assert_eq!(values_in_heap(&[0, 1], &offsets, &descriptors, 16), Ok(()));
    }

    #[test]
    fn a_layout_that_lays_a_value_past_its_heap_is_refused() {
        // Every value of a stage, as a bind checks them: value 2 runs past.
        let descriptors = [f32s(2), f32s(2), f32s(4)];
        let offsets = [0, 8, 16];
        let all = [0, 1, 2];
        assert!(values_in_heap(&all, &offsets, &descriptors, 32).is_ok());
        let why = values_in_heap(&all, &offsets, &descriptors, 24).expect_err("refused");
        assert!(why.contains("value 2"), "{why}");
    }

    #[test]
    fn a_put_past_its_heap_is_refused() {
        // Value 1 ends at byte 20 of a 16-byte heap: the rest is another
        // instance's.
        let descriptors = [f32s(3), f32s(2)];
        let offsets = [0, 12];
        assert!(values_in_heap(&[1], &offsets, &descriptors, 16).is_err());
        assert!(values_in_heap(&[0], &offsets, &descriptors, 16).is_ok());
        // A put with no span in its stage.
        assert!(values_in_heap(&[2], &offsets, &descriptors, 16).is_err());
    }

    /// Two stages: stage 0 lays its first value on the device as local 0
    /// and holds its second on the host; stage 1 lays its one value.
    fn two_stages(puts: Vec<LaunchPut>) -> LaunchPackage {
        use eta_compiler::codegen::launch::LaunchStagePlan;
        LaunchPackage {
            plans: vec![
                LaunchStagePlan {
                    value_map: vec![0, u32::MAX],
                    ..LaunchStagePlan::default()
                },
                LaunchStagePlan {
                    value_map: vec![0],
                    ..LaunchStagePlan::default()
                },
            ],
            stages: vec![
                LaunchStage {
                    puts,
                    ..LaunchStage::default()
                },
                LaunchStage::default(),
            ],
            ..LaunchPackage::default()
        }
    }

    #[test]
    fn a_stage_reads_back_its_device_puts_and_no_host_held_one() {
        let put = |value| LaunchPut { channel: 0, value };
        let package = two_stages(vec![put(0), put(1)]);
        assert_eq!(device_puts(&package, 0).unwrap(), [0]);
        assert_eq!(device_puts(&package, 1).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn a_put_of_another_stages_value_or_an_unlisted_stage_is_refused() {
        // Global value 2 is stage 1's.
        let package = two_stages(vec![LaunchPut {
            channel: 0,
            value: 2,
        }]);
        assert!(device_puts(&package, 0).is_err());
        assert!(device_puts(&package, 2).is_err());
    }

    #[test]
    fn a_status_is_read_in_the_runtimes_layout() {
        use super::read_status;
        use eta_exec::Status;
        let status = |state, fault| Status {
            state,
            fault,
            ..Status::default()
        };
        assert_eq!(read_status(status(0, 0)), Ok(None));
        assert_eq!(read_status(status(3, 0x2a)), Ok(Some(0x2a)));
        // What DX12 read back: top-p's temperature and the vocab where the
        // state lies are an overwritten status, not fault codes.
        assert!(read_status(status(0x3f33_3333, 0)).is_err());
        assert!(read_status(status(151_936, 0)).is_err());
    }
    /// A guest's write past its heap, run on the device: an `op_copy` into a
    /// value whose offset lies at its heap's end. The runtime refuses every
    /// store (the heap is bound at its exact span, beside words it must not
    /// reach), and the first fault's status reads Fault, OUT_OF_BOUNDS, the
    /// index and the limit.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_guest_write_past_its_heap_faults_out_of_bounds_and_lands_nowhere() {
        use super::{Library, STEP_BINDING};
        use crate::device::{Buffer, Context, present};
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&crate::api::DeviceBoot::default(), allowance).unwrap();
        let library = Library::build(&device, 4096, 1).unwrap();
        let call = eta_compiler::codegen::wgsl::step_calls()
            .iter()
            .position(|text| text.contains("op_copy("))
            .expect("a copy step");
        let pipeline = library.pipeline(u32::try_from(call).unwrap()).unwrap();

        let words =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let owner = engine_gpu::books::owners::TEST;
        let buffer = |bytes: &[u8]| {
            let mut buffer = Buffer::zeroed(&device, bytes.len() as u64, owner).unwrap();
            buffer.write(0, bytes).unwrap();
            buffer
        };
        // Two f32 values of four: `a` at the heap's start, `o` at its end.
        const HEAP: u64 = 64;
        const SENTINEL: u32 = 0xdead_beef;
        let f32_tag =
            u32::from(eta_ir::types::to_wire(eta_ir::Dtype::F32).expect("f32 has a wire tag"));
        let desc = |len: u32| [len, 1, len, 1, f32_tag, len, 1, 1, 1];
        let descs = buffer(&words(&[desc(4), desc(4)].concat()));
        let mut op = [0u32; 16];
        op[1] = 0; // a
        op[4] = 1; // o
        let params = buffer(&words(&op));
        // Each value's offset, then the one op's area (none).
        let offs = buffer(&words(&[0, u32::try_from(HEAP).unwrap(), 0]));
        let heap_words = u32::try_from(HEAP / 4).unwrap();
        let cfg = buffer(&eta_exec::tables::Cfg {
            value_count: 2,
            op_count: 1,
            heap_words,
            ..eta_exec::tables::Cfg::default()
        }
        .to_le_bytes());
        let mut heap_image: Vec<u32> = vec![0; 32];
        heap_image[..4].copy_from_slice(&[
            1.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            4.0f32.to_bits(),
        ]);
        heap_image[16..].fill(SENTINEL);
        let heap = buffer(&words(&heap_image));
        let status = buffer(&[0u8; 16]);
        let step = buffer(&[0u8; 256]);

        fn whole(buffer: &Buffer, size: u64) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer.slab().buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(size),
            })
        }
        let entry = |binding: u32, resource| wgpu::BindGroupEntry { binding, resource };
        let group = device
            .core()
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pie guest bounds test"),
                layout: &library.binds,
                entries: &[
                    entry(0, whole(&status, 16)),
                    entry(1, whole(&descs, descs.bytes())),
                    entry(2, whole(&params, params.bytes())),
                    entry(3, whole(&offs, offs.bytes())),
                    // The heap at its exact span, the sentinel words past it.
                    entry(4, whole(&heap, HEAP)),
                    entry(5, whole(&cfg, cfg.bytes())),
                    entry(
                        STEP_BINDING,
                        whole(&step, u64::from(eta_compiler::codegen::wgsl::STEP_BYTES)),
                    ),
                ],
            });
        device.core().take_error("create_bind_group").unwrap();
        let frame = device.frame().unwrap();
        frame
            .dispatch_at("bounds test", &pipeline, &group, &[0], [1, 1, 1])
            .unwrap();
        frame.commit().unwrap();

        let read = |buffer: &Buffer, n: usize| -> Vec<u32> {
            let mut bytes = vec![0u8; n * 4];
            buffer.read(0, &mut bytes).unwrap();
            bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        let status = read(&status, 4);
        assert_eq!(status[0], 3, "the stage faulted: {status:x?}");
        assert_eq!(
            status[1],
            eta_compiler::codegen::fault::OUT_OF_BOUNDS,
            "{status:x?}"
        );
        assert!(
            (heap_words..heap_words + 4).contains(&status[2]),
            "the index of a refused store of `o`: {status:x?}"
        );
        assert_eq!(status[3], heap_words, "the limit: {status:x?}");
        let heap = read(&heap, 32);
        assert!(
            heap[16..].iter().all(|&w| w == SENTINEL),
            "nothing past the heap: {heap:x?}"
        );
        assert_eq!(heap[0], 1.0f32.to_bits(), "`a` stands: {heap:x?}");
    }
}
