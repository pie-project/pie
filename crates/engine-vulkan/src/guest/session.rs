#![allow(unsafe_code)]
use std::sync::Arc;

use ash::vk;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{Extents, ValueDesc};

use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame, Pending};
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::{CHANNEL, INSTANCE};

use crate::serve::GuestRing;
use engine_gpu::backend::Buffer as _;
use engine::guest_claims::{Claims, Holder, PairClaim, Run};
use engine_gpu::guest::held::{self, TableLayout, consecutive};
use engine_gpu::guest::ring::{Berth, PassLayout};
use engine_gpu::guest::session::{
    Bases, HEAP_BINDING, STATUS_BINDING, STATUS_SPAN, STEP_BINDING, StageTables, TABLES_BINDING,
    run_bases, stage_tables, step_groups, step_records, step_stride,
};
use engine_gpu::guest::state::{GuestLayout, Side};

/// Where an instance's or its pair's state lies
/// ([`engine_gpu::guest::state::Placed`]).
pub type Pair = engine_gpu::guest::state::Placed<crate::device::Api>;

/// A span of a pool slot (or run) a guest holds.
pub type Held = held::Held<crate::device::Api>;

/// Every guest pipeline: one a call of the closed set
/// ([`engine_gpu::guest::step_calls`]), each from a module of its own, all
/// on one layout. Built at load, before the engine serves, and shared by
/// every session: a program registered while serving dispatches these and
/// builds none. One a bind finds missing it builds then and counts
/// ([`Library::built_serving`]).
pub struct Library {
    core: Arc<Core>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    /// By call; why one is missing where it did not lower or build.
    pipelines: std::sync::Mutex<Vec<std::result::Result<vk::Pipeline, String>>>,
    /// Pipelines [`Library::forget`] took out, which a session may still
    /// dispatch: destroyed with the library.
    retired: std::sync::Mutex<Vec<vk::Pipeline>>,
    /// Pipelines built after the load, at a bind.
    serving: std::sync::atomic::AtomicU64,
    /// Where every live instance's slots and heap lie ([`Claims`]).
    claims: Claims,
}

impl Drop for Library {
    fn drop(&mut self) {
        let d = &self.core.device;
        // SAFETY: the last `Arc<Library>` is gone with the plane and every
        // session that ran its pipelines; each session settles its frames
        // before it drops, so no pending command uses them.
        let built = self.pipelines.get_mut().expect("library lock poisoned");
        let retired = self.retired.get_mut().expect("library lock poisoned");
        unsafe {
            for pipeline in built.iter().flatten().chain(retired.iter()) {
                d.destroy_pipeline(*pipeline, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
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
    /// Lowers and builds every call, spread over the host's cores: a driver
    /// takes tens of milliseconds a pipeline.
    pub fn build(device: &Context) -> Result<Library> {
        let core = device.core();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = crate::guest::stage_layout()
            .into_iter()
            .map(|(i, slot)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(crate::guest::descriptor_type(slot))
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let d = &core.device;
        // SAFETY: `d` is live; the create infos and what they borrow are
        // locals that outlive their calls; the set layout is destroyed if
        // the pipeline layout is refused.
        let (set_layout, layout) = unsafe {
            let set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
            let set_layouts = [set_layout];
            match d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            ) {
                Ok(layout) => (set_layout, layout),
                Err(e) => {
                    d.destroy_descriptor_set_layout(set_layout, None);
                    return Err(core.fault("vkCreatePipelineLayout", e));
                }
            }
        };
        let lowered = lowered_calls();
        let calls = lowered.len();
        let threads = host_threads(calls);
        let cache = device.pipeline_cache();
        let handle = cache.handle();
        let pipelines = spread(calls, threads, |call| {
            lowered[call]
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|code| build_call(core, handle, layout, code))
        });
        cache.built();
        for (call, why) in pipelines
            .iter()
            .enumerate()
            .filter_map(|(at, built)| built.as_ref().err().map(|why| (at, why)))
        {
            tracing::warn!(call, %why, "a guest step's pipeline did not build");
        }
        Ok(Library {
            core: Arc::clone(core),
            layout,
            set_layout,
            pipelines: std::sync::Mutex::new(pipelines),
            retired: std::sync::Mutex::default(),
            serving: std::sync::atomic::AtomicU64::new(0),
            claims: Claims::default(),
        })
    }

    /// The calls whose pipelines are built.
    #[must_use]
    pub fn built(&self) -> usize {
        let pipelines = self.pipelines.lock().expect("library lock poisoned");
        pipelines.iter().filter(|built| built.is_ok()).count()
    }

    /// Why call `call` has no pipeline built, or `None` where it has one.
    #[must_use]
    pub fn unbuilt(&self, call: u32) -> Option<String> {
        let pipelines = self.pipelines.lock().expect("library lock poisoned");
        match pipelines.get(call as usize) {
            Some(Ok(_)) => None,
            Some(Err(why)) => Some(why.clone()),
            None => Some(format!(
                "call {call} is past the {} the library holds",
                pipelines.len()
            )),
        }
    }

    /// The pipelines built after the load, each at a bind that found its
    /// call missing: none, where the load built every call.
    #[must_use]
    pub fn built_serving(&self) -> u64 {
        self.serving.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Takes call `call`'s pipeline out, so the next bind that dispatches
    /// it builds it: the counter's control.
    pub fn forget(&self, call: u32) {
        let mut pipelines = self.pipelines.lock().expect("library lock poisoned");
        if let Some(slot) = pipelines.get_mut(call as usize) {
            let was = std::mem::replace(slot, Err("forgotten".to_owned()));
            if let Ok(pipeline) = was {
                self.retired
                    .lock()
                    .expect("library lock poisoned")
                    .push(pipeline);
            }
        }
    }

    /// Call `call`'s pipeline; one the load did not build is built now,
    /// counted, and named.
    fn pipeline(&self, device: &Context, call: u32) -> Result<vk::Pipeline> {
        let mut pipelines = self.pipelines.lock().expect("library lock poisoned");
        let count = pipelines.len();
        let slot = pipelines
            .get_mut(call as usize)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("guest step call {call} is past the {count} the library holds"),
            })?;
        if let Ok(pipeline) = slot {
            return Ok(*pipeline);
        }
        let built = lowered_calls()[call as usize]
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|code| {
                build_call(
                    &self.core,
                    device.pipeline_cache().handle(),
                    self.layout,
                    code,
                )
            });
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

/// The host threads a build of `items` spreads over.
fn host_threads(items: usize) -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .clamp(1, items.max(1))
}

/// Every call's SPIR-V, lowered once a process (it names no device), over
/// the host's cores.
fn lowered_calls() -> &'static [std::result::Result<Vec<u32>, String>] {
    static LOWERED: std::sync::OnceLock<Vec<std::result::Result<Vec<u32>, String>>> =
        std::sync::OnceLock::new();
    LOWERED.get_or_init(|| {
        let calls = engine_gpu::guest::step_calls();
        spread(calls, host_threads(calls), |call| {
            engine_gpu::guest::step_module(call)
                .ok_or_else(|| format!("no step call {call}"))
                .and_then(|source| {
                    <crate::device::Api as engine_gpu::guest::Lower>::lower(call, source)
                        .map_err(|why| why.to_string())
                })
        })
    })
}

/// `work` of every index below `items`, over `threads` host threads, each
/// taking the next index left: the costly ones (a sort's, a select's) do
/// not queue behind one another on one thread.
fn spread<T: Send>(items: usize, threads: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut done: Vec<(usize, T)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if at >= items {
                            return mine;
                        }
                        mine.push((at, work(at)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a build thread panicked"))
            .collect()
    });
    done.sort_unstable_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, value)| value).collect()
}

/// Builds the pipeline of a call's SPIR-V on `layout`.
fn build_call(
    core: &Arc<Core>,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    code: &[u32],
) -> std::result::Result<vk::Pipeline, String> {
    let d = &core.device;
    let name =
        std::ffi::CString::new(engine_gpu::guest::ENTRY).expect("the entry name holds no nul");
    // SAFETY: `d` is live and its pipeline cache is internally synchronised;
    // `code` is SPIR-V naga emitted for `stage_layout`'s bindings, which
    // `layout` binds; the module is destroyed only after the pipeline is
    // built from it.
    unsafe {
        let module = d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(code), None)
            .map_err(|e| core.fault("vkCreateShaderModule", e).to_string())?;
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(&name),
            )
            .layout(layout);
        let built = d.create_compute_pipelines(cache, &[info], None);
        d.destroy_shader_module(module, None);
        built
            .map(|pipelines| pipelines[0])
            .map_err(|(_, e)| core.fault("vkCreateComputePipelines", e).to_string())
    }
}

struct Stage {
    /// Its steps' pipelines, in dispatch order.
    steps: Vec<vk::Pipeline>,
    /// Its steps' records in the instance's `Instance` slots, `stride`
    /// apart ([`STEP_BINDING`]), each with the instance's bases in the run
    /// its tables lie in: written once, at the bind.
    args: Held,
    stride: u64,

    /// Each step's workgroups ([`engine_gpu::guest::session::step_groups`]).
    groups: Vec<u32>,
    set: vk::DescriptorSet,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,
    /// The stage's bind-time tables in the instance's `Instance` slots, in
    /// the order they lie: descriptors, op params, offsets, config.
    tables: [Held; 4],

    /// Its heap ([`engine_gpu::guest::state`]): the side whose state it
    /// lies in, its first byte there and its bytes.
    heap: (Side, u64, u64),
}

/// A pass of one stage: its berth in the guest ring the load laid, which
/// its host-held roots stage through and its status and the values the
/// host reads land in, held until the pass lands.
pub struct Pass {
    berth: Berth<Buffer>,
    layout: PassLayout,
}

impl std::fmt::Debug for Pass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass")
            .field("at", &self.berth.at())
            .field("bytes", &self.berth.bytes())
            .finish()
    }
}

impl Pass {
    /// The fault code the landed pass's status holds, if it faulted.
    pub fn status(&self) -> Result<Option<u32>> {
        let at = self.layout.status_at().ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: "a pass's status was read before it was copied back".into(),
        })?;
        let mut word = [0u8; 4];
        self.berth.buffer().read(self.berth.at() + at, &mut word)?;
        let code = u32::from_le_bytes(word);
        Ok((code != 0).then_some(code))
    }

    /// The bytes of `value` the landed pass read back.
    pub fn value(&self, value: u32) -> Result<Vec<u8>> {
        let (at, bytes) = self.layout.read_at(value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} was not read back by this pass"),
        })?;
        self.read(at, bytes)
    }

    /// Bytes `[at, at + bytes)` of its berth.
    pub fn read(&self, at: u64, bytes: u64) -> Result<Vec<u8>> {
        let mut out = vec![0u8; bytes as usize];
        self.berth.buffer().read(self.berth.at() + at, &mut out)?;
        Ok(out)
    }

    /// Where value `value` was read back to in its berth, and its bytes.
    #[must_use]
    pub fn read_at(&self, value: u32) -> Option<(u64, u64)> {
        self.layout.read_at(value)
    }
}

/// The cells of channels the device keeps between fires, by channel: one
/// set for the engine, so a pass reads what another program's pass kept.
/// A channel's cell lies in the `Channel` slots the runtime took for it
/// when it was registered (or its instance bound), whether or not a pass
/// keeps a cell in it now.
pub type Kept = Arc<std::sync::Mutex<std::collections::HashMap<engine::channel::ChannelId, Cell>>>;

/// Where a channel keeps a cell on the device.
#[derive(Debug)]
pub struct Cell {
    place: CellPlace,
    bytes: u64,
    /// A pass kept a cell here the host has not superseded.
    live: bool,
}

#[derive(Debug)]
enum CellPlace {
    /// In the channel's `Channel` slots (one run where the cell is past a
    /// slot).
    Pooled(Held),
    /// A cell past a chunk, which the contract states no slots for: a
    /// buffer of the engine's own.
    Own(Buffer),
}

impl Cell {
    /// A cell of `bytes` in `slots` (`Channel` slot refs, one run where
    /// `bytes` is past a slot).
    pub fn pooled(pools: &Pools<crate::device::Api>, slots: &[u32], bytes: u64) -> Result<Cell> {
        let count = u32::try_from(slots.len()).unwrap_or(u32::MAX);
        let first = *slots.first().ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: "a channel cell placed in no Channel slots".into(),
        })?;
        if !consecutive(slots) {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!("a channel cell of {bytes} bytes in slots {slots:?}, not one run"),
            });
        }
        Ok(Cell {
            place: CellPlace::Pooled(Held::in_run(pools, CHANNEL, first, count, 0, bytes)?),
            bytes,
            live: false,
        })
    }

    /// A cell past a chunk, in a buffer of its own.
    pub fn own(device: &Context, bytes: u64) -> Result<Cell> {
        Ok(Cell {
            place: CellPlace::Own(Buffer::zeroed(
                device,
                bytes,
                engine_gpu::books::owners::GUEST_STATE,
            )?),
            bytes,
            live: false,
        })
    }

    fn span(&self) -> (&Buffer, u64) {
        match &self.place {
            CellPlace::Pooled(held) => (held.buffer(), held.offset()),
            CellPlace::Own(buffer) => (buffer, 0),
        }
    }

    /// The cell as the forward reads it.
    #[must_use]
    pub fn device_cell(&self) -> engine_gpu::serve::DeviceCell<crate::device::Api> {
        let (buffer, at) = self.span();
        engine_gpu::serve::DeviceCell {
            buffer: buffer.clone(),
            at,
            bytes: self.bytes,
        }
    }

    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        match &mut self.place {
            CellPlace::Pooled(held) => held.rebind(pools),
            CellPlace::Own(_) => Ok(()),
        }
    }

    /// Moves a pooled cell to `slots`, whose bytes the runtime copied there.
    pub fn retarget(&mut self, pools: &Pools<crate::device::Api>, slots: &[u32]) -> Result<()> {
        let need = held::item_slots(
            self.bytes,
            pools.slot_bytes(CHANNEL)?,
            pools.slots_per_chunk(CHANNEL)?,
        )
        .map_or(0, |(slots, _)| slots);
        match &mut self.place {
            CellPlace::Pooled(held) if slots.len() == need as usize && consecutive(slots) => {
                held.retarget(pools, slots[0])
            }
            _ => Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "a channel cell of {} bytes rebound to {slots:?}, which is not a run of the                      {need} slot(s) it holds",
                    self.bytes
                ),
            }),
        }
    }
}

/// Where an instance's tables go: the `Instance` slots the runtime took
/// for them, and a table's alignment there.
pub struct Tables<'a> {
    pub pools: &'a Pools<crate::device::Api>,
    pub slots: &'a [u32],
    pub align: u64,
}

pub struct Session {
    core: Arc<Core>,

    /// The status word the stages share, in the instance's `Instance`
    /// slots: a pass copies it into its berth.
    status: Held,
    stages: Vec<Stage>,
    /// The instance's state and its pair's ([`engine_gpu::guest::state`]),
    /// which its stages' heaps and the pair's score room lie in, their
    /// contents dead between fires.
    instance: Pair,
    pair: Pair,
    /// The score room's first byte in the pair's state and its bytes.
    scores: Option<(u64, u64)>,
    pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    /// What the stages' pipelines and layouts belong to.
    library: Arc<Library>,
    /// The `Instance` slots the stages' tables lie in: one run.
    slots: Vec<u32>,
    /// An `Instance` slot's bytes.
    slot_bytes: u64,
    /// The name its slots and heap are claimed by ([`Claims`]).
    claim: u64,
    /// The group bound on its pair's taking that it is a member of: the
    /// group's name and the taking ([`Session::join_pair`]).
    pair_group: Option<(u64, u64)>,

    kept: Kept,
    /// The engine's channel behind each of the program's.
    channels: Vec<engine::channel::ChannelId>,

    /// Passes submitted and not yet waited on, oldest first.
    airborne: std::collections::VecDeque<Pending>,
    /// The ring a pass takes its berth of ([`Session::pass`]).
    ring: GuestRing,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field("heap", &self.stages.iter().map(|s| s.heap.2).sum::<u64>())
            .finish()
    }
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Context,
        package: &LaunchPackage,
        code: &[engine_gpu::guest::Lowered],
        extents: &Extents,
        library: &Arc<Library>,
        kept: Kept,
        channels: Vec<engine::channel::ChannelId>,
        tables: &Tables<'_>,
        [instance, pair]: [Pair; 2],
        layout: &GuestLayout,
        ring: GuestRing,
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
                    resolved
                        .iter()
                        .map(|stage| stage.scratch)
                        .collect::<Vec<_>>()
                ),
            });
        }
        // A pass of each stage takes a berth of the ring the load laid:
        // a stage whose pass could outgrow the whole ring binds no session.
        for (at, stage) in resolved.iter().enumerate() {
            let need = engine_gpu::guest::ring::stage_need(package, at, STATUS_SPAN, |value| {
                stage
                    .descriptors
                    .get(value as usize)
                    .map(ValueDesc::device_bytes)
            })?;
            if need > ring.bytes() {
                return Err(Fault::Ceiling {
                    what: "bytes a pass of one guest stage may take of the guest ring the load laid",
                    need,
                    have: ring.bytes(),
                });
            }
        }
        // Every stage's tables and step records, then the status word, in
        // one run of the instance's slots
        // ([`engine_gpu::guest::session::instance_table_sizes`]), written at
        // once. A record states its tables' words in the run, which the
        // run's layout fixes before anything is written, and which hold
        // wherever the run moves.
        let stride = step_stride(device.table_align());
        let slot_bytes = tables.pools.slot_bytes(INSTANCE)?;
        let mut sizes: Vec<u64> = Vec::with_capacity(plans.len() * 5 + 1);
        for (stage, lowered) in resolved.iter().zip(code) {
            sizes.extend(stage.tables.iter().map(|table| table.len() as u64));
            sizes.push(lowered.steps.len().max(1) as u64 * stride);
        }
        sizes.push(STATUS_SPAN);
        let places = TableLayout::of(&sizes, slot_bytes, tables.align).places;
        let word = |at: u64| {
            u32::try_from(at / 4).map_err(|_| Fault::Program {
                at: "guest::session",
                why: format!(
                    "a table at byte {at} of its run lies past the words a kernel indexes"
                ),
            })
        };
        let mut bytes: Vec<Vec<u8>> = Vec::with_capacity(sizes.len());
        let mut bases = Vec::with_capacity(plans.len());
        for (at, (stage, lowered)) in resolved.iter().zip(code).enumerate() {
            let laid = &places[at * 5..at * 5 + 4];
            let stated = Bases {
                status: 0,
                descs: word(laid[0])?,
                params: word(laid[1])?,
                offs: word(laid[2])?,
                cfg: word(laid[3])?,
                heap: 0,
            };
            bytes.extend(stage.tables.iter().cloned());
            bytes.push(step_records(&lowered.steps, &stage.areas, stride, stated));
            bases.push(stated);
        }
        bytes.push(vec![0u8; eta_exec::STATUS_BYTES]);
        let placed =
            held::place_tables(tables.pools, INSTANCE, tables.slots, &bytes, tables.align)?;
        let mut placed = placed.into_iter();
        let status = placed
            .next_back()
            .expect("the status word is the last table placed");
        let mut stages = Vec::with_capacity(plans.len());
        for ((((lowered, resolved), &heap), stated), plan) in code
            .iter()
            .zip(resolved)
            .zip(&layout.stages)
            .zip(bases)
            .zip(plans.iter())
        {
            let tables = [(); 4].map(|()| placed.next().expect("four tables a stage"));
            let args = placed.next().expect("a stage's step records");
            // The records state the words the run's layout laid the tables
            // at: the bind checks the tables lie there.
            let (laid, _) = run_bases(&tables, slot_bytes)?;
            if laid != stated {
                return Err(Fault::Program {
                    at: "guest::session",
                    why: format!(
                        "a stage's tables lie at {laid:?} in their run and its step records \
                         state {stated:?}"
                    ),
                });
            }
            stages.push(build_stage(
                device, plan, lowered, resolved, tables, library, args, stride, heap,
            )?);
        }

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3 * stages.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(stages.len() as u32),
        ];
        // A package with no stages holds no sets, and Vulkan has no empty pool.
        let pool = if stages.is_empty() {
            vk::DescriptorPool::null()
        } else {
            // SAFETY: `core.device` is live and the create info and `sizes`
            // outlive the call.
            unsafe {
                core.device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(stages.len() as u32)
                            .pool_sizes(&sizes),
                        None,
                    )
                    .map_err(|e| core.fault("vkCreateDescriptorPool", e))?
            }
        };

        let mut session = Session {
            core,
            status,
            stages,
            pool,
            set_layout: library.set_layout,
            layout: library.layout,
            library: Arc::clone(library),
            slots: tables.slots.to_vec(),
            slot_bytes,
            claim: library.claims.instance(),
            pair_group: None,
            instance,
            pair,
            scores: layout.scores,
            kept,
            channels,
            airborne: std::collections::VecDeque::new(),
            ring,
        };
        session.bind_sets()?;
        Ok(session)
    }

    fn bind_sets(&mut self) -> Result<()> {
        let d = &self.core.device;
        for at in 0..self.stages.len() {
            // SAFETY: `&mut self` owns the pool, so its external sync holds;
            // it was sized for one set per stage and this runs once, at
            // construction. The layout is the stage's live set layout.
            let set = unsafe {
                let layouts = [self.set_layout];
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| self.core.fault("vkAllocateDescriptorSets", e))?[0]
            };
            self.stages[at].set = set;
        }
        self.write_sets()
    }

    /// Points each stage's set at its buffers: the status word, the run of
    /// slots its tables lie in now, its heap at its exact span, and its
    /// step's record; and claims the instance's slots and heap
    /// ([`Claims`]). Nothing on the queue uses the sets: at construction
    /// none has run, and a rebind settles first.
    fn write_sets(&self) -> Result<()> {
        // Claimed before a set is pointed there.
        self.claim_runs()?;
        let d = &self.core.device;
        for (at, stage) in self.stages.iter().enumerate() {
            let (heap, base) = self.heap_at(at)?;
            // The run moved with the tables, and their bases in it did not.
            let (_, (_, start, end)) = run_bases(&stage.tables, self.slot_bytes)?;
            let buffers = [
                (
                    STATUS_BINDING,
                    (
                        self.status.buffer().slab().buffer,
                        self.status.offset(),
                        STATUS_SPAN,
                    ),
                    vk::DescriptorType::STORAGE_BUFFER,
                ),
                (
                    TABLES_BINDING,
                    (stage.tables[0].buffer().slab().buffer, start, end - start),
                    vk::DescriptorType::STORAGE_BUFFER,
                ),
                (
                    HEAP_BINDING,
                    (heap.slab().buffer, base, stage.heap.2),
                    vk::DescriptorType::STORAGE_BUFFER,
                ),
                (
                    STEP_BINDING,
                    (
                        stage.args.buffer().slab().buffer,
                        stage.args.offset(),
                        u64::from(eta_compiler::codegen::wgsl::STEP_BYTES),
                    ),
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                ),
            ];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|&(_, (buffer, offset, range), _)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffers
                .iter()
                .zip(&infos)
                .map(|((binding, _, kind), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(stage.set)
                        .dst_binding(*binding)
                        .descriptor_type(*kind)
                        .buffer_info(info)
                })
                .collect();
            // SAFETY: no command on the queue uses `stage.set` (see above);
            // each write names a live buffer (a pool segment at a table's,
            // the status's or the records' offset, aligned to the device's
            // storage and uniform descriptor offset alignments by the table
            // layout, or a heap's) with the descriptor type its layout
            // binding declares, and the `infos` it points into outlive the
            // call.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
        Ok(())
    }

    /// Each stage's run of table slots.
    fn table_runs(&self) -> Result<Vec<Run>> {
        let mut runs = Vec::new();
        for stage in &self.stages {
            let (_, (segment, start, end)) = run_bases(&stage.tables, self.slot_bytes)?;
            runs.push(Run {
                segment,
                start,
                end,
            });
        }
        Ok(runs)
    }

    /// The runs the instance's own state takes: each stage's run of table
    /// slots, and its state, its heaps' place.
    fn own_runs(&self) -> Result<Vec<Run>> {
        let mut runs = self.table_runs()?;
        if let Some((buffer, at)) = self.instance.span() {
            runs.push(Run {
                segment: buffer.slab_id(),
                start: at,
                end: at + self.instance.bytes(),
            });
        }
        Ok(runs)
    }

    /// The runs its pair's state takes, which the group of instances bound
    /// on its taking claims ([`Claims::claim_with_pair`]).
    fn pair_runs(&self) -> Vec<Run> {
        self.pair
            .span()
            .map(|(buffer, at)| Run {
                segment: buffer.slab_id(),
                start: at,
                end: at + self.pair.bytes(),
            })
            .into_iter()
            .collect()
    }

    /// Claims its own state's runs, and joins its pair's group where it is
    /// bound on one.
    fn claim_runs(&self) -> Result<()> {
        let runs = self.own_runs()?;
        match self.pair_group {
            Some((pair, taking)) => {
                self.library.claims.claim_with_pair(
                    self.claim,
                    &runs,
                    &PairClaim {
                        pair,
                        taking,
                        runs: self.pair_runs(),
                    },
                )?;
            }
            None => self.library.claims.claim(self.claim, &runs)?,
        }
        Ok(())
    }

    /// Joins the group bound on its pair's taking `group` (its name and the
    /// taking, [`engine_gpu::guest::state::Takings::group`]), claiming its
    /// own runs with it: refused by name, and its claims as they stood,
    /// where either overlaps another holder's.
    pub fn join_pair(&mut self, group: Option<(u64, u64)>) -> Result<()> {
        let was = self.pair_group;
        self.pair_group = group;
        if let Err(why) = self.claim_runs() {
            self.pair_group = was;
            return Err(why);
        }
        Ok(())
    }

    /// Points the stages' sets at the chunks their tables' slots and the
    /// pair's state sit in now, after an evacuation; the passes on the
    /// queue land first.
    pub fn rebind(&mut self, pools: &Pools<crate::device::Api>) -> Result<()> {
        self.settle()?;
        for stage in &mut self.stages {
            for table in stage
                .tables
                .iter_mut()
                .chain(std::iter::once(&mut stage.args))
            {
                table.rebind(pools)?;
            }
        }
        self.status.rebind(pools)?;
        self.instance.rebind(pools)?;
        self.pair.rebind(pools)?;
        // The evacuation moved the pair's state for every instance on it:
        // the group's move, claimed as one before this member's own.
        if let Some((pair, _)) = self.pair_group {
            self.library
                .claims
                .claim_all(&[(Holder::Pair(pair), self.pair_runs())])?;
        }
        self.write_sets()
    }

    /// Stage `stage`'s heap: the segment buffer it lies in, and its first
    /// byte there.
    fn heap_at(&self, stage: usize) -> Result<(&Buffer, u64)> {
        let stage = self.stages.get(stage).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {stage} is past this package's {}", self.stages.len()),
        })?;
        let (side, at, _) = stage.heap;
        let place = match side {
            Side::Instance => &self.instance,
            Side::Pair => &self.pair,
        };
        let (buffer, base) = place.span().ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("a stage's heap on the {side:?}'s side, which keeps no state"),
        })?;
        Ok((buffer, base + at))
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

    /// Moves the tables to `slots` (`Instance` slot refs, as many as they
    /// lie in, in order), whose bytes the runtime copied there, and the
    /// instance's state and its pair's to `scratch` and `pair`, fresh room
    /// whose contents are dead.
    /// Zeroes the instance's state and, where `pair` (a new taking of the
    /// pair's slots), the pair's: room another owner left as it stood.
    pub fn clear(&self, pair: bool) -> Result<()> {
        self.instance.clear()?;
        if pair {
            self.pair.clear()?;
        }
        Ok(())
    }

    /// [`Self::clear`] at an unpark: the instance's place past its resident
    /// state, which the runtime copied in with it, and the pair's where
    /// `pair`.
    pub fn clear_moved(&self, pair: bool) -> Result<()> {
        self.instance.clear_dead()?;
        if pair {
            self.pair.clear()?;
        }
        Ok(())
    }

    /// Where its pair's state lies.
    #[must_use]
    pub fn pair_place(&self) -> &engine::program::ScratchPlace {
        self.pair.place()
    }

    pub fn retarget(
        &mut self,
        pools: &Pools<crate::device::Api>,
        slots: &[u32],
        scratch: &engine::program::ScratchPlace,
        pair: &engine::program::ScratchPlace,
        pair_group: Option<(u64, u64)>,
    ) -> Result<()> {
        if slots.len() != self.slots.len() || !consecutive(slots) {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "tables in a run of {} instance slot(s) rebound to {slots:?}",
                    self.slots.len()
                ),
            });
        }
        self.settle()?;
        let held = self
            .stages
            .iter_mut()
            .flat_map(|stage| {
                stage
                    .tables
                    .iter_mut()
                    .chain(std::iter::once(&mut stage.args))
            })
            .chain(std::iter::once(&mut self.status));
        for table in held {
            let at = self
                .slots
                .iter()
                .position(|&slot| slot == table.slot())
                .expect("a table lies in one of the session's slots");
            table.retarget(pools, slots[at])?;
        }
        self.slots = slots.to_vec();
        self.instance.retarget(pools, scratch)?;
        self.pair.retarget(pools, pair)?;
        self.pair_group = pair_group;
        self.write_sets()
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
        self.heap_at(stage).ok()
    }

    /// A pass of stage `stage` over its roots `roots` and the values
    /// `wanted` the host may read back (global ids of `package`): a berth of
    /// the guest ring of what its layout ([`engine_gpu::guest::ring::pass_layout`],
    /// the computation the bind sized the stage's most by) lays, taken now
    /// and given back as the pass drops.
    pub fn pass(
        &self,
        package: &LaunchPackage,
        stage: usize,
        roots: &[u32],
        wanted: &[u32],
    ) -> Result<Pass> {
        let layout = engine_gpu::guest::ring::pass_layout(
            package,
            stage,
            roots,
            wanted,
            STATUS_SPAN,
            |value| self.span(stage, value).map(|(_, bytes)| bytes),
        )?;
        Ok(Pass {
            berth: self.ring.take(layout.bytes())?,
            layout,
        })
    }

    pub fn stage_in(
        &mut self,
        pass: &mut Pass,
        stage: usize,
        value: u32,
        bytes: &[u8],
    ) -> Result<()> {
        let (_, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
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
        // A pass still on the queue may read the heap the copy writes.
        self.settle()?;
        let (at, _) = pass.layout.stage(value)?;
        pass.berth
            .buffer()
            .write_shared(pass.berth.at() + at, bytes)?;
        Ok(())
    }

    /// Copies what `pass` staged into stage `at_stage`'s heap.
    pub fn flush(&mut self, frame: &mut Frame, pass: &Pass, at_stage: usize) -> Result<()> {
        let (heap, base) = self.heap_at(at_stage)?;
        for (value, at, len) in pass.layout.staged() {
            let (into, _) = self.span(at_stage, value).ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} is not stage {at_stage}'s"),
            })?;
            frame.copy(
                pass.berth.buffer(),
                pass.berth.at() + at,
                heap,
                base + into,
                len,
            )?;
        }
        Ok(())
    }

    /// Copies the status word into `pass`'s berth, behind the stage's
    /// dispatch.
    pub fn read_status(&self, frame: &mut Frame, pass: &mut Pass) -> Result<()> {
        let at = pass.layout.status();
        frame.copy(
            self.status.buffer(),
            self.status.offset(),
            pass.berth.buffer(),
            pass.berth.at() + at,
            STATUS_SPAN,
        )?;
        Ok(())
    }

    /// Copies `value` of stage `stage` into `pass`'s berth, behind the
    /// stage's dispatch.
    pub fn read_back(
        &self,
        frame: &mut Frame,
        pass: &mut Pass,
        stage: usize,
        value: u32,
    ) -> Result<()> {
        let (from, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let (heap, base) = self.heap_at(stage)?;
        let (at, _) = pass.layout.read_back(value)?;
        frame.copy(
            heap,
            base + from,
            pass.berth.buffer(),
            pass.berth.at() + at,
            len,
        )?;
        Ok(())
    }

    /// Copies `value` into the device's cell for `channel`, in place of a
    /// read back: the next fire's take of the channel restores it.
    pub fn keep(
        &mut self,
        device: &Context,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        channel: u32,
    ) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let _ = device;
        let id = self.channel(channel)?;
        let mut kept = self.kept.lock().expect("kept cells lock poisoned");
        let cell = kept.get_mut(&id).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!(
                "channel {id} keeps a cell on the device and holds no place for one: its \
                 registration took no Channel slots"
            ),
        })?;
        if cell.bytes < len {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "channel {id} keeps a {len}-byte value in a {}-byte cell",
                    cell.bytes
                ),
            });
        }
        let (into, into_at) = cell.span();
        let (heap, base) = self.heap_at(stage)?;
        frame.copy(heap, base + at, into, into_at, len)?;
        cell.live = true;
        Ok(())
    }

    /// The engine's id of the program's `channel`.
    fn channel(&self, channel: u32) -> Result<engine::channel::ChannelId> {
        self.channels
            .get(channel as usize)
            .copied()
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!(
                    "channel {channel} is past the {} this instance binds",
                    self.channels.len()
                ),
            })
    }

    /// The device cell a kept channel holds.
    #[must_use]
    pub fn kept_cell(
        &self,
        channel: u32,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        let id = self.channels.get(channel as usize)?;
        self.kept
            .lock()
            .expect("kept cells lock poisoned")
            .get(id)
            .filter(|cell| cell.live)
            .map(Cell::device_cell)
    }

    pub fn depart(&mut self, pending: Pending) {
        while self.airborne.front().is_some_and(Pending::landed) {
            self.airborne.pop_front();
        }
        self.airborne.push_back(pending);
    }

    /// Waits for every pass this session has on the queue.
    pub fn settle(&mut self) -> Result<()> {
        while let Some(pending) = self.airborne.pop_front() {
            pending.wait()?;
        }
        Ok(())
    }

    #[must_use]
    pub fn keeps(&self, channel: u32) -> bool {
        self.kept_cell(channel).is_some()
    }

    /// Drops the device's cell for `channel`, so the next take reads the
    /// ring, which the host has written.
    pub fn forget(&mut self, channel: u32) {
        if let Some(id) = self.channels.get(channel as usize)
            && let Some(cell) = self
                .kept
                .lock()
                .expect("kept cells lock poisoned")
                .get_mut(id)
        {
            cell.live = false;
        }
    }

    /// Copies `channel`'s kept cell into `value`; false when the device keeps
    /// no cell for it.
    pub fn restore(
        &mut self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        channel: u32,
    ) -> Result<bool> {
        let Some(cell) = self.kept_cell(channel) else {
            return Ok(false);
        };
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let (heap, base) = self.heap_at(stage)?;
        frame.copy(&cell.buffer, cell.at, heap, base + at, len.min(cell.bytes))?;
        Ok(true)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {at} is past this package's {}", self.stages.len()),
        })?;
        let groups = stage
            .groups
            .iter()
            .map(|&wanted| {
                engine_gpu::workgroups(
                    "guest",
                    [wanted, 1, 1],
                    self.core.limits.max_compute_work_group_count,
                )
                .map(|[groups, ..]| groups)
            })
            .collect::<Result<Vec<u32>>>()?;
        let d = &self.core.device;
        // SAFETY: `cmd` is the open frame's buffer, recorded only on this
        // thread (`Frame` is not `Sync`). The set, layout and pipelines are
        // this session's own and stay alive until it settles the frame; the
        // set was written once at construction and matches the layout.
        unsafe {
            let cmd = frame.cmd();

            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );

            let between = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            for (step, &pipeline) in stage.steps.iter().enumerate() {
                if step > 0 {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[between],
                        &[],
                        &[],
                    );
                }
                // The step's node, rung and selector, at its own offset.
                let offset = u32::try_from(step as u64 * stage.stride)
                    .expect("a stage's step uniforms fit a dynamic offset");
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[stage.set],
                    &[offset],
                );
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_dispatch(cmd, groups[step], 1, 1);
            }

            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::HOST_READ,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Its process parked (`engine::Rebind::Parked`): the passes on the
    /// queue land, and it claims only its tables, where they stayed in
    /// place (`kept`); its state and its share of its pair are another
    /// owner's to take until [`Self::retarget`] claims where it lies again.
    pub fn park(&mut self, kept: bool) -> Result<()> {
        self.settle()?;
        let runs = if kept { self.table_runs()? } else { Vec::new() };
        self.library.claims.park(self.claim, &runs);
        self.pair_group = None;
        Ok(())
    }

    /// Clears the status word a faulted stage left in the instance's slots.
    pub fn clear_status(&mut self) -> Result<()> {
        self.status.write(0, &[0u8; eta_exec::STATUS_BYTES])
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.settle();
        self.library.claims.release(self.claim);
        // SAFETY: `settle` waited every frame this session sent out, so no
        // pending command uses its descriptor sets; destroying the pool frees
        // them with it.
        unsafe {
            let d = &self.core.device;
            d.destroy_descriptor_pool(self.pool, None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_stage(
    device: &Context,
    plan: &eta_compiler::codegen::launch::LaunchStagePlan,
    lowered: &engine_gpu::guest::Lowered,
    resolved: StageTables,
    tables: [Held; 4],
    library: &Library,
    args: Held,
    stride: u64,
    heap: (Side, u64, u64),
) -> Result<Stage> {
    let StageTables {
        descriptors,
        offsets,
        ..
    } = resolved;
    let steps = lowered
        .steps
        .iter()
        .map(|step| library.pipeline(device, step.call))
        .collect::<Result<Vec<_>>>()?;
    Ok(Stage {
        steps,
        args,
        stride,
        groups: if lowered.strides_the_grid {
            step_groups(
                plan,
                &descriptors,
                &lowered.steps,
                device.resident().ok_or_else(|| Fault::Device {
                    call: "guest stage",
                    why: "a guest step strides the grid by the workgroups the device holds, which the load measures before it binds a guest"
                        .into(),
                })?,
            )
        } else {
            vec![1; lowered.steps.len()]
        },
        set: vk::DescriptorSet::null(),
        offsets,
        descriptors,
        tables,
        heap,
    })
}
