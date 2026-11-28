use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchPackage;
use eta_exec::{Extents, ValueDesc};

use super::outbox::{Outbox, Pool};
use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::{Core, Frame};
use engine_gpu::backend::Buffer as _;
use engine_gpu::error::{Fault, Result};
use engine_gpu::store::Pools;
use engine_gpu::store::layout::INSTANCE;

use engine::guest_claims::{Claims, Holder, PairClaim, Run};
use engine_gpu::guest::held::{self, consecutive};
use engine_gpu::guest::session::{
    HEAP_BINDING, STATUS_BINDING, STATUS_SPAN, STEP_BINDING, StageTables, TABLES_BINDING, bases,
    stage_tables, step_groups, step_records, step_stride,
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
/// (the step's record), and the same binding declared read_write storage.
/// The tables and the records lie in the segment the status binds read_write
/// (and the heap may too), and a WebGPU usage scope takes one buffer one way
/// only: a stage's module declares them read_write, and its layout binds
/// every binding read_write storage. No step stores to them (the runtime's
/// isolation facts, held by the step modules' naga test).
const WIDENED: [(&str, &str); 2] = [
    (
        "@binding(1) var<storage, read>       tables",
        "@binding(1) var<storage, read_write> tables",
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

/// Every guest pipeline: one a call of the closed set
/// ([`engine_gpu::guest::step_calls`]), each from a module of its own, all
/// on one layout. Built at load, before the engine serves, and shared by
/// every session: a program registered while serving dispatches these and
/// builds none. One a bind finds missing it builds then and counts
/// ([`Library::built_serving`]).
///
/// Where no two bindings of a buffer may overlap and the pool is one
/// segment (the tab: `pipelines::exact_guest`), the status and the step
/// records cannot share the pool's buffer with the heap, which binds it
/// whole: each lies in a buffer of its own here, made at load and entered
/// in the books by name (`owners::GUEST_MIRRORS`), and the tables, in the
/// same buffer as the heap, are read through the heap's binding (the step
/// modules' merged form, [`merged`]). The instance's slots still hold the
/// tables, and its status and records' places there go unused. One group
/// binds all three, made at load; a session takes a status word and its
/// stages' records here at its bind, and a park or an evacuation rewrites
/// its records, never a group.
pub struct Mirrors {
    status: Buffer,
    records: Buffer,
    group: Option<wgpu::BindGroup>,
    /// The places taken: status words by index, records by byte span.
    taken: std::sync::Mutex<MirrorTaken>,
}

#[derive(Default)]
struct MirrorTaken {
    status: Vec<bool>,
    records: Vec<(u64, u64)>,
}

/// A status word's stride in its mirror, and a stage's records' alignment:
/// a dynamic offset's.
const MIRROR_ALIGN: u64 = 256;

impl Mirrors {
    /// The status words' and step records' bytes the tab's mirrors hold.
    pub const STATUS_BYTES: u64 = 1 << 20;
    pub const RECORD_BYTES: u64 = 4 << 20;

    /// The two buffers, entered in the books by name.
    pub fn new(device: &Context) -> Result<Mirrors> {
        let made = |bytes| {
            Buffer::with(
                device,
                bytes,
                crate::device::Memory::Device,
                engine_gpu::books::owners::GUEST_MIRRORS,
            )
        };
        let mirrors = Mirrors {
            status: made(Self::STATUS_BYTES)?,
            records: made(Self::RECORD_BYTES)?,
            group: None,
            taken: std::sync::Mutex::new(MirrorTaken {
                status: vec![false; (Self::STATUS_BYTES / MIRROR_ALIGN) as usize],
                records: Vec::new(),
            }),
        };
        tracing::info!(
            status_bytes = Self::STATUS_BYTES,
            record_bytes = Self::RECORD_BYTES,
            "the guest's status and step records lie in mirrors of their own (one-segment \
             form: no two bindings of the pool's buffer overlap), entered as \
             `guest status / step records mirrors (one-segment form)`"
        );
        Ok(mirrors)
    }

    /// A status word's first byte and each of `stages`' records' (their
    /// bytes), taken; refused by name where the mirrors are full.
    fn take(&self, stages: &[u64]) -> Result<Mirror> {
        let mut taken = self.taken.lock().expect("mirror lock poisoned");
        let full = |what: &str| Fault::Ceiling {
            what: if what == "status" {
                "status words of the tab's guest status mirror"
            } else {
                "bytes of the tab's guest step-record mirror"
            },
            need: 1,
            have: 0,
        };
        let word = taken
            .status
            .iter()
            .position(|held| !held)
            .ok_or_else(|| full("status"))?;
        let mut records = Vec::with_capacity(stages.len());
        let mut placed: Vec<(u64, u64)> = Vec::with_capacity(stages.len());
        for &bytes in stages {
            let bytes = bytes.max(1).next_multiple_of(MIRROR_ALIGN);
            let mut spans = taken.records.clone();
            spans.extend(placed.iter().copied());
            spans.sort_unstable();
            let mut at = 0;
            for &(from, to) in &spans {
                if from >= at + bytes {
                    break;
                }
                at = at.max(to);
            }
            if at + bytes > Self::RECORD_BYTES {
                return Err(full("records"));
            }
            placed.push((at, at + bytes));
            records.push(at);
        }
        taken.status[word] = true;
        taken.records.extend(placed);
        Ok(Mirror {
            status: word as u64 * MIRROR_ALIGN,
            records,
        })
    }

    fn give_back(&self, mirror: &Mirror) {
        let mut taken = self.taken.lock().expect("mirror lock poisoned");
        if let Some(held) = taken
            .status
            .get_mut((mirror.status / MIRROR_ALIGN) as usize)
        {
            *held = false;
        }
        taken
            .records
            .retain(|(from, _)| !mirror.records.contains(from));
    }
}

/// A session's places in the [`Mirrors`]: its status word's first byte and
/// each stage's records'.
#[derive(Debug, Clone)]
struct Mirror {
    status: u64,
    records: Vec<u64>,
}

/// `source` in the one-segment form ([`Mirrors`]): the tables read through
/// the heap's binding, which spans the same buffer from the same byte, at
/// the same word; the tables' binding declared no more.
fn merged(call: usize, source: String) -> std::result::Result<String, String> {
    const DECL: &str = "@group(0) @binding(1) var<storage, read_write> tables  : array<u32>;";
    let Some(at) = source.find(DECL) else {
        return Err(format!(
            "guest step call {call}'s module declares no `{DECL}`, which the one-segment form \
             reads through the heap"
        ));
    };
    let mut out = String::with_capacity(source.len());
    out.push_str(&source[..at]);
    out.push_str(&source[at + DECL.len()..]);
    Ok(out.replace("tables[", "heap["))
}

pub struct Library {
    core: Arc<Core>,
    /// One group for each pair of the pool's segments: the one an
    /// instance's slots lie in (its status, tables and step records) and the
    /// one its heap does, each bound whole, by their slab ids. Made at load,
    /// with every segment; a dispatch picks its instance's and binds its
    /// step's record at a dynamic offset.
    groups: std::collections::HashMap<(u64, u64), wgpu::BindGroup>,
    /// The one-segment form's ([`Mirrors`]), where the library binds it.
    mirrors: Option<Mirrors>,
    /// Where every live instance's slots and heap lie ([`Claims`]).
    claims: Claims,
    layout: wgpu::PipelineLayout,
    /// By call; why one is missing where it did not build.
    pipelines: std::sync::Mutex<Vec<std::result::Result<wgpu::ComputePipeline, String>>>,
    /// Pipelines built after the load, at a bind.
    serving: std::sync::atomic::AtomicU64,
    /// The readback buffers made at load: a frame's passes copy their
    /// status and puts into the one it takes ([`super::outbox`]).
    outboxes: Arc<Pool>,
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
    /// The guest pipelines, reading back through `outboxes` (which the
    /// load's shelf reserved), and a bind group for each pair of the pool's
    /// `segments`.
    pub fn build(
        device: &Context,
        outboxes: Arc<Pool>,
        segments: &[&Buffer],
        mirrors: Option<Mirrors>,
    ) -> Result<Library> {
        let began = crate::device::host::Instant::now();
        let core = device.core().clone();
        let one = mirrors.is_some();
        if one && segments.len() != 1 {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "the guest's one-segment form binds one segment and the pool has {}",
                    segments.len()
                ),
            });
        }
        let binds = bind_layout(&core, one);
        let layout = core
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pie guest"),
                bind_group_layouts: &[Some(&binds)],
                immediate_size: 0,
            });
        core.take_error("create_pipeline_layout")?;
        let started = crate::device::host::Instant::now();
        let calls = engine_gpu::guest::step_calls();
        tracing::info!(calls, "building every guest step's pipeline");
        let pipelines: Vec<_> = (0..calls)
            .map(|call| {
                let at = crate::device::host::Instant::now();
                let built = build_call(&core, &layout, call, one);
                tracing::info!(
                    call,
                    of = calls,
                    built = built.is_ok(),
                    ms = at.elapsed().as_millis() as u64,
                    "a guest step's pipeline"
                );
                built
            })
            .collect();
        tracing::info!("binding the guest steps' segment groups");
        let mut groups = std::collections::HashMap::with_capacity(segments.len().pow(2));
        let mut mirrors = mirrors;
        match (&mut mirrors, segments) {
            (Some(mirrors), [segment]) => {
                mirrors.group = Some(mirror_group(&core, &binds, segment, mirrors)?);
            }
            _ => {
                for slots in segments {
                    for heap in segments {
                        groups.insert(
                            (slots.slab_id(), heap.slab_id()),
                            segment_group(&core, &binds, slots, heap)?,
                        );
                    }
                }
            }
        }
        let library = Library {
            core: core.clone(),
            groups,
            mirrors,
            claims: Claims::default(),
            layout,
            pipelines: std::sync::Mutex::new(pipelines),
            serving: std::sync::atomic::AtomicU64::new(0),
            outboxes,
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
            total_ms = began.elapsed().as_millis() as u64,
            "every guest step's pipeline is built"
        );
        Ok(library)
    }

    /// The group of an instance whose slots lie in `slots` and heap in
    /// `heap`; refused by name where a segment is not the pool's.
    fn group(&self, slots: &Buffer, heap: &Buffer) -> Result<&wgpu::BindGroup> {
        self.groups
            .get(&(slots.slab_id(), heap.slab_id()))
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: "an instance's slots or heap lie in a buffer that is none of the pool's \
                      segments the library binds"
                    .into(),
            })
    }

    /// The calls whose pipelines are built.
    #[must_use]
    pub fn built(&self) -> usize {
        let pipelines = self.pipelines.lock().expect("library lock poisoned");
        pipelines.iter().filter(|built| built.is_ok()).count()
    }

    /// The bytes one frame's passes read back at most: each readback
    /// buffer's.
    #[must_use]
    pub fn outbox_bytes(&self) -> u64 {
        self.outboxes.bytes()
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

    /// Claims every holder's runs where an evacuation left them, in one
    /// step ([`Claims::claim_all`]): refused by name, every claim as it
    /// stood, where one lands on another's.
    pub fn claim_moved(&self, moved: &[(Holder, Vec<Run>)]) -> Result<()> {
        self.claims.claim_all(moved)?;
        Ok(())
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
        let built = build_call(
            &self.core,
            &self.layout,
            call as usize,
            self.mirrors.is_some(),
        );
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
    /// Its steps' pipelines, in dispatch order.
    steps: Vec<wgpu::ComputePipeline>,
    /// Its steps' records in the instance's `Instance` slots, `stride`
    /// apart ([`STEP_BINDING`]); each states where the instance lies.
    args: Held,
    /// The steps the records are for.
    lowered: Vec<eta_compiler::codegen::wgsl::Step>,
    /// Each node's temporary area, which its steps' records carry
    /// (`StageTables::areas`).
    areas: Vec<u32>,
    stride: u64,

    /// Each step's workgroups ([`engine_gpu::guest::session::step_groups`]).
    groups: Vec<u32>,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,

    /// The stage's bind-time tables in the instance's `Instance` slots,
    /// in the order they lie: descriptors, op params, offsets, config.
    tables: [Held; 4],

    /// Its heap ([`engine_gpu::guest::state`]): the side whose state it
    /// lies in, its first byte there and its bytes.
    heap: (Side, u64, u64),
}

/// What a pass read back: its span of its frame's readback buffer (its
/// status, then each value the host reads), collected once the frame lands.
#[derive(Debug, Clone)]
pub struct Pass {
    outbox: Arc<Outbox>,
    at: u64,
    /// (value, byte from `at`, bytes) each value read back.
    layout: Vec<(u32, u64, u64)>,
}

impl Pass {
    /// The frame's readback buffer this pass read back into.
    #[must_use]
    pub fn outbox(&self) -> &Arc<Outbox> {
        &self.outbox
    }

    /// Its stage's fault code, if it faulted, the frame collected: the
    /// runtimes' status layout ([`eta_exec::Status`]), a fault the state
    /// `Fault` with its code in the word after. A stage that ran clean
    /// leaves its status unset; any other state is a status something else
    /// overwrote, refused by name rather than read as a code.
    pub fn status(&self) -> Result<Option<u32>> {
        let raw = self
            .outbox
            .collected(self.at, STATUS_SPAN)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: "a pass's status was read before its frame was collected".into(),
            })?;
        let status = eta_exec::Status::read(raw).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: "a pass's status is short".into(),
        })?;
        read_status(status).map_err(|why| Fault::Program {
            at: "guest::session",
            why: format!("a pass's {why}"),
        })
    }

    /// Value `value`'s bytes, `held` of them, the frame collected.
    pub fn value(&self, value: u32, held: u64) -> Result<Vec<u8>> {
        let &(_, into, len) = self
            .layout
            .iter()
            .find(|&&(seated, _, _)| seated == value)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} was not read back by its pass"),
            })?;
        self.outbox
            .collected(self.at + into, len.min(held))
            .map(<[u8]>::to_vec)
            .ok_or_else(|| Fault::Program {
                at: "guest::session",
                why: format!("value {value} was read before its frame was collected"),
            })
    }
}

/// Where an evacuation leaves a session ([`Session::resolve`]), to put it
/// there once the copies are committed ([`Session::install`]).
pub struct Rebound {
    status: Held,
    stages: Vec<([Held; 4], Held)>,
    instance: Pair,
    pair: Pair,
    /// Each stage's step records as they name the session there; none for
    /// a parked session, whose unpark writes them.
    records: Option<Vec<Vec<u8>>>,
    /// The runs its own state takes there, by its claims' name.
    pub own: (Holder, Vec<Run>),
    /// Its pair group's there, where it is bound on one.
    pub pair_runs: Option<(Holder, Vec<Run>)>,
}

pub struct Session {
    /// The status word the stages share, in the instance's `Instance` slots.
    status: Held,
    /// Its places in the library's [`Mirrors`], where it binds them: its
    /// status and records are read and written there.
    mirror: Option<Mirror>,
    stages: Vec<Stage>,
    /// The `Instance` slots the stages' tables lie in: one run.
    slots: Vec<u32>,
    /// An `Instance` slot's bytes.
    slot_bytes: u64,
    /// The name its slots and heap are claimed by ([`Claims`]).
    claim: u64,
    /// The group bound on its pair's taking that it is a member of: the
    /// group's name and the taking ([`Session::join_pair`]).
    pair_group: Option<(u64, u64)>,
    /// The instance's state and its pair's ([`engine_gpu::guest::state`]),
    /// which its stages' heaps and the pair's score room lie in, their
    /// contents dead between fires.
    instance: Pair,
    pair: Pair,
    /// The score room's first byte in the pair's state and its bytes.
    scores: Option<(u64, u64)>,

    library: Arc<Library>,

    /// The cells its channels keep on the device, by the program's index.
    cells: engine_gpu::guest::defer::Cells<crate::device::Api>,
    /// Passes submitted and not yet waited on, oldest first.
    airborne: std::collections::VecDeque<crate::device::Pending>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field("heap", &self.stages.iter().map(|s| s.heap.2).sum::<u64>())
            .finish()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Its passes on the queue land before its slots go back.
        let _ = self.settle();
        self.library.claims.release(self.claim);
        if let (Some(mirrors), Some(mirror)) = (&self.library.mirrors, &self.mirror) {
            mirrors.give_back(mirror);
        }
    }
}

impl Session {
    /// Its process parked (`engine::Rebind::Parked`), its passes landed:
    /// it claims only the run of slots its status and tables lie in, where
    /// they stayed in place (`kept`); its state and its share of its pair
    /// are another owner's to take until [`Self::retarget`] claims where it
    /// lies again.
    pub fn park(&mut self, kept: bool) {
        let runs = if kept {
            self.own_runs().into_iter().take(1).collect()
        } else {
            Vec::new()
        };
        self.library.claims.park(self.claim, &runs);
        self.pair_group = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Context,
        package: &LaunchPackage,
        code: &[engine_gpu::guest::Lowered],
        library: Arc<Library>,
        extents: &Extents,
        tables: &Tables<'_>,
        [instance, pair]: [Pair; 2],
        layout: &GuestLayout,
        cells: engine_gpu::guest::defer::Cells<crate::device::Api>,
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

        let resolved = plans
            .iter()
            .map(|plan| {
                stage_tables(plan, extents).map_err(|why| Fault::Program {
                    at: "guest::session",
                    why,
                })
            })
            .collect::<Result<Vec<StageTables>>>()?;
        // Every stage's tables and step records, then the status word, in
        // one run of the instance's slots
        // ([`engine_gpu::guest::session::instance_table_sizes`]), written
        // at once. The records lie in a pool segment beside storage, so they
        // bind as storage too, each at an offset storage's alignment allows;
        // each states where the instance lies, so they are written once the
        // tables are placed ([`Session::locate`]).
        let stride = step_stride(device.table_align());
        let mut bytes: Vec<Vec<u8>> = resolved
            .iter()
            .zip(code)
            .flat_map(|(stage, lowered)| {
                let mut tables = stage.tables.to_vec();
                tables.push(vec![0u8; lowered.steps.len().max(1) * stride as usize]);
                tables
            })
            .collect();
        bytes.push(vec![0u8; eta_exec::STATUS_BYTES]);
        let placed =
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
                    resolved
                        .iter()
                        .map(|stage| stage.scratch)
                        .collect::<Vec<_>>()
                ),
            });
        }
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
            stages.push(build_stage(
                device, &plans[at], &library, resolved, tables, lowered, args, stride, heap,
            )?);
        }

        let mirror = library
            .mirrors
            .as_ref()
            .map(|mirrors| {
                mirrors.take(
                    &stages
                        .iter()
                        .map(|stage| stage.steps.len().max(1) as u64 * stage.stride)
                        .collect::<Vec<_>>(),
                )
            })
            .transpose()?;
        let claim = library.claims.instance();
        let session = Session {
            status,
            mirror,
            stages,
            slots: tables.slots.to_vec(),
            slot_bytes: tables.pools.slot_bytes(INSTANCE)?,
            claim,
            pair_group: None,
            instance,
            pair,
            scores: layout.scores,
            library,
            cells,
            airborne: std::collections::VecDeque::new(),
        };
        session.locate()?;
        Ok(session)
    }

    /// Claims its slots and heap ([`Claims`]), then writes its step
    /// records there ([`Self::write_records`]): at the bind and a rebind.
    fn locate(&self) -> Result<()> {
        // Claimed before a record is written there.
        self.claim_runs()?;
        let records = self.records()?;
        self.write_records(&records)
    }

    /// Each stage's step records as they name where the instance lies now,
    /// each base checked ([`bases`]) and its bind group at hand; nothing
    /// written.
    fn records(&self) -> Result<Vec<Vec<u8>>> {
        let places = [&self.instance, &self.pair];
        let mut records = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let mut bases = bases(
                &self.status,
                &stage.tables,
                &stage.args,
                heap_span(places, stage.heap)?,
                self.slot_bytes,
            )?;
            match &self.mirror {
                // The status word in its mirror, which the status binds whole.
                Some(mirror) => {
                    bases.status =
                        u32::try_from(mirror.status / 4).map_err(|_| Fault::Program {
                            at: "guest::session",
                            why: "a status word lies past the words a kernel indexes".into(),
                        })?;
                }
                None => {
                    self.library
                        .group(self.status.buffer(), heap_span(places, stage.heap)?.0)?;
                }
            }
            records.push(step_records(
                &stage.lowered,
                &stage.areas,
                stage.stride,
                bases,
            ));
        }
        Ok(records)
    }

    /// Writes each stage's step records ([`Self::records`]).
    fn write_records(&self, records: &[Vec<u8>]) -> Result<()> {
        if let (Some(mirrors), Some(mirror)) = (&self.library.mirrors, &self.mirror) {
            for (&at, records) in mirror.records.iter().zip(records) {
                mirrors.records.write_shared(at, records)?;
            }
            return Ok(());
        }
        for (stage, records) in self.stages.iter().zip(records) {
            stage.args.write(0, records)?;
        }
        Ok(())
    }

    /// The runs the instance's own state takes: the run of slots its status
    /// and tables lie in, and its state, its heaps' place.
    fn own_runs(&self) -> Vec<Run> {
        let (start, end) = self.status.run_span(self.slot_bytes);
        let mut runs = vec![Run {
            segment: self.status.buffer().slab_id(),
            start,
            end,
        }];
        if let Some((buffer, at)) = self.instance.span() {
            runs.push(Run {
                segment: buffer.slab_id(),
                start: at,
                end: at + self.instance.bytes(),
            });
        }
        runs
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
        let runs = self.own_runs();
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

    /// Where the stages' tables and records, the status and, bound, the
    /// guest's state lie under `pools`' chunk table, an evacuation's not yet
    /// committed, each checked as [`Self::locate`] checks it, with the runs
    /// the session and its pair would hold there; nothing changed. A park
    /// that kept its tables in place (`parked`) holds their run alone, as
    /// its park claimed: its state and its share of its pair went back at
    /// its park, and its unpark writes its records. The caller claims every
    /// holder's runs in one step ([`Library::claim_moved`]), then installs
    /// each ([`Self::install`]).
    pub fn resolve(&mut self, pools: &Pools<crate::device::Api>, parked: bool) -> Result<Rebound> {
        let mut rebound = Rebound {
            status: self.status.clone(),
            stages: self
                .stages
                .iter()
                .map(|stage| (stage.tables.clone(), stage.args.clone()))
                .collect(),
            instance: self.instance.clone(),
            pair: self.pair.clone(),
            records: None,
            own: (Holder::Instance(self.claim), Vec::new()),
            pair_runs: None,
        };
        for (tables, args) in &mut rebound.stages {
            for table in tables {
                table.rebind(pools)?;
            }
            args.rebind(pools)?;
        }
        rebound.status.rebind(pools)?;
        if !parked {
            rebound.instance.rebind(pools)?;
            rebound.pair.rebind(pools)?;
        }
        // Checked as the session would lie there, then put back.
        self.swap(&mut rebound);
        let records = (!parked).then(|| self.records()).transpose();
        let own = self.own_runs();
        let pair = self.pair_runs();
        self.swap(&mut rebound);
        rebound.records = records?;
        if parked {
            rebound.own.1 = own.into_iter().take(1).collect();
        } else {
            rebound.own.1 = own;
            // The pair's state moved for every instance on it: the group's.
            rebound.pair_runs = self
                .pair_group
                .map(|(group, _)| (Holder::Pair(group), pair));
        }
        Ok(rebound)
    }

    /// Puts the session where [`Self::resolve`] resolved it, the
    /// evacuation's copies committed and its runs claimed: the step records
    /// name it there, where it is bound.
    pub fn install(&mut self, mut rebound: Rebound) -> Result<()> {
        self.swap(&mut rebound);
        match &rebound.records {
            Some(records) => self.write_records(records),
            None => Ok(()),
        }
    }

    /// The session's places and `rebound`'s, each the other's.
    fn swap(&mut self, rebound: &mut Rebound) {
        std::mem::swap(&mut self.status, &mut rebound.status);
        for (stage, (tables, args)) in self.stages.iter_mut().zip(&mut rebound.stages) {
            std::mem::swap(&mut stage.tables, tables);
            std::mem::swap(&mut stage.args, args);
        }
        std::mem::swap(&mut self.instance, &mut rebound.instance);
        std::mem::swap(&mut self.pair, &mut rebound.pair);
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
                    self.slots.len(),
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
        self.pair_group = pair_group;
        self.locate()
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
        let (group, records) = match (&self.library.mirrors, &self.mirror) {
            (Some(mirrors), Some(mirror)) => (
                mirrors.group.as_ref().ok_or_else(|| Fault::Program {
                    at: "guest::session",
                    why: "the guest's mirrors are bound at the library's build".into(),
                })?,
                mirror.records[at],
            ),
            _ => {
                let heap = heap_span([&self.instance, &self.pair], stage.heap)?.0;
                (
                    self.library.group(self.status.buffer(), heap)?,
                    stage.args.offset(),
                )
            }
        };
        for (step, pipeline) in stage.steps.iter().enumerate() {
            // The step's record, at its own offset in the slots' segment
            // (or the records mirror).
            let offset = u32::try_from(records + step as u64 * stage.stride).map_err(|_| {
                Fault::Program {
                    at: "guest::session",
                    why: "a step's record lies past a dynamic offset's reach".into(),
                }
            })?;
            frame.dispatch_at(
                "guest",
                pipeline,
                group,
                &[offset],
                [stage.groups[step], 1, 1],
            )?;
        }
        Ok(())
    }

    /// Copies the stage's status and `values` into the frame's readback
    /// buffer, each from a word: what the pass reads back once its frame
    /// lands, which it holds until it drops.
    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, values: &[u32]) -> Result<Pass> {
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
        let outbox = frame.guest_outbox(&self.library.outboxes)?;
        let into = outbox.reserve(cursor)?;
        let (status, status_at) = match (&self.library.mirrors, &self.mirror) {
            (Some(mirrors), Some(mirror)) => (&mirrors.status, mirror.status),
            _ => (self.status.buffer(), self.status.offset()),
        };
        frame.copy(status, status_at, outbox.buffer(), into, STATUS_SPAN)?;
        for &(_, at, from, len) in &layout {
            frame.copy(&heap, base + at, outbox.buffer(), into + from, len)?;
        }
        Ok(Pass {
            outbox,
            at: into,
            layout: layout
                .into_iter()
                .map(|(value, _, from, len)| (value, from, len))
                .collect(),
        })
    }

    /// Copies `value` into the device's cell for `channel`, in place of a
    /// read back: the next fire's take of the channel restores it.
    pub fn keep(
        &mut self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        channel: u32,
    ) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let (heap, base, _) = self.heap_at(stage)?;
        self.cells.keep(frame, heap, base + at, len, channel)
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
        if !self.cells.keeps(channel) {
            return Ok(false);
        }
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let (heap, base, _) = self.heap_at(stage)?;
        self.cells.restore(frame, heap, base + at, len, channel)
    }

    /// Drops the device's cell for `channel`, so the next take reads the
    /// ring, which the host has written.
    pub fn forget(&mut self, channel: u32) {
        self.cells.forget(channel);
    }

    /// The device cell a kept channel holds.
    #[must_use]
    pub fn kept_cell(
        &self,
        channel: u32,
    ) -> Option<engine_gpu::serve::DeviceCell<crate::device::Api>> {
        self.cells.kept_cell(channel)
    }

    #[must_use]
    pub fn keeps(&self, channel: u32) -> bool {
        self.cells.keeps(channel)
    }

    /// The cells its channels keep on the device.
    #[must_use]
    pub fn cells(&self) -> &engine_gpu::guest::defer::Cells<crate::device::Api> {
        &self.cells
    }

    /// Holds a submitted pass until it lands or the session settles.
    pub fn depart(&mut self, pending: crate::device::Pending) {
        while self
            .airborne
            .front()
            .is_some_and(|pending| pending.landed())
        {
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

    pub fn clear_status(&mut self) -> Result<()> {
        if let (Some(mirrors), Some(mirror)) = (&self.library.mirrors, &self.mirror) {
            return mirrors
                .status
                .write_shared(mirror.status, &[0u8; eta_exec::STATUS_BYTES]);
        }
        self.status
            .buffer()
            .write_shared(self.status.offset(), &[0u8; eta_exec::STATUS_BYTES])
    }
}

fn bind_layout(core: &Core, one: bool) -> wgpu::BindGroupLayout {
    let storage = |binding: u32, dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        // Read_write storage, every one ([`WIDENED`]).
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: dynamic,
            min_binding_size: if dynamic {
                std::num::NonZeroU64::new(u64::from(eta_compiler::codegen::wgsl::STEP_BYTES))
            } else {
                None
            },
        },
        count: None,
    };
    let entries = [
        storage(STATUS_BINDING, false),
        storage(TABLES_BINDING, false),
        storage(HEAP_BINDING, false),
        storage(STEP_BINDING, true),
    ];
    // The one-segment form reads its tables through the heap ([`merged`]).
    let entries: Vec<_> = entries
        .into_iter()
        .filter(|entry| !one || entry.binding != TABLES_BINDING)
        .collect();
    core.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pie guest"),
            entries: &entries,
        })
}

/// Call `call`'s module with only what its entry reaches, in the tab: every
/// step's module carries the whole guest runtime (154 functions, the
/// select's workgroup tables), which SwiftShader in Chrome takes minutes on;
/// naga drops what the entry never reaches and writes the rest back. Native
/// drivers strip it themselves, and serve the module as emitted faster.
/// A native test takes the tab's forms with `PIE_TEST_GUEST_COMPACT` (this)
/// and `PIE_TEST_GUEST_ROLLED` (the pivot finish rolled), to compare them.
fn compacted(call: usize, source: &str) -> std::result::Result<String, String> {
    #[cfg(feature = "test-limits")]
    let source = &if std::env::var_os("PIE_TEST_GUEST_ROLLED").is_some() {
        eta_compiler::codegen::wgsl::roll_pivot(source)
    } else {
        source.to_string()
    };
    #[cfg(feature = "test-limits")]
    let tab = cfg!(target_arch = "wasm32") || std::env::var_os("PIE_TEST_GUEST_COMPACT").is_some();
    #[cfg(not(feature = "test-limits"))]
    let tab = cfg!(target_arch = "wasm32");
    if !tab {
        return Ok(source.to_string());
    }
    let mut module = naga::front::wgsl::parse_str(source).map_err(|e| {
        format!(
            "guest step call {call} does not parse: {}",
            e.emit_to_string(source)
        )
    })?;
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("guest step call {call} compacted does not validate: {e}"))?;
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| format!("guest step call {call} compacted does not write: {e}"))
}

/// Builds call `call`'s pipeline from its own module.
fn build_call(
    core: &Arc<Core>,
    pipeline_layout: &wgpu::PipelineLayout,
    call: usize,
    one: bool,
) -> std::result::Result<wgpu::ComputePipeline, String> {
    let source =
        engine_gpu::guest::step_module(call).ok_or_else(|| format!("no step call {call}"))?;
    let source = widened(call, source)?;
    let source = <crate::device::Api as engine_gpu::guest::Lower>::lower(call, source)
        .map_err(|why| why.to_string())?;
    let source = if one { merged(call, source)? } else { source };
    let source = compacted(call, &source)?;
    // Said before the module is made, so the last line before a device's
    // loss names what it was making.
    tracing::info!(
        call,
        wgsl_bytes = source.len(),
        "a guest step's pipeline: creating"
    );
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

/// The group of every instance whose slots lie in segment `slots` and heap
/// in segment `heap`: the status and the tables over the whole of `slots`,
/// the heap over the whole of `heap`, and a step's record in `slots` at the
/// dynamic offset its dispatch names.
fn segment_group(
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    slots: &Buffer,
    heap: &Buffer,
) -> Result<wgpu::BindGroup> {
    fn whole(segment: &Buffer) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &segment.slab().buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(segment.bytes()),
        })
    }
    fn bound(binding: u32, segment: &Buffer, dynamic: bool) -> crate::device::pipelines::Bound<'_> {
        crate::device::pipelines::Bound {
            binding,
            buffer: &segment.slab().buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(segment.bytes()),
            writable: true,
            dynamic,
        }
    }
    crate::device::pipelines::aliased(
        "pie guest",
        &[
            bound(STATUS_BINDING, slots, false),
            bound(TABLES_BINDING, slots, false),
            bound(HEAP_BINDING, heap, false),
            bound(STEP_BINDING, slots, true),
        ],
        core.one_segment(),
    )?;
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pie guest"),
        layout: binds,
        entries: &[
            wgpu::BindGroupEntry {
                binding: STATUS_BINDING,
                resource: whole(slots),
            },
            wgpu::BindGroupEntry {
                binding: TABLES_BINDING,
                resource: whole(slots),
            },
            wgpu::BindGroupEntry {
                binding: HEAP_BINDING,
                resource: whole(heap),
            },
            wgpu::BindGroupEntry {
                binding: STEP_BINDING,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &slots.slab().buffer,
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

/// The one-segment form's group ([`Mirrors`]): the status mirror whole, the
/// pool's one segment whole as the heap (the tables read through it), and
/// a step's record in the records mirror at the dynamic offset its
/// dispatch names. No two of them share a buffer.
fn mirror_group(
    core: &Arc<Core>,
    binds: &wgpu::BindGroupLayout,
    segment: &Buffer,
    mirrors: &Mirrors,
) -> Result<wgpu::BindGroup> {
    fn whole(buffer: &Buffer) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &buffer.slab().buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(buffer.bytes()),
        })
    }
    fn bound(binding: u32, buffer: &Buffer, size: u64) -> crate::device::pipelines::Bound<'_> {
        crate::device::pipelines::Bound {
            binding,
            buffer: &buffer.slab().buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(size),
            writable: true,
            dynamic: binding == STEP_BINDING,
        }
    }
    crate::device::pipelines::aliased(
        "pie guest",
        &[
            bound(STATUS_BINDING, &mirrors.status, mirrors.status.bytes()),
            bound(HEAP_BINDING, segment, segment.bytes()),
            bound(STEP_BINDING, &mirrors.records, mirrors.records.bytes()),
        ],
        core.one_segment(),
    )?;
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pie guest"),
        layout: binds,
        entries: &[
            wgpu::BindGroupEntry {
                binding: STATUS_BINDING,
                resource: whole(&mirrors.status),
            },
            wgpu::BindGroupEntry {
                binding: HEAP_BINDING,
                resource: whole(segment),
            },
            wgpu::BindGroupEntry {
                binding: STEP_BINDING,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &mirrors.records.slab().buffer,
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

/// Stage heap `(side, at, bytes)`: the segment buffer it lies in, its first
/// byte there and its bytes.
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
    plan: &eta_compiler::codegen::launch::LaunchStagePlan,
    library: &Library,
    resolved: StageTables,
    tables: [Held; 4],
    lowered: &engine_gpu::guest::Lowered,
    args: Held,
    stride: u64,
    heap: (Side, u64, u64),
) -> Result<Stage> {
    let StageTables {
        descriptors,
        offsets,
        areas,
        ..
    } = resolved;
    let steps = lowered
        .steps
        .iter()
        .map(|step| library.pipeline(step.call))
        .collect::<Result<Vec<_>>>()?;

    Ok(Stage {
        steps,
        args,
        lowered: lowered.steps.clone(),
        areas,
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
        offsets,
        descriptors,
        tables,
        heap,
    })
}

#[cfg(test)]
mod tests {
    use super::{device_puts, read_status, stage_read_bytes, values_in_heap};
    use eta_compiler::codegen::launch::{
        LaunchChannel, LaunchPackage, LaunchPut, LaunchStage, LaunchStagePlan,
    };

    /// A put the stage does not hold is refused on the host: into a channel
    /// the package does not declare, past its heap, of another stage's value
    /// or of a stage the package does not list.
    #[test]
    fn a_put_the_stage_does_not_hold_is_refused() {
        let put = |channel, value| LaunchPut { channel, value };
        let undeclared = LaunchPackage {
            channels: vec![LaunchChannel {
                id: 0,
                shape: vec![4],
                ..LaunchChannel::default()
            }],
            stages: vec![LaunchStage {
                puts: vec![put(7, 0)],
                ..LaunchStage::default()
            }],
            ..LaunchPackage::default()
        };
        let why = stage_read_bytes(&undeclared).expect_err("refused");
        assert!(why.contains("channel 7"), "{why}");

        // Value 1 ends at byte 20 of a 16-byte heap; value 2 has no span.
        let f32s = |len| eta_exec::ValueDesc {
            len,
            ..eta_exec::ValueDesc::default()
        };
        let (descriptors, offsets) = ([f32s(3), f32s(2)], [0, 12]);
        assert!(values_in_heap(&[0], &offsets, &descriptors, 16).is_ok());
        assert!(values_in_heap(&[1], &offsets, &descriptors, 16).is_err());
        assert!(values_in_heap(&[2], &offsets, &descriptors, 16).is_err());

        // Stage 0 holds values 0 and 1; global value 2 is stage 1's.
        let two_stages = LaunchPackage {
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
                    puts: vec![put(0, 2)],
                    ..LaunchStage::default()
                },
                LaunchStage::default(),
            ],
            ..LaunchPackage::default()
        };
        assert!(device_puts(&two_stages, 0).is_err());
        assert!(device_puts(&two_stages, 2).is_err());
    }

    /// Regression (DX12): a status word something else overwrote (top-p's
    /// temperature, the vocab) is named, not read as a fault code.
    #[test]
    fn a_status_is_read_in_the_runtimes_layout() {
        use eta_exec::Status;
        let status = |state, fault| Status {
            state,
            fault,
            ..Status::default()
        };
        assert_eq!(read_status(status(0, 0)), Ok(None));
        assert_eq!(read_status(status(3, 0x2a)), Ok(Some(0x2a)));
        assert!(read_status(status(0x3f33_3333, 0)).is_err());
        assert!(read_status(status(151_936, 0)).is_err());
    }

    /// Guest bounds, on the device: two instances' heaps side by side in
    /// one segment; A copies past its heap's end onto B's. A faults
    /// OUT_OF_BOUNDS and writes nothing; B runs clean and byte-identical.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_guest_write_past_its_heap_faults_and_leaves_its_neighbour_byte_identical() {
        use super::Library;
        use crate::device::{Buffer, Context, present};
        use eta_compiler::codegen::steps::{Bases, StepUniform};
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&crate::api::DeviceBoot::default(), allowance).unwrap();
        let owner = engine_gpu::books::owners::TEST;
        let outbox = Buffer::with(
            &device,
            4096,
            crate::device::alloc::Memory::Readback,
            engine_gpu::books::owners::GUEST,
        )
        .unwrap();
        let mut segment = Buffer::zeroed(&device, 8192, owner).unwrap();
        let library =
            Library::build(&device, super::Pool::new(vec![outbox]), &[&segment], None).unwrap();
        let call = eta_compiler::codegen::wgsl::step_calls()
            .iter()
            .position(|text| text.contains("op_copy("))
            .expect("a copy step");
        let pipeline = library.pipeline(u32::try_from(call).unwrap()).unwrap();

        let words =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        // Two f32 values of four a stage, `a` at its heap's start and `o` at
        // `o_at`: an instance's tables from `at`, its record at `at + 512`.
        const HEAP: u64 = 64;
        const SENTINEL: u32 = 0xdead_beef;
        let f32_tag =
            u32::from(eta_ir::types::to_wire(eta_ir::Dtype::F32).expect("f32 has a wire tag"));
        let desc = |len: u32| [len, 1, len, 1, f32_tag, len, 1, 1, 1];
        let heap_words = u32::try_from(HEAP / 4).unwrap();
        let mut lay = |at: u64, o_at: u32, heap_at: u64| -> (Bases, u64) {
            let mut op = [0u32; 16];
            op[1] = 0; // a
            op[4] = 1; // o
            for (from, bytes) in [
                (at, words(&[desc(4), desc(4)].concat())),
                (at + 128, words(&op)),
                (at + 256, words(&[0, o_at])),
                (
                    at + 320,
                    eta_exec::tables::Cfg {
                        value_count: 2,
                        heap_words,
                        op_count: 1,
                        ..eta_exec::tables::Cfg::default()
                    }
                    .to_le_bytes(),
                ),
            ] {
                segment.write(from, &bytes).unwrap();
            }
            let word = |byte: u64| u32::try_from(byte / 4).unwrap();
            let bases = Bases {
                status: word(at + 384),
                descs: word(at),
                params: word(at + 128),
                offs: word(at + 256),
                cfg: word(at + 320),
                heap: word(heap_at),
            };
            let record = StepUniform::default().bytes(bases);
            segment.write(at + 512, &record).unwrap();
            (bases, at + 512)
        };
        // A's `o` at its heap's end, on B's first words; B's inside its own.
        let (_, a_record) = lay(0, u32::try_from(HEAP).unwrap(), 4096);
        let (_, b_record) = lay(1024, 16, 4096 + HEAP);
        let values = |x: f32| [x, x + 1.0, x + 2.0, x + 3.0].map(f32::to_bits);
        let mut heaps = vec![SENTINEL; 2 * HEAP as usize / 4 + 16];
        heaps[..4].copy_from_slice(&values(1.0));
        heaps[16..20].copy_from_slice(&values(5.0));
        segment.write(4096, &words(&heaps)).unwrap();

        let read = |at: u64, n: usize| -> Vec<u32> {
            let mut bytes = vec![0u8; n * 4];
            segment.read(at, &mut bytes).unwrap();
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        let b_before = read(1024, 256);
        let group = library.group(&segment, &segment).unwrap();
        let frame = device.frame().unwrap();
        for record in [a_record, b_record] {
            frame
                .dispatch_at(
                    "neighbours test",
                    &pipeline,
                    group,
                    &[u32::try_from(record).unwrap()],
                    [1, 1, 1],
                )
                .unwrap();
        }
        frame.commit().unwrap();

        let a_status = read(384, 4);
        assert_eq!(a_status[0], 3, "A faulted: {a_status:x?}");
        assert_eq!(
            a_status[1],
            eta_compiler::codegen::fault::OUT_OF_BOUNDS,
            "{a_status:x?}"
        );
        assert_eq!(
            read(1024, 256),
            b_before,
            "B's tables, status and record stand"
        );
        let heaps = read(4096, 2 * HEAP as usize / 4);
        // A's `a` and B's `a` stand; B's copy landed in its own heap.
        assert_eq!(heaps[..4], values(1.0), "{heaps:x?}");
        assert_eq!(
            heaps[16..24],
            [values(5.0), values(5.0)].concat(),
            "{heaps:x?}"
        );
        assert!(
            heaps[4..16]
                .iter()
                .chain(&heaps[24..])
                .all(|&w| w == SENTINEL),
            "nothing else written: {heaps:x?}"
        );
    }

    /// Guest bounds in the one-segment form ([`super::Mirrors`]): the tables
    /// read through the heap's binding, the status and records in their
    /// mirrors. A's heap ends where B's tables start; A copies past its
    /// heap's end onto B's first table word. A faults OUT_OF_BOUNDS and
    /// writes nothing: B's tables stand byte-identical, and B runs clean,
    /// its copy landing in its own heap.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_guest_write_past_its_heap_in_the_one_segment_form_leaves_every_table_word() {
        use super::{Library, Mirrors};
        use crate::device::{Buffer, Context, present};
        use eta_compiler::codegen::steps::{Bases, StepUniform};
        if !present() {
            return;
        }
        let allowance = engine::Allowance {
            budget: Some(1 << 30),
        };
        let device = Context::bind(&crate::api::DeviceBoot::default(), allowance).unwrap();
        let owner = engine_gpu::books::owners::TEST;
        let outbox = Buffer::with(
            &device,
            4096,
            crate::device::alloc::Memory::Readback,
            engine_gpu::books::owners::GUEST,
        )
        .unwrap();
        let mut segment = Buffer::zeroed(&device, 8192, owner).unwrap();
        let mirrors = Mirrors::new(&device).unwrap();
        let library = Library::build(
            &device,
            super::Pool::new(vec![outbox]),
            &[&segment],
            Some(mirrors),
        )
        .unwrap();
        let call = eta_compiler::codegen::wgsl::step_calls()
            .iter()
            .position(|text| text.contains("op_copy("))
            .expect("a copy step");
        let pipeline = library.pipeline(u32::try_from(call).unwrap()).unwrap();
        let mirrors = library.mirrors.as_ref().expect("the one-segment form");

        let words =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        const HEAP: u64 = 64;
        const SENTINEL: u32 = 0xdead_beef;
        let f32_tag =
            u32::from(eta_ir::types::to_wire(eta_ir::Dtype::F32).expect("f32 has a wire tag"));
        let desc = |len: u32| [len, 1, len, 1, f32_tag, len, 1, 1, 1];
        let heap_words = u32::try_from(HEAP / 4).unwrap();
        let word = |byte: u64| u32::try_from(byte / 4).unwrap();
        // An instance's tables from `at`; its status word `status` bytes
        // into the status mirror, its record `record` into the records'.
        let mut lay = |at: u64, o_at: u32, heap_at: u64, status: u64, record: u64| {
            let mut op = [0u32; 16];
            op[1] = 0; // a
            op[4] = 1; // o
            for (from, bytes) in [
                (at, words(&[desc(4), desc(4)].concat())),
                (at + 128, words(&op)),
                (at + 256, words(&[0, o_at])),
                (
                    at + 320,
                    eta_exec::tables::Cfg {
                        value_count: 2,
                        heap_words,
                        op_count: 1,
                        ..eta_exec::tables::Cfg::default()
                    }
                    .to_le_bytes(),
                ),
            ] {
                segment.write(from, &bytes).unwrap();
            }
            let bases = Bases {
                status: word(status),
                descs: word(at),
                params: word(at + 128),
                offs: word(at + 256),
                cfg: word(at + 320),
                heap: word(heap_at),
            };
            mirrors
                .records
                .write_shared(record, &StepUniform::default().bytes(bases))
                .unwrap();
        };
        // A's heap is the 64 bytes under B's tables; its `o` at its end is
        // B's first descriptor word. B's heap and `o` are its own.
        lay(0, u32::try_from(HEAP).unwrap(), 1024 - HEAP, 0, 0);
        lay(1024, 16, 4096, 256, 256);
        let values = |x: f32| [x, x + 1.0, x + 2.0, x + 3.0].map(f32::to_bits);
        segment.write(1024 - HEAP, &words(&values(1.0))).unwrap();
        let mut heap = vec![SENTINEL; HEAP as usize / 4];
        heap[..4].copy_from_slice(&values(5.0));
        segment.write(4096, &words(&heap)).unwrap();
        mirrors.status.write_shared(0, &[0u8; 512]).unwrap();

        let read = |buffer: &Buffer, at: u64, n: usize| -> Vec<u32> {
            let mut bytes = vec![0u8; n * 4];
            buffer.read(at, &mut bytes).unwrap();
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect()
        };
        let b_tables = read(&segment, 1024, 96);
        let group = mirrors.group.as_ref().expect("bound at the build");
        let frame = device.frame().unwrap();
        for record in [0u32, 256] {
            frame
                .dispatch_at(
                    "one-segment bounds test",
                    &pipeline,
                    group,
                    &[record],
                    [1, 1, 1],
                )
                .unwrap();
        }
        frame.commit().unwrap();

        let a_status = read(&mirrors.status, 0, 4);
        assert_eq!(a_status[0], 3, "A faulted: {a_status:x?}");
        assert_eq!(
            a_status[1],
            eta_compiler::codegen::fault::OUT_OF_BOUNDS,
            "{a_status:x?}"
        );
        assert_eq!(read(&mirrors.status, 256, 4)[0], 0, "B ran clean");
        assert_eq!(
            read(&segment, 1024, 96),
            b_tables,
            "B's tables stand: A wrote no table word"
        );
        let heap = read(&segment, 4096, HEAP as usize / 4);
        assert_eq!(heap[..8], [values(5.0), values(5.0)].concat(), "{heap:x?}");
        assert!(heap[8..].iter().all(|&w| w == SENTINEL), "{heap:x?}");
    }
}
