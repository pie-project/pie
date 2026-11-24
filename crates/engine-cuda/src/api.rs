#![allow(unsafe_code)]
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use engine::transfer::{KvCopy, MemoryDomain};
use eta_ir::registry::{GeometryClass, Port, PortMask};
use poem_compiler::{Budget, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Knobs, Lane, Seated, Shell};

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct World {
    pub rank: u32,
    pub size: u32,
}

impl Default for World {
    fn default() -> World {
        World { rank: 0, size: 1 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceBoot {
    pub ordinal: i32,
    pub world: World,
    pub comm: Option<Arc<crate::comm::Comm>>,
    pub knobs: Knobs,
    pub cache_dir: Option<std::path::PathBuf>,
    pub adapter_dir: Option<std::path::PathBuf>,
}

pub(crate) struct PendingStep {
    readout: Vec<Readout>,
    settled: crate::serve::Settled,
}

pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,
    channels: BTreeMap<ChannelId, Arc<crate::program::Endpoint>>,
    /// The channels each live instance was bound on (`binding.channels`).
    attached: BTreeMap<InstanceId, Vec<ChannelId>>,
    /// Channels closed while an instance was still bound on them: kept
    /// registered, slots and all, so a rebind of their ring is served,
    /// until the last of those instances closes ([`Engine::close_channel`]).
    closing: std::collections::BTreeSet<ChannelId>,
    sink: Option<engine::CompletionSink>,
    tap_sink: Option<engine::TapSink>,
    /// Where a tier copy still landing on the swap stream is told landed.
    landed: Option<engine::LandingSink>,
    pending: Option<(FrameId, Vec<PendingStep>)>,
    adapters: BTreeMap<InstanceId, crate::Binding>,
    replay: Option<crate::replay::Replay>,
    /// This rank's place in its expert-parallel group (`size` 1 outside
    /// one): which experts' planes its banks load.
    expert: World,
    /// Run between a load's staging and its arming: an expert-parallel
    /// group wires its ranks there, since arming fires their collectives.
    before_arming: Option<BeforeArming>,
}

pub(crate) type BeforeArming = Box<dyn FnMut(&mut Shell) -> std::result::Result<(), String> + Send>;

impl Cuda {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Cuda {
        crate::serve::diag::publish(&boot.knobs.diagnostics);
        Cuda {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            channels: BTreeMap::new(),
            attached: BTreeMap::new(),
            closing: std::collections::BTreeSet::new(),
            sink: None,
            tap_sink: None,
            landed: None,
            pending: None,
            adapters: BTreeMap::new(),
            replay: None,
            expert: World::default(),
            before_arming: None,
        }
    }

    pub(crate) fn before_arming(&mut self, hook: BeforeArming) {
        self.before_arming = Some(hook);
    }

    /// Places this engine at `rank` of an expert-parallel group of `size`,
    /// before it loads.
    pub(crate) fn set_expert_rank(&mut self, rank: u32, size: u32) {
        self.expert = World { rank, size };
    }

    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    #[must_use]
    pub fn rank(&self) -> u32 {
        self.boot.world.rank
    }

    /// Whether a live instance is bound on channel `id`.
    fn bound_on(&self, id: ChannelId) -> bool {
        self.attached
            .values()
            .any(|channels| channels.contains(&id))
    }

    /// Whether channel `id` is registered on this engine: open, or closed
    /// with an instance still bound on it.
    #[must_use]
    pub fn has_channel(&self, id: ChannelId) -> bool {
        self.channels.contains_key(&id)
    }

    pub fn endpoint(&self, id: ChannelId) -> Option<Arc<crate::program::Endpoint>> {
        self.channels.get(&id).cloned()
    }

    pub fn adopt_channel(
        &mut self,
        id: ChannelId,
        endpoint: Arc<crate::program::Endpoint>,
    ) -> EngineResult<RegisteredChannel> {
        if self.channels.contains_key(&id) {
            return Err(Error::Program(format!(
                "channel {id} is already registered on this engine"
            )));
        }
        self.channels.insert(id, endpoint);
        Ok(RegisteredChannel {
            id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id: id.0,
            })
    }

    fn settle(
        &self,
        trace: &Trace,
        path: &Path,
    ) -> EngineResult<(ModelContract, crate::experts::Plan)> {
        let contract = (self.contract_for)(trace, path).map_err(Error::Load)?;
        let prospect =
            crate::weights::prospect(trace, &contract, path, self.target()).map_err(fault)?;
        // Dense weights always live on the device. The weight cache streams
        // the routed banks it takes an expert at a time: the plan seats none
        // itself.
        let cached = if self.boot.knobs.dense_experts() {
            Vec::new()
        } else {
            crate::store::layout::streamed_experts(trace)
        };
        let ranking = prospect.ranking.clone().caching(&cached);
        Ok((contract, crate::experts::Plan::cut(&ranking)))
    }

    fn target(&self) -> checkpoint::plan::StorageTarget {
        checkpoint::plan::StorageTarget {
            ep_rank: self.expert.rank,
            ..checkpoint::plan::StorageTarget::for_backend(
                checkpoint::types::BackendKind::Cuda,
                self.boot.world.rank,
                self.boot.world.size,
            )
        }
    }

    /// A frame an expert-parallel rank brings to its group: validated, its
    /// id minted, and the token bucket each of its steps needs. A step the
    /// replay would split fires twice, which the group's other ranks would
    /// not match: refused.
    pub(crate) fn lockstep_frame(
        &mut self,
        frame: &FrameSubmission,
    ) -> EngineResult<(FrameId, Vec<u32>)> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::Serves::NONE, engine::Serves::of),
        )?;
        // The runtime's stamp of each step's stated shape, cross-checked
        // against this load's stated set in debug builds (CONTRACT-BC 24).
        if let Some(caps) = self.caps.as_ref() {
            frame.debug_check_shapes(&caps.steps);
        }
        self.validate_captures(frame)?;
        if let Some((replay, shell)) = self.replay.zip(self.shell.as_ref())
            && frame.steps.iter().any(|step| {
                replay
                    .split(step, |word| {
                        shell.trace().facts.plain(word) && !shell.wide(word)
                    })
                    .is_some()
            })
        {
            return Err(Error::Unsupported {
                verb: "an encoder replay split under expert parallelism",
                engine: "cuda",
            });
        }
        let shell = self.loaded_mut()?;
        let buckets = frame
            .steps
            .iter()
            .map(|step| shell.bucket_for(step.rows()))
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(|| Error::Invalid("a step wider than the widest bucket".into()))?;
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);
        self.pending = None;
        Ok((id, buckets))
    }

    /// Fires `step` at its group's `bucket`.
    pub(crate) fn fire_step_at(
        &mut self,
        step: &Step,
        at: engine::StepDone,
        bucket: u32,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        self.loaded_mut()?.at_bucket(bucket);
        self.fire_step(step, at, true)
    }

    /// Fires a step with no lanes at the group's `bucket`.
    pub(crate) fn fire_idle(&mut self, bucket: u32) -> EngineResult<()> {
        self.loaded_mut()?.fire_idle(bucket).map_err(fault)
    }

    /// The steps of frame `id` its group fired, to settle as a frame.
    pub(crate) fn adopt(&mut self, id: FrameId, settled: Vec<PendingStep>) {
        self.pending = Some((id, settled));
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the cuda engine has no model loaded".into()))
    }
}

fn fault(fault: Fault) -> Error {
    // A fault one lane earns refuses that lane alone; the step's other
    // lanes are the runtime's to submit again (`engine::Error::Lanes`).
    if let Some(lane) = fault.lane() {
        return Error::lane(lane, fault);
    }
    // A program overreaching its attachment refuses every lane it spans.
    if let Fault::Overreach {
        instance, lanes, ..
    } = &fault
    {
        let reason = fault.to_string();
        return Error::Lanes(
            lanes
                .clone()
                .map(|lane| engine::LaneFault {
                    lane,
                    instance: Some(*instance),
                    reason: reason.clone(),
                })
                .collect(),
        );
    }
    match fault {
        Fault::Runtimeless | Fault::Device { .. } | Fault::Unsupported(_) => {
            Error::Device(fault.to_string())
        }
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Param { .. }
        | Fault::Unbound { .. }
        | Fault::Unlowered { .. }
        | Fault::Region { .. }
        | Fault::Golden { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::OutOfMemory { need, have } => Error::Exhausted {
            resource: "device memory",
            wanted: need,
            available: have,
        },
        Fault::Residency(_) | Fault::Card(_) => Error::Impossible(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Fragmented { .. } | Fault::Integrity { .. } => Error::Device(fault.to_string()),
        Fault::TierFull(_) => Error::TierFull(fault.to_string()),
        Fault::Overcarved { .. } => Error::Invalid(fault.to_string()),
        Fault::Straddled { .. } | Fault::Recipe(_) => Error::Load(fault.to_string()),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskUnread { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapterless { .. }
        | Fault::AdapterWord { .. }
        | Fault::KvWrite { .. }
        | Fault::Overreach { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Claimed(refusal) => Error::Claimed(refusal),
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) | Fault::PatchPayload { .. } | Fault::VoxelPayload { .. } => {
            Error::Invalid(fault.to_string())
        }
    }
}

fn extents(stated: &BindExtents) -> eta_exec::Extents {
    eta_exec::Extents {
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The backend the host emits fused regions for: the HIP dialect (wave64)
/// on ROCm.
#[cfg(not(feature = "rocm"))]
pub const CODEGEN: &str = "cuda";
#[cfg(feature = "rocm")]
pub const CODEGEN: &str = "hip";

/// Powers of two from the floor up to `max_tokens`, with `steps - 1` more
/// between each pair; from `fine_from` rows (0 never), at least `fine_steps`
/// a doubling. A prefill's GEMMs run its bucket's
/// rows, so a 2K prompt padded to 4096 costs a second prompt, while buckets
/// this tall are few.
#[must_use]
pub(crate) fn default_lattice(
    max_tokens: u32,
    steps: u32,
    fine_from: u32,
    fine_steps: u32,
) -> Vec<u32> {
    let steps = steps.max(1);
    let mut lattice: Vec<u32> = Vec::new();
    let push = |point: f64, lattice: &mut Vec<u32>| -> bool {
        if point >= f64::from(max_tokens) {
            return false;
        }
        let point = point as u32;
        if lattice.last() != Some(&point) {
            lattice.push(point);
        }
        true
    };
    for i in 0u32.. {
        let point = (f64::from(LATTICE_FLOOR) * 2f64.powf(f64::from(i) / f64::from(steps))).ceil();
        if fine_from > 0 && point >= f64::from(fine_from) {
            break;
        }
        if !push(point, &mut lattice) {
            lattice.push(max_tokens);
            return lattice;
        }
    }
    let fine = steps.max(fine_steps);
    for i in 0u32.. {
        let point = (f64::from(fine_from) * 2f64.powf(f64::from(i) / f64::from(fine))).ceil();
        if !push(point, &mut lattice) {
            break;
        }
    }
    lattice.push(max_tokens);
    lattice
}

pub(crate) const LATTICE_FLOOR: u32 = 1;

#[must_use]
pub(crate) fn lattice(
    stated: Vec<u32>,
    max_tokens: u32,
    steps: u32,
    fine_from: u32,
    fine_steps: u32,
) -> Vec<u32> {
    if stated.is_empty() {
        default_lattice(max_tokens, steps, fine_from, fine_steps)
    } else {
        stated
    }
}

/// The rows a patch ladder is derived from when a load states no tokens.
pub(crate) const DEFAULT_FORWARD_TOKENS: u32 = 8192;

/// The rung lattice under `engine` up to `top`: [`default_lattice`]'s
/// points below it and `top` itself. With `top` the load's context, the
/// candidates a load with no stated buckets weighs its fire top among.
#[must_use]
pub(crate) fn rung_lattice(engine: &crate::chip::Engine, top: u32) -> Vec<u32> {
    default_lattice(
        top,
        engine.lattice_steps,
        engine.prefill_lattice_from,
        engine.prefill_lattice_steps,
    )
}

/// The widest rung of [`rung_lattice`] to `top` at or below `tokens` (its
/// first where none is): a load asked other tokens (the walk fit,
/// `engine.fire_rows`, a stated `max_forward_tokens`, a refit) lays no last
/// point off the grid.
#[must_use]
pub(crate) fn on_lattice(engine: &crate::chip::Engine, top: u32, tokens: u32) -> u32 {
    let lattice = rung_lattice(engine, top);
    lattice
        .iter()
        .rev()
        .copied()
        .find(|&rung| rung <= tokens)
        .unwrap_or(lattice[0])
}

/// The load's budgets for `trace`: its logit readouts sized to what the
/// trace drafts, not the general 16 a lane, so a vocab-wide head carves only
/// the rows a fire can read out.
pub(crate) fn bake_budgets(budgets: &LoadBudgets, trace: &Trace) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::readouts_per_lane(trace),
    }
}

pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    const DERIVED_PATCH_CEILING: u32 = 4096;

    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, poem_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(poem_ir::RowAxis::Patches))
    });
    if !declares_patches {
        return None;
    }

    let max_patches = budgets
        .max_patches
        .unwrap_or_else(|| {
            let tokens = match budgets.max_tokens {
                0 => DEFAULT_FORWARD_TOKENS,
                tokens => tokens,
            };
            tokens.min(DERIVED_PATCH_CEILING)
        })
        .max(PATCH_LATTICE_FLOOR);
    let mut buckets = Vec::new();
    let mut rung = PATCH_LATTICE_FLOOR;
    while rung < max_patches {
        buckets.push(rung);
        rung = rung.saturating_mul(2);
    }
    buckets.push(max_patches);
    Some(PatchLadder {
        max_images: budgets
            .max_images
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

fn adapter_of(
    shell: &mut Shell,
    program: engine::program::ProgramId,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::Binding>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        let wire = seeds
            .iter()
            .find(|(seeded, _)| seeded == channel)
            .map(|(_, bytes)| bytes.as_slice())
            .ok_or_else(|| {
                Error::Load(format!(
                    "this program's `lora` sink reads its `{}` plane out of channel \
                     {channel} and this bind seeded nothing into it; an adapter's \
                     weights are the seed, because the fire path never reads the cell \
                     (alto adapter §6.1), so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<crate::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| crate::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .bind_adapter(crate::AdapterSource::Own {
            instance,
            planes: &planes,
        })
        .map(Some)
        .map_err(fault)
}

/// A registered channel's cell bytes, as its ring holds them, and its
/// capacity.
fn cell_of(registration: &ChannelRegistration) -> EngineResult<(u32, u32)> {
    let numel = registration
        .shape
        .iter()
        .map(|&dim| dim as usize)
        .product::<usize>()
        .max(1);
    let dtype = registration.dtype.program_dtype();
    let bytes = if registration.host_role == eta_ir::container::HostRole::None {
        crate::program::launch::native_cell_bytes(dtype, numel)
    } else {
        eta_exec::wire_cell_bytes(dtype, numel)
    };
    let bytes = u32::try_from(bytes).map_err(|_| {
        Error::Program(format!(
            "channel {}'s cell is wider than a u32 counts",
            registration.id
        ))
    })?;
    Ok((bytes, registration.capacity.max(1)))
}

impl Engine for Cuda {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn cover_fingerprint(&self) -> Option<String> {
        let print = self.shell.as_ref()?.cover_fingerprint();
        tracing::info!(
            "the host cover's key: {print} (the device, its kernels' code, the \
             guest emitter's version, the decode products' picks and a fire's lanes and tokens)"
        );
        Some(print)
    }

    fn committed_covers(&self) -> &'static [&'static str] {
        crate::chip::COVERS
    }

    fn measured_at_load(&self) -> Vec<String> {
        self.shell
            .as_ref()
            .map(|shell| shell.measured_at_load().to_vec())
            .unwrap_or_default()
    }

    fn skips_measuring(&self) -> bool {
        self.boot.knobs.measuring == engine::rates::Measuring::Skip
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn export_kv_handle(&self) -> Option<engine::KvHandle> {
        self.caps.as_ref()?.kv_handle.clone()
    }

    fn live(&self) -> engine::live::Live {
        self.shell
            .as_ref()
            .map_or_else(engine::live::Live::armed, Shell::live)
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this cuda engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight,
            steps,
        } = request;

        // A rank of an expert-parallel group of `size` holds part of a bank
        // only in a trace lowered for that many ranks.
        let size = self.expert.size;
        if size > 1
            && !trace
                .params
                .iter()
                .any(|param| param.experts.is_some_and(|slots| slots.ep == size))
        {
            return Err(Error::Load(format!(
                "this engine is a rank of an expert-parallel group of {size}, and its trace \
                 holds no bank lowered for {size} ranks: load the deployment's `-ep{size}` trace"
            )));
        }
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        refuse_an_artifact_for_another_deployment(
            &path,
            trace.platform.backend(),
            width_free(&trace.name),
        )?;
        let (contract, plan) = self.settle(&trace, &path)?;
        let kv_budget = match (residency.device_kv_budget, residency.device_kv_tokens) {
            (Some(_), Some(_)) => {
                return Err(Error::Load(
                    "the load states the device pool both in bytes (`device_kv_budget`) and \
                     in kv tokens (`[engine.memory] kv_device_tokens`); state one"
                        .into(),
                ));
            }
            (Some(bytes), None) => Some(crate::store::KvSize::Bytes(bytes)),
            (None, Some(tokens)) => Some(crate::store::KvSize::Tokens(tokens)),
            (None, None) => None,
        };
        // Every rank on the host seats its own tier: a tensor-parallel
        // group's and each expert-parallel rank's.
        let ranks = self.boot.world.size * self.expert.size;
        // On a device whose memory is the host's (`cudaDevAttrIntegrated`)
        // no host tier is seated, nor its staging ring pinned: suspended kv
        // goes to the disk, and a stated tier is refused by name.
        let bound = if ordinal >= 0 {
            ordinal
        } else {
            self.boot.ordinal
        };
        let memory = crate::device::card::whose_memory(bound).map_err(fault)?;
        let host_tier = memory
            .host_tier(residency.host_kv_budget)
            .map_err(|why| Error::Load(format!("engine-cuda: {why}")))?;
        // The host tier's staging ring is page-locked here, before anything
        // measures the device, so what it holds is counted gone before the
        // device's free bytes are read. A ring that cannot be pinned refuses
        // the load, here and not after ready.
        let frames = engine::runahead::Runahead::of(frames_in_flight).frames_in_flight;
        let ring = match host_tier {
            engine::HostTier::Absent => None,
            engine::HostTier::Seated(stated) => crate::store::staging_ring_for(stated, ranks, frames)
                .and_then(|size| size.map(crate::store::Ring::pin).transpose())
                .map_err(fault)?,
        };
        if let Some(ring) = &ring {
            let size = ring.size();
            tracing::info!(
                "the host tier's staging ring: {} MiB pinned, {} pieces of {} MiB (a \
                 piece for each of 2 stages for each of {frames} frames in flight)",
                size.bytes() >> 20,
                size.pieces,
                size.piece >> 20
            );
        }
        // A declared context is cut to the load's room where the load reads
        // it, past this process's context and modules (`Shell::stage`).
        let patches = patch_ladder(&trace, &budgets);
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let budget = bake_budgets(&budgets, &trace);
        let mut staged = Shell::stage(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget,
            patches,
            voxels,
            page_size: budgets.page_size,
            context: budgets.max_context,
            context_declared: budgets.context_declared && kv_budget.is_none(),
            slots: budgets.slots,
            kv_budget,
            allowance: residency.device,
            ordinal: bound,
            knobs: crate::Knobs {
                weights_held: residency.weights_held,
                ..self.boot.knobs.clone()
            },
            cache_dir: self.boot.cache_dir.as_deref(),
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            residency: plan,
            world: self.boot.world,
            expert: self.expert,
            comm: self
                .boot
                .comm
                .as_ref()
                .map_or(core::ptr::null(), Arc::as_ptr),
            steps: steps.clone(),
        })
        .map_err(fault)?;
        if let Some(hook) = self.before_arming.as_mut() {
            hook(&mut staged.shell).map_err(Error::Load)?;
        }
        // Another worker reads the kv pages in place: the pool's chunks lie
        // in one run from its base.
        if residency.kv_export {
            staged.shell.keep_contiguous();
        }
        let mut shell = staged.finish().map_err(fault)?;

        shell.mount_adapters(self.boot.adapter_dir.clone());
        // The context the load serves, a declared one as the load's room cut it.
        let context = shell.paging().context();
        if let engine::HostTier::Seated(stated) = host_tier {
            shell
                .seat_host_kv(
                    stated,
                    residency.kv_seats.unwrap_or(budgets.max_lanes),
                    context,
                    ranks,
                    ring,
                )
                .map_err(fault)?;
        }
        let kv_handle = if residency.kv_export {
            Some(shell.export_kv(self.boot.world).map_err(fault)?)
        } else {
            None
        };
        let disk_pages = shell
            .seat_disk_kv(
                residency.disk_kv_budget,
                residency.disk_kv_dir.as_deref(),
                residency.disk_kv_persistent,
                residency.kv_seats.unwrap_or(budgets.max_lanes),
                context,
                self.boot.world,
            )
            .map_err(fault)?;
        if memory.unified() {
            tracing::info!(
                "{} ({} disk kv pages)",
                engine::DeviceMemory::NO_HOST_TIER,
                disk_pages
            );
        }

        let mut chunk_facts = shell.chunk_facts().map_err(fault)?;
        chunk_facts.record_bytes = shell.record_bytes();
        chunk_facts.windowed = shell.windowed_layers();
        let windowed_state = !chunk_facts.windowed.is_empty();
        chunk_facts.disk_persistent = residency.disk_kv_persistent && chunk_facts.disk_chunks > 0;
        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let (pool_committed_bytes, pool_high_water_bytes) = shell.elastic();
        let weights_from_cache = shell.weights_from_cache();
        let weights_resident = shell.weights_resident();
        let paging = shell.paging();
        let state_rows = shell
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, poem_ir::CacheRow::State { .. }));
        let trace = shell.trace();
        let drafter = trace.drafter;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cuda".to_string(),
                domain: MemoryDomain::CudaDevice(u32::try_from(shell.ordinal()).unwrap_or(0)),
                sms: shell.sms(),
                unified_memory: memory.unified(),
                fp8_native: false,
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: Some(CODEGEN.to_string()),
                // its CUDA / HIP emitters serve every form
                library_forms: engine::LibraryForms::ALL,
            },
            pools: PoolFacts {
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                window_tokens: paging.window.map_or(0, |window| window.tokens),
            },
            chunks: chunk_facts,
            limits: FireLimits {
                max_lanes: shell.fire_lanes(),
                max_tokens: shell.budget().max_tokens,
                max_page_refs: paging
                    .pages_per_slot
                    .saturating_mul(shell.budget().max_lanes),
                max_context: paging.context(),
                prefill_rows: 0,
                prefill_lanes: shell.prefill_lanes(),
                max_readouts: u32::try_from(poem_compiler::arena::readouts_ceiling(shell.budget()))
                    .unwrap_or(u32::MAX),
                // Each lane's rung bounds what it replays.
                // A lane's replayed rows are spans of its buffer and a beam's
                // resolve through its group's list (`kernels_cuda::ssm::Carry`):
                // no table holds a row, so a fire replays what its lanes'
                // contexts hold.
                replayed_rows: shell.fire_lanes().saturating_mul(paging.context()),
            },
            has_lora: true,
            has_attn_page_mask: false,
            draft_block: drafter.map_or(0, |d| d.rows),
            draft_mask_token: drafter.map_or(0, |d| d.mask_token),
            draft_proposals_from: drafter.map_or(1, |d| d.proposals_from),
            ports: PortMask::DEVICE_GEOMETRY
                .with(Port::AttnMask)
                .with(Port::Fold)
                .with(Port::Parents),
            geometry: GeometryClass::DeviceGeometry,
            kv_handle,
            media_encode: false,
            rs_verbs: true,
            windowed_state,
            // A load of rings alone lands its snapshots itself
            // (`serve::ring`); one of recurrent state too, as its state
            // lands them.
            rs_snapshots: if (state_rows && shell.lands_snapshots())
                || (windowed_state && !state_rows)
            {
                crate::store::rs::SNAPSHOTS
            } else {
                0
            },
            bidirectional_attention: true,
            adapter_attachments: true,
            facts: shell.trace().facts.clone(),
            inputs: shell.inputs(),
            classes: shell.classes(),
            // CUDA states no bound on one submission (no watchdog on a
            // compute device).
            guest_cost: engine::GuestCost::default(),
            // A beam group's page list (`beam_list`), read at a search's finish.
            host_ops: engine::HostOps { beam_list: true },
            mixed_lanes: engine::MixedLanes::Any,
            frames_in_flight: u32::from(
                engine::runahead::Runahead::of(frames_in_flight).frames_in_flight,
            ),
            steps: engine::StatedSteps::default(),
        };
        // The set as the load states it, from its stated facts, with those
        // the arm walk left eager (CONTRACT-BC 24). It is the set the walk
        // took: a fire's stamped shape names the walk's shape.
        let stated = engine::StepShapes::of(
            &steps,
            &engine::StepFacts::stated(&caps, &readout_facts()),
        );
        if stated != *shell.steps() {
            return Err(Error::Load(format!(
                "the load states the step set {} and its arm walk took {}: a fire's stated \
                 shape would name another walk's shape",
                stated.census(),
                shell.steps().census()
            )));
        }
        let mut caps = caps;
        caps.steps = shell.stated_steps(stated);
        tracing::info!(
            "step set: {}; {} stated eager",
            caps.steps.shapes.census(),
            caps.steps.eager.len(),
        );

        // A fire's lanes, rows and readouts bound the scheduler's waves: a
        // decode wave seats at most `max_lanes` lanes.
        tracing::info!(
            "fire limits: {} lanes, {} rows, {} readouts",
            caps.limits.max_lanes, caps.limits.max_tokens, caps.limits.max_readouts,
        );
        self.replay = crate::replay::Replay::of(&shell);
        if let Some(sink) = &self.tap_sink {
            shell.on_tapped(sink.clone());
        }
        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                weights_resident,
                weights_from_cache,
                arena_bytes,
                pool_bytes,
                input_bytes,
                pool_committed_bytes,
                pool_high_water_bytes,
            },
            caps,
        })
    }

    fn register_adapter(
        &mut self,
        registration: &engine::adapter::AdapterRegistration,
    ) -> EngineResult<()> {
        let planes: Vec<crate::AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| crate::AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn commit_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.commit_segments(segments).map_err(fault)
    }

    fn release_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.release_segments(segments).map_err(fault)
    }

    fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> EngineResult<u64> {
        self.loaded_mut()?
            .weight_residency(residency)
            .map_err(fault)
    }

    fn weight_uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        self.loaded_mut()?.weight_uses().map_err(fault)
    }

    /// All or nothing: an `Err` moved nothing (`Shell::evacuate_with`).
    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        let endpoints: Vec<_> = self
            .channels
            .values()
            .filter(|endpoint| !endpoint.slots().is_empty())
            .cloned()
            .collect();
        let channels: Vec<(Vec<u32>, u64)> = endpoints
            .iter()
            .map(|endpoint| {
                (
                    endpoint.slots(),
                    crate::program::Endpoint::cell_span(endpoint.wire_bytes(), endpoint.cap1() - 1),
                )
            })
            .collect();
        let at = self
            .loaded_mut()?
            .evacuate_with(evacuation, &channels)
            .map_err(fault)?;
        for (endpoint, at) in endpoints.iter().zip(at) {
            endpoint.rebase(at);
        }
        Ok(())
    }

    fn chunk_table(&self) -> EngineResult<engine::chunk::ChunkTable> {
        self.shell
            .as_ref()
            .map(|shell| shell.chunk_table().clone())
            .ok_or_else(|| Error::Load("the cuda engine has no model loaded".into()))
    }

    fn program_facts(&self, program: ProgramId) -> EngineResult<engine::program::ProgramFacts> {
        match self.shell.as_ref() {
            Some(shell) => shell.program_facts(program).map_err(fault),
            None => Ok(engine::program::ProgramFacts::default()),
        }
    }

    fn readout_facts(&self) -> EngineResult<engine::ReadoutFacts> {
        Ok(readout_facts())
    }

    fn guest_state(
        &self,
        program: ProgramId,
        extents: &engine::program::BindExtents,
        // The class bound: the engine half states per-class arms.
        _class: engine::ports::GeometryClass,
    ) -> EngineResult<engine::GuestState> {
        match self.shell.as_ref() {
            Some(shell) => shell
                .guest_state(program, self::extents(extents))
                .map_err(fault),
            None => Ok(engine::GuestState::default()),
        }
    }

    /// Every registration: a device channel's ring in its `Channel` slots,
    /// a host one's in the mirror it publishes.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        true
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::program::RingFacts>> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(None);
        };
        if registration.host_role != eta_ir::container::HostRole::None {
            return Ok(None);
        }
        let (cell_bytes, capacity) = cell_of(registration)?;
        Ok(Some(engine::program::RingFacts {
            channel: 0,
            cell_bytes: u64::from(cell_bytes),
            capacity,
            footprint: shell.guest_footprint(
                engine::chunk::SlotRole::Channel,
                crate::program::Endpoint::cell_span(cell_bytes, capacity),
            ),
        }))
    }

    fn rebind_slots(&mut self, rebind: engine::program::Rebind) -> EngineResult<()> {
        match rebind {
            engine::program::Rebind::Instance {
                id,
                instance,
                rings,
                scratch,
                pair,
                pair_generation,
                ..
            } => self
                .loaded_mut()?
                .rebind_program(
                    id,
                    crate::program::Slots {
                        instance,
                        rings,
                        scratch,
                        pair,
                        pair_generation,
                    },
                )
                .map_err(fault),
            engine::program::Rebind::Parked { id, kept } => {
                self.loaded_mut()?.park_program(id, kept).map_err(fault)
            }
            engine::program::Rebind::Channel { id, slots } => {
                let old = self.channels.get(&id).cloned().ok_or_else(|| {
                    Error::Program(format!("channel {id} is not registered on this engine"))
                })?;
                let shell = self.loaded_mut()?;
                let at = shell
                    .guest_span(
                        engine::chunk::SlotRole::Channel,
                        &slots,
                        crate::program::Endpoint::cell_span(old.wire_bytes(), old.cap1() - 1),
                    )
                    .map_err(fault)?;
                old.rebind(at, slots);
                Ok(())
            }
        }
    }

    /// Waits, on the host, for every move between the device and the host
    /// tier still landing; every other copy lands inside the call or in the
    /// main stream's order, which a later fire or move stands behind.
    fn settle_copies(&mut self) -> EngineResult<()> {
        self.loaded_mut()?.settle_moves().map_err(fault)
    }

    /// A move from the device to the host tier rides the swap stream and is
    /// still reading its device slots when the call returns: told landed
    /// once the stream has run it, by a waiter on an event recorded behind
    /// it (`serve::landing`), which enqueues nothing a later move would wait
    /// on. Every other copy's sources are free at once: a move from the host
    /// reads host slots, which the next move into them (on the same swap
    /// stream) or a host write (behind every move still landing) reaches
    /// only after it; the slot file's moves read inside the call; a move
    /// within the device runs on the main stream, ahead of everything after.
    fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        token: u64,
    ) -> EngineResult<bool> {
        let host = |domain| matches!(domain, MemoryDomain::HostPinned);
        let device = |domain| matches!(domain, MemoryDomain::CudaDevice(_));
        let disk = |domain| matches!(domain, MemoryDomain::LocalDisk);
        // A move between the host tier and the slot file runs on the tier's
        // landing thread and is told how it ended: the engine thread only
        // enqueues it.
        let host_disk = (host(copy.src) && disk(copy.dst)) || (disk(copy.src) && host(copy.dst));
        if host_disk && let Some(sink) = self.landed.clone() {
            let told: crate::serve::landing::Told =
                Box::new(move |landing| sink(token, landing));
            self.loaded_mut()?.defer_disk_tell(Some(told));
            let copied = self.copy_slots(copy);
            // Not taken (nothing to move, or refused before enqueueing):
            // answered in the call.
            let untaken = self.loaded_mut()?.defer_disk_tell(None).is_some();
            copied?;
            return Ok(untaken);
        }
        self.copy_slots(copy)?;
        let parked = device(copy.src) && host(copy.dst);
        let Some(sink) = self.landed.clone().filter(|_| parked) else {
            return Ok(true);
        };
        let told = self
            .loaded_mut()?
            .tell_swaps_landed(Box::new(move |landing| sink(token, landing)))
            .map_err(fault)?;
        Ok(!told)
    }

    fn on_landed(&mut self, sink: engine::LandingSink) {
        self.landed = Some(sink);
    }

    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        use crate::store::Tier;
        use crate::store::layout::{FIXED, LIST, RING, WINDOW};
        let tier = |domain| match domain {
            MemoryDomain::CudaDevice(_) => Some(Tier::Device),
            MemoryDomain::HostPinned => Some(Tier::Host),
            MemoryDomain::LocalDisk => Some(Tier::Disk),
            _ => None,
        };
        let (Some(from), Some(to)) = (tier(copy.src), tier(copy.dst)) else {
            return Err(Error::Unsupported {
                verb: "copy_slots to or from a domain other than the device, the host tier \
                       and the slot file",
                engine: "cuda",
            });
        };
        let shell = self.loaded_mut()?;
        shell.take_refs();
        if from != Tier::Device || to != Tier::Device {
            // Between tiers (CONTRACT-BC 22): whole slots, each laid alike.
            if !copy.runs.is_empty() {
                return Err(Error::Invalid(
                    "token runs move within the device; between tiers a slot moves whole".into(),
                ));
            }
            return shell
                .copy_tier_slots(copy.shape, from, to, &copy.slots)
                .map_err(fault);
        }
        if copy.shape == FIXED || copy.shape == RING {
            if !copy.runs.is_empty() {
                return Err(Error::Invalid(
                    "a fixed record is copied whole, not a run of its tokens".into(),
                ));
            }
            // A record is one unit across its shapes: named by any shape's
            // first piece, copied whole.
            for &(src, dst) in &copy.slots {
                shell
                    .fork_state(shell.record_key(src), shell.record_key(dst))
                    .map_err(fault)?;
            }
            return Ok(());
        }
        if copy.shape != LIST && copy.shape != WINDOW {
            // Any other shape this load states (the rs list): whole slots.
            if !copy.runs.is_empty() {
                return Err(Error::Invalid(format!(
                    "copy_slots names token runs of shape {}, whose slots are copied whole",
                    copy.shape
                )));
            }
            let pairs: Vec<[u32; 2]> = copy.slots.iter().map(|&(src, dst)| [src, dst]).collect();
            return shell.copy_whole_slots(copy.shape, &pairs).map_err(fault);
        }
        let page = shell.paging().page_size;
        let runs: Vec<engine::transfer::KvRun> = copy
            .slots
            .iter()
            .map(|&(src, dst)| engine::transfer::KvRun {
                src_page: src,
                src_token: 0,
                dst_page: dst,
                dst_token: 0,
                tokens: page,
            })
            .chain(copy.runs.iter().map(|run| engine::transfer::KvRun {
                src_page: run.src,
                src_token: run.src_token,
                dst_page: run.dst,
                dst_token: run.dst_token,
                tokens: run.tokens,
            }))
            .collect();
        shell
            .copy_token_slots(copy.shape == WINDOW, &runs)
            .map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        let submitted = self.submit_frame(frame);
        // A frame whose step refused its lanes by name fired the steps
        // before it alone: a refusal too.
        if submitted
            .as_ref()
            .is_ok_and(|ticket| !ticket.refused.is_empty())
        {
            engine::Engine::live(self).tally(engine::live::Tally::Refused, 1);
        }
        self.refused(submitted)
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let registered = self
            .loaded_mut()
            .and_then(|shell| shell.register_program(registration).map_err(fault));
        self.refused(registered)
    }

    fn prewarm_programs(&mut self, registrations: &[ProgramRegistration]) -> EngineResult<()> {
        self.loaded_mut()
            .and_then(|shell| shell.prewarm_programs(registrations).map_err(fault))
    }

    /// The program adopted and its build planned here; its regions compile
    /// (NVRTC, or the kernel cache's files) in the returned compile, off
    /// the engine's thread, while fires go on. Their modules load into the
    /// context at [`Engine::finish_program`].
    fn prepare_program(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let prepared = self
            .loaded_mut()
            .and_then(|shell| shell.prepare_program(registration).map_err(fault));
        let mut pending = self.refused(prepared)?;
        Ok(Box::new(move || {
            pending.build();
            engine::Prepared(Box::new(pending))
        }))
    }

    fn finish_program(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(pending) = prepared.0.downcast::<crate::program::Pending>() else {
            return Err(Error::Program(
                "a prepared program CUDA did not prepare".to_string(),
            ));
        };
        let registered = self
            .loaded_mut()
            .and_then(|shell| shell.finish_program(*pending).map_err(fault));
        self.refused(registered)
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        self.loaded_mut()?.close_program(program).map_err(fault)
    }

    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    /// One rank's planes hold a band of each layer's heads, so a
    /// tensor-parallel rank taps nothing.
    fn tap_groups(&self) -> Option<engine::TapGroups> {
        self.shell.as_ref()?.tap_groups()
    }

    fn on_tapped(&mut self, sink: engine::TapSink) {
        if let Some(shell) = self.shell.as_mut() {
            shell.on_tapped(sink.clone());
        }
        self.tap_sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already carved \
                 over its arena rectangles",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 the readout plane is one arena rectangle that every fire carves over. Ask \
                 for a frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut settled) = self.pending.take().expect("checked just above");
        let shell = self.loaded_mut()?;
        for step in &mut settled {
            shell
                .read_out_rows(&mut step.settled, &step.readout)
                .map_err(fault)?;
        }
        for (receipt, step) in ticket.steps.iter_mut().zip(&settled) {
            receipt.readouts = readouts_of(step);
        }
        self.pending = Some((ticket.id, settled));
        Ok(())
    }

    fn publish_landed(&mut self) -> EngineResult<()> {
        match self.shell.as_mut() {
            Some(shell) => shell.reap_landed_guests().map_err(fault),
            None => Ok(()),
        }
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        if self.channels.contains_key(&registration.id) {
            return Err(Error::Program(format!(
                "channel {} is already registered on this engine",
                registration.id
            )));
        }
        let device_only = registration.host_role == eta_ir::container::HostRole::None;
        let (cell_bytes, capacity) = cell_of(registration)?;
        let shell = self.loaded_mut()?;
        let stream = shell.stream();
        let cells = device_only
            .then(|| {
                shell
                    .guest_span(
                        engine::chunk::SlotRole::Channel,
                        &registration.slots,
                        crate::program::Endpoint::cell_span(cell_bytes, capacity),
                    )
                    .map(|at| (at, registration.slots.clone()))
            })
            .transpose()
            .map_err(fault)?;
        let endpoint = Arc::new(
            crate::program::Endpoint::open(
                registration.host_role,
                cell_bytes,
                capacity,
                cells,
                stream,
            )
            .map_err(fault)?,
        );
        let mirror = (!device_only).then(|| HostMirror {
            mirror: endpoint.mirror_host(),
            words: endpoint.words_host(),
            cell_bytes,
            capacity,
        });
        self.channels.insert(registration.id, endpoint);
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and \
                 not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }
        if let Some(id) = binding.channels.iter().find(|id| self.closing.contains(id)) {
            return Err(Error::Program(format!(
                "channel {id} is closed: it is kept only for the instances already bound \
                 on it, and a new instance cannot bind on it"
            )));
        }
        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let adopted: Vec<Option<Arc<crate::program::Endpoint>>> = binding
            .channels
            .iter()
            .map(|id| self.channels.get(id).cloned())
            .collect();
        let shell = self.loaded_mut()?;
        let id = shell
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
                binding.device_resolved,
                &adopted,
                &binding.channels,
                crate::program::Slots {
                    instance: binding.instance_slots.clone(),
                    rings: binding.ring_slots.clone(),
                    scratch: binding.scratch.clone(),
                    pair: binding.pair.clone(),
                    pair_generation: binding.pair_generation,
                },
            )
            .map_err(fault)?;
        let landed = adapter_of(shell, binding.program, id, &seeds);
        let bound = match landed {
            Ok(bound) => bound,
            Err(why) => {
                let _ = shell.close_program_instance(id);
                return Err(why);
            }
        };
        if let Some(bound) = bound {
            self.adapters.insert(id, bound);
        }
        if !binding.channels.is_empty() {
            self.attached.insert(id, binding.channels.clone());
        }
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<u64> {
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(held) = held {
            shell.release_adapter(held);
        }
        // CUDA lays its guests' tables in their pairs' state, which the
        // runtime holds: it keeps none standing.
        shell.close_program_instance(id).map_err(fault)?;
        // A closed channel this was the last instance on is released now.
        for channel in self.attached.remove(&id).unwrap_or_default() {
            if self.closing.contains(&channel) && !self.bound_on(channel) {
                self.closing.remove(&channel);
                self.channels.remove(&channel);
            }
        }
        Ok(0)
    }

    /// Closes channel `id`. With an instance still bound on it, the channel
    /// is only marked closing: its registration and slots stay, and a
    /// rebind of its ring is served, until the last such instance closes
    /// ([`Engine::close_channel`]).
    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        if self.bound_on(id) {
            self.closing.insert(id);
        } else {
            self.closing.remove(&id);
            self.channels.remove(&id);
        }
        Ok(())
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        let stream = self.loaded_mut()?.stream();
        self.instance(instance)?
            .publish(channel, cell, stream)
            .map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    fn beam_list(&mut self, instance: InstanceId, beam: u32) -> EngineResult<Vec<u32>> {
        self.loaded_mut()?.beam_list(instance, beam).map_err(fault)
    }

    fn host_kv_layout(&self) -> Option<state::PageLayout> {
        self.shell.as_ref()?.host_kv_layout(1)
    }

    fn read_host_kv(&mut self, slots: &[u32]) -> EngineResult<Vec<u8>> {
        self.loaded_mut()?.read_host_kv(slots).map_err(fault)
    }

    fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> EngineResult<()> {
        self.loaded_mut()?
            .write_host_kv(slots, bytes)
            .map_err(fault)
    }

    fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> EngineResult<Vec<u8>> {
        self.loaded_mut()?.read_tier_slots(slots).map_err(fault)
    }

    fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> EngineResult<()> {
        self.loaded_mut()?
            .write_tier_slots(slots, bytes)
            .map_err(fault)
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        // The host and disk tiers move through `copy_slots` (CONTRACT-BC
        // 22): a kv copy is within the device.
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if device.is_none_or(|device| copy.src != device || copy.dst != device) {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "cuda",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        // Windowed pages a write into a shared one copies first ride the
        // pre-launch copy beside the kv list's (CONTRACT-BC 19).
        let windowed: Vec<engine::transfer::KvRun> = copy
            .windowed
            .iter()
            .map(|&(src, dst)| engine::transfer::KvRun {
                src_page: src,
                src_token: 0,
                dst_page: dst,
                dst_token: 0,
                tokens: page_size,
            })
            .collect();
        self.loaded_mut()?
            .copy_kv_with(&moves, &windowed, !copy.moves.is_empty())
            .map_err(fault)
    }
}

impl Cuda {
    /// Refuses a step whose capturing lane no reading program is attached
    /// at ([`engine::fire::Step::validate_captures`]).
    fn validate_captures(&self, frame: &FrameSubmission) -> EngineResult<()> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(());
        };
        let facts = &shell.trace().facts;
        for step in &frame.steps {
            step.validate_captures(|word| facts.holds(word, poem_ir::Builtin::Scores))?;
        }
        Ok(())
    }

    /// `result`, its refusal counted ([`engine::live::Tally::Refused`]).
    fn refused<T>(&self, result: EngineResult<T>) -> EngineResult<T> {
        if result.is_err() {
            engine::Engine::live(self).tally(engine::live::Tally::Refused, 1);
        }
        result
    }

    fn submit_frame(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::Serves::NONE, engine::Serves::of),
        )?;
        // The runtime's stamp of each step's stated shape, cross-checked
        // against this load's stated set in debug builds (CONTRACT-BC 24).
        if let Some(caps) = self.caps.as_ref() {
            frame.debug_check_shapes(&caps.steps);
        }
        self.validate_captures(frame)?;
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut settled = Vec::with_capacity(frame.steps.len());
        let mut refused = None;
        for (index, step) in frame.steps.iter().enumerate() {
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            // A causal encoder-decoder's long prompts: their leading rows
            // through the encoder alone first, unannounced, then the step
            // with each such prompt's last window (`crate::replay`).
            let split = self
                .replay
                .zip(self.shell.as_ref())
                .and_then(|(replay, shell)| {
                    replay.split(step, |word| {
                        shell.trace().facts.plain(word) && !shell.wide(word)
                    })
                });
            let fired = match &split {
                // The head's lanes are the step's cut ones, in order.
                Some((head, _)) => self.fire_step(head, at, false).map(drop).map_err(|error| {
                    let cut = self
                        .replay
                        .map(|replay| {
                            replay.cut_lanes(step, |word| {
                                self.shell.as_ref().is_some_and(|shell| {
                                    shell.trace().facts.plain(word) && !shell.wide(word)
                                })
                            })
                        })
                        .unwrap_or_default();
                    of_lanes(error, &cut)
                }),
                None => Ok(()),
            };
            let step = split.as_ref().map_or(step, |(_, tail)| tail);
            let headed = split.is_some();
            let fired = fired.and_then(|()| {
                self.fire_step(step, at, true).map_err(|error| match error {
                    // Its head's encoder rows ran: the step is not untouched,
                    // so its lanes' refusal refuses the frame.
                    Error::Lanes(_) if headed => Error::Invalid(format!(
                        "{error}, after the step's encoder-only head fired"
                    )),
                    other => other,
                })
            });
            let (ticket, step_settled) = match fired {
                Ok(both) => both,
                // A step's lanes refused past the first step: the frame is
                // the steps before it (`engine::Error::Lanes`).
                Err(error) if error.stops_frame_at(index) => {
                    refused = Some(error);
                    break;
                }
                Err(error) => {
                    self.pending = None;
                    let airborne = self.shell.as_ref().map_or(0, Shell::airborne_steps);
                    return Err(attributed(error, airborne));
                }
            };
            steps.push(ticket);
            settled.push(step_settled);
        }
        self.pending = Some((id, settled));
        Ok(FrameTicket::of(id, steps, refused))
    }

    /// `submission` fired; its completion told the sink where `announce`.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
        announce: bool,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let id = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);

        let done = self
            .sink
            .as_ref()
            .filter(|_| announce)
            .map(|sink| crate::serve::Done {
                at,
                sink: std::sync::Arc::clone(sink),
            });
        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        if !self.adapters.is_empty() {
            for attachment in &submission.attachments {
                let Some(bound) = self.adapters.get(&attachment.instance) else {
                    continue;
                };
                // The attachment is its pass's lanes, `lanes` of them from `lane`.
                for at in attachment.lane..attachment.lane + attachment.lanes.max(1) {
                    if let Some(slot) = lane_adapters.get_mut(at as usize) {
                        *slot = Some(bound.slot);
                    }
                }
            }
        }
        let shell = self.loaded_mut()?;
        let mut words: Vec<u64> = submission.lanes.iter().map(|lane| lane.word).collect();
        for (at, slot) in lane_adapters.iter().enumerate() {
            if slot.is_none() {
                continue;
            }
            words[at] = shell.adapted_word(words[at]).ok_or_else(|| {
                Error::lane(u32::try_from(at).unwrap_or(u32::MAX), format!(
                    "lane {at} is attached to an instance that bound an adapter, and \
                     this load's model text has no corrected class for its fact word \
                     {:#x}: the text declares no `linear.lora_correct` arm, or the arm's \
                     window is not decided by one fact. A lane that asked for a \
                     correction and got the base model is the one wrong answer this \
                     axis refuses to give",
                    words[at]
                ))
            })?;
        }
        // A lane that names slot refs (`Lane::tables`, `Lane::fixed`) is the
        // runtime's shared pool speaking: its kv and window list tables are its
        // kv refs, and its recurrent record is the pairs `fixed` names, a
        // piece each, copied onto the slots the fire writes where the two
        // differ.
        use crate::store::layout::{FIXED, LIST, RING, WINDOW};
        let by_refs = submission.lanes.iter().any(|lane| {
            !lane.tables.is_empty()
                || !matches!(lane.record, engine::fire::LaneRecord::Unstated)
                || !lane.slot_copies.is_empty()
        });
        if by_refs {
            shell.take_refs();
        }
        // Every lane's slot copies land before any lane of the fire reads
        // or writes: a shared tail slot's rows onto the slot it then writes.
        for lane in &submission.lanes {
            for (shape, pairs) in &lane.slot_copies {
                shell.copy_whole_slots(*shape, pairs).map_err(fault)?;
            }
        }
        // Each lane's record: the one it reads and the one its fire writes.
        let mut records: Vec<Option<(u32, u32)>> = Vec::with_capacity(submission.lanes.len());
        // The lanes stated to carry no state: they read and write the null
        // record.
        let mut nulls: Vec<bool> = Vec::with_capacity(submission.lanes.len());
        let mut forked: Vec<(u32, u32)> = Vec::new();
        for lane in &submission.lanes {
            let pieces = match &lane.record {
                engine::fire::LaneRecord::Pieces(pieces) => pieces.as_slice(),
                engine::fire::LaneRecord::Null => &[],
                engine::fire::LaneRecord::Unstated if shell.null_record().is_some() => {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} states no record, and this load holds recurrent \
                         state: the runtime states every lane's record, or that it has none",
                        lane.slot
                    )));
                }
                engine::fire::LaneRecord::Unstated => &[],
            };
            nulls.push(matches!(lane.record, engine::fire::LaneRecord::Null));
            let record = match pieces {
                [] => None,
                // A record's pieces: its rings' first, then its state's
                // (CONTRACT-BC 21), named as one record by its first.
                stated
                    if stated.iter().all(|(shape, _)| *shape == FIXED || *shape == RING)
                        && stated.windows(2).all(|pair| pair[0].0 == RING && pair[1].0 == FIXED) =>
                {
                    let pieces: Vec<[u32; 2]> = stated
                        .iter()
                        .flat_map(|(_, pieces)| pieces.iter().copied())
                        .collect();
                    let runs: Vec<u32> = pieces.iter().map(|[run, _]| *run).collect();
                    let written: Vec<u32> = pieces.iter().map(|[_, written]| *written).collect();
                    let run = shell.name_record(&runs).map_err(fault)?;
                    let written = shell.name_record(&written).map_err(fault)?;
                    // The lanes of one working set name one pair: copied once.
                    if run != written && !forked.contains(&(run, written)) {
                        shell.fork_state(run, written).map_err(fault)?;
                        forked.push((run, written));
                    }
                    Some((run, written))
                }
                _ => {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} names fixed shapes this load does not state",
                        lane.slot
                    )));
                }
            };
            for snap in lane
                .rs_snapshots
                .iter()
                .filter(|snap| !snap.pieces.is_empty() || !snap.fixed.is_empty())
            {
                // A record of rings and state names its pieces a shape at a
                // time: the rings' first.
                let pieces: Vec<u32> = if snap.fixed.is_empty() {
                    snap.pieces.clone()
                } else {
                    snap.fixed
                        .iter()
                        .filter(|(shape, _)| *shape == RING)
                        .chain(snap.fixed.iter().filter(|(shape, _)| *shape == FIXED))
                        .flat_map(|(_, pieces)| pieces.iter().copied())
                        .collect()
                };
                let named = shell.name_record(&pieces).map_err(fault)?;
                if named != snap.slot {
                    return Err(Error::Invalid(format!(
                        "lane in slot {} snapshots into the record at slot ref {}, whose \
                         first piece it names {named}",
                        lane.slot, snap.slot
                    )));
                }
            }
            records.push(record);
        }
        // The lane's refs of `shape`: the kv list's, or a window list's.
        fn table(lane: &engine::fire::Lane, shape: engine::chunk::ShapeId) -> Option<&[u32]> {
            lane.tables
                .iter()
                .find(|(id, _)| *id == shape)
                .map(|(_, refs)| refs.as_slice())
        }
        // A lane's list kind says what its list table holds: its pages, or
        // the working set's mapped pages its program's page indexes address
        // (a device-geometry instance's lane).
        let placed: Vec<bool> = submission
            .lanes
            .iter()
            .map(|lane| lane.list == engine::fire::ListKind::Translation)
            .collect();
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    positions: &lane.positions,
                    pages: if placed[at] {
                        &[]
                    } else {
                        table(lane, LIST).unwrap_or(&[])
                    },
                    held: table(lane, LIST)
                        .filter(|_| !placed[at])
                        .map(|_| lane.kv.held),
                    folded: lane.kv.folded,
                    readable: lane.kv.readable,
                    set: lane.kv.set,
                    kv_less: lane.kv_less,
                    translation: if placed[at] {
                        table(lane, LIST).unwrap_or(&[])
                    } else {
                        &lane.reserve
                    },
                    write_end: lane.kv.write_end,
                    window: table(lane, WINDOW).unwrap_or(&[]),
                    window_fold: lane.window.clone(),
                    mask: lane.mask.as_ref(),
                    adapter: lane_adapters[at].or(lane.adapter),
                    rs: lane.rs.clone(),
                    rs_reset: lane.rs_reset,
                    rs_slot: records[at]
                        .map(|(_, written)| written)
                        .or(nulls[at].then_some(0)),
                    rs_read: records[at].map(|(run, _)| run),
                    rs_snapshots: &lane.rs_snapshots,
                    replay_stated: lane.replay_stated,
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    group: lane.group,
                    inputs: &lane.inputs,
                    reports: lane.reports,
                    tap: lane.tap.as_ref(),
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                lanes: attachment.lanes,
                instance: attachment.instance,
                adapter_only: attachment.adapter_only,
            })
            .collect();

        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = shell.patch_element() else {
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            staged.reserve(submission.media.len());
            for row in &submission.media {
                staged.push(patch_bytes(&row.patches, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: "cuda",
                    }
                })?);
            }
        }
        let mut voxel_bytes: Vec<Vec<u8>> = Vec::new();
        if !submission.voxels.is_empty() {
            let Some(element) = shell.voxel_element() else {
                return Err(fault(crate::error::Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Vaeless {
                        lane: submission.voxels[0].lane,
                    },
                ))));
            };
            voxel_bytes.reserve(submission.voxels.len());
            for row in &submission.voxels {
                voxel_bytes.push(crate::voxels::input_bytes(&row.payload, element).map_err(
                    |why| Error::Unsupported {
                        verb: why,
                        engine: "cuda",
                    },
                )?);
            }
        }
        let voxels: Vec<crate::serve::Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_bytes)
            .map(|(row, payload)| crate::serve::Clips {
                lane: row.lane,
                clips: &row.clips,
                payload,
            })
            .collect();
        let media: Vec<crate::serve::Media<'_>> = submission
            .media
            .iter()
            .zip(&staged)
            .map(|(row, patches)| crate::serve::Media {
                lane: row.lane,
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                grids: &row.grids,
                token_positions: &row.token_positions,
            })
            .collect();

        let settled = crate::serve::diag::guarded(|| {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                shell,
                crate::serve::StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                    voxels: &voxels,
                    shape: submission.shape,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            shell.settle_step(enqueued, done).map_err(fault)
        })?;

        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            PendingStep {
                readout: submission
                    .lanes
                    .iter()
                    .map(|lane| lane.readout.clone())
                    .collect(),
                settled,
            },
        ))
    }
}

fn patch_bytes(
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
    }
}

fn kv_copy_direction(src: MemoryDomain, dst: MemoryDomain) -> &'static str {
    match (src, dst) {
        (MemoryDomain::HostPinned, MemoryDomain::HostPinned) => {
            "`copy_kv` host-pinned to host-pinned, which is the caller's own memmove"
        }
        (MemoryDomain::HostPinned, _) => {
            "`copy_kv` out of host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (_, MemoryDomain::HostPinned) => {
            "`copy_kv` into host-pinned memory, which needs a swap pool this load does not \
             reserve"
        }
        (MemoryDomain::CudaDevice(_), MemoryDomain::CudaDevice(_)) => {
            "`copy_kv` between two CUDA ordinals, which needs a peer mapping this load has not \
             opened"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

/// A refusal of a split step's head, its lanes named by the step's lanes
/// `cut` (the head's lane `i` is the step's lane `cut[i]`).
fn of_lanes(error: Error, cut: &[u32]) -> Error {
    match error {
        Error::Lanes(lanes) => Error::Lanes(
            lanes
                .into_iter()
                .map(|mut fault| {
                    fault.lane = cut.get(fault.lane as usize).copied().unwrap_or(fault.lane);
                    fault
                })
                .collect(),
        ),
        other => other,
    }
}

fn attributed(error: Error, airborne: u64) -> Error {
    if airborne == 0 {
        return error;
    }
    let note = format!(
        " (detected while {airborne} earlier step(s) were still airborne; device \
         faults are asynchronous, so this may name the step that DETECTED the \
         fault rather than the step that caused it)"
    );
    match error {
        Error::Device(why) => Error::Device(format!("{why}{note}")),
        Error::Program(why) => Error::Program(format!("{why}{note}")),
        other => other,
    }
}

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let mut out = Vec::with_capacity(step.readout.len());
    for (lane, want) in step.readout.iter().enumerate() {
        let output = step.settled.outputs.get(lane).cloned().unwrap_or_default();
        let values = step.settled.logits.get(lane).cloned().unwrap_or_default();
        let rows = step.settled.rows.get(lane).copied().unwrap_or(0);
        let width = if rows == 0 {
            0
        } else {
            u32::try_from(values.len() / rows as usize).unwrap_or(u32::MAX)
        };
        if let Some((pixels, clips)) = step.settled.pixels.get(lane).filter(|(_, c)| !c.is_empty())
        {
            let voxels: usize = clips
                .iter()
                .map(|[t, h, w]| *t as usize * *h as usize * *w as usize)
                .sum();
            out.push(LaneReadout {
                rows: u32::try_from(voxels).unwrap_or(u32::MAX),
                width: u32::try_from(pixels.len() / voxels.max(1)).unwrap_or(u32::MAX),
                values: pixels.clone(),
                output,
                clips: clips.clone(),
            });
            continue;
        }
        out.push(match want {
            Readout::None => LaneReadout {
                output,
                ..LaneReadout::default()
            },
            Readout::Last | Readout::Rows(_) => LaneReadout {
                output,
                rows,
                width,
                values,
                clips: Vec::new(),
            },
        });
    }
    out
}

// SAFETY: `Shell` holds raw device handles and `Cell`s inline, none of
// which the compiler lets cross threads. The handles name driver objects that
// any thread may use once it binds the context, and every verb that touches
// the device, the stream or a `Cell` takes `&mut self` (and `bind_thread`
// binds the context first), so one thread drives a shell at a time.
unsafe impl Send for Cuda {}

#[cfg(test)]
mod tests {
    use super::patch_ladder;
    use engine::load::Budgets as LoadBudgets;
    use poem_compiler::PATCH_LATTICE_FLOOR;
    use poem_ir::{Def, Dim, Dtype, RuntimeInput, Trace, Ty, ValueDecl};

    fn trace_with(shape: Vec<Dim>) -> Trace {
        Trace {
            facts: Default::default(),
            name: "gate".into(),
            platform: poem_ir::Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: vec![ValueDecl {
                def: Def::Input(RuntimeInput::Tokens),
                ty: Ty::Tensor {
                    shape,
                    dtype: Dtype::I32,
                },
            }],
            nodes: Vec::new(),
            outputs: Vec::new(),
            drafter: None,
            inputs: Vec::new(),
        }
    }

    #[test]
    fn api_every_case() {
        a_tower_plan_derives_a_ladder_from_nothing_but_its_own_declaration();
        a_stated_ceiling_wins_and_still_gets_its_rungs();
    }

    fn a_tower_plan_derives_a_ladder_from_nothing_but_its_own_declaration() {
        let ladder = patch_ladder(
            &trace_with(vec![Dim::Patches, Dim::Const(768)]),
            &LoadBudgets::default(),
        )
        .expect("a plan that states patch rows gets a ladder");
        assert_eq!(
            ladder.max_patches, 4096,
            "two whole images at the native grid"
        );
        assert_eq!(
            ladder.buckets,
            vec![64, 128, 256, 512, 1024, 2048, 4096],
            "rungs double from the patch lattice's floor to the ceiling"
        );
        assert_eq!(
            ladder.max_images,
            4096 / PATCH_LATTICE_FLOOR,
            "as many images as the ceiling holds at the smallest whole image"
        );

        for shape in [vec![Dim::Images], vec![Dim::ImagesPlus(1)]] {
            assert!(patch_ladder(&trace_with(shape), &LoadBudgets::default()).is_some());
        }
    }

    fn a_stated_ceiling_wins_and_still_gets_its_rungs() {
        let budgets = LoadBudgets {
            max_patches: Some(1024),
            max_images: Some(3),
            ..LoadBudgets::default()
        };
        let ladder = patch_ladder(&trace_with(vec![Dim::Patches]), &budgets).expect("a ladder");
        assert_eq!(ladder.max_patches, 1024);
        assert_eq!(ladder.max_images, 3);
        assert_eq!(ladder.buckets, vec![64, 128, 256, 512, 1024]);

        let tiny = LoadBudgets {
            max_patches: Some(8),
            ..LoadBudgets::default()
        };
        let raised = patch_ladder(&trace_with(vec![Dim::Patches]), &tiny).expect("a ladder");
        assert_eq!(raised.max_patches, PATCH_LATTICE_FLOOR);
        assert_eq!(raised.buckets, vec![PATCH_LATTICE_FLOOR]);
    }
}

fn width_free(deployment: &str) -> &str {
    match deployment.rsplit_once("-tp") {
        Some((base, width)) if !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit()) => {
            base
        }
        _ => deployment,
    }
}

fn refuse_an_artifact_for_another_deployment(
    path: &std::path::Path,
    backend: &str,
    deployment: &str,
) -> EngineResult<()> {
    let stamp = match checkpoint::file::serve::stamp_of(path) {
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        Err(why) => return Err(Error::Load(why.to_string())),
    };
    let wanted = checkpoint::serving::Stamp::of(backend, deployment);
    stamp
        .check(&wanted)
        .map_err(|mismatch| Error::Load(mismatch.refuse(&path.display().to_string())))
}

#[cfg(test)]
mod serving_stamp_tests {
    use super::refuse_an_artifact_for_another_deployment as refuse;
    use checkpoint::file::emit::{self, Object};
    use checkpoint::serving::Stamp;
    use std::collections::BTreeMap;

    fn artifact(dir: &std::path::Path, backend: &str, deployment: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{backend}-{deployment}.zt"));
        let bytes = vec![7u8; 8192];
        emit::write(
            &path,
            &Stamp::of(backend, deployment),
            &BTreeMap::new(),
            4096,
            &[Object::leaf("embed", vec![8192], ztensor::Leaf::U8, &bytes)],
            |o, p, _| panic!("{o}/{p} is not streamed here"),
        )
        .expect("the fixture artifact writes");
        path
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cuda_stamp_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_artifact_for_another_shell_is_refused_before_anything_is_opened() {
        let dir = tmp("cross");
        let foreign = artifact(&dir, "metal", "qwen_3");
        let why =
            refuse(&foreign, "cuda", "qwen_3").expect_err("a metal artifact is not servable here");
        let said = format!("{why}");
        for wanted in [
            "backend",
            "\"metal\"",
            "\"cuda\"",
            "pie model import --force",
        ] {
            assert!(
                said.contains(wanted),
                "the refusal does not say {wanted:?}: {said}"
            );
        }
        refuse(&artifact(&dir, "cuda", "qwen_3"), "cuda", "qwen_3")
            .expect("a cuda artifact serves on cuda");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// CUDA's readouts ([`engine::Engine::readout_facts`]): the rows a fire
/// reads out land in a slot of the compiled activation arena, carved with
/// the rung (`serve::prepare`, `arena.readout_ceiling_of`), inside the span
/// the rungs state already: no term beside it, and no store of its own
/// bounds them.
pub(crate) fn readout_facts() -> engine::ReadoutFacts {
    engine::ReadoutFacts {
        landing: engine::ReadoutLanding::InSpan,
        store_rows: None,
    }
}
