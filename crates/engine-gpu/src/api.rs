//! The runtime's [`Engine`] over a backend's shell: loading a model,
//! submitting and settling frames, and the guest program plane around them.

use std::collections::BTreeMap;
use std::path::Path;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, MemoryDomain};
use eta_ir::registry::{GeometryClass, PortMask};
use poem_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::program::{Place, Programs, Refusal};
use crate::serve::{Boot, Clips, Landed, Lane, Media, Seated, Serve, Shell, StepView};
use crate::settle::Done;

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// A step fired and not yet read: each lane's readout policy, and its rows
/// once the host has them.
pub struct PendingStep {
    pub readout: Vec<Readout>,

    pub landed: Landed,

    pub rows: Option<Vec<Vec<f32>>>,
}

/// What a backend crate supplies beside its [`Serve`] to stand behind the
/// runtime's [`Engine`]: its name, its program plane, and how a step's guest
/// passes run around its forward.
pub trait Host: Serve {
    /// The engine's name, as the runtime and a message state it.
    const KIND: &'static str;

    /// Whether a guest's device channel commits on the device.
    const DEVICE_CHANNEL_COMMIT: bool;

    type Programs: Programs<Self>;

    /// The memory domain of the device `boot` opens.
    fn domain(boot: &Self::DeviceBoot) -> MemoryDomain;

    /// The ports a load resolves on the device.
    fn ports() -> PortMask;

    fn settles_asynchronously(gpu: &Gpu<Self>) -> bool;

    /// Fires `step` ([`Gpu::fire_step`]) and runs what attaches to it.
    fn step(
        gpu: &mut Gpu<Self>,
        step: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)>;

    /// Every step of a frame is submitted.
    fn submitted(gpu: &Gpu<Self>);

    /// Before `step` resolves its inputs: the passes whose cells its forward
    /// reads.
    fn run_prologues(gpu: &mut Gpu<Self>, step: &Step) -> EngineResult<()>;

    /// The tokens, masks and token feeds `step`'s attachments resolve for
    /// their lanes.
    fn envelope(
        gpu: &mut Gpu<Self>,
        step: &Step,
        envelope: &mut Envelope<Self>,
    ) -> EngineResult<()>;

    /// The guest passes that land with `step`'s rows.
    fn guests(gpu: &mut Gpu<Self>, step: &Step) -> EngineResult<Option<Self::Guests>>;
}

/// What a step's attachments resolve for each of its lanes, at the lane.
pub struct Envelope<B: Serve> {
    pub tokens: Vec<Option<Vec<u32>>>,

    pub masks: Vec<Option<engine::fire::Masking>>,

    /// A device cell holding the lane's one token ([`Seated::feed`]).
    pub feeds: Vec<Option<crate::serve::DeviceCell<B>>>,
}

/// The runtime's engine over a backend's device: one shell, its program
/// plane, and the frame in flight.
pub struct Gpu<B: Host> {
    pub boot: B::DeviceBoot,
    contract_for: ContractFor,
    pub shell: Option<Shell<B>>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,

    pub sink: Option<engine::CompletionSink>,

    pending: Option<(FrameId, Vec<PendingStep>)>,

    pub programs: B::Programs,

    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,

    /// A device the host handed over, adopted at the load.
    handed: Option<B::Handed>,
}

impl<B: Host> Gpu<B> {
    #[must_use]
    pub fn new(boot: B::DeviceBoot, contract_for: ContractFor) -> Gpu<B> {
        Gpu {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            pending: None,
            programs: B::Programs::default(),
            adapters: BTreeMap::new(),
            handed: None,
        }
    }

    /// An engine over a device the host already opened.
    #[must_use]
    pub fn with_device(
        boot: B::DeviceBoot,
        contract_for: ContractFor,
        handed: B::Handed,
    ) -> Gpu<B> {
        Gpu {
            handed: Some(handed),
            ..Gpu::new(boot, contract_for)
        }
    }

    #[must_use]
    pub fn boot(&self) -> &B::DeviceBoot {
        &self.boot
    }

    #[must_use]
    pub fn contract_for(&self) -> ContractFor {
        self.contract_for
    }

    pub fn shell_mut(&mut self) -> Option<&mut Shell<B>> {
        self.shell.as_mut()
    }

    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    pub fn loaded(&self) -> EngineResult<&Shell<B>> {
        self.shell.as_ref().ok_or_else(unloaded::<B>)
    }

    pub fn loaded_mut(&mut self) -> EngineResult<&mut Shell<B>> {
        self.shell.as_mut().ok_or_else(unloaded::<B>)
    }
}

/// Where `shell`'s guests place their tables and cells.
fn place<B: Host>(shell: &Shell<B>) -> Place<'_, B> {
    Place {
        device: shell.device(),
        pools: shell.pools(),
        align: B::table_align(shell.device()),
    }
}

fn unloaded<B: Host>() -> Error {
    Error::Load(format!("the {} engine has no model loaded", B::KIND))
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
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

#[must_use]
pub fn refusal(refusal: Refusal) -> Error {
    match refusal {
        Refusal::Closed { what, id } => Error::Closed { what, id },
        Refusal::Program(why) => Error::Program(why),
    }
}

#[must_use]
pub fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),

        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),

        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Backing { .. }
        | Fault::Mapped { .. }
        | Fault::Param { .. }
        | Fault::Recipe(_)
        | Fault::Shader { .. }
        | Fault::Unbound { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),

        Fault::Fragmented { .. } => Error::Device(fault.to_string()),

        Fault::Unaffine { .. } | Fault::Unstructured { .. } => Error::Load(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),

        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),

        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => Error::Invalid(fault.to_string()),

        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },

        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Vulkan { .. }
        | Fault::Wgpu { .. }
        | Fault::NoDevice { .. }
        | Fault::Blocking { .. } => Error::Device(fault.to_string()),
        Fault::Fire(_) => Error::Invalid(fault.to_string()),

        Fault::Residency(_) => Error::Impossible(fault.to_string()),
    }
}

const DEFAULT_STATE_SLOTS: u32 = 256;

/// The load's budgets: its logit readouts sized to what `trace` reads, not a
/// general figure, so a vocab-wide head carves only the rows a fire can read.
fn bake_budgets(budgets: &LoadBudgets, trace: &Trace) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::readouts_per_lane(trace),
    }
}

#[must_use]
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
        .unwrap_or_else(|| budgets.max_tokens.min(DERIVED_PATCH_CEILING))
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

pub fn patch_bytes(
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
    }
}

pub fn voxel_bytes(
    payload: &[f32],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::Bf16 => Ok(payload
            .iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                (((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()),
        poem_ir::Dtype::F32 => Ok(payload.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err("a voxel payload against a plan whose voxel element is neither bf16 nor f32"),
    }
}

fn adapter_of<B: Host>(
    shell: &mut Shell<B>,
    package: &eta_compiler::codegen::launch::LaunchPackage,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::adapter::Binding>> {
    let Some(sink) = crate::adapter::sink_of(package).map_err(fault)? else {
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
                     weights are the seed, because the fire path never reads the cell, \
                     so an unseeded plane is a correction of zero \
                     that nobody asked for",
                    role.bank()
                ))
            })?;
        built.extend(crate::adapter::planes_of(*role, site, wire, &seats).map_err(fault)?);
    }
    let planes: Vec<crate::weights::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| crate::weights::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .bind_adapter(crate::adapter::Source::Own {
            instance,
            planes: &planes,
        })
        .map(Some)
        .map_err(fault)
}

impl<B: Host> Engine for Gpu<B>
where
    Gpu<B>: Send,
{
    fn kind(&self) -> &'static str {
        B::KIND
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(format!(
                "this {} engine already has a model loaded; one shell per engine",
                B::KIND
            )));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight,
        } = request;

        if ordinal > 0 {
            return Err(Error::unsupported(B::KIND, "device ordinal selection"));
        }

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(format!(
                "the {} shell lands a checkpoint or nothing runs; `Checkpoint::None` has no \
                 weightless path here",
                B::KIND
            )));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let patches = patch_ladder(&trace, &budgets);
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let slots = poem_compiler::state_slots(budgets.slots, &trace, DEFAULT_STATE_SLOTS);
        let (disk_kv_budget, disk_kv_dir, seats) = (
            residency.disk_kv_budget,
            residency.disk_kv_dir.clone(),
            residency.kv_seats.unwrap_or(budgets.max_lanes),
        );
        let budget = bake_budgets(&budgets, &trace);
        let max_readouts =
            u32::try_from(poem_compiler::arena::readouts_ceiling(&budget)).unwrap_or(u32::MAX);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget,
            patches,

            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots,

            runahead: engine::runahead::Runahead::of(frames_in_flight),
            voxels,
            residency,
            device: &self.boot,
            handed: self.handed.take(),
        })
        .map_err(fault)?;
        let disk_kv_pages = shell
            .seat_disk_kv(
                disk_kv_budget,
                disk_kv_dir.as_deref(),
                seats,
                budgets.max_context,
            )
            .map_err(fault)?;

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let weights_resident = !shell.weights_stream();

        let paging = shell.paging();

        let pools = shell.pool_facts(disk_kv_pages);
        let chunks = shell.chunk_facts().map_err(fault)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: B::KIND.to_string(),
                domain: B::domain(&self.boot),
                sms: shell.cores(),
                unified_memory: false,

                fp8_native: false,
                native_mxfp4_moe: false,

                storage_alignment: 256,

                storage_max_tile_bytes: shell.max_buffer(),

                codegen_backend: None,
            },
            pools,
            chunks,
            limits: FireLimits {
                max_unfolded_rows: 0,
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts,
            },
            has_lora: true,
            has_attn_page_mask: false,
            draft_block: shell.trace().drafter.map_or(0, |d| d.rows),
            draft_mask_token: shell.trace().drafter.map_or(0, |d| d.mask_token),
            draft_proposals_from: shell.trace().drafter.map_or(1, |d| d.proposals_from),

            ports: B::ports(),
            geometry: GeometryClass::DecodeEnvelope,

            kv_handle: None,
            media_encode: false,
            device_channel_commit: B::DEVICE_CHANNEL_COMMIT,
            rs_verbs: shell.serves_rs_verbs(),
            // The prompt scans land the state as of rows inside a lane, a
            // record a snapshot in pieces (CONTRACT-BC 16).
            rs_snapshots: crate::rs::snapshots(shell.trace()),
            bidirectional_attention: false,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.inputs(),
            classes: shell.classes(),
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,

                weights_resident,
                weights_from_cache: false,
                arena_bytes,
                pool_bytes,
                input_bytes,

                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn commit_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.commit_segments(segments).map_err(fault)
    }

    fn release_segments(&mut self, segments: u32) -> EngineResult<()> {
        self.loaded_mut()?.release_segments(segments).map_err(fault)
    }

    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        let Gpu {
            shell, programs, ..
        } = self;
        let shell = shell.as_mut().ok_or_else(unloaded::<B>)?;
        programs.quiesce().map_err(refusal)?;
        shell.evacuate(evacuation).map_err(fault)?;
        // Guest tables and cells bind their slots' chunks by place.
        programs.rebind(shell.pools()).map_err(refusal)
    }

    fn program_facts(&self, program: ProgramId) -> EngineResult<engine::program::ProgramFacts> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(engine::program::ProgramFacts::default());
        };
        self.programs.facts(program, &place(shell)).map_err(refusal)
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::program::RingFacts>> {
        let Some(shell) = self.shell.as_ref() else {
            return Ok(None);
        };
        self.programs
            .ring_facts(registration, &place(shell))
            .map_err(refusal)
    }

    fn rebind_slots(&mut self, rebind: engine::program::Rebind) -> EngineResult<()> {
        let shell = self.shell.as_ref().ok_or_else(unloaded::<B>)?;
        self.programs
            .rebind_slots(shell.pools(), &rebind)
            .map_err(refusal)
    }

    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        self.loaded_mut()?.copy_slots(copy).map_err(fault)
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

    fn weight_residency(&mut self, residency: &engine::chunk::WeightResidency) -> EngineResult<()> {
        self.loaded_mut()?
            .weight_residency(residency)
            .map_err(fault)
    }

    fn weight_uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        self.loaded_mut()?.weight_uses().map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        frame.validate_for(
            self.caps
                .as_ref()
                .map_or(engine::fire::Serves::NONE, engine::fire::Serves::of),
        )?;
        let id = FrameId(self.next_frame);
        self.next_frame = self.next_frame.wrapping_add(1);

        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        tracing::debug!(
            engine = B::KIND,
            frame = id.0,
            steps = frame.steps.len(),
            "submit"
        );
        let mut pending = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }

            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let fired = B::step(self, step, at);
            match fired {
                Ok((ticket, step_pending)) => {
                    steps.push(ticket);
                    pending.push(step_pending);
                }
                Err(error) => {
                    self.pending = None;
                    return Err(error);
                }
            }
        }
        B::submitted(self);
        let mut ticket = FrameTicket { id, steps };
        self.pending = Some((id, pending));

        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.programs.register(registration).map_err(refusal)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        let place = self.shell.as_ref().map(place);
        self.programs
            .register_channel(registration, place.as_ref())
            .map_err(refusal)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| Error::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(Error::Program(format!(
                "this load resolves {:?} on the device, so it binds at most {:?} and not {:?}",
                caps.ports, caps.geometry, binding.geometry
            )));
        }

        let place = self.shell.as_ref().map(place);
        let bound = self
            .programs
            .bind(binding, place.as_ref())
            .map_err(refusal)?;

        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let package = self.programs.package(binding.program).cloned();
        let landed = match (package, self.shell.as_mut()) {
            (Some(package), Some(shell)) => adapter_of(shell, &package, bound.id, &seeds),
            _ => Ok(None),
        };
        match landed {
            Ok(Some(binding)) => {
                self.adapters.insert(bound.id, binding);
            }
            Ok(None) => {}
            Err(why) => {
                let _ = self.programs.close_instance(bound.id);
                return Err(why);
            }
        }
        Ok(bound)
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        if let (Some(held), Some(shell)) = (self.adapters.remove(&id), self.shell.as_mut()) {
            shell.release_adapter(&held);
        }
        self.programs.close_instance(id).map_err(refusal)
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        let planes: Vec<crate::weights::AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| crate::weights::AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        self.programs.close_channel(id).map_err(refusal)
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.programs
            .publish(instance, channel, cell)
            .map_err(refusal)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.programs.take(instance, channel).map_err(refusal)
    }

    fn settles_asynchronously(&self) -> bool {
        B::settles_asynchronously(self)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has already taken \
                 its readout seats",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s numbers are gone: frame {id} has been submitted since, and \
                 a step's rows live in the readout seat its arm owns. Ask for a \
                 frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut pending) = self.pending.take().expect("checked just above");
        let mut refused = None;
        match self.shell.as_mut() {
            None => {
                refused = Some(unloaded::<B>());
            }
            Some(shell) => {
                for step in &mut pending {
                    if step.rows.is_some() {
                        continue;
                    }
                    match shell.rows_of(&step.landed) {
                        Ok(rows) => step.rows = Some(rows),
                        Err(why) => {
                            refused = Some(fault(why));
                            break;
                        }
                    }
                }
            }
        }

        self.pending = Some((ticket.id, pending));
        if let Some(error) = refused {
            return Err(error);
        }
        let (_, pending) = self.pending.as_ref().expect("just put back");
        for (receipt, step) in ticket.steps.iter_mut().zip(pending) {
            receipt.readouts = readouts_of(step);
        }
        Ok(())
    }

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
        }
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        if device.is_none_or(|device| copy.src != device || copy.dst != device) {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: B::KIND,
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }
}

impl<B: Host> Gpu<B> {
    /// Fires `submission`'s forward; with `defers` its completion is left
    /// for the guest pass behind it to report.
    pub fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
        defers: bool,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        B::run_prologues(self, submission)?;
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = self.loaded()?.patch_element() else {
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
                        engine: B::KIND,
                    }
                })?);
            }
        }
        let voxel_element = self.loaded()?.voxel_element();
        if !submission.voxels.is_empty() && voxel_element.is_none() {
            return Err(Error::Invalid(
                "this step submits VAE clips and the loaded plan states no voxel row".into(),
            ));
        }
        let voxel_payloads = submission
            .voxels
            .iter()
            .map(|row| {
                voxel_bytes(&row.payload, voxel_element.unwrap_or(poem_ir::Dtype::Bf16)).map_err(
                    |why| Error::Unsupported {
                        verb: why,
                        engine: B::KIND,
                    },
                )
            })
            .collect::<EngineResult<Vec<_>>>()?;
        let clips: Vec<Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&voxel_payloads)
            .map(|(row, payload)| Clips {
                lane: row.lane,
                boxes: &row.clips,
                payload,
            })
            .collect();

        let media: Vec<Media<'_>> = submission
            .media
            .iter()
            .zip(&staged)
            .map(|(row, patches)| Media {
                lane: row.lane,
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                grids: &row.grids,
                token_positions: &row.token_positions,
            })
            .collect();

        let id = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);

        let lanes = submission.lanes.len();
        let mut envelope = Envelope {
            tokens: vec![None; lanes],
            masks: vec![None; lanes],
            feeds: vec![None; lanes],
        };
        B::envelope(self, submission, &mut envelope)?;
        let Envelope {
            tokens: resolved,
            masks: resolved_masks,
            feeds,
        } = envelope;

        let mut verbs: Vec<engine::fire::RsVerb> = submission
            .lanes
            .iter()
            .map(|lane| lane.rs.clone())
            .collect();
        for attachment in &submission.attachments {
            if let engine::fire::RsVerb::Buffer { fold, .. } = &mut verbs[attachment.lane as usize]
                && *fold == engine::fire::FoldLen::Device
                && let Some(n) = self
                    .programs
                    .envelope_fold(attachment.instance)
                    .map_err(refusal)?
            {
                *fold = engine::fire::FoldLen::Host(n);
            }
        }

        let mut input_cells: Vec<Vec<(engine::fire::InputFeed, crate::serve::InputCell<B>)>> =
            vec![Vec::new(); submission.lanes.len()];
        for (at, lane) in submission.lanes.iter().enumerate() {
            if lane.inputs.is_empty() {
                continue;
            }
            let instance = submission
                .attachments
                .iter()
                .find(|attachment| {
                    (attachment.lane..attachment.lane + attachment.lanes.max(1))
                        .contains(&(at as u32))
                })
                .map(|attachment| attachment.instance)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "lane {at} feeds guest inputs and has no attachment; an input is read \
                         through the instance the lane is attached to"
                    ))
                })?;
            for feed in &lane.inputs {
                let cell = self
                    .programs
                    .input_cell(instance, feed.channel)
                    .map_err(refusal)?;
                input_cells[at].push((*feed, cell));
            }
        }

        let guests = B::guests(self, submission)?;

        let mut reads: Vec<Vec<String>> = vec![Vec::new(); submission.lanes.len()];
        for attachment in program_attachments(submission) {
            for (name, k) in self.programs.reads(attachment.instance) {
                if let Some(lane) = reads.get_mut((attachment.lane + k) as usize) {
                    lane.push(name);
                }
            }
        }

        let sink = self.sink.clone();

        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        if !self.adapters.is_empty() {
            for attachment in &submission.attachments {
                let Some(bound) = self.adapters.get(&attachment.instance) else {
                    continue;
                };
                if let Some(slot) = lane_adapters.get_mut(attachment.lane as usize) {
                    *slot = Some(bound.slot);
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
                Error::Invalid(format!(
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
        // A lane stating its slot tables names its kv pages and its record
        // there (CONTRACT-BC 4); until the runtime states them, the legacy
        // fields carry the same slot refs.
        let mut pages_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut record_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut run_of: Vec<Option<Vec<u32>>> = Vec::with_capacity(submission.lanes.len());
        let mut forks: Vec<(u32, u32)> = Vec::new();
        for lane in &submission.lanes {
            pages_of.push(
                lane.tables
                    .iter()
                    .find(|(shape, _)| *shape == crate::store::layout::LIST)
                    .map(|(_, refs)| refs.clone()),
            );
            let fixed = lane
                .fixed
                .iter()
                .find(|(shape, _)| *shape == crate::store::layout::FIXED);
            run_of.push(fixed.and_then(|(_, pieces)| {
                pieces
                    .iter()
                    .any(|&[run, cow]| run != cow)
                    .then(|| pieces.iter().map(|&[run, _]| run).collect())
            }));
            record_of.push(fixed.map(|(_, pieces)| {
                // `cow != run`: the record is copied before the fire, once
                // for every lane of a group naming the same pair.
                for &[run, cow] in pieces {
                    if run != cow && !forks.contains(&(run, cow)) {
                        forks.push((run, cow));
                    }
                }
                pieces.iter().map(|&[_, cow]| cow).collect()
            }));
        }
        shell.copy_state(&forks).map_err(fault)?;
        // Tail copies (an rs-list or kv slot a fork shares) land before any
        // lane of the fire reads or writes (CONTRACT-BC 20).
        let slot_copies: Vec<_> = submission
            .lanes
            .iter()
            .flat_map(|lane| lane.slot_copies.iter().cloned())
            .collect();
        shell.copy_slot_pairs(&slot_copies).map_err(fault)?;
        let seated: Vec<Seated<'_, B>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| Seated {
                lane: Lane {
                    slot: lane.slot,

                    word: words[at],
                    tokens: resolved[at].as_deref().unwrap_or(&lane.tokens),
                },
                pages: pages_of[at].as_deref().unwrap_or(&lane.kv.pages),
                held: (!lane.kv.pages.is_empty() || pages_of[at].is_some()).then_some(lane.kv.held),
                kv_less: lane.kv_less,

                mask: resolved_masks[at].as_ref().or(lane.mask.as_ref()),

                adapter: lane_adapters[at].or(lane.adapter),
                positions: &lane.positions,

                readout: match &lane.readout {
                    Readout::Rows(rows) => Some(rows.as_slice()),
                    Readout::Last | Readout::None => None,
                },
                reads: &reads[at],

                translation: &lane.kv.translation,

                rs: &verbs[at],
                rs_reset: lane.rs_reset,
                rs_slot: lane.rs_slot,
                rs_snapshots: &lane.rs_snapshots,
                refs: true,
                windows: &lane.tables,
                record: record_of[at].as_deref(),
                record_run: run_of[at].as_deref(),
                feed: feeds[at].as_ref(),
                group: lane.group,
                inputs: &input_cells[at],
            })
            .collect();

        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.filter(|_| !defers).map(|sink| Done { at, sink });
            let prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &[],
                    media: &media,
                    clips: &clips,
                    done,
                    guests,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
        };

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
                landed,
                rows: None,
            },
        ))
    }
}

fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| {
            let count = match policy {
                Readout::None => return LaneReadout::default(),
                Readout::Last => 1,
                Readout::Rows(list) => list.len().max(1),
            };
            let values = rows.get(lane).cloned().unwrap_or_default();
            LaneReadout {
                rows: u32::try_from(count).unwrap_or(u32::MAX),
                width: u32::try_from(values.len() / count).unwrap_or(u32::MAX),
                values,
                ..LaneReadout::default()
            }
        })
        .collect()
}

/// The attachments whose program runs on this fire.
pub fn program_attachments(step: &Step) -> impl Iterator<Item = &engine::fire::Attachment> {
    step.attachments
        .iter()
        .filter(|attachment| !attachment.adapter_only)
}
