//! The `Engine` the runtime drives.
//!
//! Fires settle asynchronously (frames in flight): `submit` stages each
//! step, enqueues its fire and its guest passes on the device and returns;
//! a step's `StepDone` comes from the plugin's thread once the cells its
//! passes put have landed in their rings. The readout stays on the device:
//! eta guest programs attached to a lane run there (`Plane::fire_guests`,
//! lowered to StableHLO, lanes of one program batched), and only a row the
//! host reads — a guest that did not lower, a lane's readout asked for in
//! `settle_frame` — is copied out.
//!
//! Ordering: a step's guest passes bind their roots only after every pass
//! still in flight has landed, and that wait comes after the step's own
//! fire is enqueued, so the device always has the next fire queued behind
//! the pass it waits on. A fire whose `embed_tokens` cell is still in
//! flight reads it on the device, straight out of the pass's output (a
//! scatter into its input pack); any other port that reads an in-flight
//! cell waits for it to land. `PIE_XLA_SYNC=1` settles every frame inside
//! `submit`, for an A/B.
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, MemoryDomain, StateCopy};
use eta_ir::registry::{GeometryClass, PortMask, Stage};
use poem_compiler::{Budget, PATCH_LATTICE_FLOOR, PatchLadder};
use poem_ir::Trace;

use crate::error::Fault;
use crate::serve::{Boot, Lane, Seated, Shell};

pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBoot {
    /// The PJRT plugin; `None` finds one (`PIE_XLA_PLUGIN`,
    /// `TPU_LIBRARY_PATH`, then `libtpu.so`).
    pub plugin: Option<PathBuf>,

    /// Which addressable device of the plugin's client.
    pub ordinal: u32,

    /// `$PIE_HOME/profiles`, where a user's TPU profiles live.
    pub profiles: Option<PathBuf>,

    /// Trace and compile the first fire shapes at load (`serve/warm.rs`).
    pub prewarm: bool,

    /// Windowed kv rows live in their own pool; `false` keeps them in full
    /// pages, for a test that compares the two.
    pub window_pool: bool,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            plugin: None,
            ordinal: 0,
            profiles: None,
            prewarm: false,
            window_pool: true,
        }
    }
}

/// Fields drop in order: the pending frame's kept readouts and the program
/// plane (guest executables and buffers made on the shell's device) go
/// before the shell that owns the client.
pub struct Xla {
    /// The last frame submitted: what `settle_frame` reads its steps'
    /// readouts from. Its kept readouts are device buffers, so it drops
    /// before the shell that owns the client.
    pending: Option<(FrameId, Vec<StepSettle>)>,
    programs: crate::program::Plane,
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,
}

/// What a step's readouts are made from once it settles.
struct StepSettle {
    kept: Option<std::sync::Arc<crate::readout::Kept>>,
    pixels: Vec<crate::serve::Pixels>,
    on_device: Vec<bool>,
    policies: Vec<Readout>,
}

/// `PIE_XLA_SYNC=1`: every frame settles inside `submit`.
fn synchronous() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_XLA_SYNC").is_some_and(|v| v != "0"))
}

impl Xla {
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Xla {
        Xla {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            programs: crate::program::Plane::default(),
            adapters: BTreeMap::new(),
            pending: None,
        }
    }

    #[must_use]
    pub fn boot(&self) -> &DeviceBoot {
        &self.boot
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

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the xla engine has no model loaded".into()))
    }
}

fn refusal(refusal: crate::program::Refusal) -> Error {
    match refusal {
        crate::program::Refusal::Closed { what, id } => Error::Closed { what, id },
        crate::program::Refusal::Program(why) => Error::Program(why),
    }
}

fn port_error(error: engine::ports::PortError) -> Error {
    fault(Fault::Program {
        at: "program::ports",
        why: error.0,
    })
}

pub(crate) fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Deviceless | Fault::Device { .. } | Fault::NoDevice { .. } => {
            Error::Device(fault.to_string())
        }
        Fault::Xla { .. } | Fault::Blocking { .. } | Fault::Fragmented { .. } => {
            Error::Device(fault.to_string())
        }
        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Backing { .. }
        | Fault::Mapped { .. }
        | Fault::Param { .. }
        | Fault::Recipe(_)
        | Fault::Shader { .. }
        | Fault::Unbound { .. }
        | Fault::Unaffine { .. }
        | Fault::Unstructured { .. }
        | Fault::Straddled { .. }
        | Fault::Adapter { .. }
        | Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. }
        | Fault::Positions { .. }
        | Fault::Adapterless { .. }
        | Fault::AdapterWord { .. }
        | Fault::Fire(_) => Error::Invalid(fault.to_string()),
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
    }
}

fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
        readouts_per_lane: poem_compiler::arena::READOUTS_PER_LANE,
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

fn patch_bytes(
    patches: &[u8],
    element: poem_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        poem_ir::Dtype::U8 => Ok(patches.to_vec()),
        _ => Err("a media submission against a plan whose pixel rows are not u8 bytes"),
    }
}

fn adapter_of(
    shell: &mut Shell,
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
                     {channel} and this bind seeded nothing into it",
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

impl Engine for Xla {
    fn kind(&self) -> &'static str {
        "xla"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this xla engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            frames_in_flight: _,
        } = request;
        let mut boot = self.boot.clone();
        if ordinal > 0 {
            boot.ordinal = u32::try_from(ordinal).unwrap_or(0);
        }
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the xla shell lands a checkpoint or nothing runs".into(),
            ));
        };
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;
        let patches = patch_ladder(&trace, &budgets);
        let voxels = crate::dit::voxel_ladder(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let shell = Shell::load_with(
            Boot {
                patches,
                trace,
                contract: &contract,
                checkpoint: &path,
                budget: bake_budgets(&budgets),
                page_size: budgets.page_size,
                context: budgets.max_context,
                slots: budgets.slots,
                kv_budget: residency.device_kv_budget,
                memory_share: residency.device_memory_share,
                device: &boot,
            },
            voxels,
        )
        .map_err(fault)?;
        let mut shell = shell;
        if boot.prewarm {
            shell.prewarm().map_err(fault)?;
        }
        let trace_name = shell.trace().name.clone();
        let (weight_bytes, pool_bytes) = shell.footprint();
        let paging = shell.paging();
        let state_rows = shell.has_state();
        let drafter = shell.trace().drafter;
        let caps = Capabilities {
            device: DeviceFacts {
                backend: "xla".to_string(),
                domain: MemoryDomain::XlaDevice(boot.ordinal),
                sms: 1,
                unified_memory: false,
                fp8_native: shell.tpu().fp8_native(),
                native_mxfp4_moe: false,
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: None,
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                buffer_pages: shell.buffer_pages(),
                state_slot_bytes: shell.state_slot_bytes(),
                adapter_banks: u32::try_from(shell.banks().len()).unwrap_or(u32::MAX),
                elastic_page_bytes: 0,
                elastic_budget_pages: 0,
                window_pages: paging.window.map_or(0, |_| {
                    u32::try_from(paging.window_pages()).unwrap_or(u32::MAX)
                }),
                window_tokens: paging.window.map_or(0, |window| window.tokens),
                ..PoolFacts::default()
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
                prefill_rows: 0,
                max_readouts: u32::try_from(poem_compiler::arena::readouts_ceiling(&bake_budgets(
                    &budgets,
                )))
                .unwrap_or(u32::MAX),
            },
            has_lora: true,
            has_attn_page_mask: false,
            draft_block: drafter.map_or(0, |d| d.rows),
            draft_mask_token: drafter.map_or(0, |d| d.mask_token),
            draft_proposals_from: drafter.map_or(1, |d| d.proposals_from),
            ports: PortMask::DEVICE_GEOMETRY
                .with(eta_ir::registry::Port::AttnMask)
                .with(eta_ir::registry::Port::Fold),
            geometry: GeometryClass::DeviceGeometry,
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            device_channel_commit: true,
            rs_verbs: shell.serves_rs_verbs(),
            rs_snapshots: 0,
            bidirectional_attention: true,
            adapter_attachments: false,
            facts: shell.trace().facts.clone(),
            inputs: shell.input_decls(),
            classes: shell.classes(),
        };
        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                weights_resident: true,
                weights_from_cache: false,
                arena_bytes: 0,
                pool_bytes,
                input_bytes: 0,
                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
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
        tracing::debug!(
            frame = id.0,
            steps = frame.steps.len(),
            airborne = self.programs.airborne(),
            "xla submit"
        );
        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut settles = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let latch = crate::program::Latch::new(at, self.sink.clone());
            let (ticket, settle) = self.fire_step(step, &latch)?;
            // The submitter's hold: the step is done once its passes land.
            latch.release(Ok(()));
            steps.push(ticket);
            settles.push(settle);
        }
        let mut ticket = FrameTicket { id, steps };
        self.pending = Some((id, settles));
        tracing::debug!(frame = id.0, "xla submitted");
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    fn settles_asynchronously(&self) -> bool {
        !synchronous()
    }

    fn settle_frame(&mut self, ticket: &mut FrameTicket) -> EngineResult<()> {
        let Some((id, _)) = self.pending.as_ref() else {
            return Err(Error::Invalid(format!(
                "frame {}'s readouts are gone: nothing is pending, so either it was \
                 never submitted to this engine or a later frame has been submitted since",
                ticket.id
            )));
        };
        if *id != ticket.id {
            return Err(Error::Invalid(format!(
                "frame {}'s readouts are gone: frame {id} has been submitted since. Ask \
                 for a frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        // Every guest pass lands its cells before the frame answers.
        self.programs.settle();
        let (_, settles) = self.pending.as_ref().expect("checked just above");
        for (receipt, settle) in ticket.steps.iter_mut().zip(settles) {
            receipt.readouts = readouts_of(settle)?;
        }
        Ok(())
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.programs.register(registration).map_err(refusal)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.programs
            .register_channel(registration)
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
        let bound = self.programs.bind(binding).map_err(refusal)?;
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

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        let served = self
            .caps
            .as_ref()
            .is_some_and(|caps| copy.src == caps.device.domain && copy.dst == caps.device.domain);
        if !served {
            return Err(Error::Unsupported {
                verb: "`copy_kv` between domains other than this load's device",
                engine: "xla",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves = copy.runs(page_size)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }

    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        copy.device_only("xla")?;
        for (at, move_) in copy.moves.iter().enumerate() {
            if move_.src_token_offset != 0 || move_.dst_token_offset != 0 {
                return Err(Error::Invalid(format!(
                    "state move {at} names a token offset; this engine moves whole slots"
                )));
            }
        }
        let shell = self.loaded_mut()?;
        for move_ in &copy.moves {
            shell
                .copy_state(move_.src_slot_id, move_.dst_slot_id)
                .map_err(fault)?;
        }
        let pages: Vec<(u32, u32)> = copy
            .pages
            .iter()
            .map(|m| (m.src_slot_id, m.dst_slot_id))
            .collect();
        shell.copy_pages(&pages).map_err(fault)
    }
}

/// Per lane its input cells (id, bytes in the input's element); per voxel row its payload.
struct LaneFeeds {
    cells: Vec<Vec<(u32, Vec<u8>)>>,
    voxels: Vec<Vec<u8>>,
}

/// `PIE_XLA_CAPTURE=<dir>` writes every submitted step there as JSON
/// (`step-<n>.json`), for replaying a served request offline.
fn capture(step: &engine::fire::Step) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(dir) = std::env::var_os("PIE_XLA_CAPTURE") else {
        return;
    };
    let at = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::path::Path::new(&dir).join(format!("step-{at:05}.json"));
    match serde_json::to_vec(step) {
        Ok(bytes) => {
            if let Err(why) = std::fs::write(&path, bytes) {
                tracing::warn!("PIE_XLA_CAPTURE: {}: {why}", path.display());
            }
        }
        Err(why) => tracing::warn!("PIE_XLA_CAPTURE: step {at}: {why}"),
    }
}

/// One lane of a device-resolved fire, as its instance's descriptor ports
/// state it (engine-cuda `serve/prepare.rs`, the `envelope_of` lanes).
#[derive(Debug, Default)]
struct DeviceLane {
    tokens: Vec<u32>,
    positions: Option<Vec<u32>>,
    /// The lane's own page table, in pool ids.
    pages: Option<Vec<u32>>,
    /// The rows the cache holds before this fire, when the ports state it.
    have: Option<u32>,
    writes: Option<(Vec<u32>, Vec<u32>)>,
    first: Option<u32>,
    mask: Option<engine::fire::Masking>,
    /// The lane's windowed ids for `pages`, when the runtime handed any.
    window: Option<Vec<u32>>,
}

impl Xla {
    /// Resolves every lane an instance bound in `DeviceGeometry` describes
    /// on its descriptor ports: its tokens, positions, KV extent, the slots
    /// its rows land in (whose page table and write descriptor come from
    /// the working set's table, in pool ids) and attention mask.
    fn device_lanes(
        &self,
        submission: &engine::fire::Step,
    ) -> EngineResult<Vec<Option<DeviceLane>>> {
        let lanes = &submission.lanes;
        let mut out: Vec<Option<DeviceLane>> = (0..lanes.len()).map(|_| None).collect();
        let program = |why: String| Error::Program(why);
        let page_tokens = self
            .shell
            .as_ref()
            .map_or(1, |shell| shell.paging().page_size)
            .max(1);
        for attached in &submission.attachments {
            if self.programs.geometry_of(attached.instance) != Some(GeometryClass::DeviceGeometry) {
                continue;
            }
            let Some(envelope) = self
                .programs
                .device_envelope(attached.instance)
                .map_err(refusal)?
            else {
                continue;
            };
            let first = attached.lane as usize;
            let carried = envelope.lanes();
            if first + carried > lanes.len() {
                return Err(program(format!(
                    "instance {} is attached at lane {first} and its `embed_indptr` port \
                     describes {carried} lane(s), which runs past the {} this fire carries",
                    attached.instance,
                    lanes.len()
                )));
            }
            for at in 0..carried {
                let source = first + at;
                if out[source].is_some() {
                    return Err(program(format!(
                        "lane {source} is claimed by two attached instances; a lane's \
                         descriptor ports have one author"
                    )));
                }
                let lane = &lanes[source];
                let ports = envelope.lane(at, source).map_err(port_error)?;
                let table = &lane.kv.translation;
                let translate = |page: u32, port: &str| -> EngineResult<u32> {
                    table.get(page as usize).copied().ok_or_else(|| {
                        program(format!(
                            "lane {source}'s `{port}` port names working-set page {page} and \
                             the table this fire was handed maps {} page(s); a guest holds \
                             relative indexes and the pool's ids are the runtime's, so an \
                             index past the table addresses somebody else's cache",
                            table.len()
                        ))
                    })
                };
                let rows = if ports.owns_slots() {
                    ports.rows() as usize
                } else {
                    lane.tokens.len()
                };
                let slots = ports.slots(rows).map_err(port_error)?;
                let first = slots.and_then(|slots| slots.first().copied());
                let (pages, window, writes) = match (slots, ports.reach(rows).map_err(port_error)?)
                {
                    (Some(slots), Some(reach)) => {
                        let relative: Vec<u32> = (0..reach.div_ceil(page_tokens)).collect();
                        let pages = relative
                            .iter()
                            .map(|&page| translate(page, "slots"))
                            .collect::<EngineResult<Vec<u32>>>()?;
                        // The windowed ids are aligned with the translation:
                        // the same relative index picks a page's windowed id.
                        let window = (!lane.kv.window.is_empty()).then(|| {
                            relative
                                .iter()
                                .map(|&page| {
                                    lane.kv.window.get(page as usize).copied().unwrap_or(0)
                                })
                                .collect::<Vec<u32>>()
                        });
                        let writes = (
                            slots
                                .iter()
                                .map(|&slot| translate(slot / page_tokens, "slots"))
                                .collect::<EngineResult<Vec<u32>>>()?,
                            slots.iter().map(|&slot| slot % page_tokens).collect(),
                        );
                        (Some(pages), window, Some(writes))
                    }
                    _ => (None, None, None),
                };
                let mask = match ports.mask(rows).map_err(port_error)? {
                    Some(_) if rows != 1 => {
                        return Err(program(format!(
                            "lane {source} resolves its attention mask from a channel and \
                             carries {rows} query rows; the expansion intersects each row \
                             with the order the cache is written in, and a lane that states \
                             its own slots has no such order this shell can derive"
                        )));
                    }
                    Some((cells, stride)) => Some(engine::ports::from_dense(cells, stride)),
                    None => None,
                };
                let have = match () {
                    () if lane.kv_less => Some(0),
                    () if ports.owns_slots() => {
                        let after = ports.reach(rows).map_err(port_error)?.ok_or_else(|| {
                            program(format!(
                                "lane {source} states where its rows land and reaches no \
                                 slot; the page count, the last page's fill and the \
                                 attention schedules are all carved from what it reads"
                            ))
                        })?;
                        if (after as usize) < rows {
                            return Err(program(format!(
                                "lane {source} reads a KV extent of {after} and this fire \
                                 writes {rows} row(s) into it; the extent is AFTER the \
                                 append, so it can never be shorter than what the append adds"
                            )));
                        }
                        Some(after - rows as u32)
                    }
                    () => (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                };
                if let Some(have) = have {
                    ports
                        .check_extent(have.saturating_add(rows as u32))
                        .map_err(port_error)?;
                }
                let tokens = ports.tokens_for(rows).map_err(port_error)?.to_vec();
                let positions = match have {
                    Some(have) => ports
                        .positions_for(have, rows)
                        .map_err(port_error)?
                        .map(<[u32]>::to_vec),
                    None => None,
                };
                out[source] = Some(DeviceLane {
                    tokens,
                    positions,
                    pages,
                    have,
                    writes,
                    first,
                    mask,
                    window,
                });
            }
        }
        Ok(out)
    }

    /// Runs each attached prologue first: the forward reads the cells a prologue put.
    fn run_prologues(&mut self, submission: &engine::fire::Step) -> EngineResult<()> {
        let prologues: Vec<(InstanceId, u32)> = submission
            .attachments
            .iter()
            .filter(|attachment| {
                !attachment.adapter_only && self.programs.prologue_runs(attachment.instance)
            })
            .map(|attachment| (attachment.instance, attachment.lane))
            .collect();
        if prologues.is_empty() {
            return Ok(());
        }
        let device = self
            .shell
            .as_ref()
            .ok_or_else(|| Error::Load("the xla engine has no model loaded".into()))?
            .device();
        self.programs
            .fire_guests(device, &prologues, None, Stage::Prologue)
            .map_err(refusal)?;
        Ok(())
    }

    fn lane_feeds(&self, submission: &engine::fire::Step) -> EngineResult<LaneFeeds> {
        let lanes = &submission.lanes;
        let instance_of = |lane: usize| {
            submission
                .attachments
                .iter()
                .find(|attached| {
                    (attached.lane as usize..(attached.lane + attached.lanes) as usize)
                        .contains(&lane)
                })
                .map(|attached| attached.instance)
        };
        let shell = self.shell.as_ref();
        let mut cells: Vec<Vec<(u32, Vec<u8>)>> = vec![Vec::new(); lanes.len()];
        let mut voxel_cells: Vec<Option<Vec<u8>>> = vec![None; lanes.len()];
        for (at, lane) in lanes.iter().enumerate() {
            for feed in &lane.inputs {
                let seat = shell
                    .and_then(|shell| shell.input_seat(feed.id))
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "lane {at} feeds guest input {} and this plan reads no such input",
                            feed.id
                        ))
                    })?;
                let instance = instance_of(at).ok_or_else(|| {
                    Error::Program(format!(
                        "lane {at} feeds guest input {} off channel {} but attaches no \
                         instance; an input cell is read through the instance that carries the \
                         channel",
                        feed.id, feed.channel
                    ))
                })?;
                let cell = self
                    .programs
                    .feed_cell(instance, feed.channel)
                    .map_err(refusal)?;
                let bytes = crate::dit::cell_bytes(&cell, seat.dtype).map_err(|why| {
                    Error::Program(format!(
                        "channel {} (lane {at}'s guest input {}) holds a {:?} cell for a {:?} \
                         input: {why}",
                        feed.channel,
                        feed.id,
                        cell.dtype(),
                        seat.dtype
                    ))
                })?;
                if seat.voxel() {
                    voxel_cells[at] = Some(bytes);
                } else {
                    cells[at].push((feed.id, bytes));
                }
            }
        }

        let mut voxels = Vec::with_capacity(submission.voxels.len());
        for row in &submission.voxels {
            if row.payload.is_empty() {
                voxels.push(
                    voxel_cells
                        .get(row.lane as usize)
                        .cloned()
                        .flatten()
                        .unwrap_or_default(),
                );
                continue;
            }
            let Some(element) = shell.and_then(Shell::voxel_element) else {
                return Err(Error::Invalid(format!(
                    "lane {} carries a voxel payload and this plan reads no voxel input",
                    row.lane
                )));
            };
            voxels.push(
                crate::dit::payload_bytes(&row.payload, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: "xla",
                    }
                })?,
            );
        }

        Ok(LaneFeeds { cells, voxels })
    }

    fn fire_step(
        &mut self,
        submission: &engine::fire::Step,
        latch: &std::sync::Arc<crate::program::Latch>,
    ) -> EngineResult<(FireTicket, StepSettle)> {
        capture(submission);
        // Checked before a prologue moves the cursors the caller predicted.
        for attachment in &submission.attachments {
            if let Some(lane) = submission.lanes.get(attachment.lane as usize)
                && !lane.channels.is_empty()
                && let Some(why) = self
                    .programs
                    .disagreeing_ticket(attachment.instance, &lane.channels)
            {
                return Err(Error::Program(format!(
                    "this fire's channel predictions and the engine's disagree: {why}"
                )));
            }
        }
        self.run_prologues(submission)?;
        let fed = self.lane_feeds(submission)?;
        let input_cells: Vec<Vec<crate::serve::InputCell<'_>>> = fed
            .cells
            .iter()
            .map(|lane| {
                lane.iter()
                    .map(|(id, bytes)| crate::serve::InputCell { id: *id, bytes })
                    .collect()
            })
            .collect();
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            let Some(element) = self.shell.as_ref().and_then(Shell::patch_element) else {
                return Err(fault(Fault::from(poem_exec::Error::Fire(
                    poem_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            for row in &submission.media {
                staged.push(patch_bytes(&row.patches, element).map_err(|why| {
                    Error::Unsupported {
                        verb: why,
                        engine: "xla",
                    }
                })?);
            }
        }
        let mut media_of: Vec<Option<crate::serve::Media<'_>>> = vec![None; submission.lanes.len()];
        for (row, patches) in submission.media.iter().zip(&staged) {
            let Some(slot) = media_of.get_mut(row.lane as usize) else {
                return Err(Error::Invalid(format!(
                    "a media row names lane {} of the {} this fire has",
                    row.lane,
                    submission.lanes.len()
                )));
            };
            if slot.is_some() {
                return Err(Error::Invalid(format!(
                    "lane {} carries two media rows",
                    row.lane
                )));
            }
            *slot = Some(crate::serve::Media {
                rows: &row.rows,
                patches,
                routes: &row.routes,
                positions: &row.positions,
                grids: &row.grids,
                token_positions: &row.token_positions,
            });
        }
        let id = FireId(self.next_fire);
        self.next_fire = self.next_fire.wrapping_add(1);

        let device = self.device_lanes(submission)?;
        let mut resolved: Vec<Option<Vec<u32>>> = vec![None; submission.lanes.len()];
        let mut resolved_masks: Vec<Option<engine::fire::Masking>> =
            vec![None; submission.lanes.len()];
        let mut token_feeds: Vec<crate::serve::TokenFeed> = Vec::new();
        for attachment in &submission.attachments {
            let device_resolved = self.programs.geometry_of(attachment.instance)
                == Some(GeometryClass::DeviceGeometry);
            let lane_count = if device_resolved {
                1
            } else {
                self.programs
                    .lane_count(attachment.instance)
                    .map_err(refusal)?
            };
            for lane_at in 0..lane_count {
                let lane_index = attachment.lane as usize + lane_at;
                let Some(lane) = submission.lanes.get(lane_index) else {
                    return Err(Error::Invalid(format!(
                        "attachment names lane {} of a {}-lane step",
                        lane_index,
                        submission.lanes.len()
                    )));
                };
                if device_resolved {
                    continue;
                }
                resolved[lane_index] = match self
                    .programs
                    .envelope_tokens(attachment.instance, lane.tokens.len(), lane_at)
                    .map_err(refusal)?
                {
                    None => None,
                    Some(crate::program::Tokens::Host(ids)) => Some(ids),
                    Some(crate::program::Tokens::Device { out, words }) => {
                        let stand_in = vec![0; words.len()];
                        token_feeds.push(crate::serve::TokenFeed {
                            lane: lane_index,
                            src: out,
                            words,
                        });
                        Some(stand_in)
                    }
                };
                if lane.mask.is_none() {
                    resolved_masks[lane_index] = self
                        .programs
                        .envelope_mask(attachment.instance, lane.tokens.len(), lane_at)
                        .map_err(refusal)?;
                }
            }
        }
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

        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        for attachment in &submission.attachments {
            if let Some(bound) = self.adapters.get(&attachment.instance)
                && let Some(slot) = lane_adapters.get_mut(attachment.lane as usize)
            {
                *slot = Some(bound.slot);
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
                    "lane {at} is attached to an instance that bound an adapter, and this \
                     load's model text has no corrected class for its fact word {:#x}",
                    words[at]
                ))
            })?;
        }
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| Seated {
                lane: Lane {
                    slot: lane.slot,
                    word: words[at],
                    tokens: match &device[at] {
                        Some(resolved) => &resolved.tokens,
                        None => resolved[at].as_deref().unwrap_or(&lane.tokens),
                    },
                },
                pages: match &device[at] {
                    Some(DeviceLane {
                        pages: Some(pages), ..
                    }) => pages,
                    _ => &lane.kv.pages,
                },
                held: match &device[at] {
                    Some(DeviceLane {
                        have: Some(have), ..
                    }) => Some(*have),
                    _ => (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                },
                mask: device[at]
                    .as_ref()
                    .and_then(|resolved| resolved.mask.as_ref())
                    .or(resolved_masks[at].as_ref())
                    .or(lane.mask.as_ref()),
                adapter: lane_adapters[at].or(lane.adapter),
                positions: match &device[at] {
                    Some(DeviceLane {
                        positions: Some(positions),
                        ..
                    }) => positions,
                    _ => &lane.positions,
                },
                readout: match &lane.readout {
                    Readout::Rows(rows) => Some(rows.as_slice()),
                    Readout::Last | Readout::None => None,
                },
                rs_reset: lane.rs_reset,
                rs_slot: lane.rs_slot,
                rs: &verbs[at],
                media: media_of[at],
                inputs: &input_cells[at],
                group: lane.group,
                kv_less: lane.kv_less,
                writes: device[at]
                    .as_ref()
                    .and_then(|resolved| resolved.writes.as_ref())
                    .map(|(pages, offsets)| (pages.as_slice(), offsets.as_slice())),
                first: device[at].as_ref().and_then(|resolved| resolved.first),
                window: match &device[at] {
                    Some(DeviceLane {
                        window: Some(window),
                        ..
                    }) => window,
                    Some(DeviceLane { pages: Some(_), .. }) => &[],
                    _ => &lane.kv.window,
                },
                window_copies: &lane.kv.window_copies,
            })
            .collect();
        let clips: Vec<crate::serve::Clips<'_>> = submission
            .voxels
            .iter()
            .zip(&fed.voxels)
            .map(|(row, payload)| crate::serve::Clips {
                lane: row.lane,
                clips: &row.clips,
                payload,
            })
            .collect();
        // The readout stays on the device: guest passes read it there, and
        // the host copy is made only for a row something on the host reads.
        shell.feed_tokens(token_feeds);
        let fired = shell.fire_kept_with(&seated, &clips).map_err(fault)?;
        let kept = fired.kept.clone();
        // The step waits on its fire too, so a fault on the device reaches it.
        if let Some(plane) = kept.as_ref().and_then(|kept| kept.planes.first())
            && let Ok(ready) = plane.buffer.ready()
        {
            latch.hold();
            let hold = std::sync::Arc::clone(latch);
            if ready
                .on_ready(move |done| {
                    tracing::debug!("xla fire done");
                    hold.release(done.map_err(|e| e.to_string()));
                })
                .is_err()
            {
                latch.release(Ok(()));
            }
        }

        let attached: Vec<(InstanceId, u32)> = submission
            .attachments
            .iter()
            .filter(|attachment| !attachment.adapter_only)
            .map(|attachment| (attachment.instance, attachment.lane))
            .collect();
        let mut on_device = vec![false; submission.lanes.len()];
        if !attached.is_empty() {
            let device = self
                .shell
                .as_ref()
                .ok_or_else(|| Error::Load("the xla engine has no model loaded".into()))?
                .device();
            let ran = self
                .programs
                .fire_guests_with(
                    device,
                    &attached,
                    kept.as_deref(),
                    Some(latch),
                    Stage::Epilogue,
                )
                .map_err(refusal)?;
            for (&(_, lane), ran) in attached.iter().zip(ran) {
                if let Some(slot) = on_device.get_mut(lane as usize) {
                    *slot = ran;
                }
            }
        }

        let settle = StepSettle {
            kept,
            pixels: fired.pixels,
            on_device,
            policies: submission.lanes.iter().map(|l| l.readout.clone()).collect(),
        };
        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            settle,
        ))
    }
}

/// A settled step's readouts: per lane, its pixels, or the rows of its
/// read-back output (copied out of the kept planes now), or nothing when a
/// device guest read its readout rows.
fn readouts_of(settle: &StepSettle) -> EngineResult<Vec<LaneReadout>> {
    let StepSettle {
        kept,
        pixels,
        on_device,
        policies,
    } = settle;
    policies
        .iter()
        .enumerate()
        .map(|(lane, policy)| {
            // A lane that decoded clips answers its pixels.
            if let Some((values, boxes)) = pixels.get(lane).filter(|(_, boxes)| !boxes.is_empty()) {
                let voxels: usize = boxes
                    .iter()
                    .map(|[t, h, w]| *t as usize * *h as usize * *w as usize)
                    .sum();
                return Ok(LaneReadout {
                    rows: u32::try_from(voxels).unwrap_or(u32::MAX),
                    width: u32::try_from(values.len() / voxels.max(1)).unwrap_or(u32::MAX),
                    values: values.clone(),
                    output: kept
                        .as_ref()
                        .map(|kept| kept.pixels.0.clone())
                        .unwrap_or_default(),
                    clips: boxes.clone(),
                });
            }
            let Some(kept) = kept else {
                return Ok(LaneReadout::default());
            };
            // A lane whose guest ran on the device carries its answer in
            // its program's channels; its readout rows are not copied out.
            if matches!(policy, Readout::None)
                || on_device.get(lane).copied().unwrap_or(false) && kept.reads_readout(lane)
            {
                return Ok(LaneReadout::default());
            }
            let Some((output, rows, values)) = kept.lane_rows(lane).map_err(Error::Device)? else {
                return Ok(LaneReadout::default());
            };
            Ok(LaneReadout {
                output: output.to_string(),
                rows,
                width: u32::try_from(values.len() / rows.max(1) as usize).unwrap_or(u32::MAX),
                values,
                ..LaneReadout::default()
            })
        })
        .collect()
}
