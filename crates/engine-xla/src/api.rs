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
use engine::caps::{Capabilities, DeviceFacts, FireLimits};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, MemoryDomain};
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

    /// Device bytes the load leaves past its weights and pool for XLA's
    /// scratch; `None` takes the chip's (`chip::Tpu::headroom`).
    pub headroom: Option<u64>,

    /// The tokens a hybrid's state slot is balanced against the kv pages
    /// for (a sequence's expected prompt and output); `None`: the context.
    pub seat_tokens: Option<u32>,

    /// The load streams its streamable weights through the pool
    /// (`crate::store::streamed`): set by each load from
    /// `Residency::weights_held` (held: `false`, every plane lands as the
    /// base build lands it).
    pub stream_weights: bool,

    /// Chunks a load that streams nothing adds to its pool: a test that
    /// compares it with a streaming load at one pool size (the streamed
    /// weights' chunks).
    pub pool_extra_chunks: u32,
    /// The staging ring's pinned host slots: the frames a load keeps in
    /// flight, and one (`crate::store::stage`).
    pub staging_slots: u32,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            plugin: None,
            ordinal: 0,
            profiles: None,
            headroom: None,
            seat_tokens: None,
            stream_weights: false,
            pool_extra_chunks: 0,
            staging_slots: 2,
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
    /// The pending frame's step latches: under `PIE_XLA_SYNC` its settle
    /// waits on each and fails by name on a fault one reports.
    latches: Vec<std::sync::Arc<crate::program::Latch>>,
    programs: crate::program::Plane,
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: u64,
    next_frame: u64,
    sink: Option<engine::CompletionSink>,
    /// The adapter each instance bound: its id (first `Adapter` slot).
    adapters: BTreeMap<InstanceId, u32>,
    /// Per state record (its first piece's slot ref, [`record_key`]), how
    /// far a device-counted fold this engine resolved has frozen it: the
    /// runtime's `RsVerb::Buffer::folded` lags such a fold until its fire
    /// settles, so a fire staged before then replays from here, not from
    /// the stale count. A fork reads the record its parent wrote, so it
    /// finds the parent's; a device slot copy carries it; leaving the
    /// device with one unsettled is the runtime's to wait out.
    rs_ahead: BTreeMap<u64, u32>,
    /// Each set's written end as far as the fires admitted carried it,
    /// ahead of what the runtime states until they settle
    /// (`engine::extent`).
    written: engine::extent::Written,
    /// What the engine counts while it serves, read by the runtime: a
    /// program a fire waited on (Compiled), a frame or registration
    /// refused (Refused). Never Fallback: a guest runs on the device or is
    /// refused at its registration.
    /// Never Unrecorded: every serving fire runs a program, built if need
    /// be.
    live: engine::live::Live,
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
            rs_ahead: BTreeMap::new(),
            written: engine::extent::Written::default(),
            pending: None,
            latches: Vec::new(),
            live: engine::live::Live::armed(),
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

    /// Guest passes run on the device so far: every pass a guest takes.
    #[must_use]
    pub fn guest_passes(&self) -> u64 {
        self.programs.tally()
    }

    /// Guest stages compiled inside a fire, past their program's
    /// registration: zero where every registration compiled every shape a
    /// fire takes.
    #[must_use]
    pub fn guest_fire_compiles(&self) -> u64 {
        self.programs.fire_compiles()
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
    // A fault one lane earns refuses that lane alone; the step's other
    // lanes are the runtime's to submit again (`engine::Error::Lanes`).
    if let Some(lane) = fault.lane() {
        return Error::lane(lane, fault);
    }
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
        | Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskUnread { .. }
        | Fault::Positions { .. }
        | Fault::Adapterless { .. }
        | Fault::AdapterWord { .. }
        | Fault::Fire(_) => Error::Invalid(fault.to_string()),
        Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
    }
}

const DEFAULT_STATE_SLOTS: u32 = 256;

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

/// Lands the adapter an instance of `program` carries, its planes the
/// bind's seeds, in the `Adapter` slots the runtime took (`slots`): where
/// another instance landed the same adapter there it joins it. Its adapter
/// id, what its lanes route by; None where it carries none.
fn adapter_of(
    shell: &mut Shell,
    package: &eta_compiler::codegen::launch::LaunchPackage,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
    slots: &[u32],
) -> EngineResult<Option<u32>> {
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
    shell
        .land_adapter(instance, slots, &built)
        .map(Some)
        .map_err(fault)
}

impl Xla {
    /// [`Engine::submit`]'s frame, fired.
    fn submit_frame(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
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
        let mut latches = Vec::with_capacity(frame.steps.len());
        let mut refused = None;
        for (index, step) in frame.steps.iter().enumerate() {
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let latch = crate::program::Latch::new(
                at,
                u32::try_from(step.lanes.len()).unwrap_or(u32::MAX),
                self.sink.clone(),
            );
            // A step refused before it fires leaves no device fold's
            // position behind: what it noted of where its folds would leave
            // their states goes with it (its latched report is never told).
            let ahead = self.rs_ahead.clone();
            // Nor any set's written end.
            let written = self.written.clone();
            let fired = if step.lanes.iter().any(|l| !l.rs_snapshots.is_empty()) {
                self.fire_snapshotting(step, &latch)
            } else {
                self.fire_step(step, &latch)
            };
            let (ticket, settle) = match fired.inspect_err(|_| {
                self.rs_ahead = ahead;
                self.written = written;
            }) {
                Ok(both) => both,
                // A step's lanes refused past the first step: the frame is
                // the steps before it (`engine::Error::Lanes`); its latch is
                // never told.
                Err(error) if error.stops_frame_at(index) => {
                    refused = Some(error);
                    break;
                }
                Err(error) => return Err(error),
            };
            // The submitter's hold: the step is done once its passes land.
            latch.release(Ok(()));
            steps.push(ticket);
            settles.push(settle);
            latches.push(latch);
        }
        self.latches = latches;
        let mut ticket = FrameTicket::of(id, steps, refused);
        self.pending = Some((id, settles));
        tracing::debug!(frame = id.0, "xla submitted");
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    /// `result`, its refusal counted ([`engine::live::Tally::Refused`]).
    fn refused<T>(&self, result: EngineResult<T>) -> EngineResult<T> {
        if result.is_err() {
            self.live.tally(engine::live::Tally::Refused, 1);
        }
        result
    }
}

/// What a program's compile made off the engine's thread
/// (`Xla::prepare_program`).
struct Compiled {
    pending: crate::program::Pending,
    staged: std::result::Result<Vec<std::sync::Arc<crate::guest::run::Staged>>, String>,
}

impl Engine for Xla {
    fn kind(&self) -> &'static str {
        "xla"
    }

    fn live(&self) -> engine::live::Live {
        self.live.clone()
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
            frames_in_flight,
        } = request;
        let mut boot = self.boot.clone();
        // A fire's inputs wait in the staging ring until it runs: one slot
        // for each frame in flight, and the one being written.
        boot.staging_slots = u32::from(frames_in_flight.max(1)) + 1;
        boot.stream_weights = !residency.weights_held;
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
        let voxels = poem_compiler::VoxelLadder::of(
            &trace,
            budgets.max_voxels,
            budgets.max_clips,
            budgets.max_lanes,
        );
        let slots = poem_compiler::state_slots(budgets.slots, &trace, DEFAULT_STATE_SLOTS);
        let shell = Shell::load_with(
            Boot {
                patches,
                trace,
                contract: &contract,
                checkpoint: &path,
                budget: bake_budgets(&budgets),
                page_size: budgets.page_size,
                context: budgets.max_context,
                slots,
                kv_budget: residency.device_kv_budget,
                allowance: residency.device,
                device: &boot,
            },
            voxels,
        )
        .map_err(fault)?;
        let mut shell = shell;
        shell.set_host_programs(residency.host_programs);
        // Every maintenance program a fire, a fork or a park runs later, one
        // a slot size, compiled now whether or not the ladder runs: none is
        // built after the load.
        shell.warm_maintenance().map_err(fault)?;
        // The warm ladder, every fire shape the load serves, built before
        // ready: none is built after it (a fire outside it is a ladder miss,
        // named with its shape).
        shell.prewarm().map_err(fault)?;
        // The program a bind or a host pass lands guest cells with.
        let (rows, _) = shell.pool_geometry();
        crate::guest::run::cells_program(shell.device(), rows).map_err(Error::Load)?;
        // from here a program built is a fire the ladder missed
        shell.device().seal();
        let trace_name = shell.trace().name.clone();
        let (weight_bytes, pool_bytes) = shell.footprint();
        let paging = shell.paging();
        let drafter = shell.trace().drafter;
        let pools = shell.pool_facts();
        shell
            .seat_tiers(
                residency.host_kv_budget,
                residency.disk_kv_dir.as_deref().map(|dir| {
                    (
                        dir,
                        residency.disk_kv_budget.unwrap_or(0),
                        residency.disk_kv_persistent,
                    )
                }),
            )
            .map_err(fault)?;
        let chunks = shell.tiered_facts();
        shell.log_chunk_recipe(&chunks);
        if let engine::MixedLanes::Laid(rungs) = shell.mixed_lanes() {
            tracing::info!(
                pairs = ?rungs.iter().map(|r| (r.prompt_lanes, r.decode_lanes)).collect::<Vec<_>>(),
                max_tokens = budgets.max_tokens,
                max_lanes = budgets.max_lanes,
                "xla mixed fires laid: prompt_lanes -> decode_lanes"
            );
        }
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
                // its guest lowering (`guest::lower`) lowers a draw over a
                // top-p / top-k cut and a bf16 row's cut as one region each
                library_forms: engine::LibraryForms {
                    draw: true,
                    bf16_cut: true,
                },
            },
            pools,
            chunks,
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: u32::try_from(poem_compiler::arena::readouts_ceiling(&bake_budgets(
                    &budgets,
                )))
                .unwrap_or(u32::MAX),
                // Each lane's rung bounds what it replays.
                replayed_rows: 0,
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
            kv_handle: None,
            media_encode: false,
            rs_verbs: shell.serves_rs_verbs(),
            rs_snapshots: shell.snapshots(),
            bidirectional_attention: true,
            adapter_attachments: true,
            facts: shell.trace().facts.clone(),
            inputs: shell.input_decls(),
            classes: shell.classes(),
            // XLA states no bound on one submission.
            guest_cost: engine::GuestCost::default(),
            // No `beam_list`: a beam group's bind is refused by name.
            host_ops: engine::HostOps::NONE,
            mixed_lanes: shell.mixed_lanes(),
            // the staging ring's slots less the one being written
            frames_in_flight: engine::stated_frames(
                (shell.staging_facts().ring_slots as usize).saturating_sub(1),
            )?,
        };
        self.programs.seat(budgets.max_lanes);
        self.programs.seat_readouts(shell.readouts());
        let (rows, sink_row) = shell.pool_geometry();
        self.programs.seat_pool(rows, sink_row);
        self.programs.seat_planes(shell.guest_planes());
        // From here a program a fire waits on, a pass the host interprets,
        // is served worse than the load prepared: counted.
        shell.device().report_to(self.live.clone());
        self.programs.report_to(self.live.clone());
        shell.state_limits(caps.limits);
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
        let submitted = self.submit_frame(frame);
        // A frame whose step refused its lanes by name fired the steps
        // before it alone: a refusal too.
        if submitted
            .as_ref()
            .is_ok_and(|ticket| !ticket.refused.is_empty())
        {
            self.live.tally(engine::live::Tally::Refused, 1);
        }
        self.refused(submitted)
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
        // A settle in the submit (`PIE_XLA_SYNC`) waits on every step's
        // device events too: a fault one reports (a fire's ready event, a
        // guest pass's download) fails the frame by name, never dropped.
        if synchronous() {
            for (at, latch) in self.latches.iter().enumerate() {
                if let Some(why) = latch.wait() {
                    return Err(Error::Device(format!(
                        "frame {}'s step {at} faulted on the device: {why}",
                        ticket.id
                    )));
                }
            }
        }
        let (_, settles) = self.pending.as_ref().expect("checked just above");
        for (receipt, settle) in ticket.steps.iter_mut().zip(settles) {
            receipt.readouts = readouts_of(settle)?;
        }
        Ok(())
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        let device = self.shell.as_ref().map(|shell| shell.device());
        let registered = self
            .programs
            .register(registration, device)
            .map_err(refusal);
        self.forget_evicted();
        self.refused(registered)
    }

    /// The program checked and its stage shapes named here; its stages
    /// compile in the returned compile, off the engine's thread, while
    /// fires go on (`Plane::prepare`).
    /// The runtime's standard samplers' guest stages, compiled before ready
    /// at every batch rung exactly as their registrations compile them, so a
    /// request that registers one finds every stage compiled (the device's
    /// program cache holds them) and builds nothing after ready. The
    /// programs compile side by side, each stage admitted by the guest
    /// compile ledger (`guest::run`), so a cold boot is bounded by host
    /// memory, not run in turn. A program that does not lower or compile
    /// fails the load by name.
    fn prewarm_programs(&mut self, registrations: &[ProgramRegistration]) -> EngineResult<()> {
        let device = self
            .shell
            .as_ref()
            .map(Shell::device_shared)
            .ok_or_else(|| Error::Program("the standard samplers prewarmed before load".into()))?;
        let started = std::time::Instant::now();
        let mut pending = Vec::with_capacity(registrations.len());
        for registration in registrations {
            pending.push(self.programs.prepare(registration).map_err(refusal)?);
        }
        let compiled: Vec<Result<(usize, f64), String>> = std::thread::scope(|scope| {
            let jobs: Vec<_> = pending
                .iter()
                .map(|program| {
                    let device = &device;
                    scope.spawn(move || {
                        let at = std::time::Instant::now();
                        program
                            .compile(device)
                            .map(|staged| (staged.len(), at.elapsed().as_secs_f64()))
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| {
                    job.join()
                        .unwrap_or_else(|_| Err("a standard sampler's compile panicked".into()))
                })
                .collect()
        });
        let mut stages = 0;
        let mut slowest = 0f64;
        for (registration, done) in registrations.iter().zip(compiled) {
            let (count, seconds) = done.map_err(|why| {
                Error::Load(format!(
                    "the standard sampler program 0x{:016x}'s guest stages: {why}",
                    registration.program_hash
                ))
            })?;
            stages += count;
            slowest = slowest.max(seconds);
        }
        tracing::info!(
            programs = registrations.len(),
            stages,
            slowest_s = slowest,
            s = started.elapsed().as_secs_f64(),
            "xla standard samplers' guest stages compiled before ready"
        );
        Ok(())
    }

    fn prepare_program(
        &mut self,
        registration: &ProgramRegistration,
    ) -> EngineResult<engine::ProgramCompile> {
        let pending = self.programs.prepare(registration).map_err(refusal);
        let pending = self.refused(pending)?;
        let device = self.shell.as_ref().map(Shell::device_shared);
        Ok(Box::new(move || {
            let staged = match &device {
                Some(device) => pending.compile(device),
                None => Ok(Vec::new()),
            };
            engine::Prepared(Box::new(Compiled { pending, staged }))
        }))
    }

    /// The stages a prepared program's compile made held, and the program
    /// registered (`Plane::finish`).
    fn finish_program(&mut self, prepared: engine::Prepared) -> EngineResult<ProgramId> {
        let Ok(compiled) = prepared.0.downcast::<Compiled>() else {
            return Err(Error::Program(
                "a prepared program XLA did not prepare".to_string(),
            ));
        };
        let Compiled { pending, staged } = *compiled;
        let registered = self.programs.finish(pending, staged).map_err(refusal);
        self.forget_evicted();
        self.refused(registered)
    }

    fn close_program(&mut self, program: ProgramId) -> EngineResult<()> {
        self.programs.close_program(program).map_err(refusal)
    }

    /// XLA's readouts ([`engine::Engine::readout_facts`]): every output
    /// lands in a ring laid at load (`Pools::lay_readouts`), inside the
    /// bytes the load reserved, so no term beside the rungs; a sequence
    /// readout's ring holds the lane rung's rows a fire, the store the
    /// runtime cuts a fire's sequence lanes to. Float lanes' rows land in a
    /// ring of the top row rung's rows and are not bounded here.
    fn readout_facts(&self) -> EngineResult<engine::ReadoutFacts> {
        Ok(engine::ReadoutFacts {
            landing: engine::ReadoutLanding::InSpan,
            store_rows: self.shell.as_ref().and_then(Shell::readout_store_rows),
        })
    }

    /// The rings an instance keeps in the pool's `Channel` slots: its
    /// program's carried channels on the device; every other ring lives on
    /// the host. No `Instance` slots: the bind-time tables stay host-side.
    fn program_facts(&self, program: ProgramId) -> EngineResult<engine::program::ProgramFacts> {
        // The adapter its `lora` sink lands: one plane a bank row, an
        // `Adapter` slot each, taken once a content key and shared by
        // every holder.
        let sink = match self.programs.package(program) {
            Some(package) => crate::adapter::sink_of(package).map_err(fault)?,
            None => None,
        };
        let adapter = match (sink, self.shell.as_ref()) {
            (Some(sink), Some(shell)) => {
                let banks =
                    crate::adapter::plane_banks(&sink, &shell.bank_seats()).map_err(fault)?;
                engine::program::Footprint {
                    bytes: banks.len() as u64 * shell.pools().adapter_slot(),
                    slots: u32::try_from(banks.len()).unwrap_or(u32::MAX),
                    contiguous: false,
                }
            }
            _ => engine::program::Footprint::default(),
        };
        Ok(engine::program::ProgramFacts {
            rings: self.programs.compiled_form(program).map_or_else(Vec::new, |c| c.rings()),
            adapter,
            ..Default::default()
        })
    }

    /// The adapter a binding lands, keyed by its content
    /// ([`crate::adapter::adapter_digest`]): its sink's site word and, in
    /// the sink's plane order, each plane's bank role and seed.
    fn adapter_key(&self, binding: &InstanceBinding) -> EngineResult<Option<u128>> {
        let Some(package) = self.programs.package(binding.program) else {
            return Ok(None);
        };
        let Some(sink) = crate::adapter::sink_of(package).map_err(fault)? else {
            return Ok(None);
        };
        Ok(Some(crate::adapter::adapter_digest(
            sink.sites,
            sink.planes.iter().map(|(role, channel)| {
                let wire = binding
                    .seeds
                    .iter()
                    .find(|seed| seed.channel == *channel)
                    .map_or(&[][..], |seed| seed.bytes.as_slice());
                (role.bank(), wire)
            }),
        )))
    }

    /// A registered channel an instance may carry or feed on the device
    /// keeps its ring in the pool's `Channel` slots, the rows every instance
    /// bound to it reads; one only the host reads or writes takes none.
    /// Every registration: its ring on the host, mirrored where the host
    /// reads or writes it.
    fn holds_channel(&self, registration: &ChannelRegistration) -> bool {
        let _ = registration;
        true
    }

    fn ring_facts(
        &self,
        registration: &ChannelRegistration,
    ) -> EngineResult<Option<engine::program::RingFacts>> {
        if self.shell.is_none() {
            return Ok(None);
        }
        Ok(crate::program::Plane::ring_facts(registration))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        self.programs
            .register_channel(registration)
            .map_err(refusal)
    }

    /// What a fire holding lanes of `program` takes beside the model's own:
    /// its stages' temp past the load's largest (they share the runtime's
    /// one reservation) and their outputs, as registration compiled them.
    /// The same at any extents: every stage was compiled for every rung.
    fn fire_facts(
        &self,
        program: ProgramId,
        extents: &engine::program::BindExtents,
    ) -> EngineResult<engine::program::FireFacts> {
        let _ = extents;
        let load_temp = self.shell.as_ref().map_or(0, |shell| shell.device().load_temp());
        Ok(self.programs.fire_facts(program, load_temp))
    }

    /// An instance's carried rings, or a registered channel's ring, moved to
    /// fresh `Channel` slots (a park and its return): the cells are read and
    /// written there from now on, by every instance bound to them.
    fn rebind_slots(&mut self, rebind: engine::program::Rebind) -> EngineResult<()> {
        match rebind {
            engine::program::Rebind::Instance {
                id, rings, adapter, ..
            } => {
                if let (Some(held), Some(shell)) = (self.adapters.get_mut(&id), self.shell.as_mut())
                    && !adapter.is_empty()
                {
                    *held = shell.rebind_adapter(*held, &adapter).ok_or_else(|| {
                        Error::Program(format!(
                            "instance {id:?}'s adapter moved to slots {adapter:?}, which \
                             hold no adapter it landed"
                        ))
                    })?;
                }
                let shell = self.shell.as_ref();
                let cell_row = |slot: u32| shell.and_then(|shell| shell.channel_row(slot));
                self.programs.rebind(id, &rings, &cell_row).map_err(refusal)
            }
            // This engine claims nothing at an instance's slots and an
            // evacuation repoints none of a guest's: its next rebind names
            // where it lies, before any fire of it.
            engine::program::Rebind::Parked { .. } => Ok(()),
            engine::program::Rebind::Channel { id, slots } => {
                let shell = self.shell.as_ref();
                let cell_row = |slot: u32| shell.and_then(|shell| shell.channel_row(slot));
                (self.programs)
                    .rebind_channel(id, &slots, &cell_row)
                    .map_err(refusal)
            }
        }
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
        let shell = self.shell.as_ref();
        let cell_row = |slot: u32| shell.and_then(|shell| shell.channel_row(slot));
        let bound = self.programs.bind(binding, &cell_row).map_err(refusal)?;
        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        let package = self.programs.package(binding.program).cloned();
        let landed = match (package, self.shell.as_mut()) {
            (Some(package), Some(shell)) => {
                adapter_of(shell, &package, bound.id, &seeds, &binding.adapter_slots)
            }
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

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<u64> {
        if let (Some(held), Some(shell)) = (self.adapters.remove(&id), self.shell.as_mut()) {
            shell.release_adapter(id, held);
        }
        if let Some(shell) = self.shell.as_mut() {
            let (_, pool) = shell.device_and_pool();
            self.programs
                .flush_for_close(id, pool.as_ref())
                .map_err(refusal)?;
        }
        self.programs.close_instance(id).map_err(refusal)?;
        Ok(0)
    }

    /// Refused: an adapter is an instance's, landed at its bind in the
    /// pool's `Adapter` slots, never one the operator seats by id.
    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        Err(Error::Invalid(format!(
            "adapter {} registered by id: an adapter lands at the bind of the program \
             instance that carries it, in the pool's Adapter slots, keyed by its \
             content; this engine seats none by id",
            registration.id
        )))
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
        let shell = self.loaded_mut()?;
        let page_size = shell.paging().page_size;
        let moves = copy.runs(page_size)?;
        shell.copy_kv(&moves).map_err(fault)?;
        // Windowed slots a write into a shared one copies first, whole
        // (CONTRACT-BC 21, degenerate form; a pool states them as
        // `Lane::slot_copies`).
        if let Some(id) = shell.pools().window_shape() {
            let pairs: Vec<(engine::chunk::ShapeId, u32, u32)> = copy
                .windowed
                .iter()
                .map(|&(src, dst)| (id, src, dst))
                .collect();
            shell.copy_slots(&pairs).map_err(fault)?;
        }
        Ok(())
    }

    /// The pool is fixed at load: one segment, committed whole.
    fn commit_segments(&mut self, segments: u32) -> EngineResult<()> {
        if segments <= 1 {
            return Ok(());
        }
        Err(Error::Invalid(format!(
            "this pool is one segment fixed at load; {segments} cannot be committed"
        )))
    }

    fn release_segments(&mut self, segments: u32) -> EngineResult<()> {
        if segments >= 1 {
            return Ok(());
        }
        Err(Error::Invalid(
            "this pool is one segment fixed at load and is never released".into(),
        ))
    }

    /// Moves chunks within the fixed pool (the runtime does not ask it of a
    /// `Growth::Fixed` pool; served for a defragmenting caller).
    fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> EngineResult<()> {
        self.loaded_mut()?.evacuate(evacuation).map_err(fault)
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

    /// Whole-slot copies on the device. HOST/DISK TIERS HOOK: a copy whose
    /// source or destination is a host or disk chunk (`SlotCopy`'s tier,
    /// from sc/tier-shared's contract) lands here; until that contract is
    /// merged only device-to-device copies are served.
    fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> EngineResult<u64> {
        // This engine holds back none of the chunks it states
        // (`ChunkFacts::held_bytes` is 0 at load): the answer repeats it.
        self.loaded_mut()?
            .weight_residency(residency)
            .map(|()| 0)
            .map_err(fault)
    }

    fn weight_uses(&mut self) -> EngineResult<engine::chunk::WeightUses> {
        Ok(self.loaded_mut()?.weight_uses())
    }

    /// Its copies land inside the call (see [`Engine::copy_slots_landing`]):
    /// nothing to wait for.
    fn settle_copies(&mut self) -> EngineResult<()> {
        Ok(())
    }

    /// Lands inside the call: a move between tiers reads its source slots
    /// into host memory before it writes anywhere, and a move within the
    /// device runs as a pool program, ordered ahead of every program (a
    /// fire, a bind's seeds, a channel's cells) enqueued after it.
    fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        _token: u64,
    ) -> EngineResult<bool> {
        self.copy_slots(copy)?;
        Ok(true)
    }

    fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> EngineResult<()> {
        use crate::serve::Tier;
        let device = self.caps.as_ref().map(|caps| caps.device.domain);
        let tier = |domain| match domain {
            MemoryDomain::HostPinned => Some(Tier::Host),
            MemoryDomain::LocalDisk => Some(Tier::Disk),
            d if Some(d) == device => Some(Tier::Device),
            _ => None,
        };
        let (Some(from), Some(to)) = (tier(copy.src), tier(copy.dst)) else {
            return Err(Error::Unsupported {
                verb: "`copy_slots` to or from a domain other than the device, the host tier \
                       and the slot file",
                engine: "xla",
            });
        };
        // A state record copied carries where a device fold left it, keyed
        // by its tier too: each tier numbers its slot refs from 0, so a
        // host ref is not the device ref of the same number.
        for &(src, dst) in &copy.slots {
            let (src, dst) = (tier_key(from, copy.shape, src), tier_key(to, copy.shape, dst));
            match self.rs_ahead.get(&src).copied() {
                Some(at) => self.rs_ahead.insert(dst, at),
                None => self.rs_ahead.remove(&dst),
            };
        }
        let shell = self.loaded_mut()?;
        if from != Tier::Device || to != Tier::Device {
            // Between tiers (CONTRACT-BC 22): whole slots, laid alike.
            if !copy.runs.is_empty() {
                return Err(Error::Invalid(
                    "token runs move within the device; between tiers a slot moves whole".into(),
                ));
            }
            return shell
                .copy_tier(copy.shape, from, to, &copy.slots)
                .map_err(fault);
        }
        let pairs: Vec<(engine::chunk::ShapeId, u32, u32)> = copy
            .slots
            .iter()
            .map(|&(src, dst)| (copy.shape, src, dst))
            .collect();
        shell.copy_slots(&pairs).map_err(fault)?;
        if copy.runs.is_empty() {
            return Ok(());
        }
        if shell.pools().kv_shape() != Some(copy.shape) {
            return Err(Error::Unsupported {
                verb: "`copy_slots` token runs outside the kv list",
                engine: "xla",
            });
        }
        let runs: Vec<engine::transfer::KvRun> = copy
            .runs
            .iter()
            .map(|r| engine::transfer::KvRun {
                src_page: r.src,
                src_token: r.src_token,
                dst_page: r.dst,
                dst_token: r.dst_token,
                tokens: r.tokens,
            })
            .collect();
        shell.copy_kv(&runs).map_err(fault)
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
    /// One past the last slot the lane's rows land at, in its set's own
    /// slot numbering: its stated slots' highest, else what it held and its
    /// rows (`engine::KvWrites::end`).
    end: Option<u32>,
    /// The slots its rows land at, in its set's own numbering, where its
    /// program states them.
    slots: Option<Vec<u32>>,
}

impl Xla {
    /// Drops from the device's cache the compiled stages the guest stage
    /// cache evicted, so their memory is freed with them.
    fn forget_evicted(&mut self) {
        let evicted = self.programs.take_evicted();
        if let Some(shell) = &self.shell {
            for key in &evicted {
                shell.device().forget_key(key);
            }
        }
    }

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
        let window_shape = self
            .shell
            .as_ref()
            .and_then(|shell| shell.pools().window_shape());
        let kv_shape = self.shell.as_ref().and_then(|shell| shell.pools().kv_shape());
        for attached in &submission.attachments {
            if self.programs.geometry_of(attached.instance) != Some(GeometryClass::DeviceGeometry) {
                continue;
            }
            let Some(mut envelope) = self
                .programs
                .device_envelope(attached.instance)
                .map_err(refusal)?
            else {
                continue;
            };
            let first = attached.lane as usize;
            envelope.host_rows = lanes
                .iter()
                .skip(first)
                .take(attached.lanes.max(1) as usize)
                .map(|lane| lane.tokens.len() as u32)
                .collect();
            let carried = envelope.lanes();
            let owned = attached.lanes.max(1) as usize;
            if carried > owned {
                return Err(program(format!(
                    "instance {} is attached over {owned} lane(s) at lane {first} and its \
                     `embed_indptr` port describes {carried}: a program's ports state its own \
                     lanes' geometry, never the next request's",
                    attached.instance
                )));
            }
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
                let table: &[u32] = match lane.list {
                    engine::fire::ListKind::Translation => {
                        kv_shape.and_then(|id| table_of(lane, id)).unwrap_or(&[])
                    }
                    engine::fire::ListKind::Pages => &[],
                };
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
                        let windows = window_shape
                            .and_then(|id| table_of(lane, id))
                            .unwrap_or(&[]);
                        let window = (!windows.is_empty()).then(|| {
                            relative
                                .iter()
                                .map(|&page| windows.get(page as usize).copied().unwrap_or(0))
                                .collect::<Vec<u32>>()
                        });
                        // A row lands only in what the lane may write: below
                        // its floor is a frozen prefix other sets read, and
                        // past its reservation is not its own.
                        if let Some(why) = engine::fire::misplaced_write(lane.kv.writable(), slots)
                        {
                            return Err(Error::lane(source as u32, why));
                        }
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
                    () => holds_kv(lane, kv_shape).then_some(lane.kv.held),
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
                let end = match slots {
                    Some(slots) => slots.iter().max().map(|&slot| slot + 1),
                    None => have.map(|have| have + rows as u32),
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
                    end,
                    slots: slots.map(<[u32]>::to_vec),
                });
            }
        }
        Ok(out)
    }

    /// Runs each attached prologue first: the forward reads the cells a prologue put.
    fn run_prologues(
        &mut self,
        submission: &engine::fire::Step,
        latch: &std::sync::Arc<crate::program::Latch>,
    ) -> EngineResult<()> {
        let prologues: Vec<(InstanceId, u32, u32)> = submission
            .attachments
            .iter()
            .filter(|attachment| {
                !attachment.adapter_only && self.programs.prologue_runs(attachment.instance)
            })
            .map(|attachment| (attachment.instance, attachment.lane, attachment.lanes.max(1)))
            .collect();
        if prologues.is_empty() {
            return Ok(());
        }
        let (device, pool) = self
            .shell
            .as_mut()
            .ok_or_else(|| Error::Load("the xla engine has no model loaded".into()))?
            .device_and_pool();
        let (_, faults) = self
            .programs
            .fire_guests_with(device, &prologues, None, pool, None, Stage::Prologue)
            .map_err(refusal)?;
        for (instance, why) in faults {
            latch.fault_instances(&[instance], &why);
        }
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

    /// A step whose lanes snapshot their recurrent state inside the pass,
    /// fired in pieces (`crate::snapshot`): the last piece's ticket.
    fn fire_snapshotting(
        &mut self,
        step: &engine::fire::Step,
        latch: &std::sync::Arc<crate::program::Latch>,
    ) -> EngineResult<(FireTicket, StepSettle)> {
        let shell = self.shell.as_ref().ok_or_else(|| {
            Error::Load("the xla engine has no model loaded; load one first".into())
        })?;
        let records: Vec<engine::chunk::ShapeId> =
            shell.pools().fixed_shapes().map(|(id, _)| id).collect();
        let bits = crate::snapshot::Bits::of(&shell.trace().facts);
        let pieces = crate::snapshot::pieces(step, bits, &records).map_err(fault)?;
        let last = pieces.len() - 1;
        for (at, piece) in pieces.into_iter().enumerate() {
            let fired = self.fire_step(&piece.step, latch)?;
            if !piece.copies.is_empty() {
                self.loaded_mut()?
                    .copy_slots(&piece.copies)
                    .map_err(fault)?;
            }
            if at == last {
                return Ok(fired);
            }
        }
        unreachable!("a step cuts into one piece at least")
    }

    fn fire_step(
        &mut self,
        submission: &engine::fire::Step,
        latch: &std::sync::Arc<crate::program::Latch>,
    ) -> EngineResult<(FireTicket, StepSettle)> {
        capture(submission);
        // Each attached instance's lanes: where its pass's fault lands.
        latch.seat_spans(
            (submission.attachments.iter())
                .map(|a| (a.instance, a.lane..a.lane + a.lanes.max(1)))
                .collect(),
        );
        self.run_prologues(submission, latch)?;
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
        // A page holds what its last owner left until its set writes it: no
        // lane reads past what its set holds written (`engine::extent`).
        let kv_shape = self.shell.as_ref().and_then(|shell| shell.pools().kv_shape());
        let mut extents = engine::extent::FireExtents::default();
        for (at, lane) in submission.lanes.iter().enumerate() {
            if lane.kv_less {
                continue;
            }
            let set = (lane.kv.set, lane.kv.readable);
            match device[at].as_ref() {
                Some(DeviceLane {
                    have: Some(have),
                    slots,
                    tokens,
                    ..
                }) => {
                    let reach = have.saturating_add(tokens.len() as u32);
                    match slots {
                        Some(slots) => extents.lane(
                            &self.written,
                            at,
                            lane.slot,
                            set,
                            slots.iter().map(|&slot| slot..slot.saturating_add(1)),
                            reach,
                        ),
                        None => extents.lane(
                            &self.written,
                            at,
                            lane.slot,
                            set,
                            std::iter::once(*have..reach),
                            reach,
                        ),
                    }
                }
                Some(_) => {}
                None if holds_kv(lane, kv_shape) => {
                    let reach = lane.kv.held.saturating_add(lane.rows());
                    extents.lane(
                        &self.written,
                        at,
                        lane.slot,
                        set,
                        std::iter::once(lane.kv.held..reach),
                        reach,
                    );
                }
                None => {}
            }
        }
        extents.check().map_err(Error::Program)?;
        self.written.commit(extents);
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
                    Some(crate::program::Tokens::Device { at }) => {
                        let stand_in = vec![0; at.len()];
                        token_feeds.push(crate::serve::TokenFeed {
                            lane: lane_index,
                            at,
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
        // The lanes whose fold length was the device's, resolved here.
        let mut device_folds = vec![false; submission.lanes.len()];
        // A state a device-counted fold froze past what the runtime counts
        // replays from where the fold left it.
        for (lane, verb) in submission.lanes.iter().zip(verbs.iter_mut()) {
            let Some((read, write)) = record_key(lane) else {
                continue;
            };
            // A state starting afresh holds nothing an earlier fold froze.
            let fresh = match lane.rs_reset {
                engine::fire::RsReset::Fresh => true,
                engine::fire::RsReset::Inferred => lane.kv.held == 0,
                engine::fire::RsReset::Held => false,
            };
            if fresh {
                self.rs_ahead.remove(&write);
                continue;
            }
            if let engine::fire::RsVerb::Buffer { folded, .. } = verb {
                match self.rs_ahead.get(&read).copied() {
                    Some(ahead) if ahead > *folded => *folded = ahead,
                    Some(_) if read == write => {
                        self.rs_ahead.remove(&read);
                    }
                    _ => {}
                }
            }
        }
        // A reporting lane with no recurrent state to fold (a KV-only set)
        // is folded by the count its program's `fold` port states just the
        // same: the count, as read, is its report's fold.
        let mut kv_folds: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        for attachment in &submission.attachments {
            let at = attachment.lane as usize;
            match &mut verbs[at] {
                engine::fire::RsVerb::Buffer { fold, .. } if *fold == engine::fire::FoldLen::Device => {
                    if let Some(n) = self
                        .programs
                        .envelope_fold(attachment.instance)
                        .map_err(refusal)?
                    {
                        *fold = engine::fire::FoldLen::Host(n);
                        device_folds[at] = true;
                    }
                }
                engine::fire::RsVerb::Buffer { .. } => {}
                _ if submission.lanes.get(at).is_some_and(|lane| lane.reports != 0) => {
                    kv_folds[at] = self
                        .programs
                        .envelope_fold(attachment.instance)
                        .map_err(refusal)?;
                }
                _ => {}
            }
        }
        // Where each device-counted fold leaves its state, for the fires
        // staged before this one settles.
        for (at, lane) in submission.lanes.iter().enumerate() {
            if let (
                engine::fire::RsVerb::Buffer {
                    folded,
                    fold: engine::fire::FoldLen::Host(n),
                    ..
                },
                Some((_, slot)),
                true,
            ) = (&verbs[at], record_key(lane), device_folds[at])
            {
                let first = device
                    .get(at)
                    .and_then(Option::as_ref)
                    .and_then(|p| p.first.or(p.have))
                    .unwrap_or(lane.kv.held);
                let to = folded
                    .saturating_add(*n)
                    .min(first + lane.tokens.len() as u32);
                self.rs_ahead.insert(slot, to);
            }
        }
        // What each reported lane wrote, told when the step lands: one past
        // the last slot its rows landed at (the runtime bounds what a
        // device-placed lane carries by it, `engine::fire::rung_rows`), and
        // where its fold length was the device's, the slots this fire's fold
        // froze, clamped to its rows' end as the plan clamps it
        // (`engine::rs::LanePlan::of`).
        latch.report(
            submission
                .lanes
                .iter()
                .enumerate()
                .filter(|(_, lane)| lane.reports != 0)
                .map(|(at, lane)| {
                    let rows = lane.tokens.len() as u32;
                    let placed = device.get(at).and_then(Option::as_ref);
                    let first = placed
                        .and_then(|p| p.first.or(p.have))
                        .unwrap_or(lane.kv.held);
                    let end = match placed {
                        Some(placed) => placed.end,
                        None => (!lane.kv_less).then_some(first + rows),
                    };
                    let fold = match &verbs[at] {
                        engine::fire::RsVerb::Buffer {
                            folded,
                            fold: engine::fire::FoldLen::Host(n),
                            ..
                        } if device_folds[at] => {
                            let to = folded.saturating_add(*n).min(first + rows);
                            Some(to.saturating_sub(*folded))
                        }
                        _ => kv_folds[at],
                    };
                    engine::KvWrites {
                        tag: lane.reports,
                        fold,
                        end,
                        ..engine::KvWrites::default()
                    }
                })
                .collect(),
        );

        let mut lane_adapters: Vec<Option<u32>> = vec![None; submission.lanes.len()];
        for attachment in &submission.attachments {
            if let Some(bound) = self.adapters.get(&attachment.instance)
                && let Some(slot) = lane_adapters.get_mut(attachment.lane as usize)
            {
                *slot = Some(*bound);
            }
        }
        let (kv_shape, window_shape) = self.shell.as_ref().map_or((None, None), |shell| {
            (shell.pools().kv_shape(), shell.pools().window_shape())
        });
        let shell = self.loaded_mut()?;
        // A load that holds state is told every lane's record: its pieces,
        // or none (`LaneRecord::Null`, which names no piece).
        if shell.state_slot_bytes() > 0
            && let Some(lane) = submission
                .lanes
                .iter()
                .find(|lane| matches!(lane.record, engine::fire::LaneRecord::Unstated))
        {
            return Err(Error::Invalid(format!(
                "lane in slot {} states no record, and this load holds recurrent state: the \
                 runtime states every lane's record, or that it has none",
                lane.slot
            )));
        }
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
                    _ => kv_shape.and_then(|id| table_of(lane, id)).unwrap_or(&[]),
                },
                held: match &device[at] {
                    Some(DeviceLane {
                        have: Some(have), ..
                    }) => Some(*have),
                    _ => holds_kv(lane, kv_shape).then_some(lane.kv.held),
                },
                folded: lane.kv.folded,
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
                quiet: matches!(lane.readout, Readout::None),
                rs_reset: lane.rs_reset,
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
                    _ => window_shape
                        .and_then(|id| table_of(lane, id))
                        .unwrap_or(&[]),
                },
                fixed: lane.record.pieces(),
                copies: &lane.slot_copies,
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

        let attached: Vec<(InstanceId, u32, u32)> = submission
            .attachments
            .iter()
            .filter(|attachment| !attachment.adapter_only)
            .map(|attachment| (attachment.instance, attachment.lane, attachment.lanes.max(1)))
            .collect();
        let mut on_device = vec![false; submission.lanes.len()];
        if !attached.is_empty() {
            let (device, pool) = self
                .shell
                .as_mut()
                .ok_or_else(|| Error::Load("the xla engine has no model loaded".into()))?
                .device_and_pool();
            let (ran, faults) = self
                .programs
                .fire_guests_with(
                    device,
                    &attached,
                    kept.as_deref(),
                    pool,
                    Some(latch),
                    Stage::Epilogue,
                )
                .map_err(refusal)?;
            for (instance, why) in faults {
                latch.fault_instances(&[instance], &why);
            }
            for (attachment, ran) in submission
                .attachments
                .iter()
                .filter(|attachment| !attachment.adapter_only)
                .zip(ran)
            {
                for k in 0..attachment.lanes.max(1) {
                    if let Some(slot) = on_device.get_mut((attachment.lane + k) as usize) {
                        *slot = ran;
                    }
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

/// Lane `lane`'s table of list shape `id` (`Lane::tables`), when it states
/// one.
fn table_of(lane: &engine::fire::Lane, id: engine::chunk::ShapeId) -> Option<&[u32]> {
    lane.tables
        .iter()
        .find(|(held, _)| *held == id)
        .map(|(_, table)| table.as_slice())
}

/// A device slot ref of shape `shape` as one key.
fn ref_key(shape: engine::chunk::ShapeId, slot: u32) -> u64 {
    (u64::from(shape.0) << 32) | u64::from(slot)
}

/// A slot ref of shape `shape` on `tier` as one key: a device ref's is its
/// [`ref_key`], what a lane's record reads; the host tier's and the slot
/// file's stand apart from it.
fn tier_key(tier: crate::serve::Tier, shape: engine::chunk::ShapeId, slot: u32) -> u64 {
    let tag: u64 = match tier {
        crate::serve::Tier::Device => 0,
        crate::serve::Tier::Host => 1,
        crate::serve::Tier::Disk => 2,
    };
    (tag << 62) | ref_key(shape, slot)
}

/// The state record a lane reads and the one it writes: its first fixed
/// shape's first piece, `[run, cow]` (a fork's `cow` differs); none where
/// it holds no record.
fn record_key(lane: &engine::fire::Lane) -> Option<(u64, u64)> {
    let (id, &[run, cow]) =
        lane.record.pieces().first().and_then(|(id, pieces)| Some((*id, pieces.first()?)))?;
    Some((ref_key(id, run), ref_key(id, cow)))
}

/// Whether the lane holds kv pages: its kv list's table names any.
fn holds_kv(lane: &engine::fire::Lane, kv: Option<engine::chunk::ShapeId>) -> bool {
    kv.and_then(|id| table_of(lane, id)).is_some_and(|table| !table.is_empty())
}

#[cfg(test)]
mod tier_key_tests {
    use super::{ref_key, tier_key};
    use crate::serve::Tier;

    /// A host or slot-file ref never keys as the device ref of the same
    /// number, and a device ref keys as what a lane's record reads.
    #[test]
    fn a_tier_keys_its_refs_apart() {
        let shape = engine::chunk::ShapeId((1 << 17) + 2);
        assert_eq!(tier_key(Tier::Device, shape, 3), ref_key(shape, 3));
        assert_ne!(tier_key(Tier::Host, shape, 3), ref_key(shape, 3));
        assert_ne!(tier_key(Tier::Disk, shape, 3), ref_key(shape, 3));
        assert_ne!(tier_key(Tier::Host, shape, 3), tier_key(Tier::Disk, shape, 3));
    }
}
