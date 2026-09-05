//! `impl Engine for Cuda` — the shell behind the contract.
//!
//! [`Cuda`] wraps a `Shell` that has not happened yet: the device knobs a
//! boot config states, an `Option<Shell>` that `load` fills, and the
//! [`Capabilities`] that load answered. Every verb before a load is a
//! refusal with a sentence. [`LoadRequest`] carries no `ModelContract`, so
//! the contract lookup is a function pointer supplied when the engine is
//! opened, by the party that already links the model catalog. `encode`
//! answers [`Error::Unsupported`]: this shell carries no multimodal encoder.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, HostMirror, RegisteredChannel};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout, Step,
};
use engine::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use engine::transfer::{KvCopy, MemoryDomain, StateCopy};
use eta_ir::registry::{GeometryClass, ModelProfile, Port, PortMask};
use eta_ir::types::Dtype;
use model_compiler::{Budget, DeviceProfile, PATCH_LATTICE_FLOOR, PatchLadder};
use model_ir::Trace;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Graphs, Knobs, Lane, Seated, Shell};

/// How a caller answers "what does this checkpoint's bytes mean for this
/// plan" — supplied by the party that links the model catalog, since this
/// crate must not know a model family.
pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// The catalog's classifier for a SKU name, or `None` for one it does not ship.
pub type ClassifyFor = fn(&str) -> Option<model_ir::ClassifyFn>;

/// The device knobs a boot config states, before any model is loaded.
/// Machine/deployment properties; the model's own ceilings arrive later on
/// [`LoadRequest::budgets`].
// `Eq` left with [`Knobs`]'s: the knobs carry an `f64`, and total equality
// over a float is a claim neither struct should make.
/// Which rank of how wide a tensor-parallel group this shell is. A single
/// device is rank 0 of 1.
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

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    /// Which device to bind.
    pub ordinal: i32,
    /// This shell's rank and the group's width: the band of every cut weight
    /// it lands, and whether the plan's collectives have peers.
    pub world: World,
    /// The group's communicator for this rank, opened by
    /// [`open_group`](crate::open_group); `None` on a single device.
    pub comm: Option<Arc<crate::comm::Comm>>,
    /// How much of a fire to record, from `[engine] graphs`.
    pub graphs: Graphs,
    /// The shell's own words, from the boot document's `[engine]` table.
    /// Stated once when the engine is opened and carried onto every
    /// [`Boot`] it makes.
    pub knobs: Knobs,
    /// Where this deployment keeps its caches, from `[cache] dir` — the
    /// root; each consumer joins its own subdirectory. `None` disables the
    /// cache: every kernel compiles through NVRTC and nothing is stored.
    pub cache_dir: Option<std::path::PathBuf>,
    /// Where this deployment keeps its shared adapters, from
    /// `model.adapter_dir` — a read-only directory of adapter
    /// subdirectories. `None` disables shared binds.
    pub adapter_dir: Option<std::path::PathBuf>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            ordinal: 0,
            world: World::default(),
            comm: None,
            graphs: Graphs::default(),
            knobs: Knobs::default(),
            cache_dir: None,
            adapter_dir: None,
        }
    }
}

/// The CUDA shell, behind [`Engine`].
/// One submitted step, held for a caller that comes back for numbers: the
/// readback plan the shell computed at `settle`, plus the per-lane
/// `Readout` policy the submission stated. Carried here and handed down at
/// the numbers door because the shell is the only party that knows where a
/// lane's row run sits in the arena rectangle.
struct PendingStep {
    readout: Vec<Readout>,
    settled: crate::serve::Settled,
}

pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    classify_for: ClassifyFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    /// The host end of every registered channel, keyed by the caller's
    /// channel id (an instance names them by dense index instead —
    /// `InstanceBinding::channels` is the map between the two). Control
    /// plane: nothing here is touched between `bind_instance` and
    /// `close_channel`.
    channels: BTreeMap<ChannelId, Arc<crate::program::Endpoint>>,
    /// Where step completions go, installed once by the thread that owns
    /// this engine. `None` is a caller that does not want to hear.
    sink: Option<engine::CompletionSink>,
    /// The last submitted frame's per-step readback plans. One frame's
    /// worth and no more: the arena rectangles a frame's numbers live in
    /// are carved over by the next fire, so holding two would answer with
    /// bytes the device has overwritten.
    pending: Option<(FrameId, Vec<PendingStep>)>,
    /// Which adapter slot each bound instance routes to. Landed once, at
    /// [`Engine::bind_instance`], never at fire time; an instance with no
    /// adapter is absent from this map. Removed at `close_instance`, so a
    /// slot is reclaimable.
    adapters: BTreeMap<InstanceId, crate::Binding>,
}

impl Cuda {
    /// An engine bound to nothing yet.
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor, classify_for: ClassifyFor) -> Cuda {
        Cuda {
            boot,
            contract_for,
            classify_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            channels: BTreeMap::new(),
            sink: None,
            pending: None,
            adapters: BTreeMap::new(),
        }
    }

    /// The loaded shell, for a caller that wants the native surface — the
    /// A/B mode switch, the graph statistics, the footprint.
    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    /// The loaded shell, mutably.
    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    /// What this load can do, once there is one.
    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    /// Open a slot for a fresh sequence. Not a trait verb: a shell that
    /// owns a lane's page table (`KvDelta::pages` empty) opens the slot
    /// itself; a runtime with its own page table never calls this.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] before a load, and whatever the pools said.
    pub fn open(&mut self, slot: u32) -> EngineResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    /// One bound instance's session, for the two channel doors.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] before a load, [`Error::Closed`] for an instance
    /// this plane does not carry.
    /// This shell's rank in its tensor-parallel group (0 when alone).
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.boot.world.rank
    }

        /// Rank 0's endpoint for channel `id`, for a follower to adopt.
    #[must_use]
    pub fn endpoint(&self, id: ChannelId) -> Option<Arc<crate::program::Endpoint>> {
        self.channels.get(&id).cloned()
    }

    /// A tensor-parallel follower's `register_channel`: the host end is rank
    /// 0's, shared. This rank's passes pull the guest's cells out of it and
    /// never write a word or a cell of it (their sessions bind as shadows).
    ///
    /// # Errors
    ///
    /// [`Error::Program`] for an id already registered here.
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
            // The mirror is published by rank 0's answer.
            mirror: None,
        })
    }

    /// Every bound instance's predicted channel cursors, in instance order.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] before a load.
    pub fn channel_predictions(&mut self) -> EngineResult<Vec<(u64, Vec<crate::program::Cursor>)>> {
        Ok(self.loaded_mut()?.channel_predictions())
    }

    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id,
            })
    }

    /// What a load settles before it binds a device: the model's own load
    /// contract, and the residency plan the two budgets decided, with the
    /// tier admission that plan is checked against. Nothing here touches a
    /// device or allocates.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] for a checkpoint this build has no import contract
    /// for, and [`Error::Impossible`] for a budget pair no plan meets.
    fn settle(
        &self,
        trace: &Trace,
        path: &Path,
        residency: &engine::Residency,
    ) -> EngineResult<(ModelContract, crate::experts::Plan)> {
        let contract = (self.contract_for)(trace, path).map_err(Error::Load)?;
        // Residency, before a byte is allocated. Two tiers: routed expert
        // banks may live in a device slab of `n < experts` slots over a
        // pinned host copy behind a device-resident indirection table;
        // everything else is resident, whole. `Plan::of` reads the budget
        // against the trace alone and answers the residency this load will
        // have; `Residency::admit` is then asked what the plan demands, not
        // what the checkpoint holds, since the host demand (every expert of
        // every streamed bank) is the real question. A packed bank neither
        // budget holds is planned onto the mapped artifact instead of
        // refusing the load, if that artifact exists.
        let prospect =
            crate::weights::prospect(trace, &contract, path, self.target()).map_err(fault)?;
        let plan = crate::experts::Plan::cut(
            &prospect.ranking,
            crate::experts::Budgets {
                device: residency.device_weight_budget,
                host: residency.host_weight_budget,
            },
        )
        .map_err(fault)?;
        // The model's own `.zt` holds every plane of the trace, so it is
        // the one file that can be the spill source. `Serving::open` takes
        // the contract because some planes carry the SKU's own names and
        // others the checkpoint's, and it is the dictionary between the
        // two. This predicate and `weights::resident`'s selection must ask
        // exactly the same question, or a load is admitted here and finds
        // nothing there.
        let sourced = plan.spill_demand() > 0
            && crate::checkpoint_serving::Serving::open(path, trace).is_some();
        residency.admit_tiers(engine::load::Tiers {
            device: plan.device_demand(),
            host: plan.host_demand(),
            spilled: plan.spill_demand(),
            sourced,
        })?;
        Ok((contract, plan))
    }

    /// The band of the checkpoint this rank lands: rank `world.rank` of
    /// `world.size` (every `Shard::Cut` segment whole on a single device).
    fn target(&self) -> checkpoint::plan::StorageTarget {
        checkpoint::plan::StorageTarget::for_backend(
            checkpoint::types::BackendKind::Cuda,
            self.boot.world.rank,
            self.boot.world.size,
        )
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the cuda engine has no model loaded".into()))
    }
}

/// The shell's refusal, in the contract's vocabulary. `Exhausted` and
/// `Impossible` are scheduling answers the runtime's lane loop acts on;
/// everything else is a failure it logs. A [`Fault::Ceiling`] is
/// `Impossible`, not `Exhausted`, because every ceiling was reserved at load.
fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Runtimeless | Fault::Device { .. } => Error::Device(fault.to_string()),
        // A region baked behind a conditional node, or a body that answers
        // something other than its own walk, is a fault caught at boot.
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Param { .. }
        | Fault::Unbound { .. }
        | Fault::Unlowered { .. }
        | Fault::Golden { .. } => Error::Load(fault.to_string()),
        // A mount that lies is load-class: the fix is on disk.
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        Fault::OutOfMemory { need, have } => Error::Exhausted {
            resource: "device memory",
            wanted: need,
            available: have,
        },
        // A budget the tiers cannot meet is not a pool the deployment can
        // free its way out of.
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        Fault::Fragmented { .. } | Fault::Integrity { .. } => Error::Device(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        // The caller stated it and can state it differently: a retry with a
        // mask of the lane's own extent is a real answer.
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. } => Error::Invalid(fault.to_string()),
        // A registration the banks cannot seat is `Load`: capacity is model
        // text, so nothing the caller frees makes room.
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Draftless { .. }
        | Fault::DraftWord { .. }
        | Fault::Scoreless { .. }
        | Fault::ScoreWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        // The one adapter-axis refusal a caller can clear: a bind that
        // waits for another instance to finish is a real answer.
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) | Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling — field for field,
/// so adding a role to one without the other is a compile error.
fn extents(stated: &BindExtents) -> eta_exec::Extents {
    eta_exec::Extents {
        kv_len: stated.kv_len,
        page_count: stated.page_count,
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The lattice a deployment gets when it states none: powers of two from
/// [`LATTICE_FLOOR`] up to and including `max_tokens`, so a fire above the
/// floor never computes more than twice its own rows.
#[must_use]
pub(crate) fn default_lattice(max_tokens: u32) -> Vec<u32> {
    let mut lattice: Vec<u32> =
        core::iter::successors(Some(LATTICE_FLOOR), |point| point.checked_mul(2))
            .take_while(|point| *point < max_tokens)
            .collect();
    lattice.push(max_tokens);
    lattice
}

/// Where the default lattice starts. One, so a lone decode lane runs its
/// linear layers at M=1 and takes the gemv arm the tuner picks for it,
/// rather than riding an eight-row tile.
pub(crate) const LATTICE_FLOOR: u32 = 1;

/// The shape lattice policy, at the door: a caller that stated a lattice
/// keeps it exactly; one that stated none gets [`default_lattice`], since
/// the empty lattice is not a fit default. Idempotent by construction.
#[must_use]
pub(crate) fn lattice(stated: Vec<u32>, max_tokens: u32) -> Vec<u32> {
    if stated.is_empty() {
        default_lattice(max_tokens)
    } else {
        stated
    }
}

/// The ceilings the compiler bakes against, out of the ones the load
/// states. The other three fields (`page_size`, `max_context`, `slots`) are
/// the pools' and go to `Boot` directly.
fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// The second row axis's ladder, derived from the text that needs one.
/// `None` for a plan with no `Dim::Patches`. Floor is
/// [`PATCH_LATTICE_FLOOR`]; ceiling is the token rectangle's, capped at two
/// whole images; rungs double. `pub` so a gate can boot a tower against the
/// ladder this engine would derive for it.
pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    /// Two whole images at the catalog towers' native 48 x 48 grid.
    const DERIVED_PATCH_CEILING: u32 = 4096;

    // The plan is what asks: read off the types a text already wrote, never
    // off a flag.
    let declares_patches = trace.values.iter().any(|decl| {
        matches!(&decl.ty, model_ir::Ty::Tensor { shape, .. }
            if shape.first().and_then(|dim| dim.axis()) == Some(model_ir::RowAxis::Patches))
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
            .unwrap_or_else(|| max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

/// The guest-visible profile of a loaded plan, read off the plan and the
/// budgets (`num_layers` from node `layer` stamps, `vocab` from the `out`
/// seam's width) rather than reconstructed from capability flags.
fn profile(shell: &Shell, budgets: &LoadBudgets) -> EngineResult<ModelProfile> {
    let trace = shell.trace();
    let layers = trace
        .nodes
        .iter()
        .filter_map(|node| node.layer)
        .max()
        .map_or(0, |top| top + 1);
    let vocab = u32::try_from(shell.out_width().map_err(fault)?).unwrap_or(u32::MAX);
    Ok(ModelProfile {
        vocab,
        page_size: budgets.page_size,
        num_layers: layers,
        // Interpreter-visible materialization: the device's own activation
        // type is bf16 and is not what a guest program reads.
        activation: Dtype::F32,
        // Does this load's model text declare a draft head.
        has_mtp_logits: shell.drafts(),
        // `MtpDrafts` is `[k]` i32 token ids, an argmax the guest can take
        // for itself off `MtpLogits`; no device path here produces it.
        mtp_depth: 0,
        draft_block: 0,
        draft_mask_token: 0,
        draft_bidirectional: false,
        draft_proposals_from: 1,
        has_value_head: false,
        // Does this load export a capture column, and did the slab that
        // observes it get carved.
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        // `true` because the sink is consumed: `Cuda::bind_instance` lands
        // seeded weights in a slot, `Cuda::fire_step` stamps it onto every
        // attached lane. A load with no bank refuses by name
        // (`Fault::Adapterless`).
        has_lora: true,
        kernels: Vec::new(),
    })
}

/// One instance's adapter, landed off its seeds. `Ok(None)`: this program
/// declares no `lora` sink. `Ok(Some(binding))`: the weights crossed once,
/// here, on the host, naming the slot every attached lane routes to.
///
/// Reads the seed, not the ring: a seeded channel's cell is already on
/// this side of the wire at bind, so a guest publishing new weights
/// mid-pass is not serving a new adapter — swapping one is a re-bind.
///
/// # Errors
///
/// [`Error::Load`] for a sink channel this bind seeded nothing into;
/// [`Fault::AdapterSlots`] when every slot is pinned; [`Fault::Adapter`]
/// when the planes are not this load's banks.
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
    // Which site the guest asked for, checked against the banks by `planes_of`.
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        // A channel the sink names and the bind did not seed is a refusal,
        // not a plane of zeros: a zero `A` is the identity adapter, and
        // accepting an unseeded channel would silently answer the base model.
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

impl Engine for Cuda {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    /// The lane thread says it is the one now. `Shell::load` binds the
    /// device on the boot thread; the runtime then moves this engine onto
    /// its own lane thread, and `cudaSetDevice` is per-thread, so the lane
    /// thread announces itself once and this rebinds. Legal before a load
    /// too — nothing to do.
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
        } = request;
        let trace = model_ir::fuse::residual_norm(trace);

        // Serving eagerly is a choice a deployment may make but never one
        // it should make silently: an uncaptured decode pays hundreds of
        // kernel launches per token-step of pure CPU time.
        if !self.boot.graphs.records() {
            eprintln!(
                "engine-cuda: serving without CUDA graph capture ([engine] graphs = \
{:?}, not \"on\"): every fire launches eagerly, which costs per-step host time; \
intended for diagnostics, not serving",
                self.boot.graphs
            );
        }

        // `Checkpoint::None` has no path here: a `WeightTable` of nulls
        // would panic at the first dispatch rather than at the load.
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        // Before the settle, before a plan, before any buffer: an artifact
        // written for another shell or another degree is refused by the
        // field that disagrees, not discovered by the tokens it produces.
        refuse_an_artifact_for_another_deployment(
            &path,
            trace.platform.backend(),
            width_free(&trace.name),
        )?;
        let (contract, plan) = self.settle(&trace, &path, &residency)?;

        // Derived BEFORE the trace moves into the boot — the ladder is a
        // reading of the plan, so it is taken while the plan is still here.
        let patches = patch_ladder(&trace, &budgets);
        let classify = (self.classify_for)(&trace.name).ok_or_else(|| {
            Error::Load(format!("this build ships no classifier for {:?}", trace.name))
        })?;
        let mut shell = Shell::load(Boot {
            classify,
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            // The plan's own declaration decides; a text-only SKU still
            // gets the literal `None` G4 depends on.
            patches,
            // `None` takes this device's measured SM count.
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            pages: budgets.pages,
            // The request's ordinal wins over the boot config's default.
            ordinal: if ordinal >= 0 {
                ordinal
            } else {
                self.boot.ordinal
            },
            graphs: self.boot.graphs,
            knobs: self.boot.knobs,
            cache_dir: self.boot.cache_dir.as_deref(),
            // Clamps what the free-slot word cannot carry.
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            // Carried whole rather than re-derived, so it cannot disagree
            // with the numbers `admit` was asked about.
            residency: plan,
            world: self.boot.world,
            comm: self
                .boot
                .comm
                .as_ref()
                .map_or(core::ptr::null_mut(), |comm| comm.raw()),
        })
        .map_err(fault)?;

        // Not a `Boot` field: it is not a property of the bake, and where
        // the shared adapters live outlives every load.
        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        // `pool_bytes` above is the ceiling the arenas' address space was
        // reserved at; these are what admission has actually mapped, read
        // here as the floor a fresh load starts from.
        let (pool_committed_bytes, pool_high_water_bytes, elastic_page_bytes, elastic_budget_pages) =
            shell.elastic();
        let weights_from_cache = shell.weights_from_cache();
        // Read beside the other facts, while the shell is still here to ask.
        let weights_resident = shell.weights_resident();
        let paging = shell.paging();
        // Only a trace with state rows seats sequences; the KV pool is shared.
        let state_rows = shell
            .trace()
            .caches
            .iter()
            .any(|row| matches!(row, model_ir::CacheRow::State { .. }));
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cuda".to_string(),
                domain: MemoryDomain::CudaDevice(u32::try_from(shell.ordinal()).unwrap_or(0)),
                sms: shell.sms(),
                unified_memory: false,
                // Neither is probed; nothing in this shell reads them yet.
                fp8_native: false,
                native_mxfp4_moe: false,
                // What `cudaMalloc` guarantees and cuBLAS wants — the same
                // 256 the weight store aligns to.
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: Some("cuda".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                // What one recurrent slot costs across the plan's state rows.
                // Zero for a plan with none, which is how the runtime tells a
                // hybrid model from a pure-attention one (`RsCaps::state_size`).
                state_slot_bytes: shell.state_slot_bytes(),
                // What the load actually seats, read off the plan: the
                // smallest capacity any one bank declares, since an id must
                // fit every site it is written into. Zero for a model whose
                // text declares no correction.
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                // The pools are virtual: one logical page of the elastic
                // supply, and the most this load may ever map.
                elastic_page_bytes,
                elastic_budget_pages,
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                // Every lane may name its whole slot's block, and a fire may
                // carry `max_lanes` of them.
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // The whole fire geometry, plus the mask and the fold length:
            // `crate::program::ports` reads every port in this set off the
            // attached instance's own rings at fire time. The page family and
            // write descriptor are claimed because `geometry_with` already
            // takes a caller-stated page table per lane; the mask because a
            // fire whose ancestry is device data has nowhere else to state it;
            // the fold length because it belongs to no class at all
            // (`ports::resolves` answers it for both) and a recurrent guest
            // that cannot state it falls back to a host-serialized pass.
            ports: PortMask::DEVICE_GEOMETRY
                .with(Port::AttnMask)
                .with(Port::RsFoldLen),
            geometry: GeometryClass::DeviceGeometry,
            // `copy_kv` moves cells device-to-device inside this load's own
            // pools; the other three directions need a pool or mapping this
            // shell does not reserve, and are refused by name.
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            // The rings advance on the device: `channel::commit_bump` is the
            // only writer of durable ring state, so a caller may predict
            // cursors by counting.
            device_channel_commit: true,
            // This shell allocates the buffered-activation pool at load and
            // serves `RsVerb::Buffer` and `RsVerb::FoldBuffered` against it.
            rs_verbs: true,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                // `false` says this load opened the routed-expert tier, so
                // `weight_bytes` above is what is resident rather than what
                // the checkpoint holds. `true` is every load with no budget.
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
        // Banks are declared by the model text and reserved at load; this
        // is the write. No graph is touched, since a bank's contents are
        // not in a fire's key, so this costs a copy and leaves every
        // recorded graph valid.
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

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        // Every step is checked before any of them runs. A prediction this
        // shell validates on the device is accepted, and one it could only
        // ignore is refused by name (`Lane::validate_for`).
        frame.validate_for(engine::Serves {
            device_channel_commit: self
                .caps
                .as_ref()
                .is_some_and(|caps| caps.device_channel_commit),
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
        })?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        // The previous frame's numbers die here: the out seam is an arena
        // rectangle this frame is about to carve over, so a caller that
        // wanted numbers had to ask before now.
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut settled = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            // A step that faults poisons the frame's remaining steps: the
            // loop stops, but steps already airborne settle normally since
            // they are real device work.
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            let (ticket, step_settled) = match self.fire_step(step, at) {
                Ok(both) => both,
                Err(error) => {
                    // The steps already in flight keep their settlements; what
                    // this frame cannot do is hand back numbers for a frame it
                    // did not finish.
                    self.pending = None;
                    let airborne = self.shell.as_ref().map_or(0, Shell::airborne_steps);
                    return Err(attributed(error, airborne));
                }
            };
            steps.push(ticket);
            settled.push(step_settled);
        }
        self.pending = Some((id, settled));
        Ok(FrameTicket { id, steps })
    }

    /// Yes: `submit` returns with the device still running, launches on
    /// the compute stream and settlement registered behind an event on the
    /// notify stream, with no host read between them.
    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    /// Fill in the last submitted frame's readouts. Waits for the compute
    /// stream and takes the same reads off the same rectangles, in the
    /// same order, so the result is byte-identical to depth-1 execution.
    /// Refuses a frame the arena no longer holds by name, since a caller
    /// that submits again before it asks has asked too late.
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
                 the out seam is one arena rectangle that every fire carves over. Ask \
                 for a frame's readouts before submitting the next one",
                ticket.id
            )));
        }
        let (_, mut settled) = self.pending.take().expect("checked just above");
        let shell = self.loaded_mut()?;
        for step in &mut settled {
            // The row list the submission stated, handed down.
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

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        // Registration is where the host end is allocated. A guest's cell
        // crosses by device access to mapped pinned memory
        // (`channel::pull_validate` / `channel::scatter_publish`), which
        // requires the engine and the caller to agree on which bytes: the
        // engine allocates the mirror and the four control words, and the
        // caller's ring is a view of them. A channel with no host end
        // allocates nothing. `Endpoint::open` refuses a ring the control
        // kernels' arithmetic cannot carry before anything is allocated.
        if self.channels.contains_key(&registration.id) {
            return Err(Error::Program(format!(
                "channel {} is already registered on this engine",
                registration.id
            )));
        }
        // Every role gets an endpoint here, cut once, so a ring two passes
        // share is not two rings that never met. A `HostRole::None` channel
        // publishes no `HostMirror` (no guest end to point at it) but does
        // set `HOST_READER`: its mirror is a pinned shadow of the committed
        // cell, so a descriptor port resolved off a device-only ring is a
        // load out of mapped memory rather than a blocking `cudaMemcpy`.
        // Its width is the slab's, hence the native `cell_bytes` below.
        let numel = registration
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        let device_only = registration.host_role == eta_ir::container::HostRole::None;
        let cell_bytes = u32::try_from(if device_only {
            // The slab the emitted kernels index, not the wire the mirror
            // holds — see `endpoints_for`.
            crate::program::launch::native_cell_bytes(registration.dtype.program_dtype(), numel)
        } else {
            eta_exec::wire_cell_bytes(registration.dtype.program_dtype(), numel)
        })
        .map_err(|_| {
            Error::Program(format!(
                "channel {}'s cell is wider than a u32 counts",
                registration.id
            ))
        })?;
        let capacity = registration.capacity.max(1);
        let endpoint = Arc::new(
            crate::program::Endpoint::open(registration.host_role, cell_bytes, capacity)
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
            // This shell keeps no waker table (parking and waking are the
            // runtime's), so it mints no wait slot.
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        // Refused at bind: a class claims which descriptor ports the device
        // resolves, checked through `Capabilities::admits` rather than
        // re-deriving the subset test here.
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
        let seeds: Vec<(u32, Vec<u8>)> = binding
            .seeds
            .iter()
            .map(|seed| (seed.channel, seed.bytes.clone()))
            .collect();
        // The extents are the caller's answer, not a guess: every stage's
        // fire-path buffers are carved at this call from these numbers.
        //
        // `InstanceBinding` names this instance's channels in the package's
        // declaration order, the only place caller ids and dense slots are
        // related, so registered endpoints are gathered into dense order
        // here. A channel this engine never registered contributes `None`.
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
                &adopted,
                &binding.channels,
            )
            .map_err(fault)?;
        // The adapter lands here, and nowhere else: whether, what and which
        // slot are all host questions, asked once at bind, so a channel is
        // never a weight transport at fire time. A refused landing closes
        // the instance it was for.
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
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        // The bind is given back before the instance is. The slot keeps
        // its contents (eviction is under pressure, not eager); the
        // release only makes the slot reclaimable.
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(held) = held {
            shell.release_adapter(held);
        }
        shell.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        // The pinned half dies here, and only here. Closing a channel this
        // engine never registered is not an error: idempotent by construction.
        self.channels.remove(&id);
        Ok(())
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        self.instance(instance)?
            .publish(channel, cell)
            .map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    /// Move recurrent state between slots. `StateMove`'s slot ids are read
    /// as this shell's own seat ids (the number `Lane::slot` carries), with
    /// no translation: a runtime whose RS store keeps a second id space
    /// owes that translation on its own side. Whole slots only — a
    /// recurrent bank is a folded summary of a prefix, not per-token
    /// entries, so a move with a token offset is refused rather than
    /// rounded off. Buffered activations are not copied: a fork's buffer
    /// is the runtime's to re-derive or abandon.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a partial move, [`Error::Device`] for a slot
    /// past the pool or the copy itself.
    fn copy_state(&mut self, copy: &StateCopy) -> EngineResult<()> {
        for (at, move_) in copy.moves.iter().enumerate() {
            if move_.src_token_offset != 0 || move_.dst_token_offset != 0 {
                return Err(Error::Invalid(format!(
                    "state move {at} names a token offset, and a recurrent bank is a folded \
                     summary of a prefix rather than an array of per-token entries — this \
                     engine moves whole slots"
                )));
            }
        }
        let shell = self.loaded_mut()?;
        for move_ in &copy.moves {
            shell
                .copy_state(move_.src_slot_id, move_.dst_slot_id)
                .map_err(fault)?;
        }
        Ok(())
    }

    /// Move KV pages inside this device's pools — device-to-device only
    /// (what a fork, a graft and a prefix-cache hit are); the other three
    /// domain pairs are refused by name, as [`Capabilities::kv_copy`]
    /// already states. `src_page_ids`/`dst_page_ids` (whole pages) and
    /// `moves` (single token cells) both flatten into [`store::Move`] runs;
    /// consecutive cell moves coalesce into one `cudaMemcpyAsync` per
    /// plane. Page ids are the caller's; this shell does not translate
    /// them. Enqueued on the fire stream and not synchronized.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a domain pair this shell has no storage
    /// for, [`Error::Invalid`] for a malformed plan, [`Error::Impossible`]
    /// for a page past the pool.
    ///
    /// [`Capabilities::kv_copy`]: engine::caps::Capabilities
    /// [`store::Move`]: crate::store::Move
    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        // The domain pair, before anything is built. `Unsupported`, not
        // `Invalid`: what is missing is storage on this engine.
        let ordinal = self
            .caps
            .as_ref()
            .map(|caps| caps.device.domain)
            .and_then(MemoryDomain::ordinal);
        let served = matches!(
            (copy.src, copy.dst),
            (MemoryDomain::CudaDevice(src), MemoryDomain::CudaDevice(dst))
                if Some(src) == ordinal && Some(dst) == ordinal
        );
        if !served {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "cuda",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let mut moves: Vec<crate::store::Move> =
            Vec::with_capacity(copy.src_page_ids.len() + copy.moves.len());
        // The whole-page half: every token slot, both sides at offset zero.
        // A page's live length is not a number this verb is handed.
        for (src, dst) in copy.src_page_ids.iter().zip(&copy.dst_page_ids) {
            moves.push(crate::store::Move {
                src_page: *src,
                src_token: 0,
                dst_page: *dst,
                dst_token: 0,
                tokens: page_size,
            });
        }
        // The token-granular half, coalesced into runs.
        for (at, cell) in copy.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(Error::Invalid(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                )));
            }
            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = moves.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => moves.push(crate::store::Move {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        // Overlapping ends are the caller's error: a device copy whose ends
        // overlap is undefined. A shift needs a staging page and two moves.
        for run in &moves {
            if run.src_page != run.dst_page {
                continue;
            }
            let (lo, hi) = (
                u32::min(run.src_token, run.dst_token),
                u32::max(run.src_token, run.dst_token),
            );
            if hi - lo < run.tokens {
                return Err(Error::Invalid(format!(
                    "a kv move of {} tokens reads page {} from token {} and writes the same \
                     page at token {} — the two ends overlap, and a device copy whose ends \
                     overlap is undefined rather than a shift",
                    run.tokens, run.src_page, run.src_token, run.dst_token
                )));
            }
        }
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }

    // `encode` takes the trait's default body: this shell carries no
    // multimodal encoder.
}

impl Cuda {
    /// One step of an admitted frame, run to completion. Readiness is not
    /// a fire-path question: the runtime proves it over the whole frame at
    /// `submit` (`validate_frame`), so a pass that does not commit here is
    /// a fault naming the instance and channel, never a replay.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        let done = self.sink.as_ref().map(|sink| crate::serve::Done {
            at,
            sink: std::sync::Arc::clone(sink),
        });
        // Which lane carries which adapter: the slot its attached instance
        // landed at bind, never a channel this fire reads or a number a
        // guest names. An instance with no adapter contributes nothing.
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
        // The word moves with it: a lane's fact word and its adapter are
        // one reading of one lane, so the word is re-stated here into the
        // class this bake's correction window covers, refusing by name
        // rather than firing the lane uncorrected.
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
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| {
                if !lane.positions.is_empty() {
                    // The shell derives positions as `held .. held + rows`.
                    // An explicit list means a speculative fire re-feeding
                    // rejected positions or an mRoPE lane, and both need a
                    // staged `positions` vector this fire path does not take.
                    return Err(Error::Unsupported {
                        verb: "explicit lane positions",
                        engine: "cuda",
                    });
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    // `pages` is already pool ids; this is the table the
                    // ports resolved off the rings still have to go through.
                    translation: &lane.kv.translation,
                    // The plan decides whether anything reads the mask: a
                    // mask against an artifact with no masked arm is
                    // `Fault::Maskless`, named at the fire.
                    mask: lane.mask.as_ref(),
                    // Derived from the binding, not the submission:
                    // `Lane::adapter` is `None` on the runtime's path, and a
                    // bind that landed wins over it.
                    adapter: lane_adapters[at].or(lane.adapter),
                    drafts: lane.drafts,
                    captures_scores: lane.captures_scores,
                    rs: lane.rs.clone(),
                    rs_reset: lane.rs_reset,
                    // The row list crosses to the device half too, for a
                    // guest epilogue reading `IntrinsicId::Logits` on the
                    // device. `Last` and `None` both cross as `None`, the
                    // lane's last row.
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        // The caller's prediction, checked against this engine's own,
        // minted from the same counting: the caller's is the only one that
        // can be right, so a disagreement refuses loudly.
        for attachment in &submission.attachments {
            let Some(lane) = submission.lanes.get(attachment.lane as usize) else {
                continue;
            };
            if lane.channels.is_empty() {
                continue;
            }
            if let Some(why) =
                shell.program_ticket_disagreement(attachment.instance, &lane.channels)
            {
                return Err(Error::Program(format!(
                    "this fire's channel predictions and the engine's disagree: {why}"
                )));
            }
        }

        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                instance: attachment.instance,
                at: attachment.at,
            })
            .collect();
        // Three enqueue-only calls, not one wait: `prepare` makes every
        // host decision and claims a staging slot; `enqueue` puts the step
        // on the compute stream; `settle_step` records an event and hangs
        // the completion callback off it. Bodies arm at load
        // (`Shell::arm_bodies`), so there is no arming instant on this path.

        // The contract's media rows in, `serve::Media` borrows out — the
        // same fields, plus the one conversion: a payload is `f32` until it
        // meets a plan, which computes in the element its text declares, so
        // it is converted here, where `Shell::patch_element` is a value
        // this shell reads off its own load.
        //
        // A text-only fire pays nothing: the two vectors below are never
        // allocated for it.
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            // A media submission against a load whose plan states no patch
            // row has no element or tower to convert for. Refused at the
            // first instant it is knowable.
            let Some(element) = shell.patch_element() else {
                return Err(fault(crate::error::Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Towerless {
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
                embed_rows: &row.embed_rows,
                embed_weights: &row.embed_weights,
                token_positions: &row.token_positions,
            })
            .collect();

        let settled = {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                shell,
                crate::serve::StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            shell.settle_step(enqueued, done).map_err(fault)?
        };

        // Empty readouts, per `FireTicket`'s own doc: the numbers, for a
        // caller that wants them, come from `Engine::settle_frame`. The
        // runtime never asks, since a guest reads its logits on the device
        // through the epilogue's intrinsic.
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

/// A payload row's `f32` numbers, in the element the plan computes in,
/// little-endian. Round to nearest even via [`crate::adapter::bf16_bits`],
/// not truncated, so parity claims about the tower stay about the right
/// numbers.
///
/// # Errors
///
/// The `&'static str` an [`Error::Unsupported`] carries, for a plan whose
/// activation element this marshal cannot write.
fn patch_bytes(
    patches: &[f32],
    element: model_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(patches
            .iter()
            .flat_map(|&v| crate::adapter::bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(patches.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err(
            "a media submission against a plan whose activation element is neither \
                  `bf16` nor `f32`, which is the pair every tower in this catalog computes in",
        ),
    }
}

/// The verb name a refused `copy_kv` direction is refused under. A
/// `&'static str` (not a format) so a caller can match on the refusal
/// rather than only print it.
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

/// Error attribution. With settlement asynchronous, a fault detected while
/// `n` earlier steps are still airborne may belong to any of them, so the
/// message says so — a note, not a different error, since `Exhausted` and
/// `Impossible` are decided on the host and pass through untouched.
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
        // Everything else was decided on the host before a kernel ran, so
        // there is nothing to be uncertain about.
        other => other,
    }
}

/// One step's readouts, as the contract shapes them. `Readout` chooses
/// which logits rows come back; capture is a different column with a
/// different reader. The row count is `Shell::read_out_rows`'s own answer
/// (`Settled::rows`), not a second reading of the policy here.
fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let mut out = Vec::with_capacity(step.readout.len());
    for (lane, want) in step.readout.iter().enumerate() {
        let scores = step.settled.scores.get(lane).cloned().unwrap_or_default();
        let values = step.settled.logits.get(lane).cloned().unwrap_or_default();
        let rows = step.settled.rows.get(lane).copied().unwrap_or(0);
        let width = if rows == 0 {
            0
        } else {
            u32::try_from(values.len() / rows as usize).unwrap_or(u32::MAX)
        };
        out.push(match want {
            // The shell mirrored nothing, but the capture still crosses.
            Readout::None => LaneReadout {
                scores,
                ..LaneReadout::default()
            },
            Readout::Last | Readout::Rows(_) => LaneReadout {
                rows,
                width,
                values,
                scores,
            },
        });
    }
    out
}

// SAFETY: `Shell` holds raw device handles inline, none of which is `Send`
// to the compiler. Sound because every verb takes `&mut self`, so exactly
// one thread touches a shell at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

#[cfg(test)]
mod tests {
    use super::patch_ladder;
    use engine::load::Budgets as LoadBudgets;
    use model_compiler::PATCH_LATTICE_FLOOR;
    use model_ir::{Def, Dim, Dtype, RuntimeInput, Trace, Ty, ValueDecl};

    /// A trace holding one runtime input of the stated shape and nothing else.
    fn trace_with(shape: Vec<Dim>) -> Trace {
        Trace {
            name: "gate".into(),
            platform: model_ir::Platform::Cuda,
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
            seams: Vec::new(),
            drafter: None,
        }
    }

    /// A plan that states a patch row serves with zero configuration.
    #[test]
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

        // Every other patch-axis dim reaches the same answer, because the
        // reading is `Dim::axis` and not a list of variants.
        for shape in [vec![Dim::Images], vec![Dim::ImagesPlus(1)]] {
            assert!(patch_ladder(&trace_with(shape), &LoadBudgets::default()).is_some());
        }
    }

    /// A stated ceiling wins and still gets its rungs.
    #[test]
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

        // And a stated ceiling under the floor is raised to it rather than
        // refused: a rung below the smallest whole image rounds up to a fire
        // that cannot exist.
        let tiny = LoadBudgets {
            max_patches: Some(8),
            ..LoadBudgets::default()
        };
        let raised = patch_ladder(&trace_with(vec![Dim::Patches]), &tiny).expect("a ladder");
        assert_eq!(raised.max_patches, PATCH_LATTICE_FLOOR);
        assert_eq!(raised.buckets, vec![PATCH_LATTICE_FLOOR]);
    }

}

/// The artifact must be for this deployment, asked before a plane lands.
/// A repack moves no value, so a `.zt` converted for one backend/SKU and
/// served on another has identical object names, shapes, spans and part
/// digests to the right artifact — nothing about the bytes tells them
/// apart. The stamp is the only thing that can. An artifact with no stamp
/// is not an error (an ordinary checkpoint); a broken stamp is refused.
/// Runs on the path's name (`serve::read_head`, positioned reads only), so
/// no device buffer or host mapping has happened when it refuses.
/// A SKU name without its `-tp<n>` width: a serving artifact holds whole
/// tensors and each rank reads its band, so the artifact stamped for the
/// one-rank row serves every width of the same text.
fn width_free(sku: &str) -> &str {
    match sku.rsplit_once("-tp") {
        Some((base, width)) if !width.is_empty() && width.bytes().all(|b| b.is_ascii_digit()) => {
            base
        }
        _ => sku,
    }
}

fn refuse_an_artifact_for_another_deployment(
    path: &std::path::Path,
    backend: &str,
    sku: &str,
) -> EngineResult<()> {
    // Three outcomes, asked as three: `Ok(None)` (no serving key, proceed),
    // `Ok(Some(stamp))` (checked below), `Err` (a file claiming to be
    // servable with a rotted stamp).
    let stamp = match checkpoint::file::serve::stamp_of(path) {
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        Err(why) => return Err(Error::Load(why.to_string())),
    };
    let deployment = checkpoint::serving::Stamp::of(backend, sku);
    stamp
        .check(&deployment)
        .map_err(|mismatch| Error::Load(mismatch.refuse(&path.display().to_string())))
}

#[cfg(test)]
mod serving_stamp_tests {
    use super::refuse_an_artifact_for_another_deployment as refuse;
    use checkpoint::file::emit::{self, Object};
    use checkpoint::serving::Stamp;
    use std::collections::BTreeMap;

    fn artifact(dir: &std::path::Path, backend: &str, sku: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{backend}-{sku}.zt"));
        let bytes = vec![7u8; 8192];
        emit::write(
            &path,
            &Stamp::of(backend, sku),
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

    /// The shell refuses an artifact written for another deployment, by
    /// the field that disagrees, before anything is opened.
    #[test]
    fn an_artifact_for_another_shell_is_refused_before_anything_is_opened() {
        let dir = tmp("cross");
        let foreign = artifact(&dir, "metal", "qwen_3");
        let why = refuse(&foreign, "cuda", "qwen_3")
            .expect_err("a metal artifact is not servable here");
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
        // Its own deployment takes it.
        refuse(&artifact(&dir, "cuda", "qwen_3"), "cuda", "qwen_3")
            .expect("a cuda artifact serves on cuda");
        std::fs::remove_dir_all(&dir).ok();
    }

}
