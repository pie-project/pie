//! `impl Engine for Metal` — the shell behind the contract; a boot config with no model until
//! `Engine::load` fills it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::adapter::AdapterRegistration;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
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
use crate::experts;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Landed, Lane, Seated, Shell, StepView};
use crate::settle::Done;
use crate::weights::AdapterPlane;

/// How a caller answers what this checkpoint's bytes mean for this plan; supplied by the party
/// that links the model catalog, since this crate must not know a model family.
pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// Device knobs stated before any model loads. No `ordinal` (Metal has one GPU) and no `graphs`
/// knob (a fire's encode is its own capture).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    /// The fraction of `recommendedMaxWorkingSetSize` this device may hold
    /// resident — weights, kv pool and scratch. See
    /// [`store::accounting`](crate::store::accounting).
    pub gpu_mem_utilization: f64,
    /// Where this deployment's shared adapters live, or `None` if none are mounted.
    pub adapter_dir: Option<std::path::PathBuf>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            gpu_mem_utilization: crate::store::accounting::DEFAULT_GPU_MEM_UTILIZATION,
            adapter_dir: None,
        }
    }
}

/// The readback plan one submitted step left behind, plus the per-lane readout policy the
/// submission stated.
struct PendingStep {
    /// What each lane asked for, in submission order.
    readout: Vec<Readout>,
    /// The receipt the shell minted for this step.
    landed: Landed,
    /// The rows, once taken. Cached since `Engine::settle_frame` hands a step's answer over
    /// exactly once.
    rows: Option<Vec<Vec<f32>>>,
}

/// The Metal shell, behind [`Engine`].
pub struct Metal {
    /// Device-wide boot knobs (see [`DeviceBoot`]).
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    /// Where step completions go, installed once by the engine's owning thread. `None` means
    /// nobody's listening.
    sink: Option<engine::CompletionSink>,
    /// The last submitted frame's per-step readback plans, for a caller that comes back for
    /// numbers ([`Engine::settle_frame`]). One frame's worth: the next submit replaces it.
    pending: Option<(FrameId, Vec<PendingStep>)>,
    /// Which adapter slot each bound instance routes to. An instance with no adapter is absent
    /// from this map.
    adapters: BTreeMap<InstanceId, crate::adapter::Binding>,
}

impl Metal {
    /// An engine bound to nothing yet.
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Metal {
        Metal {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            sink: None,
            pending: None,
            adapters: BTreeMap::new(),
        }
    }

    /// The loaded shell, for a caller that wants the native surface — the
    /// compiled-pipeline counter, the footprint, the guest-program pass.
    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    /// The loaded shell, mutably — also the door to a guest pass fired on its own
    /// ([`Shell::fire_program`]), for a program with no `logits` intrinsic.
    pub fn shell_mut(&mut self) -> Option<&mut Shell> {
        self.shell.as_mut()
    }

    /// What this load can do, once there is one.
    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.caps.as_ref()
    }

    /// Open a slot for a fresh sequence. Not a trait verb: a shell whose lane's `pages` are
    /// empty owns that lane's page table itself.
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
    /// [`Error::Load`] before a load, [`Error::Closed`] for an
    /// instance this plane does not carry.
    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
            .ok_or(Error::Closed {
                what: "instance",
                id,
            })
    }

    #[allow(dead_code)]
    fn loaded(&self) -> EngineResult<&Shell> {
        self.shell
            .as_ref()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))
    }

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| Error::Load("the metal engine has no model loaded".into()))
    }
}

/// The verb name a refused `copy_kv` direction is refused under — a `&'static str` so callers
/// can match on the taxonomy rather than only print it.
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
        (MemoryDomain::MetalPrivate, _) | (_, MemoryDomain::MetalPrivate) => {
            "`copy_kv` with a Metal PRIVATE end, and every reservation this load made is \
             Shared — there are no private pages here to read or write"
        }
        _ => "`copy_kv` between the domains named, neither of which is this load's own device",
    }
}

/// The shell's refusal, in the contract's vocabulary. `Exhausted`/`Impossible` are scheduling
/// answers the runtime acts on (retry, or drop); everything else it logs.
fn fault(fault: Fault) -> Error {
    match fault {
        // The machine could not answer, not the submission.
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),
        // A well-formed request with wrong numbers.
        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
        // Load axis: discovered once at load, not fixed by retrying.
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
        // A bake-integrity break, not fixed by retrying.
        Fault::Fragmented { .. } => Error::Device(fault.to_string()),
        // Binding-recipe derivation failures, fixed at load time.
        Fault::Unaffine { .. } | Fault::Unstructured { .. } => {
            Error::Load(fault.to_string())
        }
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        // A differently-shaped or -positioned mask is a real retry.
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),
        // Retryable with a consistent adapter id/word.
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // Every seat pinned; none reclaimable at any width.
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        Fault::Scoreless { .. } | Fault::ScoreWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // Bank shape is a model-text fact; only the text (not a retry) fixes it.
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        // Guest-program plane, not a model-fire condition.
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) => Error::Invalid(fault.to_string()),
        // Nothing the deployment frees changes this.
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling — converted field for field so an added
/// role must be added to both.
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

/// The ceilings the compiler bakes against. `Budget` takes four of the load's seven numbers;
/// `page_size`, `max_context`, `slots` go straight to [`Boot`].
fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// The second row axis's ceilings, derived from the trace's own types — `None` for a plan with
/// no patch row.
///
/// `max_patches` is the deployment's if stated, else the token ceiling capped at two whole
/// images on the catalog towers' native 48x48 grid, never below [`PATCH_LATTICE_FLOOR`]. Rungs
/// double from that floor; `max_images` is the ceiling at the floor.
#[must_use]
pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    /// Two whole images at the catalog towers' native 48 x 48 grid.
    const DERIVED_PATCH_CEILING: u32 = 4096;

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
            .unwrap_or(max_patches / PATCH_LATTICE_FLOOR)
            .max(1),
        max_patches,
        buckets,
    })
}

/// A payload row's `f32` numbers, encoded in the plan's activation element, little-endian.
/// Rounds to nearest even.
///
/// # Errors
///
/// An `&'static str` for a plan whose activation element this marshal cannot write.
fn patch_bytes(
    patches: &[f32],
    element: model_ir::Dtype,
) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(patches
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(patches.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err("a media submission against a plan whose activation element is neither \
                  `bf16` nor `f32`, which is the pair every tower in this catalog computes in"),
    }
}

/// One `f32` as `bf16` bits, round to nearest even.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// The guest-visible profile of a loaded plan, read off the plan and budgets rather than
/// reconstructed from flags.
///
/// `has_mtp_logits` follows [`Shell::drafts`]; `has_attn_score` follows
/// [`Shell::observes_scores`]; `mtp_depth` follows [`Shell::mtp_depth`]. `has_value_head` is always `false` (no device
/// path produces them). `has_attn_page_mask` is unrelated to `Lane::mask`.
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
        // Interpreter-visible dtype, not this device's own bf16 activation type.
        activation: Dtype::F32,
        has_mtp_logits: shell.drafts(),
        mtp_depth: shell.mtp_depth(),
        // The block drafter's facts, stated by the text on its trace.
        draft_block: shell.trace().drafter.map_or(0, |d| d.rows),
        draft_mask_token: shell.trace().drafter.map_or(0, |d| d.mask_token),
        draft_bidirectional: shell.trace().drafter.is_some_and(|d| d.bidirectional),
        draft_proposals_from: shell.trace().drafter.map_or(1, |d| d.proposals_from),
        has_value_head: false,
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        // The `lora` guest-sink gate; a model with no bank refuses by name
        // (`Fault::Adapterless`).
        has_lora: true,
        kernels: Vec::new(),
    })
}

/// One instance's adapter, landed off its seeds. `Ok(None)` if the program declares no `lora`
/// sink. `Ok(Some(binding))` names the slot every lane attached to this instance routes to.
///
/// Weights are taken from the seeded channel's cell at bind time, not read from the ring at fire
/// time, so re-publishing new adapter weights mid-pass does nothing; swapping means re-binding.
///
/// # Errors
///
/// Whatever [`crate::adapter::sink_of`] and [`crate::adapter::planes_of`] say,
/// [`Error::Load`] for a sink channel this bind seeded nothing into, and
/// whatever the landing said — [`Fault::AdapterSlots`] when every slot is
/// pinned by a live bind, [`Fault::Adapter`] when the planes are not this
/// load's banks.
fn adapter_of(
    shell: &mut Shell,
    program: engine::program::ProgramId,
    instance: InstanceId,
    seeds: &[(u32, Vec<u8>)],
) -> EngineResult<Option<crate::adapter::Binding>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        // An unseeded channel is refused, not treated as a zero (identity) plane.
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
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
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

impl Engine for Metal {
    fn kind(&self) -> &'static str {
        "metal"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    /// No thread to bind: `MTLDevice`/`MTLCommandQueue` are documented thread-safe, unlike
    /// CUDA's per-thread-bound context.
    fn bind_thread(&mut self) -> EngineResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> EngineResult<Loaded> {
        if self.shell.is_some() {
            return Err(Error::Load(
                "this metal engine already has a model loaded; one shell per engine".into(),
            ));
        }
        let LoadRequest {
            trace,
            checkpoint,
            budgets,
            residency,
            ordinal,
            // Carves the A/B seats: one resident-input plane and one readout seat per
            // in-flight step.
            frames_in_flight,
        } = request;

        // Metal serves only the system default device (ordinal 0).
        if ordinal > 0 {
            return Err(Error::unsupported("metal", "device ordinal selection"));
        }

        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the metal shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        // Residency is planned and admitted before any byte lands. Host demand is zero
        // here: unified memory has no separate pinned-copy address space.
        let planes = crate::weights::attachments(&trace, &contract, &path).map_err(fault)?;
        // The gathered class is planned first: a statically-demanded dense plane (e.g. the
        // PLE n-gram table) must be excluded from the table the expert slab is sized
        // against, or an expert plan counting it in its dense floor would refuse early.
        // `max_tokens` sizes the slab directly and is not a budget.
        let gathered = crate::gather::Plan::of(
            &trace,
            &planes,
            residency.device_weight_budget,
            budgets.max_tokens,
        )
        .map_err(fault)?;
        let mut residency_plan =
            experts::Plan::beside(&trace, &planes, residency.device_weight_budget, gathered)
                .map_err(fault)?;

        // The wired ceiling is checked before any byte lands: a GPU-touched Shared page is
        // wired and never evicted on Apple Silicon, so `device_weight_budget` is the only
        // lever bounding the weight tier. A throwaway bind reads
        // `recommendedMaxWorkingSetSize` to compute the effective weight budget.
        {
            let working_set = crate::device::Context::bind().map_err(fault)?.working_set();
            let util = self.boot.gpu_mem_utilization;
            let paging =
                crate::store::kv::Paging::of(
                    budgets.page_size,
                    budgets.max_context,
                    budgets.slots,
                    u64::from(budgets.pages),
                )
                    .map_err(|error| fault(Fault::from(error)))?;
            let kv_pool = crate::store::pool_demand(&trace, paging).map_err(fault)?;

            // The gather slab is weight bytes, reserved in the same store as expert
            // seats, so the ceiling is checked against the sum.
            let acct = crate::store::accounting::Accounting::of(
                working_set,
                util,
                residency_plan.device_demand(),
                kv_pool,
            );
            if acct.admit(residency.device_weight_budget, util).is_err() {
                // Over the wired ceiling: shrink the weight tier to the headroom left
                // after the kv pool and floor, if there's a streamable tier to shrink.
                let headroom = acct.weight_headroom();
                // Re-planned in the same order: gather first (does not shrink), then
                // the expert slab over what is left.
                let regathered =
                    crate::gather::Plan::of(&trace, &planes, Some(headroom), budgets.max_tokens);
                let regathered = match regathered {
                    Ok(plan) => plan,
                    Err(_) => {
                        return Err(fault(
                            acct.admit(residency.device_weight_budget, util)
                                .expect_err("admit errored above"),
                        ));
                    }
                };
                match experts::Plan::beside(&trace, &planes, Some(headroom), regathered) {
                    Ok(shrunk) => {
                        // Re-admit catches a dense floor the headroom itself can't hold.
                        let re = crate::store::accounting::Accounting::of(
                            working_set,
                            util,
                            shrunk.device_demand(),
                            kv_pool,
                        );
                        re.admit(Some(headroom), util).map_err(fault)?;
                        residency_plan = shrunk;
                    }
                    Err(_) => {
                        // Nothing to stream, or the minimal slab is still over.
                        return Err(fault(
                            acct.admit(residency.device_weight_budget, util)
                                .expect_err("admit errored above"),
                        ));
                    }
                }
            }
        }

        residency.admit(residency_plan.device_demand(), residency_plan.host_demand())?;
        let streams = residency_plan.streams();

        let patches = patch_ladder(&trace, &budgets);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            patches,
            // `None` takes this device's own core count and zero side streams.
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            pages: budgets.pages,
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            residency: residency_plan,
        })
        .map_err(fault)?;

        // A deployment property, not part of the bake — outlives every load.
        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        // Read here since the shell moves into `self` before the facts are assembled.
        let weights_warm = shell.weights_warm();

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
                backend: "metal".to_string(),
                // `Context::bind` refuses a device without unified memory, so
                // `MetalShared` is the only domain reachable here.
                domain: MemoryDomain::MetalShared,
                // Not measured: Metal publishes no SM/core count.
                sms: 0,
                unified_memory: true,
                // `kernels-metal` stamps one dtype (bf16) across the op set.
                fp8_native: false,
                native_mxfp4_moe: false,
                // What `weights.rs`/`inputs.rs` carve to; the tighter of the two.
                storage_alignment: 256,
                // `maxBufferLength`, the ceiling `Context::reserve` enforces.
                storage_max_tile_bytes: shell.max_buffer(),
                codegen_backend: Some("metal".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                state_slots: if state_rows { paging.slots } else { 0 },
                // Non-zero is what tells the runtime this model folds a recurrent
                // state and makes its passes hybrid (`crate::rs`).
                state_slot_bytes: shell.state_slot_bytes(),
                // The smallest capacity any one bank declares; zero if the model text
                // declares no correction.
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                // The pools aren't virtual, so `resize_pool` isn't served.
                elastic_page_bytes: 0,
                elastic_budget_pages: 0,
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // This shell resolves the whole fire geometry on the device, letting a
            // sampled token feed the next decode step without a host round-trip.
            ports: PortMask::DEVICE_GEOMETRY.with(Port::AttnMask).with(Port::RsFoldLen),
            geometry: GeometryClass::DeviceGeometry,
            // Host directions need a pinned swap pool this load reserves none of.
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            // Rings advance on the host; a channel prediction is refused, not ignored.
            device_channel_commit: false,
            // The recurrent-state device half (`crate::rs`): a buffered page slab, committed
            // scans that persist the bank only as far as the verb says, and the read path
            // that replays a buffered prefix ahead of a fire's rows.
            rs_verbs: shell.serves_rs_verbs(),
            // The sdpa shaders apply their own causal bound beside the
            // staged mask plane, so a bidirectional lane is refused by name.
            bidirectional_attention: false,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                // `true`: full residency, one buffer that never moves. `false`: routed
                // bands went to the wired-slab tier and fires are cut into segments.
                weights_resident: !streams,
                // `true` when this load mapped its checkpoint (the warm arm) instead of
                // reading it; `false` for a streamed plan or anything that couldn't map.
                weights_from_cache: weights_warm,
                arena_bytes,
                pool_bytes,
                input_bytes,
                // This plane's pools are one reservation, so committed is the ceiling.
                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        // No graph is touched: a bank's contents aren't part of the composition key, so
        // registering between fires is a memcpy. Borrowed, not copied; no scaling
        // happens here — `α/r` is folded into the up bank's contents already.
        let planes: Vec<AdapterPlane<'_>> = registration
            .planes
            .iter()
            .map(|plane| AdapterPlane {
                bank: plane.bank.as_str(),
                bytes: &plane.bytes,
            })
            .collect();
        self.loaded_mut()?
            .register_adapter(registration.id, &planes)
            .map_err(fault)
    }

    fn submit(&mut self, frame: &FrameSubmission) -> EngineResult<FrameTicket> {
        // Every step is validated before any of them runs — against what this
        // load actually serves: the recurrent verbs where the plan has state
        // to buffer (`crate::rs`), never a channel prediction.
        frame.validate_for(engine::fire::Serves {
            device_channel_commit: false,
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
            bidirectional: false,
        })?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        // The previous frame's numbers are dropped: a caller wanting them must ask
        // before the next submit.
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut pending = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            // Advisory hint to the engine about its own successor step.
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }
            // A step that faults poisons the frame's remaining steps; steps already
            // committed still settle normally.
            let at = engine::StepDone {
                frame: id,
                step: index as u32,
            };
            match self.fire_step(step, at) {
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
        let mut ticket = FrameTicket { id, steps };
        self.pending = Some((id, pending));
        // At depth one, numbers are filled before `submit` returns.
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    /// `true` above depth one: `submit` returns with the device still running, and outcomes
    /// arrive via [`Engine::on_complete`]'s sink. `false` at depth one, where `submit` waited.
    fn settles_asynchronously(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.frames_in_flight() > 1)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    /// Fills in the last submitted frame's readouts, waiting for whatever the host hasn't caught
    /// up with. Refuses a frame whose readout seats a later `submit` has already taken back.
    /// Idempotent: rows are cached on first read.
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
                refused = Some(Error::Load("the metal engine has no model loaded".into()));
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
        // Put back either way, so a device fault still leaves the answered steps readable.
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

    /// Harvests every already-finished in-flight step ([`Shell::reap`], non-blocking) so the
    /// next `prepare` finds a free arm. Correctness never depends on it.
    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            let _ = shell.reap();
        }
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    /// Registers only device-only channels (`HostRole::None`); the two host-visible roles are
    /// the runtime's own ring, which this plane publishes no mirror for. A device-only ring is
    /// cut once here, per channel id, so multiple attachments share one ring.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for the two host-visible roles,
    /// [`Error::Program`] for an id already registered or a load that has not
    /// happened, and whatever the reservation said.
    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> EngineResult<RegisteredChannel> {
        if registration.host_role != eta_ir::container::HostRole::None {
            return Err(self.unsupported("register_channel"));
        }
        // A cell-width mismatch is refused at bind, against this shape.
        let shape = crate::program::ChannelShape {
            capacity: registration.capacity.max(1),
            numel: registration
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .product::<usize>()
                .max(1),
            dtype: registration.dtype.program_dtype(),
        };
        self.loaded_mut()?
            .register_shared_channel(registration.id, shape)
            .map_err(fault)?;
        Ok(RegisteredChannel {
            id: registration.id,
            // No waker table kept here.
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        // Checked against `Capabilities::admits` rather than re-derived here.
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
        // Extents are the caller's stated answer; every stage's fire-path buffers are
        // carved from these numbers here. A device-only channel is adopted once, so
        // every attachment of one id shares the same ring.
        let shell = self.loaded_mut()?;
        let id = shell
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
                &binding.channels,
            )
            .map_err(fault)?;
        // A refused landing closes the instance: an unlanded adapter would silently fire
        // the base model.
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
        // The bind is released before the instance, or the slot stays pinned forever.
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(bound) = held.as_ref() {
            shell.release_adapter(bound);
        }
        shell.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        // Only for channels this plane registered (device-only rings). The ring itself
        // outlives the entry while an attachment still holds one.
        if self.loaded_mut()?.close_shared_channel(id) {
            return Ok(());
        }
        Err(self.unsupported("close_channel"))
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

    /// Move kv cells between pages of this load's own pools — a fork, a graft, or a
    /// prefix-cache hit.
    ///
    /// # One direction; the other three refused by name
    ///
    /// Only `MetalShared -> MetalShared` is served. A host-pinned end needs a swap pool this
    /// load doesn't reserve; a `MetalPrivate` end is a storage mode nothing here reserves.
    /// [`Capabilities::kv_copy`] states this ahead of time.
    ///
    /// Page ids are the caller's; this shell does not translate them.
    ///
    /// # Ordering
    ///
    /// The moves go into their own command buffer on the fire queue, unsynchronized: behind
    /// every step already committed, ahead of every step committed after this returns.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a domain pair this shell has no storage for,
    /// [`Error::Invalid`] for a malformed plan — unequal page lists, an offset
    /// past the page, or a move whose two ends overlap — and
    /// [`Error::Impossible`] for a page past the pool.
    ///
    /// [`Capabilities::kv_copy`]: engine::caps::Capabilities
    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        // `Unsupported` (not `Invalid`): the plan is well-formed, what's missing is
        // storage on this engine.
        let served = self
            .caps
            .as_ref()
            .is_some_and(|caps| copy.src == caps.device.domain && copy.dst == caps.device.domain);
        if !served {
            return Err(Error::Unsupported {
                verb: kv_copy_direction(copy.src, copy.dst),
                engine: "metal",
            });
        }
        let page_size = self.loaded_mut()?.paging().page_size;
        let moves =
            crate::store::Move::plan(copy, page_size).map_err(Error::Invalid)?;
        self.loaded_mut()?.copy_kv(&moves).map_err(fault)
    }

    /// Move recurrent state between slots — the device half of a recurrent fork.
    ///
    /// `StateMove`'s slot ids are read as this shell's own seat ids, with no translation.
    /// Whole slots only: a recurrent bank is a folded summary of a prefix, not per-token
    /// entries, so a move with a token offset is refused rather than rounded off, and
    /// `token_count` is not read. Buffered activations are not copied; a fork's buffer is
    /// the runtime's to re-derive or abandon. An attention-only load has no banks and
    /// answers `Ok`.
    ///
    /// # Ordering
    ///
    /// As [`Engine::copy_kv`] on this engine: one command buffer on the fire queue,
    /// unsynchronized — behind every step already committed, ahead of every step committed
    /// after this returns.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a partial move, [`Error::Impossible`] for a slot past the
    /// pool, [`Error::Device`] when the queue would not take the copy.
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
        let moves: Vec<(u32, u32)> = copy
            .moves
            .iter()
            .map(|move_| (move_.src_slot_id, move_.dst_slot_id))
            .collect();
        self.loaded_mut()?.copy_state(&moves).map_err(fault)
    }

    // `resize_pool` and `encode` use the trait's default (refusing) bodies: no virtual
    // pools, no multimodal encoder here.
}

impl Metal {
    /// One step of an admitted frame — `submit`'s per-step body.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        // Attachment refusals (instance readiness, double attachment, an unsupported
        // column) are checked by `Shell::admit_attachments` at `prepare`.
        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                instance: attachment.instance,
                at: attachment.at,
            })
            .collect();

        // The marshal (media door): a payload is `f32` until it meets a plan, and only
        // this shell knows the plan's activation element (`Shell::patch_element`). A
        // text-only fire (`submission.media` empty) allocates nothing extra.
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            // A media submission against a plan with no patch row has no tower to
            // convert for; refused as soon as it's knowable.
            let Some(element) = self.loaded()?.patch_element() else {
                return Err(fault(crate::error::Fault::from(model_exec::Error::Fire(
                    model_exec::fire::Fault::Towerless {
                        lane: submission.media[0].lane,
                    },
                ))));
            };
            staged.reserve(submission.media.len());
            for row in &submission.media {
                staged.push(
                    patch_bytes(&row.patches, element).map_err(|why| Error::Unsupported {
                        verb: why,
                        engine: "metal",
                    })?,
                );
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

        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        // The readout policy is checked before anything runs: a row list needs the
        // logits rectangle blitted into a load-time-reserved seat of exactly `max_lanes`
        // rows, fixed at load. Served for a lane with an epilogue attached (delivered on
        // the device); refused otherwise.
        for (index, lane) in submission.lanes.iter().enumerate() {
            let listed = matches!(lane.readout, Readout::Rows(_));
            let served = attached
                .iter()
                .any(|a| a.lane as usize == index && a.at == engine::fire::Boundary::Epilogue);
            if listed && !served {
                return Err(Error::unsupported("metal", "row-selected readout"));
            }
        }

        // The sink is cloned out before the shell is borrowed: one `Arc` bump.
        let sink = self.sink.clone();
        // A lane's adapter is the slot its attached instance landed at bind.
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
        // The lane's fact word is re-stated into the correction's window when it
        // carries an adapter; refuses by name (`Fault::AdapterWord`) rather than firing
        // uncorrected.
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
                // `drafts` rides the WORD (`stamp_lane_words` folds it in beside the
                // mask and adapter facts), which is what selects the draft arm's
                // class; the `mtp` seam's rectangle is bound to the reading
                // attachment at the fire (`Shell::drafts`). Nothing else to carry.
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        // The caller's own word, or the correction window's.
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    captures_scores: lane.captures_scores,
                    // Expanded into the dense sdpa plane (`crate::mask`) and
                    // cross-checked against the artifact at the fire.
                    mask: lane.mask.as_ref(),
                    // A guest sink's bound slot wins over `Lane::adapter`.
                    adapter: lane_adapters[at].or(lane.adapter),
                    positions: &lane.positions,
                    // A row list crosses to the device half only. `Last`/`None` both
                    // cross as `None` (the lane's own last row).
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    // Empty except for `DeviceGeometry` lanes.
                    translation: &lane.kv.translation,
                    // The recurrent verb and the store's own reset classification
                    // (`crate::rs`), verbatim.
                    rs: &lane.rs,
                    rs_reset: lane.rs_reset,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        // Three calls, no wait: `prepare` makes host decisions and takes a seat,
        // `enqueue` encodes and commits without waiting, `settle` files the flight.
        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.map(|sink| Done { at, sink });
            let prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &attached,
                    media: &media,
                    done,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
        };

        // Readouts are empty here; numbers come from `Engine::settle_frame`.
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

/// One step's rows, per lane. The shell answers one row per lane (the last one). A
/// `Readout::Rows` lane answers an empty record: those rows were delivered on the device to the
/// epilogue that named them.
fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| match policy {
            // `scores` stays empty: served on the device with no host mirror.
            Readout::None | Readout::Rows(_) => LaneReadout::default(),
            Readout::Last => {
                let values = rows.get(lane).cloned().unwrap_or_default();
                LaneReadout {
                    rows: 1,
                    width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    values,
                    ..LaneReadout::default()
                }
            }
        })
        .collect()
}

// SAFETY: `Engine` requires `Send + Sync`; a loaded `Shell` holds retained Objective-C
// objects `objc2` doesn't mark `Send`, plus `RefCell`s written through `&self`. Sound
// because every verb that reaches the shell takes `&mut self`; `kind`/`device_facts`
// (the only `&self` verbs) never touch the shell.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}
