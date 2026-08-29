//! `impl Engine for Cuda` — the shell behind the contract.
//!
//! # Why a wrapper and not `impl Engine for Shell`
//!
//! Because a [`Shell`] **is a loaded model**. `Shell::load` binds the device,
//! compiles the plan, lands the checkpoint and reserves the pools in one
//! call, and every other method on it is about that load. The contract's
//! [`Engine`] is the other shape: a caller opens an engine first
//! (`runtime::engine::backend::open::cuda`, from a boot config that has no
//! model in it), registers it, and only then calls
//! [`Engine::load`] with a traced `Trace`. There is no `Shell` to have an
//! `Engine` impl on until the verb that makes one has been called.
//!
//! So [`Cuda`] is a `Shell` that has not happened yet: the device knobs a
//! boot config states, an `Option<Shell>` that `load` fills, and the
//! [`Capabilities`] that load answered. Every verb before a load is a
//! refusal with a sentence.
//!
//! # The contract the wrapper cannot state, and how it is supplied
//!
//! [`LoadRequest`] carries `{ trace, checkpoint, budgets, ordinal }` and NOT a
//! `ModelContract` — deliberately, because `engine-api`'s dependency floor is
//! `model-ir`, `tensor-ir`, `serde`, `thiserror` (its own header), and a
//! contract type in it would put `model-loader` in the graph of everyone who
//! reads a `KvHandle`. But [`Weights::resident`](crate::weights::Weights)
//! needs one: how a checkpoint's tensors become this plan's params is the
//! MODEL's declaration, and the shell must not grow an arm per family to
//! rediscover it (`weights.rs`'s own header).
//!
//! The resolution is a function pointer, installed when the engine is opened:
//!
//! ```text
//!   runtime (links `model`)                     engine-cuda (links no family)
//!   -----------------------                     -----------------------------
//!   fn contract_for(trace, path) -> Contract ──▶ Cuda::new(boot, contract_for)
//!     model::import_of(trace.name)                  … load(request) calls it
//! ```
//!
//! One pointer, resolved by the party that already links the catalog, and no
//! model name anywhere in this crate outside its own dev-dependencies.
//!
//! # What this engine does not serve, and says so
//!
//! `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
//! default bodies, which answer [`Error::Unsupported`]. That is the
//! honest report of what the v1 shell has: its pools are not virtual, it has
//! no peer-copy path, no recurrent-state mover and no multimodal encoder.
//! Stubbing any of them to `Ok(())` would make a prefix-cache hit, a swap or
//! an image prompt *appear* to work and silently read the wrong bytes, which
//! is the failure mode the contract's "refusal is a value" section exists to
//! prevent.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use engine::engine_api::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::engine_api::channel::{
    ChannelId, ChannelRegistration, HostMirror, RegisteredChannel,
};
use engine::engine_api::error::{Error, Result as EngineResult};
use engine::engine_api::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout,
    Step,
};
use engine::engine_api::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use engine::engine_api::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use engine::engine_api::transfer::MemoryDomain;
use engine::engine_api::Engine;
use model_compiler::{Budget, DeviceProfile};
use model_ir::Trace;
use model_loader::contract::ModelContract;
use engine::engine_api::tensor_ir::registry::{GeometryClass, ModelProfile, PortMask};
use engine::engine_api::tensor_ir::types::DType;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Attached, Boot, Graphs, Lane, Seated, Shell};

/// How a caller answers "what does this checkpoint's bytes mean for this
/// plan".
///
/// See the module header: the contract has no seat for a `ModelContract` and
/// this crate must not know a model family, so the party that links the
/// catalog supplies the lookup.
pub type ContractFor =
    fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

/// The device knobs a boot config states, before any model is loaded.
///
/// Everything here is a property of the MACHINE and the deployment, which is
/// what an engine is opened with; the model's own ceilings arrive later on
/// [`LoadRequest::budgets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceBoot {
    /// Which device to bind.
    pub ordinal: i32,
    /// How much of a fire to record. Overridden by `PIE_CUDA_GRAPHS`.
    pub graphs: Graphs,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            ordinal: 0,
            graphs: Graphs::default(),
        }
    }
}

/// The CUDA shell, behind [`Engine`].
/// One submitted step, held for a caller that comes back for numbers.
///
/// The readback plan the shell computed at `settle`, plus the per-lane
/// `Readout` policy the SUBMISSION stated — which the shell never sees,
/// because which rows a caller wants back is a contract question and not a
/// device one.
struct PendingStep {
    readout: Vec<Readout>,
    settled: crate::serve::Settled,
}

pub struct Cuda {
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    /// **THE HOST END OF EVERY REGISTERED CHANNEL** (alto design §5).
    ///
    /// Registration allocates the mapped pinned mirror and the four control
    /// words a channel's host end is, and publishes their addresses; the bind
    /// that follows hands the same allocation to the instance's session, so
    /// the guest writes through the very bytes `channel::pull_validate` reads.
    /// Keyed by the caller's channel id, because the caller names channels by
    /// id and an instance names them by dense index — `InstanceBinding`'s own
    /// `channels` list is the map between the two.
    ///
    /// **CONTROL PLANE, NOT FIRE PATH** (article 9): nothing here is touched
    /// between `bind_instance` and `close_channel`.
    channels: BTreeMap<ChannelId, Arc<crate::program::Endpoint>>,
    /// **Where step completions go**, installed once by the thread that owns
    /// this engine ([`Engine::on_complete`]). `None` is a caller that does not
    /// want to hear — the smoke tests, a bench — and costs the settlement
    /// callback one branch.
    sink: Option<engine::engine_api::CompletionSink>,
    /// **The last submitted frame's per-step readback plans**, held for a
    /// caller that comes back for numbers ([`Engine::settle_frame`]).
    ///
    /// One frame's worth and no more, and that is a statement about the arena
    /// rather than a cache policy: the out seam and the export columns are
    /// arena rectangles the NEXT fire carves over, so a frame's numbers exist
    /// only until the frame after it is enqueued. Holding two would be
    /// offering to answer with bytes the device has overwritten.
    pending: Option<(FrameId, Vec<PendingStep>)>,
}

impl Cuda {
    /// An engine bound to nothing yet.
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Cuda {
        Cuda {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
            next_frame: 1,
            channels: BTreeMap::new(),
            sink: None,
            pending: None,
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

    /// Open a slot for a fresh sequence.
    ///
    /// Not a trait verb: the contract has no `open`, because
    /// [`KvDelta`](engine::engine_api::KvDelta) says a lane whose `pages` are
    /// empty is one whose page table the SHELL owns, and a shell that owns it
    /// opens the slot itself. A runtime that keeps its own page table never
    /// calls this — it states the same fact as `KvDelta::held == 0`, and the
    /// fire path clears the slot's recurrent banks on reading it.
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
    /// instance this plane does not carry — `Closed` and not `Program`
    /// because "this handle is gone" is what the caller can act on, and a
    /// channel door is exactly where a torn-down instance is discovered.
    fn instance(&mut self, id: InstanceId) -> EngineResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .ok_or(Error::Closed {
                what: "instance",
                id,
            })
    }

    // `fn loaded(&self) -> EngineResult<&Shell>` STOOD HERE, AND IT EXISTED
    // FOR THE DOUBLE DOOR. Its one caller was `fire`'s readiness loop, which
    // needed a SHARED borrow of the shell to ask `program_ready` a second
    // time before handing the shell a `&mut` to fire with. The question is
    // asked once now, inside the shell's own prepare phase, so nothing on
    // this side needs the shell without needing to drive it.

    fn loaded_mut(&mut self) -> EngineResult<&mut Shell> {
        self.shell.as_mut().ok_or_else(|| {
            Error::Load("the cuda engine has no model loaded".into())
        })
    }
}

/// The shell's refusal, in the contract's vocabulary.
///
/// **THE TAXONOMY IS THE POINT.** `Exhausted` and `Impossible` are scheduling
/// answers the runtime's lane loop acts on — retry behind something that frees
/// pages, or drop the request — and everything else is a failure it logs. A
/// [`Fault::Ceiling`] is `Impossible` and not `Exhausted` because every
/// ceiling this shell states was reserved at LOAD: no amount of freeing makes
/// a graph recorded for 8192 rows take 9000, which is exactly the distinction
/// `engine_api::error`'s header draws.
fn fault(fault: Fault) -> Error {
    match fault {
        Fault::Runtimeless | Fault::Device { .. } | Fault::Schedule { .. } => {
            Error::Device(fault.to_string())
        }
        // `Unlowered` joins them: a region baked behind a conditional node is
        // an ARTIFACT this shell cannot record, so the recovery is a different
        // profile at load time and not a different fire.
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Param { .. }
        | Fault::Unbound { .. }
        | Fault::Unlowered { .. } => Error::Load(fault.to_string()),
        // AN EXHAUSTION, AND THE CONTRACT HAS THE SHAPE FOR IT. `Exhausted`
        // carries the two numbers structurally rather than only in a sentence,
        // which is what a control plane deciding where to place a model wants.
        // Its `is_scheduling()` reading costs nothing here: that predicate is
        // consulted on the FIRE path, and every allocation this shell makes is
        // made at load.
        Fault::OutOfMemory { need, have } => Error::Exhausted {
            resource: "device memory",
            wanted: need,
            available: have,
        },
        Fault::Ceiling { what, need, have } => Error::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        // A region whose classes this fire's order does not make consecutive
        // is a BAKE-integrity break, not a submission the caller can fix; so
        // is a schedule built over more classes than its reader runs.
        Fault::Fragmented { .. } => Error::Device(fault.to_string()),
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        // A mask that does not describe its lane, or one against a plan with
        // no masked arm, is the SUBMISSION's — the caller stated it and the
        // caller can state it differently. `Invalid`, not `Impossible`: a
        // retry with a mask of the lane's own extent is a real answer.
        Fault::Mask { .. } | Fault::Maskless { .. } | Fault::MaskWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // The adapter axis's three, sorted the same way. A lane routed
        // against an artifact with no correction, or against a word that puts
        // it outside the correction's window, is the SUBMISSION's — a retry
        // that states the two consistently is a real answer. A registration
        // the banks cannot seat is `Load`: capacity is a shape the model text
        // declared, so nothing the caller frees makes room and the fix is the
        // model text.
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // The two export axes' four, sorted by the same rule and for the same
        // reason: a lane that asked for a draft or a capture the artifact does
        // not declare, or that asked in a way its word does not agree with, is
        // the SUBMISSION's. A retry that states the intent and the word as one
        // reading of one lane is a real answer — and on the runtime's path
        // there is only one reading, because `stamp_lane_words` computes the
        // word FROM the intent.
        Fault::Draftless { .. }
        | Fault::DraftWord { .. }
        | Fault::Scoreless { .. }
        | Fault::ScoreWord { .. } => Error::Invalid(fault.to_string()),
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        // **THE ONE REFUSAL THAT CROSSES AS A SCHEDULING ANSWER**, and the
        // reason it is a variant rather than a sentence. `Fault::Blocked` is
        // the readiness gate in `serve`'s prepare phase saying an attached
        // guest's ring has no room *right now*; the host drains it and the
        // identical frame is admitted. The numbers are the CHANNEL's: a
        // blocked instance wants one more cell than the ring can give it,
        // which is either an input that has not arrived or an output the host
        // has not taken, and naming the channel is what tells a log those two
        // apart. This is exactly what `Cuda::fire` used to compute by asking
        // `program_ready` a second time before the shell asked it at all.
        Fault::Blocked { channel, .. } => Error::Exhausted {
            resource: "guest channel cells",
            wanted: u64::from(channel) + 1,
            available: u64::from(channel),
        },
        Fault::Compile(_) | Fault::Program { .. } => Error::Program(fault.to_string()),
        Fault::Fire(_) => Error::Invalid(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling.
///
/// Two names for one seven-role vector: [`ExtentRole`] is the tag space both
/// are written in, and the conversion is field for field so that adding a
/// role to one without the other is a compile error rather than a silently
/// unresolved axis.
fn extents(stated: &BindExtents) -> engine::Extents {
    engine::Extents {
        kv_len: stated.kv_len,
        page_count: stated.page_count,
        row_count: stated.row_count,
        token_count: stated.token_count,
        sampled_rows: stated.sampled_rows,
        query_len: stated.query_len,
        key_len: stated.key_len,
    }
}

/// The ceilings the compiler bakes against, out of the ones the load states.
///
/// The contract carries seven numbers and `model_compiler::Budget` takes
/// four; the other three (`page_size`, `max_context`, `slots`) are the POOLS'
/// and go to `Boot` directly. Converted in one place, which is the whole
/// reason `engine-api` states its own `Budget` rather than depending on the
/// compiler (`load.rs`'s note).
fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// The guest-visible profile of a loaded plan.
///
/// **CARRIED, NOT RECONSTRUCTED** (design §7 on `caps`): the runtime used to
/// rebuild a `ModelProfile` at bind time out of eight `has_*` booleans on a
/// flat capability struct. Everything below is read off the plan and the
/// budgets — `num_layers` from the nodes' own `layer` stamps, `vocab` from
/// the width of the `out` seam — so there is one copy and nothing to keep in
/// step.
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
        // The INTERPRETER-VISIBLE materialization, which is what the field
        // means (`ModelProfile::activation`'s own doc): the device's own
        // activation type is bf16 and is not what a guest program reads.
        activation: DType::F32,
        // Every one of these is a MODEL-GATED intrinsic the guest-program
        // plane would have to point at a buffer of this fire's, and the
        // attachment binds exactly ONE — `IntrinsicId::Logits`, at the arena's
        // out-seam rectangle (`Shell::fire_attached`). There is no second
        // rectangle for an mtp draft, a value head or an attention score to
        // stand in, so advertising one would let a program bind against it and
        // then fail at its first fire, which is the opposite of a bind-time
        // contract.
        // **THE DRAFT COLUMN NOW HAS SOMEWHERE TO STAND** (palo C3b). This
        // was `false` with the note above, and the note was true: there was
        // one rectangle and one binding. There are two now — the `mtp` export
        // is a column of its own, held open past the last node by
        // `model_compiler::arena`'s delivery tail and pointed at by
        // `IntrinsicId::MtpLogits` in `Shell::fire_captured` — so this is
        // exactly "does this load's model text declare a draft head", which
        // is what a bind-time contract has to mean.
        has_mtp_logits: shell.drafts(),
        // Still `false`, and still for the reason above: `MtpDrafts` is `[k]`
        // I32 TOKEN IDS, which is an argmax the guest can take for itself off
        // `MtpLogits` and which no device path in this shell produces.
        has_mtp_drafts: false,
        has_value_head: false,
        // Still `false`, and NOT because the column is missing — the capture
        // arm writes one and `LaneReadout::scores` reads it. `AttnScore` is
        // registered at `Stage::OnAttn`, a mid-graph tap, and design §9
        // abolished the third boundary; it also promises `[num_heads, kv_len]`
        // per-key softmax weights, where this axis exports a per-query mass.
        // Advertising it would let a program bind against a shape and a place
        // that do not exist here.
        has_attn_score: false,
        has_attn_page_mask: false,
        has_lora: false,
        kernels: Vec::new(),
    })
}

impl Engine for Cuda {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    /// **THE LANE THREAD SAYS IT IS THE ONE NOW.** `Shell::load` binds the
    /// device on the thread that CALLED it, which is the worker's boot
    /// thread; the runtime then moves this engine onto its own lane thread and
    /// runs every verb after the load from there. `cudaSetDevice` is
    /// per-thread and does not travel with the value, so the lane thread
    /// announces itself once and this rebinds.
    ///
    /// Before a load there is no device to bind to and nothing to do — the
    /// announcement is legal at any point and answers for what it can.
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
            ordinal,
            frames_in_flight,
        } = request;

        // `Checkpoint::None` — bind and bake, land nothing — is a shape the
        // contract states and this shell has no path for: `Weights::resident`
        // is what reserves the store, and a `WeightTable` of nulls would
        // panic at the first dispatch rather than at the load that asked for
        // it. Refused by name.
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the cuda shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        let shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            // `None` takes this device's measured SM count. Costs are input,
            // not knowledge (design §6): a deployment that has measured its
            // own would state them, and one that has not still bakes
            // something that runs.
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
            // The REQUEST's ordinal wins over the boot config's when it names
            // one: `LoadRequest::ordinal` is the contract's field for "which
            // device, when the shell serves more than one", and a boot config
            // that also named one is the deployment's default.
            ordinal: if ordinal >= 0 { ordinal } else { self.boot.ordinal },
            graphs: self.boot.graphs,
            // **THE DEPTH CROSSES ONCE AND IS DERIVED FROM HERE ON** (article
            // 8). `Runahead::of` clamps what the free-slot word cannot carry;
            // the deployment's config layer refuses an out-of-range depth by
            // name long before it reaches this line.
            runahead: engine::runahead::Runahead::of(frames_in_flight),
        })
        .map_err(fault)?;

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let paging = shell.paging();
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "cuda".to_string(),
                domain: MemoryDomain::CudaDevice(
                    u32::try_from(shell.ordinal()).unwrap_or(0),
                ),
                sms: shell.sms(),
                unified_memory: false,
                // Neither is probed. Both are load-time answers about the
                // device's ARITHMETIC, and nothing in this shell reads them
                // yet — the mxfp4 MoE arm is the dispatch plane's decision
                // (design §6), not a capability the caller selects.
                fp8_native: false,
                native_mxfp4_moe: false,
                // What `cudaMalloc` itself guarantees, and what a matrix
                // operand wants under cuBLAS — the same 256 the weight store
                // aligns to.
                storage_alignment: 256,
                storage_max_tile_bytes: u64::MAX,
                codegen_backend: Some("cuda".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                // The recurrent pool is reserved off the plan's own
                // `CacheRow::State` rows and the shell keeps no count of its
                // slots apart from the seats it opened, which is `slots`.
                state_slots: paging.slots,
                state_slot_bytes: 0,
                // `palo C2`: what the LOAD actually seats, read off the plan
                // rather than off the request. `Budget::max_adapters` is what
                // the deployment asked for and `compile` already refused a
                // plan that could not seat it; this is the answer, which is
                // the smallest capacity any one bank of this model declares —
                // an id must fit every site it will be written into, so the
                // minimum is the honest ceiling. Zero for a model whose text
                // declares no correction, and then `Lane::adapter` has
                // nowhere to go and is refused by name.
                adapter_banks: shell
                    .banks()
                    .iter()
                    .map(|&(_, adapters, _)| adapters)
                    .min()
                    .unwrap_or(0),
                // The pools are not virtual, so `resize_pool` is not served
                // and zero is what says so.
                elastic_page_bytes: 0,
                elastic_budget_pages: 0,
            },
            limits: FireLimits {
                max_lanes: budgets.max_lanes,
                max_tokens: budgets.max_tokens,
                // Every lane may name its whole slot's block, and a fire may
                // carry `max_lanes` of them.
                max_page_refs: paging
                    .pages_per_slot
                    .saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // THE DECODE ENVELOPE IS SERVED AND NOTHING WIDER IS (`palo
            // B3`). `crate::program::ports` reads `embed_tokens`, `positions`
            // and `kv_len` off the attached instance's own device rings at
            // fire time — the token because the device DECIDED it and no host
            // can know it, the other two because a port that is read is a
            // port that is served, and both are checked against the seat the
            // page arithmetic is carved from.
            //
            // The four ports above that — `pages`, `page_indptr`, `w_slot`,
            // `w_off` — are NOT claimed, and the mask says so. A
            // decode-envelope lane's page table is this shell's own
            // (`KvDelta::pages` empty is the submission saying so), and
            // `store::kv::geometry_with` derives all four from the seat;
            // claiming them would mean letting a guest's page ids reach the
            // pool, which is the pooled device-geometry class and a different
            // piece of work. A `DeviceGeometry` binding is still refused at
            // bind, by name.
            ports: PortMask::DECODE_ENVELOPE,
            geometry: GeometryClass::DecodeEnvelope,
            kv_copy: KvCopyDomains::default(),
            kv_handle: None,
            media_encode: false,
            // **F2a: THE RINGS ADVANCE ON THE DEVICE.** `register_channel`
            // below hands out the pinned host half of every channel, the
            // fire's tickets are validated by `channel::pull_validate`, and
            // `channel::commit_bump` is the only writer of durable ring
            // state. So a caller may predict cursors by counting, and its
            // pump has nothing left to carry.
            device_channel_commit: true,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                arena_bytes,
                pool_bytes,
                input_bytes,
            },
            caps,
        })
    }

    fn register_adapter(
        &mut self,
        registration: &engine::engine_api::adapter::AdapterRegistration,
    ) -> EngineResult<()> {
        // `palo C2`: design §8's banks are declared by the model text and
        // reserved at load; this is the write. No graph is touched — the key
        // is a fire's composition and a bank's contents are not in it — so a
        // registration between two fires costs a copy and leaves every
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
        // ── ARTICLE 4, THE ONLY DOOR. Every step is checked before any of
        //    them runs, and `Step::validate` is where the check lives — the
        //    contract wrote the arithmetic once and a second spelling of it
        //    here would be a second thing to keep in step.
        // The engine's own answer about channel tickets rides in: a
        // prediction this shell validates on the device is accepted, and one
        // it could only ignore is refused by name (`Lane::validate_for`).
        frame.validate_for(self.caps.as_ref().is_some_and(|caps| caps.device_channel_commit))?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        // **THE PREVIOUS FRAME'S NUMBERS DIE HERE** and the drop is the whole
        // of the rule: the out seam is an arena rectangle this frame is about
        // to carve over, so a caller that wanted numbers had to ask before now
        // (`Cuda::settle_frame`, which says so by refusing).
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut settled = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            // ── THE NEXT STEP, STATED (`Engine::expect_fire`, advisory).
            //    The fold's prebind applies the SUCCESSOR's cached binding to
            //    a seat that is not in flight, inside this step's own hidden
            //    window, so the successor has to be on the table before this
            //    step fires. It used to be stated by the runtime's per-step
            //    loop; a frame that crosses whole is a frame whose successors
            //    the engine already knows, so the hint is derived here instead
            //    of restated there. The lookahead ACROSS frames stays the
            //    runtime's, because only the scheduler can see the launch
            //    queued behind this one.
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }
            // ── ARTICLE 1, AND THE ERROR ARM IS ARTICLE 4's OTHER HALF. A
            //    step that faults POISONS THE FRAME'S REMAINING STEPS — the
            //    loop simply stops, so nothing after it is prepared or
            //    enqueued — and the steps already airborne settle normally,
            //    because they are real work the device is really doing and
            //    pretending otherwise would leave their staging slots and
            //    their events held forever. The runtime hears one failure for
            //    the frame (it settles every terminal cell FAILED) and drops
            //    the frame's registration, so the completions that arrive
            //    afterwards for steps 0..k find nothing to resurrect.
            let at = engine::engine_api::StepDone {
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

    /// **Yes** — and this is the fact the runtime branches on (design §2,
    /// article 1).
    ///
    /// `submit` returns with the device still running: every step's launches
    /// are on the compute stream, every step's settlement is registered behind
    /// an event on the notify stream, and not one host read stands between
    /// them. The receipts carry ids and empty readouts; the outcomes arrive on
    /// [`Engine::on_complete`]'s sink.
    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn on_complete(&mut self, sink: engine::engine_api::CompletionSink) {
        self.sink = Some(sink);
    }

    /// **Fill in the last submitted frame's readouts** (design §4's readback
    /// obligation).
    ///
    /// Waits for the compute stream and takes the same two reads F1's settle
    /// took, off the same rectangles, in the same order — so what a caller
    /// gets here is byte-identical to depth-1 execution.
    ///
    /// **AND IT REFUSES A FRAME THE ARENA NO LONGER HOLDS.** The out seam and
    /// the export columns are arena rectangles the next fire carves over, so a
    /// caller that submits again before it asks has asked too late. That is a
    /// named refusal rather than a silent wrong answer, because the bytes at
    /// those addresses after a second submit are a real logits rectangle
    /// belonging to somebody else's fire and nothing about them looks wrong.
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
            shell.read_out(&mut step.settled).map_err(fault)?;
        }
        for (receipt, step) in ticket.steps.iter_mut().zip(&settled) {
            receipt.readouts = readouts_of(step);
        }
        self.pending = Some((ticket.id, settled));
        Ok(())
    }

    /// The contract's advisory, landed on [`Shell::expect`]: the fold's
    /// prebind seam (`.wiki/palo/cuda-abi.md` §6d) applies the hinted
    /// composition's cached binding to an exec that is not in flight, after
    /// the next fire's launch and before its sync.
    ///
    /// Only the composition crosses — `Shell::expect` reads each lane's
    /// `word` and `tokens.len()` and nothing else, so the borrow of the
    /// token vectors here is a shape statement, not a content one. Nothing
    /// of `fire`'s validation runs: a hint the artifact cannot compose is
    /// dropped inside `expect` (the fire that actually submits it will say
    /// why), and a hint before a load has no shell to warm and warms
    /// nothing — both are the advisory contract's "a wrong hint costs only
    /// the hidden work", not errors.
    fn expect_fire(&mut self, submission: &Step) {
        let Ok(shell) = self.loaded_mut() else {
            return;
        };
        let lanes: Vec<Lane<'_>> = submission
            .lanes
            .iter()
            .map(|lane| Lane {
                slot: lane.slot,
                word: lane.word,
                tokens: &lane.tokens,
            })
            .collect();
        shell.expect(&lanes);
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
        // **REGISTRATION IS WHERE THE HOST END IS ALLOCATED** (alto design
        // §5, and it is what changed in F2a). What stood here refused: a
        // channel's DEVICE ring is carved inside `Session::bind`, from the
        // package's own declarations, so a standalone registration had
        // nothing to allocate and the caller owned the host ring it pumped
        // cells across from.
        //
        // The pump is what died. A guest's cell now crosses by DEVICE ACCESS
        // to mapped pinned memory — `channel::pull_validate` reads it where
        // the guest wrote it and `channel::scatter_publish` writes the answer
        // where the guest will read it — and the one thing that requires is
        // that the two agree on WHICH BYTES. This is that agreement: the
        // engine allocates the mirror and the four control words, and the
        // caller makes its ring a view of them rather than a second
        // allocation.
        //
        // A channel with no host end allocates nothing and answers no
        // mirror: its cells never leave the device.
        //
        // **THE CAPACITY REFUSAL LIVES HERE** (track-K finding 3). The
        // control kernels take `cap1 - 1` unsigned and `expected_head %
        // cap1`, reproducing dev's arithmetic unchanged, so a ring the
        // arithmetic cannot carry has to be refused before anything is
        // allocated against it — `Endpoint::open` is where both bounds are
        // written down.
        if self.channels.contains_key(&registration.id) {
            return Err(Error::Program(format!(
                "channel {} is already registered on this engine",
                registration.id
            )));
        }
        let mirror = if registration.host_role == engine::tensor_ir::container::HostRole::None {
            None
        } else {
            let numel = registration
                .shape
                .iter()
                .map(|&dim| dim as usize)
                .product::<usize>()
                .max(1);
            let cell_bytes = u32::try_from(engine::wire_cell_bytes(
                registration.dtype.program_dtype(),
                numel,
            ))
            .map_err(|_| {
                Error::Program(format!(
                    "channel {}'s wire cell is wider than a u32 counts",
                    registration.id
                ))
            })?;
            let capacity = registration.capacity.max(1);
            let endpoint = Arc::new(
                crate::program::Endpoint::open(registration.host_role, cell_bytes, capacity)
                    .map_err(fault)?,
            );
            let published = HostMirror {
                mirror: endpoint.mirror_host(),
                words: endpoint.words_host(),
                cell_bytes,
                capacity,
            };
            self.channels.insert(registration.id, endpoint);
            Some(published)
        };
        Ok(RegisteredChannel {
            id: registration.id,
            // ZERO, AND THE CONTRACT SAYS WHAT ZERO MEANS: this shell keeps no
            // waker table — parking and waking are the runtime's, on its own
            // threads — so it mints no wait slot and says so rather than
            // inventing an id it would never signal.
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        // REFUSED AT BIND, WHICH IS THE CONTRACT'S OWN READING: a class is a
        // claim about which descriptor ports the device resolves, and the
        // caps this load answered say which those are. Asked through
        // `Capabilities::admits` rather than by re-deriving the subset test
        // here, because the contract wrote that negotiation down once and a
        // second spelling of it is a second thing to keep in step. A program
        // bound in a class this load does not serve would otherwise fail at
        // its first fire, against a descriptor nobody wrote.
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
        // THE EXTENTS ARE THE CALLER'S ANSWER, NOT A GUESS (Build log 15).
        // Every stage's fire-path buffers are carved at this call, from these
        // numbers; `sampled_rows` is the one a model fire's epilogue reads,
        // and one carved for a single row when the fire hands it four
        // zero-fills three rows that no launch faults on. Converted here
        // because `BindExtents` is the contract's spelling and `Extents` is
        // the plane's — the same seven roles, and the tags are
        // `ExtentRole`'s in both.
        // **THE DENSE INDEX MEETS THE CHANNEL ID, ONCE.** `InstanceBinding`
        // names this instance's channels in the package's declaration order,
        // which is the only place the caller's ids and the program's dense
        // slots are related — so the endpoints registration allocated are
        // gathered into dense order here and nowhere else. A channel this
        // engine never registered contributes `None` and the session opens a
        // fresh host end for it, which is the shape a caller driving
        // `bind_instance` directly is in.
        let adopted: Vec<Option<Arc<crate::program::Endpoint>>> = binding
            .channels
            .iter()
            .map(|id| self.channels.get(id).cloned())
            .collect();
        let id = self
            .loaded_mut()?
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
                &adopted,
                &binding.channels,
            )
            .map_err(fault)?;
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> EngineResult<()> {
        self.loaded_mut()?
            .close_program_instance(id)
            .map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        // **THE PINNED HALF DIES HERE, AND ONLY HERE.** A caller that adopted
        // the mirror holds raw addresses into this allocation, so freeing it
        // is a fact the caller has to have asked for. Closing a channel this
        // engine never registered is not an error — a channel with no host end
        // allocated nothing — so this is idempotent by construction rather
        // than by a tolerated refusal.
        self.channels.remove(&id);
        Ok(())
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> EngineResult<bool> {
        // THE DOOR THAT REPLACED THE POINTER. The runtime used to write a cell
        // into this ring itself, through the addresses `ChannelBinding`
        // published; it owns a host ring of its own now and hands the bytes
        // over here (`runtime::engine::channel`'s header, and the trait's).
        self.instance(instance)?.publish(channel, cell).map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> EngineResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    // `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
    // default bodies. See the module header: the v1 shell genuinely has
    // none of the four, and a stub that answered `Ok(())` would make a
    // prefix-cache hit, a swap or an image prompt appear to work.
}

impl Cuda {
    /// One step of an admitted frame, run to completion.
    ///
    /// **THE DOUBLE DOOR, CLOSED** (alto design §9). What stood here was a
    /// loop asking `program_ready` over every attachment so that a blocked
    /// guest could be answered `Error::Exhausted` — and then `serve`'s own
    /// gate asked the identical question a few microseconds later, because it
    /// has to: an epilogue that discovers its rings are not ready AFTER the
    /// forward is a fire nobody can retry. Two doors, one question, and the
    /// only reason for the first one was that the shell spoke `Fault` and
    /// `Exhausted` is a contract word. `Fault::Blocked` is that word in the
    /// shell's language and `fault()` is the translation, so the question is
    /// asked once now, where the answer is load-bearing.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::engine_api::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

        let done = self
            .sink
            .as_ref()
            .map(|sink| crate::serve::Done {
                at,
                sink: std::sync::Arc::clone(sink),
            });
        let shell = self.loaded_mut()?;
        let seated: Vec<Seated<'_>> = submission
            .lanes
            .iter()
            .map(|lane| {
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
                if !lane.kv.translation.is_empty() {
                    // A fork moved this lane's pages, and the runtime states
                    // both the old ids and the new. The shell reads the page
                    // table it is given and never the one before it, so a
                    // translation is either already applied to `pages` — in
                    // which case stating it is redundant — or it is not, in
                    // which case honouring it needs a page mover this shell
                    // does not have (`copy_kv`, unsupported below).
                    return Err(Error::Unsupported {
                        verb: "kv page translation",
                        engine: "cuda",
                    });
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: lane.word,
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    // `palo B-mask`, closed on this side. The `masked` fact
                    // is a declared axis (design §0/§8) and the mask itself
                    // is a runtime input, so the shell carries the bits and
                    // the PLAN decides whether anything reads them: a mask
                    // against an artifact with no `attention.masked` arm is
                    // `Fault::Maskless`, named at the fire rather than
                    // refused for every model here. The word the runtime
                    // stamped is what puts the lane in the masked class, and
                    // `compose` refuses a word this artifact has no class for
                    // — so a mask and a word that disagree cannot both pass.
                    mask: lane.mask.as_ref(),
                    // `palo C2`, closed on this side, and by the same
                    // argument the mask closed by one line up: an adapter is
                    // a declared axis, the id is a runtime input, and the
                    // PLAN decides whether anything reads it. An id against
                    // an artifact with no `linear.lora_correct` arm is
                    // `Fault::Adapterless`; an id that disagrees with the
                    // word the runtime stamped is `Fault::AdapterWord`. Both
                    // are named at the fire, before anything launches, rather
                    // than refused for every model here.
                    adapter: lane.adapter,
                    // `palo C3b`/`C4b`, closed the same way once more, and
                    // the only thing that changes is that these two carry no
                    // runtime input at all. What the submission states is the
                    // INTENT, and the plan decides whether the artifact has
                    // an arm for it: a draft ask against a text that declares
                    // no head is `Fault::Draftless`, a capture ask against
                    // one with no capture arm is `Fault::Scoreless`, and
                    // either ask disagreeing with the word the runtime stamped
                    // is `Fault::DraftWord` / `Fault::ScoreWord`.
                    drafts: lane.drafts,
                    captures_scores: lane.captures_scores,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        // ── THE CALLER'S PREDICTION, CHECKED AGAINST THIS ENGINE'S
        //    (article 3). A lane carries the tickets the runtime minted for
        //    the instance attached to it; this shell mints its own from the
        //    same counting, validates THOSE on the device, and refuses loudly
        //    when the two disagree — because in F2b the caller's is the only
        //    one that can be right, and a silent resolution in the engine's
        //    favour now would be a wrong cell then.
        for attachment in &submission.attachments {
            let Some(lane) = submission.lanes.get(attachment.lane as usize) else {
                continue;
            };
            if lane.channels.is_empty() {
                continue;
            }
            if let Some(why) = shell.program_ticket_disagreement(attachment.instance, &lane.channels)
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
        // ── THE THREE PHASES, AND THE VERB RETURNS BEFORE THE DEVICE DOES
        //    (articles 1 and 2). `prepare` makes every host decision and
        //    claims a staging slot; `enqueue` puts the whole step on the
        //    compute stream and nothing else; `settle_step` records an event,
        //    points the notify stream at it and hangs the completion callback
        //    off that — three enqueue-only calls and not one wait.
        //
        //    `Shell::fire_captured` is the SYNCHRONOUS spelling of the same
        //    three, with `read_out` on the end; it is what the native surface
        //    and the arming pass use, and it is deliberately not what this
        //    path calls.
        // ── THE FOLD'S ARMING INSTANT, BEFORE ANY OF THIS FIRE'S STAGING
        //    (design §4: arming is control plane). It used to be the first
        //    line of `Shell::fire_captured`, which was the only door onto the
        //    phases; this path drives them directly, so it calls the door by
        //    name. Nothing here can fail a fire — see `Shell::arm_if_due`.
        shell.arm_if_due(&seated);

        let settled = {
            use engine::frame::Shell as FrameShell;
            let prepared = FrameShell::prepare(
                shell,
                crate::serve::StepView {
                    lanes: &seated,
                    attachments: &attached,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            shell.settle_step(enqueued, done).map_err(fault)?
        };

        // **EMPTY READOUTS, AND THE CONTRACT ALREADY SAID SO** (`FireTicket`'s
        // own doc: "an asynchronous shell answers with the id and an empty
        // readout list"). The numbers, for a caller that wants them, come from
        // `Engine::settle_frame`; the runtime never asks, because a guest
        // reads its logits on the device through the epilogue's intrinsic.
        for lane in &submission.lanes {
            if let Readout::Rows(_) = lane.readout {
                // palo B-readout: reading an interior row means keeping the
                // whole logits rectangle addressable after the walk, which the
                // arena does — `slots.0[out]` is the rectangle — but the row
                // list has to reach the read-back loop, and `Shell::read_out`
                // answers one row per lane by design.
                return Err(Error::Unsupported {
                    verb: "row-selected readout",
                    engine: "cuda",
                });
            }
        }
        Ok((
            FireTicket {
                id,
                readouts: Vec::new(),
            },
            PendingStep {
                readout: submission.lanes.iter().map(|lane| lane.readout.clone()).collect(),
                settled,
            },
        ))
    }
}

/// **ERROR ATTRIBUTION, WHICH IS WHAT THE SYNC USED TO BUY** (alto F2b).
///
/// F1's settle ended in `cudaStreamSynchronize` and one of the five things
/// that sync guarded was the name on a fault: every launch is enqueue-only, so
/// an asynchronous device fault surfaces at the next BLOCKING call, and a
/// per-fire sync is what made "the next blocking call" be this fire's own.
///
/// With settlement asynchronous there is no such call, and the honest thing is
/// not to pretend: a fault detected while `n` earlier steps are still airborne
/// may belong to any of them. So the sentence says so. It is a note on the
/// message rather than a different error, because WHICH error it is — a
/// scheduling refusal the caller retries, a permanent one it does not — is a
/// separate question that this must not disturb; `Exhausted` and `Impossible`
/// are decided on the host before anything launches and pass through
/// untouched.
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

/// **One step's readouts, as the contract shapes them.**
///
/// The shell answers one row of logits per lane and the capture column beside
/// it; which of those a lane actually asked for is `Readout`, which is the
/// SUBMISSION's word and never crosses into the shell.
///
/// **A LANE THAT CAPTURED IS ANSWERED WHATEVER ITS `Readout` SAYS.** `Readout`
/// chooses which LOGITS rows come back; the capture is a different column with
/// a different reader, and a lane that asked for its mass and set
/// `Readout::None` for its logits asked for both of those things and meant
/// them.
fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let mut out = Vec::with_capacity(step.readout.len());
    for (lane, want) in step.readout.iter().enumerate() {
        let scores = step.settled.scores.get(lane).cloned().unwrap_or_default();
        let values = step.settled.logits.get(lane).cloned().unwrap_or_default();
        out.push(match want {
            Readout::None => LaneReadout {
                scores,
                ..LaneReadout::default()
            },
            Readout::Last => LaneReadout {
                rows: 1,
                width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                values,
                scores,
            },
            // Refused at submit; a `Vec` of nothing is the only honest answer
            // if one ever reached here.
            Readout::Rows(_) => LaneReadout::default(),
        });
    }
    out
}

// STILL `unsafe impl`, and for the reason `Shell` has always needed one: it
// holds the device's own raw handles inline (a `cublasContext`, CUDA events,
// the arena's `c_void` bases), none of which is `Send` to the compiler, and
// taking C out of the CALL did not change what a CUDA context is. What makes
// it sound is the engine's own rule: every verb takes `&mut self`, so exactly
// one thread touches a shell at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}
