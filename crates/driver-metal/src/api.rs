//! `impl Driver for Metal` — the shell behind the contract.
//!
//! # Why a wrapper and not `impl Driver for Shell`
//!
//! Because a [`Shell`] **is a loaded model**. `Shell::load` binds the device,
//! compiles the plan, lands the checkpoint and reserves the pools in one
//! call, and every other method on it is about that load. The contract's
//! [`Driver`] is the other shape: a caller opens a driver first
//! (`engine::driver::backend::open::metal`, from a boot config that has no
//! model in it), registers it, and only then calls [`Driver::load`] with a
//! traced `Plan`. There is no `Shell` to have a `Driver` impl on until the
//! verb that makes one has been called.
//!
//! So [`Metal`] is a `Shell` that has not happened yet: the device knobs a
//! boot config states, an `Option<Shell>` that `load` fills, and the
//! [`Capabilities`] that load answered. Every verb before a load is a
//! refusal with a sentence.
//!
//! # The contract the wrapper cannot state, and how it is supplied
//!
//! [`LoadRequest`] carries `{ plan, checkpoint, budgets, ordinal }` and NOT a
//! `ModelContract` — deliberately, because `driver-api`'s dependency floor is
//! `model-ir`, `tensor-ir`, `serde`, `thiserror` (its own header), and a
//! contract type in it would put `model-loader` in the graph of everyone who
//! reads a `KvHandle`. But [`Weights::resident`](crate::weights::Weights)
//! needs one: how a checkpoint's tensors become this plan's params is the
//! MODEL's declaration, and the shell must not grow an arm per family to
//! rediscover it (`weights.rs`'s own header).
//!
//! The resolution is a function pointer, installed when the driver is opened:
//!
//! ```text
//!   engine (links `model`)                    driver-metal (links no family)
//!   ----------------------                    ------------------------------
//!   fn contract_for(plan, path) -> Contract ─▶ Metal::new(boot, contract_for)
//!     model::import_of(plan.name)                 … load(request) calls it
//! ```
//!
//! One pointer, resolved by the party that already links the catalog, and no
//! model name anywhere in this crate outside its own dev-dependencies.
//!
//! # What this driver does not serve, and says so by name
//!
//! **EVERY ABSENCE BELOW IS A REFUSAL, NOT A SILENT DROP.** The metal
//! [`Shell`] is genuinely smaller than the CUDA one — `Seated` is
//! `{ lane, pages, held, mask }` and `Shell::fire_seated(&[Seated])` is the
//! whole fire door — so this wrapper is handed submission fields that have
//! nowhere on this plane to go. Dropping one would make an adapter-routed
//! lane, a draft ask or a score capture *appear* to have been honoured and
//! then answer the plain continuation, which is the failure mode the
//! contract's "refusal is a value" section exists to prevent.
//!
//! * `register_adapter` — design §8's banks. The RESIDENCY exists:
//!   `weights.rs` reserves and zeroes a bank for any plan that declares one,
//!   and [`Weights::register_adapter`](crate::weights::Weights) writes planes
//!   into it. What does not exist is anything that READS one — `kernels-metal`
//!   stubs `linear.lora_correct` (`serve.rs`'s own "what this plane refuses")
//!   — and, consequently, no [`Shell`] door onto that write. §8's standing
//!   open item.
//! * A lane's `adapter`, `drafts` and `captures_scores` — the three declared
//!   export axes. The metal `Seated` carries none of them, for the reason one
//!   line up: the dispatch layer stubs the ops they would run in.
//! * `attachments` — a guest program at a fire BOUNDARY. The guest-program
//!   plane is fully served (register, bind, publish, take, fire, stats), but
//!   it fires BESIDE a model fire and never inside one: there is no
//!   `fire_attached` here, because attaching means binding
//!   `IntrinsicId::Logits` at the arena's out-seam rectangle and this fire
//!   path reads that rectangle back to the host instead. A caller that wants
//!   a pass runs it through [`Shell::fire_program`], which
//!   [`Metal::shell_mut`] hands over.
//! * `LoadRequest::ordinal` — see [`DeviceBoot`]. `MTLCreateSystemDefaultDevice`
//!   takes no ordinal, so a request that names one is refused rather than
//!   quietly given the default device.
//! * `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
//!   default bodies, which answer [`DriverError::Unsupported`]. The pools are
//!   not virtual, there is no peer-copy path, no recurrent-state mover and no
//!   multimodal encoder.
//! * `register_channel` and `close_channel` — as on the CUDA plane, and for
//!   the same reason: binding IS registration here.

use std::path::{Path, PathBuf};

use driver::driver_api::Driver;
use driver::driver_api::adapter::AdapterRegistration;
use driver::driver_api::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use driver::driver_api::channel::{ChannelId, ChannelRegistration, RegisteredChannel};
use driver::driver_api::error::{DriverError, Result as DriverResult};
use driver::driver_api::fire::{FireId, FireSubmission, FireTicket, LaneReadout, Readout};
use driver::driver_api::load::{Budgets as LoadBudgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
use driver::driver_api::program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
use driver::driver_api::tensor_ir::registry::{GeometryClass, ModelProfile, PortMask};
use driver::driver_api::tensor_ir::types::DType;
use driver::driver_api::transfer::MemoryDomain;
use model_compiler::{Budgets, DeviceProfile};
use model_ir::Plan;
use model_loader::contract::ModelContract;

use crate::error::Fault;
use crate::program::Session as ProgramSession;
use crate::serve::{Boot, Lane, Seated, Shell};

/// How a caller answers "what does this checkpoint's bytes mean for this
/// plan".
///
/// See the module header: the contract has no seat for a `ModelContract` and
/// this crate must not know a model family, so the party that links the
/// catalog supplies the lookup. Identical to the CUDA sibling's type on
/// purpose — one door, one signature, whichever shell is behind it.
pub type ContractFor = fn(&Plan, &Path) -> std::result::Result<ModelContract, String>;

/// The device knobs a boot config states, before any model is loaded.
///
/// **IT IS EMPTY TODAY, AND THAT IS THE HONEST ANSWER RATHER THAN AN
/// OVERSIGHT.** The CUDA twin carries two fields and this plane has neither:
///
/// * no `ordinal`, because Metal does not select by number.
///   `MTLCreateSystemDefaultDevice` names the machine's one GPU and takes no
///   argument ([`Context::bind`](crate::device::Context::bind)), so there is
///   nothing for a deployment to state and nothing for a request to override.
/// * no `graphs`, because there is no capture to choose a mode of. Design §6
///   rules it out in the tree itself — *"no record.rs: dispatch is
///   encode-only, so `EagerSink` per fire IS encoding"* — and `serve.rs`'s
///   header argues why: a Metal dispatch is already only an encode into a
///   command buffer, so a fire IS the capture, submitted instead of replayed.
///
/// The type is kept rather than dissolved into a one-argument
/// [`Metal::new`] for one reason and it is a seam reason: the engine's
/// backend door opens every driver the same shape
/// (`Metal::new(DeviceBoot::default(), contract_for)`), and the knob this
/// plane will grow first — the *indirect command buffer* `serve.rs` names as
/// a future note — is a boot-time choice that lands here. An empty seat that
/// is documented as empty costs one `::default()` at the door; a signature
/// that changes when the first knob arrives costs the door.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceBoot;

/// The Metal shell, behind [`Driver`].
pub struct Metal {
    /// Empty today — see [`DeviceBoot`]. Held rather than dropped so the
    /// first knob has somewhere to arrive without moving the seam.
    #[allow(dead_code)]
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
}

impl Metal {
    /// A driver bound to nothing yet.
    #[must_use]
    pub fn new(boot: DeviceBoot, contract_for: ContractFor) -> Metal {
        Metal {
            boot,
            contract_for,
            shell: None,
            caps: None,
            next_fire: 1,
        }
    }

    /// The loaded shell, for a caller that wants the native surface — the
    /// compiled-pipeline counter, the footprint, the guest-program pass.
    #[must_use]
    pub fn shell(&self) -> Option<&Shell> {
        self.shell.as_ref()
    }

    /// The loaded shell, mutably.
    ///
    /// **THE ONE DOOR TO A GUEST PASS ON THIS PLANE.** The contract fires a
    /// program by ATTACHING it to a model fire, and this shell has no
    /// attachment path (module header); [`Shell::fire_program`] runs one on
    /// its own and is reached through here.
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
    /// [`KvDelta`](driver::driver_api::KvDelta) says a lane whose `pages` are
    /// empty is one whose page table the SHELL owns, and a shell that owns it
    /// opens the slot itself. An engine that keeps its own page table never
    /// calls this — it states the same fact as `KvDelta::held == 0`, and the
    /// fire path clears the slot's recurrent banks on reading it.
    ///
    /// # Errors
    ///
    /// [`DriverError::Load`] before a load, and whatever the pools said.
    pub fn open(&mut self, slot: u32) -> DriverResult<()> {
        self.loaded_mut()?.open(slot).map_err(fault)
    }

    /// One bound instance's session, for the two channel doors.
    ///
    /// # Errors
    ///
    /// [`DriverError::Load`] before a load, [`DriverError::Closed`] for an
    /// instance this plane does not carry — `Closed` and not `Program`
    /// because "this handle is gone" is what the caller can act on, and a
    /// channel door is exactly where a torn-down instance is discovered.
    fn instance(&mut self, id: InstanceId) -> DriverResult<&mut ProgramSession> {
        self.loaded_mut()?
            .program_instance(id)
            .ok_or(DriverError::Closed {
                what: "instance",
                id,
            })
    }

    #[allow(dead_code)]
    fn loaded(&self) -> DriverResult<&Shell> {
        self.shell
            .as_ref()
            .ok_or_else(|| DriverError::Load("the metal driver has no model loaded".into()))
    }

    fn loaded_mut(&mut self) -> DriverResult<&mut Shell> {
        self.shell
            .as_mut()
            .ok_or_else(|| DriverError::Load("the metal driver has no model loaded".into()))
    }
}

/// The shell's refusal, in the contract's vocabulary.
///
/// **THE TAXONOMY IS THE POINT.** `Exhausted` and `Impossible` are scheduling
/// answers the engine's lane loop acts on — retry behind something that frees
/// pages, or drop the request — and everything else is a failure it logs. A
/// [`Fault::Ceiling`] is `Impossible` and not `Exhausted` because every
/// ceiling this shell states was reserved at LOAD: no amount of freeing makes
/// a pool carved for 256 slots seat a 257th, which is exactly the distinction
/// `driver_api::error`'s header draws.
///
/// **THERE IS NO `Exhausted` ARM HERE, AND THE ABSENCE IS THE MAP.** The CUDA
/// twin has one because `cudaMalloc` reports a shortfall as two numbers
/// (`Fault::OutOfMemory { need, have }`). Metal has no such variant to map
/// from: a device that will not give up bytes answers `nil` from
/// `newBufferWithLength:options:` and becomes [`Fault::Device`], and the one
/// place this shell knows both numbers — a reservation past
/// `maxBufferLength` — is a [`Fault::Ceiling`], which is a device property
/// and not a moment. Inventing an `Exhausted` out of either would tell the
/// lane loop to retry something that cannot change.
fn fault(fault: Fault) -> DriverError {
    match fault {
        // The machine's, not the submission's. `Deviceless` is a build with
        // no Metal in it — a non-Apple target, or an Apple one with no GPU
        // published — and it reaches a caller through exactly the same door
        // as a Metal call that refused, because both mean "the device half
        // could not answer".
        Fault::Deviceless | Fault::Device { .. } => DriverError::Device(fault.to_string()),
        // The load axis: a plan these budgets do not admit, a checkpoint the
        // contract does not fit, a param that never published, a seat the
        // plan names and this shell binds none of. `Shader` joins them, and
        // it is `Unbound`'s sibling rather than a device condition: an
        // entrypoint this crate does not SHIP is a fact about the build and
        // the model text, discovered at the node that needs it, and no retry
        // of any submission changes it. Mapped through `to_string` so the
        // Metal compiler's own paragraph — the reason `error.rs`'s header
        // keeps a `String` where CUDA keeps an `i32` — survives the crossing.
        Fault::Bake(_)
        | Fault::Load(_)
        | Fault::Param { .. }
        | Fault::Shader { .. }
        | Fault::Unbound { .. } => DriverError::Load(fault.to_string()),
        Fault::Ceiling { what, need, have } => DriverError::Impossible(format!(
            "this fire wants {need} {what} and the load reserved {have}"
        )),
        // A region whose classes this fire's order does not make consecutive
        // is a BAKE-integrity break, not a submission the caller can fix; so
        // is a schedule built over more classes than its reader runs.
        Fault::Fragmented { .. } => DriverError::Device(fault.to_string()),
        Fault::Straddled { .. } => DriverError::Load(fault.to_string()),
        // A mask that does not describe its lane, one against a plan with no
        // masked arm, or one whose word says the other thing, is the
        // SUBMISSION's — the caller stated it and the caller can state it
        // differently. `Invalid`, not `Impossible`: a retry with a mask of
        // the lane's own extent is a real answer. On this plane
        // `Fault::Maskless` also covers the unconditional refusal
        // `Seated::mask` documents — no mask bits are staged yet — and it is
        // still the caller's field, still fixable by dropping it.
        Fault::Mask { .. } | Fault::Maskless { .. } | Fault::MaskWord { .. } => {
            DriverError::Invalid(fault.to_string())
        }
        // An adapter registration this load's banks cannot seat is the
        // CALLER's too, and for the same reason the mask arms are: the bank
        // is a shape the model text declared and the planes are bytes the
        // caller assembled, so a plane of one slot's width is a real retry.
        // It reaches this shell at all because a bank is a
        // `ParamSource::Registered` weight and `Weights` reserves one for any
        // plan that declares it — what does not run is the CORRECTION, which
        // `linear.lora_correct` refuses by name one dispatch later.
        Fault::Adapter { .. } => DriverError::Invalid(fault.to_string()),
        // The guest-program plane's two, both `Program`, which is the word
        // the contract reserves for it. `Compile` carries `driver::Failure`'s
        // deterministic/retryable split and `Program` names the entry that
        // refused; neither is a model-fire condition and neither should reach
        // the lane loop as one.
        Fault::Compile(_) | Fault::Program { .. } => DriverError::Program(fault.to_string()),
        Fault::Fire(_) => DriverError::Invalid(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling.
///
/// Two names for one seven-role vector: [`ExtentRole`](driver::Role) is the
/// tag space both are written in, and the conversion is field for field so
/// that adding a role to one without the other is a compile error rather than
/// a silently unresolved axis.
fn extents(stated: &BindExtents) -> driver::Extents {
    driver::Extents {
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
/// The contract carries seven numbers and `model_compiler::Budgets` takes
/// four; the other three (`page_size`, `max_context`, `slots`) are the POOLS'
/// and go to [`Boot`] directly. Converted in one place, which is the whole
/// reason `driver-api` states its own `Budgets` rather than depending on the
/// compiler (`load.rs`'s note).
///
/// `max_adapters` crosses unchanged even though this plane seats no bank: it
/// is a BAKE input, and the compiler is entitled to refuse a plan that cannot
/// carve what the deployment asked for. What the load then ADVERTISES is a
/// different number, and it is zero — see [`Driver::load`].
fn bake_budgets(budgets: &LoadBudgets) -> Budgets {
    Budgets {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// The guest-visible profile of a loaded plan.
///
/// **CARRIED, NOT RECONSTRUCTED** (design §7 on `caps`): the engine used to
/// rebuild a `ModelProfile` at bind time out of eight `has_*` booleans on a
/// flat capability struct. Everything below is read off the plan and the
/// budgets — `num_layers` from the nodes' own `layer` stamps, `vocab` from
/// the width of the `out` seam — so there is one copy and nothing to keep in
/// step.
///
/// **EVERY MODEL-GATED INTRINSIC IS `false` ON THIS PLANE, AND ONE ANSWER
/// COVERS ALL OF THEM.** Each one names a rectangle a guest program would
/// bind a buffer of this fire's against, and the metal fire path produces
/// exactly one rectangle — the `out` seam's logits, which
/// [`Shell::fire_seated`] reads the last row of and hands back to the host.
/// There is no second column for an mtp draft, a value head or an attention
/// mass to stand in, and no attachment door to bind one through
/// (module header). Advertising any of them would let a program bind at
/// bind time and fail at its first fire, which is the opposite of a
/// bind-time contract.
fn profile(shell: &Shell, budgets: &LoadBudgets) -> DriverResult<ModelProfile> {
    let plan = shell.plan();
    let layers = plan
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
        // means (`ModelProfile::activation`'s own doc): this device's own
        // activation type is bf16 — the one dtype `kernels-metal` stamps —
        // and it is not what a guest program reads.
        activation: DType::F32,
        // See the item doc: one rectangle, no attachment, so no intrinsic.
        has_mtp_logits: false,
        has_mtp_drafts: false,
        has_value_head: false,
        has_attn_score: false,
        has_attn_page_mask: false,
        // And this one is `false` twice over: no `attn_page_mask` sink to
        // honour, and no `linear.lora_correct` arm to apply a delta at
        // (`serve.rs`'s "what this plane refuses"). §8's standing open item.
        has_lora: false,
        kernels: Vec::new(),
    })
}

impl Driver for Metal {
    fn kind(&self) -> &'static str {
        "metal"
    }

    fn device_facts(&self) -> Option<&DeviceFacts> {
        self.caps.as_ref().map(|caps| &caps.device)
    }

    /// **THERE IS NO THREAD TO BIND, AND THAT IS A PLATFORM FACT.** The CUDA
    /// twin rebinds here because `cudaSetDevice` is per-thread state and a
    /// context bound on the worker's boot thread strands every call the lane
    /// thread makes. An `MTLDevice` and an `MTLCommandQueue` are objects,
    /// documented thread-safe, and moving a loaded shell onto a lane thread
    /// costs nothing ([`Context::bind_thread`](crate::device::Context::bind_thread)).
    /// The verb is answered rather than refused because the engine's call
    /// order is one shape across backends, and "nothing to do" is a real
    /// answer to "you are the lane thread now".
    fn bind_thread(&mut self) -> DriverResult<()> {
        match self.shell.as_ref() {
            Some(shell) => shell.bind_thread().map_err(fault),
            None => Ok(()),
        }
    }

    fn load(&mut self, request: LoadRequest) -> DriverResult<Loaded> {
        if self.shell.is_some() {
            return Err(DriverError::Load(
                "this metal driver already has a model loaded; one shell per driver".into(),
            ));
        }
        let LoadRequest {
            plan,
            checkpoint,
            budgets,
            ordinal,
        } = request;

        // THE ORDINAL IS REFUSED RATHER THAN IGNORED. `LoadRequest::ordinal`
        // is the contract's field for "which device, when the shell serves
        // more than one", and this one serves the system default and only it
        // — `MTLCreateSystemDefaultDevice` takes no argument. A negative
        // ordinal is the contract's "unspecified" and zero is the only device
        // there is; anything above that is a caller asking for a machine this
        // plane cannot address, and answering it with the default device
        // would place a model somewhere nobody chose.
        if ordinal > 0 {
            return Err(DriverError::unsupported("metal", "device ordinal selection"));
        }

        // `Checkpoint::None` — bind and bake, land nothing — is a shape the
        // contract states and this shell has no path for: `Weights::resident`
        // is what reserves the store, and a `WeightTable` of nulls would
        // fault at the first dispatch rather than at the load that asked for
        // it. Refused by name.
        let Checkpoint::Path(path) = checkpoint else {
            return Err(DriverError::Load(
                "the metal shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        let contract = (self.contract_for)(&plan, &path).map_err(DriverError::Load)?;

        let shell = Shell::load(Boot {
            plan,
            contract: &contract,
            checkpoint: &path,
            budgets: bake_budgets(&budgets),
            // `None` takes this device's own core count AND `side_streams: 0`
            // — the metal reading of P6, stated in `Shell::load` rather than
            // here so that a reader of the shell learns it. Costs are input,
            // not knowledge (design §6): a deployment that has measured its
            // own would state them, and one that has not still bakes
            // something that runs.
            profile: None::<DeviceProfile>,
            page_size: budgets.page_size,
            context: budgets.max_context,
            slots: budgets.slots,
        })
        .map_err(fault)?;

        let plan_name = shell.plan().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        let paging = shell.paging();
        let profile = profile(&shell, &budgets)?;

        let caps = Capabilities {
            device: DeviceFacts {
                backend: "metal".to_string(),
                // **SHARED, AND IT IS ASSERTED RATHER THAN ASSUMED.** Every
                // reservation this shell makes is `StorageModeShared`, and
                // `Context::bind` REFUSES a device that answers `false` to
                // `hasUnifiedMemory` — because `Buffer` writes and reads
                // through `contents()`, which is only the device's own bytes
                // on a machine where the CPU and the GPU share them. So
                // `MetalShared` is not the optimistic half of a
                // `MetalShared`/`MetalPrivate` pair: it is the only domain a
                // load that got this far can have.
                domain: MemoryDomain::MetalShared,
                // **NOT MEASURED, AND ZERO SAYS SO.** Metal publishes no SM
                // count and no core count; `Context::cores` is a stated
                // stand-in that feeds the compiler's cost model and nothing a
                // kernel argument reads. Reporting that stand-in here would
                // dress a constant as a probe, so the caps say what is true:
                // this plane does not measure its parallel width.
                sms: 0,
                unified_memory: true,
                // Neither is probed, and neither is reachable. `kernels-metal`
                // stamps ONE dtype (bf16) across the whole op set, so there is
                // no fp8 arithmetic arm and no mxfp4 MoE GEMM to select — the
                // question a caller would be asking has no second answer here.
                fp8_native: false,
                native_mxfp4_moe: false,
                // What `weights.rs` and `inputs.rs` both carve to (`ALIGN`),
                // and therefore what a view minted into either store actually
                // satisfies. A `newBufferWithLength:options:` reservation is
                // page-aligned underneath it, so this is the ceiling of the
                // two, not the weaker of them.
                storage_alignment: 256,
                // The real ceiling is `maxBufferLength`, which the device
                // states and `Context::reserve` enforces by refusing past it
                // with `Fault::Ceiling`. This field is where it is STATED: a caller
                // planning a landing against one huge tile wants to know the
                // ceiling before it builds one, not after `Context::reserve`
                // refuses. Read off the device rather than invented — this is
                // `maxBufferLength`, which is a real device property and the
                // one `Fault::Ceiling` is raised against.
                storage_max_tile_bytes: shell.max_buffer(),
                // The guest-program plane compiles MSL through
                // `MTLDevice::newLibraryWithSource:`, which is exactly what
                // `tensor_compiler::codegen::Backend::Metal` emits and what
                // it advertises under this name.
                codegen_backend: Some("metal".to_string()),
            },
            pools: PoolFacts {
                kv_pages: u32::try_from(paging.pages()).unwrap_or(u32::MAX),
                kv_page_size: paging.page_size,
                // The recurrent pool is reserved off the plan's own
                // `CacheRow::State` rows and the shell keeps no count of its
                // slots apart from the seats it opened, which is `slots`.
                state_slots: paging.slots,
                state_slot_bytes: 0,
                // **ZERO, AND NOT BECAUSE THE MODEL DECLARED NOTHING.** The
                // CUDA twin answers the smallest capacity any one bank of the
                // model declares, which is the id ceiling a lane may route
                // against. `weights.rs` reserves those same banks here — a
                // bank is a `ParamSource::Registered` param and gets its
                // zeroed residency like any other — but nothing on this plane
                // READS one: `kernels-metal` stubs `linear.lora_correct`, so
                // `Lane::adapter` is refused at the fire and
                // `register_adapter` is refused at the door.
                //
                // This field is what a caller PLANS against, so it answers
                // the routes that will be served rather than the bytes that
                // were reserved. A non-zero count would tell a control plane
                // it may place a corrected request here, which is the one
                // thing it must not conclude. Design §8's standing open item.
                adapter_banks: 0,
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
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // **THIS SHELL RESOLVES NO DESCRIPTOR PORT ON THE DEVICE, AND
            // `NONE` IS THE WHOLE OF THAT SENTENCE.** The CUDA twin claims
            // `PortMask::DECODE_ENVELOPE` because `crate::program::ports`
            // reads `embed_tokens`, `positions` and `kv_len` off an ATTACHED
            // instance's own device rings at fire time. There are no attached
            // instances here (module header): a guest pass runs beside a fire
            // and never inside one, so no port of a guest's is ever resolved
            // against a model fire's buffers, and there is nothing for a
            // decode envelope to be carved from.
            //
            // The consequence is the one `Capabilities::admits` is written
            // for, and it is a BIND-TIME refusal rather than a fire-time
            // surprise: a program bound in `GeometryClass::DecodeEnvelope` or
            // `DeviceGeometry` is refused at `bind_instance`, by name, and
            // `GeometryClass::Host` — the class that asks for nothing — is
            // what this load serves.
            ports: PortMask::NONE,
            geometry: GeometryClass::Host,
            // **NO DIRECTION IS SERVED, AND UNIFIED MEMORY DOES NOT CHANGE
            // THAT.** It is tempting to claim the host directions on a plane
            // where `contents()` makes host and device the same bytes — a
            // "copy" to the host is not even a distinct direction here. But
            // `KvCopyDomains` describes `copy_kv`, and `copy_kv` takes the
            // trait's default body: there is no page mover on this shell at
            // all. A `true` in this record would be a promise a verb one call
            // later refuses.
            kv_copy: KvCopyDomains::default(),
            // The pools are `MTLBuffer`s reserved for this process. Metal has
            // an `MTLSharedEvent`/`IOSurface` story for cross-process
            // sharing; nothing in this shell writes one, so there is no
            // handle to export.
            kv_handle: None,
            media_encode: false,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                plan_name,
                weight_bytes,
                arena_bytes,
                pool_bytes,
                input_bytes,
            },
            caps,
        })
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> DriverResult<()> {
        // **REFUSED BY NAME, AND NOT BECAUSE THE WRITE IS MISSING.** Design
        // §8's banks are declared by the model text and reserved at load, and
        // that half is here: `Weights::register_adapter` looks a bank up,
        // checks the id against its declared capacity and the plane against
        // its slot width, and copies. What is missing is the READER —
        // `kernels-metal` stubs `linear.lora_correct` (`serve.rs`'s "what
        // this plane refuses") — so a registration that succeeded would land
        // bytes no dispatch ever reaches. The [`Shell`] publishes no door
        // onto that write for exactly this reason, and
        // `PoolFacts::adapter_banks` answers zero.
        //
        // `Ok(())` is the dangerous answer and it is the one this refuses,
        // and note that it would be dangerous even though the copy would
        // SUCCEED: a caller that registered an adapter and got a success
        // would run every following fire against the base weights and be
        // told the correction was applied. §8's standing open item, and it
        // stays open until the dispatch layer has an arm — at which point
        // this body is three lines and a `Shell` forward, not a redesign.
        let _ = registration;
        Err(self.unsupported("register_adapter"))
    }

    fn fire(&mut self, submission: &FireSubmission) -> DriverResult<FireTicket> {
        submission.validate()?;

        // ATTACHMENTS FIRST, AND BEFORE A FIRE ID IS SPENT. The CUDA twin
        // asks its readiness gate here — is every attached instance's ring
        // ready, and is a block a scheduling answer rather than a failure —
        // because it is about to run those instances INSIDE the fire. This
        // shell has no `fire_attached`: a guest pass runs beside a model
        // fire, at neither of its boundaries, through
        // `Shell::fire_program`. So there is no gate to ask, and the
        // attachment itself is what is refused — silently ignoring one would
        // run the model, answer logits, and never tell the caller that the
        // program it attached did not execute.
        if !submission.attachments.is_empty() {
            return Err(DriverError::unsupported(
                "metal",
                "guest-program attachment at a fire boundary",
            ));
        }

        let id = self.next_fire;
        self.next_fire = self.next_fire.wrapping_add(1);

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
                    return Err(DriverError::unsupported("metal", "explicit lane positions"));
                }
                if !lane.kv.translation.is_empty() {
                    // A fork moved this lane's pages, and the engine states
                    // both the old ids and the new. The shell reads the page
                    // table it is given and never the one before it, so a
                    // translation is either already applied to `pages` — in
                    // which case stating it is redundant — or it is not, in
                    // which case honouring it needs a page mover this shell
                    // does not have (`copy_kv`, unsupported above).
                    return Err(DriverError::unsupported("metal", "kv page translation"));
                }
                // THE THREE AXES THE METAL `Seated` DOES NOT CARRY, REFUSED
                // ONE BY ONE RATHER THAN DROPPED. On the CUDA plane each of
                // these is a declared axis with a runtime input, and the
                // refusal is deferred to the fire so it can name the MODEL
                // TEXT ("this artifact declares no `linear.lora_correct`
                // arm"). Here the answer does not depend on the artifact:
                // the dispatch layer stubs the ops all three would run in, so
                // no plan this shell can load has an arm for any of them, and
                // the refusal belongs at the door where it is cheapest and
                // clearest.
                //
                // What makes it worth three separate refusals is what each
                // silent drop would look like: an adapter dropped answers the
                // base model as if corrected, a draft dropped answers a
                // one-token step to a speculator expecting `k`, and a capture
                // dropped answers an empty `scores` that reads as "this lane
                // had no attention mass".
                if lane.adapter.is_some() {
                    return Err(DriverError::unsupported("metal", "adapter-routed lane"));
                }
                if lane.drafts {
                    return Err(DriverError::unsupported("metal", "mtp draft readout"));
                }
                if lane.captures_scores {
                    return Err(DriverError::unsupported("metal", "attention score capture"));
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        word: lane.word,
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    // **THE ONE AXIS THAT IS CARRIED RATHER THAN REFUSED
                    // HERE, AND IT IS STILL REFUSED — ONE LAYER DOWN.** The
                    // `masked` fact is a declared axis (design §0/§8) and the
                    // mask itself is a runtime input, so the shell carries
                    // the bits and answers for them at the fire: this plane
                    // stages no mask bits yet (`Seated::mask`'s own doc — the
                    // metal sdpa shaders read a mask plane indexed by the
                    // launch's local row, a different ABI from the CUDA
                    // shell's packed runs plus an indptr), so a lane carrying
                    // one is `Fault::Maskless`.
                    //
                    // Passed through instead of refused up here on purpose:
                    // the shell also checks the WORD against the class table
                    // in the same pass (`Fault::MaskWord`), which catches a
                    // lane whose word puts it in a masked class while its
                    // mask says otherwise. Refusing at this door would answer
                    // the first half and lose the second.
                    mask: lane.mask.as_ref(),
                })
            })
            .collect::<DriverResult<Vec<_>>>()?;

        let rows = shell.fire_seated(&seated).map_err(fault)?;

        // THE SHELL READS THE LAST ROW AND ONLY THE LAST ROW, which is
        // `Readout::Last` — the default, and the reason a prefill does not
        // hand back half a megabyte per teacher-forced position.
        let mut readouts = Vec::with_capacity(rows.len());
        for (lane, values) in submission.lanes.iter().zip(rows) {
            readouts.push(match &lane.readout {
                // `scores` is empty and stays empty: no lane on this plane
                // captured, because `captures_scores` was refused above.
                Readout::None => LaneReadout::default(),
                Readout::Last => LaneReadout {
                    rows: 1,
                    width: u32::try_from(values.len()).unwrap_or(u32::MAX),
                    values,
                    ..LaneReadout::default()
                },
                Readout::Rows(_) => {
                    // Reading an interior row means keeping the whole logits
                    // rectangle addressable after the walk, which the arena
                    // does — `slots.0[out]` is the rectangle — but the row
                    // list has to reach the read-back loop, and
                    // `Shell::fire_seated` answers one row per lane by
                    // design.
                    return Err(DriverError::unsupported("metal", "row-selected readout"));
                }
            });
        }
        Ok(FireTicket { id, readouts })
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> DriverResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> DriverResult<RegisteredChannel> {
        // BINDING IS REGISTRATION HERE, and the verb's own doc says so. A
        // channel's DEVICE ring is allocated inside `Session::bind` —
        // `program/launch.rs` carves every instance's rings when the instance
        // is bound, from the package's own channel declarations — so a
        // standalone `register_channel` has nothing to allocate. What the
        // engine registers before a bind is its HOST ring, which it owns
        // (`engine::driver::channel`), and the two are joined by
        // `publish_channel`/`take_channel` rather than by a second
        // allocation here.
        let _ = registration;
        Err(self.unsupported("register_channel"))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> DriverResult<BoundInstance> {
        // REFUSED AT BIND, WHICH IS THE CONTRACT'S OWN READING: a class is a
        // claim about which descriptor ports the device resolves, and the
        // caps this load answered say which those are — none, on this plane
        // (see `load`). Asked through `Capabilities::admits` rather than by
        // re-deriving the subset test here, because the contract wrote that
        // negotiation down once and a second spelling of it is a second thing
        // to keep in step. A program bound in a class this load does not
        // serve would otherwise fail at its first fire, against a descriptor
        // nobody wrote.
        let caps = self
            .caps
            .as_ref()
            .ok_or_else(|| DriverError::Program("bind_instance before load".to_string()))?;
        if !caps.admits(binding.geometry) {
            return Err(DriverError::Program(format!(
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
        // the plane's — the same seven roles, and the tags are `Role`'s in
        // both.
        let id = self
            .loaded_mut()?
            .bind_program(
                binding.program,
                &seeds,
                extents(&binding.extents),
                binding.geometry,
            )
            .map_err(fault)?;
        Ok(BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn close_instance(&mut self, id: InstanceId) -> DriverResult<()> {
        self.loaded_mut()?.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> DriverResult<()> {
        // As `register_channel`: a channel's life is its instance's, and
        // closing one on its own is a door this plane does not have.
        let _ = id;
        Err(self.unsupported("close_channel"))
    }

    fn publish_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
        cell: &[u8],
    ) -> DriverResult<bool> {
        // THE DOOR THAT REPLACED THE POINTER. The engine used to write a cell
        // into this ring itself, through the addresses `ChannelBinding`
        // published; it owns a host ring of its own now and hands the bytes
        // over here (`engine::driver::channel`'s header, and the trait's).
        self.instance(instance)?
            .publish(channel, cell)
            .map_err(fault)
    }

    fn take_channel(
        &mut self,
        instance: InstanceId,
        channel: u32,
    ) -> DriverResult<Option<Vec<u8>>> {
        self.instance(instance)?.take(channel).map_err(fault)
    }

    // `copy_kv`, `copy_state`, `resize_pool` and `encode` take the trait's
    // default bodies. See the module header: this shell genuinely has none of
    // the four, and a stub that answered `Ok(())` would make a prefix-cache
    // hit, a swap or an image prompt appear to work.
}

// STILL `unsafe impl`, and the CUDA sibling's rule holds over different
// contents. `Driver` is `Send + Sync` and a loaded `Shell` is neither to the
// compiler, for two separate reasons:
//
//   * it holds retained Objective-C objects inline — the `MTLDevice` and its
//     queue, every `MTLBuffer` the arena, the pools, the weight store and the
//     resident inputs reserved, every `MTLComputePipelineState` the pipeline
//     cache compiled. Apple documents all of them as thread-safe; `objc2`
//     does not mark them `Send`, which is why `Context` and `Frame` already
//     carry an `unsafe impl` each.
//   * it holds two `RefCell`s, which is what costs `Sync` rather than `Send`:
//     `Pipelines` is written through `Encode::fire`'s `&self` when a shader
//     point is compiled, and `Handles` mints a handle from `&self` when a
//     window cuts one.
//
// What makes both sound is the driver's own rule: every verb that reaches
// either takes `&mut self`, so exactly one thread touches a shell at a time.
// `kind` and `device_facts` are the only `&self` verbs on this impl and
// neither goes near the shell.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}
