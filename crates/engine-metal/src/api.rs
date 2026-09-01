//! `impl Engine for Metal` — the shell behind the contract.
//!
//! # Why a wrapper and not `impl Engine for Shell`
//!
//! Because a [`Shell`] **is a loaded model**. `Shell::load` binds the device,
//! compiles the plan, lands the checkpoint and reserves the pools in one
//! call, and every other method on it is about that load. The contract's
//! [`Engine`] is the other shape: a caller opens an engine first
//! (`runtime::engine::backend::open::metal`, from a boot config that has no
//! model in it), registers it, and only then calls [`Engine::load`] with a
//! traced `Trace`. There is no `Shell` to have an `Engine` impl on until the
//! verb that makes one has been called.
//!
//! So [`Metal`] is a `Shell` that has not happened yet: the device knobs a
//! boot config states, an `Option<Shell>` that `load` fills, and the
//! [`Capabilities`] that load answered. Every verb before a load is a
//! refusal with a sentence.
//!
//! # The contract the wrapper cannot state, and how it is supplied
//!
//! [`LoadRequest`] carries `{ trace, checkpoint, budgets, ordinal }` and NOT a
//! `ModelContract` — deliberately, because `engine`'s dependency floor is
//! `model-ir`, `eta-ir`, `serde`, `thiserror` (its own header), and a
//! contract type in it would put `checkpoint` in the graph of everyone who
//! reads a `KvHandle`. But [`Weights::resident`](crate::weights::Weights)
//! needs one: how a checkpoint's tensors become this plan's params is the
//! MODEL's declaration, and the shell must not grow an arm per family to
//! rediscover it (`weights.rs`'s own header).
//!
//! The resolution is a function pointer, installed when the engine is opened:
//!
//! ```text
//!   runtime (links `model`)                   engine-metal (links no family)
//!   -----------------------                   ------------------------------
//!   fn contract_for(trace, path) -> Contract ─▶ Metal::new(boot, contract_for)
//!     models::import_of(trace.name)                 … load(request) calls it
//! ```
//!
//! One pointer, resolved by the party that already links the catalog, and no
//! model name anywhere in this crate outside its own dev-dependencies.
//!
//! # The settlement protocol, stated once
//!
//! This engine used to settle inside `submit`: every step committed a command
//! buffer and blocked on it, `settles_asynchronously` was `false`, and the
//! receipts came back with their readouts already filled. It answers two
//! shapes now and the deployment's `frames_in_flight` chooses between them.
//!
//! ```text
//! depth 1 (the eager shell, `Runahead::F1`)
//!   submit  -> prepare/enqueue/settle each step, then `settle_frame` itself
//!           -> returns with every readout filled; `settles_asynchronously`
//!              is false and the runtime settles its own cells on the spot
//!
//! depth >= 2 (article 1's floor, and the default)
//!   submit  -> prepare/enqueue/settle each step and RETURN, device running
//!           -> receipts carry ids and EMPTY readouts
//!   device  -> each command buffer's completion handler, on Metal's own
//!              thread, publishes `StepOutcome` to `Engine::on_complete`'s
//!              sink; that is what retires the frame for the runtime
//!   numbers -> `Engine::settle_frame`, for a caller that came for them —
//!              it waits for whatever the host has not caught up with and
//!              reads the rows out of the arm's readout seat
//! ```
//!
//! **A FRAME'S NUMBERS LIVE UNTIL THE FRAMES BEHIND IT TAKE ITS SEATS.** There
//! are as many readout seats as there are arms, so `settle_frame` refuses a
//! frame a later `submit` has displaced, by name — the bytes in that seat
//! afterwards are a real logits row belonging to somebody else's step, and
//! nothing about them looks wrong.
//!
//! # What this engine does not serve, and says so by name
//!
//! **EVERY ABSENCE BELOW IS A REFUSAL, NOT A SILENT DROP.** The metal
//! [`Shell`] is genuinely smaller than the CUDA one — `Seated` is
//! `{ lane, pages, held, mask, adapter, positions }` and
//! `Shell::fire_seated(&[Seated])` is the whole fire door — so this wrapper
//! is handed submission fields that have nowhere on this plane to go.
//! Dropping one would make a draft ask or a score capture *appear* to have
//! been honoured and then answer the plain continuation, which is the failure
//! mode the contract's "refusal is a value" section exists to prevent.
//!
//! * A lane's `drafts` — ONE declared export axis the metal `Seated` still
//!   does not carry, and the list used to hold two. **THE REASON WAS ALWAYS
//!   NARROWER THAN IT LOOKED.** It used to be that neither answer had
//!   anywhere to be DELIVERED: the M2 emitter bound ONE intrinsic buffer for
//!   every `INTRINSIC_VAL` op, so a draft column and a score plane had no
//!   second rectangle to stand in. The slot table closed that —
//!   `program::launch` binds a rectangle per intrinsic and `serve.rs` points
//!   `IntrinsicId::MtpLogits` at the `mtp` export — and what was left was a
//!   different smaller thing for each axis. `drafts` is a DECLARATION the
//!   shell cannot cross-check: the CUDA sibling holds `Seated::drafts`
//!   against whether the fire's class word actually runs the draft arm, and
//!   this `Seated` has no such field, so accepting the flag would be
//!   accepting something nothing reads. (The arm itself is reachable — a lane
//!   whose `word` carries the draft class is seated and its rows are written
//!   — and an epilogue reading `mtp_logits` is served against them.) Serving
//!   an arm and dropping its column is the silent success this refusal
//!   prevents.
//! * `captures_scores` is NOT in this list any more. Its two blockers are
//!   both closed: `engine_metal::scores` carves the observability slab the
//!   capture arm writes, and `ptir_m1_runtime.metal`'s `0xA0` handler grew an
//!   arm on the intrinsic id, so an F32 score plane is read as F32
//!   (`.wiki/alto/attn-score.md` §4). The flag crosses as
//!   `Seated::captures_scores` and is cross-checked at the fire against the
//!   artifact — `Fault::Scoreless` for a bake with no capture column,
//!   `Fault::ScoreWord` for a lane whose word and whose ask disagree — which
//!   is exactly the check whose absence kept it here.
//! * `register_adapter` and `Lane::adapter` are NOT in this list any more.
//!   Design §8's banks were half here for two waves — `weights.rs` reserves
//!   and zeroes a bank for any plan that declares one and
//!   [`Weights::register_adapter`](crate::weights::Weights) writes planes
//!   into it — and what was missing was the READER. `kernels-metal`'s
//!   `linear::lora::correct` is it, the dispatch layer calls it at
//!   `Linear::LoraCorrect`, and `serve.rs` stages the per-row routes it
//!   indexes with. So the verb forwards to the shell, a routed lane crosses
//!   as `Seated::adapter`, and [`PoolFacts::adapter_banks`] answers the
//!   smallest capacity this load's banks declare rather than zero. What is
//!   still refused, by name and at the FIRE, is an id against an artifact
//!   that bakes no correction (`Fault::Adapterless`) and an id whose lane's
//!   word puts its rows outside the correction's window, or a word that puts
//!   them inside with no id to route with (`Fault::AdapterWord`).
//! * `Lane::positions` and `Lane::mask` are NOT in this list any more. A
//!   stated position run reaches rope's seat verbatim and a mask of either
//!   form expands into the sdpa entries' own plane; what each still refuses
//!   is a shape that does not describe its lane, by name, at the fire.
//! * `attachments` are NOT in this list any more. A guest program at a fire
//!   BOUNDARY is served: an epilogue's `IntrinsicId::Logits` is bound at the
//!   arena's out-seam rectangle, at the row its lane asked for, and its pass
//!   is encoded into the model fire's own command buffer
//!   ([`Shell::fire_attached`], `Session::stage_into`). The verdict is read
//!   one frame later, from the harvest, which is the only place a proof
//!   exists that the kernels ran. What is still refused, by name and before
//!   anything is staged, is `Boundary::Prologue` (a prologue's writes are
//!   INPUTS to the forward and this shell stages every fire input on the
//!   host, before it opens a command buffer, so there is no point in the step
//!   to encode one at), a program that reads `mtp_logits` against a load
//!   whose model text bakes no `mtp` seam (there is a second rectangle now —
//!   what there may not be is a draft column in THIS artifact),
//!   an instance attached twice, an attachment naming a lane this fire does
//!   not have, an instance whose rings are not ready, and a readout row list
//!   that is not one ascending run.
//! * **A GUEST WITH MORE THAN TWELVE CHANNELS is NOT in this list any more.**
//!   It never crossed the contract as a refusal — it crossed as a COMPILE
//!   failure, because the M2 fused kernel binds each channel's two cells as
//!   argument indices (`7 + 2k`, `8 + 2k`) against Metal's last index of 30,
//!   the emitter declined a wider region by name, and this shell had no other
//!   form to run it in. `engine_metal::program::compile::Form::Grouped` is
//!   that form: the channels move into a lane table of device addresses
//!   ([`crate::device::Buffer::address_at`]), so the ceiling that applies is
//!   the lane table's twenty-nine and no guest in the corpus is near it. The
//!   same form is what runs the emitted grouped nucleus and top-k samplers
//!   and what splits a vocabulary-wide gather across a threadgroup instead of
//!   walking it on one thread.
//! * `LoadRequest::ordinal` — see [`DeviceBoot`]. `MTLCreateSystemDefaultDevice`
//!   takes no ordinal, so a request that names one is refused rather than
//!   quietly given the default device.
//! * `copy_state`, `resize_pool` and `encode` take the trait's default
//!   bodies, which answer [`Error::Unsupported`]. The pools are not virtual,
//!   there is no recurrent-state mover and no multimodal encoder.
//! * `copy_kv` IS served, and in one direction: `MetalShared` to
//!   `MetalShared`, cells moved between pages of this load's own pools. A
//!   host-pinned end names a swap pool this load does not reserve and a
//!   private end names a storage mode it reserves nothing in; both are refused
//!   by name (`kv_copy_direction`), and `Capabilities::kv_copy` says so ahead
//!   of time.
//! * `register_channel` and `close_channel` **for a host-visible channel**,
//!   and for the reason they always were: the runtime owns that ring's host
//!   half and the cells cross through `publish_channel`/`take_channel`, so
//!   binding IS registration and there is nothing standalone to allocate.
//!   **A `HostRole::None` CHANNEL IS SERVED NOW**, because its ring is the
//!   one thing a per-instance allocation cannot stand in for — design §5's
//!   device-only ring shared by ≤8 attachments, which two passes meet on.
//!   See the verb.

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
use engine::transfer::{KvCopy, MemoryDomain};
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

/// How a caller answers "what does this checkpoint's bytes mean for this
/// plan".
///
/// See the module header: the contract has no seat for a `ModelContract` and
/// this crate must not know a model family, so the party that links the
/// catalog supplies the lookup. Identical to the CUDA sibling's type on
/// purpose — one door, one signature, whichever shell is behind it.
pub type ContractFor = fn(&Trace, &Path) -> std::result::Result<ModelContract, String>;

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
/// [`Metal::new`] for one reason and it is a seam reason: the runtime's
/// backend door opens every engine the same shape
/// (`Metal::new(DeviceBoot::default(), contract_for)`), and the knob this
/// plane will grow first — the *indirect command buffer* `serve.rs` names as
/// a future note — is a boot-time choice that lands here. An empty seat that
/// is documented as empty costs one `::default()` at the door; a signature
/// that changes when the first knob arrives costs the door.
///
/// **THE FIRST KNOB HAS ARRIVED, AND IT IS `gpu_mem_utilization`.** The wired
/// ceiling this shell now admits against (`store::accounting`) needs the same
/// fraction the CUDA plane's `[engine] gpu_mem_utilization` states, and it is a
/// device-wide boot choice rather than a per-model one — so it lands here,
/// read by [`crate::boot::open`] out of `[metal] gpu_mem_utilization` and
/// answered at 0.90 when the document is silent, exactly as CUDA's is.
///
/// **AND THE SECOND IS `adapter_dir`** (alto adapter §3.3), read out of
/// `[model] adapter_dir` and handed to [`crate::serve::Shell::mount_adapters`]
/// after the load. It is a `Clone` rather than a `Copy` field, which is what
/// took the `Copy` off this struct: a path is bytes, and a mount is stated
/// once per boot.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    /// The fraction of `recommendedMaxWorkingSetSize` this device may hold
    /// resident — weights, kv pool and scratch. See
    /// [`store::accounting`](crate::store::accounting).
    pub gpu_mem_utilization: f64,
    /// **WHERE THIS DEPLOYMENT'S SHARED ADAPTERS LIVE**, or `None` for a
    /// deployment that mounted none — the feature off, and every shared bind
    /// a refusal that says so ([`crate::blob`]).
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

/// **The readback plan one submitted step left behind**, plus the per-lane
/// policy the SUBMISSION stated.
///
/// The shell answers one row per lane and the contract asks for a `Readout`
/// per lane, and the two meet here rather than on the fire path: which rows a
/// caller wants back is a contract question, and at the instant the step is
/// enqueued nobody is asking it yet.
struct PendingStep {
    /// What each lane asked for, in submission order.
    readout: Vec<Readout>,
    /// The receipt the shell minted for this step.
    landed: Landed,
    /// The rows, once somebody has come for them. Cached rather than re-taken,
    /// because `Engine::settle_frame` is documented idempotent and the shell's
    /// settled ring hands a step's answer over exactly once.
    rows: Option<Vec<Vec<f32>>>,
}

/// The Metal shell, behind [`Engine`].
pub struct Metal {
    /// The device-wide boot knobs — today, the wired-ceiling utilization
    /// fraction [`Metal::load`] admits against (see [`DeviceBoot`]).
    boot: DeviceBoot,
    contract_for: ContractFor,
    shell: Option<Shell>,
    caps: Option<Capabilities>,
    next_fire: FireId,
    next_frame: FrameId,
    /// **Where step completions go**, installed once by the thread that owns
    /// this engine ([`Engine::on_complete`]). `None` is a caller that does not
    /// want to hear — a smoke test, a bench — and costs the completion handler
    /// one branch.
    sink: Option<engine::CompletionSink>,
    /// **The last submitted frame's per-step readback plans**, held for a
    /// caller that comes back for numbers ([`Engine::settle_frame`]).
    ///
    /// One frame's worth and no more, and that is a statement about the
    /// readout seats rather than a cache policy: there are as many of them as
    /// there are arms, and the frame after next reuses them.
    pending: Option<(FrameId, Vec<PendingStep>)>,
    /// **WHICH ADAPTER SLOT EACH BOUND INSTANCE ROUTES TO** (alto adapter
    /// §6.4: the plan says WHETHER, the bind says WHICH).
    ///
    /// An instance whose program declares the `lora` sink has its weights
    /// landed at bind and its slot recorded here; every lane attached to it
    /// carries that slot into the fire. Instances with no adapter are not in
    /// this map at all, and a fire whose lanes are all of that kind walks a
    /// map that is empty in every deployment that never bound one — which is
    /// what keeps the axis costing an adapterless fire nothing.
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

    /// The loaded shell, mutably.
    ///
    /// **THE DOOR TO A GUEST PASS FIRED ON ITS OWN.** The contract fires a
    /// program by ATTACHING it to a model fire, and `submit` serves that
    /// directly now (module header); [`Shell::fire_program`] is the other
    /// spelling — a pass beside no fire, which is what a program with no
    /// `logits` intrinsic wants — and it is reached through here.
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
    /// [`KvDelta`](engine::KvDelta) says a lane whose `pages` are
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
        // **THE FENCE RIDES ON THIS DOOR** (`Shell::program_instance`): a
        // channel read or write while that instance's epilogue is still
        // airborne would see the cursors of the fire before last.
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

/// **The verb name a refused `copy_kv` direction is refused under.**
///
/// A `&'static str` because [`Error::Unsupported`] carries one, so the pairs
/// this shell can be asked for are enumerated rather than formatted. That is
/// not a limitation being worked around: a refusal a caller can MATCH ON is
/// worth more than one it can only print, and there are seven domains and one
/// served pair. Written to the CUDA sibling's shape (`kv_copy_direction`
/// there) so that a caller reading two workers' refusals reads one taxonomy.
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

/// The shell's refusal, in the contract's vocabulary.
///
/// **THE TAXONOMY IS THE POINT.** `Exhausted` and `Impossible` are scheduling
/// answers the runtime's lane loop acts on — retry behind something that frees
/// pages, or drop the request — and everything else is a failure it logs. A
/// [`Fault::Ceiling`] is `Impossible` and not `Exhausted` because every
/// ceiling this shell states was reserved at LOAD: no amount of freeing makes
/// a pool carved for 256 slots seat a 257th, which is exactly the distinction
/// `engine::error`'s header draws.
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
fn fault(fault: Fault) -> Error {
    match fault {
        // The machine's, not the submission's. `Deviceless` is a build with
        // no Metal in it — a non-Apple target, or an Apple one with no GPU
        // published — and it reaches a caller through exactly the same door
        // as a Metal call that refused, because both mean "the device half
        // could not answer".
        Fault::Deviceless | Fault::Device { .. } => Error::Device(fault.to_string()),
        // **THE SUBMISSION'S**, and not the machine's: a lane whose media
        // geometry and media payload disagree is a well-formed request with
        // wrong numbers in it, which is exactly what `Invalid` means. Its
        // CUDA twin answers the same variant from the same shape of refusal.
        Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
        // The load axis: a plan these budgets do not admit, a checkpoint the
        // contract does not fit, a param that never published, a seat the
        // plan names and this shell binds none of. `Shader` joins them, and
        // it is `Unbound`'s sibling rather than a device condition: an
        // entrypoint this crate does not SHIP is a fact about the build and
        // the model text, discovered at the node that needs it, and no retry
        // of any submission changes it. Mapped through `to_string` so the
        // Metal compiler's own paragraph — the reason `error.rs`'s header
        // keeps a `String` where CUDA keeps an `i32` — survives the crossing.
        //
        // `Backing` joins them and does NOT join an `Exhausted` that this
        // shell deliberately does not have (the paragraph above). A streamed
        // load whose temporary file will not open, size or map is a LOAD
        // answer for the reason the rest of this arm is one: it is discovered
        // once, at the load, and the fix — room in the temporary directory,
        // descriptors, address space — is the deployment's rather than a
        // resubmission's. Telling the lane loop to retry behind something that
        // frees DEVICE pages would be the wrong instruction, and it is the
        // only other word available.
        // `Mapped` joins them on the same argument one step further out: an
        // artifact that will not open, stat or map is discovered once, at the
        // load, and the fix is the deployment's — the file's permissions, its
        // path, this process's address space. (Its read-only REFUSAL arm —
        // `write` on a mapped reservation — cannot reach here at all: it is a
        // shell bug, caught by `crate::mapping`'s own gate, not a load a
        // caller can submit.)
        // `Recipe` joins them and belongs nowhere else: a serving artifact
        // stamped for another deployment is discovered once, at the load,
        // before a byte is reserved, and the fix is the operator's — re-import
        // the checkpoint for THIS deployment, which is the command the
        // forwarded sentence spells. No resubmission and nothing the machine
        // frees changes the answer.
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
        // A region whose classes this fire's order does not make consecutive
        // is a BAKE-integrity break, not a submission the caller can fix; so
        // is a schedule built over more classes than its reader runs.
        Fault::Fragmented { .. } => Error::Device(fault.to_string()),
        // The two derivation refusals (`crate::abi`) are LOAD answers, and
        // deliberately: the binding recipe is derived once, from synthetic
        // descriptors, before a `FireId` is ever spent — so a quantity that
        // is not affine in the descriptor, or two compositions that do not
        // walk the same template, is a fact about this artifact on this
        // plane, discovered at load and unchanged by any retry.
        Fault::Unaffine { .. } | Fault::Unstructured { .. } => {
            Error::Load(fault.to_string())
        }
        Fault::Straddled { .. } => Error::Load(fault.to_string()),
        // **EVERY MASK REFUSAL IS `Invalid` NOW, AND THE ONE THAT WAS NOT IS
        // THE MEASURE OF THE WAVE.** A mask that does not describe its lane,
        // a per-row mask of the wrong height, one against a plan with no
        // masked arm, one whose word says the other thing, and a stated
        // position run that is not the lane's height are all the SUBMISSION's
        // — the caller stated it and the caller can state it differently. A
        // retry with a mask of the lane's own extent, or with as many rows as
        // the lane feeds, is a real answer, which is what `Invalid` means and
        // `Impossible` does not.
        //
        // `Fault::MaskRows` answered `Unsupported` here while the per-row form
        // had no expansion on this plane — "the caller's answer is another
        // engine, not another mask". It has one ([`crate::mask`]), so the
        // sentence is a submission's again and the verb it named is served.
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. }
        | Fault::Positions { .. } => Error::Invalid(fault.to_string()),
        // The adapter axis's two fire refusals, sorted with the mask's for
        // the mask's reason: a lane routed against an artifact with no
        // correction, or against a word that puts it outside the
        // correction's window, is the SUBMISSION's, and a retry that states
        // the id and the word as one reading of one lane is a real answer.
        Fault::Adapterless { .. } | Fault::AdapterWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // One slot wanted, none reclaimable — the numbers are about what is
        // FREE and not what exists, because a table whose every seat is
        // pinned has no free one whatever its width. Sorted as the CUDA twin
        // sorts it, because a control plane retrying on `Exhausted` is
        // retrying the same fact on either plane.
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        // The observability axis's two, sorted with the adapter's for the
        // adapter's reason: a lane asking to be observed against an artifact
        // that declares no capture column, or against a word that puts it
        // outside the capture window, is the SUBMISSION's, and a retry that
        // states the ask and the word as one reading of one lane is a real
        // answer (`.wiki/alto/attn-score.md` §4).
        Fault::Scoreless { .. } | Fault::ScoreWord { .. } => {
            Error::Invalid(fault.to_string())
        }
        // The REGISTRATION's is `Load`, and the twin sorts it there for the
        // reason its own sentence gives: a bank's capacity and a slot's width
        // are shapes the model text declared, so nothing the caller frees or
        // restates makes room and the fix is the model text. It was `Invalid`
        // here while the verb was refused at the door and the variant was
        // unreachable; the door is open, so the two planes answer one word.
        Fault::Adapter { .. } => Error::Load(fault.to_string()),
        // The shared-adapter mount's, and it sorts with the registration
        // above rather than with slot exhaustion below. A blob refusal is the
        // DEPLOYMENT's — a name that is not in the mount, a manifest that
        // disagrees with the model text's banks — and nothing a caller frees
        // or restates changes it. Sorted as the CUDA twin sorts it, because a
        // control plane reading the word is reading one axis on either plane.
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        // The guest-program plane's two, both `Program`, which is the word
        // the contract reserves for it. `Compile` carries `eta_exec::Failure`'s
        // deterministic/retryable split and `Program` names the entry that
        // refused; neither is a model-fire condition and neither should reach
        // the lane loop as one.
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        Fault::Fire(_) => Error::Invalid(fault.to_string()),
        // **A RESIDENCY THIS SHELL CANNOT ARRANGE IS `Impossible`, NEVER
        // `Exhausted`.** It is the same ruling `engine::load::Residency::admit`
        // makes for the budget it owns, for the same reason: the refusal is
        // about a tier this build does not have — dense planes that do not
        // stream, a segment whose distinct experts outnumber its seats, a bake
        // whose regions carry two mixtures — and nothing the deployment frees
        // changes the answer. The sentence already names both numbers.
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling.
///
/// Two names for one seven-role vector: [`SymbolicExtent`](eta_exec::Role) is
/// the tag space both are written in, and the conversion is field for field so
/// that adding a role to one without the other is a compile error rather than
/// a silently unresolved axis.
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

/// The ceilings the compiler bakes against, out of the ones the load states.
///
/// The contract carries seven numbers and `model_compiler::Budget` takes
/// four; the other three (`page_size`, `max_context`, `slots`) are the POOLS'
/// and go to [`Boot`] directly. Converted in one place, which is the whole
/// reason `engine` states its own `Budget` rather than depending on the
/// compiler (`load.rs`'s note).
///
/// `max_adapters` crosses unchanged: it is a BAKE input, and the compiler is
/// entitled to refuse a plan that cannot carve what the deployment asked for.
/// What the load then ADVERTISES is a different number — the smallest
/// capacity this plan's banks actually declare, which is what a lane's id is
/// checked against (`PoolFacts::adapter_banks`, see [`Engine::load`]).
fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// **THE SECOND ROW AXIS'S CEILINGS, DERIVED** (multimodal §5.5), or `None`
/// for a plan that states no patch row.
///
/// **THE PLAN IS WHAT ASKS.** Read off the types a model text already wrote
/// (`Dim::axis`) and never off a flag: a value on the patch axis is the whole
/// of what makes a plan a two-unit one, and this follows the same reading
/// `model_compiler::unit` does. A text-only plan gets `None`, reserves not one
/// byte of the axis, and bakes the artifact it baked before the axis existed.
///
/// What is derived, and what a deployment states instead:
///
/// * **`max_patches`** is the deployment's if it named one, and otherwise the
///   token ceiling capped at two whole images on the catalog towers' native
///   48 × 48 grid. It is never below [`PATCH_LATTICE_FLOOR`], which is the
///   smallest whole image a patch-16 / merge-2 resize policy admits — a rung
///   below it would round up to a fire that cannot exist.
/// * **the rungs double** from that floor to the ceiling, which is the token
///   lattice's shape and not its numbers.
/// * **`max_images` is the ceiling at the floor** — as many images as the
///   patch ceiling holds if every one of them is the smallest whole image.
///
/// `pub` so a gate can boot a tower against the ladder this engine would
/// derive for it rather than against one the gate invented, which is the
/// difference between proving the derivation serves and proving some ladder
/// does. It is the CUDA twin's function, transcribed, because the derivation
/// is a statute about images and not about a plane.
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

/// **THE ONE ARITHMETIC THE MEDIA DOOR LEFT** (media-door §6): a payload row's
/// `f32` numbers, in the element the plan computes in, little-endian.
///
/// **ROUND TO NEAREST EVEN, AND STATED RATHER THAN TRUNCATED.** A truncating
/// conversion would land a slightly different image than the one the
/// front-end computed, and every parity claim about the tower below it would
/// then be about the wrong numbers.
///
/// # Errors
///
/// The `&'static str` an [`Error::Unsupported`] carries, for a plan whose
/// activation element this marshal cannot write. `Fp8` and the quantized codes
/// are WEIGHT elements and no activation is stated in one, so the arm that
/// would encode them is a refusal rather than a guess.
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

/// The guest-visible profile of a loaded plan.
///
/// **CARRIED, NOT RECONSTRUCTED** (design §7 on `caps`): the runtime used to
/// rebuild a `ModelProfile` at bind time out of eight `has_*` booleans on a
/// flat capability struct. Everything below is read off the plan and the
/// budgets — `num_layers` from the nodes' own `layer` stamps, `vocab` from
/// the width of the `out` seam — so there is one copy and nothing to keep in
/// step.
///
/// **THE MODEL-GATED INTRINSICS USED TO BE `false` UNDER ONE ANSWER, AND ONE
/// OF THEM IS NOT ANY MORE.** The answer was the ABI: the metal fire path
/// produced exactly ONE rectangle a guest could be pointed at — the `out`
/// seam's logits, bound as `IntrinsicId::Logits` — because the M2 emitter
/// made one buffer at index 6 the first argument of EVERY `INTRINSIC_VAL`
/// op. A second column had nowhere to go whatever the model text said.
///
/// **`has_mtp_logits` FOLLOWS THE BAKE NOW, WHICH IS WHAT IT ALWAYS MEANT.**
/// `eta_compiler::codegen::metal::intrinsics` gives each intrinsic an
/// argument index of its own, `program::launch` carries a slot table, and
/// `Shell::enqueue` points `IntrinsicId::MtpLogits` at the `mtp` export's own
/// rectangle at the attached lane's rows. So this is exactly the question the
/// CUDA sibling has always asked — does this load's model text declare a
/// draft head ([`Shell::drafts`]) — which is what a bind-time contract has to
/// mean. A load that bakes no `mtp` seam still answers `false`, and
/// `serve::prepare` refuses an attachment that would read one by name.
///
/// **`has_attn_score` FOLLOWS THE BAKE TOO NOW, AND FOR ITS OWN THREE
/// REASONS.** It needed two things the slot table did not buy — an
/// observability slab for the graph to write, and an element type, because a
/// score plane is F32 and the emitted `0xA0` handler read `bfloat` with no
/// second arm. `crate::scores` is the first: a slab the shell owns, carved off
/// the `attn.scores` seam the model text already wrote, that the capture arm
/// (`kernels_metal::attn::score`) fills as the graph runs. The second is an
/// arm on the intrinsic id in `ptir_m1_runtime.metal`, which is where the
/// element type has to live on a plane whose bindings are objects rather than
/// addresses. So this asks what the CUDA sibling asks
/// ([`Shell::observes_scores`]): does a fire of THIS load write scores.
///
/// The rest stay `false` and each keeps its own reason. `has_mtp_drafts` is
/// `[k]` I32 TOKEN IDS, an argmax the guest can take for itself off
/// `mtp_logits` and which no device path in this shell produces — the same
/// sentence the CUDA sibling gives. And `has_value_head` has no export seam
/// at all.
///
/// **AND `has_attn_page_mask` IS NOT ABOUT `Lane::mask`.** It gates the eta
/// vocabulary's `attn_page_mask` SINK — a guest program writing a
/// page-granular eviction mask from inside an attention stage — which is a
/// different door from the lane's own run-length mask this shell now stages
/// end to end. The CUDA plane answers `false` here too, for the same reason:
/// there is no sink to honour. A lane-level mask arriving through
/// `Lane::mask` is served whatever this bit says.
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
        // means (`ModelProfile::activation`'s own doc): this device's own
        // activation type is bf16 — the one dtype `kernels-metal` stamps —
        // and it is not what a guest program reads.
        activation: Dtype::F32,
        // **THE DRAFT COLUMN HAS SOMEWHERE TO STAND** — see the item doc.
        // This was `false` under "one rectangle and one intrinsic buffer",
        // and that sentence was true until the M2 slot table landed. It is
        // the bake's question now, and nothing else's.
        has_mtp_logits: shell.drafts(),
        // `[k]` I32 token ids, which is an argmax the guest takes for itself
        // off `mtp_logits`; no device path in this shell produces them.
        has_mtp_drafts: false,
        has_value_head: false,
        // **THE PER-KEY RECTANGLE HAS SOMEWHERE TO BE WRITTEN AND SOMEWHERE
        // TO BE READ** — see the item doc. Two conditions in one question,
        // which is what `Shell::observes_scores` answers: this load's text
        // declares a capture column AND the slab that observes it was carved.
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        // **HONEST, AND NOW OPEN** (alto adapter §6.4, wave A2 on this
        // plane). This was `false` under a note saying the correction class
        // was served but no GUEST SINK was honoured, and that note was true:
        // `has_lora` is the ETA guest-sink gate — "this backend can consume a
        // `lora` sink's A/B/SITES configuration and apply the delta at the
        // sites a guest program declared" (`eta_ir::validate`'s own check) —
        // and nothing in this crate read one. A program carrying the sink was
        // refused at bind, which was the right answer for a shell that would
        // otherwise have fired the BASE MODEL under a program that asked for
        // a correction.
        //
        // It is `true` because the sink is consumed now. `Metal::bind_instance`
        // reads it off the launch package (`crate::adapter::sink_of`), takes
        // the weights off the cells the guest seeded, converts them into the
        // banks' own bf16 and lands them in a slot; `Metal::fire_step` stamps
        // that slot onto every lane attached to the instance, with the fact
        // word moved into the correction's window beside it. A load whose
        // model text declares no bank still refuses by name
        // (`Fault::Adapterless`), which is a sentence and not a silent zero.
        has_lora: true,
        kernels: Vec::new(),
    })
}

/// **ONE INSTANCE'S ADAPTER, LANDED OFF ITS SEEDS** (alto adapter §6.1, §6.4).
///
/// `Ok(None)` is the ordinary answer: this program declares no `lora` sink, so
/// nothing about the adapter axis touches it. `Ok(Some(binding))` says the
/// weights crossed — once, here, on the host — and names the slot every lane
/// attached to this instance routes to.
///
/// # Why the SEED and not the ring
///
/// The sink's operands are `chan_read`s, and reading them at fire time is
/// exactly what §6.1 ruled out: a cell materialised into per-lane scratch on
/// every launch is a weight transported per fire. A seeded channel's cell is
/// already on this side of the wire at bind ([`InstanceBinding::seeds`]), so
/// the resolver takes it from there, converts it into the banks' own dtype,
/// and never looks at the ring again. A guest that publishes new adapter
/// weights mid-pass is therefore NOT serving a new adapter — the honest
/// reading of "swapping an adapter is re-seeding", which is a re-BIND.
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
    // **WHICH SITE THE GUEST ASKED FOR**, read off the sink's placement
    // constant once and checked against the banks by `planes_of`: a text that
    // names its site refuses a mismatch, a text that names none means what it
    // always meant.
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        // **A CHANNEL THE SINK NAMES AND THE BIND DID NOT SEED IS A
        // REFUSAL**, not a plane of zeros. A zero `A` is the IDENTITY adapter
        // — it is the construction every bank gate starts from — so accepting
        // an unseeded channel would answer the base model under a program
        // that asked for a correction, and answer it silently.
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

    /// **THERE IS NO THREAD TO BIND, AND THAT IS A PLATFORM FACT.** The CUDA
    /// twin rebinds here because `cudaSetDevice` is per-thread state and a
    /// context bound on the worker's boot thread strands every call the lane
    /// thread makes. An `MTLDevice` and an `MTLCommandQueue` are objects,
    /// documented thread-safe, and moving a loaded shell onto a lane thread
    /// costs nothing ([`Context::bind_thread`](crate::device::Context::bind_thread)).
    /// The verb is answered rather than refused because the runtime's call
    /// order is one shape across backends, and "nothing to do" is a real
    /// answer to "you are the lane thread now".
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
            // **SPENT NOW, AND IT IS THE WHOLE OF THIS SHELL'S RUN-AHEAD.**
            // It used to be stated and dropped, with a comment saying so:
            // this shell settled inside `submit`, so it had exactly one step
            // in flight whatever the deployment asked for. It carves the A/B
            // seats now — one resident-input plane and one readout seat per
            // in-flight step — and article 1's floor of two is what
            // `Runahead::default` answers when a deployment states nothing.
            frames_in_flight,
            // **§M-4c'S TWO FACTS, READ NOW.** They are the pair a shell
            // cannot look up — a shell must not know a model family, which is
            // why this crate's `models` edge is a DEV one — and they cross to
            // `Boot` unchanged, to be checked against the artifact's own stamp
            // as the first thing `Weights::resident` does with the path. The
            // other three the stamp compares are already in `trace` and are
            // not restated here, for `request_of`'s reason.
            tp_size,
            precision,
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
            return Err(Error::unsupported("metal", "device ordinal selection"));
        }

        // `Checkpoint::None` — bind and bake, land nothing — is a shape the
        // contract states and this shell has no path for: `Weights::resident`
        // is what reserves the store, and a `WeightTable` of nulls would
        // fault at the first dispatch rather than at the load that asked for
        // it. Refused by name.
        let Checkpoint::Path(path) = checkpoint else {
            return Err(Error::Load(
                "the metal shell lands a checkpoint or nothing runs; \
                 `Checkpoint::None` has no weightless path here"
                    .into(),
            ));
        };
        let path = PathBuf::from(path);
        let contract = (self.contract_for)(&trace, &path).map_err(Error::Load)?;

        // ── RESIDENCY (alto design §7), PLANNED AND ADMITTED BEFORE A BYTE
        //    LANDS. The plan is decided off the trace and the load plan's
        //    pairings — a quantized bank's factors and zero points are part of
        //    an expert's seat, so the decision cannot be made off the trace
        //    alone (`crate::experts`'s header says why moving the codes alone
        //    is wrong rather than merely partial). It costs one metadata parse
        //    and one plan compile, and reads no tensor bytes.
        //
        //    **AND THE HOST DEMAND IS HONESTLY ZERO.** The CUDA twin admits a
        //    pinned tier here because a device address and a host address are
        //    different addresses there. On unified memory they are not: an
        //    expert's source bytes are host bytes the process holds either way
        //    (a `Vec<u8>` today, the mapped artifact in the next wave), and no
        //    byte of them is a second copy the device reads through. So the
        //    second argument is zero, and `Plan::host_demand` is where the
        //    sentence lives.
        let planes = crate::weights::attachments(&trace, &contract, &path).map_err(fault)?;
        let mut residency_plan =
            experts::Plan::of(&trace, &planes, residency.device_weight_budget).map_err(fault)?;

        // ── THE WIRED CEILING (alto streaming §3 item 5, wave W-6), CHECKED
        //    BEFORE A BYTE LANDS. On Apple Silicon a GPU-touched Shared page is
        //    WIRED and the pager never evicts it (`.wiki/alto/streaming.md`), so
        //    `device_weight_budget` is the ONLY lever that bounds the weight
        //    tier — and a load whose resident weights plus kv pool exceed what
        //    the device holds does not page, it resets the box. This is the
        //    Metal twin of `engine_cuda::store::admit_the_card`: bind the device
        //    now, read `recommendedMaxWorkingSetSize`, and turn the operator's
        //    `gpu_mem_utilization` fraction and this model's kv pool into the
        //    effective weight budget the slab is shrunk to.
        //
        //    The bind is a throwaway — it reserves not one byte of model memory,
        //    and `Shell::load` binds its own device below. What it buys is the
        //    one number no pure plan can know, read at the one instant admission
        //    can still change the plan rather than the box.
        {
            let working_set = crate::device::Context::bind().map_err(fault)?.working_set();
            let util = self.boot.gpu_mem_utilization;
            let paging =
                crate::store::kv::Paging::of(budgets.page_size, budgets.max_context, budgets.slots)
                    .map_err(|error| fault(Fault::from(error)))?;
            let kv_pool = crate::store::pool_demand(&trace, paging).map_err(fault)?;

            let acct = crate::store::accounting::Accounting::of(
                working_set,
                util,
                residency_plan.device_demand(),
                kv_pool,
            );
            if acct.admit(residency.device_weight_budget, util).is_err() {
                // Over the wired ceiling. Shrink the weight tier to the headroom
                // the ceiling leaves after the kv pool and the floor, if there
                // is a streamable tier to shrink; a `Plan::of` refusal here
                // means either nothing streams or even a one-seat slab is too
                // big, and the ceiling's own six-number sentence — which names
                // the lever — is the better answer than the budget sentence.
                let headroom = acct.weight_headroom();
                match experts::Plan::of(&trace, &planes, Some(headroom)) {
                    Ok(shrunk) => {
                        // The shrink met the ceiling by construction (its demand
                        // is at most `headroom`); re-admit only to catch a dense
                        // floor that the headroom itself cannot hold, which is
                        // the six-number refusal.
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
                        // Nothing to stream, or the minimal slab is still over:
                        // the ceiling refuses, by name and with all six numbers.
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

        // Derived before the trace moves into `Boot`, and off the trace's own
        // types — see `patch_ladder`.
        let patches = patch_ladder(&trace, &budgets);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            // §M-4c, carried across unchanged. `Weights::resident` is where
            // they are spent, and it spends them BEFORE it reserves the store:
            // an artifact written for another shell or another degree is
            // refused by the field that disagrees, not discovered by the
            // tokens it produces.
            tp_size,
            precision,
            budget: bake_budgets(&budgets),
            // **THE SECOND ROW AXIS'S CEILINGS**, derived off the TRACE's own
            // types rather than off a flag: a value on the patch axis is the
            // whole of what makes a plan a two-unit one, and a plan that
            // states none gets `None` and an artifact bit-identical to the
            // one this door baked before the axis existed.
            patches,
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
            // ARTICLE 9: the knob is typed here, not read from anywhere. The
            // clamp is `Runahead::of`'s — one place knows the bound a seat
            // ring can carry, and a caller that states zero means one.
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            residency: residency_plan,
        })
        .map_err(fault)?;

        // **THE SHARED-ADAPTER MOUNT, STATED AFTER THE LOAD** (alto adapter
        // §3.3). It is not a `Boot` field because it is not a property of the
        // BAKE: the banks are, and they came off the model text above. Where
        // the shared adapters live is the deployment's, it outlives every
        // load, and §3.3's hot-add is a file drop into it — so it arrives as
        // a verb, typed off the boot document, and never out of the
        // environment (article 9). The CUDA door states it in the same place
        // for the same sentence.
        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        // Read here rather than at the field, because the shell moves into
        // `self` before the facts are assembled. See `weights_from_cache`.
        let weights_warm = shell.weights_warm();

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
                // `eta_compiler::codegen::Backend::Metal` emits and what
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
                // **WHAT THE LOAD ACTUALLY SEATS, READ OFF THE PLAN** — the
                // twin's answer, and now the honest one here too. The
                // smallest capacity any one bank of this model declares is
                // the id ceiling: an adapter occupies one slot of EVERY bank
                // it fills, so a load whose `A` seats eight and whose `B`
                // seats four holds four. Zero for a model whose text declares
                // no correction, and then `Lane::adapter` has nowhere to go
                // and is refused by name at the fire (`Fault::Adapterless`).
                //
                // This field is what a caller PLANS against, so it answers
                // the routes that will be SERVED. It answered zero while
                // nothing read a bank; `kernels-metal`'s correction entry and
                // this shell's routes staging are what changed the truth,
                // not this line.
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
                max_page_refs: paging.pages_per_slot.saturating_mul(budgets.max_lanes),
                max_context: paging.context(),
            },
            profile,
            // **THIS SHELL RESOLVES THE WHOLE FIRE GEOMETRY ON THE DEVICE,
            // AND NINE PORTS ARE THE WHOLE OF THAT SENTENCE.** `serve::stage`
            // reads every one of them off an attached instance's own device
            // rings at step 0b — before `compose`, behind the fence
            // `admit_attachments` takes — and steps 0c and 3b are where the
            // answers replace what a seat would have derived:
            //
            // ```text
            // embed_indptr  the member's lane CSR  which lanes, which rows
            // embed_tokens  the sampled id         the device DECIDED it
            // positions     what reaches RoPE      the guest may renumber
            // kv_len        the extent AFTER       `have` is derived back
            //                                     from it; the page count and
            //                                     the last page's fill follow
            // pages         the lane's page run    cut out of one flat run by
            // page_indptr   the run's bounds       the CSR beside it
            // w_slot        where a row lands      `have + row` cannot spell
            // w_off         and at what offset     B lanes into one pool
            // attn_mask     a dense bool rectangle run-length encoded and
            //                                     expanded by `crate::mask`
            //                                     into the same plane a
            //                                     host-stated mask expands to
            // ```
            //
            // The narrow claim — `embed_tokens`, `positions`, `kv_len` — is
            // the guest telling the model what to run rather than the model
            // handing the guest its answer, and what it buys is the one thing
            // a host cannot buy: the sampled token stops travelling out
            // through `take_channel`, through a guest await, and back in
            // through `publish_channel`, so a second decode step can be
            // submitted behind the first inside one frame.
            //
            // **THE FOUR BEYOND IT WERE WITHHELD WHILE THE PAGE IDS WERE THIS
            // SHELL'S ALONE.** `store::kv::geometry_with` derived all four
            // from the seat, and reading the guest's copy would have been
            // reading a second opinion about a table the guest did not hold.
            // They are claimed now because that same call already took a
            // caller-stated table per lane (`KvDelta::pages` non-empty), and a
            // guest that states its pages on a CHANNEL is that same caller
            // reaching the pool one phase later — through
            // `KvDelta::translation`, which is what keeps the guest's
            // working-set-relative indexes out of the pool's space. The mask
            // is claimed with them because a fire whose ancestry is device
            // data — a beam search's `gather(mask, parent)`, a sliding
            // window's rebuilt row — has nowhere else to state it, and the
            // runtime's bind-time classifier asks for exactly this union
            // before it will admit one (`inferlet::host::forward`'s
            // `devgeo_capable`).
            //
            // What `Capabilities::admits` still buys is the BIND-TIME refusal
            // rather than a fire-time surprise, for whatever a later load
            // cannot serve.
            ports: PortMask::DEVICE_GEOMETRY.with(Port::AttnMask),
            geometry: GeometryClass::DeviceGeometry,
            // **ONE DIRECTION, AND UNIFIED MEMORY DOES NOT ADD THE OTHERS.**
            // `copy_kv` moves cells between pages of THESE pools, in a command
            // buffer on the fire queue (`Pools::copy_kv`), which is what a
            // fork, a graft and a prefix-cache hit are. It is tempting to claim
            // the host directions too on a plane where `contents()` makes host
            // and device the same bytes — a "copy" to the host is not even a
            // distinct direction here. But what those two name is a PINNED SWAP
            // POOL at the other end, and this load reserves none: unified
            // memory makes the pools host-addressable, it does not supply a
            // second place to put a page. Each of the three is refused by name
            // in `Metal::copy_kv`, and this is where a caller reads that
            // without having to try one.
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            // The pools are `MTLBuffer`s reserved for this process. Metal has
            // an `MTLSharedEvent`/`IOSurface` story for cross-process
            // sharing; nothing in this shell writes one, so there is no
            // handle to export.
            kv_handle: None,
            media_encode: false,
            // **FALSE, AND THE CONTROL KERNELS ARE WHY** (alto design §5).
            // This shell has no pull-validate and no commit-bump: its rings
            // advance on the host, so a caller's channel prediction is
            // something it could only ignore, and `Lane::validate_for`
            // refuses a stated one by name. The caller's pump stays the path
            // — `publish_channel`/`take_channel` at the fire's boundary —
            // until the MSL half of the two kernels exists.
            device_channel_commit: false,
            // **FALSE FOR THE SAME KIND OF REASON** (alto design §6, wave
            // F3). This shell allocates no buffered-activation pool and its
            // `ssm` entries seat neither `commit_len` nor a fold predicate,
            // so `RsVerb::Buffer` and `RsVerb::FoldBuffered` are things it
            // could only serve as a destructive fold — which is the one
            // outcome a speculating caller must never be handed. Refused by
            // name at `Lane::validate_for` until the MSL half exists.
            rs_verbs: false,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                // **WHAT THIS LOAD ACTUALLY HOLDS** (alto design §7). `true`
                // is a full-residency load — every byte of the table in one
                // shared buffer that never moves — and `false` says the
                // routed bands went to the wired-slab tier and the fires are
                // cut into segments around it (`engine_metal::experts`).
                //
                // THERE IS STILL ONLY ONE TIER, AND THE HOST BUDGET STILL
                // ADMITS EVERYTHING. Unified memory has no second address
                // space for a pinned copy to live in, so a streamed load's
                // host demand is zero and `host_weight_budget` has nothing to
                // refuse — which is the platform, not a gap.
                weights_resident: !streams,
                // **TRUE WHEN THIS LOAD MAPPED ITS CHECKPOINT INSTEAD OF
                // READING IT** — the warm arm (§M-5), landed.
                //
                // What stood here was an honest `false` and a plan: this
                // plane had no warm boot, so every Metal load was a cold one
                // and said so. The plan's two remaining steps are both in —
                // M-2 is `checkpoint::file::serve::Artifact` (the CUDA
                // node's, as recorded: this crate added call sites and did
                // not create it), and M-5 is `weights::warm`, which maps the
                // serving artifact once and mints one handle per plane at
                // the offset its manifest states.
                //
                // So the field means what its name says, on both sides. A
                // serving artifact whose stamp agrees with this deployment
                // is served off its own pages — no store reservation the
                // size of the model, no host arena the size of the image,
                // and no `memcpy` into pages the GPU was about to wire
                // anyway — and reports `true`. Measured on this box against
                // the four-bit 0.8B: 27 ms warm against 155 ms cold, the
                // same nine greedy tokens, and a wired delta at the BOOT of
                // +0.001 GiB — the touch is what wires, and it happens at
                // the first fire (`a_warm_load_is_the_artifact_mapped`).
                //
                // Everything else reports `false`, and a file that CLAIMED
                // to be servable prints which fact sent it down the cold
                // road: a streamed plan (the mapped door is the
                // fits-in-memory one, and `crate::mapping` carries the
                // measurement that makes that a rule), a sharded payload, an
                // alignment this device would not want, a plane at a width
                // the plan does not predict, or a residue of computed planes
                // too large for the word "warm" to survive. A raw snapshot
                // and an ordinary `.zt` are quiet: they never claimed it.
                weights_from_cache: weights_warm,
                arena_bytes,
                pool_bytes,
                input_bytes,
                // **THIS PLANE'S POOLS ARE ONE RESERVATION**, so what is
                // committed IS the ceiling and there is no high water to
                // report separately. Stated as `pool_bytes` rather than left
                // at zero, because zero would read as "nothing is backed".
                pool_committed_bytes: pool_bytes,
                pool_high_water_bytes: pool_bytes,
            },
            caps,
        })
    }

    fn register_adapter(&mut self, registration: &AdapterRegistration) -> EngineResult<()> {
        // **THE DOOR IS OPEN, AND IT IS THREE LINES AND A `Shell` FORWARD.**
        // This body refused for one reason — nothing on this plane READ a
        // bank, so a registration that succeeded would have landed bytes no
        // dispatch reaches and told the caller the correction was applied.
        // `kernels-metal`'s `linear::lora::correct` is the reader, the
        // dispatch layer's `Linear::LoraCorrect` arm is where it is called,
        // and `serve.rs` stages the routes the arm indexes with. So the
        // dangerous answer is now the true one.
        //
        // No graph is touched: the composition is what keys a fire and a
        // bank's contents are not in it, so a registration between two fires
        // costs a memcpy through the shared mapping and leaves every recorded
        // walk valid (decision 17).
        //
        // BORROWED, NOT COPIED. The contract's `AdapterPlane` owns its bytes
        // and this crate's borrows them; the conversion is a reborrow per
        // plane, and no scaling happens at either side of it — `α/r` is
        // folded into the up bank's contents by whoever assembled the plane
        // (`model_ir::Linear::LoraCorrect`'s own note), exactly as on the
        // CUDA plane.
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
        // ── ARTICLE 4, MECHANICALLY: every step validated before any of them
        //    runs, one id for the frame, the steps in order. `Step::validate`
        //    is where the check lives — the contract wrote the arithmetic once
        //    and a second spelling of it here would be a second thing to keep
        //    in step.
        frame.validate()?;
        let id = self.next_frame;
        self.next_frame = self.next_frame.wrapping_add(1);
        // **THE PREVIOUS FRAME'S NUMBERS DIE HERE**, and the drop is the rule:
        // there are as many readout seats as there are arms, so a frame's rows
        // survive exactly until the frames behind it have taken the seats
        // back. A caller that wanted numbers had to ask before now
        // ([`Metal::settle_frame`], which says so by refusing).
        self.pending = None;

        let mut steps = Vec::with_capacity(frame.steps.len());
        let mut pending = Vec::with_capacity(frame.steps.len());
        for (index, step) in frame.steps.iter().enumerate() {
            // ── THE NEXT STEP, STATED (`Engine::expect_fire`, advisory). Two
            //    parties know a successor and the frame verb splits them: the
            //    runtime states the launch queued behind this FRAME, and a
            //    frame that crosses whole is one whose successors the engine
            //    already knows.
            if let Some(next) = frame.steps.get(index + 1) {
                self.expect_fire(next);
            }
            // ── ARTICLE 1, AND THE ERROR ARM IS ARTICLE 4's OTHER HALF. A
            //    step that faults POISONS THE FRAME'S REMAINING STEPS — the
            //    loop stops, so nothing after it is prepared or enqueued — and
            //    the steps already committed settle normally, because they are
            //    real work the device is really doing and pretending otherwise
            //    would leave their arms held forever.
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

        // ── **THE EAGER SHELL, KEPT REACHABLE AND KEPT HONEST.** At
        //    `frames_in_flight == 1` there is one arm, one step may be in
        //    flight, and this shell answers filled readouts before it returns
        //    — which is byte for byte what it did before the asynchronous wave
        //    and is the golden model a divergence at depth two is bisected
        //    against. `settles_asynchronously` answers `false` for it, so the
        //    runtime settles its own bookkeeping on the spot and never waits
        //    for a completion that already happened.
        //
        //    Above depth one this line is not taken and the verb returns with
        //    the device still running: article 1, and the exit from the
        //    registered exception this shell was the last holder of.
        if !self.settles_asynchronously() {
            self.settle_frame(&mut ticket)?;
        }
        Ok(ticket)
    }

    /// **Yes above depth one, and the depth is the deployment's**
    /// (design §2, article 1).
    ///
    /// At `frames_in_flight >= 2` `submit` returns with the device still
    /// running: every step's launches are on the command queue, every step's
    /// completion handler is armed, and not one host read stands between them.
    /// The receipts carry ids and empty readouts; the outcomes arrive on
    /// [`Engine::on_complete`]'s sink, called from Metal's own completion
    /// thread.
    ///
    /// At depth one it is `false` and it is the truth: that shell waits.
    fn settles_asynchronously(&self) -> bool {
        self.shell
            .as_ref()
            .is_some_and(|shell| shell.frames_in_flight() > 1)
    }

    fn on_complete(&mut self, sink: engine::CompletionSink) {
        self.sink = Some(sink);
    }

    /// **Fill in the last submitted frame's readouts** (design §4's readback
    /// obligation).
    ///
    /// Waits for the steps the host has not caught up with and reads the rows
    /// their own command buffers copied out — the same rows, off the same
    /// rectangle, in the same order F1's in-fire readback took, so what a
    /// caller gets here is byte-identical to depth-one execution.
    ///
    /// **AND IT REFUSES A FRAME WHOSE SEATS ARE GONE.** A step's answer lives
    /// in the readout seat its arm owns, and the frames behind it take those
    /// seats back; a caller that submits again before it asks has asked too
    /// late. That is a named refusal rather than a silent wrong answer,
    /// because the bytes in those seats afterwards are a real logits row
    /// belonging to somebody else's step and nothing about them looks wrong.
    ///
    /// Idempotent: the rows are cached on the pending record the first time
    /// they are taken, because the shell hands a step's answer over once.
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
        // The record goes back whatever happened, so a caller that hits a
        // device fault can still read the steps that did answer.
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

    /// **THE HINT IS SPENT ON THE SEATS, WHICH IS THE ONE THING THIS SHELL
    /// HAS TO WARM.**
    ///
    /// This body was empty, and the comment that stood here said why: nothing
    /// on this plane keys prepared state on a composition, because there is no
    /// capture and every fire encodes its own command buffer. That is still
    /// true and it is no longer the whole story. What a step needs before it
    /// can be staged is an ARM — a resident-input plane and a readout seat the
    /// device is not still reading — and an arm comes free when the step that
    /// held it is harvested. So the hint is taken as *another step is coming*,
    /// and the answer is to harvest every flight the device has ALREADY
    /// finished, without waiting for any of them
    /// ([`Shell::reap`](crate::Shell::reap), which asks `status` and never
    /// blocks).
    ///
    /// The win is the one run-ahead exists for: at steady state the next
    /// `prepare` finds a free seat and never enters the kernel, where without
    /// the hint it would have discovered the shortage at the moment it had
    /// work to do.
    ///
    /// **CORRECTNESS NEVER DEPENDS ON IT** (the contract's own promise). A
    /// hint for a fire that never comes costs one non-blocking status query
    /// per in-flight step; a fire that arrives unhinted harvests inside
    /// `prepare` exactly as it would have. Nothing here reads the submission
    /// at all, which is why a hint stated before the tokens are sampled is
    /// as good as any other.
    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
        if let Some(shell) = self.shell.as_mut() {
            // A refusal here is a device fault on a step that has ALREADY
            // finished, and it belongs to whoever comes for that step's
            // numbers — this verb answers `()` and has nowhere to put it.
            let _ = shell.reap();
        }
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> EngineResult<ProgramId> {
        self.loaded_mut()?
            .register_program(registration)
            .map_err(fault)
    }

    /// **The device-only channels are registered here now, and only they**
    /// (design §5, mirroring `engine-cuda`'s `27de300fa`).
    ///
    /// BINDING WAS REGISTRATION HERE, and for the two host-visible roles it
    /// still is: a `Writer`/`Reader` channel's ring is the RUNTIME's — it
    /// owns the host half (`runtime::engine::channel`) and the cells cross
    /// through `publish_channel`/`take_channel` — and this plane publishes no
    /// [`HostMirror`](engine::channel::HostMirror) for it to adopt, so there
    /// is nothing standalone to allocate and the refusal stands.
    ///
    /// A [`HostRole::None`] channel is the one that could not survive that
    /// reading. Its ring was cut inside whichever `Session` bound it, which
    /// is right for a ring ONE pass owns and silently wrong for the shape
    /// design §5 names by hand — a device-only ring shared by up to eight
    /// attachments, one putting and another taking. Two sessions cut two
    /// slabs and the handoff crossed nothing: the bench guest's decode pass
    /// read a `tok_in` that was empty forever, and every request died at its
    /// first decode frame. So the ring belongs to the CHANNEL, and this is
    /// where it is cut — once, per global id, before any instance binds.
    ///
    /// What comes back names no mirror and no wait slots. There is no guest
    /// end to point at a device-only ring, so the runtime mints its own
    /// slots (the contract's own reading of a zero) and keeps no host ring
    /// for it at all.
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
        // THE SHAPE IS THE REGISTRATION'S, AND `Rings::allocate` HOLDS EVERY
        // ATTACHMENT'S OWN DECLARATION AGAINST IT — a ring cut at one cell
        // width and addressed at another is a wrong cell and never a fault,
        // so the disagreement is refused there, at the bind that would have
        // made it.
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
            // ZERO, AND THE CONTRACT SAYS WHAT ZERO MEANS: this shell keeps
            // no waker table, so the caller mints its own slots.
            reader_wait_id: 0,
            writer_wait_id: 0,
            // No host end at either side is the whole of what `None` means.
            mirror: None,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> EngineResult<BoundInstance> {
        // REFUSED AT BIND, WHICH IS THE CONTRACT'S OWN READING: a class is a
        // claim about which descriptor ports the device resolves, and the caps
        // this load answered say which those are (see `load`, where this plane
        // now claims all nine). Asked through `Capabilities::admits` rather
        // than by re-deriving the subset test here, because the contract wrote
        // that negotiation down once and a second spelling of it is a second
        // thing to keep in step. A program bound in a class this load does not
        // serve would otherwise fail at its first fire, against a descriptor
        // nobody wrote — which is what the check is for even now that the
        // widest class passes it: the answer is the LOAD's, and a load whose
        // artifact cannot carry a class is refused here rather than there.
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
        // the plane's — the same seven roles, and the tags are `Role`'s in
        // both.
        // **THE DENSE SLOT MEETS THE CHANNEL ID, ONCE** (design §5).
        // `InstanceBinding` names this instance's channels in the package's
        // declaration order, which is the only place the caller's ids and the
        // program's dense slots are related — so a device-only channel this
        // shell registered is adopted HERE and nowhere else, and every
        // attachment of one id gets the same ring however differently the two
        // programs number their channels.
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
        // ── **THE ADAPTER LANDS HERE, AND NOWHERE ELSE** (alto adapter §6.1,
        //    §6.4). The plan says WHETHER this program carries a `lora` sink;
        //    the cells the guest seeded say what the weights ARE; the store
        //    says WHICH slot they go in. All three are host questions and all
        //    three are asked once, at bind — because §6.1's ruling is that a
        //    channel is a naming device and never a weight transport: a cell
        //    materialised into per-lane scratch EVERY FIRE is the cost this
        //    instant exists to refuse.
        //
        //    A refused landing closes the instance it was for. An instance
        //    bound with an adapter that did not arrive would fire the base
        //    model under a program that asked for a correction, which is the
        //    silently-wrong answer the whole axis is written against.
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
        // **THE BIND IS GIVEN BACK BEFORE THE INSTANCE IS** (alto adapter
        // §3.3), and it is given back as the BINDING rather than as the
        // instance's number: a shared bind's slot is held under the blob's
        // identity, so the id this instance was closed by does not name it. A
        // close that skipped this would pin a slot forever and the bank would
        // fill up with adapters nobody is using, which is a refusal that looks
        // like a leak.
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(bound) = held.as_ref() {
            shell.release_adapter(bound);
        }
        shell.close_program_instance(id).map_err(fault)
    }

    fn close_channel(&mut self, id: ChannelId) -> EngineResult<()> {
        // As `register_channel`: the door exists for the channels this plane
        // registered — the device-only rings — and for nothing else.
        //
        // **THE ENTRY GOES; THE RING OUTLIVES IT WHEN AN ATTACHMENT STILL
        // HOLDS ONE.** That is the property a per-session ring never had: a
        // pipeline may close its prefill pass while its decode pass is still
        // reading the ring the prefill filled, in either order.
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
        // THE DOOR THAT REPLACED THE POINTER. The runtime used to write a cell
        // into this ring itself, through the addresses `ChannelBinding`
        // published; it owns a host ring of its own now and hands the bytes
        // over here (`runtime::engine::channel`'s header, and the trait's).
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

    /// **Move kv cells between pages of this load's own pools.**
    ///
    /// A fork, a graft and a prefix-cache hit are all this verb: a page run two
    /// sequences share is copied onto fresh page ids, and the partial page at
    /// the boundary has its live tokens copied out so the child can append past
    /// them without writing into the parent's cells.
    ///
    /// # One direction, and the other three refused by name
    ///
    /// ```text
    /// MetalShared -> MetalShared    served: cells move inside the pools
    /// host-pinned at either end     a pinned swap pool this load does not
    ///                               reserve — and unified memory does not
    ///                               supply one, it only makes the pools
    ///                               themselves host-addressable
    /// MetalPrivate at either end    a storage mode this shell reserves
    ///                               nothing in (`Context::bind` asserts
    ///                               unified memory and every buffer is
    ///                               Shared)
    /// anything else                 a domain this load has no bytes in
    /// ```
    ///
    /// [`Capabilities::kv_copy`] states the same thing ahead of time, so a
    /// caller that reads capabilities never has to discover this by being
    /// refused.
    ///
    /// # The page ids are the CALLER'S, and this shell does not translate
    ///
    /// Article 8: page ids are the runtime's policy and the bytes under them
    /// are the engine's supply. A page id here indexes the same pools
    /// `KvDelta::pages` indexes, which is what makes "copy the pages, then fire
    /// against the copies" mean anything.
    ///
    /// # Ordering
    ///
    /// The moves go into a command buffer of their own on the fire queue and
    /// nothing is synchronized (article 2): they run behind every step already
    /// committed — which may still be reading the source pages — and in front
    /// of every step committed after this returns, which is the queue's own
    /// order and the same property two frames in flight already rest on. See
    /// [`Shell::copy_kv`](crate::serve::Shell::copy_kv).
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
        // THE DOMAIN PAIR, BEFORE ANYTHING IS BUILT. `Unsupported` and not
        // `Invalid`: the plan is a plan this contract describes, and what is
        // missing is storage on THIS engine — which is exactly the difference
        // the two variants carry.
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

    // `copy_state`, `resize_pool` and `encode` take the trait's default
    // bodies. See the module header: this shell genuinely has no recurrent-slot
    // mover, no virtual pools and no multimodal encoder, and a stub that
    // answered `Ok(())` would make a speculative rollback, a swap or an image
    // prompt appear to work.
}

impl Metal {
    /// One step of an admitted frame.
    ///
    /// `submit`'s per-step body, in its own inherent method for the same
    /// reason the CUDA twin's is: the contract's verb is the FRAME, and a
    /// step is this shell's private unit inside it.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
    ) -> EngineResult<(FireTicket, PendingStep)> {
        // ATTACHMENTS FIRST, AND BEFORE A FIRE ID IS SPENT. This is a
        // field-for-field lift and nothing more: every question an attachment
        // can be refused on is a question about the SHELL's state — is this
        // instance ready, is it attached twice, does its program read a
        // column this ABI has no second buffer for — and none of them is a
        // fact about the submission alone. `Shell::admit_attachments` asks
        // all six, at `prepare`, before anything is staged.
        //
        // **THE BLANKET REFUSAL THAT STOOD HERE IS GONE**, and what removed
        // it was a fire path that can encode a guest pass into the model
        // fire's own command buffer (`Session::stage_into`) and read its
        // verdict from the harvest (`Session::settle_launched`). While the
        // only spelling of a guest pass was one that opened its own command
        // buffer and WAITED, there was no point inside `enqueue` to put one
        // at, and refusing was the only honest answer.
        let attached: Vec<Attached> = submission
            .attachments
            .iter()
            .map(|attachment| Attached {
                lane: attachment.lane,
                instance: attachment.instance,
                at: attachment.at,
            })
            .collect();

        // ── **THE MARSHAL** (media-door §6). The contract's media rows in,
        //    `serve::Media` borrows out — the same eight fields, owned there
        //    and borrowed here — plus the ONE conversion no party above the
        //    load could have made.
        //
        //    **A PAYLOAD IS `f32` UNTIL IT MEETS A PLAN.** A front-end
        //    computes real numbers; a plan computes in the element its text
        //    declares, and `RuntimeInput::Patches` is where that is written
        //    down. No party above the load holds a trace, so a submission
        //    stated in bytes would have had to guess an element and would have
        //    guessed it in the runtime, for every engine at once. It is
        //    converted here, where `Shell::patch_element` is a value this
        //    shell reads off its own load.
        //
        //    **THE BLANKET REFUSAL THAT STOOD HERE IS GONE.** It said this
        //    shell binds no patch seat — nothing reserved
        //    `RuntimeInput::Patches`, nothing staged a payload, nothing
        //    carried a second row axis through `compose` — and every clause of
        //    that was true when it was written. `Inputs::reserve` carves the
        //    six regions now, `Shell::prepare` seriates them through
        //    `compose_axes`, and `Run::cut` cuts a patch rectangle at its own
        //    window.
        //
        //    **AND A TEXT-ONLY FIRE PAYS NOTHING FOR IT.** `submission.media`
        //    is empty for every fire this engine served before the door, so
        //    the two vectors below are never allocated and the `StepView` is
        //    handed the same empty slice it always was.
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            // A media submission against a load whose plan states no patch row
            // has no element to convert into and no tower to convert for. The
            // shell's own refusal, taken at the first instant it is knowable
            // rather than after a rectangle has been sized against a zero.
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

        // **THE READOUT POLICY IS CHECKED BEFORE ANYTHING RUNS** (article 4).
        // It used to be read in a loop after the fire, which was harmless when
        // the fire was over by then; under run-ahead a refusal there would
        // arrive with the step already on the device.
        //
        // **AND IT IS THE READOUT SEAT'S ARITHMETIC THAT REFUSES, NOT A
        // MISSING KERNEL.** Reading an interior row needs the logits
        // rectangle addressable after the walk, which it is (`slots.0[out]`)
        // — but under run-ahead the step behind this one carves over that
        // rectangle, so every row a reader wants is blitted into the arm's
        // own seat inside this step's command buffer, and that seat is a
        // load-time reservation of exactly `max_lanes` rows of the out seam.
        // A row LIST has no such ceiling: its bound is `max_tokens`, which is
        // 8192 at the default budget and, at a qwen vocabulary, 2.4 GiB of
        // `MTLBuffer` per arm for a column almost every fire leaves cold.
        // Serving it wants either a stated readout ceiling in `Boot` or a
        // seat that grows, and a seat that grew would move bytes a committed
        // command buffer had already been told to copy into.
        //
        // **AND THE REFUSAL IS SPLIT, BECAUSE A FIRE'S LOGITS HAVE TWO
        // READERS AND ONLY ONE OF THEM IS THIS SEAT.** Everything above is
        // about the HOST mirror — the seat, its ceiling, the copy that fills
        // it — and it is all still true. The other reader is a GUEST: an
        // epilogue that reads `intrinsics::logits()` inside a `fwd.epilogue`
        // and argmaxes on the DEVICE, which is how every sampler and every
        // speculative verifier in the corpus gets its tokens. It reads its
        // rows out of the arena rectangle at an offset the shell binds, so
        // the seat's ceiling is not its ceiling and never was — the paragraph
        // above said as much while the attachment path was refused and there
        // was nothing to do about it.
        //
        // So a row list is served for the lane whose epilogue asked for it,
        // and refused for a lane that has no epilogue at all. The rows of an
        // attached lane are DELIVERED — on the device, to the program that
        // named them — and the empty `LaneReadout` `readouts_of` answers for
        // it is not a drop: it is the same answer `Readout::None` gets, and
        // the caller asking for those rows was the guest.
        for (index, lane) in submission.lanes.iter().enumerate() {
            let listed = matches!(lane.readout, Readout::Rows(_));
            let served = attached
                .iter()
                .any(|a| a.lane as usize == index && a.at == engine::fire::Boundary::Epilogue);
            if listed && !served {
                return Err(Error::unsupported("metal", "row-selected readout"));
            }
        }

        // The sink is cloned out before the shell is borrowed: one `Arc`
        // bump, and it is what a completed command buffer publishes through.
        let sink = self.sink.clone();
        // ── **WHICH LANE CARRIES WHICH ADAPTER** (alto adapter §6.4: the plan
        //    says WHETHER, the bind says WHICH). A lane's adapter is the slot
        //    its ATTACHED INSTANCE landed at bind — never a channel this fire
        //    reads, and never a number a guest names. `Lane::adapter` arrives
        //    from the runtime as `None` for every lane (the ETA port
        //    vocabulary has no adapter port), so this is where the axis
        //    becomes real for a guest-declared correction.
        //
        //    An instance with no adapter contributes nothing, and a fire whose
        //    lanes are all of that kind walks a `BTreeMap` that is empty in
        //    every deployment that never bound one — which is what keeps an
        //    adapterless fire byte- and launch-count-identical to the one it
        //    was before this wave.
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
        // ── **AND THE WORD MOVES WITH IT.** A lane's fact word and its adapter
        //    are one reading of one lane — `Fault::AdapterWord` refuses a fire
        //    where the two disagree — and the runtime stamped this word before
        //    anybody knew there was a slot. So the word is re-stated here, into
        //    the class this bake's correction window actually covers, and a
        //    bake that has no such class refuses BY NAME rather than firing the
        //    lane uncorrected.
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
                // THE ONE AXIS THE METAL `Seated` DOES NOT CARRY, REFUSED
                // RATHER THAN DROPPED. **THE ADAPTER USED TO BE THE THIRD AND
                // THE CAPTURE THE SECOND, AND NEITHER IS ANY MORE**, and the
                // difference is exactly the difference the paragraph below
                // draws: an adapter is a declared axis with a runtime input
                // and an arm that now runs, so the answer depends on the
                // ARTIFACT and the refusal moved to the fire, where it can
                // name the model text (`Fault::Adapterless`,
                // `Fault::AdapterWord`). `captures_scores` has just made the
                // same journey — the shell carves an observability slab, the
                // capture arm writes it and an epilogue's `attn_score`
                // intrinsic is bound at it — so it crosses as
                // `Seated::captures_scores` and is answered at the fire
                // against the artifact (`Fault::Scoreless`,
                // `Fault::ScoreWord`), which is what a declared axis's
                // refusal has to be able to say. `drafts` does not depend on
                // the artifact in that way, so its refusal stays at the door
                // where it is cheapest and clearest.
                //
                // What makes it worth refusing rather than dropping is what a
                // silent drop would look like: a draft dropped answers a
                // one-token step to a speculator expecting `k`.
                //
                // **`drafts` IS THE ONE WHOSE REFUSAL IS ABOUT THE
                // DECLARATION AND NOT THE READER ANY MORE.** It used to be
                // the reader: the M2 emitter gave every intrinsic one buffer
                // at index 6, so a draft column had no second rectangle and
                // `serve::prepare` refused to attach a program that read one.
                // The slot table ended that — `IntrinsicId::MtpLogits` is
                // bound at the `mtp` export's rectangle for a guest epilogue,
                // exactly as the CUDA shell does it — and the arm was never
                // the problem: the draft head is ordinary text and `compose`
                // puts a lane whose WORD carries the draft class in the `mtp`
                // window without complaint.
                //
                // What this flag is, and what this plane cannot yet do with
                // it, is the CROSS-CHECK. `Seated::drafts` on the CUDA side
                // is held against whether the fire's class word actually runs
                // the draft arm, and against whether the load exports the
                // seam at all; this `Seated` carries no such field, so
                // accepting the flag would be accepting a declaration nothing
                // compares to anything. A caller that means to draft says so
                // in `Lane::word`, and its epilogue is served.
                if lane.drafts {
                    return Err(Error::unsupported("metal", "mtp draft readout"));
                }
                Ok(Seated {
                    lane: Lane {
                        slot: lane.slot,
                        // **THE RESTATED WORD**, which is the caller's own for
                        // every lane that bound no adapter and the correction
                        // window's for one that did.
                        word: words[at],
                        tokens: &lane.tokens,
                    },
                    pages: &lane.kv.pages,
                    held: (!lane.kv.pages.is_empty()).then_some(lane.kv.held),
                    // **THE OBSERVABILITY AXIS, CARRIED AND ANSWERED AT THE
                    // FIRE** (`.wiki/alto/attn-score.md` §4), exactly as the
                    // masked one below it. The ask is a declaration; whether
                    // this lane's word really puts its rows in the capture
                    // window is the shell's to check, and it does.
                    captures_scores: lane.captures_scores,
                    // **THE MASKED AXIS, CARRIED AND ANSWERED AT THE FIRE.**
                    // The `masked` fact is a declared axis (design §0/§8) and
                    // the mask itself is a runtime input, so the shell takes
                    // the bits and expands them (`crate::mask`) into the
                    // dense plane the metal sdpa entries read. What is
                    // answered at the fire rather than here is everything
                    // that depends on the ARTIFACT and on the composition: a
                    // mask against a plan that bakes no masked class
                    // (`Fault::Maskless`), a mask whose presence and whose
                    // word say different things (`Fault::MaskWord`), and a
                    // mask that does not describe the lane it rides on
                    // (`Fault::Mask`, `Fault::MaskRows`). None of those can
                    // be asked at this door, because none of them is a fact
                    // about the submission alone.
                    mask: lane.mask.as_ref(),
                    // **THE ADAPTER AXIS, CARRIED AND ANSWERED AT THE FIRE**,
                    // by the argument the mask closed one field up: an
                    // adapter is a declared axis, the id is a runtime input,
                    // and the PLAN decides whether anything reads it. An id
                    // against an artifact that bakes no correction is
                    // `Fault::Adapterless`; an id that disagrees with the
                    // word the runtime stamped is `Fault::AdapterWord`. Both
                    // are facts about the artifact and the composition, so
                    // neither can be asked at this door.
                    //
                    // **AND THE GUEST SINK'S SLOT WINS WHERE THERE IS ONE.**
                    // A lane attached to an instance that bound a `lora` sink
                    // routes to the slot that bind landed at; a lane with no
                    // such attachment keeps whatever the control plane named
                    // through `Lane::adapter`, which is the door
                    // `Engine::register_adapter` has always served.
                    adapter: lane_adapters[at].or(lane.adapter),
                    // **EMPTY IS THE DERIVED RUN AND IT CROSSES AS EMPTY.**
                    // The contract spells "absent" as a zero-length vector
                    // rather than as `None`, and this hands the slice over
                    // unchanged so the shell makes the same distinction on
                    // the same evidence — `Seated::positions` argues which
                    // seat a stated run reaches and which ones it does not.
                    positions: &lane.positions,
                    // **THE ROW LIST CROSSES TO THE DEVICE HALF, AND ONLY
                    // THERE.** `readouts_of` fills the host mirror from the
                    // arm's seat, which holds one row per lane; this hands
                    // the row indices to the fire, because the other reader
                    // of a fire's logits is a guest epilogue and the shell is
                    // the only party that knows where a lane's run sits in
                    // the arena rectangle.
                    //
                    // `Last` and `None` both cross as `None`, which is the
                    // lane's last row: that is the row every epilogue was
                    // given before a list could be stated, and a lane that
                    // asked for no host mirror may still carry one.
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
                    // **THE WORKING SET'S TABLE, QUOTED AND NOT INTERPRETED.**
                    // Empty for every lane whose page references the runtime
                    // already resolved, which is every lane of every class but
                    // `DeviceGeometry`; for that one it is the only thing that
                    // maps the guest's relative indexes onto pool page ids,
                    // and the shell may index it and nothing else
                    // (`Seated::translation` argues both spaces).
                    translation: &lane.kv.translation,
                })
            })
            .collect::<EngineResult<Vec<_>>>()?;

        // ── **THE THREE PHASES, AND THE VERB RETURNS BEFORE THE DEVICE
        //    DOES** (articles 1 and 2). `prepare` makes every host decision,
        //    takes an A/B seat and commits the step's demand against the
        //    pools; `enqueue` encodes the whole walk, copies the answer out of
        //    the arena into that seat and commits the command buffer without
        //    waiting; `settle` files the flight. Three calls and not one wait.
        //
        //    `Shell::fire_seated` is the SYNCHRONOUS spelling of the same
        //    three with the harvest on the end; it is what the native surface
        //    and the smoke tests use, and it is deliberately not what this
        //    path calls.
        let landed = {
            use engine::frame::Shell as FrameShell;
            let done = sink.map(|sink| Done { at, sink });
            let prepared = FrameShell::prepare(
                shell,
                StepView {
                    lanes: &seated,
                    attachments: &attached,
                    // **AND THE IMAGES CROSS.**
                    media: &media,
                    done,
                },
                None,
            )
            .map_err(fault)?;
            let enqueued = FrameShell::enqueue(shell, prepared).map_err(fault)?;
            FrameShell::settle(shell, enqueued).map_err(fault)?
        };

        // **EMPTY READOUTS, AND THE CONTRACT ALREADY SAID SO** (`FireTicket`'s
        // own doc: "an asynchronous shell answers with the id and an empty
        // readout list"). The numbers, for a caller that wants them, come from
        // `Engine::settle_frame` — which `submit` itself calls at depth one,
        // so the eager shell's receipts are filled exactly as they always
        // were.
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

/// **One step's rows, in the shape each lane asked for.**
///
/// The shell answers one row per lane — the last one, which is what a sampler
/// wants and the reason a prefill does not hand back half a megabyte per
/// teacher-forced position — and this is where that becomes the contract's
/// per-lane record.
///
/// **A `Readout::Rows` LANE ANSWERS AN EMPTY RECORD, AND THAT IS NOT A
/// DROP.** A row list reaches this shell only for a lane whose epilogue asked
/// for it — `fire_step` refuses one that has no epilogue — and those rows
/// were delivered: on the device, into the guest program that named them, at
/// the arena rectangle `serve::enqueue` bound its `logits` intrinsic to. The
/// host mirror is a different reader with a different ceiling (one row per
/// lane, the arm's seat), and it is the reader this lane did not ask.
fn readouts_of(step: &PendingStep) -> Vec<LaneReadout> {
    let rows: &[Vec<f32>] = step.rows.as_deref().unwrap_or(&[]);
    step.readout
        .iter()
        .enumerate()
        .map(|(lane, policy)| match policy {
            // `scores` is empty and stays empty, and that is no longer a
            // statement about whether anything captured. `LaneReadout::scores`
            // is the per-QUERY column palo C4b named — `LayerScores { layer,
            // rows, heads, lse }` — and the observability door publishes a
            // different number (per-KEY mass, `.wiki/alto/attn-score.md` §2.3
            // calls pointing one at the other "a lie that computes"). A
            // capturing lane on this plane is served on the DEVICE, where the
            // epilogue's `attn_score` intrinsic reads the slab in place; the
            // host mirror of the other column has no writer here.
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

// STILL `unsafe impl`, and the CUDA sibling's rule holds over different
// contents. `Engine` is `Send + Sync` and a loaded `Shell` is neither to the
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
// What makes both sound is the engine's own rule: every verb that reaches
// either takes `&mut self`, so exactly one thread touches a shell at a time.
// `kind` and `device_facts` are the only `&self` verbs on this impl and
// neither goes near the shell.
unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}
