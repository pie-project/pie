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
//! `encode` takes the trait's default body, which answers
//! [`Error::Unsupported`]: this shell carries no multimodal encoder. Stubbing
//! it to `Ok(())` would make an image prompt *appear* to work and silently
//! read the wrong bytes, which is the failure mode the contract's "refusal is
//! a value" section exists to prevent.
//!
//! `copy_kv` and `copy_state` stood beside it and no longer do — the shell has
//! a page mover ([`Pools::copy_kv`](crate::store::Pools::copy_kv)) and a slot
//! mover ([`Pools::copy_slot`](crate::store::Pools::copy_slot)) now. `copy_kv`
//! serves DEVICE-TO-DEVICE inside this load's own pools and refuses every
//! other direction BY NAME: a host-pinned end is a swap pool this shell does
//! not reserve, and a second device's is a peer mapping it has not opened.
//! [`Capabilities::kv_copy`](engine::caps::Capabilities) is where
//! that is stated rather than discovered. `resize_pool` is not on the trait at
//! all any more (design §8, wave C: an elastic pool grows as a side effect of
//! frame admission).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::{
    ChannelId, ChannelRegistration, HostMirror, RegisteredChannel,
};
use engine::error::{Error, Result as EngineResult};
use engine::fire::{
    FireId, FireTicket, FrameId, FrameSubmission, FrameTicket, LaneReadout, Readout,
    Step,
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
// `Eq` left with [`Knobs`]'s: the knobs carry `gpu_mem_utilization`, an `f64`,
// and a total equality over a float is a claim neither struct should make.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBoot {
    /// Which device to bind.
    pub ordinal: i32,
    /// How much of a fire to record, from `[engine] graphs`. Nothing
    /// overrides it any more (article 9: shells read no environment).
    pub graphs: Graphs,
    /// **THE SHELL'S OWN WORDS**, from the boot document's `[engine]` table —
    /// what nine `PIE_CUDA_*` environment reads were before article 9.
    ///
    /// A deployment fact like the ordinal beside it: the knobs describe how
    /// this machine's shell fires, not what one model asks for, so they are
    /// stated once when the engine is opened and carried onto every
    /// [`Boot`] it makes. [`Knobs::default`] is what an
    /// absent variable meant, byte for byte.
    pub knobs: Knobs,
    /// **Where this deployment keeps its warm-boot weight artifacts** (alto
    /// design §7's T2 tier), from `[model] weight_cache_dir`.
    ///
    /// A deployment fact, which is why it sits here beside the ordinal rather
    /// than on the load: the directory belongs to the machine and the
    /// operator, and every model this engine ever loads writes into the same
    /// one under its own key.
    ///
    /// **TYPED, NOT `getenv`** (article 9). The worker has been writing this
    /// key into every CUDA boot document since the palo rewrite and nothing
    /// read it; this is the field it arrives at. `None` is the honest answer
    /// for a deployment that named no directory — the feature is off, no
    /// artifact is read and none is written.
    pub weight_cache_dir: Option<std::path::PathBuf>,
    /// **Where this deployment keeps the guest-program plane's compiled
    /// cubins**, from the boot document's `[cache] dir`.
    ///
    /// A deployment fact beside [`DeviceBoot::weight_cache_dir`], and typed
    /// for its reason: `program::compile::Disk` used to resolve
    /// `$PIE_HOME/cache/ptir-cuda` itself, which was the last environment read
    /// in the shell (article 9). `None` is the feature off — every program
    /// compiles through NVRTC and nothing is stored, which costs time and
    /// never an answer.
    pub program_cache_dir: Option<std::path::PathBuf>,
    /// **Where this deployment keeps its shared adapters** (alto adapter
    /// §3.3), from `[model] adapter_dir`.
    ///
    /// A read-only directory whose subdirectories are adapters: one
    /// `adapter.toml` and the plane files it names. It is a deployment fact
    /// for the same reason the two cache directories above it are — the
    /// mount belongs to the machine and the operator, every model this engine
    /// loads reads the same one, and adding a LoRA to a serving box is
    /// writing a file into it rather than calling a verb.
    ///
    /// `None` is the feature off: shared binds refuse by name and a
    /// byte-seeded adapter is unaffected. **TYPED, NOT `getenv`**
    /// (article 9).
    pub adapter_dir: Option<std::path::PathBuf>,
}

impl Default for DeviceBoot {
    fn default() -> DeviceBoot {
        DeviceBoot {
            ordinal: 0,
            graphs: Graphs::default(),
            knobs: Knobs::default(),
            weight_cache_dir: None,
            program_cache_dir: None,
            adapter_dir: None,
        }
    }
}

/// The CUDA shell, behind [`Engine`].
/// One submitted step, held for a caller that comes back for numbers.
///
/// The readback plan the shell computed at `settle`, plus the per-lane
/// `Readout` policy the SUBMISSION stated.
///
/// **THE POLICY IS CARRIED HERE AND HANDED DOWN AT THE NUMBERS DOOR**
/// (`palo B-readout`, closed). It used to be a list the shell never saw, on
/// the argument that which rows a caller wants back is a contract question
/// and not a device one — true of `Last` versus `None`, which are two ways of
/// spending the same one row the shell read anyway, and false the moment
/// `Rows` names three: the shell is the only party that knows where a lane's
/// row run sits in the arena rectangle, so the list has to reach the loop that
/// indexes it. `settle_frame` passes it into `Shell::read_out_rows`; nothing
/// on the FIRE path reads it, which is the part that was load-bearing.
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
    sink: Option<engine::CompletionSink>,
    /// **The last submitted frame's per-step readback plans**, held for a
    /// caller that comes back for numbers ([`Engine::settle_frame`]).
    ///
    /// One frame's worth and no more, and that is a statement about the arena
    /// rather than a cache policy: the out seam and the export columns are
    /// arena rectangles the NEXT fire carves over, so a frame's numbers exist
    /// only until the frame after it is enqueued. Holding two would be
    /// offering to answer with bytes the device has overwritten.
    pending: Option<(FrameId, Vec<PendingStep>)>,
    /// **WHICH ADAPTER SLOT EACH BOUND INSTANCE ROUTES TO** (alto adapter
    /// §6.4: the plan says WHETHER, the bind says WHICH).
    ///
    /// An instance whose program declares the `lora` sink has its weights
    /// landed ONCE, at [`Engine::bind_instance`], out of the cells the guest
    /// seeded — never at fire time, which is §6.1's whole ruling. The slot the
    /// store answered is kept here, and every lane a fire attaches to that
    /// instance is stamped with it while the fire is composed. Instances with
    /// no adapter are not in this map at all, and a fire whose lanes are all
    /// of that kind reads it once, finds nothing, and costs the axis nothing.
    ///
    /// **A BIND IS A REFERENCE AND IS GIVEN BACK AT `close_instance`**, which
    /// is what makes a slot reclaimable; a map that only ever grew would fill
    /// the bank and refuse the ninth instance of a serving box that had long
    /// since finished the first eight.
    adapters: BTreeMap<InstanceId, crate::Binding>,
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
        self.loaded_mut()?
            .program_instance(id)
            .map_err(fault)?
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
/// `engine::error`'s header draws.
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
        // **THE RESIDENCY REFUSAL IS `Impossible` AND SAYS SO IN ONE PLACE**
        // (alto design §7). A budget the tiers cannot meet is not a pool the
        // deployment can free its way out of; the sentence already carries
        // both numbers and what would have to change.
        Fault::Residency(_) => Error::Impossible(fault.to_string()),
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
        Fault::Mask { .. }
        | Fault::MaskRows { .. }
        | Fault::Maskless { .. }
        | Fault::MaskWord { .. } => Error::Invalid(fault.to_string()),
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
        // The shared-adapter mount's two, and they sort apart. A blob refusal
        // is the deployment's — a name that is not in the mount, a manifest
        // that disagrees with the model text's banks — and nothing a caller
        // frees changes it, so it is `Load` beside the registration above.
        // Slot exhaustion is the only one of the axis that a caller CAN clear:
        // `slots` bounds concurrent residency, so a bind that waits for
        // another instance to finish is a real answer.
        Fault::Blob { .. } => Error::Load(fault.to_string()),
        // One slot wanted, none reclaimable — the numbers are about what is
        // FREE and not what exists, because a table whose every seat is
        // pinned has no free one whatever its width.
        Fault::AdapterSlots { .. } => Error::Exhausted {
            resource: "adapter slots",
            wanted: 1,
            available: 0,
        },
        // A `Fault::Blocked` arm stood here and crossed as `Error::Exhausted`
        // — an attached guest's ring with no room right now, which the host
        // was to drain before re-submitting the identical frame. Article 4
        // has no such outcome: `validate_frame` proves ring occupancy,
        // host-writer staging and reader pressure statically at submit, so a
        // readiness miss past admission is a contract violation and
        // `serve::committed_or` names it. `Fault::OutOfMemory` above is the
        // only exhaustion this shell reports, and it is about device bytes.
        Fault::Compile(_) | Fault::Program { .. } | Fault::Interpret(_) => {
            Error::Program(fault.to_string())
        }
        // A submission whose two halves disagree is the caller's statement
        // and not the device's condition, so it crosses as `Invalid` beside
        // the fire faults it is the shell-side twin of — no amount of freeing
        // makes a short patch payload the geometry it claimed.
        Fault::Fire(_) | Fault::PatchPayload { .. } => Error::Invalid(fault.to_string()),
    }
}

/// The contract's bind extents, in the plane's spelling.
///
/// Two names for one seven-role vector: [`SymbolicExtent`](eta_exec::Role) is
/// the tag space both
/// are written in, and the conversion is field for field so that adding a
/// role to one without the other is a compile error rather than a silently
/// unresolved axis.
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

/// **THE LATTICE A DEPLOYMENT GETS WHEN IT STATES NONE**: the powers of two
/// from [`LATTICE_FLOOR`] up to and including `max_tokens`.
///
/// `Budget::buckets` is a deployment's dial and `Budget::new` leaves it
/// empty, which `compose::bucket_of` reads — correctly — as "one graph per
/// exact size, and the honest bucket for a fire of `rows` rows is `rows`".
/// That answer makes every consumer of the lattice a no-op: P4's fallback menu
/// collapses to one bucket at position 0, and D4's padding rounds a fire up to
/// itself. A shell whose whole business is firing on a real device should not
/// ship that as its default, because a dial nobody set is not a measurement of
/// the dial's zero.
///
/// **POWERS OF TWO, BECAUSE GEOMETRIC IS WHAT BOUNDS THE TAIL.** Above the
/// floor a fire never computes more than twice its own rows, which is D4's
/// whole cost argument stated as a ratio rather than as a hope. It is also the
/// spacing of the fourteen-point lattice `crate::window`'s header prices the
/// copy/split crossover on and `every_sku_walks_its_classes` walks, so the two
/// consumers of `Budget::buckets` are looking at the same kind of object.
///
/// The ceiling is included even when it is not a power of two, because a fire
/// AT `max_tokens` must have a bucket and `Fault::NoBucket` is the refusal for
/// a fire above the lattice, not for the largest one the budget admits.
#[must_use]
pub(crate) fn default_lattice(max_tokens: u32) -> Vec<u32> {
    let mut lattice: Vec<u32> =
        core::iter::successors(Some(LATTICE_FLOOR), |point| point.checked_mul(2))
            .take_while(|point| *point < max_tokens)
            .collect();
    lattice.push(max_tokens);
    lattice
}

/// **WHERE THE DEFAULT LATTICE STARTS, AND WHY IT IS NOT 1.**
///
/// A lattice is free to name every small size, and the fourteen-point one this
/// tree quotes does. D4 asks for the opposite at the bottom, for two reasons
/// the census measured:
///
/// * **The arm flip at one row is the whole point.** `.wiki/palo/cuda-abi.md`
///   §1: a one-lane decode takes the gemv arm and a two-lane one does not —
///   127 launches change kernel across that boundary, and the 423-node
///   topology it produces is a shape of its own. §3's promise is that "the
///   gemv↔gemm arm flip at ×1 dies (M ≥ 2 always)", and a lattice naming 1
///   keeps it alive. §3's own worked example — "decode 3 lanes padded to 8" —
///   is a lattice whose first point is this one.
/// * **A boundary is where two fires stop agreeing.** Padding does not remove
///   the arithmetic drift between two compositions; it QUANTIZES it (two fires
///   compute the same numbers iff they share a bucket — see
///   `a_padded_fire_is_in_bounds_and_says_something_true`). Every extra point
///   at the bottom is one more place where a lane fired alone and the same
///   lane fired beside two others land on different sides, and at decode
///   scale that is the commonest pair a deployment has.
///
/// What it costs is the tail on the smallest fires, where the cost argument is
/// strongest rather than weakest: a decode fire's linear layers are
/// weight-bound — 1.40 GiB of weight reads against eight rows of activation —
/// so the rows below the floor ride reads that were happening anyway.
/// `the_tail_a_padded_decode_computes_rides_the_weight_reads` is that claim
/// with a number on it, and a deployment that measures otherwise on its own
/// hardware states its own `Budget::buckets`.
pub(crate) const LATTICE_FLOOR: u32 = 8;


/// **THE SHAPE LATTICE POLICY, AT THE DOOR** (alto wave P, article 9).
///
/// It was `lattice_from_env` in `serve.rs` — `PIE_CUDA_BUCKETS` read at load,
/// under a header claiming the file had no logic. Two things were wrong with
/// that and only one was the environment: which buckets exist is a COMPILER
/// input, so the policy belongs beside [`bake_budgets`], where this crate
/// turns a load request's ceilings into the `Budget` the bake is handed.
///
/// A caller that stated a lattice keeps it, exactly. One that stated none —
/// which is every `Budget::new` caller, and so every gate and the worker's own
/// embedded engine — gets [`default_lattice`], because the empty lattice makes
/// P4's bucket ranges collapse to one position and D4's padding round every
/// fire up to itself, and a dial nobody set is not a measurement of the dial's
/// zero.
///
/// Idempotent by construction: a filled lattice is stated, and a stated
/// lattice is kept.
#[must_use]
pub(crate) fn lattice(stated: Vec<u32>, max_tokens: u32) -> Vec<u32> {
    if stated.is_empty() {
        default_lattice(max_tokens)
    } else {
        stated
    }
}

/// The ceilings the compiler bakes against, out of the ones the load states.
///
/// The contract carries seven numbers and `model_compiler::Budget` takes
/// four; the other three (`page_size`, `max_context`, `slots`) are the POOLS'
/// and go to `Boot` directly. Converted in one place, which is the whole
/// reason `engine` states its own `Budget` rather than depending on the
/// compiler (`load.rs`'s note).
fn bake_budgets(budgets: &LoadBudgets) -> Budget {
    Budget {
        max_lanes: budgets.max_lanes,
        max_tokens: budgets.max_tokens,
        buckets: budgets.buckets.clone(),
        max_adapters: budgets.max_adapters,
    }
}

/// **THE SECOND ROW AXIS'S LADDER, DERIVED FROM THE TEXT THAT NEEDS ONE**
/// (multimodal §5.5) — `bake_budgets`' twin for the patch axis.
///
/// `None` for a plan that states no `Dim::Patches`, which is every SKU served
/// before the towers and is G4's invariant restated at the door: the ladder
/// only exists when a text asks for it, so a text-only artifact bakes through
/// `model_compiler::compile` exactly as it always did.
///
/// # The rungs, and the argument for each number
///
/// * **the floor is [`PATCH_LATTICE_FLOOR`]**, which is the module's own — the
///   smallest whole image a resize policy admits (64 rows at patch-16 /
///   merge-2). A rung below it rounds up to a fire that cannot exist.
/// * **the ceiling is the token rectangle's, capped at two whole images.**
///   Every tower row ends up scattered into a TOKEN row, so a fire can never
///   usefully carry more patch rows than `max_tokens`; and the catalog's
///   towers store a 48 x 48 position grid, so one image at its native grid is
///   at most 2304 patch rows and [`DERIVED_PATCH_CEILING`] admits two of them.
///   The smaller of the two wins, because a deployment that stated a small
///   token rectangle meant it.
/// * **the rungs double**, as the token lattice does and for the reason it
///   does not: patches-per-image is fixed by the resize policy, so a handful
///   of rungs covers "one image", "two", "four" without a rung that no fire
///   lands on.
/// * **`max_images` is the ceiling at the floor** — as many images as the
///   patch ceiling holds if every one of them is the smallest whole image.
///   That is the honest bound rather than a guess, and it costs
///   `(images + 1) * 4` bytes of indptr to be generous with.
///
/// An operator who has measured their traffic states
/// [`Budgets::max_patches`](LoadBudgets::max_patches) and
/// [`max_images`](LoadBudgets::max_images) instead, and then this function
/// only supplies the rungs.
///
/// `pub` so a gate can boot a tower against the ladder this engine would
/// derive for it, rather than against a ladder the gate invented — which is
/// the difference between proving the derivation serves and proving some
/// ladder does.
pub fn patch_ladder(trace: &Trace, budgets: &LoadBudgets) -> Option<PatchLadder> {
    /// Two whole images at the catalog towers' native 48 x 48 grid.
    const DERIVED_PATCH_CEILING: u32 = 4096;

    // **THE PLAN IS WHAT ASKS.** Read off the types a text already wrote
    // (`Dim::axis`), never off a flag: a value on the patch axis is the whole
    // of what makes a plan a two-unit one, and the ladder follows the same
    // reading `model_compiler::unit` does.
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
        activation: Dtype::F32,
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
        // **AND NOW THE SCORE COLUMN HAS SOMEWHERE TO STAND TOO**
        // (`.wiki/alto/attn-score.md` §4, wave S1). This was `false` under a
        // note that named two blockers, and both are gone rather than
        // waived. `AttnScore` was registered at `Stage::OnAttn`, a mid-graph
        // tap design §9 abolished — it is registered at `Stage::Epilogue`
        // now, and the capture arm ACCUMULATES the per-key rectangle in-graph
        // instead of a boundary being torn open to compute it. And it
        // promised per-key softmax weights where this axis exported a
        // per-query mass — it gets per-key softmax weights now, from
        // `attention.score_capture`, written into the shell's own slab.
        //
        // So this is exactly "does this load export a capture column, and did
        // the slab that observes it get carved", which is what a bind-time
        // contract has to mean. A text with no `attn.scores` seam still
        // answers `false`, and every refusal downstream still fires by name.
        has_attn_score: shell.observes_scores(),
        has_attn_page_mask: false,
        // **HONEST, AND NOW OPEN** (alto adapter §6.4, wave A2). This was
        // `false`, and `false` was the truth: `plan.needs.lora` had zero
        // readers in this crate, so a program carrying the sink would have
        // bound, compiled, fired and quietly answered the BASE MODEL — the one
        // wrong answer this axis must never give, and the reason the
        // capability existed shut rather than open.
        //
        // It is `true` because the sink is consumed now. `Cuda::bind_instance`
        // reads the sink off the launch package, takes the weights off the
        // cells the guest seeded, converts them into the banks' own dtype and
        // lands them in a slot; `Cuda::fire_step` stamps that slot onto every
        // lane attached to the instance, with the fact word moved into the
        // correction's window beside it. A load whose model text declares no
        // bank still refuses by name (`Fault::Adapterless`), which is a
        // sentence and not a silent zero.
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
/// exactly what §6.1 ruled out. A seeded channel's cell is already on this
/// side of the wire at bind ([`InstanceBinding::seeds`]), so the resolver
/// takes it from there, converts it into the banks' own dtype, and never looks
/// at the ring again. A guest that publishes new adapter weights mid-pass is
/// therefore NOT serving a new adapter — the honest reading of "swapping an
/// adapter is re-seeding", which is a re-BIND.
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
) -> EngineResult<Option<crate::Binding>> {
    let Some(sink) = shell.program_adapter_sink(program).map_err(fault)? else {
        return Ok(None);
    };
    let seats = shell.bank_seats();
    // **WHICH SITE THE GUEST ASKED FOR** (alto next B3), read off the sink's
    // placement constant once and checked against the banks by `planes_of`:
    // a text that names its site refuses a mismatch, a text that names none
    // means what it always meant.
    let site = sink.site().map_err(fault)?;
    let mut built: Vec<(String, Vec<u8>)> = Vec::new();
    for (role, channel) in &sink.planes {
        // **A CHANNEL THE SINK NAMES AND THE BIND DID NOT SEED IS A REFUSAL**,
        // not a plane of zeros. A zero `A` is the IDENTITY adapter — it is the
        // construction every bank gate starts from — so accepting an unseeded
        // channel would answer the base model under a program that asked for a
        // correction, and answer it silently.
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
        .bind_adapter(crate::AdapterSource::Own { instance, planes: &planes })
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
            residency,
            ordinal,
            frames_in_flight,
        } = request;

        // Serving eagerly is a choice a deployment may make — for a bisect,
        // for a golden diff — but never one it should make silently: an
        // uncaptured decode pays hundreds of kernel launches per token-step
        // of pure CPU time (~25% of single-stream latency measured on
        // qwen35-d0.8b, 2026-08-29). `Graphs::On` is the default; this warn
        // is the receipt for overriding it.
        if !self.boot.graphs.records() {
            eprintln!(
                "engine-cuda: serving without CUDA graph capture ([engine] graphs = \
{:?}, not \"on\"): every fire launches eagerly, which costs per-step host time; \
intended for diagnostics, not serving",
                self.boot.graphs
            );
        }

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

        // ── RESIDENCY, BEFORE A BYTE IS ALLOCATED (alto design §7).
        //    This shell has TWO tiers now, and which of them a plane can live
        //    in is a property of the plane rather than a setting:
        //
        //      routed expert banks   a device slab of `n < experts` slots over
        //                            a pinned host copy of all of them, behind
        //                            a device-resident indirection table
        //      everything else       resident, whole, as it always was
        //
        //    `Plan::of` reads the budget against the trace ALONE — no device
        //    is bound and no byte is allocated — and answers the residency
        //    this load will actually have: the empty plan for an uncapped
        //    budget or one that covers the whole table (dev's `place_all`),
        //    a per-bank slot count for one that does not, and
        //    `Fault::Residency` -> `Error::Impossible` for a budget under the
        //    dense planes, which do not stream in this wave.
        //
        //    `Residency::admit` is then asked with what the PLAN demands
        //    rather than with what the checkpoint holds — the device demand
        //    it fits by construction, and the host demand is the real
        //    question, because the pinned tier holds every expert of every
        //    streamed bank and a `host_weight_budget` under it is a load this
        //    machine cannot hold either.
        //    **AND THE PLAN IS READ OFF THE LOAD PLAN'S PAIRINGS TOO**, not
        //    off the trace alone: a QUANTIZED bank is codes beside factors,
        //    two params under one `Def::Weight`, and a residency decision that
        //    moved one of them would leave the other reading somebody else's
        //    expert (`crate::experts`'s header states it whole). The pairing
        //    is the loader's own record; `weights::prospect` reads it for one
        //    metadata parse and one plan compile, and no tensor bytes.
        //
        //    **AND THERE IS A THIRD TIER NOW** (alto streaming §2, wave W-1).
        //    Both budgets reach the planner, so a packed bank neither of them
        //    holds is planned onto the MAPPED artifact rather than refusing
        //    the load. Whether that artifact exists is a question about a file
        //    and is answered here, once, by opening it; `admit_tiers` is the
        //    statute that turns "spilled bytes, no source" into an
        //    `Impossible` naming the one thing an operator can do about it.
        let prospect = crate::weights::prospect(&trace, &contract, &path).map_err(fault)?;
        let plan = crate::experts::Plan::of(
            &trace,
            &prospect.planes,
            crate::experts::Budgets {
                device: residency.device_weight_budget,
                host: residency.host_weight_budget,
            },
        )
        .map_err(fault)?;
        let sourced = plan.spill_demand() > 0
            && crate::weights::spill_source(
                self.boot.weight_cache_dir.as_deref(),
                prospect.resident_key,
            )
            .is_some();
        residency.admit_tiers(engine::load::Tiers {
            device: plan.device_demand(),
            host: plan.host_demand(),
            spilled: plan.spill_demand(),
            sourced,
        })?;

        // Derived BEFORE the trace moves into the boot — the ladder is a
        // reading of the plan, so it is taken while the plan is still here.
        let patches = patch_ladder(&trace, &budgets);
        let mut shell = Shell::load(Boot {
            trace,
            contract: &contract,
            checkpoint: &path,
            budget: bake_budgets(&budgets),
            // **AND THE SECOND AXIS'S, WHEN THE TEXT ASKS FOR ONE.** A literal
            // `None` until now, which made every vision SKU a load that could
            // not happen (`Error::Unsized`, named at the door); it is the
            // plan's own declaration that decides, so a text-only SKU still
            // gets the literal `None` G4 depends on.
            patches,
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
            // The deployment's words, unchanged per load — the `[engine]`
            // table is a fact about this machine's shell, not about one model.
            knobs: self.boot.knobs,
            // The deployment's directory, unchanged per load: every model
            // this engine loads keys its own artifact inside it.
            weight_cache_dir: self.boot.weight_cache_dir.as_deref(),
            // The deployment's cubin cache, unchanged per load — every program
            // this engine ever compiles keys its own file inside it.
            program_cache_dir: self.boot.program_cache_dir.as_deref(),
            // **THE DEPTH CROSSES ONCE AND IS DERIVED FROM HERE ON** (article
            // 8). `Runahead::of` clamps what the free-slot word cannot carry;
            // the deployment's config layer refuses an out-of-range depth by
            // name long before it reaches this line.
            runahead: engine::runahead::Runahead::of(frames_in_flight),
            // The plan the two budgets decided, carried whole rather than
            // re-derived: a shell that recomputed it could disagree with the
            // numbers `admit` was asked about.
            residency: plan,
        })
        .map_err(fault)?;

        // **THE SHARED-ADAPTER MOUNT, STATED AFTER THE LOAD** (alto adapter
        // §3.3). It is not a `Boot` field because it is not a property of the
        // BAKE: the banks are, and they came off the model text above. Where
        // the shared adapters live is the deployment's, it outlives every
        // load, and §3.3's hot-add is a file drop into it — so it arrives as
        // a verb, typed off the boot document, and never out of the
        // environment (article 9).
        shell.mount_adapters(self.boot.adapter_dir.clone());

        let trace_name = shell.trace().name.clone();
        let (weight_bytes, arena_bytes, pool_bytes, input_bytes) = shell.footprint();
        // **WHAT IS RESERVED AND WHAT IS MAPPED ARE TWO NUMBERS** (wave C).
        // `pool_bytes` above is the ceiling the arenas' address space was
        // reserved at; these are what admission has actually put physical
        // pages behind. Read here, at load, they are the floor a fresh load
        // starts from — the interesting reading is later, and it comes out of
        // the same accessor.
        let (pool_committed_bytes, pool_high_water_bytes, elastic_page_bytes, elastic_budget_pages) =
            shell.elastic();
        let weights_from_cache = shell.weights_from_cache();
        // Read beside the other facts, while the shell is still here to ask.
        let weights_resident = shell.weights_resident();
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
                // **THE POOLS ARE VIRTUAL NOW** (wave C), and these two
                // say so: one logical page of the elastic supply, and the
                // most of them this load may ever map. Zero was the
                // reservation model's honest answer and is no longer this
                // shell's.
                elastic_page_bytes,
                elastic_budget_pages,
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
            // **THE WHOLE FIRE GEOMETRY IS SERVED, AND THE MASK BESIDE IT.**
            // `crate::program::ports` reads every port in this set off the
            // attached instance's own rings at fire time, before anything has
            // launched, and `serve::prepare` uses each:
            //
            // ```text
            // embed_tokens  the ids                the device DECIDED them
            // embed_indptr  the member's lane CSR  which lanes, which rows
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
            //                                     into the same slab a
            //                                     host-stated mask expands to
            // ```
            //
            // The four beyond the decode envelope — the page family and the
            // write descriptor — were withheld while the page ids were this
            // shell's alone (`palo B3`). They are claimed now because
            // `geometry_with` already took a caller-stated table per lane
            // (`KvDelta::pages` non-empty), and a guest that states its pages
            // on a CHANNEL is that same caller reaching the pool one phase
            // later. The mask is claimed with them because a fire whose
            // ancestry is device data — a beam search's `gather(mask, parent)`
            // — has nowhere else to state it, and the runtime's bind-time
            // classifier asks for exactly this union before it will admit one
            // (`inferlet::host::forward`'s `devgeo_capable`).
            ports: PortMask::DEVICE_GEOMETRY.with(Port::AttnMask),
            geometry: GeometryClass::DeviceGeometry,
            // **THE ONE DIRECTION THIS SHELL HAS BYTES FOR.** `copy_kv`
            // moves cells between pages of THESE pools, on this stream, as
            // device-to-device copies (`Pools::copy_kv`). The other three
            // names a swap pool or a peer mapping: a host-pinned end needs a
            // pinned page pool this load does not reserve, and a second
            // device's ordinal needs a peer mapping it has not opened. Each
            // is refused by name in `Cuda::copy_kv`, and this is where a
            // caller reads that without having to try one.
            kv_copy: KvCopyDomains {
                device_to_device: true,
                device_to_host: false,
                host_to_device: false,
                host_to_host: false,
            },
            kv_handle: None,
            media_encode: false,
            // **F2a: THE RINGS ADVANCE ON THE DEVICE.** `register_channel`
            // below hands out the pinned host half of every channel, the
            // fire's tickets are validated by `channel::pull_validate`, and
            // `channel::commit_bump` is the only writer of durable ring
            // state. So a caller may predict cursors by counting, and its
            // pump has nothing left to carry.
            device_channel_commit: true,
            // **THE RS DEVICE HALF, STATED** (alto design §6, waves F3 and
            // F3b). This shell allocates the buffered-activation pool at
            // load, seats the `commit_len`, `write_state_mask` and segment
            // origin the chunked scans honour, and serves `RsVerb::Buffer`
            // and `RsVerb::FoldBuffered` against them — including the mixed
            // row, whose interior fold boundary is run as the head that folds
            // and the tail that continues from it.
            rs_verbs: true,
        };

        self.shell = Some(shell);
        self.caps = Some(caps.clone());
        Ok(Loaded {
            facts: LoadFacts {
                trace_name,
                weight_bytes,
                // **ANSWERED, NOT ASSUMED** (alto design §7). `false` says
                // this load opened the routed-expert tier: some bank of it is
                // a device slab smaller than the bank, over pinned host
                // bytes, and `weight_bytes` above is then what is RESIDENT
                // rather than what the checkpoint holds. `true` is the
                // degenerate case and every load that states no budget.
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
        frame.validate_for(engine::Serves {
            device_channel_commit: self
                .caps
                .as_ref()
                .is_some_and(|caps| caps.device_channel_commit),
            rs_verbs: self.caps.as_ref().is_some_and(|caps| caps.rs_verbs),
        })?;
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

    fn on_complete(&mut self, sink: engine::CompletionSink) {
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
            // The row lists the submission stated, handed down: `Last` for a
            // lane that wanted its sampler's row, the named rows for a
            // spec-decode verifier, nothing at all for a lane that ran for
            // its cache writes.
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
        // The HINT sees the same words the fire will, adapters included: the
        // prebind caches a composition, and a composition is a set of classes
        // — so a hint stated in the unadapted classes would warm the wrong
        // one and the fire it was for would find nothing. Advisory either way
        // (a wrong hint costs only the hidden work), which is why the lookup
        // here refuses nothing and simply leaves the word alone.
        let adapted: Vec<bool> = if self.adapters.is_empty() {
            vec![false; submission.lanes.len()]
        } else {
            let mut adapted = vec![false; submission.lanes.len()];
            for attachment in &submission.attachments {
                if self.adapters.contains_key(&attachment.instance)
                    && let Some(lane) = adapted.get_mut(attachment.lane as usize)
                {
                    *lane = true;
                }
            }
            adapted
        };
        let Ok(shell) = self.loaded_mut() else {
            return;
        };
        let lanes: Vec<Lane<'_>> = submission
            .lanes
            .iter()
            .enumerate()
            .map(|(at, lane)| Lane {
                slot: lane.slot,
                word: if adapted[at] {
                    shell.adapted_word(lane.word).unwrap_or(lane.word)
                } else {
                    lane.word
                },
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
        // **EVERY ROLE GETS AN ENDPOINT NOW, AND ONLY TWO OF THEM GET A
        // MIRROR** (design §5). A `HostRole::None` channel used to allocate
        // nothing here and register nothing, so its ring was cut inside
        // whichever session bound it — and a ring two passes SHARE was then
        // two rings that never met, which is the whole of the
        // "device-only private ring shared by <=8 attachments" gap. The ring
        // belongs to the channel now: this is where it is cut, once, and
        // `endpoints_for` hands the same one to every attachment.
        //
        // What a `None` channel still has no part of is the CROSSING. It
        // publishes no `HostMirror`, because there is no guest end to point at
        // it, and `mint` never sets `HOST_WRITER` on its tickets — nothing on
        // the host writes into it, so there is nothing to pull.
        //
        // It DOES set `HOST_READER`, and the mirror it opens below is not
        // dead. Nothing crosses to a guest through it: it is a pinned SHADOW
        // of the committed cell, written by `channel::scatter_publish` at the
        // same instant it writes a real guest's, so that a descriptor port
        // resolved off a device-only ring is a load out of mapped memory
        // rather than a blocking four-byte `cudaMemcpy` inside `prepare`
        // (`Session::mint`, and `Rings::read_cell`). Its width is the SLAB's,
        // which is why `cell_bytes` below is the native one.
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
        // `SymbolicExtent`'s in both.
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
        // ── **THE ADAPTER LANDS HERE, AND NOWHERE ELSE** (alto adapter §6.1,
        //    §6.4). The plan says WHETHER this program carries a `lora` sink;
        //    the cells the guest seeded say what the weights ARE; the store
        //    says WHICH slot they go in. All three are host questions and all
        //    three are asked once, at bind — because §6.1's ruling is that a
        //    channel is a naming device and never a weight transport: a
        //    12 MiB cell materialised into per-lane scratch and dragged over
        //    mapped-pinned PCIe EVERY FIRE is the cost this instant exists to
        //    refuse.
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
        // §3.3). The slot KEEPS its contents — eviction is under pressure and
        // not eager, so an adapter somebody comes back to does not re-pay its
        // H2D — and what the release changes is only that the slot is now
        // reclaimable. A close that skipped this would pin a slot forever and
        // the bank would fill up with adapters nobody is using, which is a
        // refusal that looks like a leak.
        let held = self.adapters.remove(&id);
        let shell = self.loaded_mut()?;
        if let Some(held) = held {
            shell.release_adapter(held);
        }
        shell.close_program_instance(id).map_err(fault)
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

    /// **Move recurrent state between slots** (alto survey §9's gap list,
    /// wave F3) — the verb an RS fork dispatched into nothing until now.
    ///
    /// # The two id spaces, and which one this reads (survey §9, gap 6)
    ///
    /// `StateMove` carries `src_slot_id`/`dst_slot_id`, and the runtime's RS
    /// store has an `RsSlotId` space of its own that covers folded states AND
    /// buffered pages. **This shell reads them as its own SEAT ids** — the
    /// same number `Lane::slot` carries and `Pools::clear` indexes — and does
    /// not translate, because there is nothing here to translate WITH: the
    /// engine has never been told the runtime's mapping, and a shell that
    /// invented one would move the wrong bytes silently. The aliasing is
    /// therefore stated rather than hidden: **a caller's slot id means, to
    /// this engine, the seat that caller's lanes name.** A runtime whose RS
    /// store keeps a second space owes the translation on its own side, which
    /// is where the mapping lives.
    ///
    /// # Whole slots only
    ///
    /// A recurrent bank is a folded summary of a prefix, not an array of
    /// per-token entries — "the first `n` tokens of a slot" names nothing that
    /// exists — so `src_token_offset`, `dst_token_offset` and `token_count`
    /// have no meaning here and a move that states them is refused rather
    /// than rounded off. dev's `copy_slot_d2d` is whole-slot for the same
    /// reason.
    ///
    /// **THE BUFFERED ACTIVATIONS ARE NOT COPIED**, which is also dev's
    /// answer: a fork's buffer is the runtime's to re-derive or to abandon,
    /// and the folded state is the only thing a fork cannot recompute.
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

    /// **Move KV pages inside this device's pools** (alto survey §9's gap
    /// list) — the verb a prefix-tree fork dispatched into nothing until now.
    ///
    /// # One direction, and the other three refused by name
    ///
    /// Device-to-device WITHIN THIS LOAD'S OWN POOLS, which is what a fork,
    /// a graft and a prefix-cache hit are. The other three domain pairs are
    /// refused with the pair spelled out, because each names storage this
    /// shell does not hold:
    ///
    /// ```text
    /// device -> device (this ordinal)   served: cells move inside the arenas
    /// device -> host / host -> device   a pinned swap pool this load does
    ///                                   not reserve (dev's KvSwapPool)
    /// host   -> host                    two buffers neither end of which is
    ///                                   ours; the caller owns that memmove
    /// device -> another ordinal         a peer mapping nothing has opened
    /// ```
    ///
    /// [`Capabilities::kv_copy`] states the same thing ahead of time, so a
    /// caller that reads capabilities never has to discover this by being
    /// refused.
    ///
    /// # The two spellings are one move
    ///
    /// `src_page_ids`/`dst_page_ids` are whole pages and `moves` are single
    /// token cells; both flatten into [`store::Move`] runs, which is a
    /// `(page, token, tokens)` pair per side. **Consecutive cell moves are
    /// COALESCED** — a fork copying a partial page's live tokens states one
    /// `KvMove` per token, and the run they form is one `cudaMemcpyAsync` per
    /// plane rather than one per token per plane. The merge is exact
    /// (identical pages on both sides, both offsets continuing the previous
    /// run by one) and order-preserving, so a caller whose moves do not form
    /// runs gets the same bytes at more copies, never different bytes.
    ///
    /// # The page ids are the CALLER'S, and this shell does not translate
    ///
    /// Article 8: page ids are the runtime's policy and the bytes under them
    /// are the engine's supply. A page id here indexes the same arenas
    /// [`KvDelta::pages`](engine::fire::KvDelta) indexes, which is
    /// what makes "copy the pages, then fire against the copies" mean
    /// anything.
    ///
    /// # Ordering
    ///
    /// Enqueued on the fire stream and NOT synchronized (article 2). The
    /// copies queue behind every step still airborne — which may be reading
    /// the source pages — and in front of every fire submitted after this
    /// returns. See [`Shell::copy_kv`](crate::serve::Shell::copy_kv).
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for a domain pair this shell has no storage
    /// for, [`Error::Invalid`] for a malformed plan — unequal page lists, or
    /// a move whose two ends overlap — and [`Error::Impossible`] for a page
    /// past the pool.
    ///
    /// [`Capabilities::kv_copy`]: engine::caps::Capabilities
    /// [`store::Move`]: crate::store::Move
    fn copy_kv(&mut self, copy: &KvCopy) -> EngineResult<()> {
        copy.validate()?;
        // THE DOMAIN PAIR, BEFORE ANYTHING IS BUILT. `Unsupported` and not
        // `Invalid`: the plan is a plan this contract describes, and what is
        // missing is storage on THIS engine — which is exactly the difference
        // the two variants carry.
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
        // The whole-page half: every token slot of the page, both sides at
        // offset zero. A page's LIVE length is the runtime's bookkeeping and
        // not a number this verb is handed, so the whole page moves — which is
        // also what dev's `copy_d2d_async` did, page by page.
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
        // **OVERLAPPING ENDS ARE THE CALLER'S ERROR AND ARE NAMED HERE.** Both
        // ends of a run live in the same arena, so a run that reads and writes
        // overlapping cells of one page is a device copy with overlapping
        // ends — undefined, and silently so. A caller that means "shift a
        // page's tokens" states a staging page and two moves.
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

    // `encode` takes the trait's default body. See the module header: the
    // shell genuinely carries no multimodal encoder, and a stub that answered
    // `Ok(())` would make an image prompt appear to work.
}

impl Cuda {
    /// One step of an admitted frame, run to completion.
    ///
    /// **BOTH DOORS, CLOSED** (alto design §9, then §1 article 4). What stood
    /// here was a loop asking `program_ready` over every attachment so that a
    /// blocked guest could be answered `Error::Exhausted`; F2a deleted it and
    /// left `serve`'s own gate asking the identical question. Wave E deletes
    /// that one too. Readiness is not a fire-path question at all: the
    /// runtime proves it over the whole frame at `submit`
    /// (`validate_frame`), and past that door a pass that does not commit is
    /// a fault naming the instance and the channel — never a replay.
    fn fire_step(
        &mut self,
        submission: &Step,
        at: engine::StepDone,
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
        // ── **WHICH LANE CARRIES WHICH ADAPTER** (alto adapter §6.4: the plan
        //    says WHETHER, the bind says WHICH). A lane's adapter is the slot
        //    its ATTACHED INSTANCE landed at bind — never a channel this fire
        //    reads, and never a number a guest names. `Lane::adapter` arrives
        //    from the runtime as `None` for every lane (the ETA port
        //    vocabulary has no adapter port, and §3.1 rejected adding one), so
        //    this is where the axis becomes real.
        //
        //    An instance with no adapter contributes nothing, and a fire whose
        //    lanes are all of that kind walks a `BTreeMap` that is empty in
        //    every deployment that never bound one — which is what makes A-5's
        //    "byte- and launch-count-identical" claim hold on this side.
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
                    // The two spaces, both crossing here: `pages` is already
                    // pool ids and this is the table the ports resolved off
                    // the rings still have to go through.
                    translation: &lane.kv.translation,
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
                    // **DERIVED FROM THE BINDING, NOT FROM THE SUBMISSION**
                    // (alto adapter §6.4). `Lane::adapter` in the contract is
                    // still the door a caller driving this engine directly
                    // comes through; what arrives on the runtime's path is
                    // always `None`, and the slot above is the instance's own.
                    // The two are taken in that order — a bind that landed
                    // wins, because it is the one that knows a slot exists.
                    adapter: lane_adapters[at].or(lane.adapter),
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
                    // `alto F3`, closed on this side: the verb and the reset
                    // fact cross unchanged, and the SHELL decides what they
                    // mean. A verb against a plan with no chunked recurrence
                    // is `Fault::Unbound` at the fire, named there rather
                    // than refused for every model here — the mask's rule and
                    // the adapter's, for the sixth time.
                    rs: lane.rs.clone(),
                    rs_reset: lane.rs_reset,
                    // **THE ROW LIST CROSSES TO THE DEVICE HALF TOO**
                    // (`palo B-readout`). `settle_frame` hands the same
                    // `Readout` to `Shell::read_out_rows` for the HOST mirror;
                    // this hands the row indices to the fire, because the
                    // other reader of a fire's logits is a guest epilogue
                    // reading `IntrinsicId::Logits` on the device and the
                    // shell is the only party that knows where a lane's row
                    // run sits in the arena rectangle.
                    //
                    // `Last` and `None` both cross as `None`, which is the
                    // lane's last row: that is the row every epilogue was
                    // given before a list could be stated, and a lane that
                    // asked for no host mirror may still carry an epilogue.
                    readout: match &lane.readout {
                        Readout::Rows(rows) => Some(rows.as_slice()),
                        Readout::Last | Readout::None => None,
                    },
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

        // ── **THE MARSHAL** (media-door §6, wave MD-C). The contract's media
        //    rows in, `serve::Media` borrows out — the same eight fields,
        //    owned there and borrowed here — plus the ONE conversion MD-A
        //    deliberately did not guess.
        //
        //    **A PAYLOAD IS `f32` UNTIL IT MEETS A PLAN.** A front-end
        //    computes real numbers; a plan computes in the element its text
        //    declares, and `RuntimeInput::Patches` is where that is written
        //    down. No party above the load holds a trace, so no party above
        //    the load could have converted — a submission stated in bytes
        //    would have had to guess an element and would have guessed it in
        //    the runtime, for every engine at once. It is converted here,
        //    where `Shell::patch_element` is a value this shell reads off its
        //    own load.
        //
        //    **AND A TEXT-ONLY FIRE PAYS NOTHING FOR THIS.** `submission.media`
        //    is empty for every fire this engine served before the door, so
        //    the two vectors below are never allocated and the `StepView` is
        //    handed the same empty slice it always was.
        let mut staged: Vec<Vec<u8>> = Vec::new();
        if !submission.media.is_empty() {
            // A media submission against a load whose plan states no patch row
            // has no element to convert into, and no tower to convert for. The
            // shell's own refusal, taken at the first instant it is knowable
            // rather than after a rectangle has been sized against a zero.
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
                    // **AND THE IMAGES CROSS** (media-door §6). This line read
                    // `&[]` under a note saying the contract's `Step` had no
                    // media rows, so this door submitted none and every lane
                    // through it was text-only. It has them, and the shell
                    // path behind this seam was complete before they existed.
                    media: &media,
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
        //
        //    **AND `Readout::Rows` IS NOT REFUSED HERE ANY MORE** (`palo
        //    B-readout`, closed). What stood at this line was a loop refusing
        //    every lane that named interior rows, on the true observation that
        //    `Shell::read_out` answered one row per lane. The rectangle was
        //    always addressable after the walk — `slots.0[out]` is it, and the
        //    capture columns are already read at a lane's own row run — so
        //    what was missing was the row LIST, which `PendingStep` was
        //    already carrying to settlement and now hands to
        //    `Shell::read_out_rows`. A row past the lane's own rows is refused
        //    by `Lane::validate_for` at submit, and again by name in the
        //    read-back loop.
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

/// **THE ONE ARITHMETIC THE MEDIA DOOR LEFT** (media-door §6): a payload row's
/// `f32` numbers, in the element the plan computes in, little-endian.
///
/// **ROUND TO NEAREST EVEN, AND STATED RATHER THAN TRUNCATED**, for the reason
/// [`crate::adapter::bf16_bits`] gives where it does the same to an adapter's
/// planes: a truncating conversion would land a slightly different image than
/// the one the front-end computed, and every parity claim about the tower
/// below it would then be about the wrong numbers. It is that same function,
/// called rather than restated.
///
/// # Errors
///
/// The `&'static str` an [`Error::Unsupported`] carries, for a plan whose
/// activation element this marshal cannot write. `Fp8` and the quantized
/// codes are weight elements and no activation is stated in one, so the arm
/// that would encode them is a refusal rather than a guess.
fn patch_bytes(patches: &[f32], element: model_ir::Dtype) -> std::result::Result<Vec<u8>, &'static str> {
    match element {
        model_ir::Dtype::Bf16 => Ok(patches
            .iter()
            .flat_map(|&v| crate::adapter::bf16_bits(v).to_le_bytes())
            .collect()),
        model_ir::Dtype::F32 => Ok(patches.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        _ => Err("a media submission against a plan whose activation element is neither \
                  `bf16` nor `f32`, which is the pair every tower in this catalog computes in"),
    }
}

/// **The verb name a refused `copy_kv` direction is refused under.**
///
/// A `&'static str` because [`Error::Unsupported`] carries one, so the pairs
/// this shell can be asked for are enumerated rather than formatted. That is
/// not a limitation being worked around: a refusal a caller can MATCH ON is
/// worth more than one it can only print, and there are seven domains and one
/// served pair.
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
///
/// **THE ROW COUNT IS THE SHELL'S ANSWER AND NOT THIS FUNCTION'S GUESS.**
/// `Shell::read_out_rows` was handed the same `Readout` list and reports what
/// it actually mirrored in `Settled::rows`; the width is then one division
/// rather than a second reading of the same policy. That is what keeps a
/// three-row readout from being described as one row of triple width by a
/// function that never saw the rectangle.
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
            // The shell mirrored nothing for this lane, so `values` is already
            // empty and `rows` already zero; the arm is written out anyway
            // because the CAPTURE still crosses and the default is what
            // carries it.
            Readout::None => LaneReadout {
                scores,
                ..LaneReadout::default()
            },
            // One row under `Last`, `n` rows in the caller's own order under
            // `Rows(n)`, and the same three fields describe both.
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

// STILL `unsafe impl`, and for the reason `Shell` has always needed one: it
// holds the device's own raw handles inline (a `cublasContext`, CUDA events,
// the arena's `c_void` bases), none of which is `Send` to the compiler, and
// taking C out of the CALL did not change what a CUDA context is. What makes
// it sound is the engine's own rule: every verb takes `&mut self`, so exactly
// one thread touches a shell at a time.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

#[cfg(test)]
mod tests {
    use super::{LATTICE_FLOOR, default_lattice, lattice, patch_ladder};
    use engine::load::Budgets as LoadBudgets;
    use model_compiler::PATCH_LATTICE_FLOOR;
    use model_ir::{Def, Dim, Dtype, RuntimeInput, Trace, Ty, ValueDecl};

    /// A trace holding one runtime input of the stated shape and nothing else
    /// — the ladder reads types, so a type is all a gate needs to hand it.
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
        }
    }

    /// **G4 AT THE ENGINE DOOR.** A plan that states no patch row gets no
    /// ladder, which is the literal `None` this field held before it was
    /// derived and the reason a text-only artifact is bit-identical either
    /// way.
    #[test]
    fn a_text_only_plan_gets_no_ladder_at_all() {
        assert_eq!(
            patch_ladder(&trace_with(vec![Dim::Tokens]), &LoadBudgets::default()),
            None
        );
        // `Lanes` is the token axis too — `Dim::axis` is what decides, not the
        // variant's name.
        assert_eq!(
            patch_ladder(&trace_with(vec![Dim::Lanes]), &LoadBudgets::default()),
            None
        );
    }

    /// A plan that states a patch row serves with ZERO configuration, and the
    /// rungs it gets are whole images from the patch lattice's own floor.
    #[test]
    fn a_tower_plan_derives_a_ladder_from_nothing_but_its_own_declaration() {
        let ladder = patch_ladder(&trace_with(vec![Dim::Patches, Dim::Const(768)]), &LoadBudgets::default())
            .expect("a plan that states patch rows gets a ladder");
        assert_eq!(ladder.max_patches, 4096, "two whole images at the native grid");
        assert_eq!(
            ladder.buckets,
            vec![64, 128, 256, 512, 1024, 2048, 4096],
            "rungs double from the patch lattice's floor to the ceiling"
        );
        assert_eq!(
            ladder.max_images, 4096 / PATCH_LATTICE_FLOOR,
            "as many images as the ceiling holds at the smallest whole image"
        );

        // Every other patch-axis dim reaches the same answer, because the
        // reading is `Dim::axis` and not a list of variants.
        for shape in [vec![Dim::Images], vec![Dim::ImagesPlus(1)]] {
            assert!(patch_ladder(&trace_with(shape), &LoadBudgets::default()).is_some());
        }
    }

    /// **THE TOKEN RECTANGLE BOUNDS THE PATCH ONE**, because every tower row
    /// ends up scattered into a token row — so a deployment that stated a
    /// small `max_tokens` meant it, and the floor still wins under both.
    #[test]
    fn the_derived_ceiling_never_outruns_the_token_rectangle() {
        for (max_tokens, want) in [(512u32, 512u32), (8192, 4096), (65536, 4096), (16, 64)] {
            let budgets = LoadBudgets {
                max_tokens,
                ..LoadBudgets::default()
            };
            let ladder = patch_ladder(&trace_with(vec![Dim::Patches]), &budgets)
                .expect("a tower plan gets a ladder");
            assert_eq!(
                ladder.max_patches, want,
                "at max_tokens = {max_tokens} the patch ceiling should be {want}"
            );
            assert!(
                ladder.buckets.iter().all(|rung| *rung <= ladder.max_patches),
                "a rung past the ceiling is what `model_compiler` refuses by name"
            );
            assert!(
                ladder.buckets.windows(2).all(|pair| pair[0] < pair[1]),
                "the rungs ascend strictly, which is the ladder's own rule"
            );
            assert_eq!(*ladder.buckets.last().expect("a rung"), ladder.max_patches);
        }
    }

    /// An operator who has measured their traffic states the two numbers, and
    /// then the derivation only supplies the rungs.
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

    /// The lattice a `Boot` that stated none is served, spelled out: geometric
    /// above a floor of eight, so that no fire computes more than twice its own
    /// rows and no decode fire lands on a bucket boundary its solo twin missed.
    #[test]
    fn the_default_lattice_is_geometric_above_the_floor() {
        assert_eq!(
            default_lattice(8192),
            vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        );
    }

    /// **THE ARM FLIP AT ONE ROW IS WHAT THE FLOOR EXISTS TO KILL** (the
    /// cuda-abi census: a one-lane decode takes the gemv arm, a two-lane one
    /// does not, and 127 launches change kernel across that boundary). A
    /// lattice naming 1 would leave it exactly where it was.
    #[test]
    fn no_default_lattice_names_a_bucket_that_keeps_the_gemv_arm_alive() {
        for ceiling in [8u32, 16, 64, 256, 8192] {
            assert!(
                default_lattice(ceiling).iter().all(|point| *point > 1),
                "a lattice for {ceiling} rows puts a fire on M=1"
            );
        }
    }

    /// Two properties P0 refuses a lattice for (`model_compiler`'s `accept`):
    /// it must strictly ascend, and no point may pass the token ceiling. A
    /// default that could not be baked would turn every unstated lattice into
    /// `Fault::Bake`.
    #[test]
    fn the_default_lattice_ascends_and_stops_at_the_ceiling() {
        for ceiling in [1u32, 2, 3, 4, 63, 64, 65, 256, 511, 8192] {
            let lattice = default_lattice(ceiling);
            assert_eq!(
                *lattice.last().expect("a lattice is never empty"),
                ceiling,
                "a fire AT the ceiling must have a bucket"
            );
            assert!(
                lattice.windows(2).all(|pair| pair[0] < pair[1]),
                "{lattice:?} does not strictly ascend"
            );
            assert!(
                lattice.iter().all(|point| *point <= ceiling),
                "{lattice:?} names a bucket past the token ceiling"
            );
        }
    }

    /// The waste D4 pays is bounded by the lattice's ratio, and a geometric
    /// lattice is what makes that a sentence with a number in it: no fire ever
    /// computes more than twice the rows it has.
    #[test]
    fn no_fire_above_the_floor_is_padded_past_twice_its_own_rows() {
        let lattice = default_lattice(8192);
        for rows in LATTICE_FLOOR..=8192 {
            let bucket = lattice
                .iter()
                .copied()
                .find(|point| *point >= rows)
                .expect("every row count up to the ceiling has a bucket");
            assert!(
                u64::from(bucket) < 2 * u64::from(rows),
                "a fire of {rows} rows pads to {bucket}"
            );
        }
    }

    /// A stated lattice is kept, and an unstated one is filled — the whole of
    /// [`lattice`], and the property that made `PIE_CUDA_BUCKETS` unnecessary.
    #[test]
    fn a_stated_lattice_is_kept_and_an_unstated_one_is_filled() {
        assert_eq!(lattice(vec![4, 9, 33], 64), vec![4, 9, 33], "stated wins");
        assert_eq!(lattice(Vec::new(), 64), default_lattice(64));
    }
}
