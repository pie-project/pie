//! The runtime↔engine contract: what an engine *is*, in types.
//!
//! ```text
//! engine.rs    trait Engine — load, register_program, register_channel,
//!              bind_instance, close_*, submit, copy_kv, copy_state,
//!              resize_pool, encode
//! error.rs     Error                       ← the PIE_STATUS_* graveyard
//! load.rs      LoadRequest { plan, checkpoint, budgets } -> Loaded { facts, caps }
//! fire.rs      FrameSubmission { steps: Vec<Step { lanes, attachments }> }
//!              -> FrameTicket                    ← the execution plane's unit
//! program.rs   the LaunchPackage lineage, purified
//! channel.rs   the typed channel declaration
//! adapter.rs   AdapterRegistration — the correction class's residency verb
//! transfer.rs  KvHandle / KvLayout / the three movement verbs' arguments
//! caps.rs      DeviceFacts, Capabilities
//! ```
//!
//! # What this crate was
//!
//! A C header's ghost. There is no C on either side of this boundary any more
//! — every engine in the workspace is Rust, linked into the same process, and
//! called through a `&mut dyn Engine` — but the crate kept the shape of one
//! for years after the C++ went:
//!
//! * an `i32` status ladder (`PIE_STATUS_OK` … `PIE_STATUS_IMPOSSIBLE`), so a
//!   caller needed a table to read a failure and a second, unrelated string to
//!   learn what it was about;
//! * `PIE_DRIVER_ABI_VERSION: u32 = 25`, stamped into every capability record
//!   and checked on every load — an ABI version on a call between two crates
//!   Cargo compiles together;
//! * `type DeviceDomain = u32` with seven `PIE_MEMORY_DOMAIN_*` constants and
//!   a `pie_memory_domain_is_valid` predicate, sitting three files away from
//!   `enum MemoryDomain`, which is the same axis and cannot be invalid;
//! * forty-odd `u8` tag constants re-spelling `tensor-ir`'s own vocabulary
//!   (dtypes, host roles, extern directions, readiness, stages, ports) with
//!   nothing checking that the two numberings agreed;
//! * thirteen `PIE_DEVICE_PORT_*` bits in a private numbering that disagreed
//!   with the port registry's;
//! * and an 807-line completion broker — a waker table, a recycling pool of
//!   `#[repr(C)]` atomic terminal cells, per-work-item leases — living inside
//!   the description of what an engine is.
//!
//! # What it is now
//!
//! The verb set survived; the encoding did not. The five decisions the rewrite
//! executed (palo design §7, decisions 18–20):
//!
//! 1. **`model-ir` is a dependency.** The runtime traces, `Trace` crosses at
//!    load, `CompiledModel` never crosses, and `model_ir::Dtype` is *the* dtype — the
//!    `KvDtype` that spelled five of its variants a second time is gone.
//! 2. **The completion broker went to the runtime.** Run-ahead is a scheduling
//!    policy, and this crate keeps only the receipt
//!    ([`fire::FireTicket`]).
//! 3. **The ports went to `tensor-ir`.** `PIE_DEVICE_PORT_*` and
//!    `GeometryClass` live in `tensor_ir::registry` beside the `Port` enum
//!    they were a second numbering of. This crate names them; it does not
//!    re-export them.
//! 4. **Every noun is serde and the trait is object-safe.** There is no wire
//!    version here: remote is a property of a transport, not an encoding of a
//!    contract.
//! 5. **The dead `LaunchPlan`-era types are gone**, and the ones with living
//!    consumers got typed successors.
//!
//! # The dependency floor
//!
//! `model-ir`, `tensor-ir`, `serde`, `thiserror`. All four are leaves. What
//! this crate no longer drags into the graph of everyone who reads a
//! `KvHandle`: `tarpc` (and tokio, and its wasm/windows platform closure),
//! `waker` (and `loom`, and a C compiler), `anyhow`, `crossbeam-queue`. Both
//! of the crate's Cargo features existed to hold those back, and both are
//! gone with them.

#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]
#![forbid(unsafe_code)]

// Re-exported so a consumer that only reads a `KvLayout` does not have to
// declare a `model-ir` dependency of its own to name the dtype inside it, and
// so `tensor_ir::registry`'s port vocabulary is reachable from the contract
// without this crate re-exporting the ports themselves (decision 19).
pub use model_ir;
pub use tensor_ir;

pub mod adapter;
pub mod caps;
pub mod channel;
pub mod engine;
pub mod error;
pub mod fire;
pub mod load;
pub mod program;
pub mod transfer;

pub use adapter::{AdapterPlane, AdapterRegistration};
pub use caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
pub use channel::{
    ChannelId, ChannelRegistration, ChannelSeed, HostMirror, RegisteredChannel, Ticket,
};
pub use engine::{CompletionSink, Engine, StepDone, StepOutcome};
pub use error::{Error, Result};
pub use fire::{
    Attachment, Boundary, FireId, FireTicket, FoldLen, FrameId, FrameSubmission, FrameTicket,
    KvDelta, Lane, LaneReadout, LayerScores, Mask, MediaEncode, Readout, RsVerb, Step,
};
pub use load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded};
pub use program::{
    Axis, BindExtents, BoundInstance, DirectArgmax, EmittedKernel, ExtentRole, InstanceBinding,
    InstanceId,
    KernelKind, LaunchChannel, LaunchChannelRule, LaunchOp, LaunchPackage, LaunchPlanValue,
    LaunchPort, LaunchPut, LaunchRegion, LaunchStage, LaunchStagePlan, LaunchValue, LibraryOp,
    ProgramId, ProgramRegistration, RegionAnalysis, RegionKind, StageNeeds, ValueSource,
};
pub use transfer::{
    KvCopy, KvExport, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain, PageRange,
    Pool, PoolResize, StateCopy, StateMove,
};
