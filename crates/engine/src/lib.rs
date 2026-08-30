//! The runtime↔engine contract: what an engine *is*, in types.
//!
//! ```text
//! engine.rs    trait Engine — load, register_program, register_channel,
//!              bind_instance, close_*, submit, copy_kv, copy_state,
//!              encode
//! error.rs     Error                       ← the PIE_STATUS_* graveyard
//! load.rs      LoadRequest { plan, checkpoint, budgets, residency } -> Loaded { facts, caps }
//! fire.rs      FrameSubmission { steps: Vec<Step { lanes, attachments }> }
//!              -> FrameTicket                    ← the execution plane's unit
//! frame.rs     Prepared / Enqueued / Shell — the typed prepare/enqueue/
//!              settle seam every shell steps through
//! runahead.rs  Runahead — every run-ahead depth, derived from one number
//! program.rs   ProgramRegistration / InstanceBinding / BoundInstance /
//!              BindExtents — what a caller states and the engine answers.
//!              The LaunchPackage lineage it used to hold is `eta-compiler`'s
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
//! * forty-odd `u8` tag constants re-spelling `eta-ir`'s own vocabulary
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
//! 3. **The ports went to `eta-ir`.** `PIE_DEVICE_PORT_*` and
//!    `GeometryClass` live in `eta_ir::registry` beside the `Port` enum
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
//! `model-ir`, `eta-ir`, `eta-compiler`, `serde`, `thiserror`.
//!
//! **Four of the five are leaves; `eta-compiler` is not, and that is the
//! honest version of a sentence this header used to get away with.** The
//! launch package is the compiler's output artifact, and it was declared here
//! only because this crate was the one both sides could name — which forced
//! the producer to depend on the contract to describe its own output, and left
//! five types (`LibraryOp`, `RegionKind`, `KernelKind`, `EmittedKernel`,
//! `RegionAnalysis`) declared once on each side with conversions bridging
//! them. [`program`] names `eta_compiler`'s now, and `eta-compiler` names
//! nothing here.
//!
//! What that costs a reader's graph is `eta-compiler` and nothing else.
//! Measured feature-resolved, `eta-compiler`'s own closure after losing its
//! edge to this crate is `{eta-ir, serde}` — a strict subset of what a
//! `KvHandle` reader already carried — so the edge is added and no crate is.
//!
//! What this crate still does not drag into that graph: `tarpc` (and tokio,
//! and its wasm/windows platform closure), `waker` (and `loom`, and a C
//! compiler), `anyhow`, `crossbeam-queue`. Both of the crate's Cargo features
//! existed to hold those back, and both are gone with them.

#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]
#![forbid(unsafe_code)]

// `pub use model_ir; pub use eta_ir; pub use eta_compiler;` STOOD HERE, and
// the argument for them was that a consumer naming a `KvLayout`'s dtype or a
// `ProgramRegistration`'s package should not have to declare an edge of its
// own to spell the type inside it.
//
// What the argument cost, measured when the substrate's own pass-through
// re-export came out: twenty-nine call sites reaching a leaf through this
// crate — `engine::model_ir::Platform`, `engine::eta_ir::registry::Stage` —
// and half of them in `runtime`, which DECLARES `eta-ir` already and was
// paying the extra hop for nothing. The saving was never real either: a
// consumer that names a type is a consumer of the crate that owns it, and
// writing that down is one manifest line, not a dependency it did not have.
//
// So the leaves are named directly now, by everyone. `transport` came out
// better than the rest — it wanted an element type and not an IR, so it
// declares `dtype` and this crate is no longer between it and the enum.

pub mod adapter;
pub mod caps;
pub mod channel;
pub mod engine;
pub mod error;
pub mod fire;
pub mod frame;
pub mod load;
pub mod program;
pub mod runahead;
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
    KvDelta, Lane, LaneReadout, LayerScores, Mask, Masking, MediaEncode, Readout, RsReset, RsVerb,
    Serves, Step,
};
pub use load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded, Residency};
pub use program::{
    BindExtents, BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration,
};
pub use transfer::{
    KvCopy, KvExport, KvHandle, KvLayout, KvLayoutKind, KvMove, KvRegion, MemoryDomain, StateCopy,
    StateMove,
};
