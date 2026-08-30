//! The menlo Metal engine's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `kernels-metal`
//! (design §8, decisions #13–#16).
//!
//! **Lineage.** This crate took `engine-metal`'s name when the string-plan
//! shell it re-imagines was deleted with the rest of the old stack (design,
//! porting order step 6); serving plumbing (bind, device, serve) rejoins it
//! as the fabric is rewired.
//!
//! **Portability.** No Metal API is named here. The `Run` encodes through
//! [`kernels_metal::Ctx`] (`dyn Encode`), which the real shell
//! implements against its command buffers — so this crate, like the walk
//! that drives it (`model_exec::fire::walk`), builds and tests on any OS; only
//! the shell behind the sink is macOS-bound.
//!
//! **The seam** — the engine side of the `MENLO-SEAM` markers in
//! `kernels_metal::attn`. The ledger is shorter than it was: the appenders'
//! write tables are op-named now (`write_page`/`write_offset` ride the
//! `kv_append` ops as inputs), so that entry is closed and gone. A `Run`
//! still binds more than the ops name:
//!
//! - the fire's shared **indptr**: `qo_indptr` is no longer a runtime input
//!   (design §5), so ragged views for the boundary-aware entries are
//!   assembled here, from [`FireBindings::indptr`];
//! - the **fire tables** (`positions`, `request_of_token`, `mask`,
//!   `mask_enabled`, `mask_stride`): the sdpa plan builders read them, yet
//!   `attention.plan_*` names kv geometry those builders never touch — the
//!   plan-building arms bridge the two, binding the tables from
//!   [`FireBindings::tables`]. `mask` is half-closed: `attention.masked`
//!   names it and the resolver routes `RuntimeInput::Mask` onto the same
//!   seat, but the causal launches still read it plan-carried, beside the
//!   `mask_enabled`/`mask_stride` no op names.

pub mod abi;
pub mod api;
pub mod arena;
/// Reading a boot document — this crate's half of it, in this crate.
pub mod boot;
pub mod device;
mod dispatch;
pub mod encode;
mod error;
pub mod experts;
/// The indirect command buffer — Metal-only, and the one module of this
/// crate that is.
///
/// Everything else here compiles on any target (the standing doctrine: the
/// shell type-checks without an Apple SDK), and the recorder and the derived
/// `DescriptorAbi` above are portable BECAUSE they are about the walk rather
/// than about the device. An `MTLIndirectCommandBuffer` is not: there is no
/// portable half of it worth writing, and a refusing twin would be a type
/// with no callers off Apple.
#[cfg(target_vendor = "apple")]
pub mod icb;
pub mod rebind;
pub mod inputs;
pub mod mask;
pub mod program;
pub mod record;
pub mod run;
/// The observability slab the attention capture arm writes and an
/// epilogue's `attn_score` intrinsic reads (`.wiki/alto/attn-score.md` §4).
pub mod scores;
pub mod scratch;
pub mod serve;
/// The settlement plane: the run-ahead counters, the completion seam and
/// the A/B seat ring the asynchronous fire path is built on.
pub mod settle;
pub mod store;
pub mod weights;
pub mod window;

pub use abi::{Armed, At as AbiAt, Axis, DescriptorAbi, Law, SlotAbi, Survey};
pub use api::{ContractFor, DeviceBoot, Metal};
pub use boot::open;
pub use arena::Arena;
pub use device::{Buffer, Context, Handles, Pipelines};
pub use encode::Sink;
pub use error::{Fault, Result};
pub use experts::{GroupResidency, Plan as ResidencyPlan};
#[cfg(target_vendor = "apple")]
pub use icb::{Icb, Rebound};
pub use inputs::Inputs;
pub use program::{Fired, Launched, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{Arg, Point, Recording, Slot, Tape};
pub use run::{
    CacheGeometry, CachePool, CacheTable, FireBindings, FireTables, Run, SlotTable, StructSlot,
    WeightRow, WeightTable,
};
pub use scores::ScoreSeat;
pub use scratch::Scratch;
pub use serve::{
    Attached, Boot, Enqueued, FireCost, Landed, Lane, Prepared, Seated, Shell, StepView,
};
pub use settle::{Airborne, Arms, Done};
pub use store::Pools;
pub use weights::{AdapterPlane, Weights};
pub use window::{Copies, Cursor, Gathered, GatheredSpace, Window, Windows};
