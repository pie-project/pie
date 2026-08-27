//! The menlo Metal driver's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `kernels-metal`
//! (design §8, decisions #13–#16).
//!
//! **Lineage.** This crate took `driver-metal`'s name when the string-plan
//! shell it re-imagines was deleted with the rest of the old stack (design,
//! porting order step 6); serving plumbing (bind, device, serve) rejoins it
//! as the fabric is rewired.
//!
//! **Portability.** No Metal API is named here. The `Run` encodes through
//! [`kernels_metal::Ctx`] (`dyn Encode`), which the real shell
//! implements against its command buffers — so this crate, like the walk
//! that drives it (`driver::fire::walk`), builds and tests on any OS; only
//! the shell behind the sink is macOS-bound.
//!
//! **The seam** — the driver side of the `MENLO-SEAM` markers in
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

pub mod api;
pub mod arena;
pub mod device;
mod dispatch;
pub mod encode;
mod error;
pub mod inputs;
pub mod mask;
pub mod program;
pub mod run;
pub mod serve;
pub mod store;
pub mod weights;
pub mod window;

pub use api::{ContractFor, DeviceBoot, Metal};
pub use arena::Arena;
pub use device::{Buffer, Context, Handles, Pipelines};
pub use encode::Sink;
pub use error::{Fault, Result};
pub use inputs::Inputs;
pub use program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
pub use run::{
    CacheGeometry, CachePool, CacheTable, FireBindings, FireTables, Run, SlotTable, StructSlot,
    WeightRow, WeightTable,
};
pub use serve::{Boot, Lane, Seated, Shell};
pub use store::Pools;
pub use weights::Weights;
pub use window::{Cursor, Window, Windows};
