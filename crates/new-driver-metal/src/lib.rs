//! The menlo Metal driver's dispatch layer: the [`Run`] that resolves plan
//! ids to device handles, and its `impl Dispatch*` set — every op family
//! answered by destructure → resolve → call into `new-kernels-metal`
//! (design §8, decisions #13–#16).
//!
//! **Successor note.** This crate is the menlo rewrite of `driver-metal`'s
//! execution layer — the resolving and dispatching business its `walk/`,
//! `bind/`, and `baker/` did for the string-plan stack. It merges into
//! `driver-metal` when the old stack is deleted in one commit (design,
//! porting order step 6).
//!
//! **Portability.** No Metal API is named here. The `Run` encodes through
//! [`new_kernels_metal::Ctx`] (`dyn Encode`), which the real shell
//! implements against its command buffers — so this crate, like the old
//! walk, builds and tests on any OS; only the shell behind the sink is
//! macOS-bound.
//!
//! **The seam** — the driver side of the `MENLO-SEAM` markers in
//! `new_kernels_metal::attn`. A `Run` binds more than the ops name:
//!
//! - the fire's shared **indptr**: `qo_indptr` is no longer a runtime input
//!   (design §5), so ragged views for the boundary-aware entries are
//!   assembled here, from [`FireBindings::indptr`];
//! - the **fire tables** (`positions`, `request_of_token`, `mask`,
//!   `mask_enabled`, `mask_stride`): the sdpa plan builders read them, yet
//!   `attention.plan_*` names kv geometry those builders never touch — the
//!   plan-building arms bridge the two, binding the tables from
//!   [`FireBindings::tables`];
//! - the pool rows' **write_page/write_offset**: the appenders address by
//!   them while the IR states `kv_indices`/`positions`; the shell derives
//!   those tables from the same inputs when it builds the [`CacheTable`].

mod dispatch;
pub mod run;

pub use run::{
    CacheGeometry, CachePool, CacheTable, FireBindings, FireTables, Run, SlotTable, StructSlot,
    WeightTable,
};

/// The walk is `new-driver`'s, written once and generic over any `Dispatch`;
/// re-exported so a shell driving this `Run` needs one crate in scope.
pub use new_driver::{Phases, fire, phases, walk};
