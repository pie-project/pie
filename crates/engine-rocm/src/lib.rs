pub mod adapter;
pub mod api;
pub mod arena;
pub mod blob;
pub mod boot;
pub mod checkpoint_serving;
pub mod chip;
pub mod comm;
pub mod device;
mod dispatch;
mod error;
pub mod experts;
pub mod exports;
pub mod group;
pub mod inputs;
pub mod mask;
pub mod program;
pub mod record;
pub mod rotate;
pub mod run;
pub mod scores;
pub mod serve;
pub mod settle;
pub mod staged_h2d;
pub mod store;
pub mod tune;
pub mod voxels;
pub mod weights;
pub mod window;

pub const EXCLUSIVE: [&str; 0] = [];

pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fused kernels this backend ships; the compiler forms only these. The
/// CUDA engine's other fusions have no AMD kernel yet, so their primitives
/// stay apart here.
pub const FUSED: [&str; 2] = ["elementwise.residual_add_rmsnorm", "linear.mlp_swiglu"];

/// The fused kernels left on with `fuse-chains=off`.
pub const UNCHAINED: [&str; 1] = ["elementwise.residual_add_rmsnorm"];

#[must_use]
pub fn shifted(op: &str) -> bool {
    matches!(
        kernels_rocm::seat::reads(op),
        Reads::Rows | Reads::RowsAndLanes
    )
}

pub const PLANNED: [&str; 2] = ["attention.plan_decode", "attention.plan_prefill"];

#[must_use]
pub fn lane_shifted(op: &str) -> bool {
    kernels_rocm::seat::reads(op) == Reads::RowsAndLanes
}

pub use api::{ContractFor, Cuda, DeviceBoot, World};
pub use boot::{open, ordinal_of};
pub use error::{Fault, Result};
pub use group::{Group, open_group};
pub use kernels_rocm::{EntryInfo, Reads};
pub use mask::{LaneMask, Staged as StagedMask};
pub use program::{Fired, Plane as ProgramPlane, Session as ProgramSession};
pub use record::{
    AxisKey, Bodies as GraphCache, BodyCensus, BodyKey, BodyStats, BodyTally, LastCapture,
};
pub use run::{
    CacheGeometry, CachePlanning, CachePool, CacheTable, FireBindings, FireTables, Planning,
    PoolSlabs, Run, SlotTable, StructSlot, WeightRow, WeightTable,
};
pub use serve::{
    Armed, Boot, DEFAULT_BODIES_MEGABYTES, Diagnostics, FireCost, Golden, Graphs, Knobs, Lane,
    Media, Recording, Seal, Seated, Shell,
};

pub use blob::{
    Adapters, Binding, Site as AdapterSite, Source as AdapterSource, layer_of, role_of, site_of,
};
pub use engine::fire::LayerScores;
pub use weights::{AdapterPlane, BankSeat};
pub use window::{Cursor, Window, WindowShape, Windows};
