#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("`cuda` and `rocm` are each this engine's device runtime; a build carries one");

// The engine names its kernels `kernels_cuda`; on ROCm they are kernels-rocm's.
#[cfg(feature = "rocm")]
extern crate kernels_rocm as kernels_cuda;

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
#[cfg(not(feature = "rocm"))]
pub mod grouped_decode;
pub mod inputs;
pub mod mask;
pub mod narrow;
pub mod program;
pub mod record;
pub mod replay;
pub mod rotate;
pub mod run;
pub mod scores;
pub mod serve;
pub mod settle;
#[cfg(feature = "rocm")]
pub mod split_arm;
pub mod staged_h2d;
pub mod store;
pub mod tune;
pub mod voxels;
pub mod weights;
pub mod window;

pub const EXCLUSIVE: [&str; 0] = [];

pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fused kernels this backend ships; the compiler forms only these. A
/// rule it does not list (`linear.mlp_swiglu`, which Metal splits across
/// engines) leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub const FUSED: [&str; 30] = [
    "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
    "elementwise.rmsnorm_residual_add",
    "elementwise.embed_scale_add_select",
    "elementwise.embed_scale_add",
    "elementwise.residual_add_rmsnorm",
    "linear.matmul_geglu",
    "linear.moe_experts",
    "linear.moe_select_glu",
    "linear.matmul_swiglu",
    "linear.lm_head_softcap",
    "linear.matmul_bias",
    "linear.matmul_split_rows",
    "elementwise.gated_residual_norm_modulate",
    "elementwise.norm_modulate",
    "elementwise.rmsnorm_rope_partial_q",
    "elementwise.norm_rope_last",
    "elementwise.norm_rope_last_append",
    "elementwise.split_rope_yarn_append",
    "elementwise.rope_yarn_append",
    "elementwise.qg_qknorm_rope_append",
    "attention.lse_sink",
    "attention.pool_attention",
    "attention.pool_compress",
    "attention.mla_prefill_expanded",
    "elementwise.norm_rope_pair",
    "attention.index_topk_rope",
    "elementwise.hc_prenorm_project",
    "elementwise.hc_gates_norm",
    "elementwise.res_blend_rmsnorm",
    "elementwise.hc_pre",
];
#[cfg(feature = "rocm")]
pub const FUSED: [&str; 16] = [
    "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
    "elementwise.rmsnorm_residual_add",
    "elementwise.embed_scale_add_select",
    "elementwise.embed_scale_add",
    "elementwise.residual_add_rmsnorm",
    "linear.matmul_geglu",
    "linear.lm_head_softcap",
    "linear.matmul_bias",
    "linear.mlp_swiglu",
    "linear.moe_experts",
    "elementwise.gated_residual_norm_modulate",
    "elementwise.norm_modulate",
    "elementwise.rmsnorm_rope_partial_q",
    "elementwise.hc_norm_project",
    "attention.ssm_gdn_step",
    "attention.q_gate_k_norm_rope_append",
];

/// The fused kernels left on with `fuse-chains=off`.
#[cfg(not(feature = "rocm"))]
pub const UNCHAINED: [&str; 2] = [
    "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
    "elementwise.residual_add_rmsnorm",
];
#[cfg(feature = "rocm")]
pub const UNCHAINED: [&str; 1] = ["elementwise.residual_add_rmsnorm"];

#[must_use]
pub fn shifted(op: &str) -> bool {
    matches!(
        kernels_cuda::seat::reads(op),
        Reads::Rows | Reads::RowsAndLanes
    )
}

pub const PLANNED: [&str; 2] = ["attention.plan_decode", "attention.plan_prefill"];

#[must_use]
pub fn lane_shifted(op: &str) -> bool {
    kernels_cuda::seat::reads(op) == Reads::RowsAndLanes
}

pub use api::{ContractFor, Cuda, DeviceBoot, World};
pub use boot::{open, ordinal_of};
pub use error::{Fault, Result};
pub use group::{Group, open_group};
pub use kernels_cuda::{EntryInfo, Reads};
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
