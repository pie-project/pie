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
pub mod tap;
pub mod tune;
pub mod voxels;
pub mod weights;
pub mod window;

pub const EXCLUSIVE: [&str; 0] = [];

pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fused kernels this backend ships; the compiler forms only these. A
/// rule it does not list (`fused.matmul_swiglu_matmul`, which Metal splits across
/// engines) leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub const FUSED: [&str; 31] = [
    "fused.split_qkv_fused",
    "fused.qkv_fused_qknorm_rope_vnorm_write",
    "fused.rmsnorm_residual_add",
    "fused.embed_scale_add_select",
    "fused.embed_scale_add",
    "fused.residual_add_rmsnorm",
    "fused.matmul_geglu",
    "fused.moe_experts",
    "fused.moe_select_glu",
    "fused.matmul_swiglu",
    "fused.matmul_softcap",
    "fused.matmul_bias",
    "fused.matmul_split_rows",
    "fused.gated_residual_norm_modulate",
    "fused.norm_modulate",
    "fused.rmsnorm_rope_partial_q",
    "fused.norm_rope_last",
    "fused.norm_rope_last_append",
    "fused.split_rope_yarn_append",
    "fused.rope_yarn_append",
    "fused.qg_qknorm_rope_append",
    "fused.lse_sink",
    "fused.pool_attention",
    "fused.pool_compress",
    "fused.mla_prefill_expanded",
    "fused.norm_rope_pair",
    "fused.index_topk_rope",
    "fused.hc_prenorm_project",
    "fused.hc_gates_norm",
    "fused.res_blend_rmsnorm",
    "fused.hc_pre",
];
#[cfg(feature = "rocm")]
pub const FUSED: [&str; 16] = [
    "fused.qkv_fused_qknorm_rope_vnorm_write",
    "fused.rmsnorm_residual_add",
    "fused.embed_scale_add_select",
    "fused.embed_scale_add",
    "fused.residual_add_rmsnorm",
    "fused.matmul_geglu",
    "fused.matmul_softcap",
    "fused.matmul_bias",
    "fused.matmul_swiglu_matmul",
    "fused.moe_experts",
    "fused.gated_residual_norm_modulate",
    "fused.norm_modulate",
    "fused.rmsnorm_rope_partial_q",
    "fused.hc_norm_project",
    "fused.gdn_step",
    "fused.q_gate_norm_rope_append",
];

/// The fused kernels left on with `fuse-chains=off`.
#[cfg(not(feature = "rocm"))]
pub const UNCHAINED: [&str; 2] = [
    "fused.qkv_fused_qknorm_rope_vnorm_write",
    "fused.residual_add_rmsnorm",
];
#[cfg(feature = "rocm")]
pub const UNCHAINED: [&str; 1] = ["fused.residual_add_rmsnorm"];

/// What `op` reads of a seat's rows: the seat table's entry for its name,
/// but a gated unit of routed rows (each times its route's weight) reads
/// routed rows, none of the seat's, a predicted route only the prediction's,
/// an expert product that quantizes its activations its own staging, and a
/// combine that adds a shared row none of the seat's.
fn reads(op: &poem_ir::Operation) -> Reads {
    match op {
        poem_ir::Operation::Linear(
            poem_ir::Linear::Glu {
                weights: Some(_), ..
            }
            | poem_ir::Linear::MoeRoute {
                score: poem_ir::RouteScore::Predict,
                ..
            }
            | poem_ir::Linear::MoeSelect { quant: Some(_), .. }
            | poem_ir::Linear::MoeCombine {
                shared: Some(_), ..
            },
        ) => Reads::Nothing,
        _ => kernels_cuda::seat::reads(poem_ir::Operands::name(op)),
    }
}

#[must_use]
pub fn shifted(op: &poem_ir::Operation) -> bool {
    matches!(reads(op), Reads::Rows | Reads::RowsAndLanes)
}

pub const PLANNED: [&str; 2] = ["attn.plan_decode", "attn.plan_prefill"];

#[must_use]
pub fn lane_shifted(op: &poem_ir::Operation) -> bool {
    reads(op) == Reads::RowsAndLanes
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
