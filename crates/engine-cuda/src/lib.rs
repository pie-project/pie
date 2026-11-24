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
pub mod check;
pub mod checkpoint_serving;
pub mod chip;
pub mod comm;
pub mod device;
mod dispatch;
pub mod dissect;
pub mod ep;
mod error;
pub mod experts;
pub mod exports;
pub mod group;
#[cfg(not(feature = "rocm"))]
pub mod grouped_decode;
pub mod inputs;
pub mod lockstep;
pub mod mask;
pub mod narrow;
pub mod program;
pub mod rates;
pub mod record;
pub mod replay;
pub mod run;
pub mod scores;
pub mod serve;
pub mod settle;
pub mod sizing;
#[cfg(feature = "rocm")]
pub mod split_arm;
pub mod staged_h2d;
pub(crate) mod landing_ring;
pub mod store;
pub mod stored;
pub mod synth;
pub mod tap;
pub mod top;
pub mod tune;
pub mod voxels;
pub mod weights;
pub mod window;

pub const EXCLUSIVE: [&str; 0] = [];

pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fusions this backend implements: the compiler runs these rules alone,
/// so each forms only fused ops this backend's dispatch fires. A rule it does
/// not hold (`matmul_swiglu_matmul`, which Metal splits across engines)
/// leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules;
        vec![
            rules::qkv_fused_padded(),
            rules::residual_add_rmsnorm(),
            rules::moe_mega(),
            rules::moe_experts(),
            rules::moe_select_glu(),
            rules::matmul_moe_route(),
            rules::matmul_split_rows(),
            rules::norm_rope_last_append_rows(),
            rules::lse_sink(),
            rules::pool_attention(),
            rules::pool_compress(),
            rules::mla_prefill_expanded(),
            rules::norm_rope_pair(),
            rules::index_topk_rope(),
            rules::hc_prenorm_project(),
            rules::hc_gates_norm(),
            rules::res_blend_rmsnorm(),
            rules::hc_fold_pre(),
            rules::hc_pre(),
        ]
    });
#[cfg(feature = "rocm")]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules::{self, Rotated};
        use poem_ir::GluAct;
        use poem_ir::ops::fused::Intermediates;
        vec![
            rules::qkv_fused_qknorm_rope_vnorm_write(),
            rules::rmsnorm_residual_add(),
            rules::residual_add_rmsnorm_matmul(),
            rules::residual_add_rmsnorm_matmul_swiglu_matmul(),
            rules::embed_scale_add_select(),
            rules::embed_scale_add(),
            rules::residual_add_rmsnorm(),
            rules::matmul_glu(&[GluAct::GegluTanh]),
            rules::matmul_softcap(),
            rules::matmul_bias(),
            rules::matmul_swiglu_matmul(),
            rules::moe_experts(),
            rules::gated_residual_norm_modulate(),
            rules::norm_modulate(),
            rules::norm_rope(Rotated::First),
            rules::hc_norm_project(),
            rules::gdn_step(),
            rules::q_gate_norm_rope_append(Intermediates::Written),
        ]
    });

/// The hand rules CUDA does not hold: every site one forms folds into a
/// generated row kernel or an anchored region instead.
#[cfg(not(feature = "rocm"))]
#[must_use]
pub fn retired() -> Vec<poem_compiler::fuse::Rule> {
    use poem_compiler::fuse::rules;
    vec![
        rules::norm_rope(rules::Rotated::Last),
        rules::norm_rope(rules::Rotated::First),
        rules::matmul_glu(&[poem_ir::GluAct::GegluTanh, poem_ir::GluAct::Swiglu]),
        rules::matmul_softcap(),
        rules::matmul_bias(),
        rules::qkv_fused_qknorm_rope_vnorm_write(),
        rules::norm_rope_last_append(),
        rules::rope_yarn_append(),
        rules::split_rope_yarn_append(),
        rules::q_gate_norm_rope_append(poem_ir::ops::fused::Intermediates::Unwritten),
        rules::embed_scale_add_select(),
        rules::embed_scale_add(),
        rules::norm_modulate(),
        rules::gated_residual_norm_modulate(),
    ]
}

#[cfg(not(feature = "rocm"))]
const RESIDUAL_NORM: &str = "fused.residual_add_rmsnorm";

#[cfg(not(feature = "rocm"))]
#[must_use]
pub fn fusions(chip: &profile::Profile) -> Vec<&'static poem_compiler::fuse::Rule> {
    poem_compiler::fuse::flags::on(chip, &FUSIONS)
        .into_iter()
        .filter(|rule| rule.kernel() != RESIDUAL_NORM)
        .collect()
}

#[cfg(not(feature = "rocm"))]
#[must_use]
pub fn residual_norms(
    chip: &profile::Profile,
    synth_regions: bool,
) -> Vec<&'static poem_compiler::fuse::Rule> {
    poem_compiler::fuse::flags::on(chip, &FUSIONS)
        .into_iter()
        .filter(|rule| !synth_regions && rule.kernel() == RESIDUAL_NORM)
        .collect()
}

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
    Armed, Boot, Diagnostics, EveryRung, Golden, Graphs, Knobs, Lane, Media, QueueMasks, Recording, Seal,
    Seated, Shell, WeightPin,
};

pub use blob::{
    Adapters, Binding, Site as AdapterSite, Source as AdapterSource, layer_of, role_of, site_of,
};
pub use weights::{AdapterPlane, BankSeat};
pub use window::{Cursor, Window, WindowShape, Windows};
