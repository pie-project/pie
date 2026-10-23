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

/// The fusions this backend implements: the compiler runs these rules alone,
/// so each forms only fused ops this backend's dispatch fires. A rule it does
/// not hold (`matmul_swiglu_matmul`, which Metal splits across engines)
/// leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules;
        vec![
            rules::split_qkv_fused(),
            rules::qkv_fused_qknorm_rope_vnorm_write(),
            rules::rmsnorm_residual_add(),
            rules::embed_scale_add_select(),
            rules::embed_scale_add(),
            rules::residual_add_rmsnorm(),
            rules::matmul_geglu(),
            rules::moe_experts(),
            rules::moe_select_glu(),
            rules::matmul_swiglu(),
            rules::matmul_softcap(),
            rules::matmul_bias(),
            rules::matmul_split_rows(),
            rules::gated_residual_norm_modulate(),
            rules::norm_modulate(),
            rules::rmsnorm_rope_partial_q(),
            rules::norm_rope_last(),
            rules::norm_rope_last_append(),
            rules::split_rope_yarn_append(),
            rules::rope_yarn_append(),
            rules::qg_qknorm_rope_append(),
            rules::lse_sink(),
            rules::pool_attention(),
            rules::pool_compress(),
            rules::mla_prefill_expanded(),
            rules::norm_rope_pair(),
            rules::index_topk_rope(),
            rules::hc_prenorm_project(),
            rules::hc_gates_norm(),
            rules::res_blend_rmsnorm(),
            rules::hc_pre(),
        ]
    });
#[cfg(feature = "rocm")]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules;
        vec![
            rules::qkv_fused_qknorm_rope_vnorm_write(),
            rules::rmsnorm_residual_add(),
            rules::embed_scale_add_select(),
            rules::embed_scale_add(),
            rules::residual_add_rmsnorm(),
            rules::matmul_geglu(),
            rules::matmul_softcap(),
            rules::matmul_bias(),
            rules::matmul_swiglu_matmul(),
            rules::moe_experts(),
            rules::gated_residual_norm_modulate(),
            rules::norm_modulate(),
            rules::rmsnorm_rope_partial_q(),
            rules::hc_norm_project(),
            rules::gdn_step(),
            rules::q_gate_norm_rope_append(),
        ]
    });

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

#[cfg(test)]
mod tests {
    use poem_ir::Operation;

    /// Every table this backend keys by op name names an op: an entry under
    /// a name no op carries is never read, and the op it meant falls to the
    /// table's default without a word.
    #[test]
    fn every_name_keyed_entry_names_an_op() {
        let stale: Vec<&str> = kernels_cuda::seat::ENTRIES
            .iter()
            .map(|entry| entry.name)
            .chain(super::FUSIONS.iter().map(poem_compiler::fuse::Rule::kernel))
            .chain(super::UNCHAINED)
            .chain(super::GROUPED)
            .chain(super::PLANNED)
            .chain(super::EXCLUSIVE)
            .filter(|name| !Operation::names_an_op(name))
            .collect();
        assert!(stale.is_empty(), "named by no op: {stale:?}");
    }

    /// Every fused op this backend's rules form on a catalog deployment is
    /// one its dispatch fires: a rule it holds and a form its dispatch turns
    /// away cannot both stand.
    #[test]
    fn every_fused_op_its_rules_form_is_one_dispatch_fires() {
        let mut refused = std::collections::BTreeSet::new();
        for row in poem_compiler::catalog::deployments().chain(poem_compiler::catalog::splits()) {
            let Ok(trace) = row.try_trace(poem_ir::Platform::Cuda) else {
                continue;
            };
            #[cfg(feature = "rocm")]
            let trace = crate::split_arm::split_into_arms(trace);
            let fused = poem_compiler::fuse::fuse(trace, super::FUSIONS.iter());
            for node in &fused.nodes {
                if let Operation::Fused(op) = &node.op
                    && crate::dispatch::refuses(op)
                {
                    refused.insert(format!("{}: {}", row.name, poem_ir::Operands::name(op)));
                }
            }
        }
        assert!(refused.is_empty(), "formed, then refused: {refused:?}");
    }
}
