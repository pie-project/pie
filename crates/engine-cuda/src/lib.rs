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
pub mod synth;
pub mod tap;
pub mod tune;
pub mod voxels;
pub mod weights;
pub mod window;

pub const EXCLUSIVE: [&str; 0] = [];

/// Ops one launch serves over the runs of rows a region's classes hold, by a
/// segment list. Each reads every row's own route and leaves a row that has
/// none untouched, so a launch may also walk the runs' whole union: a
/// captured body does, its live rows read off the device, so one body
/// answers whichever of the region's classes a fire seats.
pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fusions this backend implements: the compiler runs these rules alone,
/// so each forms only fused ops this backend's dispatch fires. A rule it does
/// not hold (`matmul_swiglu_matmul`, which Metal splits across engines)
/// leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules::{self, Rotated};
        use poem_ir::GluAct;
        use poem_ir::ops::fused::Intermediates;
        vec![
            rules::split_qkv_fused(),
            rules::qkv_fused_qknorm_rope_vnorm_write(),
            rules::embed_scale_add_select(),
            rules::embed_scale_add(),
            rules::matmul_glu(&[GluAct::GegluTanh, GluAct::Swiglu]),
            rules::moe_mega(),
            rules::moe_experts(),
            rules::moe_select_glu(),
            rules::matmul_softcap(),
            rules::matmul_bias(),
            rules::matmul_moe_route(),
            rules::matmul_split_rows(),
            rules::gated_residual_norm_modulate(),
            rules::norm_modulate(),
            rules::norm_rope(Rotated::First),
            rules::norm_rope(Rotated::Last),
            rules::norm_rope_last_append(),
            rules::split_rope_yarn_append(),
            rules::rope_yarn_append(),
            rules::q_gate_norm_rope_append(Intermediates::Unwritten),
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

/// The fused kernels left on with `fuse-chains=off`.
#[cfg(not(feature = "rocm"))]
pub const UNCHAINED: [&str; 1] = ["fused.qkv_fused_qknorm_rope_vnorm_write"];
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

    /// No `fused.region` of a catalog deployment, fused by this backend's
    /// rules, folded and planned, lands an output over one of its inputs: a
    /// generated kernel reads its parameters read-only on that. The arena
    /// holds a value over every node that reads or writes it, both ends
    /// included, so an input the region reads last and an output it writes
    /// first are live together at its node; and a region aliases no output
    /// into an input.
    #[cfg(not(feature = "rocm"))]
    #[test]
    fn no_region_of_a_catalog_deployment_lands_over_its_inputs() {
        use poem_ir::Fused;
        let (mut wrong, mut held, mut planned, mut unplanned) =
            (Vec::new(), 0usize, 0usize, Vec::new());
        let profile = poem_compiler::DeviceProfile::default();
        for row in poem_compiler::catalog::deployments() {
            let Ok(trace) = row.try_trace(poem_ir::Platform::Cuda) else {
                continue;
            };
            let fused = poem_compiler::fuse::fuse(trace, super::FUSIONS.iter());
            let (folded, _) = poem_compiler::fuse::chain::fold(fused);
            let regions: Vec<(Vec<poem_ir::ValueId>, Vec<poem_ir::ValueId>)> = folded
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    Operation::Fused(Fused::Region { ins, outs, .. }) => {
                        Some((ins.clone(), outs.clone()))
                    }
                    _ => None,
                })
                .collect();
            if regions.is_empty() {
                continue;
            }
            let load = engine::load::Budgets {
                max_lanes: 4,
                max_tokens: 2048,
                ..engine::load::Budgets::default()
            };
            let mut budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(4, 2048));
            if let Some(ladder) = crate::api::patch_ladder(&folded, &load) {
                budgets = budgets.with_patches(ladder);
            }
            if let Some(ladder) = crate::api::voxel_ladder(&folded, &load) {
                budgets = budgets.with_voxels(ladder);
            }
            let Ok(compiled) = poem_compiler::compile_axes(&folded, &budgets, &profile) else {
                unplanned.push(row.name.clone());
                continue;
            };
            planned += 1;
            let arena = &compiled.arena;
            let fire = poem_compiler::FireRows::ceilings(&budgets);
            for (ins, outs) in regions {
                for out in &outs {
                    for input in &ins {
                        let shared = arena.root(*out) == arena.root(*input);
                        let overlap = match (arena.window(*out, fire), arena.window(*input, fire)) {
                            (Some(a), Some(b)) => {
                                held += 1;
                                a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes
                            }
                            _ => false,
                        };
                        if shared || overlap {
                            wrong.push(format!("{}: {out:?} over {input:?}", row.name));
                        }
                    }
                }
            }
        }
        eprintln!(
            "{planned} deployments planned, {held} region reads the arena holds, not planned: \
             {unplanned:?}"
        );
        assert!(held > 0, "a region reads a value the arena holds");
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }
}
