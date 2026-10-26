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

pub const GROUPED: [&str; 1] = ["linear.lora_correct"];

/// The fusions this backend implements: the compiler runs these rules alone,
/// so each forms only fused ops this backend's dispatch fires. A rule it does
/// not hold (`matmul_swiglu_matmul`, which Metal splits across engines)
/// leaves its primitives alone here.
#[cfg(not(feature = "rocm"))]
pub static FUSIONS: std::sync::LazyLock<Vec<poem_compiler::fuse::Rule>> =
    std::sync::LazyLock::new(|| {
        use poem_compiler::fuse::rules::{self, Rotated};
        use poem_ir::ops::fused::Intermediates;
        vec![
            rules::split_qkv_fused(),
            rules::qkv_fused_qknorm_rope_vnorm_write(),
            rules::moe_mega(),
            rules::moe_experts(),
            rules::moe_select_glu(),
            rules::matmul_moe_route(),
            rules::matmul_split_rows(),
            rules::norm_rope(Rotated::First),
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

/// The hand row rules CUDA's chains replace: each site one formed folds
/// into a generated row kernel (`fuse::chain`) that lands its ops' bits
/// and, at the default knobs, outruns the rule's hand kernel. A head's norm
/// and the turn of its tail (`norm_rope`'s `Last`: dots3's k row,
/// DeepSeek-V4's q). Its `First` (Gemma 4's split turn) stays: a 512-wide
/// head's region trails the hand kernel at a decode's rows. A scaled
/// embedding's chain (`embed_scale_add`, Gemma 4's per-layer rows picked
/// from their stack with `embed_scale_add_select`) and a DiT block's norm
/// and modulation (`norm_modulate`, after the gated residual
/// `gated_residual_norm_modulate`): each a generated kernel that ties the
/// hand one at a decode's rows and outruns it from 512. And the hand
/// product rules CUDA's anchored regions replace (`fuse::anchor`): a
/// product's bias, its soft cap and a packed bank's GEGLU and SwiGLU, each
/// the epilogue of its contraction skeleton, timed against the product and
/// the epilogue on the row skeleton. The qkv writes stay: a packed qkv
/// product's split, norms and turn are an anchored region (Qwen3's, every
/// site no rule holds), but the hand kernel writes k and v into the paged
/// cache in the same launch, an epilogue lands tiles, and Gemma 4's
/// 512-wide global heads and MiMo's padded split are no epilogue's span;
/// Muse Spark's `qkv | gate` is `matmul_split_rows`'s first, whose long
/// fires land each half by a GEMM of its own, no product landed and cut.
#[cfg(not(feature = "rocm"))]
#[must_use]
pub fn retired() -> Vec<poem_compiler::fuse::Rule> {
    use poem_compiler::fuse::rules;
    vec![
        rules::norm_rope(rules::Rotated::Last),
        rules::matmul_glu(&[poem_ir::GluAct::GegluTanh, poem_ir::GluAct::Swiglu]),
        rules::matmul_softcap(),
        rules::matmul_bias(),
        rules::embed_scale_add_select(),
        rules::embed_scale_add(),
        rules::norm_modulate(),
        rules::gated_residual_norm_modulate(),
    ]
}

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

    /// The hand rules CUDA no longer holds, their chains generated row
    /// kernels and their products' epilogues anchored regions: on every
    /// catalog deployment, fused by this backend's rules and folded (the
    /// anchors, then the chains), none of them finds a match left that it
    /// fused, so each such site is a region (`fuse::chain`) or an anchored
    /// one (`fuse::anchor`). A product's rule fused only bf16 rows against a
    /// weight stored bf16 (its dispatch ran the product and its op apart over
    /// any other), so a site over another keeps its ops; so does one whose
    /// product is no whole runs of eight wide (a two-column head), which no
    /// row skeleton carries and the rule ran apart past one row. A retired
    /// rule's op may be another rule's too (`norm_rope`'s two turns), so its
    /// sites are what it forms past the backend's.
    #[cfg(not(feature = "rocm"))]
    #[test]
    fn every_site_a_retired_rule_fused_folds_into_a_region() {
        use poem_ir::{Def, Dim, Dtype, Fused, Trace, Ty, ValueId};
        let mut wrong = Vec::new();
        let mut census = std::collections::BTreeMap::<&str, (usize, usize, usize)>::new();
        let bf16 = |t: &Trace, v: &ValueId| {
            let decl = &t.values[v.0 as usize];
            match decl.def {
                Def::Weight(p) => t.params[p as usize].dtype == Dtype::Bf16,
                _ => matches!(
                    decl.ty,
                    Ty::Tensor {
                        dtype: Dtype::Bf16,
                        ..
                    }
                ),
            }
        };
        let narrow = |t: &Trace, v: &ValueId| {
            matches!(&t.values[v.0 as usize].ty,
                Ty::Tensor { shape, .. } if matches!(shape.last(), Some(Dim::Const(n)) if n % 8 != 0))
        };
        for rule in super::retired() {
            let kernel = rule.kernel();
            let only = [rule];
            for row in poem_compiler::catalog::deployments() {
                let Ok(trace) = row.try_trace(poem_ir::Platform::Cuda) else {
                    continue;
                };
                // The sites the rule forms in `t` that it fused (its product
                // bf16 against a bf16 weight), and of those the ones too
                // narrow for a row skeleton.
                let sites = |t: &Trace| {
                    let mut held = (0usize, 0usize);
                    for n in &t.nodes {
                        let Operation::Fused(op) = &n.op else {
                            continue;
                        };
                        if poem_ir::Operands::name(op) != kernel {
                            continue;
                        }
                        let (fused, thin) = match op {
                            Fused::MatmulBias { act, w, y, .. }
                            | Fused::MatmulSoftcap { act, w, y, .. } => {
                                (bf16(t, act) && bf16(t, w), narrow(t, y))
                            }
                            Fused::MatmulGlu { act, w, y, .. } => {
                                (bf16(t, act) && bf16(t, w), narrow(t, y))
                            }
                            _ => (true, false),
                        };
                        if fused {
                            held.0 += 1;
                            held.1 += usize::from(thin);
                        }
                    }
                    held
                };
                let formed = |t: Trace| {
                    let before = sites(&t);
                    let after = sites(&poem_compiler::fuse::fuse(t, only.iter()));
                    (after.0 - before.0, after.1 - before.1)
                };
                let fused = poem_compiler::fuse::fuse(trace, super::FUSIONS.iter());
                let (here, _) = formed(fused.clone());
                let (anchored, _) = crate::synth::fold_anchors(fused);
                let (folded, _) = poem_compiler::fuse::chain::fold(anchored);
                let (left, thin) = formed(folded);
                let held = census.entry(kernel).or_default();
                held.0 += here;
                held.1 += usize::from(here > 0);
                held.2 += thin;
                if left > thin {
                    wrong.push(format!(
                        "{}: {} of {here} `{kernel}` sites left",
                        row.name,
                        left - thin
                    ));
                }
            }
        }
        eprintln!("retired sites (fused, deployments, too narrow to fold): {census:?}");
        assert!(
            census.values().all(|(sites, ..)| *sites > 0),
            "each retired rule fused somewhere: {census:?}"
        );
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
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

    /// No `fused.region` or `fused.anchored` of a catalog deployment, fused
    /// by this backend's rules, folded and planned, lands an output over one
    /// of its inputs: a generated kernel reads its parameters read-only on
    /// that. The arena
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
            let (anchored, _) = crate::synth::fold_anchors(fused);
            let (folded, _) = poem_compiler::fuse::chain::fold(anchored);
            let regions: Vec<(Vec<poem_ir::ValueId>, Vec<poem_ir::ValueId>)> = folded
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    Operation::Fused(Fused::Region { ins, outs, .. }) => {
                        Some((ins.clone(), outs.clone()))
                    }
                    Operation::Fused(Fused::Anchored { act, ins, outs, .. }) => Some((
                        [*act].into_iter().chain(ins.iter().copied()).collect(),
                        outs.clone(),
                    )),
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
