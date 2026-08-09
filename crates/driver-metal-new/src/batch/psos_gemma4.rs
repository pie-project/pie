//! Gemma 4's PSO plan: the family with the most kernels of its own.
//!
//! Two facts shape the list. Every matvec runs the TAIL form — the PLE
//! projections' K is 256, which no reduction block divides — and the
//! checkpoint may carry a SECOND affine format: mlx_lm quantizes per
//! tensor, and the 26B's predicate has shipped two different exemption
//! sets (lmstudio's QAT spares the dense FFN and the router at 8 bits,
//! mlx-community's spares only the router). One format over the other's
//! bytes compiles, binds, dispatches, is fast, and every token is wrong
//! — the C++ records the router's logits at cosine 0.10 while every
//! tensor feeding them agreed at 0.9999. So which kinds read the alt
//! format is geometry, and the plan claims them against the alt entry.
//!
//! Both attention widths are the geometry's: sliding layers at
//! `head_dim`, full layers at `global_head_dim`, both through the SWA
//! kernel (a full layer is window 0). An uninstantiated width fails BY
//! NAME at load.

use super::abi::Kernel;
use super::gemma4::Gemma4Geometry;
use super::psos::{DecodePsoPlan, PsoRequest};

/// The whole M=1 compile list, self-contained as the other families'.
#[must_use]
#[allow(clippy::too_many_lines)] // one list; splitting hides what is compiled
pub fn gemma4_step_plan(g: &Gemma4Geometry) -> DecodePsoPlan {
    let mut plan = DecodePsoPlan::default();
    let mut want = |file: &'static str, entry: String, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry,
            kinds: kinds.to_vec(),
        });
    };
    let suffix = g.quant.kernel_suffix();

    // The scaled embed gather serves BOTH tables: the token embedding
    // (scaled √hidden) and the PLE table (scaled √ple_dim) — the scale
    // is a constant, not a kernel.
    want(
        "layout/embed_gather.metal",
        format!("embed_gather_scaled_4bit{suffix}"),
        &[Kernel::EmbedGather, Kernel::G4PleTokenGather],
    );
    // Every matvec is the tail form; the alt-format kinds are peeled
    // off below when the checkpoint carries a second format.
    let mut tail_kinds = vec![
        Kernel::QmvQ,
        Kernel::QmvK,
        Kernel::QmvV,
        Kernel::QmvO,
        Kernel::QmvLmHead,
        Kernel::G4PleProjGemv,
        Kernel::G4PleGateGemv,
        Kernel::G4PleProjLayerGemv,
    ];
    let mut alt_kinds = Vec::new();
    let ffn = [Kernel::QmvGate, Kernel::QmvUp, Kernel::QmvDown];
    if g.has_alt_quant() && g.alt_quant_ffn {
        alt_kinds.extend(ffn);
    } else {
        tail_kinds.extend(ffn);
    }
    if g.is_moe() {
        if g.has_alt_quant() && g.alt_quant_router {
            alt_kinds.push(Kernel::G4Router);
        } else {
            tail_kinds.push(Kernel::G4Router);
        }
    }
    want(
        "quant/qmv.metal",
        format!("affine_qmv_tail{suffix}"),
        &tail_kinds,
    );
    if !alt_kinds.is_empty() {
        // A second width means a second copy of the ONE kernel whose
        // selection depends on the width: which K counts as aligned is
        // `32·(32/bits)·2`, so the same projection can need the tail at
        // one width and not the other.
        want(
            "quant/qmv.metal",
            format!("affine_qmv_tail{}", g.ffn_quant.kernel_suffix()),
            &alt_kinds,
        );
    }
    // The norms: one single-row kernel, ten kinds — the sandwich's
    // fused halves get their own entries below.
    want(
        "norm/rms.metal",
        "rms_single_row_bfloat16".to_owned(),
        &[
            Kernel::Rms,
            Kernel::QNorm,
            Kernel::KNorm,
            Kernel::G4FfnPreNorm,
            Kernel::G4PleProjNorm,
            Kernel::G4RouterNorm,
            Kernel::G4MoeNorm,
            Kernel::G4DenseBranchNorm,
            Kernel::G4MoeBranchNorm,
            Kernel::FinalRms,
        ],
    );
    want(
        "norm/rms.metal",
        "rms_residual_bfloat16".to_owned(),
        &[Kernel::G4AttnPostResidual, Kernel::G4FfnPostResidual],
    );
    want(
        "norm/rms.metal",
        "rms_residual_scaled_bfloat16".to_owned(),
        &[Kernel::G4PleResidualScaled],
    );
    want(
        "norm/vector.metal",
        "vnorm_single_row_bfloat16".to_owned(),
        &[Kernel::G4VNorm, Kernel::G4VNormFromK],
    );
    // Partial rotary rides the `prop` rope: full layers rotate a
    // quarter of a 512-wide head, and the proportion is a constant.
    want(
        "rope/neox.metal",
        "neox_prop_decode_bfloat16".to_owned(),
        &[Kernel::Rope, Kernel::RopeK],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_bfloat16".to_owned(),
        &[Kernel::KvAppend],
    );
    // Both widths through the SWA kernel — a full layer is window 0.
    want(
        "attn/sdpa_sliding.metal",
        format!("sdpa_vector_decode_swa_bfloat16_d_{}", g.head_dim),
        &[Kernel::G4SdpaSliding],
    );
    want(
        "attn/sdpa_sliding.metal",
        format!("sdpa_vector_decode_swa_bfloat16_d_{}", g.global_head_dim),
        &[Kernel::Sdpa],
    );
    // GeGLU-tanh three ways from one kernel: the dense MLP, the sorted
    // stack (a sorted row IS a slot), and the PLE gate — at M=1 the
    // layer's slice of the table is a buffer offset, so the dense form
    // serves; the strided MB form is the prefill's.
    want(
        "mlp/gated.metal",
        "geglu_tanh_bfloat16".to_owned(),
        &[Kernel::G4Geglu, Kernel::G4ExpertGeglu, Kernel::G4PleGeglu],
    );
    want(
        "norm/layer_scalar.metal",
        "layer_scalar_mul_bfloat16".to_owned(),
        &[Kernel::G4LayerScalar],
    );
    want(
        "layout/ple_combine.metal",
        "ple_combine_bfloat16".to_owned(),
        &[Kernel::G4PleCombine],
    );
    want(
        "attn/logit_softcap.metal",
        "logit_softcap_bfloat16".to_owned(),
        &[Kernel::G4Softcap],
    );
    if g.is_moe() {
        // The scaled top-k: gemma's learned per-expert gain is what
        // makes this its router and not gpt-oss's.
        want(
            "moe/route.metal",
            "router_topk_scaled_bfloat16".to_owned(),
            &[Kernel::G4RouterTopK],
        );
        want(
            "quant/qmv.metal",
            format!("affine_qmv_routed{suffix}"),
            &[
                Kernel::G4ExpertGate,
                Kernel::G4ExpertUp,
                Kernel::G4ExpertDown,
            ],
        );
        want(
            "moe/route.metal",
            "route_sort".to_owned(),
            &[Kernel::G4MoeSort],
        );
        want(
            "moe/route.metal",
            "route_gather".to_owned(),
            &[Kernel::G4MoeGather],
        );
        want(
            "moe/route.metal",
            "combine_sorted".to_owned(),
            &[Kernel::G4ExpertCombine],
        );
        want(
            "norm/residual_add.metal",
            "residual_add_bfloat16".to_owned(),
            &[Kernel::G4BranchAdd],
        );
    }
    want(
        "sample/argmax.metal",
        "argmax_logits_bfloat16".to_owned(),
        &[Kernel::Argmax],
    );
    plan
}

/// The M>1 compile list: the M=1 plan with the batched forms claimed
/// AFTER their base entries. The paged attention's by-kind claim is the
/// SLIDING width's; the full layers' d512 instantiation is slot-keyed
/// in the shared lattice and the step's table resolves it from the
/// layer — one kind, two widths, the same rule as everywhere: the
/// launch is identical, only the instantiation moves.
#[must_use]
pub fn gemma4_mb_plan(g: &Gemma4Geometry) -> DecodePsoPlan {
    let mut plan = gemma4_step_plan(g);
    let suffix = g.quant.kernel_suffix();
    let mut want = |file: &'static str, entry: String, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry,
            kinds: kinds.to_vec(),
        });
    };
    want(
        "layout/embed_gather.metal",
        format!("embed_gather_scaled_mb_4bit{suffix}"),
        &[Kernel::EmbedGather, Kernel::G4PleTokenGather],
    );
    want(
        "rope/neox.metal",
        "neox_prop_mb_bfloat16".to_owned(),
        &[Kernel::Rope, Kernel::RopeK],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_paged_bfloat16".to_owned(),
        &[Kernel::KvAppendPaged],
    );
    want(
        "attn/sdpa_paged.metal",
        format!("sdpa_paged_decode_bfloat16_d_{}", g.head_dim),
        &[Kernel::SdpaPaged],
    );
    // The PLE gate is the one elementwise kernel whose operands have
    // different row pitches at M>1: `up` strides by the whole table.
    want(
        "mlp/gated.metal",
        "geglu_tanh_strided_bfloat16".to_owned(),
        &[Kernel::G4PleGeglu],
    );
    want(
        "layout/row_gather.metal",
        "row_gather_bfloat16".to_owned(),
        &[Kernel::G4RowGather],
    );
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::build_gemma4_dag;
    use crate::batch::geometry::AffineFormat;

    fn moe_26b() -> Gemma4Geometry {
        Gemma4Geometry {
            n_kv_heads: 8,
            attention_k_eq_v: true,
            n_global_kv_heads: 2,
            enable_moe: true,
            n_experts: 128,
            experts_per_token: 4,
            moe_intermediate: 704,
            ..Gemma4Geometry::default()
        }
    }

    #[test]
    fn the_plan_claims_every_kind_each_shape_dispatches() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        for g in [Gemma4Geometry::default(), moe_26b()] {
            let plan = gemma4_step_plan(&g);
            let dag = build_gemma4_dag(&g, true);
            for d in &dag {
                assert!(
                    plan.source_of(d.kind).is_some(),
                    "{:?} has no compiled pipeline — the fire would refuse at prepare",
                    d.kind
                );
            }
            for request in &plan.requests {
                assert!(table.contains(&request.entry), "{}", request.entry);
            }
        }
    }

    #[test]
    fn the_second_format_peels_exactly_the_exempted_kinds() {
        let entry_of = |plan: &DecodePsoPlan, kind: Kernel| {
            let i = plan.source_of(kind).expect("claimed");
            plan.requests[i].entry.clone()
        };
        // mlx-community's 26B: only the router is spared at 8 bits.
        let router_only = Gemma4Geometry {
            ffn_quant: AffineFormat { bits: 8, group: 64 },
            alt_quant_router: true,
            ..moe_26b()
        };
        let plan = gemma4_step_plan(&router_only);
        assert!(entry_of(&plan, Kernel::G4Router).contains("_b_8"));
        assert!(entry_of(&plan, Kernel::QmvGate).contains("_b_4"));
        // lmstudio's QAT build: the dense FFN moves too.
        let ffn_too = Gemma4Geometry {
            alt_quant_ffn: true,
            ..router_only
        };
        let plan = gemma4_step_plan(&ffn_too);
        assert!(entry_of(&plan, Kernel::QmvGate).contains("_b_8"));
        assert!(entry_of(&plan, Kernel::QmvQ).contains("_b_4"));
        // One format: nothing peels, and no alt entry is compiled.
        let one = gemma4_step_plan(&moe_26b());
        assert!(entry_of(&one, Kernel::G4Router).contains("_b_4"));
        // Both attention widths are spelled from the geometry.
        assert!(entry_of(&one, Kernel::G4SdpaSliding).ends_with("_d_256"));
        assert!(entry_of(&one, Kernel::Sdpa).ends_with("_d_512"));
    }
}
