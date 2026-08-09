//! GPT-OSS's PSO plan: which entrypoints this family compiles, keyed by
//! the three facts the staged tensors decided.
//!
//! The geometry decides and every choice is refused-not-defaulted, because
//! either wrong answer RUNS: `router_bits` selects the router's matvec (8
//! for the width mlx_lm's predicate usually leaves, 4 for a uniform
//! checkpoint — either kernel over the other's packing is fluent wrong
//! text); `mxfp4_experts` selects which routed matvec reads the bank; and
//! `head_dim` names the attention instantiation — this was once a literal
//! 64 while the geometry read the config, so a variant shipping any other
//! width would have run a d=64 pipeline over its heads, striding past the
//! end of each and writing zeros. Spelled from the geometry, an
//! uninstantiated width fails to build BY NAME at load.

use super::abi::Kernel;
use super::geometry::AffineFormat;
use super::gptoss::GptOssGeometry;
use super::psos::{DecodePsoPlan, PsoRequest};

/// The head width the matrix-unit attention is instantiated at.
pub const SDPA_MMA_HEAD_DIM: u32 = 64;

/// Which table slot a compiled gpt-oss pipeline lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)] // the slot names ARE the C++ field names
pub enum GptOssSlot {
    QmvTail,
    QmvTailBias,
    QmvRoutedBias,
    QmvRouter,
    RouterTopK,
    MoeSort,
    MoeGather,
    MoeCombine,
    SwiGlu,
    SdpaSink,
    SdpaSinkPaged,
    SdpaSinkPagedTiled,
    SdpaSinkPagedMma,
    RopeFreqs,
    RopeFreqsMb,
    RowGather,
    /// The routed MXFP4 GEMM: `width` indexes the sort's tile widths
    /// (16/32/64), `bn` the column tiles. Slot-keyed, not kind-keyed —
    /// which entry a routed dispatch runs is `qmm_bn`/`qmm_bm` on the
    /// dispatch itself, decided where the launch was.
    QmmRoutedBias {
        width: usize,
        bn: usize,
    },
}

/// One entrypoint the plan wants compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GptOssPsoRequest {
    /// Where the pipeline goes.
    pub slot: GptOssSlot,
    /// The shader file, relative to the kernels directory.
    pub file: &'static str,
    /// The entrypoint name.
    pub entry: String,
}

fn suffix(bits: u32, group: u32) -> String {
    AffineFormat { bits, group }.kernel_suffix()
}

/// Lay out the compile list for `g`.
///
/// K = 2880 is a whole number of quantization groups but not of any
/// reduction block, so every projection here runs the TAIL-handling
/// matvec. The mixture movers and the top-k are the shared kernels — the
/// shape is a property of `router_topk`, not of gpt-oss.
#[must_use]
pub fn plan_gptoss_psos(g: &GptOssGeometry) -> Vec<GptOssPsoRequest> {
    let mut out = Vec::new();
    let mut want = |slot: GptOssSlot, file: &'static str, entry: String| {
        out.push(GptOssPsoRequest { slot, file, entry });
    };
    let qmv = "quant/qmv.metal";
    want(
        GptOssSlot::QmvTail,
        qmv,
        format!("affine_qmv_tail{}", suffix(g.proj_bits, 64)),
    );
    want(
        GptOssSlot::QmvTailBias,
        qmv,
        format!("affine_qmv_tail_bias{}", suffix(g.proj_bits, 64)),
    );
    // The bank's matvec: the checkpoint's own MXFP4 (block exponents,
    // group 32, no zero point) or the loader's affine U4.
    want(
        GptOssSlot::QmvRoutedBias,
        qmv,
        if g.mxfp4_experts {
            format!("mxfp4_qmv_routed_bias{}", suffix(4, 32))
        } else {
            format!("affine_qmv_routed_bias{}", suffix(4, 64))
        },
    );
    want(
        GptOssSlot::QmvRouter,
        qmv,
        format!("affine_qmv_tail_bias{}", suffix(g.router_bits, 64)),
    );
    want(
        GptOssSlot::RouterTopK,
        "moe/route.metal",
        "router_topk_bfloat16".to_string(),
    );
    want(
        GptOssSlot::MoeSort,
        "moe/route.metal",
        "route_sort".to_string(),
    );
    want(
        GptOssSlot::MoeGather,
        "moe/route.metal",
        "route_gather".to_string(),
    );
    want(
        GptOssSlot::MoeCombine,
        "moe/route.metal",
        "combine_sorted".to_string(),
    );
    want(
        GptOssSlot::SwiGlu,
        "mlp/gated.metal",
        "gptoss_swiglu_bfloat16".to_string(),
    );
    let d = format!("_d_{}", g.head_dim);
    want(
        GptOssSlot::SdpaSink,
        "attn/sdpa_sliding.metal",
        format!("sdpa_vector_decode_sink_bfloat16{d}"),
    );
    want(
        GptOssSlot::SdpaSinkPaged,
        "attn/sdpa_paged.metal",
        format!("sdpa_paged_decode_sink_bfloat16{d}"),
    );
    want(
        GptOssSlot::SdpaSinkPagedTiled,
        "attn/sdpa_paged.metal",
        format!("sdpa_paged_tiled_sink_bfloat16{d}"),
    );
    if g.head_dim == SDPA_MMA_HEAD_DIM {
        want(
            GptOssSlot::SdpaSinkPagedMma,
            "attn/sdpa_paged_mma.metal",
            format!("sdpa_paged_mma_sink_bfloat16{d}"),
        );
    }
    want(
        GptOssSlot::RopeFreqs,
        "rope/neox.metal",
        "neox_freqs_decode_bfloat16".to_string(),
    );
    want(
        GptOssSlot::RopeFreqsMb,
        "rope/neox.metal",
        "neox_freqs_mb_bfloat16".to_string(),
    );
    want(
        GptOssSlot::RowGather,
        "layout/row_gather.metal",
        "row_gather_bfloat16".to_string(),
    );
    // The routed GEMM table exists only for the bank that has an MXFP4
    // GEMM instantiation; the affine bank keeps the matvec at every
    // batch. `bm` is spelled from the shared tile widths, not restated —
    // it is the same number the sort pads each expert's run to and the
    // same number the grid divides the sorted rows by, and the C++ notes
    // that hardcoding it here was the one place that did not follow the
    // constant.
    if g.mxfp4_experts {
        for (width, &rows) in super::psos_mb::MOE_TILE_WIDTHS.iter().enumerate() {
            for bn in 0..3usize {
                want(
                    GptOssSlot::QmmRoutedBias { width, bn },
                    "quant/qmm_t.metal",
                    format!(
                        "mxfp4_qmm_t_routed_bias_bfloat16_bm_{rows}_bn_{}",
                        16u32 << bn
                    ),
                );
            }
        }
    }
    out
}

/// The kinds a slot's pipeline serves in the M=1 DAG.
///
/// The C++ `pso_for` was a second switch at ENCODE time — every fire
/// re-decided the mapping, and a missed arm fell through to an empty
/// `Pso{}` that faulted at dispatch. Here the fan-out is data on the
/// plan: the encode walk stays the shared per-kind lookup, and a kind
/// nothing claims is refused when the step is prepared, not when it
/// fires. Empty for the slots the M=1 ring never dispatches — the paged
/// sinks are the paged fire's, `RopeFreqsMb` and `RowGather` are the
/// prefill's.
#[must_use]
pub fn gptoss_kinds(slot: GptOssSlot) -> &'static [Kernel] {
    match slot {
        GptOssSlot::QmvTail => &[Kernel::LmHeadUntied],
        GptOssSlot::QmvTailBias => &[
            Kernel::GoQmvQ,
            Kernel::GoQmvK,
            Kernel::GoQmvV,
            Kernel::GoQmvO,
        ],
        GptOssSlot::QmvRoutedBias => &[
            Kernel::GoExpertGate,
            Kernel::GoExpertUp,
            Kernel::GoExpertDown,
        ],
        GptOssSlot::QmvRouter => &[Kernel::GoRouter],
        GptOssSlot::RouterTopK => &[Kernel::GoRouterTopK],
        GptOssSlot::MoeSort => &[Kernel::LlMoeSort],
        GptOssSlot::MoeGather => &[Kernel::LlMoeGather],
        GptOssSlot::MoeCombine => &[Kernel::GoExpertCombine],
        GptOssSlot::SwiGlu => &[Kernel::GoSwiGlu],
        GptOssSlot::SdpaSink => &[Kernel::GoSdpaSink],
        GptOssSlot::RopeFreqs => &[Kernel::Rope, Kernel::RopeK],
        GptOssSlot::RopeFreqsMb
        | GptOssSlot::SdpaSinkPaged
        | GptOssSlot::SdpaSinkPagedTiled
        | GptOssSlot::SdpaSinkPagedMma
        | GptOssSlot::RowGather
        | GptOssSlot::QmmRoutedBias { .. } => &[],
    }
}

/// [`gptoss_kinds`] for the M>1 DAG: the ring attention and the decode
/// rope give way to their paged and freqs-table forms, and the sampled-row
/// compaction joins the tail. The matvec entries keep their kinds — a
/// dispatch that tiled carries `qmm_bn > 0` and is served by slot, so the
/// kind-keyed table is its FALLBACK, not its contradiction. The tiled and
/// matrix sinks stay unclaimed with the shared family's — the scalar paged
/// sink serves every fleet width until that arm is opened for both.
#[must_use]
pub fn gptoss_mb_kinds(slot: GptOssSlot) -> &'static [Kernel] {
    match slot {
        GptOssSlot::SdpaSink | GptOssSlot::RopeFreqs => &[],
        GptOssSlot::SdpaSinkPaged => &[Kernel::GoSdpaSinkPaged],
        GptOssSlot::RopeFreqsMb => &[Kernel::Rope, Kernel::RopeK],
        GptOssSlot::RowGather => &[Kernel::G4RowGather],
        other => gptoss_kinds(other),
    }
}

/// The whole M>1 compile list, as the shared plan type — the MB twin of
/// [`gptoss_step_plan`], claiming exactly the kinds
/// [`build_gptoss_dag_mb`](super::build_gptoss_dag_mb) dispatches. Same
/// self-containment argument: the C++ MB encoder borrowed THREE tables
/// (qwen's base, the shared multibatch set, its own), and a kind that fell
/// through every switch dispatched an empty `Pso{}`.
#[must_use]
pub fn gptoss_mb_plan(g: &GptOssGeometry) -> DecodePsoPlan {
    let mut plan = DecodePsoPlan::default();
    for request in plan_gptoss_psos(g) {
        let kinds = gptoss_mb_kinds(request.slot);
        if kinds.is_empty() {
            continue;
        }
        plan.requests.push(PsoRequest {
            file: request.file,
            entry: request.entry,
            kinds: kinds.to_vec(),
        });
    }
    let mut want = |file: &'static str, entry: String, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry,
            kinds: kinds.to_vec(),
        });
    };
    // The MB embed reads a row per token off `grid.y`; the M=1 gather has
    // no row axis at all. Same historical `4bit` prefix, same rule: the
    // suffix carries the truth.
    want(
        "layout/embed_gather.metal",
        format!("embed_gather_mb_4bit{}", suffix(g.proj_bits, 64)),
        &[Kernel::EmbedUntied],
    );
    want(
        "norm/rms.metal",
        "rms_single_row_bfloat16".to_owned(),
        &[Kernel::Rms, Kernel::FfnRms, Kernel::FinalRms],
    );
    want(
        "norm/residual_add.metal",
        "residual_add_bfloat16".to_owned(),
        &[Kernel::Residual, Kernel::LayerOut],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_paged_bfloat16".to_owned(),
        &[Kernel::KvAppendPaged],
    );
    want(
        "sample/argmax.metal",
        "argmax_logits_bfloat16".to_owned(),
        &[Kernel::Argmax],
    );
    plan
}

/// The whole M=1 compile list, as the shared plan type: the family's own
/// pipelines fanned out by [`gptoss_kinds`], plus the shared kinds the
/// DAG dispatches (norms, residuals, the KV append, the embed ends, the
/// argmax).
///
/// The C++ could not fire this family alone: `pso_for` fell back to a
/// BASE table for the norms and residuals, so gpt-oss ran only while
/// qwen's pipelines were loaded beside it. This plan is self-contained —
/// `load_step_psos` compiles it verbatim and the encode walk never asks
/// another family for a pipeline.
#[must_use]
pub fn gptoss_step_plan(g: &GptOssGeometry) -> DecodePsoPlan {
    let mut plan = DecodePsoPlan::default();
    for request in plan_gptoss_psos(g) {
        let kinds = gptoss_kinds(request.slot);
        if kinds.is_empty() {
            continue;
        }
        plan.requests.push(PsoRequest {
            file: request.file,
            entry: request.entry,
            kinds: kinds.to_vec(),
        });
    }
    let mut want = |file: &'static str, entry: String, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry,
            kinds: kinds.to_vec(),
        });
    };
    // `embed_gather_4bit` is the entrypoint FAMILY's name, not its width:
    // the 8-bit instantiation lives under the same prefix, and the suffix
    // carries the truth.
    want(
        "layout/embed_gather.metal",
        format!("embed_gather_4bit{}", suffix(g.proj_bits, 64)),
        &[Kernel::EmbedUntied],
    );
    want(
        "norm/rms.metal",
        "rms_single_row_bfloat16".to_owned(),
        &[Kernel::Rms, Kernel::FfnRms, Kernel::FinalRms],
    );
    want(
        "norm/residual_add.metal",
        "residual_add_bfloat16".to_owned(),
        &[Kernel::Residual, Kernel::LayerOut],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_bfloat16".to_owned(),
        &[Kernel::KvAppend],
    );
    want(
        "sample/argmax.metal",
        "argmax_logits_bfloat16".to_owned(),
        &[Kernel::Argmax],
    );
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::dispatch_gptoss::build_gptoss_dag;

    #[test]
    fn every_variant_the_solver_can_pick_is_a_shipped_entrypoint() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        let variants = [
            GptOssGeometry::default(), // affine experts, router 8, proj 4
            GptOssGeometry {
                mxfp4_experts: true,
                router_bits: 4,
                proj_bits: 8,
                ..GptOssGeometry::default()
            },
        ];
        for g in variants {
            for request in plan_gptoss_psos(&g) {
                assert!(
                    table.contains(&request.entry),
                    "{} is not in the signature table (slot {:?}, mxfp4 {})",
                    request.entry,
                    request.slot,
                    g.mxfp4_experts
                );
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("crates/")
                    .join("kernels-metal/kernels")
                    .join(request.file);
                assert!(path.exists(), "{} does not exist", path.display());
            }
        }
    }

    #[test]
    fn the_step_plan_claims_every_kind_the_dag_dispatches() {
        let g = GptOssGeometry::default();
        let plan = gptoss_step_plan(&g);
        let dag = build_gptoss_dag(&g, true);
        for d in &dag {
            assert!(
                plan.source_of(d.kind).is_some(),
                "{:?} has no compiled pipeline — the fire would refuse at prepare",
                d.kind
            );
        }
        // And self-contained: the C++ borrowed qwen's base table for the
        // norms and residuals; this plan compiles its own.
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        for request in &plan.requests {
            assert!(table.contains(&request.entry), "{}", request.entry);
        }
        // The head is the one UNBIASED projection; everything else that
        // multiplies weights carries its bias in the kernel.
        let entry_of = |kind: Kernel| {
            let i = plan.source_of(kind).expect("claimed above");
            plan.requests[i].entry.clone()
        };
        assert!(entry_of(Kernel::LmHeadUntied).starts_with("affine_qmv_tail_bfloat16"));
        assert!(entry_of(Kernel::GoQmvQ).starts_with("affine_qmv_tail_bias"));
        // The 8-bit embed lives under the historical `4bit` prefix; the
        // suffix carries the truth.
        let eight = GptOssGeometry {
            proj_bits: 8,
            ..GptOssGeometry::default()
        };
        let plan8 = gptoss_step_plan(&eight);
        let i = plan8.source_of(Kernel::EmbedUntied).expect("the embed");
        assert_eq!(
            plan8.requests[i].entry,
            "embed_gather_4bit_bfloat16_gs_64_b_8"
        );
        assert!(table.contains(&plan8.requests[i].entry));
    }

    #[test]
    fn the_mb_plan_claims_every_kind_the_mb_dag_dispatches() {
        let g = GptOssGeometry {
            mxfp4_experts: true,
            ..GptOssGeometry::default()
        };
        let plan = gptoss_mb_plan(&g);
        let dag = crate::batch::build_gptoss_dag_mb(
            &g,
            &crate::tuning::Tuning::default(),
            16,
            2,
            0,
            true,
        );
        for d in &dag {
            assert!(
                plan.source_of(d.kind).is_some(),
                "{:?} has no compiled pipeline — the MB fire would refuse at prepare",
                d.kind
            );
        }
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        for request in &plan.requests {
            assert!(table.contains(&request.entry), "{}", request.entry);
        }
        // The swaps are the paged pair, the freqs-table rope, the MB
        // embed and the compaction; the ring sink and the decode rope do
        // not survive into this plan.
        let entry_of = |kind: Kernel| {
            let i = plan.source_of(kind).expect("claimed above");
            plan.requests[i].entry.clone()
        };
        assert!(entry_of(Kernel::GoSdpaSinkPaged).starts_with("sdpa_paged_decode_sink"));
        assert_eq!(entry_of(Kernel::Rope), "neox_freqs_mb_bfloat16");
        assert!(entry_of(Kernel::EmbedUntied).starts_with("embed_gather_mb_4bit"));
        assert_eq!(entry_of(Kernel::G4RowGather), "row_gather_bfloat16");
        assert_eq!(entry_of(Kernel::KvAppendPaged), "kv_append_paged_bfloat16");
        assert!(plan.source_of(Kernel::GoSdpaSink).is_none());
        assert!(plan.source_of(Kernel::KvAppend).is_none());
    }

    #[test]
    fn the_routed_gemm_table_belongs_to_the_bank_that_has_one() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        let mxfp4 = GptOssGeometry {
            mxfp4_experts: true,
            ..GptOssGeometry::default()
        };
        let routed: Vec<_> = plan_gptoss_psos(&mxfp4)
            .into_iter()
            .filter(|r| matches!(r.slot, GptOssSlot::QmmRoutedBias { .. }))
            .collect();
        assert_eq!(routed.len(), 9, "three sort widths by three column tiles");
        for r in &routed {
            assert!(table.contains(&r.entry), "{} is not shipped", r.entry);
        }
        // The affine bank compiles none — a table for it would name
        // entrypoints that do not exist, and the matvec is its every
        // batch.
        let affine = GptOssGeometry::default();
        assert!(
            plan_gptoss_psos(&affine)
                .iter()
                .all(|r| !matches!(r.slot, GptOssSlot::QmmRoutedBias { .. }))
        );
    }

    #[test]
    fn an_unusual_head_width_skips_the_matrix_unit_rather_than_lying() {
        let wide = GptOssGeometry {
            head_dim: 128,
            ..GptOssGeometry::default()
        };
        let plan = plan_gptoss_psos(&wide);
        assert!(
            plan.iter().all(|r| r.slot != GptOssSlot::SdpaSinkPagedMma),
            "the MMA pipeline is instantiated at d=64; other widths fall back"
        );
        // And the named instantiation carries the width, so an
        // uninstantiated one fails BY NAME at load instead of striding
        // past every head.
        let sink = plan
            .iter()
            .find(|r| r.slot == GptOssSlot::SdpaSink)
            .unwrap();
        assert!(sink.entry.ends_with("_d_128"));
    }
}
