//! The M>1 PSO plan: which entrypoints a multibatch decode compiles.
//!
//! The portable half of `load_multibatch_psos` (`decode_psos.cpp`): the
//! feature gating and the name grammar, produced as a request list the
//! device half compiles. Kept separate from the M=1 plan (`psos.rs`) as the
//! C++ keeps `MultiBatchPsos` separate from `DecodeStepPsos`, so the M=1
//! table stays byte-untouched.
//!
//! Two decisions from the C++ carry over as facts about the LIST rather
//! than code:
//!
//! * Requests are grouped by file. `quant/qmm_t.metal` alone yields ~25
//!   entrypoints out of one 1800-line source, and compiling them one at a
//!   time re-parsed the whole file for each; the device half batches each
//!   file's requests and front-ends it exactly once.
//! * The row-tile axis IS `QMM_BMS`. The C++ static_asserts its PSO
//!   table's extent against the kernel header's list; here the slots carry
//!   their rung index and the one list is this module's.
//!
//! Names are grammar products (`base + affine + tile…`), the same grammar
//! the shaders were stamped from. Whether a product actually exists in the
//! shader tree is pinned by the dev test against
//! `kernels_metal::entrypoints()`, exactly as for the M=1 plan.

use crate::tuning::Tuning;

use super::geometry::AffineFormat;

/// The GEMM row tiles the shaders are compiled for, narrow first — the
/// same three `TILE_M` declares in `kernels-metal/src/axes.rs`.
pub const QMM_BMS: [u32; 3] = [16, 32, 64];

/// The split-K GEMM's fixed column tile.
pub const QMM_SPLIT_BN: u32 = 32;

/// The routed GEMM's row tiles — the widths the expert sort pads to.
pub const MOE_TILE_WIDTHS: [u32; 3] = [16, 32, 64];

fn tile(bm: u32, bn: u32) -> String {
    format!("_bm_{bm}_bn_{bn}")
}

/// Which table slot a compiled PSO lands in. Array slots carry their rung
/// indices: `bm` indexes [`QMM_BMS`] (or [`MOE_TILE_WIDTHS`] for the routed
/// form), `bn` counts 16/32/64 column tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)] // the slot names ARE the C++ field names; each is documented there
pub enum MbSlot {
    EmbedMb,
    RopeMb,
    KvAppendPaged,
    SdpaPaged,
    SdpaPagedD512,
    SdpaPagedTiled,
    SdpaPagedTiledD512,
    SdpaPagedTiledStrided,
    GdnPrepSlotted,
    GdnRecurrentSlotted,
    GdnPrepPrefill,
    GdnCorePrefill,
    GatedRmsStrided,
    QmmT { bm: usize, bn: usize },
    QmmTFp16 { bm: usize, bn: usize },
    QmmTResidual { bm: usize, bn: usize },
    QmmTResidualFp16 { bm: usize, bn: usize },
    QmmTBias { bm: usize, bn: usize },
    QmmTBiasFp16 { bm: usize, bn: usize },
    QmmRouted { width: usize, bn: usize },
    QmmTSplitk { bm: usize },
    QmmTSplitkF32 { bm: usize },
    QmmTSplitkFp16 { bm: usize },
    QmmTSplitkFp16F32 { bm: usize },
    QmmCastBf16F16,
    QmmSplitkReduce,
    QmmSplitkReduceF32,
    QmmTStrided { bm: usize },
    QmmTStridedResidual { bm: usize },
    QmmTStridedFp16 { bm: usize },
    QmmTStridedFp16Residual { bm: usize },
    QmmTStridedCast,
    QmvWideStrided,
    RmsStrided,
    RmsStridedHead,
    RopeStrided,
    SiluMulStrided,
    ResidualAddStrided,
    SharedExpertCombineStrided,
}

/// One entrypoint the plan wants compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MbRequest {
    /// Where the compiled PSO goes.
    pub slot: MbSlot,
    /// The shader file, relative to the kernels directory.
    pub file: &'static str,
    /// The entrypoint name.
    pub entry: String,
}

/// Which extensions the family's multibatch step asks for.
/// (`MultiBatchPsoFeatures`, field for field.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MbFeatures {
    /// gemma4's 512-wide full-attention heads.
    pub d512: bool,
    /// The 256-wide paged SDPA pair.
    pub sdpa_d256: bool,
    /// The GDN prep/recurrent kernels, slotted and prefill.
    pub gdn: bool,
    /// GEMMs with the residual add folded into the store.
    pub residual: bool,
    /// GEMMs with a Linear's bias broadcast down the tile — gpt-oss biases
    /// every projection, so without it the batched path is a GEMM plus a
    /// dispatch that rewrites the whole output to add one vector.
    pub bias: bool,
    /// The mixture's batched GEMM, weight stack indexed per TILE.
    pub routed: bool,
    /// The split-K decode GEMM pair and its reduces.
    pub splitk: bool,
    /// The prefill's strided kernels: one dispatch per prompt, not per row.
    pub strided: bool,
    /// FP16-staged GEMM tiles (M1's native matrix path).
    pub fp16_precast: bool,
    /// The strided GEMM's FP16 forms.
    pub fp16_strided: bool,
}

/// Lay out the multibatch compile list.
///
/// `quant` has no default for the reason the M=1 loader states: when width
/// and group were two defaulted ints, call sites passed one and let the
/// other fall back, which compiled, bound, dispatched, and answered
/// wrongly. `tuning` decides two names: the GDN prefill scan's unroll, and
/// whether the routed GEMM takes its FP16 form — a NAME choice only, since
/// both forms share buffers, grid and `tile_expert` contract. (The one
/// family that must not take FP16 routing is llama, whose routed top-k
/// moved under it; llama builds its own routed table and never reaches
/// this.)
#[must_use]
pub fn plan_multibatch_psos(
    quant: AffineFormat,
    features: MbFeatures,
    tuning: &Tuning,
) -> Vec<MbRequest> {
    let affine = quant.kernel_suffix();
    let qmm = "quant/qmm_t.metal";
    // The C++ writes this gate out at each site; it is one fact — the FP16
    // staging kernels are instantiated only for the g64/b4 body format.
    let fp16_body = quant.group == 64 && quant.bits == 4;
    let mut out = Vec::new();
    let mut want = |slot: MbSlot, file: &'static str, entry: String| {
        out.push(MbRequest { slot, file, entry });
    };

    for (bm, &bm_rows) in QMM_BMS.iter().enumerate() {
        for bn in 0..3usize {
            let bn_cols = 16u32 << bn;
            let at = |base: &str| format!("{base}{affine}{}", tile(bm_rows, bn_cols));
            want(MbSlot::QmmT { bm, bn }, qmm, at("affine_qmm_t"));
            if features.fp16_precast && fp16_body {
                want(
                    MbSlot::QmmTFp16 { bm, bn },
                    qmm,
                    at("affine_qmm_t_fp16_precast"),
                );
            }
            if features.residual {
                want(
                    MbSlot::QmmTResidual { bm, bn },
                    qmm,
                    at("affine_qmm_t_residual"),
                );
            }
            if features.residual && features.fp16_precast && fp16_body {
                want(
                    MbSlot::QmmTResidualFp16 { bm, bn },
                    qmm,
                    at("affine_qmm_t_residual_fp16_precast"),
                );
            }
            if features.bias {
                want(MbSlot::QmmTBias { bm, bn }, qmm, at("affine_qmm_t_bias"));
            }
            if features.bias && features.fp16_precast && fp16_body {
                want(
                    MbSlot::QmmTBiasFp16 { bm, bn },
                    qmm,
                    at("affine_qmm_t_bias_fp16_precast"),
                );
            }
        }
        if features.splitk {
            let split = |base: &str| format!("{base}{affine}{}", tile(bm_rows, QMM_SPLIT_BN));
            want(MbSlot::QmmTSplitk { bm }, qmm, split("affine_qmm_t_splitk"));
            want(
                MbSlot::QmmTSplitkF32 { bm },
                qmm,
                split("affine_qmm_t_splitk_f32"),
            );
            if features.fp16_precast && fp16_body {
                want(
                    MbSlot::QmmTSplitkFp16 { bm },
                    qmm,
                    split("affine_qmm_t_splitk_fp16_precast"),
                );
                want(
                    MbSlot::QmmTSplitkFp16F32 { bm },
                    qmm,
                    split("affine_qmm_t_splitk_fp16_precast_f32"),
                );
            }
        }
    }
    if features.fp16_precast && fp16_body {
        want(
            MbSlot::QmmCastBf16F16,
            qmm,
            "cast_qmm_input_bfloat16_to_float16".to_string(),
        );
    }
    if features.routed {
        // The row tile is the number the sort padded every expert's run to
        // — a tile that disagreed would read one expert's weights for
        // another's rows — so every padded width is compiled and the fire
        // picks.
        let routed = if tuning.fp16_qmm && fp16_body {
            "affine_qmm_t_routed_fp16"
        } else {
            "affine_qmm_t_routed"
        };
        for (width, &rows) in MOE_TILE_WIDTHS.iter().enumerate() {
            for bn in 0..3usize {
                want(
                    MbSlot::QmmRouted { width, bn },
                    qmm,
                    format!("{routed}{affine}{}", tile(rows, 16 << bn)),
                );
            }
        }
    }
    if features.splitk {
        want(
            MbSlot::QmmSplitkReduce,
            qmm,
            "qmm_splitk_reduce_bfloat16".to_string(),
        );
        want(
            MbSlot::QmmSplitkReduceF32,
            qmm,
            "qmm_splitk_reduce_f32_bfloat16".to_string(),
        );
    }
    // The strided rungs' column tile is fixed at 32.
    if features.strided {
        for (bm, &bm_rows) in QMM_BMS.iter().enumerate() {
            want(
                MbSlot::QmmTStrided { bm },
                qmm,
                format!("affine_qmm_t_strided{affine}{}", tile(bm_rows, 32)),
            );
            if features.residual {
                want(
                    MbSlot::QmmTStridedResidual { bm },
                    qmm,
                    format!("affine_qmm_t_strided_residual{affine}{}", tile(bm_rows, 32)),
                );
            }
        }
    }
    if features.fp16_strided && fp16_body {
        for (bm, &bm_rows) in QMM_BMS.iter().enumerate() {
            want(
                MbSlot::QmmTStridedFp16 { bm },
                qmm,
                format!(
                    "affine_qmm_t_strided_fp16_precast{affine}{}",
                    tile(bm_rows, 32)
                ),
            );
            want(
                MbSlot::QmmTStridedFp16Residual { bm },
                qmm,
                format!(
                    "affine_qmm_t_strided_fp16_precast_residual{affine}{}",
                    tile(bm_rows, 32)
                ),
            );
        }
        want(
            MbSlot::QmmTStridedCast,
            qmm,
            "cast_qmm_input_strided_bfloat16_to_float16".to_string(),
        );
    }
    // The wide matvec, asked for the CHECKPOINT's own format rather than
    // only the 4-bit one the fp16 block happens to also want. Gating it on
    // `fp16_strided` tied it to a feature it has nothing to do with, which
    // left an alt-quant kind with no batched shape at all.
    if features.strided && quant.group == 64 && (quant.bits == 4 || quant.bits == 8) {
        want(
            MbSlot::QmvWideStrided,
            qmm,
            format!("affine_qmv_wide_strided{affine}_v_4_kl_8"),
        );
    }
    want(
        MbSlot::EmbedMb,
        "layout/embed_gather.metal",
        format!("embed_gather_mb_4bit{affine}"),
    );
    want(
        MbSlot::RopeMb,
        "rope/neox.metal",
        "neox_mb_bfloat16".to_string(),
    );
    want(
        MbSlot::KvAppendPaged,
        "attn/kv_write.metal",
        "kv_append_paged_bfloat16".to_string(),
    );
    if features.sdpa_d256 {
        want(
            MbSlot::SdpaPaged,
            "attn/sdpa_paged.metal",
            "sdpa_paged_decode_bfloat16_d_256".to_string(),
        );
        want(
            MbSlot::SdpaPagedTiled,
            "attn/sdpa_paged.metal",
            "sdpa_paged_tiled_bfloat16_d_256".to_string(),
        );
    }
    if features.d512 {
        want(
            MbSlot::SdpaPagedD512,
            "attn/sdpa_paged.metal",
            "sdpa_paged_decode_bfloat16_d_512".to_string(),
        );
        want(
            MbSlot::SdpaPagedTiledD512,
            "attn/sdpa_paged.metal",
            "sdpa_paged_tiled_bfloat16_d_512".to_string(),
        );
    }
    if features.gdn {
        want(
            MbSlot::GdnPrepSlotted,
            "ssm/gdn_prep.metal",
            "gdn_prep_slotted_bfloat16".to_string(),
        );
        want(
            MbSlot::GdnRecurrentSlotted,
            "ssm/gdn_prep.metal",
            "gdn_core_recurrent_slotted_bfloat16".to_string(),
        );
        want(
            MbSlot::GdnPrepPrefill,
            "ssm/gdn_prep.metal",
            "gdn_prep_prefill_bfloat16".to_string(),
        );
        want(
            MbSlot::GdnCorePrefill,
            "ssm/gdn_prep.metal",
            format!(
                "gdn_core_recurrent_prefill_bfloat16_l_{}_v_{}",
                tuning.gdn_scan_lanes, tuning.gdn_scan_rows
            ),
        );
        want(
            MbSlot::GatedRmsStrided,
            "norm/gated_rms.metal",
            "gated_rms_strided_bfloat16".to_string(),
        );
    }
    if features.strided {
        want(
            MbSlot::RmsStrided,
            "norm/rms.metal",
            "rms_strided_row_bfloat16".to_string(),
        );
        want(
            MbSlot::RmsStridedHead,
            "norm/rms.metal",
            "rms_strided_head_row_bfloat16".to_string(),
        );
        want(
            MbSlot::RopeStrided,
            "rope/neox.metal",
            "neox_strided_bfloat16".to_string(),
        );
        want(
            MbSlot::SiluMulStrided,
            "mlp/gated.metal",
            "silu_mul_strided_bfloat16".to_string(),
        );
        // The two per-token elementwise lines a prefill was still
        // dispatching once per row: on a 40-layer model they were ~80k
        // dispatches in a 1024-row fire.
        want(
            MbSlot::ResidualAddStrided,
            "norm/residual_add.metal",
            "residual_add_strided_bfloat16".to_string(),
        );
        want(
            MbSlot::SharedExpertCombineStrided,
            "moe/route.metal",
            "shared_expert_combine_strided".to_string(),
        );
        if features.sdpa_d256 {
            want(
                MbSlot::SdpaPagedTiledStrided,
                "attn/sdpa_paged.metal",
                "sdpa_paged_tiled_strided_bfloat16_d_256".to_string(),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_features() -> MbFeatures {
        MbFeatures {
            d512: true,
            sdpa_d256: true,
            gdn: true,
            residual: true,
            bias: true,
            routed: true,
            splitk: true,
            strided: true,
            fp16_precast: true,
            fp16_strided: true,
        }
    }

    #[test]
    fn every_emittable_name_is_a_shader_entrypoint_and_its_file_exists() {
        let plan = plan_multibatch_psos(AffineFormat::G64_B4, all_features(), &Tuning::default());
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        for request in &plan {
            assert!(
                table.contains(&request.entry),
                "{} is not in the signature table (slot {:?})",
                request.entry,
                request.slot
            );
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("crates/")
                .join("kernels-metal/kernels")
                .join(request.file);
            assert!(path.exists(), "{} does not exist", path.display());
        }
    }

    #[test]
    fn the_fp16_staging_names_exist_only_for_the_body_format() {
        let g32b8 = AffineFormat { bits: 8, group: 32 };
        let plan = plan_multibatch_psos(g32b8, all_features(), &Tuning::default());
        assert!(
            plan.iter()
                .all(|r| !matches!(r.slot, MbSlot::QmmTFp16 { .. } | MbSlot::QmmCastBf16F16)),
            "fp16 staging is instantiated for g64/b4 alone"
        );
        // And the wide matvec follows the checkpoint's format, not fp16's:
        // g64/b8 gets a batched shape even though no fp16 kernel exists
        // for it.
        let g64b8 = AffineFormat { bits: 8, group: 64 };
        let plan = plan_multibatch_psos(g64b8, all_features(), &Tuning::default());
        assert!(
            plan.iter()
                .any(|r| matches!(r.slot, MbSlot::QmvWideStrided)),
            "an alt-quant kind must not be left with no batched shape"
        );
    }

    #[test]
    fn the_routed_form_is_a_name_choice_decided_by_tuning() {
        let mut tuning = Tuning {
            fp16_qmm: true,
            ..Tuning::default()
        };
        let fp16 = plan_multibatch_psos(AffineFormat::G64_B4, all_features(), &tuning);
        let routed_name = |plan: &[MbRequest]| {
            plan.iter()
                .find(|r| matches!(r.slot, MbSlot::QmmRouted { width: 0, bn: 0 }))
                .unwrap()
                .entry
                .clone()
        };
        assert!(routed_name(&fp16).starts_with("affine_qmm_t_routed_fp16"));
        tuning.fp16_qmm = false;
        let bf16 = plan_multibatch_psos(AffineFormat::G64_B4, all_features(), &tuning);
        assert!(routed_name(&bf16).starts_with("affine_qmm_t_routed_bfloat16"));
    }

    #[test]
    fn a_minimal_step_still_gets_the_model_independent_substrate() {
        let plan = plan_multibatch_psos(
            AffineFormat::G64_B4,
            MbFeatures::default(),
            &Tuning::default(),
        );
        for slot in [MbSlot::EmbedMb, MbSlot::RopeMb, MbSlot::KvAppendPaged] {
            assert!(
                plan.iter().any(|r| r.slot == slot),
                "{slot:?} is the substrate MultiBatchPsos::valid() checks"
            );
        }
        // The base GEMM table always compiles; the extensions do not.
        assert_eq!(
            plan.iter()
                .filter(|r| matches!(r.slot, MbSlot::QmmT { .. }))
                .count(),
            9
        );
        assert!(
            plan.iter()
                .all(|r| !matches!(r.slot, MbSlot::QmmRouted { .. })),
        );
    }
}
