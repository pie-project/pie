//! Gemma 4's device consts walk — the one that cannot be a switch on
//! the kind alone: four of its constants depend on the LAYER (the head
//! width, the rope base, the rotated fraction, the MLP width), so the
//! walk asks the geometry per dispatch. `Dispatch::layer` is what makes
//! that possible.
//!
//! Two single-value findings the C++ paid for and this walk keeps:
//! the router norm's gain is `1/√hidden` and must NOT share a case with
//! the other norms (without the root every logit is 53× too large,
//! which SURVIVES the top-k — a uniform positive scale does not reorder
//! — and comes out of the softmax as a one-hot: the mixture degenerates
//! to its single best expert); and the attention scale is 1.0, not
//! `1/√head_dim` — gemma folds the scale into the q-norm's learned
//! weights, and dividing again scales an already-scaled query.

use crate::Result;
use crate::batch::{
    Dispatch, GegluParams, Gemma4Geometry, Kernel, LayerScalarParams, MoeRouteParams,
    PleCombineParams, RmsParams, RouterParams, RowGatherParams, SoftcapParams, VNormParams,
    gemma4_qmv_kn, sorted_rows,
};
use crate::tuning::Tuning;

use super::bind::ConstSlots;
use super::context::Context;
use super::tables::Tables;

/// The bind ordinals this walk writes beyond the shared slot module.
#[allow(missing_docs)] // names are the C++ bind:: constants, one-to-one
pub mod slot {
    pub const QMM_M: u8 = 7;
    pub const EMBED_SCALE: u8 = 6;
    pub const SDPA_SLIDING_WINDOW: u8 = 11;
    pub const SDPA_SLIDING_Q_ROW_STRIDE: u8 = 12;
    pub const SDPA_SLIDING_O_ROW_STRIDE: u8 = 13;
    pub const VNORM_PARAMS: u8 = 2;
    pub const GEGLU_PARAMS: u8 = 3;
    pub const LAYER_SCALAR_PARAMS: u8 = 3;
    pub const PLE_COMBINE_PARAMS: u8 = 3;
    pub const SOFTCAP_PARAMS: u8 = 2;
}

/// Bind every gemma4 constant, by ordinal — the M=1 walk (`rows`/
/// `head_rows` carried for the shapes that scale, as in the other
/// families' walks).
///
/// # Errors
///
/// An allocation refusal for a constant slot.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn bind_gemma4_consts(
    context: &Context,
    tables: &mut Tables,
    consts: &mut ConstSlots,
    dag: &[Dispatch],
    g: &Gemma4Geometry,
    tuning: &Tuning,
    max_ctx: u32,
    rows: u32,
    head_rows: u32,
) -> Result<()> {
    let r = rows.max(1);
    let s = if head_rows == 0 { r } else { head_rows.min(r) };
    let k = g.experts_per_token.max(1);
    let pairs = r * k;
    let tile = if g.is_moe() {
        tuning.moe_tile_rows(pairs, g.n_experts)
    } else {
        1
    };
    let sorted = if g.is_moe() {
        u32::try_from(sorted_rows(pairs, g.n_experts, tile)).expect("a sort is bounded")
    } else {
        r
    };
    let plus_one = 0u32; // gemma4 stores PLAIN weights; (1+w) is an earlier gemma's
    let rms = |axis: u32| RmsParams {
        eps: g.eps,
        axis_size: axis,
        w_stride: 1,
        plus_one,
        gain: 1.0,
    };

    for d in dag {
        let ord = d.ordinal;
        let kind = d.kind;
        let layer = d.layer;
        let hd = layer.map_or(g.head_dim, |l| g.head_dim_of(l));
        let head_stride = u64::from(max_ctx) * u64::from(hd);

        let kn = gemma4_qmv_kn(kind, g, layer);
        if kn.n != 0 {
            consts.bind(
                context,
                tables,
                ord,
                super::bind::slot::QMV_K,
                &(kn.k as i32),
            )?;
            consts.bind(
                context,
                tables,
                ord,
                super::bind::slot::QMV_N,
                &(kn.n as i32),
            )?;
            // Every matvec here is the tail form, which reads the routed
            // stride slots with the expert axis switched off — so every
            // projection binds them, rather than this walk re-deriving
            // the pipeline choice the plan already made.
            consts.bind(
                context,
                tables,
                ord,
                super::bind::slot::GO_QMV_X_SLOT_STRIDE,
                &0i32,
            )?;
            consts.bind(
                context,
                tables,
                ord,
                super::bind::slot::GO_QMV_X_ROW_STRIDE,
                &(kn.k as i32),
            )?;
            consts.bind(
                context,
                tables,
                ord,
                super::bind::slot::GO_QMV_SLOTS_PER_ROW,
                &1i32,
            )?;
            // The GEMM shares Qmv's ordinals and appends M. Bound
            // unconditionally: at rows==1 the matvec never reads slot 7,
            // and an unbound slot on the prefill path is a row count out
            // of uninitialized memory.
            let routed = matches!(
                kind,
                Kernel::G4ExpertGate | Kernel::G4ExpertUp | Kernel::G4ExpertDown
            );
            let m = if routed {
                sorted
            } else if matches!(kind, Kernel::QmvLmHead | Kernel::LmHeadUntied) {
                s
            } else {
                r
            };
            consts.bind(context, tables, ord, slot::QMM_M, &(m as i32))?;
            continue;
        }
        match kind {
            Kernel::Rms
            | Kernel::G4FfnPreNorm
            | Kernel::FinalRms
            | Kernel::G4AttnPostResidual
            | Kernel::G4FfnPostResidual
            | Kernel::G4PleResidualScaled
            | Kernel::G4MoeNorm
            | Kernel::G4DenseBranchNorm
            | Kernel::G4MoeBranchNorm => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::RMS_PARAMS,
                    &rms(g.hidden),
                )?;
            }
            // The ONLY norm with a gain; it must not share a case with
            // the others — see the module docs for what sharing cost.
            Kernel::G4RouterNorm => {
                let mut params = rms(g.hidden);
                params.gain = 1.0 / (g.hidden as f32).sqrt();
                consts.bind(context, tables, ord, super::bind::slot::RMS_PARAMS, &params)?;
            }
            Kernel::QNorm | Kernel::KNorm => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::RMS_PARAMS,
                    &rms(hd),
                )?;
            }
            // The two PLE norms are NOT the same width: the projection
            // norm runs ple_dim-wide rows on the table; the post-input
            // norm is hidden-wide, back in the stream (it is the fused
            // PleResidualScaled above).
            Kernel::G4PleProjNorm => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::RMS_PARAMS,
                    &rms(g.per_layer_emb_dim),
                )?;
            }
            Kernel::G4VNorm | Kernel::G4VNormFromK => {
                let params = VNormParams {
                    eps: g.eps,
                    axis_size: hd,
                };
                consts.bind(context, tables, ord, slot::VNORM_PARAMS, &params)?;
            }
            Kernel::Rope | Kernel::RopeK => {
                consts.bind(context, tables, ord, super::bind::slot::ROPE_SCALE, &1.0f32)?;
                // log2(theta), not theta — bound as theta, exp2(-d·1e6)
                // is 0 for every frequency but the first.
                let theta = layer.map_or(g.rope_theta_global, |l| g.rope_theta_of(l));
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROPE_BASE,
                    &theta.log2(),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROPE_HEAD_DIM,
                    &(hd as i32),
                )?;
            }
            // Both attention types run an `sdpa_vector_decode_swa`
            // instantiation — they differ by head width, not kernel — so
            // BOTH read the window slot. Binding it only for sliding
            // layers left layers 4, 9, 14… reading an unbound window:
            // wrong attention, not a crash. 0 attends all.
            Kernel::Sdpa | Kernel::G4SdpaSliding => {
                let nkv = layer.map_or(g.n_kv_heads, |l| g.n_kv_heads_of(l));
                let gqa = g.n_q_heads.checked_div(nkv).unwrap_or(1) as i32;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_GQA_FACTOR,
                    &gqa,
                )?;
                // 1.0: the scale lives in the q-norm's learned weights.
                consts.bind(context, tables, ord, super::bind::slot::SDPA_SCALE, &1.0f32)?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_K_SEQ_STRIDE,
                    &u64::from(hd),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_V_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_V_SEQ_STRIDE,
                    &u64::from(hd),
                )?;
                let window = if kind == Kernel::G4SdpaSliding {
                    g.sliding_window as i32
                } else {
                    0
                };
                consts.bind(context, tables, ord, slot::SDPA_SLIDING_WINDOW, &window)?;
                let row = (g.n_q_heads * hd) as i32;
                consts.bind(context, tables, ord, slot::SDPA_SLIDING_Q_ROW_STRIDE, &row)?;
                consts.bind(context, tables, ord, slot::SDPA_SLIDING_O_ROW_STRIDE, &row)?;
            }
            Kernel::SdpaPaged => {
                let nkv = layer.map_or(g.n_kv_heads, |l| g.n_kv_heads_of(l));
                let gqa = g.n_q_heads.checked_div(nkv).unwrap_or(1) as i32;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_GQA_FACTOR,
                    &gqa,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_PAGE_SIZE,
                    &(g.kv_page_size as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_N_KV_HEADS,
                    &(nkv as i32),
                )?;
                // 1.0: folded into the q-norm weights, as on the ring.
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_SCALE,
                    &1.0f32,
                )?;
                // The kind no longer carries the attention type; the
                // geometry answers per layer. 0 attends all.
                let window = if layer.is_some_and(|l| g.is_sliding(l)) {
                    g.sliding_window as i32
                } else {
                    0
                };
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_WINDOW,
                    &window,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_ROWS,
                    &(r as i32),
                )?;
            }
            Kernel::KvAppendPaged => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_HEAD_DIM,
                    &(hd as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_K_SEQ_STRIDE,
                    &u64::from(hd),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_PAGE_SIZE,
                    &(g.kv_page_size as i32),
                )?;
                let nkv = layer.map_or(g.n_kv_heads, |l| g.n_kv_heads_of(l));
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_N_KV_HEADS,
                    &(nkv as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_SRC_ROW_STRIDE,
                    &0i32,
                )?;
            }
            Kernel::KvAppend => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_HEAD_DIM,
                    &(hd as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_K_SEQ_STRIDE,
                    &u64::from(hd),
                )?;
            }
            Kernel::G4RouterTopK => {
                let params = RouterParams {
                    n_experts: g.n_experts,
                    experts_per_token: k,
                    softmax_over_all: 0,
                    logits_pitch: 0,
                };
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROUTER_TOPK_PARAMS,
                    &params,
                )?;
            }
            Kernel::G4MoeSort | Kernel::G4MoeGather => {
                let params = MoeRouteParams {
                    n: pairs,
                    n_experts: g.n_experts,
                    experts_per_token: k,
                    tile_rows: tile,
                    padded: sorted,
                    width: g.hidden,
                    x_pitch: 0,
                };
                let index = if kind == Kernel::G4MoeSort {
                    super::bind::slot::MOE_SORT_PARAMS
                } else {
                    super::bind::slot::MOE_ROWS_PARAMS
                };
                consts.bind(context, tables, ord, index, &params)?;
            }
            Kernel::G4ExpertCombine => {
                let params = crate::batch::ExpertCombineParams {
                    width: g.hidden,
                    experts_per_token: k,
                    out_pitch: 0,
                };
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::EXPERT_COMBINE_PARAMS,
                    &params,
                )?;
            }
            // The sorted stack, flat: sized off the SORTED count, because
            // the sort pads every expert's run and the padding rows are
            // real rows of what this reads.
            Kernel::G4ExpertGeglu => {
                let params = GegluParams {
                    n: sorted * g.moe_intermediate,
                };
                consts.bind(context, tables, ord, slot::GEGLU_PARAMS, &params)?;
            }
            Kernel::G4Geglu => {
                let width = layer.map_or(g.intermediate, |l| g.intermediate_of(l));
                let params = GegluParams { n: r * width };
                consts.bind(context, tables, ord, slot::GEGLU_PARAMS, &params)?;
            }
            // At M=1 the layer's slice of the table is a buffer offset,
            // so the flat params serve; the strided pitches are the
            // prefill's.
            Kernel::G4PleGeglu => {
                let params = GegluParams {
                    n: r * g.per_layer_emb_dim,
                };
                consts.bind(context, tables, ord, slot::GEGLU_PARAMS, &params)?;
            }
            Kernel::G4LayerScalar => {
                let params = LayerScalarParams { n: r * g.hidden };
                consts.bind(context, tables, ord, slot::LAYER_SCALAR_PARAMS, &params)?;
            }
            Kernel::G4PleCombine => {
                let params = PleCombineParams {
                    inv_sqrt2: std::f32::consts::FRAC_1_SQRT_2,
                    n: r * g.n_layers * g.per_layer_emb_dim,
                };
                consts.bind(context, tables, ord, slot::PLE_COMBINE_PARAMS, &params)?;
            }
            Kernel::G4Softcap => {
                let params = SoftcapParams {
                    cap: g.final_softcap,
                    n: s * g.vocab,
                };
                consts.bind(context, tables, ord, slot::SOFTCAP_PARAMS, &params)?;
            }
            // The embedding scale cannot fold into the table: the LM
            // head reads the same tied weights, UNSCALED.
            Kernel::EmbedGather => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::EMBED_HIDDEN,
                    &(g.hidden as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::EMBED_SCALE,
                    &(g.hidden as f32).sqrt(),
                )?;
            }
            Kernel::G4PleTokenGather => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::EMBED_HIDDEN,
                    &((g.n_layers * g.per_layer_emb_dim) as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::EMBED_SCALE,
                    &(g.per_layer_emb_dim as f32).sqrt(),
                )?;
            }
            Kernel::G4RowGather => {
                let params = RowGatherParams {
                    width: g.hidden,
                    rows: s,
                };
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::ROW_GATHER_PARAMS,
                    &params,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}
