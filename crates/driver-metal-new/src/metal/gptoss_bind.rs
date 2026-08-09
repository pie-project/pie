//! GPT-OSS's device consts walk: the YaRN table bound once, and the
//! family's own parameter blocks.
//!
//! The weight, state and IO binds REUSE the shared pass — the Go* kinds
//! carry their names in `weight_binds`, the sink attention reads the ring
//! at the plain SDPA's slots, and an all-full-attention geometry gives
//! every layer its KV pair — so what this module owns is only what the
//! shared consts walk would get WRONG for this family: the rope is a
//! frequencies TABLE (YaRN), not a base scalar; the attention carries a
//! window and a row stride; the mixture's params use this family's counts.

use crate::Result;
use crate::batch::{
    Dispatch, ExpertCombineParams, GptOssGeometry, Kernel, MoeRouteParams, RmsParams, RouterParams,
    RowGatherParams, SwiGluParams, gptoss_moe_sorted_rows, gptoss_qmv_kn, yarn_inv_freq,
    yarn_mscale,
};
use crate::region::Region as _;
use crate::tuning::Tuning;

use super::bind::ConstSlots;
use super::context::Context;
use super::handle::Handle;
use super::ring::allocate;
use super::tables::Tables;

/// The bind ordinals this walk writes (`decode_abi.hpp`'s `bind::`).
#[allow(missing_docs)] // names are the C++ constants, one-to-one
pub mod slot {
    pub const ROPE_FREQS_SCALE: u8 = 2;
    pub const ROPE_FREQS_INV_FREQ: u8 = 3;
    pub const ROPE_FREQS_HEAD_DIM: u8 = 4;
    pub const ROPE_FREQS_MSCALE: u8 = 5;
    pub const ROPE_FREQS_ROW_STRIDE: u8 = 6;
    pub const SDPA_SINK_GQA_FACTOR: u8 = 4;
    pub const SDPA_SINK_K_HEAD_STRIDE: u8 = 6;
    pub const SDPA_SINK_K_SEQ_STRIDE: u8 = 7;
    pub const SDPA_SINK_V_HEAD_STRIDE: u8 = 8;
    pub const SDPA_SINK_V_SEQ_STRIDE: u8 = 9;
    pub const SDPA_SINK_SCALE: u8 = 10;
    pub const SDPA_SINK_WINDOW: u8 = 11;
    pub const SDPA_SINK_Q_ROW_STRIDE: u8 = 12;
    pub const SDPA_SINK_O_ROW_STRIDE: u8 = 13;
    pub const GO_SWIGLU_PARAMS: u8 = 3;
}

/// Bind every gpt-oss constant, by ordinal. Returns the YaRN table's
/// buffer, which must stay alive as long as the tables reference it.
///
/// ONE walk for the ring and the paged fire both: the C++ took a `paged`
/// bool and re-decided per arm, but here the DAG's kinds already say
/// which ABI each dispatch is on, so the flag would be a second copy of
/// what the dispatch list states. `rows` is the fire's token count and
/// `head_rows` how many the sampler reads (0 = all); at `rows == 1` every
/// row-dependent value collapses to the M=1 constants this walk always
/// bound.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn bind_gptoss_consts(
    context: &Context,
    tables: &mut Tables,
    consts: &mut ConstSlots,
    dag: &[Dispatch],
    g: &GptOssGeometry,
    tuning: &Tuning,
    max_ctx: u32,
    rows: u32,
    head_rows: u32,
) -> Result<Handle> {
    // The YaRN table: computed once, one buffer serving every rope
    // dispatch in the model.
    let inv_freq = yarn_inv_freq(g);
    let freqs = allocate(context, (inv_freq.len() * 4) as u64, "yarn table")?;
    let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
    // SAFETY: freshly allocated; the GPU has no reference yet.
    unsafe { freqs.write(0, &bytes)? };
    let mscale = yarn_mscale(g);

    let head_stride = u64::from(max_ctx) * u64::from(g.head_dim);
    let seq_stride = u64::from(g.head_dim);
    let sdpa_scale = 1.0 / (g.head_dim as f32).sqrt();
    let k = g.experts_per_token;
    let r = rows.max(1);
    let s = if head_rows == 0 { r } else { head_rows.min(r) };
    // The SAME numbers the MB builder wrote into the launches: the pair
    // count, the tile the sort pads to, the padded stack. At one row the
    // tile is 1 and this is the pure grouping the decode always bound.
    let pairs = r * k;
    let tile = tuning.moe_tile_rows(pairs, g.n_experts);
    let sorted = gptoss_moe_sorted_rows(g, tuning, r);

    for d in dag {
        let ord = d.ordinal;
        let kind = d.kind;
        let kn = gptoss_qmv_kn(kind, g);
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
            let routed = matches!(
                kind,
                Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoExpertDown
            );
            if routed {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::GO_QMV_X_SLOT_STRIDE,
                    &0i32,
                )?;
            }
            // Bound for EVERY matvec, routed or not: at M=1 the row is 0
            // and neither is read, and an unbound constant on the M>1 path
            // is a stride out of uninitialized memory.
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
            continue;
        }
        match kind {
            Kernel::Rms | Kernel::FfnRms | Kernel::FinalRms => {
                let params = RmsParams {
                    eps: g.eps,
                    axis_size: g.hidden,
                    w_stride: 1,
                    plus_one: 0,
                    gain: 1.0,
                };
                consts.bind(context, tables, ord, super::bind::slot::RMS_PARAMS, &params)?;
            }
            Kernel::Rope | Kernel::RopeK => {
                consts.bind(context, tables, ord, slot::ROPE_FREQS_SCALE, &1.0f32)?;
                tables.bind_address(
                    context,
                    ord,
                    slot::ROPE_FREQS_INV_FREQ as usize,
                    freqs.gpu_address(),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::ROPE_FREQS_HEAD_DIM,
                    &(g.head_dim as i32),
                )?;
                consts.bind(context, tables, ord, slot::ROPE_FREQS_MSCALE, &mscale)?;
                // q and k have different head counts and share the kernel,
                // so its grid cannot supply the row pitch.
                let stride = if kind == Kernel::Rope {
                    g.q_dim()
                } else {
                    g.kv_dim()
                };
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::ROPE_FREQS_ROW_STRIDE,
                    &(stride as i32),
                )?;
            }
            Kernel::KvAppend => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_HEAD_DIM,
                    &(g.head_dim as i32),
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
                    &seq_stride,
                )?;
            }
            Kernel::KvAppendPaged => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_HEAD_DIM,
                    &(g.head_dim as i32),
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
                    &seq_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_PAGE_SIZE,
                    &(g.kv_page_size as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_N_KV_HEADS,
                    &(g.n_kv_heads as i32),
                )?;
                // Packed rows: the batched step lays k_new/v_new out as
                // [N, n_kv_heads, head_dim]. Explicit because an ordinal
                // the kernel DOES declare and nobody wrote is a source
                // pitch read out of whatever the table held — the wrong
                // rows appended, not a crash.
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::KV_APPEND_PAGED_SRC_ROW_STRIDE,
                    &0i32,
                )?;
            }
            // The paged ABI puts different meanings at the ring's slots
            // (`SdpaSink::N` is `SdpaPaged::PositionIds`, a length read as
            // a pointer), so it is bound as its own thing, not a subset.
            Kernel::GoSdpaSinkPaged => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_GQA_FACTOR,
                    &(g.gqa_factor() as i32),
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
                    &(g.n_kv_heads as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_SCALE,
                    &sdpa_scale,
                )?;
                let window = if g.is_sliding(d.layer.unwrap_or(0)) {
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
                // N, for the tiled pipeline's partial last tile. Bound
                // whether or not this fire tiles: the bind table is per
                // kind and the pipeline choice is per row count. Unbound,
                // the tiled kernel decides which rows exist from a stale
                // ordinal — wrong attention, not a crash.
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_ROWS,
                    &(r as i32),
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
            Kernel::GoSdpaSink => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_GQA_FACTOR,
                    &(g.gqa_factor() as i32),
                )?;
                // 1/sqrt(head_dim) on q; no q-norm to fold it into.
                consts.bind(context, tables, ord, slot::SDPA_SINK_SCALE, &sdpa_scale)?;
                // 0 attends all — the shared kernel reads the slot either way.
                let window = if g.is_sliding(d.layer.unwrap_or(0)) {
                    g.sliding_window as i32
                } else {
                    0
                };
                consts.bind(context, tables, ord, slot::SDPA_SINK_WINDOW, &window)?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_K_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_K_SEQ_STRIDE,
                    &seq_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_V_HEAD_STRIDE,
                    &head_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_V_SEQ_STRIDE,
                    &seq_stride,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_Q_ROW_STRIDE,
                    &(g.q_dim() as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    slot::SDPA_SINK_O_ROW_STRIDE,
                    &(g.q_dim() as i32),
                )?;
            }
            Kernel::GoRouterTopK => {
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
            Kernel::LlMoeSort | Kernel::LlMoeGather => {
                let params = MoeRouteParams {
                    n: pairs,
                    n_experts: g.n_experts,
                    experts_per_token: k,
                    tile_rows: tile,
                    padded: sorted,
                    width: g.hidden,
                    x_pitch: 0,
                };
                let index = if kind == Kernel::LlMoeSort {
                    super::bind::slot::MOE_SORT_PARAMS
                } else {
                    super::bind::slot::MOE_ROWS_PARAMS
                };
                consts.bind(context, tables, ord, index, &params)?;
            }
            Kernel::GoSwiGlu => {
                let params = SwiGluParams {
                    count: sorted * g.intermediate,
                    limit: g.swiglu_limit,
                    alpha: g.swiglu_alpha,
                };
                consts.bind(context, tables, ord, slot::GO_SWIGLU_PARAMS, &params)?;
            }
            Kernel::GoExpertCombine => {
                let params = ExpertCombineParams {
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
            Kernel::EmbedUntied => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::EMBED_HIDDEN,
                    &(g.hidden as i32),
                )?;
            }
            _ => {}
        }
    }
    Ok(freqs)
}
