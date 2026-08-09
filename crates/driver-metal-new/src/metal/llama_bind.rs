//! The llama families' device consts walk.
//!
//! Every slot a dispatch declares is bound, including ones the M=1
//! kernel never reads — an unbound slot is not a crash: the argument
//! table holds whatever was there last, so a stride comes back as a
//! pointer or a row count out of uninitialised memory.
//!
//! The rope is the walk's sharpest edge: the freq-table form and the
//! base form are a DIFFERENT ABI, not a different value. Buffer 3 means
//! `inv_freq` (a pointer) to one and `base` (a float) to the other, so
//! the two arms bind nothing in common — and the base is `log2(theta)`,
//! not theta: bound as theta, `exp2(-d·5e5)` underflows to zero for
//! every frequency but the first, which is a rope that does nothing.

use crate::Result;
use crate::batch::{
    Dispatch, ExpertCombineParams, Kernel, LlamaGeometry, MoeRouteParams, RmsParams, RouterParams,
    RowGatherParams, llama_qmv_kn, llama3_inv_freq, sorted_rows,
};
use crate::region::Region as _;
use crate::tuning::Tuning;

use super::bind::ConstSlots;
use super::context::Context;
use super::gptoss_bind::slot as freqs_slot;
use super::handle::Handle;
use super::ring::allocate;
use super::tables::Tables;

/// Bind every llama constant, by ordinal. Returns the llama3 frequency
/// table's buffer when the geometry carries one — the rope tables hold
/// its GPU address, so the handle must outlive them — and `None` for a
/// checkpoint whose frequencies really are a geometric series.
///
/// ONE kind-driven walk for the ring and the paged fire both, as with
/// gpt-oss: the DAG's kinds already say which ABI each dispatch is on.
///
/// # Errors
///
/// An allocation refusal for the frequency table or a constant slot.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn bind_llama_consts(
    context: &Context,
    tables: &mut Tables,
    consts: &mut ConstSlots,
    dag: &[Dispatch],
    g: &LlamaGeometry,
    tuning: &Tuning,
    max_ctx: u32,
    rows: u32,
    head_rows: u32,
) -> Result<Option<Handle>> {
    let freqs = if g.rope_freq_table {
        let inv_freq = llama3_inv_freq(g);
        let table = allocate(context, (inv_freq.len() * 4) as u64, "llama3 rope table")?;
        let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
        // SAFETY: freshly allocated; the GPU has no reference yet.
        unsafe { table.write(0, &bytes)? };
        Some(table)
    } else {
        None
    };

    let r = rows.max(1);
    let s = if head_rows == 0 { r } else { head_rows.min(r) };
    let k = g.experts_per_token.max(1);
    let pairs = r * k;
    let tile = tuning.moe_tile_rows(pairs, g.n_experts);
    let sorted = u32::try_from(sorted_rows(pairs, g.n_experts, tile)).expect("a sort is bounded");
    let head_stride = u64::from(max_ctx) * u64::from(g.head_dim);
    let seq_stride = u64::from(g.head_dim);
    let gqa = g.n_q_heads.checked_div(g.n_kv_heads).unwrap_or(1) as i32;
    // 1/sqrt(head_dim), unlike gemma4 — which folds its scale into the
    // q-norm weights and so passes 1.0. Llama does not.
    let sdpa_scale = 1.0 / (g.head_dim as f32).sqrt();

    for d in dag {
        let ord = d.ordinal;
        let kind = d.kind;
        let kn = llama_qmv_kn(kind, g);
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
            // Only the routed matvec carries the GoQmv stride slots; the
            // dense form is a different kernel with no such ABI. The
            // sorted stack is one row per (token, slot) pair, which
            // collapses all three to the dense case: no slot axis, a row
            // pitch that is the input width, one expert per row — the
            // sort is what made the pair axis disappear.
            if matches!(
                kind,
                Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown
            ) {
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
            }
            continue;
        }
        match kind {
            Kernel::Rms | Kernel::FfnRms | Kernel::FinalRms => {
                // `plus_one` is 0: llama applies the learned gain
                // directly, where gemma stores `w - 1`. Wrong here is
                // every norm off by one, which still produces text.
                let params = RmsParams {
                    eps: g.eps,
                    axis_size: g.hidden,
                    w_stride: 1,
                    plus_one: 0,
                    gain: 1.0,
                };
                consts.bind(context, tables, ord, super::bind::slot::RMS_PARAMS, &params)?;
            }
            Kernel::QNorm | Kernel::KNorm => {
                // Per HEAD: the axis is head_dim, not the projection.
                // Normalising across heads would mix them — wrong
                // attention, not a crash.
                let params = RmsParams {
                    eps: g.eps,
                    axis_size: g.head_dim,
                    w_stride: 1,
                    plus_one: 0,
                    gain: 1.0,
                };
                consts.bind(context, tables, ord, super::bind::slot::RMS_PARAMS, &params)?;
            }
            Kernel::Rope | Kernel::RopeK => {
                if let Some(table) = &freqs {
                    consts.bind(context, tables, ord, freqs_slot::ROPE_FREQS_SCALE, &1.0f32)?;
                    tables.bind_address(
                        context,
                        ord,
                        freqs_slot::ROPE_FREQS_INV_FREQ as usize,
                        table.gpu_address(),
                    )?;
                    consts.bind(
                        context,
                        tables,
                        ord,
                        freqs_slot::ROPE_FREQS_HEAD_DIM,
                        &(g.head_dim as i32),
                    )?;
                    // llama3 has no attention-temperature correction;
                    // the kernel multiplies by mscale unconditionally.
                    consts.bind(context, tables, ord, freqs_slot::ROPE_FREQS_MSCALE, &1.0f32)?;
                    let stride = if kind == Kernel::Rope {
                        g.q_width()
                    } else {
                        g.kv_width()
                    };
                    consts.bind(
                        context,
                        tables,
                        ord,
                        freqs_slot::ROPE_FREQS_ROW_STRIDE,
                        &(stride as i32),
                    )?;
                } else {
                    // The linear position divisor rides Scale.
                    let scale = if g.rope_scale != 0.0 {
                        1.0 / g.rope_scale
                    } else {
                        1.0
                    };
                    consts.bind(context, tables, ord, super::bind::slot::ROPE_SCALE, &scale)?;
                    consts.bind(
                        context,
                        tables,
                        ord,
                        super::bind::slot::ROPE_BASE,
                        &g.rope_theta.log2(),
                    )?;
                    consts.bind(
                        context,
                        tables,
                        ord,
                        super::bind::slot::ROPE_HEAD_DIM,
                        &(g.head_dim as i32),
                    )?;
                }
            }
            Kernel::Sdpa => {
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_GQA_FACTOR,
                    &gqa,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_SCALE,
                    &sdpa_scale,
                )?;
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
                    &seq_stride,
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
                    &seq_stride,
                )?;
            }
            Kernel::SdpaPaged => {
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
                    &(g.n_kv_heads as i32),
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_SCALE,
                    &sdpa_scale,
                )?;
                // Full attention, but the window is still BOUND: one
                // paged kernel serves the sliding families and this one,
                // and <= 0 is how "no window" is spelled. Unbound, the
                // slot holds a window nobody asked for, which truncates
                // attention rather than crashing.
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_WINDOW,
                    &0i32,
                )?;
                consts.bind(
                    context,
                    tables,
                    ord,
                    super::bind::slot::SDPA_PAGED_ROWS,
                    &(r as i32),
                )?;
            }
            Kernel::KvAppend | Kernel::KvAppendPaged => {
                let (dim, khs, kss, page, heads, src) = if kind == Kernel::KvAppend {
                    (
                        super::bind::slot::KV_APPEND_HEAD_DIM,
                        super::bind::slot::KV_APPEND_K_HEAD_STRIDE,
                        super::bind::slot::KV_APPEND_K_SEQ_STRIDE,
                        None,
                        None,
                        None,
                    )
                } else {
                    (
                        super::bind::slot::KV_APPEND_PAGED_HEAD_DIM,
                        super::bind::slot::KV_APPEND_PAGED_K_HEAD_STRIDE,
                        super::bind::slot::KV_APPEND_PAGED_K_SEQ_STRIDE,
                        Some(super::bind::slot::KV_APPEND_PAGED_PAGE_SIZE),
                        Some(super::bind::slot::KV_APPEND_PAGED_N_KV_HEADS),
                        Some(super::bind::slot::KV_APPEND_PAGED_SRC_ROW_STRIDE),
                    )
                };
                consts.bind(context, tables, ord, dim, &(g.head_dim as i32))?;
                consts.bind(context, tables, ord, khs, &head_stride)?;
                consts.bind(context, tables, ord, kss, &seq_stride)?;
                if let (Some(page), Some(heads), Some(src)) = (page, heads, src) {
                    consts.bind(context, tables, ord, page, &(g.kv_page_size as i32))?;
                    consts.bind(context, tables, ord, heads, &(g.n_kv_heads as i32))?;
                    // The C++ llama walk leaves this slot UNBOUND, and
                    // the kernel declares and reads it (`src_row_stride
                    // > 0 ? …`): the very class this walk's C++ header
                    // opens by naming. Packed rows are spelled 0.
                    consts.bind(context, tables, ord, src, &0i32)?;
                }
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
                // One struct for both, so the sort's padding and the
                // gather's bounds cannot disagree.
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
            Kernel::LlMoeCombine => {
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
            // The row WIDTH: `embed_gather_4bit` derives the packed row
            // pitch and the group count from it, so a stale value does
            // not truncate the embedding — it reads a different row.
            Kernel::EmbedGather | Kernel::EmbedUntied => {
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
