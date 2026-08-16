//! What happens when a trace states one of `attn`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::attn::qkv_fused::{
    qkv_decode_qk_norm_rope_write_kv_bf16, qkv_packed_qk_norm_rope_vnorm_write_kv_bf16,
};
use kernels_cuda::attn::*;

use super::super::cx::Cx;
use super::Bound;

/// `attn::lse_log2_to_ln`
fn lse_log2_to_ln_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
    let Ok(n) = usize::try_from(elems) else {
        return Err(Refusal::Empty { what: "lse elements" });
    };
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    lse_log2_to_ln(&ctx, cx.arg_out(0)?.cast::<f32>(), n)
}

/// `attn::attention_sink_rescale_bf16`
fn attention_sink_rescale_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attention_sink_rescale::<bf16>(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.rows().count,
        cx.num_q_heads()?,
        cx.head_dim()?,
    )
}

/// `attn::attn_res_blend_bf16`
fn attn_res_blend_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let h = cx.out_width(0)?;
    let b = cx.in_width(1)? / h;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attn_res_blend::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_in(2)?.cast_const().cast::<bf16>(),
        cx.arg_in(3)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        b,
        h,
        cx.rows().count,
        cx.rms_eps()?,
    )
}

/// `attn::pad_head_dim_bf16`
fn pad_head_dim_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    let packed_width = cx.in_width(0)?;
    let num_heads = packed_width / head_dim;
    if num_heads <= 0 {
        return Err(Refusal::Narrow { what: "in_width(0)", at: i64::from(packed_width) });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    pad_head_dim::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        num_heads,
        head_dim,
        cx.out_width(0)? / num_heads,
    )
}

/// `attn::strip_head_dim_bf16`
fn strip_head_dim_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    let packed_width = cx.out_width(0)?;
    let num_heads = packed_width / head_dim;
    if num_heads <= 0 {
        return Err(Refusal::Narrow { what: "out_width(0)", at: i64::from(packed_width) });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    strip_head_dim::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        num_heads,
        head_dim,
        cx.in_width(0)? / num_heads,
    )
}

/// `attn::logit_softcap_bf16`
fn logit_softcap_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let cap = cx.final_logit_softcap()?;
    let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
    let Ok(n) = usize::try_from(elems) else {
        return Err(Refusal::Narrow { what: "logit elements", at: i64::from(elems) });
    };
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    logit_softcap::<bf16>(&ctx, cx.arg_out(0)?.cast::<bf16>(), cap, n)
}

/// `attn::kimi_split_q_b_bf16`
fn kimi_split_q_b_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
    let heads = param(0)?;
    let nope = param(1)?;
    let rope = param(2)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kimi_split_q_b::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.rows().count,
        heads,
        nope,
        rope,
    )
}

/// `attn::kimi_split_kv_a_norm_bf16`
fn kimi_split_kv_a_norm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kimi_split_kv_a_norm::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.out_width(1)?,
        cx.in_width(0)?,
        cx.rms_eps()?,
    )
}

/// `attn::dsa_index_topk_mask`
fn dsa_index_topk_mask_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dsa_index_topk_mask(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_in(2)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<u8>(),
        cx.rows().count,
        param(0)?,
        param(1)?,
        param(2)?,
    )
}

/// `attn::split_qkv_bf16_devwin`
fn split_qkv_devwin_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let win = cx.peel_window()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    split_qkv_bf16_devwin(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.arg_out(2)?.cast::<bf16>(),
        win.as_ptr().cast_const(),
        cx.rows().total,
        cx.out_width(0)?,
        cx.out_width(1)?,
    )
}

/// `attn::split_qkv_bf16`
///
/// The plain twin of [`split_qkv_devwin_arm`] below, and the difference is
/// where the token count comes from: this one is told, the device-window one
/// is handed a pointer to a window the DEVICE writes. Both split one packed
/// `[tokens, q_dim + 2 * kv_dim]` bank into three.
///
/// The body is `driver_internal`'s rather than a routine, because it takes
/// `*const c_void` where the crossed `_devwin` twin takes `*const bf16` --
/// the driver hands a `void*` and the module exists to hold exactly that
/// difference. Declaring the symbol was one change; this is the other, and
/// the registry entry that stood here said so in the words this replaces.
fn split_qkv_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kernels_cuda::driver_internal::split_qkv_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        cx.arg_out(0)?,
        cx.arg_out(1)?,
        cx.arg_out(2)?,
        cx.rows().count,
        cx.out_width(0)?,
        cx.out_width(1)?,
    )
}

/// `attn::combine_attn_outputs_bf16`
fn combine_attn_outputs_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
    let num_heads = param(0)?;
    let head_dim = param(1)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    combine_attn_outputs::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<bf16>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        num_heads,
        head_dim,
    )
}

/// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`
fn qkv_packed_post_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    if layer.head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    let num_q_heads = cx.out_width(0)? / layer.head_dim;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        layer.k_pages.cast::<bf16>(),
        layer.v_pages.cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.weight(1)?.cast_const().cast::<bf16>(),
        cx.positions()?,
        plan.kv_page_indices,
        plan.kv_page_indptr,
        plan.kv_last_page_lens,
        plan.row_valid,
        cx.rows().count,
        num_q_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.page_size,
        layer.hnd,
        cx.theta()?,
        cx.rms_eps()?,
    )
}

/// `attn::write_kv_to_pages`
fn write_kv_to_pages_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kv_paged::write_kv_to_pages(
        &ctx,
        &layer,
        cx.arg_in(0)?.cast::<bf16>().cast_const(),
        cx.arg_in(1)?.cast::<bf16>().cast_const(),
        plan.qo_indptr,
        plan.kv_page_indices,
        plan.kv_page_indptr,
        plan.kv_last_page_lens,
        cx.rows().count,
        plan.requests,
        plan.row_valid,
        cx.first_token()?,
    )
}

/// `attn::write_kv_explicit_bf16`
fn write_kv_explicit_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kv_paged::write_kv_explicit_bf16(
        &ctx,
        &layer,
        cx.arg_in(0)?.cast::<bf16>().cast_const(),
        cx.arg_in(1)?.cast::<bf16>().cast_const(),
        cx.w_page_d()?,
        cx.w_off_d()?,
        cx.rows().count,
        plan.row_valid,
    )
}

/// `attn::attention_naive_paged`
fn attention_naive_paged_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attention_naive_paged(
        &ctx,
        &layer,
        cx.arg_in(0)?.cast::<bf16>().cast_const(),
        cx.arg_out(0)?.cast::<bf16>(),
        plan.qo_indptr,
        plan.kv_page_indices,
        plan.kv_page_indptr,
        plan.kv_last_page_lens,
        cx.rows().count,
        plan.requests,
        cx.in_width(0)?,
        cx.window_left()?,
        cx.sm_scale()?,
        cx.logits_soft_cap()?,
        cx.lse_out()?,
    )
}

/// `attn::qkv_decode_qk_norm_rope_write_kv_bf16`
fn qkv_decode_fused_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    let w_page = cx.w_page_d().unwrap_or(core::ptr::null());
    let w_off = cx.w_off_d().unwrap_or(core::ptr::null());
    let head_dim = layer.head_dim;
    let num_kv_heads = layer.num_kv_heads;
    let num_q_heads = (cx.in_width(0)? - 2 * num_kv_heads * head_dim) / head_dim;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    qkv_decode_qk_norm_rope_write_kv_bf16(
        &ctx,
        cx.arg_in(0)?.cast::<bf16>().cast_const(),
        cx.q_out()?.cast::<bf16>(),
        layer.k_pages.cast::<bf16>(),
        layer.v_pages.cast::<bf16>(),
        cx.weight(0)?.cast::<bf16>().cast_const(),
        cx.weight(1)?.cast::<bf16>().cast_const(),
        cx.positions()?,
        cx.arg_in(1)?.cast::<f32>().cast_const(),
        plan.kv_page_indices,
        plan.kv_page_indptr,
        plan.kv_last_page_lens,
        w_page,
        w_off,
        plan.row_valid,
        cx.rows().count,
        num_q_heads,
        num_kv_heads,
        head_dim,
        layer.page_size,
        layer.hnd,
        cx.theta()?,
        cx.rms_eps()?,
    )
}

/// `attn::dequant_kv_cache_layer_to_bf16_active`
fn dequant_kv_active_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    // A BF16 CACHE HAS NOTHING TO DEQUANTISE, and that is a no-op rather
    // than a refusal.
    //
    // The statement is unconditional: `lower::walk` describes this kernel as
    // the one that *"stages a quantized cache before a prefill dispatch"*,
    // and the text names it whether or not the deployment quantised its
    // pages -- `[driver] kv_cache_dtype` is a BOOT choice and a trace cannot
    // see one. So every bf16 deployment reaches here, which is the default
    // and was every fire of this benchmark.
    //
    // The host program answers `Refusal::Absent { what: "quantised pages on
    // a bf16 layer" }`, which is correct FOR IT -- it declines a case it does
    // not cover, exactly as `write_kv_to_pages` declines a quantised writer
    // for native storage. What it cannot know is whether the caller wanted
    // that case. This one did not: dequantising bf16 to bf16 is the identity,
    // so the work is already done and the arm says so.
    //
    // Tested here rather than by catching the refusal, because the two are
    // not the same claim. Catching `Absent` would swallow any future decline
    // this host program grows; asking `is_native_bf16` says which case is
    // being called a no-op and would not survive that meaning changing.
    if layer.is_native_bf16 {
        return Ok(());
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kv_paged::dequant_kv_cache_layer_to_bf16_active(
        &ctx,
        &layer,
        cx.plan()?.kv_page_indices,
        cx.num_pages_in_batch()?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound { symbol: "attn::split_qkv_bf16", arm: Some(split_qkv_arm), unbound: None },

    Bound { symbol: "attn::lse_log2_to_ln", arm: Some(lse_log2_to_ln_arm), unbound: None },
    Bound {
        symbol: "attn::attention_sink_rescale_bf16",
        arm: Some(attention_sink_rescale_arm),
        unbound: None,
    },
    Bound { symbol: "attn::attn_res_blend_bf16", arm: Some(attn_res_blend_arm), unbound: None },
    Bound { symbol: "attn::pad_head_dim_bf16", arm: Some(pad_head_dim_arm), unbound: None },
    Bound { symbol: "attn::strip_head_dim_bf16", arm: Some(strip_head_dim_arm), unbound: None },
    Bound { symbol: "attn::logit_softcap_bf16", arm: Some(logit_softcap_arm), unbound: None },
    Bound { symbol: "attn::kimi_split_q_b_bf16", arm: Some(kimi_split_q_b_arm), unbound: None },
    Bound {
        symbol: "attn::kimi_split_kv_a_norm_bf16",
        arm: Some(kimi_split_kv_a_norm_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dsa_index_topk_mask",
        arm: Some(dsa_index_topk_mask_arm),
        unbound: None,
    },
    Bound { symbol: "attn::split_qkv_bf16_devwin", arm: Some(split_qkv_devwin_arm), unbound: None },
    Bound {
        symbol: "attn::combine_attn_outputs_bf16",
        arm: Some(combine_attn_outputs_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        arm: Some(qkv_packed_post_arm),
        unbound: None,
    },
    Bound { symbol: "attn::write_kv_to_pages", arm: Some(write_kv_to_pages_arm), unbound: None },
    Bound {
        symbol: "attn::write_kv_explicit_bf16",
        arm: Some(write_kv_explicit_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::attention_naive_paged",
        arm: Some(attention_naive_paged_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        arm: Some(qkv_decode_fused_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dequant_kv_cache_layer_to_bf16_active",
        arm: Some(dequant_kv_active_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dsa_index_q_rope_bf16",
        arm: None,
        unbound: Some(
            "the indexer's query rotation is not a shape this trace states: \
         `dsl::cuda::dsa_index_q_rope` records ONE input and NO parameters, \
         and puts `heads` and `head_dim` into the RESULT SHAPE only -- so \
         `out_width(0)` is their product and nothing splits it -- while \
         `rope_dim` appears in no statement, no shape and no context at all. \
         The host program is written, in `x::attn::dsa_index_q_rope_bf16`, \
         and what it is waiting for is a statement rather than a query",
        ),
    },
    Bound {
        symbol: "attn::dsa_index_knorm_rope_bf16",
        arm: None,
        unbound: Some(
            "the key half is blocked by the same statement and by one more: \
         `dsl::cuda::dsa_index_knorm_rope` names NO weight bank, and the \
         kernel reads a LayerNorm weight AND a bias -- two operands with \
         nothing to come from, on top of the `rope_dim` its sibling also \
         lacks. `head_dim` alone is statable, as `out_width(0)`. The host \
         program is `x::attn::dsa_index_knorm_rope_bf16` and is complete",
        ),
    },
    Bound {
        symbol: "attn::attn_score_fold_heads",
        arm: None,
        unbound: Some(
            "`score_indptr_d` -- the score-capture CSR, \
        which says where each request's rows begin in the folded sink. Eight of \
        the nine operands are queries that exist: `scores` is `arg_in(0)`, \
        `folded` is `arg_out(0)`, three come off `plan()`, `num_q_heads` and \
        `page_size` off `num_q_heads()` and `kv_layer()`. The ninth is an \
        `AttnCtx` field with a real producer -- `attn_score::DecodeScoreCapturePlan` \
        publishes it as an arena-stable device base -- and no `Cx` query reaches \
        it. Same shape as `first_token` and `w_page_d` before they landed, and \
        NOT `Cx::mla_layer`'s shape, which refuses because nothing fills it",
        ),
    },
    Bound {
        symbol: "attn::compact_page_csr",
        arm: None,
        unbound: Some(
            "the statement declares ONE result and the kernel writes THREE CSR \
         arrays plus a scratch: `dsl::cuda::compact_page_csr` records one \
         input, no parameters, a `StateRef` and a single `[Requests] I32` \
         result, so `arg_out(0)` answers one of `page_indptr_out`, \
         `last_page_lens_out` and `page_indices_out` and there is no way to \
         say WHICH -- while `scratch_counts`, the buffer that carries the \
         dependency BETWEEN the two launches, and `keep_stride` have nothing \
         at all. Six of eleven ARE answered: `keep` is `arg_in(0)`, the three \
         CSR inputs and `num_requests` come off `plan()`. The host program is \
         `x::attn::compact_page_csr`, both launches in order with both \
         refusals hoisted ahead of the first, and it is complete",
        ),
    },
    Bound {
        symbol: "attn::mtp_shift_hidden_bf16",
        arm: None,
        unbound: Some(
            "ONE operand of nine, and it is `slot_ids`: the only query that \
         reaches a request->slot map is `Cx::gdn()`, whose `slot_ids_d` is \
         exactly this pointer, and `Facts::gdn` answers `None` unless the \
         fire has a RECURRENT shape. An MTP head on a dense transformer has \
         none, so the query refuses for the fire that needs it. Everything \
         else is answered: `target_hidden` and `pending_hidden` are \
         `arg_in(0)` and `arg_in(1)` -- the statement hands the pending slab \
         over as an INPUT, so no `Slab` variant is wanted here -- `out` is \
         `arg_out(0)`, `qo_indptr` and `num_requests` come off `plan()`, \
         `total_tokens` is `rows()` and `hidden_size` is `out_width(0)`. The \
         host program is `x::attn::mtp_shift_hidden_bf16` and is complete",
        ),
    },
    Bound {
        symbol: "attn::mtp_update_pending_hidden_bf16",
        arm: None,
        unbound: Some(
            "its twin's `slot_ids`, and one more of a different kind: this \
         statement records NO result and a `StateRef { store: \
         RecurrentState }`, so `pending_hidden` -- which this kernel WRITES \
         -- is a slab reference rather than an argument, and `Slab` has two \
         variants, `Conv` and `Recurrent`, neither of which is the MTP \
         pending-hidden row. `RecurrentStateCache` carries it as a third \
         half, `Buffer::MtpHidden`, addressed by SLOT and not by layer, so it \
         is a slab kind rather than a stride on an existing one -- which is \
         the change `Slab`'s own doc asks for: `the next person to add a slab \
         kind adds a stride to Gdn in the same change`. `target_hidden` is \
         `arg_in(0)`, `hidden_size` is `in_width(0)`, `qo_indptr` and \
         `num_requests` come off `plan()`. The host program is \
         `x::attn::mtp_update_pending_hidden_bf16` and is complete",
        ),
    },
    Bound {
        symbol: "attn::mla_prepare_bf16",
        arm: None,
        unbound: Some(
            "`Cx::mla_layer` refuses, and it is the whole blocker: the two page \
         arrays, `page_size`, `kv_lora_rank` and `qk_rope_head_dim` all come \
         out of one view, so five of this kernel's thirty operands go \
         together or not at all. That query's refusal is STRUCTURAL and its \
         own doc says so -- `AttnCtx` carries `layers: Vec<KvCacheLayerView>` \
         and no MLA equivalent, and the views come from \
         `pools::mla_cache::MlaCachePool::layer_view`, which no `Fire` can \
         reach. This is `ATTENTION_MLA`'s refusal, one kernel earlier in the \
         same lane, and it is a DIFFERENT SHAPE from the `dsv4` three's \
         ratio: the ratio has no producer anywhere, and this has a producer \
         no fire reaches. Everything else is answered -- the four query \
         outputs and two KV outputs are `arg_out(0..5)`, `kv_a`/`q_b` are \
         `arg_in`, the norm weight is `weight(0)`, the four CSR arrays and \
         `row_valid` come off `plan()`, `eps` is `rms_eps()`, `theta` is \
         `rope_theta()`, `interleaved` is `rope_interleaved()` and `yarn` is \
         `yarn()`. The host program is `x::attn::mla_prepare_bf16` and is \
         complete",
        ),
    },
    Bound {
        symbol: "attn::write_mla_to_pages",
        arm: None,
        unbound: Some(
            "the same view, and nothing else missing: this kernel's thirteen \
         operands are two inputs, the four CSR arrays, `row_valid`, \
         `num_requests` -- all answered -- and the five that ARE the layer \
         view. `serve/load.rs` refuses every MLA checkpoint at load today, so \
         the refusal this states is the one a model would meet anyway, one \
         layer lower and in a sentence. The host program is \
         `x::attn::write_mla_to_pages` and is complete",
        ),
    },
    Bound {
        symbol: "attn::dsv4_boundary_meta_decode",
        arm: None,
        unbound: Some(
            "the compression RATIO is not a value this trace carries: \
         `dsl::cuda::dsv4_boundary_meta` records its inputs with \
         `record_many` and NO parameters, so the one integer the kernel \
         DIVIDES BY has no operand — and it appears in no `AttnCtx` field, \
         no `DispatchCtx` field and no `Facts` query either, so there is \
         nothing to answer it with. Everything ELSE is statable: `positions` \
         is `arg_in(0)`, the three outputs are `arg_out(0..2)`, `row_valid` \
         and `requests` come off `plan()`, and `n` is `rows()`. The host \
         program is `x::attn::dsv4_boundary_meta_decode` and is complete",
        ),
    },
    Bound {
        symbol: "attn::dsv4_boundary_meta_paged",
        arm: None,
        unbound: Some(
            "its twin's ratio, and nothing else: `qo_indptr` and `num_requests` \
         BOTH come off `plan()`, so the prefill form's two extra operands \
         are the two that are already answered. One statement carries both \
         rows -- `dsl::cuda::dsv4_boundary_meta` -- so a parameter added \
         there lands on both at once. The host program is \
         `x::attn::dsv4_boundary_meta_paged` and is complete",
        ),
    },
    Bound {
        symbol: "attn::attention_compressed_paged_bf16",
        arm: None,
        unbound: Some(
            "the same ratio and two buffers with no producer anywhere: \
         `comp_kv_pages` is deepseek_v4's COMPRESSED cache, which no pool \
         allocates and no context carries, and `req_of_token` is a \
         per-token request map that nothing in `driver-cuda` builds. \
         `sm_scale` is the one blocker of a different kind -- it HAS a \
         producer, `AttnCtx::sm_scale` at `bind/mod.rs:1489`, and six \
         generated arms read it -- so it is a query that could exist and \
         does not, where the other three are values that do not exist. \
         `q`, `o`, `lse_out`, `positions`, the two page arrays, \
         `total_tokens`, `num_q_heads`, `head_dim` and `page_size` are all \
         answered today. The host program is \
         `x::attn::attention_compressed_paged_bf16` and is complete",
        ),
    },
    Bound {
        symbol: "attn::dsv4_compress_gather_paged_bf16",
        arm: None,
        unbound: Some(
            "deepseek_v4's compression state is not a value this trace names: \
         `dsl::cuda::dsv4_compress_gather_paged` records ONE input \
         (`boundary_pos`) and NO parameters for a kernel that reads five \
         buffers and three integers, so `state_kv`, `state_score`, `ape`, \
         `boundary_req`, `ratio` and `coff` have no operand to come from — \
         the host program is written, in `x::attn::dsv4_compress::\
         dsv4_compress_gather_paged_bf16`, and what it is waiting for is a \
         statement rather than a query",
        ),
    },
    Bound {
        symbol: "attn::dsv4_store_comp_entries_bf16",
        arm: None,
        unbound: Some(
            "the commit half is blocked by the same statement as the gather: \
         `dsl::cuda::dsv4_store_comp_entries` names `entries` and \
         `boundary_pos`, and the kernel also reads `boundary_req` — \
         `dsv4_boundary_meta_*`'s second output, which the trace discards — \
         and needs `head_dim` and `page_size` besides; the host program is \
         `x::attn::dsv4_compress::dsv4_store_comp_entries_bf16`",
        ),
    },
    Bound {
        symbol: "attn::dispatch_attention_mla_bf16",
        arm: None,
        unbound: Some(
            "attention over the latent cache cannot be bound because this driver \
             does not build one: `fire/launch.rs`' `kv_pools_for` refuses \
             `KvStyle::Mla` and `serve/load.rs` refuses the checkpoint at model \
             load, so `Cx::mla_layer` and `Cx::mla_plan` have nothing to answer \
             with. BOTH host programs are now in the kernels crate and both \
             fire: `x::attn::mla_fa2::fire` over the six instantiations its \
             `mod inst` names, and `x::attn::mla_naive::fire` over the \
             Blackwell pair. Choosing between them is `Ctx::compute_capability\
             _major` — `>= 10` picks naive, because FA2 MLA writes ZERO OUTPUT \
             on sm_100 — so the capability is no longer the missing fact \
             either. What is missing is a caller: nothing reaches these two \
             until a latent cache exists to attend over",
        ),
    },
];
