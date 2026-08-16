//! What happens when a trace states one of `moe`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::moe::*;

use super::super::cx::Cx;
use super::Bound;

/// `moe::apply_per_expert_scale_bf16`
fn apply_per_expert_scale_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = cx.in_width(1)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    apply_per_expert_scale::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.arg_in(1)?.cast::<f32>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.rows().count.saturating_mul(top_k),
    )
}

/// `moe::topk_sqrtsoftplus_bf16`
fn topk_sqrtsoftplus_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let correction_bias = cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    topk_sqrtsoftplus::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<f32>(),
        correction_bias,
        cx.rows().count,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.moe_norm_topk()?,
        cx.moe_routed_scaling()?,
    )
}

/// `moe::hash_route_lookup`
fn hash_route_lookup_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    hash_route_lookup(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.weight(0)?.cast_const().cast::<i64>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.vocab()?,
        cx.in_width(1)?,
        cx.out_width(0)?,
        cx.moe_norm_topk()?,
        cx.moe_routed_scaling()?,
    )
}

/// `moe::topk_sigmoid_bias_fp32`
fn topk_sigmoid_bias_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    topk_sigmoid_bias_fp32(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.weight(0)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.moe_norm_topk()?,
        cx.moe_routed_scaling()?,
    )
}

/// `moe::moe_align_decode`
fn moe_align_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
    let num_routes = cx.rows().count.saturating_mul(cx.in_width(0)?);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    moe_align_decode(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<i32>(),
        cx.arg_out(2)?.cast::<i32>(),
        num_routes,
        param(0)?,
        param(1)?,
        param(2)?,
        core::ptr::null_mut(),
    )
}

/// `moe::gather_moe_aligned_inputs_bf16`
fn gather_moe_aligned_inputs_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    gather_moe_aligned_inputs::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<i32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count.saturating_mul(top_k),
        cx.in_rows(1)?,
        top_k,
        cx.out_width(0)?,
        -1,
        cx.rows().count,
    )
}

/// `moe::reorder_moe_aligned_output_bf16`
fn reorder_moe_aligned_output_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    reorder_moe_aligned_output::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<i32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count.saturating_mul(top_k),
        cx.in_rows(1)?,
        cx.in_width(0)?,
        -1,
        cx.rows().count,
        core::ptr::null_mut(),
    )
}

/// `moe::topk_sigmoid_bf16`
fn topk_sigmoid_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let correction_bias = cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    topk_sigmoid::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<f32>(),
        correction_bias,
        cx.rows().count,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.moe_norm_topk()?,
        cx.moe_routed_scaling()?,
    )
}

/// `moe::topk_softmax_bf16`
fn topk_softmax_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    topk_softmax::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<i32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
        cx.out_width(0)?,
    )
}

/// `moe::moe_gate_up_decode_gemv_bf16`
fn moe_gate_up_gemv_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = cx.in_width(0)?;
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the route width" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    moe_gate_up_decode_gemv::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        top_k,
        cx.in_width(1)?,
        cx.out_width(0)? / top_k,
    )
}

/// `moe::moe_down_decode_gemv_bf16`
fn moe_down_gemv_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = cx.in_width(0)?;
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the route width" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    moe_down_decode_gemv::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        top_k,
        cx.out_width(0)? / top_k,
        cx.in_width(1)?,
    )
}

/// `moe::token_batched_weighted_sum_bf16`
fn moe_weighted_sum_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    token_batched_weighted_sum::<bf16>(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.in_width(1)?,
        cx.out_width(0)?,
    )
}

/// `moe::token_batched_weighted_sum_add_bf16`
fn moe_weighted_sum_add_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    token_batched_weighted_sum_add::<bf16>(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.in_width(1)?,
        cx.out_width(0)?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "moe::apply_per_expert_scale_bf16",
        arm: Some(apply_per_expert_scale_arm),
        unbound: None,
    },
    Bound {
        symbol: "moe::topk_sqrtsoftplus_bf16",
        arm: Some(topk_sqrtsoftplus_arm),
        unbound: None,
    },
    Bound { symbol: "moe::hash_route_lookup", arm: Some(hash_route_lookup_arm), unbound: None },
    Bound {
        symbol: "moe::topk_sigmoid_bias_fp32",
        arm: Some(topk_sigmoid_bias_arm),
        unbound: None,
    },
    Bound { symbol: "moe::moe_align_decode", arm: Some(moe_align_arm), unbound: None },
    Bound {
        symbol: "moe::gather_moe_aligned_inputs_bf16",
        arm: Some(gather_moe_aligned_inputs_arm),
        unbound: None,
    },
    Bound {
        symbol: "moe::reorder_moe_aligned_output_bf16",
        arm: Some(reorder_moe_aligned_output_arm),
        unbound: None,
    },
    Bound { symbol: "moe::topk_sigmoid_bf16", arm: Some(topk_sigmoid_arm), unbound: None },
    Bound { symbol: "moe::topk_softmax_bf16", arm: Some(topk_softmax_arm), unbound: None },
    Bound {
        symbol: "moe::moe_gate_up_decode_gemv_bf16",
        arm: Some(moe_gate_up_gemv_arm),
        unbound: None,
    },
    Bound { symbol: "moe::moe_down_decode_gemv_bf16", arm: Some(moe_down_gemv_arm), unbound: None },
    Bound {
        symbol: "moe::token_batched_weighted_sum_bf16",
        arm: Some(moe_weighted_sum_arm),
        unbound: None,
    },
    Bound {
        symbol: "moe::token_batched_weighted_sum_add_bf16",
        arm: Some(moe_weighted_sum_add_arm),
        unbound: None,
    },
    Bound {
        symbol: "moe::add_moe_route_bias_bf16",
        arm: None,
        unbound: Some(
            "Nothing states the destination's row \
        PITCH, and that is the whole of it. Two of this kernel's three \
        numbers are reachable and an earlier version of this sentence said \
        otherwise: dsl.rs:4300 passes topk_idx as input 1, so num_routes is \
        the fire's rows times that operand's own width, exactly as \
        moe::moe_align_decode's is, and cols is the result's own width. \
        out_stride is not reachable -- the kernel writes a slice of a wider \
        rectangle, and a stride is the caller's arithmetic rather than an \
        operand's extent, so no Source spelled one and no Cx query answers \
        one. WHAT WOULD MAKE IT FIRE: a lowering that states the \
        destination pitch, and then a model that calls the wrapper, \
        because today nothing under crates/model/src does",
        ),
    },
    Bound {
        symbol: "moe::transpose_expert_scales_u8",
        arm: None,
        unbound: Some(
            "Weight preparation is not a trace \
        statement, and this one is the proof: dsl.rs:4418 records it with \
        inputs vec![], THE ONLY STATEMENT IN THIS FAMILY WITH NO INPUTS AT \
        ALL. It rewrites a checkpoint's per-expert group-scale planes from \
        [experts, k_groups, n] to [experts, n, k_groups] once, over \
        weights, before any fire exists; its row stated no Source on any of \
        its five operands because there is no statement to read one from, \
        and its three numbers are the RESULT's three dims where Cx answers \
        only a width. WHAT WOULD MAKE IT FIRE: nothing from the trace, ever \
        -- it wants the driver-op shape, a call from driver-cuda's weight \
        loader with the host fn above as its body, which is where \
        moe::flashinfer_cutlass_moe_bf16 already sits. A none: here is \
        permanent unless that call is written",
        ),
    },
    Bound {
        symbol: "moe::moe_bucket_exact",
        arm: None,
        unbound: Some(
            "AHEAD OF A CALLER, AND NO LONGER AHEAD OF \
        A DECLARATION. The statement declared TWO results while the kernel \
        writes THREE buffers: moe_dispatch.cuh:907 takes topk_idx, \
        sorted_route_ids, route_to_sorted_row, counts_out, and the inverse \
        map was named nowhere in this crate. Passing a null for it would \
        not be a wrong answer but a write to null -- the store at :952 has \
        no null guard, which is the one place this kernel differs from its \
        padded twin, whose route_to_aligned_row IS guarded and IS therefore \
        optional. dsl.rs:5121 declares three now, in the kernel's own \
        parameter order, so a binding reads straight down: sorted_route_ids, \
        route_to_sorted_row, counts. The route count was NOT the gap and an \
        earlier version of this sentence said it was: topk_idx IS input 0 \
        and IS [Tokens, top_k], so num_routes reads exactly as \
        moe::moe_align_decode's does, and num_experts is the THIRD result's \
        own extent now that counts has moved behind the inverse map. WHAT \
        WOULD MAKE IT FIRE: a caller. Nothing under crates/model/src names \
        this symbol, and a bind nothing exercises is a claim nothing checks",
        ),
    },
];
