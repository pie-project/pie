//! What happens when a trace states one of `ssm`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::Slab;
use kernels_cuda_new::x::abi::{MaybeConst, bf16};
use kernels_cuda_new::x::ssm::*;

use super::super::cx::Cx;
use super::Bound;

/// `ssm::nemotron_mamba_split_bf16`
fn nemotron_mamba_split_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    nemotron_mamba_split_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        cx.arg_out(0)?,
        cx.arg_out(1)?,
        cx.arg_out(2)?,
        cx.rows().count,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.out_width(1)?,
        cx.out_width(2)?,
    )
}

/// `ssm::nemotron_prepare_mamba_params`
fn nemotron_prepare_mamba_params_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    nemotron_prepare_mamba_params(
        &ctx,
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.weight(1)?.cast_const().cast::<bf16>(),
        cx.weight(2)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.arg_out(2)?.cast::<f32>(),
        gdn.v_h,
    )
}

/// `ssm::nemotron_prepare_mamba_dt_da`
fn nemotron_prepare_mamba_dt_da_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    nemotron_prepare_mamba_dt_da(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.aux(3)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
        0.0,
    )
}

/// `ssm::nemotron_mamba_ssm_batched_bf16`
fn nemotron_mamba_ssm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    let rows = cx.rows().count;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    nemotron_mamba_ssm_batched_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        cx.aux(0)?.cast_const(),
        cx.aux(1)?.cast_const().cast::<f32>(),
        cx.aux(2)?.cast_const().cast::<f32>(),
        cx.aux(3)?.cast_const().cast::<f32>(),
        MaybeConst::new(cx.arg_in(1)?.cast_const().cast::<f32>()),
        MaybeConst::new(cx.aux(5)?.cast_const().cast::<f32>()),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        plan.qo_indptr,
        cx.arg_out(0)?,
        plan.requests,
        gdn.v_h,
        gdn.v_d,
        gdn.k_d,
        gdn.n_groups,
        gdn.conv_dim,
        gdn.v_h.saturating_mul(gdn.v_d),
        0.0,
        rows != plan.requests,
    )
}

/// `ssm::kda_gate_beta_bf16`
fn kda_gate_beta_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let d =
        i32::try_from(cx.param(0)?).map_err(|_| Refusal::Unstated { what: "the KDA head dim" })?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kda_gate_beta_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<f32>(),
        cx.weight(1)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.out_width(1)?,
        d,
        0.0,
    )
}

/// `ssm::kda_o_norm_gated_bf16`
fn kda_o_norm_gated_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let h = i32::try_from(cx.param(0)?)
        .map_err(|_| Refusal::Unstated { what: "the KDA head count" })?;
    let d =
        i32::try_from(cx.param(1)?).map_err(|_| Refusal::Unstated { what: "the KDA head dim" })?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    kda_o_norm_gated_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        h,
        d,
        cx.rms_eps()?,
    )
}

/// `ssm::causal_conv1d_update_batched_bf16`
fn gdn_conv_update_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let bias = cx
        .weight_bias()
        .map_or_else(MaybeConst::none, |p| MaybeConst::new(p.cast_const().cast::<bf16>()));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    causal_conv1d_update_batched_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        bias,
        cx.slab(Slab::Conv)?.cast::<bf16>(),
        gdn.slot_ids_d,
        gdn.conv_stride_elems,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        gdn.conv_dim,
        gdn.conv_k,
    )
}

/// `ssm::causal_conv1d_prefill_batched_bf16`
fn gdn_conv_prefill_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    let bias = cx
        .weight_bias()
        .map_or_else(MaybeConst::none, |p| MaybeConst::new(p.cast_const().cast::<bf16>()));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    causal_conv1d_prefill_batched_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        bias,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.slab(Slab::Conv)?.cast::<bf16>(),
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.conv_stride_elems,
        plan.requests,
        gdn.conv_dim,
        gdn.conv_k,
        gdn.write_state,
        MaybeConst::none(),
        MaybeConst::none(),
    )
}

/// `ssm::recurrent_gated_delta_step_batched`
fn gdn_step_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    recurrent_gated_delta_step_batched(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?.cast::<f32>(),
        gdn.slot_ids_d,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        cx.plan()?.requests,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
    )
}

/// `ssm::recurrent_gated_delta_step_batched_gqa`
fn gdn_step_gqa_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    recurrent_gated_delta_step_batched_gqa(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?.cast::<f32>(),
        gdn.slot_ids_d,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        cx.plan()?.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
    )
}

/// `ssm::recurrent_gated_delta_step_batched_state_bf16`
fn gdn_step_state_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    recurrent_gated_delta_step_batched_state_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        cx.rows().count,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
    )
}

/// `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16`
fn gdn_step_gqa_state_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    recurrent_gated_delta_step_batched_gqa_state_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        cx.plan()?.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
    )
}

/// `ssm::chunk_gated_delta_prefill_batched`
fn gdn_prefill_fla_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?.cast::<f32>(),
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        MaybeConst::none(),
        MaybeConst::none(),
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_state_bf16`
fn gdn_prefill_fla_state_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched_state_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        MaybeConst::none(),
        MaybeConst::none(),
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_cached`
fn gdn_prefill_cached_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched_cached(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?.cast::<f32>(),
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        MaybeConst::none(),
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16`
fn gdn_prefill_cached_state_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched_cached_state_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        MaybeConst::none(),
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa`
fn gdn_prefill_warp_tiled_gqa_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched_warp_tiled_gqa(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?.cast::<f32>(),
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        core::ptr::null(),
    )
}

/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16`
fn gdn_prefill_warp_tiled_gqa_state_bf16_arm(
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    let plan = cx.plan()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_in(4)?.cast_const().cast::<f32>(),
        cx.slab(Slab::Recurrent)?,
        gdn.slot_ids_d,
        plan.qo_indptr,
        gdn.state_stride_elems,
        cx.result(0)?.cast::<f32>(),
        plan.requests,
        gdn.k_h,
        gdn.v_h,
        gdn.k_d,
        gdn.v_d,
        gdn.write_state,
        core::ptr::null(),
    )
}

/// `ssm::repeat_interleave_heads_fp32`
fn repeat_interleave_heads_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    repeat_interleave_heads_fp32(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.result(0)?.cast::<f32>(),
        cx.rows().count,
        gdn.k_h,
        gdn.v_h,
        gdn.v_d,
    )
}

/// `ssm::l2norm_scale_bf16_to_fp32`
fn l2norm_scale_to_f32_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    l2norm_scale_bf16_to_fp32(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)?,
        1.0,
        cx.rms_eps()?,
    )
}

/// `ssm::bf16_to_fp32`
fn bf16_to_f32_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = usize::try_from(cx.rows().count)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    bf16_to_fp32(&ctx, cx.arg_in(0)?.cast_const(), cx.arg_out(0)?.cast::<f32>(), n)
}

/// `ssm::fp32_to_bf16`
fn f32_to_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = usize::try_from(cx.rows().count)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(cx.out_width(0)?).unwrap_or(0));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    fp32_to_bf16(&ctx, cx.arg_in(0)?.cast_const().cast::<f32>(), cx.arg_out(0)?, n)
}

/// `ssm::zamba_rmsnorm_gated_bf16`
fn zamba_rmsnorm_gated_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let gdn = cx.gdn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    zamba_rmsnorm_gated_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.in_width(0)?,
        cx.in_width(1)?,
        gdn.n_groups,
        cx.rms_eps()?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "ssm::nemotron_mamba_split_bf16",
        arm: Some(nemotron_mamba_split_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::nemotron_prepare_mamba_params",
        arm: Some(nemotron_prepare_mamba_params_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::nemotron_prepare_mamba_dt_da",
        arm: Some(nemotron_prepare_mamba_dt_da_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::nemotron_mamba_ssm_batched_bf16",
        arm: Some(nemotron_mamba_ssm_arm),
        unbound: None,
    },
    Bound { symbol: "ssm::kda_gate_beta_bf16", arm: Some(kda_gate_beta_arm), unbound: None },
    Bound { symbol: "ssm::kda_o_norm_gated_bf16", arm: Some(kda_o_norm_gated_arm), unbound: None },
    Bound {
        symbol: "ssm::causal_conv1d_update_batched_bf16",
        arm: Some(gdn_conv_update_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::causal_conv1d_prefill_batched_bf16",
        arm: Some(gdn_conv_prefill_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::recurrent_gated_delta_step_batched",
        arm: Some(gdn_step_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::recurrent_gated_delta_step_batched_gqa",
        arm: Some(gdn_step_gqa_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::recurrent_gated_delta_step_batched_state_bf16",
        arm: Some(gdn_step_state_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
        arm: Some(gdn_step_gqa_state_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched",
        arm: Some(gdn_prefill_fla_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched_state_bf16",
        arm: Some(gdn_prefill_fla_state_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched_cached",
        arm: Some(gdn_prefill_cached_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
        arm: Some(gdn_prefill_cached_state_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        arm: Some(gdn_prefill_warp_tiled_gqa_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
        arm: Some(gdn_prefill_warp_tiled_gqa_state_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::repeat_interleave_heads_fp32",
        arm: Some(repeat_interleave_heads_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::l2norm_scale_bf16_to_fp32",
        arm: Some(l2norm_scale_to_f32_arm),
        unbound: None,
    },
    Bound { symbol: "ssm::bf16_to_fp32", arm: Some(bf16_to_f32_arm), unbound: None },
    Bound { symbol: "ssm::fp32_to_bf16", arm: Some(f32_to_bf16_arm), unbound: None },
    Bound {
        symbol: "ssm::zamba_rmsnorm_gated_bf16",
        arm: Some(zamba_rmsnorm_gated_arm),
        unbound: None,
    },
    Bound {
        symbol: "ssm::kda_recurrent_step_batched",
        arm: None,
        unbound: Some(
            "kda_recurrent_step needs a KDA state \
        arena and the per-request slot ids that index it; a trace states \
        neither, and no operand source names a driver-allocated slab",
        ),
    },
    Bound {
        symbol: "ssm::kda_prefill_batched",
        arm: None,
        unbound: Some(
            "kda_prefill needs a KDA state arena, the \
        per-request slot ids that index it, and the query-offset plan the \
        driver assembles between statements; a trace states none of them",
        ),
    },
    Bound {
        symbol: "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        arm: None,
        unbound: Some(
            "build_nemotron_moe_ptrs_aligned \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the padded block layout a counting sort produced; a trace states \
        none of them and no operand source names a scratch slab",
        ),
    },
    Bound {
        symbol: "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        arm: None,
        unbound: Some(
            "build_nemotron_moe_ptrs_decode \
        needs six driver-allocated pointer arrays, two expert weight tables \
        and the decode intermediates the MoE path allocates between \
        statements; a trace states none of them",
        ),
    },
];
