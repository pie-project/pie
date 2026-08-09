//! What happens when a trace states one of `quant`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::abi::{bf16, f16};
use kernels_cuda_new::x::quant::*;

use super::super::cx::Cx;
use super::Bound;

/// `quant::dequant_wna16_int4b8_to_bf16`
fn dequant_wna16_int4b8_to_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_wna16_int4b8_to_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        group_size,
    )
}

/// `quant::cast_fp32_to_bf16`
fn cast_fp32_to_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let rows = cx.rows().count;
    let width = cx.out_width(0)?;
    if rows <= 0 || width <= 0 {
        return Err(Refusal::Empty { what: "the output rectangle" });
    }
    let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    cast_fp32_to_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        n,
    )
}

/// `quant::mxfp4_scales_to_marlin_e8m0`
fn mxfp4_scales_to_marlin_e8m0_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    mxfp4_scales_to_marlin_e8m0(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u8>(),
        cx.arg_out(0)?.cast::<u8>(),
        param(0)?,
        param(1)?,
        cx.rows().count,
        param(2)?,
        param(3)?,
        param(4)?,
        param(5)?,
        cx.out_width(0)?,
        param(6)?,
    )
}

/// `quant::dequant_fp8_e4m3_to_bf16`
fn dequant_fp8_e4m3_to_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let rows = cx.rows().count;
    let width = cx.out_width(0)?;
    if rows <= 0 || width <= 0 {
        return Err(Refusal::Empty { what: "the output rectangle" });
    }
    let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_fp8_e4m3_to_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u8>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.param_f32(0)?,
        n,
    )
}

/// `quant::dequant_fp8_e4m3_to_bf16_per_channel`
fn dequant_fp8_e4m3_to_bf16_per_channel_arm(
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_fp8_e4m3_to_bf16_per_channel(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u8>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `quant::dequant_fp8_e4m3_to_bf16_per_group`
fn dequant_fp8_e4m3_to_bf16_per_group_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_fp8_e4m3_to_bf16_per_group(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u8>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)?,
        group_size,
    )
}

/// `quant::dequant_mxfp4_to_bf16`
fn dequant_mxfp4_to_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_mxfp4_to_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u8>(),
        cx.arg_in(1)?.cast_const().cast::<u8>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `quant::bf16_to_fp16`
fn bf16_to_fp16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let rows = cx.rows().count;
    let width = cx.out_width(0)?;
    if rows <= 0 || width <= 0 {
        return Err(Refusal::Empty { what: "the output rectangle" });
    }
    let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    bf16_to_fp16(&ctx, cx.arg_in(0)?.cast_const().cast::<bf16>(), cx.arg_out(0)?.cast::<f16>(), n)
}

/// `quant::scale_rows_bf16`
fn scale_rows_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    scale_rows_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `quant::quantize_bf16_to_mxfp4_e2m1_per_block`
fn quantize_bf16_to_mxfp4_e2m1_per_block_arm(
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    quantize_bf16_to_mxfp4_e2m1_per_block(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<u8>(),
        cx.arg_out(1)?.cast::<u8>(),
        cx.rows().count,
        cx.in_width(0)?,
    )
}

/// `quant::quantize_bf16_to_fp8_e4m3_per_channel`
fn quantize_bf16_to_fp8_e4m3_per_channel_arm(
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    quantize_bf16_to_fp8_e4m3_per_channel(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<u8>(),
        cx.arg_out(1)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
    )
}

/// `quant::mxfp4_moe_gate_up_decode_bf16`
fn mxfp4_moe_gate_up_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = cx.in_width(0)?;
    let hidden = cx.in_width(1)?;
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the routed fanout" });
    }
    let intermediate = cx.out_width(0)? / top_k;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    mxfp4_moe_gate_up_decode_bf16(
        &ctx,
        cx.arg_in(1)?.cast_const().cast::<f16>(),
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.weight(0)?.cast_const().cast::<*const u8>(),
        cx.weight_suffixed("_scales")
            .ok_or(Refusal::Absent { what: "scale_ptrs" })?
            .cast_const()
            .cast::<*const u8>(),
        cx.weight_suffixed("_gate_bias")
            .unwrap_or(core::ptr::null_mut())
            .cast_const()
            .cast::<*const c_void>(),
        cx.weight_suffixed("_up_bias")
            .unwrap_or(core::ptr::null_mut())
            .cast_const()
            .cast::<*const c_void>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        None,
        cx.glu_limit()?,
        cx.glu_alpha()?,
        cx.rows().count,
        top_k,
        hidden,
        intermediate,
    )
}

/// `quant::mxfp4_moe_down_decode_bf16`
fn mxfp4_moe_down_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let top_k = cx.in_width(0)?;
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the routed fanout" });
    }
    let hidden = cx.out_width(0)? / top_k;
    let intermediate = cx.in_width(1)? / top_k;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    mxfp4_moe_down_decode_bf16(
        &ctx,
        cx.arg_in(1)?.cast_const().cast::<f16>(),
        cx.arg_in(0)?.cast_const().cast::<i32>(),
        cx.weight(0)?.cast_const().cast::<*const u8>(),
        cx.weight_suffixed("_scales")
            .ok_or(Refusal::Absent { what: "scale_ptrs" })?
            .cast_const()
            .cast::<*const u8>(),
        cx.weight_bias().unwrap_or(core::ptr::null_mut()).cast_const().cast::<*const c_void>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        top_k,
        hidden,
        intermediate,
    )
}

/// `quant::wna16_gate_up_decode_bf16`
fn wna16_gate_up_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    wna16_gate_up_decode_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f16>(),
        cx.arg_in(1)?.cast_const().cast::<i32>(),
        cx.weight(0)?.cast_const().cast::<*const i32>(),
        cx.weight(1)?.cast_const().cast::<*const c_void>(),
        cx.weight(2)?.cast_const().cast::<*const i32>(),
        cx.weight(3)?.cast_const().cast::<*const c_void>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.rows().count,
        cx.in_width(1)?,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.wna16_group_size()?,
    )
}

/// `quant::wna16_down_decode_bf16`
fn wna16_down_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    wna16_down_decode_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<f16>(),
        cx.arg_in(1)?.cast_const().cast::<i32>(),
        cx.weight(0)?.cast_const().cast::<*const i32>(),
        cx.weight(1)?.cast_const().cast::<*const c_void>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.in_width(1)?,
        cx.out_width(0)?,
        cx.in_width(0)?,
        cx.wna16_group_size()?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "quant::dequant_wna16_int4b8_to_bf16",
        arm: Some(dequant_wna16_int4b8_to_bf16_arm),
        unbound: None,
    },
    Bound { symbol: "quant::cast_fp32_to_bf16", arm: Some(cast_fp32_to_bf16_arm), unbound: None },
    Bound {
        symbol: "quant::mxfp4_scales_to_marlin_e8m0",
        arm: Some(mxfp4_scales_to_marlin_e8m0_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::dequant_fp8_e4m3_to_bf16",
        arm: Some(dequant_fp8_e4m3_to_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::dequant_fp8_e4m3_to_bf16_per_channel",
        arm: Some(dequant_fp8_e4m3_to_bf16_per_channel_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::dequant_fp8_e4m3_to_bf16_per_group",
        arm: Some(dequant_fp8_e4m3_to_bf16_per_group_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::dequant_mxfp4_to_bf16",
        arm: Some(dequant_mxfp4_to_bf16_arm),
        unbound: None,
    },
    Bound { symbol: "quant::bf16_to_fp16", arm: Some(bf16_to_fp16_arm), unbound: None },
    Bound { symbol: "quant::scale_rows_bf16", arm: Some(scale_rows_bf16_arm), unbound: None },
    Bound {
        symbol: "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
        arm: Some(quantize_bf16_to_mxfp4_e2m1_per_block_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::quantize_bf16_to_fp8_e4m3_per_channel",
        arm: Some(quantize_bf16_to_fp8_e4m3_per_channel_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::mxfp4_moe_gate_up_decode_bf16",
        arm: Some(mxfp4_moe_gate_up_decode_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::mxfp4_moe_down_decode_bf16",
        arm: Some(mxfp4_moe_down_decode_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::wna16_gate_up_decode_bf16",
        arm: Some(wna16_gate_up_decode_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "quant::wna16_down_decode_bf16",
        arm: Some(wna16_down_decode_bf16_arm),
        unbound: None,
    },
];
