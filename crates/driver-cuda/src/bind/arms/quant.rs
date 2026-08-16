//! What a trace that states one of `quant`'s symbols binds to.
//!
//! The rows that keep a hand arm are weight walkers: each reads a CHECKPOINT
//! MATRIX's extents where `Cx` answers a fire's rectangle. Two of them are
//! passes `model-loader` runs over a checkpoint, with no statement behind them
//! at all.
//!
//! Two `Lit` temptations are declined: `wna16_*`'s `group_size` comes off the
//! checkpoint despite being 128 almost everywhere, and a `Lit(Null)` for
//! `mxfp4_moe_down_decode`'s `bias_ptrs` would have frozen a null over a plane
//! the checkpoint ships.

use core::ffi::c_void;

use kernels::Refusal;
use kernels::keys::{self, Fact};
use kernels::routine::{Bank, Env, In, Out, Param};
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::{bf16, f16};
use kernels_cuda::quant::*;

use super::super::cx::Cx;
use super::Bound;




/// `quant::dequant_fp8_e4m3_to_bf16_per_channel`
fn dequant_fp8_e4m3_to_bf16_per_channel_arm(
    cx: &Cx<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    dequant_fp8_e4m3_to_bf16_per_channel(
        &ctx,
        kernels::routine::In { ptr: cx.arg_in(0)?.cast_const().cast::<u8>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: cx.arg_out(0)?.cast::<bf16>(), rows: 0, width: 0 },
        kernels::routine::In { ptr: cx.arg_in(1)?.cast_const().cast::<f32>(), rows: 0, width: 0 },
        Param(cx.rows().count),
        Param(cx.out_width(0)?),
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
        kernels::routine::In { ptr: cx.arg_in(0)?.cast_const().cast::<bf16>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: cx.arg_out(0)?.cast::<u8>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: cx.arg_out(1)?.cast::<u8>(), rows: 0, width: 0 },
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
        kernels::routine::In { ptr: cx.arg_in(0)?.cast_const().cast::<bf16>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: cx.arg_out(0)?.cast::<u8>(), rows: 0, width: 0 },
        kernels::routine::Out { ptr: cx.arg_out(1)?.cast::<f32>(), rows: 0, width: 0 },
        cx.rows().count,
        cx.in_width(0)?,
    )
}

/// `quant::mxfp4_moe_gate_up_decode_bf16`
fn mxfp4_moe_gate_up_decode_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // `_gate_bias`/`_up_bias` must bind NULL when absent, and absence here is
    // which export was loaded -- not something a CUDA statement sees. Hence
    // `unwrap_or(null_mut())`; marking them would kill a live row.
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    mxfp4_moe_gate_up_decode_bf16(
        &ctx,
        In {
            ptr: cx.arg_in(1)?.cast_const().cast::<f16>(),
            rows: cx.rows().count,
            width: cx.in_width(1).unwrap_or(0),
        },
        In {
            ptr: cx.arg_in(0)?.cast_const().cast::<i32>(),
            rows: cx.rows().count,
            width: cx.in_width(0).unwrap_or(0),
        },
        Bank { ptr: cx.weight(0)?.cast_const().cast::<*const u8>() },
        keys::WeightScales::env(
            cx.weight_suffixed("_scales")
                .ok_or(Refusal::Absent { what: "scale_ptrs" })?
                .cast_const()
                .cast::<u8>(),
        ),
        Env(cx
            .weight_suffixed("_gate_bias")
            .unwrap_or(core::ptr::null_mut())
            .cast_const()
            .cast::<*const c_void>()),
        Env(cx
            .weight_suffixed("_up_bias")
            .unwrap_or(core::ptr::null_mut())
            .cast_const()
            .cast::<*const c_void>()),
        Out {
            ptr: cx.arg_out(0)?.cast::<bf16>(),
            rows: cx.rows().count,
            width: cx.out_width(0).unwrap_or(0),
        },
        Out {
            ptr: cx.arg_out(1)?.cast::<bf16>(),
            rows: cx.rows().count,
            width: cx.out_width(1).unwrap_or(0),
        },
        keys::GluLimit::env(cx.glu_limit()?),
        keys::GluAlpha::env(cx.glu_alpha()?),
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    // A weight walker: `out_dim`/`in_dim` are the checkpoint matrix's extents.
    Bound::derived("quant::dequant_wna16_int4b8_to_bf16"),
    // Its `n` is a `usize` and `operand()` mints `ArgValue::I32`, so the call
    // refuses `Refusal::Kind`. Narrowing is wrong here: `model-loader` casts
    // whole checkpoint tensors, whose counts are genuinely `usize`.
    Bound::derived("quant::cast_fp32_to_bf16"),
    // `selected_rows` is the trap: `out.rows` is the same number today for a
    // reason unrelated to what the parameter means.
    Bound::derived("quant::mxfp4_scales_to_marlin_e8m0"),
    // A weight walker, plus `scale` from `cx.param_f32(0)?`: two independent
    // refusals on one row.
    Bound::derived("quant::dequant_fp8_e4m3_to_bf16"),
    // The POINTERS derive; `rows`/`cols` are the MATRIX's, and a right pointer
    // beside a wrong extent launches over the wrong rectangle.
    Bound {
        symbol: "quant::dequant_fp8_e4m3_to_bf16_per_channel",
        arm: Some(dequant_fp8_e4m3_to_bf16_per_channel_arm),
        unbound: None,
    },
    // Crossed; the extent problem above is now visible in the signature.
    Bound::derived("quant::dequant_fp8_e4m3_to_bf16_per_group"),
    Bound::derived("quant::dequant_mxfp4_to_bf16"),

    // A non-positive width now refuses `Absent`, not `Empty`.
    Bound::derived("quant::bf16_to_fp16"),

    // No `in_place` needed: that is a claim about ALIASING, where
    // `#[source(In(1))]` is a claim about WHICH INPUT. `x` is scaled in place,
    // so the first genuine `*const` is the second input.
    Bound::derived("quant::scale_rows_bf16"),

    // The loader quantisers: no statement stands behind either, so both extents
    // derive `None`.
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

    // The operand order inverts against the W4A16 twin below: the parameter
    // order is the CUDA signature's, the operand order the model text's. Both
    // are stated. `packed_ptrs` reads the POSITIONAL bank, not `Weight<0, _>`.
    // What blocks the row is `_gate_bias`/`_up_bias`, which must bind NULL on
    // the export that has neither.
    Bound {
        symbol: "quant::mxfp4_moe_gate_up_decode_bf16",
        arm: Some(mxfp4_moe_gate_up_decode_bf16_arm),
        unbound: None,
    },
    // `scale_ptrs` is NOT `Weight(1)`: the positional bank is a separate slice
    // from the suffixed lookup `keys::WeightScales` names.
    Bound::derived("quant::mxfp4_moe_down_decode_bf16"),

    // The positional weights are marked because `In(2)`..`In(5)` are ANSWERABLE
    // -- they resolve the moment a statement places six operands, to buffers
    // these launchers do not want. Both rows are `arm: None` so they refuse at
    // LOAD rather than dying mid-fire with `NoArm`.
    Bound {
        symbol: "quant::wna16_gate_up_decode_bf16",
        arm: None,
        unbound: Some(
            "a WNA16 group size, which no model contract states and no `QuantMeta` here builds",
        ),
    },
    Bound {
        symbol: "quant::wna16_down_decode_bf16",
        arm: None,
        unbound: Some(
            "a WNA16 group size; see `quant::wna16_gate_up_decode_bf16`.",
        ),
    },
];
