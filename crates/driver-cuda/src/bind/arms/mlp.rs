//! What happens when a trace states one of `mlp`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::abi::bf16;
use kernels_cuda_new::x::mlp::*;

use super::super::cx::Cx;
use super::Bound;

/// `Source::OutElements(0)` — the region's rows times the result's width.
fn elements(cx: &Cx<'_>) -> Result<i32, Refusal> {
    Ok(cx.rows().count.saturating_mul(cx.out_width(0)?))
}

/// `mlp::chunked_swiglu_bf16`
fn chunked_swiglu_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunked_swiglu_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        false,
    )
}

/// `mlp::relu2_bf16`
fn relu2_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = elements(cx)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    relu2_bf16(&ctx, cx.arg_in(0)?.cast_const().cast::<bf16>(), cx.arg_out(0)?.cast::<bf16>(), n)
}

/// `mlp::geglu_tanh_bf16`
fn geglu_tanh_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = elements(cx)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    geglu_tanh_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        n,
    )
}

/// `mlp::chunked_geglu_tanh_bf16`
fn chunked_geglu_tanh_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    chunked_geglu_tanh_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        false,
    )
}

/// `mlp::gpt_oss_glu_bf16`
fn gpt_oss_glu_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = elements(cx)?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    gpt_oss_glu_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        None,
        n,
        cx.param_f32(0)?,
        GPT_OSS_GLU_ALPHA,
    )
}

/// `mlp::sigmoid_dot_scalar_gate_add_bf16`
fn moe_shared_gate_dot_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    sigmoid_dot_scalar_gate_add_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(2)?.cast_const().cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound { symbol: "mlp::chunked_swiglu_bf16", arm: Some(chunked_swiglu_bf16_arm), unbound: None },
    Bound { symbol: "mlp::relu2_bf16", arm: Some(relu2_bf16_arm), unbound: None },
    Bound { symbol: "mlp::geglu_tanh_bf16", arm: Some(geglu_tanh_bf16_arm), unbound: None },
    Bound {
        symbol: "mlp::chunked_geglu_tanh_bf16",
        arm: Some(chunked_geglu_tanh_bf16_arm),
        unbound: None,
    },
    Bound { symbol: "mlp::gpt_oss_glu_bf16", arm: Some(gpt_oss_glu_bf16_arm), unbound: None },
    Bound {
        symbol: "mlp::sigmoid_dot_scalar_gate_add_bf16",
        arm: Some(moe_shared_gate_dot_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "mlp::swiglu_bf16",
        arm: None,
        unbound: Some(
            "this kernel reads the up projection's second \
        half, and a trace that leaves the projection packed never names it \
        -- the driver's op join supplies it, and a bind cannot ask the join \
        for anything. FLOOR: `up` is Source::Or(In(1), Aux(0)) and `Cx` \
        answers only In(1); needs `Facts::aux(i) -> Option<*mut c_void>`, \
        which is `join_aux(spec, i, frame, resolver)` and one defaulted \
        method",
        ),
    },
    Bound {
        symbol: "mlp::swiglu_clamp_bf16",
        arm: None,
        unbound: Some(
            "this kernel needs the model's GLU clamp \
        limit and the up projection's second half, and a bind can ask for \
        neither. FLOOR: DispatchCtx::glu_limit (bind/mod.rs:1193) and the \
        join's foreign operands; needs `Facts::glu_limit()` and \
        `Facts::aux(i)`",
        ),
    },
    Bound {
        symbol: "mlp::chunked_swiglu_clamp_bf16",
        arm: None,
        unbound: Some(
            "this kernel needs the model's GLU \
        clamp limit, and a bind cannot ask for it. FLOOR: \
        DispatchCtx::glu_limit (bind/mod.rs:1193); needs \
        `Facts::glu_limit()`, one defaulted method over a field the driver \
        already holds",
        ),
    },
    Bound {
        symbol: "mlp::situ_bf16",
        arm: None,
        unbound: Some(
            "this kernel needs the model's two SITU betas and \
        the up projection's second half, and a bind can ask for neither. \
        FLOOR: DispatchCtx::situ_beta and situ_linear_beta \
        (bind/mod.rs:1200-1202) and the join's foreign operands; needs \
        `Facts::situ() -> Option<(f32, f32)>` and `Facts::aux(i)`",
        ),
    },
    Bound {
        symbol: "mlp::chunked_situ_bf16",
        arm: None,
        unbound: Some(
            "this kernel needs the model's two SITU \
        betas and which half of the packed projection is the gate, and a \
        bind can ask for neither. FLOOR: DispatchCtx::situ_beta, \
        situ_linear_beta (bind/mod.rs:1200-1202) and gate_second \
        (bind/mod.rs:1149); needs `Facts::situ()` and \
        `Facts::gate_second() -> bool`",
        ),
    },
    Bound {
        symbol: "mlp::gaussian_topk_bf16",
        arm: None,
        unbound: Some(
            "this kernel needs the layer's altup \
        standard-deviation multiplier, which is a per-layer model constant, \
        and a bind cannot ask for it. FLOOR: the row bound \
        Source::CtxByLayer(\"altup_std_mult\") and \
        DispatchCtx::altup_std_mult(layer) (bind/mod.rs:1310) is the \
        accessor; needs `Facts::altup_std_mult(layer) -> Option<f32>`, and \
        `Cx::layer()` already answers the index",
        ),
    },
];
