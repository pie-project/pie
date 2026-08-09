//! What happens when a trace states one of `norm`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::abi::bf16;
use kernels_cuda_new::x::norm::*;

use super::super::cx::Cx;
use super::Bound;

/// `Source::Isqrt` — the exact integer square root, or `0`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn isqrt_exact(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut r = f64::from(n).sqrt() as i32;
    while r > 0 && r.saturating_mul(r) > n {
        r -= 1;
    }
    while (r + 1).saturating_mul(r + 1) <= n {
        r += 1;
    }
    if r.saturating_mul(r) == n { r } else { 0 }
}

/// `Source::OutElements(0)` — the region's rows times the result's width.
fn elements(cx: &Cx<'_>) -> Result<i32, Refusal> {
    Ok(cx.rows().count.saturating_mul(cx.out_width(0)?))
}

/// `norm::rmsnorm_strided_bf16`
fn rmsnorm_strided_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm_strided_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.in_width(0)?,
        cx.out_width(0)?,
        cx.rms_eps()?,
    )
}

/// `norm::residual_add_rmsnorm_bf16`
fn residual_add_rmsnorm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    residual_add_rmsnorm_bf16(
        &ctx,
        cx.arg_in(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.rms_eps()?,
    )
}

/// `norm::rmsnorm_residual_add_bf16`
fn norm_residual_add_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm_residual_add_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.rms_eps()?,
    )
}

/// `norm::add_bias_bf16`
fn add_bias_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    add_bias_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.weight_named(0)?.cast_const().cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `norm::hc_rmsnorm_to_f32`
fn hc_rmsnorm_to_f32_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    hc_rmsnorm_to_f32(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.rms_eps()?,
    )
}

/// `norm::hc_expand_bf16`
fn hc_expand_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let hidden = cx.in_width(0)?;
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "the hidden width" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    hc_expand_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)? / hidden,
        hidden,
    )
}

/// `norm::hc_post_bf16`
fn hc_post_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let hidden = cx.in_width(0)?;
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "the hidden width" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    hc_post_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.arg_in(2)?.cast_const().cast::<f32>(),
        cx.arg_in(3)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)? / hidden,
        hidden,
    )
}

/// `norm::per_head_rmsnorm_bf16`
fn per_head_rmsnorm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    if head_dim <= 0 {
        return Err(Refusal::Narrow { what: "head_dim", at: i64::from(head_dim) });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    per_head_rmsnorm_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)? / head_dim,
        head_dim,
        cx.rms_eps()?,
    )
}

/// `norm::attn_sink_correction_bf16`
fn attn_sink_correction_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    if head_dim <= 0 {
        return Err(Refusal::Narrow { what: "head_dim", at: i64::from(head_dim) });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attn_sink_correction_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.weight(0)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)? / head_dim,
        head_dim,
    )
}

/// `norm::altup_unpack_predict_coefs`
fn altup_unpack_predict_coefs_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    altup_unpack_predict_coefs(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        isqrt_exact(cx.in_width(0)?),
    )
}

/// `norm::altup_unpack_correct_coefs`
fn altup_unpack_correct_coefs_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    altup_unpack_correct_coefs(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
    )
}

/// `norm::compute_rms_bf16`
fn compute_rms_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    compute_rms_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        cx.in_width(0)?,
        ALTUP_EPS,
    )
}

/// `norm::magnitude_rescale_bf16`
fn magnitude_rescale_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    magnitude_rescale_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<f32>(),
        cx.rows().count,
        cx.out_width(0)?,
        ALTUP_EPS,
    )
}

/// `norm::residual_add_bf16`
fn residual_add_cuda_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = usize::try_from(elements(cx)?).unwrap_or(0);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    residual_add_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        n,
    )
}

/// `norm::tanh_bf16`
fn tanh_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    tanh_bf16(&ctx, cx.arg_out(0)?.cast::<bf16>(), elements(cx)?)
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound { symbol: "norm::rmsnorm_strided_bf16", arm: Some(rmsnorm_strided_arm), unbound: None },
    Bound {
        symbol: "norm::residual_add_rmsnorm_bf16",
        arm: Some(residual_add_rmsnorm_arm),
        unbound: None,
    },
    Bound {
        symbol: "norm::rmsnorm_residual_add_bf16",
        arm: Some(norm_residual_add_arm),
        unbound: None,
    },
    Bound { symbol: "norm::add_bias_bf16", arm: Some(add_bias_arm), unbound: None },
    Bound { symbol: "norm::hc_rmsnorm_to_f32", arm: Some(hc_rmsnorm_to_f32_arm), unbound: None },
    Bound { symbol: "norm::hc_expand_bf16", arm: Some(hc_expand_arm), unbound: None },
    Bound { symbol: "norm::hc_post_bf16", arm: Some(hc_post_arm), unbound: None },
    Bound { symbol: "norm::per_head_rmsnorm_bf16", arm: Some(per_head_rmsnorm_arm), unbound: None },
    Bound {
        symbol: "norm::attn_sink_correction_bf16",
        arm: Some(attn_sink_correction_arm),
        unbound: None,
    },
    Bound {
        symbol: "norm::altup_unpack_predict_coefs",
        arm: Some(altup_unpack_predict_coefs_arm),
        unbound: None,
    },
    Bound {
        symbol: "norm::altup_unpack_correct_coefs",
        arm: Some(altup_unpack_correct_coefs_arm),
        unbound: None,
    },
    Bound { symbol: "norm::compute_rms_bf16", arm: Some(compute_rms_arm), unbound: None },
    Bound {
        symbol: "norm::magnitude_rescale_bf16",
        arm: Some(magnitude_rescale_arm),
        unbound: None,
    },
    Bound { symbol: "norm::residual_add_bf16", arm: Some(residual_add_cuda_arm), unbound: None },
    Bound { symbol: "norm::tanh_bf16", arm: Some(tanh_arm), unbound: None },
    Bound {
        symbol: "norm::rmsnorm_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the statement's per-head width. \
        The deleted row read Source::IfPresent(PerHeadDim, ..) on both num_rows \
        and hidden, because OpKind::RmsnormPerHead lowers to this same symbol \
        and norms rows x (width / head_dim) rows of head_dim where the plain \
        kind norms rows of width; without the query this fn would norm \
        gemma-4's q/k heads as one row each. Needs `Facts::per_head_dim() -> \
        Option<i32>` over LaunchSpec::per_head_dim (bind/mod.rs:1798), which \
        the driver already holds",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_gemma_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the statement's per-head \
        width, exactly as for norm::rmsnorm_bf16 — same operand contract, \
        different arithmetic. Needs `Facts::per_head_dim()`",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_bf16_with_fp16",
        arm: None,
        unbound: Some(
            "The deleted row stated no Source on any of \
        its eight operands, so there is nothing to read a binding from: it \
        described the launcher's C signature and never said where y_fp16, or \
        anything else, comes from. The host program above is complete and \
        proven; what is missing is a statement that names the fp16 copy. Needs \
        a lowering that produces two results, and then Source::Out(1)",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_no_scale_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the statement's per-head \
        width; this is the V-norm and the per-head reading is the only one it \
        is ever fired with. Needs `Facts::per_head_dim()`",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_gated_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the statement's \
        per-head width. Needs `Facts::per_head_dim()`",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_gated_fp32_in_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the gated-delta-net \
        head width. The deleted row bound hidden from Source::Gdn(\"v_d\") and \
        families/norm.rs records the correction it wanted -- spec.per_head_dim \
        set from gdn.v_d where the statement is a gated norm -- so this needs \
        `Facts::per_head_dim()` and the driver-side assignment, not a new \
        Source",
        ),
    },
    Bound {
        symbol: "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the layer's \
        residual scale. Every other operand of gemma-4's fused landing is \
        available -- two weights, two results, the row count and the width -- \
        and the one that is not is Source::LayerScale, the per-layer constant \
        the binder reads off the model. Needs `Facts::layer_scale() -> \
        Option<f32>`. This is the family's most expensive refusal: the host \
        program above is the three-arm vectorised form measured at -38%, -49% \
        and -53% against the shipping scalar kernel",
        ),
    },
    Bound {
        symbol: "norm::hc_pre_postprocess_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- nor for the \
        two scratch buffers this kernel hands to norm::hc_post_bf16, nor for \
        the model constants sinkhorn_iters and hc_post_alpha. The deleted row \
        stated no Source on any of its thirteen operands and that was the \
        honest spelling: a half-bound row generates exactly as much as an \
        unbound one while claiming bindings nobody checked. Needs a lowering \
        that states the slabs, which is a design question and not an accessor",
        ),
    },
    Bound {
        symbol: "norm::hc_head_postprocess_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- or for \
        hc_eps. Same shape as norm::hc_pre_postprocess_bf16 and the same \
        answer",
        ),
    },
    Bound {
        symbol: "norm::altup_predict_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the AltUp stream count. \
        `streams` is [t, k*h] with the streams interleaved, so only the fire \
        knows how that row divides and the deleted row read \
        Source::Ctx(\"altup_streams\"); DispatchCtx::altup_streams \
        (bind/mod.rs:1244) is the accessor. Needs `Facts::altup_streams() -> \
        Option<i32>`",
        ),
    },
    Bound {
        symbol: "norm::altup_correct_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for which AltUp stream was run \
        through the real layer. Every extent on this statement comes off its \
        own values -- k from input 2's width, h from input 1's -- and the one \
        that does not is active_idx, DispatchCtx::altup_active \
        (bind/mod.rs:1246). Needs `Facts::altup_active() -> Option<i32>`",
        ),
    },
    Bound {
        symbol: "norm::mean_streams_bf16",
        arm: None,
        unbound: Some(
            "Cx has no query for the AltUp stream count, and \
        here it is not an extent at all: the streams arrive interleaved and \
        only the fire knows how the row divides, which is why the deleted row \
        said CtxNonZero rather than Ctx -- declining is better than dividing \
        by zero. Needs `Facts::altup_streams()`",
        ),
    },
    Bound {
        symbol: "norm::scalar_mul_bf16",
        arm: None,
        unbound: Some(
            "Cx can read a stated scale but not a named one. \
        The deleted row said Source::Or(ParamF32(0), NamedScale): a statement \
        that carries the number binds today through Cx::param_f32(0), and one \
        that carries a NAME -- which is what gemma-3n and gemma-2 state -- has \
        nowhere to read it from. Binding only the first half would make this \
        symbol work for some models and refuse at fire for exactly the two the \
        deleted row named, which is worse than refusing at load. Needs \
        `Facts::named_scale() -> Option<f32>`",
        ),
    },
];
