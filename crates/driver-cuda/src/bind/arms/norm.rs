//! What happens when a trace states one of `norm`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::norm::*;

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

/// The gamma of a norm, whichever of the TWO ways its statement carries it.
///
/// `norm::rmsnorm_bf16` and `norm::rmsnorm_gemma_bf16` are each reached by
/// two different kinds of statement, and the two spell their weight in
/// different places:
///
/// * an `OpKind::Launch` — what `dsl::cuda::rmsnorm` emits when the handle
///   has no `per_head` — states it as an `Arg::Weight` in the operand list,
///   so it arrives POSITIONALLY, at `n_in + n_out`.
/// * the SEMANTIC `OpKind::Rmsnorm` / `RmsnormPerHead` — what the same call
///   emits when the handle does carry `per_head` — has no operand for it.
///   The name rides `LaunchSpec::weight`, resolved into `w_named`.
///
/// Reading only the first is what made gemma-4 refuse at its PLE prologue:
/// `ple_model_norm` has `per_head: Some(ple_dim)`, so it took the semantic
/// path and the positional slot was simply not there — *"the fire does not
/// carry a weight"*, for a statement that names one. Two launches later the
/// same symbol bound fine, because `layer.0.attn_norm` has `per_head: None`
/// and went the other way. One symbol, one weight, two spellings.
///
/// Positional FIRST, because that is where a statement that has both puts
/// the one it means: `OpKind::Launch` also fills `spec.weight` from
/// `weights.first()`, so a named-first order would work today and would
/// stop being obviously right the day a launch's first `Arg::Weight` is not
/// its gamma.
fn gamma(cx: &Cx<'_>) -> Result<*mut c_void, Refusal> {
    match cx.weight(0) {
        Ok(w) => Ok(w),
        Err(positional) => cx.weight_named(0).map_err(|_| positional),
    }
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

/// `norm::rmsnorm_bf16`
///
/// One symbol, two statements. `OpKind::Rmsnorm` norms `rows` rows of
/// `width`; `OpKind::RmsnormPerHead` norms `rows * (width / head_dim)` rows
/// of `head_dim`, and they lower to this same name — so the ONLY thing that
/// separates them is `Cx::per_head_dim`, which is why this arm could not be
/// written until that query existed and why the registry entry that stood
/// here named it as the floor.
///
/// The branch is not taken here. `norm::rmsnorm_bf16` opens with
/// `let hidden = if per_head_dim == 0 { width } else { per_head_dim };`, so
/// the absence is spelled zero and the kernel decides — which keeps the row
/// arithmetic beside the `<<<>>>` that depends on it, and keeps this arm
/// from being a second place where the two statements are told apart.
fn rmsnorm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        gamma(cx)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.per_head_dim().unwrap_or(0),
        cx.rms_eps()?,
    )
}

/// `norm::rmsnorm_gemma_bf16` — the `(1 + w)` fold.
///
/// [`rmsnorm_arm`] exactly, down to the argument list; the whole difference
/// is on the device, where gamma is read as `1 + w`. The registry entry that
/// stood here said so — *"same operand contract, different arithmetic"* —
/// and named `Facts::per_head_dim()` as the one thing missing. That query
/// exists, so the entry's own condition is met and this is what it asked to
/// become.
///
/// Reached by the SEMANTIC kinds under `NormVariant::Gemma`, which is why it
/// takes [`gamma`] rather than the positional weight: gemma-2's and
/// gemma-3n's norm handles are the ones that carry `per_head`, and those are
/// exactly the statements with no operand slot for their weight. gemma-4
/// reaches none of them — its fourteen norm sites are all `Plain`, which its
/// own forward says outright — so this arm's first real caller is another
/// family.
fn rmsnorm_gemma_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm_gemma::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        gamma(cx)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.per_head_dim().unwrap_or(0),
        cx.rms_eps()?,
    )
}

/// `norm::rmsnorm_no_scale_bf16` — gemma-4's V-norm, `v / rms(v)`.
///
/// No gamma, so no [`gamma`]: the dsl's `rmsnorm_no_scale` takes no `NormW`
/// at all, *"a norm handle contributes a name and a layer, and this kernel
/// reads neither"*. That is also why it is its own symbol rather than a
/// variant of the semantic `Rmsnorm` — there is no weight for a variant to
/// describe.
///
/// The registry entry that stood here wanted `Facts::per_head_dim()` and
/// said the per-head reading *"is the only one it is ever fired with"*. The
/// query exists; the arm still passes what the statement said rather than
/// assuming per-head, because the kernel spells the absence as zero and
/// deciding here would put the row arithmetic in two places.
fn rmsnorm_no_scale_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm_no_scale::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.per_head_dim().unwrap_or(0),
        cx.rms_eps()?,
    )
}

/// `norm::rmsnorm_gated_bf16` — norm times a sigmoid gate.
///
/// [`rmsnorm_no_scale_arm`]'s reason for existing now, plus two operands:
/// the gate is input 1, and gamma is **f32 here rather than bf16** — the one
/// place in this file where the weight's element type is not the
/// activation's, which is the routine's signature and not a choice made
/// here.
///
/// Positional gamma, not [`gamma`]: this symbol is reached only by
/// `dsl::cuda::rmsnorm_gated_launch`, which states the weight as an
/// `Arg::Weight` operand. No semantic kind lowers to it, so the second
/// spelling cannot arise and reaching for it would suggest otherwise.
fn rmsnorm_gated_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rmsnorm_gated::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_in(1)?.cast_const().cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<f32>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.per_head_dim().unwrap_or(0),
        cx.rms_eps()?,
    )
}

/// `norm::scalar_mul_bf16` — multiply a rectangle by a load-time constant.
///
/// # The NUMBER binds; the NAME never had to
///
/// The registry entry that stood here asked for `Facts::named_scale() ->
/// Option<f32>`, on the reading that `Source::Or(ParamF32(0), NamedScale)`
/// meant a driver had to be able to resolve `scale.<name>` to a number.
/// That is a fair reading of the row and it is not what the dsl does.
///
/// `dsl::cuda::scalar_mul(x, scale, by)` puts the NAME in the weight slot
/// and the NUMBER in the params, and its own doc says why the name is
/// there: *"a statement that did not say WHICH would leave an executor with
/// four identical launches and no way to tell them apart"*. The name is for
/// a READER. The `scale.` prefix is precisely the marker that no binder
/// should look the name up — `driver-wgpu` spells the same fact as
/// `Unbindable::Constant`. So there is nothing for a `named_scale()` query
/// to read: the host already derived the number and put it in the params,
/// and a table here would re-derive on the driver side what the model
/// crate had computed from its own dims.
///
/// # `by: None` is a family saying it does not know yet, and it still refuses
///
/// The worry the entry raised — that binding "only the first half" would
/// serve some models and refuse at fire for others — is real and is
/// answered by WHERE the refusal lands, not by declining to bind. Three
/// call sites pass `None`: gemma-2's `layer.N.query_scale` and gemma-3n's
/// altup and laurel scales, all per-layer constants nothing on the host has
/// derived. Those statements carry no param, so `param_f32(0)` refuses, and
/// `DispatchPlan::unfireable` asks that question at LOAD — which is the
/// load-time refusal the entry wanted, reached by binding rather than by
/// staying unbound. Staying unbound refused gemma-4 too, and gemma-4 passes
/// `Some(..)` at every one of its four sites.
///
/// In place, on operand 0: `x` is `*mut` and the routine takes one pointer.
fn scalar_mul_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let n = elements(cx)?;
    if n <= 0 {
        return Err(Refusal::Empty { what: "the scaled rectangle" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    scalar_mul::<bf16>(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.param_f32(0)?,
        n.unsigned_abs() as usize,
    )
}

/// `norm::residual_add_rmsnorm_bf16`
fn residual_add_rmsnorm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    residual_add_rmsnorm::<bf16>(
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
    rmsnorm_residual_add::<bf16>(
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
    add_bias::<bf16>(
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
    hc_expand::<bf16>(
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
    hc_post::<bf16>(
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
    per_head_rmsnorm::<bf16>(
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
    attn_sink_correction::<bf16>(
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
    compute_rms::<bf16>(
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
    magnitude_rescale::<bf16>(
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
    residual_add::<bf16>(
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
    tanh::<bf16>(&ctx, cx.arg_out(0)?.cast::<bf16>(), elements(cx)?)
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
    Bound { symbol: "norm::rmsnorm_bf16", arm: Some(rmsnorm_arm), unbound: None },
    Bound { symbol: "norm::rmsnorm_gemma_bf16", arm: Some(rmsnorm_gemma_arm), unbound: None },
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
    Bound { symbol: "norm::rmsnorm_no_scale_bf16", arm: Some(rmsnorm_no_scale_arm), unbound: None },
    Bound { symbol: "norm::rmsnorm_gated_bf16", arm: Some(rmsnorm_gated_arm), unbound: None },
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
    Bound { symbol: "norm::scalar_mul_bf16", arm: Some(scalar_mul_arm), unbound: None },
];
