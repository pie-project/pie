//! What happens when a trace states one of `rope`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda-new`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::abi::bf16;
use kernels_cuda_new::x::rope::*;

use super::super::cx::Cx;
use super::Bound;

/// `rope::rope_standard_table`
fn rope_standard_table_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rope_standard_table(
        &ctx,
        cx.positions()?,
        cx.arg_out(0)?.cast::<f32>(),
        cx.rows().count,
        cx.head_dim()?,
        cx.rope_theta()?,
    )
}

/// `rope::rope_bf16`
fn rope_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rope_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.positions()?,
        cx.rows().count,
        cx.num_q_heads()?,
        cx.num_kv_heads()?,
        cx.head_dim()?,
        cx.rope_theta()?,
        cx.rope_interleaved(),
    )
}

/// `rope::qk_rmsnorm_rope_bf16`
fn qk_rmsnorm_rope_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    qk_rmsnorm_rope_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.weight(1)?.cast_const().cast::<bf16>(),
        cx.positions()?,
        cx.rows().count,
        cx.out_width(0)? / head_dim,
        cx.out_width(1)? / head_dim,
        head_dim,
        cx.theta()?,
        cx.rms_eps()?,
    )
}

/// `rope::qk_rmsnorm_rope_bf16_devwin`
fn qk_rmsnorm_rope_bf16_devwin_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    let n_max = cx.rows().total;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    qk_rmsnorm_rope_bf16_devwin(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.weight(0)?.cast_const().cast::<bf16>(),
        cx.weight(1)?.cast_const().cast::<bf16>(),
        cx.positions()?,
        cx.peel_window()?.as_ptr().cast_const(),
        n_max,
        cx.out_width(0)? / head_dim,
        cx.out_width(1)? / head_dim,
        head_dim,
        cx.theta()?,
        cx.rms_eps()?,
    )
}

/// `rope::rope_partial_last_bf16`
fn rope_partial_last_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let kv = cx.kv_layer()?;
    if kv.head_dim <= 0 {
        return Err(Refusal::Empty { what: "kv head_dim" });
    }
    let q = cx.arg_out(0)?.cast::<bf16>();
    let kv_heads = cx.out_width(1).map_or(0, |w| w / kv.head_dim);
    let yarn = cx.yarn().unwrap_or(kernels_cuda_new::x::Yarn::NONE);
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rope_partial_last_bf16(
        &ctx,
        q,
        cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
        cx.positions()?,
        cx.rows().count,
        cx.out_width(0)? / kv.head_dim,
        kv_heads,
        kv.head_dim,
        cx.rotary_width()?,
        cx.theta()?,
        false,
        cx.rope_interleaved(),
        yarn.factor,
        yarn.beta_fast,
        yarn.beta_slow,
        yarn.original_max_position,
    )
}

/// `rope::rope_partial_bf16`
fn rope_partial_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let kv = cx.kv_layer()?;
    if kv.head_dim <= 0 {
        return Err(Refusal::Empty { what: "kv head_dim" });
    }
    let q = cx.arg_out(0)?.cast::<bf16>();
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rope_partial_bf16(
        &ctx,
        q,
        cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
        cx.positions()?,
        0,
        cx.rows().count,
        cx.out_width(0)? / kv.head_dim,
        cx.out_width(1).map_or(0, |w| w / kv.head_dim),
        kv.head_dim,
        cx.rotary_width()?,
        cx.theta()?,
    )
}

/// `rope::qk_rmsnorm_rope_bf16_rounded`
fn qk_rmsnorm_rope_bf16_rounded_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let kv = cx.kv_layer()?;
    if kv.head_dim <= 0 {
        return Err(Refusal::Empty { what: "kv head_dim" });
    }
    let k = cx.arg_out(1).unwrap_or(core::ptr::null_mut()).cast::<bf16>();
    let k_weight = cx.weight(1).unwrap_or(core::ptr::null_mut()).cast_const().cast::<bf16>();
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    qk_rmsnorm_rope_bf16_rounded(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        k,
        cx.weight(0)?.cast_const().cast::<bf16>(),
        k_weight,
        cx.positions()?,
        cx.rows().count,
        cx.out_width(0)? / kv.head_dim,
        cx.out_width(1).map_or(0, |w| w / kv.head_dim),
        kv.head_dim,
        cx.theta()?,
        cx.rms_eps()?,
    )
}

/// `rope::rope_yarn_original_bf16`
fn rope_yarn_original_bf16_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let head_dim = cx.head_dim()?;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    let yarn = cx.yarn()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    rope_yarn_original_bf16(
        &ctx,
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.positions()?,
        cx.rows().count,
        cx.out_width(0)? / head_dim,
        cx.out_width(1)? / head_dim,
        head_dim,
        cx.rope_theta()?,
        yarn.factor,
        yarn.beta_fast,
        yarn.beta_slow,
        yarn.attention_factor,
        yarn.original_max_position,
        cx.rope_interleaved(),
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound {
        symbol: "rope::rope_standard_table",
        arm: Some(rope_standard_table_arm),
        unbound: None,
    },
    Bound { symbol: "rope::rope_bf16", arm: Some(rope_bf16_arm), unbound: None },
    Bound {
        symbol: "rope::qk_rmsnorm_rope_bf16",
        arm: Some(qk_rmsnorm_rope_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "rope::qk_rmsnorm_rope_bf16_devwin",
        arm: Some(qk_rmsnorm_rope_bf16_devwin_arm),
        unbound: None,
    },
    Bound {
        symbol: "rope::rope_partial_last_bf16",
        arm: Some(rope_partial_last_bf16_arm),
        unbound: None,
    },
    Bound { symbol: "rope::rope_partial_bf16", arm: Some(rope_partial_bf16_arm), unbound: None },
    Bound {
        symbol: "rope::qk_rmsnorm_rope_bf16_rounded",
        arm: Some(qk_rmsnorm_rope_bf16_rounded_arm),
        unbound: None,
    },
    Bound {
        symbol: "rope::rope_yarn_original_bf16",
        arm: Some(rope_yarn_original_bf16_arm),
        unbound: None,
    },
    Bound {
        symbol: "rope::rope_yarn_bf16",
        arm: None,
        unbound: Some(
            "rope_yarn: llama-3's low_freq_factor/high_freq_factor. No statement \
         and no context carries them, and the YaRN quartet the context does \
         carry is a different scheme with the same arity",
        ),
    },
    Bound {
        symbol: "rope::qk_rmsnorm_mrope_bf16",
        arm: None,
        unbound: Some(
            "qk_rmsnorm_mrope: the (t, h, w) section split. A property of a \
         vision checkpoint that no statement and no context carries",
        ),
    },
    Bound {
        symbol: "rope::rope_partial_bf16_position_delta",
        arm: None,
        unbound: Some(
            "rope_partial_position_delta: the offset added to every position. A \
         fact about a draft/verify pairing that no statement carries",
        ),
    },
    Bound {
        symbol: "rope::rope_write_kv_bf16",
        arm: None,
        unbound: Some(
            "rope_write_kv: the contract states no in_place pair, so which \
         addresses q and k rotate at is not something the declaration \
         determines. Every other operand is reachable",
        ),
    },
];
