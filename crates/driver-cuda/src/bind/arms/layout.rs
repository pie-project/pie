//! What happens when a trace states one of `layout`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through [`Cx`], so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::bf16;
use kernels_cuda::layout::*;

use super::super::cx::Cx;
use super::Bound;

/// `layout::split_bf16_rows`
fn split_rows_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    split_bf16_rows(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.out_width(1)?,
    )
}

/// `layout::split_qwen_gdn_ba_bf16`
fn split_qwen_gdn_ba_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    split_qwen_gdn_ba::<bf16>(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.arg_out(1)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `layout::embed_bf16`
fn embed_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    embed_bf16(
        &ctx,
        cx.token_ids()?,
        cx.weight_named(0)?.cast_const().cast::<bf16>(),
        cx.arg_out(0)?.cast::<bf16>(),
        cx.rows().count,
        cx.out_width(0)?,
        cx.vocab()?,
    )
}

/// `layout::gather_bf16_rows`
fn gather_rows_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    gather_bf16_rows(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u16>(),
        cx.sampling_indices()?,
        cx.arg_out(0)?.cast::<u16>(),
        cx.rows().count,
        cx.out_width(0)?,
    )
}

/// `layout::transpose_bf16_nld_to_lnd`
fn transpose_nld_to_lnd_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let ple_dim = cx.ple_dim()?;
    if ple_dim <= 0 {
        return Err(Refusal::Empty { what: "ple_dim" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    transpose_bf16_nld_to_lnd(
        &ctx,
        cx.arg_in(0)?.cast_const().cast::<u16>(),
        cx.arg_out(0)?.cast::<u16>(),
        cx.rows().count,
        cx.in_width(0)? / ple_dim,
        ple_dim,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    // DECLARED IN `sigs()` AND ARMED BY NOBODY, which is a state this
    // registry can hold and a bare absence cannot. Before the declaration
    // landed a fire naming it refused `NoArm` -- a message about dispatch,
    // naming neither what was missing nor who would supply it.
    Bound {
        symbol: "layout::split_q_gate_bf16",
        arm: None,
        unbound: Some(
            "this symbol has a HOST PROGRAM and no arm. \
             kernels_cuda::driver_internal::split_q_gate_bf16 is the \
             body -- a plain pub fn the driver is meant to call by path -- \
             and nothing in this crate calls it. The gap is not the kernel: \
             OpKind::SplitQGate arrives with no bind written, so a fire \
             reaching it refused NoArm and named neither the body nor its \
             module. FLOOR: call driver_internal::split_q_gate_bf16 from an \
             arm here, over the packed bank in and the q/gate halves out",
        ),
    },

    Bound { symbol: "layout::split_bf16_rows", arm: Some(split_rows_arm), unbound: None },
    Bound {
        symbol: "layout::split_qwen_gdn_ba_bf16",
        arm: Some(split_qwen_gdn_ba_arm),
        unbound: None,
    },
    Bound { symbol: "layout::embed_bf16", arm: Some(embed_arm), unbound: None },
    Bound { symbol: "layout::gather_bf16_rows", arm: Some(gather_rows_arm), unbound: None },
    Bound {
        symbol: "layout::transpose_bf16_nld_to_lnd",
        arm: Some(transpose_nld_to_lnd_arm),
        unbound: None,
    },
];
