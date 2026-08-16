//! What happens when a trace states one of `xqa`'s symbols.
//!
//! These were `bind!` arms inside `kernels-cuda`. They read the driver's
//! own vocabulary through `Cx`, so they belong on this side of the seam:
//! the kernels crate exposes routines, and joining a statement to one is the
//! driver's job.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::jit::Ctx;
use kernels_cuda::attn::xqa::{XqaIoHead, attention_xqa_decode_bf16_prepared};

use super::super::cx::Cx;
use super::Bound;

/// `attn::attention_xqa_decode_bf16_prepared`
///
/// # `max_pages_per_seq` is the fire's TOTAL page count, and that is a bound
///
/// The routine wants the dense page table's row stride to cover every
/// request, and the driver names no per-request maximum: `Cx` answers
/// `num_pages_in_batch`, which is `kv_page_indptr`'s tail — the pages the
/// whole fire's CSR holds — and that is `>=` any one request's share by
/// construction. The tight number would be
/// `max(indptr[r + 1] - indptr[r])`, and reading it means reading a DEVICE
/// array on the host, which is a synchronise a fire may not make; the host
/// mirror `AttnCtx::kv_page_indptr_h` exists only when a planless prefill is
/// stated, and no `Cx` query answers a host pointer anyway.
///
/// What the looseness costs is the page table, which is
/// `num_requests * page_bucket(stride) * 4` bytes and is carved out of the
/// attention workspace — so a request set wide enough to overflow it is a
/// [`Refusal::Wide`] naming the carve rather than a wrong answer.
/// `page_bucket` clamps the stride at 4096, so the cost is bounded. Nothing
/// else widens: the bucketed `maxSeqLen` only raises the multi-block split's
/// second term, and the multiprocessor count already dominates it.
fn xqa_decode_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let layer = cx.kv_layer()?;
    let plan = cx.plan()?;
    let workspace = cx.attn_workspace()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    attention_xqa_decode_bf16_prepared(
        &ctx,
        cx.arg_in(0)?.cast::<XqaIoHead>().cast_const(),
        cx.arg_out(0)?.cast::<XqaIoHead>(),
        layer.k_pages,
        layer.v_pages,
        plan.kv_page_indices,
        plan.kv_page_indptr,
        plan.kv_last_page_lens,
        workspace.float_buffer,
        workspace.float_bytes,
        workspace.int_buffer,
        workspace.int_bytes,
        plan.requests,
        cx.num_q_heads()?,
        layer.num_kv_heads,
        layer.head_dim,
        layer.page_size,
        cx.num_pages_in_batch()?,
        cx.sm_scale()?,
    )
}

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[Bound {
    symbol: "attn::attention_xqa_decode_bf16_prepared",
    arm: Some(xqa_decode_arm),
    unbound: None,
}];
