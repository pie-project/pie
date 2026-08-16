//! TENSOR PARALLELISM — the shard shapes, and the collectives that
//! recombine them.
//!
//! This divider used to sit 1,290 lines from the end of `cuda.rs` with
//! everything after it under its heading, which is how the most-stated
//! surface in the crate -- `swiglu`, `rope`, the flashinfer dispatches,
//! the KV writes -- came to be filed under "tensor parallelism". The
//! collectives are what the heading was ever about; the rest is
//! `base.rs`.

use super::*;

// ── tensor-parallel shapes ─────────────────────────────────────

/// `kernels::norm::residual_add_rmsnorm_bf16`: the residual add and the
/// next block's pre-norm, fused.
///
/// `hidden = round_bf16(hidden + residual)` then
/// `norm_out = rmsnorm(hidden, weight)`. The kernel's own header states
/// that the rounding matches `kernels::norm::residual_add_bf16`'s, so this is
/// numerically the two-kernel sequence and not an approximation of it —
/// which is what makes it a BINDING choice a declaration may state rather
/// than a different computation.
pub fn residual_add_rmsnorm(x: &Val, residual: &Val, weight: &str, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::residual_add_rmsnorm_bf16",
        vec![weight.to_string()],
        None,
        vec![x.id, residual.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the fused norm produces its value")
}


// ── TENSOR PARALLELISM ─────────────────────────────────────────
//
// A collective is a STATEMENT. It is real device work with operands
// and a result, and the only reason it has not been one is that the
// hand-written passes reached for `tp->` directly.
//
// Sharding itself needs no vocabulary: a rank's trace states ITS
// widths, and the text divides by `tp_size` from the facts the way
// it already divides by anything else. What needs vocabulary is the
// point where the shards are recombined, because that is a launch.

/// `comm::all_reduce_bf16`: the NVLink P2P sum, out of place.
///
/// ONE ARM OF A CHOICE THE DRIVER USED TO MAKE. `NcclComm::
/// all_reduce_bf16` asks `can_handle(bytes)` and routes to this
/// kernel below the threshold, `ncclAllReduce` above it — an `if`
/// inside a driver method picking between two implementations,
/// which is the shape this whole arc removes.
///
/// So a text states the pair as a GUARD, the way qwen3.5's
/// recurrence states its three spellings and the fused landing
/// states its two. The predicate is the message size, which is
/// `TokensLE` — the threshold is bytes, and a row of `hidden` bf16
/// elements is a fixed number of them, so the token count IS the
/// test once the deployment's hidden size is known.
///
/// What does NOT reduce to the predicate is buffer REGISTRATION:
/// the P2P kernel reads only buffers handed to `register_buffer`,
/// which is a placement fact of the deployment rather than a
/// property of the fire. It belongs on the facts beside
/// `gate_up_fused` — a load-time answer that erases into the trace.
pub fn all_reduce_p2p(x: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "comm::all_reduce_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the collective produces its value")
}

/// `dist::all_reduce_bf16`: sum this value across ranks, in place.
///
/// The in-place form, which is what a post-norm landing takes: the
/// partial is summed where it lies and the statement's result is the
/// same bytes.
///
/// THE OTHER ARM of [`all_reduce_p2p`]'s choice, and the one that is
/// not a kernel: NCCL is the comm plane, and `custom_all_reduce.hpp`
/// says in as many words where that knowledge belongs — with the
/// caller, not with a compute kernel. So this symbol has no
/// `kernel!` operand signature and cannot get one without moving
/// NCCL down a layer, which is a decision that was already made in
/// the other direction once.
///
/// It is still a STATEMENT. What a symbol needs to be stated is a
/// name the declaration can choose and an arm that binds it; a
/// generated ABI entry point is a separate benefit that this one
/// does not get.
pub fn all_reduce(x: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "dist::all_reduce_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the collective produces its value")
}

/// `dist::all_reduce_bf16_out`: sum this value across ranks into a
/// SEPARATE destination.
///
/// The two-step landing's first half. It reads as the same
/// collective and it is; what differs is that the result is not the
/// operand's bytes, because the residual add downstream needs both.
pub fn all_reduce_out(x: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "dist::all_reduce_bf16_out",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the collective produces its value")
}

/// `dist::all_gather_bf16`: concatenate this value's shards along
/// its row width. The result is `parts` times as wide.
pub fn all_gather(x: &Val, parts: u32, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "dist::all_gather_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(width * parts)]),
            DType::BF16,
        )),
    )
    .expect("the collective produces its value")
}

/// `comm::all_reduce_residual_rmsnorm_bf16`: the FUSED landing —
/// sum the shards, add the residual, and norm, in one launch.
///
/// TWO results, because the kernel has two effects: the residual
/// stream is updated IN PLACE (operand 1, which the `kernel!` row
/// aliases output 0 over) and the normed activation is written
/// fresh. Returned in that order.
///
/// WHETHER TO FUSE IS A GUARD, not a driver test. The hand-written
/// pass asks `can_fuse_residual_rmsnorm(tokens, hidden, stream)` at
/// fire time; `hidden` and the buffer registration are load-time
/// facts that resolve into the trace, and what is left —
/// `tokens` — is exactly `GuardPred::TokensLE`. So a text states
/// the fused arm under that predicate and the two-step form as the
/// else, the same shape qwen3.5's recurrence uses for its three
/// spellings.
pub fn all_reduce_residual_rmsnorm(
    x: &Val,
    residual: &Val,
    weight: &NormW,
    hidden: u32,
) -> (Val, Val) {
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let outs = x.t.with(x.layer, |b| {
        b.launch(
            "comm::all_reduce_residual_rmsnorm_bf16",
            vec![weight.name.clone()],
            None,
            vec![x.id, residual.id],
            vec![shape.clone(), shape],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    (mk(outs[0]), mk(outs[1]))
}
