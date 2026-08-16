//! QWEN 3.5 — multi-token prediction, and the statements its
//! recurrence needs that no other generation states.

use super::*;

// ── qwen3_5: multi-token prediction ────────────────────────────
//
// MTP drafts several tokens per step and repairs when a draft is
// rejected, which needs two things the rest of the model does not: an
// attention that can see a HISTORY buffer alongside the pages (the
// rejected tokens are not committed), and a per-slot pending-hidden
// shuffle. All four address through `slot_ids` or `qo_indptr`, so all
// four are `whole`.

// `dsl::cuda::attention_mtp_paged_history` WAS HERE, stating `attn::attention_mtp_paged_history_bf16`.
//
// The row it named is gone -- `kernels-cuda/src/table/attn.rs` carries
// the tombstone and the reason -- and the reason was that its whole
// consumer set was this wrapper, which nothing called. Deleting one
// half left the other stating a symbol no table declares, which is
// not a harmless leftover: `check_plan` refuses an undeclared symbol
// at LOAD, so a text that reached for this would fail late and for a
// reason with no bearing on what it was trying to do.
//
// Re-adding is a row and a wrapper, together. Either alone is a trap.

/// `kernels::attn::mtp_shift_hidden_bf16`: the previous step's pending
/// hidden, shifted into this step's rows.
pub fn mtp_shift_hidden(target: &Val, pending: &Val, hidden: u32) -> Val {
    record(
        &target.t,
        target.layer,
        "attn::mtp_shift_hidden_bf16",
        vec![],
        None,
        vec![target.id, pending.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the shift produces its value")
}

/// `kernels::attn::mtp_update_pending_hidden_bf16`: stash each request's
/// last hidden for the next step.
pub fn mtp_update_pending_hidden(target: &Val, l: u32) {
    record(
        &target.t,
        Some(l),
        "attn::mtp_update_pending_hidden_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![target.id],
        None,
    );
}

// ── FOUR WRAPPERS WERE HERE AND ARE DELETED. §54. ──────────────
//
// `copy_if_valid_slot`, `concat_rows`, `deinterleave_rows` and
// `deinterleave_vec`, for `layout::copy_if_valid_slot`,
// `layout::concat_bf16_rows`, `layout::deinterleave_rows_bf16` and
// `layout::deinterleave_vec_bf16`. Each had zero callers in
// `crates/model/src`, and so did the four symbols they recorded.
//
// They are recorded here rather than simply removed because their
// existence was the BUG, not an accident of it. §28: this surface was
// generated from launcher headers, so a wrapper existed for every
// launcher whether or not any model wanted one -- and a wrapper reads as
// demand to every tool that stops at it. Four `table::layout` rows were
// held live by nothing but these four functions. The rows went in the
// same edit, because `model/tests/kernels_table.rs::
// the_table_covers_the_dsl_surface` asserts this surface and that table
// are the same set, and half the edit fails it.
//
// If a model ever wants one of them back, the kernel is still there:
// `families::layout`'s device rows and the `.cuh` text are untouched,
// and `kernels-cuda/tests/launch_rules.rs` still fires
// `copy_if_valid_slot` three times as the tree's only witness for
// `LaunchRule::Single`. What has to come back is a row and a wrapper
// TOGETHER, with a caller.

// ── qwen3_5: the rest ──────────────────────────────────────────

/// `kernels::norm::rmsnorm_gated_bf16`: the gated RMS norm, in its own
/// launch rather than folded into a projection.
pub fn rmsnorm_gated_launch(x: &Val, gate: &Val, weight: &str, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "norm::rmsnorm_gated_bf16",
        vec![weight.to_string()],
        None,
        vec![x.id, gate.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the norm produces its value")
}

/// `kernels::moe::moe_grouped_gemm_bf16`: the grouped expert GEMM.
/// The bank is named, like any other matmul's weight. It is ONE tensor
/// (`[E, N, K]`) that the kernel indexes by the block's expert id, not a
/// per-expert selection, so the traced name carries the `{e}` the family
/// spells it with and the binding resolves to the whole bank. Without
/// this the statement said "a grouped GEMM" and left which weights
/// entirely to the executor -- readable, but not a declaration.
/// The two block numbers ride the PARAM channel, for the reason
/// [`gather_moe_aligned_inputs`]'s `top_k` does: the kernel takes
/// `max_blocks` and the per-block row count `m`, and the statement's
/// operands carry only their PRODUCT — the aligned rectangle's
/// leading extent. `n` and `k` are the result's and the operand's row
/// widths and need no help; these two do. Same numbers
/// [`moe_align`] already states, and it is the same permutation.
pub fn moe_grouped_gemm(
    act: &Val,
    expert_ids: &Val,
    stage: &Val,
    aligned: Dim,
    width: u32,
    bank: &str,
    block_size: u32,
    max_blocks: u32,
) -> Val {
    record_with_params(
        &act.t,
        act.layer,
        "moe::moe_grouped_gemm_bf16",
        vec![bank.to_string()],
        None,
        vec![block_size, max_blocks],
        // The second operand is the ALIGN's per-block expert id --
        // what the kernel indexes the bank by. It used to be the
        // sorted route order, which the kernel never reads: the
        // statement named one array and the executor bound another
        // (`mw.aligned_expert_ids`), so the declaration could not be
        // checked against the call. The third is the DESTINATION,
        // named by the pointer build above and written in place.
        vec![act.id, expert_ids.id, stage.id],
        // Block-major rows, not tokens: the operand this multiplies is
        // the gathered aligned bank, and saying `Tokens` here made the
        // routed leg's values indistinguishable from the shared
        // expert's -- which is exactly the question an executor has to
        // answer to pick a buffer.
        Some((Shape(vec![aligned, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the gemm produces its value")
}

/// `kernels::sample::lm_head_gemv_argmax_int8`: the readout and the argmax
/// in one launch, over an int8 head with a per-channel scale.
///
/// It produces TOKEN IDS, not logits. A greedy-decode fast path that
/// never materializes the vocab-wide row -- which is why it is its own
/// statement rather than `lm_head` followed by an argmax.
pub fn lm_head_gemv_argmax_int8(x: &Val, weight: &str, scale: &str) -> Val {
    record(
        &x.t,
        None,
        "sample::lm_head_gemv_argmax_int8",
        vec![weight.to_string(), scale.to_string()],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Requests]), DType::I32)),
    )
    .expect("the readout produces its value")
}
