//! ATTENTION FORMS — the device-window spellings and the head-dim
//! padding a paged kernel needs.

use super::*;

// ── the DEVICE-WINDOW forms ────────────────────────────────────
//
// A hooked pure-decode fire is graph-CAPTURED, and its hook split rides a
// DEVICE word (`win_d`) rather than a host row range. That is what makes
// these their own statements: the window is not a number the lowering
// knows, so it cannot be expressed as a rectangle -- every one is
// `whole`, and for a reason no other `whole` row in this table gives.
//
// `.wiki/tart/dsl.md`'s step 2e found this path by surveying before
// deleting; these are the statements that survey was about.

/// `kernels::rope::qk_rmsnorm_rope_bf16_devwin`: the fused q/k norm and
/// rope, over a device-carried window.
pub fn qk_rmsnorm_rope_devwin(q: &Val, k: &Val, q_w: &str, k_w: &str, q_width: u32) -> Val {
    record(
        &q.t,
        q.layer,
        "rope::qk_rmsnorm_rope_bf16_devwin",
        vec![q_w.to_string(), k_w.to_string()],
        None,
        vec![q.id, k.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the norm+rope produces its value")
}

/// `kernels::attn::write_kv_explicit_bf16_devwin`: the explicit-slot
/// write, over a device-carried window.
pub fn write_kv_explicit_devwin(k: &Val, v: &Val, l: u32) {
    record(
        &k.t,
        Some(l),
        "attn::write_kv_explicit_bf16_devwin",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![k.id, v.id],
        None,
    );
}

// ── head-dim padding, and the rest ─────────────────────────────

/// `kernels::attn::pad_head_dim_bf16` / `kernels::attn::strip_head_dim_bf16`:
/// widen each head to the padded width a kernel demands, and narrow back.
///
/// The pair is what `head_dim_padded` COSTS, and stating it is what turns
/// `if (c.head_dim_padded)` in the model body into a fact the trace
/// carries. Row-shaped: each token's heads are padded independently.
pub fn pad_head_dim(x: &Val, heads: u32, head_dim_padded: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "attn::pad_head_dim_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(heads),
                Dim::Const(head_dim_padded),
            ]),
            DType::BF16,
        )),
    )
    .expect("the pad produces its value")
}

/// The inverse of [`pad_head_dim`](crate::cuda::pad_head_dim).
pub fn strip_head_dim(x: &Val, heads: u32, head_dim: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "attn::strip_head_dim_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the strip produces its value")
}

// `dsl::cuda::merge_attention_states` WAS HERE, stating `attn::merge_attention_states_bf16`.
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

/// `kernels::attn::compact_page_csr`: drop the pages a keep-mask
/// excludes, rewriting the CSR.
///
/// `whole`: it rewrites `[R+1]` indptr arrays, so a row window would
/// compact the wrong requests' page lists.
pub fn compact_page_csr(t: &Trace, l: u32, keep: &Val) -> Val {
    record(
        t,
        Some(l),
        "attn::compact_page_csr",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![keep.id],
        Some((Shape(vec![Dim::Requests]), DType::I32)),
    )
    .expect("the compaction produces its value")
}

/// `kernels::attn::attn_score_fold_heads`: fold the captured per-head scores
/// into the per-request form an observer reads.
pub fn attn_score_fold_heads(scores: &Val, heads: u32) -> Val {
    record(
        &scores.t,
        scores.layer,
        "attn::attn_score_fold_heads",
        vec![],
        None,
        vec![scores.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(heads)]), DType::F32)),
    )
    .expect("the fold produces its value")
}

/// `kernels::gemm::mla_absorb_q_to_latent_bf16`: project the query into the latent
/// space MLA attends in.
///
/// A cuBLAS op, not a raw launch -- and that is why the vocabulary audit
/// missed it twice: a launcher is anything that issues DEVICE work, and
/// there are two ways to do that here.
/// `v_head_dim` rides the PARAM channel: the bank is sliced by it and
/// this direction's result does not carry it.
pub fn mla_absorb_q_to_latent(
    q_nope: &Val,
    w: &str,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    qk_nope_dim: u32,
) -> Val {
    record_with_params(
        &q_nope.t,
        q_nope.layer,
        "gemm::mla_absorb_q_to_latent_bf16",
        vec![w.to_string()],
        None,
        vec![heads, qk_nope_dim, v_head_dim, kv_lora_rank],
        vec![q_nope.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(heads),
                Dim::Const(kv_lora_rank),
            ]),
            DType::BF16,
        )),
    )
    .expect("the absorb produces its value")
}

/// `kernels::gemm::mla_absorb_latent_to_v_bf16`: project the latent attention
/// output back to the value space.
pub fn mla_absorb_latent_to_v(
    latent: &Val,
    w: &str,
    heads: u32,
    v_head_dim: u32,
    qk_nope_dim: u32,
    kv_lora_rank: u32,
) -> Val {
    record_with_params(
        &latent.t,
        latent.layer,
        "gemm::mla_absorb_latent_to_v_bf16",
        vec![w.to_string()],
        None,
        vec![heads, qk_nope_dim, v_head_dim, kv_lora_rank],
        vec![latent.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(v_head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the absorb produces its value")
}

// `dsl::cuda::flashinfer_mamba_ssu` WAS HERE, stating `ssm::flashinfer_mamba_ssu_bf16`.
//
// The row it named is gone -- `kernels-cuda/src/table/ssm.rs` carries
// the tombstone and the reason -- and the reason was that its whole
// consumer set was this wrapper, which nothing called. Deleting one
// half left the other stating a symbol no table declares, which is
// not a harmless leftover: `check_plan` refuses an undeclared symbol
// at LOAD, so a text that reached for this would fail late and for a
// reason with no bearing on what it was trying to do.
//
// Re-adding is a row and a wrapper, together. Either alone is a trap.

/// `kernels::gemm::act_x_wt_bf16_out_fp32`: the same, accumulating to fp32.
pub fn gemm_out_fp32(act: &Val, w: &str, n: u32) -> Val {
    record(
        &act.t,
        act.layer,
        "gemm::act_x_wt_bf16_out_fp32",
        vec![w.to_string()],
        None,
        vec![act.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::F32)),
    )
    .expect("the gemm produces its value")
}

/// `kernels::gemm::act_x_wt_bf16`: the plain `x · Wᵀ`.
///
/// Stated because families FIRE it — glm5's projections, nemotron_h's,
/// qwen3_5's router. It went missing from the table for as long as it
/// did because it is an `inline void` forwarder in `ops/gemm.hpp`, and
/// the audit's launcher regex required the return type to start the
/// line; the fix to that regex is what surfaced this.
///
/// Distinct from the ordinary `Matmul` op: this is the CUDA reading
/// for a projection whose weight the family names directly rather
/// than through the `layer.{l}.{field}` binding — the DSA indexer's,
/// for one.
pub fn gemm_xwt(act: &Val, w: &str, n: u32) -> Val {
    record(
        &act.t,
        act.layer,
        "gemm::act_x_wt_bf16",
        vec![w.to_string()],
        None,
        vec![act.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
    )
    .expect("the gemm produces its value")
}

// `dsl::cuda::gemm_batched_xwt` WAS HERE, stating `gemm::batched_act_x_wt_bf16`.
//
// The row it named is gone -- `kernels-cuda/src/table/gemm.rs` carries
// the tombstone and the reason -- and the reason was that its whole
// consumer set was this wrapper, which nothing called. Deleting one
// half left the other stating a symbol no table declares, which is
// not a harmless leftover: `check_plan` refuses an undeclared symbol
// at LOAD, so a text that reached for this would fail late and for a
// reason with no bearing on what it was trying to do.
//
// Re-adding is a row and a wrapper, together. Either alone is a trap.

/// `kernels::gemm::grouped_act_x_wt_bf16`: one GEMM per group, batched.
///
/// `whole`: the group boundaries (`M_array`) are fire-global, so a row
/// window would cut a group in half.
pub fn gemm_grouped(act: &Val, w: &str, n: u32) -> Val {
    record(
        &act.t,
        act.layer,
        "gemm::grouped_act_x_wt_bf16",
        vec![w.to_string()],
        None,
        vec![act.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(n)]), DType::BF16)),
    )
    .expect("the gemm produces its value")
}

/// `kernels::layout::split_bf16_rows`: split `[N, l+r]` into `[N, l]` and
/// `[N, r]`.
///
/// Its inverse, `concat_rows`, was deleted in §54 — it had no caller and
/// no dispatch arm, and this one has both. An operation being half of a
/// pair is not a reason to keep the other half.
pub fn split_rows(src: &Val, left_dim: u32, right_dim: u32) -> (Val, Val) {
    let outs = record_many(
        &src.t,
        src.layer,
        "layout::split_bf16_rows",
        vec![],
        vec![src.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(left_dim)]), DType::BF16),
            (Shape(vec![Dim::Tokens, Dim::Const(right_dim)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let l = it.next().expect("the split states two outputs");
    let r = it.next().expect("the split states two outputs");
    (l, r)
}

/// `kernels::layout::split_qwen_gdn_ba_bf16`: split the GDN `ba`
/// projection into its beta and alpha halves.
pub fn split_qwen_gdn_ba(ba: &Val, v_h: u32) -> (Val, Val) {
    let outs = record_many(
        &ba.t,
        ba.layer,
        "layout::split_qwen_gdn_ba_bf16",
        vec![],
        vec![ba.id],
        vec![
            (Shape(vec![Dim::Tokens, Dim::Const(v_h)]), DType::BF16),
            (Shape(vec![Dim::Tokens, Dim::Const(v_h)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let b = it.next().expect("the split states two outputs");
    let a = it.next().expect("the split states two outputs");
    (b, a)
}
