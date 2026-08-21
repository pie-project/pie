//! Attention CUDA statements with device windows, cache writes, or input-shaped outputs.

use super::*;



/// `kernels::rope::qk_rmsnorm_rope_bf16_devwin`.
/// `outputs[0]` = q, `outputs[1]` = k; shapes are read from inputs.
///
/// The run is `[head_dim, theta, eps, n_max, win_start, win_len]`: `n_max`
/// is the fire's row count (a spliced extent), and the window pair is the
/// peel split — lowering-spliced (design-no-ask §3 category E), so the
/// statement carries zero placeholders at those slots.
#[must_use]
pub fn qk_rmsnorm_rope_devwin(
    q: &Val,
    k: &Val,
    q_w: &str,
    k_w: &str,
    head_dim: u32,
    theta: f32,
    eps: f32,
) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch_devwin(
            "rope::qk_rmsnorm_rope_bf16_devwin",
            vec![q_w.to_string(), k_w.to_string()],
            None,
            vec![head_dim, theta.to_bits(), eps.to_bits(), 0, 0, 0],
            vec![(3, Shape(vec![Dim::Tokens]))],
            // The walk fills 4/5 with this launch's own rectangle — the
            // peel split no statement can state.
            Some((4, 5)),
            vec![q.id, k.id, positions],
            vec![(q_sh, DType::BF16), (k_sh, DType::BF16)],
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::attn::write_kv_explicit_bf16_devwin`.
/// Writes the KV cache and returns no value.
///
/// Run `[num_kv_heads, head_dim, n_max, win_start, win_len]`: `n_max` is a
/// spliced extent, the window pair is the peel split the lowering writes
/// (zero placeholders here).
pub fn write_kv_explicit_devwin(k: &Val, v: &Val, l: u32, num_kv_heads: u32, head_dim: u32) {
    let kvc = rt_object(&k.t, "kv_cache", Some(l));
    let row_valid = rt_tokens(&k.t, "row_valid");
    record_devwin(
        &k.t,
        Some(l),
        "attn::write_kv_explicit_bf16_devwin",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![num_kv_heads, head_dim, 0, 0, 0],
        vec![tokens_extent(2)],
        // The walk fills 3/4 with this launch's own rectangle.
        Some((3, 4)),
        vec![k.id, v.id, kvc, row_valid],
        None,
    );
}


builder! {
    /// `kernels::attn::pad_head_dim` / `kernels::attn::strip_head_dim`:
    /// `params[0]` is the PACKED head dim — the padded one is read off the
    /// result's width.
    pub fn pad_head_dim(x: &Val, heads: u32, head_dim_padded: u32, head_dim: u32) -> Val {
        symbol: "attn::pad_head_dim",
        on: x,
        params: [head_dim],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim_padded)] as BF16,
        made: "the pad produces its value",
    }


    /// The inverse of [`pad_head_dim`](crate::cuda::pad_head_dim); the same
    /// packed-head-dim scalar.
    pub fn strip_head_dim(x: &Val, heads: u32, head_dim: u32) -> Val {
        symbol: "attn::strip_head_dim",
        on: x,
        params: [head_dim],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the strip produces its value",
    }



    /// `kernels::attn::attn_score_fold_heads`.
    pub fn attn_score_fold_heads(scores: &Val, heads: u32) -> Val {
        symbol: "attn::attn_score_fold_heads",
        on: scores,
        inputs: [scores],
        out: [Dim::Tokens, Dim::Const(heads)] as F32,
        made: "the fold produces its value",
    }


    /// `kernels::gemm::act_x_wt_bf16_out_fp32`: the same, accumulating to fp32.
    pub fn gemm_out_fp32(act: &Val, w: &str, n: u32) -> Val {
        symbol: "gemm::act_x_wt_bf16_out_fp32",
        on: act,
        weights: [w],
        inputs: [act],
        out: [Dim::Tokens, Dim::Const(n)] as F32,
        made: "the gemm produces its value",
    }


    /// `kernels::gemm::act_x_wt_bf16`: the plain `x · Wᵀ`.
    pub fn gemm_xwt(act: &Val, w: &str, n: u32) -> Val {
        symbol: "gemm::act_x_wt_bf16",
        on: act,
        weights: [w],
        inputs: [act],
        out: [Dim::Tokens, Dim::Const(n)] as BF16,
        made: "the gemm produces its value",
    }



}

/// `kernels::gemm::grouped_act_x_wt_bf16`: one GEMM per group, batched.
///
/// The swept signature reads everything through the raised
/// `In<Struct<GemmGroups>>` view — pointer arrays and the host M array —
/// and writes through the view's `out_ptrs`, so the statement declares no
/// result and places no activation or weight of its own. The run is
/// `[group_count, beta, n, k]`.
pub fn gemm_grouped(
    t: &Trace,
    layer: Option<u32>,
    group_count: u32,
    beta: f32,
    n: u32,
    k: u32,
) {
    let groups = rt_object(t, "gemm.groups", layer);
    record_with_params(
        t,
        layer,
        "gemm::grouped_act_x_wt_bf16",
        vec![],
        None,
        vec![group_count, beta.to_bits(), n, k],
        vec![groups],
        None,
    );
}


/// `kernels::attn::compact_page_csr`.
/// No results: the launcher's output arrays are `PageMask` arena buffers.
pub fn compact_page_csr(t: &Trace, l: u32, keep: &Val) {
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
        None,
    );
}

/// `kernels::gemm::mla_absorb_q_to_latent_bf16`.
/// `params[0..5] = [heads, qk_nope_dim, v_head_dim, kv_lora_rank, tokens]`;
/// the token count is the fire's and spliced by the lowering.
pub fn mla_absorb_q_to_latent(
    q_nope: &Val,
    w: &str,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    qk_nope_dim: u32,
) -> Val {
    record_with_extents(
        &q_nope.t,
        q_nope.layer,
        "gemm::mla_absorb_q_to_latent_bf16",
        vec![w.to_string()],
        None,
        vec![heads, qk_nope_dim, v_head_dim, kv_lora_rank, 0],
        vec![tokens_extent(4)],
        vec![q_nope.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(kv_lora_rank)]),
            DType::BF16,
        )),
    )
    .expect("the absorb produces its value")
}

/// `kernels::gemm::mla_absorb_latent_to_v_bf16`; the same five-scalar run as
/// [`mla_absorb_q_to_latent`].
pub fn mla_absorb_latent_to_v(
    latent: &Val,
    w: &str,
    heads: u32,
    v_head_dim: u32,
    qk_nope_dim: u32,
    kv_lora_rank: u32,
) -> Val {
    record_with_extents(
        &latent.t,
        latent.layer,
        "gemm::mla_absorb_latent_to_v_bf16",
        vec![w.to_string()],
        None,
        vec![heads, qk_nope_dim, v_head_dim, kv_lora_rank, 0],
        vec![tokens_extent(4)],
        vec![latent.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(v_head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the absorb produces its value")
}



builder! {
    /// `kernels::layout::split_bf16_rows`.
    pub fn split_rows(src: &Val, left_dim: u32, right_dim: u32) -> (Val, Val) {
        symbol: "layout::split_bf16_rows",
        on: src,
        inputs: [src],
        outs: [
            [Dim::Tokens, Dim::Const(left_dim)] as BF16,
            [Dim::Tokens, Dim::Const(right_dim)] as BF16,
        ],
        made: "the split states two outputs",
    }


    /// `kernels::layout::split_qwen_gdn_ba`.
    pub fn split_qwen_gdn_ba(ba: &Val, v_h: u32) -> (Val, Val) {
        symbol: "layout::split_qwen_gdn_ba",
        on: ba,
        inputs: [ba],
        outs: [
            [Dim::Tokens, Dim::Const(v_h)] as BF16,
            [Dim::Tokens, Dim::Const(v_h)] as BF16,
        ],
        made: "the split states two outputs",
    }
}
