//! Attention CUDA statements with device windows, cache writes, or input-shaped outputs.

use super::*;



/// `kernels::rope::qk_rmsnorm_rope_bf16_devwin`.
/// `outputs[0]` = q, `outputs[1]` = k; shapes are read from inputs.
#[must_use]
pub fn qk_rmsnorm_rope_devwin(q: &Val, k: &Val, q_w: &str, k_w: &str) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch(
            "rope::qk_rmsnorm_rope_bf16_devwin",
            vec![q_w.to_string(), k_w.to_string()],
            None,
            vec![q.id, k.id],
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


builder! {
    /// `kernels::attn::pad_head_dim_bf16` / `kernels::attn::strip_head_dim_bf16`:
    pub fn pad_head_dim(x: &Val, heads: u32, head_dim_padded: u32) -> Val {
        symbol: "attn::pad_head_dim_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim_padded)] as BF16,
        made: "the pad produces its value",
    }


    /// The inverse of [`pad_head_dim`](crate::cuda::pad_head_dim).
    pub fn strip_head_dim(x: &Val, heads: u32, head_dim: u32) -> Val {
        symbol: "attn::strip_head_dim_bf16",
        on: x,
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



    /// `kernels::gemm::grouped_act_x_wt_bf16`: one GEMM per group, batched.
    pub fn gemm_grouped(act: &Val, w: &str, n: u32) -> Val {
        symbol: "gemm::grouped_act_x_wt_bf16",
        on: act,
        weights: [w],
        inputs: [act],
        out: [Dim::Tokens, Dim::Const(n)] as BF16,
        made: "the gemm produces its value",
    }
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

builder! {
    /// `kernels::gemm::mla_absorb_q_to_latent_bf16`.
    pub fn mla_absorb_q_to_latent(
        q_nope: &Val,
        w: &str,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        qk_nope_dim: u32,
    ) -> Val {
        symbol: "gemm::mla_absorb_q_to_latent_bf16",
        on: q_nope,
        weights: [w],
        // params[0] = heads, [1] = qk_nope_dim, [2] = v_head_dim, [3] = kv_lora_rank.
        params: [heads, qk_nope_dim, v_head_dim, kv_lora_rank],
        inputs: [q_nope],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(kv_lora_rank)] as BF16,
        made: "the absorb produces its value",
    }

    /// `kernels::gemm::mla_absorb_latent_to_v_bf16`.
    pub fn mla_absorb_latent_to_v(
        latent: &Val,
        w: &str,
        heads: u32,
        v_head_dim: u32,
        qk_nope_dim: u32,
        kv_lora_rank: u32,
    ) -> Val {
        symbol: "gemm::mla_absorb_latent_to_v_bf16",
        on: latent,
        weights: [w],
        // params[0] = heads, [1] = qk_nope_dim, [2] = v_head_dim, [3] = kv_lora_rank.
        params: [heads, qk_nope_dim, v_head_dim, kv_lora_rank],
        inputs: [latent],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(v_head_dim)] as BF16,
        made: "the absorb produces its value",
    }
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


    /// `kernels::layout::split_qwen_gdn_ba_bf16`.
    pub fn split_qwen_gdn_ba(ba: &Val, v_h: u32) -> (Val, Val) {
        symbol: "layout::split_qwen_gdn_ba_bf16",
        on: ba,
        inputs: [ba],
        outs: [
            [Dim::Tokens, Dim::Const(v_h)] as BF16,
            [Dim::Tokens, Dim::Const(v_h)] as BF16,
        ],
        made: "the split states two outputs",
    }
}
