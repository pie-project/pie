//! Latent-attention CUDA statements.

use super::*;


builder! {
    /// `kernels::attn::kimi_split_kv_a_norm_bf16`.
    pub fn kimi_split_kv_a_norm(
        kv_a: &Val,
        norm_weight: &str,
        kv_lora_rank: u32,
        qk_rope_dim: u32,
    ) -> (Val, Val) {
        symbol: "attn::kimi_split_kv_a_norm_bf16",
        on: kv_a,
        weights: [norm_weight],
        inputs: [kv_a],
        outs: [
            [Dim::Tokens, Dim::Const(kv_lora_rank)] as BF16,
            [Dim::Tokens, Dim::Const(qk_rope_dim)] as BF16,
        ],
        made: "the split states two outputs",
    }


    /// `kernels::attn::kimi_split_q_b_bf16`.
    pub fn kimi_split_q_b(q_b: &Val, heads: u32, qk_nope_dim: u32, qk_rope_dim: u32) -> (Val, Val) {
        symbol: "attn::kimi_split_q_b_bf16",
        on: q_b,
        // params[0] = heads, [1] = qk_nope_dim, [2] = qk_rope_dim.
        params: [heads, qk_nope_dim, qk_rope_dim],
        inputs: [q_b],
        outs: [
            [Dim::Tokens, Dim::Const(heads), Dim::Const(qk_nope_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads), Dim::Const(qk_rope_dim)] as BF16,
        ],
        made: "the split states two outputs",
    }
}


builder! {
    /// `kernels::attn::dsa_index_q_rope_bf16`.
    pub fn dsa_index_q_rope(idx_q: &Val, heads: u32, head_dim: u32) -> Val {
        symbol: "attn::dsa_index_q_rope_bf16",
        on: idx_q,
        inputs: [idx_q],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)] as BF16,
        made: "the rope produces its value",
    }


    /// `kernels::attn::dsa_index_knorm_rope_bf16`.
    pub fn dsa_index_knorm_rope(idx_k: &Val, head_dim: u32) -> Val {
        symbol: "attn::dsa_index_knorm_rope_bf16",
        on: idx_k,
        inputs: [idx_k],
        out: [Dim::Tokens, Dim::Const(head_dim)] as BF16,
        made: "the norm+rope produces its value",
    }



    /// `kernels::attn::dsa_index_topk_mask`.
    pub fn dsa_index_topk_mask(
        idx_q: &Val,
        idx_k: &Val,
        idx_w: &Val,
        n_heads: u32,
        head_dim: u32,
        top_k: u32,
    ) -> Val {
        symbol: "attn::dsa_index_topk_mask",
        on: idx_q,
        // params[0] = n_heads, [1] = head_dim, [2] = top_k.
        params: [n_heads, head_dim, top_k],
        inputs: [idx_q, idx_k, idx_w],
        out: [Dim::Tokens, Dim::Tokens] as I32,
        made: "the indexer produces its mask",
    }
}


builder! {
    /// `kernels::attn::mla_prepare_bf16`.
    pub fn mla_prepare(
        kv_a: &Val,
        q_b: &Val,
        heads: u32,
        kv_lora_rank: u32,
        qk_nope_dim: u32,
        qk_rope_dim: u32,
    ) -> (Val, Val, Val, Val) {
        symbol: "attn::mla_prepare_bf16",
        on: kv_a,
        inputs: [kv_a, q_b],
        outs: [
            [Dim::Tokens, Dim::Const(kv_lora_rank)] as BF16,
            [Dim::Tokens, Dim::Const(qk_rope_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads), Dim::Const(qk_nope_dim)] as BF16,
            [Dim::Tokens, Dim::Const(heads), Dim::Const(qk_rope_dim)] as BF16,
        ],
        made: "mla_prepare states four outputs",
    }
}

/// `kernels::attn::write_mla_to_pages`.
pub fn write_mla_to_pages(kv_c: &Val, k_pe: &Val, l: u32) {
    record(
        &kv_c.t,
        Some(l),
        "attn::write_mla_to_pages",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![kv_c.id, k_pe.id],
        None,
    );
}


builder! {
    /// `kernels::attn::dispatch_attention_mla_bf16`: attention over the latent cache.
    pub fn attention_mla(q_nope: &Val, q_pe: &Val, l: u32, heads: u32, kv_lora_rank: u32) -> Val {
        symbol: "attn::dispatch_attention_mla_bf16",
        on: q_nope,
        layer: Some(l),
        state: Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        inputs: [q_nope, q_pe],
        out: [Dim::Tokens, Dim::Const(heads), Dim::Const(kv_lora_rank)] as BF16,
        made: "the attention produces its value",
    }
}
