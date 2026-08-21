//! Latent-attention CUDA statements.

use super::*;


builder! {
    /// `kernels::attn::kimi_split_kv_a_norm`; the epsilon rides the run.
    pub fn kimi_split_kv_a_norm(
        kv_a: &Val,
        norm_weight: &str,
        kv_lora_rank: u32,
        qk_rope_dim: u32,
        eps: f32,
    ) -> (Val, Val) {
        symbol: "attn::kimi_split_kv_a_norm",
        on: kv_a,
        weights: [norm_weight],
        params: [eps.to_bits()],
        inputs: [kv_a],
        outs: [
            [Dim::Tokens, Dim::Const(kv_lora_rank)] as BF16,
            [Dim::Tokens, Dim::Const(qk_rope_dim)] as BF16,
        ],
        made: "the split states two outputs",
    }


    /// `kernels::attn::kimi_split_q_b`.
    pub fn kimi_split_q_b(q_b: &Val, heads: u32, qk_nope_dim: u32, qk_rope_dim: u32) -> (Val, Val) {
        symbol: "attn::kimi_split_q_b",
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


/// `kernels::attn::dsa_index_q_rope`.
/// `[n_heads, head_dim, rope_dim, theta]` is the run; positions minted by name.
#[must_use]
pub fn dsa_index_q_rope(
    idx_q: &Val,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Val {
    let positions = rt_tokens(&idx_q.t, "positions");
    record_with_params(
        &idx_q.t,
        idx_q.layer,
        "attn::dsa_index_q_rope",
        vec![],
        None,
        vec![heads, head_dim, rope_dim, theta.to_bits()],
        vec![idx_q.id, positions],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the rope produces its value")
}

/// `kernels::attn::dsa_index_knorm_rope`.
///
/// THE NORM IS A LAYERNORM, so it takes a weight AND a bias, and both
/// are operands rather than facts. The kernel
/// (`attn/dsa_indexer.cuh`'s `index_knorm_rope`) subtracts the row mean
/// before scaling and its last statement is
/// `row[d] = x * w[d] + b[d]` -- two banks it dereferences per element,
/// not one bank and a constant. A statement that places neither leaves
/// the arm binding whatever the two weight slots happen to hold.
///
/// They arrive by NAME because the DSA indexer's tensors have no
/// spelling in any manifest: `glm_5/project.rs` says why, and says it
/// about this exact group of weights -- the checkpoint's own names for
/// the indexer are not written down anywhere in this tree, so a
/// manifest row for one would be a guess that turns a matching
/// checkpoint into a `Fault::Missing`. `tests/seam_names.rs` records
/// them as names no builder can yet emit, which is where this pair
/// goes too.
///
/// `[rope_dim, theta, eps]` is the run, in the routine's order; the
/// positions stream is minted by name.
#[must_use]
pub fn dsa_index_knorm_rope(
    idx_k: &Val,
    k_norm_weight: &str,
    k_norm_bias: &str,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
    eps: f32,
) -> Val {
    let positions = rt_tokens(&idx_k.t, "positions");
    record_with_params(
        &idx_k.t,
        idx_k.layer,
        "attn::dsa_index_knorm_rope",
        vec![k_norm_weight.to_string(), k_norm_bias.to_string()],
        None,
        vec![rope_dim, theta.to_bits(), eps.to_bits()],
        vec![idx_k.id, positions],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::BF16)),
    )
    .expect("the norm+rope produces its value")
}

builder! {
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
