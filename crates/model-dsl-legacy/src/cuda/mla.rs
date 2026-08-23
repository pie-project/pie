//! Latent-attention CUDA statements.

use super::*;

/// `kernels::attn::mla_prepare_bf16` — untraced (no generated twin).
#[must_use]
pub fn mla_prepare(
    kv_a: &Val,
    q_b: &Val,
    heads: u32,
    kv_lora_rank: u32,
    qk_nope_dim: u32,
    qk_rope_dim: u32,
) -> (Val, Val, Val, Val) {
    let outs = record_many(
        &kv_a.t,
        kv_a.layer,
        "attn::mla_prepare_bf16",
        vec![],
        vec![kv_a.id, q_b.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(kv_lora_rank)]),
                DType::BF16,
            ),
            (
                Shape(vec![Dim::Tokens, Dim::Const(qk_rope_dim)]),
                DType::BF16,
            ),
            (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(qk_nope_dim),
                ]),
                DType::BF16,
            ),
            (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(heads),
                    Dim::Const(qk_rope_dim),
                ]),
                DType::BF16,
            ),
        ],
    );
    let mut it = outs.into_iter();
    let mut next = || it.next().expect("mla_prepare states four outputs");
    let kv_c = next();
    let k_pe = next();
    let q_nope = next();
    let q_pe = next();
    (kv_c, k_pe, q_nope, q_pe)
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

/// `kernels::attn::dispatch_attention_mla_bf16`: attention over the latent
/// cache — no traced routine row (the driver's own dispatch), so no
/// generated twin.
#[must_use]
pub fn attention_mla(q_nope: &Val, q_pe: &Val, l: u32, heads: u32, kv_lora_rank: u32) -> Val {
    record(
        &q_nope.t,
        Some(l),
        "attn::dispatch_attention_mla_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::KvCache,
            layer: l,
        }),
        vec![q_nope.id, q_pe.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(heads),
                Dim::Const(kv_lora_rank),
            ]),
            DType::BF16,
        )),
    )
    .expect("the attention produces its value")
}
