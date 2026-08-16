//! LATENT ATTENTION — the kimi splits, DSA's lightning indexer, and
//! MLA proper.

use super::*;

// ── MLA: the kimi splits ───────────────────────────────────────
//
// The unfused counterpart of [`mla_prepare`]. Kimi projects `q_a` and
// `kv_a` in one GEMM and splits afterwards; both statements are row-shaped
// (`tokens` is their only extent), so unlike the fused prepare they are
// NOT `whole` — which is the whole reason a deployment might bind them
// instead.

/// `kernels::attn::kimi_split_kv_a_norm_bf16`: split the latent KV
/// projection into `(kv_c, k_pe)`, norming the compressed half on the way.
///
/// The norm is folded in, so this is one statement where the semantic
/// reading is two (`rmsnorm` then a split).
pub fn kimi_split_kv_a_norm(
    kv_a: &Val,
    norm_weight: &str,
    kv_lora_rank: u32,
    qk_rope_dim: u32,
) -> (Val, Val) {
    let outs = record_many(
        &kv_a.t,
        kv_a.layer,
        "attn::kimi_split_kv_a_norm_bf16",
        vec![norm_weight.to_string()],
        vec![kv_a.id],
        vec![
            (
                Shape(vec![Dim::Tokens, Dim::Const(kv_lora_rank)]),
                DType::BF16,
            ),
            (
                Shape(vec![Dim::Tokens, Dim::Const(qk_rope_dim)]),
                DType::BF16,
            ),
        ],
    );
    let mut it = outs.into_iter();
    let kv_c = it.next().expect("the split states two outputs");
    let k_pe = it.next().expect("the split states two outputs");
    (kv_c, k_pe)
}

/// `kernels::attn::kimi_split_q_b_bf16`: split the query projection into
/// its nope and rope halves.
pub fn kimi_split_q_b(q_b: &Val, heads: u32, qk_nope_dim: u32, qk_rope_dim: u32) -> (Val, Val) {
    let outs = record_many_with_params(
        &q_b.t,
        q_b.layer,
        "attn::kimi_split_q_b_bf16",
        vec![],
        vec![heads, qk_nope_dim, qk_rope_dim],
        vec![q_b.id],
        vec![
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
    let q_nope = it.next().expect("the split states two outputs");
    let q_pe = it.next().expect("the split states two outputs");
    (q_nope, q_pe)
}


// ── DSA: the lightning indexer ─────────────────────────────────
//
// glm5 attends SPARSELY: a small side network scores every (query, key)
// pair, and only the top-k keys per query are attended. The two rope
// statements prepare that indexer's own q and k; the third scores and
// thresholds, and its output is the mask MLA's `index_mask` reads.
//
// The mask is the one statement here that is `whole`, and the reason is
// the algebra rather than the addressing: query `i` scores keys `0..=i`,
// so a row window that starts anywhere but zero cannot see the keys it
// must rank against.

/// `kernels::attn::dsa_index_q_rope_bf16`: interleaved rope on each
/// index head of the indexer's queries.
pub fn dsa_index_q_rope(idx_q: &Val, heads: u32, head_dim: u32) -> Val {
    record(
        &idx_q.t,
        idx_q.layer,
        "attn::dsa_index_q_rope_bf16",
        vec![],
        None,
        vec![idx_q.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(heads), Dim::Const(head_dim)]),
            DType::BF16,
        )),
    )
    .expect("the rope produces its value")
}

/// `kernels::attn::dsa_index_knorm_rope_bf16`: layernorm then rope on the
/// indexer's keys.
///
/// A LayerNorm with a bias, not the RMS norm the rest of the model uses —
/// which is why it is its own statement rather than `rmsnorm` followed by
/// `rope`.
pub fn dsa_index_knorm_rope(idx_k: &Val, head_dim: u32) -> Val {
    record(
        &idx_k.t,
        idx_k.layer,
        "attn::dsa_index_knorm_rope_bf16",
        vec![],
        None,
        vec![idx_k.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::BF16)),
    )
    .expect("the norm+rope produces its value")
}

/// `kernels::attn::dsa_index_topk_mask`: score every causal (query, key)
/// pair and keep the top-k per query.
///
/// `logit[i,j] = Σ_h relu(idx_q[i,h,·] · idx_k[j,·]) · idx_w[i,h]`, then
/// the mask is 1 for the top-`k` of `j <= i`. The output is `[T, T]`, and
/// it is what MLA's `index_mask` consumes.
/// `top_k` rides the PARAM channel: it is a load-time number and no
/// operand shape carries it — the same reading `moe_align` and the
/// aligned gather already use.
pub fn dsa_index_topk_mask(
    idx_q: &Val,
    idx_k: &Val,
    idx_w: &Val,
    n_heads: u32,
    head_dim: u32,
    top_k: u32,
) -> Val {
    record_with_params(
        &idx_q.t,
        idx_q.layer,
        "attn::dsa_index_topk_mask",
        vec![],
        None,
        vec![n_heads, head_dim, top_k],
        vec![idx_q.id, idx_k.id, idx_w.id],
        Some((Shape(vec![Dim::Tokens, Dim::Tokens]), DType::I32)),
    )
    .expect("the indexer produces its mask")
}


// ── MLA: latent attention ──────────────────────────────────────
//
// deepseek_v4, glm5 and kimi_k3 all attend through a LATENT KV: the
// cache stores a `kv_lora_rank`-wide compressed row plus a small
// rope-carrying `qk_rope_head_dim` row, and the heads are reconstructed
// on the way in. It is a different attention algebra, not a different
// head count -- which is why none of the flashinfer statements above can
// stand in for it, and why it gets its own `Prepare::MlaPlan`.

/// `kernels::attn::mla_prepare_bf16`: one launch that turns the two
/// projections into the four operands MLA attends over.
///
/// Returns `(kv_c, k_pe, q_nope, q_pe)` — the compressed KV row, its
/// rope-carrying companion, and the query split the same way. It is one
/// statement rather than four because the kernel is one launch, and the
/// trace records launches.
///
/// `whole`: it addresses through `qo_indptr` / `kv_page_indptr` /
/// `kv_last_page_lens`, which are R-shaped. A row window would leave that
/// arithmetic pointing at the wrong request.
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
    let kv_c = it.next().expect("mla_prepare states four outputs");
    let k_pe = it.next().expect("mla_prepare states four outputs");
    let q_nope = it.next().expect("mla_prepare states four outputs");
    let q_pe = it.next().expect("mla_prepare states four outputs");
    (kv_c, k_pe, q_nope, q_pe)
}

/// `kernels::attn::write_mla_to_pages`: commit the compressed KV row and
/// its rope companion to the paged latent cache.
///
/// The MLA counterpart of `write_kv_to_pages`, and `whole` for the same
/// reason `mla_prepare` is: page addressing is per-request.
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

/// `kernels::attn::dispatch_attention_mla_bf16`: attention over the latent cache.
///
/// `needs` `Prepare::MlaPlan` — its own kind of plan, built from the
/// latent geometry (`kv_lora_rank`, `qk_rope_head_dim`) that no other
/// prepare has a field for, and cached in an `MlaPlanCache` rather than
/// in the shared attention workspace.
///
/// `lacks Scores`: there is no capture variant of this dispatch, so a
/// program whose `attn.out` seam wants the score matrix cannot be served
/// over rows this kernel covers. It publishes an LSE, which is a
/// different thing and not what the capability names.
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
