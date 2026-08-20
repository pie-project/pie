//! What a trace that states one of `attn`'s symbols binds to.
//!
//! The `head_dim` here is `keys::KvHeadDim`, the width the cache was allocated
//! at, and not `keys::HeadDim`, however much the two agree.
//!
//! # The null rule
//!
//! Picking wrong between a key that null-checks and one that binds the pointer
//! as it stands is silent both ways. The question is what the row does with
//! the null, not whether the query is available.
//!
//! * Bind the null when another fact on the same row announces the absence:
//!   `keys::KvHasEnvelopes` IS `!k_env_min.is_null() && !k_env_max.is_null()`,
//!   so refusing would refuse exactly the rows it reports `false` for.
//! * Refuse when nothing announces it -- `arms/fa2.rs`'s `lse_slab`.
//! * Do neither when one row wants each: `keys::KvWritePage` refuses on null,
//!   which `write_kv_explicit` wants and `qkv_decode_fused` does not, its
//!   kernel branching on the null itself.
//!
//! Three residues have no expression in a column: `no_join_extras` (a `Source`
//! binds a slot, it cannot assert one is empty), `dequant_prelude` (a second
//! launch) and `lse_slab`.

use super::Bound;


/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound::derived("attn::split_qkv_bf16"),

    Bound::derived("attn::lse_log2_to_ln"),
    Bound::derived("attn::attention_sink_rescale_bf16"),
    Bound::derived("attn::attn_res_blend_bf16"),
    Bound::derived("attn::pad_head_dim_bf16"),
    Bound::derived("attn::strip_head_dim_bf16"),
    Bound::derived("attn::logit_softcap_bf16"),
    Bound::derived("attn::kimi_split_q_b_bf16"),
    Bound::derived("attn::kimi_split_kv_a_norm_bf16"),
    // The launcher never reads the mask width; derived regions carry exactly
    // what the arm did.
    Bound::derived("attn::dsa_index_topk_mask"),
    // `total`, not `rows.count`: the `_devwin` forms exist for fires where the
    // two differ.
    Bound::derived("attn::split_qkv_bf16_devwin"),
    // Those params are the statement's merge shape, not the global head facts.
    Bound::derived("attn::combine_attn_outputs_bf16"),
    // The `head_dim` division and its `Refusal::Empty` guard live in the
    // launcher; `attn/mod.rs`' closing pin asserts the slots.
    Bound::derived("attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16"),
    // The write is split in two, one argument list each, and the boot's KV
    // storage choice picks between them; the name model texts state,
    // `attn::write_kv_to_pages`, is resolved once at load. Envelope planes bind
    // as they stand, null included -- see the null rule above.
    Bound::derived("attn::write_kv_to_pages_bf16"),
    // The declaration itself, so the registry mentions the name model texts
    // state. It carries no arm because nothing ever fires this spelling:
    // `Boot::route` resolves it to one of the two rows above at load, and
    // `dispatch` binds the ROUTED row's own symbol. Listing it is what tells a
    // reader the absence is routing rather than an omission.
    Bound {
        symbol: "attn::write_kv_to_pages",
        arm: None,
        unbound: Some(
            "nothing: it is an `untraced!` declaration the boot's KV storage choice \
             resolves to `_bf16` or `_quantised` before any fire names it",
        ),
    },
    // `keys::KvSchemeByte`/`KvStorageDtype` arrive as `i32` and the launcher
    // rebuilds the byte: `KvScheme` is a backend type, not a fact.
    Bound::derived("attn::write_kv_to_pages_quantised"),
    // `keys::KvWritePage`/`KvWriteOffset` null-check, which is why they fit
    // here and not on `qkv_decode_qk_norm_rope_write_kv_bf16`.
    Bound::derived("attn::write_kv_explicit_bf16"),
    // No host struct implements `Arg`, so this takes `&KvLayer`'s ten leaves.
    Bound::derived("attn::attention_naive_paged"),
    // `_or_null` write coordinates: the kernel branches on the null, so the
    // refusing spelling would decline a valid fire.
    Bound::derived("attn::qkv_decode_qk_norm_rope_write_kv_bf16"),
    // `keys::KvPagesInBatch` is the per-batch bound, deliberately not `xqa`'s
    // per-request maximum.
    // NOT `derived`: the routine takes nothing from the statement, so there
    // is no column. See the arm.
    Bound {
        symbol: "attn::dequant_kv_cache_layer_to_bf16_active",
        arm: Some(super::fa2::dequant_kv_cache_layer_arm),
        unbound: None,
    },
    Bound {
        symbol: "attn::dsa_index_q_rope_bf16",
        arm: None,
        unbound: Some("the indexer's `rope_dim`, which appears in no statement, shape or context"),
    },
    Bound {
        symbol: "attn::dsa_index_knorm_rope_bf16",
        arm: None,
        unbound: Some("the same `rope_dim`, and a LayerNorm weight and bias no statement names"),
    },
    Bound {
        symbol: "attn::attn_score_fold_heads",
        arm: None,
        unbound: Some("`score_indptr_d`, the score-capture CSR, which no `Cx` query reaches"),
    },
    Bound {
        symbol: "attn::compact_page_csr",
        arm: None,
        unbound: Some(
            "which of the three CSR results `arg_out(0)` is, plus `scratch_counts` and the stride",
        ),
    },
    Bound {
        symbol: "attn::mtp_shift_hidden_bf16",
        arm: None,
        unbound: Some(
            "`slot_ids`, reachable only through `Cx::gdn()`, which wants a recurrent shape",
        ),
    },
    Bound {
        symbol: "attn::mtp_update_pending_hidden_bf16",
        arm: None,
        unbound: Some(
            "its twin's `slot_ids`, and `pending_hidden`, a slab kind `Slab` has no variant for",
        ),
    },
    Bound {
        symbol: "attn::mla_prepare_bf16",
        arm: None,
        unbound: Some(
            "the MLA layer view `Cx::mla_layer` refuses, which five of these operands come out of",
        ),
    },
    Bound {
        symbol: "attn::write_mla_to_pages",
        arm: None,
        unbound: Some("the same MLA layer view, which five of these thirteen operands are"),
    },
    Bound {
        symbol: "attn::dsv4_boundary_meta_decode",
        arm: None,
        unbound: Some(
            "the compression ratio the kernel divides by, which no statement or query carries",
        ),
    },
    Bound {
        symbol: "attn::dsv4_boundary_meta_paged",
        arm: None,
        unbound: Some("its twin's compression ratio, and nothing else"),
    },
    Bound {
        symbol: "attn::attention_compressed_paged_bf16",
        arm: None,
        unbound: Some(
            "the same ratio, plus `comp_kv_pages` and `req_of_token`, which nothing here builds",
        ),
    },
    Bound {
        symbol: "attn::dsv4_compress_gather_paged_bf16",
        arm: None,
        unbound: Some(
            "the compression state: `state_kv`, `state_score`, `ape`, `ratio`, `coff`, none stated",
        ),
    },
    Bound {
        symbol: "attn::dsv4_store_comp_entries_bf16",
        arm: None,
        unbound: Some("`page_size`, and the compression ratio its gather half is blocked on"),
    },
    Bound {
        symbol: "attn::dispatch_attention_mla_bf16",
        arm: None,
        unbound: Some("a latent cache to attend over, which this driver builds none of"),
    },
];
