//! Rows for the symbols that have NOT crossed to the routine shape.
//!
//! The other half of [`crate::sigs`] is DERIVED: a `Routine`'s columns are
//! read off the `fn` that runs it, so it cannot disagree with the code. These
//! symbols have no such `fn` to read. Their host programs are `driver-cuda`'s
//! — the hand dispatch in `bind/mod.rs`, the arm registries under
//! `bind/arms/`, `bind/service.rs` and `fire/` — or, for a handful, nothing
//! anywhere. They still fire, and `dsl::cuda` still records every one of them,
//! so `model-compiler`'s `check_plan` coverage rule (*every launched symbol
//! must be declared*) refuses a model text at LOAD without a row. Hence
//! stated rather than derived, and hence a file of their own: the distinction
//! is the point.
//!
//! **Every row here is a debt.** One leaves each time
//! `.wiki/kernel-x/refactor-plan.md` §6.3 lands a family; when the list is
//! empty this module is deletable and `sigs()` is the derived half alone. A
//! symbol may not be in both halves at once — [`crate::sigs`]' own test
//! refuses that — because two claims on one symbol is two contracts, and the
//! way that fails is not that one is wrong: each is right for whichever half
//! of the tree its tests exercise, so nothing goes red until a model text
//! picks the other one.
//!
//! **Only the columns with a live reader are stated**, which is the same set
//! the derived half fills, so no consumer can tell the two apart: the symbol
//! it looks up by, `whole` (the Peel refusal and the row-window split),
//! `depth_prefix_plan` (the union-tail plan swap) and `in_place` (buffer
//! aliasing). `args` is the one column a derived row has and a stated row
//! cannot — it comes off a parameter list — and it fills itself when the
//! symbol crosses. Everything else `KernelSig` can carry had no reader when
//! the consumer audit went looking, and stating it here would make these rows
//! claim more than a derived one can.

use kernels::{KernelSig, kernel};

/// The stated half of [`crate::sigs`], with the host program that answers
/// each row named beside it.
pub const NOT_YET_CROSSED: &[KernelSig] = &[
    // ── the FlashInfer dispatch lattice ─────────────────────────────────
    //
    // Six symbols, six hand arms in `bind/mod.rs` (`fa2_decode` through
    // `fa2_prefill_planless`), all of them over
    // `fire::flashinfer_fa2_dispatch`. This is §6.3's largest single debt.
    kernel!(fa2_decode "attn::dispatch_attention_flashinfer_decode",
        depth_prefix_plan = true),
    kernel!(fa2_decode_capture "attn::dispatch_attention_flashinfer_decode_capture"),
    kernel!(fa2_prefill "attn::dispatch_attention_flashinfer_prefill_bf16"),
    kernel!(fa2_prefill_capture "attn::dispatch_attention_flashinfer_prefill_capture_bf16"),
    kernel!(fa2_prefill_custom "attn::dispatch_attention_flashinfer_prefill_custom"),
    // The planless form plans over the whole fire on the way in, so it owes
    // its caller nothing and cannot be handed a row window.
    kernel!(fa2_prefill_planless "attn::attention_flashinfer_prefill", whole = true),
    // The head dims FlashInfer's prefill template rejects (gemma-4's 512)
    // take this naive paged kernel instead: no plan at all, fire-shaped.
    // `bind/arms/attn.rs`'s `attention_naive_paged_arm`.
    kernel!(attention_naive_paged "attn::attention_naive_paged", whole = true),
    // XQA's paged decode. `bind/arms/xqa.rs` DECLINES it: its host program is
    // written (`x::xqa::xqa_decode_bf16` under `fire/xqa.rs`'s workspace
    // carve) and what is missing is the ORDER -- the dense page table
    // `attn::build_xqa_metadata` must have written earlier in the same fire,
    // which is the `Prepare::FireWide` obligation nothing reads.
    kernel!(xqa_decode "attn::attention_xqa_decode_bf16_prepared", whole = true),

    // ── the KV writes, the dequant and the score fold ───────────────────
    //
    // The first three are armed in `bind/arms/attn.rs` over
    // `x::attn::kv_paged`. Those host programs are in this crate and are
    // still not routines: each takes a `&KvLayer`, which has no `Arg` impl
    // because a trace statement cannot supply a KV-cache layer descriptor.
    kernel!(write_kv_to_pages "attn::write_kv_to_pages"),
    kernel!(write_kv_explicit "attn::write_kv_explicit_bf16"),
    kernel!(dequant "attn::dequant_kv_cache_layer_to_bf16_active"),
    // The device-window twin has the host program and no arm on either side.
    kernel!(write_kv_explicit_devwin "attn::write_kv_explicit_bf16_devwin", whole = true),
    // The fold is DECLINED in `bind/arms/attn.rs` for one operand -- the
    // score-capture CSR, which has a producer and no `Cx` query -- and fires
    // out of band from `fire::attn_score`, at the point on the stream where
    // the capture dispatch used to issue it.
    kernel!(attn_score_fold_heads "attn::attn_score_fold_heads", whole = true),

    // ── MLA ─────────────────────────────────────────────────────────────
    //
    // `bind/arms/attn.rs` declines the first three: `Cx::mla_layer` and
    // `Cx::mla_plan` have producers no `Fire` reaches, and `serve/load.rs`
    // refuses an MLA checkpoint at load anyway. The row is what makes that a
    // refusal in a sentence instead of a missing declaration. The prepare and
    // the page write are `x::attn`'s own `fn`s; the dispatch's two arms are
    // `x::attn::mla_fa2` and `driver-cuda/src/fire/mla_naive.rs`.
    //
    // The first two are `whole` for the addressing rather than the algebra:
    // they walk `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which
    // are R-shaped, so a row window leaves that arithmetic pointing at
    // another request. The dispatch is not -- it reads a plan built over the
    // whole fire and still covers a row range, like the FlashInfer ones.
    kernel!(mla_prepare "attn::mla_prepare_bf16", whole = true),
    kernel!(write_mla_to_pages "attn::write_mla_to_pages", whole = true),
    kernel!(attention_mla "attn::dispatch_attention_mla_bf16"),
    // The absorb pair -- cuBLAS calls, which is why a launcher is anything
    // that issues DEVICE work and not anything taking a stream. Armed by
    // `bind/mod.rs`'s `mla_absorb` over `x::attn::mla_absorb_*`. They carry
    // `gemm`'s namespace on `attn`'s host programs, so no `Family` resolves
    // them and a routine alone would not retire these two rows.
    kernel!(mla_absorb_q_to_latent "gemm::mla_absorb_q_to_latent_bf16"),
    kernel!(mla_absorb_latent_to_v "gemm::mla_absorb_latent_to_v_bf16"),

    // ── the quantised GEMMs ─────────────────────────────────────────────
    //
    // `bind/service.rs`'s three `gemm_act_x_wt_*` over `bind/quant_gemm.rs`,
    // which is where the weight-representation routing lives. The dense
    // `gemm::act_x_wt_bf16` crossed and these did not, because the view they
    // build is the driver's vocabulary and not a trace's.
    kernel!(gemm_xwt_channel_scaled "gemm::act_x_wt_channel_scaled"),
    kernel!(gemm_xwt_grouped_scaled "gemm::act_x_wt_grouped_scaled"),
    kernel!(gemm_xwt_mxfp4_marlin "gemm::act_x_wt_mxfp4_marlin"),

    // ── the adapter correction ──────────────────────────────────────────
    //
    // `bind/mod.rs`'s hand arm over `fire::lora`'s `LoraState::apply`. No
    // device text anywhere for it: the LoRA seam is batched cuBLAS, so there
    // is no `__global__` and never was.
    kernel!(lora_qkv_correction "pie_lora_qkv_correction"),

    // ── the collectives ─────────────────────────────────────────────────
    //
    // Every one is `whole`, and for a reason stronger than "a reduction is
    // over the whole value": every rank must enter the same collective the
    // same number of times, so a row window that split one rank's launch and
    // not another's would DEADLOCK rather than compute a wrong answer. The
    // refusal is not an optimisation. They are also synchronisation points,
    // which the graph-capture rules have to know.
    //
    // The `comm::` pair are the P2P arm and have host programs --
    // `bind/service.rs`'s `comm_all_reduce_bf16` and
    // `comm_all_reduce_residual_rmsnorm_bf16` over `fire::all_reduce`. The
    // three `dist::` rows are NCCL and there is no NCCL in this tree, so
    // nothing answers them at all; `serve/load.rs` refuses `tp_size > 1` at
    // model load for exactly that reason, which is the refusal these rows
    // keep visible one layer up.
    kernel!(all_reduce "dist::all_reduce_bf16", whole = true, in_place = &[(0, 0)]),
    // The out-of-place sum: same collective, a separate destination, and no
    // alias pair. That absence is the whole difference from the row above.
    kernel!(all_reduce_out "dist::all_reduce_bf16_out", whole = true),
    kernel!(all_gather "dist::all_gather_bf16", whole = true),
    kernel!(all_reduce_p2p "comm::all_reduce_bf16", whole = true),
    // The fused landing -- sum, add the residual, norm. Two results, so it
    // needs a pair list rather than a single alias: the residual stream is
    // updated in place and the normed activation is the other.
    kernel!(all_reduce_residual_rmsnorm "comm::all_reduce_residual_rmsnorm_bf16",
        whole = true, in_place = &[(0, 1)]),

    // ── the pseudo-symbols, and the leg that retired ────────────────────
    //
    // The verify-stash pair name no `__global__`, carry no family namespace
    // and have no arm on either side of the seam. They are statements the
    // trace makes about a driver-side stash, and the row is all there is to
    // be: nothing to cross, and nothing that would make crossing meaningful.
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
    // The fused CUTLASS MoE leg retired with `fire::flashinfer_moe` and the
    // aligned leg replaced it, but `dsl::cuda` still records this symbol, so
    // a trace can still state it and nothing would run it.
    kernel!(moe_fused_cutlass "moe::flashinfer_cutlass_moe_bf16"),
];
