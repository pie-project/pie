//! The three hand-stated columns of every row of [`sigs`], pinned against the
//! last commit that stated them.
//!
//! # The columns, and why only these three can drift
//!
//! A derived row reads its columns off the `fn` that runs it: `args` comes
//! from the parameter list, the trace namespace from the module path, the C++
//! spelling from the `Abi` impl. None of those can disagree with the code,
//! because they ARE the code.
//!
//! `whole`, `in_place` and `depth_prefix_plan` are the exception. No signature
//! carries them — whether a statement consumes its whole operand, which
//! operands must be given the same address, whether it joins the union-tail
//! plan swap — so each is typed by hand beside a `routine!` line. They were
//! transcribed there out of the `contract!` and `table/` rows the kernel-x
//! sweep deleted, and `.wiki/kernel-x/refactor-plan.md` §12c's
//! contract-agreement twin was the check that the transcription was faithful.
//! That twin *"die\[d\] with `unit!` in the sweep — by then the strings have
//! been proven once."* The strings had been. **The columns had not**, and
//! `every_instantiation_compiles` does not look at them: it compiles what a
//! body names, and a wrong `whole` compiles perfectly.
//!
//! Four rows had drifted by the time anyone compared, all `false → true`, all
//! in `attn`. `src/not_yet_crossed.rs` had already recovered ITS columns from
//! the same commit rather than defaulting them, *"because a defaulted `whole`
//! or `in_place` is a silently different lowering"*. This is that recovery for
//! the derived half, which is the larger half and the one nothing had checked.
//!
//! # Where the expected data lives, and the two shapes rejected
//!
//! It is [`AT_9E3936FB9`] below: 176 rows, generated once, checked in.
//!
//! **A test that shells out to `git show` at run time was the first
//! candidate, and it is not slow — it is impossible.** `9e3936fb9^` does not
//! compile on this workspace's pinned toolchain: `x/norm.rs:1129`'s `const fn
//! route_rows` calls `<u32 as Ord>::max` and `::min` in a const context, which
//! rustc 1.97.1 refuses with `E0658` ("cannot call conditionally-const
//! method"). So the columns cannot be recovered by building that tree at all,
//! by a test or by anything else. They can only be recovered by lifting the
//! DECLARATIONS out of it into a crate that does compile — which is a
//! procedure, not an expression, and not something a `#[test]` can do
//! in-process. A second reason stands behind that one and would have decided
//! it anyway: a test needing a git object needs a repository, and a vendored
//! or `cargo package`d copy of this crate has neither the commit nor `.git`.
//!
//! **A hand-typed table was the second, and it is the defect this file
//! exists to find.** The four drifted columns are the fourth transcription of
//! one fact; a table typed from a terminal would be the fifth, and nothing
//! would check it against anything.
//!
//! Generated-once-and-checked-in is the honest form, and it is honest only if
//! anyone can regenerate it. The recipe is below, and it was run twice —
//! the second run produced a byte-identical file.
//!
//! ## Regenerating [`AT_9E3936FB9`]
//!
//! ```text
//! git worktree add /tmp/old 9e3936fb9^
//! ```
//!
//! Then build a throwaway binary crate whose only dependency is that
//! commit's own `kernels` (`path = "/tmp/old/crates/kernels"`, which still
//! has `Prepare`, `publishes_aux`, `lowered_as` and `returns`), holding:
//!
//! * `Contract`, `Contract::DEFAULT`, `Contract::sig` and `SIG_BASE`, copied
//!   from `/tmp/old/crates/kernels-cuda/src/x/contract.rs`, and the
//!   `contract!` macro from that tree's `src/x/macros.rs` — both with
//!   `$crate::x::` rewritten to `$crate::` and nothing else changed;
//! * one module per family holding the `contract! { .. }` block lifted
//!   byte-for-byte out of `/tmp/old/.../src/x/<family>.rs`, for the twelve
//!   families `x::SIGS` listed: `rope layout sample adapter quant mlp norm
//!   ssm moe gemm attn xqa`;
//! * `/tmp/old/.../src/table/attn.rs` unedited but for `//!` → `//`, which is
//!   the whole of that commit's `table::ROW_TABLES`.
//!
//! `main` walks `table_attn::KERNELS` and then each family's generated
//! `SIGS`, printing `symbol`, `whole`, `depth_prefix_plan` and `in_place`,
//! and sorts. Those two lists ARE `table::KERNELS` at that commit — its
//! `concat_lists` is `ROW_TABLES` followed by `x::SIGS` — so the dump is the
//! complete declared set and nothing is filtered.
//!
//! **The compiler is the oracle for the extraction itself**, which is the
//! property that makes a lift trustworthy where a regex would not be. Every
//! column below is what `Contract::sig()` and `kernel!` COMPUTED from text
//! taken verbatim out of that commit; a botched lift is a build failure
//! naming the line, not a silently wrong row.
//!
//! It printed **176 rows, 176 distinct**, which is the number
//! `refactor-plan-followup.md` §9 records for that commit.
//!
//! # The one row that deliberately disagrees
//!
//! [`DIVERGED`]. Three of the four drifted columns have been put back;
//! `attn::dsv4_boundary_meta_paged` keeps `whole = true` on purpose, and the
//! argument is in that constant. A deliberate change and a transcription slip
//! look identical in a diff, so the difference is written down rather than
//! left for the next reader to re-derive — which is the same reason the old
//! table wrote a sentence beside `dsa_index_topk_mask` and none beside its
//! two neighbours.
//!
//! # What makes this file deletable, and it is NOT met
//!
//! The condition is that no `routine!` line states a column by hand any
//! more — at which point there is no transcription left to check and this
//! pin is a second copy of `sigs()` under another name.
//!
//! `refactor-plan-followup.md` §2 expects §6.3 to bring that about. **It does
//! not, and the honest statement is that it cannot.** §6.3 emptied
//! `not_yet_crossed.rs`, which is the STATED half; these three columns are
//! hand-typed on the DERIVED half and stay hand-typed however many families
//! cross. Emptying that file MOVED the transcriptions, it did not remove
//! them: a `driver_bound!(all_reduce_bf16, whole)` types `whole` by hand
//! exactly as `kernel!(all_reduce_p2p "comm::all_reduce_bf16", whole = true)`
//! did.
//!
//! This paragraph used to end with a prediction, and the prediction was
//! wrong: *"A per-row classification of `not_yet_crossed`'s twenty-one
//! symbols also says that file will not empty: seven are blocked on how a
//! trace states a KV-layer view, which is an open design question and not a
//! port; twelve need a driver resource or name no `__global__` at all; one is
//! deliberately excluded and one is a single symbol with two arms."* **The
//! file is deleted.** Every one of those classifications was true and none of
//! them was a reason to state a row: `kernels::driver_bound!` derives a
//! declaration from the `fn` that runs it whether or not a STATEMENT can bind
//! it, which is a different question from the one each of those twenty-one
//! sentences was answering. The prediction is kept because being wrong about
//! it is the finding.
//!
//! So this file goes when `whole`, `in_place` and `depth_prefix_plan` become
//! derivable — from the device text, or from a type on the `fn` — and not
//! before.
//!
//! # What this pin can never cover, which is worth knowing before trusting it
//!
//! 171 of the 200 rows `sigs()` now holds — 176 pinned less the five in
//! [`DEPARTED`]. The rest arrived after
//! `9e3936fb9^` and have no older statement to be checked against; neither
//! will the next one. **The coverage shrinks with every new routine and
//! cannot be made to grow**, which is exactly why this is a one-time
//! reconciliation rather than a permanent invariant, and why the paragraph
//! above is about deleting it rather than extending it.
//!
//! Four of the 28 arrived by DECLARATION rather than by anyone writing a
//! kernel: `attn::split_qkv_bf16`, `layout::split_q_gate_bf16`,
//! `mlp::sigmoid_gate_inplace_bf16` and `ssm::qwen_gdn_post_conv_prep_bf16`
//! are `driver_bound!` lines over `kernels_cuda::driver_internal`'s
//! `fn`s, and live model texts had been lowering to all four while nothing
//! declared them. They are the sharpest case of what "arrived after" does not
//! mean: every one of the four is older than `9e3936fb9^` as CODE, and new
//! only as a row.

use std::collections::HashMap;

use kernels_cuda::sigs;

/// One row's three stated columns, as `9e3936fb9^` stated them.
///
/// Only the columns a live reader consumes, which is the same set both halves
/// of [`sigs`] fill: `model-ir/src/kernels.rs:108` refuses a model at
/// LOAD if a `whole` kernel is stated inside a Peel region, `lower.rs:1086`
/// raises `Uncovered::WholeKernelSplit` if an arm emits one over a row
/// window, `depth_prefix_plan` drives the union-tail plan swap and `in_place`
/// is buffer aliasing. `args` is deliberately absent: a `contract!` row could
/// not carry one, so there is nothing at that commit to compare against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stated {
    symbol: &'static str,
    whole: bool,
    depth_prefix_plan: bool,
    in_place: &'static [(u32, u32)],
}

/// A row that stated nothing but its symbol.
///
/// The builders below are what a row states ON TOP of that, so the table
/// reads the way the `contract!` blocks it came from read: a declaration says
/// only what is unusual about it, and a reader's eye goes to exactly that.
/// Spelling all three columns on all 176 rows would be 528 words of `false`
/// and `&[]` with the 41 that matter buried in them.
const fn stated(symbol: &'static str) -> Stated {
    Stated { symbol, whole: false, depth_prefix_plan: false, in_place: &[] }
}

impl Stated {
    /// This statement consumes its whole operand, not a row range.
    const fn whole(mut self) -> Self {
        self.whole = true;
        self
    }

    /// This statement participates in the depth-prefix plan.
    const fn depth_prefix_plan(mut self) -> Self {
        self.depth_prefix_plan = true;
        self
    }

    /// `(input, output)` pairs that must be given the same address.
    const fn in_place(mut self, pairs: &'static [(u32, u32)]) -> Self {
        self.in_place = pairs;
        self
    }
}

/// Every symbol `9e3936fb9^` declared, with the three columns it stated.
///
/// Generated — see this module's header for the commit, the procedure and
/// the two shapes rejected. Sorted by symbol, which is the generator's own
/// order and what [`the_pinned_table_is_the_one_that_was_generated`] checks.
const AT_9E3936FB9: &[Stated] = &[
    stated("attn::attention_compressed_paged_bf16").whole(),
    stated("attn::attention_flashinfer_prefill").whole(),
    stated("attn::attention_naive_paged").whole(),
    stated("attn::attention_sink_rescale_bf16").in_place(&[(0, 0)]),
    stated("attn::attention_xqa_decode_bf16_prepared").whole(),
    stated("attn::attn_res_blend_bf16"),
    stated("attn::attn_score_fold_heads").whole(),
    stated("attn::combine_attn_outputs_bf16"),
    stated("attn::compact_page_csr").whole(),
    stated("attn::dequant_kv_cache_layer_to_bf16_active"),
    stated("attn::dispatch_attention_flashinfer_decode").depth_prefix_plan(),
    stated("attn::dispatch_attention_flashinfer_decode_capture"),
    stated("attn::dispatch_attention_flashinfer_prefill_bf16"),
    stated("attn::dispatch_attention_flashinfer_prefill_capture_bf16"),
    stated("attn::dispatch_attention_flashinfer_prefill_custom"),
    stated("attn::dispatch_attention_mla_bf16"),
    stated("attn::dsa_index_knorm_rope_bf16"),
    stated("attn::dsa_index_q_rope_bf16"),
    stated("attn::dsa_index_topk_mask").whole(),
    stated("attn::dsv4_boundary_meta_decode"),
    stated("attn::dsv4_boundary_meta_paged"),
    stated("attn::dsv4_compress_gather_paged_bf16"),
    stated("attn::dsv4_store_comp_entries_bf16").whole(),
    stated("attn::kimi_split_kv_a_norm_bf16"),
    stated("attn::kimi_split_q_b_bf16"),
    stated("attn::logit_softcap_bf16").in_place(&[(0, 0)]),
    stated("attn::lse_log2_to_ln").in_place(&[(0, 0)]),
    stated("attn::mla_prepare_bf16").whole(),
    stated("attn::mtp_shift_hidden_bf16").whole(),
    stated("attn::mtp_update_pending_hidden_bf16").whole(),
    stated("attn::pad_head_dim_bf16"),
    stated("attn::qkv_decode_qk_norm_rope_write_kv_bf16"),
    stated("attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16"),
    stated("attn::split_qkv_bf16_devwin"),
    stated("attn::strip_head_dim_bf16"),
    stated("attn::write_kv_explicit_bf16"),
    stated("attn::write_kv_explicit_bf16_devwin").whole(),
    stated("attn::write_kv_to_pages"),
    stated("attn::write_mla_to_pages").whole(),
    stated("comm::all_reduce_bf16").whole(),
    stated("comm::all_reduce_residual_rmsnorm_bf16").whole().in_place(&[(0, 1)]),
    stated("dist::all_gather_bf16").whole(),
    stated("dist::all_reduce_bf16").whole().in_place(&[(0, 0)]),
    stated("dist::all_reduce_bf16_out").whole(),
    stated("gemm::act_x_wt_bf16"),
    stated("gemm::act_x_wt_bf16_out_fp32"),
    stated("gemm::act_x_wt_bias_bf16"),
    stated("gemm::act_x_wt_channel_scaled"),
    stated("gemm::act_x_wt_grouped_scaled"),
    stated("gemm::act_x_wt_mxfp4_marlin"),
    stated("gemm::grouped_act_x_wt_bf16").whole(),
    stated("gemm::mla_absorb_latent_to_v_bf16"),
    stated("gemm::mla_absorb_q_to_latent_bf16"),
    stated("layout::embed_bf16"),
    stated("layout::gather_bf16_rows"),
    stated("layout::split_bf16_rows"),
    stated("layout::split_qwen_gdn_ba_bf16"),
    stated("layout::transpose_bf16_nld_to_lnd"),
    stated("mlp::chunked_geglu_tanh_bf16"),
    stated("mlp::chunked_situ_bf16"),
    stated("mlp::chunked_swiglu_bf16").in_place(&[(0, 1)]),
    stated("mlp::chunked_swiglu_clamp_bf16"),
    stated("mlp::gaussian_topk_bf16").in_place(&[(0, 0)]),
    stated("mlp::geglu_tanh_bf16").in_place(&[(0, 0)]),
    stated("mlp::gpt_oss_glu_bf16").in_place(&[(0, 0)]),
    stated("mlp::relu2_bf16"),
    stated("mlp::sigmoid_dot_scalar_gate_add_bf16").in_place(&[(0, 1)]),
    stated("mlp::situ_bf16"),
    stated("mlp::swiglu_bf16"),
    stated("mlp::swiglu_clamp_bf16"),
    stated("moe::add_moe_route_bias_bf16").whole(),
    stated("moe::apply_per_expert_scale_bf16").in_place(&[(0, 1)]),
    stated("moe::build_moe_ptrs_aligned_bf16").whole(),
    stated("moe::flashinfer_cutlass_moe_bf16"),
    stated("moe::gather_moe_aligned_inputs_bf16").whole(),
    stated("moe::hash_route_lookup"),
    stated("moe::moe_align_decode").whole(),
    stated("moe::moe_bucket_exact").whole(),
    stated("moe::moe_down_decode_gemv_bf16"),
    stated("moe::moe_gate_up_decode_gemv_bf16"),
    stated("moe::moe_grouped_gemm_bf16").in_place(&[(0, 2)]),
    stated("moe::reorder_moe_aligned_output_bf16").whole(),
    stated("moe::scatter_add_weighted_bf16").whole(),
    stated("moe::token_batched_weighted_sum_add_bf16").in_place(&[(0, 2)]),
    stated("moe::token_batched_weighted_sum_bf16"),
    stated("moe::topk_sigmoid_bf16"),
    stated("moe::topk_sigmoid_bias_fp32"),
    stated("moe::topk_softmax_bf16"),
    stated("moe::topk_sqrtsoftplus_bf16"),
    stated("moe::transpose_expert_scales_u8"),
    stated("norm::add_bias_bf16").in_place(&[(0, 0)]),
    stated("norm::altup_correct_bf16"),
    stated("norm::altup_predict_bf16"),
    stated("norm::altup_unpack_correct_coefs"),
    stated("norm::altup_unpack_predict_coefs"),
    stated("norm::attn_sink_correction_bf16").in_place(&[(0, 0)]),
    stated("norm::compute_rms_bf16"),
    stated("norm::hc_expand_bf16"),
    stated("norm::hc_head_postprocess_bf16"),
    stated("norm::hc_post_bf16"),
    stated("norm::hc_pre_postprocess_bf16"),
    stated("norm::hc_rmsnorm_to_f32"),
    stated("norm::magnitude_rescale_bf16").in_place(&[(0, 0)]),
    stated("norm::mean_streams_bf16"),
    stated("norm::per_head_rmsnorm_bf16").in_place(&[(0, 0)]),
    stated("norm::residual_add_bf16").in_place(&[(0, 0)]),
    stated("norm::residual_add_rmsnorm_bf16"),
    stated("norm::rmsnorm_bf16"),
    stated("norm::rmsnorm_bf16_with_fp16"),
    stated("norm::rmsnorm_gated_bf16"),
    stated("norm::rmsnorm_gated_fp32_in_bf16"),
    stated("norm::rmsnorm_gemma_bf16"),
    stated("norm::rmsnorm_no_scale_bf16").in_place(&[(0, 0)]),
    stated("norm::rmsnorm_residual_add_bf16").in_place(&[(0, 1)]),
    stated("norm::rmsnorm_residual_add_scale_rmsnorm_bf16").in_place(&[(0, 1)]),
    stated("norm::rmsnorm_strided_bf16"),
    stated("norm::scalar_mul_bf16").in_place(&[(0, 0)]),
    stated("norm::tanh_bf16").in_place(&[(0, 0)]),
    stated("pie_lora_qkv_correction"),
    stated("quant::bf16_to_fp16"),
    stated("quant::cast_fp32_to_bf16"),
    stated("quant::dequant_fp8_e4m3_to_bf16"),
    stated("quant::dequant_fp8_e4m3_to_bf16_per_channel"),
    stated("quant::dequant_fp8_e4m3_to_bf16_per_group"),
    stated("quant::dequant_mxfp4_to_bf16"),
    stated("quant::dequant_wna16_int4b8_to_bf16"),
    stated("quant::mxfp4_moe_down_decode_bf16"),
    stated("quant::mxfp4_moe_gate_up_decode_bf16"),
    stated("quant::mxfp4_scales_to_marlin_e8m0"),
    stated("quant::quantize_bf16_to_fp8_e4m3_per_channel"),
    stated("quant::quantize_bf16_to_mxfp4_e2m1_per_block"),
    stated("quant::scale_rows_bf16"),
    stated("quant::wna16_down_decode_bf16"),
    stated("quant::wna16_gate_up_decode_bf16"),
    stated("qwen35_verify_stash_load"),
    stated("qwen35_verify_stash_store"),
    stated("rope::qk_rmsnorm_mrope_bf16"),
    stated("rope::qk_rmsnorm_rope_bf16").in_place(&[(0, 0),  (1, 1)]),
    stated("rope::qk_rmsnorm_rope_bf16_devwin").whole().in_place(&[(0, 0),  (1, 1)]),
    stated("rope::qk_rmsnorm_rope_bf16_rounded").in_place(&[(0, 0),  (1, 1)]),
    stated("rope::rope_bf16").in_place(&[(0, 0),  (1, 1)]),
    stated("rope::rope_partial_bf16").in_place(&[(0, 0),  (1, 1)]),
    stated("rope::rope_partial_bf16_position_delta"),
    stated("rope::rope_partial_last_bf16"),
    stated("rope::rope_standard_table"),
    stated("rope::rope_write_kv_bf16").whole(),
    stated("rope::rope_yarn_bf16"),
    stated("rope::rope_yarn_original_bf16").in_place(&[(0, 0),  (1, 1)]),
    stated("sample::lm_head_gemv_argmax_int8"),
    stated("ssm::bf16_to_fp32"),
    stated("ssm::build_nemotron_moe_ptrs_aligned_bf16").whole(),
    stated("ssm::build_nemotron_moe_ptrs_decode_batched_bf16").whole(),
    stated("ssm::causal_conv1d_prefill_batched_bf16"),
    stated("ssm::causal_conv1d_update_batched_bf16"),
    stated("ssm::chunk_gated_delta_prefill_batched"),
    stated("ssm::chunk_gated_delta_prefill_batched_cached"),
    stated("ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"),
    stated("ssm::chunk_gated_delta_prefill_batched_state_bf16"),
    stated("ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"),
    stated("ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"),
    stated("ssm::fp32_to_bf16"),
    stated("ssm::kda_gate_beta_bf16"),
    stated("ssm::kda_o_norm_gated_bf16"),
    stated("ssm::kda_prefill_batched").whole(),
    stated("ssm::kda_recurrent_step_batched").whole(),
    stated("ssm::l2norm_scale_bf16_to_fp32"),
    stated("ssm::nemotron_mamba_split_bf16"),
    stated("ssm::nemotron_mamba_ssm_batched_bf16").whole(),
    stated("ssm::nemotron_prepare_mamba_dt_da"),
    stated("ssm::nemotron_prepare_mamba_params"),
    stated("ssm::recurrent_gated_delta_step_batched"),
    stated("ssm::recurrent_gated_delta_step_batched_gqa"),
    stated("ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"),
    stated("ssm::recurrent_gated_delta_step_batched_state_bf16"),
    stated("ssm::repeat_interleave_heads_fp32"),
    stated("ssm::zamba_rmsnorm_gated_bf16"),
];

/// A symbol that left `sigs()` since `9e3936fb9^`, and why it was allowed to.
///
/// Pinned rather than tolerated. A symbol leaving the declared set is not a
/// cosmetic change: `model-compiler`'s `check_plan` coverage rule refuses a
/// model text at LOAD for naming a symbol nothing declares, so a row that
/// disappears without anyone deciding it should is a deployment that stops
/// loading. Two have left, both deliberately, and a third would fail the test
/// below rather than pass unnoticed.
struct Departed {
    symbol: &'static str,
    /// The decision, in one sentence, with the address that holds it.
    why: &'static str,
}

/// These, and nothing else, may be missing from [`sigs`].
const DEPARTED: &[Departed] = &[
    Departed {
        symbol: "moe::scatter_add_weighted_bf16",
        why: "`refactor-plan.md` §6.4's deliberate deletion of a confirmed \
              orphan: `WeightedSum` lowers to `moe::token_batched_weighted_\
              sum_bf16`, the DSL builder had zero callers, and the device \
              kernel, the row, the host fn, the contract, the arm and the \
              builder went in one commit",
    },
    Departed {
        symbol: "rope::rope_partial_bf16_position_delta",
        why: "unreachable from either end -- its arm is `unbound` at \
              `driver-cuda/src/bind/arms/rope.rs:261` (\"the offset added to \
              every position ... no statement carries\") and no `dsl::cuda` \
              builder records it, which is why `model/tests/kernels_table.rs` \
              lists it in `UNSTATED_ROWS`. See that list for the second, \
              currently-masked failure this departure causes there",
    },
    // THE NEXT TWO ARE A RENAME, WHICH IS THE ONE DEPARTURE SHAPE THIS
    // STRUCT CANNOT DISTINGUISH FROM A DELETION. Both left `sigs()`; both
    // are still there under another name, so nothing stopped loading and
    // nothing was orphaned. It is recorded here rather than made invisible
    // because the difference is exactly what a reader six months from now
    // will need, and `Departed` holding it costs one sentence.
    Departed {
        symbol: "qwen35_verify_stash_store",
        why: "RENAMED to `ssm::verify_stash_store`. It carried no namespace \
              and named a MODEL, which breaks the crate's one derivation -- \
              `module_path!()` is the trace namespace -- twice: a bare symbol \
              has no family to derive from, and a symbol naming a deployment \
              grows the table once per model. It is the in-proj triple of a \
              LINEAR layer, declared four lines from \
              `ssm::repeat_interleave_heads_fp32`; the DSL's own builders were \
              called `verify_stash_store` all along and only the wire name \
              lagged. Renamed in `model-dsl`, declared in `ssm.rs`",
    },
    Departed {
        symbol: "qwen35_verify_stash_load",
        why: "RENAMED to `ssm::verify_stash_load`, with its store; the load's \
              contract is only meaningful against the store's layout, which is \
              why the DSL declares the pair together and why they move together",
    },
    Departed {
        symbol: "pie_lora_qkv_correction",
        why: "RENAMED to `gemm::lora_qkv_correction`, the third and last bare \
              symbol, and the one whose old name misdescribed it as well as \
              breaking the derivation. `dsl::cuda` called it a PSEUDO-SYMBOL \
              because it names no `__global__`; so does `gemm::mla_absorb_q_to_\
              latent_bf16`, which is two `cublasGemmStridedBatchedEx` and an \
              ordinary derived routine -- a cuBLAS host program is a host \
              program. The `pie_` was the prefix of a C ABI this tree no longer \
              has, and while it stood no family could offer the string at all \
              (`Family::symbol` is a namespace, `::` and a name). The launch \
              half -- three passes of matmul over a staged lane set -- is \
              `gemm::lora`; the staging stays in `driver-cuda`, which is where \
              the per-fire arena it draws from lives",
    },
];

/// A column this tree states differently from `9e3936fb9^`, on purpose.
///
/// **Not an expected-failure list.** An expected-failure list is a question
/// with the answer deferred; every entry here is a decision that has been
/// made, and the entry exists because a deliberate change and a transcription
/// slip are indistinguishable in a diff. `was`/`is` are both recorded so the
/// entry cannot outlive its subject: if someone puts the old value back, the
/// test below fails asking for this entry to be deleted.
struct Diverged {
    symbol: &'static str,
    /// What `9e3936fb9^` stated.
    was: bool,
    /// What this tree states, and means to.
    is: bool,
    /// The argument. Prose, because the reason is not derivable from either
    /// value.
    why: &'static str,
}

/// The `whole` column, on one row.
///
/// # The four that drifted, and why they are not one case
///
/// All four went `false → true` in the sweep with no comment saying why, and
/// the temptation is to treat them as one mistake and revert all four. Their
/// evidence differs, and reading the host programs is what separates them:
///
/// * `dsa_index_q_rope_bf16` and `dsa_index_knorm_rope_bf16` are
///   `Launch::per_row(tokens, ..)` over the indexer's query rows, with no
///   operand that addresses across a row. A rope over per-row data is per-row
///   by construction. The old table argued this from the other side, in the
///   sentence it wrote beside their neighbour `dsa_index_topk_mask` and not
///   beside them: *"`whole`, and here the reason is the ALGEBRA rather than
///   the addressing: query `i` scores keys `0..=i`, so a row window starting
///   anywhere but zero cannot see the keys it must rank against."* "Here"
///   distinguishes it from these two. Both are back to `false`.
/// * `dsv4_boundary_meta_decode` is `Launch::flat(n, ..)` over `(positions,
///   n, ratio)`, and every row's compressed-block index is `position /
///   ratio` — its own position and nothing else's. Back to `false`.
/// * `dsv4_boundary_meta_paged` is the entry below, and it is the odd one.
///
/// **The direction matters and is asymmetric.** `false → true` cannot corrupt
/// a result — it only ever refuses — but it CAN refuse a model that used to
/// load. `true → false` is the reverse and is the dangerous one. So the three
/// reverts above are the direction that needs the argument, and each has one.
const DIVERGED: &[Diverged] = &[Diverged {
    symbol: "attn::dsv4_boundary_meta_paged",
    was: false,
    is: true,
    why: "It takes `qo_indptr` and `num_requests` and walks them to write \
          `out_req`, which is R-shaped addressing: a row window would leave \
          that search reading a `qo_indptr` describing the whole fire while \
          `n` counts only the window's rows, so it would answer with another \
          request's index. That is exactly the argument the old table made \
          for the two paged MLA statements -- \"`whole` because they address \
          through `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which \
          are R-shaped\" -- and the row it was written on is the one row of \
          the four that has that shape. Its decode twin does not take \
          `qo_indptr` at all, which is the whole difference between them and \
          is why they are `_decode` and `_paged`. So the old row is the one \
          that is wrong here, and this is a correction rather than a slip.",
}];

/// The pinned table is the file that was generated, not one that was edited.
///
/// Three properties a hand edit breaks and the oracle below would not notice:
/// the count, uniqueness, and sort order. The count is `refactor-plan-
/// followup.md` §9's independently recorded number for that commit, so an
/// added or dropped line fails here against a figure from outside this file.
#[test]
fn the_pinned_table_is_the_one_that_was_generated() {
    assert_eq!(
        AT_9E3936FB9.len(),
        176,
        "`9e3936fb9^` declared 176 symbols; this table holds {}, so it is not \
         the generated file",
        AT_9E3936FB9.len()
    );
    let mut symbols: Vec<&str> = AT_9E3936FB9.iter().map(|r| r.symbol).collect();
    let unsorted = symbols.clone();
    symbols.sort_unstable();
    assert_eq!(unsorted, symbols, "the generator sorts; this table is out of order");
    symbols.dedup();
    assert_eq!(symbols.len(), AT_9E3936FB9.len(), "a symbol is pinned twice");
}

/// Every column `9e3936fb9^` stated, [`sigs`] still states — or [`DIVERGED`]
/// says why not.
///
/// The oracle. It is the whole of what this file is for, and everything else
/// here exists to keep it honest.
#[test]
fn every_stated_column_still_says_what_it_said() {
    let now: HashMap<&str, &kernels::KernelSig> =
        sigs().iter().map(|row| (row.symbol, row)).collect();
    let departed: Vec<&str> = DEPARTED.iter().map(|d| d.symbol).collect();
    let diverged: HashMap<&str, &Diverged> = DIVERGED.iter().map(|d| (d.symbol, d)).collect();

    let mut wrong: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for pinned in AT_9E3936FB9 {
        if departed.contains(&pinned.symbol) {
            continue;
        }
        let Some(row) = now.get(pinned.symbol) else {
            wrong.push(format!(
                "{}: declared at 9e3936fb9^ and gone from `sigs()` with no `DEPARTED` entry \
                 -- a model text naming it is refused at LOAD",
                pinned.symbol
            ));
            continue;
        };
        checked += 1;
        // A row in `DIVERGED` has its `whole` checked by
        // `every_recorded_divergence_is_still_a_divergence` and NOT here, so
        // that one fault produces one message. Comparing against the
        // divergence's `is` instead was the first shape and it was wrong in
        // the case that matters: reversing the decision made this test report
        // *"`whole` was false at 9e3936fb9^, is false now"*, which is not a
        // statement about anything, alongside the divergence test's correct
        // one. `whole` is also the only column any row has ever diverged on,
        // so a drift on the other two lands in `wrong` and stays there until
        // someone extends this file -- the right amount of friction for a
        // column nothing has yet needed to change.
        if !diverged.contains_key(pinned.symbol) && row.whole != pinned.whole {
            wrong.push(format!(
                "{}: `whole` was {} at 9e3936fb9^, is {} now",
                pinned.symbol, pinned.whole, row.whole
            ));
        }
        if row.depth_prefix_plan != pinned.depth_prefix_plan {
            wrong.push(format!(
                "{}: `depth_prefix_plan` was {} at 9e3936fb9^, is {} now",
                pinned.symbol, pinned.depth_prefix_plan, row.depth_prefix_plan
            ));
        }
        if row.in_place != pinned.in_place {
            wrong.push(format!(
                "{}: `in_place` was {:?} at 9e3936fb9^, is {:?} now",
                pinned.symbol, pinned.in_place, row.in_place
            ));
        }
    }

    // DERIVED, not typed. It was `174` — the number that happened to be right
    // when this file was written — and the first rename to reach `DEPARTED`
    // made it wrong, which is the same defect in miniature that the whole
    // file exists to catch: a fact transcribed once and then maintained by
    // whoever remembers. `AT_9E3936FB9` is the declared set at that commit
    // and `DEPARTED` is what left it, so the join is their difference and
    // cannot disagree with either.
    let survivors = AT_9E3936FB9.len() - DEPARTED.len();
    assert_eq!(
        checked, survivors,
        "the join should be {survivors} rows ({} pinned less {} departed) and was \
         {checked}; `DEPARTED` and the pinned table disagree about which symbols survive",
        AT_9E3936FB9.len(),
        DEPARTED.len()
    );
    assert!(
        wrong.is_empty(),
        "{} stated column(s) no longer say what the last commit that stated them said. \
         Each is either a transcription that drifted -- put it back -- or a decision, \
         which belongs in `DIVERGED` with the argument beside it:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// A recorded divergence is still a divergence.
///
/// Without this, [`DIVERGED`] is a list that can only grow: an entry whose
/// subject was quietly reverted would keep excusing a row that no longer
/// needs excusing, and the next reader would find an argument for a state the
/// tree is not in.
#[test]
fn every_recorded_divergence_is_still_a_divergence() {
    let now: HashMap<&str, &kernels::KernelSig> =
        sigs().iter().map(|row| (row.symbol, row)).collect();
    for d in DIVERGED {
        assert_ne!(d.was, d.is, "{}: `DIVERGED` records no difference", d.symbol);
        assert!(!d.why.is_empty(), "{}: a divergence with no argument is a bug report", d.symbol);
        let pinned = AT_9E3936FB9
            .iter()
            .find(|r| r.symbol == d.symbol)
            .unwrap_or_else(|| panic!("{}: diverges from a row 9e3936fb9^ never stated", d.symbol));
        assert_eq!(pinned.whole, d.was, "{}: `was` disagrees with the pinned table", d.symbol);
        let row = now
            .get(d.symbol)
            .unwrap_or_else(|| panic!("{}: diverges from `sigs()` and is not in it", d.symbol));
        assert_eq!(
            row.whole, d.is,
            "{}: `DIVERGED` says this row states `whole = {}` on purpose, and it states {}. \
             If the decision was reversed, delete the entry.",
            d.symbol, d.is, row.whole
        );
    }
}

/// Exactly the symbols that were meant to leave have left.
///
/// The other direction of the join, and the one with a deployment behind it:
/// `check_plan` refuses a model text that names a symbol nothing declares, so
/// a row silently leaving `sigs()` is a model that stops loading. Arrivals
/// are not pinned — 22 symbols have arrived since `9e3936fb9^` and a new
/// routine is not a regression — but departures are.
#[test]
fn exactly_the_symbols_that_were_meant_to_leave_have_left() {
    let now: Vec<&str> = sigs().iter().map(|row| row.symbol).collect();
    let mut gone: Vec<&str> = AT_9E3936FB9
        .iter()
        .map(|r| r.symbol)
        .filter(|s| !now.contains(s))
        .collect();
    gone.sort_unstable();
    let mut expected: Vec<&str> = DEPARTED.iter().map(|d| d.symbol).collect();
    expected.sort_unstable();
    assert_eq!(
        gone, expected,
        "the set of symbols that left `sigs()` has changed. One that arrived here needs a \
         `DEPARTED` entry naming the decision; one that left needs its entry deleted."
    );
    for d in DEPARTED {
        assert!(!d.why.is_empty(), "{}: a departure with no reason is a deletion", d.symbol);
    }
}
