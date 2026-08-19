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
//! them: a `untraced!(all_reduce_bf16, whole)` types `whole` by hand
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
//! them was a reason to state a row: `kernels::untraced!` derives a
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
//! are `untraced!` lines over `kernels_cuda::driver_internal`'s
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

    /// `(output, input)` pairs that must be given the same address.
    ///
    /// OUTPUT FIRST -- see `kernels::routine::Routine::in_place`.
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
/// loading. Every entry below is deliberate, and one more would fail the test
/// below rather than pass unnoticed.
///
/// THREE SHAPES ARE IN HERE and the struct cannot tell them apart, which is
/// why each `why` says which it is: a DELETION (the symbol is gone), a RENAME
/// (it is still there under another name) and a DEMOTION (the `fn` is still
/// there and still fires, and what it stopped being is a `Routine`).
struct Departed {
    symbol: &'static str,
    /// The decision, in one sentence, with the address that holds it.
    ///
    /// **File, not `file:line`, wherever the sentence is also quoted.** One
    /// of these citations rotted three times against the same moving text
    /// -- `arms/rope.rs`, still written `:647` when the line had reached
    /// `:854` -- and every repair restored a number that was going to be
    /// wrong again by the next refactor. The quotation beside it never
    /// rotted once, because it moves WITH the text it names.
    ///
    /// A line number is only worth carrying where there is nothing else to
    /// find the text by. Where a verbatim quote is already present, the
    /// number is a second address for one thing, and the second address is
    /// the one that decays.
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
        why: "unreachable from either end -- its arm is `unbound` in \
              `driver-cuda/src/bind/arms/rope.rs` (\"the offset added to \
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
    // THE LOADER'S TWO, WHICH LEFT AS A PAIR AND NOT AS A DELETION. Both
    // still exist, still fire, and still have their symbol carried in a
    // plan; what they stopped being is a ROUTINE. Nothing that loads a model
    // notices, because `check_plan`'s coverage rule reads the symbols a
    // `dsl::cuda` builder RECORDS, and no builder ever recorded either.
    Departed {
        symbol: "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
        why: "LEFT `ROUTINES`, not the tree. A `Routine` is what a TRACE \
              states and no trace states a load-time weight transform: \
              `model-loader`'s tile plan runs this once over a CHECKPOINT \
              MATRIX, so its `In`/`Out` wrappers promised a statement that \
              did not exist and every caller passed `rows: 0, width: 0` \
              through them. It is a plain `unsafe fn` in `kernels_cuda::\
              quant` now (\"the six that take raw pointers\"), reached by \
              path from `executor/cuda.rs`, and the row it had in \
              `driver-cuda/src/bind/arms/quant.rs` went with it",
    },
    Departed {
        symbol: "quant::quantize_bf16_to_fp8_e4m3_per_channel",
        why: "LEFT `ROUTINES` with the MXFP4 quantiser above, for its reason \
              and in the same commit; the plan names both as strings in \
              `plan/passes/tile.rs`, which is the LOADER's vocabulary and \
              not this crate's registry",
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

/// The `in_place` column, on one row — the twin of [`DIVERGED`], and it
/// exists because a pin was holding the corrupting value.
///
/// # A pin is not evidence that the column was right
///
/// That sentence is the whole reason this constant is here rather than the
/// two pins being quietly edited. `rope::rope_yarn_bf16` and
/// `rope::qk_rmsnorm_mrope_bf16` stated no `in_place` at `9e3936fb9^`, and
/// this file faithfully recorded it — while the launchers had declared
/// `q: Out<0>` and `k: Out<1>` since Stage 2 and the device code
/// (`prelude/rope.cuh`'s `rotate_pair`) reads and assigns the same two
/// cells. **The pinned value was the defect**: with no alias declared, the
/// planner hands the kernel two fresh buffers and `walk.rs:611`'s
/// zero-query bug is manufactured, which is precisely the corruption
/// `qk_rmsnorm_rope_bf16_devwin` was repaired for one commit earlier.
///
/// So the direction argument at [`DIVERGED`] has a counterpart here and it
/// points the other way. There, `false → true` on `whole` can only refuse,
/// so the reverts are what needed the argument. **Adding an alias cannot
/// refuse and cannot corrupt** — it tells the planner two buffers are one,
/// which they demonstrably are — while its absence corrupts silently. The
/// asymmetry is opposite because the column means something different.
///
/// # Why a parallel constant rather than a field on `Diverged`
///
/// `Diverged` carries `was`/`is` as `bool`, and this file says so in as many
/// words: *"a drift on the other two lands in `wrong` and stays there until
/// someone extends this file."* This is that extension, done as the shape
/// that comment implies rather than by generalising `Diverged` over a
/// column type — two rows do not pay for a trait, and the `why` strings
/// are what a reader comes here for.
const DIVERGED_IN_PLACE: &[DivergedInPlace] = &[
    // ── THE FIVE THE `Env` -> `Const` MIGRATION SETTLED ──
    //
    // Two directions and one reason each, and none of them is a slip. See
    // `.wiki/migration.md` §11.5 and §11.6.
    DivergedInPlace {
        symbol: "quant::scale_rows_bf16",
        was: &[],
        is: &[(0, 0)],
        why: "The row declared no alias and the launcher's own comment \
              always did: *`dsl::cuda::scale_rows` hands back the same \
              buffer it took as input 0*. The kernel fires ONE pointer for \
              the data -- `&[buf.ptr, l_bf16, buf.stride]` -- so a fresh \
              result buffer would be scaled from bytes nobody wrote. \
              `InOut<Tensor<T>>` says it at the parameter, and saying it \
              there is also what puts `l_bf16` back at input 1 where its \
              `In<1, T>` used to spell it.",
    },
    DivergedInPlace {
        symbol: "rope::rope_write_kv_bf16",
        was: &[],
        is: &[(0, 0)],
        why: "As `scale_rows`: the row declared none and the numbering \
              required one. `q` was `Out<0>` while `k` and `v` were \
              `In<1>`/`In<2>`, so the statement's input 0 -- `q` itself -- \
              was claimed by no parameter. Under positional derivation that \
              gap closes the wrong way and `k` binds the query's buffer. \
              `InOut` is the mark that claims both slots at one address, \
              which is what the statement does: it places `q` and declares \
              the rotated `q` as its one result.",
    },
    DivergedInPlace {
        symbol: "mlp::geglu_tanh_bf16",
        was: &[(0, 0)],
        is: &[],
        why: "A MAY-ALIAS, and the four marks have no word for one. The \
              pair was an allocator INSTRUCTION -- give result 0 operand \
              0's offset -- and the kernel does not require it: it takes \
              the gate and the destination as two pointers, and \
              `tower::gemma4_vision` calls this launcher directly with \
              three distinct buffers. `InOut` claims ONE ADDRESS, which \
              would be false at that caller, so the hint is dropped rather \
              than mis-stated. The cost is an arena that reuses one buffer \
              less; the alternative is a claim that is not true.",
    },
    DivergedInPlace {
        symbol: "mlp::gpt_oss_glu_bf16",
        was: &[(0, 0)],
        is: &[],
        why: "As `geglu_tanh_bf16`, and for the same caller.",
    },
    DivergedInPlace {
        symbol: "norm::rmsnorm_no_scale_bf16",
        was: &[(0, 0)],
        is: &[],
        why: "As `geglu_tanh_bf16`. `tower::gemma4_vision` normalises the \
              pooled rows out of place -- `pooled` in, `pn` out, two \
              scratch allocations -- so the kernel's two pointers are two \
              buffers there and the alias was the statement's convenience \
              rather than the launcher's requirement.",
    },
    DivergedInPlace {
        symbol: "moe::add_moe_route_bias_bf16",
        was: &[],
        is: &[(0, 0)],
        why: "the kernel ACCUMULATES (`moe_dispatch.cuh:1132` reads the \
          destination cell before adding the expert bias into it), so \
          result 0 must be the buffer input 0 already holds. Without the \
          pair the allocator hands the result a fresh rectangle and the \
          add runs over whatever was in it -- garbage, silently, with \
          every operand resolving. §6.2 refuses it earlier and for a \
          different reason: `reads = 2` against `placed = 3` once the \
          bias became `Bank<0, T>`, which is the read side of the same \
          missing pair.",
    },
    DivergedInPlace {
        symbol: "rope::rope_yarn_bf16",
        was: &[],
        is: &[(0, 0), (1, 1)],
        why: "`rotate_yarn` (`rope.cuh:597`) forms `qp`/`kp` and calls \
          `rotate_pair`, which reads `h_ptr[i]` and `h_ptr[i + half]` \
          and assigns those same two cells. The scaling was the right \
          reason to check this separately from `rope_bf16` and it \
          changes nothing: neither plane is read from one operand and \
          written to another. `arity_problem` moves from a band of \
          [2, 2] against `reads = 0` to [0, 2], so the read side was \
          firing too -- and the filing that opened this named only the \
          write side, which is how the other half survived the repair.",
    },
    DivergedInPlace {
        symbol: "rope::qk_rmsnorm_mrope_bf16",
        was: &[],
        is: &[(0, 0), (1, 1)],
        why: "`qk_rmsnorm_rotate_mrope` (`rope.cuh:429`) selects `row` from \
          q or k, reduces over `row[i]`, and writes `row[i]` and \
          `row[i + half]`. The normalisation happens before the \
          rotation and both happen in place. Its band moves from \
          [4, 4] against `reads = 2` to [2, 4]; the two extra reads \
          over its twin are its two `Bank`s, so `reads = 2` is not \
          slack in the new floor, it is exactly on it.",
    },
    // AND THIS ONE NAMES A RESULT THE STATEMENT NEVER DECLARES, which is a
    // shape neither pin above has.
    // `norm::residual_add_rmsnorm_bf16` STOOD HERE and the divergence is
    // over: the row is back to the `[]` the pinned table records.
    //
    // Its `(1, 0)` named an output index one past the only one there is --
    // the old vocabulary's way of saying `hidden` is a pointer the statement
    // PLACES and the launcher WRITES THROUGH, with no result declared for it.
    // The four marks say that at the parameter instead: `hidden` is
    // `In<Tensor<T>>` and the body spells the mutation where the kernel is
    // called. `InOut` would be the wrong word — it claims a RESULT slot too,
    // which would push `norm_out` to `Out(1)`, a slot no statement fills.
    // `.wiki/migration.md` §11.6.
    // AND THESE TWO WERE ALREADY DIVERGENT WHEN THE MECHANISM WAS BUILT,
    // which is the argument for building it rather than editing two pins.
    // `4c3843dd3` repaired them with a full case at the `routine!` line and
    // this file has been failing ever since, because the only way to record
    // a deliberate `in_place` change was a list that did not exist. **A gate
    // with no exemption path does not stop a change; it stops REPORTING.**
    DivergedInPlace {
        symbol: "attn::dsa_index_q_rope_bf16",
        was: &[],
        is: &[(0, 0)],
        why: "`dsa_indexer.cuh:156-158` reads `row[d]` into a register buffer, rotates it \
              with `rope_interleave_inplace`, and writes `row[d]` back -- one buffer read \
              and written, and the launcher takes ONE pointer for both. It was left alone \
              once because *adding `in_place` changes how a real fire is planned, which is \
              a live repair with its own blast radius*, and half that blast radius was \
              `alias()`: an `in_place` shifted every `In(n)` in the derived column, so a \
              truthful declaration could move a launcher that takes the buffer twice onto \
              the wrong operand. `operands` counts now -- the remap runs only where the \
              column names FEWER `In`s than the statement has inputs -- and `alias()` is \
              deleted outright. The reason expired before the pin did.",
    },
    DivergedInPlace {
        symbol: "attn::dsa_index_knorm_rope_bf16",
        was: &[],
        is: &[(0, 0)],
        why: "Its twin, one layernorm earlier, and the same `row[d]` read and written \
              through one pointer. This pair is also where the `whole` argument above was \
              worked out, so they are the two rows carrying a recorded divergence on BOTH \
              columns in opposite directions: `whole` was reverted to `false` because \
              `false -> true` can refuse a model that used to load, and `in_place` moved \
              to true because its absence corrupts. Neither decision constrains the other, \
              and having them side by side is the clearest statement in this file that the \
              two columns are not the same kind of claim.",
    },
];

/// One row's `in_place`, before and after — see [`DIVERGED_IN_PLACE`].
struct DivergedInPlace {
    symbol: &'static str,
    /// What `9e3936fb9^` stated.
    was: &'static [(u32, u32)],
    /// What `sigs()` states now.
    is: &'static [(u32, u32)],
    /// Why the change is a correction and not a slip.
    why: &'static str,
}

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
    let diverged_in_place: HashMap<&str, &DivergedInPlace> =
        DIVERGED_IN_PLACE.iter().map(|d| (d.symbol, d)).collect();

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
        // Exempted exactly as `whole` is above, and for a stronger reason:
        // the two rows in `DIVERGED_IN_PLACE` had the pin holding the value
        // that corrupts. See that constant.
        // DERIVED NOW, AND THAT IS WHAT THIS PIN IS WORTH. The forty-one
        // values below were written by hand on the rows; they come off the
        // `InOut` marks in the signatures since, and this comparison is what
        // says the two agree. `KernelSig` carries no `in_place` method of its
        // own -- `kernels::routine::aliased` is the one definition
        // `Routine::in_place` and `Declared::in_place` both read `sources`
        // through, and `sigs()`'s rows are a third shape that reads it the
        // same way.
        if !diverged_in_place.contains_key(pinned.symbol)
            && kernels::routine::aliased(row.sources) != pinned.in_place
        {
            wrong.push(format!(
                "{}: `in_place` was {:?} at 9e3936fb9^, is {:?} now",
                pinned.symbol,
                pinned.in_place,
                kernels::routine::aliased(row.sources)
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

/// The same guard for [`DIVERGED_IN_PLACE`], and it is not optional.
///
/// Without it that constant is a list that can only grow -- the exact
/// failure the test above exists to prevent, and an exemption for an alias
/// is worse than one for `whole` in the one way that matters. A stale
/// `whole` exemption excuses a row that would only ever REFUSE. A stale
/// `in_place` exemption excuses a row whose alias was reverted, which is
/// the direction that corrupts, and it would excuse it with a paragraph
/// arguing the kernel writes in place.
#[test]
fn every_recorded_alias_divergence_is_still_a_divergence() {
    let now: HashMap<&str, &kernels::KernelSig> =
        sigs().iter().map(|row| (row.symbol, row)).collect();
    for d in DIVERGED_IN_PLACE {
        assert_ne!(d.was, d.is, "{}: `DIVERGED_IN_PLACE` records no difference", d.symbol);
        assert!(!d.why.is_empty(), "{}: a divergence with no argument is a bug report", d.symbol);
        let pinned = AT_9E3936FB9
            .iter()
            .find(|r| r.symbol == d.symbol)
            .unwrap_or_else(|| panic!("{}: diverges from a row 9e3936fb9^ never stated", d.symbol));
        assert_eq!(
            pinned.in_place, d.was,
            "{}: `was` disagrees with the pinned table",
            d.symbol
        );
        let row = now
            .get(d.symbol)
            .unwrap_or_else(|| panic!("{}: diverges from `sigs()` and is not in it", d.symbol));
        assert_eq!(
            kernels::routine::aliased(row.sources), d.is,
            "{}: `DIVERGED_IN_PLACE` says this row states `in_place = {:?}` on purpose, and it \
             states {:?}. If the alias was reverted, delete the entry -- but read the `why` \
             first, because these two were reverted once already by never being stated.",
            d.symbol,
            d.is,
            kernels::routine::aliased(row.sources)
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

// ── A MARK WHOSE KEY ALREADY FITS ────────────────────────────────────────
//
// Five times this session a `#[source(X)]` mark was converted to
// `Env<keys::X>` and the conversion needed NOTHING except someone noticing
// that `keys::X` was already there and already declared over the parameter's
// type. `rope.rs`'s `kv_page_indices` and `kv_page_indptr` had been
// convertible since the keys were minted. `mlp.rs`'s four GLU scalars were
// convertible the day `GluAlpha` and `GluLimit` landed.
//
// **NOBODY WAS WRONG ABOUT ANY OF THEM. There was simply no question being
// asked**, and that is what a gate is for: the tree has a `keys.rs` full of
// facts and a scattering of marks naming the same facts, and the only thing
// that ever compared the two lists was a person reading both.
//
// THE EXEMPTIONS ARE THE INTERESTING PART, because a key existing is not a
// key FITTING and the three ways it can fail to fit are three different
// repairs in three different files:
//
//   - THE KEY IS WRONG. `keys::KvHndLayout` was declared over `i32` against
//     a `bool` parameter. `rope.rs` diagnosed it, named `keys.rs` as the
//     file that had to change, and `keys.rs` changed. The declaration had
//     been wrong where nothing could contradict it -- `operand()` blocks
//     that source through the allowlist, so no fire ever compared the
//     declared `Ty::I32` against the `ArgValue::Bool` two lines beside it.
//   - THE PARAMETER IS WRONG. `keys::KvKeys` is `*mut u8` against a
//     `*mut bf16`. Here the KEY is right: a KV page is a run of bytes whose
//     element type is the layer's dtype, and a key declared `*mut bf16`
//     would be true of the bf16 launcher and false of its fp8 sibling.
//     **A fact that is true of one instantiation is not a fact.**
//   - NEITHER IS WRONG AND THE MARK IS BLOCKED ANYWAY. `Attn(..)` has no
//     `operand()` arm at all; it appears only on the blocked allowlist.
//
// So the gate lists what it excuses and why, and a mark that appears here
// without an entry is a mark someone could have converted and did not.
//
// THE CENSUS METHOD IS PART OF THE GATE AND NOT AN IMPLEMENTATION DETAIL.
// `rope.rs`'s header counted its own marks four times -- fifty-two, then
// sixty-five, then sixteen, then seven -- and every count was too high in
// the same direction, because a naive `grep '#[source('` on that file
// answers THIRTY and only TWO are marks. The other twenty-eight are
// paragraphs talking about marks. Comment lines are stripped below for that
// reason, and string-literal contents would be too if any mark could hide
// in one.
#[test]
fn no_mark_names_a_fact_a_key_already_carries() {
    // Marks that name a fact `keys.rs` declares, and the reason each one
    // cannot take it. Removing a line here without converting the mark makes
    // the test fail, which is the point: the exemption and the repair are
    // the same edit.
    const EXCUSED: &[(&str, &str)] = &[
        // `"KvKeys"` AND `"KvValues"` STOOD HERE, AND THE NOTATION GREW TO
        // REACH THEM. The excuse read *"the key is `*mut u8` and right; the
        // parameter is `*mut bf16` ... the route is a cast at the parameter,
        // in thirty launchers, which is the shape of change that gets half
        // made"*. That was a statement about a mark that carried a POINTER
        // and therefore an element it could disagree with the key about.
        //
        // `Tensor<E>` is the carrier now and a body ASKS: `ctx.ask::<Tensor
        // <bf16>, keys::KvKeys>()` names the element at the call, so the two
        // spellings meet in one expression and there is nothing left for the
        // key to disagree with. The thirty casts are gone with them.
        // `.wiki/migration.md` §11.1.
        // `"Attn"` STOOD HERE AND ITS EXCUSE WAS FALSIFIED BY BEING FIXED.
        // It read *"`operand()` has no `Source::Attn` arm; the variant
        // reaches the binder only through the blocked allowlist ... the mark
        // is blocked by design and the key would be too"*. Every clause of
        // that is now wrong and none of it was wrong when written:
        // `operand()` gained `Source::Named(<keys::SmScale as keys::Fact>::KEY) => f.sm_scale()`, the
        // allowlist lost its `Attn` entry in the same commit, and
        // `keys::SmScale` was minted -- so `attn/xqa.rs`'s `sm_scale` is
        // `Env<keys::SmScale>` and no `#[source(Attn` remains in the tree.
        //
        // **AN EXCUSE THAT NAMES ITS OWN BLOCKER IS THE ONLY KIND THAT CAN
        // EXPIRE.** The two above say the key does not FIT -- `*mut u8`
        // against a `*mut bf16` parameter -- a standing fact about two types
        // that expires only if one of them changes. This one said the binder
        // does not ANSWER, which is a fact about code somebody could go and
        // write, and somebody did. The reverse check below is what turned
        // that from a comment into a deadline: it failed the build the
        // moment the mark left, quoting the excuse back.
    ];

    let keyed: Vec<&str> = include_str!("../../kernels/src/keys.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .filter_map(|l| l.split("=> Source::").nth(1))
        .map(|r| {
            r.trim_end_matches(|c: char| !c.is_alphanumeric())
                .split(|c: char| !c.is_alphanumeric())
                .next()
                .unwrap_or("")
        })
        .filter(|v| !v.is_empty())
        .collect();
    assert!(
        keyed.len() > 20,
        "the scrape found {} keys, which means `keys.rs` changed shape and \
         this gate is now measuring nothing -- an empty scrape passes every \
         assertion below",
        keyed.len(),
    );

    let mut unexcused: Vec<(&str, &str)> = Vec::new();
    for (file, src) in [
        ("attn/mod.rs", include_str!("../src/attn/mod.rs")),
        ("attn/xqa.rs", include_str!("../src/attn/xqa.rs")),
        ("gemm/mod.rs", include_str!("../src/gemm/mod.rs")),
        ("layout.rs", include_str!("../src/layout.rs")),
        ("mlp.rs", include_str!("../src/mlp.rs")),
        ("moe.rs", include_str!("../src/moe.rs")),
        ("norm.rs", include_str!("../src/norm.rs")),
        ("quant.rs", include_str!("../src/quant.rs")),
        ("rope.rs", include_str!("../src/rope.rs")),
        ("ssm.rs", include_str!("../src/ssm.rs")),
    ] {
        for line in src.lines() {
            // THE STRIP THAT MAKES THE NUMBER MEAN ANYTHING. Without it
            // `rope.rs` alone contributes twenty-eight phantom marks.
            if line.trim_start().starts_with("//") {
                continue;
            }
            let Some(rest) = line.split("#[source(").nth(1) else {
                continue;
            };
            let variant = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            if variant.is_empty() || !keyed.contains(&variant) {
                continue;
            }
            if EXCUSED.iter().any(|(v, _)| *v == variant) {
                continue;
            }
            unexcused.push((file, variant));
        }
    }

    assert!(
        unexcused.is_empty(),
        "these marks name a fact `keys.rs` already declares, so each is an \
         `Env<keys::_>` parameter nobody wrote: {unexcused:?}",
    );

    // AND THE OTHER DIRECTION, which is the one that rots. An excused
    // variant that no longer appears anywhere has had its mark converted or
    // deleted, and the excuse outlived the thing it excused -- the same rot
    // as a comment citing a line that moved, except that an excuse decays
    // into PERMISSION rather than into noise.
    let all: String = [
        include_str!("../src/attn/mod.rs"),
        include_str!("../src/rope.rs"),
        include_str!("../src/attn/xqa.rs"),
    ]
    .concat();
    for (variant, why) in EXCUSED {
        assert!(
            all.contains(&format!("#[source({variant}")),
            "`{variant}` is excused for `{why}` and no longer appears; the \
             excuse is now a standing permission for a mark nobody has",
        );
    }
}


/// **§3.11 CANNOT DELETE `take_source_attr`, AND THIS IS THE PROOF, HELD OPEN.**
///
/// `.wiki/kilimanjaro2.md` §3.11 folds `Source` to three variants and its
/// plan for the attribute is deletion: every mark becomes a wrapper, so the
/// parser has nothing left to parse. The named half of that is done -- marks
/// went from fifty-two to seven across this refactor, and four families
/// (`mlp`, `ssm`, `norm`, `moe`) reached zero.
///
/// The last seven do not convert, and not because nobody has got to them.
/// Each is inexpressible in the wrapper notation for a reason recorded at its
/// own site, and the reasons fall into four classes that have nothing to do
/// with each other. **A PLAN THAT ENDS IN DELETION HAS TO BE TOLD WHEN THE
/// REMAINDER STOPS SHRINKING**, because a backlog of seven and a floor of
/// seven produce the same census reading, and the difference is the whole
/// question of whether the next pass has work in it.
///
/// So the attribute survives, BOUNDED: this test pins the residue exactly.
/// A new mark fails it, and a pinned mark that leaves without its pin also
/// fails it. `take_source_attr` stops being an open door and becomes an
/// escape hatch with a written list of who is allowed through.
///
/// THE SHARPEST OF THE FOUR CLASSES WAS CREATED BY THIS REFACTOR, NOT FOUND
/// BY IT. §3.11's answer for a width mark is E1, *"a body that wants a width
/// asks the region it already holds"*. `gemm/mod.rs`'s `n` and `k` cannot,
/// because the operands they measure are `Unbound<*const *const c_void>` --
/// host arrays of device pointers -- and **`Unbound<T>` is precisely the
/// declaration that the body does NOT hold a region.** Before those
/// parameters were wrapped, one commit ago, they sat among bare pointers and
/// these two marks looked like ordinary backlog. The conversion that made the
/// type say it is what proved the marks unconvertible: *two commits that
/// looked independent, and one is the other's receipt.*
#[test]
fn the_marks_that_remain_are_the_ones_that_cannot_convert() {
    // (file, variant, class). The class is the load-bearing column: a mark
    // with no class is backlog, and a mark with one is a floor.
    const RESIDUE: &[(&str, &str, &str)] = &[
        // `("ssm.rs", "WeightNamed2")` STOOD HERE, AND THE NOTATION GREW TO
        // REACH IT. The residue read *"the key does not fit ... `keys::
        // NamedWeight2` is a non-nullable `*const u8`; the parameter is
        // `MaybeConst<T>`"* -- a mismatch between what the KEY declared and
        // what the parameter could hold.
        //
        // `Const<Tensor<MaybeConst<T>>>` holds both: the mark says the
        // statement places it, `Tensor` says which element, and `MaybeConst`
        // says it may be absent. The chain it derives IS `weight2`'s, and
        // nullability is read off the carrier rather than off the key -- so
        // qwen3.5's `bias=False` is a deployment the column states, which is
        // what this entry said it wanted.
        // `("gemm/mod.rs", "OutWidth")` AND ITS INPUT-SIDE TWIN STOOD HERE,
        // AND THE NOTATION GREW TO REACH THEM -- by a different route than
        // the one they were blocked on.
        //
        // The residue read *"NO REGION TO ASK. E1 needs a body holding a
        // region; the operand is `Unbound<*const *const c_void>` ... a device
        // pointer inside a HOST array has no rectangle"*. Every clause is
        // still true: `grouped_act_x_wt_bf16` still takes four bare pointer
        // arrays and still holds no region.
        //
        // What changed is that a body no longer has to DERIVE a fact from a
        // rectangle to reach it. `ctx.ask::<i32, keys::OutWidth0>()` asks the
        // fire directly, which is what `#[source(OutWidth(0))]` was asking
        // for through the column, and neither reading needs an operand to
        // hang on. The blocked route stayed blocked; a second one opened.
        // `.wiki/migration.md` §4.4.
        // `("quant.rs", "OutElements")` STOOD HERE AND ITS SITE ANSWERED IT.
        // The residue read *"NO RECTANGLE EITHER ... a byte run longer than
        // `i32::MAX` does not fit `Extent`"*, and the repair it named was
        // widening `Extent` to `i64`.
        //
        // `cast_fp32_to` took a different one. `keys::OutElements0` is *"rows
        // times the result's row width"* -- the two numbers the operand
        // already carries -- so the mark was asking a driver for a product of
        // its own extents. `Out<Tensor<T>>` carries both, and the body reads
        // `Region::elements` off the value the caller placed, which is also
        // what makes the routine reachable from a hand-written caller: an ask
        // needs a fire behind it and `Ctx::on(stream)` has none.
        //
        // `Extent` is still `i32`, and that limit is still there. It is no
        // longer in this row's way.
        // `("rope.rs", "KvKeys")` AND ITS TWIN STOOD HERE, AND THE NOTATION
        // GREW TO REACH THEM -- the same growth that retired the pair in
        // `EXCUSED` above, for the same reason.
        //
        // The residue read *"the key does not fit: `*mut u8` against a
        // `*mut bf16` parameter, and the key is the correct one -- a KV page
        // is bytes whose element type is the layer's dtype, so a fact true of
        // one instantiation is not a fact"*. That is a mismatch between a
        // key's declared value type and a PARAMETER's, and it only exists
        // while the fact has to arrive as a parameter.
        //
        // It does not: `let k_pages = ctx.ask::<*mut bf16, keys::KvKeys>()?`
        // names the reading at the call. The key still declares the bytes and
        // this instantiation still reads them as bf16; the two now meet in
        // one expression, where a reader can see both.
        // `.wiki/migration.md` §11.1.
        // `("attn/mod.rs", "Weight")` WAS THE LAST ENTRY, AND THE LIST IS
        // EMPTY BECAUSE THE NOTATION IT MEASURED IS GONE.
        //
        // Every entry here named a `#[source(..)]` attribute — the escape
        // the five marks needed when a parameter's source could not be read
        // off its type. The four marks derive it from the type in every case
        // (`kernels/src/routine.rs`'s `resolve`), so the attribute has no
        // remaining use and no remaining site: the scan below finds only
        // prose mentions, which it strips.
        //
        // The defect this last entry named is NOT retired by that. The DSL
        // still places no weight for `mla_prepare`, and the parameter is
        // still there — it is `kv_a_norm_weight: Const<Tensor<bf16>>` now,
        // deriving `Or(Named("weight"), Slot(Weight, 0))` where it used to
        // say `#[source(Weight(0))]`. What changed is only that a scan for
        // attributes cannot see it. `model-ir`'s `arity_problem` can, and
        // that is the gate that should carry it.
        // `("attn/xqa.rs", "Attn")` STOOD HERE AND CONVERTED, WHICH IS THE
        // ONLY WAY AN ENTRY IS SUPPOSED TO LEAVE THIS LIST. Its class was
        // *"blocked by design, pending the four-part binder template -- the
        // only entry expected to convert"*, and the template landed: `Cx`
        // accessor, `Facts` field, `facts()` fill, `operand()` arm, with
        // `Source::Named(<keys::SmScale as keys::Fact>::KEY)` off the blocked allowlist in the same
        // change. `sm_scale` is now `Env<keys::SmScale>`.
        //
        // **THE GATE CAUGHT ITS OWN SUBJECT MOVING.** The reverse check
        // failed the build with the mark's class quoted back, which is what
        // the class column is for: six of the seven were floor, this one was
        // backlog, and the list said which before anybody touched it.
    ];

    // WALKED AT RUNTIME, NOT `include_str!`ed. Every other scrape in this
    // file names its inputs, and every one of them has drifted at least once
    // -- the list one gate above just failed the build by naming a `layer.rs`
    // this crate does not have. A named list also cannot see a NEW file, and
    // a new file is exactly where a new mark would arrive. `CARGO_MANIFEST_DIR`
    // is set for tests and the sources are on disk beside them, so the gate
    // reads the crate rather than a description of it.
    fn walk(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        for e in std::fs::read_dir(dir).expect("the crate's src/ is readable") {
            let p = e.expect("a readable dir entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let rel = p
                    .strip_prefix(concat!(env!("CARGO_MANIFEST_DIR"), "/src/"))
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, std::fs::read_to_string(&p).expect("readable")));
            }
        }
    }
    let mut files: Vec<(String, String)> = Vec::new();
    walk(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
        &mut files,
    );
    assert!(
        files.len() > 20,
        "the walk found {} files, so it is measuring nothing",
        files.len(),
    );

    let mut found: Vec<(&str, &str)> = Vec::new();
    for (name, src) in &files {
        for line in src.lines() {
            let t = line.trim_start();
            // THREE CLAUSES, ONE PER DEFECT THIS SESSION MEASURED. Strip
            // comments (`rope.rs` alone contributes twenty-eight phantoms);
            // require the attribute to OPEN the trimmed line, which is what
            // keeps `kernels-macros`' `compile_error!` text and two prose
            // paragraphs inside multi-line string literals from counting --
            // four of eleven raw hits tree-wide were exactly that.
            if t.starts_with("//") || !t.starts_with("#[source(") {
                continue;
            }
            let variant = t["#[source(".len()..]
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            if !variant.is_empty() {
                found.push((name.as_str(), variant));
            }
        }
    }

    let mut extra: Vec<&(&str, &str)> = found
        .iter()
        .filter(|(f, v)| !RESIDUE.iter().any(|(rf, rv, _)| rf == f && rv == v))
        .collect();
    extra.sort();
    assert!(
        extra.is_empty(),
        "a `#[source(..)]` mark appeared that is not in the pinned residue: \
         {extra:?}. Either convert it to a wrapper -- which is what §3.11 \
         wants and what forty-five other marks did -- or add it here WITH \
         the class of inexpressibility that stops it, because an unclassed \
         entry turns this floor back into a backlog.",
    );

    // The direction that rots. A residue entry whose mark has gone is a
    // deletion nobody recorded, and it reads in a diff exactly like an entry
    // that is still holding -- the same ambiguity that made the numeric pins
    // single-owner.
    for (file, variant, why) in RESIDUE {
        assert!(
            found.iter().any(|(f, v)| f == file && v == variant),
            "`{file}` no longer marks `{variant}`, pinned as `{why}`. If the \
             notation grew to reach it, say so here and take the line out in \
             the same commit; if the mark was merely deleted, the row it \
             bound is now sourced by POSITION and nothing said so.",
        );
    }

    // AND THE COUNT, said out loud, because the two assertions above both
    // pass on an empty scrape -- the failure mode `keyed.len() > 20` exists
    // to catch one gate over.
    assert_eq!(
        found.len(),
        RESIDUE.len(),
        "the scrape found {} marks against {} pinned; if this is zero the \
         `include_str!` list has drifted from the crate and this test is \
         measuring nothing",
        found.len(),
        RESIDUE.len(),
    );
}
