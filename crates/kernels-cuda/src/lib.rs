//! CUDA's kernel signature table — one row per launcher symbol in `csrc/`.
//!
//! The rows live here, beside the `.cu` files they describe, so that adding a
//! kernel is one source file and one table row in the same directory and the
//! same diff hunk. The words a row is written in — [`KernelSig`], `whole`,
//! `needs`, `lacks`, `sink` — are `kernels`', which is also where the reasons
//! for each of them are.
//!
//! ## Reading this without a GPU
//!
//! The table is the crate's `default-features = false` surface, and that is
//! deliberate: `model-compiler` reads it on every trace, and a compiler dev
//! loop must not pay nvcc to look up a symbol's contract. Turning on
//! `native` adds the CMake build of `csrc/` and nothing to what is below.
//!
//! The table is kept honest from the other end: `model-compiler`'s
//! `kernels::check_plan` refuses any `OpKind::Launch` symbol no row declares,
//! so a kernel cannot be stated by a model text without its contract.

pub use kernels::{Cap, KernelSig, Prepare};
use kernels::kernel;

/// Every kernel a lowered declaration may state.
pub static KERNELS: &[KernelSig] = &[
    // ── attention ──────────────────────────────────────────────────
    kernel!(flashinfer_decode "dispatch_attention_flashinfer_decode",
        needs = Prepare::DecodePlan, sink = Some("kv.pages"),
        depth_prefix_plan = true),
    kernel!(flashinfer_decode_capture "dispatch_attention_flashinfer_decode_capture",
        needs = Prepare::DecodePlan, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill "dispatch_attention_flashinfer_prefill_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    // The plan-free prefill wrapper: it builds an R-shaped plan on the
    // way in, so it owes its caller nothing and cannot be handed a row
    // window — `whole`, and `FireWide` for the same reason XQA is.
    kernel!(flashinfer_prefill_planless "ops::launch_attention_flashinfer_prefill",
        whole = true, needs = Prepare::FireWide, sink = Some("kv.pages")),
    // Head dims flashinfer's prefill template rejects (gemma-4's 512)
    // take a naive paged kernel instead. No plan at all; fire-shaped.
    kernel!(attention_naive_paged "ops::launch_attention_naive_paged",
        whole = true, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill_capture "dispatch_attention_flashinfer_prefill_capture_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    kernel!(flashinfer_custom "dispatch_attention_flashinfer_prefill_custom",
        needs = Prepare::CustomPlan, sink = Some("kv.pages")),
    // XQA: its prepare is fire-wide (R-shaped), so the kernel cannot be
    // given a row window — `whole`. And no capture variant of it
    // exists, so it cannot publish scores — `lacks Scores`. Both are
    // hand-written rules today: the first is the model body's
    // `window_one && c.xqa_decode` test, the second a C++ throw.
    kernel!(xqa_decode "launch_attention_xqa_decode_bf16_prepared",
        whole = true, needs = Prepare::FireWide, lacks = &[Cap::Scores]),
    kernel!(dequant "launch_dequant_kv_cache_layer_to_bf16_active"),

    // ── qkv / norms / rope / kv write ──────────────────────────────
    kernel!(rope_standard_table "launch_rope_standard_table"),
    kernel!(qk_rmsnorm_rope "launch_qk_rmsnorm_rope_bf16"),
    kernel!(qkv_decode_fused "launch_qkv_decode_qk_norm_rope_write_kv_bf16"),
    kernel!(write_kv_explicit "launch_write_kv_explicit_bf16"),
    kernel!(write_kv_to_pages "launch_write_kv_to_pages"),

    // ── mlp ────────────────────────────────────────────────────────
    // Two spellings of one arithmetic, and the BINDING picks: a packed
    // gate‖up bank feeds the chunked form, two narrow buffers the pair
    // form. A load-time fact, so the declaration states it.
    kernel!(chunked_swiglu "launch_chunked_swiglu_bf16"),
    kernel!(swiglu "launch_swiglu_bf16"),

    // ── gemma-3n: AltUp ────────────────────────────────────────────
    // A rank-K residual stream: K parallel streams predicted from each
    // other, one of them run through the real layer, the rest corrected
    // from the difference. See `dsl::cuda`'s AltUp block for the algebra.
    //
    // Not one of these carries a contract clause, and that is a claim
    // rather than an omission: every one is row-shaped -- token `t`'s
    // output reads only token `t`'s inputs -- so a peel may split it, it
    // obligates no host plan, and there is no seam capability for it to
    // refuse.
    kernel!(altup_predict "launch_altup_predict_bf16"),
    kernel!(altup_correct "launch_altup_correct_bf16"),
    kernel!(altup_unpack_predict_coefs "launch_altup_unpack_predict_coefs"),
    kernel!(altup_unpack_correct_coefs "launch_altup_unpack_correct_coefs"),
    kernel!(mean_streams "launch_mean_streams_bf16"),
    kernel!(compute_rms "launch_compute_rms_bf16"),
    kernel!(magnitude_rescale "launch_magnitude_rescale_bf16"),
    kernel!(tanh "launch_tanh_bf16"),
    kernel!(gaussian_topk "launch_gaussian_topk_bf16"),

    // ── gemma-4 ────────────────────────────────────────────────────
    // GeGLU-tanh is not a swiglu variant: `gelu_pytorch_tanh` on the
    // gate is a different function. The packed/pair split is the same
    // binding question.
    kernel!(geglu_tanh "launch_geglu_tanh_bf16"),
    kernel!(chunked_geglu_tanh "launch_chunked_geglu_tanh_bf16"),
    // Weightless per-head norm (the V-norm) — no gamma, so no variant.
    kernel!(rmsnorm_no_scale "launch_rmsnorm_no_scale_bf16"),
    // Four statements in one launch, and two: gemma-4 fuses the next
    // block's input norm into the previous block's landing, which is why
    // its layer body appears to be missing one.
    kernel!(norm_residual_scale_norm "launch_rmsnorm_residual_add_scale_rmsnorm_bf16"),
    kernel!(norm_residual_add "launch_rmsnorm_residual_add_bf16"),
    kernel!(scalar_mul "launch_scalar_mul_bf16"),
    kernel!(logit_softcap "launch_logit_softcap_bf16"),
    // Q-only rotation: a KV-shared layer's K was rotated at its source
    // layer. One operand is the statement.
    kernel!(rope_partial_q_only "launch_rope_partial_bf16"),
    // Six statements in one launch; the only value that survives is q.
    kernel!(qkv_packed_post "launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        sink = Some("kv.pages")),
    // gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    // numbers come out — so the symbol IS the statement.
    kernel!(qk_rmsnorm_rope_rounded "launch_qk_rmsnorm_rope_bf16_rounded"),
    // The PLE relay: [N, L, D] -> [L, N, D], so a layer reads a
    // contiguous slice. Addressing, not arithmetic.
    kernel!(transpose_nld_to_lnd "launch_transpose_bf16_nld_to_lnd"),

    // ── MoE ────────────────────────────────────────────────────────
    // The router's top-k, then the decode GEMV leg's two routed
    // projections and its combine. The expert axis rides INSIDE the
    // value on this leg, so the whole branch stays a list of rectangles;
    // the grouped-GEMM and host-routed legs reach the same numbers by
    // shapes no `Dim` spells, and are named refusals, not entries.
    kernel!(topk_softmax "launch_topk_softmax_bf16"),
    // The whole routed block as one call — permute, both grouped GEMMs,
    // the activation and the weighted finalize. The leg decode actually
    // takes, and the only one that is a single rectangle.
    // Namespaced because it is not a `kernels::launch_*` at all: it is an
    // `ops::` entry point that installs tactics and runs a CUTLASS
    // pipeline. The symbol says so.
    kernel!(moe_fused_cutlass "ops::flashinfer_cutlass_moe_bf16"),
    kernel!(moe_gate_up_gemv "launch_moe_gate_up_decode_gemv_bf16"),
    kernel!(moe_down_gemv "launch_moe_down_decode_gemv_bf16"),
    kernel!(moe_shared_gate_dot "launch_sigmoid_dot_scalar_gate_add_bf16"),
    kernel!(residual_add_cuda "launch_residual_add_bf16"),
    // The combine folds the residual when the MoE output lands straight
    // on the stream (tp=1) — one launch where the semantic text has a
    // WeightedSum and a ResidualAdd.
    kernel!(moe_weighted_sum "launch_token_batched_weighted_sum_bf16"),
    kernel!(moe_weighted_sum_add "launch_token_batched_weighted_sum_add_bf16"),

    // ── gpt-oss ────────────────────────────────────────────────────
    // The sink rescale, and the fp32 LSE it eats. The LSE has no row of
    // its own: it is a second OUTPUT of the decode dispatch, requested
    // by an argument, so the kernel that changes is none.
    // A projection with its bias in the EPILOGUE — one launch where a
    // matmul plus an AddBias is two, and a different accumulation order.
    kernel!(gemm_bias "ops::gemm_act_x_wt_bias_bf16"),
    // YaRN, as its paper spells it. A deployment's scaling is a load-time
    // config answer, so it picks a kernel here rather than an argument.
    kernel!(rope_yarn_original "launch_rope_yarn_original_bf16"),
    kernel!(attention_sink_rescale "launch_attention_sink_rescale_bf16"),
    kernel!(bf16_to_fp16 "launch_bf16_to_fp16"),
    // The routed MXFP4 GEMVs. Like qwen3_5's GEMV leg the expert axis
    // rides INSIDE the value, so each is one rectangle over `N * k`
    // routes; unlike it, the weight slot names a per-expert POINTER
    // BANK, which is a binding question and not a shape one.
    kernel!(mxfp4_moe_gate_up "launch_mxfp4_moe_gate_up_decode_bf16"),
    kernel!(mxfp4_moe_down "launch_mxfp4_moe_down_decode_bf16"),
    // SwiGLU with a clamp. `swiglu_limit` is a config constant, so this
    // is a different kernel and not a different argument.
    kernel!(gpt_oss_glu "launch_gpt_oss_glu_bf16"),

    // ── adapters ───────────────────────────────────────────────────
    kernel!(lora_qkv_correction "pie_lora_qkv_correction"),

    // ── gdn: conv, recurrence, stash ───────────────────────────────
    kernel!(gdn_conv_update "launch_causal_conv1d_update_batched_bf16"),
    kernel!(gdn_conv_prefill "launch_causal_conv1d_prefill_batched_bf16"),
    kernel!(gdn_step "launch_recurrent_gated_delta_step_batched"),
    kernel!(gdn_step_gqa "launch_recurrent_gated_delta_step_batched_gqa"),
    kernel!(gdn_step_state_bf16 "launch_recurrent_gated_delta_step_batched_state_bf16"),
    kernel!(gdn_step_gqa_state_bf16 "launch_recurrent_gated_delta_step_batched_gqa_state_bf16"),
    kernel!(gdn_prefill_fla "launch_chunk_gated_delta_prefill_batched"),
    kernel!(gdn_prefill_fla_state_bf16 "launch_chunk_gated_delta_prefill_batched_state_bf16"),
    kernel!(gdn_prefill_cached "launch_chunk_gated_delta_prefill_batched_cached"),
    kernel!(gdn_prefill_cached_state_bf16
        "launch_chunk_gated_delta_prefill_batched_cached_state_bf16"),
    kernel!(gdn_prefill_warp_tiled_gqa "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa"),
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"),
    kernel!(repeat_interleave_heads "launch_repeat_interleave_heads_fp32"),
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
];
