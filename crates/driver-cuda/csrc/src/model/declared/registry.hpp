#pragma once

// THE REGISTRY — every launcher symbol a declared trace may state, and
// the one lookup that resolves it.
//
// There were four of these: `LaunchKernel` (llama_like, 28 symbols),
// `Q35Kernel` (qwen3.5, 39), `G4Kernel` (gemma-4, 20) and `GoKernel`
// (gpt-oss, 15), each with its own enum and its own chain of string
// compares. Eighteen symbols appeared in more than one, so eighteen
// contracts were written between two and four times.
//
// Four copies of a table is not only duplication -- it is four places a
// symbol can mean something different, and it did. `AttnFlashinferPrefill`
// named the PLANNED dispatch in qwen3.5 and the PLAN-FREE wrapper in
// gemma-4 and gpt-oss: two different kernels, one enumerator name, and
// nothing to notice it until the tables met. It is `AttnFlashinferPrefill`
// and `AttnFlashinferPrefillPlanless` here, which is what the dsl surface
// called them all along.
//
// An enumerator is a SYMBOL, never a family's use of one. Two families
// that state the same symbol with different arities (gemma-4's rope
// passes `num_kv_heads = 0` where llama_like passes a real count) share
// the entry and the arm reads the arity off the statement -- which is
// binding, not choosing.
//
// A symbol NOT here is a trace that drifted from this driver, and
// `resolve` says so by name at model load rather than mid-fire.

#include <stdexcept>
#include <string>
#include <string_view>

namespace pie_cuda_driver::model::declared {

enum class Kernel {
    // ── norm ──
    ResidualAdd,                             // gemma4,mixtral
    ResidualAddRmsnorm,                      // llama_like
    RmsnormRow,                              // llama_like,qwen3_5,gemma4,mixtral
    RmsnormRowGemma,                         // llama_like,qwen3_5,gemma4,mixtral
    RmsnormNoScale,                          // gemma4
    NormResidualAdd,                         // gemma4
    NormResidualScaleNorm,                   // gemma4
    ScalarMul,                               // gemma4
    // ── rope ──
    QkRmsnormRope,                           // llama_like
    QkRmsnormRopeRounded,                    // gemma4
    RopeFull,                                // llama_like,qwen3_5,gemma4
    RopePartial,                             // llama_like,qwen3_5,gemma4
    RopeStandardTable,                       // llama_like
    RopeYarnOriginal,                        // mixtral
    // ── attn ──
    AttnFlashinferPrefillPlanless,           // gemma4,mixtral
    AttnNaivePaged,                          // gemma4
    AttnSinkRescale,                         // mixtral
    AttentionXqaDecodePrepared,              // llama_like
    DequantKvCacheLayerToBf16Active,         // llama_like
    AttnFlashinferDecode,                    // llama_like,qwen3_5,gemma4,mixtral
    AttentionFlashinferDecodeCapture,        // llama_like
    AttnFlashinferPrefill,                   // llama_like,qwen3_5
    AttentionFlashinferPrefillCapture,       // llama_like
    AttentionFlashinferPrefillCustom,        // llama_like
    AttentionMla,                            // unclaimed
    LogitSoftcap,                            // gemma4
    MlaPrepare,                              // unclaimed
    PadHeadDim,                              // llama_like
    QkvDecodeQkNormRopeWriteKv,              // llama_like
    QkvPackedPost,                           // gemma4
    StripHeadDim,                            // llama_like
    WriteKvExplicit,                         // llama_like,qwen3_5
    WriteKvToPages,                          // llama_like,qwen3_5,gemma4,mixtral
    WriteMlaToPages,                         // unclaimed
    // ── gemm ──
    GemmBias,                                // mixtral
    MatmulChannelScaled,                     // llama_like,qwen3_5
    MatmulGroupedScaled,                     // llama_like,qwen3_5
    MatmulMxfp4Marlin,                       // llama_like,qwen3_5
    MatmulTensorScaled,                      // llama_like,qwen3_5
    MlaAbsorbLatentToV,                      // unclaimed
    MlaAbsorbQToLatent,                      // unclaimed
    // ── mlp ──
    ChunkedGegluTanh,                        // gemma4
    ChunkedSwiglu,                           // llama_like,qwen3_5
    GegluTanh,                               // gemma4
    GptOssGlu,                               // mixtral
    SigmoidDotScalarGateAdd,                 // qwen3_5
    Swiglu,                                  // llama_like,qwen3_5
    // ── moe ──
    MoeBuildPtrsAligned,                     // qwen3_5
    MoeGatherAligned,                        // qwen3_5
    MoeAlignDecode,                          // qwen3_5
    MoeDownDecodeGemv,                       // unclaimed
    MoeGateUpDecodeGemv,                     // unclaimed
    MoeGroupedGemm,                          // qwen3_5
    MoeReorderAligned,                       // qwen3_5
    MoeWeightedSum,                          // qwen3_5
    WeightedSum,                             // mixtral
    TopkSigmoid,                             // unclaimed
    TopkSoftmax,                             // qwen3_5,mixtral
    // ── ssm ──
    ConvPrefillBatched,                      // qwen3_5
    ConvUpdateBatched,                       // qwen3_5
    PrefillFla,                              // qwen3_5
    PrefillCached,                           // qwen3_5
    PrefillCachedBf16,                       // qwen3_5
    PrefillFlaBf16,                          // qwen3_5
    PrefillWarpTiledGqa,                     // qwen3_5
    PrefillWarpTiledGqaBf16,                 // qwen3_5
    StepBatched,                             // qwen3_5
    StepBatchedGqa,                          // qwen3_5
    StepBatchedGqaBf16,                      // qwen3_5
    StepBatchedBf16,                         // qwen3_5
    RepeatInterleave,                        // qwen3_5
    // ── layout ──
    SplitRows,                               // qwen3_5
    SplitGdnBa,                              // qwen3_5
    TransposeNldToLnd,                       // gemma4
    // ── quant ──
    Bf16ToFp16,                              // mixtral
    Mxfp4Down,                               // mixtral
    Mxfp4GateUp,                             // mixtral
    // ── dist ──
    AllReduce,                               // llama_like
    AllReduceOut,                            // llama_like
    // ── pie_lora_qkv_correction ──
    LoraQkvCorrection,                       // llama_like,gemma4
    // ── qwen35_verify_stash_load ──
    VerifyStashLoad,                         // qwen3_5
    // ── qwen35_verify_stash_store ──
    VerifyStashStore,                        // qwen3_5
};

// The lookup. A linear chain of `==` over string_views, which is what
// each of the four was: the list is short, the compare is a length check
// plus a memcmp, and this runs once per op rather than per row.
inline Kernel resolve_kernel(std::string_view k) {
    if (k == "norm::residual_add_bf16") return Kernel::ResidualAdd;
    if (k == "norm::residual_add_rmsnorm_bf16") return Kernel::ResidualAddRmsnorm;
    if (k == "norm::rmsnorm_bf16") return Kernel::RmsnormRow;
    if (k == "norm::rmsnorm_gemma_bf16") return Kernel::RmsnormRowGemma;
    if (k == "norm::rmsnorm_no_scale_bf16") return Kernel::RmsnormNoScale;
    if (k == "norm::rmsnorm_residual_add_bf16") return Kernel::NormResidualAdd;
    if (k == "norm::rmsnorm_residual_add_scale_rmsnorm_bf16") return Kernel::NormResidualScaleNorm;
    if (k == "norm::scalar_mul_bf16") return Kernel::ScalarMul;
    if (k == "rope::qk_rmsnorm_rope_bf16") return Kernel::QkRmsnormRope;
    if (k == "rope::qk_rmsnorm_rope_bf16_rounded") return Kernel::QkRmsnormRopeRounded;
    if (k == "rope::rope_bf16") return Kernel::RopeFull;
    if (k == "rope::rope_partial_bf16") return Kernel::RopePartial;
    if (k == "rope::rope_standard_table") return Kernel::RopeStandardTable;
    if (k == "rope::rope_yarn_original_bf16") return Kernel::RopeYarnOriginal;
    if (k == "attn::attention_flashinfer_prefill") return Kernel::AttnFlashinferPrefillPlanless;
    if (k == "attn::attention_naive_paged") return Kernel::AttnNaivePaged;
    if (k == "attn::attention_sink_rescale_bf16") return Kernel::AttnSinkRescale;
    if (k == "attn::attention_xqa_decode_bf16_prepared") return Kernel::AttentionXqaDecodePrepared;
    if (k == "attn::dequant_kv_cache_layer_to_bf16_active") return Kernel::DequantKvCacheLayerToBf16Active;
    if (k == "attn::dispatch_attention_flashinfer_decode") return Kernel::AttnFlashinferDecode;
    if (k == "attn::dispatch_attention_flashinfer_decode_capture") return Kernel::AttentionFlashinferDecodeCapture;
    if (k == "attn::dispatch_attention_flashinfer_prefill_bf16") return Kernel::AttnFlashinferPrefill;
    if (k == "attn::dispatch_attention_flashinfer_prefill_capture_bf16") return Kernel::AttentionFlashinferPrefillCapture;
    if (k == "attn::dispatch_attention_flashinfer_prefill_custom") return Kernel::AttentionFlashinferPrefillCustom;
    if (k == "attn::dispatch_attention_mla_bf16") return Kernel::AttentionMla;
    if (k == "attn::logit_softcap_bf16") return Kernel::LogitSoftcap;
    if (k == "attn::mla_prepare_bf16") return Kernel::MlaPrepare;
    if (k == "attn::pad_head_dim_bf16") return Kernel::PadHeadDim;
    if (k == "attn::qkv_decode_qk_norm_rope_write_kv_bf16") return Kernel::QkvDecodeQkNormRopeWriteKv;
    if (k == "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16") return Kernel::QkvPackedPost;
    if (k == "attn::strip_head_dim_bf16") return Kernel::StripHeadDim;
    if (k == "attn::write_kv_explicit_bf16") return Kernel::WriteKvExplicit;
    if (k == "attn::write_kv_to_pages") return Kernel::WriteKvToPages;
    if (k == "attn::write_mla_to_pages") return Kernel::WriteMlaToPages;
    if (k == "gemm::act_x_wt_bias_bf16") return Kernel::GemmBias;
    if (k == "gemm::act_x_wt_channel_scaled") return Kernel::MatmulChannelScaled;
    if (k == "gemm::act_x_wt_grouped_scaled") return Kernel::MatmulGroupedScaled;
    if (k == "gemm::act_x_wt_mxfp4_marlin") return Kernel::MatmulMxfp4Marlin;
    if (k == "gemm::act_x_wt_tensor_scaled") return Kernel::MatmulTensorScaled;
    if (k == "gemm::mla_absorb_latent_to_v_bf16") return Kernel::MlaAbsorbLatentToV;
    if (k == "gemm::mla_absorb_q_to_latent_bf16") return Kernel::MlaAbsorbQToLatent;
    if (k == "mlp::chunked_geglu_tanh_bf16") return Kernel::ChunkedGegluTanh;
    if (k == "mlp::chunked_swiglu_bf16") return Kernel::ChunkedSwiglu;
    if (k == "mlp::geglu_tanh_bf16") return Kernel::GegluTanh;
    if (k == "mlp::gpt_oss_glu_bf16") return Kernel::GptOssGlu;
    if (k == "mlp::sigmoid_dot_scalar_gate_add_bf16") return Kernel::SigmoidDotScalarGateAdd;
    if (k == "mlp::swiglu_bf16") return Kernel::Swiglu;
    if (k == "moe::build_moe_ptrs_aligned_bf16") return Kernel::MoeBuildPtrsAligned;
    if (k == "moe::gather_moe_aligned_inputs_bf16") return Kernel::MoeGatherAligned;
    if (k == "moe::moe_align_decode") return Kernel::MoeAlignDecode;
    if (k == "moe::moe_down_decode_gemv_bf16") return Kernel::MoeDownDecodeGemv;
    if (k == "moe::moe_gate_up_decode_gemv_bf16") return Kernel::MoeGateUpDecodeGemv;
    if (k == "moe::moe_grouped_gemm_bf16") return Kernel::MoeGroupedGemm;
    if (k == "moe::reorder_moe_aligned_output_bf16") return Kernel::MoeReorderAligned;
    if (k == "moe::token_batched_weighted_sum_add_bf16") return Kernel::MoeWeightedSum;
    if (k == "moe::token_batched_weighted_sum_bf16") return Kernel::WeightedSum;
    if (k == "moe::topk_sigmoid_bf16") return Kernel::TopkSigmoid;
    if (k == "moe::topk_softmax_bf16") return Kernel::TopkSoftmax;
    if (k == "ssm::causal_conv1d_prefill_batched_bf16") return Kernel::ConvPrefillBatched;
    if (k == "ssm::causal_conv1d_update_batched_bf16") return Kernel::ConvUpdateBatched;
    if (k == "ssm::chunk_gated_delta_prefill_batched") return Kernel::PrefillFla;
    if (k == "ssm::chunk_gated_delta_prefill_batched_cached") return Kernel::PrefillCached;
    if (k == "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16") return Kernel::PrefillCachedBf16;
    if (k == "ssm::chunk_gated_delta_prefill_batched_state_bf16") return Kernel::PrefillFlaBf16;
    if (k == "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa") return Kernel::PrefillWarpTiledGqa;
    if (k == "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16") return Kernel::PrefillWarpTiledGqaBf16;
    if (k == "ssm::recurrent_gated_delta_step_batched") return Kernel::StepBatched;
    if (k == "ssm::recurrent_gated_delta_step_batched_gqa") return Kernel::StepBatchedGqa;
    if (k == "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16") return Kernel::StepBatchedGqaBf16;
    if (k == "ssm::recurrent_gated_delta_step_batched_state_bf16") return Kernel::StepBatchedBf16;
    if (k == "ssm::repeat_interleave_heads_fp32") return Kernel::RepeatInterleave;
    if (k == "layout::split_bf16_rows") return Kernel::SplitRows;
    if (k == "layout::split_qwen_gdn_ba_bf16") return Kernel::SplitGdnBa;
    if (k == "layout::transpose_bf16_nld_to_lnd") return Kernel::TransposeNldToLnd;
    if (k == "quant::bf16_to_fp16") return Kernel::Bf16ToFp16;
    if (k == "quant::mxfp4_moe_down_decode_bf16") return Kernel::Mxfp4Down;
    if (k == "quant::mxfp4_moe_gate_up_decode_bf16") return Kernel::Mxfp4GateUp;
    if (k == "dist::all_reduce_bf16") return Kernel::AllReduce;
    if (k == "dist::all_reduce_bf16_out") return Kernel::AllReduceOut;
    if (k == "pie_lora_qkv_correction") return Kernel::LoraQkvCorrection;
    if (k == "qwen35_verify_stash_load") return Kernel::VerifyStashLoad;
    if (k == "qwen35_verify_stash_store") return Kernel::VerifyStashStore;
    throw std::runtime_error(
        "declared forward: stated kernel '" + std::string(k) +
        "' is not in the registry (the trace and the driver drifted)");
}

}  // namespace pie_cuda_driver::model::declared
