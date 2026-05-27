#include "model/deepseek_v4_forward.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>

#include "model/qwen3.hpp"  // for make_weight_view
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/kimi_mla.hpp"
#include "kernels/dsv4_routing.hpp"
#include "kernels/dsv4_hc.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

DsV4Workspace DsV4Workspace::allocate(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size)
{
    const int N = max_tokens;
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int q_lora = cfg.q_lora_rank;
    const int head_dim = cfg.head_dim;
    const int num_heads = cfg.num_attention_heads / std::max(1, tp_size);
    const int qk_rope = cfg.qk_rope_head_dim;
    const int o_lora = cfg.dsv4_o_lora_rank;
    const int o_groups = cfg.dsv4_o_groups;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int moe_I = cfg.moe_intermediate_size;
    const int M = cfg.dsv4_hc_mult;
    const int mix_hc = (2 + M) * M;

    DsV4Workspace ws;
    ws.hc_residual = DeviceTensor::allocate(DType::BF16,{N, M * H});
    ws.y           = DeviceTensor::allocate(DType::BF16,{N, H});
    ws.norm_x      = DeviceTensor::allocate(DType::BF16,{N, H});
    ws.norm_y      = DeviceTensor::allocate(DType::BF16,{N, H});

    // HC scratch
    ws.hc_mixes_f32 = DeviceTensor::allocate(DType::FP32, {N, M * H});
    ws.hc_gemm_out  = DeviceTensor::allocate(DType::FP32, {N, mix_hc});
    ws.hc_post_mix  = DeviceTensor::allocate(DType::FP32, {N, M});
    ws.hc_comb_mix  = DeviceTensor::allocate(DType::FP32, {N, M * M});
    ws.hc_head_gemm = DeviceTensor::allocate(DType::FP32, {N, M});

    ws.q_a       = DeviceTensor::allocate(DType::BF16,{N, q_lora});
    ws.q         = DeviceTensor::allocate(DType::BF16,{N, num_heads * head_dim});
    ws.kv        = DeviceTensor::allocate(DType::BF16,{N, head_dim});
    ws.q_rope    = DeviceTensor::allocate(DType::BF16,{N, num_heads * qk_rope});
    ws.k_rope    = DeviceTensor::allocate(DType::BF16,{N, qk_rope});
    ws.attn_out  = DeviceTensor::allocate(DType::BF16,{N, num_heads * head_dim});

    ws.wo_a_out  = DeviceTensor::allocate(DType::BF16,{N, o_groups * o_lora});
    ws.wo_b_out  = DeviceTensor::allocate(DType::BF16,{N, H});

    ws.router_logits = DeviceTensor::allocate(DType::BF16,{N, E});
    ws.topk_idx      = DeviceTensor::allocate(DType::INT32, {N, K});
    ws.topk_weights  = DeviceTensor::allocate(DType::FP32, {N, K});
    ws.moe_out       = DeviceTensor::allocate(DType::BF16,{N, H});

    const int local_shared_I = moe_I / std::max(1, tp_size);
    ws.shared_gate = DeviceTensor::allocate(DType::BF16,{N, local_shared_I});
    ws.shared_up   = DeviceTensor::allocate(DType::BF16,{N, local_shared_I});
    ws.shared_act  = DeviceTensor::allocate(DType::BF16,{N, local_shared_I});
    ws.shared_out  = DeviceTensor::allocate(DType::BF16,{N, H});

    const int O = max_logit_rows > 0 ? max_logit_rows : N;
    ws.logits = DeviceTensor::allocate(DType::BF16,{O, V});

    return ws;
}

void dsv4_forward_paged(
    const DsV4Weights& w,
    const HfConfig& cfg,
    const DsV4ForwardCfg& fwd_cfg,
    DsV4Workspace& ws,
    KvCache& kv_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    void* logits_out,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    const int N = total_tokens;
    const int H = cfg.hidden_size;
    const int L = cfg.num_hidden_layers;
    const float eps = cfg.rms_norm_eps;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int num_heads = cfg.num_attention_heads / T;
    const int head_dim = cfg.head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int qk_rope = cfg.qk_rope_head_dim;
    const int o_lora = cfg.dsv4_o_lora_rank;
    const int o_groups = cfg.dsv4_o_groups;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int moe_I = cfg.moe_intermediate_size;
    const int local_shared_I = moe_I / T;
    const int M = cfg.dsv4_hc_mult;
    const int mix_hc = (2 + M) * M;
    const float hc_eps = cfg.dsv4_hc_eps;
    const float hc_post_alpha = 2.0f;
    const int sinkhorn_iters = 20; // from config hc_sinkhorn_iters

    cudaStream_t stream = nullptr;  // default stream

    // ── Embedding ────────────────────────────────────────────────────
    if (w.embed_tp_sharded) {
        kernels::launch_embed_bf16_vocab_shard(
            token_ids, w.embed->data(), ws.y.data(),
            N, H, static_cast<int>(w.embed->shape()[0]),
            w.embed_tp_vocab_offset, stream);
    } else {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(), N, H, cfg.vocab_size, stream);
    }

    // Expand embedding [N, H] → [N, hc_mult, H]
    if (M > 1) {
        kernels::launch_hc_expand_bf16(
            ws.y.data(), ws.hc_residual.data(),
            N, M, H, stream);
    } else {
        cudaMemcpyAsync(ws.hc_residual.data(), ws.y.data(),
                        static_cast<std::size_t>(N) * H * 2, cudaMemcpyDeviceToDevice, stream);
    }

    // ── Per-layer processing ─────────────────────────────────────────
    for (int li = 0; li < L; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];

        // ── HC pre (attention) ───────────────────────────────────────
        // Extract layer_input [N, H] from multi-stream residual
        if (M > 1 && Lw.hc_attn_fn != nullptr) {
            // RMSNorm the flattened residual → F32
            kernels::launch_hc_rmsnorm_to_f32(
                ws.hc_residual.data(), static_cast<float*>(ws.hc_mixes_f32.data()),
                N, M * H, eps, stream);

            // GEMM: [N, M*H] F32 x [mix_hc, M*H]^T F32 → [N, mix_hc] F32
            // Using cublasSgemm
            {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSetStream(cublas.handle(), stream);
                cublasSgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    mix_hc, N, M * H,
                    &alpha,
                    static_cast<const float*>(Lw.hc_attn_fn->data()), M * H,
                    static_cast<const float*>(ws.hc_mixes_f32.data()), M * H,
                    &beta,
                    static_cast<float*>(ws.hc_gemm_out.data()), mix_hc);
            }

            // Post-process: sigmoid, sinkhorn, weighted sum
            kernels::launch_hc_pre_postprocess_bf16(
                static_cast<const float*>(ws.hc_gemm_out.data()),
                static_cast<const float*>(Lw.hc_attn_scale->data()),
                static_cast<const float*>(Lw.hc_attn_base->data()),
                ws.hc_residual.data(),
                static_cast<float*>(ws.hc_post_mix.data()),
                static_cast<float*>(ws.hc_comb_mix.data()),
                ws.norm_x.data(),   // layer_input output → norm_x
                N, M, H, hc_eps, hc_post_alpha, sinkhorn_iters, stream);
        } else {
            // No HC: just RMSNorm the residual
            kernels::launch_rmsnorm_bf16(
                ws.hc_residual.data(), Lw.attn_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
        }

        // ── Attention block ──────────────────────────────────────────
        // At this point ws.norm_x = layer_input [N, H] (from HC pre or RMSNorm)

        // Apply attn RMSNorm if HC was used (HC pre already normalizes)
        if (M > 1 && Lw.hc_attn_fn != nullptr) {
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), Lw.attn_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
        }

        // Q path: wq_a → q_norm → wq_b
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.wq_a, Lw.wq_a_quant), ws.q_a.data(),
            N, q_lora, H);

        kernels::launch_rmsnorm_bf16(
            ws.q_a.data(), Lw.q_norm->data(), ws.q_a.data(),
            N, q_lora, eps, stream);

        ops::gemm_act_x_w(cublas.handle(),
            ws.q_a.data(), make_weight_view(Lw.wq_b, Lw.wq_b_quant), ws.q.data(),
            N, num_heads * head_dim, q_lora);

        // KV path: wkv → kv_norm
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.wkv, Lw.wkv_quant), ws.kv.data(),
            N, head_dim, H);

        kernels::launch_rmsnorm_bf16(
            ws.kv.data(), Lw.kv_norm->data(), ws.kv.data(),
            N, head_dim, eps, stream);

        // TODO: V4 GPT-J interleaved RoPE on last qk_rope dims.
        // Standard RoPE for now (applied to Q's last rope dims and K's last rope dims).
        // V4's KV has head_dim=512 with last 64 dims being RoPE.
        // For correctness we'd split Q into [nope, rope] and K into [nope, rope],
        // apply RoPE to rope dims, then recombine. For now, skip RoPE (Phase 1).

        if (Lw.compress_ratio == 0) {
            // SWA layer: write KV to cache, run FlashInfer attention
            auto lv = kv_cache.layer_view(li);
            kernels::launch_write_kv_to_pages_bf16(
                lv.k_pages,
                lv.v_pages,
                ws.kv.data(),       // K [N, 1, head_dim]
                ws.kv.data(),       // V (same tensor for MQA)
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                N, num_requests,
                lv.page_size,
                1,                  // num_kv_heads = 1
                head_dim,
                false,              // hnd_layout
                stream);

            // Naive paged attention (supports HEAD_DIM=512)
            ops::launch_attention_naive_paged_bf16(
                ws.q.data(),        // Q [N, num_heads, head_dim]
                lv.k_pages,
                lv.v_pages,
                ws.attn_out.data(), // O [N, num_heads, head_dim]
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                N, num_requests,
                num_heads,
                1,                  // num_kv_heads = 1
                head_dim,
                lv.page_size,
                stream);
            // TODO: wo_a is a grouped BMM, not a standard GEMM.
            // For now, skip the output projection (identity residual).
            cudaMemsetAsync(ws.y.data(), 0,
                            static_cast<std::size_t>(N) * H * 2, stream);
        } else {
            // Compressed layers (C4/C128): skip attention entirely
            // (identity residual — no attention contribution)
            cudaMemsetAsync(ws.y.data(), 0,
                            static_cast<std::size_t>(N) * H * 2, stream);
        }

        // ── HC post (attention) ──────────────────────────────────────
        if (M > 1 && Lw.hc_attn_fn != nullptr) {
            kernels::launch_hc_post_bf16(
                ws.y.data(),             // layer output [N, H]
                ws.hc_residual.data(),   // residual [N, M, H]
                static_cast<const float*>(ws.hc_post_mix.data()),
                static_cast<const float*>(ws.hc_comb_mix.data()),
                ws.hc_residual.data(),   // output (in-place)
                N, M, H, stream);
        } else {
            kernels::launch_residual_add_bf16(
                ws.hc_residual.data(), ws.y.data(), N * H, stream);
        }

        // ── HC pre (FFN) ─────────────────────────────────────────────
        if (M > 1 && Lw.hc_ffn_fn != nullptr) {
            kernels::launch_hc_rmsnorm_to_f32(
                ws.hc_residual.data(), static_cast<float*>(ws.hc_mixes_f32.data()),
                N, M * H, eps, stream);

            {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSetStream(cublas.handle(), stream);
                cublasSgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    mix_hc, N, M * H,
                    &alpha,
                    static_cast<const float*>(Lw.hc_ffn_fn->data()), M * H,
                    static_cast<const float*>(ws.hc_mixes_f32.data()), M * H,
                    &beta,
                    static_cast<float*>(ws.hc_gemm_out.data()), mix_hc);
            }

            kernels::launch_hc_pre_postprocess_bf16(
                static_cast<const float*>(ws.hc_gemm_out.data()),
                static_cast<const float*>(Lw.hc_ffn_scale->data()),
                static_cast<const float*>(Lw.hc_ffn_base->data()),
                ws.hc_residual.data(),
                static_cast<float*>(ws.hc_post_mix.data()),
                static_cast<float*>(ws.hc_comb_mix.data()),
                ws.norm_y.data(),
                N, M, H, hc_eps, hc_post_alpha, sinkhorn_iters, stream);
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.hc_residual.data(), Lw.ffn_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
        }

        // Apply FFN RMSNorm
        if (M > 1 && Lw.hc_ffn_fn != nullptr) {
            kernels::launch_rmsnorm_bf16(
                ws.norm_y.data(), Lw.ffn_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
        }

        // ── FFN (MoE) block ──────────────────────────────────────────
        // Router
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_y.data(), ops::WeightView(*Lw.router), ws.router_logits.data(),
            N, E, H);

        // MoE routing
        if (Lw.is_hash_layer && Lw.tid2eid != nullptr) {
            kernels::launch_hash_route_lookup(
                token_ids,
                static_cast<const std::int64_t*>(Lw.tid2eid->data()),
                static_cast<std::int32_t*>(ws.topk_idx.data()),
                static_cast<float*>(ws.topk_weights.data()),
                N, cfg.vocab_size, K,
                1.0f / static_cast<float>(K),
                stream);
        } else {
            kernels::launch_topk_sqrtsoftplus_bf16(
                ws.router_logits.data(),
                static_cast<std::int32_t*>(ws.topk_idx.data()),
                static_cast<float*>(ws.topk_weights.data()),
                Lw.router_bias
                    ? static_cast<const float*>(Lw.router_bias->data())
                    : nullptr,
                N, E, K,
                cfg.norm_topk_prob,
                cfg.routed_scaling_factor,
                stream);
        }

        // TODO: Routed expert dispatch (MXFP4 experts). For now, only
        // shared expert runs.
        cudaMemsetAsync(ws.moe_out.data(), 0,
                        static_cast<std::size_t>(N) * H * 2, stream);

        // Shared expert (sharded by TP)
        if (Lw.shared_w1 != nullptr) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(Lw.shared_w1, Lw.shared_w1_quant), ws.shared_gate.data(),
                N, local_shared_I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(Lw.shared_w3, Lw.shared_w3_quant), ws.shared_up.data(),
                N, local_shared_I, H);
            kernels::launch_swiglu_bf16(
                ws.shared_gate.data(), ws.shared_up.data(),
                ws.shared_act.data(),
                static_cast<std::size_t>(N) * local_shared_I, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.shared_act.data(), make_weight_view(Lw.shared_w2, Lw.shared_w2_quant), ws.shared_out.data(),
                N, H, local_shared_I);
            kernels::launch_residual_add_bf16(
                ws.moe_out.data(), ws.shared_out.data(), N * H, stream);
        }

        // ── HC post (FFN) ────────────────────────────────────────────
        if (M > 1 && Lw.hc_ffn_fn != nullptr) {
            kernels::launch_hc_post_bf16(
                ws.moe_out.data(),
                ws.hc_residual.data(),
                static_cast<const float*>(ws.hc_post_mix.data()),
                static_cast<const float*>(ws.hc_comb_mix.data()),
                ws.hc_residual.data(),
                N, M, H, stream);
        } else {
            kernels::launch_residual_add_bf16(
                ws.hc_residual.data(), ws.moe_out.data(), N * H, stream);
        }
    }

    // ── HC head: collapse multi-stream → single-stream ───────────────
    if (M > 1 && w.hc_head_fn != nullptr) {
        kernels::launch_hc_rmsnorm_to_f32(
            ws.hc_residual.data(), static_cast<float*>(ws.hc_mixes_f32.data()),
            N, M * H, eps, stream);

        // GEMM: [N, M*H] F32 x [M, M*H]^T → [N, M] F32
        {
            const float alpha = 1.0f, beta = 0.0f;
            cublasSetStream(cublas.handle(), stream);
            cublasSgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, M * H,
                &alpha,
                static_cast<const float*>(w.hc_head_fn->data()), M * H,
                static_cast<const float*>(ws.hc_mixes_f32.data()), M * H,
                &beta,
                static_cast<float*>(ws.hc_head_gemm.data()), M);
        }

        kernels::launch_hc_head_postprocess_bf16(
            static_cast<const float*>(ws.hc_head_gemm.data()),
            static_cast<const float*>(w.hc_head_scale->data()),
            static_cast<const float*>(w.hc_head_base->data()),
            ws.hc_residual.data(),
            ws.y.data(),
            N, M, H, stream);
    } else {
        // No HC: residual is already [N, H] in the first H elements
        cudaMemcpyAsync(ws.y.data(), ws.hc_residual.data(),
                        static_cast<std::size_t>(N) * H * 2,
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ── Final output ─────────────────────────────────────────────────
    int rows = N;
    void* final_in = ws.y.data();
    if (logit_row_indices_d != nullptr && num_logit_rows > 0 && num_logit_rows < N) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            num_logit_rows, H, stream);
        final_in = ws.norm_x.data();
        rows = num_logit_rows;
    }

    kernels::launch_rmsnorm_bf16(
        final_in, w.final_norm->data(), ws.norm_y.data(),
        rows, H, eps, stream);

    if (fwd_cfg.emit_logits) {
        const int local_vocab = static_cast<int>(w.lm_head->shape()[0]);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_y.data(), ops::WeightView(*w.lm_head), logits_out,
            rows, local_vocab, H);
    }
}

}  // namespace pie_cuda_driver::model
