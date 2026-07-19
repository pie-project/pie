#include "model/deepseek_v4/deepseek_v4_forward.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cublas_v2.h>

#include "model/llama_like/qwen3.hpp"  // for make_weight_view
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_fp8.hpp"
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
#include "kernels/dsv4_compress.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>> weights;
};

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx,
    const std::vector<float>& topk_w,
    int N, int K, int E)
{
    ExpertRouting r;
    r.token_idx.resize(static_cast<std::size_t>(E));
    r.weights.resize(static_cast<std::size_t>(E));
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx[static_cast<std::size_t>(n) * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[static_cast<std::size_t>(e)].push_back(n);
            r.weights[static_cast<std::size_t>(e)].push_back(
                topk_w[static_cast<std::size_t>(n) * K + k]);
        }
    }
    return r;
}

}  // namespace

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
    // V4: weights are replicated (not TP-sharded), so each rank uses full heads.
    const int num_heads = cfg.num_attention_heads;
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

    ws.attn_lse      = DeviceTensor::allocate(DType::FP32, {N, num_heads});

    // Compressed attention workspace.
    // coff: C4 has coff=2 (overlap in original), C128 has coff=1.
    // The compressor weights project to coff*head_dim, so we need that much space.
    int coff_max = 0;  // max coff across all compressed layers
    int min_ratio = 0;
    if (!cfg.dsv4_compress_ratios.empty()) {
        for (int r : cfg.dsv4_compress_ratios) {
            if (r > 0) {
                if (min_ratio == 0 || r < min_ratio) min_ratio = r;
                const int coff = (r == 4) ? 2 : 1;
                if (coff > coff_max) coff_max = coff;
            }
        }
    }
    const int max_comp_tokens = min_ratio > 0 ? (N / min_ratio + 1) : 0;
    if (max_comp_tokens > 0 && coff_max > 0) {
        ws.comp_kv_proj    = DeviceTensor::allocate(DType::BF16, {N, coff_max * head_dim});
        ws.comp_score_proj = DeviceTensor::allocate(DType::BF16, {N, coff_max * head_dim});
        ws.comp_kv         = DeviceTensor::allocate(DType::BF16, {max_comp_tokens, head_dim});
        ws.comp_attn_out   = DeviceTensor::allocate(DType::BF16, {N, num_heads * head_dim});
        ws.comp_attn_lse   = DeviceTensor::allocate(DType::FP32, {N, num_heads});
    }

    // Routed experts: weights are sharded on intermediate dim → moe_I / T.
    const int local_moe_I_ws = moe_I / std::max(1, tp_size);
    ws.expert_in     = DeviceTensor::allocate(DType::BF16,{N, H});
    ws.expert_gate_w = DeviceTensor::allocate(DType::BF16,{local_moe_I_ws, H});
    ws.expert_up_w   = DeviceTensor::allocate(DType::BF16,{local_moe_I_ws, H});
    ws.expert_down_w = DeviceTensor::allocate(DType::BF16,{H, local_moe_I_ws});
    ws.expert_gate   = DeviceTensor::allocate(DType::BF16,{N, local_moe_I_ws});
    ws.expert_up     = DeviceTensor::allocate(DType::BF16,{N, local_moe_I_ws});
    ws.expert_out    = DeviceTensor::allocate(DType::BF16,{N, H});
    ws.route_idx     = DeviceTensor::allocate(DType::INT32, {N * K});
    ws.route_w       = DeviceTensor::allocate(DType::FP32, {N * K});

    const int O = max_logit_rows > 0 ? max_logit_rows : N;
    ws.logits = DeviceTensor::allocate(DType::BF16,{O, V});

    return ws;
}

std::size_t dsv4_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size) {
    ScopedDeviceAllocationCounter counter;
    {
        auto workspace = DsV4Workspace::allocate(
            cfg, max_tokens, max_logit_rows, tp_size);
    }
    return counter.allocated_bytes();
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
    // V4: weights are replicated, so each rank uses full heads.
    const int num_heads = cfg.num_attention_heads;
    const int head_dim = cfg.head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int qk_rope = cfg.qk_rope_head_dim;
    const int o_lora = cfg.dsv4_o_lora_rank;
    const int o_groups = cfg.dsv4_o_groups;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int moe_I = cfg.moe_intermediate_size;
    // Shared experts are sharded on intermediate dim (column/row parallel).
    const int local_shared_I = moe_I / T;
    // Routed experts are also sharded the same way.
    const int local_moe_I = moe_I / T;
    const int M = cfg.dsv4_hc_mult;
    const int mix_hc = (2 + M) * M;
    const float hc_eps = cfg.dsv4_hc_eps;
    const float hc_post_alpha = 2.0f;
    const int sinkhorn_iters = 20; // from config hc_sinkhorn_iters

    cudaStream_t stream = nullptr;  // default stream

    // TP all-reduce helper (safe: no-op when tp_comm is null)
    auto tp_all_reduce = [&](void* buf, std::size_t count) {
        if (fwd_cfg.tp_comm != nullptr) {
            fwd_cfg.tp_comm->all_reduce_bf16(
                buf, count, ncclSum, stream);
        }
    };

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
        invoke_stage_hook(
            StageHookPoint::OnAttnProj, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(num_heads * head_dim),
            static_cast<std::uint32_t>(li), stream);

        // Per-head RMSNorm on Q (no gamma weight) — reference: q *= rsqrt(...)
        kernels::launch_per_head_rmsnorm_bf16(
            ws.q.data(), N, num_heads, head_dim, eps, stream);

        // KV path: wkv → kv_norm
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.wkv, Lw.wkv_quant), ws.kv.data(),
            N, head_dim, H);

        kernels::launch_rmsnorm_bf16(
            ws.kv.data(), Lw.kv_norm->data(), ws.kv.data(),
            N, head_dim, eps, stream);

        // Partial RoPE on last qk_rope dims of Q and KV
        kernels::launch_rope_partial_last_bf16(
            ws.q.data(), ws.kv.data(),
            positions, N, num_heads, 1, head_dim, qk_rope,
            cfg.rope_theta, stream);
        {
            // SWA attention on ALL layers (every layer has a sliding window).
            auto lv = kv_cache.layer_view(li);
            kernels::launch_write_kv_to_pages_bf16(
                lv.k_pages, lv.v_pages,
                ws.kv.data(), ws.kv.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, num_requests, lv.page_size,
                1, head_dim, false, stream);

            ops::launch_attention_naive_paged_bf16(
                ws.q.data(), lv.k_pages, lv.v_pages,
                ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, num_requests, num_heads, 1, head_dim, lv.page_size, stream,
                /*window_left=*/-1, /*sm_scale=*/-1.f,
                /*logits_soft_cap=*/0.f,
                /*lse_out=*/static_cast<float*>(ws.attn_lse.data()));

            // ── Compressed attention (C4/C128 layers, prefill only) ──
            const int comp_ratio = Lw.compress_ratio;
            if (comp_ratio > 0 && !is_pure_decode &&
                Lw.compressor.wkv != nullptr &&
                Lw.compressor.wgate != nullptr &&
                !ws.comp_kv.empty())
            {
                const bool is_c4 = (comp_ratio == 4);
                const int coff = is_c4 ? 2 : 1;
                const int proj_dim = coff * head_dim;  // output dim of compressor GEMMs

                // Step 1: Project hidden states through compressor.wkv and wgate
                // norm_x is [N, H], compressor.wkv is [proj_dim, H]
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.compressor.wkv->data(),
                    ws.comp_kv_proj.data(),
                    N, proj_dim, H);

                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.compressor.wgate->data(),
                    ws.comp_score_proj.data(),
                    N, proj_dim, H);

                // Step 2: For C4 (coff=2) without overlap simplification,
                // we use only the second half of the projected dims (the "current
                // window" part). For C128 (coff=1), we use all dims.
                //
                // We need to handle this per-request: each request has its own
                // token range and produces its own compressed entries.
                //
                // For simplicity in this first pass, process per-request on host:
                // download proj data, do gated pooling + APE, upload result.
                //
                // Per-request compressed entry counts:
                // Request r has tokens in [qo_indptr[r], qo_indptr[r+1]).
                // Number of compressed entries = floor(num_tokens_r / comp_ratio).

                // Copy host-side qo_indptr
                std::vector<std::uint32_t> qo_h(static_cast<std::size_t>(num_requests) + 1);
                std::memcpy(qo_h.data(), qo_indptr_h,
                    qo_h.size() * sizeof(std::uint32_t));

                // Download projected kv and score to host
                const std::size_t proj_bytes = static_cast<std::size_t>(N) * proj_dim * 2;
                std::vector<std::uint16_t> kv_proj_h(static_cast<std::size_t>(N) * proj_dim);
                std::vector<std::uint16_t> score_proj_h(static_cast<std::size_t>(N) * proj_dim);
                CUDA_CHECK(cudaMemcpyAsync(kv_proj_h.data(), ws.comp_kv_proj.data(),
                    proj_bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(score_proj_h.data(), ws.comp_score_proj.data(),
                    proj_bytes, cudaMemcpyDeviceToHost, stream));

                // Download APE to host
                std::vector<float> ape_h;
                if (Lw.compressor.ape != nullptr) {
                    const std::size_t ape_numel =
                        static_cast<std::size_t>(comp_ratio) * proj_dim;
                    ape_h.resize(ape_numel);
                    CUDA_CHECK(cudaMemcpyAsync(ape_h.data(), Lw.compressor.ape->data(),
                        ape_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
                }

                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Compute compressed KV on host
                // For C4: offset into second half of proj_dim (skip first head_dim dims)
                const int dim_offset = is_c4 ? head_dim : 0;

                // Build compressed entries for all requests
                int total_comp = 0;
                std::vector<int> comp_offsets_h(static_cast<std::size_t>(num_requests));
                std::vector<int> comp_lens_h(static_cast<std::size_t>(num_requests));
                std::vector<int> comp_ratios_h(static_cast<std::size_t>(num_requests), comp_ratio);

                for (int r = 0; r < num_requests; ++r) {
                    const int tok_lo = static_cast<int>(qo_h[r]);
                    const int tok_hi = static_cast<int>(qo_h[r + 1]);
                    const int n_tok = tok_hi - tok_lo;
                    const int n_comp = n_tok / comp_ratio;
                    comp_offsets_h[r] = total_comp;
                    comp_lens_h[r] = n_comp;
                    total_comp += n_comp;
                }

                if (total_comp > 0) {
                    // Allocate host buffer for compressed KV
                    // Each compressed entry has head_dim bf16 elements
                    std::vector<std::uint16_t> comp_kv_h(
                        static_cast<std::size_t>(total_comp) * head_dim, 0);

                    // Helper: convert bf16 bits to float
                    auto bf16_to_f32 = [](std::uint16_t bits) -> float {
                        union { std::uint32_t u; float f; } tmp;
                        tmp.u = static_cast<std::uint32_t>(bits) << 16;
                        return tmp.f;
                    };
                    auto f32_to_bf16 = [](float val) -> std::uint16_t {
                        union { float f; std::uint32_t u; } tmp;
                        tmp.f = val;
                        // Round to nearest even (add 0x7fff + bit[16])
                        std::uint32_t rounding_bias = ((tmp.u >> 16) & 1) + 0x7FFF;
                        return static_cast<std::uint16_t>((tmp.u + rounding_bias) >> 16);
                    };

                    for (int r = 0; r < num_requests; ++r) {
                        const int tok_lo = static_cast<int>(qo_h[r]);
                        const int n_comp = comp_lens_h[r];
                        const int comp_off = comp_offsets_h[r];

                        for (int g = 0; g < n_comp; ++g) {
                            // Group g covers tokens [tok_lo + g*ratio, tok_lo + (g+1)*ratio)
                            const int base_tok = tok_lo + g * comp_ratio;

                            for (int d = 0; d < head_dim; ++d) {
                                // Softmax over ratio scores at this dim
                                const int proj_d = dim_offset + d;

                                // Find max score for numerical stability
                                float max_s = -std::numeric_limits<float>::infinity();
                                for (int i = 0; i < comp_ratio; ++i) {
                                    const std::size_t idx =
                                        static_cast<std::size_t>(base_tok + i) * proj_dim + proj_d;
                                    float s = bf16_to_f32(score_proj_h[idx]);
                                    // Add APE
                                    if (!ape_h.empty()) {
                                        s += ape_h[static_cast<std::size_t>(i) * proj_dim + proj_d];
                                    }
                                    if (s > max_s) max_s = s;
                                }

                                // Compute softmax-weighted sum
                                float sum_exp = 0.f;
                                float weighted_sum = 0.f;
                                for (int i = 0; i < comp_ratio; ++i) {
                                    const std::size_t idx =
                                        static_cast<std::size_t>(base_tok + i) * proj_dim + proj_d;
                                    float s = bf16_to_f32(score_proj_h[idx]);
                                    if (!ape_h.empty()) {
                                        s += ape_h[static_cast<std::size_t>(i) * proj_dim + proj_d];
                                    }
                                    const float e = std::exp(s - max_s);
                                    sum_exp += e;

                                    const float v = bf16_to_f32(kv_proj_h[idx]);
                                    weighted_sum += v * e;
                                }

                                const float result = sum_exp > 0.f ? weighted_sum / sum_exp : 0.f;
                                comp_kv_h[static_cast<std::size_t>(comp_off + g) * head_dim + d] =
                                    f32_to_bf16(result);
                            }
                        }
                    }

                    // Upload compressed KV to device
                    CUDA_CHECK(cudaMemcpyAsync(ws.comp_kv.data(), comp_kv_h.data(),
                        static_cast<std::size_t>(total_comp) * head_dim * 2,
                        cudaMemcpyHostToDevice, stream));

                    // Step 3: RMSNorm with compressor.norm weight
                    if (Lw.compressor.norm != nullptr) {
                        kernels::launch_rmsnorm_bf16(
                            ws.comp_kv.data(), Lw.compressor.norm->data(),
                            ws.comp_kv.data(),
                            total_comp, head_dim, eps, stream);
                    }

                    // Step 4: RoPE on last qk_rope dims of compressed KV.
                    // Build position indices for compressed RoPE.
                    // Reference uses freqs_cis[:cutoff:ratio] which gives positions
                    // 0, ratio, 2*ratio, ... within each sequence. For prefill
                    // (start_pos=0), group g gets position g*ratio.
                    // TODO: For continuation prefills, read actual positions from
                    // the device `positions` array.
                    std::vector<std::int32_t> comp_positions_h(static_cast<std::size_t>(total_comp));
                    for (int r = 0; r < num_requests; ++r) {
                        const int comp_off_r = comp_offsets_h[r];
                        const int n_comp = comp_lens_h[r];
                        for (int g = 0; g < n_comp; ++g) {
                            // Position = g * ratio within each request's sequence
                            comp_positions_h[comp_off_r + g] =
                                static_cast<std::int32_t>(g * comp_ratio);
                        }
                    }

                    // Upload positions and apply RoPE
                    // Reuse route_idx device buffer for positions (it's [N*K] int32, large enough)
                    CUDA_CHECK(cudaMemcpyAsync(ws.route_idx.data(), comp_positions_h.data(),
                        static_cast<std::size_t>(total_comp) * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream));

                    // Apply RoPE to compressed KV (single head, partial last dims)
                    // Use compress_rope_theta if available, else base theta
                    const float comp_theta =
                        cfg.dsv4_compress_rope_theta > 0.f
                            ? cfg.dsv4_compress_rope_theta
                            : cfg.rope_theta;
                    kernels::launch_rope_partial_last_bf16(
                        ws.comp_kv.data(),  // "q" — will be rotated
                        ws.comp_kv.data(),  // "k" — dummy (same buffer, 0 heads below)
                        static_cast<const std::int32_t*>(ws.route_idx.data()),
                        total_comp,
                        1,      // num_q_heads: treat as single-head to rotate just the KV
                        0,      // num_kv_heads: 0 means skip K rotation
                        head_dim,
                        qk_rope,
                        comp_theta,
                        stream);

                    // Step 5: Dense attention over compressed KV entries
                    // Build host-side qo_indptr as int for the compressed attention kernel
                    std::vector<int> qo_int(static_cast<std::size_t>(num_requests) + 1);
                    for (int i = 0; i <= num_requests; ++i) {
                        qo_int[i] = static_cast<int>(qo_h[i]);
                    }

                    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                    kernels::launch_attention_compressed_bf16(
                        ws.q.data(),
                        ws.comp_kv.data(),
                        ws.comp_attn_out.data(),
                        static_cast<float*>(ws.comp_attn_lse.data()),
                        qo_int.data(),
                        comp_offsets_h.data(),
                        comp_lens_h.data(),
                        comp_ratios_h.data(),
                        N, num_requests, num_heads, head_dim,
                        sm_scale, stream);

                    // Step 6: Combine SWA and compressed attention outputs
                    kernels::launch_combine_attn_outputs_bf16(
                        ws.attn_out.data(),
                        static_cast<const float*>(ws.attn_lse.data()),
                        ws.comp_attn_out.data(),
                        static_cast<const float*>(ws.comp_attn_lse.data()),
                        ws.attn_out.data(),   // write combined result back to attn_out
                        static_cast<float*>(ws.attn_lse.data()),  // update LSE
                        N, num_heads, head_dim, stream);
                }
            }

            // Attention sink correction: o *= sigmoid(lse - sink_h)
            if (Lw.attn_sink != nullptr) {
                kernels::launch_attn_sink_correction_bf16(
                    ws.attn_out.data(),
                    static_cast<const float*>(ws.attn_lse.data()),
                    static_cast<const float*>(Lw.attn_sink->data()),
                    N, num_heads, head_dim, stream);
            }
            invoke_stage_hook(
                StageHookPoint::OnAttn, ws.q.data(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(num_heads * head_dim),
                static_cast<std::uint32_t>(li), stream);

            // Inverse RoPE on attention output (removes position info before projection)
            kernels::launch_rope_partial_last_bf16(
                ws.attn_out.data(), ws.attn_out.data(),
                positions, N, num_heads, 0, head_dim, qk_rope,
                cfg.rope_theta, stream, /*inverse=*/true);

            // Grouped output projection: per-group GEMMs with strided I/O.
            //
            // attn_out is [N, num_heads*head_dim] BF16.  Conceptually
            // reshaped to [N, local_groups, per_group_dim].  wo_a is
            // [o_groups*o_lora, per_group_dim] (FP8 or BF16); the local
            // slice for TP rank covers rows [gg_start*o_lora,
            // (gg_start+local_groups)*o_lora).  For each group g we
            // compute:
            //   wo_a_out[:, g*o_lora:(g+1)*o_lora] =
            //       attn_out[:, g*pgd:(g+1)*pgd] @ wo_a_g^T
            // using cublasGemmEx with custom leading dimensions so the
            // strided slices are accessed in-place without copies.
            // V4: weights replicated, so each rank handles all o_groups locally.
            const int local_groups = o_groups;
            const int out_dim = local_groups * o_lora;
            const int gg_start = 0;
            const int K_wo_a = static_cast<int>(Lw.wo_a->shape().back());
            const int per_group_dim = K_wo_a;  // alias for clarity
            const int attn_stride = num_heads * head_dim;  // full row width

            const bool wo_a_is_fp8 =
                Lw.wo_a->dtype() == DType::FP8_E4M3;
            const int elem_bytes_wo_a = wo_a_is_fp8 ? 1 : 2;

            // Workspace for dequantised group slice when FP8.
            // Reuse expert_gate_w ([moe_I, H] BF16) which is large
            // enough for one [o_lora, per_group_dim] BF16 slice.
            void* dequant_buf = ws.expert_gate_w.data();

            cublasSetStream(cublas.handle(), stream);

            for (int g = 0; g < local_groups; ++g) {
                const int global_g = gg_start + g;

                // --- Weight pointer for this group: [o_lora, pgd] ---
                const char* wo_a_g_raw = static_cast<const char*>(
                    Lw.wo_a->data()) +
                    static_cast<std::size_t>(global_g) * o_lora *
                    per_group_dim * elem_bytes_wo_a;

                // Pointer to BF16 weight used by cuBLAS (either the
                // original tensor or the dequant buffer).
                const void* wo_a_g_bf16 = nullptr;

                if (wo_a_is_fp8) {
                    // Dequant [o_lora, pgd] FP8 -> BF16 into workspace.
                    const auto* fp8_ptr =
                        reinterpret_cast<const std::uint8_t*>(wo_a_g_raw);

                    if (Lw.wo_a_quant.has_value() &&
                        Lw.wo_a_quant->kind == QuantMeta::Kind::PerGroup) {
                        const auto& qm = *Lw.wo_a_quant;
                        const int gs = qm.group_size > 0
                            ? qm.group_size : 128;
                        // Scale tensor layout: [ceil(total_rows/gs),
                        //   ceil(pgd/gs)].  The local group g starts at
                        //   row = global_g * o_lora in the full weight.
                        const int sc_cols =
                            (per_group_dim + gs - 1) / gs;
                        const int sc_row_start =
                            (global_g * o_lora) / gs;
                        const std::size_t scale_off =
                            static_cast<std::size_t>(sc_row_start) *
                            sc_cols * sizeof(float);
                        const auto* sc_ptr =
                            static_cast<const float*>(static_cast<const void*>(
                                static_cast<const char*>(
                                    qm.scale->data()) + scale_off));

                        kernels::launch_dequant_fp8_e4m3_to_bf16_per_group(
                            fp8_ptr, dequant_buf, sc_ptr,
                            o_lora, per_group_dim, gs, stream);
                    } else if (Lw.wo_a_quant.has_value() &&
                               Lw.wo_a_quant->kind ==
                                   QuantMeta::Kind::PerChannel) {
                        const auto& qm = *Lw.wo_a_quant;
                        const auto* sc_ptr =
                            static_cast<const float*>(
                                qm.scale->data()) + global_g * o_lora;
                        kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
                            fp8_ptr, dequant_buf, sc_ptr,
                            o_lora, per_group_dim, stream);
                    } else {
                        // PerTensor or no quant metadata — use scale 1.
                        float scale_val = 1.0f;
                        if (Lw.wo_a_quant.has_value() &&
                            Lw.wo_a_quant->scale != nullptr) {
                            // Single FP32 scalar on device — copy it.
                            CUDA_CHECK(cudaMemcpyAsync(
                                &scale_val,
                                Lw.wo_a_quant->scale->data(),
                                sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                        kernels::launch_dequant_fp8_e4m3_to_bf16(
                            fp8_ptr, dequant_buf, scale_val,
                            static_cast<std::size_t>(o_lora) *
                                per_group_dim,
                            stream);
                    }
                    wo_a_g_bf16 = dequant_buf;
                } else {
                    // Weight is already BF16 — use directly.
                    wo_a_g_bf16 = wo_a_g_raw;
                }

                // --- Strided GEMM via cublasGemmEx ---
                // Row-major: C[N, o_lora] = A[N, pgd] @ W[o_lora, pgd]^T
                // Col-major equiv:
                //   C'[o_lora, N] = W'[pgd, o_lora]^T_T * A'[pgd, N]
                //                 = W (CUBLAS_OP_T) * A (CUBLAS_OP_N)
                //
                // A = wo_a_g: [o_lora, pgd] row-major = [pgd, o_lora]
                //     col-major, lda = pgd.
                // B = attn_out group slice: starts at column g*pgd in the
                //     [N, attn_stride] row-major layout.  Col-major view
                //     is [attn_stride, N]; group slice at row offset
                //     g*pgd gives ldb = attn_stride.
                // C = wo_a_out group slice: column g*o_lora in
                //     [N, out_dim] row-major.  Col-major view is
                //     [out_dim, N]; group slice at row offset g*o_lora
                //     gives ldc = out_dim.
                const float alpha = 1.0f, beta = 0.0f;

                const void* b_ptr = static_cast<const char*>(
                    ws.attn_out.data()) +
                    static_cast<std::size_t>(g) * per_group_dim * 2;
                void* c_ptr = static_cast<char*>(
                    ws.wo_a_out.data()) +
                    static_cast<std::size_t>(g) * o_lora * 2;

                cublasGemmEx(cublas.handle(),
                    CUBLAS_OP_T,       // opA: transpose W
                    CUBLAS_OP_N,       // opB: no transpose on act
                    o_lora,            // m = output rows (col-major)
                    N,                 // n = batch
                    per_group_dim,     // k = contraction dim
                    &alpha,
                    wo_a_g_bf16,       // A = W [pgd, o_lora] col-major
                    CUDA_R_16BF,
                    per_group_dim,     // lda = pgd
                    b_ptr,             // B = act group slice
                    CUDA_R_16BF,
                    attn_stride,       // ldb = full row width
                    &beta,
                    c_ptr,             // C = output group slice
                    CUDA_R_16BF,
                    out_dim,           // ldc = full output width
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT);
            }

            ops::gemm_act_x_w(cublas.handle(),
                ws.wo_a_out.data(),
                make_weight_view(Lw.wo_b, Lw.wo_b_quant),
                ws.y.data(),
                N, H, out_dim);

            // TODO: all-reduce wo_b output across TP ranks
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

        // Routed expert dispatch (MXFP4 weights, dequant to bf16)
        cudaMemsetAsync(ws.moe_out.data(), 0,
                        static_cast<std::size_t>(N) * H * 2, stream);
        if (!Lw.experts.empty() && Lw.experts[0].w1 != nullptr) {
            std::vector<std::int32_t> topk_idx_h(
                static_cast<std::size_t>(N) * K);
            std::vector<float> topk_w_h(static_cast<std::size_t>(N) * K);
            CUDA_CHECK(cudaMemcpyAsync(
                topk_idx_h.data(), ws.topk_idx.data(),
                topk_idx_h.size() * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                topk_w_h.data(), ws.topk_weights.data(),
                topk_w_h.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
            for (int e = 0; e < E; ++e) {
                const auto& tok_idx = routing.token_idx[static_cast<std::size_t>(e)];
                const int Ne = static_cast<int>(tok_idx.size());
                if (Ne == 0) continue;
                const auto& ew = Lw.experts[static_cast<std::size_t>(e)];
                if (!ew.w1 || !ew.w1_scale) continue;
                const auto& wts = routing.weights[static_cast<std::size_t>(e)];

                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(ew.w1->data()),
                    static_cast<const std::uint8_t*>(ew.w1_scale->data()),
                    ws.expert_gate_w.data(), local_moe_I, H, stream);
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(ew.w3->data()),
                    static_cast<const std::uint8_t*>(ew.w3_scale->data()),
                    ws.expert_up_w.data(), local_moe_I, H, stream);
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(ew.w2->data()),
                    static_cast<const std::uint8_t*>(ew.w2_scale->data()),
                    ws.expert_down_w.data(), H, local_moe_I, stream);

                CUDA_CHECK(cudaMemcpyAsync(
                    ws.route_idx.data(), tok_idx.data(),
                    static_cast<std::size_t>(Ne) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    ws.route_w.data(), wts.data(),
                    static_cast<std::size_t>(Ne) * sizeof(float),
                    cudaMemcpyHostToDevice, stream));

                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.norm_y.data()),
                    static_cast<const std::int32_t*>(ws.route_idx.data()),
                    static_cast<std::uint16_t*>(ws.expert_in.data()),
                    Ne, H, stream);

                ops::gemm_act_x_w(cublas.handle(),
                    ws.expert_in.data(),
                    ops::WeightView::raw(ws.expert_gate_w.data(), DType::BF16),
                    ws.expert_gate.data(), Ne, local_moe_I, H);
                ops::gemm_act_x_w(cublas.handle(),
                    ws.expert_in.data(),
                    ops::WeightView::raw(ws.expert_up_w.data(), DType::BF16),
                    ws.expert_up.data(), Ne, local_moe_I, H);
                kernels::launch_swiglu_bf16(
                    ws.expert_gate.data(), ws.expert_up.data(),
                    ws.expert_gate.data(),
                    static_cast<std::size_t>(Ne) * local_moe_I, stream);
                ops::gemm_act_x_w(cublas.handle(),
                    ws.expert_gate.data(),
                    ops::WeightView::raw(ws.expert_down_w.data(), DType::BF16),
                    ws.expert_out.data(), Ne, H, local_moe_I);

                kernels::launch_scatter_add_weighted_bf16(
                    ws.moe_out.data(), ws.expert_out.data(),
                    static_cast<const std::int32_t*>(ws.route_idx.data()),
                    static_cast<const float*>(ws.route_w.data()),
                    Ne, H, stream);
            }
        }

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

        // All-reduce moe_out across TP ranks (routed + shared experts).
        if (fwd_cfg.tp_comm != nullptr) {
            fwd_cfg.tp_comm->all_reduce_bf16(
                ws.moe_out.data(),
                static_cast<std::size_t>(N) * H,
                ncclSum, stream);
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
            N, M, H, stream, hc_eps);
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
