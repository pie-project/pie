#include "model/qwen3_5_moe_forward.hpp"

#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

Qwen3_5MoeMlpWorkspace Qwen3_5MoeMlpWorkspace::allocate(
    int max_tokens, int hidden, int num_experts, int top_k,
    int moe_intermediate, int shared_intermediate)
{
    Qwen3_5MoeMlpWorkspace ws;
    const std::size_t N    = static_cast<std::size_t>(max_tokens);
    const std::size_t maxR = N * top_k;            // worst-case routes
    const std::size_t H    = static_cast<std::size_t>(hidden);
    const std::size_t I    = static_cast<std::size_t>(moe_intermediate);
    const std::size_t Ish  = static_cast<std::size_t>(shared_intermediate);

    ws.router_logits = DeviceBuffer<std::uint16_t>::alloc(N * num_experts);
    ws.topk_idx      = DeviceBuffer<std::int32_t>::alloc(N * top_k);
    ws.topk_weights  = DeviceBuffer<float>::alloc(N * top_k);

    ws.expert_in      = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_gate_up = DeviceBuffer<std::uint16_t>::alloc(maxR * 2 * I);
    ws.expert_act     = DeviceBuffer<std::uint16_t>::alloc(maxR * I);
    ws.expert_out     = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_idx     = DeviceBuffer<std::int32_t>::alloc(maxR);
    ws.expert_w       = DeviceBuffer<float>::alloc(maxR);

    ws.shared_gate       = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_up         = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_act        = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_out        = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.shared_gate_logit = DeviceBuffer<std::uint16_t>::alloc(N * 1);

    ws.moe_out = DeviceBuffer<std::uint16_t>::alloc(N * H);
    return ws;
}

namespace {

// `linear_attn_body` and `full_attn_body` below are near-clones of the
// helpers in `qwen3_5_forward.cpp`. The only difference is the
// per-layer-weights type they consume (`Qwen3_5MoeLayerWeights` vs
// `Qwen3_5LayerWeights`). De-duplicating via a template would require
// hoisting the helpers out of the anonymous namespace and parameter-
// izing on the layer struct; that's a defensible refactor for later
// but we keep the small amount of copied code local to each arch
// while the schemas may still drift.

// Build per-expert routing lists from device-side topk decisions.
struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>>        weights;
};
ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx_h,
    const std::vector<float>& topk_w_h,
    int N, int K, int E)
{
    ExpertRouting r;
    r.token_idx.assign(E, {});
    r.weights.assign(E, {});
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx_h[n * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[e].push_back(n);
            r.weights[e].push_back(topk_w_h[n * K + k]);
        }
    }
    return r;
}

// Linear-attn body (replica of qwen3_5_forward.cpp's logic, against
// MoeLayerWeights). Reads `ws.norm_x`, writes contribution into
// `ws.norm_y`.
void linear_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    Qwen3_5StateCache& state_cache,
    int layer_idx, int N, bool is_pure_decode,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int H        = cfg.hidden_size;
    const int K_h      = cfg.linear_num_key_heads;
    const int V_h      = cfg.linear_num_value_heads;
    const int K_d      = cfg.linear_key_head_dim;
    const int V_d      = cfg.linear_value_head_dim;
    const int K_dim    = K_h * K_d;
    const int V_dim    = V_h * V_d;
    const int conv_dim = 2 * K_dim + V_dim;
    const int conv_K   = cfg.linear_conv_kernel_dim;

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_qkv->data(),
        la.mixed_qkv.data(), N, conv_dim, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_z->data(),
        la.z.data(), N, V_dim, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_a->data(),
        la.a.data(), N, V_h, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_b->data(),
        la.b.data(), N, V_h, H);

    if (is_pure_decode) {
        kernels::launch_causal_conv1d_update_bf16(
            la.mixed_qkv.data(), Lw.la_conv1d_w->data(),
            Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
            state_cache.conv_state(layer_idx), la.mixed_qkv_post.data(),
            conv_dim, conv_K, stream);
    } else {
        kernels::launch_causal_conv1d_prefill_bf16(
            la.mixed_qkv.data(), Lw.la_conv1d_w->data(),
            Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
            la.mixed_qkv_post.data(), state_cache.conv_state(layer_idx),
            N, conv_dim, conv_K, stream);
    }

    auto* qkv_base = la.mixed_qkv_post.data();
    const std::size_t bf16 = sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(
        la.q_raw.data(), K_dim * bf16,
        qkv_base, conv_dim * bf16,
        K_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        la.k_raw.data(), K_dim * bf16,
        qkv_base + K_dim, conv_dim * bf16,
        K_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        la.v_raw.data(), V_dim * bf16,
        qkv_base + 2 * K_dim, conv_dim * bf16,
        V_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));

    const float scale = 1.f / std::sqrt(static_cast<float>(K_d));
    kernels::launch_l2norm_scale_bf16_to_fp32(
        la.q_raw.data(), la.q_pre.data(), N * K_h, K_d, scale, /*eps=*/1e-6f, stream);
    kernels::launch_l2norm_scale_bf16_to_fp32(
        la.k_raw.data(), la.k_pre.data(), N * K_h, K_d, /*scale=*/1.f, /*eps=*/1e-6f, stream);

    if (V_h != K_h) {
        kernels::launch_repeat_interleave_heads_fp32(
            la.q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d, stream);
        kernels::launch_repeat_interleave_heads_fp32(
            la.k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            la.q_norm.data(), la.q_pre.data(),
            (std::size_t)N * K_h * K_d * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            la.k_norm.data(), la.k_pre.data(),
            (std::size_t)N * K_h * K_d * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    kernels::launch_bf16_to_fp32(
        la.v_raw.data(), la.v_fp32.data(),
        (std::size_t)N * V_dim, stream);

    kernels::launch_gated_delta_g_beta(
        la.a.data(), la.b.data(),
        Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
        la.g_log.data(), la.beta.data(),
        N, V_h, stream);

    if (is_pure_decode) {
        kernels::launch_recurrent_gated_delta_step(
            la.q_norm.data(), la.k_norm.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            state_cache.recurrent_state(layer_idx),
            la.core_out.data(),
            /*B=*/1, V_h, K_d, V_d, stream);
    } else {
        kernels::launch_chunk_gated_delta_prefill(
            la.q_norm.data(), la.k_norm.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            state_cache.recurrent_state(layer_idx),
            la.core_out.data(),
            N, V_h, K_d, V_d, /*chunk_size=*/64, stream);
    }

    kernels::launch_fp32_to_bf16(
        la.core_out.data(), la.core_out_bf16.data(),
        (std::size_t)N * V_dim, stream);
    kernels::launch_rmsnorm_gated_bf16(
        la.core_out_bf16.data(), la.z.data(), Lw.la_norm_w_fp32,
        la.core_out_bf16.data(),
        N * V_h, V_d, /*eps=*/cfg.rms_norm_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        la.core_out_bf16.data(), Lw.la_out_proj->data(),
        ws.norm_y.data(), N, H, V_dim);
}

// Full-attention body (replica of qwen3_5_forward.cpp's logic).
void full_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache, AttentionWorkspace& attn_ws,
    int kv_layer, int N, int R, bool is_pure_decode,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int H  = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int d  = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_q_proj->data(),
        la.fa_qg_packed.data(), N, 2 * Hq, H);
    kernels::launch_split_q_gate_bf16(
        la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
        N, cfg.num_attention_heads, d, stream);

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_k_proj->data(),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_v_proj->data(),
        ws.v.data(), N, Hk, H);

    kernels::launch_rmsnorm_gemma_bf16(
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * cfg.num_attention_heads, d, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * cfg.num_key_value_heads, d, eps, stream);

    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, cfg.num_attention_heads, cfg.num_key_value_heads,
        d, rotary_dim, cfg.rope_theta, stream);

    kernels::launch_write_kv_to_pages_bf16(
        cache.k(kv_layer), cache.v(kv_layer), ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, cache.page_size(), cfg.num_key_value_heads, d, stream);

    // Same prefill-only path as qwen3_5_forward — see comment there.
    (void)is_pure_decode;
    ops::launch_attention_flashinfer_prefill_bf16(
        ws.q.data(), cache.k(kv_layer), cache.v(kv_layer), ws.attn_out.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        qo_indptr_h, kv_page_indptr_h,
        N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
        cache.page_size(), attn_ws, stream);

    kernels::launch_sigmoid_gate_inplace_bf16(
        ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.attn_out.data(), Lw.fa_o_proj->data(),
        ws.norm_y.data(), N, H, Hq);
}

// MoE block: routed experts + shared expert with sigmoid gate.
// Reads `ws.norm_x`, writes contribution into `moe_ws.moe_out`.
void moe_block(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int H = cfg.hidden_size;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int Im = cfg.moe_intermediate_size;
    const int Is = cfg.shared_expert_intermediate_size;

    // ── Routed experts ────────────────────────────────────────────
    // 1. Router logits.
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.moe_router->data(),
        moe_ws.router_logits.data(), N, E, H);
    // 2. Top-K + softmax + renormalize.
    kernels::launch_topk_softmax_bf16(
        moe_ws.router_logits.data(),
        moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
        N, E, K, stream);

    // 3. D2H copy of routing.
    std::vector<std::int32_t> topk_idx_h((std::size_t)N * K);
    std::vector<float>        topk_w_h((std::size_t)N * K);
    CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), moe_ws.topk_idx.data(),
                               topk_idx_h.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), moe_ws.topk_weights.data(),
                               topk_w_h.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);

    // 4. Zero moe_out (we'll scatter-accumulate per-expert outputs).
    CUDA_CHECK(cudaMemsetAsync(moe_ws.moe_out.data(), 0,
        (std::size_t)N * H * sizeof(std::uint16_t), stream));

    // 5. Per-expert dispatch.
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert in gate_up_proj
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;       // bf16 elements per expert in down_proj
    for (int e = 0; e < E; ++e) {
        const auto& tok_idx = routing.token_idx[e];
        const auto& wts     = routing.weights[e];
        const int Ne = static_cast<int>(tok_idx.size());
        if (Ne == 0) continue;

        CUDA_CHECK(cudaMemcpyAsync(
            moe_ws.expert_idx.data(), tok_idx.data(),
            Ne * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            moe_ws.expert_w.data(), wts.data(),
            Ne * sizeof(float), cudaMemcpyHostToDevice, stream));

        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.norm_x.data()),
            moe_ws.expert_idx.data(),
            moe_ws.expert_in.data(),
            Ne, H, stream);

        // gate_up_proj[e]: pointer offset into the fused [E, 2*Im, H] tensor.
        auto* gate_up_w = static_cast<const std::uint16_t*>(
                              Lw.moe_gate_up_proj->data())
                          + e * expert_stride_gu;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            moe_ws.expert_in.data(), gate_up_w,
            moe_ws.expert_gate_up.data(), Ne, 2 * Im, H);

        // Chunked SwiGLU: silu(gate) * up.
        kernels::launch_chunked_swiglu_bf16(
            moe_ws.expert_gate_up.data(),
            moe_ws.expert_act.data(),
            Ne, Im, stream);

        // down_proj[e]: offset into [E, H, Im] tensor.
        auto* down_w = static_cast<const std::uint16_t*>(
                           Lw.moe_down_proj->data())
                       + e * expert_stride_dn;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            moe_ws.expert_act.data(), down_w,
            moe_ws.expert_out.data(), Ne, H, Im);

        kernels::launch_scatter_add_weighted_bf16(
            moe_ws.moe_out.data(), moe_ws.expert_out.data(),
            moe_ws.expert_idx.data(), moe_ws.expert_w.data(),
            Ne, H, stream);
    }

    // ── Shared expert (always-on, dense MLP + sigmoid gate) ───────
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.shared_gate_proj->data(),
        moe_ws.shared_gate.data(), N, Is, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.shared_up_proj->data(),
        moe_ws.shared_up.data(), N, Is, H);
    kernels::launch_swiglu_bf16(
        moe_ws.shared_gate.data(), moe_ws.shared_up.data(),
        moe_ws.shared_act.data(),
        N * Is, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        moe_ws.shared_act.data(), Lw.shared_down_proj->data(),
        moe_ws.shared_out.data(), N, H, Is);

    // shared_gate logit [N, 1] = norm_x @ shared_gate.weight.T
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.shared_gate->data(),
        moe_ws.shared_gate_logit.data(), N, 1, H);

    // shared_out *= sigmoid(scalar_gate[n]) per token, broadcast across
    // all H channels.
    kernels::launch_sigmoid_scalar_gate_inplace_bf16(
        moe_ws.shared_out.data(), moe_ws.shared_gate_logit.data(),
        N, H, stream);

    // moe_out += shared_out (residual-add style).
    kernels::launch_residual_add_bf16(
        moe_ws.moe_out.data(), moe_ws.shared_out.data(),
        (std::size_t)N * H, stream);

    // Copy moe_out → ws.norm_y so the caller can do `y += norm_y`.
    CUDA_CHECK(cudaMemcpyAsync(
        ws.norm_y.data(), moe_ws.moe_out.data(),
        (std::size_t)N * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

void qwen3_5_moe_forward_paged(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    Qwen3_5StateCache& state_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens, int num_requests,
    bool is_pure_decode,
    const std::uint8_t* /*mask_d*/,
    const std::int32_t* /*mask_indptr_d*/)
{
    if (num_requests != 1) {
        throw std::runtime_error(
            "qwen3_5_moe_forward_paged: multi-request not supported yet");
    }
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    if (!is_pure_decode) state_cache.reset(stream);

    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];

        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);

        if (Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn) {
            linear_attn_body(
                Lw, cfg, ws, la_ws, state_cache,
                static_cast<int>(L), N, is_pure_decode, cublas, stream);
        } else {
            full_attn_body(
                Lw, cfg, ws, la_ws, cache, attn_ws, Lw.kv_layer,
                N, num_requests, is_pure_decode,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                cublas, stream);
        }
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            (std::size_t)N * H, stream);

        // Post-attention norm + MoE block + residual.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        moe_block(Lw, cfg, ws, moe_ws, N, cublas, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            (std::size_t)N * H, stream);
    }

    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(),
        ws.logits.data(), N, V, H);
}

}  // namespace pie_cuda_driver::model
