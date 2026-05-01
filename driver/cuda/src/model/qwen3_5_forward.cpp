#include "model/qwen3_5_forward.hpp"

#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

Qwen3_5LinearAttnWorkspace Qwen3_5LinearAttnWorkspace::allocate(
    int max_tokens, int conv_dim, int v_h, int k_d, int v_d)
{
    Qwen3_5LinearAttnWorkspace ws;
    const std::size_t N = static_cast<std::size_t>(max_tokens);
    const std::size_t v_dim = static_cast<std::size_t>(v_h) * v_d;
    ws.mixed_qkv      = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.mixed_qkv_post = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.z              = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    ws.a              = DeviceBuffer<std::uint16_t>::alloc(N * v_h);
    ws.b              = DeviceBuffer<std::uint16_t>::alloc(N * v_h);
    ws.q_norm   = DeviceBuffer<float>::alloc(N * (std::size_t)v_h * k_d);
    ws.k_norm   = DeviceBuffer<float>::alloc(N * (std::size_t)v_h * k_d);
    ws.v_fp32   = DeviceBuffer<float>::alloc(N * v_dim);
    ws.g_log    = DeviceBuffer<float>::alloc(N * v_h);
    ws.beta     = DeviceBuffer<float>::alloc(N * v_h);
    ws.core_out = DeviceBuffer<float>::alloc(N * v_dim);
    ws.core_out_bf16 = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    return ws;
}

namespace {

// Linear-attn layer body. Reads `ws.norm_x` (post-input-layernorm
// activations) and writes the layer's contribution into `ws.norm_y`
// (which the caller adds to `ws.y` as the residual).
void linear_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    Qwen3_5StateCache& state_cache,
    int layer_idx,
    int N,
    bool is_pure_decode,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
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

    // ── In-projections ────────────────────────────────────────────
    // mixed_qkv [N, conv_dim] = norm_x @ in_proj_qkv.T
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_qkv->data(),
        la.mixed_qkv.data(), N, conv_dim, H);
    // z [N, V_dim] = norm_x @ in_proj_z.T
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_z->data(),
        la.z.data(), N, V_dim, H);
    // a [N, V_h] = norm_x @ in_proj_a.T   (b symmetric)
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_a->data(),
        la.a.data(), N, V_h, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.la_in_proj_b->data(),
        la.b.data(), N, V_h, H);

    // ── Causal depthwise conv1d (kernel=K, fused silu) ────────────
    // Per-request conv_state lives in `state_cache.conv_state(layer)`.
    // Layout: conv_state is [conv_K, conv_dim] bf16.
    if (is_pure_decode) {
        // N == 1; single-token update.
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

    // ── Split mixed_qkv_post into q_raw, k_raw, v_raw ─────────────
    // mixed_qkv_post[N, conv_dim] packs per-token channels as
    // [q_raw(K_dim) | k_raw(K_dim) | v_raw(V_dim)]; l2norm wants a
    // contiguous [N*K_h, K_d] flat view, so copy each segment into its
    // own dense buffer via stride-aware memcpy2D.
    auto* qkv_base = la.mixed_qkv_post.data();
    DeviceBuffer<std::uint16_t> q_raw =
        DeviceBuffer<std::uint16_t>::alloc((std::size_t)N * K_dim);
    DeviceBuffer<std::uint16_t> k_raw =
        DeviceBuffer<std::uint16_t>::alloc((std::size_t)N * K_dim);
    DeviceBuffer<std::uint16_t> v_raw =
        DeviceBuffer<std::uint16_t>::alloc((std::size_t)N * V_dim);

    const std::size_t bf16 = sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpy2DAsync(
        q_raw.data(), K_dim * bf16,
        qkv_base, conv_dim * bf16,
        K_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        k_raw.data(), K_dim * bf16,
        qkv_base + K_dim, conv_dim * bf16,
        K_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(
        v_raw.data(), V_dim * bf16,
        qkv_base + 2 * K_dim, conv_dim * bf16,
        V_dim * bf16, N, cudaMemcpyDeviceToDevice, stream));

    // ── L2-normalise + scale q, l2-normalise k, widen v to fp32 ─────
    // q has [N, K_h, K_d] layout, l2norm operates on K_d. After
    // l2norm we'll repeat-interleave to V_h heads.
    //
    // Pre-repeat staging: [N, K_h, K_d] fp32.
    DeviceBuffer<float> q_pre = DeviceBuffer<float>::alloc((std::size_t)N * K_h * K_d);
    DeviceBuffer<float> k_pre = DeviceBuffer<float>::alloc((std::size_t)N * K_h * K_d);

    const float scale = 1.f / std::sqrt(static_cast<float>(K_d));
    kernels::launch_l2norm_scale_bf16_to_fp32(
        q_raw.data(), q_pre.data(),
        N * K_h, K_d, scale, /*eps=*/1e-6f, stream);
    kernels::launch_l2norm_scale_bf16_to_fp32(
        k_raw.data(), k_pre.data(),
        N * K_h, K_d, /*scale=*/1.f, /*eps=*/1e-6f, stream);

    // repeat_interleave from K_h → V_h heads.
    if (V_h != K_h) {
        kernels::launch_repeat_interleave_heads_fp32(
            q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d, stream);
        kernels::launch_repeat_interleave_heads_fp32(
            k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            la.q_norm.data(), q_pre.data(),
            (std::size_t)N * K_h * K_d * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            la.k_norm.data(), k_pre.data(),
            (std::size_t)N * K_h * K_d * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }

    // v: bf16 → fp32 (no l2norm).
    kernels::launch_bf16_to_fp32(
        v_raw.data(), la.v_fp32.data(),
        (std::size_t)N * V_dim, stream);

    // ── g, β per token per head ────────────────────────────────────
    kernels::launch_gated_delta_g_beta(
        la.a.data(), la.b.data(),
        Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
        la.g_log.data(), la.beta.data(),
        N, V_h, stream);

    // ── Recurrent update ───────────────────────────────────────────
    if (is_pure_decode) {
        kernels::launch_recurrent_gated_delta_step(
            la.q_norm.data(), la.k_norm.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            state_cache.recurrent_state(layer_idx),
            la.core_out.data(),
            /*B=*/1, V_h, K_d, V_d, stream);
    } else {
        // Sequential per-token loop (Phase-5 simplification).
        kernels::launch_chunk_gated_delta_prefill(
            la.q_norm.data(), la.k_norm.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            state_cache.recurrent_state(layer_idx),
            la.core_out.data(),
            N, V_h, K_d, V_d, /*chunk_size=*/64, stream);
    }

    // ── core_out → bf16, then RMSNormGated with z ──────────────────
    // core_out has [N, V_h, V_d] layout = [N, V_dim] flat. We want
    // RMSNormGated over V_d per (n, h). Treat as [N*V_h, V_d].
    kernels::launch_fp32_to_bf16(
        la.core_out.data(), la.core_out_bf16.data(),
        (std::size_t)N * V_dim, stream);

    // RMSNormGated: y = weight * x_hat * silu(gate) where weight is
    // per-V_d. z must align — it's [N, V_dim] in token-row layout, so
    // viewing it as [N*V_h, V_d] gives the correct broadcast.
    kernels::launch_rmsnorm_gated_bf16(
        la.core_out_bf16.data(), la.z.data(), Lw.la_norm_w_fp32,
        la.core_out_bf16.data(),
        N * V_h, V_d, /*eps=*/cfg.rms_norm_eps, stream);

    // ── out_proj: [N, V_dim] → [N, H], write into ws.norm_y ────────
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        la.core_out_bf16.data(), Lw.la_out_proj->data(),
        ws.norm_y.data(), N, H, V_dim);
}

// Full-attention layer body. Reads `ws.norm_x`, writes contribution
// to `ws.norm_y`. KV cache and flashinfer mirror the qwen3 path.
void full_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    int kv_layer,
    int N, int R,
    bool is_pure_decode,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    const int H  = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int d  = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;

    // ── q/k/v projections (q is 2× wide for the output gate) ──────
    // qg_packed [N, 2*Hq] = norm_x @ q_proj.T
    DeviceBuffer<std::uint16_t> qg_packed =
        DeviceBuffer<std::uint16_t>::alloc((std::size_t)N * 2 * Hq);
    DeviceBuffer<std::uint16_t> gate =
        DeviceBuffer<std::uint16_t>::alloc((std::size_t)N * Hq);

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_q_proj->data(),
        qg_packed.data(), N, 2 * Hq, H);
    kernels::launch_split_q_gate_bf16(
        qg_packed.data(), ws.q.data(), gate.data(),
        N, cfg.num_attention_heads, d, stream);

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_k_proj->data(),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_v_proj->data(),
        ws.v.data(), N, Hk, H);

    // ── q_norm / k_norm (gemma-style (1+w)·x_hat) ─────────────────
    kernels::launch_rmsnorm_gemma_bf16(
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * cfg.num_attention_heads, d, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * cfg.num_key_value_heads, d, eps, stream);

    // ── Partial RoPE ──────────────────────────────────────────────
    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, cfg.num_attention_heads, cfg.num_key_value_heads,
        d, rotary_dim, cfg.rope_theta, stream);

    // ── Write K/V to paged cache ──────────────────────────────────
    kernels::launch_write_kv_to_pages_bf16(
        cache.k(kv_layer), cache.v(kv_layer), ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, cache.page_size(), cfg.num_key_value_heads, d, stream);

    // ── Flashinfer attention ──────────────────────────────────────
    // Always route through the prefill kernel — it handles qo_len=1 (the
    // decode shape) correctly. Wiring the dedicated decode kernel + plan
    // is an optional perf win; correctness is identical.
    (void)is_pure_decode;
    ops::launch_attention_flashinfer_prefill_bf16(
        ws.q.data(), cache.k(kv_layer), cache.v(kv_layer), ws.attn_out.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        qo_indptr_h, kv_page_indptr_h,
        N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
        cache.page_size(), attn_ws, stream);

    // ── Output gate: attn_out *= sigmoid(gate) ────────────────────
    kernels::launch_sigmoid_gate_inplace_bf16(
        ws.attn_out.data(), gate.data(), N * Hq, stream);

    // ── o_proj → ws.norm_y ────────────────────────────────────────
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.attn_out.data(), Lw.fa_o_proj->data(),
        ws.norm_y.data(), N, H, Hq);
}

}  // namespace

void qwen3_5_forward_paged(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
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
            "qwen3_5_forward_paged: multi-request batching not yet "
            "supported on the linear-attn path; got R=" +
            std::to_string(num_requests));
    }
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    // Prefill pulls fresh state — reset linear-attn caches.
    if (!is_pure_decode) {
        state_cache.reset(stream);
    }

    // 1. Embed.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    // 2. Per-layer.
    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];

        // Pre-attention norm: y → norm_x.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);

        if (Lw.kind == Qwen3_5LayerWeights::Kind::LinearAttn) {
            linear_attn_layer_body(
                Lw, cfg, ws, la_ws, state_cache,
                static_cast<int>(L), N, is_pure_decode, cublas, stream);
        } else {
            full_attn_layer_body(
                Lw, cfg, ws, cache, attn_ws, Lw.kv_layer,
                N, num_requests, is_pure_decode,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                cublas, stream);
        }

        // Residual: y += norm_y.
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            (std::size_t)N * H, stream);

        // Post-attention norm + SwiGLU MLP + residual.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        const int I = cfg.intermediate_size;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), Lw.gate_proj->data(),
            ws.gate.data(), N, I, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), Lw.up_proj->data(),
            ws.up.data(), N, I, H);
        kernels::launch_swiglu_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.gate.data(), Lw.down_proj->data(),
            ws.norm_y.data(), N, H, I);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            (std::size_t)N * H, stream);
    }

    // 3. Final norm.
    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);

    // 4. lm_head.
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(),
        ws.logits.data(), N, V, H);
}

}  // namespace pie_cuda_driver::model
