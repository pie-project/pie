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
    int max_tokens, int conv_dim, int v_h, int k_h, int k_d, int v_d,
    int hq)
{
    Qwen3_5LinearAttnWorkspace ws;
    const std::size_t N     = static_cast<std::size_t>(max_tokens);
    const std::size_t v_dim = static_cast<std::size_t>(v_h) * v_d;
    const std::size_t k_dim = static_cast<std::size_t>(k_h) * k_d;
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
    ws.q_raw = DeviceBuffer<std::uint16_t>::alloc(N * k_dim);
    ws.k_raw = DeviceBuffer<std::uint16_t>::alloc(N * k_dim);
    ws.v_raw = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    ws.q_pre = DeviceBuffer<float>::alloc(N * (std::size_t)k_h * k_d);
    ws.k_pre = DeviceBuffer<float>::alloc(N * (std::size_t)k_h * k_d);
    ws.fa_qg_packed = DeviceBuffer<std::uint16_t>::alloc(N * (std::size_t)2 * hq);
    ws.fa_gate      = DeviceBuffer<std::uint16_t>::alloc(N * (std::size_t)hq);
    return ws;
}

namespace {

// Linear-attn layer body. Reads `ws.norm_x` (post-input-layernorm
// activations) and writes the layer's contribution into `ws.norm_y`
// (which the caller adds to `ws.y` as the residual).
//
// Multi-request: every state-touching kernel is the batched variant —
// causal_conv1d_{update,prefill} and {recurrent,chunk}_gated_delta_*
// each fire as a single grid indirected through `slot_ids_d` (and on
// prefill `qo_indptr_d` too) to address each request's slab. One
// launch per kernel per layer regardless of R, eliminating
// `R × num_layers × {2,4}` per-token / per-fire launch overhead.
//
// `slot_ids_h` and `slot_ids_d` describe the same mapping in host and
// device memory respectively. When both are nullptr (parity /
// single-request callers), every kernel takes its legacy single-request
// path against slot 0 — matches the R=1 layout exactly.
void linear_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    Qwen3_5StateCache& state_cache,
    int layer_idx,
    int N, int R,
    bool is_pure_decode,
    const std::int32_t*  slot_ids_h,
    const std::int32_t*  slot_ids_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* qo_indptr_d,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    // TP-local dims for linear-attention. tp_size == 1 keeps everything
    // unsharded. The K/V head counts must divide tp_size (checked at
    // engine load); each rank operates on its 1/T head share.
    const int T        = std::max(1, fwd_cfg.tp_size);
    const int H        = cfg.hidden_size;
    const int K_h      = cfg.linear_num_key_heads / T;
    const int V_h      = cfg.linear_num_value_heads / T;
    const int K_d      = cfg.linear_key_head_dim;
    const int V_d      = cfg.linear_value_head_dim;
    const int K_dim    = K_h * K_d;
    const int V_dim    = V_h * V_d;
    const int conv_dim = 2 * K_dim + V_dim;
    const int conv_K   = cfg.linear_conv_kernel_dim;

    auto slot_for = [&](int r) -> int {
        return slot_ids_h ? slot_ids_h[r] : 0;
    };

    // ── In-projections ────────────────────────────────────────────
    // Linear-attn projections stay bf16 (no QuantMeta companion in
    // Qwen3_5LayerWeights) — the implicit WeightView ctor pulls the
    // bf16 tensor through unchanged.
    // mixed_qkv [N, conv_dim] = norm_x @ in_proj_qkv.T
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *Lw.la_in_proj_qkv,
        la.mixed_qkv.data(), N, conv_dim, H);
    // z [N, V_dim] = norm_x @ in_proj_z.T
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *Lw.la_in_proj_z,
        la.z.data(), N, V_dim, H);
    // a [N, V_h] = norm_x @ in_proj_a.T   (b symmetric)
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *Lw.la_in_proj_a,
        la.a.data(), N, V_h, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *Lw.la_in_proj_b,
        la.b.data(), N, V_h, H);

    // ── Causal depthwise conv1d (kernel=K, fused silu) ────────────
    // Per-request conv_state lives in `state_cache.conv_state(layer, slot)`.
    // Layout: conv_state is [conv_K, conv_dim] bf16 per slot.
    auto* qkv_in_base   = la.mixed_qkv.data();
    auto* qkv_post_base = la.mixed_qkv_post.data();
    if (is_pure_decode) {
        // N == R; one token per request. Decode hot path → batched
        // kernel: one launch picks per-request slot via slot_ids_d.
        if (slot_ids_d != nullptr) {
            kernels::launch_causal_conv1d_update_batched_bf16(
                qkv_in_base, Lw.la_conv1d_w->data(),
                Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                state_cache.conv_state(layer_idx, /*slot=*/0),
                slot_ids_d,
                static_cast<long long>(state_cache.conv_kernel()) *
                    state_cache.conv_dim(),
                qkv_post_base,
                R, conv_dim, conv_K, stream);
        } else {
            // Legacy single-request path (parity entrypoint, slot 0).
            kernels::launch_causal_conv1d_update_bf16(
                qkv_in_base, Lw.la_conv1d_w->data(),
                Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                state_cache.conv_state(layer_idx, 0),
                qkv_post_base,
                conv_dim, conv_K, stream);
        }
    } else {
        // Prefill: batched kernel — one launch over (C, R) blocks; each
        // block reads its own (t0_r, Nr_r) window from qo_indptr_d and
        // walks tokens internally, persisting the trailing K-window
        // into the request's slot. Falls back to host-loop for the
        // legacy single-request parity entrypoint.
        if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
            kernels::launch_causal_conv1d_prefill_batched_bf16(
                qkv_in_base, Lw.la_conv1d_w->data(),
                Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                qkv_post_base,
                state_cache.conv_state(layer_idx, /*slot=*/0),
                slot_ids_d, qo_indptr_d,
                static_cast<long long>(state_cache.conv_kernel()) *
                    state_cache.conv_dim(),
                R, conv_dim, conv_K, stream);
        } else {
            for (int r = 0; r < R; ++r) {
                const int t0 = static_cast<int>(qo_indptr_h[r]);
                const int Nr = static_cast<int>(qo_indptr_h[r + 1]) - t0;
                if (Nr <= 0) continue;
                const std::size_t off = static_cast<std::size_t>(t0) * conv_dim;
                kernels::launch_causal_conv1d_prefill_bf16(
                    qkv_in_base + off, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    qkv_post_base + off,
                    state_cache.conv_state(layer_idx, slot_for(r)),
                    Nr, conv_dim, conv_K, stream);
            }
        }
    }

    // ── Split mixed_qkv_post into q_raw, k_raw, v_raw ─────────────
    // mixed_qkv_post[N, conv_dim] packs per-token channels as
    // [q_raw(K_dim) | k_raw(K_dim) | v_raw(V_dim)]; l2norm wants a
    // contiguous [N*K_h, K_d] flat view, so copy each segment into its
    // own dense buffer via stride-aware memcpy2D.
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

        // ── L2-normalise + scale q, l2-normalise k, widen v to fp32 ─────
        const float scale = 1.f / std::sqrt(static_cast<float>(K_d));
        kernels::launch_l2norm_scale_bf16_to_fp32(
            la.q_raw.data(), la.q_pre.data(),
            N * K_h, K_d, scale, /*eps=*/1e-6f, stream);
        kernels::launch_l2norm_scale_bf16_to_fp32(
            la.k_raw.data(), la.k_pre.data(),
            N * K_h, K_d, /*scale=*/1.f, /*eps=*/1e-6f, stream);

        // repeat_interleave from K_h → V_h heads.
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

        // v: bf16 → fp32 (no l2norm).
        kernels::launch_bf16_to_fp32(
            la.v_raw.data(), la.v_fp32.data(),
            (std::size_t)N * V_dim, stream);

        // ── g, β per token per head ────────────────────────────────────
        kernels::launch_gated_delta_g_beta(
            la.a.data(), la.b.data(),
            Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
            la.g_log.data(), la.beta.data(),
            N, V_h, stream);

        // ── Recurrent update ───────────────────────────────────────────
        // Both decode and prefill: one batched launch over (R, V_h) blocks
        // with per-block slot (and on prefill, qo_indptr) indirection.
        // Decode = one token per request; prefill = each block walks its
        // own T_r-token window from `qo_indptr_d` along the recurrence.
        {
            const std::size_t qk_step = static_cast<std::size_t>(V_h) * K_d;
            const std::size_t v_step  = static_cast<std::size_t>(V_dim);
            const std::size_t gh_step = static_cast<std::size_t>(V_h);
            if (is_pure_decode) {
                if (slot_ids_d != nullptr) {
                    kernels::launch_recurrent_gated_delta_step_batched(
                        la.q_norm.data(),
                        la.k_norm.data(),
                        la.v_fp32.data(),
                        la.g_log.data(),
                        la.beta.data(),
                        state_cache.recurrent_state(layer_idx, /*slot=*/0),
                        slot_ids_d,
                        static_cast<long long>(
                            state_cache.recurrent_slot_stride_floats()),
                        la.core_out.data(),
                        R, V_h, K_d, V_d, stream);
                } else {
                    kernels::launch_recurrent_gated_delta_step(
                        la.q_norm.data(),
                        la.k_norm.data(),
                        la.v_fp32.data(),
                        la.g_log.data(),
                        la.beta.data(),
                        state_cache.recurrent_state(layer_idx, 0),
                        la.core_out.data(),
                        /*B=*/1, V_h, K_d, V_d, stream);
                }
            } else {
                if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
                    kernels::launch_chunk_gated_delta_prefill_batched(
                        la.q_norm.data(),
                        la.k_norm.data(),
                        la.v_fp32.data(),
                        la.g_log.data(),
                        la.beta.data(),
                        state_cache.recurrent_state(layer_idx, /*slot=*/0),
                        slot_ids_d, qo_indptr_d,
                        static_cast<long long>(
                            state_cache.recurrent_slot_stride_floats()),
                        la.core_out.data(),
                        R, V_h, K_d, V_d, stream);
                } else {
                    for (int r = 0; r < R; ++r) {
                        const int t0 = static_cast<int>(qo_indptr_h[r]);
                        const int Nr = static_cast<int>(qo_indptr_h[r + 1]) - t0;
                        if (Nr <= 0) continue;
                        const std::size_t qk_off = static_cast<std::size_t>(t0) * qk_step;
                        const std::size_t v_off  = static_cast<std::size_t>(t0) * v_step;
                        const std::size_t gh_off = static_cast<std::size_t>(t0) * gh_step;
                        kernels::launch_chunk_gated_delta_prefill(
                            la.q_norm.data() + qk_off,
                            la.k_norm.data() + qk_off,
                            la.v_fp32.data() + v_off,
                            la.g_log.data()  + gh_off,
                            la.beta.data()   + gh_off,
                            state_cache.recurrent_state(layer_idx, slot_for(r)),
                            la.core_out.data() + v_off,
                            Nr, V_h, K_d, V_d, /*chunk_size=*/64, stream);
                    }
                }
            }
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

    // ── out_proj: [N, V_dim] → [N, H]. On TP=1 we fuse the residual via
    //    beta=1; on TP>1 the proj is row-parallel so we write to scratch,
    //    all-reduce, then residual-add into y.
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    if (T == 1) {
        ops::gemm_act_x_w(cublas.handle(),
            la.core_out_bf16.data(), *Lw.la_out_proj,
            ws.y.data(), N, H, V_dim, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            la.core_out_bf16.data(), *Lw.la_out_proj,
            ws.norm_y.data(), N, H, V_dim, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

// Full-attention layer body. Reads `ws.norm_x`, writes contribution
// to `ws.norm_y`. KV cache and flashinfer mirror the qwen3 path.
void full_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    const ops::DecodePlanCachePtr& decode_plan,  // non-null on decode path
    int kv_layer,
    int N, int R,
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
    const int T  = std::max(1, fwd_cfg.tp_size);
    const int H  = cfg.hidden_size;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int Hq = num_q_heads_local * cfg.head_dim;
    const int Hk = num_kv_heads_local * cfg.head_dim;
    const int d  = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

    // ── q/k/v projections (q is 2× wide for the output gate) ──────
    // qg_packed [N, 2*Hq] = norm_x @ q_proj.T
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
        la.fa_qg_packed.data(), N, 2 * Hq, H);
    kernels::launch_split_q_gate_bf16(
        la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
        N, num_q_heads_local, d, stream);

    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        ws.v.data(), N, Hk, H);

    // ── q_norm / k_norm (gemma-style (1+w)·x_hat) ─────────────────
    kernels::launch_rmsnorm_gemma_bf16(
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * num_q_heads_local, d, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * num_kv_heads_local, d, eps, stream);

    // ── Partial RoPE ──────────────────────────────────────────────
    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, num_q_heads_local, num_kv_heads_local,
        d, rotary_dim, cfg.rope_theta, stream);

    // ── Write K/V to paged cache ──────────────────────────────────
    auto kv_view = cache.layer_view(kv_layer);
    kernels::launch_write_kv_to_pages(
        kv_view, ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, stream);

    // ── Flashinfer attention ──────────────────────────────────────
    // Decode path: pre-planned (graph-friendly). Prefill path: includes
    // host work inside the launcher (PrefillPlan), so non-graph-capturable.
    if (decode_plan) {
        ops::dispatch_attention_flashinfer_decode(
            *decode_plan,
            ws.q.data(), kv_view, ws.attn_out.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else {
        ops::launch_attention_flashinfer_prefill(
            ws.q.data(), kv_view, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, attn_ws, stream);
    }

    // ── Output gate: attn_out *= sigmoid(gate) ────────────────────
    kernels::launch_sigmoid_gate_inplace_bf16(
        ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);

    // ── o_proj fused with post-attn residual on TP=1; on TP>1 row-
    //    parallel: write to scratch, all-reduce, residual-add to y.
    if (T == 1) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.y.data(), N, H, Hq, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.norm_y.data(), N, H, Hq, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

}  // namespace

void prepare_qwen3_5_decode_plan(
    Qwen3_5PlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    bool is_pure_decode,
    cudaStream_t stream)
{
    if (!is_pure_decode || fwd_cfg.force_prefill_path) {
        // Body uses the prefill kernel — no plan to compute.
        state.decode_plan.reset();
        return;
    }
    if (!state.decode_plan) {
        state.decode_plan = ops::make_decode_plan();
    }
    // The decode kernel runs on per-rank slices of Q / KV — its tile
    // geometry must be planned for the per-rank head count, not the
    // full unsharded count. (Mistral-7B TP=2 happens to work with
    // either value because gqa is invariant under sharding *and* the
    // sharded Q tile size still rounds favorably; Qwen3.6-MoE
    // (head_dim=256, num_heads=16, num_kv=2 → 16/2=8 per-rank-q,
    // 1 per-rank-kv) does not — flashinfer's chunk-metadata read
    // overruns its 256-byte allocation when the full 16/2 plan meets
    // a small per-rank kv_chunks count.)
    const int T = std::max(1, fwd_cfg.tp_size);
    ops::plan_attention_flashinfer_decode(
        *state.decode_plan, kv_page_indptr_h, num_requests,
        cfg.num_attention_heads / T,
        cfg.num_key_value_heads / T,
        cfg.head_dim,
        cache.page_size(), attn_ws, stream,
        /*enable_cuda_graph=*/true,
        /*full_attention_variant=*/false,
        cache.hnd_layout());
}

std::uint32_t qwen3_5_decode_graph_layout(
    const Qwen3_5PlanState& state)
{
    if (!state.decode_plan) return 0;
    return ops::decode_plan_graph_layout(*state.decode_plan);
}

void qwen3_5_forward_paged(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
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
    const std::int32_t* /*mask_indptr_d*/,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d)
{
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const int R  = num_requests;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    // Per-slot reset for any request whose slot was just (re)assigned.
    // The runtime guarantees a slot is_fresh only on the first fire of a
    // context (which is always a prefill), so zeroing here matches the
    // "fresh state before consumption" semantic without disturbing
    // continuing decodes that share the fire. Legacy path (slot_ids_h
    // null) keeps the old "reset all on prefill" behaviour for the
    // parity entry point; max_slots == 1 there makes the two equivalent.
    if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
        for (int r = 0; r < R; ++r) {
            if (is_fresh_h[r]) {
                state_cache.reset_slot(slot_ids_h[r], stream);
            }
        }
    } else if (!is_pure_decode) {
        state_cache.reset(stream);
    }

    // Decode plan was refreshed by `prepare_qwen3_5_decode_plan` before
    // this body call (in serving) or as part of the host-side parity
    // setup. Reading it from `plan_state` keeps host work — and its
    // attendant cudaMemcpyAsync H2D from a stack-allocated indptr_h_buf
    // — out of any cudaStream capture region.
    const ops::DecodePlanCachePtr& decode_plan = plan_state.decode_plan;

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
                Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                static_cast<int>(L), N, R, is_pure_decode,
                slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                cublas, stream);
        } else {
            full_attn_layer_body(
                Lw, cfg, fwd_cfg, ws, la_ws, cache, attn_ws, decode_plan, Lw.kv_layer,
                N, num_requests,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                cublas, stream);
        }

        // (Post-attention residual is fused into the body's final GEMM
        //  via beta=1 on tp_size==1, or all-reduce + residual_add for
        //  tp_size>1. Either way ws.y has the post-attn state at this
        //  point.)

        // Post-attention norm + SwiGLU MLP + residual.
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        const int T_mlp = std::max(1, fwd_cfg.tp_size);
        const int I = cfg.intermediate_size / T_mlp;
        NcclComm* tp_mlp = (T_mlp > 1) ? fwd_cfg.tp_comm : nullptr;
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.gate_proj, Lw.gate_proj_quant),
            ws.gate.data(), N, I, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.up_proj, Lw.up_proj_quant),
            ws.up.data(), N, I, H);
        kernels::launch_swiglu_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);
        // down_proj: TP=1 fuses residual via beta=1; TP>1 is row-parallel
        // — write to scratch, all-reduce, residual-add.
        if (T_mlp == 1) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.gate.data(), make_weight_view(Lw.down_proj, Lw.down_proj_quant),
                ws.y.data(), N, H, I, /*beta=*/1.f);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.gate.data(), make_weight_view(Lw.down_proj, Lw.down_proj_quant),
                ws.norm_y.data(), N, H, I, /*beta=*/0.f);
            tp_mlp->all_reduce_bf16(ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, stream);
        }
    }

    // 3. Final norm.
    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);

    // 4. lm_head.
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *w.lm_head,
        ws.logits.data(), N, V, H);
}

}  // namespace pie_cuda_driver::model
