#include "model/qwen3_5_moe_forward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

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

namespace {

// RMSNorm dispatch: Qwen3.5 / 3.6-MoE store gamma centered at zero and
// apply `(1 + w) * x_hat` (Gemma-style); Qwen3-MoE (Qwen3-30B-A3B) uses
// the standard `w * x_hat`. The bind layer wires the same struct for
// both so the forward picks the right kernel based on `cfg.model_type`.
inline bool uses_gemma_rmsnorm(const HfConfig& cfg) {
    return cfg.model_type != "qwen3_moe";
}

inline void rmsnorm_bf16_dispatch(
    const HfConfig& cfg,
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    if (uses_gemma_rmsnorm(cfg)) {
        kernels::launch_rmsnorm_gemma_bf16(x, weight, y,
            num_rows, hidden, eps, stream);
    } else {
        kernels::launch_rmsnorm_bf16(x, weight, y,
            num_rows, hidden, eps, stream);
    }
}

}  // namespace

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
    ws.a_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_gu_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.a_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_dn_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.batch_weights = DeviceBuffer<float>::alloc(top_k);
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
// `ws.norm_y`. Multi-request semantics match qwen3_5_forward's
// linear_attn_layer_body — see the comment block there.
void linear_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    Qwen3_5StateCache& state_cache,
    int layer_idx, int N, int R, bool is_pure_decode,
    const std::int32_t*  slot_ids_h,
    const std::int32_t*  slot_ids_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* qo_indptr_d,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
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
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    auto slot_for = [&](int r) -> int {
        return slot_ids_h ? slot_ids_h[r] : 0;
    };

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

    {
        auto* qkv_in_base   = la.mixed_qkv.data();
        auto* qkv_post_base = la.mixed_qkv_post.data();
        if (is_pure_decode) {
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
                kernels::launch_causal_conv1d_update_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(layer_idx, 0),
                    qkv_post_base,
                    conv_dim, conv_K, stream);
            }
        } else {
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

    kernels::launch_fp32_to_bf16(
        la.core_out.data(), la.core_out_bf16.data(),
        (std::size_t)N * V_dim, stream);
    kernels::launch_rmsnorm_gated_bf16(
        la.core_out_bf16.data(), la.z.data(), Lw.la_norm_w_fp32,
        la.core_out_bf16.data(),
        N * V_h, V_d, /*eps=*/cfg.rms_norm_eps, stream);
    // out_proj: TP=1 fuses residual via beta=1; TP>1 row-parallel +
    // all-reduce + residual-add.
    if (T == 1) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            la.core_out_bf16.data(), Lw.la_out_proj->data(),
            ws.y.data(), N, H, V_dim, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            la.core_out_bf16.data(), Lw.la_out_proj->data(),
            ws.norm_y.data(), N, H, V_dim, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

// Full-attention body (replica of qwen3_5_forward.cpp's logic).
void full_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache, AttentionWorkspace& attn_ws,
    const ops::DecodePlanCachePtr& decode_plan,
    int kv_layer, int N, int R,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    ops::CublasHandle& cublas, cudaStream_t stream)
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

    // Qwen3.5 / 3.6-MoE fuse the per-head sigmoid output gate into
    // q_proj as a [2*Hq, H] tensor — rows [0,Hq) are q, rows [Hq,2*Hq)
    // are the gate logits. Qwen3-MoE (Qwen3-30B-A3B) ships plain q_proj
    // [Hq, H] with no output gate, so the GEMM goes straight into ws.q.
    if (cfg.attn_output_gate) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), Lw.fa_q_proj->data(),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        kernels::launch_split_q_gate_bf16(
            la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
            N, num_q_heads_local, d, stream);
    } else {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), Lw.fa_q_proj->data(),
            ws.q.data(), N, Hq, H);
    }

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_k_proj->data(),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.fa_v_proj->data(),
        ws.v.data(), N, Hk, H);

    rmsnorm_bf16_dispatch(cfg,
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * num_q_heads_local, d, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * num_kv_heads_local, d, eps, stream);

    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, num_q_heads_local, num_kv_heads_local,
        d, rotary_dim, cfg.rope_theta, stream);

    if (decode_plan) {
        kernels::launch_write_kv_decode_to_pages_bf16(
            cache.k(kv_layer), cache.v(kv_layer), ws.k.data(), ws.v.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            R, cache.page_size(), num_kv_heads_local, d, stream);
    } else {
        kernels::launch_write_kv_to_pages_bf16(
            cache.k(kv_layer), cache.v(kv_layer), ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, cache.page_size(), num_kv_heads_local, d, stream);
    }

    // Decode path: pre-planned (graph-friendly). Prefill: includes host
    // work in the launcher (PrefillPlan), so non-graph-capturable.
    if (decode_plan) {
        ops::dispatch_attention_flashinfer_decode_bf16(
            *decode_plan,
            ws.q.data(), cache.k(kv_layer), cache.v(kv_layer), ws.attn_out.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else {
        ops::launch_attention_flashinfer_prefill_bf16(
            ws.q.data(), cache.k(kv_layer), cache.v(kv_layer), ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), attn_ws, stream);
    }

    if (cfg.attn_output_gate) {
        kernels::launch_sigmoid_gate_inplace_bf16(
            ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
    }

    // o_proj: TP=1 fuses residual via beta=1; TP>1 row-parallel +
    // all-reduce + residual-add.
    if (T == 1) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), Lw.fa_o_proj->data(),
            ws.y.data(), N, H, Hq, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), Lw.fa_o_proj->data(),
            ws.norm_y.data(), N, H, Hq, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

// MoE block: routed experts + shared expert with sigmoid gate.
// Reads `ws.norm_x`, writes the combined routed-expert + shared-expert
// contribution directly into `ws.norm_y` (the residual buffer the caller
// will add into `ws.y`).
void moe_block(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    // Both routed and shared experts shard along the intermediate axis
    // (column-parallel gate/up + row-parallel down). The engine load loop
    // streams per-rank slices of `experts.gate_up_proj` / `experts.down_proj`
    // straight from the safetensors mmap, so each rank only allocates its
    // own Im_local-sized portion and the per-expert GEMMs run at the
    // sharded width. We do one all-reduce at the end of the block,
    // covering both routed and shared partial sums.
    const int Im = cfg.moe_intermediate_size / T;            // routed: sharded
    const int Is = cfg.shared_expert_intermediate_size / T;  // shared: sharded
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

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

    // 3. Routing decisions. The N=1 (decode) path stays entirely on-device
    //    (so the layer is graph-capturable); the general path needs them
    //    on-host to bucket tokens per expert and we D2H-sync below.
    std::vector<std::int32_t> topk_idx_h;
    std::vector<float>        topk_w_h;
    if (N != 1) {
        topk_idx_h.resize((std::size_t)N * K);
        topk_w_h.resize((std::size_t)N * K);
        CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), moe_ws.topk_idx.data(),
                                   topk_idx_h.size() * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), moe_ws.topk_weights.data(),
                                   topk_w_h.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // 4. (For the N>1 path only: zero moe_out before scatter_add.) The
    //    N=1 fast-path's `batched_weighted_sum` overwrites moe_out, so the
    //    memset there would be wasted work.

    // 5. Per-expert dispatch.
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert in gate_up_proj
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;       // bf16 elements per expert in down_proj

    if (N == 1) {
        // Decode fast-path. Fully on-device pipeline (graph-capturable):
        //   1. `build_moe_ptrs_decode` reads topk_idx + topk_w and writes
        //      the gate_up + down per-expert cuBLAS pointer arrays
        //      directly to fixed device buffers — no D2H, no host loop.
        //   2. `cublasGemmBatchedEx` for gate_up (top_k batches, M=1).
        //   3. `chunked_swiglu` over [top_k, 2*Im].
        //   4. `cublasGemmBatchedEx` for down_proj (top_k batches, M=1).
        //   5. `batched_weighted_sum` collapses [top_k, H] → moe_out[0].
        //
        // Every step has fixed kernel topology and stable device-pointer
        // arguments, so the executor's graph-capture path can fire
        // for the whole forward.
        kernels::launch_build_moe_ptrs_decode_bf16(
            moe_ws.topk_idx.data(),
            moe_ws.topk_weights.data(),
            Lw.moe_gate_up_proj->data(),
            Lw.moe_down_proj->data(),
            ws.norm_x.data(),
            moe_ws.expert_gate_up.data(),
            moe_ws.expert_act.data(),
            moe_ws.expert_out.data(),
            reinterpret_cast<const void**>(moe_ws.a_gu_ptrs.data()),
            reinterpret_cast<const void**>(moe_ws.b_gu_ptrs.data()),
            reinterpret_cast<void**>(moe_ws.c_gu_ptrs.data()),
            reinterpret_cast<const void**>(moe_ws.a_dn_ptrs.data()),
            reinterpret_cast<const void**>(moe_ws.b_dn_ptrs.data()),
            reinterpret_cast<void**>(moe_ws.c_dn_ptrs.data()),
            moe_ws.batch_weights.data(),
            K, H, Im, stream);

        // gate_up batched GEMM: M=1, N=2*Im, K=H, batch=top_k.
        ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
            reinterpret_cast<const void* const*>(moe_ws.b_gu_ptrs.data()),
            reinterpret_cast<const void* const*>(moe_ws.a_gu_ptrs.data()),
            reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
            /*M=*/1, /*N=*/2 * Im, /*K=*/H,
            /*batch_count=*/K);

        // SwiGLU on [top_k, 2*Im] → [top_k, Im].
        kernels::launch_chunked_swiglu_bf16(
            moe_ws.expert_gate_up.data(),
            moe_ws.expert_act.data(),
            K, Im, stream);

        // down_proj batched GEMM: M=1, N=H, K=Im, batch=top_k.
        ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
            reinterpret_cast<const void* const*>(moe_ws.b_dn_ptrs.data()),
            reinterpret_cast<const void* const*>(moe_ws.a_dn_ptrs.data()),
            reinterpret_cast<void* const*>(moe_ws.c_dn_ptrs.data()),
            /*M=*/1, /*N=*/H, /*K=*/Im,
            /*batch_count=*/K);

        // Sum K outputs into moe_out.
        kernels::launch_batched_weighted_sum_bf16(
            ws.norm_y.data(), moe_ws.expert_out.data(),
            moe_ws.batch_weights.data(),
            K, H, stream);
    } else {
        // General path (prefill / multi-token). Build per-expert routing
        // lists on host and gather/scatter via the existing kernels.
        // Zero moe_out before the scatter_add accumulation.
        CUDA_CHECK(cudaMemsetAsync(ws.norm_y.data(), 0,
            (std::size_t)N * H * sizeof(std::uint16_t), stream));
        const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
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

            const auto* gate_up_w = static_cast<const std::uint16_t*>(
                                        Lw.moe_gate_up_proj->data())
                                    + e * expert_stride_gu;
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                moe_ws.expert_in.data(), gate_up_w,
                moe_ws.expert_gate_up.data(), Ne, 2 * Im, H);

            kernels::launch_chunked_swiglu_bf16(
                moe_ws.expert_gate_up.data(),
                moe_ws.expert_act.data(),
                Ne, Im, stream);

            const auto* down_w = static_cast<const std::uint16_t*>(
                                     Lw.moe_down_proj->data())
                                 + e * expert_stride_dn;
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                moe_ws.expert_act.data(), down_w,
                moe_ws.expert_out.data(), Ne, H, Im);

            kernels::launch_scatter_add_weighted_bf16(
                ws.norm_y.data(), moe_ws.expert_out.data(),
                moe_ws.expert_idx.data(), moe_ws.expert_w.data(),
                Ne, H, stream);
        }
    }

    // ── Shared expert (Qwen3.5 / 3.6-MoE: always-on dense MLP + sigmoid
    //    gate). Qwen3-MoE has no shared expert — skip the whole block
    //    when the bind didn't wire `shared_*` pointers (Is == 0).
    if (Is > 0 && Lw.shared_gate_proj != nullptr) {
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

        // shared_out *= sigmoid(scalar_gate[n]) per token, broadcast
        // across all H channels.
        kernels::launch_sigmoid_scalar_gate_inplace_bf16(
            moe_ws.shared_out.data(), moe_ws.shared_gate_logit.data(),
            N, H, stream);

        // moe_out += shared_out — both terms are this rank's partial
        // sum (routed: row-parallel down_proj; shared: row-parallel
        // down_proj). We accumulate into ws.norm_y first and all-reduce
        // the combined partial once, so the block fires a single
        // collective regardless of routed/shared split.
        kernels::launch_residual_add_bf16(
            ws.norm_y.data(), moe_ws.shared_out.data(),
            (std::size_t)N * H, stream);
    }

    if (T > 1) {
        tp->all_reduce_bf16(ws.norm_y.data(),
            (std::size_t)N * H, ncclSum, stream);
    }
}

}  // namespace

void qwen3_5_moe_forward_paged(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
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
    const std::int32_t* /*mask_indptr_d*/,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d)
{
    // Pure-Qwen3-MoE (Qwen3-30B-A3B, model_type == "qwen3_moe") has no
    // linear-attn layers; the per-slot state cache is unused. Qwen3.5 /
    // 3.6-MoE additionally fires the linear-attn body — those layers
    // consume slot_ids_h / is_fresh_h to drive per-request state.
    const bool has_linear_attn_layers = std::any_of(
        w.layers.begin(), w.layers.end(),
        [](const Qwen3_5MoeLayerWeights& Lw) {
            return Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn;
        });
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const int R  = num_requests;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    if (has_linear_attn_layers) {
        if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
            for (int r = 0; r < R; ++r) {
                if (is_fresh_h[r]) {
                    state_cache.reset_slot(slot_ids_h[r], stream);
                }
            }
        } else if (!is_pure_decode) {
            state_cache.reset(stream);
        }
    }

    // Decode plan was refreshed by `prepare_qwen3_5_decode_plan` before
    // this body call. Avoids host work inside any cudaStream capture.
    const ops::DecodePlanCachePtr& decode_plan = plan_state.decode_plan;

    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];

        rmsnorm_bf16_dispatch(cfg,
            ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);

        if (Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn) {
            linear_attn_body(
                Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                static_cast<int>(L), N, R, is_pure_decode,
                slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                cublas, stream);
        } else {
            full_attn_body(
                Lw, cfg, fwd_cfg, ws, la_ws, cache, attn_ws, decode_plan,
                Lw.kv_layer,
                N, num_requests,
                positions, qo_indptr, kv_page_indices, kv_page_indptr,
                kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                cublas, stream);
        }
        // (Post-attention residual fused into the body's final GEMM
        //  via beta=1 on TP=1; on TP>1 the body did the all-reduce and
        //  residual_add itself. ws.y holds the post-attention state.)

        // Post-attention norm + MoE block + residual.
        rmsnorm_bf16_dispatch(cfg,
            ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        moe_block(Lw, cfg, fwd_cfg, ws, moe_ws, N, cublas, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            (std::size_t)N * H, stream);
    }

    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(),
        ws.logits.data(), N, V, H);
}

}  // namespace pie_cuda_driver::model
