#include "model/nemotron_h/nemotron_h_forward.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "distributed.hpp"
#include "store/kv_cache.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/nemotron_h.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/flashinfer_mamba.hpp"
#include "ops/flashinfer_moe.hpp"

namespace pie_cuda_driver::model {

namespace {

constexpr int kMambaIntermediate = 4096;

bool nemotron_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_decode_conv_update_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_DISABLE_DECODE_CONV_UPDATE");
        return v == nullptr || v[0] == '\0' || v[0] == '0';
    }();
    return enabled;
}

bool nemotron_grouped_moe_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_ENABLE_GROUPED_MOE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_flashinfer_moe_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_DECODE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_decode_dt_precompute_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_PRECOMPUTE_DECODE_DT");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_fused_ar_norm_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_FUSED_AR_NORM");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_exact_decode_moe_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_DISABLE_EXACT_DECODE_MOE");
        return v == nullptr || v[0] == '\0' || v[0] == '0';
    }();
    return enabled;
}

int nemotron_exact_prefill_moe_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_NEMOTRON_EXACT_PREFILL_MOE_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

bool nemotron_aligned_prefill_moe_enabled() {
    static const bool enabled = [] {
        const char* disabled =
            std::getenv("PIE_NEMOTRON_DISABLE_ALIGNED_PREFILL_MOE");
        if (disabled != nullptr && disabled[0] != '\0' &&
            disabled[0] != '0') {
            return false;
        }
        const char* enabled =
            std::getenv("PIE_NEMOTRON_ENABLE_ALIGNED_PREFILL_MOE");
        return enabled == nullptr || enabled[0] == '\0' || enabled[0] != '0';
    }();
    return enabled;
}

bool nemotron_aligned_decode_moe_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_ENABLE_ALIGNED_DECODE_MOE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_route_batched_decode_moe_enabled() {
    static const bool enabled = [] {
        const char* v =
            std::getenv("PIE_NEMOTRON_ENABLE_ROUTE_BATCHED_DECODE_MOE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool nemotron_grouped_down_exact_moe_enabled() {
    static const bool enabled = [] {
        const char* v =
            std::getenv("PIE_NEMOTRON_ENABLE_GROUPED_DOWN_EXACT_MOE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

int nemotron_aligned_moe_block_size() {
    static const int block_size = [] {
        const char* v = std::getenv("PIE_NEMOTRON_ALIGNED_MOE_BLOCK_SIZE");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::max(1, std::atoi(v));
    }();
    return block_size;
}

std::size_t nemotron_moe_aligned_rows_capacity(
    std::size_t routes,
    std::size_t num_experts)
{
    if (!nemotron_aligned_prefill_moe_enabled()) return routes;
    const std::size_t block =
        static_cast<std::size_t>(nemotron_aligned_moe_block_size());
    return routes + num_experts * (block - 1);
}

std::uint64_t nemotron_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_NEMOTRON_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{16};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

std::uint64_t nemotron_profile_skip() {
    static const std::uint64_t skip = [] {
        const char* v = std::getenv("PIE_NEMOTRON_PROFILE_SKIP");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{0};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return skip;
}

bool nemotron_profile_all_ranks() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_NEMOTRON_PROFILE_ALL_RANKS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

struct NemotronForwardProfile {
    bool enabled = false;
    int tp_rank = 0;
    int N = 0;
    int R = 0;
    bool pure_decode = false;
    int mamba_layers = 0;
    int attn_layers = 0;
    int moe_layers = 0;

    double embed_ms = 0.0;
    double norm_ms = 0.0;
    double attn_ms = 0.0;
    double mamba_inproj_ms = 0.0;
    double mamba_split_ms = 0.0;
    double mamba_conv_ms = 0.0;
    double mamba_ssm_ms = 0.0;
    double mamba_norm_ms = 0.0;
    double mamba_outproj_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_routed_ms = 0.0;
    double moe_shared_ms = 0.0;
    double moe_allreduce_ms = 0.0;
    double lm_head_ms = 0.0;
    double forward_ms = 0.0;

    cudaEvent_t forward_start = nullptr;
    cudaEvent_t forward_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~NemotronForwardProfile() {
        if (forward_start != nullptr) cudaEventDestroy(forward_start);
        if (forward_stop != nullptr) cudaEventDestroy(forward_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
    }

    void ensure_events() {
        if (forward_start != nullptr) return;
        CUDA_CHECK(cudaEventCreate(&forward_start));
        CUDA_CHECK(cudaEventCreate(&forward_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
    }

    void begin(int n, int r, bool decode, int rank, cudaStream_t stream) {
        enabled = nemotron_profile_enabled();
        if (!enabled) return;
        ensure_events();
        tp_rank = rank;
        N = n;
        R = r;
        pure_decode = decode;
        mamba_layers = attn_layers = moe_layers = 0;
        embed_ms = norm_ms = attn_ms = 0.0;
        mamba_inproj_ms = mamba_split_ms = mamba_conv_ms = 0.0;
        mamba_ssm_ms = mamba_norm_ms = mamba_outproj_ms = 0.0;
        moe_router_ms = moe_routed_ms = moe_shared_ms = moe_allreduce_ms = 0.0;
        lm_head_ms = forward_ms = 0.0;
        CUDA_CHECK(cudaEventRecord(forward_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(forward_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(forward_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, forward_start, forward_stop));
        forward_ms = ms;
    }
};

template <class F>
void profile_cuda_stage(
    NemotronForwardProfile* profile,
    double* dst,
    cudaStream_t stream,
    F&& fn)
{
    if (profile == nullptr || !profile->enabled || dst == nullptr) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile->stage_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile->stage_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile->stage_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile->stage_start, profile->stage_stop));
    *dst += static_cast<double>(ms);
}

void maybe_print_profile(const NemotronForwardProfile& p) {
    if (!p.enabled) return;
    if (p.tp_rank != 0 && !nemotron_profile_all_ranks()) return;
    static std::uint64_t seq = 0;
    ++seq;
    if (seq <= nemotron_profile_skip()) return;
    const std::uint64_t limit = nemotron_profile_print_limit();
    if (limit == 0 || seq > nemotron_profile_skip() + limit) return;

    const double mamba_ms =
        p.mamba_inproj_ms + p.mamba_split_ms + p.mamba_conv_ms +
        p.mamba_ssm_ms + p.mamba_norm_ms + p.mamba_outproj_ms;
    const double moe_ms =
        p.moe_router_ms + p.moe_routed_ms + p.moe_shared_ms +
        p.moe_allreduce_ms;
    const double named =
        p.embed_ms + p.norm_ms + p.attn_ms + mamba_ms + moe_ms +
        p.lm_head_ms;
    const double other = p.forward_ms > named ? p.forward_ms - named : 0.0;
    std::cerr
        << "[pie-nemotron-profile] seq=" << seq
        << " rank=" << p.tp_rank
        << " N=" << p.N
        << " R=" << p.R
        << " decode=" << (p.pure_decode ? 1 : 0)
        << " layers_mamba=" << p.mamba_layers
        << " layers_attn=" << p.attn_layers
        << " layers_moe=" << p.moe_layers
        << " total_ms=" << p.forward_ms
        << " embed_ms=" << p.embed_ms
        << " norm_ms=" << p.norm_ms
        << " attn_ms=" << p.attn_ms
        << " mamba_ms=" << mamba_ms
        << " mamba_inproj_ms=" << p.mamba_inproj_ms
        << " mamba_split_ms=" << p.mamba_split_ms
        << " mamba_conv_ms=" << p.mamba_conv_ms
        << " mamba_ssm_ms=" << p.mamba_ssm_ms
        << " mamba_norm_ms=" << p.mamba_norm_ms
        << " mamba_outproj_ms=" << p.mamba_outproj_ms
        << " moe_ms=" << moe_ms
        << " moe_router_ms=" << p.moe_router_ms
        << " moe_routed_ms=" << p.moe_routed_ms
        << " moe_shared_ms=" << p.moe_shared_ms
        << " moe_allreduce_ms=" << p.moe_allreduce_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
}

struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> route_ids;
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>> weights;
};

const void* maybe_tp_data(const DeviceTensor* full, const DeviceTensor& tp) {
    return tp.empty() ? full->data() : tp.data();
}

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx_h,
    const std::vector<float>& topk_w_h,
    int N, int K, int E)
{
    ExpertRouting r;
    r.route_ids.assign(E, {});
    r.token_idx.assign(E, {});
    r.weights.assign(E, {});
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx_h[static_cast<std::size_t>(n) * K + k];
            if (e < 0 || e >= E) continue;
            r.route_ids[static_cast<std::size_t>(e)].push_back(n * K + k);
            r.token_idx[static_cast<std::size_t>(e)].push_back(n);
            r.weights[static_cast<std::size_t>(e)].push_back(
                topk_w_h[static_cast<std::size_t>(n) * K + k]);
        }
    }
    return r;
}

void attention_layer(
    const NemotronHLayerWeights& Lw,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int model_layer,
    int N, int R, bool is_pure_decode,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const void* next_norm_w,
    float eps,
    cudaStream_t stream,
    bool* produced_next_norm)
{
    if (produced_next_norm != nullptr) *produced_next_norm = false;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int num_q_heads_local = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int Hq = num_q_heads_local * cfg.head_dim;
    const int Hk = num_kv_heads_local * cfg.head_dim;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.q_proj->data(), ws.q.data(), N, Hq, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.k_proj->data(), ws.k.data(), N, Hk, H);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), Lw.v_proj->data(), ws.v.data(), N, Hk, H);
    invoke_stage_hook(
        StageHookPoint::OnAttnProj, ws.q.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(Hq),
        static_cast<std::uint32_t>(model_layer), stream);

    kernels::launch_rope_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, num_q_heads_local, num_kv_heads_local,
        cfg.head_dim, cfg.rope_theta, stream);
    auto kv_view = cache.layer_view(Lw.kv_layer);
    kernels::launch_write_kv_to_pages(
        kv_view, ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, stream);

    const bool use_decode_path =
        is_pure_decode && !fwd_cfg.force_prefill_path;
    const auto* decode_plan = plan_state.decode_plan
        ? plan_state.decode_plan.get()
        : nullptr;
    const auto* prefill_plan = plan_state.prefill_plan
        ? plan_state.prefill_plan.get()
        : nullptr;

    if (use_decode_path && decode_plan != nullptr) {
        ops::dispatch_attention_flashinfer_decode(
            *decode_plan, ws.q.data(), kv_view, ws.attn_out.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else if (custom_mask_d) {
        ops::launch_attention_flashinfer_prefill_custom(
            ws.q.data(), kv_view, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            custom_mask_d, custom_mask_indptr_d,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, attn_ws, stream);
    } else if (plan_state.use_prefill_plan && prefill_plan != nullptr) {
        const int num_pages_in_batch = kv_page_indptr_h[R];
        kernels::launch_dequant_kv_cache_layer_to_bf16_active(
            kv_view, kv_page_indices, num_pages_in_batch, stream);
        ops::dispatch_attention_flashinfer_prefill_bf16(
            *prefill_plan, ws.q.data(), kv_view.k_bf16_pages,
            kv_view.v_bf16_pages, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else {
        ops::launch_attention_flashinfer_prefill(
            ws.q.data(), kv_view, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, attn_ws, stream);
    }
    invoke_stage_hook(
        StageHookPoint::OnAttn, ws.q.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(Hq),
        static_cast<std::uint32_t>(model_layer), stream);

    if (T == 1) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), Lw.o_proj->data(), ws.y.data(),
            N, H, Hq, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), Lw.o_proj->data(), ws.norm_y.data(),
            N, H, Hq, /*beta=*/0.f);
        auto* fused_ar = tp->custom_all_reduce();
        if (next_norm_w != nullptr && fused_ar != nullptr &&
            fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
            fused_ar->all_reduce_residual_rmsnorm_bf16_exact(
                ws.norm_y.data(), ws.y.data(), next_norm_w,
                ws.norm_x.data(), N, H, eps, stream);
            if (produced_next_norm != nullptr) *produced_next_norm = true;
        } else {
            tp->all_reduce_bf16_out(
                ws.norm_y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            if (next_norm_w != nullptr) {
                kernels::launch_residual_add_rmsnorm_bf16(
                    ws.y.data(), ws.norm_x.data(), next_norm_w,
                    ws.norm_x.data(), N, H, eps, stream);
                if (produced_next_norm != nullptr) *produced_next_norm = true;
            } else {
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, stream);
            }
        }
    }
}

void mamba_layer(
    const NemotronHLayerWeights& Lw,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    Workspace& ws,
    NemotronHWorkspace& nem_ws,
    RecurrentStateCache& state_cache,
    ops::CublasHandle& cublas,
    const std::uint32_t* qo_indptr,
    int layer_idx,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* slot_ids_d,
    const void* next_norm_w,
    float eps,
    cudaStream_t stream,
    NemotronForwardProfile* profile,
    bool* produced_next_norm)
{
    if (produced_next_norm != nullptr) *produced_next_norm = false;
    const int T = std::max(1, fwd_cfg.tp_size);
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const int H = cfg.hidden_size;
    const int m_heads = Lw.mamba_tp_sharded
        ? cfg.mamba_num_heads / T
        : cfg.mamba_num_heads;
    const int m_groups = Lw.mamba_tp_sharded
        ? cfg.mamba_n_groups / T
        : cfg.mamba_n_groups;
    const int m_head_dim = cfg.mamba_head_dim;
    const int m_intermediate = m_heads * m_head_dim;
    const int conv_dim =
        m_intermediate + 2 * m_groups * cfg.mamba_state_size;
    const int projection_dim = m_intermediate + conv_dim + m_heads;
    const void* in_proj_w =
        maybe_tp_data(Lw.mamba_in_proj, Lw.mamba_in_proj_tp);
    const void* conv_w =
        maybe_tp_data(Lw.mamba_conv_w, Lw.mamba_conv_w_tp);
    const void* conv_b =
        maybe_tp_data(Lw.mamba_conv_b, Lw.mamba_conv_b_tp);
    const void* D_bf16 =
        maybe_tp_data(Lw.mamba_D, Lw.mamba_D_tp);
    const void* dt_bias_bf16 =
        maybe_tp_data(Lw.mamba_dt_bias, Lw.mamba_dt_bias_tp);
    const void* norm_w =
        maybe_tp_data(Lw.mamba_norm_w, Lw.mamba_norm_w_tp);
    const void* out_proj_w =
        maybe_tp_data(Lw.mamba_out_proj, Lw.mamba_out_proj_tp);

    profile_cuda_stage(profile, profile ? &profile->mamba_inproj_ms : nullptr,
        stream, [&] {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), in_proj_w,
            nem_ws.mamba_projected.data(), N, projection_dim, H);
    });
    profile_cuda_stage(profile, profile ? &profile->mamba_split_ms : nullptr,
        stream, [&] {
        kernels::launch_nemotron_mamba_split_bf16(
            nem_ws.mamba_projected.data(),
            nullptr,
            nem_ws.mamba_conv_in.data(),
            nem_ws.mamba_dt.data(),
            N, projection_dim, m_intermediate, conv_dim, m_heads, stream);
    });

    profile_cuda_stage(profile, profile ? &profile->mamba_conv_ms : nullptr,
        stream, [&] {
        if (is_pure_decode && nemotron_decode_conv_update_enabled()) {
            if (slot_ids_d != nullptr) {
                kernels::launch_causal_conv1d_update_batched_bf16(
                    nem_ws.mamba_conv_in.data(),
                    conv_w,
                    conv_b,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    slot_ids_d,
                    static_cast<long long>(state_cache.conv_slot_stride_bytes() /
                                           sizeof(std::uint16_t)),
                    nem_ws.mamba_conv_out.data(),
                    R, conv_dim, cfg.mamba_conv_kernel, stream);
            } else {
                kernels::launch_causal_conv1d_update_bf16(
                    nem_ws.mamba_conv_in.data(),
                    conv_w,
                    conv_b,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    nem_ws.mamba_conv_out.data(),
                    conv_dim, cfg.mamba_conv_kernel, stream);
            }
        } else {
            kernels::launch_causal_conv1d_prefill_batched_bf16(
                nem_ws.mamba_conv_in.data(),
                conv_w,
                conv_b,
                nem_ws.mamba_conv_out.data(),
                state_cache.conv_state(layer_idx, /*slot=*/0),
                slot_ids_d,
                qo_indptr,
                static_cast<long long>(state_cache.conv_slot_stride_bytes() /
                                       sizeof(std::uint16_t)),
                R, conv_dim, cfg.mamba_conv_kernel, stream);
        }
    });

    profile_cuda_stage(profile, profile ? &profile->mamba_ssm_ms : nullptr,
        stream, [&] {
        const float* dt_precomputed = nullptr;
        const float* dA_precomputed = nullptr;
        if (!is_pure_decode || nemotron_decode_dt_precompute_enabled()) {
            kernels::launch_nemotron_prepare_mamba_dt_da(
                nem_ws.mamba_dt.data(),
                Lw.mamba_A.data(),
                Lw.mamba_dt_bias_f32.data(),
                nem_ws.mamba_dt_f32.data(),
                nem_ws.mamba_dA_f32.data(),
                N, m_heads, 0.f, stream);
            dt_precomputed = nem_ws.mamba_dt_f32.data();
            dA_precomputed = nem_ws.mamba_dA_f32.data();
        }
        const bool used_flashinfer_ssu =
            is_pure_decode && dt_precomputed == nullptr &&
            ops::flashinfer_mamba_ssu_bf16(
                nem_ws.mamba_conv_out.data(),
                nem_ws.mamba_dt.data(),
                Lw.mamba_A.data(),
                static_cast<const std::uint16_t*>(D_bf16),
                static_cast<const std::uint16_t*>(dt_bias_bf16),
                static_cast<std::uint16_t*>(
                    state_cache.recurrent_state_raw(layer_idx, /*slot=*/0)),
                slot_ids_d,
                nem_ws.mamba_core.data(),
                R, m_heads, m_head_dim, cfg.mamba_state_size,
                m_groups, conv_dim, m_intermediate,
                state_cache.max_slots(), stream);
        if (!used_flashinfer_ssu) {
            kernels::launch_nemotron_mamba_ssm_batched_bf16(
                nem_ws.mamba_conv_out.data(),
                nem_ws.mamba_dt.data(),
                Lw.mamba_A.data(),
                Lw.mamba_D_f32.data(),
                Lw.mamba_dt_bias_f32.data(),
                dt_precomputed,
                dA_precomputed,
                state_cache.recurrent_state_raw(layer_idx, /*slot=*/0),
                slot_ids_d,
                qo_indptr,
                nem_ws.mamba_core.data(),
                R, m_heads, m_head_dim, cfg.mamba_state_size,
                m_groups, conv_dim, m_intermediate,
                0.f, !is_pure_decode, stream);
        }
    });

    profile_cuda_stage(profile, profile ? &profile->mamba_norm_ms : nullptr,
        stream, [&] {
        kernels::launch_zamba_rmsnorm_gated_bf16(
            nem_ws.mamba_core.data(), nem_ws.mamba_projected.data(),
            norm_w, nem_ws.mamba_core.data(),
            N, m_intermediate, projection_dim,
            m_intermediate / m_groups,
            cfg.rms_norm_eps, stream);
    });

    profile_cuda_stage(profile, profile ? &profile->mamba_outproj_ms : nullptr,
        stream, [&] {
        if (Lw.mamba_tp_sharded && tp != nullptr) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                nem_ws.mamba_core.data(), out_proj_w,
                ws.norm_y.data(), N, H, m_intermediate, /*beta=*/0.f);
            auto* fused_ar = tp->custom_all_reduce();
            if (next_norm_w != nullptr && fused_ar != nullptr &&
                fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                fused_ar->all_reduce_residual_rmsnorm_bf16_exact(
                    ws.norm_y.data(), ws.y.data(), next_norm_w,
                    ws.norm_x.data(), N, H, eps, stream);
                if (produced_next_norm != nullptr) *produced_next_norm = true;
            } else {
                tp->all_reduce_bf16_out(
                    ws.norm_y.data(), ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
                if (next_norm_w != nullptr) {
                    kernels::launch_residual_add_rmsnorm_bf16(
                        ws.y.data(), ws.norm_x.data(), next_norm_w,
                        ws.norm_x.data(), N, H, eps, stream);
                    if (produced_next_norm != nullptr) *produced_next_norm = true;
                } else {
                    kernels::launch_residual_add_bf16(
                        ws.y.data(), ws.norm_x.data(),
                        static_cast<std::size_t>(N) * H, stream);
                }
            }
        } else {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                nem_ws.mamba_core.data(), out_proj_w,
                ws.y.data(), N, H, m_intermediate, /*beta=*/1.f);
        }
    });
}

void moe_layer(
    const NemotronHLayerWeights& Lw,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    Workspace& ws,
    NemotronHWorkspace& nem_ws,
    ops::CublasHandle& cublas,
    int N,
    bool is_pure_decode,
    const void* next_norm_w,
    float eps,
    cudaStream_t stream,
    NemotronForwardProfile* profile,
    bool* produced_next_norm)
{
    if (produced_next_norm != nullptr) *produced_next_norm = false;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int I = cfg.moe_intermediate_size / T;
    const int Is = cfg.shared_expert_intermediate_size / T;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const int tp_rank = tp ? tp->rank() : 0;

    profile_cuda_stage(profile, profile ? &profile->moe_router_ms : nullptr,
        stream, [&] {
        ops::gemm_act_x_wt_bf16_out_fp32(cublas.handle(),
            ws.norm_x.data(), Lw.router->data(),
            nem_ws.router_logits.data(), N, E, H);
        if (cfg.n_group != 1 || cfg.topk_group != 1) {
            throw std::runtime_error(
                "nemotron_h: CUDA router currently supports n_group=topk_group=1");
        }
        kernels::launch_topk_sigmoid_bias_fp32(
            nem_ws.router_logits.data(),
            static_cast<const float*>(Lw.router_correction_bias->data()),
            nem_ws.topk_idx.data(),
            nem_ws.topk_weights.data(),
            N, E, K, cfg.norm_topk_prob,
            cfg.routed_scaling_factor, stream);
    });

    profile_cuda_stage(profile, profile ? &profile->moe_routed_ms : nullptr,
        stream, [&] {
        const int routes = N * K;
        if (ops::flashinfer_cutlass_moe_enabled() &&
            (!is_pure_decode || nemotron_flashinfer_moe_decode_enabled()) &&
            Lw.expert_up_packed != nullptr &&
            Lw.expert_down_packed != nullptr &&
            !nem_ws.flashinfer_moe_workspace.empty() &&
            !nem_ws.flashinfer_moe_map.empty()) {
            const bool ran = ops::flashinfer_cutlass_moe_bf16_relu2(
                static_cast<const std::uint16_t*>(ws.norm_x.data()),
                nem_ws.topk_idx.data(),
                nem_ws.topk_weights.data(),
                static_cast<const std::uint16_t*>(Lw.expert_up_packed->data()),
                static_cast<const std::uint16_t*>(Lw.expert_down_packed->data()),
                static_cast<std::uint16_t*>(ws.norm_y.data()),
                nem_ws.flashinfer_moe_workspace.data(),
                nem_ws.flashinfer_moe_workspace_bytes,
                nem_ws.flashinfer_moe_map.data(),
                N, H, I, E, K, T, tp_rank, stream);
            if (ran) return;
        }
        if (is_pure_decode &&
            (N == 1 || nemotron_route_batched_decode_moe_enabled())) {
            kernels::launch_build_nemotron_moe_ptrs_decode_batched_bf16(
                nem_ws.topk_idx.data(),
                nem_ws.topk_weights.data(),
                reinterpret_cast<const void* const*>(Lw.expert_up_ptrs.data()),
                reinterpret_cast<const void* const*>(Lw.expert_down_ptrs.data()),
                ws.norm_x.data(),
                nem_ws.expert_up.data(),
                nem_ws.expert_act.data(),
                nem_ws.expert_out.data(),
                reinterpret_cast<const void**>(nem_ws.a_up_ptrs.data()),
                reinterpret_cast<const void**>(nem_ws.b_up_ptrs.data()),
                reinterpret_cast<void**>(nem_ws.c_up_ptrs.data()),
                reinterpret_cast<const void**>(nem_ws.a_down_ptrs.data()),
                reinterpret_cast<const void**>(nem_ws.b_down_ptrs.data()),
                reinterpret_cast<void**>(nem_ws.c_down_ptrs.data()),
                nem_ws.route_weights.data(),
                N, K, H, I, stream);

            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                reinterpret_cast<const void* const*>(nem_ws.b_up_ptrs.data()),
                reinterpret_cast<const void* const*>(nem_ws.a_up_ptrs.data()),
                reinterpret_cast<void* const*>(nem_ws.c_up_ptrs.data()),
                /*M=*/1, /*N=*/I, /*K=*/H, routes);
            kernels::launch_relu2_bf16(
                nem_ws.expert_up.data(), nem_ws.expert_act.data(),
                routes * I, stream);
            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                reinterpret_cast<const void* const*>(nem_ws.b_down_ptrs.data()),
                reinterpret_cast<const void* const*>(nem_ws.a_down_ptrs.data()),
                reinterpret_cast<void* const*>(nem_ws.c_down_ptrs.data()),
                /*M=*/1, /*N=*/H, /*K=*/I, routes);
            kernels::launch_token_batched_weighted_sum_bf16(
                ws.norm_y.data(), nem_ws.expert_out.data(),
                nem_ws.route_weights.data(), N, K, H, stream);
        } else {
            const bool use_aligned_decode =
                is_pure_decode && N > 1 &&
                nemotron_aligned_decode_moe_enabled();
            if ((is_pure_decode && !use_aligned_decode &&
                 nemotron_exact_decode_moe_enabled()) ||
                (!is_pure_decode && N <= nemotron_exact_prefill_moe_max_tokens())) {
                std::int32_t* sorted_route_ids = nem_ws.expert_idx.data();
                std::int32_t* route_to_sorted_row =
                    sorted_route_ids + routes;
                std::int32_t* counts_d = route_to_sorted_row + routes;
                kernels::launch_moe_bucket_exact(
                    nem_ws.topk_idx.data(),
                    sorted_route_ids,
                    route_to_sorted_row,
                    counts_d,
                    routes, E, stream);

                std::vector<std::int32_t> counts(static_cast<std::size_t>(E));
                CUDA_CHECK(cudaMemcpyAsync(
                    counts.data(), counts_d,
                    counts.size() * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<const std::uint16_t*> act_ptrs;
                std::vector<const std::uint16_t*> weight_ptrs;
                std::vector<std::uint16_t*> out_ptrs;
                std::vector<const std::uint16_t*> down_act_ptrs;
                std::vector<const std::uint16_t*> down_weight_ptrs;
                std::vector<std::uint16_t*> down_out_ptrs;
                std::vector<int> rows_per_expert;
                std::vector<int> active_experts;
                std::vector<int> row_offsets;
                act_ptrs.reserve(static_cast<std::size_t>(E));
                weight_ptrs.reserve(static_cast<std::size_t>(E));
                out_ptrs.reserve(static_cast<std::size_t>(E));
                down_act_ptrs.reserve(static_cast<std::size_t>(E));
                down_weight_ptrs.reserve(static_cast<std::size_t>(E));
                down_out_ptrs.reserve(static_cast<std::size_t>(E));
                rows_per_expert.reserve(static_cast<std::size_t>(E));
                active_experts.reserve(static_cast<std::size_t>(E));
                row_offsets.reserve(static_cast<std::size_t>(E));

                int offset = 0;
                for (int e = 0; e < E; ++e) {
                    const int Ne = counts[static_cast<std::size_t>(e)];
                    if (Ne <= 0) continue;
                    active_experts.push_back(e);
                    row_offsets.push_back(offset);
                    rows_per_expert.push_back(Ne);
                    act_ptrs.push_back(
                        nem_ws.expert_in.data() + static_cast<long long>(offset) * H);
                    weight_ptrs.push_back(
                        static_cast<const std::uint16_t*>(
                            Lw.expert_up[static_cast<std::size_t>(e)]->data()));
                    out_ptrs.push_back(
                        nem_ws.expert_up.data() + static_cast<long long>(offset) * I);
                    down_act_ptrs.push_back(
                        nem_ws.expert_act.data() +
                        static_cast<long long>(offset) * I);
                    down_weight_ptrs.push_back(
                        static_cast<const std::uint16_t*>(
                            Lw.expert_down[static_cast<std::size_t>(e)]->data()));
                    down_out_ptrs.push_back(
                        nem_ws.expert_out.data() +
                        static_cast<long long>(offset) * H);
                    offset += Ne;
                }

                if (offset > 0) {
                    kernels::launch_gather_moe_aligned_inputs_bf16(
                        ws.norm_x.data(), sorted_route_ids,
                        nem_ws.expert_in.data(), routes, offset,
                        K, H, stream);
                    nem_ws.b_up_ptrs.copy_from_host(
                        std::span<const std::uint16_t* const>(
                            act_ptrs.data(), act_ptrs.size()));
                    nem_ws.a_up_ptrs.copy_from_host(
                        std::span<const std::uint16_t* const>(
                            weight_ptrs.data(), weight_ptrs.size()));
                    nem_ws.c_up_ptrs.copy_from_host(
                        std::span<std::uint16_t* const>(
                            out_ptrs.data(), out_ptrs.size()));

                    ops::gemm_grouped_act_x_wt_bf16(
                        cublas.handle(),
                        reinterpret_cast<const void* const*>(
                            nem_ws.b_up_ptrs.data()),
                        reinterpret_cast<const void* const*>(
                            nem_ws.a_up_ptrs.data()),
                        reinterpret_cast<void* const*>(
                            nem_ws.c_up_ptrs.data()),
                        rows_per_expert.data(),
                        static_cast<int>(rows_per_expert.size()), I, H);

                    kernels::launch_relu2_bf16(
                        nem_ws.expert_up.data(), nem_ws.expert_act.data(),
                        offset * I, stream);

                    if (nemotron_grouped_down_exact_moe_enabled()) {
                        nem_ws.b_down_ptrs.copy_from_host(
                            std::span<const std::uint16_t* const>(
                                down_act_ptrs.data(), down_act_ptrs.size()));
                        nem_ws.a_down_ptrs.copy_from_host(
                            std::span<const std::uint16_t* const>(
                                down_weight_ptrs.data(),
                                down_weight_ptrs.size()));
                        nem_ws.c_down_ptrs.copy_from_host(
                            std::span<std::uint16_t* const>(
                                down_out_ptrs.data(), down_out_ptrs.size()));
                        ops::gemm_grouped_act_x_wt_bf16(
                            cublas.handle(),
                            reinterpret_cast<const void* const*>(
                                nem_ws.b_down_ptrs.data()),
                            reinterpret_cast<const void* const*>(
                                nem_ws.a_down_ptrs.data()),
                            reinterpret_cast<void* const*>(
                                nem_ws.c_down_ptrs.data()),
                            rows_per_expert.data(),
                            static_cast<int>(rows_per_expert.size()), H, I);
                    } else {
                        for (std::size_t i = 0; i < active_experts.size(); ++i) {
                            const int e = active_experts[i];
                            const int row_offset = row_offsets[i];
                            const int Ne = rows_per_expert[i];
                            ops::gemm_act_x_wt_bf16(cublas.handle(),
                                nem_ws.expert_act.data() +
                                    static_cast<long long>(row_offset) * I,
                                Lw.expert_down[static_cast<std::size_t>(e)]->data(),
                                nem_ws.expert_out.data() +
                                    static_cast<long long>(row_offset) * H,
                                Ne, H, I);
                        }
                    }

                    kernels::launch_token_batched_weighted_sum_aligned_bf16(
                        ws.norm_y.data(), nem_ws.expert_out.data(),
                        nem_ws.topk_weights.data(), route_to_sorted_row,
                        N, K, H, stream);
                }
                return;
            }

            if (use_aligned_decode ||
                (!is_pure_decode && nemotron_aligned_prefill_moe_enabled())) {
                const int block_size = nemotron_aligned_moe_block_size();
                const int max_blocks =
                    (routes + E * (block_size - 1) + block_size - 1) /
                    block_size;
                const int aligned_rows = max_blocks * block_size;
                std::int32_t* sorted_route_ids = nem_ws.expert_idx.data();
                std::int32_t* expert_ids = sorted_route_ids + aligned_rows;
                std::int32_t* route_to_aligned_row = expert_ids + max_blocks;

                kernels::launch_moe_align_decode(
                    nem_ws.topk_idx.data(), sorted_route_ids, expert_ids,
                    route_to_aligned_row,
                    routes, E, block_size, max_blocks, stream);
                kernels::launch_gather_moe_aligned_inputs_bf16(
                    ws.norm_x.data(), sorted_route_ids,
                    nem_ws.expert_in.data(), routes, aligned_rows,
                    K, H, stream);
                kernels::launch_build_nemotron_moe_ptrs_aligned_bf16(
                    expert_ids,
                    reinterpret_cast<const void* const*>(Lw.expert_up_ptrs.data()),
                    reinterpret_cast<const void* const*>(Lw.expert_down_ptrs.data()),
                    nem_ws.expert_in.data(),
                    nem_ws.expert_up.data(),
                    nem_ws.expert_act.data(),
                    nem_ws.expert_out.data(),
                    reinterpret_cast<const void**>(nem_ws.a_up_ptrs.data()),
                    reinterpret_cast<const void**>(nem_ws.b_up_ptrs.data()),
                    reinterpret_cast<void**>(nem_ws.c_up_ptrs.data()),
                    reinterpret_cast<const void**>(nem_ws.a_down_ptrs.data()),
                    reinterpret_cast<const void**>(nem_ws.b_down_ptrs.data()),
                    reinterpret_cast<void**>(nem_ws.c_down_ptrs.data()),
                    max_blocks, block_size, H, I, stream);
                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(nem_ws.b_up_ptrs.data()),
                    reinterpret_cast<const void* const*>(nem_ws.a_up_ptrs.data()),
                    reinterpret_cast<void* const*>(nem_ws.c_up_ptrs.data()),
                    block_size, I, H, max_blocks);
                kernels::launch_relu2_bf16(
                    nem_ws.expert_up.data(), nem_ws.expert_act.data(),
                    aligned_rows * I, stream);
                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(nem_ws.b_down_ptrs.data()),
                    reinterpret_cast<const void* const*>(nem_ws.a_down_ptrs.data()),
                    reinterpret_cast<void* const*>(nem_ws.c_down_ptrs.data()),
                    block_size, H, I, max_blocks);
                kernels::launch_token_batched_weighted_sum_aligned_bf16(
                    ws.norm_y.data(), nem_ws.expert_out.data(),
                    nem_ws.topk_weights.data(), route_to_aligned_row,
                    N, K, H, stream);
                return;
            }

            std::vector<std::int32_t> topk_idx_h(static_cast<std::size_t>(N) * K);
            std::vector<float> topk_w_h(static_cast<std::size_t>(N) * K);
            CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), nem_ws.topk_idx.data(),
                                       topk_idx_h.size() * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), nem_ws.topk_weights.data(),
                                       topk_w_h.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
            if (nemotron_grouped_moe_enabled()) {
                std::vector<std::int32_t> sorted_route_ids;
                std::vector<const std::uint16_t*> act_ptrs;
                std::vector<const std::uint16_t*> weight_ptrs;
                std::vector<std::uint16_t*> out_ptrs;
                std::vector<int> rows_per_expert;
                std::vector<int> active_experts;
                std::vector<int> row_offsets;
                std::vector<std::int32_t> route_to_sorted_row(
                    static_cast<std::size_t>(N) * K);
                sorted_route_ids.reserve(static_cast<std::size_t>(N) * K);
                act_ptrs.reserve(static_cast<std::size_t>(E));
                weight_ptrs.reserve(static_cast<std::size_t>(E));
                out_ptrs.reserve(static_cast<std::size_t>(E));
                rows_per_expert.reserve(static_cast<std::size_t>(E));
                active_experts.reserve(static_cast<std::size_t>(E));
                row_offsets.reserve(static_cast<std::size_t>(E));

                for (int e = 0; e < E; ++e) {
                    const auto& route_ids =
                        routing.route_ids[static_cast<std::size_t>(e)];
                    const int Ne = static_cast<int>(route_ids.size());
                    if (Ne == 0) continue;
                    const std::size_t offset = sorted_route_ids.size();
                    for (std::size_t j = 0; j < route_ids.size(); ++j) {
                        const int route = route_ids[j];
                        route_to_sorted_row[static_cast<std::size_t>(route)] =
                            static_cast<std::int32_t>(
                                offset + j);
                    }
                    sorted_route_ids.insert(
                        sorted_route_ids.end(), route_ids.begin(), route_ids.end());
                    active_experts.push_back(e);
                    row_offsets.push_back(static_cast<int>(offset));
                    rows_per_expert.push_back(Ne);
                    act_ptrs.push_back(
                        nem_ws.expert_in.data() + static_cast<long long>(offset) * H);
                    weight_ptrs.push_back(
                        static_cast<const std::uint16_t*>(
                            Lw.expert_up[static_cast<std::size_t>(e)]->data()));
                    out_ptrs.push_back(
                        nem_ws.expert_up.data() + static_cast<long long>(offset) * I);
                }

                const int routed_rows = static_cast<int>(sorted_route_ids.size());
                if (routed_rows > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        nem_ws.expert_idx.data(), sorted_route_ids.data(),
                        sorted_route_ids.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream));
                    std::int32_t* route_to_sorted_row_d =
                        nem_ws.expert_idx.data() + routed_rows;
                    CUDA_CHECK(cudaMemcpyAsync(
                        route_to_sorted_row_d, route_to_sorted_row.data(),
                        route_to_sorted_row.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream));
                    kernels::launch_gather_moe_aligned_inputs_bf16(
                        ws.norm_x.data(), nem_ws.expert_idx.data(),
                        nem_ws.expert_in.data(), N * K, routed_rows,
                        K, H, stream);
                    nem_ws.b_up_ptrs.copy_from_host(
                        std::span<const std::uint16_t* const>(
                            act_ptrs.data(), act_ptrs.size()));
                    nem_ws.a_up_ptrs.copy_from_host(
                        std::span<const std::uint16_t* const>(
                            weight_ptrs.data(), weight_ptrs.size()));
                    nem_ws.c_up_ptrs.copy_from_host(
                        std::span<std::uint16_t* const>(
                            out_ptrs.data(), out_ptrs.size()));

                    ops::gemm_grouped_act_x_wt_bf16(
                        cublas.handle(),
                        reinterpret_cast<const void* const*>(
                            nem_ws.b_up_ptrs.data()),
                        reinterpret_cast<const void* const*>(
                            nem_ws.a_up_ptrs.data()),
                        reinterpret_cast<void* const*>(
                            nem_ws.c_up_ptrs.data()),
                        rows_per_expert.data(),
                        static_cast<int>(rows_per_expert.size()), I, H);

                    kernels::launch_relu2_bf16(
                        nem_ws.expert_up.data(), nem_ws.expert_act.data(),
                        routed_rows * I, stream);

                    for (std::size_t i = 0; i < active_experts.size(); ++i) {
                        const int e = active_experts[i];
                        const int offset = row_offsets[i];
                        const int Ne = rows_per_expert[i];
                        ops::gemm_act_x_wt_bf16(cublas.handle(),
                            nem_ws.expert_act.data() +
                                static_cast<long long>(offset) * I,
                            Lw.expert_down[static_cast<std::size_t>(e)]->data(),
                            nem_ws.expert_out.data() +
                                static_cast<long long>(offset) * H,
                            Ne, H, I);
                    }

                    kernels::launch_token_batched_weighted_sum_aligned_bf16(
                        ws.norm_y.data(), nem_ws.expert_out.data(),
                        nem_ws.topk_weights.data(), route_to_sorted_row_d,
                        N, K, H, stream);
                }
            } else {
                std::vector<std::int32_t> packed_token_idx;
                std::vector<float> packed_weights;
                std::vector<int> expert_offsets(static_cast<std::size_t>(E) + 1, 0);
                packed_token_idx.reserve(static_cast<std::size_t>(N) * K);
                packed_weights.reserve(static_cast<std::size_t>(N) * K);

                for (int e = 0; e < E; ++e) {
                    expert_offsets[static_cast<std::size_t>(e)] =
                        static_cast<int>(packed_token_idx.size());
                    const auto& tok_idx =
                        routing.token_idx[static_cast<std::size_t>(e)];
                    const auto& wts =
                        routing.weights[static_cast<std::size_t>(e)];
                    packed_token_idx.insert(
                        packed_token_idx.end(), tok_idx.begin(), tok_idx.end());
                    packed_weights.insert(
                        packed_weights.end(), wts.begin(), wts.end());
                }
                expert_offsets[static_cast<std::size_t>(E)] =
                    static_cast<int>(packed_token_idx.size());
                if (!packed_token_idx.empty()) {
                    nem_ws.expert_idx.copy_from_host(
                        std::span<const std::int32_t>(
                            packed_token_idx.data(), packed_token_idx.size()));
                    nem_ws.expert_w.copy_from_host(
                        std::span<const float>(
                            packed_weights.data(), packed_weights.size()));
                }

                CUDA_CHECK(cudaMemsetAsync(ws.norm_y.data(), 0,
                    static_cast<std::size_t>(N) * H * sizeof(std::uint16_t), stream));
                for (int e = 0; e < E; ++e) {
                    const int offset = expert_offsets[static_cast<std::size_t>(e)];
                    const int Ne =
                        expert_offsets[static_cast<std::size_t>(e) + 1] - offset;
                    if (Ne == 0) continue;
                    const std::int32_t* expert_idx_d =
                        nem_ws.expert_idx.data() + offset;
                    const float* expert_w_d =
                        nem_ws.expert_w.data() + offset;
                    kernels::launch_gather_bf16_rows(
                        static_cast<const std::uint16_t*>(ws.norm_x.data()),
                        expert_idx_d,
                        nem_ws.expert_in.data(), Ne, H, stream);
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        nem_ws.expert_in.data(),
                        Lw.expert_up[static_cast<std::size_t>(e)]->data(),
                        nem_ws.expert_up.data(), Ne, I, H);
                    kernels::launch_relu2_bf16(
                        nem_ws.expert_up.data(), nem_ws.expert_act.data(),
                        Ne * I, stream);
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        nem_ws.expert_act.data(),
                        Lw.expert_down[static_cast<std::size_t>(e)]->data(),
                        nem_ws.expert_out.data(), Ne, H, I);
                    kernels::launch_scatter_add_weighted_bf16(
                        ws.norm_y.data(), nem_ws.expert_out.data(),
                        expert_idx_d, expert_w_d,
                        Ne, H, stream);
                }
            }
        }
    });

    profile_cuda_stage(profile, profile ? &profile->moe_shared_ms : nullptr,
        stream, [&] {
    if (Is > 0) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), Lw.shared_up->data(),
            nem_ws.shared_up.data(), N, Is, H);
        kernels::launch_relu2_bf16(
            nem_ws.shared_up.data(), nem_ws.shared_act.data(),
            N * Is, stream);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            nem_ws.shared_act.data(), Lw.shared_down->data(),
            nem_ws.shared_out.data(), N, H, Is);
        kernels::launch_residual_add_bf16(
            ws.norm_y.data(), nem_ws.shared_out.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
    });

    if (T > 1) {
        profile_cuda_stage(profile, profile ? &profile->moe_allreduce_ms : nullptr,
            stream, [&] {
            auto* fused_ar = tp->custom_all_reduce();
            if (next_norm_w != nullptr && fused_ar != nullptr &&
                fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                fused_ar->all_reduce_residual_rmsnorm_bf16_exact(
                    ws.norm_y.data(), ws.y.data(), next_norm_w,
                    ws.norm_x.data(), N, H, eps, stream);
                if (produced_next_norm != nullptr) *produced_next_norm = true;
            } else {
                tp->all_reduce_bf16_out(
                    ws.norm_y.data(), ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
                if (next_norm_w != nullptr) {
                    kernels::launch_residual_add_rmsnorm_bf16(
                        ws.y.data(), ws.norm_x.data(), next_norm_w,
                        ws.norm_x.data(), N, H, eps, stream);
                    if (produced_next_norm != nullptr) *produced_next_norm = true;
                }
            }
        });
        if (produced_next_norm == nullptr || !*produced_next_norm) {
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, stream);
        }
    } else {
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

}  // namespace

NemotronHWorkspace NemotronHWorkspace::allocate(
    const HfConfig& cfg, int max_tokens, int tp_size)
{
    NemotronHWorkspace ws;
    const int T = std::max(1, tp_size);
    const bool shard_mamba = nemotron_h_tp_mamba_sharding_enabled(T);
    const std::size_t N = static_cast<std::size_t>(max_tokens);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t m_heads = static_cast<std::size_t>(
        shard_mamba ? cfg.mamba_num_heads / T : cfg.mamba_num_heads);
    const std::size_t m_groups = static_cast<std::size_t>(
        shard_mamba ? cfg.mamba_n_groups / T : cfg.mamba_n_groups);
    const std::size_t m_intermediate =
        m_heads * cfg.mamba_head_dim;
    const std::size_t conv_dim =
        m_intermediate +
        2ull * m_groups * cfg.mamba_state_size;
    const std::size_t projection_dim = m_intermediate + conv_dim + m_heads;
    ws.mamba_projected = DeviceBuffer<std::uint16_t>::alloc(N * projection_dim);
    ws.mamba_gate = DeviceBuffer<std::uint16_t>::alloc(N * m_intermediate);
    ws.mamba_conv_in = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.mamba_conv_out = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.mamba_core = DeviceBuffer<std::uint16_t>::alloc(N * m_intermediate);
    ws.mamba_dt = DeviceBuffer<std::uint16_t>::alloc(N * m_heads);
    ws.mamba_dt_f32 = DeviceBuffer<float>::alloc(N * m_heads);
    ws.mamba_dA_f32 = DeviceBuffer<float>::alloc(N * m_heads);

    const std::size_t E = static_cast<std::size_t>(cfg.num_experts);
    const std::size_t K = static_cast<std::size_t>(cfg.num_experts_per_tok);
    const std::size_t routes = N * K;
    const std::size_t moe_rows = nemotron_moe_aligned_rows_capacity(routes, E);
    const std::size_t moe_blocks =
        (moe_rows + static_cast<std::size_t>(
                        std::max(1, nemotron_aligned_moe_block_size())) - 1) /
        static_cast<std::size_t>(std::max(1, nemotron_aligned_moe_block_size()));
    const std::size_t I =
        static_cast<std::size_t>(cfg.moe_intermediate_size / T);
    const std::size_t Is =
        static_cast<std::size_t>(cfg.shared_expert_intermediate_size / T);
    ws.router_logits = DeviceBuffer<float>::alloc(N * E);
    ws.topk_idx = DeviceBuffer<std::int32_t>::alloc(routes);
    ws.topk_weights = DeviceBuffer<float>::alloc(routes);
    ws.expert_in = DeviceBuffer<std::uint16_t>::alloc(moe_rows * H);
    ws.expert_up = DeviceBuffer<std::uint16_t>::alloc(moe_rows * I);
    ws.expert_act = DeviceBuffer<std::uint16_t>::alloc(moe_rows * I);
    ws.expert_out = DeviceBuffer<std::uint16_t>::alloc(moe_rows * H);
    ws.expert_idx = DeviceBuffer<std::int32_t>::alloc(
        moe_rows + moe_blocks + routes);
    ws.expert_w = DeviceBuffer<float>::alloc(routes);
    ws.shared_up = DeviceBuffer<std::uint16_t>::alloc(N * Is);
    ws.shared_act = DeviceBuffer<std::uint16_t>::alloc(N * Is);
    ws.shared_out = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.a_up_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(routes);
    ws.b_up_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(routes);
    ws.c_up_ptrs = DeviceBuffer<std::uint16_t*>::alloc(routes);
    ws.a_down_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(routes);
    ws.b_down_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(routes);
    ws.c_down_ptrs = DeviceBuffer<std::uint16_t*>::alloc(routes);
    ws.route_weights = DeviceBuffer<float>::alloc(routes);
    if (ops::flashinfer_cutlass_moe_enabled()) {
        ws.flashinfer_moe_workspace_bytes =
            ops::flashinfer_cutlass_moe_workspace_bytes(
                static_cast<int>(N), static_cast<int>(H),
                static_cast<int>(I), static_cast<int>(E),
                static_cast<int>(K), T, 0);
        ws.flashinfer_moe_workspace =
            DeviceBuffer<std::uint8_t>::alloc(ws.flashinfer_moe_workspace_bytes);
        ws.flashinfer_moe_map = DeviceBuffer<std::int32_t>::alloc(routes);
    }
    return ws;
}

std::size_t nemotron_h_workspace_bytes(
    const HfConfig& cfg, int max_tokens, int tp_size)
{
    const int T = std::max(1, tp_size);
    const bool shard_mamba = nemotron_h_tp_mamba_sharding_enabled(T);
    const std::size_t N = static_cast<std::size_t>(max_tokens);
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t m_heads = static_cast<std::size_t>(
        shard_mamba ? cfg.mamba_num_heads / T : cfg.mamba_num_heads);
    const std::size_t m_groups = static_cast<std::size_t>(
        shard_mamba ? cfg.mamba_n_groups / T : cfg.mamba_n_groups);
    const std::size_t m_intermediate =
        m_heads * cfg.mamba_head_dim;
    const std::size_t conv_dim =
        m_intermediate +
        2ull * m_groups * cfg.mamba_state_size;
    const std::size_t projection_dim = m_intermediate + conv_dim + m_heads;
    const std::size_t E = static_cast<std::size_t>(cfg.num_experts);
    const std::size_t K = static_cast<std::size_t>(cfg.num_experts_per_tok);
    const std::size_t routes = N * K;
    const std::size_t moe_rows = nemotron_moe_aligned_rows_capacity(routes, E);
    const std::size_t block =
        static_cast<std::size_t>(std::max(1, nemotron_aligned_moe_block_size()));
    const std::size_t moe_blocks = (moe_rows + block - 1) / block;
    const std::size_t I =
        static_cast<std::size_t>(cfg.moe_intermediate_size / T);
    const std::size_t Is =
        static_cast<std::size_t>(cfg.shared_expert_intermediate_size / T);
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto i32 = [](std::size_t elems) { return elems * 4; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    std::size_t bytes = 0;
    bytes += u16(N * projection_dim);
    bytes += u16(N * m_intermediate);
    bytes += u16(N * conv_dim);
    bytes += u16(N * conv_dim);
    bytes += u16(N * m_intermediate);
    bytes += u16(N * m_heads);
    bytes += fp32(N * m_heads);
    bytes += fp32(N * m_heads);
    bytes += fp32(N * E);
    bytes += i32(routes);
    bytes += fp32(routes);
    bytes += u16(moe_rows * H);
    bytes += u16(moe_rows * I);
    bytes += u16(moe_rows * I);
    bytes += u16(moe_rows * H);
    bytes += i32(moe_rows + moe_blocks + routes);
    bytes += fp32(routes);
    bytes += u16(N * Is);
    bytes += u16(N * Is);
    bytes += u16(N * H);
    bytes += routes * (6 * sizeof(void*) + sizeof(float));
    if (ops::flashinfer_cutlass_moe_enabled()) {
        bytes += ops::flashinfer_cutlass_moe_workspace_bytes(
            static_cast<int>(N), static_cast<int>(H), static_cast<int>(I),
            static_cast<int>(E), static_cast<int>(K), T, 0);
        bytes += i32(routes);
    }
    return bytes;
}

int nemotron_h_attention_layers(const HfConfig& cfg) {
    return static_cast<int>(std::count(
        cfg.layer_types.begin(), cfg.layer_types.end(), "attention"));
}

void nemotron_h_forward_paged(
    const NemotronHWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    NemotronHWorkspace& nem_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    int N,
    int R,
    bool is_pure_decode,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    NemotronForwardProfile profile;
    profile.begin(N, R, is_pure_decode, fwd_cfg.tp_comm ? fwd_cfg.tp_comm->rank() : 0, stream);

    if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
        for (int r = 0; r < R; ++r) {
            if (is_fresh_h[r]) {
                state_cache.reset_slot(slot_ids_h[r], stream);
            }
        }
    } else if (!is_pure_decode) {
        state_cache.reset(stream);
    }

    profile_cuda_stage(&profile, &profile.embed_ms, stream, [&] {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(),
            N, H, cfg.vocab_size, stream);
    });

    bool have_norm_x = false;
    for (std::size_t li = 0; li < w.layers.size(); ++li) {
        const auto& Lw = w.layers[li];
        if (have_norm_x) {
            have_norm_x = false;
        } else {
            profile_cuda_stage(&profile, &profile.norm_ms, stream, [&] {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), Lw.norm->data(), ws.norm_x.data(),
                    N, H, eps, stream);
            });
        }

        const bool try_fused_next_norm =
            nemotron_fused_ar_norm_enabled() && li + 1 < w.layers.size();
        const void* next_norm_w = try_fused_next_norm
            ? w.layers[li + 1].norm->data()
            : nullptr;
        bool produced_next_norm = false;

        if (Lw.kind == NemotronHLayerWeights::Kind::Mamba) {
            ++profile.mamba_layers;
            mamba_layer(Lw, cfg, fwd_cfg, ws, nem_ws, state_cache, cublas,
                qo_indptr, static_cast<int>(li), N, R, is_pure_decode,
                slot_ids_d, next_norm_w, eps, stream, &profile,
                &produced_next_norm);
        } else if (Lw.kind == NemotronHLayerWeights::Kind::Attention) {
            ++profile.attn_layers;
            profile_cuda_stage(&profile, &profile.attn_ms, stream, [&] {
                attention_layer(Lw, cfg, fwd_cfg, plan_state, ws, cache, attn_ws,
                    cublas, positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                    static_cast<int>(li),
                    N, R, is_pure_decode, custom_mask_d, custom_mask_indptr_d,
                    next_norm_w, eps, stream, &produced_next_norm);
            });
        } else {
            ++profile.moe_layers;
            moe_layer(Lw, cfg, fwd_cfg, ws, nem_ws, cublas,
                N, is_pure_decode, next_norm_w, eps, stream, &profile,
                &produced_next_norm);
        }
        have_norm_x = produced_next_norm;
    }

    if (!fwd_cfg.emit_logits) {
        profile.end(stream);
        maybe_print_profile(profile);
        return;
    }

    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        int lm_head_rows = N;
        const void* lm_head_input = ws.norm_y.data();
        if (compact_logits) {
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.y.data()),
                logit_row_indices_d,
                static_cast<std::uint16_t*>(ws.norm_x.data()),
                num_logit_rows, H, stream);
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
                num_logit_rows, H, eps, stream);
            lm_head_rows = num_logit_rows;
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), w.final_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
        }
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            lm_head_input, w.lm_head->data(),
            ws.logits.data(), lm_head_rows, V, H);
    });
    profile.end(stream);
    maybe_print_profile(profile);
}

int nemotron_h_mamba_layers(const HfConfig& cfg) {
    return static_cast<int>(std::count(
        cfg.layer_types.begin(), cfg.layer_types.end(), "mamba"));
}

std::size_t kv_page_bytes_nemotron_h(const HfConfig& cfg,
                                     int tp_size,
                                     const ::pie_cuda_driver::KvCacheFormat& format) {
    const int attention_layers = static_cast<int>(std::count(
        cfg.layer_types.begin(), cfg.layer_types.end(), "attention"));
    const int kv_heads = cfg.num_key_value_heads / std::max(1, tp_size);
    return static_cast<std::size_t>(attention_layers) *
           kv_cache_device_bytes_per_page(
               format, 1, kv_heads, cfg.head_dim_kernel);
}

std::size_t nemotron_h_state_slot_bytes(const HfConfig& cfg,
                                        int mamba_layers,
                                        int tp_size) {
    if (mamba_layers <= 0) return 0;
    const int T = std::max(1, tp_size);
    const bool shard_mamba = nemotron_h_tp_mamba_sharding_enabled(T);
    const std::size_t m_intermediate =
        static_cast<std::size_t>(
            std::max(0, shard_mamba ? cfg.mamba_num_heads / T
                                    : cfg.mamba_num_heads)) *
        static_cast<std::size_t>(std::max(0, cfg.mamba_head_dim));
    const std::size_t m_groups =
        static_cast<std::size_t>(
            std::max(0, shard_mamba ? cfg.mamba_n_groups / T
                                    : cfg.mamba_n_groups));
    const std::size_t conv_dim =
        m_intermediate +
        2ull * m_groups *
            static_cast<std::size_t>(std::max(0, cfg.mamba_state_size));
    const std::size_t per_slot_conv =
        static_cast<std::size_t>(std::max(0, cfg.mamba_conv_kernel)) *
        conv_dim * sizeof(std::uint16_t);
    const std::size_t per_slot_recurrent =
        static_cast<std::size_t>(
            std::max(0, shard_mamba ? cfg.mamba_num_heads / T
                                    : cfg.mamba_num_heads)) *
        static_cast<std::size_t>(std::max(0, cfg.mamba_head_dim)) *
        static_cast<std::size_t>(std::max(0, cfg.mamba_state_size)) *
        sizeof(std::uint16_t);
    return static_cast<std::size_t>(mamba_layers) *
           (per_slot_conv + per_slot_recurrent);
}

}  // namespace pie_cuda_driver::model
