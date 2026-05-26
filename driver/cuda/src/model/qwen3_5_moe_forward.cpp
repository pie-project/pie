#include "model/qwen3_5_moe_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
#include "ops/attention_naive.hpp"
#include "ops/attention_naive_paged.hpp"
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

bool qwen35_moe_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t qwen35_moe_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool qwen35_moe_profile_all_ranks() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE_ALL_RANKS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool mtp_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t mtp_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

int qwen35_gdn_cached_prefill_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

int qwen35_gdn_warp_tiled_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

int qwen35_moe_aligned_decode_block_size() {
    static const int block = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK");
        if (v == nullptr || v[0] == '\0') return 16;
        char* end = nullptr;
        long parsed_long = std::strtol(v, &end, 10);
        if (end == v) return 16;
        int parsed = static_cast<int>(parsed_long);
        if (parsed <= 1) return 0;
        if (parsed < 2) parsed = 2;
        if (parsed > 64) parsed = 64;
        return parsed;
    }();
    return block;
}

int qwen35_moe_aligned_decode_min_routes() {
    static const int min_routes = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::clamp(std::atoi(v), 0, 4096);
    }();
    return min_routes;
}

int qwen35_moe_decode_fast_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_DECODE_FAST_N");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::clamp(std::atoi(v), 0, 128);
    }();
    return max_tokens;
}

bool qwen35_moe_wmma_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_WMMA_DECODE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

enum class MtpMoeMode {
    Full,
    SharedOnly,
    Skip,
};

MtpMoeMode mtp_moe_mode() {
    static const MtpMoeMode mode = [] {
        const char* v = std::getenv("PIE_MTP_MOE_MODE");
        if (v == nullptr || v[0] == '\0') return MtpMoeMode::Full;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "shared" || s == "shared_only" || s == "shared-only") {
            return MtpMoeMode::SharedOnly;
        }
        if (s == "skip" || s == "none" || s == "off") {
            return MtpMoeMode::Skip;
        }
        return MtpMoeMode::Full;
    }();
    return mode;
}

struct Qwen35MoeForwardProfile {
    bool enabled = false;
    int tp_rank = 0;
    int N = 0;
    int R = 0;
    bool pure_decode = false;
    int linear_layers = 0;
    int full_layers = 0;
    int moe_layers = 0;

    double embed_ms = 0.0;
    double norm_ms = 0.0;
    double linear_attn_ms = 0.0;
    double linear_proj_ms = 0.0;
    double linear_conv_ms = 0.0;
    double linear_prep_ms = 0.0;
    double linear_recur_ms = 0.0;
    double linear_post_ms = 0.0;
    double full_attn_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_routed_ms = 0.0;
    double moe_route_setup_ms = 0.0;
    double moe_gate_up_ms = 0.0;
    double moe_act_ms = 0.0;
    double moe_down_ms = 0.0;
    double moe_reduce_ms = 0.0;
    double moe_shared_ms = 0.0;
    double moe_shared_gate_up_ms = 0.0;
    double moe_shared_down_ms = 0.0;
    double moe_shared_gate_ms = 0.0;
    double moe_allreduce_ms = 0.0;
    double residual_ms = 0.0;
    double lm_head_ms = 0.0;
    double forward_ms = 0.0;

    cudaEvent_t forward_start = nullptr;
    cudaEvent_t forward_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;
    cudaEvent_t detail_start = nullptr;
    cudaEvent_t detail_stop = nullptr;

    ~Qwen35MoeForwardProfile() {
        if (forward_start != nullptr) cudaEventDestroy(forward_start);
        if (forward_stop != nullptr) cudaEventDestroy(forward_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
        if (detail_start != nullptr) cudaEventDestroy(detail_start);
        if (detail_stop != nullptr) cudaEventDestroy(detail_stop);
    }

    void ensure_events() {
        if (forward_start != nullptr) return;
        CUDA_CHECK(cudaEventCreate(&forward_start));
        CUDA_CHECK(cudaEventCreate(&forward_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
        CUDA_CHECK(cudaEventCreate(&detail_start));
        CUDA_CHECK(cudaEventCreate(&detail_stop));
    }

    void begin(int n, int r, bool decode, int rank, cudaStream_t stream) {
        enabled = qwen35_moe_profile_enabled();
        if (!enabled) return;
        ensure_events();
        tp_rank = rank;
        N = n;
        R = r;
        pure_decode = decode;
        linear_layers = 0;
        full_layers = 0;
        moe_layers = 0;
        embed_ms = norm_ms = linear_attn_ms = full_attn_ms = 0.0;
        linear_proj_ms = linear_conv_ms = linear_prep_ms = 0.0;
        linear_recur_ms = linear_post_ms = 0.0;
        moe_router_ms = moe_routed_ms = moe_shared_ms = moe_allreduce_ms = 0.0;
        moe_route_setup_ms = moe_gate_up_ms = moe_act_ms = moe_down_ms = 0.0;
        moe_reduce_ms = moe_shared_gate_up_ms = moe_shared_down_ms = 0.0;
        moe_shared_gate_ms = 0.0;
        residual_ms = lm_head_ms = forward_ms = 0.0;
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

    void add(double& dst, float ms) {
        dst += static_cast<double>(ms);
    }
};

template <class F>
void profile_cuda_stage(
    Qwen35MoeForwardProfile* profile,
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
    profile->add(*dst, ms);
}

template <class F>
void profile_cuda_detail_stage(
    Qwen35MoeForwardProfile* profile,
    double* dst,
    cudaStream_t stream,
    F&& fn)
{
    if (profile == nullptr || !profile->enabled || dst == nullptr) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile->detail_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile->detail_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile->detail_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile->detail_start, profile->detail_stop));
    profile->add(*dst, ms);
}

void maybe_print_profile(const Qwen35MoeForwardProfile& p) {
    if (!p.enabled) return;
    if (p.tp_rank != 0 && !qwen35_moe_profile_all_ranks()) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = qwen35_moe_profile_print_limit();
    if (limit == 0 || seq > limit) return;

    const double named =
        p.embed_ms + p.norm_ms + p.linear_attn_ms + p.full_attn_ms +
        p.moe_router_ms + p.moe_routed_ms + p.moe_shared_ms +
        p.moe_allreduce_ms + p.residual_ms + p.lm_head_ms;
    const double other = p.forward_ms > named ? p.forward_ms - named : 0.0;
    std::cerr
        << "[pie-qwen35-moe-profile] seq=" << seq
        << " rank=" << p.tp_rank
        << " N=" << p.N
        << " R=" << p.R
        << " decode=" << (p.pure_decode ? 1 : 0)
        << " layers_linear=" << p.linear_layers
        << " layers_full=" << p.full_layers
        << " layers_moe=" << p.moe_layers
        << " total_ms=" << p.forward_ms
        << " embed_ms=" << p.embed_ms
        << " norm_ms=" << p.norm_ms
        << " linear_attn_ms=" << p.linear_attn_ms
        << " linear_proj_ms=" << p.linear_proj_ms
        << " linear_conv_ms=" << p.linear_conv_ms
        << " linear_prep_ms=" << p.linear_prep_ms
        << " linear_recur_ms=" << p.linear_recur_ms
        << " linear_post_ms=" << p.linear_post_ms
        << " full_attn_ms=" << p.full_attn_ms
        << " moe_router_ms=" << p.moe_router_ms
        << " moe_routed_ms=" << p.moe_routed_ms
        << " moe_route_setup_ms=" << p.moe_route_setup_ms
        << " moe_gate_up_ms=" << p.moe_gate_up_ms
        << " moe_act_ms=" << p.moe_act_ms
        << " moe_down_ms=" << p.moe_down_ms
        << " moe_reduce_ms=" << p.moe_reduce_ms
        << " moe_shared_ms=" << p.moe_shared_ms
        << " moe_shared_gate_up_ms=" << p.moe_shared_gate_up_ms
        << " moe_shared_down_ms=" << p.moe_shared_down_ms
        << " moe_shared_gate_ms=" << p.moe_shared_gate_ms
        << " moe_allreduce_ms=" << p.moe_allreduce_ms
        << " residual_ms=" << p.residual_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
}

struct MtpProfile {
    bool enabled = false;
    int N = 0;
    double input_fc_ms = 0.0;
    double attn_ms = 0.0;
    double moe_ms = 0.0;
    double lm_head_ms = 0.0;
    double total_ms = 0.0;
    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~MtpProfile() {
        if (total_start != nullptr) cudaEventDestroy(total_start);
        if (total_stop != nullptr) cudaEventDestroy(total_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
    }

    void ensure_events() {
        if (total_start != nullptr) return;
        CUDA_CHECK(cudaEventCreate(&total_start));
        CUDA_CHECK(cudaEventCreate(&total_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
    }

    void begin(int n, cudaStream_t stream) {
        enabled = mtp_profile_enabled();
        if (!enabled) return;
        ensure_events();
        N = n;
        input_fc_ms = attn_ms = moe_ms = lm_head_ms = total_ms = 0.0;
        CUDA_CHECK(cudaEventRecord(total_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(total_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        total_ms = static_cast<double>(ms);
    }
};

template <class F>
void profile_mtp_stage(
    MtpProfile& profile,
    double& dst,
    cudaStream_t stream,
    F&& fn)
{
    if (!profile.enabled) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile.stage_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile.stage_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile.stage_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile.stage_start, profile.stage_stop));
    dst += static_cast<double>(ms);
}

void maybe_print_mtp_profile(const MtpProfile& p) {
    if (!p.enabled) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = mtp_profile_print_limit();
    if (limit == 0 || seq > limit) return;
    const double named = p.input_fc_ms + p.attn_ms + p.moe_ms + p.lm_head_ms;
    const double other = p.total_ms > named ? p.total_ms - named : 0.0;
    std::cerr
        << "[pie-mtp-profile] seq=" << seq
        << " N=" << p.N
        << " total_ms=" << p.total_ms
        << " input_fc_ms=" << p.input_fc_ms
        << " attn_ms=" << p.attn_ms
        << " moe_ms=" << p.moe_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
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
    ws.shared_gate_up    = DeviceBuffer<std::uint16_t>::alloc(N * (2 * Ish + 1));
    ws.shared_act        = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_out        = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.shared_gate_logit = DeviceBuffer<std::uint16_t>::alloc(N * 1);

    ws.moe_out = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.a_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(maxR);
    ws.b_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(maxR);
    ws.c_gu_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(maxR);
    ws.a_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(maxR);
    ws.b_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(maxR);
    ws.c_dn_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(maxR);
    ws.batch_weights = DeviceBuffer<float>::alloc(maxR);

    ws.aligned_block_size = qwen35_moe_aligned_decode_block_size();
    if (ws.aligned_block_size > 1 && maxR > 0 && num_experts > 0) {
        const std::size_t active_expert_cap =
            std::min<std::size_t>(static_cast<std::size_t>(num_experts), maxR);
        const std::size_t block =
            static_cast<std::size_t>(ws.aligned_block_size);
        const std::size_t max_blocks =
            (maxR + active_expert_cap * (block - 1) + block - 1) / block;
        ws.aligned_rows_capacity = max_blocks * block;
        ws.aligned_route_ids =
            DeviceBuffer<std::int32_t>::alloc(ws.aligned_rows_capacity);
        ws.aligned_expert_ids =
            DeviceBuffer<std::int32_t>::alloc(max_blocks);
        ws.aligned_expert_in =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * H);
        ws.aligned_gate_up =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * 2 * I);
        ws.aligned_act =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * I);
        ws.aligned_out =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * H);
    }
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
    RecurrentStateCache& state_cache,
    int layer_idx, int N, int R, bool is_pure_decode,
    const std::int32_t*  slot_ids_h,
    const std::int32_t*  slot_ids_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* qo_indptr_d,
    ops::CublasHandle& cublas, cudaStream_t stream,
    Qwen35MoeForwardProfile* profile)
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
    const int snapshot_base_slot =
        (R == 1) ? state_cache.spec_snapshot_base_slot() : -1;
    const int snapshot_count =
        snapshot_base_slot >= 0 ? state_cache.spec_snapshot_count() : 0;

    const void* z_data = la.z.data();
    const void* a_data = la.a.data();
    const void* b_data = la.b.data();
    void* qkv_in_data = la.mixed_qkv.data();
    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_proj_ms : nullptr,
        stream, [&] {
            if (Lw.la_in_proj_qkvz != nullptr && Lw.la_in_proj_ba != nullptr) {
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.la_in_proj_qkvz->data(),
                    la.mixed_qkvz.data(), N, conv_dim + V_dim, H);
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.la_in_proj_ba->data(),
                    la.ba.data(), N, 2 * V_h, H);
                kernels::launch_split_qwen_gdn_projections_bf16(
                    la.mixed_qkvz.data(), la.ba.data(),
                    la.mixed_qkv.data(), la.z.data(), la.b.data(), la.a.data(),
                    N, conv_dim, V_dim, V_h, stream);
            } else {
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
            }
        });

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_conv_ms : nullptr,
        stream, [&] {
        auto* qkv_in_base   = static_cast<std::uint16_t*>(qkv_in_data);
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
                kernels::launch_causal_conv1d_prefill_batched_snapshot_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    qkv_post_base,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    slot_ids_d, qo_indptr_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    R, conv_dim, conv_K,
                    snapshot_base_slot, snapshot_count, stream);
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
    });

    auto* qkv_base = la.mixed_qkv_post.data();
    const bool use_warp_tiled_recurrent =
        !is_pure_decode &&
        slot_ids_d != nullptr &&
        qo_indptr_d != nullptr &&
        N <= qwen35_gdn_warp_tiled_max_tokens() &&
        K_d <= 256;
    const bool use_decode_gqa_recurrent =
        is_pure_decode &&
        slot_ids_d != nullptr &&
        V_h != K_h &&
        V_h % K_h == 0;
    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_prep_ms : nullptr,
        stream, [&] {
        kernels::launch_qwen_gdn_post_conv_prep_bf16(
            qkv_base, a_data, b_data,
            Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
            la.q_pre.data(), la.k_pre.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            N, K_h, V_h, K_d, V_d, conv_dim, stream);

        if (V_h != K_h && !use_warp_tiled_recurrent &&
            !use_decode_gqa_recurrent) {
            kernels::launch_repeat_interleave_heads_fp32(
                la.q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d, stream);
            kernels::launch_repeat_interleave_heads_fp32(
                la.k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d, stream);
        }
    });
    const float* q_recur_full =
        (V_h == K_h) ? la.q_pre.data() : la.q_norm.data();
    const float* k_recur_full =
        (V_h == K_h) ? la.k_pre.data() : la.k_norm.data();

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_recur_ms : nullptr,
        stream, [&] {
            const std::size_t qk_step = static_cast<std::size_t>(V_h) * K_d;
            const std::size_t v_step  = static_cast<std::size_t>(V_dim);
            const std::size_t gh_step = static_cast<std::size_t>(V_h);
            const bool state_bf16 = state_cache.recurrent_state_bf16();
            void* state_slot0 = state_cache.recurrent_state_raw(
                layer_idx, /*slot=*/0);
            const auto slot_stride = static_cast<long long>(
                state_cache.recurrent_slot_stride_floats());
            if (is_pure_decode) {
                if (slot_ids_d != nullptr) {
                    if (use_decode_gqa_recurrent) {
                        if (state_bf16) {
                            kernels::launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d, stream);
                        } else {
                            kernels::launch_recurrent_gated_delta_step_batched_gqa(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d, stream);
                        }
                    } else {
                        if (state_bf16) {
                            kernels::launch_recurrent_gated_delta_step_batched_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        } else {
                            kernels::launch_recurrent_gated_delta_step_batched(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        }
                    }
                } else {
                    if (state_bf16) {
                        kernels::launch_recurrent_gated_delta_step_state_bf16(
                            q_recur_full,
                            k_recur_full,
                            la.v_fp32.data(),
                            la.g_log.data(),
                            la.beta.data(),
                            state_slot0,
                            la.core_out.data(),
                            /*B=*/1, V_h, K_d, V_d, stream);
                    } else {
                        kernels::launch_recurrent_gated_delta_step(
                            q_recur_full,
                            k_recur_full,
                            la.v_fp32.data(),
                            la.g_log.data(),
                            la.beta.data(),
                            static_cast<float*>(state_slot0),
                            la.core_out.data(),
                            /*B=*/1, V_h, K_d, V_d, stream);
                    }
                }
            } else {
                if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
                    if (use_warp_tiled_recurrent && V_h != K_h) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot_state_bf16(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else if (use_warp_tiled_recurrent) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else if (N <= qwen35_gdn_cached_prefill_max_tokens()) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached_snapshot_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached_snapshot(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        }
                    }
                } else {
                    for (int r = 0; r < R; ++r) {
                        const int t0 = static_cast<int>(qo_indptr_h[r]);
                        const int Nr = static_cast<int>(qo_indptr_h[r + 1]) - t0;
                        if (Nr <= 0) continue;
                        const std::size_t qk_off = static_cast<std::size_t>(t0) * qk_step;
                        const std::size_t v_off  = static_cast<std::size_t>(t0) * v_step;
                        const std::size_t gh_off = static_cast<std::size_t>(t0) * gh_step;
                        void* state_slot = state_cache.recurrent_state_raw(
                            layer_idx, slot_for(r));
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_state_bf16(
                                q_recur_full + qk_off,
                                k_recur_full + qk_off,
                                la.v_fp32.data() + v_off,
                                la.g_log.data()  + gh_off,
                                la.beta.data()   + gh_off,
                                state_slot,
                                la.core_out.data() + v_off,
                                Nr, V_h, K_d, V_d, /*chunk_size=*/64, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill(
                                q_recur_full + qk_off,
                                k_recur_full + qk_off,
                                la.v_fp32.data() + v_off,
                                la.g_log.data()  + gh_off,
                                la.beta.data()   + gh_off,
                                static_cast<float*>(state_slot),
                                la.core_out.data() + v_off,
                                Nr, V_h, K_d, V_d, /*chunk_size=*/64, stream);
                        }
                    }
                }
            }
        });

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_post_ms : nullptr,
        stream, [&] {
    kernels::launch_fp32_to_bf16(
        la.core_out.data(), la.core_out_bf16.data(),
        (std::size_t)N * V_dim, stream);
    kernels::launch_rmsnorm_gated_bf16(
        la.core_out_bf16.data(), z_data, Lw.la_norm_w_fp32,
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
    });
}

// Full-attention body (replica of qwen3_5_forward.cpp's logic).
void full_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache, AttentionWorkspace& attn_ws,
    const ops::DecodePlanCache* decode_plan,
    const ops::PrefillPlanCache* prefill_plan,
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
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        kernels::launch_split_q_gate_bf16(
            la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
            N, num_q_heads_local, d, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            ws.q.data(), N, Hq, H);
    }

    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
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

    auto kv_view = cache.layer_view(kv_layer);
    kernels::launch_write_kv_to_pages(
        kv_view, ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, stream);

    // Decode and planned-prefill paths are graph-friendly: the host-side
    // FlashInfer planning was hoisted to the executor prepare hook.
    const bool use_small_prefill_naive =
        decode_plan == nullptr &&
        prefill_plan == nullptr &&
        fwd_cfg.small_prefill_naive_attention_max_tokens > 0 &&
        N <= fwd_cfg.small_prefill_naive_attention_max_tokens &&
        kv_view.is_native_bf16() && !kv_view.hnd_layout;
    if (decode_plan) {
        ops::dispatch_attention_flashinfer_decode(
            *decode_plan,
            ws.q.data(), kv_view, ws.attn_out.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else if (prefill_plan) {
        ops::dispatch_attention_flashinfer_prefill_bf16(
            *prefill_plan,
            ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
            ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else if (use_small_prefill_naive) {
        ops::launch_attention_naive_paged_bf16(
            ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
            ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), stream);
    } else {
        ops::launch_attention_flashinfer_prefill(
            ws.q.data(), kv_view, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, attn_ws, stream);
    }

    if (cfg.attn_output_gate) {
        kernels::launch_sigmoid_gate_inplace_bf16(
            ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
    }

    // o_proj: TP=1 fuses residual via beta=1; TP>1 row-parallel +
    // all-reduce + residual-add.
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

// MoE block: routed experts + shared expert with sigmoid gate.
// Reads `ws.norm_x`, writes the combined routed-expert + shared-expert
// contribution directly into `ws.norm_y` (the residual buffer the caller
// will add into `ws.y`).
bool moe_block(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    bool is_pure_decode,
    ops::CublasHandle& cublas, cudaStream_t stream,
    Qwen35MoeForwardProfile* profile)
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
    const bool use_decode_fast_path =
        is_pure_decode ||
        (N > 0 && N <= qwen35_moe_decode_fast_max_tokens());
    const bool add_to_residual = (T == 1) && use_decode_fast_path;
    void* moe_out = add_to_residual ? ws.y.data() : ws.norm_y.data();

    // ── Routed experts ────────────────────────────────────────────
    // 1. Router logits.
    profile_cuda_stage(profile, profile ? &profile->moe_router_ms : nullptr,
        stream, [&] {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), Lw.moe_router->data(),
                moe_ws.router_logits.data(), N, E, H);
            // 2. Top-K + softmax + renormalize.
            kernels::launch_topk_softmax_bf16(
                moe_ws.router_logits.data(),
                moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
                N, E, K, stream);
        });

    // 3. Routing decisions. The default pure-decode path stays entirely
    //    on-device (so the layer is graph-capturable). The prefill/mixed
    //    path needs host routing to bucket tokens per expert.
    std::vector<std::int32_t> topk_idx_h;
    std::vector<float>        topk_w_h;
    if (!use_decode_fast_path) {
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

    // 4. (For the prefill/mixed path only: zero moe_out before scatter_add.)
    //    Decode fast-path weighted-sum overwrites norm_y, so the memset
    //    there would be wasted work.

    // 5. Per-expert dispatch.
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert in gate_up_proj
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;       // bf16 elements per expert in down_proj

    if (use_decode_fast_path) {
        // Decode fast-path. Fully on-device pipeline (graph-capturable):
        //   1. Build gate_up/down cuBLAS pointer arrays for every
        //      token/expert route (N*K rows) with no D2H sync.
        //   2. `cublasGemmBatchedEx` for gate_up (N*K batches, M=1).
        //   3. `chunked_swiglu` over [N*K, 2*Im].
        //   4. `cublasGemmBatchedEx` for down_proj (N*K batches, M=1).
        //   5. Weighted sum collapses [N, K, H] -> [N, H].
        //
        // Every step has fixed kernel topology and stable device-pointer
        // arguments, so the executor's graph-capture path can fire
        // for the whole forward.
        profile_cuda_stage(profile, profile ? &profile->moe_routed_ms : nullptr,
            stream, [&] {
                const int routes = N * K;
                const int block = moe_ws.aligned_block_size;
                const bool use_aligned_decode =
                    block > 1 &&
                    routes >= qwen35_moe_aligned_decode_min_routes() &&
                    !moe_ws.aligned_expert_in.empty();
                if (use_aligned_decode) {
                    const int active_expert_cap = std::min(E, routes);
                    const int max_blocks =
                        (routes + active_expert_cap * (block - 1) +
                         block - 1) / block;
                    const int aligned_rows = max_blocks * block;
                    if (static_cast<std::size_t>(aligned_rows) >
                        moe_ws.aligned_rows_capacity) {
                        throw std::runtime_error(
                            "qwen3.5-moe aligned decode scratch too small");
                    }

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_route_setup_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_align_decode(
                                moe_ws.topk_idx.data(),
                                moe_ws.aligned_route_ids.data(),
                                moe_ws.aligned_expert_ids.data(),
                                /*route_to_aligned_row=*/nullptr,
                                routes, E, block, max_blocks, stream);
                            kernels::launch_gather_moe_aligned_inputs_bf16(
                                ws.norm_x.data(), moe_ws.aligned_route_ids.data(),
                                moe_ws.aligned_expert_in.data(),
                                routes, aligned_rows, K, H, stream);
                            kernels::launch_build_moe_ptrs_aligned_bf16(
                                moe_ws.aligned_expert_ids.data(),
                                Lw.moe_gate_up_proj->data(),
                                Lw.moe_down_proj->data(),
                                moe_ws.aligned_expert_in.data(),
                                moe_ws.aligned_gate_up.data(),
                                moe_ws.aligned_act.data(),
                                moe_ws.aligned_out.data(),
                                reinterpret_cast<const void**>(moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_dn_ptrs.data()),
                                max_blocks, block, H, Im, stream);
                        });

                    // Aligned gate_up: M=block_size, N=2*Im, K=H.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
                                block, 2 * Im, H, max_blocks);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.aligned_gate_up.data(),
                                moe_ws.aligned_act.data(),
                                aligned_rows, Im, stream);
                        });

                    // Aligned down_proj: M=block_size, N=H, K=Im.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_dn_ptrs.data()),
                                block, H, Im, max_blocks);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            kernels::launch_reorder_moe_aligned_output_bf16(
                                moe_ws.aligned_out.data(),
                                moe_ws.aligned_route_ids.data(),
                                moe_ws.expert_out.data(),
                                routes, aligned_rows, H, stream);
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                } else if (qwen35_moe_wmma_decode_enabled() &&
                           (H % 16) == 0 && (Im % 16) == 0) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_gate_up_decode_wmma_bf16(
                                moe_ws.topk_idx.data(),
                                ws.norm_x.data(),
                                Lw.moe_gate_up_proj->data(),
                                moe_ws.expert_gate_up.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                routes, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_down_decode_wmma_bf16(
                                moe_ws.topk_idx.data(),
                                moe_ws.expert_act.data(),
                                Lw.moe_down_proj->data(),
                                moe_ws.expert_out.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                } else {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_route_setup_ms : nullptr,
                        stream, [&] {
                            kernels::launch_build_moe_ptrs_decode_batched_bf16(
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
                                N, K, H, Im, stream);
                        });

                    // gate_up batched GEMM: M=1, N=2*Im, K=H, batch=N*top_k.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
                                /*M=*/1, /*N=*/2 * Im, /*K=*/H,
                                /*batch_count=*/routes);
                        });

                    // SwiGLU on [N*top_k, 2*Im] -> [N*top_k, Im].
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                routes, Im, stream);
                        });

                    // down_proj batched GEMM: M=1, N=H, K=Im, batch=N*top_k.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_dn_ptrs.data()),
                                /*M=*/1, /*N=*/H, /*K=*/Im,
                                /*batch_count=*/routes);
                        });

                    // Sum each token's K routed outputs into norm_y.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.batch_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.batch_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                }
            });
    } else {
        // General path (prefill / multi-token). Build per-expert routing
        // lists on host and gather/scatter via the existing kernels.
        // Zero moe_out before the scatter_add accumulation.
        profile_cuda_stage(profile, profile ? &profile->moe_routed_ms : nullptr,
            stream, [&] {
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
            });
    }

    // ── Shared expert (Qwen3.5 / 3.6-MoE: always-on dense MLP + sigmoid
    //    gate). Qwen3-MoE has no shared expert — skip the whole block
    //    when the bind didn't wire `shared_*` pointers (Is == 0).
    if (Is > 0 && Lw.shared_gate_proj != nullptr) {
        profile_cuda_stage(profile, profile ? &profile->moe_shared_ms : nullptr,
            stream, [&] {
                const bool fused_shared_scalar_gate =
                    Lw.shared_gate_up_gate_proj != nullptr;
                if (fused_shared_scalar_gate) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                ops::WeightView(*Lw.shared_gate_up_gate_proj),
                                moe_ws.shared_gate_up.data(), N, 2 * Is + 1, H);
                            kernels::launch_chunked_swiglu_strided_bf16(
                                moe_ws.shared_gate_up.data(),
                                moe_ws.shared_act.data(), N, Is, 2 * Is + 1, stream);
                        });
                } else if (Lw.shared_gate_up_proj != nullptr) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_proj),
                                moe_ws.shared_gate_up.data(), N, 2 * Is, H);
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.shared_gate_up.data(),
                                moe_ws.shared_act.data(), N, Is, stream);
                        });
                } else {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(
                                    Lw.shared_gate_proj, Lw.shared_gate_proj_quant),
                                moe_ws.shared_gate.data(), N, Is, H);
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(
                                    Lw.shared_up_proj, Lw.shared_up_proj_quant),
                                moe_ws.shared_up.data(), N, Is, H);
                            kernels::launch_swiglu_bf16(
                                moe_ws.shared_gate.data(), moe_ws.shared_up.data(),
                                moe_ws.shared_act.data(),
                                N * Is, stream);
                        });
                }
                profile_cuda_detail_stage(
                    profile, profile ? &profile->moe_shared_down_ms : nullptr,
                    stream, [&] {
                        ops::gemm_act_x_w(cublas.handle(),
                            moe_ws.shared_act.data(),
                            make_weight_view(
                                Lw.shared_down_proj, Lw.shared_down_proj_quant),
                            moe_ws.shared_out.data(), N, H, Is);
                    });

                // shared_gate logit [N, 1] = norm_x @ shared_gate.weight.T
                profile_cuda_detail_stage(
                    profile, profile ? &profile->moe_shared_gate_ms : nullptr,
                    stream, [&] {
                        if (fused_shared_scalar_gate) {
                            const auto* scalar_gate =
                                moe_ws.shared_gate_up.data() +
                                static_cast<std::size_t>(2 * Is);
                            kernels::launch_sigmoid_scalar_gate_strided_add_bf16(
                                moe_out, moe_ws.shared_out.data(),
                                scalar_gate,
                                N, H, 2 * Is + 1, stream);
                        } else if (Lw.shared_gate != nullptr &&
                                   N <= qwen35_moe_decode_fast_max_tokens() &&
                                   !Lw.shared_gate_quant.has_value()) {
                            kernels::launch_sigmoid_dot_scalar_gate_add_bf16(
                                ws.norm_x.data(),
                                Lw.shared_gate->data(),
                                moe_out,
                                moe_ws.shared_out.data(),
                                N, H, stream);
                        } else {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(Lw.shared_gate, Lw.shared_gate_quant),
                                moe_ws.shared_gate_logit.data(), N, 1, H);

                            // shared_out *= sigmoid(scalar_gate[n]) per token,
                            // broadcast across all H channels.
                            kernels::launch_sigmoid_scalar_gate_add_bf16(
                                moe_out, moe_ws.shared_out.data(),
                                moe_ws.shared_gate_logit.data(),
                                N, H, stream);
                        }
                    });
            });
    }

    if (T > 1) {
        profile_cuda_stage(profile, profile ? &profile->moe_allreduce_ms : nullptr,
            stream, [&] {
                tp->all_reduce_bf16(ws.norm_y.data(),
                    (std::size_t)N * H, ncclSum, stream);
            });
    }
    return add_to_residual;
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
    int total_tokens, int num_requests,
    bool is_pure_decode,
    const std::uint8_t* /*mask_d*/,
    const std::int32_t* /*mask_indptr_d*/,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    // Pure-Qwen3-MoE (Qwen3-30B-A3B, model_type == "qwen3_moe") has no
    // linear-attn layers; the per-slot rs_cache is unused. Qwen3.5 /
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
    cudaStream_t stream = cublas.stream();
    Qwen35MoeForwardProfile profile;
    const int tp_rank = (fwd_cfg.tp_comm != nullptr) ? fwd_cfg.tp_comm->rank() : 0;
    profile.begin(N, R, is_pure_decode, tp_rank, stream);

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
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        (plan_state.use_prefill_plan && plan_state.prefill_plan)
            ? plan_state.prefill_plan.get()
            : nullptr;

    profile_cuda_stage(&profile, &profile.embed_ms, stream, [&] {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(),
            N, H, cfg.vocab_size, stream);
    });

    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];

        profile_cuda_stage(&profile, &profile.norm_ms, stream, [&] {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });

        if (Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn) {
            ++profile.linear_layers;
            profile_cuda_stage(&profile, &profile.linear_attn_ms, stream, [&] {
                linear_attn_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                    static_cast<int>(L), N, R, is_pure_decode,
                    slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                    cublas, stream, &profile);
            });
        } else {
            ++profile.full_layers;
            profile_cuda_stage(&profile, &profile.full_attn_ms, stream, [&] {
                full_attn_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, cache, attn_ws,
                    decode_plan, prefill_plan, Lw.kv_layer,
                    N, num_requests,
                    positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                    cublas, stream);
            });
        }
        // (Post-attention residual fused into the body's final GEMM
        //  via beta=1 on TP=1; on TP>1 the body did the all-reduce and
        //  residual_add itself. ws.y holds the post-attention state.)

        // Post-attention norm + MoE block + residual.
        profile_cuda_stage(&profile, &profile.norm_ms, stream, [&] {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });
        ++profile.moe_layers;
        const bool moe_added_to_residual = moe_block(
            Lw, cfg, fwd_cfg, ws, moe_ws, N, is_pure_decode,
            cublas, stream, &profile);
        if (!moe_added_to_residual) {
            profile_cuda_stage(&profile, &profile.residual_ms, stream, [&] {
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_y.data(),
                    (std::size_t)N * H, stream);
            });
        }
    }

    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        int lm_head_rows = N;
        const void* lm_head_input = ws.norm_x.data();
        if (compact_logits) {
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.y.data()),
                logit_row_indices_d,
                static_cast<std::uint16_t*>(ws.norm_y.data()),
                num_logit_rows, H, stream);
            rmsnorm_bf16_dispatch(cfg,
                ws.norm_y.data(), w.final_norm->data(), ws.norm_x.data(),
                num_logit_rows, H, eps, stream);
            lm_head_rows = num_logit_rows;
        } else {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
        }
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            lm_head_input, w.lm_head->data(),
            ws.logits.data(), lm_head_rows, V, H);
        if (!compact_logits) {
            CUDA_CHECK(cudaMemcpyAsync(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice, stream));
        }
    });
    profile.end(stream);
    maybe_print_profile(profile);
}

namespace {

void mtp_full_attn_no_cache_moe(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    int N,
    int draft_step,
    const std::int32_t* position_ids,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int max_global_tokens,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int q_heads = cfg.num_attention_heads / T;
    const int kv_heads = cfg.num_key_value_heads / T;
    const int Hq = q_heads * cfg.head_dim;
    const int Hk = kv_heads * cfg.head_dim;
    const int d = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const std::size_t kv_step_offset =
        static_cast<std::size_t>(draft_step) * N * Hk;
    auto* k_step = static_cast<std::uint16_t*>(ws.k.data()) + kv_step_offset;
    auto* v_step = static_cast<std::uint16_t*>(ws.v.data()) + kv_step_offset;

    if (cfg.attn_output_gate) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        kernels::launch_split_q_gate_bf16(
            la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
            N, q_heads, cfg.head_dim, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            ws.q.data(), N, Hq, H);
    }

    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        k_step, N, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        v_step, N, Hk, H);
    rmsnorm_bf16_dispatch(cfg,
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * q_heads, d, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        k_step, Lw.fa_k_norm->data(), k_step,
        N * kv_heads, d, eps, stream);
    kernels::launch_rope_partial_bf16(
        ws.q.data(), k_step, position_ids,
        N, q_heads, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
    const auto mtp_kv = cache.layer_view(Lw.kv_layer);
    ops::launch_attention_mtp_paged_history_bf16(
        ws.q.data(), mtp_kv.k_bf16_pages, mtp_kv.v_bf16_pages,
        ws.k.data(), ws.v.data(), ws.attn_out.data(),
        position_ids, request_ids,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, draft_step + 1, N, max_global_tokens, cache.page_size(),
        q_heads, kv_heads, d, mtp_kv.hnd_layout,
        fwd_cfg.mtp_global_cache_uses_prefix_position, stream);
    if (cfg.attn_output_gate) {
        kernels::launch_sigmoid_gate_inplace_bf16(
            ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
    }

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

bool mtp_shared_expert_only_moe(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int Is = cfg.shared_expert_intermediate_size / T;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    if (Is <= 0 || Lw.shared_gate_proj == nullptr) return false;

    const bool fused_shared_scalar_gate =
        Lw.shared_gate_up_gate_proj != nullptr;
    if (fused_shared_scalar_gate) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_gate_proj),
            moe_ws.shared_gate_up.data(), N, 2 * Is + 1, H);
        kernels::launch_chunked_swiglu_strided_bf16(
            moe_ws.shared_gate_up.data(), moe_ws.shared_act.data(),
            N, Is, 2 * Is + 1, stream);
    } else if (Lw.shared_gate_up_proj != nullptr) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_proj),
            moe_ws.shared_gate_up.data(), N, 2 * Is, H);
        kernels::launch_chunked_swiglu_bf16(
            moe_ws.shared_gate_up.data(), moe_ws.shared_act.data(),
            N, Is, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_gate_proj, Lw.shared_gate_proj_quant),
            moe_ws.shared_gate.data(), N, Is, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_up_proj, Lw.shared_up_proj_quant),
            moe_ws.shared_up.data(), N, Is, H);
        kernels::launch_swiglu_bf16(
            moe_ws.shared_gate.data(), moe_ws.shared_up.data(),
            moe_ws.shared_act.data(), N * Is, stream);
    }
    ops::gemm_act_x_w(cublas.handle(),
        moe_ws.shared_act.data(),
        make_weight_view(Lw.shared_down_proj, Lw.shared_down_proj_quant),
        moe_ws.shared_out.data(), N, H, Is);
    if (fused_shared_scalar_gate) {
        const auto* scalar_gate =
            moe_ws.shared_gate_up.data() + static_cast<std::size_t>(2 * Is);
        kernels::launch_sigmoid_scalar_gate_strided_inplace_bf16(
            moe_ws.shared_out.data(), scalar_gate, N, H, 2 * Is + 1, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_gate, Lw.shared_gate_quant),
            moe_ws.shared_gate_logit.data(), N, 1, H);
        kernels::launch_sigmoid_scalar_gate_inplace_bf16(
            moe_ws.shared_out.data(), moe_ws.shared_gate_logit.data(),
            N, H, stream);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        ws.norm_y.data(), moe_ws.shared_out.data(),
        static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
    if (T > 1) {
        tp->all_reduce_bf16(
            ws.norm_y.data(), static_cast<std::size_t>(N) * H,
            ncclSum, stream);
    }
    return true;
}

}  // namespace

void qwen3_5_moe_mtp_process_cache(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::int32_t* slot_ids_d,
    const std::int32_t* source_row_indices,
    int total_tokens,
    int num_requests)
{
    if (!w.mtp || total_tokens <= 0 || num_requests <= 0) return;
    const auto& mtp = *w.mtp;
    const auto& Lw = mtp.layer;
    if (Lw.kv_layer < 0) return;

    const int H = cfg.hidden_size;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int kv_heads = cfg.num_key_value_heads / T;
    const int Hk = kv_heads * cfg.head_dim;
    const int d = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    void* pending = state_cache.mtp_pending_hidden(0);
    const void* target_hidden = ws.y.data();
    if (source_row_indices != nullptr) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            source_row_indices,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            total_tokens, H, stream);
        target_hidden = ws.norm_x.data();
    }
    ops::launch_mtp_shift_hidden_bf16(
        target_hidden, pending, qo_indptr, slot_ids_d, ws.norm_y.data(),
        total_tokens, num_requests, H, stream);
    ops::launch_mtp_update_pending_hidden_bf16(
        target_hidden, pending, qo_indptr, slot_ids_d, num_requests, H, stream);
    kernels::launch_embed_bf16(
        token_ids, mtp.embed->data(), ws.norm_x.data(),
        total_tokens, H, cfg.vocab_size, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        total_tokens, H, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), mtp.pre_fc_norm_hidden->data(), ws.attn_out.data(),
        total_tokens, H, eps, stream);
    kernels::launch_concat_bf16_rows(
        ws.q.data(), ws.attn_out.data(), ws.mtp_concat.data(),
        total_tokens, H, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.mtp_concat.data(), *mtp.fc, ws.norm_y.data(),
        total_tokens, H, 2 * H);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        total_tokens, H, eps, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), total_tokens, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        ws.v.data(), total_tokens, Hk, H);
    rmsnorm_bf16_dispatch(cfg,
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        total_tokens * kv_heads, d, eps, stream);
    kernels::launch_rope_partial_bf16(
        /*q=*/nullptr, ws.k.data(), positions,
        total_tokens, 0, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
    kernels::launch_write_kv_to_pages(
        cache.layer_view(Lw.kv_layer),
        ws.k.data(), ws.v.data(), qo_indptr, kv_page_indices,
        kv_page_indptr, kv_last_page_lens, total_tokens, num_requests,
        stream);
}

void qwen3_5_moe_mtp_forward(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::int32_t* sampled_token_ids,
    int num_tokens,
    int draft_step,
    int max_global_tokens)
{
    (void)sampled_token_ids;
    if (!w.mtp || num_tokens <= 0) return;
    const auto& mtp = *w.mtp;
    const auto& Lw = mtp.layer;
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    MtpProfile profile;
    profile.begin(num_tokens, stream);

    profile_mtp_stage(profile, profile.input_fc_ms, stream, [&] {
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(ws.y.data()),
        base_hidden_row_indices,
        static_cast<std::uint16_t*>(ws.norm_y.data()),
        num_tokens, H, stream);
    kernels::launch_embed_bf16(
        token_ids, mtp.embed->data(), ws.norm_x.data(),
        num_tokens, H, cfg.vocab_size, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        num_tokens, H, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), mtp.pre_fc_norm_hidden->data(), ws.y.data(),
        num_tokens, H, eps, stream);
    kernels::launch_concat_bf16_rows(
        ws.q.data(), ws.y.data(), ws.mtp_concat.data(),
        num_tokens, H, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.mtp_concat.data(), *mtp.fc, ws.y.data(),
        num_tokens, H, 2 * H);
    });

    profile_mtp_stage(profile, profile.attn_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    mtp_full_attn_no_cache_moe(
        Lw, cfg, fwd_cfg, ws, la_ws, cache, num_tokens, draft_step,
        position_ids, request_ids, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, max_global_tokens, cublas, stream);
    });

    profile_mtp_stage(profile, profile.moe_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    const MtpMoeMode mode = mtp_moe_mode();
    bool add_moe_residual = false;
    if (mode == MtpMoeMode::Full) {
        const bool moe_added_to_residual = moe_block(
            Lw, cfg, fwd_cfg, ws, moe_ws, num_tokens,
            /*is_pure_decode=*/true, cublas, stream, /*profile=*/nullptr);
        add_moe_residual = !moe_added_to_residual;
    } else if (mode == MtpMoeMode::SharedOnly) {
        add_moe_residual = mtp_shared_expert_only_moe(
            Lw, cfg, fwd_cfg, ws, moe_ws, num_tokens, cublas, stream);
    }
    if (add_moe_residual) {
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(num_tokens) * H, stream);
    }
    });

    profile_mtp_stage(profile, profile.lm_head_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), mtp.norm->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(),
        ws.logits.data(), num_tokens, V, H);
    CUDA_CHECK(cudaMemcpyAsync(
        ws.y.data(), ws.norm_x.data(),
        static_cast<std::size_t>(num_tokens) * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
    });
    profile.end(stream);
    maybe_print_mtp_profile(profile);
}

std::size_t qwen3_5_moe_workspace_bytes(const HfConfig& cfg,
                                        int N, int tp_size) {
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.moe_intermediate_size <= 0) {
        return 0;
    }
    const int T = std::max(1, tp_size);
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t maxR = n * cfg.num_experts_per_tok;
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t I =
        static_cast<std::size_t>(cfg.moe_intermediate_size / T);
    const std::size_t Ish =
        static_cast<std::size_t>(
            std::max(0, cfg.shared_expert_intermediate_size / T));
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto i32 = [](std::size_t elems) { return elems * 4; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    auto aligned_decode_block = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK");
        if (v == nullptr || v[0] == '\0') return 16;
        char* end = nullptr;
        long parsed_long = std::strtol(v, &end, 10);
        if (end == v) return 16;
        int parsed = static_cast<int>(parsed_long);
        if (parsed <= 1) return 0;
        if (parsed < 4) parsed = 4;
        if (parsed > 64) parsed = 64;
        return parsed;
    };
    std::size_t bytes = 0;
    bytes += u16(n * cfg.num_experts);
    bytes += i32(n * cfg.num_experts_per_tok);
    bytes += fp32(n * cfg.num_experts_per_tok);
    bytes += u16(maxR * H);
    bytes += u16(maxR * 2 * I);
    bytes += u16(maxR * I);
    bytes += u16(maxR * H);
    bytes += i32(maxR);
    bytes += fp32(maxR);
    bytes += u16(n * Ish);
    bytes += u16(n * Ish);
    bytes += u16(n * Ish);
    bytes += u16(n * H);
    bytes += u16(n);
    bytes += u16(n * H);
    bytes += maxR * (6 * sizeof(void*) + sizeof(float));
    const int aligned_block = aligned_decode_block();
    if (aligned_block > 1 && maxR > 0) {
        const std::size_t block = static_cast<std::size_t>(aligned_block);
        const std::size_t active_expert_cap =
            std::min<std::size_t>(static_cast<std::size_t>(cfg.num_experts), maxR);
        const std::size_t max_blocks =
            (maxR + active_expert_cap * (block - 1) + block - 1) / block;
        const std::size_t aligned_rows = max_blocks * block;
        bytes += i32(aligned_rows);
        bytes += i32(max_blocks);
        bytes += u16(aligned_rows * H);
        bytes += u16(aligned_rows * 2 * I);
        bytes += u16(aligned_rows * I);
        bytes += u16(aligned_rows * H);
    }
    return bytes;
}

}  // namespace pie_cuda_driver::model
