#include "model/kimi/kimi_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/argmax.hpp"
#include "kernels/dequant_wna16.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kimi_mla.hpp"
#include "kernels/mla_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"

namespace pie_cuda_driver::model {

namespace {

bool kimi_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_KIMI_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t kimi_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_KIMI_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool kimi_profile_all_ranks() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_KIMI_PROFILE_ALL_RANKS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool kimi_dump_logits_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_KIMI_DUMP_LOGITS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

void dump_top_logits(const void* logits_bf16, int rows, int cols,
                     int tp_rank, int vocab_offset, cudaStream_t stream) {
    if (!kimi_dump_logits_enabled() || tp_rank != 0) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    std::vector<std::uint16_t> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), logits_bf16,
        n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        std::vector<std::pair<float, int>> vals;
        for (int c = 0; c < cols; ++c) {
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(
                &host[static_cast<std::size_t>(r) * cols + c]);
            vals.emplace_back(__bfloat162float(*p), vocab_offset + c);
        }
        std::sort(vals.begin(), vals.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
        std::cerr << "[pie-logits] rank=" << tp_rank << " row=" << r;
        for (int i = 0; i < std::min(10, static_cast<int>(vals.size())); ++i) {
            std::cerr << " " << vals[i].second << ":" << vals[i].first;
        }
        std::cerr << "\n";
    }
}

void dump_hidden_norm(const void* hidden_bf16, int tokens, int hidden,
                      int layer, const char* tag, int tp_rank,
                      cudaStream_t stream) {
    if (!kimi_dump_logits_enabled() || tp_rank != 0) return;
    if (tokens > 1) return;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
    std::vector<std::uint16_t> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), hidden_bf16,
        n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    double sum_sq = 0.0;
    float max_abs = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        const float v = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&host[i]));
        sum_sq += static_cast<double>(v) * v;
        max_abs = std::max(max_abs, std::abs(v));
    }
    const float rms = std::sqrt(static_cast<float>(sum_sq / n));
    std::cerr << "[pie-hidden] layer=" << layer
              << " tag=" << tag
              << " rms=" << rms
              << " max=" << max_abs
              << "\n";
}

struct KimiForwardProfile {
    bool enabled = false;
    int tp_rank = 0;
    int N = 0;
    int R = 0;
    bool pure_decode = false;
    int dense_layers = 0;
    int moe_layers = 0;

    double embed_ms = 0.0;
    double attn_ms = 0.0;
    double dense_mlp_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_gate_up_ms = 0.0;
    double moe_swiglu_ms = 0.0;
    double moe_down_ms = 0.0;
    double moe_weighted_sum_ms = 0.0;
    double moe_prefill_ms = 0.0;
    double moe_shared_ms = 0.0;
    double moe_allreduce_ms = 0.0;
    double residual_ms = 0.0;
    double lm_head_ms = 0.0;
    double forward_ms = 0.0;

    cudaEvent_t forward_start = nullptr;
    cudaEvent_t forward_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~KimiForwardProfile() {
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
        enabled = kimi_profile_enabled();
        if (!enabled) return;
        ensure_events();
        tp_rank = rank;
        N = n;
        R = r;
        pure_decode = decode;
        dense_layers = 0;
        moe_layers = 0;
        embed_ms = attn_ms = dense_mlp_ms = 0.0;
        moe_router_ms = moe_gate_up_ms = moe_swiglu_ms = moe_down_ms = 0.0;
        moe_weighted_sum_ms = moe_prefill_ms = moe_shared_ms = moe_allreduce_ms = 0.0;
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
};

template <class F>
void profile_cuda_stage(
    KimiForwardProfile* profile,
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

void maybe_print_profile(const KimiForwardProfile& p) {
    if (!p.enabled) return;
    if (p.tp_rank != 0 && !kimi_profile_all_ranks()) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = kimi_profile_print_limit();
    if (limit == 0 || seq > limit) return;

    const double named =
        p.embed_ms + p.attn_ms + p.dense_mlp_ms + p.moe_router_ms +
        p.moe_gate_up_ms + p.moe_swiglu_ms + p.moe_down_ms +
        p.moe_weighted_sum_ms + p.moe_prefill_ms + p.moe_shared_ms +
        p.moe_allreduce_ms + p.residual_ms + p.lm_head_ms;
    const double other = p.forward_ms > named ? p.forward_ms - named : 0.0;
    std::cerr
        << "[pie-kimi-profile] seq=" << seq
        << " rank=" << p.tp_rank
        << " N=" << p.N
        << " R=" << p.R
        << " decode=" << (p.pure_decode ? 1 : 0)
        << " layers_dense=" << p.dense_layers
        << " layers_moe=" << p.moe_layers
        << " total_ms=" << p.forward_ms
        << " embed_ms=" << p.embed_ms
        << " attn_ms=" << p.attn_ms
        << " dense_mlp_ms=" << p.dense_mlp_ms
        << " moe_router_ms=" << p.moe_router_ms
        << " moe_gate_up_ms=" << p.moe_gate_up_ms
        << " moe_swiglu_ms=" << p.moe_swiglu_ms
        << " moe_down_ms=" << p.moe_down_ms
        << " moe_weighted_sum_ms=" << p.moe_weighted_sum_ms
        << " moe_prefill_ms=" << p.moe_prefill_ms
        << " moe_shared_ms=" << p.moe_shared_ms
        << " moe_allreduce_ms=" << p.moe_allreduce_ms
        << " residual_ms=" << p.residual_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
}

struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>> weights;
};

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx,
    const std::vector<float>& topk_w,
    int N,
    int K,
    int E)
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

void dequant_expert_w4(
    const KimiExpertWeights& e,
    KimiWorkspace& ws,
    int H,
    int I,
    cudaStream_t stream)
{
    constexpr int group = 32;
    kernels::launch_dequant_wna16_int4b8_to_bf16(
        static_cast<const std::int32_t*>(e.gate_packed->data()),
        e.gate_scale->data(), ws.expert_gate_w.data(), I, H, group, stream);
    kernels::launch_dequant_wna16_int4b8_to_bf16(
        static_cast<const std::int32_t*>(e.up_packed->data()),
        e.up_scale->data(), ws.expert_up_w.data(), I, H, group, stream);
    kernels::launch_dequant_wna16_int4b8_to_bf16(
        static_cast<const std::int32_t*>(e.down_packed->data()),
        e.down_scale->data(), ws.expert_down_w.data(), H, I, group, stream);
}

}  // namespace

KimiWorkspace KimiWorkspace::allocate(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size)
{
    const int T = std::max(1, tp_size);
    const int N = std::max(1, max_tokens);
    const int O = std::max(1, max_logit_rows > 0 ? max_logit_rows : max_tokens);
    const int H = cfg.hidden_size;
    const int local_heads = cfg.num_attention_heads / T;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;
    const int dense_I =
        cfg.intermediate_size > 0 ? cfg.intermediate_size / T : 0;
    const int routed_I =
        cfg.moe_intermediate_size > 0 ? cfg.moe_intermediate_size / T : 0;
    const int shared_I =
        cfg.shared_expert_intermediate_size > 0
            ? cfg.shared_expert_intermediate_size / T
            : 0;
    const int max_I = std::max(1, std::max(dense_I, routed_I));
    const int Ktop = std::max(1, cfg.num_experts_per_tok);
    const int routes = N * Ktop;

    if (H <= 0 || local_heads <= 0 || q_nope <= 0 || q_rope <= 0 ||
        v_dim <= 0 || q_lora <= 0 || kv_lora <= 0) {
        throw std::runtime_error("kimi: cannot allocate workspace with unset dimensions");
    }

    KimiWorkspace ws;
    ws.y             = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_x        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.q_a           = DeviceTensor::allocate(DType::BF16, {N, q_lora});
    ws.q_b           = DeviceTensor::allocate(DType::BF16, {N, local_heads * (q_nope + q_rope)});
    ws.q_nope        = DeviceTensor::allocate(DType::BF16, {N, local_heads * q_nope});
    ws.kv_a_mqa      = DeviceTensor::allocate(DType::BF16, {N, kv_lora + q_rope});
    ws.kv_c          = DeviceTensor::allocate(DType::BF16, {N, kv_lora});
    ws.k_pe          = DeviceTensor::allocate(DType::BF16, {N, q_rope});
    ws.q_nope_latent = DeviceTensor::allocate(DType::BF16, {N, local_heads * kv_lora});
    ws.q_pe          = DeviceTensor::allocate(DType::BF16, {N, local_heads * q_rope});
    ws.attn_latent   = DeviceTensor::allocate(DType::BF16, {N, local_heads * kv_lora});
    ws.attn_v        = DeviceTensor::allocate(DType::BF16, {N, local_heads * v_dim});
    ws.attn_out      = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_y        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.gate          = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.up            = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.expert_gate_w = DeviceTensor::allocate(DType::BF16, {std::max(1, routed_I), H});
    ws.expert_up_w   = DeviceTensor::allocate(DType::BF16, {std::max(1, routed_I), H});
    ws.expert_down_w = DeviceTensor::allocate(DType::BF16, {H, std::max(1, routed_I)});
    ws.router_logits = DeviceTensor::allocate(DType::BF16, {N, std::max(1, cfg.num_experts)});
    ws.topk_idx      = DeviceTensor::allocate(DType::INT32, {N, Ktop});
    ws.topk_weights  = DeviceTensor::allocate(DType::FP32, {N, Ktop});
    ws.route_idx     = DeviceTensor::allocate(DType::INT32, {routes});
    ws.route_w       = DeviceTensor::allocate(DType::FP32, {routes});
    ws.expert_in     = DeviceTensor::allocate(DType::BF16, {routes, H});
    ws.expert_gate   = DeviceTensor::allocate(DType::BF16, {routes, max_I});
    ws.expert_up     = DeviceTensor::allocate(DType::BF16, {routes, max_I});
    ws.expert_out    = DeviceTensor::allocate(DType::BF16, {routes, H});
    ws.moe_out       = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.shared_gate   = DeviceTensor::allocate(DType::BF16, {N, std::max(1, 2 * shared_I)});
    ws.shared_up     = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_act    = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_out    = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.logits        = DeviceTensor::allocate(DType::BF16, {O, cfg.vocab_size});
    ws.probs         = DeviceTensor::allocate(DType::FP32, {O, cfg.vocab_size});
    return ws;
}

void prepare_kimi_mla_plan(
    KimiPlanState& state,
    AttentionWorkspace& attn_ws,
    const MlaCache& cache,
    const HfConfig& cfg,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_h,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    bool causal,
    int tp_size)
{
    if (!state.mla_plan) state.mla_plan = ops::make_mla_plan();
    if (kimi_dump_logits_enabled()) {
        static int plan_seq = 0;
        int seq = plan_seq++;
        int pages = kv_page_indptr_h[1] - kv_page_indptr_h[0];
        int page_sz = cache.page_size();
        int kv_len = (pages - 1) * page_sz + kv_last_page_lens_h[0];
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[pie-plan] seq=%d N=%d R=%d causal=%d kv_len=%d\n",
            seq, total_tokens, num_requests, causal ? 1 : 0, kv_len);
        write(2, buf, std::strlen(buf));
    }
    ops::plan_attention_mla_bf16(
        *state.mla_plan,
        qo_indptr_h,
        kv_page_indptr_h,
        kv_last_page_lens_h,
        total_tokens,
        num_requests,
        cfg.num_attention_heads / std::max(1, tp_size),
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
        cache.page_size(),
        attn_ws,
        0,
        causal,
        1.0f / std::sqrt(static_cast<float>(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)));
    (void)kv_page_indices_d;
    (void)kv_page_indptr_d;
    (void)kv_last_page_lens_d;
}

void kimi_forward_paged(
    const KimiWeights& w,
    const HfConfig& cfg,
    const KimiForwardCfg& fwd_cfg,
    const KimiPlanState& plan_state,
    KimiWorkspace& kimi_ws,
    MlaCache& mla_cache,
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
    (void)qo_indptr_h;
    (void)kv_page_indptr_h;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const int heads = cfg.num_attention_heads / T;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int dense_I = cfg.intermediate_size / T;
    const int routed_I = cfg.moe_intermediate_size / T;
    const int shared_I = cfg.shared_expert_intermediate_size / T;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    KimiForwardProfile profile;
    profile.begin(total_tokens, num_requests, is_pure_decode,
        tp != nullptr ? tp->rank() : 0, stream);

    profile_cuda_stage(&profile, &profile.embed_ms, stream, [&] {
        if (w.embed_tp_sharded) {
            if (tp == nullptr) {
                throw std::runtime_error("kimi: sharded embed requires TP communicator");
            }
            kernels::launch_embed_bf16_vocab_shard(
                token_ids, w.embed->data(), kimi_ws.y.data(),
                total_tokens, H, static_cast<int>(w.embed->shape()[0]),
                w.embed_tp_vocab_offset, stream);
            tp->all_reduce_bf16(kimi_ws.y.data(),
                static_cast<std::size_t>(total_tokens) * static_cast<std::size_t>(H),
                ncclSum, stream);
            if (kimi_dump_logits_enabled() && (tp == nullptr || tp->rank() == 0)) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                // Dump positions
                std::vector<std::int32_t> pos_h(total_tokens);
                CUDA_CHECK(cudaMemcpy(pos_h.data(), positions,
                    total_tokens * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                std::cerr << "[pie-pos] N=" << total_tokens << " positions:";
                for (int i = 0; i < std::min(total_tokens, 5); ++i)
                    std::cerr << " " << pos_h[i];
                if (total_tokens > 5) std::cerr << " ...";
                std::cerr << "\n";
                // Dump token IDs
                std::vector<std::int32_t> tok_h(total_tokens);
                CUDA_CHECK(cudaMemcpy(tok_h.data(), token_ids,
                    total_tokens * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
                std::cerr << "[pie-tokens] N=" << total_tokens << " ids:";
                for (int i = 0; i < std::min(total_tokens, 5); ++i)
                    std::cerr << " " << tok_h[i];
                if (total_tokens > 5) std::cerr << " ... " << tok_h[total_tokens-1];
                std::cerr << "\n";
                // Dump embed
                if (total_tokens <= 1) {
                    std::vector<std::uint16_t> hbuf(static_cast<std::size_t>(H));
                    CUDA_CHECK(cudaMemcpy(hbuf.data(), kimi_ws.y.data(),
                        H * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                    std::cerr << "[pie-embed] first10:";
                    for (int i = 0; i < 10; ++i) {
                        float v = __bfloat162float(
                            *reinterpret_cast<const __nv_bfloat16*>(&hbuf[i]));
                        std::cerr << " " << v;
                    }
                    std::cerr << "\n";
                }
            }
        } else {
            kernels::launch_embed_bf16(
                token_ids, w.embed->data(), kimi_ws.y.data(),
                total_tokens, H, cfg.vocab_size, stream);
        }
    });

    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];

        profile_cuda_stage(&profile, &profile.attn_ms, stream, [&] {
            kernels::launch_rmsnorm_bf16(
                kimi_ws.y.data(), Lw.attn_norm->data(), kimi_ws.norm_x.data(),
                total_tokens, H, eps, stream);

            if (Lw.q_kv_a_fused != nullptr) {
                // Fused q_a + kv_a projection: one GEMM instead of two
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.q_kv_a_fused,
                    kimi_ws.q_a.data(), total_tokens, q_lora + kv_lora + q_rope, H);
                // Split the fused output: q_a is first q_lora rows, kv_a is the rest
                // q_a is already in place; copy kv_a part to kv_a_mqa buffer
                CUDA_CHECK(cudaMemcpyAsync(
                    kimi_ws.kv_a_mqa.data(),
                    static_cast<const char*>(kimi_ws.q_a.data()) +
                        static_cast<std::size_t>(total_tokens) * q_lora * sizeof(std::uint16_t),
                    static_cast<std::size_t>(total_tokens) * (kv_lora + q_rope) * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream));
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.q_a_proj,
                    kimi_ws.q_a.data(), total_tokens, q_lora, H);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.kv_a_proj_with_mqa,
                    kimi_ws.kv_a_mqa.data(), total_tokens, kv_lora + q_rope, H);
            }
            kernels::launch_rmsnorm_bf16(
                kimi_ws.q_a.data(), Lw.q_a_norm->data(), kimi_ws.q_a.data(),
                total_tokens, q_lora, eps, stream);
            ops::gemm_act_x_w(cublas.handle(),
                kimi_ws.q_a.data(), *Lw.q_b_proj,
                kimi_ws.q_b.data(), total_tokens, heads * (q_nope + q_rope), q_lora);
            kernels::launch_kimi_split_kv_a_norm_bf16(
                kimi_ws.kv_a_mqa.data(), Lw.kv_a_norm->data(),
                kimi_ws.kv_c.data(), kimi_ws.k_pe.data(),
                total_tokens, kv_lora, q_rope, eps, stream);

            kernels::launch_kimi_split_q_b_bf16(
                kimi_ws.q_b.data(), kimi_ws.q_nope.data(), kimi_ws.q_pe.data(),
                total_tokens, heads, q_nope, q_rope, stream);
            if (kimi_dump_logits_enabled() && li == 0 && (tp == nullptr || tp->rank() == 0)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "[pie-rope] has_scaling=%d kind=%d factor=%.1f beta_fast=%.1f beta_slow=%.1f attn_factor=%.3f orig_max=%d\n",
                    cfg.has_rope_scaling ? 1 : 0,
                    static_cast<int>(cfg.rope_scaling_kind),
                    cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
                    cfg.rope_attention_factor, cfg.rope_original_max_position);
                write(2, buf, std::strlen(buf));
            }
            if (cfg.has_rope_scaling &&
                cfg.rope_scaling_kind == HfConfig::RopeScaling::OriginalYaRN) {
                kernels::launch_rope_yarn_original_bf16(
                    kimi_ws.q_pe.data(), kimi_ws.k_pe.data(), positions,
                    total_tokens, heads, 1, q_rope, cfg.rope_theta,
                    cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
                    cfg.rope_attention_factor,
                    cfg.rope_original_max_position, stream);
            } else {
                kernels::launch_rope_bf16(
                    kimi_ws.q_pe.data(), kimi_ws.k_pe.data(), positions,
                    total_tokens, heads, 1, q_rope, cfg.rope_theta, stream);
            }
            auto layer_view = mla_cache.layer_view(li);
            kernels::launch_write_mla_to_pages(
                layer_view, kimi_ws.kv_c.data(), kimi_ws.k_pe.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                total_tokens, num_requests, stream);

            kernels::launch_kimi_q_nope_to_latent_bf16(
                kimi_ws.q_nope.data(), Lw.kv_b_proj->data(),
                kimi_ws.q_nope_latent.data(),
                total_tokens, heads, q_nope, v_dim, kv_lora, stream);

            if (!plan_state.mla_plan) {
                throw std::runtime_error("kimi: MLA plan missing; prepare hook did not run");
            }
            ops::dispatch_attention_mla_bf16(
                *plan_state.mla_plan,
                kimi_ws.q_nope_latent.data(),
                kimi_ws.q_pe.data(),
                layer_view,
                kimi_ws.attn_latent.data(),
                kv_page_indices,
                attn_ws,
                stream,
                /*lse_out=*/nullptr,
                qo_indptr, kv_page_indptr, kv_last_page_lens);
            kernels::launch_kimi_latent_to_v_bf16(
                kimi_ws.attn_latent.data(), Lw.kv_b_proj->data(),
                kimi_ws.attn_v.data(),
                total_tokens, heads, q_nope, v_dim, kv_lora, stream);

            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.attn_v.data(), *Lw.o_proj,
                    kimi_ws.y.data(), total_tokens, H, heads * v_dim, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.attn_v.data(), *Lw.o_proj,
                    kimi_ws.norm_x.data(), total_tokens, H, heads * v_dim);
                tp->all_reduce_bf16(kimi_ws.norm_x.data(),
                    static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
                kernels::launch_residual_add_bf16(
                    kimi_ws.y.data(), kimi_ws.norm_x.data(),
                    static_cast<std::size_t>(total_tokens) * H, stream);
            }
        });

        kernels::launch_rmsnorm_bf16(
            kimi_ws.y.data(), Lw.mlp_norm->data(), kimi_ws.norm_y.data(),
            total_tokens, H, eps, stream);

        if (!Lw.is_moe) {
            ++profile.dense_layers;
            profile_cuda_stage(&profile, &profile.dense_mlp_ms, stream, [&] {
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_y.data(), *Lw.dense_gate_proj,
                    kimi_ws.gate.data(), total_tokens, dense_I, H);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_y.data(), *Lw.dense_up_proj,
                    kimi_ws.up.data(), total_tokens, dense_I, H);
                kernels::launch_swiglu_bf16(
                    kimi_ws.gate.data(), kimi_ws.up.data(), kimi_ws.gate.data(),
                    total_tokens * dense_I, stream);
                if (T == 1) {
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.gate.data(), *Lw.dense_down_proj,
                        kimi_ws.y.data(), total_tokens, H, dense_I, /*beta=*/1.f);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.gate.data(), *Lw.dense_down_proj,
                        kimi_ws.norm_x.data(), total_tokens, H, dense_I);
                    tp->all_reduce_bf16(kimi_ws.norm_x.data(),
                        static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
                    kernels::launch_residual_add_bf16(
                        kimi_ws.y.data(), kimi_ws.norm_x.data(),
                        static_cast<std::size_t>(total_tokens) * H, stream);
                }
            });
            continue;
        }

        ++profile.moe_layers;
        profile_cuda_stage(&profile, &profile.moe_router_ms, stream, [&] {
            ops::gemm_act_x_w(cublas.handle(),
                kimi_ws.norm_y.data(), *Lw.router,
                kimi_ws.router_logits.data(), total_tokens, E, H);
            kernels::launch_topk_sigmoid_bf16(
                kimi_ws.router_logits.data(),
                static_cast<std::int32_t*>(kimi_ws.topk_idx.data()),
                static_cast<float*>(kimi_ws.topk_weights.data()),
                Lw.e_score_correction_bias != nullptr
                    ? static_cast<const float*>(Lw.e_score_correction_bias->data())
                    : nullptr,
                total_tokens, E, K, cfg.norm_topk_prob,
                cfg.routed_scaling_factor, stream);
        });

        static const bool force_prefill_moe = [] {
            const char* v = std::getenv("PIE_CUDA_KIMI_FORCE_PREFILL_MOE");
            return v != nullptr && v[0] != '\0';
        }();
        if (is_pure_decode && !force_prefill_moe) {
            const int routes = total_tokens * K;
            profile_cuda_stage(&profile, &profile.moe_gate_up_ms, stream, [&] {
                kernels::launch_wna16_gate_up_decode_bf16(
                    kimi_ws.norm_y.data(),
                    static_cast<const std::int32_t*>(kimi_ws.topk_idx.data()),
                    Lw.expert_gate_packed_ptrs.data(),
                    Lw.expert_gate_scale_ptrs.data(),
                    Lw.expert_up_packed_ptrs.data(),
                    Lw.expert_up_scale_ptrs.data(),
                    kimi_ws.expert_gate.data(),
                    kimi_ws.expert_up.data(),
                    total_tokens, K, H, routed_I, 32, stream);
            });
            profile_cuda_stage(&profile, &profile.moe_swiglu_ms, stream, [&] {
                kernels::launch_swiglu_bf16(
                    kimi_ws.expert_gate.data(), kimi_ws.expert_up.data(),
                    kimi_ws.expert_gate.data(), routes * routed_I, stream);
            });
            profile_cuda_stage(&profile, &profile.moe_down_ms, stream, [&] {
                kernels::launch_wna16_down_decode_bf16(
                    kimi_ws.expert_gate.data(),
                    static_cast<const std::int32_t*>(kimi_ws.topk_idx.data()),
                    Lw.expert_down_packed_ptrs.data(),
                    Lw.expert_down_scale_ptrs.data(),
                    kimi_ws.expert_out.data(),
                    total_tokens, K, H, routed_I, 32, stream);
            });
            profile_cuda_stage(&profile, &profile.moe_weighted_sum_ms, stream, [&] {
                kernels::launch_token_batched_weighted_sum_bf16(
                    kimi_ws.moe_out.data(), kimi_ws.expert_out.data(),
                    static_cast<const float*>(kimi_ws.topk_weights.data()),
                    total_tokens, K, H, stream);
            });
        } else {
            profile_cuda_stage(&profile, &profile.moe_prefill_ms, stream, [&] {
            std::vector<std::int32_t> topk_idx_h(
                static_cast<std::size_t>(total_tokens) * K);
            std::vector<float> topk_w_h(static_cast<std::size_t>(total_tokens) * K);
            CUDA_CHECK(cudaMemcpyAsync(
                topk_idx_h.data(), kimi_ws.topk_idx.data(),
                topk_idx_h.size() * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                topk_w_h.data(), kimi_ws.topk_weights.data(),
                topk_w_h.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemsetAsync(kimi_ws.moe_out.data(), 0,
                static_cast<std::size_t>(total_tokens) * H * sizeof(std::uint16_t),
                stream));
            const auto routing =
                build_routing(topk_idx_h, topk_w_h, total_tokens, K, E);
            for (int e = 0; e < E; ++e) {
                const auto& tok_idx = routing.token_idx[static_cast<std::size_t>(e)];
                const int Ne = static_cast<int>(tok_idx.size());
                if (Ne == 0) continue;
                const auto& wts = routing.weights[static_cast<std::size_t>(e)];
                dequant_expert_w4(
                    Lw.experts[static_cast<std::size_t>(e)],
                    kimi_ws, H, routed_I, stream);
                CUDA_CHECK(cudaMemcpyAsync(
                    kimi_ws.route_idx.data(), tok_idx.data(),
                    static_cast<std::size_t>(Ne) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    kimi_ws.route_w.data(), wts.data(),
                    static_cast<std::size_t>(Ne) * sizeof(float),
                    cudaMemcpyHostToDevice, stream));
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(kimi_ws.norm_y.data()),
                    static_cast<const std::int32_t*>(kimi_ws.route_idx.data()),
                    static_cast<std::uint16_t*>(kimi_ws.expert_in.data()),
                    Ne, H, stream);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.expert_in.data(),
                    ops::WeightView::raw(kimi_ws.expert_gate_w.data(), DType::BF16),
                    kimi_ws.expert_gate.data(), Ne, routed_I, H);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.expert_in.data(),
                    ops::WeightView::raw(kimi_ws.expert_up_w.data(), DType::BF16),
                    kimi_ws.expert_up.data(), Ne, routed_I, H);
                kernels::launch_swiglu_bf16(
                    kimi_ws.expert_gate.data(), kimi_ws.expert_up.data(),
                    kimi_ws.expert_gate.data(), Ne * routed_I, stream);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.expert_gate.data(),
                    ops::WeightView::raw(kimi_ws.expert_down_w.data(), DType::BF16),
                    kimi_ws.expert_out.data(), Ne, H, routed_I);
                kernels::launch_scatter_add_weighted_bf16(
                    kimi_ws.moe_out.data(), kimi_ws.expert_out.data(),
                    static_cast<const std::int32_t*>(kimi_ws.route_idx.data()),
                    static_cast<const float*>(kimi_ws.route_w.data()),
                    Ne, H, stream);
            }
            });
        }

        if (shared_I > 0 && Lw.shared_gate_proj != nullptr) {
            profile_cuda_stage(&profile, &profile.moe_shared_ms, stream, [&] {
                if (Lw.shared_gate_up_fused != nullptr && total_tokens == 1) {
                    // Fused gate+up for single-token decode: one GEMM
                    // Output: [1, 2*shared_I] = [gate..., up...]
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_gate_up_fused,
                        kimi_ws.shared_gate.data(), 1, 2 * shared_I, H);
                    // SwiGLU reads gate from shared_gate, up from shared_gate+shared_I
                    // For N=1, shared_up can point into the fused buffer
                } else if (Lw.shared_gate_up_fused != nullptr) {
                    // Multi-token: fall back to unfused (need interleaved layout)
                    // TODO: implement strided SwiGLU for fused gate+up
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_gate_up_fused,
                        kimi_ws.shared_gate.data(), total_tokens, 2 * shared_I, H);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_gate_proj,
                        kimi_ws.shared_gate.data(), total_tokens, shared_I, H);
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_up_proj,
                        kimi_ws.shared_up.data(), total_tokens, shared_I, H);
                }
                {
                    // For fused gate+up with N=1: gate at offset 0, up at offset shared_I
                    const void* up_ptr = (Lw.shared_gate_up_fused != nullptr && total_tokens == 1)
                        ? static_cast<const char*>(kimi_ws.shared_gate.data()) +
                          static_cast<std::size_t>(shared_I) * sizeof(std::uint16_t)
                        : kimi_ws.shared_up.data();
                    kernels::launch_swiglu_bf16(
                        kimi_ws.shared_gate.data(), up_ptr,
                        kimi_ws.shared_act.data(), total_tokens * shared_I, stream);
                }
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.shared_act.data(), *Lw.shared_down_proj,
                    kimi_ws.shared_out.data(), total_tokens, H, shared_I);
                kernels::launch_residual_add_bf16(
                    kimi_ws.moe_out.data(), kimi_ws.shared_out.data(),
                    static_cast<std::size_t>(total_tokens) * H, stream);
            });
        }
        if (T > 1) {
            profile_cuda_stage(&profile, &profile.moe_allreduce_ms, stream, [&] {
                tp->all_reduce_bf16(kimi_ws.moe_out.data(),
                    static_cast<std::size_t>(total_tokens) * H, ncclSum, stream);
            });
        }
        profile_cuda_stage(&profile, &profile.residual_ms, stream, [&] {
            kernels::launch_residual_add_bf16(
                kimi_ws.y.data(), kimi_ws.moe_out.data(),
                static_cast<std::size_t>(total_tokens) * H, stream);
        });
        dump_hidden_norm(kimi_ws.y.data(), total_tokens, H, li,
            "post_moe", tp != nullptr ? tp->rank() : 0, stream);
    }

    const bool use_tp_greedy =
        fwd_cfg.tp_greedy_argmax && T > 1 && tp != nullptr &&
        w.lm_head_tp_sharded &&
        w.lm_head != nullptr &&
        w.lm_head->shape().size() == 2 &&
        w.lm_head->shape()[0] > 0 &&
        fwd_cfg.greedy_pairs != nullptr &&
        fwd_cfg.greedy_pairs_all != nullptr &&
        T <= 8;
    if (!fwd_cfg.emit_logits && !use_tp_greedy) {
        profile.end(stream);
        maybe_print_profile(profile);
        return;
    }

    const bool compact_logits =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < total_tokens;
    const int rows = compact_logits ? num_logit_rows : total_tokens;
    const void* final_in = kimi_ws.y.data();
    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        if (compact_logits) {
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(kimi_ws.y.data()),
                logit_row_indices_d,
                static_cast<std::uint16_t*>(kimi_ws.norm_x.data()),
                num_logit_rows, H, stream);
            final_in = kimi_ws.norm_x.data();
        }
        kernels::launch_rmsnorm_bf16(
            final_in, w.final_norm->data(), kimi_ws.norm_y.data(),
            rows, H, eps, stream);
    });
    if (use_tp_greedy) {
        const int V_local = static_cast<int>(w.lm_head->shape()[0]);
        profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
            ops::gemm_act_x_w(cublas.handle(),
                kimi_ws.norm_y.data(), *w.lm_head, logits_out,
                rows, V_local, H);
            dump_top_logits(logits_out, rows, V_local,
                tp != nullptr ? tp->rank() : 0,
                w.lm_head_tp_vocab_offset, stream);
            kernels::launch_argmax_bf16_tile_pair(
                logits_out,
                reinterpret_cast<std::uint64_t*>(fwd_cfg.greedy_pairs),
                rows, V_local, w.lm_head_tp_vocab_offset, stream);
            tp->all_gather_bytes(
                fwd_cfg.greedy_pairs, fwd_cfg.greedy_pairs_all,
                static_cast<std::size_t>(rows) * sizeof(std::uint64_t),
                stream);
        });
        profile.end(stream);
        maybe_print_profile(profile);
        return;
    }
    if (w.lm_head_tp_sharded) {
        throw std::runtime_error(
            "kimi: sharded lm_head requires TP greedy argmax for logits");
    }
    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        ops::gemm_act_x_w(cublas.handle(),
            kimi_ws.norm_y.data(), *w.lm_head, logits_out,
            rows, V, H);
        dump_top_logits(logits_out, rows, V,
            tp != nullptr ? tp->rank() : 0, 0, stream);
    });
    profile.end(stream);
    maybe_print_profile(profile);
}

}  // namespace pie_cuda_driver::model
