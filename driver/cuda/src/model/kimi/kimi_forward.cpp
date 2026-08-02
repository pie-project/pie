#include "model/kimi/kimi_forward.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/argmax.hpp"
#include "model/act_dump.hpp"
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
#include "ops/flashinfer_moe.hpp"

namespace pie_cuda_driver::model {


// The per-route decode GEMV wins only at a single token. It dequantises int4
// with scalar FP32 ALU and re-reads each routed expert per token, while the
// batched/fused paths run on tensor cores over a materialised BF16 stack.
// Measured on kimi26-mini (output tok/s, GEMV vs batched): c=1 564/566,
// c=2 896/1088, c=4 1301/2089, c=8 1718/4103. Override with
// `PIE_MOE_GEMV_MAX_TOKENS`.
constexpr int kKimiMoeGemvMaxTokens = 1;
// Per-layer budget for the materialised BF16 expert stack. The mini
// checkpoints (E=8) need ~0.7 GiB; a full 384-expert Kimi would need ~34 GiB,
// so it stays on the W4A16 path where each expert is touched at most a few
// times per step anyway.

namespace {

constexpr bool kimi_profile_enabled() { return false; }

constexpr std::uint64_t kimi_profile_print_limit() { return 8; }

constexpr bool kimi_profile_all_ranks() { return false; }

constexpr bool kimi_dump_logits_enabled() { return false; }

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
    double attn_proj_ms = 0.0;
    double attn_absorb_ms = 0.0;
    double attn_core_ms = 0.0;
    double attn_oproj_ms = 0.0;
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
        attn_absorb_ms = attn_core_ms = attn_oproj_ms = attn_proj_ms = 0.0;
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
        p.embed_ms + p.attn_proj_ms + p.attn_absorb_ms + p.attn_core_ms +
        p.attn_oproj_ms + p.dense_mlp_ms + p.moe_router_ms +
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
        << " attn_proj_ms=" << p.attn_proj_ms
        << " attn_absorb_ms=" << p.attn_absorb_ms
        << " attn_core_ms=" << p.attn_core_ms
        << " attn_oproj_ms=" << p.attn_oproj_ms
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
    ws.qkv_a         = DeviceTensor::allocate(DType::BF16, {N, q_lora + kv_lora + q_rope});
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
    // fp16 activation staging for the W4A16 decode GEMVs, whose inner loop is
    // pure `__hfma2` and so wants its activation already in fp16.
    //
    // Sized for N, not for `moe_gemv_max_tokens`. The decode-GEMV branch is not
    // only reached below that threshold: `want_batched` also requires
    // `routes > 4 * min(E, routes)` and a bf16 stacked expert weight, so a
    // 2-to-4 token decode -- or any decode at all on a checkpoint that ships no
    // bf16 copy -- lands here too. Under-sizing these and guarding the branch
    // drops those cases into the prefill path, which synchronises the stream and
    // therefore fails outright during CUDA-graph capture.
    ws.norm_y_fp16 = DeviceTensor::allocate(DType::FP16, {N, H});
    ws.expert_act_fp16 = DeviceTensor::allocate(DType::FP16, {routes, max_I});
    ws.shared_gate   = DeviceTensor::allocate(DType::BF16, {N, std::max(1, 2 * shared_I)});
    ws.shared_up     = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_act    = DeviceTensor::allocate(DType::BF16, {N, std::max(1, shared_I)});
    ws.shared_out    = DeviceTensor::allocate(DType::BF16, {N, H});

    // Aligned MoE scratch (batched-GEMM path). Worst case is every route
    // landing in its own expert block, each padded up to `block` rows.
    if (routed_I > 0 && cfg.num_experts > 0) {
        const int E = cfg.num_experts;
        // `block` is picked per forward from that batch's route count, so the
        // scratch has to cover both extremes: the smallest block produces the
        // most blocks, the largest produces the most padded rows.
        const int block = kernels::moe_aligned_block(routes, E);
        const int active_expert_cap = std::min(E, routes);
        const int max_blocks =
            (routes + active_expert_cap * (kernels::kMoeAlignedBlockMin - 1) +
             kernels::kMoeAlignedBlockMin - 1) /
            kernels::kMoeAlignedBlockMin;
        const int aligned_rows =
            std::max(max_blocks * kernels::kMoeAlignedBlockMin,
                     ((routes + active_expert_cap * (block - 1) + block - 1) /
                      block) * block);
        ws.aligned_block_size = block;
        ws.aligned_max_blocks = max_blocks;
        ws.aligned_route_ids  = DeviceTensor::allocate(DType::INT32, {aligned_rows});
        ws.aligned_expert_ids = DeviceTensor::allocate(DType::INT32, {max_blocks});
        ws.aligned_expert_in  = DeviceTensor::allocate(DType::BF16, {aligned_rows, H});
        ws.aligned_gate_up    = DeviceTensor::allocate(DType::BF16, {aligned_rows, 2 * routed_I});
        ws.aligned_act        = DeviceTensor::allocate(DType::BF16, {aligned_rows, routed_I});
        ws.aligned_out        = DeviceTensor::allocate(DType::BF16, {aligned_rows, H});
        const std::int64_t pw =
            static_cast<std::int64_t>(max_blocks) * sizeof(void*) / sizeof(std::int64_t);
        for (DeviceTensor* t : {&ws.a_gu_ptrs, &ws.b_gu_ptrs, &ws.c_gu_ptrs,
                                &ws.a_dn_ptrs, &ws.b_dn_ptrs, &ws.c_dn_ptrs}) {
            *t = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        }
    }
    if (routed_I > 0 && cfg.num_experts > 0 && Ktop > 0 &&
        ops::flashinfer_cutlass_moe_enabled() && kimi_moe_gate_up_swapped()) {
        ws.cutlass_max_rows = std::min(N, ops::flashinfer_cutlass_moe_max_rows());
        const std::size_t bytes = ops::flashinfer_cutlass_moe_workspace_bytes(
            ops::MoeActivation::Swiglu, ws.cutlass_max_rows, H, routed_I,
            cfg.num_experts, Ktop, /*tp_size=*/1, /*tp_rank=*/0);
        if (bytes > 0) {
            ws.cutlass_ws = DeviceTensor::allocate(
                DType::UINT8, {static_cast<std::int64_t>(bytes)});
            ws.cutlass_row_map = DeviceTensor::allocate(
                DType::INT32,
                {static_cast<std::int64_t>(ws.cutlass_max_rows) * Ktop});
        }
    }
    ws.logits        = DeviceTensor::allocate(DType::BF16, {O, cfg.vocab_size});
    ws.probs         = DeviceTensor::allocate(DType::FP32, {O, cfg.vocab_size});
    return ws;
}

std::size_t kimi_workspace_bytes(
    const HfConfig& cfg,
    int max_tokens,
    int max_logit_rows,
    int tp_size) {
    ScopedDeviceAllocationCounter counter;
    {
        auto workspace = KimiWorkspace::allocate(
            cfg, max_tokens, max_logit_rows, tp_size);
    }
    return counter.allocated_bytes();
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
        (1.0f / std::sqrt(static_cast<float>(
             cfg.qk_nope_head_dim + cfg.qk_rope_head_dim))) *
            cfg.rope_mla_softmax_mscale * cfg.rope_mla_softmax_mscale);
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
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const StageHooks* hooks)
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
    act_dump_step_begin(stream);

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
        if (li == 0) {
            act_dump_i32("tokens", token_ids, total_tokens, 1, stream);
            act_dump_i32("positions", positions, total_tokens, 1, stream);
            act_dump_bf16("embed", kimi_ws.y.data(), total_tokens, H, stream);
        }
        profile_cuda_stage(&profile, &profile.attn_proj_ms, stream, [&] {
            kernels::launch_rmsnorm_bf16(
                kimi_ws.y.data(), Lw.attn_norm->data(), kimi_ws.norm_x.data(),
                total_tokens, H, eps, stream);
            act_dump_bf16(act_dump_layer_tag("norm_x", li).c_str(),
                kimi_ws.norm_x.data(), total_tokens, H, stream);

            // Where the kv half ends up, and its row pitch. The fused path
            // leaves both halves interleaved per token in `qkv_a`; the split
            // path writes a compact `kv_a_mqa`.
            const void* kv_a_src = kimi_ws.kv_a_mqa.data();
            int kv_a_row_stride = 0;
            if (Lw.q_kv_a_fused != nullptr) {
                // Fused q_a + kv_a projection: one GEMM instead of two. The
                // result is row-major `[T, q_lora + kv_lora + q_rope]`, so the
                // halves interleave per token and neither is a contiguous
                // block of the buffer -- both consumers below take a pitch.
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.q_kv_a_fused,
                    kimi_ws.qkv_a.data(), total_tokens, q_lora + kv_lora + q_rope, H);
                kv_a_src = static_cast<const char*>(kimi_ws.qkv_a.data()) +
                    static_cast<std::size_t>(q_lora) * sizeof(std::uint16_t);
                kv_a_row_stride = q_lora + kv_lora + q_rope;
                kernels::launch_rmsnorm_strided_bf16(
                    kimi_ws.qkv_a.data(), Lw.q_a_norm->data(), kimi_ws.q_a.data(),
                    total_tokens, q_lora,
                    /*x_row_stride=*/q_lora + kv_lora + q_rope,
                    /*y_row_stride=*/q_lora, eps, stream);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.q_a_proj,
                    kimi_ws.q_a.data(), total_tokens, q_lora, H);
                ops::gemm_act_x_w(cublas.handle(),
                    kimi_ws.norm_x.data(), *Lw.kv_a_proj_with_mqa,
                    kimi_ws.kv_a_mqa.data(), total_tokens, kv_lora + q_rope, H);
                kernels::launch_rmsnorm_bf16(
                    kimi_ws.q_a.data(), Lw.q_a_norm->data(), kimi_ws.q_a.data(),
                    total_tokens, q_lora, eps, stream);
            }
            act_dump_bf16(act_dump_layer_tag("q_a", li).c_str(),
                kimi_ws.q_a.data(), total_tokens, q_lora, stream);
            ops::gemm_act_x_w(cublas.handle(),
                kimi_ws.q_a.data(), *Lw.q_b_proj,
                kimi_ws.q_b.data(), total_tokens, heads * (q_nope + q_rope), q_lora);
            act_dump_bf16(act_dump_layer_tag("q_b", li).c_str(),
                kimi_ws.q_b.data(), total_tokens, heads * (q_nope + q_rope), stream);
            invoke_stage_hook(
                hooks,
                StageHookPoint::OnAttnProj, kimi_ws.q_b.data(),
                static_cast<std::uint32_t>(total_tokens),
                static_cast<std::uint32_t>(heads * (q_nope + q_rope)),
                static_cast<std::uint32_t>(li), stream);
            auto layer_view = mla_cache.layer_view(li);
            const bool yarn =
                cfg.has_rope_scaling &&
                cfg.rope_scaling_kind == HfConfig::RopeScaling::OriginalYaRN;
            const bool fuse_prepare =
                kernels::mla_prepare_supported(q_rope) && !act_dump_enabled() &&
                (!cfg.has_rope_scaling || yarn);
            if (fuse_prepare) {
                kernels::YarnOriginalParams yp{};
                yp.factor = cfg.rope_factor;
                yp.beta_fast = cfg.rope_beta_fast;
                yp.beta_slow = cfg.rope_beta_slow;
                yp.attention_factor = cfg.rope_attention_factor;
                yp.original_max_position = cfg.rope_original_max_position;
                kernels::launch_mla_prepare_bf16(
                    layer_view,
                    kv_a_src, Lw.kv_a_norm->data(), kimi_ws.q_b.data(),
                    kimi_ws.kv_c.data(), kimi_ws.k_pe.data(),
                    kimi_ws.q_nope.data(), kimi_ws.q_pe.data(),
                    positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, total_tokens, num_requests,
                    heads, q_nope, eps, cfg.rope_theta, /*interleaved=*/true,
                    kv_a_row_stride, yarn ? &yp : nullptr, stream, row_valid_d);
            } else {
            kernels::launch_kimi_split_kv_a_norm_bf16(
                kv_a_src, Lw.kv_a_norm->data(),
                kimi_ws.kv_c.data(), kimi_ws.k_pe.data(),
                total_tokens, kv_lora, q_rope, eps, stream, kv_a_row_stride);
            act_dump_bf16(act_dump_layer_tag("kv_c", li).c_str(),
                kimi_ws.kv_c.data(), total_tokens, kv_lora, stream);

            kernels::launch_kimi_split_q_b_bf16(
                kimi_ws.q_b.data(), kimi_ws.q_nope.data(), kimi_ws.q_pe.data(),
                total_tokens, heads, q_nope, q_rope, stream);
            act_dump_bf16(act_dump_layer_tag("q_pe_pre", li).c_str(),
                kimi_ws.q_pe.data(), total_tokens, heads * q_rope, stream);
            act_dump_bf16(act_dump_layer_tag("k_pe_pre", li).c_str(),
                kimi_ws.k_pe.data(), total_tokens, q_rope, stream);
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
            // DeepSeek-V2/V3 (and Kimi-K2, same arch) rotate *adjacent* dim
            // pairs on q_pe/k_pe. HF `modeling_deepseek.py` writes it as an
            // interleave-transpose followed by a NeoX `rotate_half`, which is
            // the identical rotation; vLLM builds the rope with
            // `is_neox_style=False`. The NeoX half/half pairing this used to
            // apply is a different rotation and made attention drift with
            // position (row 0 exact, later rows increasingly wrong).
            if (cfg.has_rope_scaling &&
                cfg.rope_scaling_kind == HfConfig::RopeScaling::OriginalYaRN) {
                kernels::launch_rope_yarn_original_bf16(
                    kimi_ws.q_pe.data(), kimi_ws.k_pe.data(), positions,
                    total_tokens, heads, 1, q_rope, cfg.rope_theta,
                    cfg.rope_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
                    cfg.rope_attention_factor,
                    cfg.rope_original_max_position, stream,
                    /*interleaved=*/true);
            } else {
                kernels::launch_rope_bf16(
                    kimi_ws.q_pe.data(), kimi_ws.k_pe.data(), positions,
                    total_tokens, heads, 1, q_rope, cfg.rope_theta, stream,
                    /*interleaved=*/true);
            }
            act_dump_bf16(act_dump_layer_tag("q_pe", li).c_str(),
                kimi_ws.q_pe.data(), total_tokens, heads * q_rope, stream);
            act_dump_bf16(act_dump_layer_tag("k_pe", li).c_str(),
                kimi_ws.k_pe.data(), total_tokens, q_rope, stream);
            kernels::launch_write_mla_to_pages(
                layer_view, kimi_ws.kv_c.data(), kimi_ws.k_pe.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                total_tokens, num_requests, stream, row_valid_d);
            }

            profile_cuda_stage(&profile, &profile.attn_absorb_ms, stream, [&] {
            ops::mla_absorb_q_to_latent_bf16(cublas.handle(),
                kimi_ws.q_nope.data(), Lw.kv_b_proj->data(),
                kimi_ws.q_nope_latent.data(),
                total_tokens, heads, q_nope, v_dim, kv_lora);
            });

            if (!plan_state.mla_plan) {
                throw std::runtime_error("kimi: MLA plan missing; prepare hook did not run");
            }
            profile_cuda_stage(&profile, &profile.attn_core_ms, stream, [&] {
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
            });
            profile_cuda_stage(&profile, &profile.attn_absorb_ms, stream, [&] {
            ops::mla_absorb_latent_to_v_bf16(cublas.handle(),
                kimi_ws.attn_latent.data(), Lw.kv_b_proj->data(),
                kimi_ws.attn_v.data(),
                total_tokens, heads, q_nope, v_dim, kv_lora);
            });
            act_dump_bf16(act_dump_layer_tag("attn_v", li).c_str(),
                kimi_ws.attn_v.data(), total_tokens, heads * v_dim, stream);
            invoke_stage_hook(
                hooks,
                StageHookPoint::OnAttn, kimi_ws.q_b.data(),
                static_cast<std::uint32_t>(total_tokens),
                static_cast<std::uint32_t>(heads * (q_nope + q_rope)),
                static_cast<std::uint32_t>(li), stream);

            profile_cuda_stage(&profile, &profile.attn_oproj_ms, stream, [&] {
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
        });

        kernels::launch_rmsnorm_bf16(
            kimi_ws.y.data(), Lw.mlp_norm->data(), kimi_ws.norm_y.data(),
            total_tokens, H, eps, stream);
        act_dump_bf16(act_dump_layer_tag("post_attn", li).c_str(),
            kimi_ws.y.data(), total_tokens, H, stream);
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
            act_dump_bf16(act_dump_layer_tag("out", li).c_str(),
                kimi_ws.y.data(), total_tokens, H, stream);
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

        constexpr bool force_prefill_moe = false;

        // The W4A16 GEMVs dequantise int4 with scalar FP32 ALU and re-read
        // every routed expert once per token; the batched path materialises a
        // BF16 stack and runs on tensor cores. Its weight traffic is 4x, but it
        // is roughly 30x faster per token, so the crossover sits far below what
        // `routes > 4 * active_experts` predicts. The traffic model still gates
        // the compiled-in default; `PIE_MOE_GEMV_MAX_TOKENS=0` retires the GEMV.
        const int routes = total_tokens * K;
        const int gemv_max = ops::moe_gemv_max_tokens(kKimiMoeGemvMaxTokens);
        const bool want_batched =
            Lw.moe_gate_up_bf16 != nullptr && kimi_ws.aligned_block_size > 0 &&
            total_tokens > gemv_max &&
            (gemv_max <= 0 || routes > 4 * std::min(E, routes)) &&
            !force_prefill_moe;

        // flashinfer's fused grouped GEMM does the batched path's whole job in
        // one call and without its padding: it permutes rows by expert, runs
        // both GEMMs over the actual row counts, and folds SwiGLU and the top-k
        // weighted sum into GEMM2's FINALIZE epilogue. The batched path has to
        // provision `max_blocks` for worst-case routing skew and run every
        // block unconditionally, which is what makes it lose at high E.
        bool fused_moe_ran = false;
        if (want_batched && !kimi_ws.cutlass_ws.empty() &&
            total_tokens <= kimi_ws.cutlass_max_rows &&
            total_tokens >= ops::flashinfer_cutlass_moe_min_rows()) {
            profile_cuda_stage(&profile, &profile.moe_prefill_ms, stream, [&] {
                fused_moe_ran = ops::flashinfer_cutlass_moe_bf16(
                    ops::MoeActivation::Swiglu,
                    static_cast<const std::uint16_t*>(kimi_ws.norm_y.data()),
                    static_cast<const std::int32_t*>(kimi_ws.topk_idx.data()),
                    static_cast<const float*>(kimi_ws.topk_weights.data()),
                    static_cast<const std::uint16_t*>(Lw.moe_gate_up_bf16->data()),
                    static_cast<const std::uint16_t*>(Lw.moe_down_bf16->data()),
                    static_cast<std::uint16_t*>(kimi_ws.moe_out.data()),
                    static_cast<std::uint8_t*>(kimi_ws.cutlass_ws.data()),
                    static_cast<std::size_t>(kimi_ws.cutlass_ws.nbytes()),
                    static_cast<std::int32_t*>(kimi_ws.cutlass_row_map.data()),
                    total_tokens, H, routed_I, E, K,
                    /*tp_size=*/1, /*tp_rank=*/0, stream);
            });
        }

        if (fused_moe_ran) {
            // FINALIZE already applied `topk_weights` and summed the K experts.
        } else if (want_batched) {
            profile_cuda_stage(&profile, &profile.moe_prefill_ms, stream, [&] {
                const int block = std::min(kimi_ws.aligned_block_size,
                                           kernels::moe_aligned_block(routes, E));
                const int active_expert_cap = std::min(E, routes);
                const int max_blocks =
                    (routes + active_expert_cap * (block - 1) + block - 1) / block;
                const int aligned_rows = max_blocks * block;
                if (max_blocks > kimi_ws.aligned_max_blocks ||
                    aligned_rows >
                        static_cast<int>(kimi_ws.aligned_expert_in.shape()[0])) {
                    throw std::runtime_error("kimi: aligned MoE scratch too small");
                }
                kernels::launch_moe_align_decode(
                    static_cast<const std::int32_t*>(kimi_ws.topk_idx.data()),
                    static_cast<std::int32_t*>(kimi_ws.aligned_route_ids.data()),
                    static_cast<std::int32_t*>(kimi_ws.aligned_expert_ids.data()),
                    /*route_to_aligned_row=*/nullptr,
                    routes, E, block, max_blocks, /*num_tokens_past_padded=*/nullptr, stream);
                kernels::launch_gather_moe_aligned_inputs_bf16(
                    kimi_ws.norm_y.data(),
                    static_cast<const std::int32_t*>(kimi_ws.aligned_route_ids.data()),
                    kimi_ws.aligned_expert_in.data(),
                    routes, aligned_rows, K, H,
                    /*shared_row_begin=*/-1, total_tokens, stream);
                kernels::launch_build_moe_ptrs_aligned_bf16(
                    static_cast<const std::int32_t*>(kimi_ws.aligned_expert_ids.data()),
                    Lw.moe_gate_up_bf16->data(), Lw.moe_down_bf16->data(),
                    kimi_ws.aligned_expert_in.data(), kimi_ws.aligned_gate_up.data(),
                    kimi_ws.aligned_act.data(), kimi_ws.aligned_out.data(),
                    reinterpret_cast<const void**>(kimi_ws.a_gu_ptrs.data()),
                    reinterpret_cast<const void**>(kimi_ws.b_gu_ptrs.data()),
                    reinterpret_cast<void**>(kimi_ws.c_gu_ptrs.data()),
                    reinterpret_cast<const void**>(kimi_ws.a_dn_ptrs.data()),
                    reinterpret_cast<const void**>(kimi_ws.b_dn_ptrs.data()),
                    reinterpret_cast<void**>(kimi_ws.c_dn_ptrs.data()),
                    max_blocks, block, H, routed_I,
                    /*routed_blocks=*/max_blocks, nullptr, nullptr, stream);

                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(kimi_ws.b_gu_ptrs.data()),
                    reinterpret_cast<const void* const*>(kimi_ws.a_gu_ptrs.data()),
                    reinterpret_cast<void* const*>(kimi_ws.c_gu_ptrs.data()),
                    block, 2 * routed_I, H, max_blocks);
                kernels::launch_chunked_swiglu_bf16(
                    kimi_ws.aligned_gate_up.data(), kimi_ws.aligned_act.data(),
                    aligned_rows, routed_I, stream,
                    /*gate_second=*/kimi_moe_gate_up_swapped());
                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(kimi_ws.b_dn_ptrs.data()),
                    reinterpret_cast<const void* const*>(kimi_ws.a_dn_ptrs.data()),
                    reinterpret_cast<void* const*>(kimi_ws.c_dn_ptrs.data()),
                    block, H, routed_I, max_blocks);
                kernels::launch_reorder_moe_aligned_output_bf16(
                    kimi_ws.aligned_out.data(),
                    static_cast<const std::int32_t*>(kimi_ws.aligned_route_ids.data()),
                    kimi_ws.expert_out.data(),
                    routes, aligned_rows, H,
                    /*shared_row_begin=*/-1, total_tokens, nullptr, stream);
                kernels::launch_token_batched_weighted_sum_bf16(
                    kimi_ws.moe_out.data(), kimi_ws.expert_out.data(),
                    static_cast<const float*>(kimi_ws.topk_weights.data()),
                    total_tokens, K, H, stream);
            });
        } else if (is_pure_decode && !force_prefill_moe) {
            profile_cuda_stage(&profile, &profile.moe_gate_up_ms, stream, [&] {
                kernels::launch_bf16_to_fp16(
                    kimi_ws.norm_y.data(), kimi_ws.norm_y_fp16.data(),
                    static_cast<std::size_t>(total_tokens) * H, stream);
                kernels::launch_wna16_gate_up_decode_bf16(
                    kimi_ws.norm_y_fp16.data(),
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
                kernels::launch_bf16_to_fp16(
                    kimi_ws.expert_gate.data(),
                    kimi_ws.expert_act_fp16.data(),
                    static_cast<std::size_t>(routes) * routed_I, stream);
            });
            profile_cuda_stage(&profile, &profile.moe_down_ms, stream, [&] {
                kernels::launch_wna16_down_decode_bf16(
                    kimi_ws.expert_act_fp16.data(),
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

        if (shared_I > 0 &&
            (Lw.shared_gate_proj != nullptr || Lw.shared_gate_up_fused != nullptr)) {
            profile_cuda_stage(&profile, &profile.moe_shared_ms, stream, [&] {
                if (Lw.shared_gate_up_fused != nullptr) {
                    // One GEMM into `[T, 2*shared_I]` (gate half first, up half
                    // second) followed by the chunked SwiGLU that reads both
                    // halves of a row. `shared_gate` is allocated at
                    // `2 * shared_I` columns precisely for this.
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_gate_up_fused,
                        kimi_ws.shared_gate.data(), total_tokens, 2 * shared_I, H);
                    kernels::launch_chunked_swiglu_bf16(
                        kimi_ws.shared_gate.data(), kimi_ws.shared_act.data(),
                        total_tokens, shared_I, stream);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_gate_proj,
                        kimi_ws.shared_gate.data(), total_tokens, shared_I, H);
                    ops::gemm_act_x_w(cublas.handle(),
                        kimi_ws.norm_y.data(), *Lw.shared_up_proj,
                        kimi_ws.shared_up.data(), total_tokens, shared_I, H);
                    kernels::launch_swiglu_bf16(
                        kimi_ws.shared_gate.data(), kimi_ws.shared_up.data(),
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
        act_dump_bf16(act_dump_layer_tag("out", li).c_str(),
            kimi_ws.y.data(), total_tokens, H, stream);
    }

    if (!fwd_cfg.emit_logits) {
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
    if (w.lm_head_tp_sharded) {
        throw std::runtime_error(
            "kimi: sharded lm_head cannot emit full-vocab logits");
    }
    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        ops::gemm_act_x_w(cublas.handle(),
            kimi_ws.norm_y.data(), *w.lm_head, logits_out,
            rows, V, H);
        dump_top_logits(logits_out, rows, V,
            tp != nullptr ? tp->rank() : 0, 0, stream);
        act_dump_bf16("final_norm", kimi_ws.norm_y.data(), rows, H, stream);
        act_dump_bf16("logits", logits_out, rows, V, stream);
    });
    profile.end(stream);
    maybe_print_profile(profile);
}

}  // namespace pie_cuda_driver::model
