#include "model/deepseek_v4/deepseek_v4_forward.hpp"

#include "loader/group_stream_cache.hpp"
#include "model/act_dump.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cublas_v2.h>

#include "model/llama_like/qwen3.hpp"  // for make_weight_view
#include "kernels/attn_sink.hpp"
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
#include "ops/flashinfer_moe.hpp"

namespace pie_cuda_driver::model {

// Expert-block granularity for the device-side aligned MoE, and the token
// count below which the per-route GEMV path beats a batched GEMM. The GEMV is
// one warp per output row doing scalar FP32 math, so it only wins at a single
// token; from two tokens up the batched GEMM's tensor cores dominate. Measured
// at c=8 (output tok/s, GEMV vs batched): dsv4pro-mini 805/1318,
// dsv4flash-mini 1614/2137. Override with `PIE_MOE_GEMV_MAX_TOKENS`.
static constexpr int kDsV4MoeGemvMaxTokens = 1;

inline void dsv4_chunked_swiglu(const void* packed, void* y, int rows, int I,
                                float limit, cudaStream_t stream) {
    if (limit > 0.f) {
        kernels::launch_chunked_swiglu_clamp_bf16(packed, y, rows, I, limit, stream);
    } else {
        kernels::launch_chunked_swiglu_bf16(packed, y, rows, I, stream);
    }
}


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
        const std::size_t base = static_cast<std::size_t>(n) * K;
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx[base + k];
            if (e < 0 || e >= E) continue;
            // DeepSeek-V4's hash router reads its expert ids out of a
            // `tid2eid[token]` row, which — unlike a top-k selection — may
            // name the same expert more than once. Emitting one routed row
            // per occurrence would make several blocks of the non-atomic
            // `scatter_add_weighted_bf16` read-modify-write the same output
            // row concurrently and silently drop all but one contribution.
            // Fold the duplicates into a single row with the summed weight.
            bool dup = false;
            for (int k2 = 0; k2 < k; ++k2) {
                if (topk_idx[base + k2] == e) { dup = true; break; }
            }
            if (dup) continue;
            float w = 0.f;
            for (int k2 = k; k2 < K; ++k2) {
                if (topk_idx[base + k2] == e) w += topk_w[base + k2];
            }
            r.token_idx[static_cast<std::size_t>(e)].push_back(n);
            r.weights[static_cast<std::size_t>(e)].push_back(w);
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
    // A boundary token needs (pos + 1) % ratio == 0, so a request of n tokens
    // contributes at most ceil(n / ratio) entries. Summed over requests that is
    // bounded by N (and by N/ratio + num_requests), so N is a hard safe bound.
    const int max_comp_tokens = min_ratio > 0 ? N : 0;
    if (max_comp_tokens > 0 && coff_max > 0) {
        ws.comp_kv_proj    = DeviceTensor::allocate(DType::BF16, {N, coff_max * head_dim});
        ws.comp_score_proj = DeviceTensor::allocate(DType::BF16, {N, coff_max * head_dim});
        ws.comp_kv         = DeviceTensor::allocate(DType::BF16, {max_comp_tokens, head_dim});
        ws.comp_attn_out   = DeviceTensor::allocate(DType::BF16, {N, num_heads * head_dim});
        ws.comp_attn_lse   = DeviceTensor::allocate(DType::FP32, {N, num_heads});
        // Three index slices per distinct compress ratio, plus one slice for
        // the per-token request index. Every slice is N wide.
        std::vector<int> distinct_ratios;
        for (int r : cfg.dsv4_compress_ratios) {
            if (r <= 0) continue;
            if (std::find(distinct_ratios.begin(), distinct_ratios.end(), r) ==
                distinct_ratios.end()) {
                distinct_ratios.push_back(r);
            }
        }
        ws.comp_meta = DeviceTensor::allocate(
            DType::INT32,
            {(3 * static_cast<int>(distinct_ratios.size()) + 1) * N});
    }

    // `wo_a` is replicated, so this is deliberately NOT divided by tp_size.
    if (cfg.dsv4_o_lora_rank > 0 && cfg.dsv4_o_groups > 0) {
        const int per_group_dim =
            cfg.num_attention_heads * cfg.head_dim / cfg.dsv4_o_groups;
        ws.wo_a_dequant = DeviceTensor::allocate(
            DType::BF16, {cfg.dsv4_o_lora_rank, per_group_dim});
    }

    // Routed experts: weights are sharded on intermediate dim → moe_I / T.
    const int local_moe_I_ws = moe_I / std::max(1, tp_size);
    ws.expert_in     = DeviceTensor::allocate(DType::BF16,{N, H});
    ws.expert_gate_w = DeviceTensor::allocate(DType::BF16,{local_moe_I_ws, H});
    ws.expert_up_w   = DeviceTensor::allocate(DType::BF16,{local_moe_I_ws, H});
    ws.expert_down_w = DeviceTensor::allocate(DType::BF16,{H, local_moe_I_ws});
    ws.expert_gate   = DeviceTensor::allocate(DType::BF16,{N, local_moe_I_ws});
    ws.expert_up     = DeviceTensor::allocate(DType::BF16,{N, local_moe_I_ws});
    // One row per route, not per token: both the aligned reorder and the
    // decode GEMV scatter into row `token * K + k`.
    ws.expert_out    = DeviceTensor::allocate(
        DType::BF16, {static_cast<std::int64_t>(N) * K, H});
    ws.route_idx     = DeviceTensor::allocate(DType::INT32, {N * K});
    ws.route_w       = DeviceTensor::allocate(DType::FP32, {N * K});

    // Aligned MoE scratch: worst case is every route landing in its own
    // expert block, each padded up to `block` rows.
    if (local_moe_I_ws > 0 && E > 0 && K > 0) {
        const int routes = N * K;
        // Sized for both extremes because the block is chosen per forward:
        // the minimum block yields the most blocks, this one the most rows.
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
        ws.aligned_gate_up    = DeviceTensor::allocate(DType::BF16, {aligned_rows, 2 * local_moe_I_ws});
        ws.aligned_act        = DeviceTensor::allocate(DType::BF16, {aligned_rows, local_moe_I_ws});
        ws.aligned_out        = DeviceTensor::allocate(DType::BF16, {aligned_rows, H});
        const std::int64_t pw =
            static_cast<std::int64_t>(max_blocks) * sizeof(void*) / sizeof(std::int64_t);
        for (DeviceTensor* t : {&ws.a_gu_ptrs, &ws.b_gu_ptrs, &ws.c_gu_ptrs,
                                &ws.a_dn_ptrs, &ws.b_dn_ptrs, &ws.c_dn_ptrs}) {
            *t = DeviceTensor::allocate(DType::INT64, {std::max<std::int64_t>(pw, 1)});
        }
    }

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
    DsV4CompressCache& comp_cache,
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
    // Read the local extents off the bound weights, not off `moe_I / T`.
    //
    // Dividing the config by the TP degree re-derives, here, a decision the
    // contract already made over there, and nothing checks that the two agree.
    // When they disagree the shapes still look plausible and every kernel
    // still launches -- the answer is just wrong, which is the failure this
    // family had at TP=2. `embed` and `lm_head` in this same function already
    // read `shape()`; this is that, for the MoE.
    auto extent_of = [](const DeviceTensor* t, std::size_t axis, int fallback) {
        return (t != nullptr && axis < t->shape().size())
                   ? static_cast<int>(t->shape()[axis])
                   : fallback;
    };
    int local_shared_I = moe_I / T;
    int local_moe_I = moe_I / T;
    for (const auto& L : w.layers) {
        // `shared_w1` is `[I_local, H]`; the stacked routed slab is
        // `[E, 2*I_local, H]`, so its gate and up halves come apart on axis 1.
        if (L.shared_w1 != nullptr) {
            local_shared_I = extent_of(L.shared_w1, 0, local_shared_I);
        }
        if (L.moe_gate_up_bf16 != nullptr) {
            local_moe_I = extent_of(L.moe_gate_up_bf16, 1, local_moe_I * 2) / 2;
        }
        if (L.shared_w1 != nullptr && L.moe_gate_up_bf16 != nullptr) break;
    }
    const int M = cfg.dsv4_hc_mult;
    const int mix_hc = (2 + M) * M;
    const float hc_eps = cfg.dsv4_hc_eps;
    const float hc_post_alpha = 2.0f;
    const int sinkhorn_iters = 20; // from config hc_sinkhorn_iters
    // Every V4 layer attends to a sliding window of raw tokens; vLLM uses
    // `swa_len = min(position + 1, sliding_window)`, i.e. the query plus
    // `sliding_window - 1` predecessors.
    const int swa_window_left =
        cfg.dsv4_sliding_window > 0 ? cfg.dsv4_sliding_window - 1 : -1;

    // Must be cublas's stream, not the null stream: CUDA-graph capture runs on
    // a private stream and only work enqueued there is recorded into the graph.
    cudaStream_t stream = cublas.stream();
    act_dump_step_begin(stream);

    // Env-gated per-stage timer. `PIE_DSV4_PROFILE=1` records CUDA events
    // and syncs at every probe, giving GPU time per stage. `=2` instead
    // takes host wall-clock around each stage with no sync, giving the
    // *submit* cost — the two answer different questions, and DSV4 decode
    // is submit-bound, so both are needed.
    static const int prof_mode = [] {
        const char* v = std::getenv("PIE_DSV4_PROFILE");
        return (v == nullptr || v[0] == '\0') ? 0 : std::atoi(v);
    }();
    const bool prof_on = prof_mode == 1;
    const bool prof_host = prof_mode == 2;
    static std::vector<std::pair<std::string, double>> prof_acc;
    static int prof_steps = 0;
    static std::chrono::steady_clock::time_point ph_a;
    cudaEvent_t pe_a = nullptr, pe_b = nullptr;
    if (prof_host) prof_acc.clear();
    if (prof_on) {
        CUDA_CHECK(cudaEventCreate(&pe_a));
        CUDA_CHECK(cudaEventCreate(&pe_b));
        prof_acc.clear();
    }
    auto prof_add = [&](const char* label, float ms) {
        for (auto& kv : prof_acc) {
            if (kv.first == label) { kv.second += ms; return; }
        }
        prof_acc.emplace_back(label, ms);
    };
    auto prof_beg = [&] {
        if (prof_host) { ph_a = std::chrono::steady_clock::now(); return; }
        if (prof_on) CUDA_CHECK(cudaEventRecord(pe_a, stream));
    };
    auto prof_end = [&](const char* label) {
        if (prof_host) {
            prof_add(label, std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - ph_a).count());
            return;
        }
        if (!prof_on) return;
        CUDA_CHECK(cudaEventRecord(pe_b, stream));
        CUDA_CHECK(cudaEventSynchronize(pe_b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pe_a, pe_b));
        prof_add(label, ms);
    };

    prof_beg();
    // ── Compressor boundary bookkeeping ──────────────────────────────
    // Compressing layers all need the same three small per-entry index
    // arrays, which depend only on the compress ratio — so they are built
    // once per distinct ratio instead of once per layer. `comp_meta` holds
    // them back to back, with the last slice reserved for the per-token
    // request index that the compressed attention needs.
    struct CompBoundaries {
        int count = 0;
        const std::int32_t* pos_d = nullptr;       // boundary absolute position
        const std::int32_t* req_d = nullptr;       // owning request
        const std::int32_t* rope_pos_d = nullptr;  // (pos / ratio) * ratio
    };
    std::vector<std::pair<int, CompBoundaries>> comp_boundary_cache;
    const std::int32_t* req_of_token_d = nullptr;
    int comp_meta_slices_used = 0;
    const int comp_meta_stride =
        ws.comp_meta.empty() ? 0 : static_cast<int>(ws.comp_meta.shape()[0] / N);
    std::vector<std::int32_t> pos_h;

    // Pure decode (one token per request) takes a fully device-side path: no
    // D2H of `positions`, no host compaction, no blocking H2D — which is what
    // makes the layer capturable into a CUDA graph. Everything else keeps the
    // host path, where compaction is worth the sync because N can be large.
    static const bool dev_boundary = [] {
        const char* v = std::getenv("PIE_DSV4_DEV_BOUNDARY");
        return v == nullptr || v[0] != '0';
    }();
    const bool decode_only = dev_boundary && (N == num_requests) && N > 0;

    auto comp_boundaries = [&](int ratio) -> const CompBoundaries& {
        for (auto& kv : comp_boundary_cache) {
            if (kv.first == ratio) return kv.second;
        }
        std::int32_t* meta_d = static_cast<std::int32_t*>(ws.comp_meta.data());

        if (decode_only) {
            if (req_of_token_d == nullptr) {
                req_of_token_d =
                    meta_d + static_cast<std::size_t>(comp_meta_stride - 1) * N;
            }
            CompBoundaries cb;
            cb.count = N;
            std::int32_t* base =
                meta_d + static_cast<std::size_t>(comp_meta_slices_used) * N;
            comp_meta_slices_used += 3;
            kernels::launch_dsv4_boundary_meta_decode(
                static_cast<const std::int32_t*>(positions),
                base, base + N, base + 2 * N, N, ratio, stream, row_valid_d);
            // The compressed attention wants a token->request map; in pure
            // decode that is the identity, which the same kernel just wrote.
            CUDA_CHECK(cudaMemcpyAsync(
                const_cast<std::int32_t*>(req_of_token_d), base + N,
                static_cast<std::size_t>(N) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice, stream));
            cb.pos_d = base;
            cb.req_d = base + N;
            cb.rope_pos_d = base + 2 * N;
            comp_boundary_cache.emplace_back(ratio, cb);
            return comp_boundary_cache.back().second;
        }
        if (pos_h.empty()) {
            pos_h.resize(static_cast<std::size_t>(N));
            CUDA_CHECK(cudaMemcpyAsync(pos_h.data(), positions,
                static_cast<std::size_t>(N) * sizeof(std::int32_t),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Last slice: request index of every token in the batch.
            std::vector<std::int32_t> req_of_token(static_cast<std::size_t>(N), 0);
            for (int r = 0; r < num_requests; ++r) {
                for (std::uint32_t t = qo_indptr_h[r]; t < qo_indptr_h[r + 1]; ++t) {
                    req_of_token[t] = r;
                }
            }
            req_of_token_d = meta_d + static_cast<std::size_t>(comp_meta_stride - 1) * N;
            // Blocking copy on purpose: the source is a stack-local vector that
            // dies at the end of this lambda, so an async copy out of pageable
            // memory is a use-after-free race. It only shows up when the rest
            // of the layer is device-side (no stream sync to hide it).
            CUDA_CHECK(cudaMemcpy(const_cast<std::int32_t*>(req_of_token_d),
                req_of_token.data(),
                static_cast<std::size_t>(N) * sizeof(std::int32_t),
                cudaMemcpyHostToDevice));
        }

        std::vector<std::int32_t> b_pos, b_req, b_rope;
        for (int r = 0; r < num_requests; ++r) {
            for (std::uint32_t t = qo_indptr_h[r]; t < qo_indptr_h[r + 1]; ++t) {
                const int p = pos_h[t];
                if (((p + 1) % ratio) != 0) continue;
                b_pos.push_back(p);
                b_req.push_back(r);
                b_rope.push_back((p / ratio) * ratio);
            }
        }

        CompBoundaries cb;
        cb.count = static_cast<int>(b_pos.size());
        if (cb.count > 0) {
            std::int32_t* base =
                meta_d + static_cast<std::size_t>(comp_meta_slices_used) * N;
            comp_meta_slices_used += 3;
            const std::size_t nbytes =
                static_cast<std::size_t>(cb.count) * sizeof(std::int32_t);
            CUDA_CHECK(cudaMemcpy(base, b_pos.data(), nbytes,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(base + N, b_req.data(), nbytes,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(base + 2 * N, b_rope.data(), nbytes,
                cudaMemcpyHostToDevice));
            cb.pos_d = base;
            cb.req_d = base + N;
            cb.rope_pos_d = base + 2 * N;
        }
        comp_boundary_cache.emplace_back(ratio, cb);
        return comp_boundary_cache.back().second;
    };

    // Resolve every ratio up front. Each of the copies above is blocking on
    // the default stream, so resolving lazily inside the layer loop drained
    // the whole pipeline mid-forward — once per distinct ratio, each time
    // waiting on every layer enqueued so far. Doing it here costs nothing:
    // the positions D2H already synced, so the stream is empty.
    if (comp_meta_stride > 0 && N > 0) {
        bool any_compressing = false;
        for (int li = 0; li < static_cast<int>(w.layers.size()); ++li) {
            const auto& Lw = w.layers[li];
            if (Lw.compress_ratio > 0 && Lw.compressor.wkv != nullptr &&
                Lw.compressor.wgate != nullptr && comp_cache.has_layer(li)) {
                any_compressing = true;
                break;
            }
        }
        if (any_compressing) {
            for (int li = 0; li < static_cast<int>(w.layers.size()); ++li) {
                const auto& Lw = w.layers[li];
                if (Lw.compress_ratio > 0 && Lw.compressor.wkv != nullptr &&
                    Lw.compressor.wgate != nullptr && comp_cache.has_layer(li)) {
                    comp_boundaries(Lw.compress_ratio);
                }
            }
        }
    }

    prof_end("comp_bnd");
    prof_beg();
    // ── Sliding-window attention ─────────────────────────────────────
    // Every layer attends with the same geometry, so the FlashInfer plan is
    // built once per fire — in DsV4Model::prepare(), which runs outside any
    // CUDA-graph capture region. The naive fallback kernel this replaces
    // walked the KV scalar-wise (one `load_kv_scalar` call per element per
    // query head) and cost ~65% of the whole step.
    static const bool swa_force_naive = [] {
        const char* v = std::getenv("PIE_DSV4_NAIVE_SWA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    const bool swa_use_flashinfer = !swa_force_naive && ws.swa_plan_valid;
    prof_end("swa_plan");
    prof_beg();
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

    act_dump_i32("tokens", token_ids, N, 1, stream);
    act_dump_bf16("embed", ws.y.data(), N, H, stream);
    act_dump_bf16("hc_expand", ws.hc_residual.data(), N, M * H, stream);

    // vLLM `build_deepseek_v4_rope`: one rope per layer, whose base is
    // `compress_rope_theta` when the layer compresses (`max(1, ratio) > 1`)
    // and `rope_theta` otherwise, built with `is_neox_style=False` (GPT-J
    // pairing) and the YaRN inv_freq ramp. `mscale`/`mscale_all_dim` are both
    // forced to 0 there ("Disable mscale"), so cos/sin are unscaled.
    const bool rope_yarn =
        cfg.rope_scaling_kind == HfConfig::RopeScaling::OriginalYaRN;
    const float rope_yarn_factor = rope_yarn ? cfg.rope_factor : 1.f;
    auto layer_rope_theta = [&](int layer) -> float {
        const int ratio = (layer < static_cast<int>(cfg.dsv4_compress_ratios.size()))
            ? std::max(1, cfg.dsv4_compress_ratios[static_cast<std::size_t>(layer)])
            : 1;
        return (ratio > 1 && cfg.dsv4_compress_rope_theta > 0.f)
            ? cfg.dsv4_compress_rope_theta
            : cfg.rope_theta;
    };

    // ── Per-layer processing ─────────────────────────────────────────
    for (int li = 0; li < L; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];
        const float rope_theta_l = layer_rope_theta(li);

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

        act_dump_bf16(act_dump_layer_tag("norm_x", li).c_str(),
            ws.norm_x.data(), N, H, stream);

        prof_end("pre_layers");
        prof_beg();
        // Q path: wq_a → q_norm → wq_b
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.wq_a, Lw.wq_a_quant), ws.q_a.data(),
            N, q_lora, H);

        kernels::launch_rmsnorm_bf16(
            ws.q_a.data(), Lw.q_norm->data(), ws.q_a.data(),
            N, q_lora, eps, stream);

        act_dump_bf16(act_dump_layer_tag("q_a", li).c_str(),
            ws.q_a.data(), N, q_lora, stream);
        ops::gemm_act_x_w(cublas.handle(),
            ws.q_a.data(), make_weight_view(Lw.wq_b, Lw.wq_b_quant), ws.q.data(),
            N, num_heads * head_dim, q_lora);
        act_dump_bf16(act_dump_layer_tag("q_b", li).c_str(),
            ws.q.data(), N, num_heads * head_dim, stream);
        invoke_stage_hook(
            hooks,
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
        act_dump_bf16(act_dump_layer_tag("kv", li).c_str(),
            ws.kv.data(), N, head_dim, stream);

        // Partial RoPE on last qk_rope dims of Q and KV
        kernels::launch_rope_partial_last_bf16(
            ws.q.data(), ws.kv.data(),
            positions, N, num_heads, 1, head_dim, qk_rope,
            rope_theta_l, stream, /*inverse=*/false, /*interleaved=*/true,
            rope_yarn_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
            cfg.rope_original_max_position);
            prof_end("qkv_proj");
            prof_beg();
        {
            // SWA attention on ALL layers (every layer has a sliding window).
            auto lv = kv_cache.layer_view(li);
            kernels::launch_write_kv_to_pages_bf16(
                lv.k_pages, lv.v_pages,
                ws.kv.data(), ws.kv.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, num_requests, lv.page_size,
                1, head_dim, false, stream, row_valid_d);

            if (swa_use_flashinfer) {
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *ws.swa_plan,
                    ws.q.data(), lv.k_pages, lv.v_pages, ws.attn_out.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, attn_ws, stream,
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f,
                    static_cast<float*>(ws.attn_lse.data()));
                // FlashInfer emits `m + log2(d)`; the compressed-attention
                // LSE this is merged with is a natural log.
                kernels::launch_lse_log2_to_ln(
                    static_cast<float*>(ws.attn_lse.data()),
                    N * num_heads, stream);
            } else {
                ops::launch_attention_naive_paged_bf16(
                    ws.q.data(), lv.k_pages, lv.v_pages,
                    ws.attn_out.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    N, num_requests, num_heads, 1, head_dim, lv.page_size, stream,
                    /*window_left=*/swa_window_left, /*sm_scale=*/-1.f,
                    /*logits_soft_cap=*/0.f,
                    /*lse_out=*/static_cast<float*>(ws.attn_lse.data()));
            }

            prof_end("swa_attn");
            prof_beg();
            // ── Compressed attention (C4/C128 layers) ────────────────
            // Every compressing layer keeps two page-parallel caches alive
            // across forward passes: the per-token compressor state and the
            // finished compressed entries. Prefill and decode therefore run
            // exactly the same code — a decode step saves its one token's
            // state, emits an entry if that token closes a window, and then
            // attends to every entry the request has produced so far.
            const int comp_ratio = Lw.compress_ratio;
            if (comp_ratio > 0 &&
                Lw.compressor.wkv != nullptr &&
                Lw.compressor.wgate != nullptr &&
                comp_cache.has_layer(li))
            {
                const int coff = (comp_ratio == 4) ? 2 : 1;
                const int proj_dim = coff * head_dim;
                const int comp_page_size = comp_cache.page_size();

                // Step 1: project hidden states through compressor.wkv/wgate.
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.compressor.wkv->data(),
                    ws.comp_kv_proj.data(),
                    N, proj_dim, H);

                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), Lw.compressor.wgate->data(),
                    ws.comp_score_proj.data(),
                    N, proj_dim, H);

                // Step 2: persist this batch's state. The KV page writer works
                // verbatim with kv/score standing in for k/v.
                kernels::launch_write_kv_to_pages_bf16(
                    comp_cache.state_kv(li), comp_cache.state_score(li),
                    ws.comp_kv_proj.data(), ws.comp_score_proj.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    N, num_requests, comp_page_size,
                    1, proj_dim, false, stream, row_valid_d);

                const CompBoundaries& cb = comp_boundaries(comp_ratio);
                if (cb.count > 0) {
                    // Step 3: pool each new window into one entry, RMSNorm it,
                    // and RoPE it at (position / ratio) * ratio.
                    kernels::launch_dsv4_compress_gather_paged_bf16(
                        comp_cache.state_kv(li), comp_cache.state_score(li),
                        Lw.compressor.ape != nullptr
                            ? static_cast<const float*>(Lw.compressor.ape->data())
                            : nullptr,
                        cb.pos_d, cb.req_d, kv_page_indices, kv_page_indptr,
                        ws.comp_kv.data(),
                        cb.count, head_dim, comp_ratio, coff, comp_page_size,
                        stream);

                    if (Lw.compressor.norm != nullptr) {
                        kernels::launch_rmsnorm_bf16(
                            ws.comp_kv.data(), Lw.compressor.norm->data(),
                            ws.comp_kv.data(),
                            cb.count, head_dim, eps, stream);
                    }

                    kernels::launch_rope_partial_last_bf16(
                        ws.comp_kv.data(),  // "q" — will be rotated
                        ws.comp_kv.data(),  // "k" — dummy (0 kv heads below)
                        cb.rope_pos_d,
                        cb.count,
                        1,      // num_q_heads: single head, rotate just the KV
                        0,      // num_kv_heads: 0 means skip K rotation
                        head_dim,
                        qk_rope,
                        rope_theta_l,
                        stream,
                        /*inverse=*/false, /*interleaved=*/true,
                        rope_yarn_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
                        cfg.rope_original_max_position);

                    kernels::launch_dsv4_store_comp_entries_bf16(
                        ws.comp_kv.data(), comp_cache.comp_kv(li),
                        cb.pos_d, cb.req_d, kv_page_indices, kv_page_indptr,
                        cb.count, head_dim, comp_page_size, stream);
                }

                // Step 4: attend to every entry this request has produced,
                // then merge with the sliding-window attention by LSE.
                const float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                kernels::launch_attention_compressed_paged_bf16(
                    ws.q.data(), comp_cache.comp_kv(li),
                    ws.comp_attn_out.data(),
                    static_cast<float*>(ws.comp_attn_lse.data()),
                    positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    req_of_token_d,
                    N, num_heads, head_dim, comp_ratio, comp_page_size,
                    sm_scale, stream);

                kernels::launch_combine_attn_outputs_bf16(
                    ws.attn_out.data(),
                    static_cast<const float*>(ws.attn_lse.data()),
                    ws.comp_attn_out.data(),
                    static_cast<const float*>(ws.comp_attn_lse.data()),
                    ws.attn_out.data(),   // write combined result back
                    static_cast<float*>(ws.attn_lse.data()),  // update LSE
                    N, num_heads, head_dim, stream);
            }

            prof_end("comp_attn");
            prof_beg();
            // Attention sink correction: o *= sigmoid(lse - sink_h)
            if (Lw.attn_sink != nullptr) {
                kernels::launch_attn_sink_correction_bf16(
                    ws.attn_out.data(),
                    static_cast<const float*>(ws.attn_lse.data()),
                    static_cast<const float*>(Lw.attn_sink->data()),
                    N, num_heads, head_dim, stream);
            }
            invoke_stage_hook(
                hooks,
                StageHookPoint::OnAttn, ws.q.data(),
                static_cast<std::uint32_t>(N),
                static_cast<std::uint32_t>(num_heads * head_dim),
                static_cast<std::uint32_t>(li), stream);

            act_dump_bf16(act_dump_layer_tag("attn_out", li).c_str(),
                ws.attn_out.data(), N, num_heads * head_dim, stream);

            // Inverse RoPE on attention output (removes position info before projection)
            kernels::launch_rope_partial_last_bf16(
                ws.attn_out.data(), ws.attn_out.data(),
                positions, N, num_heads, 0, head_dim, qk_rope,
                rope_theta_l, stream, /*inverse=*/true, /*interleaved=*/true,
                rope_yarn_factor, cfg.rope_beta_fast, cfg.rope_beta_slow,
                cfg.rope_original_max_position);

            act_dump_bf16(act_dump_layer_tag("attn_irope", li).c_str(),
                ws.attn_out.data(), N, num_heads * head_dim, stream);

            prof_end("attn_epilogue");
            prof_beg();
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

            // Workspace for the dequantised group slice when FP8. Sized for
            // this weight rather than borrowed from the expert scratch, whose
            // extent is a function of tp_size and so cannot be relied on to
            // cover a replicated weight.
            void* dequant_buf = ws.wo_a_dequant.data();
            if (dequant_buf == nullptr ||
                ws.wo_a_dequant.nbytes() <
                    static_cast<std::size_t>(o_lora) * per_group_dim * 2) {
                throw std::runtime_error(
                    "deepseek_v4: wo_a dequant workspace too small");
            }

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

            act_dump_bf16(act_dump_layer_tag("wo_a_in", li).c_str(),
                ws.attn_out.data(), N, num_heads * head_dim, stream);
            act_dump_bf16(act_dump_layer_tag("wo_a_out", li).c_str(),
                ws.wo_a_out.data(), N, out_dim, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.wo_a_out.data(),
                make_weight_view(Lw.wo_b, Lw.wo_b_quant),
                ws.y.data(),
                N, H, out_dim);
            act_dump_bf16(act_dump_layer_tag("o_proj", li).c_str(),
                ws.y.data(), N, H, stream);

            // TODO: all-reduce wo_b output across TP ranks
        }

        prof_end("o_proj");
        prof_beg();
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


        act_dump_bf16(act_dump_layer_tag("res_attn", li).c_str(),
            ws.hc_residual.data(), N, M * H, stream);

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


        act_dump_bf16(act_dump_layer_tag("ffn_in", li).c_str(),
            ws.norm_y.data(), N, H, stream);

        prof_end("hc_attn");
        prof_beg();
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
                ws.router_logits.data(),
                static_cast<std::int32_t*>(ws.topk_idx.data()),
                static_cast<float*>(ws.topk_weights.data()),
                N, cfg.vocab_size, E, K,
                cfg.norm_topk_prob,
                cfg.routed_scaling_factor,
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

        prof_end("moe_router");
        prof_beg();
        // Routed expert dispatch (MXFP4 weights, dequant to bf16)
        cudaMemsetAsync(ws.moe_out.data(), 0,
                        static_cast<std::size_t>(N) * H * 2, stream);
        const bool stacked = Lw.moe_gate_up_bf16 != nullptr;
        // Streamed experts take the per-expert path and only that path. The
        // two device-side fast paths index a stack by `weight_base + e*N*K`,
        // which is a statement about a slab that holds every expert in routing
        // order; a slot holds whichever expert was asked for last, so there is
        // no stride that makes them mean the right thing.
        //
        // That path is commented below as correct and slow, and it is, but the
        // comparison has changed: reading an expert off an SSD costs more than
        // any dispatch, so what used to be the fallback's penalty is now noise
        // against the transfer it exists to serve.
        const bool streamed = Lw.expert_cache != nullptr;
        if (stacked || streamed ||
            (!Lw.experts.empty() && Lw.experts[0].w1 != nullptr)) {
            // Either the contract published dense bf16 stacks or it left the
            // experts packed, and exactly one of the two is bound. cuBLAS
            // cannot read the packed form, so the packed path pays a dequant
            // of every routed expert on every layer of every step -- far more
            // bytes than the GEMMs that consume it -- which is why the stacks
            // are the default.
            if (stacked && ws.aligned_block_size > 0 &&
                N <= ops::moe_gemv_max_tokens(kDsV4MoeGemvMaxTokens) &&
                (H % 8) == 0 &&
                (local_moe_I % 8) == 0) {
                // Decode: at M=1 the routed GEMMs stream weights with no reuse,
                // so a warp-per-output-row GEMV beats a batched GEMM whose
                // tiling assumes an M worth filling.
                const int routes = N * K;
                kernels::launch_moe_gate_up_decode_gemv_bf16(
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    ws.norm_y.data(), Lw.moe_gate_up_bf16->data(),
                    ws.aligned_gate_up.data(),
                    N, K, H, local_moe_I, stream);
                dsv4_chunked_swiglu(ws.aligned_gate_up.data(),
                    ws.aligned_act.data(), routes, local_moe_I,
                    cfg.swiglu_limit, stream);
                kernels::launch_moe_down_decode_gemv_bf16(
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    ws.aligned_act.data(), Lw.moe_down_bf16->data(),
                    ws.expert_out.data(),
                    N, K, H, local_moe_I, stream);
                kernels::launch_token_batched_weighted_sum_bf16(
                    ws.moe_out.data(), ws.expert_out.data(),
                    static_cast<const float*>(ws.topk_weights.data()),
                    N, K, H, stream);
            } else if (stacked && ws.aligned_block_size > 0) {
                // Prefill: bucket routes into fixed-size expert blocks on
                // device and run two batched GEMMs. No host round-trip, no
                // stream sync, and the reduction is deterministic.
                const int routes = N * K;
                const int block = std::min(ws.aligned_block_size,
                                           kernels::moe_aligned_block(routes, E));
                const int active_expert_cap = std::min(E, routes);
                const int max_blocks =
                    (routes + active_expert_cap * (block - 1) + block - 1) / block;
                const int aligned_rows = max_blocks * block;
                if (max_blocks > ws.aligned_max_blocks ||
                    aligned_rows > static_cast<int>(ws.aligned_expert_in.shape()[0])) {
                    throw std::runtime_error("dsv4: aligned MoE scratch too small");
                }

                kernels::launch_moe_align_decode(
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    static_cast<std::int32_t*>(ws.aligned_route_ids.data()),
                    static_cast<std::int32_t*>(ws.aligned_expert_ids.data()),
                    /*route_to_aligned_row=*/nullptr,
                    routes, E, block, max_blocks, stream);
                kernels::launch_gather_moe_aligned_inputs_bf16(
                    ws.norm_y.data(),
                    static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                    ws.aligned_expert_in.data(),
                    routes, aligned_rows, K, H,
                    /*shared_row_begin=*/-1, N, stream);
                kernels::launch_build_moe_ptrs_aligned_bf16(
                    static_cast<const std::int32_t*>(ws.aligned_expert_ids.data()),
                    Lw.moe_gate_up_bf16->data(), Lw.moe_down_bf16->data(),
                    ws.aligned_expert_in.data(), ws.aligned_gate_up.data(),
                    ws.aligned_act.data(), ws.aligned_out.data(),
                    reinterpret_cast<const void**>(ws.a_gu_ptrs.data()),
                    reinterpret_cast<const void**>(ws.b_gu_ptrs.data()),
                    reinterpret_cast<void**>(ws.c_gu_ptrs.data()),
                    reinterpret_cast<const void**>(ws.a_dn_ptrs.data()),
                    reinterpret_cast<const void**>(ws.b_dn_ptrs.data()),
                    reinterpret_cast<void**>(ws.c_dn_ptrs.data()),
                    max_blocks, block, H, local_moe_I,
                    /*routed_blocks=*/max_blocks, nullptr, nullptr, stream);

                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(ws.b_gu_ptrs.data()),
                    reinterpret_cast<const void* const*>(ws.a_gu_ptrs.data()),
                    reinterpret_cast<void* const*>(ws.c_gu_ptrs.data()),
                    block, 2 * local_moe_I, H, max_blocks);
                dsv4_chunked_swiglu(ws.aligned_gate_up.data(),
                    ws.aligned_act.data(), aligned_rows, local_moe_I,
                    cfg.swiglu_limit, stream);
                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                    reinterpret_cast<const void* const*>(ws.b_dn_ptrs.data()),
                    reinterpret_cast<const void* const*>(ws.a_dn_ptrs.data()),
                    reinterpret_cast<void* const*>(ws.c_dn_ptrs.data()),
                    block, H, local_moe_I, max_blocks);
                kernels::launch_reorder_moe_aligned_output_bf16(
                    ws.aligned_out.data(),
                    static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                    ws.expert_out.data(),
                    routes, aligned_rows, H,
                    /*shared_row_begin=*/-1, N, nullptr, stream);
                kernels::launch_token_batched_weighted_sum_bf16(
                    ws.moe_out.data(), ws.expert_out.data(),
                    static_cast<const float*>(ws.topk_weights.data()),
                    N, K, H, stream);
            } else {
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
            if (streamed) {
                // The whole layer's routing is known before any of it runs, and
                // experts are paged one at a time, so say which ones are wanted
                // now: the read for the next overlaps the GEMMs for this one.
                for (int e = 0; e < E; ++e) {
                    if (routing.token_idx[static_cast<std::size_t>(e)].empty()) {
                        continue;
                    }
                    Lw.expert_cache->prefetch(
                        Lw.expert_group, static_cast<std::uint32_t>(e));
                }
            }
            for (int e = 0; e < E; ++e) {
                const auto& tok_idx = routing.token_idx[static_cast<std::size_t>(e)];
                const int Ne = static_cast<int>(tok_idx.size());
                if (Ne == 0) continue;
                const auto& wts = routing.weights[static_cast<std::size_t>(e)];

                const void* w_gate = ws.expert_gate_w.data();
                const void* w_up   = ws.expert_up_w.data();
                const void* w_down = ws.expert_down_w.data();
                if (streamed) {
                    // Page it in. The slot comes back holding exactly what the
                    // stack would have held for this expert -- the group's
                    // plan is the stack's expression with the expert axis
                    // removed, so the dequantize already ran on the way in and
                    // these are bf16 weights, not packed ones.
                    const WeightStore& slot = Lw.expert_cache->ensure_resident(
                        Lw.expert_group, static_cast<std::uint32_t>(e), stream);
                    const auto* gu_e = static_cast<const std::uint16_t*>(
                        slot.get("gate_up.weight").data());
                    w_gate = gu_e;
                    w_up   = gu_e + static_cast<std::size_t>(local_moe_I) * H;
                    w_down = slot.get("down.weight").data();
                } else if (stacked) {
                    const auto* gu_e =
                        static_cast<const std::uint16_t*>(Lw.moe_gate_up_bf16->data()) +
                        static_cast<std::size_t>(e) * 2 * local_moe_I * H;
                    w_gate = gu_e;
                    w_up   = gu_e + static_cast<std::size_t>(local_moe_I) * H;
                    w_down =
                        static_cast<const std::uint16_t*>(Lw.moe_down_bf16->data()) +
                        static_cast<std::size_t>(e) * H * local_moe_I;
                } else {
                    const auto& ew = Lw.experts[static_cast<std::size_t>(e)];
                    if (!ew.w1 || !ew.w1_scale) continue;
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
                }

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
                    ops::WeightView::raw(w_gate, DType::BF16),
                    ws.expert_gate.data(), Ne, local_moe_I, H);
                ops::gemm_act_x_w(cublas.handle(),
                    ws.expert_in.data(),
                    ops::WeightView::raw(w_up, DType::BF16),
                    ws.expert_up.data(), Ne, local_moe_I, H);
                if (cfg.swiglu_limit > 0.f) {
                    kernels::launch_swiglu_clamp_bf16(
                        ws.expert_gate.data(), ws.expert_up.data(),
                        ws.expert_gate.data(),
                        static_cast<int>(Ne) * local_moe_I,
                        cfg.swiglu_limit, stream);
                } else {
                    kernels::launch_swiglu_bf16(
                        ws.expert_gate.data(), ws.expert_up.data(),
                        ws.expert_gate.data(),
                        static_cast<std::size_t>(Ne) * local_moe_I, stream);
                }
                ops::gemm_act_x_w(cublas.handle(),
                    ws.expert_gate.data(),
                    ops::WeightView::raw(w_down, DType::BF16),
                    ws.expert_out.data(), Ne, H, local_moe_I);

                kernels::launch_scatter_add_weighted_bf16(
                    ws.moe_out.data(), ws.expert_out.data(),
                    static_cast<const std::int32_t*>(ws.route_idx.data()),
                    static_cast<const float*>(ws.route_w.data()),
                    Ne, H, stream);

                if (streamed) {
                    // Unpin here rather than at the end of the layer, so that
                    // the pin covers exactly the launches that read the slot
                    // and no more. Holding every routed expert of a layer
                    // pinned at once would make the slab's minimum size the
                    // number of experts a step happens to route to, which is
                    // not a number anyone can budget for; this way one slot is
                    // enough to be correct and more slots only buy hits.
                    //
                    // Nothing is racing: these GEMMs are queued on `stream`,
                    // and a later page-in that wants this slot synchronizes
                    // `stream` before it writes.
                    Lw.expert_cache->end_batch();
                }
            }
            }  // end per-expert fallback
        }

        act_dump_bf16(act_dump_layer_tag("routed_out", li).c_str(),
            ws.moe_out.data(), N, H, stream);

        prof_end("moe_experts");
        prof_beg();
        // Shared expert (sharded by TP)
        if (Lw.shared_w1 != nullptr) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(Lw.shared_w1, Lw.shared_w1_quant), ws.shared_gate.data(),
                N, local_shared_I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(Lw.shared_w3, Lw.shared_w3_quant), ws.shared_up.data(),
                N, local_shared_I, H);
            if (cfg.swiglu_limit > 0.f) {
                kernels::launch_swiglu_clamp_bf16(
                    ws.shared_gate.data(), ws.shared_up.data(),
                    ws.shared_act.data(),
                    N * local_shared_I, cfg.swiglu_limit, stream);
            } else {
                kernels::launch_swiglu_bf16(
                    ws.shared_gate.data(), ws.shared_up.data(),
                    ws.shared_act.data(),
                    static_cast<std::size_t>(N) * local_shared_I, stream);
            }
            ops::gemm_act_x_w(cublas.handle(),
                ws.shared_act.data(), make_weight_view(Lw.shared_w2, Lw.shared_w2_quant), ws.shared_out.data(),
                N, H, local_shared_I);
            act_dump_bf16(act_dump_layer_tag("shared_out", li).c_str(),
                ws.shared_out.data(), N, H, stream);
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

        prof_end("moe_shared");
        prof_beg();
        // ── HC post (FFN) ────────────────────────────────────────────
        act_dump_bf16(act_dump_layer_tag("ffn_out", li).c_str(),
            ws.moe_out.data(), N, H, stream);
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
        act_dump_bf16(act_dump_layer_tag("out", li).c_str(),
                      ws.hc_residual.data(), N, M * H, stream);
    }

    prof_end("hc_ffn");
    prof_beg();
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
    act_dump_bf16("hc_head", final_in, rows, H, stream);
    act_dump_bf16("final_norm", ws.norm_y.data(), rows, H, stream);

    if (fwd_cfg.emit_logits) {
        const int local_vocab = static_cast<int>(w.lm_head->shape()[0]);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_y.data(), ops::WeightView(*w.lm_head), logits_out,
            rows, local_vocab, H);
        act_dump_bf16("logits", logits_out, rows, local_vocab, stream);
    }

    if (prof_on || prof_host) {
        prof_end("head");
        double total = 0.0;
        for (auto& kv : prof_acc) total += kv.second;
        std::cerr << "[pie-dsv4-profile] step=" << prof_steps++
                  << " tokens=" << N << " requests=" << num_requests
                  << " total_ms=" << total;
        for (auto& kv : prof_acc) {
            std::cerr << " " << kv.first << "=" << kv.second;
        }
        std::cerr << "\n";
        if (prof_on) {
            CUDA_CHECK(cudaEventDestroy(pe_a));
            CUDA_CHECK(cudaEventDestroy(pe_b));
        }
    }
}

}  // namespace pie_cuda_driver::model
