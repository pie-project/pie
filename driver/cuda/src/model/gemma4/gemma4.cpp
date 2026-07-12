#include "model/gemma4/gemma4.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/argmax.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_naive_paged.hpp"

namespace pie_cuda_driver::model {

namespace {

thread_local bool g_logits_argmax_only = false;
thread_local std::int32_t* g_fused_argmax_output = nullptr;
thread_local bool g_fused_argmax_done = false;

bool fused_lmhead_argmax_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_FUSED_LMHEAD_ARGMAX");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool gemma4_forward_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_FORWARD_PROFILE");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

std::uint64_t gemma4_forward_profile_limit() {
    static const std::uint64_t limit = []() -> std::uint64_t {
        const char* v = std::getenv("PIE_GEMMA4_FORWARD_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return 64ull;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(v, &end, 10);
        if (end == v) return 64ull;
        return static_cast<std::uint64_t>(parsed);
    }();
    return limit;
}

struct Gemma4ForwardProfile {
    bool enabled = false;
    bool ended = false;
    std::uint64_t seq = 0;
    int N = 0;
    int R = 0;
    int layers = 0;
    int lm_head_rows = 0;
    bool pure_decode = false;
    bool decode_path = false;
    bool row_decode_path = false;
    bool compact_logits = false;
    bool logits_argmax_only = false;

    float total_gpu_ms = 0.f;
    float embed_ms = 0.f;
    float ple_inputs_ms = 0.f;
    float attn_prep_ms = 0.f;
    float attention_ms = 0.f;
    float attn_out_ms = 0.f;
    float mlp_ms = 0.f;
    float ple_residual_ms = 0.f;
    float final_norm_ms = 0.f;
    float lm_head_ms = 0.f;
    float softcap_ms = 0.f;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~Gemma4ForwardProfile() {
        if (total_start != nullptr) cudaEventDestroy(total_start);
        if (total_stop != nullptr) cudaEventDestroy(total_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
    }

    void begin(cudaStream_t stream) {
        static std::atomic<std::uint64_t> counter{0};
        if (!gemma4_forward_profile_enabled()) return;
        const std::uint64_t current =
            counter.fetch_add(1, std::memory_order_relaxed);
        const std::uint64_t limit = gemma4_forward_profile_limit();
        if (limit != 0 && current >= limit) return;
        enabled = true;
        seq = current;
        CUDA_CHECK(cudaEventCreate(&total_start));
        CUDA_CHECK(cudaEventCreate(&total_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
        CUDA_CHECK(cudaEventRecord(total_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled || ended) return;
        ended = true;
        CUDA_CHECK(cudaEventRecord(total_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_gpu_ms, total_start, total_stop));
    }
};

template <class Fn>
void profile_gemma4_cuda_stage(
    Gemma4ForwardProfile& profile,
    float& dst,
    cudaStream_t stream,
    Fn&& fn)
{
    if (!profile.enabled) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile.stage_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile.stage_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile.stage_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile.stage_start,
                                    profile.stage_stop));
    dst += ms;
}

void maybe_print_gemma4_forward_profile(
    Gemma4ForwardProfile& p,
    cudaStream_t stream)
{
    if (!p.enabled) return;
    p.end(stream);
    const float named =
        p.embed_ms + p.ple_inputs_ms + p.attn_prep_ms + p.attention_ms +
        p.attn_out_ms + p.mlp_ms + p.ple_residual_ms + p.final_norm_ms +
        p.lm_head_ms + p.softcap_ms;
    const float other = p.total_gpu_ms - named;
    std::cerr
        << "[pie-gemma4-forward-profile]"
        << " seq=" << p.seq
        << " N=" << p.N
        << " R=" << p.R
        << " layers=" << p.layers
        << " lm_head_rows=" << p.lm_head_rows
        << " pure_decode=" << (p.pure_decode ? 1 : 0)
        << " decode_path=" << (p.decode_path ? 1 : 0)
        << " row_decode=" << (p.row_decode_path ? 1 : 0)
        << " compact_logits=" << (p.compact_logits ? 1 : 0)
        << " logits_argmax_only=" << (p.logits_argmax_only ? 1 : 0)
        << " total_gpu_ms=" << p.total_gpu_ms
        << " embed_ms=" << p.embed_ms
        << " ple_inputs_ms=" << p.ple_inputs_ms
        << " attn_prep_ms=" << p.attn_prep_ms
        << " attention_ms=" << p.attention_ms
        << " attn_out_ms=" << p.attn_out_ms
        << " mlp_ms=" << p.mlp_ms
        << " ple_residual_ms=" << p.ple_residual_ms
        << " final_norm_ms=" << p.final_norm_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " softcap_ms=" << p.softcap_ms
        << " other_gpu_ms=" << other
        << "\n";
}

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gemma4: missing weight '" + name + "'");
    }
    return e.get(name);
}

float read_bf16_scalar_once(const DeviceTensor& t) {
    if (t.empty()) return 1.f;
    std::uint16_t bits = 0;
    CUDA_CHECK(cudaMemcpy(&bits, t.data(), sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    const std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

// HF Gemma-4 prefixes language-model tensors with `model.language_model.`
// — different from Llama / Qwen which use `model.`. The bind helpers
// below take the prefix as a string so the call sites read like the
// other binders.
constexpr const char* kPrefix = "model.language_model.";

bool gemma4_row_decode_verify_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_SPEC_ROW_DECODE");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

int gemma4_row_decode_qmax() {
    static const int qmax = [] {
        const char* v = std::getenv("PIE_GEMMA4_SPEC_ROW_DECODE_QMAX");
        // Gemma4 native MTP uses three draft tokens in vLLM, so the
        // verifier's common block is the sampled token plus three drafts.
        // Keep that q_len=4 case on the decode-style verifier by default;
        // the full causal verifier is substantially slower and follows a
        // different multi-token numerical path anyway.
        if (v == nullptr || v[0] == '\0') return 4;
        return std::max(1, std::atoi(v));
    }();
    return qmax;
}

std::size_t gemma4_row_decode_max_page_refs() {
    static const std::size_t cap = [] {
        const char* v = std::getenv("PIE_GEMMA4_SPEC_ROW_DECODE_MAX_PAGE_REFS");
        if (v == nullptr || v[0] == '\0') return static_cast<std::size_t>(1 << 20);
        const long long parsed = std::atoll(v);
        return parsed > 0 ? static_cast<std::size_t>(parsed) : static_cast<std::size_t>(0);
    }();
    return cap;
}

bool gemma4_dense_gate_up_batched_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_DENSE_GATE_UP_BATCHED");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool gemma4_dense_gate_up_fused_enabled(const HfConfig& cfg) {
    const char* v = std::getenv("PIE_GEMMA4_DENSE_GATE_UP_FUSED");
    if (v != nullptr && v[0] != '\0') return v[0] != '0';

    return !cfg.gemma4_enable_moe &&
           cfg.hidden_size == 2560 &&
           cfg.intermediate_size == 10240 &&
           cfg.num_hidden_layers == 42 &&
           cfg.num_attention_heads == 8 &&
           cfg.num_key_value_heads == 2 &&
           cfg.head_dim == 256;
}

bool gemma4_dense_gate_up_fused_for_row_decode(const HfConfig& cfg) {
    const char* v = std::getenv("PIE_GEMMA4_DENSE_GATE_UP_FUSED");
    if (v != nullptr && v[0] != '\0') return v[0] != '0';
    return gemma4_dense_gate_up_fused_enabled(cfg);
}

bool gemma4_dense_qkv_fused_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_DENSE_QKV_FUSED");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

bool gemma4_plan_debug_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_PLAN_DEBUG");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

int gemma4_plan_debug_limit() {
    static const int limit = [] {
        const char* v = std::getenv("PIE_GEMMA4_PLAN_DEBUG_LIMIT");
        if (v == nullptr || v[0] == '\0') return 32;
        return std::max(0, std::atoi(v));
    }();
    return limit;
}

DeviceTensor make_gate_up_fused_weight(const DeviceTensor& gate,
                                       const DeviceTensor& up)
{
    if (gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 ||
        gate.shape().size() != 2 || up.shape().size() != 2 ||
        gate.shape()[0] != up.shape()[0] ||
        gate.shape()[1] != up.shape()[1]) {
        throw std::runtime_error(
            "gemma4: cannot fuse gate/up projections with mismatched shapes");
    }
    const std::int64_t I = gate.shape()[0];
    const std::int64_t H = gate.shape()[1];
    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {2 * I, H});
    const std::size_t bytes =
        static_cast<std::size_t>(I) * static_cast<std::size_t>(H) *
        sizeof(std::uint16_t);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(dst, gate.data(), bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + bytes, up.data(), bytes, cudaMemcpyDeviceToDevice));
    return fused;
}

DeviceTensor make_qkv_fused_weight(const DeviceTensor& q,
                                   const DeviceTensor& k,
                                   const DeviceTensor& v)
{
    if (q.dtype() != DType::BF16 || k.dtype() != DType::BF16 ||
        v.dtype() != DType::BF16 || q.shape().size() != 2 ||
        k.shape().size() != 2 || v.shape().size() != 2 ||
        q.shape()[1] != k.shape()[1] || q.shape()[1] != v.shape()[1] ||
        k.shape()[0] != v.shape()[0]) {
        throw std::runtime_error(
            "gemma4: cannot fuse q/k/v projections with mismatched shapes");
    }
    const std::int64_t Hq = q.shape()[0];
    const std::int64_t Hk = k.shape()[0];
    const std::int64_t H = q.shape()[1];
    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {Hq + 2 * Hk, H});
    const std::size_t q_bytes =
        static_cast<std::size_t>(Hq) * static_cast<std::size_t>(H) *
        sizeof(std::uint16_t);
    const std::size_t kv_bytes =
        static_cast<std::size_t>(Hk) * static_cast<std::size_t>(H) *
        sizeof(std::uint16_t);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(dst, q.data(), q_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + q_bytes, k.data(), kv_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + q_bytes + kv_bytes, v.data(), kv_bytes,
                          cudaMemcpyDeviceToDevice));
    return fused;
}

bool prepare_row_decode_kv_table(
    Gemma4MoeMlpWorkspace& moe_ws,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int N,
    int R,
    int page_size)
{
    const bool debug = [] {
        const char* dbg = std::getenv("PIE_GEMMA4_SPEC_ROW_DECODE_DEBUG");
        return dbg != nullptr && dbg[0] != '\0' && dbg[0] != '0';
    }();
    const auto fail = [&](const char* reason, int r = -1,
                          std::uint32_t q_len = 0,
                          std::uint32_t value0 = 0,
                          std::uint32_t value1 = 0) {
        if (debug) {
            static std::atomic<int> count{0};
            const int idx = count.fetch_add(1, std::memory_order_relaxed);
            if (idx < 16) {
                std::cerr << "[pie-driver-cuda] gemma4 row-decode skip"
                          << " reason=" << reason
                          << " r=" << r
                          << " q_len=" << q_len
                          << " v0=" << value0
                          << " v1=" << value1
                          << " N=" << N
                          << " R=" << R
                          << "\n";
            }
        }
        return false;
    };
    if (!gemma4_row_decode_verify_enabled() ||
        qo_indptr_h == nullptr ||
        kv_page_indices_h == nullptr ||
        kv_page_indptr_h == nullptr ||
        kv_last_page_lens_h == nullptr ||
        N <= 0 || R <= 0 || page_size <= 0) {
        return fail("disabled-or-bad-input");
    }

    const int qmax = gemma4_row_decode_qmax();
    const std::size_t max_page_refs = gemma4_row_decode_max_page_refs();
    if (max_page_refs == 0) return fail("zero-page-ref-cap");
    bool saw_multi = false;
    moe_ws.h_row_decode_kv_page_indices.clear();
    moe_ws.h_row_decode_kv_page_indptr.clear();
    moe_ws.h_row_decode_kv_last_page_lens.clear();
    moe_ws.h_row_decode_kv_page_indices.reserve(max_page_refs);
    moe_ws.h_row_decode_kv_page_indptr.reserve(static_cast<std::size_t>(N) + 1);
    moe_ws.h_row_decode_kv_last_page_lens.reserve(static_cast<std::size_t>(N));
    moe_ws.h_row_decode_kv_page_indptr.push_back(0);

    for (int r = 0; r < R; ++r) {
        const std::uint32_t q_lo = qo_indptr_h[r];
        const std::uint32_t q_hi = qo_indptr_h[r + 1];
        if (q_hi < q_lo || q_hi > static_cast<std::uint32_t>(N)) {
            return fail("bad-qo-indptr", r, 0, q_lo, q_hi);
        }
        const std::uint32_t q_len = q_hi - q_lo;
        if (q_len == 0 || q_len > static_cast<std::uint32_t>(qmax)) {
            return fail("q-len", r, q_len, static_cast<std::uint32_t>(qmax));
        }
        if (q_len > 1) saw_multi = true;

        const std::uint32_t page_lo = kv_page_indptr_h[r];
        const std::uint32_t page_hi = kv_page_indptr_h[r + 1];
        if (page_hi <= page_lo) return fail("bad-page-indptr", r, q_len, page_lo, page_hi);
        const std::uint32_t final_last = kv_last_page_lens_h[r];
        if (final_last == 0 || final_last > static_cast<std::uint32_t>(page_size)) {
            return fail("bad-last-page-len", r, q_len, final_last,
                        static_cast<std::uint32_t>(page_size));
        }
        const std::uint32_t final_len =
            (page_hi - page_lo - 1u) * static_cast<std::uint32_t>(page_size) +
            final_last;
        if (final_len < q_len) return false;

        for (std::uint32_t j = 0; j < q_len; ++j) {
            const std::uint32_t future_rows = q_len - 1u - j;
            const std::uint32_t prefix_len = final_len - future_rows;
            const std::uint32_t needed_pages =
                (prefix_len + static_cast<std::uint32_t>(page_size) - 1u) /
                static_cast<std::uint32_t>(page_size);
            if (needed_pages == 0 || page_lo + needed_pages > page_hi) {
                return fail("needed-pages", r, q_len, needed_pages, page_hi - page_lo);
            }
            const std::size_t dst =
                moe_ws.h_row_decode_kv_page_indices.size();
            if (dst + needed_pages > max_page_refs) {
                return fail("page-ref-cap", r, q_len,
                            static_cast<std::uint32_t>(dst + needed_pages),
                            static_cast<std::uint32_t>(
                                std::min<std::size_t>(
                                    max_page_refs,
                                    std::numeric_limits<std::uint32_t>::max())));
            }
            moe_ws.h_row_decode_kv_page_indices.resize(dst + needed_pages);
            std::copy(
                kv_page_indices_h + page_lo,
                kv_page_indices_h + page_lo + needed_pages,
                moe_ws.h_row_decode_kv_page_indices.data() + dst);
            moe_ws.h_row_decode_kv_page_indptr.push_back(
                static_cast<std::uint32_t>(
                    moe_ws.h_row_decode_kv_page_indices.size()));
            std::uint32_t last =
                prefix_len % static_cast<std::uint32_t>(page_size);
            if (last == 0) last = static_cast<std::uint32_t>(page_size);
            moe_ws.h_row_decode_kv_last_page_lens.push_back(last);
        }
    }

    if (!saw_multi ||
        moe_ws.h_row_decode_kv_page_indptr.size() !=
            static_cast<std::size_t>(N + 1) ||
        moe_ws.h_row_decode_kv_last_page_lens.size() !=
            static_cast<std::size_t>(N)) {
        return fail("final-shape", -1, 0,
                    static_cast<std::uint32_t>(
                        std::min<std::size_t>(
                            moe_ws.h_row_decode_kv_page_indptr.size(),
                            std::numeric_limits<std::uint32_t>::max())),
                    static_cast<std::uint32_t>(
                        std::min<std::size_t>(
                            moe_ws.h_row_decode_kv_last_page_lens.size(),
                            std::numeric_limits<std::uint32_t>::max())));
    }

    if (moe_ws.row_decode_kv_page_indices.size() <
            moe_ws.h_row_decode_kv_page_indices.size() ||
        moe_ws.row_decode_kv_page_indptr.size() <
            moe_ws.h_row_decode_kv_page_indptr.size() ||
        moe_ws.row_decode_kv_last_page_lens.size() <
            moe_ws.h_row_decode_kv_last_page_lens.size()) {
        return fail("device-cap", -1, 0,
                    static_cast<std::uint32_t>(
                        std::min<std::size_t>(
                            moe_ws.h_row_decode_kv_page_indices.size(),
                            std::numeric_limits<std::uint32_t>::max())),
                    static_cast<std::uint32_t>(
                        std::min<std::size_t>(
                            moe_ws.h_row_decode_kv_page_indptr.size(),
                            std::numeric_limits<std::uint32_t>::max())));
    }
    moe_ws.row_decode_kv_page_indices.copy_from_host(
        std::span<const std::uint32_t>(moe_ws.h_row_decode_kv_page_indices));
    moe_ws.row_decode_kv_page_indptr.copy_from_host(
        std::span<const std::uint32_t>(moe_ws.h_row_decode_kv_page_indptr));
    moe_ws.row_decode_kv_last_page_lens.copy_from_host(
        std::span<const std::uint32_t>(moe_ws.h_row_decode_kv_last_page_lens));
    if (const char* dbg = std::getenv("PIE_GEMMA4_SPEC_ROW_DECODE_DEBUG");
        dbg != nullptr && dbg[0] != '\0' && dbg[0] != '0') {
        static std::atomic<int> count{0};
        const int idx = count.fetch_add(1, std::memory_order_relaxed);
        if (idx < 8) {
            std::cerr << "[pie-driver-cuda] gemma4 row-decode verify"
                      << " rows=" << N
                      << " page_refs="
                      << moe_ws.h_row_decode_kv_page_indices.size()
                      << " qmax=" << qmax << "\n";
        }
    }
    return true;
}

}  // namespace

void set_gemma4_logits_argmax_only(bool enabled) {
    g_logits_argmax_only = enabled;
}

void set_gemma4_fused_argmax_output(std::int32_t* ptr) {
    g_fused_argmax_output = ptr;
    g_fused_argmax_done = false;
}

bool gemma4_fused_argmax_done() {
    return g_fused_argmax_done;
}

namespace {

const Gemma4LayerWeights* first_layer_for_plan(
    const Gemma4Weights& w,
    bool full)
{
    for (const auto& layer : w.layers) {
        if (layer.is_full == full) return &layer;
    }
    return nullptr;
}

void prepare_gemma4_plan_for_layer(
    ops::DecodePlanCachePtr& plan,
    const Gemma4LayerWeights& layer,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    AttentionWorkspace& attn_ws,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int page_size,
    bool hnd_layout,
    cudaStream_t stream)
{
    if (!plan) plan = ops::make_decode_plan();
    const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int num_q_heads_local = cfg.num_attention_heads / T;
    const int num_kv_heads_local = layer.num_kv_heads / T;
    ops::plan_attention_flashinfer_decode(
        *plan, kv_page_indptr_h, num_requests,
        num_q_heads_local, num_kv_heads_local, layer.head_dim,
        page_size, attn_ws, stream,
        /*enable_cuda_graph=*/true,
        /*full_attention_variant=*/layer.is_full,
        hnd_layout);
}

ops::DecodePlanCachePtr& select_prepared_plan(
    Gemma4MoeMlpWorkspace& moe_ws,
    bool row_decode,
    bool full)
{
    if (row_decode) {
        return full ? moe_ws.row_decode_plan_full
                    : moe_ws.row_decode_plan_sliding;
    }
    return full ? moe_ws.decode_plan_full : moe_ws.decode_plan_sliding;
}

const ops::DecodePlanCachePtr& select_prepared_plan_const(
    const Gemma4MoeMlpWorkspace& moe_ws,
    bool row_decode,
    bool full)
{
    if (row_decode) {
        return full ? moe_ws.row_decode_plan_full
                    : moe_ws.row_decode_plan_sliding;
    }
    return full ? moe_ws.decode_plan_full : moe_ws.decode_plan_sliding;
}

}  // namespace

void prepare_gemma4_decode_plans(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int N,
    int R,
    bool is_pure_decode)
{
    static std::atomic<int> debug_count{0};
    const bool debug = gemma4_plan_debug_enabled();
    const int debug_idx = debug
        ? debug_count.fetch_add(1, std::memory_order_relaxed)
        : -1;
    const bool log_debug = debug && debug_idx < gemma4_plan_debug_limit();
    moe_ws.decode_plans_prepared = false;
    moe_ws.row_decode_prepared = false;
    moe_ws.row_decode_prepared_tokens = 0;
    moe_ws.row_decode_prepared_requests = 0;
    if (fwd_cfg.force_prefill_path || N <= 0 || R <= 0 ||
        kv_page_indptr_h == nullptr) {
        if (log_debug) {
            std::cerr << "[pie-driver-cuda] gemma4 plan prepare skip"
                      << " N=" << N
                      << " R=" << R
                      << " pure=" << (is_pure_decode ? 1 : 0)
                      << " force_prefill=" << (fwd_cfg.force_prefill_path ? 1 : 0)
                      << " has_kvpp=" << (kv_page_indptr_h != nullptr ? 1 : 0)
                      << "\n";
        }
        return;
    }

    const auto plan_layout = [](const ops::DecodePlanCachePtr& plan) {
        return plan ? ops::decode_plan_graph_layout(*plan) : 0u;
    };
    const auto prepare_pair = [&](const std::uint32_t* indptr,
                                  int requests,
                                  bool row_decode) {
        if (const auto* sliding = first_layer_for_plan(w, /*full=*/false)) {
            prepare_gemma4_plan_for_layer(
                select_prepared_plan(moe_ws, row_decode, /*full=*/false),
                *sliding, cfg, fwd_cfg, attn_ws, indptr, requests,
                cache.page_size(), cache.hnd_layout(), /*stream=*/nullptr);
        }
        if (const auto* full = first_layer_for_plan(w, /*full=*/true)) {
            prepare_gemma4_plan_for_layer(
                select_prepared_plan(moe_ws, row_decode, /*full=*/true),
                *full, cfg, fwd_cfg, attn_ws, indptr, requests,
                cache.page_size(), cache.hnd_layout(), /*stream=*/nullptr);
        }
    };

    if (is_pure_decode) {
        prepare_pair(kv_page_indptr_h, R, /*row_decode=*/false);
        moe_ws.decode_plans_prepared = true;
        if (log_debug) {
            std::cerr << "[pie-driver-cuda] gemma4 plan prepare"
                      << " N=" << N
                      << " R=" << R
                      << " pure=1"
                      << " decode_prepared=1"
                      << " row_prepared=0"
                      << " sliding_layout="
                      << plan_layout(moe_ws.decode_plan_sliding)
                      << " full_layout=" << plan_layout(moe_ws.decode_plan_full)
                      << " pages=" << kv_page_indptr_h[R]
                      << "\n";
        }
        return;
    }

    const bool row_ready = prepare_row_decode_kv_table(
            moe_ws, qo_indptr_h, kv_page_indices_h, kv_page_indptr_h,
            kv_last_page_lens_h, N, R, cache.page_size());
    if (row_ready) {
        prepare_pair(
            moe_ws.h_row_decode_kv_page_indptr.data(), N,
            /*row_decode=*/true);
        moe_ws.row_decode_prepared = true;
        moe_ws.row_decode_prepared_tokens = N;
        moe_ws.row_decode_prepared_requests = R;
    }
    if (log_debug) {
        std::cerr << "[pie-driver-cuda] gemma4 plan prepare"
                  << " N=" << N
                  << " R=" << R
                  << " pure=0"
                  << " row_ready=" << (row_ready ? 1 : 0)
                  << " decode_prepared=" << (moe_ws.decode_plans_prepared ? 1 : 0)
                  << " row_prepared=" << (moe_ws.row_decode_prepared ? 1 : 0)
                  << " row_tokens=" << moe_ws.row_decode_prepared_tokens
                  << " row_requests=" << moe_ws.row_decode_prepared_requests
                  << " sliding_layout="
                  << plan_layout(moe_ws.row_decode_plan_sliding)
                  << " full_layout=" << plan_layout(moe_ws.row_decode_plan_full)
                  << " row_pages="
                  << (moe_ws.h_row_decode_kv_page_indptr.empty()
                          ? 0u
                          : moe_ws.h_row_decode_kv_page_indptr.back())
                  << "\n";
    }
}

Gemma4MoeMlpWorkspace Gemma4MoeMlpWorkspace::allocate(
    int max_tokens, int hidden, int num_experts, int top_k,
    int moe_intermediate)
{
    Gemma4MoeMlpWorkspace ws;
    const std::size_t N    = static_cast<std::size_t>(max_tokens);
    const std::size_t maxR = N * top_k;
    const std::size_t H    = static_cast<std::size_t>(hidden);
    const std::size_t I    = static_cast<std::size_t>(moe_intermediate);

    ws.router_x      = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.router_logits = DeviceBuffer<std::uint16_t>::alloc(N * num_experts);
    ws.topk_idx      = DeviceBuffer<std::int32_t>::alloc(N * top_k);
    ws.topk_weights  = DeviceBuffer<float>::alloc(N * top_k);

    ws.moe_input    = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.expert_in    = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_gate_up = DeviceBuffer<std::uint16_t>::alloc(maxR * 2 * I);
    ws.expert_act   = DeviceBuffer<std::uint16_t>::alloc(maxR * I);
    ws.expert_out   = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_idx   = DeviceBuffer<std::int32_t>::alloc(maxR);
    ws.expert_w     = DeviceBuffer<float>::alloc(maxR);
    ws.moe_out      = DeviceBuffer<std::uint16_t>::alloc(N * H);

    ws.allocate_row_decode(max_tokens);

    ws.a_gu_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_gu_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_gu_ptrs    = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.a_dn_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_dn_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_dn_ptrs    = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.batch_weights = DeviceBuffer<float>::alloc(top_k);
    return ws;
}

void Gemma4MoeMlpWorkspace::allocate_row_decode(int max_tokens)
{
    if (max_tokens <= 0) return;
    // CUDA graphs capture the row-decode KV table device pointers. Allocate
    // these at workspace lifetime so growing speculative verification prefixes
    // cannot invalidate a captured graph.
    const std::size_t N = static_cast<std::size_t>(max_tokens);
    const std::size_t row_page_refs = gemma4_row_decode_max_page_refs();
    if (row_page_refs == 0) return;
    if (row_decode_kv_page_indices.size() < row_page_refs) {
        row_decode_kv_page_indices =
            DeviceBuffer<std::uint32_t>::alloc(row_page_refs);
    }
    if (row_decode_kv_page_indptr.size() < N + 1) {
        row_decode_kv_page_indptr =
            DeviceBuffer<std::uint32_t>::alloc(N + 1);
    }
    if (row_decode_kv_last_page_lens.size() < N) {
        row_decode_kv_last_page_lens =
            DeviceBuffer<std::uint32_t>::alloc(N);
    }
    // Dense Gemma4 uses the MoE pointer-array slots as tiny graph-stable
    // scratch for batching the gate/up MLP GEMMs. MoE variants allocate
    // larger arrays in allocate(); keep those when present.
    if (a_gu_ptrs.size() < 2) {
        a_gu_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(2);
    }
    if (b_gu_ptrs.size() < 2) {
        b_gu_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(2);
    }
    if (c_gu_ptrs.size() < 2) {
        c_gu_ptrs = DeviceBuffer<std::uint16_t*>::alloc(2);
    }
}

void Gemma4MoeMlpWorkspace::allocate_ple(int max_tokens, int per_layer_total)
{
    if (max_tokens <= 0 || per_layer_total <= 0) return;
    const std::size_t elems =
        static_cast<std::size_t>(max_tokens) *
        static_cast<std::size_t>(per_layer_total);
    ple_token = DeviceBuffer<std::uint16_t>::alloc(elems);
    ple_proj  = DeviceBuffer<std::uint16_t>::alloc(elems);
}

Gemma4Weights bind_gemma4(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.layer_types.empty()) {
        throw std::runtime_error(
            "gemma4: HfConfig.layer_types is empty — Gemma-4 requires "
            "the per-layer attention type from the HF config.");
    }
    const int L = cfg.num_hidden_layers;
    if (static_cast<int>(cfg.layer_types.size()) != L) {
        throw std::runtime_error("gemma4: layer_types size != num_hidden_layers");
    }

    Gemma4Weights w;
    // Reserve up front: pointers into `owned_router_combined_scales`
    // get cached on each `Gemma4LayerWeights`, so any vector reallocation
    // would dangle them. One slot per layer when MoE is on.
    if (cfg.gemma4_enable_moe) {
        w.owned_router_combined_scales.reserve(
            static_cast<std::size_t>(cfg.num_hidden_layers));
    }
    const bool fuse_dense_gate_up = gemma4_dense_gate_up_fused_enabled(cfg);
    const bool fuse_dense_qkv =
        !cfg.gemma4_enable_moe && gemma4_dense_qkv_fused_enabled();
    if (fuse_dense_gate_up) {
        w.owned_gate_up_fused.reserve(
            static_cast<std::size_t>(cfg.num_hidden_layers));
    }
    if (fuse_dense_qkv) {
        w.owned_qkv_fused.reserve(
            static_cast<std::size_t>(cfg.num_hidden_layers));
    }
    const std::string p = kPrefix;
    w.embed           = &must(engine, p + "embed_tokens.weight");
    // PLE (Per-Layer Embeddings) machinery is optional — Gemma-4 E2B /
    // E4B / 31B all ship it (`hidden_size_per_layer_input > 0`) but the
    // 26B-A4B MoE variant disables it (`hidden_size_per_layer_input ==
    // 0`). Skip the PLE tensors when the config says they're inert.
    if (cfg.gemma_hidden_size_per_layer_input > 0) {
        w.embed_per_layer = &must(engine, p + "embed_tokens_per_layer.weight");
        w.ple_model_proj  = &must(engine, p + "per_layer_model_projection.weight");
        w.ple_model_norm  = &must(engine, p + "per_layer_projection_norm.weight");
    }
    w.final_norm      = &must(engine, p + "norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "gemma4: lm_head missing and tie_word_embeddings=false");
    }

    // Per-layer dimensions. HF stores `head_dim` (sliding) and
    // `global_head_dim` (full); we read both from the config but our
    // HfConfig only carries the single `head_dim`. Recompute from
    // first-layer Q-proj shape: full-attention layers have
    // `q_proj = [num_q*global_head_dim, hidden]`, sliding have
    // `[num_q*head_dim, hidden]`.
    const int sliding_head_dim = cfg.head_dim;
    int global_head_dim = cfg.head_dim;
    {
        // First full-attention layer's q_proj reveals global_head_dim.
        for (int i = 0; i < L; ++i) {
            if (cfg.layer_types[i] == "full_attention") {
                const std::string q_name = p + "layers." + std::to_string(i) +
                                           ".self_attn.q_proj.weight";
                if (engine.has(q_name)) {
                    const auto& qt = engine.get(q_name);
                    const auto& s = qt.shape();
                    if (!s.empty()) {
                        global_head_dim = static_cast<int>(s[0]) /
                                          cfg.num_attention_heads;
                    }
                }
                break;
            }
        }
    }

    // Determine which layers are KV-shared. HF: last
    // `num_kv_shared_layers` layers reuse from earlier; given E2B has
    // 35 layers, 20 shared, the boundary is index 14. The `kv_source`
    // for a shared layer is the most recent non-shared layer of the
    // *same* attention type.
    const int num_kv_shared = std::max(0, [&]{
        // num_kv_shared_layers isn't carried in HfConfig today; we
        // derive it from the layer_types vector and the implicit
        // "last N reuse from earlier" rule. Fall back to 0 (every
        // layer computes its own K/V) when the field is missing.
        return engine.hf_config().num_kv_shared_layers;
    }());
    const int first_shared = L - num_kv_shared;  // first shared layer index

    w.layers.resize(static_cast<std::size_t>(L));
    w.per_layer_head_dim.resize(L);
    w.per_layer_intermediate.resize(L);
    w.per_layer_num_kv_heads.resize(L);
    w.kv_source_layer.resize(L);
    w.per_layer_window_left.resize(L);
    w.per_layer_rope_theta.resize(L);
    w.per_layer_partial_rotary_factor.resize(L, 1.0f);

    for (int i = 0; i < L; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        auto& Lw = w.layers[i];
        const bool is_full = (cfg.layer_types[i] == "full_attention");
        const bool is_shared = (i >= first_shared);

        Lw.is_full   = is_full;
        Lw.is_shared = is_shared;
        Lw.head_dim  = is_full ? global_head_dim : sliding_head_dim;
        w.per_layer_head_dim[i] = Lw.head_dim;
        // 26B-A4B's "k_eq_v" mode flips full-attention layers onto a
        // narrower KV head count (`num_global_key_value_heads`) and
        // skips `v_proj.weight` entirely — V is taken from the raw
        // k_proj output, before k_norm/RoPE, then v-norm. Sliding
        // layers stay on the standard `num_key_value_heads` and have
        // their own v_proj.
        Lw.use_k_as_v = cfg.gemma4_attention_k_eq_v && is_full;
        Lw.num_kv_heads = Lw.use_k_as_v
                              ? cfg.gemma4_num_global_key_value_heads
                              : cfg.num_key_value_heads;
        w.per_layer_num_kv_heads[i] = Lw.num_kv_heads;

        // KV source: same layer when not shared; most recent non-shared
        // layer of the same type when shared.
        if (!is_shared) {
            Lw.kv_source = i;
        } else {
            int src = -1;
            for (int j = first_shared - 1; j >= 0; --j) {
                if (cfg.layer_types[j] == cfg.layer_types[i]) { src = j; break; }
            }
            if (src < 0) {
                throw std::runtime_error(
                    "gemma4: no source layer found for shared layer " +
                    std::to_string(i));
            }
            Lw.kv_source = src;
        }
        w.kv_source_layer[i] = Lw.kv_source;

        // Per-layer window_left: sliding layers limit context to the
        // configured `sliding_window`; full layers run unbounded.
        w.per_layer_window_left[i] = is_full ? -1 : cfg.sliding_window;

        // Per-layer rope_theta + partial_rotary_factor. Gemma-4 nests
        // these under `rope_parameters[layer_type]` in HF; we expand
        // into vectors at parse time.
        if (i < static_cast<int>(cfg.gemma_per_layer_rope_theta.size())) {
            w.per_layer_rope_theta[i]            = cfg.gemma_per_layer_rope_theta[i];
            w.per_layer_partial_rotary_factor[i] =
                cfg.gemma_per_layer_partial_rotary_factor[i];
        } else {
            w.per_layer_rope_theta[i] =
                (is_full || cfg.rope_local_base_freq <= 0.f)
                    ? cfg.rope_theta
                    : cfg.rope_local_base_freq;
        }

        // Norms (4 per layer).
        Lw.attn_norm_pre  = &must(engine, lp + "input_layernorm.weight");
        Lw.attn_norm_post = &must(engine, lp + "post_attention_layernorm.weight");
        Lw.mlp_norm_pre   = &must(engine, lp + "pre_feedforward_layernorm.weight");
        Lw.mlp_norm_post  = &must(engine, lp + "post_feedforward_layernorm.weight");

        // Q is always present.
        Lw.q_proj = &must(engine, lp + "self_attn.q_proj.weight");
        Lw.q_norm = &must(engine, lp + "self_attn.q_norm.weight");

        // Even on shared layers HF keeps k_proj/v_proj/k_norm in the
        // file (redundant). Bind them when present so the schema
        // tolerates either dump style; the forward only consults them
        // when `is_shared == false`. On 26B-A4B's `use_k_as_v` full
        // layers v_proj is genuinely absent (V is derived from raw
        // k_proj), so a missing v_proj is expected there.
        if (engine.has(lp + "self_attn.k_proj.weight")) {
            Lw.k_proj = &engine.get(lp + "self_attn.k_proj.weight");
        }
        if (engine.has(lp + "self_attn.v_proj.weight")) {
            Lw.v_proj = &engine.get(lp + "self_attn.v_proj.weight");
        }
        if (engine.has(lp + "self_attn.k_norm.weight")) {
            Lw.k_norm = &engine.get(lp + "self_attn.k_norm.weight");
        }
        Lw.o_proj = &must(engine, lp + "self_attn.o_proj.weight");
        if (fuse_dense_qkv && !is_shared && !Lw.use_k_as_v &&
            Lw.k_proj != nullptr && Lw.v_proj != nullptr) {
            w.owned_qkv_fused.push_back(
                make_qkv_fused_weight(*Lw.q_proj, *Lw.k_proj, *Lw.v_proj));
            Lw.qkv_proj_fused = &w.owned_qkv_fused.back();
        }

        // MLP (intermediate may be 2× when use_double_wide_mlp + shared).
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj   = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.intermediate = static_cast<int>(Lw.gate_proj->shape()[0]);
        w.per_layer_intermediate[i] = Lw.intermediate;
        if (fuse_dense_gate_up) {
            w.owned_gate_up_fused.push_back(
                make_gate_up_fused_weight(*Lw.gate_proj, *Lw.up_proj));
            Lw.gate_up_proj_fused = &w.owned_gate_up_fused.back();
        }

        // PLE per-layer triple. HF names match `per_layer_input_gate`,
        // `per_layer_projection`, `post_per_layer_input_norm`. Optional
        // — see top of `bind_gemma4`.
        if (cfg.gemma_hidden_size_per_layer_input > 0) {
            Lw.ple_input_gate = &must(engine, lp + "per_layer_input_gate.weight");
            Lw.ple_projection = &must(engine, lp + "per_layer_projection.weight");
            Lw.ple_norm       = &must(engine, lp + "post_per_layer_input_norm.weight");
        }

        // Per-layer learnable scalar.
        Lw.layer_scalar = engine.has(lp + "layer_scalar")
                              ? &engine.get(lp + "layer_scalar")
                              : nullptr;
        Lw.layer_scalar_value = Lw.layer_scalar
                                  ? read_bf16_scalar_once(*Lw.layer_scalar)
                                  : 1.f;

        // ── Sparse-MoE block (Gemma-4 26B-A4B only) ───────────────────
        // The MoE variant runs in parallel with the dense MLP (HF
        // `Gemma4TextDecoderLayer.forward`). When `enable_moe_block` is
        // false the dense path runs alone and these pointers stay null.
        if (cfg.gemma4_enable_moe) {
            Lw.router_proj            = &must(engine, lp + "router.proj.weight");
            Lw.router_per_expert_scale = &must(engine, lp + "router.per_expert_scale");
            Lw.moe_gate_up_proj       = &must(engine, lp + "experts.gate_up_proj");
            Lw.moe_down_proj          = &must(engine, lp + "experts.down_proj");
            Lw.mlp_norm_post_dense    = &must(engine, lp + "post_feedforward_layernorm_1.weight");
            Lw.moe_norm_pre           = &must(engine, lp + "pre_feedforward_layernorm_2.weight");
            Lw.moe_norm_post          = &must(engine, lp + "post_feedforward_layernorm_2.weight");
            // The router pipeline does `(rmsnorm_no_scale(x) * scale) *
            // (1/sqrt(H))` then a linear. Bake `1/sqrt(H)` into the
            // per-channel `scale` here so the forward collapses the
            // first three steps into a single rmsnorm-with-weight call.
            const auto* raw_scale = &must(engine, lp + "router.scale");
            const std::int64_t H64 = raw_scale->numel();
            const float inv_sqrt_h = 1.f / std::sqrt(static_cast<float>(H64));
            std::vector<std::uint16_t> host(static_cast<std::size_t>(H64));
            CUDA_CHECK(cudaMemcpy(host.data(), raw_scale->data(),
                                  H64 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            for (auto& bits : host) {
                std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
                float v;
                std::memcpy(&v, &f32_bits, sizeof(float));
                v *= inv_sqrt_h;
                std::memcpy(&f32_bits, &v, sizeof(float));
                bits = static_cast<std::uint16_t>(f32_bits >> 16);
            }
            DeviceTensor combined = DeviceTensor::allocate(DType::BF16, {H64});
            CUDA_CHECK(cudaMemcpy(combined.data(), host.data(),
                                  H64 * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
            w.owned_router_combined_scales.push_back(std::move(combined));
            Lw.router_scale = &w.owned_router_combined_scales.back();
        }

        if (is_full) w.full_layer_indices.push_back(i);
    }

    return w;
}

// ── Gemma-4 vision tower bind ───────────────────────────────────────────────
// Mirrors `bind_gemma4` for the `model.vision_tower.` / `model.embed_vision.`
// tensors. The encoder forward (MULTIMODAL.md Phase 2.2) consumes the result;
// this only wires pointers, so it has no GPU cost beyond what the loader
// already staged. Requires the vision tensors to be present in the weight
// store (i.e. removed from `mm_skip_prefixes`) — otherwise `must` throws.
Gemma4VisionWeights bind_gemma4_vision(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (!cfg.gemma_vision.has_value()) {
        throw std::runtime_error(
            "gemma4 vision: HfConfig.gemma_vision is empty — config has no "
            "gemma4_vision vision_config");
    }
    const GemmaVisionConfig& vc = *cfg.gemma_vision;

    Gemma4VisionWeights w;
    w.config = vc;

    const std::string vp = "model.vision_tower.";
    w.patch_input_proj =
        &must(engine, vp + "patch_embedder.input_proj.weight");
    w.patch_position_embedding =
        &must(engine, vp + "patch_embedder.position_embedding_table");
    w.embed_vision_projection =
        &must(engine, "model.embed_vision.embedding_projection.weight");

    // Bind one clipped-linear: required weight + optional per-tensor clip
    // ranges (present only on quantized checkpoints, `use_clipped_linears`).
    const auto bind_clipped = [&](const std::string& base) {
        Gemma4ClippedLinear c;
        c.weight = &must(engine, base + ".linear.weight");
        const auto opt = [&](const char* suffix) -> const DeviceTensor* {
            const std::string n = base + suffix;
            return engine.has(n) ? &engine.get(n) : nullptr;
        };
        c.input_min  = opt(".input_min");
        c.input_max  = opt(".input_max");
        c.output_min = opt(".output_min");
        c.output_max = opt(".output_max");
        return c;
    };

    const int L = vc.num_hidden_layers;
    w.layers.resize(static_cast<std::size_t>(L));
    const std::string ep = vp + "encoder.layers.";
    for (int i = 0; i < L; ++i) {
        const std::string lp = ep + std::to_string(i) + ".";
        auto& Lw = w.layers[static_cast<std::size_t>(i)];

        Lw.input_layernorm =
            &must(engine, lp + "input_layernorm.weight");
        Lw.post_attention_layernorm =
            &must(engine, lp + "post_attention_layernorm.weight");
        Lw.pre_feedforward_layernorm =
            &must(engine, lp + "pre_feedforward_layernorm.weight");
        Lw.post_feedforward_layernorm =
            &must(engine, lp + "post_feedforward_layernorm.weight");

        Lw.q_proj = bind_clipped(lp + "self_attn.q_proj");
        Lw.k_proj = bind_clipped(lp + "self_attn.k_proj");
        Lw.v_proj = bind_clipped(lp + "self_attn.v_proj");
        Lw.o_proj = bind_clipped(lp + "self_attn.o_proj");
        Lw.q_norm = &must(engine, lp + "self_attn.q_norm.weight");
        Lw.k_norm = &must(engine, lp + "self_attn.k_norm.weight");

        Lw.gate_proj = bind_clipped(lp + "mlp.gate_proj");
        Lw.up_proj   = bind_clipped(lp + "mlp.up_proj");
        Lw.down_proj = bind_clipped(lp + "mlp.down_proj");
    }

    return w;
}

// ── Gemma-4 audio tower bind ────────────────────────────────────────────────
// Mirrors `bind_gemma4_vision` for `model.audio_tower.` / `model.embed_audio.`.
// All tensor names verified against scripts/gemma4_audio_parity_ref.py (the
// `audio.`-prefixed dumps map to `model.audio_tower.` here). Throws on missing.
Gemma4AudioWeights bind_gemma4_audio(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (!cfg.gemma_audio.has_value()) {
        throw std::runtime_error(
            "gemma4 audio: HfConfig.gemma_audio is empty — config has no "
            "gemma4_audio audio_config");
    }
    const GemmaAudioConfig& ac = *cfg.gemma_audio;

    Gemma4AudioWeights w;
    w.config.hidden_size              = ac.hidden_size;
    w.config.num_attention_heads      = ac.num_attention_heads;
    w.config.num_hidden_layers        = ac.num_hidden_layers;
    w.config.conv_kernel_size         = ac.conv_kernel_size;
    w.config.subsampling_conv_channels0 = ac.subsampling_conv_channels0;
    w.config.subsampling_conv_channels1 = ac.subsampling_conv_channels1;
    w.config.output_proj_dims         = ac.output_proj_dims;
    w.config.attention_chunk_size     = ac.attention_chunk_size;
    w.config.attention_context_left   = ac.attention_context_left;
    w.config.attention_context_right  = ac.attention_context_right;
    w.config.feature_size             = ac.feature_size;
    w.config.attention_logit_cap      = ac.attention_logit_cap;
    w.config.residual_weight          = ac.residual_weight;
    w.config.rms_norm_eps             = ac.rms_norm_eps;

    const std::string ap = "model.audio_tower.";

    // SSCP subsampling conv stack.
    w.sscp_layer0_conv =
        &must(engine, ap + "subsample_conv_projection.layer0.conv.weight");
    w.sscp_layer0_norm =
        &must(engine, ap + "subsample_conv_projection.layer0.norm.weight");
    w.sscp_layer1_conv =
        &must(engine, ap + "subsample_conv_projection.layer1.conv.weight");
    w.sscp_layer1_norm =
        &must(engine, ap + "subsample_conv_projection.layer1.norm.weight");
    w.sscp_input_proj =
        &must(engine, ap + "subsample_conv_projection.input_proj_linear.weight");

    w.output_proj_weight = &must(engine, ap + "output_proj.weight");
    w.output_proj_bias   = &must(engine, ap + "output_proj.bias");
    w.embed_audio_projection =
        &must(engine, "model.embed_audio.embedding_projection.weight");

    // Bind one clipped-linear: required weight + optional per-tensor clip ranges.
    const auto bind_clipped = [&](const std::string& base) {
        Gemma4AudioClippedLinear c;
        c.weight = &must(engine, base + ".linear.weight");
        const auto opt = [&](const char* suffix) -> const DeviceTensor* {
            const std::string n = base + suffix;
            return engine.has(n) ? &engine.get(n) : nullptr;
        };
        c.input_min  = opt(".input_min");
        c.input_max  = opt(".input_max");
        c.output_min = opt(".output_min");
        c.output_max = opt(".output_max");
        return c;
    };
    const auto bind_ffn = [&](const std::string& base) {
        Gemma4AudioFfnWeights f;
        f.pre_layer_norm  = &must(engine, base + ".pre_layer_norm.weight");
        f.post_layer_norm = &must(engine, base + ".post_layer_norm.weight");
        f.ffw_layer_1 = bind_clipped(base + ".ffw_layer_1");
        f.ffw_layer_2 = bind_clipped(base + ".ffw_layer_2");
        return f;
    };

    const int L = ac.num_hidden_layers;
    w.layers.resize(static_cast<std::size_t>(L));
    const std::string lp_base = ap + "layers.";
    for (int i = 0; i < L; ++i) {
        const std::string lp = lp_base + std::to_string(i) + ".";
        auto& Lw = w.layers[static_cast<std::size_t>(i)];

        Lw.feed_forward1 = bind_ffn(lp + "feed_forward1");
        Lw.feed_forward2 = bind_ffn(lp + "feed_forward2");

        Lw.norm_pre_attn  = &must(engine, lp + "norm_pre_attn.weight");
        Lw.norm_post_attn = &must(engine, lp + "norm_post_attn.weight");
        Lw.norm_out       = &must(engine, lp + "norm_out.weight");

        Lw.q_proj = bind_clipped(lp + "self_attn.q_proj");
        Lw.k_proj = bind_clipped(lp + "self_attn.k_proj");
        Lw.v_proj = bind_clipped(lp + "self_attn.v_proj");
        Lw.post   = bind_clipped(lp + "self_attn.post");
        Lw.relative_k_proj = &must(engine, lp + "self_attn.relative_k_proj.weight");
        Lw.per_dim_scale   = &must(engine, lp + "self_attn.per_dim_scale");

        Lw.lconv_pre_layer_norm = &must(engine, lp + "lconv1d.pre_layer_norm.weight");
        Lw.lconv_conv_norm      = &must(engine, lp + "lconv1d.conv_norm.weight");
        Lw.lconv_linear_start = bind_clipped(lp + "lconv1d.linear_start");
        Lw.lconv_linear_end   = bind_clipped(lp + "lconv1d.linear_end");
        Lw.lconv_depthwise_conv =
            &must(engine, lp + "lconv1d.depthwise_conv1d.weight");
    }

    return w;
}

namespace {

// Parity-dump helper: write a bf16 tensor of `numel` elements to
// `<dir>/<tag>.bin` as raw bf16 bytes. We only record the *first* fire
// of a session (typically the prefill) — subsequent decode fires would
// overwrite the prefill's intermediates with decode-shaped tensors,
// which is not what the PyTorch parity harness compares against.
inline bool& dbg_first_fire_flag() {
    static bool first = true;
    return first;
}
// True only when `PIE_GEMMA4_DUMP_DIR` is set in the environment.
// Cached on first call so per-fire checks are a single bool load.
inline bool dbg_dumps_enabled() {
    static const bool enabled = std::getenv("PIE_GEMMA4_DUMP_DIR") != nullptr;
    return enabled;
}
inline void dbg_dump_bf16(const char* tag, const void* dev_ptr,
                          std::size_t numel) {
    if (!dbg_dumps_enabled()) return;
    if (!dbg_first_fire_flag()) return;
    static const char* dir = std::getenv("PIE_GEMMA4_DUMP_DIR");
    std::vector<std::uint16_t> tmp(numel);
    cudaMemcpy(tmp.data(), dev_ptr, numel * 2, cudaMemcpyDeviceToHost);
    std::string path = std::string(dir) + "/" + tag + ".bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    out.write(reinterpret_cast<const char*>(tmp.data()), numel * 2);
}
// Sync-then-dump: paired with `dbg_dump_bf16` to guarantee the kernel
// preceding the dump has finished. **Only syncs when dumping is on.**
// Replaces the previous pattern of unconditional `cudaDeviceSynchronize()
// + dbg_dump_bf16(...)` which stalled the GPU on every layer of every
// fire even with no dump directory configured (the dumps no-op'd, but
// the syncs did not — they were the dominant per-step overhead in
// Gemma-4 release builds).
inline void dbg_sync_dump_bf16(const char* tag, const void* dev_ptr,
                               std::size_t numel) {
    if (!dbg_dumps_enabled()) return;
    if (!dbg_first_fire_flag()) return;
    cudaDeviceSynchronize();
    dbg_dump_bf16(tag, dev_ptr, numel);
}

// Per-expert routing lists from device-side topk decisions. Mirrors
// `qwen3_5_moe_forward.cpp::build_routing` — kept local because the
// layer-weights schema differs.
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

// MoE block for Gemma-4 26B-A4B: parallel branch alongside the dense
// MLP. Computes `branch_2 = post_ff_norm_2(experts(pre_ff_norm_2(y),
// router(y)))` and writes it into `moe_ws.moe_out`.
void gemma4_moe_block(
    const Gemma4LayerWeights& Lw,
    const HfConfig& cfg,
    Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int H  = cfg.hidden_size;
    const int E  = cfg.num_experts;
    const int K  = cfg.num_experts_per_tok;
    const int Im = cfg.moe_intermediate_size;
    const float eps = cfg.rms_norm_eps;

    // ── Router ────────────────────────────────────────────────────
    // Step 1+2: rmsnorm-no-scale(y) * (router_scale * 1/sqrt(H)).
    // The combined scale was baked at bind time, so this collapses to
    // a single weighted-rmsnorm call.
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), Lw.router_scale->data(), moe_ws.router_x.data(),
        N, H, eps, stream);
    // Step 3: linear projection to expert logits.
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        moe_ws.router_x.data(), Lw.router_proj->data(),
        moe_ws.router_logits.data(), N, E, H);
    // Steps 4+5: softmax over E → top-K → renormalise.
    kernels::launch_topk_softmax_bf16(
        moe_ws.router_logits.data(),
        moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
        N, E, K, stream);
    // Step 6: per-expert scalar gain on the chosen weights.
    kernels::launch_apply_per_expert_scale_bf16(
        moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
        Lw.router_per_expert_scale->data(),
        N, K, stream);

    // ── MoE input ────────────────────────────────────────────────
    // pre_feedforward_layernorm_2(y) → moe_input. Note: HF flattens the
    // residual `y`, NOT the dense MLP's pre-norm (`ws.norm_x` was
    // already overwritten by the dense path).
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), Lw.moe_norm_pre->data(), moe_ws.moe_input.data(),
        N, H, eps, stream);

    // D2H sync the routing decisions for the prefill / multi-token
    // dispatch loop. The dense Gemma-4 forward is non-graph-capturable
    // (per-layer head_dim, attention_factor lookups all run host code),
    // so an extra sync here is in the noise.
    std::vector<std::int32_t> topk_idx_h((std::size_t)N * K);
    std::vector<float>        topk_w_h((std::size_t)N * K);
    CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), moe_ws.topk_idx.data(),
        topk_idx_h.size() * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), moe_ws.topk_weights.data(),
        topk_w_h.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Zero `moe_out` before scatter-add accumulation.
    CUDA_CHECK(cudaMemsetAsync(moe_ws.moe_out.data(), 0,
        (std::size_t)N * H * sizeof(std::uint16_t), stream));

    const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;

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
            static_cast<const std::uint16_t*>(moe_ws.moe_input.data()),
            moe_ws.expert_idx.data(),
            moe_ws.expert_in.data(),
            Ne, H, stream);

        const auto* gate_up_w = static_cast<const std::uint16_t*>(
                                    Lw.moe_gate_up_proj->data())
                                + e * expert_stride_gu;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            moe_ws.expert_in.data(), gate_up_w,
            moe_ws.expert_gate_up.data(), Ne, 2 * Im, H);

        kernels::launch_chunked_geglu_tanh_bf16(
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
            moe_ws.moe_out.data(), moe_ws.expert_out.data(),
            moe_ws.expert_idx.data(), moe_ws.expert_w.data(),
            Ne, H, stream);
    }
}

}  // namespace

std::uint32_t gemma4_decode_graph_layout(
    const Gemma4MoeMlpWorkspace& moe_ws)
{
    const bool row_decode = moe_ws.row_decode_prepared;
    const bool decode = moe_ws.decode_plans_prepared;
    if (!row_decode && !decode) return 0u;

    const auto pack = [](const ops::DecodePlanCachePtr& plan) -> std::uint32_t {
        if (!plan) return 0u;
        return ops::decode_plan_graph_layout(*plan);
    };
    const std::uint32_t sliding = row_decode
        ? pack(moe_ws.row_decode_plan_sliding)
        : pack(moe_ws.decode_plan_sliding);
    const std::uint32_t full = row_decode
        ? pack(moe_ws.row_decode_plan_full)
        : pack(moe_ws.decode_plan_full);
    if (sliding == 0u && full == 0u) return 0u;

    return (row_decode ? 0x10000u : 0x20000u) |
           (sliding & 0xffu) |
           ((full & 0xffu) << 8);
}

void gemma4_forward_paged(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* /*custom_mask_indptr_d*/,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const Gemma4VisionInputs* vision_in,
    const Gemma4AudioInputs* audio_in)
{
    const int H        = cfg.hidden_size;
    const int L        = cfg.num_hidden_layers;
    const int V        = cfg.vocab_size;
    const int ple_dim  = cfg.gemma_hidden_size_per_layer_input;
    const float eps    = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    Gemma4ForwardProfile profile;
    profile.begin(stream);

    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;
    const bool prepared_row_decode =
        !use_decode_path &&
        moe_ws.row_decode_prepared &&
        moe_ws.row_decode_prepared_tokens == N &&
        moe_ws.row_decode_prepared_requests == R;
    const bool use_row_decode_path =
        !use_decode_path &&
        custom_mask_d == nullptr &&
        !fwd_cfg.force_prefill_path &&
        (prepared_row_decode ||
         prepare_row_decode_kv_table(
             moe_ws, qo_indptr_h, kv_page_indices_h, kv_page_indptr_h,
             kv_last_page_lens_h, N, R, cache.page_size()));
    if (profile.enabled) {
        profile.N = N;
        profile.R = R;
        profile.pure_decode = is_pure_decode;
        profile.decode_path = use_decode_path;
        profile.row_decode_path = use_row_decode_path;
    }

    // ── 1. Embed + √hidden scale ──────────────────────────────────────────
    // Dump input tokens to disk so the parity harness can confirm
    // it's running HF on the same prefix. Only on the first fire
    // (the prefill) — subsequent decode fires would clobber the
    // file with a single-token dump.
    {
        const char* dir = std::getenv("PIE_GEMMA4_DUMP_DIR");
        if (dir != nullptr && dbg_first_fire_flag()) {
            std::vector<std::int32_t> tmp(N);
            cudaMemcpy(tmp.data(), token_ids, N * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);
            std::ofstream out(std::string(dir) + "/tokens.bin", std::ios::binary);
            if (out) out.write(reinterpret_cast<const char*>(tmp.data()),
                               N * sizeof(std::int32_t));
        }
    }
    profile_gemma4_cuda_stage(profile, profile.embed_ms, stream, [&] {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(), N, H, V, stream);
        dbg_sync_dump_bf16("embed_pre_scale", ws.y.data(),
                      static_cast<std::size_t>(N) * H);
        kernels::launch_scalar_mul_bf16(
            ws.y.data(), std::sqrt(static_cast<float>(H)),
            static_cast<std::size_t>(N) * H, stream);
    });
    dbg_sync_dump_bf16("embed_post_scale", ws.y.data(),
                  static_cast<std::size_t>(N) * H);

    // ── 1b. Multimodal: encode images and overwrite their soft-token rows ──
    // HF `masked_scatter`s image features into inputs_embeds AFTER the
    // embed-scale, so the projected rows replace the (scaled) placeholder
    // embeddings unscaled. No-op for text-only passes.
    if (vision_in != nullptr && vision_in->num_images > 0) {
        scatter_gemma4_vision(*vision_in, static_cast<__nv_bfloat16*>(ws.y.data()),
                              N, H, stream);
    }
    // Audio is the direct analog of vision: encode each clip's log-mel features
    // and overwrite its soft-token rows. No M-RoPE / DeepStack — just scatter.
    if (audio_in != nullptr && audio_in->num_clips > 0) {
        scatter_gemma4_audio(*audio_in, static_cast<__nv_bfloat16*>(ws.y.data()),
                             N, H, stream);
    }

    // ── 2. Per-layer inputs (PLE) ────────────────────────────────────────
    // Compute once per fire; sliced per layer below. Skipped entirely
    // when `ple_dim == 0` (Gemma-4 26B-A4B disables PLE; the per-layer
    // residual block at step 3c is also gated below).
    //
    //     per_layer_token = embed_per_layer[token_ids]              [N, L*ple_dim]
    //     per_layer_token *= sqrt(ple_dim)
    //     per_layer_proj   = inputs_embeds @ ple_model_proj.T       [N, L*ple_dim]
    //     per_layer_proj  *= 1/sqrt(hidden)
    //     per_layer_proj   = rms_norm(per_layer_proj, ple_model_norm)  [per ple_dim row]
    //     per_layer_inputs = (per_layer_proj + per_layer_token) * 1/sqrt(2)
    //
    // Allocate dedicated scratch for these — `ws.gate`/`ws.up` would
    // get clobbered by the per-layer MLP GEMM in step 3 before the
    // PLE block reads back the slice for layer N. Earlier versions
    // shared the buffers and silently produced wrong PLE residuals
    // for every token (most visibly token 0).
    const int per_layer_total = L * ple_dim;
    DeviceTensor per_layer_token_buf;
    DeviceTensor per_layer_proj_buf;
    void* per_layer_token = nullptr;
    void* per_layer_proj  = nullptr;
    if (ple_dim > 0) {
        const std::size_t ple_elems =
            static_cast<std::size_t>(N) *
            static_cast<std::size_t>(per_layer_total);
        if (moe_ws.ple_token.size() >= ple_elems &&
            moe_ws.ple_proj.size() >= ple_elems) {
            per_layer_token = moe_ws.ple_token.data();
            per_layer_proj  = moe_ws.ple_proj.data();
        } else {
            per_layer_token_buf = DeviceTensor::allocate(
                DType::BF16, {N, per_layer_total});
            per_layer_proj_buf = DeviceTensor::allocate(
                DType::BF16, {N, per_layer_total});
            per_layer_token = per_layer_token_buf.data();
            per_layer_proj  = per_layer_proj_buf.data();
        }
        profile_gemma4_cuda_stage(
            profile, profile.ple_inputs_ms, stream, [&] {
        // Embed lookup into the per-layer table.
        kernels::launch_embed_bf16(
            token_ids, w.embed_per_layer->data(), per_layer_token,
            N, per_layer_total, V, stream);
        kernels::launch_scalar_mul_bf16(
            per_layer_token, std::sqrt(static_cast<float>(ple_dim)),
            static_cast<std::size_t>(N) * per_layer_total, stream);

        // Project the main embedding to the per-layer subspace.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.y.data(), w.ple_model_proj->data(), per_layer_proj,
            N, per_layer_total, H);
        kernels::launch_scalar_mul_bf16(
            per_layer_proj, 1.0f / std::sqrt(static_cast<float>(H)),
            static_cast<std::size_t>(N) * per_layer_total, stream);

        // RMSNorm per ple_dim row. We reshape mentally to
        // [N*L, ple_dim] and run our row-wise rmsnorm at that shape.
        kernels::launch_rmsnorm_bf16(
            per_layer_proj, w.ple_model_norm->data(), per_layer_proj,
            N * L, ple_dim, eps, stream);

        // (per_layer_proj + per_layer_token) * 1/sqrt(2). residual_add
        // gives us in-place add; then scale.
        kernels::launch_residual_add_bf16(
            per_layer_proj, per_layer_token,
            static_cast<std::size_t>(N) * per_layer_total, stream);
        kernels::launch_scalar_mul_bf16(
            per_layer_proj, 1.0f / std::sqrt(2.0f),
            static_cast<std::size_t>(N) * per_layer_total, stream);
        // The layer loop consumes one `[N, ple_dim]` slice at a time.
        // Re-layout once here instead of launching a slice-pack kernel
        // for every layer and fire.
        kernels::launch_transpose_bf16_nld_to_lnd(
            static_cast<const std::uint16_t*>(per_layer_proj),
            static_cast<std::uint16_t*>(per_layer_token),
            N, L, ple_dim, stream);
            });
    }
    // After this block, `per_layer_proj` keeps the original [N, L, D]
    // signal for dumps; `per_layer_token` holds the [L, N, D] layout
    // consumed by the layer loop.
    dbg_sync_dump_bf16("per_layer_inputs", per_layer_proj,
                  static_cast<std::size_t>(N) * L * ple_dim);

    // Hoist decode plan(s). Gemma-4 has dual head_dim so we plan twice
    // — once at sliding head_dim, once at global head_dim. Both reuse
    // the same workspace (flashinfer's plan info encodes only memory
    // offsets, which don't conflict across kernels run sequentially).
    // We call the kernel-level plan inline per-layer instead — cheaper
    // for small batches than maintaining two cached plans.

    // ── 3. Layer loop ────────────────────────────────────────────────────
    int debug_max_layers = L;
    if (const char* lim = getenv("PIE_GEMMA4_MAX_LAYERS")) {
        debug_max_layers = std::min(L, std::atoi(lim));
    }
    if (profile.enabled) {
        profile.layers = debug_max_layers;
    }
    bool attn_norm_precomputed = false;
    for (int l = 0; l < debug_max_layers; ++l) {
        const auto& layer = w.layers[l];
        const bool dump_this = (l == 0);
        // Pair the sync with the dump so a release run (no dump dir
        // env var) skips both — the standalone syncs that used to
        // precede each dump_l0 call were the dominant per-step
        // overhead on Gemma-4 (~3-15 ms per fire across 30 layers).
        auto dump_l0 = [&](const char* tag, const void* p, std::size_t n) {
            if (!dump_this || !dbg_dumps_enabled() || !dbg_first_fire_flag()) return;
            cudaDeviceSynchronize();
            std::string t = std::string("L0_") + tag;
            dbg_dump_bf16(t.c_str(), p, n);
        };
        // Per-layer dims sharded by tp_size on TP runs. The head/intermediate
        // counts must be divisible by tp_size — guarded at engine load.
        const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        const int d  = layer.head_dim;
        const int num_q_heads_local  = cfg.num_attention_heads / T;
        // KV-head count is now per-layer: 26B-A4B's full-attention
        // layers use num_global_key_value_heads; sliding layers and
        // every other Gemma-4 family use the standard num_key_value_heads.
        const int num_kv_heads_local = layer.num_kv_heads / T;
        const int Hq = num_q_heads_local * d;
        const int Hk = num_kv_heads_local * d;
        const int I  = layer.intermediate / T;
        NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

        // ── 3a. Attention block ─────────────────────────────────────────
        auto kv_view = cache.layer_view(l);
        profile_gemma4_cuda_stage(
            profile, profile.attn_prep_ms, stream, [&] {
        if (!attn_norm_precomputed) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), layer.attn_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        }
        attn_norm_precomputed = false;
        dump_l0("attn_norm_pre", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        // RoPE: partial rotary on full-attention layers
        // (`partial_rotary_factor < 1`), full rotation otherwise.
        const float prf = w.per_layer_partial_rotary_factor[l];
        const int rotary_dim = static_cast<int>(prf * d);
        const bool partial = (prf < 1.0f) && (rotary_dim > 0);
        const bool qk_norm_enabled = getenv("PIE_NO_QK_NORM") == nullptr;
        bool qkv_post_fused = false;

        const bool use_fused_qkv =
            !layer.is_shared &&
            gemma4_dense_qkv_fused_enabled() &&
            layer.qkv_proj_fused != nullptr &&
            !ws.qkv_fused.empty();
        if (use_fused_qkv) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), layer.qkv_proj_fused->data(),
                ws.qkv_fused.data(), N, Hq + 2 * Hk, H);
            const bool can_fuse_packed_qkv_post =
                qk_norm_enabled && !partial && !dbg_dumps_enabled() &&
                kv_view.is_native_bf16() &&
                (use_row_decode_path || use_decode_path);
            if (can_fuse_packed_qkv_post) {
                const std::uint32_t* post_kv_page_indices = use_row_decode_path
                    ? moe_ws.row_decode_kv_page_indices.data()
                    : kv_page_indices;
                const std::uint32_t* post_kv_page_indptr = use_row_decode_path
                    ? moe_ws.row_decode_kv_page_indptr.data()
                    : kv_page_indptr;
                const std::uint32_t* post_kv_last_page_lens = use_row_decode_path
                    ? moe_ws.row_decode_kv_last_page_lens.data()
                    : kv_last_page_lens;
                kernels::launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    ws.qkv_fused.data(), ws.q.data(),
                    kv_view.k_pages, kv_view.v_pages,
                    layer.q_norm->data(), layer.k_norm->data(),
                    positions, post_kv_page_indices, post_kv_page_indptr,
                    post_kv_last_page_lens, N, num_q_heads_local,
                    num_kv_heads_local, d, cache.page_size(),
                    kv_view.hnd_layout, w.per_layer_rope_theta[l], eps,
                    stream);
                qkv_post_fused = true;
            } else {
                kernels::launch_split_qkv_bf16(
                    ws.qkv_fused.data(), ws.q.data(), ws.k.data(), ws.v.data(),
                    N, Hq, Hk, stream);
            }
        } else {
            // Q-projection always runs.
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), layer.q_proj->data(), ws.q.data(),
                N, Hq, H);
            if (!layer.is_shared) {
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), layer.k_proj->data(), ws.k.data(),
                    N, Hk, H);
                if (layer.use_k_as_v) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        ws.v.data(), ws.k.data(),
                        static_cast<std::size_t>(N) * Hk *
                            sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream));
                } else {
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        ws.norm_x.data(), layer.v_proj->data(), ws.v.data(),
                        N, Hk, H);
                }
            }
        }

        // Pre-norm dumps for parity.
        if (l == 0 && !layer.is_shared) {
            dump_l0("v_pre_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * num_kv_heads_local * d);
            dump_l0("q_pre_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * num_q_heads_local * d);
        }

        const bool can_fuse_qk_norm_rope = qk_norm_enabled && !partial;
        if (qkv_post_fused) {
            // Packed QKV post-processing already produced Q and wrote K/V.
        } else if (can_fuse_qk_norm_rope) {
            if (!layer.is_shared) {
                // V-Norm: pure RMSNorm (no learnable scale) on V before the
                // KV write. Gemma-4 trained against this; skipping it
                // produces gibberish even though softmax stays well-formed.
                kernels::launch_rmsnorm_no_scale_bf16(
                    ws.v.data(), ws.v.data(),
                    N * num_kv_heads_local, d, eps, stream);
            }
            kernels::launch_qk_rmsnorm_rope_bf16_rounded(
                ws.q.data(), ws.k.data(),
                layer.q_norm->data(),
                layer.is_shared ? nullptr : layer.k_norm->data(),
                positions, N, num_q_heads_local,
                layer.is_shared ? 0 : num_kv_heads_local, d,
                w.per_layer_rope_theta[l], eps, stream);
        } else {
            // Per-head Q/K RMSNorm (Gemma-4 always has it).
            if (qk_norm_enabled) {
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), layer.q_norm->data(), ws.q.data(),
                    N * num_q_heads_local, d, eps, stream);
                if (!layer.is_shared) {
                    kernels::launch_rmsnorm_bf16(
                        ws.k.data(), layer.k_norm->data(), ws.k.data(),
                        N * num_kv_heads_local, d, eps, stream);
                    kernels::launch_rmsnorm_no_scale_bf16(
                        ws.v.data(), ws.v.data(),
                        N * num_kv_heads_local, d, eps, stream);
                }
            }

            if (!layer.is_shared) {
                if (partial) {
                    kernels::launch_rope_partial_bf16(
                        ws.q.data(), ws.k.data(), positions,
                        N, num_q_heads_local, num_kv_heads_local, d,
                        rotary_dim, w.per_layer_rope_theta[l], stream);
                } else {
                    kernels::launch_rope_bf16(
                        ws.q.data(), ws.k.data(), positions,
                        N, num_q_heads_local, num_kv_heads_local, d,
                        w.per_layer_rope_theta[l], stream);
                }
            } else {
                // Shared layers: only Q gets RoPE'd here; K was rotated at
                // its source layer (where it was written to the cache).
                if (partial) {
                    kernels::launch_rope_partial_bf16(
                        ws.q.data(), ws.q.data(), positions,
                        N, num_q_heads_local, /*num_kv_heads=*/0, d,
                        rotary_dim, w.per_layer_rope_theta[l], stream);
                } else {
                    kernels::launch_rope_bf16(
                        ws.q.data(), ws.q.data(), positions,
                        N, num_q_heads_local, /*num_kv_heads=*/0, d,
                        w.per_layer_rope_theta[l], stream);
                }
            }
        }
        if (l == 0 && !layer.is_shared) {
            dump_l0("v_post_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * num_kv_heads_local * d);
            dump_l0("q_post_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * num_q_heads_local * d);
        }

        // KV write only on non-shared layers — shared layers attend
        // through the source slot's already-populated pages.
        if (!layer.is_shared && !qkv_post_fused) {
            kernels::launch_write_kv_to_pages(
                kv_view, ws.k.data(), ws.v.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, stream);
        }
            });

        // Plan + dispatch attention. Shared layers dispatch against the
        // source slot's tensors (KvCache redirects via `kv_source_layer`).
        // Gemma-4 full-attention layers run at HEAD_DIM=512, which
        // flashinfer 0.6.x's TC prefill template rejects ("Invalid
        // configuration: NUM_MMA_D_QK=32"). For prefill at 512 we fall
        // back to a naive paged-attention kernel (much slower but
        // correct); decode at 512 still uses flashinfer.
        profile_gemma4_cuda_stage(
            profile, profile.attention_ms, stream, [&] {
                ops::DecodePlanCachePtr decode_plan;
                if (use_decode_path) {
                    ops::DecodePlanCache* plan =
                        select_prepared_plan(
                            moe_ws, /*row_decode=*/false, layer.is_full).get();
                    ops::DecodePlanCachePtr decode_plan;
                    if (plan == nullptr) {
                        decode_plan = ops::make_decode_plan();
                        plan = decode_plan.get();
                        ops::plan_attention_flashinfer_decode(
                            *plan, kv_page_indptr_h, R,
                            num_q_heads_local, num_kv_heads_local, d,
                            cache.page_size(), attn_ws, stream,
                            /*enable_cuda_graph=*/true,
                            /*full_attention_variant=*/layer.is_full,
                            cache.hnd_layout());
                    }
                    ops::dispatch_attention_flashinfer_decode(
                        *plan,
                        ws.q.data(), kv_view, ws.attn_out.data(),
                        kv_page_indices, kv_page_indptr, kv_last_page_lens,
                        attn_ws, stream,
                        /*window_left=*/w.per_layer_window_left[l],
                        /*logits_soft_cap=*/0.f,
                        /*sm_scale=*/1.0f);
                } else if (use_row_decode_path) {
                    // Short speculative-verification blocks are causal and
                    // already have K/V written above. Treat each query row as
                    // its own decode request with a prefix-specific page table.
                    ops::DecodePlanCache* plan =
                        select_prepared_plan(
                            moe_ws, /*row_decode=*/true, layer.is_full).get();
                    ops::DecodePlanCachePtr row_plan;
                    if (plan == nullptr) {
                        row_plan = ops::make_decode_plan();
                        plan = row_plan.get();
                        ops::plan_attention_flashinfer_decode(
                            *plan,
                            moe_ws.h_row_decode_kv_page_indptr.data(), N,
                            num_q_heads_local, num_kv_heads_local, d,
                            cache.page_size(), attn_ws, stream,
                            /*enable_cuda_graph=*/true,
                            /*full_attention_variant=*/layer.is_full,
                            cache.hnd_layout());
                    }
                    ops::dispatch_attention_flashinfer_decode(
                        *plan,
                        ws.q.data(), kv_view, ws.attn_out.data(),
                        moe_ws.row_decode_kv_page_indices.data(),
                        moe_ws.row_decode_kv_page_indptr.data(),
                        moe_ws.row_decode_kv_last_page_lens.data(),
                        attn_ws, stream,
                        /*window_left=*/w.per_layer_window_left[l],
                        /*logits_soft_cap=*/0.f,
                        /*sm_scale=*/1.0f);
                } else if (d == 512) {
                    ops::launch_attention_naive_paged(
                        ws.q.data(), kv_view, ws.attn_out.data(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens, N, R, kv_page_indptr_h[R],
                        num_q_heads_local, stream,
                        /*window_left=*/w.per_layer_window_left[l],
                        /*sm_scale=*/1.0f,
                        /*logits_soft_cap=*/0.f,
                        /*lse_out=*/nullptr);
                } else {
                    ops::launch_attention_flashinfer_prefill(
                        ws.q.data(), kv_view, ws.attn_out.data(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                        N, R, num_q_heads_local, attn_ws, stream,
                        /*window_left=*/w.per_layer_window_left[l],
                        /*logits_soft_cap=*/0.f,
                        /*sm_scale=*/1.0f);
                }
            });

        dump_l0("attn_out", ws.attn_out.data(),
                static_cast<std::size_t>(N) * Hq);

        // o_proj → norm_x scratch, post-attn norm, residual-add y. Under
        // TP this is row-parallel: all-reduce the partial sums before
        // post-norm sees them.
        profile_gemma4_cuda_stage(
            profile, profile.attn_out_ms, stream, [&] {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
            N, H, Hq, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        dump_l0("o_proj_out", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);
        if (!dbg_dumps_enabled()) {
            kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
                ws.norm_x.data(), layer.attn_norm_post->data(), ws.y.data(),
                1.f, layer.mlp_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.attn_norm_post->data(), ws.norm_y.data(),
                N, H, eps, stream);
            dump_l0("attn_norm_post", ws.norm_y.data(),
                    static_cast<std::size_t>(N) * H);
            kernels::launch_residual_add_rmsnorm_bf16(
                ws.y.data(), ws.norm_y.data(), layer.mlp_norm_pre->data(),
                ws.norm_x.data(), N, H, eps, stream);
        }
        dump_l0("post_attn_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);
            });

        // ── 3b. MLP block ──────────────────────────────────────────────
        profile_gemma4_cuda_stage(profile, profile.mlp_ms, stream, [&] {
        dump_l0("mlp_norm_pre", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        const bool use_gate_up_fused =
            (!use_row_decode_path ||
             gemma4_dense_gate_up_fused_for_row_decode(cfg)) &&
            layer.gate_up_proj_fused != nullptr &&
            !ws.gate_up_fused.empty();
        if (use_gate_up_fused) {
            ops::gemm_act_x_wt_bf16_cublas(cublas.handle(),
                ws.norm_x.data(), layer.gate_up_proj_fused->data(),
                ws.gate_up_fused.data(), N, 2 * I, H);
            kernels::launch_chunked_geglu_tanh_bf16(
                ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
        } else if (gemma4_dense_gate_up_batched_enabled() &&
            moe_ws.a_gu_ptrs.size() >= 2 &&
            moe_ws.b_gu_ptrs.size() >= 2 &&
            moe_ws.c_gu_ptrs.size() >= 2) {
            kernels::launch_build_dual_bf16_gemm_ptrs(
                ws.norm_x.data(),
                layer.gate_proj->data(),
                layer.up_proj->data(),
                ws.gate.data(),
                ws.up.data(),
                reinterpret_cast<const void**>(moe_ws.a_gu_ptrs.data()),
                reinterpret_cast<const void**>(moe_ws.b_gu_ptrs.data()),
                reinterpret_cast<void**>(moe_ws.c_gu_ptrs.data()),
                stream);
            ops::gemm_batched_act_x_wt_bf16(
                cublas.handle(),
                reinterpret_cast<const void* const*>(moe_ws.a_gu_ptrs.data()),
                reinterpret_cast<const void* const*>(moe_ws.b_gu_ptrs.data()),
                reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
                N, I, H, /*batch_count=*/2);
            kernels::launch_geglu_tanh_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        } else {
            if (use_row_decode_path) {
                ops::gemm_act_x_wt_bf16_cublas(cublas.handle(),
                    ws.norm_x.data(), layer.gate_proj->data(), ws.gate.data(),
                    N, I, H);
                ops::gemm_act_x_wt_bf16_cublas(cublas.handle(),
                    ws.norm_x.data(), layer.up_proj->data(),   ws.up.data(),
                    N, I, H);
            } else {
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), layer.gate_proj->data(), ws.gate.data(),
                    N, I, H);
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), layer.up_proj->data(),   ws.up.data(),
                    N, I, H);
            }
            kernels::launch_geglu_tanh_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        }
        dump_l0("mlp_geglu", ws.gate.data(),
                static_cast<std::size_t>(N) * I);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.gate.data(), layer.down_proj->data(), ws.norm_x.data(),
            N, H, I, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        dump_l0("mlp_down", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        // Gemma-4 26B-A4B's MoE block runs **alongside** the dense MLP
        // and the two branches' post-norms are summed before the final
        // `post_feedforward_layernorm`. On the dense E2B / E4B / 31B
        // variants `cfg.gemma4_enable_moe` is false and we keep the
        // straight-line dense path.
        const bool moe_active = cfg.gemma4_enable_moe &&
                                layer.router_proj != nullptr;
        if (moe_active) {
            // branch_1 = post_feedforward_layernorm_1(dense_out)
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm_post_dense->data(),
                ws.norm_y.data(), N, H, eps, stream);
            // experts → moe_ws.moe_out (raw, no post-norm).
            gemma4_moe_block(layer, cfg, ws, moe_ws, N, cublas, stream);
            // branch_2 = post_feedforward_layernorm_2(moe_out) → norm_x
            // (norm_x's prior contents — dense_out — are no longer
            // needed).
            kernels::launch_rmsnorm_bf16(
                moe_ws.moe_out.data(), layer.moe_norm_post->data(),
                ws.norm_x.data(), N, H, eps, stream);
            // combined = branch_1 + branch_2 (in norm_y).
            kernels::launch_residual_add_bf16(
                ws.norm_y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, stream);
            // final = post_feedforward_layernorm(combined) → norm_x.
            kernels::launch_rmsnorm_bf16(
                ws.norm_y.data(), layer.mlp_norm_post->data(),
                ws.norm_x.data(), N, H, eps, stream);
            dump_l0("mlp_norm_post", ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, stream);
        } else {
            if (!dbg_dumps_enabled()) {
                kernels::launch_rmsnorm_residual_add_bf16(
                    ws.norm_x.data(), layer.mlp_norm_post->data(),
                    ws.y.data(), N, H, eps, stream);
            } else {
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), layer.mlp_norm_post->data(),
                    ws.norm_y.data(), N, H, eps, stream);
                dump_l0("mlp_norm_post", ws.norm_y.data(),
                        static_cast<std::size_t>(N) * H);
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_y.data(),
                    static_cast<std::size_t>(N) * H, stream);
            }
        }
        dump_l0("post_mlp_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);
        });

        // ── 3c. PLE residual ───────────────────────────────────────────
        // Wrapped in a block so debugging can disable the whole step
        // (env `PIE_NO_PLE=1`) without touching the surrounding flow.
        // Skipped on Gemma-4 26B-A4B (`hidden_size_per_layer_input == 0`
        // → `ple_dim == 0`), which doesn't ship the per-layer triple at
        // all.
        const bool ple_active =
            ple_dim > 0 && getenv("PIE_NO_PLE") == nullptr;
        const bool layer_scalar_active =
            layer.layer_scalar && getenv("PIE_NO_LAYER_SCALAR") == nullptr &&
            std::abs(layer.layer_scalar_value - 1.f) > 1e-6f;
        const float layer_scalar =
            layer_scalar_active ? layer.layer_scalar_value : 1.f;
        bool scalar_applied_in_ple = false;
        bool next_attn_norm_ready = false;
        profile_gemma4_cuda_stage(
            profile, profile.ple_residual_ms, stream, [&] {
        if (ple_active) {
        dump_l0("ple_residual_in", ws.y.data(),
                static_cast<std::size_t>(N) * H);
        // ple_gate = ple_input_gate @ y_norm (using attn output in y)
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.y.data(), layer.ple_input_gate->data(), ws.norm_x.data(),
            N, ple_dim, H);
        dump_l0("ple_gate_pre_gelu", ws.norm_x.data(),
                static_cast<std::size_t>(N) * ple_dim);
        // GeGLU(tanh) but with the per-layer input acting as the "up"
        // signal. `per_layer_token` now holds `[L, N, ple_dim]`, so the
        // current layer's signal is already contiguous.
        const auto* ple_signal =
            static_cast<const std::uint16_t*>(per_layer_token) +
            static_cast<std::size_t>(l) * N * ple_dim;
        dump_l0("ple_signal_slice", ple_signal,
                static_cast<std::size_t>(N) * ple_dim);
        kernels::launch_geglu_tanh_bf16(
            ws.norm_x.data(), ple_signal, ws.norm_x.data(),
            N * ple_dim, stream);
        dump_l0("ple_gated", ws.norm_x.data(),
                static_cast<std::size_t>(N) * ple_dim);
        // Project back to hidden, post-norm, add to residual.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.ple_projection->data(), ws.norm_y.data(),
            N, H, ple_dim, /*beta=*/0.f);
        if (l + 1 < debug_max_layers && !dbg_dumps_enabled()) {
            kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
                ws.norm_y.data(), layer.ple_norm->data(), ws.y.data(),
                layer_scalar, w.layers[l + 1].attn_norm_pre->data(),
                ws.norm_x.data(), N, H, eps, stream);
            scalar_applied_in_ple = true;
            next_attn_norm_ready = true;
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.norm_y.data(), layer.ple_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
        if (l + 1 < debug_max_layers) {
            kernels::launch_residual_add_scale_rmsnorm_bf16(
                ws.y.data(), ws.norm_y.data(), layer_scalar,
                w.layers[l + 1].attn_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
            scalar_applied_in_ple = true;
            next_attn_norm_ready = true;
        } else {
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, stream);
        }
        }
        }  // end PLE-bypass guard
            });

        // Parity dump: residual stream after attention/MLP/PLE for
        // the first few layers.
        if (l < 4) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_ple_y", l);
            dbg_sync_dump_bf16(tag, ws.y.data(),
                               static_cast<std::size_t>(N) * H);
        }

        // ── 3d. Per-layer learnable scalar ────────────────────────────
        if (layer_scalar_active && !scalar_applied_in_ple) {
            kernels::launch_scalar_mul_bf16(
                ws.y.data(), layer_scalar,
                static_cast<std::size_t>(N) * H, stream);
        }
        attn_norm_precomputed = next_attn_norm_ready;

        // Post-layer_scalar dump for parity comparison against HF's
        // `hidden_states[layer+1]` (which is after the scalar mul).
        if (l < 4) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_scalar_y", l);
            dbg_sync_dump_bf16(tag, ws.y.data(),
                               static_cast<std::size_t>(N) * H);
        }
    }

    // ── 4. Final norm, lm_head, optional softcap ─────────────────────
    const bool compact_logits =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N;
    int lm_head_rows = N;
    const void* lm_head_input = ws.norm_x.data();
    if (compact_logits) {
        // System speculation consumes the verifier's normalized hidden
        // state by original row index. Keep ws.norm_x populated for all
        // rows, then compact only the expensive lm_head input.
        profile_gemma4_cuda_stage(
            profile, profile.final_norm_ms, stream, [&] {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                    N, H, eps, stream);
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.norm_x.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_y.data()),
                    num_logit_rows, H, stream);
            });
        lm_head_input = ws.norm_y.data();
        lm_head_rows = num_logit_rows;
    } else {
        profile_gemma4_cuda_stage(
            profile, profile.final_norm_ms, stream, [&] {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                    N, H, eps, stream);
            });
    }
    if (profile.enabled) {
        profile.compact_logits = compact_logits;
        profile.lm_head_rows = lm_head_rows;
    }
    const bool use_fused_lmhead_argmax =
        fused_lmhead_argmax_enabled() &&
        g_logits_argmax_only && g_fused_argmax_output != nullptr &&
        lm_head_rows > 0 && !dbg_dumps_enabled();
    if (use_fused_lmhead_argmax) {
        profile_gemma4_cuda_stage(profile, profile.lm_head_ms, stream, [&] {
            constexpr int MAX_TILES = 8;
            const int num_tiles = std::clamp(V / 32768, 1, MAX_TILES);
            const int tile_size = (V + num_tiles - 1) / num_tiles;
            const auto* lm_head_bf16 =
                static_cast<const std::uint16_t*>(w.lm_head->data());
            auto* pairs = reinterpret_cast<std::uint64_t*>(
                ws.greedy_pairs_all.data());
            for (int tile = 0; tile < num_tiles; ++tile) {
                const int tile_start = tile * tile_size;
                const int this_tile = std::min(tile_size, V - tile_start);
                if (this_tile <= 0) break;
                const void* tile_w =
                    lm_head_bf16 +
                    static_cast<long long>(tile_start) * H;
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    lm_head_input, tile_w, ws.logits.data(),
                    lm_head_rows, this_tile, H);
                kernels::launch_argmax_bf16_tile_pair(
                    ws.logits.data(),
                    pairs + static_cast<std::size_t>(tile) * lm_head_rows,
                    lm_head_rows, this_tile, tile_start, stream);
            }
            kernels::launch_select_global_argmax_pairs(
                pairs, g_fused_argmax_output,
                lm_head_rows, num_tiles, stream);
        });
        g_fused_argmax_done = true;
        if (profile.enabled) {
            profile.logits_argmax_only = true;
        }
    } else {
        profile_gemma4_cuda_stage(profile, profile.lm_head_ms, stream, [&] {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                lm_head_input, w.lm_head->data(), ws.logits.data(),
                lm_head_rows, V, H);
        });
        const bool skip_final_softcap =
            g_logits_argmax_only && !dbg_dumps_enabled();
        if (profile.enabled) {
            profile.logits_argmax_only = g_logits_argmax_only;
        }
        if (fwd_cfg.final_logit_softcap > 0.f && !skip_final_softcap) {
            profile_gemma4_cuda_stage(profile, profile.softcap_ms, stream, [&] {
                kernels::launch_logit_softcap_bf16(
                    ws.logits.data(), fwd_cfg.final_logit_softcap,
                    static_cast<std::size_t>(lm_head_rows) * V, stream);
            });
        }
    }
    if (lm_head_rows > 0 && !use_fused_lmhead_argmax) {
        dbg_sync_dump_bf16("logits_last",
            static_cast<const std::uint16_t*>(ws.logits.data()) +
                static_cast<std::size_t>(lm_head_rows - 1) * V,
            static_cast<std::size_t>(V));
    }
    maybe_print_gemma4_forward_profile(profile, stream);
    // After the first fire, freeze the dumps so subsequent decode
    // fires don't overwrite the prefill intermediates we want to
    // parity-check.
    dbg_first_fire_flag() = false;
}

std::size_t gemma4_moe_workspace_bytes(const HfConfig& cfg, int N) {
    if (!cfg.gemma4_enable_moe || cfg.num_experts <= 0 ||
        cfg.num_experts_per_tok <= 0 || cfg.moe_intermediate_size <= 0) {
        return 0;
    }
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t maxR = n * cfg.num_experts_per_tok;
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t I = static_cast<std::size_t>(cfg.moe_intermediate_size);
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto i32 = [](std::size_t elems) { return elems * 4; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    std::size_t bytes = 0;
    bytes += u16(n * H);
    bytes += u16(n * cfg.num_experts);
    bytes += i32(n * cfg.num_experts_per_tok);
    bytes += fp32(n * cfg.num_experts_per_tok);
    bytes += u16(n * H);
    bytes += u16(maxR * H);
    bytes += u16(maxR * 2 * I);
    bytes += u16(maxR * I);
    bytes += u16(maxR * H);
    bytes += i32(maxR);
    bytes += fp32(maxR);
    bytes += u16(n * H);
    bytes += static_cast<std::size_t>(cfg.num_experts_per_tok) *
             (6 * sizeof(void*) + sizeof(float));
    return bytes;
}

}  // namespace pie_cuda_driver::model
