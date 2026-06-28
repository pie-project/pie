#include "executor/executor.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "kv_cache.hpp"
#include "recurrent_state_cache.hpp"
#include "response_subpass.hpp"
#include "model/imodel.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_temp.hpp"
#include "sampler_type.hpp"
#include "sampling_dispatch.hpp"
#include "sampling_ir/pie_standard_samplers.h"
#include "sampling_ir/sampler_dispatch.hpp"
#include "sampling_ir/group.hpp"
#include "sampling_ir/next_input.hpp"
#include "sampling_ir/tensor_io.hpp"
#include "sampling_ir/program_recognizer.hpp"
#include "sampling_ir/param_extract.hpp"
#include "spec_expansion.hpp"

namespace pie_cuda_driver {

void ForwardFn::attach_model(model::IModel* m) {
    model = m;
    if (m == nullptr) return;
    const auto caps = m->capabilities();
    graph_safe                   = caps.graph_safe;
    supports_compact_logits      = caps.supports_compact_logits;
    supports_tp_greedy_argmax    = caps.supports_tp_greedy_argmax;
    supports_small_prefill_graph = caps.supports_small_prefill_graph;
    supports_fused_lmhead_argmax = caps.supports_fused_lmhead_argmax;
}

void ForwardFn::invoke_prepare(AttentionWorkspace& aws,
                               const PrepareInputs& in) {
    if (model) model->prepare(aws, in);
}

void ForwardFn::invoke_body(model::Qwen3Workspace& ws,
                            KvCache& kv,
                            AttentionWorkspace& aws,
                            ops::CublasHandle& cublas,
                            const ForwardInputs& in) {
    if (model) model->body(ws, kv, aws, cublas, in);
}

std::uint32_t ForwardFn::invoke_graph_layout() {
    return model ? model->graph_layout() : 0u;
}

void ForwardFn::invoke_set_logits_argmax_only(bool enabled) {
    if (model) model->set_logits_argmax_only(enabled);
}

void ForwardFn::invoke_set_fused_argmax_output(std::int32_t* ptr) {
    if (model) model->set_fused_argmax_output(ptr);
}

bool ForwardFn::invoke_fused_argmax_done() {
    return model ? model->fused_argmax_done() : false;
}

namespace {

struct TpCpuGate {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<std::uint64_t> seq{0};
};

std::mutex g_tp_cpu_gates_mu;
std::unordered_map<std::string, std::shared_ptr<TpCpuGate>> g_tp_cpu_gates;

void tp_graph_capture_barrier(const Executor& executor) {
    if (executor.tp_comm == nullptr) return;
    if (executor.tp_cpu_gate_key.empty()) return;
    const int world = executor.tp_comm->world_size();
    if (world <= 1) return;

    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<std::barrier<>>>
        registry;

    std::shared_ptr<std::barrier<>> b;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[executor.tp_cpu_gate_key + ":graph_capture"];
        if (!entry) entry = std::make_shared<std::barrier<>>(world);
        b = entry;
    }
    b->arrive_and_wait();
}

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

std::uint64_t initial_sampling_seed() {
    // Test/debug determinism: a fixed seed makes the per-fire RNG sequence
    // reproducible across processes (used by the #7 per-kind gate-on≡gate-off
    // verify to isolate dispatch from seed noise). Default stays non-deterministic.
    if (const char* fixed = std::getenv("PIE_FIXED_SAMPLING_SEED")) {
        return splitmix64(std::strtoull(fixed, nullptr, 10));
    }
    std::uint64_t seed =
        static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= reinterpret_cast<std::uintptr_t>(&seed);
    try {
        std::random_device rd;
        seed ^= static_cast<std::uint64_t>(rd()) << 32;
        seed ^= static_cast<std::uint64_t>(rd());
    } catch (...) {
        // Some libstdc++/container combinations can throw when entropy is
        // unavailable. The clock/address mix above is still enough to avoid
        // a fixed sampler sequence across server starts.
    }
    return splitmix64(seed);
}

std::uint32_t fresh_sampling_seed() {
    static std::atomic<std::uint64_t> counter{initial_sampling_seed()};
    std::uint64_t x = splitmix64(counter.fetch_add(
        0x9E3779B97F4A7C15ULL, std::memory_order_relaxed));
    std::uint32_t seed = static_cast<std::uint32_t>(x ^ (x >> 32));
    return seed == 0 ? 1u : seed;
}

struct SamplingScratch {
    std::vector<float> h_per_temp;
    std::vector<float> h_per_min_p;
    std::vector<float> h_per_top_p;
    std::vector<std::int32_t> h_per_top_k;
    std::vector<std::uint32_t> h_per_seed;
    std::vector<std::uint32_t> per_slot_type;
    std::vector<float> per_slot_temp;
    std::vector<float> per_slot_top_p;
    std::vector<float> per_slot_min_p;
    std::vector<std::int32_t> per_slot_top_k;
    std::vector<std::uint32_t> per_slot_seed;
    std::vector<std::int32_t> h_sample_idx;
};

SamplingScratch& sampling_scratch() {
    thread_local SamplingScratch scratch;
    return scratch;
}

int partitioned_argmax_parts() {
    static const int parts = [] {
        const char* v = std::getenv("PIE_ARGMAX_PARTS");
        if (v == nullptr || v[0] == '\0') return 1;
        return std::clamp(std::atoi(v), 1, 8);
    }();
    return parts;
}

int greedy_argmax_parts(int vocab) {
    const char* global = std::getenv("PIE_ARGMAX_PARTS");
    if (global != nullptr && global[0] != '\0') {
        return partitioned_argmax_parts();
    }
    return vocab >= 131072 ? 8 : 1;
}

int mtp_argmax_parts(int vocab) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MTP_ARGMAX_PARTS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::clamp(std::atoi(v), 1, 8);
    }();
    if (forced > 0) return forced;
    return greedy_argmax_parts(vocab);
}

bool graph_single_gpu_argmax_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_GRAPH_ARGMAX");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

int qwen35_small_spec_graph_tokens() {
    static const int tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_SPEC_VERIFY_GRAPH_N");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::clamp(std::atoi(v), 0, 64);
    }();
    return tokens;
}

int small_spec_graph_max_requests() {
    static const int requests = [] {
        const char* v = std::getenv("PIE_SPEC_VERIFY_GRAPH_MAX_R");
        if (v == nullptr || v[0] == '\0') return 16;
        return std::clamp(std::atoi(v), 1, 512);
    }();
    return requests;
}

int small_spec_graph_min_requests(int max_requests) {
    static const int configured = [] {
        const char* v = std::getenv("PIE_SPEC_VERIFY_GRAPH_MIN_R");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::clamp(std::atoi(v), 1, 512);
    }();
    if (configured > 0) return std::min(configured, max_requests);
    return 1;
}

bool mtp_trace_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t mtp_trace_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_MTP_TRACE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{32};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool mtp_trace_take() {
    if (!mtp_trace_enabled()) return false;
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) < mtp_trace_limit();
}

bool mtp_argmax_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t mtp_argmax_profile_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool mtp_argmax_profile_take() {
    if (!mtp_argmax_profile_enabled()) return false;
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) <
           mtp_argmax_profile_limit();
}

bool mtp_process_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_PROCESS_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t mtp_process_profile_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_MTP_PROCESS_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{16};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool mtp_process_profile_take() {
    if (!mtp_process_profile_enabled()) return false;
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) <
           mtp_process_profile_limit();
}

bool step_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_STEP_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t step_profile_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_STEP_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{32};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool step_profile_take() {
    if (!step_profile_enabled()) return false;
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) < step_profile_limit();
}

bool mtp_chain_graph_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_CHAIN_GRAPH");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

struct StepProfileTimer {
    bool enabled = false;
    const char* label = "";
    int tokens = 0;
    int requests = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    StepProfileTimer(
        const char* label_, cudaStream_t stream, int tokens_, int requests_)
        : enabled(step_profile_take()),
          label(label_),
          tokens(tokens_),
          requests(requests_)
    {
        if (!enabled) return;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    ~StepProfileTimer() {
        if (start) CUDA_CHECK(cudaEventDestroy(start));
        if (stop) CUDA_CHECK(cudaEventDestroy(stop));
    }

    void finish(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cerr << "[pie-step-profile] label=" << label
                  << " tokens=" << tokens
                  << " requests=" << requests
                  << " ms=" << ms << "\n";
        enabled = false;
    }
};

template <typename Fn>
void profile_mtp_process_call(
    const char* label,
    cudaStream_t stream,
    int total_tokens,
    int num_requests,
    Fn&& fn)
{
    const bool profile = mtp_process_profile_take();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    fn();
    if (profile) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cerr << "[pie-mtp-process-profile] label=" << label
                  << " tokens=" << total_tokens
                  << " requests=" << num_requests
                  << " process_ms=" << ms << "\n";
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

int mtp_graph_max_global_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_MTP_GRAPH_MAX_GLOBAL_TOKENS");
        if (v == nullptr || v[0] == '\0') return 1024;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

int mtp_graph_global_token_bucket(int observed_tokens) {
    const int max_tokens = mtp_graph_max_global_tokens();
    if (max_tokens <= 0) return -1;
    if (observed_tokens <= 0) return 0;

    int bucket = 1;
    while (bucket < observed_tokens && bucket < max_tokens) {
        bucket <<= 1;
    }
    return bucket >= observed_tokens ? bucket : -1;
}

void launch_mtp_argmax(Executor& executor, int rows, cudaStream_t stream) {
    const int vocab = executor.loaded_model.hf_config().vocab_size;
    const int argmax_parts = mtp_argmax_parts(vocab);
    const bool profile = mtp_argmax_profile_take();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    if (argmax_parts > 1 && !executor.ws.greedy_pairs_all.empty()) {
        kernels::launch_argmax_bf16_partitioned_pairs(
            executor.ws.logits.data(),
            reinterpret_cast<std::uint64_t*>(
                executor.ws.greedy_pairs_all.data()),
            rows, vocab, argmax_parts, stream);
        kernels::launch_select_global_argmax_pairs(
            reinterpret_cast<const std::uint64_t*>(
                executor.ws.greedy_pairs_all.data()),
            reinterpret_cast<std::int32_t*>(executor.inputs.sampled.data()),
            rows, argmax_parts, stream);
    } else {
        kernels::launch_argmax_bf16(
            executor.ws.logits.data(),
            reinterpret_cast<std::int32_t*>(executor.inputs.sampled.data()),
            rows, vocab, stream);
    }
    if (profile) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cerr << "[pie-mtp-argmax-profile] rows=" << rows
                  << " vocab=" << vocab
                  << " parts=" << argmax_parts
                  << " argmax_ms=" << ms << "\n";
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

struct MtpGraphKey {
    const Executor* executor = nullptr;
    int rows = 0;
    int draft_step = 0;
    int max_global_tokens = 0;

    bool operator==(const MtpGraphKey& o) const noexcept {
        return executor == o.executor &&
               rows == o.rows &&
               draft_step == o.draft_step &&
               max_global_tokens == o.max_global_tokens;
    }
};

struct MtpGraphKeyHash {
    std::size_t operator()(const MtpGraphKey& k) const noexcept {
        return reinterpret_cast<std::uintptr_t>(k.executor) ^
               (static_cast<std::size_t>(k.rows) << 8) ^
               (static_cast<std::size_t>(k.draft_step) << 16) ^
               (static_cast<std::size_t>(k.max_global_tokens) << 24);
    }
};

struct MtpChainGraphKey {
    const Executor* executor = nullptr;
    int rows = 0;
    int drafts = 0;
    int max_global_tokens = 0;

    bool operator==(const MtpChainGraphKey& o) const noexcept {
        return executor == o.executor &&
               rows == o.rows &&
               drafts == o.drafts &&
               max_global_tokens == o.max_global_tokens;
    }
};

struct MtpChainGraphKeyHash {
    std::size_t operator()(const MtpChainGraphKey& k) const noexcept {
        return reinterpret_cast<std::uintptr_t>(k.executor) ^
               (static_cast<std::size_t>(k.rows) << 8) ^
               (static_cast<std::size_t>(k.drafts) << 16) ^
               (static_cast<std::size_t>(k.max_global_tokens) << 24);
    }
};

cudaGraphExec_t capture_mtp_graph_exec(
    Executor& executor,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    int rows,
    int draft_step,
    int max_global_tokens)
{
    auto& pi = executor.inputs;

    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    cudaStream_t cstream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
    executor.cublas.set_stream(cstream);
    CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
    executor.system_drafter.draft_step(
        executor.ws, executor.kv_cache, executor.cublas,
        token_ids,
        positions,
        base_hidden_row_indices,
        request_ids,
        pi.kv_page_indices.data(),
        pi.kv_page_indptr.data(),
        pi.kv_last_page_lens.data(),
        executor.system_drafter.draft_step_writes_sampled_tokens
            ? reinterpret_cast<std::int32_t*>(pi.sampled.data())
            : nullptr,
        rows, draft_step, max_global_tokens);
    if (!executor.system_drafter.draft_step_writes_sampled_tokens) {
        launch_mtp_argmax(executor, rows, cstream);
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
    executor.cublas.set_stream(nullptr);

    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(exec, nullptr));
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cstream);
    return exec;
}

cudaGraphExec_t capture_mtp_chain_graph_exec(
    Executor& executor,
    int rows,
    int drafts,
    int max_global_tokens)
{
    auto& pi = executor.inputs;

    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    cudaStream_t cstream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
    executor.cublas.set_stream(cstream);
    CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
    for (int draft = 0; draft < drafts; ++draft) {
        const std::size_t offset =
            static_cast<std::size_t>(draft) * static_cast<std::size_t>(rows);
        executor.system_drafter.draft_step(
            executor.ws, executor.kv_cache, executor.cublas,
            reinterpret_cast<const std::int32_t*>(pi.tokens.data() + offset),
            reinterpret_cast<const std::int32_t*>(pi.positions.data() + offset),
            pi.sample_idx.data() + offset,
            pi.mtp_request_ids.data(),
            pi.kv_page_indices.data(),
            pi.kv_page_indptr.data(),
            pi.kv_last_page_lens.data(),
            executor.system_drafter.draft_step_writes_sampled_tokens
                ? reinterpret_cast<std::int32_t*>(pi.sampled.data())
                : nullptr,
            rows, draft, max_global_tokens);
        if (!executor.system_drafter.draft_step_writes_sampled_tokens) {
            launch_mtp_argmax(executor, rows, cstream);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            pi.tokens.data() + offset + static_cast<std::size_t>(rows),
            pi.sampled.data(),
            sizeof(std::uint32_t) * static_cast<std::size_t>(rows),
            cudaMemcpyDeviceToDevice, cstream));
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
    executor.cublas.set_stream(nullptr);

    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(exec, nullptr));
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cstream);
    return exec;
}

bool try_run_mtp_graph_with_argmax(
    Executor& executor,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    int rows,
    int draft_step,
    int max_global_tokens)
{
    if (executor.tp_comm != nullptr || rows <= 0) return false;
    if (mtp_argmax_profile_enabled() || mtp_trace_enabled()) return false;
    const int graph_tokens = mtp_graph_global_token_bucket(max_global_tokens);
    if (graph_tokens < 0) return false;

    static std::mutex mu;
    static std::unordered_map<MtpGraphKey, cudaGraphExec_t, MtpGraphKeyHash>
        cache;
    const MtpGraphKey key{&executor, rows, draft_step, graph_tokens};
    cudaGraphExec_t exec = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it == cache.end()) {
            exec = capture_mtp_graph_exec(
                executor, token_ids, positions, base_hidden_row_indices,
                request_ids, rows, draft_step, graph_tokens);
            cache.emplace(key, exec);
        } else {
            exec = it->second;
        }
    }
    CUDA_CHECK(cudaGraphLaunch(exec, nullptr));
    return true;
}

bool try_run_mtp_chain_graph_with_argmax(
    Executor& executor,
    int rows,
    int drafts,
    int max_observed_global_tokens)
{
    if (!mtp_chain_graph_enabled()) return false;
    if (executor.tp_comm != nullptr || rows <= 0 || drafts <= 1) return false;
    if (mtp_argmax_profile_enabled() || mtp_trace_enabled()) return false;
    const int graph_tokens =
        mtp_graph_global_token_bucket(max_observed_global_tokens);
    if (graph_tokens < 0) return false;

    static std::mutex mu;
    static std::unordered_map<
        MtpChainGraphKey, cudaGraphExec_t, MtpChainGraphKeyHash> cache;
    const MtpChainGraphKey key{&executor, rows, drafts, graph_tokens};
    cudaGraphExec_t exec = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it == cache.end()) {
            exec = capture_mtp_chain_graph_exec(
                executor, rows, drafts, graph_tokens);
            cache.emplace(key, exec);
        } else {
            exec = it->second;
        }
    }
    CUDA_CHECK(cudaGraphLaunch(exec, nullptr));
    return true;
}

void run_mtp_draft_with_argmax(
    Executor& executor,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    int rows,
    int draft_step,
    int max_global_tokens)
{
    if (rows <= 0) return;
    auto& pi = executor.inputs;
    if (try_run_mtp_graph_with_argmax(
            executor, token_ids, positions, base_hidden_row_indices,
            request_ids, rows, draft_step, max_global_tokens)) {
        return;
    }
    executor.system_drafter.draft_step(
        executor.ws, executor.kv_cache, executor.cublas,
        token_ids,
        positions,
        base_hidden_row_indices,
        request_ids,
        pi.kv_page_indices.data(),
        pi.kv_page_indptr.data(),
        pi.kv_last_page_lens.data(),
        executor.system_drafter.draft_step_writes_sampled_tokens
            ? reinterpret_cast<std::int32_t*>(pi.sampled.data())
            : nullptr,
        rows, draft_step, max_global_tokens);
    if (!executor.system_drafter.draft_step_writes_sampled_tokens) {
        launch_mtp_argmax(executor, rows, executor.cublas.stream());
    }
}

void run_mtp_draft_with_argmax(
    Executor& executor, int rows, int draft_step, int max_global_tokens)
{
    auto& pi = executor.inputs;
    run_mtp_draft_with_argmax(
        executor,
        reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
        reinterpret_cast<const std::int32_t*>(pi.positions.data()),
        pi.sample_idx.data(),
        pi.mtp_request_ids.data(),
        rows, draft_step, max_global_tokens);
}

void tp_broadcast_mtp_inputs(NcclComm& comm, PersistentInputs& pi,
                             int num_tokens, int draft_step,
                             cudaStream_t stream);
int tensor_rows(const DeviceTensor& t);
void tp_cpu_gate_notify(const std::string& key);

void run_step_chained_system_drafter(
    Executor& executor,
    std::span<const SystemSpecDraftRequest> requests,
    std::span<pie_driver::PerRequestOutput> per_req,
    bool mtp_global_cache)
{
    const int max_drafts = executor.system_drafter.max_drafts;
    if (max_drafts <= 0 || requests.empty() ||
        !executor.system_drafter.draft_step) {
        return;
    }

    auto& pi = executor.inputs;
    auto& ws = executor.ws;
    auto& cublas = executor.cublas;

    std::vector<std::uint32_t> mtp_tokens;
    std::vector<std::int32_t> mtp_rows;
    std::vector<std::uint32_t> mtp_input_pos;
    std::vector<std::uint32_t> mtp_output_pos;
    std::vector<int> mtp_requests;
    mtp_tokens.reserve(requests.size());
    mtp_rows.reserve(requests.size());
    mtp_input_pos.reserve(requests.size());
    mtp_output_pos.reserve(requests.size());
    mtp_requests.reserve(requests.size());

    const int draft_position_offset =
        executor.system_drafter.draft_position_offset;
    const bool prefix_global =
        executor.system_drafter.draft_global_cache_uses_prefix_position;
    for (const auto& req : requests) {
        if (req.request_index < 0 ||
            req.request_index >= static_cast<int>(per_req.size()) ||
            req.source_row < 0) {
            continue;
        }
        const std::uint64_t input_pos64 =
            static_cast<std::uint64_t>(req.source_position) +
            static_cast<std::uint64_t>(std::max(0, draft_position_offset));
        if (input_pos64 > std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        mtp_tokens.push_back(req.accepted_token);
        mtp_rows.push_back(req.source_row);
        mtp_input_pos.push_back(static_cast<std::uint32_t>(input_pos64));
        mtp_output_pos.push_back(req.first_draft_position);
        mtp_requests.push_back(req.request_index);
    }

    const int S = static_cast<int>(mtp_tokens.size());
    if (S <= 0) return;
    if (S > tensor_rows(ws.logits)) {
        throw std::runtime_error(
            "MTP draft rows exceed logits workspace capacity");
    }
    if (static_cast<std::size_t>(S) *
            static_cast<std::size_t>(max_drafts) >
        static_cast<std::size_t>(tensor_rows(ws.k))) {
        throw std::runtime_error(
            "MTP draft history exceeds workspace capacity");
    }

    std::vector<std::int32_t> mtp_request_ids(static_cast<std::size_t>(S));
    std::vector<std::int32_t> chained_rows(static_cast<std::size_t>(S));
    for (int i = 0; i < S; ++i) {
        mtp_request_ids[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(mtp_requests[static_cast<std::size_t>(i)]);
        chained_rows[static_cast<std::size_t>(i)] = i;
    }

    const std::size_t s_sz = static_cast<std::size_t>(S);
    const std::size_t drafts_sz = static_cast<std::size_t>(max_drafts);
    const bool gpu_chain_capacity =
        s_sz * (drafts_sz + 1) <= pi.tokens.size() &&
        s_sz * drafts_sz <= pi.positions.size() &&
        s_sz * drafts_sz <= pi.sample_idx.size() &&
        s_sz * drafts_sz <= pi.tokens.size();
    const bool gpu_chain =
        executor.tp_comm == nullptr && !mtp_trace_enabled() &&
        gpu_chain_capacity;

    if (gpu_chain) {
        StepProfileTimer mtp_timer(
            "mtp_draft_chain", cublas.stream(),
            static_cast<int>(s_sz * drafts_sz), S);
        std::vector<std::uint32_t> mtp_positions_flat(s_sz * drafts_sz);
        std::vector<std::int32_t> mtp_rows_flat(s_sz * drafts_sz);
        std::vector<int> mtp_max_global_tokens(drafts_sz, 0);
        int mtp_max_observed_global_tokens = 0;
        for (int draft = 0; draft < max_drafts; ++draft) {
            int max_global_tokens = 0;
            for (int i = 0; i < S; ++i) {
                const std::uint32_t input_pos =
                    mtp_input_pos[static_cast<std::size_t>(i)] +
                    static_cast<std::uint32_t>(draft);
                const std::size_t flat =
                    static_cast<std::size_t>(draft) * s_sz +
                    static_cast<std::size_t>(i);
                mtp_positions_flat[flat] = input_pos;
                mtp_rows_flat[flat] =
                    draft == 0
                        ? mtp_rows[static_cast<std::size_t>(i)]
                        : chained_rows[static_cast<std::size_t>(i)];
                if (mtp_global_cache) {
                    const int global_tokens = std::max(
                        0,
                        static_cast<int>(input_pos) -
                            (prefix_global ? draft : 0));
                    max_global_tokens = std::max(
                        max_global_tokens, global_tokens);
                }
            }
            mtp_max_global_tokens[static_cast<std::size_t>(draft)] =
                max_global_tokens;
            mtp_max_observed_global_tokens = std::max(
                mtp_max_observed_global_tokens, max_global_tokens);
        }

        pi.tokens.copy_from_host(std::span<const std::uint32_t>(mtp_tokens));
        pi.positions.copy_from_host(
            std::span<const std::uint32_t>(mtp_positions_flat));
        pi.sample_idx.copy_from_host(
            std::span<const std::int32_t>(mtp_rows_flat));
        pi.mtp_request_ids.copy_from_host(
            std::span<const std::int32_t>(mtp_request_ids));

        if (!try_run_mtp_chain_graph_with_argmax(
                executor, S, max_drafts, mtp_max_observed_global_tokens)) {
            for (int draft = 0; draft < max_drafts; ++draft) {
                const std::size_t offset =
                    static_cast<std::size_t>(draft) * s_sz;
                run_mtp_draft_with_argmax(
                    executor,
                    reinterpret_cast<const std::int32_t*>(
                        pi.tokens.data() + offset),
                    reinterpret_cast<const std::int32_t*>(
                        pi.positions.data() + offset),
                    pi.sample_idx.data() + offset,
                    pi.mtp_request_ids.data(),
                    S, draft,
                    mtp_max_global_tokens[static_cast<std::size_t>(draft)]);
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.tokens.data() + offset + s_sz, pi.sampled.data(),
                    sizeof(std::uint32_t) * s_sz,
                    cudaMemcpyDeviceToDevice, cublas.stream()));
            }
        }

        std::vector<std::uint32_t> mtp_sampled_flat(s_sz * drafts_sz);
        CUDA_CHECK(cudaMemcpyAsync(
            mtp_sampled_flat.data(), pi.tokens.data() + s_sz,
            sizeof(std::uint32_t) * s_sz * drafts_sz,
            cudaMemcpyDeviceToHost, cublas.stream()));
        mtp_timer.finish(cublas.stream());
        CUDA_CHECK(cudaStreamSynchronize(cublas.stream()));

        for (int draft = 0; draft < max_drafts; ++draft) {
            for (int i = 0; i < S; ++i) {
                auto& out = per_req[static_cast<std::size_t>(
                    mtp_requests[static_cast<std::size_t>(i)])];
                const std::size_t flat =
                    static_cast<std::size_t>(draft) * s_sz +
                    static_cast<std::size_t>(i);
                out.spec_tokens.push_back(mtp_sampled_flat[flat]);
                out.spec_positions.push_back(
                    mtp_output_pos[static_cast<std::size_t>(i)] +
                    static_cast<std::uint32_t>(draft));
            }
        }
        return;
    }

    std::vector<std::int32_t> mtp_sampled(static_cast<std::size_t>(S));
    for (int draft = 0; draft < max_drafts; ++draft) {
        int max_global_tokens = 0;
        if (mtp_global_cache) {
            for (auto pos : mtp_input_pos) {
                const int global_tokens =
                    std::max(0,
                             static_cast<int>(pos) -
                                 (prefix_global ? draft : 0));
                max_global_tokens =
                    std::max(max_global_tokens, global_tokens);
            }
        }
        pi.tokens.copy_from_host(std::span<const std::uint32_t>(mtp_tokens));
        pi.positions.copy_from_host(
            std::span<const std::uint32_t>(mtp_input_pos));
        pi.sample_idx.copy_from_host(std::span<const std::int32_t>(mtp_rows));
        pi.mtp_request_ids.copy_from_host(
            std::span<const std::int32_t>(mtp_request_ids));
        if (executor.tp_comm != nullptr) {
            tp_cpu_gate_notify(executor.tp_cpu_gate_key);
            tp_broadcast_mtp_inputs(
                *executor.tp_comm, pi, S, draft, /*stream=*/nullptr);
        }
        run_mtp_draft_with_argmax(executor, S, draft, max_global_tokens);
        CUDA_CHECK(cudaMemcpyAsync(
            mtp_sampled.data(), pi.sampled.data(),
            sizeof(std::int32_t) * static_cast<std::size_t>(S),
            cudaMemcpyDeviceToHost, cublas.stream()));
        CUDA_CHECK(cudaStreamSynchronize(cublas.stream()));
        if (mtp_trace_take()) {
            std::cerr << "[pie-mtp-trace] draft_step=" << draft
                      << " rows=" << S << " in_tok=[";
            for (int i = 0; i < S; ++i) {
                if (i) std::cerr << ",";
                std::cerr << mtp_tokens[static_cast<std::size_t>(i)];
            }
            std::cerr << "] in_pos=[";
            for (int i = 0; i < S; ++i) {
                if (i) std::cerr << ",";
                std::cerr << mtp_input_pos[static_cast<std::size_t>(i)];
            }
            std::cerr << "] out_pos=[";
            for (int i = 0; i < S; ++i) {
                if (i) std::cerr << ",";
                std::cerr << mtp_output_pos[static_cast<std::size_t>(i)];
            }
            std::cerr << "] out_tok=[";
            for (int i = 0; i < S; ++i) {
                if (i) std::cerr << ",";
                std::cerr << mtp_sampled[static_cast<std::size_t>(i)];
            }
            std::cerr << "]\n";
        }
        for (int i = 0; i < S; ++i) {
            auto& out = per_req[static_cast<std::size_t>(
                mtp_requests[static_cast<std::size_t>(i)])];
            const auto sampled =
                static_cast<std::uint32_t>(
                    mtp_sampled[static_cast<std::size_t>(i)]);
            out.spec_tokens.push_back(sampled);
            out.spec_positions.push_back(
                mtp_output_pos[static_cast<std::size_t>(i)]);
            mtp_tokens[static_cast<std::size_t>(i)] = sampled;
            mtp_input_pos[static_cast<std::size_t>(i)] += 1u;
            mtp_output_pos[static_cast<std::size_t>(i)] += 1;
        }
        mtp_rows = chained_rows;
    }
}

std::vector<int> forward_graph_request_lattice(int max_requests) {
    std::vector<int> out;
    if (max_requests <= 0) return out;
    for (int r = 1; r <= max_requests; ++r) {
        const int bucket = forward_graph_request_bucket(r, max_requests);
        if (bucket <= 0) continue;
        if (out.empty() || out.back() != bucket) out.push_back(bucket);
        if (bucket == max_requests) break;
        r = bucket;
    }
    return out;
}

cudaGraphExec_t capture_forward_graph_exec(
    Executor& executor,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    bool single_gpu_greedy_argmax,
    bool tp_greedy_argmax)
{
    auto& pi = executor.inputs;

    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    cudaStream_t cstream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
    executor.cublas.set_stream(cstream);
    CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
    {
        pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
        fwd_in.token_ids = reinterpret_cast<const std::int32_t*>(pi.tokens.data());
        fwd_in.positions = reinterpret_cast<const std::int32_t*>(pi.positions.data());
        fwd_in.qo_indptr_d         = pi.qo_indptr.data();
        fwd_in.kv_page_indices_d   = pi.kv_page_indices.data();
        fwd_in.kv_page_indptr_d    = pi.kv_page_indptr.data();
        fwd_in.kv_last_page_lens_d = pi.kv_last_page_lens.data();
        fwd_in.qo_indptr_h         = qo_indptr_h;
        fwd_in.kv_page_indices_h   = kv_page_indices_h;
        fwd_in.kv_page_indptr_h    = kv_page_indptr_h;
        fwd_in.kv_last_page_lens_h = kv_last_page_lens_h;
        fwd_in.total_tokens        = N;
        fwd_in.num_requests        = R;
        fwd_in.is_pure_decode      = is_pure_decode;
        fwd_in.slot_ids_h          = slot_ids_h;
        fwd_in.is_fresh_h          = is_fresh_h;
        fwd_in.slot_ids_d          = slot_ids_d;
        fwd_in.logit_row_indices_d = logit_row_indices_d;
        fwd_in.num_logit_rows      = num_logit_rows;
        fwd_in.tp_greedy_argmax    = tp_greedy_argmax;
        executor.forward_fn.invoke_body(
            executor.ws, executor.kv_cache, executor.attn_ws, executor.cublas,
            fwd_in);
    }
    if (single_gpu_greedy_argmax) {
        const int argmax_parts =
            greedy_argmax_parts(executor.loaded_model.hf_config().vocab_size);
        if (argmax_parts > 1 && !executor.ws.greedy_pairs_all.empty()) {
            kernels::launch_argmax_bf16_partitioned_pairs(
                executor.ws.logits.data(),
                reinterpret_cast<std::uint64_t*>(
                    executor.ws.greedy_pairs_all.data()),
                N, executor.loaded_model.hf_config().vocab_size, argmax_parts,
                cstream);
            kernels::launch_select_global_argmax_pairs(
                reinterpret_cast<const std::uint64_t*>(
                    executor.ws.greedy_pairs_all.data()),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, argmax_parts, cstream);
        } else {
            kernels::launch_argmax_bf16(
                executor.ws.logits.data(),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, executor.loaded_model.hf_config().vocab_size,
                cstream);
        }
    }
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
    if (executor.tp_comm != nullptr &&
        executor.tp_comm->custom_all_reduce() != nullptr) {
        executor.tp_comm->custom_all_reduce()
            ->register_graph_buffers(*executor.tp_comm);
    }
    executor.cublas.set_stream(nullptr);

    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(exec, nullptr));
    cudaGraphDestroy(graph);
    cudaStreamDestroy(cstream);
    return exec;
}

float bf16_to_float(std::uint16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int tensor_rows(const DeviceTensor& t) {
    if (t.shape().empty()) return 0;
    return static_cast<int>(t.shape()[0]);
}

std::int32_t* sampled_pinned_buf(std::size_t want_elems) {
    static std::int32_t* buf = nullptr;
    static std::size_t buf_capacity = 0;
    if (want_elems > buf_capacity) {
        if (buf) cudaFreeHost(buf);
        CUDA_CHECK(cudaMallocHost(&buf, want_elems * sizeof(std::int32_t)));
        buf_capacity = want_elems;
    }
    return buf;
}

std::int32_t masked_argmax_bf16(
    const std::uint16_t* row,
    int vocab_size,
    std::span<const std::uint32_t> mask_runs)
{
    bool allow = false;
    std::uint64_t pos = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    std::int32_t best_idx = -1;

    for (std::size_t i = 0; i < mask_runs.size() && pos < static_cast<std::uint64_t>(vocab_size);
         ++i) {
        const std::uint64_t run_len = mask_runs[i];
        const std::uint64_t end =
            std::min<std::uint64_t>(pos + run_len, static_cast<std::uint64_t>(vocab_size));
        if (allow) {
            for (std::uint64_t j = pos; j < end; ++j) {
                const auto idx = static_cast<std::int32_t>(j);
                const float val = bf16_to_float(row[idx]);
                if (val > best_val || (val == best_val && idx < best_idx)) {
                    best_val = val;
                    best_idx = idx;
                }
            }
        }
        pos = end;
        allow = !allow;
    }

    return best_idx;
}

void apply_logit_mask_overrides(
    model::Qwen3Workspace& ws,
    std::vector<std::int32_t>& all_sampled,
    std::span<const std::uint32_t> logit_masks,
    std::span<const std::uint32_t> logit_mask_indptr,
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> sampling_indptr,
    std::span<const std::uint32_t> sampling_indices,
    std::span<const std::uint32_t> per_slot_type,
    int R,
    int N,
    int vocab_size)
{
    if (logit_masks.empty()) return;
    if (logit_mask_indptr.size() < static_cast<std::size_t>(R + 1)) return;
    if (qo_indptr.size() < static_cast<std::size_t>(R + 1)) return;
    if (sampling_indptr.size() < static_cast<std::size_t>(R + 1)) return;

    const auto* logits_u16 = static_cast<const std::uint16_t*>(ws.logits.data());
    std::vector<std::uint16_t> row_bf16(static_cast<std::size_t>(vocab_size));

    for (int r = 0; r < R; ++r) {
        const std::uint32_t mask_lo = logit_mask_indptr[r];
        const std::uint32_t mask_hi = logit_mask_indptr[r + 1];
        if (mask_hi <= mask_lo || mask_hi > logit_masks.size()) continue;

        const auto runs = logit_masks.subspan(mask_lo, mask_hi - mask_lo);
        const std::uint32_t qo_lo = qo_indptr[r];
        for (std::uint32_t k = sampling_indptr[r]; k < sampling_indptr[r + 1]; ++k) {
            if (k >= per_slot_type.size() || !is_token_sampler(per_slot_type[k])) continue;
            const std::uint32_t row = qo_lo + sampling_indices[k];
            if (row >= static_cast<std::uint32_t>(N)) continue;

            CUDA_CHECK(cudaMemcpy(row_bf16.data(),
                                  logits_u16 + static_cast<long long>(row) * vocab_size,
                                  sizeof(std::uint16_t) * static_cast<std::size_t>(vocab_size),
                                  cudaMemcpyDeviceToHost));
            const std::int32_t masked = masked_argmax_bf16(
                row_bf16.data(), vocab_size, runs);
            if (masked >= 0) {
                all_sampled[row] = masked;
            }
        }
    }
}

std::shared_ptr<TpCpuGate> tp_cpu_gate_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_cpu_gates_mu);
    auto& gate = g_tp_cpu_gates[key];
    if (!gate) gate = std::make_shared<TpCpuGate>();
    return gate;
}

void tp_cpu_gate_notify(const std::string& key) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    gate->seq.fetch_add(1, std::memory_order_release);
    gate->cv.notify_all();
}

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

void tp_cpu_gate_wait(const std::string& key,
                      std::uint64_t& seen,
                      std::atomic<bool>& stop) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    constexpr auto spin_budget = std::chrono::microseconds(2000);
    const auto start = std::chrono::steady_clock::now();
    while (!stop.load(std::memory_order_relaxed)) {
        const std::uint64_t seq = gate->seq.load(std::memory_order_acquire);
        if (seq != seen) {
            seen = seq;
            return;
        }
        if (std::chrono::steady_clock::now() - start >= spin_budget) break;
        cpu_relax();
    }

    std::unique_lock<std::mutex> lk(gate->mu);
    gate->cv.wait(lk, [&] {
        return stop.load(std::memory_order_relaxed) ||
               gate->seq.load(std::memory_order_acquire) != seen;
    });
    seen = gate->seq.load(std::memory_order_acquire);
}

// Broadcast header sent from rank 0 → followers before each fire's
// per-fire payload. Followers parse it to size the subsequent broadcasts
// + the forward call. Two magic values:
//
//   * TP_FIRE_MAGIC: a regular fire is incoming; payload broadcasts follow.
//   * TP_MTP_MAGIC: rank 0 has accepted tokens and wants followers to run
//                   the tensor-parallel MTP head against their local shards.
//   * TP_STOP_MAGIC: shutdown sentinel; follower exits its serve loop.
//
// Sized at exactly 8 i32 so we can broadcast it as `8 * sizeof(int32_t)`
// bytes without alignment surprises across compilers.
struct TpFireHeader {
    std::int32_t magic;
    std::int32_t total_tokens;
    std::int32_t num_requests;
    std::int32_t is_pure_decode;
    std::int32_t kv_indices_count;
    std::int32_t mask_bytes;
    std::int32_t mask_indptr_count;
    // 1 = slot_ids[R] (int32) and is_fresh[R] (uint8) follow the
    // existing payload broadcasts. Inert (0) for archs that don't use
    // rs_cache — followers skip those broadcasts.
    std::int32_t has_slot_ids;
    // 1 = llama-like TP greedy decode fast path. Followers use this to
    // capture/replay the same forward variant as rank 0.
    std::int32_t tp_greedy_argmax;
    // Number of compact logit rows in pi.sample_idx.
    std::int32_t logit_rows;
};
static_assert(sizeof(TpFireHeader) == 10 * sizeof(std::int32_t),
              "TpFireHeader must pack into exactly 10 ints");
constexpr std::int32_t TP_FIRE_MAGIC = 0x55504954;  // 'TPIU' tag
constexpr std::int32_t TP_MTP_MAGIC  = 0x50544D4D;  // 'MMTP' tag
constexpr std::int32_t TP_STOP_MAGIC = 0x504F5453;  // 'STOP' tag

// Lazily-allocated 32-byte device buffer holding the broadcast header.
// Both rank 0 and followers reuse it across fires; no need to plumb it
// through Executor.
std::int32_t* tp_hdr_dev_buf() {
    thread_local std::int32_t* buf = nullptr;
    if (buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf, sizeof(TpFireHeader)));
    }
    return buf;
}

// Issue every per-fire broadcast in dependency order. Caller has already
// refilled `pi.*` with the current fire's data; this just fans them out.
// All ops run on the default stream so they sequence correctly with the
// kernels that follow inside `forward_fn.body`.
void tp_broadcast_inputs(NcclComm& comm, PersistentInputs& pi,
                         int N, int R, bool is_pure_decode,
                         int kv_indices_count,
                         int mask_bytes, int mask_indptr_count,
                         bool has_slot_ids,
                         bool tp_greedy_argmax,
                         int logit_rows,
                         cudaStream_t stream)
{
    auto* d_hdr = tp_hdr_dev_buf();
    TpFireHeader hdr{
        TP_FIRE_MAGIC, N, R, is_pure_decode ? 1 : 0,
        kv_indices_count, mask_bytes, mask_indptr_count,
        has_slot_ids ? 1 : 0,
        tp_greedy_argmax ? 1 : 0,
        logit_rows,
    };
    // Header goes first (synchronous from the followers' POV — they need
    // to parse sizes before posting matching payload broadcasts).
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                                   comm.comm(), stream),
                     comm.comm());
    // Group the payload broadcasts so NCCL submits them as a single batch
    // — tens of microseconds of host-side launch overhead saved per fire,
    // most visible at small batch sizes (decode where each broadcast is
    // sub-KB but the fixed per-op cost dominates).
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                             static_cast<std::size_t>(N) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                             static_cast<std::size_t>(N) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    if (R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                 pi.kv_last_page_lens.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
    }
    if (kv_indices_count > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                 pi.kv_page_indices.data(),
                                 static_cast<std::size_t>(kv_indices_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (mask_bytes > 0) {
        NCCL_CHECK(ncclBroadcast(pi.custom_mask.data(),
                                 pi.custom_mask.data(),
                                 static_cast<std::size_t>(mask_bytes), ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                 pi.custom_mask_indptr.data(),
                                 static_cast<std::size_t>(mask_indptr_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (has_slot_ids && R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                 static_cast<std::size_t>(R), ncclChar, 0,
                                 comm.comm(), stream));
    }
    if (logit_rows > 0) {
        NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                 static_cast<std::size_t>(logit_rows) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

void tp_broadcast_mtp_inputs(NcclComm& comm, PersistentInputs& pi,
                             int num_tokens, int draft_step,
                             cudaStream_t stream)
{
    auto* d_hdr = tp_hdr_dev_buf();
    TpFireHeader hdr{
        TP_MTP_MAGIC, num_tokens, draft_step, 0, 0, 0, 0, 0, 0, 0,
    };
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                                   comm.comm(), stream),
                     comm.comm());
    NCCL_CHECK(ncclGroupStart());
    if (num_tokens > 0) {
        NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                 static_cast<std::size_t>(num_tokens) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                 static_cast<std::size_t>(num_tokens) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                 static_cast<std::size_t>(num_tokens) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.mtp_request_ids.data(),
                                 pi.mtp_request_ids.data(),
                                 static_cast<std::size_t>(num_tokens) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

}  // namespace

struct ForwardInputViews {
    std::span<const std::uint32_t> tokens;
    std::span<const std::uint32_t> positions;
    std::span<const std::uint32_t> qo_indptr;
    std::span<const std::uint32_t> kv_page_indices;
    std::span<const std::uint32_t> kv_page_indptr;
    std::span<const std::uint32_t> kv_last_page_lens;
    int total_tokens = 0;
    int num_requests = 0;
    bool padded = false;

    std::vector<std::uint32_t> tokens_storage;
    std::vector<std::uint32_t> positions_storage;
    std::vector<std::uint32_t> qo_indptr_storage;
    std::vector<std::uint32_t> kv_page_indices_storage;
    std::vector<std::uint32_t> kv_page_indptr_storage;
    std::vector<std::uint32_t> kv_last_page_lens_storage;
};

ForwardInputViews make_forward_input_views(
    std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> kv_page_indices,
    std::span<const std::uint32_t> kv_page_indptr,
    std::span<const std::uint32_t> kv_last_page_lens,
    int real_requests,
    int graph_requests,
    int graph_pad_page)
{
    ForwardInputViews out{
        tokens,
        positions,
        qo_indptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        static_cast<int>(tokens.size()),
        real_requests,
        false,
    };

    if (graph_requests <= real_requests) return out;

    const int pad = graph_requests - real_requests;
    out.tokens_storage.assign(tokens.begin(), tokens.end());
    out.positions_storage.assign(positions.begin(), positions.end());
    out.qo_indptr_storage.assign(qo_indptr.begin(), qo_indptr.end());
    out.kv_page_indices_storage.assign(kv_page_indices.begin(),
                                       kv_page_indices.end());
    out.kv_page_indptr_storage.assign(kv_page_indptr.begin(),
                                      kv_page_indptr.end());
    out.kv_last_page_lens_storage.assign(kv_last_page_lens.begin(),
                                         kv_last_page_lens.end());

    for (int i = 0; i < pad; ++i) {
        out.tokens_storage.push_back(0);
        out.positions_storage.push_back(static_cast<std::uint32_t>(i));
        out.qo_indptr_storage.push_back(
            out.qo_indptr_storage.back() + 1u);
        out.kv_page_indices_storage.push_back(
            static_cast<std::uint32_t>(graph_pad_page));
        out.kv_page_indptr_storage.push_back(
            static_cast<std::uint32_t>(out.kv_page_indices_storage.size()));
        out.kv_last_page_lens_storage.push_back(
            static_cast<std::uint32_t>(i + 1));
    }

    out.tokens = out.tokens_storage;
    out.positions = out.positions_storage;
    out.qo_indptr = out.qo_indptr_storage;
    out.kv_page_indices = out.kv_page_indices_storage;
    out.kv_page_indptr = out.kv_page_indptr_storage;
    out.kv_last_page_lens = out.kv_last_page_lens_storage;
    out.total_tokens = graph_requests;
    out.num_requests = graph_requests;
    out.padded = true;
    return out;
}

std::size_t capture_forward_graph_lattice(Executor& executor) {
    if (executor.graph_cache == nullptr) return 0;
    if (!executor.forward_fn.graph_safe) return 0;
    if (executor.forward_fn.model == nullptr) return 0;
    if (executor.loaded_model.hf_config().model_type == "nemotron_h") {
        // Nemotron-H has recurrent Mamba state in addition to attention state.
        // Synthetic upfront capture replays incorrectly; first-use capture with
        // real slot/page metadata is correct, so leave the cache cold here.
        return 0;
    }
    const char* disable_upfront = std::getenv("PIE_CUDA_DISABLE_UPFRONT_GRAPHS");
    if (disable_upfront != nullptr && disable_upfront[0] != '\0' &&
        disable_upfront[0] != '0') {
        return 0;
    }
    const int max_requests =
        std::min(executor.max_forward_requests, executor.max_workspace_tokens);
    if (max_requests <= 0) return 0;

    auto buckets = forward_graph_request_lattice(max_requests);
    if (buckets.empty()) return 0;

    auto& pi = executor.inputs;
    const int num_pages = std::max(1, executor.kv_cache.num_pages());
    std::size_t captured = 0;
    const bool tp_greedy_argmax =
        executor.tp_comm != nullptr &&
        executor.tp_comm->world_size() <= 8 &&
        executor.forward_fn.supports_tp_greedy_argmax;
    const bool fwd_handles_argmax_precapture =
        executor.forward_fn.supports_fused_lmhead_argmax &&
        executor.tp_comm == nullptr;
    const bool single_gpu_graph_argmax =
        graph_single_gpu_argmax_enabled() && executor.tp_comm == nullptr &&
        !fwd_handles_argmax_precapture;
    const bool log_rank =
        executor.verbose &&
        (executor.tp_comm == nullptr || executor.tp_comm->rank() == 0);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    if (log_rank) {
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));
    }

    for (int R : buckets) {
        const int N = R;
        std::vector<std::uint32_t> tokens(static_cast<std::size_t>(N), 0u);
        std::vector<std::uint32_t> positions(static_cast<std::size_t>(N), 0u);
        std::vector<std::uint32_t> qo(static_cast<std::size_t>(R) + 1);
        std::vector<std::uint32_t> kvpp(static_cast<std::size_t>(R) + 1);
        std::vector<std::uint32_t> kvlpl(static_cast<std::size_t>(R), 1u);
        std::vector<std::uint32_t> kvpi(static_cast<std::size_t>(R));
        std::vector<std::int32_t> slot_ids;

        for (int r = 0; r <= R; ++r) {
            qo[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
            kvpp[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
        }
        for (int r = 0; r < R; ++r) {
            kvpi[static_cast<std::size_t>(r)] =
                static_cast<std::uint32_t>(r % num_pages);
        }
        if (executor.rs_cache != nullptr) {
            slot_ids.resize(static_cast<std::size_t>(R));
            for (int r = 0; r < R; ++r) {
                slot_ids[static_cast<std::size_t>(r)] =
                    static_cast<std::int32_t>(r);
            }
        }

        pi.tokens.copy_from_host(std::span<const std::uint32_t>(tokens));
        pi.positions.copy_from_host(std::span<const std::uint32_t>(positions));
        pi.qo_indptr.copy_from_host(std::span<const std::uint32_t>(qo));
        pi.kv_page_indices.copy_from_host(std::span<const std::uint32_t>(kvpi));
        pi.kv_page_indptr.copy_from_host(std::span<const std::uint32_t>(kvpp));
        pi.kv_last_page_lens.copy_from_host(std::span<const std::uint32_t>(kvlpl));
        if (!slot_ids.empty()) {
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids));
        }

        executor.forward_fn.invoke_prepare(
            executor.attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = qo.data(),
                .kv_page_indices_h = kvpi.data(),
                .kv_page_indices_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
                .kv_page_indptr_h = kvpp.data(),
                .kv_page_indptr_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
                .kv_last_page_lens_h = kvlpl.data(),
                .kv_last_page_lens_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
                .total_tokens = N,
                .num_requests = R,
                .is_pure_decode = true,
            });
        const std::uint32_t graph_layout =
            executor.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            (tp_greedy_argmax ? 1u : 0u) |
            (single_gpu_graph_argmax ? 2u : 0u) |
            (fwd_handles_argmax_precapture ? 4u : 0u) |
            (graph_layout << 3);
        const ForwardGraphKey key{R, N, graph_variant};
        if (executor.graph_cache->get(key) != nullptr) continue;

        if (fwd_handles_argmax_precapture) {
            executor.forward_fn.invoke_set_logits_argmax_only(true);
            executor.forward_fn.invoke_set_fused_argmax_output(
                reinterpret_cast<std::int32_t*>(pi.sampled.data()));
        }
        tp_graph_capture_barrier(executor);
        cudaGraphExec_t exec = capture_forward_graph_exec(
            executor, qo.data(), kvpi.data(), kvpp.data(), kvlpl.data(),
            N, R, /*is_pure_decode=*/true,
            /*slot_ids_h=*/nullptr, /*is_fresh_h=*/nullptr,
            executor.rs_cache != nullptr ? pi.slot_ids.data() : nullptr,
            /*logit_row_indices_d=*/nullptr,
            /*num_logit_rows=*/0, single_gpu_graph_argmax,
            tp_greedy_argmax);
        if (fwd_handles_argmax_precapture) {
            executor.forward_fn.invoke_set_fused_argmax_output(nullptr);
        }
        executor.graph_cache->put(key, exec);
        ++captured;
        tp_graph_capture_barrier(executor);
    }

    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    if (log_rank) {
        std::size_t free_after = 0;
        std::size_t total_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
        const std::size_t graph_bytes =
            free_before > free_after ? (free_before - free_after) : 0;
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[pie-driver-cuda] CUDA graph upfront capture: "
                  << captured << " decode graphs"
                  << " (cache size=" << executor.graph_cache->size()
                  << ", graph_mem~" << (graph_bytes / (1024 * 1024))
                  << " MiB"
                  << ", " << dt << " ms)\n";
    }
    return captured;
}

struct GraphShapeDecision {
    bool small_spec_graph_shape = false;
    bool graph_shape_ok = false;
    int graph_requests = 0;
};

// Sliced wire-views the sample-plan builder consumes. Bundled so the
// builder's signature stays manageable (12+ spans otherwise).
struct SamplePlanInputs {
    std::span<const std::uint8_t>  outspec_view;
    std::span<const std::uint32_t> sptr_view;
    std::span<const std::uint32_t> sidx_view;
    std::span<const std::uint32_t> rns_view;
    std::span<const std::uint32_t> types_view;
    std::span<const float>         temp_view;
    std::span<const std::uint32_t> top_k_view;
    std::span<const float>         top_p_view;
    std::span<const float>         min_p_view;
    std::span<const std::uint32_t> seed_view;
    std::span<const std::uint32_t> logit_masks_view;
    const std::uint32_t* h_qo = nullptr;
    int N = 0;
    int R = 0;
    int num_sampling = 0;
    bool is_pure_decode = false;
    bool have_custom_mask = false;
    bool has_spec_drafts = false;
    bool has_custom_program = false;
    // #12 phase-1: a standard sampler program recognized from the (migrated)
    // program contract. When set, the program's dedicated-kernel params have been
    // extracted into the fields below; the per-slot loop seeds `per_slot_*` from
    // them (the program carries no legacy slot params) so the flag-set + dispatch
    // route the fire to the SAME kernel a slot fire would (FlashInfer top-p /
    // sample_temp min-p / greedy argmax / BakedIR temp) instead of falling through
    // to CustomJIT. An unrecognized program keeps has_custom_program=true →
    // CustomJIT. Seed stays ambient (`pi.sample_seed`; seed_view empty →
    // fresh_sampling_seed, exactly as the pre-migration multisamp slot fire).
    bool program_recognized = false;
    float rec_temp = 1.0f;
    float rec_top_p = 1.0f;
    float rec_min_p = 0.0f;
    std::int32_t rec_top_k = 0;
};

// Per-fire decisions emitted by the sample-plan phase. The spans on
// `sampling_plan` point into the thread-local `sample_scratch()`
// vectors, which the builder fills as a side effect; caller must keep
// `sample_scratch()` alive for the rest of the fire (i.e. don't recurse).
struct SamplePlanResult {
    bool has_msgpack_only_slots = false;
    bool has_rich_sampler_slots = false;
    bool need_msgpack = false;
    bool any_topk_topp = false;
    bool sample_rows_are_dense = false;
    bool all_rows_greedy = false;
    bool all_slots_token = false;
    bool compact_logit_rows = false;
    bool tp_greedy_argmax = false;
    bool single_gpu_greedy_argmax = false;
    // De-hardwiring (Task #4 / WS5 #7, env-gated PIE_DEHARDWIRE_STD_SAMPLERS):
    // a fire whose every sampling row recognizes to a BakedIR kind (per the
    // dispatch scorecard — temperature today) is routed through the driver-baked
    // IR program over the full [N,V] block instead of the legacy `sample_temp`.
    // Restricted to the dense pure-decode case (contiguous sampling rows, the
    // M=1-decode MVP); mixed/compact fires fall to the legacy ladder. The IR
    // path falls back to the legacy sampler on any non-Handled status.
    bool dehardwire_baked_ir = false;
    sampling_ir::StandardSamplerKind dehardwire_kind =
        sampling_ir::StandardSamplerKind::Temperature;
    int logit_rows_required = 0;
    int prob_rows_required = 0;
};

// Inputs to the forward-dispatch phase. Built once at the call site
// and passed by-ref so the dispatcher can pick between graph replay
// and a direct `forward_fn.body` call without a 20-arg signature.
struct ForwardDispatchInputs {
    int R = 0;
    int forward_R = 0;
    int forward_N = 0;
    int N = 0;
    int num_sampling = 0;
    bool is_pure_decode = false;
    bool have_custom_mask = false;
    bool small_spec_graph_shape = false;
    bool graph_shape_ok = false;
    bool tp_greedy_argmax = false;
    bool forward_handles_argmax = false;
    bool single_gpu_greedy_argmax = false;
    bool compact_logit_rows = false;
    bool use_slots = false;
    bool padded = false;
    const std::uint32_t* h_qo_forward = nullptr;
    const std::uint32_t* h_kvpi_forward = nullptr;
    const std::uint32_t* h_kvpp_forward = nullptr;
    const std::uint32_t* h_kvlpl_forward = nullptr;
    const std::int32_t*  slot_ids_h_data = nullptr;
    const std::uint8_t*  is_fresh_h_data = nullptr;
    // Ph7 RS rs-output (W10): when rs_buffer_write, the linear layers scatter
    // their in-proj [mixed_qkv|a|b] page-major into the buffered-activation pool
    // at these per-request CSR slabs (write_state forced false). FOLD passes use
    // the separate fold-replay dispatch instead (not this path).
    const std::uint32_t* rs_buffer_slot_ids_h = nullptr;
    const std::uint32_t* rs_buffer_slot_indptr_h = nullptr;
    bool                 rs_buffer_write = false;
    // Multimodal (gemma4 vision): image side-channel, set from the view.
    const float*         image_pixels_h = nullptr;
    const std::uint32_t* image_pixel_byte_indptr_h = nullptr;
    const std::uint32_t* image_patch_positions_h = nullptr;
    const std::uint32_t* image_anchor_rows_h = nullptr;
    int                  num_images = 0;
    // Qwen3-VL: per-image (t,h,w) grids and the assembled per-token M-RoPE
    // 3-axis positions for the whole batch.
    const std::uint32_t* image_grids_h = nullptr;
    const std::uint32_t* mrope_positions_h = nullptr;
    int                  num_mrope_positions = 0;
    // Multimodal (gemma4 audio): log-mel side-channel, set from the view.
    const float*         audio_features_h = nullptr;
    const std::uint32_t* audio_feature_byte_indptr_h = nullptr;
    const std::uint32_t* audio_anchor_rows_h = nullptr;
    int                  num_clips = 0;
};

struct ForwardDispatchResult {
    std::uint32_t graph_layout = 0;
    bool graph_captures_single_gpu_argmax = false;
};

// Decide whether the active fire can replay a captured CUDA graph and,
// if so, what bucket size to request. Two paths qualify: pure-decode
// (qo_len==1 every request) and small-prefill speculative verification.
// Caller passes the shape inputs already gathered at the top of the
// fire; the planner-pad fields come from `executor`.
GraphShapeDecision decide_graph_shape(
    const Executor& executor,
    int R, int N, int page_size,
    bool is_pure_decode,
    bool has_spec_drafts,
    bool has_fresh_slot,
    std::size_t kvpi_view_size,
    std::size_t pi_kv_page_indices_size)
{
    const int small_spec_graph_tokens = qwen35_small_spec_graph_tokens();
    const int small_spec_graph_requests = small_spec_graph_max_requests();
    const int small_spec_graph_min_r =
        small_spec_graph_min_requests(small_spec_graph_requests);

    const bool pure_decode_graph_shape =
        executor.graph_cache != nullptr && is_pure_decode &&
        executor.forward_fn.graph_safe && !has_fresh_slot;
    const bool small_spec_graph_shape =
        executor.graph_cache != nullptr &&
        executor.tp_comm == nullptr &&
        !is_pure_decode &&
        has_spec_drafts &&
        executor.forward_fn.graph_safe &&
        executor.forward_fn.supports_small_prefill_graph &&
        small_spec_graph_tokens > 0 &&
        R >= small_spec_graph_min_r &&
        R <= small_spec_graph_requests &&
        N <= small_spec_graph_tokens &&
        executor.kv_cache.format().is_native_bf16() &&
        !executor.kv_cache.hnd_layout() &&
        !has_fresh_slot;

    GraphShapeDecision out;
    out.small_spec_graph_shape = small_spec_graph_shape;
    out.graph_shape_ok = pure_decode_graph_shape || small_spec_graph_shape;
    out.graph_requests = R;
    if (pure_decode_graph_shape) {
        const int bucket =
            forward_graph_request_bucket(R, executor.max_forward_requests);
        const int pad = bucket - R;
        const bool exact_bucket = (bucket == R);
        const bool can_pad =
            bucket > R &&
            executor.graph_pad_page >= 0 &&
            (executor.rs_cache == nullptr ||
             executor.graph_pad_slot >= 0) &&
            bucket <= executor.max_workspace_tokens &&
            pad <= page_size &&
            kvpi_view_size + static_cast<std::size_t>(pad) <=
                pi_kv_page_indices_size;
        out.graph_shape_ok = bucket > 0 && (exact_bucket || can_pad);
        if (out.graph_shape_ok) out.graph_requests = bucket;
    }
    return out;
}

// Hoisted sample-plan builder. Mirrors the historical "Sample-plan
// construction" block — fills the thread-local `sample_scratch()` with
// per-row/per-slot sampler params and emits the dispatch flags
// (compact-logit / TP greedy / single-GPU greedy / etc.) so the
// downstream forward + sampling launch can take the fast path when
// the request mix permits.
SamplePlanResult build_sample_plan(
    const Executor& executor,
    const SamplePlanInputs& in)
{
    SamplePlanResult out;
    for (auto t : in.types_view) {
        if (pie_cuda_driver::is_msgpack_only(t)) {
            out.has_msgpack_only_slots = true;
            out.has_rich_sampler_slots = true;
            break;
        }
    }
    out.need_msgpack = out.has_msgpack_only_slots;
    if (!out.need_msgpack) {
        for (auto f : in.outspec_view) {
            if (f) { out.need_msgpack = true; break; }
        }
    }
    if (in.has_spec_drafts) out.need_msgpack = true;

    const std::uint32_t* h_sptr  = in.sptr_view.data();
    const std::uint32_t* h_sidx  = in.sidx_view.data();
    const std::uint32_t* h_rns   = in.rns_view.data();
    const float*         h_temp  = in.temp_view.data();
    const std::uint32_t* h_top_k = in.top_k_view.data();
    const float*         h_top_p = in.top_p_view.data();
    const float*         h_min_p = in.min_p_view.data();
    const std::uint32_t* h_seed  = in.seed_view.data();

    auto& sample_scratch = sampling_scratch();
    auto& h_per_temp = sample_scratch.h_per_temp;
    auto& h_per_min_p = sample_scratch.h_per_min_p;
    auto& h_per_top_p = sample_scratch.h_per_top_p;
    auto& h_per_top_k = sample_scratch.h_per_top_k;
    auto& h_per_seed = sample_scratch.h_per_seed;
    auto& per_slot_type = sample_scratch.per_slot_type;
    auto& per_slot_temp = sample_scratch.per_slot_temp;
    auto& per_slot_top_p = sample_scratch.per_slot_top_p;
    auto& per_slot_min_p = sample_scratch.per_slot_min_p;
    auto& per_slot_top_k = sample_scratch.per_slot_top_k;
    auto& per_slot_seed = sample_scratch.per_slot_seed;
    auto& h_sample_idx = sample_scratch.h_sample_idx;

    const int R = in.R;
    const int N = in.N;
    const int num_sampling = in.num_sampling;

    bool fast_dense_greedy_argmax =
        executor.tp_comm == nullptr &&
        !in.have_custom_mask &&
        !out.need_msgpack &&
        !in.has_spec_drafts &&
        in.logit_masks_view.empty() &&
        in.is_pure_decode &&
        N == R &&
        num_sampling == R &&
        in.sptr_view.size() == static_cast<std::size_t>(R + 1) &&
        in.sidx_view.size() == static_cast<std::size_t>(R) &&
        in.rns_view.size() >= static_cast<std::size_t>(R) &&
        in.types_view.size() >= static_cast<std::size_t>(R) &&
        in.temp_view.size() >= static_cast<std::size_t>(R);
    if (fast_dense_greedy_argmax) {
        for (int r = 0; r < R; ++r) {
            if (h_sptr[r] != static_cast<std::uint32_t>(r) ||
                h_sptr[r + 1] != static_cast<std::uint32_t>(r + 1) ||
                h_sidx[r] != 0u ||
                h_rns[r] != 1u ||
                !pie_cuda_driver::is_token_sampler(in.types_view[r]) ||
                h_temp[r] > 0.f) {
                fast_dense_greedy_argmax = false;
                break;
            }
        }
    }

    out.sample_rows_are_dense = fast_dense_greedy_argmax;
    out.all_slots_token = fast_dense_greedy_argmax;
    if (!fast_dense_greedy_argmax) {
        h_per_temp.assign(static_cast<std::size_t>(N), 0.f);
        h_per_min_p.assign(static_cast<std::size_t>(N), 0.f);
        h_per_top_p.assign(static_cast<std::size_t>(N), 1.f);
        h_per_top_k.assign(static_cast<std::size_t>(N), 0);
        h_per_seed.assign(static_cast<std::size_t>(N), 0u);

        per_slot_type.assign(static_cast<std::size_t>(num_sampling), 1u);
        per_slot_temp.assign(static_cast<std::size_t>(num_sampling), 0.f);
        per_slot_top_p.assign(static_cast<std::size_t>(num_sampling), 1.f);
        per_slot_min_p.assign(static_cast<std::size_t>(num_sampling), 0.f);
        per_slot_top_k.assign(static_cast<std::size_t>(num_sampling), 0);
        per_slot_seed.assign(static_cast<std::size_t>(num_sampling), 0u);

        std::uint32_t sampler_off = 0;
        for (int r = 0; r < R; ++r) {
            const std::uint32_t ns =
                (in.rns_view.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
            const std::uint32_t lo = h_sptr[r];
            const std::uint32_t hi = h_sptr[r + 1];
            const std::uint32_t qo_lo = in.h_qo[r];
            for (std::uint32_t k = lo; k < hi; ++k) {
                const std::uint32_t s_idx = sampler_off + (k - lo);
                const std::uint32_t type =
                    (s_idx < in.types_view.size()) ? in.types_view[s_idx] : 1u;
                per_slot_type[k] = type;
                const float T = in.program_recognized ? in.rec_temp
                    : (s_idx < in.temp_view.size()) ? h_temp[s_idx] : 1.f;
                float Tp = in.program_recognized ? in.rec_top_p
                    : (s_idx < in.top_p_view.size()) ? h_top_p[s_idx] : 1.f;
                // Defensive (#7 boundary): valid top_p ∈ (0,1]; "no top-p" = 1.0
                // (the driver default — confirmed: a non-top-p fire reads pi.top_p
                // = 1.0, so top_p<=0 never occurs in prod). A degenerate top_p<=0
                // (adversarial / full-param-space) clamps to no-filter (1.0),
                // matching the recognizer's `0<top_p<1` so the dispatch agreement
                // holds on every input incl. hotel's boundary fires. Never fires
                // on a real fire; the recognizer (infer_kind) is left unchanged.
                if (Tp <= 0.f) Tp = 1.f;
                const float Mp = in.program_recognized ? in.rec_min_p
                    : (s_idx < in.min_p_view.size()) ? h_min_p[s_idx] : 0.f;
                const std::int32_t Tk_raw = in.program_recognized ? in.rec_top_k
                    : (s_idx < in.top_k_view.size())
                        ? static_cast<std::int32_t>(h_top_k[s_idx]) : 0;
                const std::int32_t Tk =
                    (Tk_raw == 0) ? executor.loaded_model.hf_config().vocab_size : Tk_raw;
                std::uint32_t s = (s_idx < in.seed_view.size()) ? h_seed[s_idx] : 0u;
                per_slot_temp[k] = T;
                per_slot_top_p[k] = Tp;
                per_slot_min_p[k] = Mp;
                per_slot_top_k[k] = Tk;
                const bool is_token = pie_cuda_driver::is_token_sampler(type);
                if (is_token) {
                    if (T > 0.f && s == 0u) {
                        s = fresh_sampling_seed();
                    }
                    per_slot_seed[k] = s;
                    const std::uint32_t row = qo_lo + h_sidx[k];
                    if (row < static_cast<std::uint32_t>(N)) {
                        h_per_temp[row]  = T;
                        h_per_top_k[row] = Tk;
                        h_per_top_p[row] = Tp;
                        h_per_min_p[row] = Mp;
                        h_per_seed[row]  = s;
                    }
                } else {
                    per_slot_seed[k] = s;
                }
            }
            sampler_off += ns;
        }

        h_sample_idx.assign(static_cast<std::size_t>(num_sampling), 0);
        int k_g = 0;
        for (int r = 0; r < R; ++r) {
            const std::uint32_t qo_lo = in.h_qo[r];
            for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k, ++k_g) {
                h_sample_idx[k_g] =
                    static_cast<std::int32_t>(qo_lo + h_sidx[k]);
            }
        }

        out.sample_rows_are_dense = (num_sampling == N);
        if (out.sample_rows_are_dense) {
            for (int i = 0; i < N; ++i) {
                if (h_sample_idx[i] != i) {
                    out.sample_rows_are_dense = false;
                    break;
                }
            }
        }
        out.all_slots_token = true;
        for (auto type : per_slot_type) {
            if (!pie_cuda_driver::is_token_sampler(type)) {
                out.all_slots_token = false;
                break;
            }
        }
    }

    // #7 (de-hardwiring LIVE): the recognizer is the SOLE dispatch flag source.
    // Per fire, recognize each sampling slot's kind (infer_sampler_kind over the
    // per-slot params) and derive the dispatch flag-set (all_rows_greedy /
    // any_topk_topp) that the ~20 downstream readers consume. The fixed
    // sampler_type→kernel flag computation is DELETED — the recognizer + the
    // dispatch table replace it (verified token-exact across every kind: both
    // safety nets green, §2f re-bench perf-neutral-or-better). On the fast-greedy
    // path per_slot is empty but the fire is all-argmax by construction.
    if (num_sampling > 0) {
        bool all_greedy = true;
        bool any_topk_topp = false;
        if (!fast_dense_greedy_argmax) {
            const std::int32_t vocab =
                executor.loaded_model.hf_config().vocab_size;
            for (int k = 0; k < num_sampling; ++k) {
                const sampling_ir::StandardSamplerKind kind =
                    sampling_ir::infer_sampler_kind(sampling_ir::params_from_slot(
                        per_slot_temp[k], per_slot_top_k[k], per_slot_top_p[k],
                        per_slot_min_p[k], vocab));
                if (kind != sampling_ir::StandardSamplerKind::Argmax)
                    all_greedy = false;
                if (kind == sampling_ir::StandardSamplerKind::TopK ||
                    kind == sampling_ir::StandardSamplerKind::TopP ||
                    kind == sampling_ir::StandardSamplerKind::TopKTopP)
                    any_topk_topp = true;
            }
        }
        out.all_rows_greedy = all_greedy;
        out.any_topk_topp = any_topk_topp;
    }

    out.compact_logit_rows =
        executor.forward_fn.supports_compact_logits &&
        !in.has_custom_program &&  // a custom IR program reads the full [N,V] block
        !in.is_pure_decode &&
        !in.have_custom_mask &&
        !out.has_msgpack_only_slots &&
        !in.has_spec_drafts &&
        in.logit_masks_view.empty() &&
        !out.any_topk_topp &&
        out.all_slots_token &&
        num_sampling > 0 &&
        num_sampling < N;
    out.tp_greedy_argmax =
        executor.tp_comm != nullptr &&
        executor.tp_comm->world_size() <= 8 &&
        executor.forward_fn.supports_tp_greedy_argmax &&
        !in.have_custom_mask &&
        !out.need_msgpack &&
        !in.has_spec_drafts &&
        in.logit_masks_view.empty() &&
        !out.any_topk_topp &&
        out.all_slots_token &&
        out.all_rows_greedy &&
        in.is_pure_decode &&
        out.sample_rows_are_dense;
    out.single_gpu_greedy_argmax =
        executor.tp_comm == nullptr &&
        !in.have_custom_mask &&
        in.logit_masks_view.empty() &&
        !out.any_topk_topp &&
        out.all_slots_token &&
        out.all_rows_greedy &&
        out.sample_rows_are_dense;
    // De-hardwiring gate (env, default-off): a fire whose every sampling row
    // recognizes to a BakedIR kind (temperature, per the dispatch scorecard) is
    // routed to the driver-baked IR program over the full [N,V] block (Task #4 —
    // temp is the ~2× win vs `sample_temp`). MVP-restricted to the dense
    // pure-decode case so the [N,V] block is contiguous (sampling row r =
    // workspace row r, sample_row 0) and `pi.sample_temp`/`pi.sample_seed`
    // (uploaded pre-sampling) index 1:1; mixed/compact fires fall to legacy.
    // `single_gpu_greedy_argmax`/`sample_temp` stay as the fallback (used iff
    // try_run doesn't Handle). The graph is disabled at its site so the forward
    // emits logits the IR samples over (parallel to the argmax path).
    // #7 DEFAULT-ON (de-hardwiring WIN live): temp routes through the baked IR
    // program by default (the 0.52× win — hotel three-rung signed-off, delta §2f
    // benched, token-exact [271,...] either way). `PIE_DEHARDWIRE_STD_SAMPLERS=0`
    // is a one-line revert escape-hatch for the first production window (delete in
    // a follow-up once stable). Only temp qualifies (dispatch_target(Temperature)
    // =BakedIR); min_p/argmax/top-k/p → DedicatedKernel → fall to their kernels.
    static const char* dehardwire_env =
        std::getenv("PIE_DEHARDWIRE_STD_SAMPLERS");
    static const bool dehardwire_std =
        dehardwire_env == nullptr || std::strcmp(dehardwire_env, "0") != 0;
    out.dehardwire_baked_ir = false;
    if (dehardwire_std && num_sampling > 0 && out.sample_rows_are_dense &&
        out.all_slots_token && !out.all_rows_greedy && in.is_pure_decode &&
        !out.any_topk_topp && !in.have_custom_mask &&
        in.logit_masks_view.empty() && !in.has_spec_drafts &&
        !in.has_custom_program) {
        const std::int32_t vocab =
            executor.loaded_model.hf_config().vocab_size;
        bool all_baked_ir = true;
        sampling_ir::StandardSamplerKind kind0 =
            sampling_ir::StandardSamplerKind::Temperature;
        for (int k = 0; k < num_sampling; ++k) {
            const sampling_ir::SamplerParams pp = sampling_ir::params_from_slot(
                per_slot_temp[k], per_slot_top_k[k], per_slot_top_p[k],
                per_slot_min_p[k], vocab);
            const sampling_ir::StandardSamplerKind kind =
                sampling_ir::infer_sampler_kind(pp);
            // #10: every row must be a BakedIR kind (temperature today; +min-p
            // after #7). Mixed BakedIR kinds are allowed — the IR-hook partitions
            // them by program. A DedicatedKernel/custom row → keep legacy.
            if (k == 0) kind0 = kind;
            if (sampling_ir::dispatch_target(kind) !=
                sampling_ir::DispatchTarget::BakedIR) {
                all_baked_ir = false;
                break;
            }
        }
        if (all_baked_ir) {
            out.dehardwire_baked_ir = true;
            out.dehardwire_kind = kind0;  // informational; IR-hook re-recognizes per group
            out.compact_logit_rows = false;  // IR needs the full [N,V] block
        }
    }
    out.logit_rows_required =
        num_sampling == 0 ? 0
        : out.compact_logit_rows ? num_sampling
        : N;
    out.prob_rows_required = out.any_topk_topp ? N : 0;
    return out;
}

// Run the per-fire forward body. Two paths: replay a captured CUDA
// graph if the active shape qualifies, or invoke `forward_fn.body`
// directly. The captured exec is cached at `executor.graph_cache`.
ForwardDispatchResult run_forward_dispatch(
    Executor& executor,
    const ForwardDispatchInputs& in)
{
    auto& ws = executor.ws;
    auto& kv_cache = executor.kv_cache;
    auto& attn_ws = executor.attn_ws;
    auto& cublas = executor.cublas;
    auto& pi = executor.inputs;
    auto& forward_fn = executor.forward_fn;

    ForwardDispatchResult out;
    out.graph_layout = forward_fn.invoke_graph_layout();
    const bool prepared_small_spec_graph =
        !in.small_spec_graph_shape || out.graph_layout != 0u;
    const bool try_graphs =
        in.graph_shape_ok && !in.have_custom_mask && prepared_small_spec_graph;
    out.graph_captures_single_gpu_argmax =
        try_graphs && in.single_gpu_greedy_argmax &&
        !in.forward_handles_argmax &&
        graph_single_gpu_argmax_enabled();
    const bool rs_verify_frozen =
        executor.rs_cache != nullptr && executor.rs_cache->verify_frozen();
    const std::uint32_t graph_variant =
        (in.tp_greedy_argmax ? 1u : 0u) |
        (out.graph_captures_single_gpu_argmax ? 2u : 0u) |
        (in.forward_handles_argmax ? 4u : 0u) |
        (out.graph_layout << 3) |
        (in.small_spec_graph_shape ? 0x200u : 0u) |
        // Frozen verify skips the GDN state writeback, baked into the captured
        // kernel — must not share a graph with a normal-writeback forward of
        // the same shape.
        (rs_verify_frozen ? 0x400u : 0u);

    if (try_graphs) {
        const ForwardGraphKey key{in.forward_R, in.forward_N, graph_variant};
        cudaGraphExec_t exec = executor.graph_cache->get(key);
        if (exec == nullptr) {
            exec = capture_forward_graph_exec(
                executor, in.h_qo_forward, in.h_kvpi_forward,
                in.h_kvpp_forward, in.h_kvlpl_forward,
                in.forward_N, in.forward_R, in.is_pure_decode,
                in.use_slots ? in.slot_ids_h_data : nullptr,
                in.use_slots ? in.is_fresh_h_data : nullptr,
                in.use_slots ? pi.slot_ids.data() : nullptr,
                in.compact_logit_rows ? pi.sample_idx.data() : nullptr,
                in.compact_logit_rows ? in.num_sampling : 0,
                out.graph_captures_single_gpu_argmax,
                in.tp_greedy_argmax);
            executor.graph_cache->put(key, exec);
            const auto sz = executor.graph_cache->size();
            if (executor.verbose && (sz <= 4 || sz % 16 == 0)) {
                std::cerr << "[pie-driver-cuda] graph captured: R="
                          << in.forward_R
                          << " N=" << in.forward_N
                          << (in.padded ? " padded" : "")
                          << " real_R=" << in.R
                          << " variant=" << graph_variant
                          << " layout=" << out.graph_layout
                          << " (cache size=" << sz << ")\n";
            }
        }
        CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
    } else {
        pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
        fwd_in.token_ids = reinterpret_cast<const std::int32_t*>(pi.tokens.data());
        fwd_in.positions = reinterpret_cast<const std::int32_t*>(pi.positions.data());
        fwd_in.qo_indptr_d         = pi.qo_indptr.data();
        fwd_in.kv_page_indices_d   = pi.kv_page_indices.data();
        fwd_in.kv_page_indptr_d    = pi.kv_page_indptr.data();
        fwd_in.kv_last_page_lens_d = pi.kv_last_page_lens.data();
        fwd_in.qo_indptr_h         = in.h_qo_forward;
        fwd_in.kv_page_indices_h   = in.h_kvpi_forward;
        fwd_in.kv_page_indptr_h    = in.h_kvpp_forward;
        fwd_in.kv_last_page_lens_h = in.h_kvlpl_forward;
        fwd_in.total_tokens        = in.forward_N;
        fwd_in.num_requests        = in.forward_R;
        fwd_in.is_pure_decode      = in.is_pure_decode;
        fwd_in.custom_mask_d        = in.have_custom_mask ? pi.custom_mask.data()        : nullptr;
        fwd_in.custom_mask_indptr_d = in.have_custom_mask ? pi.custom_mask_indptr.data() : nullptr;
        fwd_in.slot_ids_h          = in.use_slots ? in.slot_ids_h_data : nullptr;
        fwd_in.is_fresh_h          = in.use_slots ? in.is_fresh_h_data : nullptr;
        fwd_in.slot_ids_d          = in.use_slots ? pi.slot_ids.data() : nullptr;
        fwd_in.rs_buffer_slot_ids_h    = in.rs_buffer_slot_ids_h;
        fwd_in.rs_buffer_slot_indptr_h = in.rs_buffer_slot_indptr_h;
        fwd_in.rs_buffer_write         = in.rs_buffer_write;
        fwd_in.logit_row_indices_d = in.compact_logit_rows ? pi.sample_idx.data() : nullptr;
        fwd_in.num_logit_rows      = in.compact_logit_rows ? in.num_sampling : 0;
        fwd_in.emit_logits         = in.num_sampling > 0;
        fwd_in.tp_greedy_argmax    = in.tp_greedy_argmax;
        // Multimodal: image data for the encode+scatter (no-op if none). Only
        // the non-graph path carries it — image fires are prefills.
        fwd_in.image_pixels_h            = in.image_pixels_h;
        fwd_in.image_pixel_byte_indptr_h = in.image_pixel_byte_indptr_h;
        fwd_in.image_patch_positions_h   = in.image_patch_positions_h;
        fwd_in.image_anchor_rows_h       = in.image_anchor_rows_h;
        fwd_in.num_images                = in.num_images;
        fwd_in.image_grids_h             = in.image_grids_h;
        fwd_in.mrope_positions_h         = in.mrope_positions_h;
        fwd_in.num_mrope_positions       = in.num_mrope_positions;
        // Multimodal: audio data for the encode+scatter (no-op if none).
        fwd_in.audio_features_h             = in.audio_features_h;
        fwd_in.audio_feature_byte_indptr_h  = in.audio_feature_byte_indptr_h;
        fwd_in.audio_anchor_rows_h          = in.audio_anchor_rows_h;
        fwd_in.num_clips                    = in.num_clips;
        forward_fn.invoke_body(ws, kv_cache, attn_ws, cublas, fwd_in);
    }
    return out;
}

namespace {
using probe_clock = std::chrono::steady_clock;
inline std::uint32_t us_between(probe_clock::time_point a,
                                probe_clock::time_point b) {
    return static_cast<std::uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
}
inline void write_probes(pie_driver::PieForwardResponseView& out,
                         probe_clock::time_point t0, probe_clock::time_point t1,
                         probe_clock::time_point t2, probe_clock::time_point t3,
                         probe_clock::time_point t4, probe_clock::time_point t5) {
    const auto t6 = probe_clock::now();
    out.probe_wire_parse_us    = us_between(t0, t1);
    out.probe_plan_us          = us_between(t1, t2);
    out.probe_h2d_us           = us_between(t2, t3);
    out.probe_kernel_launch_us = us_between(t3, t4);
    out.probe_sync_us          = us_between(t4, t5);
    out.probe_response_build_us = us_between(t5, t6);
}

// Lazily construct the Sampling-IR JIT backend on the first program-carrying
// fire — the JitEngine ctor resolves the device arch from the CUDA context,
// which is current mid-fire. One-shot: a hard init failure (e.g. no NVRTC)
// disables programmable sampling for the process and falls back to the legacy
// sampler rather than retrying + re-logging every fire.
sampling_ir::SamplingIrBackend* ensure_sampling_ir_backend(Executor& ex) {
    if (ex.sampling_ir_runtime.has_backend()) {
        return ex.sampling_ir_backend.get();
    }
    if (ex.sampling_ir_init_attempted) {
        return nullptr;
    }
    ex.sampling_ir_init_attempted = true;
    try {
        ex.sampling_ir_backend =
            std::make_unique<sampling_ir::SamplingIrBackend>(/*batched=*/true);
        ex.sampling_ir_runtime.set_backend(ex.sampling_ir_backend.get());
        return ex.sampling_ir_backend.get();
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] sampling-ir backend init failed: "
                  << e.what()
                  << " — programmable sampling disabled (legacy path)\n";
        ex.sampling_ir_backend.reset();
        return nullptr;
    }
}
}  // namespace

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_driver::PieForwardRequestView& view,
    pie_driver::PieForwardResponseView& out_resp,
    Executor& executor,
    std::uint64_t handled)
{
    using clock = std::chrono::steady_clock;
    const auto t_entry = clock::now();

    // #27 cut #1 (a2): cleared each fire; set only if this fire takes the
    // output→tensor fast-path (deferred forward-done). The in-proc service reads
    // it after the handler to drive `out.deferred`.
    executor.last_fire_deferred = false;

    // Diagnostic trace (env-gated, zero-cost when unset): localizes where a
    // fire parks across the worker forward path. Pairs with the runtime-side
    // inproc submit/recv/slot-fill trace to map the a/b/c park location.
    const bool ir_trace = std::getenv("PIE_SAMPLING_IR_TRACE") != nullptr;
    if (ir_trace) {
        std::cerr << "[ir-trace] fire entry req_id=" << req_id << "\n";
        std::cerr.flush();
    }

    // Local references for the most-touched Executor members.
    auto& engine               = executor.loaded_model;
    auto& ws                   = executor.ws;
    auto& kv_cache             = executor.kv_cache;
    auto& attn_ws              = executor.attn_ws;
    auto& cublas               = executor.cublas;
    auto& pi                   = executor.inputs;  // persistent input slabs
    auto& forward_fn           = executor.forward_fn;
    const int max_workspace_tokens = executor.max_workspace_tokens;
    const int graph_pad_page = executor.graph_pad_page;

    // Track whether the custom-mask path was populated this fire so the
    // forward kernel knows whether to consume `pi.custom_mask`. Sizes are
    // stashed alongside so the TP broadcast knows how many bytes to fan
    // out to followers.
    bool have_custom_mask = false;
    int mask_bytes = 0;
    int mask_indptr_count = 0;

    // Multimodal (gemma4 vision): image side-channel from the view. Declared
    // before the try so it's in scope at the forward dispatch (which is after
    // the try/catch). `image_pixels` is f32 pixel_values stored as bytes.
    const float* img_pixels_h =
        reinterpret_cast<const float*>(view.image_pixels.data());
    const auto img_pix_byte_indptr = view.image_pixel_indptr.as<std::uint32_t>();
    const auto img_patch_pos       = view.image_patch_positions.as<std::uint32_t>();
    const auto img_anchor          = view.image_anchor_rows.as<std::uint32_t>();
    const int img_num_images       = static_cast<int>(img_anchor.size());
    // Qwen3-VL M-RoPE: per-image (t,h,w) grids + per-image-token 3-axis
    // positions. Assembled into a full `[N, 3]` per-token array below.
    const auto img_grids           = view.image_grids.as<std::uint32_t>();
    const auto img_mrope_pos       = view.image_mrope_positions.as<std::uint32_t>();
    const auto img_mrope_indptr    = view.image_mrope_indptr.as<std::uint32_t>();
    // Storage for the assembled per-token [N,3] M-RoPE positions (filled
    // only when this fire carries Qwen3-VL mrope image data).
    std::vector<std::uint32_t> mrope_positions_storage;

    // Multimodal (gemma4 audio): log-mel side-channel from the view.
    // `audio_features` is f32 log-mel stored as bytes.
    const float* aud_features_h =
        reinterpret_cast<const float*>(view.audio_features.data());
    const auto aud_feat_byte_indptr = view.audio_feature_indptr.as<std::uint32_t>();
    const auto aud_anchor           = view.audio_anchor_rows.as<std::uint32_t>();
    const int aud_num_clips         = static_cast<int>(aud_anchor.size());

    // Env-gated per-fire timing (PIE_FIRE_TIMING=1): logs tokens/requests/images
    // and wall duration of the whole fire. Scope guard fires on every return.
    int dbg_R = 0, dbg_N = 0;
    const bool dbg_fire = std::getenv("PIE_FIRE_TIMING") != nullptr;
    struct FireTimer {
        std::chrono::steady_clock::time_point t0; const int& R; const int& N;
        int nimg; std::uint32_t rid; bool en;
        ~FireTimer() {
            if (!en) return;
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            std::cerr << "[fire] req=" << rid << " R=" << R << " N=" << N
                      << " imgs=" << nimg << " " << ms << "ms\n";
        }
    } dbg_ft{t_entry, dbg_R, dbg_N, img_num_images, req_id, dbg_fire};

    try {
        const auto tok_view_orig   = view.token_ids.as<std::uint32_t>();
        const auto pos_view_orig   = view.position_ids.as<std::uint32_t>();
        const auto qo_view_orig    = view.qo_indptr.as<std::uint32_t>();
        const auto kvpi_view = view.kv_page_indices.as<std::uint32_t>();
        const auto kvpp_view = view.kv_page_indptr.as<std::uint32_t>();
        const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();

        const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
        const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();
        // Per-request stable context ids. Runtime-managed rs_cache slot
        // ids below are indexed in this same request order.
        const auto ctx_id_view     = view.context_ids.as<std::uint64_t>();

        // Sampler params (per-sampler arrays). Read here (rather than
        // further down) so the spec expansion below can append cloned
        // entries for the verification block.
        const auto temp_view_orig  = view.sampler_temperatures.as<float>();
        const auto top_k_view_orig = view.sampler_top_k.as<std::uint32_t>();
        const auto top_p_view_orig = view.sampler_top_p.as<float>();
        const auto min_p_view_orig = view.sampler_min_p.as<float>();
        const auto types_view_orig = view.sampler_types.as<std::uint32_t>();
        const auto seed_view_orig  = view.sampler_seeds.as<std::uint32_t>();
        const auto rns_view_orig   = view.request_num_samplers.as<std::uint32_t>();
        const auto logit_masks_view = view.logit_masks.as<std::uint32_t>();
        const auto logit_mask_indptr_view = view.logit_mask_indptr.as<std::uint32_t>();

        // Spec-decoding fields. When non-empty for some request, splice
        // drafts into the forward and append a verification block to
        // the sampling layout (one extra sample per draft + one bonus).
        // Mirrors pie_driver's `get_spec_expanded_*`.
        const auto spec_tok_view  = view.spec_token_ids.as<std::uint32_t>();
        const auto spec_pos_view  = view.spec_position_ids.as<std::uint32_t>();
        const auto spec_iptr_view = view.spec_indptr.as<std::uint32_t>();
        const bool has_spec_drafts = !spec_tok_view.empty();

        const int R = static_cast<int>(qo_view_orig.size()) - 1;

        // Spec-decoding batch expansion. When `has_spec_drafts` is false
        // the result has empty vectors and `verify_slot_start[r] == -1`
        // for every r; the active spans below fall through to the
        // original wire views.
        const SpecExpansion spec = expand_spec_batch(
            SpecExpansionInputs{
                tok_view_orig, pos_view_orig, qo_view_orig, kvlpl_view_orig,
                sidx_view_orig, sptr_view_orig, rns_view_orig,
                types_view_orig, top_k_view_orig, seed_view_orig,
                temp_view_orig, top_p_view_orig, min_p_view_orig,
                spec_tok_view, spec_pos_view, spec_iptr_view,
                kv_cache.page_size(),
            },
            R);
        const std::vector<int>& verify_slot_start = spec.verify_slot_start;
        const std::vector<int>& verify_n_drafts   = spec.verify_n_drafts;

        // Active views: spec-expanded if drafts present, else direct
        // wire. The rest of the function uses these.
        const std::span<const std::uint32_t> tok_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.tokens)               : tok_view_orig;
        const std::span<const std::uint32_t> pos_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.positions)            : pos_view_orig;
        const std::span<const std::uint32_t> qo_view    = spec.has_drafts ? std::span<const std::uint32_t>(spec.qo_indptr)            : qo_view_orig;
        const std::span<const std::uint32_t> kvlpl_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.kv_last_page_lens)    : kvlpl_view_orig;
        const std::span<const std::uint32_t> sidx_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampling_indices)     : sidx_view_orig;
        const std::span<const std::uint32_t> sptr_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampling_indptr)      : sptr_view_orig;
        const std::span<const std::uint32_t> rns_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.request_num_samplers) : rns_view_orig;
        const std::span<const std::uint32_t> types_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_types)        : types_view_orig;
        const std::span<const std::uint32_t> top_k_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_top_k)        : top_k_view_orig;
        const std::span<const std::uint32_t> seed_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_seeds)        : seed_view_orig;
        const std::span<const float>         temp_view  = spec.has_drafts ? std::span<const float>        (spec.sampler_temperatures) : temp_view_orig;
        const std::span<const float>         top_p_view = spec.has_drafts ? std::span<const float>        (spec.sampler_top_p)        : top_p_view_orig;
        const std::span<const float>         min_p_view = spec.has_drafts ? std::span<const float>        (spec.sampler_min_p)        : min_p_view_orig;

        const auto t_wire_parse_end = clock::now();

        const int N = static_cast<int>(tok_view.size());
        const int num_sampling = static_cast<int>(sidx_view.size());
        dbg_R = R; dbg_N = N;
        if (ir_trace) {
            const std::size_t n_prog_bytes = view.sampling_program_bytes.size();
            const std::size_t prog_indptr_n =
                view.sampling_program_bytes_indptr.size();
            std::cerr << "[ir-trace] fire shape req_id=" << req_id
                      << " N=" << N << " R=" << R
                      << " num_sampling=" << num_sampling
                      << " program_bytes=" << n_prog_bytes
                      << " program_indptr_len=" << prog_indptr_n << "\n";
            std::cerr.flush();
        }

        // Qwen3-VL: assemble the per-token [N,3] M-RoPE positions. Text rows
        // carry (p,p,p) from the 1-D `pos_view`; image-token rows are
        // overwritten with the staged 3-axis (t,h,w) positions for each image
        // (image i's rows start at batch row `img_anchor[i]`). Built only when
        // image mrope data is present (image prefills never carry spec drafts).
        if (img_num_images > 0 && !img_mrope_pos.empty() && !spec.has_drafts) {
            mrope_positions_storage.resize(static_cast<std::size_t>(N) * 3);
            for (int t = 0; t < N; ++t) {
                const std::uint32_t p =
                    t < static_cast<int>(pos_view.size()) ? pos_view[t] : 0u;
                mrope_positions_storage[3 * t + 0] = p;
                mrope_positions_storage[3 * t + 1] = p;
                mrope_positions_storage[3 * t + 2] = p;
            }
            for (int im = 0; im < img_num_images; ++im) {
                const std::uint32_t anchor_row = img_anchor[im];
                const std::uint32_t lo =
                    im < static_cast<int>(img_mrope_indptr.size()) - 1
                        ? img_mrope_indptr[im] : 0u;
                const std::uint32_t hi =
                    im + 1 < static_cast<int>(img_mrope_indptr.size())
                        ? img_mrope_indptr[im + 1] : 0u;
                const std::uint32_t n_tok = (hi - lo) / 3u;
                for (std::uint32_t j = 0; j < n_tok; ++j) {
                    const int row = static_cast<int>(anchor_row + j);
                    if (row < 0 || row >= N) continue;
                    mrope_positions_storage[3 * row + 0] = img_mrope_pos[lo + 3 * j + 0];
                    mrope_positions_storage[3 * row + 1] = img_mrope_pos[lo + 3 * j + 1];
                    mrope_positions_storage[3 * row + 2] = img_mrope_pos[lo + 3 * j + 2];
                }
            }
        }

        if (N == 0 || R <= 0) {
            // Empty batch — emit a zero-request response view.
            std::vector<pie_driver::PerRequestOutput> empty(std::max(R, 0));
            executor.response_builder.build(empty, out_resp);
            return;
        }
        if (N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }

        // Compute max KV length across requests for shmem sizing.
        // Also detect "pure decode" (every request has qo_len == 1) so
        // we can dispatch flashinfer's decode kernel on the hot path.
        const int page_size = kv_cache.page_size();
        int max_kv_len = 0;
        const std::uint32_t* h_kvpp  = kvpp_view.data();
        const std::uint32_t* h_kvlpl = kvlpl_view.data();
        const std::uint32_t* h_qo    = qo_view.data();
        bool is_pure_decode = (R > 0);
        for (int r = 0; r < R; ++r) {
            const int num_pages_r = static_cast<int>(h_kvpp[r + 1] - h_kvpp[r]);
            if (num_pages_r <= 0) continue;
            const int kv_len_r = (num_pages_r - 1) * page_size +
                                 static_cast<int>(h_kvlpl[r]);
            if (kv_len_r > max_kv_len) max_kv_len = kv_len_r;
            if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
        }

        const auto rs_slot_view = view.rs_slot_ids.as<std::uint32_t>();
        const auto rs_flag_view = view.rs_slot_flags.as<std::uint8_t>();
        bool use_slots =
            R > 0 && rs_slot_view.size() == static_cast<std::size_t>(R);
        if (executor.rs_cache != nullptr && R > 0 && !use_slots) {
            throw std::runtime_error(
                "rs_cache forward missing runtime-assigned slot ids");
        }
        const bool has_fresh_slot =
            use_slots && std::any_of(
                rs_flag_view.begin(), rs_flag_view.end(),
                [](std::uint8_t v) { return (v & 1u) != 0; });

        // Ph7 RS working-set buffered-activation channel. Single-role per pass
        // (v1): a FOLD pass (FOLD-bit=2 set) gathers+replays from the buffered
        // pool into recurrent_state (separate fold-replay dispatch below); an
        // rs-output write pass (FOLD-bit clear + buffered slabs present)
        // scatters in-proj [mixed_qkv|a|b] to the pool during the main forward.
        const auto rs_fold_view = view.rs_fold_lens.as<std::uint32_t>();
        const auto rs_buf_id_view = view.rs_buffer_slot_ids.as<std::uint32_t>();
        const auto rs_buf_indptr_view = view.rs_buffer_slot_indptr.as<std::uint32_t>();
        const bool rs_is_fold = use_slots && std::any_of(
            rs_flag_view.begin(), rs_flag_view.end(),
            [](std::uint8_t v) { return (v & 2u) != 0; });
        const bool rs_is_write =
            use_slots && !rs_is_fold && rs_buf_id_view.size() > 0;

        const GraphShapeDecision graph_shape = decide_graph_shape(
            executor, R, N, page_size,
            is_pure_decode, has_spec_drafts, has_fresh_slot,
            kvpi_view.size(), pi.kv_page_indices.size());
        bool graph_shape_ok = graph_shape.graph_shape_ok;
        const int graph_requests = graph_shape.graph_requests;
        const bool small_spec_graph_shape = graph_shape.small_spec_graph_shape;

        ForwardInputViews forward_inputs = make_forward_input_views(
            tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view,
            R, graph_requests, graph_pad_page);
        const int forward_N = forward_inputs.total_tokens;
        const int forward_R = forward_inputs.num_requests;
        const std::uint32_t* h_qo_forward = forward_inputs.qo_indptr.data();
        const std::uint32_t* h_kvpi_forward =
            forward_inputs.kv_page_indices.data();
        const std::uint32_t* h_kvpp_forward =
            forward_inputs.kv_page_indptr.data();
        const std::uint32_t* h_kvlpl_forward =
            forward_inputs.kv_last_page_lens.data();

        // Refill persistent device buffers with this fire's wire inputs.
        // Same device addresses every fire — required for graph-replay
        // safety; cheap (single async memcpy each) on its own.
        pi.tokens.copy_from_host(forward_inputs.tokens);
        pi.positions.copy_from_host(forward_inputs.positions);
        pi.qo_indptr.copy_from_host(forward_inputs.qo_indptr);
        pi.kv_page_indices.copy_from_host(forward_inputs.kv_page_indices);
        pi.kv_page_indptr.copy_from_host(forward_inputs.kv_page_indptr);
        pi.kv_last_page_lens.copy_from_host(forward_inputs.kv_last_page_lens);

        // BRLE attention masks. For prefill batches that aren't pure
        // causal, decode + upload a packed bitmap and route through the
        // flashinfer kCustom path. For decode-only batches the kernel
        // doesn't support custom masks; we proceed without one (a
        // limitation we'd have to fix by routing decode through the
        // prefill kernel for custom-mask inferlets).
        const auto fmask_view  = view.flattened_masks.as<std::uint32_t>();
        const auto mskptr_view = view.mask_indptr.as<std::uint32_t>();
        if (!has_spec_drafts && !is_pure_decode && !fmask_view.empty()) {
            const auto qo_span =
                std::span<const std::uint32_t>(qo_view.data(), qo_view.size());
            const auto kvpp_span =
                std::span<const std::uint32_t>(kvpp_view.data(), kvpp_view.size());
            const auto kvlpl_span =
                std::span<const std::uint32_t>(kvlpl_view.data(), kvlpl_view.size());
            if (!pie_cuda_driver::brle::is_pure_causal(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size())) {
                auto decoded = pie_cuda_driver::brle::decode(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size());
                pi.custom_mask.copy_from_host(
                    std::span<const std::uint8_t>(decoded.packed));
                pi.custom_mask_indptr.copy_from_host(
                    std::span<const std::int32_t>(decoded.mask_indptr));
                mask_bytes = static_cast<int>(decoded.packed.size());
                mask_indptr_count = static_cast<int>(decoded.mask_indptr.size());
                have_custom_mask = true;
            }
        }

        // Linear-attention rs_cache slots. Runtime owns slot assignment;
        // RS-capable models must receive one slot id per request.
        std::vector<std::int32_t> slot_ids_h;
        std::vector<std::uint8_t> is_fresh_h;
        if (use_slots) {
            const int slot_count = graph_shape_ok ? forward_R : R;
            slot_ids_h.resize(slot_count);
            is_fresh_h.resize(slot_count);
            for (int r = 0; r < R; ++r) {
                slot_ids_h[r] = static_cast<std::int32_t>(rs_slot_view[r]);
                is_fresh_h[r] = (r < static_cast<int>(rs_flag_view.size()) &&
                                 (rs_flag_view[r] & 1u))
                                    ? 1u
                                    : 0u;
            }
            for (int r = R; r < slot_count; ++r) {
                slot_ids_h[r] = executor.graph_pad_slot;
                is_fresh_h[r] = 0u;
            }
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids_h));
            pi.is_fresh.copy_from_host(std::span<const std::uint8_t>(is_fresh_h));
        }

        // RS_FLAG_FOLD (bit 1) + per-request rs_fold_lens (working-set fold(n)).
        // v1: the runtime's `inference.fold` advances the folded-state boundary
        // host-side and the mock driver no-ops the fold compute, so there is no
        // real CUDA fold to run here yet — and the existing `commit_len` GDN
        // primitive folds tokens *in-forward* (the spec commit-advance path),
        // not from the working set's separately-buffered RS slabs.
        // Ph7 (RS-real-driver): the real buffered-then-fold needs a new
        // fold-from-buffer kernel that reads the `rs_buffer_slot_ids` SoA
        // (append-only ForwardRequest field, parked with this kernel) for each
        // request flagged RS_FLAG_FOLD and folds rs_fold_lens[r] buffered tokens
        // into the folded slot `rs_slot_ids[r]`. Lower it here.
        // Frozen-verify speculative path. The verify forward walks the GDN
        // recurrent state in registers to produce correct draft outputs but
        // persists NOTHING (every linear layer sees write_state=false), so each
        // committed slot stays at its pre-verify value — the implicit
        // speculative snapshot. After sampling, one batched repair forward over
        // [input | accepted] advances each committed slot to exactly the
        // confirmed prefix. No per-request snapshot buffer → no concurrency
        // cap; the committed slot only ever reflects committed tokens. (KV is
        // untouched by freezing — only the recurrent-state writeback is gated —
        // so the verify writes KV normally and the repair's KV re-write is an
        // idempotent no-op.)
        // rs_frozen_verify gates the conv freeze + the repair/commit-advance.
        // The FLA freeze is decoupled (in linear_attn_layer_body): the fla only
        // freezes in commit-advance mode (verify_frozen AND stash enabled), so
        // the default keeps fla-drift + repair (the validated 6899 path) while
        // the commit-advance gets a truly frozen verify for lossless replay.
        bool rs_frozen_verify = false;
        if (executor.rs_cache != nullptr) {
            rs_frozen_verify = has_spec_drafts && use_slots &&
                               executor.tp_comm == nullptr;
            executor.rs_cache->set_verify_frozen(rs_frozen_verify);
        }

        // ── #12 phase-1: recognize a standard sampler program ───────────
        // The contract migration (#15) moved the sampler slot→program: a fire
        // now carries the sampler as a bytecode program (with the legacy slot
        // params EMPTY). Hash the program against the driver's own baked
        // standard programs (`pie_standard_samplers.h`); a match → extract its
        // dedicated-kernel params and plan the fire as the equivalent slot fire,
        // restoring the pre-migration dispatch (FlashInfer top-p / sample_temp
        // min-p / greedy argmax / BakedIR temp). A miss → a genuine custom
        // program → CustomJIT via `try_run`, unchanged. Without this, every
        // standard program falls through to CustomJIT (SplitMix64-Gumbel) ≠ the
        // FlashInfer the slot fire used → silent token divergence.
        bool program_recognized = false;
        sampling_ir::DedicatedParams rec_params;
        const bool prog_attached =
            view.sampling_program_bytes_indptr.size() >= 2 &&
            !view.sampling_program_bytes.empty();
        if (prog_attached && num_sampling > 0) {
            const std::uint32_t vocab = static_cast<std::uint32_t>(
                executor.loaded_model.hf_config().vocab_size);
            // Built once per vocab (fixed per model); per-fire = O(table) hash compare.
            static thread_local std::uint32_t recog_vocab = 0;
            static thread_local std::vector<sampling_ir::StandardKindEntry> recog_table;
            static thread_local std::vector<sampling_ir::StandardKindEntry> canon_table;
            if (recog_table.empty() || recog_vocab != vocab) {
                recog_table = sampling_ir::build_standard_kind_table(vocab);
                canon_table = sampling_ir::build_canonical_kind_table(vocab);
                recog_vocab = vocab;
            }
            const std::uint32_t blo = view.sampling_program_bytes_indptr.data()[0];
            const std::uint32_t bhi = view.sampling_program_bytes_indptr.data()[1];
            const std::uint8_t* const prog_bc = view.sampling_program_bytes.data() + blo;
            const std::size_t prog_len = static_cast<std::size_t>(bhi - blo);
            // Exact hash (k-invariant kinds) first; on a miss, the op-shape
            // (canonicalize RankLe→0)-then-hash for the k-bearing kinds (#12 phase-2).
            auto kind_opt = sampling_ir::recognize_standard_kind(recog_table, prog_bc, prog_len);
            if (!kind_opt)
                kind_opt = sampling_ir::recognize_canonical_kind(canon_table, prog_bc, prog_len);
            if (ir_trace && !kind_opt) {
                const std::uint64_t wh = sampling_ir::jit::fnv1a64(prog_bc, prog_len);
                const std::vector<std::uint8_t> canon =
                    sampling_ir::canonicalize_op_shape(prog_bc, prog_len);
                const std::uint64_t ch =
                    sampling_ir::jit::fnv1a64(canon.data(), canon.size());
                std::cerr << "[ir-trace] recognize MISS wire_len=" << prog_len
                          << " wire_hash=" << std::hex << wh
                          << " canon_hash=" << ch << std::dec
                          << " exact=" << recog_table.size()
                          << " canon=" << canon_table.size() << "\n";
            }
            if (kind_opt) {
                // Program 0's host-submit inputs (the WS1a slice) carry T + the
                // filter (top_p/min_p) by input ordinal; extract them. The k-bearing
                // top-k cutoff is NOT a submit input — it lives baked in the op-shape.
                std::vector<sampling_ir::SubmitInput> rec_submit;
                if (view.sampling_input_indptr.size() >= 2) {
                    const std::uint32_t ilo = view.sampling_input_indptr.data()[0];
                    const std::uint32_t ihi = view.sampling_input_indptr.data()[1];
                    const std::uint8_t* blob = view.sampling_input_blob.data();
                    for (std::uint32_t i = ilo; i < ihi; ++i) {
                        sampling_ir::SubmitInput si;
                        si.key = view.sampling_input_keys.data()[i];
                        si.data = blob + view.sampling_input_offsets.data()[i];
                        si.len_bytes = view.sampling_input_lens.data()[i];
                        rec_submit.push_back(si);
                    }
                }
                rec_params =
                    sampling_ir::extract_dedicated_params(*kind_opt, rec_submit);
                // #12 phase-2: the baked top-k cutoff comes from the op-shape
                // (the RankLe immediate), not a submit input. nullopt for the
                // k-invariant kinds (no RankLe) → top_k stays unset.
                if (auto kcut = sampling_ir::extract_rank_le_k(prog_bc, prog_len))
                    rec_params.top_k = static_cast<std::int32_t>(*kcut);
                program_recognized = true;
                if (ir_trace) {
                    std::cerr << "[ir-trace] program recognized kind="
                              << static_cast<int>(*kind_opt) << " T=" << rec_params.temp
                              << " top_p=" << rec_params.top_p
                              << " min_p=" << rec_params.min_p
                              << " top_k=" << rec_params.top_k
                              << " → dedicated dispatch\n";
                }
                // echo's free agreement guard (#7-step-1 analog, env-gated, prod
                // default-off): the extracted params MUST re-classify (param
                // recognizer + #7 normalizations) to the hash kind — a mis-extract
                // (wrong key → wrong param) is caught here, pre-HW. Under
                // PIE_RECOGNIZER_AUDIT we ALSO dump the extracted VALUES every
                // recognized fire (not just on disagreement) — the #12 stage-2
                // "param-dump" gate: HW-readable proof the route AND the values are
                // right (e.g. top_p==0.9, T==0.8), catching a wrong-VALUE extract
                // that the kind-only agreement check can't (0.5 and 0.9 both → TopP).
                if (std::getenv("PIE_RECOGNIZER_AUDIT")) {
                    const bool agree = sampling_ir::extracted_params_agree(
                        *kind_opt, rec_params,
                        executor.loaded_model.hf_config().vocab_size);
                    std::cerr << "[extract-audit] " << (agree ? "OK" : "DISAGREE")
                              << " hash_kind=" << static_cast<int>(*kind_opt)
                              << " extracted(T=" << rec_params.temp
                              << " top_p=" << rec_params.top_p
                              << " min_p=" << rec_params.min_p
                              << " top_k=" << rec_params.top_k << ")\n";
                }
            }
        }

        // ── Sample-plan construction (hoisted) ──────────────────
        // Sampling stays outside the CUDA graph because sampler/probe
        // layouts can vary even when the decode shape is identical. The
        // builder fills `sampling_scratch()` and emits dispatch flags
        // so the forward + sampling launches downstream can pick the
        // fast path when the request mix permits.
        const SamplePlanResult sp = build_sample_plan(executor, SamplePlanInputs{
            .outspec_view      = view.output_spec_flags.as<std::uint8_t>(),
            .sptr_view         = sptr_view,
            .sidx_view         = sidx_view,
            .rns_view          = rns_view,
            .types_view        = types_view,
            .temp_view         = temp_view,
            .top_k_view        = top_k_view,
            .top_p_view        = top_p_view,
            .min_p_view        = min_p_view,
            .seed_view         = seed_view,
            .logit_masks_view  = logit_masks_view,
            .h_qo              = h_qo,
            .N                 = N,
            .R                 = R,
            .num_sampling      = num_sampling,
            .is_pure_decode    = is_pure_decode,
            .have_custom_mask  = have_custom_mask,
            .has_spec_drafts   = has_spec_drafts,
            // A recognized standard program is planned as the equivalent slot
            // fire (not a custom program), so the dedicated ladder runs.
            .has_custom_program = prog_attached && !program_recognized,
            .program_recognized = program_recognized,
            .rec_temp           = rec_params.temp,
            .rec_top_p          = rec_params.top_p,
            .rec_min_p          = rec_params.min_p,
            .rec_top_k          = rec_params.top_k,
        });
        const bool has_rich_sampler_slots = sp.has_rich_sampler_slots;
        const bool need_msgpack          = sp.need_msgpack;
        const bool any_topk_topp         = sp.any_topk_topp;
        const bool sample_rows_are_dense = sp.sample_rows_are_dense;
        const bool all_rows_greedy       = sp.all_rows_greedy;
        const bool all_slots_token       = sp.all_slots_token;
        const bool compact_logit_rows    = sp.compact_logit_rows;
        const bool tp_greedy_argmax      = sp.tp_greedy_argmax;
        const bool single_gpu_greedy_argmax = sp.single_gpu_greedy_argmax;
        const bool dehardwire_baked_ir   = sp.dehardwire_baked_ir;
        // A custom IR sampling program (carrier path) owns the sampling and reads
        // `ws.logits` via `try_run` — so, exactly like de-hardwiring, the forward
        // must emit the full [N,V] logits. Otherwise a greedy custom fire takes the
        // fused lm_head-argmax-only path (logits never materialized) and the IR
        // argmax samples over unwritten logits → token 0. (Argmax routes to a
        // DedicatedKernel, so this IR output path is unexercised until a real
        // program-path argmax runs.)
        // #12: a RECOGNIZED standard program is NOT custom — it runs the dedicated
        // ladder (FlashInfer/sample_temp/argmax/BakedIR), so it follows the same
        // logits-materialization rules as the equivalent slot fire (and lets the
        // greedy-argmax fused path stay enabled when its kind is argmax).
        const bool has_custom_program =
            prog_attached && !program_recognized;
        // De-hardwiring / custom-program: force the non-graph forward path so it
        // emits the full [N,V] logits (the captured graph would fuse/own the
        // sampling); the IR program then samples over the emitted logits.
        // `graph_shape_ok` is only consumed at the forward dispatch below, so
        // overriding it here is safe.
        if (dehardwire_baked_ir || has_custom_program) {
            graph_shape_ok = false;
        }
        const int  logit_rows_required   = sp.logit_rows_required;
        const int  prob_rows_required    = sp.prob_rows_required;
        auto& sample_scratch = sampling_scratch();
        auto& h_per_temp   = sample_scratch.h_per_temp;
        auto& h_per_min_p  = sample_scratch.h_per_min_p;
        auto& h_per_top_p  = sample_scratch.h_per_top_p;
        auto& h_per_top_k  = sample_scratch.h_per_top_k;
        auto& h_per_seed   = sample_scratch.h_per_seed;
        auto& h_sample_idx = sample_scratch.h_sample_idx;
        auto& per_slot_type  = sample_scratch.per_slot_type;
        auto& per_slot_temp  = sample_scratch.per_slot_temp;
        auto& per_slot_top_k = sample_scratch.per_slot_top_k;
        auto& per_slot_top_p = sample_scratch.per_slot_top_p;
        auto& per_slot_min_p = sample_scratch.per_slot_min_p;
        const std::uint32_t* h_sptr = sptr_view.data();
        const std::uint32_t* h_sidx = sidx_view.data();
        const auto outspec_view = view.output_spec_flags.as<std::uint8_t>();
        if (logit_rows_required > tensor_rows(ws.logits)) {
            std::cerr << "[pie-driver-cuda] fire_batch needs "
                      << logit_rows_required
                      << " logit rows, exceeding workspace capacity "
                      << tensor_rows(ws.logits) << "\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }
        if (prob_rows_required > tensor_rows(ws.probs)) {
            std::cerr << "[pie-driver-cuda] fire_batch needs "
                      << prob_rows_required
                      << " probability rows, exceeding workspace capacity "
                      << tensor_rows(ws.probs) << "\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }

        const SamplingPlan sample_plan{
            any_topk_topp,
            std::span<const float>(h_per_temp),
            std::span<const float>(h_per_top_p),
            std::span<const float>(h_per_min_p),
            std::span<const std::int32_t>(h_per_top_k),
            std::span<const std::uint32_t>(h_per_seed),
            std::span<const std::int32_t>(h_sample_idx),
        };

        const auto t_plan_end = clock::now();

        if (compact_logit_rows) {
            CUDA_CHECK(cudaMemcpyAsync(pi.sample_idx.data(), h_sample_idx.data(),
                                       sizeof(std::int32_t) * num_sampling,
                                       cudaMemcpyHostToDevice, nullptr));
        }

        // TP fan-out. Rank 0 broadcasts the per-fire payload (header +
        // refilled persistent_inputs) to every follower so they can run
        // the same forward kernels against an identical view of inputs.
        // The all-reduces inside `forward_fn.body` then synchronise the
        // ranks layer-by-layer. The header includes the forward variant so
        // CUDA graph capture/replay stays lockstep across ranks.
        if (executor.tp_comm != nullptr) {
            tp_cpu_gate_notify(executor.tp_cpu_gate_key);
            tp_broadcast_inputs(*executor.tp_comm, pi,
                                forward_N, forward_R, is_pure_decode,
                                static_cast<int>(
                                    forward_inputs.kv_page_indices.size()),
                                mask_bytes, mask_indptr_count,
                                /*has_slot_ids=*/use_slots,
                                tp_greedy_argmax,
                                /*logit_rows=*/compact_logit_rows ? num_sampling : 0,
                                /*stream=*/nullptr);
        }

        // ── prepare hook ────────────────────────────────────────
        // Always run the per-arch prepare phase first (when present).
        // For graph-capable archs this updates pinned host / device
        // plan state for the captured body to read. Lives outside any
        // capture region so the host work re-runs every fire.
        forward_fn.invoke_prepare(
            attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = h_qo_forward,
                .kv_page_indices_h = h_kvpi_forward,
                .kv_page_indices_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
                .kv_page_indptr_h = h_kvpp_forward,
                .kv_page_indptr_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
                .kv_last_page_lens_h =
                    h_kvlpl_forward,
                .kv_last_page_lens_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
                .total_tokens = forward_N,
                .num_requests = forward_R,
                .is_pure_decode = is_pure_decode,
            });

        // ── Upload sampling inputs (must precede sampling launch) ──
        // Per-row sampler params land in `pi.sample_*`. Sampling runs after
        // forward, but the upload is kept here so the response path can use
        // the same prepared device buffers for captured and uncaptured
        // forward bodies.
        if (!tp_greedy_argmax && !single_gpu_greedy_argmax) {
            upload_sampling_inputs(pi, sample_plan, N, /*stream=*/nullptr);
        }
        const bool logits_argmax_only =
            forward_fn.supports_fused_lmhead_argmax &&
            !has_rich_sampler_slots &&
            logit_masks_view.empty() &&
            !any_topk_topp &&
            all_slots_token &&
            all_rows_greedy &&
            !dehardwire_baked_ir &&  // de-hardwire needs full logits, not argmax-only
            !has_custom_program;     // a custom IR program samples over ws.logits → emit them
        forward_fn.invoke_set_logits_argmax_only(logits_argmax_only);
        const bool forward_handles_argmax =
            forward_fn.supports_fused_lmhead_argmax &&
            logits_argmax_only &&
            single_gpu_greedy_argmax &&
            !dehardwire_baked_ir;  // IR owns the sampling → forward must emit logits
        forward_fn.invoke_set_fused_argmax_output(
            forward_handles_argmax
                ? reinterpret_cast<std::int32_t*>(pi.sampled.data())
                : nullptr);

        const auto t_h2d_end = clock::now();

        // ── #6 WS8 P2: device-resident next-input inject ────────────────
        // Source some of this fire's input tokens (`pi.tokens`) device-side from a
        // PRIOR producer forward's retained `pi.sampled` (the next-input link), in
        // place of a host-injected token — removing the host `output()`+copy between
        // decode passes. Runs on the forward's stream so it strictly precedes the
        // forward's `pi.tokens` read; event-gated on each producer's sample-done so
        // the retained copy is ready cross-pass. The carrier groups entries by
        // producer link id; one inject per distinct producer. A producer not in the
        // retained map (e.g. its source pass hasn't run) is skipped.
        if (!view.next_input_producer_links.empty()) {
            const std::uint32_t* prod = view.next_input_producer_links.data();
            const std::uint32_t* srcs = view.next_input_src_rows.data();
            const std::uint32_t* dsts = view.next_input_dest_slots.data();
            const std::size_t nlinks = view.next_input_producer_links.size();
            std::size_t i = 0;
            while (i < nlinks) {
                const std::uint32_t link = prod[i];
                std::vector<sampling_ir::NextInputLink> links;
                for (; i < nlinks && prod[i] == link; ++i)
                    links.push_back(sampling_ir::NextInputLink{srcs[i], dsts[i]});
                auto it = executor.retained_next_input.find(link);
                if (it != executor.retained_next_input.end()) {
                    sampling_ir::inject_next_input_after(
                        it->second.copy.data(), links, pi.tokens.data(),
                        it->second.done, cublas.stream());
                    if (ir_trace)
                        std::cerr << "[ir-trace] next-input INJECT link=" << link
                                  << " rows=" << links.size() << " → pi.tokens\n";
                } else if (ir_trace) {
                    // A consumer references a link not in the retained map — the
                    // use-after-free signature (freed too early, or producer pass
                    // hasn't run). The dest stays the placeholder ⇒ token-divergence.
                    std::cerr << "[ir-trace] next-input INJECT MISS link=" << link
                              << " rows=" << links.size()
                              << " (no retained source — UAF / unrun producer)\n";
                }
            }
        }

        // ── Forward pass ────────────────────────────────────────
        // Either replay a captured CUDA graph or invoke `forward_fn.body`
        // directly — see run_forward_dispatch for the variant/key logic.
        // The verify timer wraps body + sampling on the same stream so
        // its end-of-fire reading is the visible cost of the work the
        // graph executes.
        StepProfileTimer verify_timer(
            "verify", cublas.stream(), forward_N, forward_R);
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-begin req_id=" << req_id
                      << " forward_N=" << forward_N
                      << " forward_R=" << forward_R << "\n";
            std::cerr.flush();
        }
        // ── #27 cut #1 (b): WAR guard for single-buffer pi.sampled ───────
        // The prior fast-path fire's eager-D2H (copy stream) READS pi.sampled;
        // this fire's forward/sampler WRITES it (alloc-once, reused). Gate this
        // fire's cublas-stream work behind the prior fire's eager-D2H read so the
        // in-flight copy can't be clobbered. Placed before the forward dispatch
        // (not only the dedicated sampler tail) so it also covers the
        // fused-lm_head-argmax / captured-graph paths that write pi.sampled
        // DURING the forward. nullptr ⇒ first fire / no prior fast-path read ⇒
        // no wait. Near-zero cost: the prior one-token D2H has long completed by
        // the time this fire's matmul is enqueued (it hides under H2D + compute).
        if (executor.last_eager_d2h_done)
            CUDA_CHECK(cudaStreamWaitEvent(cublas.stream(),
                                           executor.last_eager_d2h_done, 0));

        const ForwardDispatchResult fd = run_forward_dispatch(
            executor, ForwardDispatchInputs{
                .R = R,
                .forward_R = forward_R,
                .forward_N = forward_N,
                .N = N,
                .num_sampling = num_sampling,
                .is_pure_decode = is_pure_decode,
                .have_custom_mask = have_custom_mask,
                .small_spec_graph_shape = small_spec_graph_shape,
                .graph_shape_ok = graph_shape_ok,
                .tp_greedy_argmax = tp_greedy_argmax,
                .forward_handles_argmax = forward_handles_argmax,
                .single_gpu_greedy_argmax = single_gpu_greedy_argmax,
                .compact_logit_rows = compact_logit_rows,
                .use_slots = use_slots,
                .padded = forward_inputs.padded,
                .h_qo_forward = h_qo_forward,
                .h_kvpi_forward = h_kvpi_forward,
                .h_kvpp_forward = h_kvpp_forward,
                .h_kvlpl_forward = h_kvlpl_forward,
                .slot_ids_h_data = slot_ids_h.data(),
                .is_fresh_h_data = is_fresh_h.data(),
                .rs_buffer_slot_ids_h = rs_is_write ? rs_buf_id_view.data() : nullptr,
                .rs_buffer_slot_indptr_h = rs_is_write ? rs_buf_indptr_view.data() : nullptr,
                .rs_buffer_write = rs_is_write,
                .image_pixels_h = img_pixels_h,
                .image_pixel_byte_indptr_h = img_pix_byte_indptr.data(),
                .image_patch_positions_h = img_patch_pos.data(),
                .image_anchor_rows_h = img_anchor.data(),
                .num_images = img_num_images,
                .image_grids_h = img_grids.data(),
                .mrope_positions_h = mrope_positions_storage.empty()
                    ? nullptr : mrope_positions_storage.data(),
                .num_mrope_positions = static_cast<int>(
                    mrope_positions_storage.size() / 3),
                .audio_features_h = aud_features_h,
                .audio_feature_byte_indptr_h = aud_feat_byte_indptr.data(),
                .audio_anchor_rows_h = aud_anchor.data(),
                .num_clips = aud_num_clips,
            });
        const bool graph_captures_single_gpu_argmax =
            fd.graph_captures_single_gpu_argmax;
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-returned req_id=" << req_id << "\n";
            std::cerr.flush();
        }
        // Sampling is deliberately outside the CUDA graph. The forward
        // graph key is only `R`; sampler/probe layouts vary independently
        // (for example top-p token-only decode vs. argmax + rich probes).
        // Running the current fire's sampling kernel after the graph launch
        // keeps that R-only key valid.
        // A fire that samples nothing (e.g. a multimodal image-token KV-fill
        // pass) produces no logits and no sampled tokens — skip every argmax /
        // sampling launch (they would read the unwritten, undersized
        // `ws.logits` over all N rows and fault).
        // ── Sampling-IR (programmable sampler) mode-select ──────────────
        // A request carrying a sampling program takes the IR path: a
        // JIT-compiled kernel DAG over `ws.logits`, in place of the legacy
        // per-slot sampler. MVP: single program / single sampling row / M=1
        // decode. The backend is built lazily here (CUDA context is current
        // mid-fire). A null backend or an absent program falls through to the
        // legacy ladder unchanged. The IR Token output lands in
        // `pi.sampled[sample_row]`, so the existing D2H + response build below
        // marshal it like any other sampled token.
        sampling_ir::RunStatus ir_status = sampling_ir::RunStatus::NoProgram;
        if (num_sampling > 0 &&
            view.sampling_program_bytes_indptr.size() >= 2 &&
            !view.sampling_program_bytes.empty() &&
            !program_recognized &&  // #12: recognized standard → dedicated ladder, not CustomJIT
            ensure_sampling_ir_backend(executor) != nullptr) {
            const std::uint32_t blo = view.sampling_program_bytes_indptr.data()[0];
            const std::uint32_t bhi = view.sampling_program_bytes_indptr.data()[1];
            const int sample_row = static_cast<int>(h_qo[0] + h_sidx[0]);

            // Submit-bound host inputs for program 0 (WS1a). The carrier packs
            // every program's entries into one key/offset/len index table CSR'd
            // by `sampling_input_indptr`; pull program 0's slice and point each
            // SubmitInput at its bytes in `sampling_input_blob`. The runtime
            // stages these to device before binding.
            std::vector<sampling_ir::SubmitInput> submit_inputs;
            if (view.sampling_input_indptr.size() >= 2) {
                const std::uint32_t ilo = view.sampling_input_indptr.data()[0];
                const std::uint32_t ihi = view.sampling_input_indptr.data()[1];
                const std::uint8_t* blob = view.sampling_input_blob.data();
                for (std::uint32_t i = ilo; i < ihi; ++i) {
                    sampling_ir::SubmitInput si;
                    si.key = view.sampling_input_keys.data()[i];
                    si.data = blob + view.sampling_input_offsets.data()[i];
                    si.len_bytes = view.sampling_input_lens.data()[i];
                    submit_inputs.push_back(si);
                }
            }

            // Host-late VALUES for program 0 (WS1b correctness path): the
            // late-value table mirrors the submit index-table but is keyed by
            // `sampling_late_keys`; pull program 0's keys and their bytes from
            // `sampling_late_blob`. The runtime stages these like submit inputs;
            // a declared HostLate input resolves to them, else skip.
            std::vector<sampling_ir::SubmitInput> late_value_inputs;
            if (view.sampling_late_indptr.size() >= 2 &&
                !view.sampling_late_blob.empty()) {
                const std::uint32_t llo = view.sampling_late_indptr.data()[0];
                const std::uint32_t lhi = view.sampling_late_indptr.data()[1];
                const std::uint8_t* lblob = view.sampling_late_blob.data();
                for (std::uint32_t i = llo; i < lhi; ++i) {
                    if (i >= view.sampling_late_lens.size()) break;
                    // len == 0 ⇒ no staged host value for this late key (device
                    // alias / skip-on-miss handles it); don't register it.
                    if (view.sampling_late_lens.data()[i] == 0) continue;
                    sampling_ir::SubmitInput li;
                    li.key = view.sampling_late_keys.data()[i];
                    li.data = lblob + view.sampling_late_offsets.data()[i];
                    li.len_bytes = view.sampling_late_lens.data()[i];
                    late_value_inputs.push_back(li);
                }
            }

            sampling_ir::FireContext ctx;
            ctx.program_bytecode = {
                view.sampling_program_bytes.data() + blo,
                static_cast<std::size_t>(bhi - blo)};
            ctx.submit_inputs = submit_inputs;
            ctx.late_value_inputs = late_value_inputs;
            // v4 binding manifest for program 0: the per-slot binding-map (Logits
            // intrinsic vs keyed HostTensor) the binding-free v4 bytecode omits.
            // Without it, get_or_compile routes to the v3 self-binding decoder,
            // which rejects version 4 ("decode failed: version 4").
            sampling_ir::ProgramManifest manifest;
            if (view.sampling_binding_indptr.size() >= 2) {
                const std::uint32_t mlo = view.sampling_binding_indptr.data()[0];
                const std::uint32_t mhi = view.sampling_binding_indptr.data()[1];
                manifest.reserve(mhi - mlo);
                for (std::uint32_t i = mlo; i < mhi; ++i) {
                    sampling_ir::InputBind b;
                    if (view.sampling_binding_kind.data()[i] != 0 /* KIND_TENSOR */) {
                        b.kind = sampling_ir::BindKind::HostTensor;
                        b.host_key = view.sampling_binding_key.data()[i];
                        b.ready = sampling_ir::HostAvailability::SubmitBound;
                    } else {
                        b.kind = sampling_ir::BindKind::Logits;
                    }
                    manifest.push_back(b);
                }
            }
            ctx.manifest = std::move(manifest);
            ctx.logits = ws.logits.data();
            ctx.pi = &pi;
            ctx.vocab_size = engine.hf_config().vocab_size;
            ctx.sample_row = sample_row;
            // Ambient per-row RNG seed S (Model B) for custom RNG programs (e.g.
            // mirostat's Op::Rng). The de-hardwiring/#10 path sets this at its group
            // base; the custom-program path must too, else a sampling custom's
            // RowSeed buffer is unbound at launch (jit_backend bind_external skips it
            // when args.row_seeds==nullptr). sample_row indexes the [N] seed block
            // (= the seed legacy sample_temp uses for this row). Non-RNG customs
            // (e.g. grammar's masked-argmax) declare no RowSeed and ignore it.
            ctx.row_seeds = pi.sample_seed.data() + sample_row;
            ctx.prng_offset = static_cast<std::uint64_t>(handled);
            ctx.stream = cublas.stream();
            ir_status = executor.sampling_ir_runtime.try_run(ctx);
        }
        // De-hardwiring branch (Task #4 / #10 cross-request batching): a fire whose
        // every row recognizes to a BakedIR kind (temperature today; +min-p after #7)
        // and that carries NO custom program is partitioned by program identity and
        // each group launched over its [Ng,V] block. CONTIGUOUS groups launch in
        // place (the keystone fast path, no gather); a SCATTERED group (≥2 BakedIR
        // kinds interleaved — #7's temp+min_p — or sampling rows interleaved in a
        // mixed fire) is deferred to the gather path (delta's group.hpp) → for now
        // the whole fire falls back to the legacy sampler (token-exact, kept in the
        // plan). Never fails the fire (hardwired-kernel replacement, not a program).
        if (ir_status == sampling_ir::RunStatus::NoProgram && dehardwire_baked_ir &&
            num_sampling > 0 && ensure_sampling_ir_backend(executor) != nullptr) {
            auto* ir_backend = ensure_sampling_ir_backend(executor);
            const std::uint32_t vocab =
                static_cast<std::uint32_t>(engine.hf_config().vocab_size);
            // Per-row program handle (= the canonical bytecode-hash / #9 cache key /
            // #10 group key — one mechanism): recognize kind → baked program →
            // get_or_compile. Cached per kind, so all rows of a kind share a handle.
            struct KindProg {
                sampling_ir::ProgramHandle           handle;
                sampling_ir::StandardSamplerProgram  prog;
                sampling_ir::StandardSamplerKind     kind;
            };
            std::unordered_map<int, KindProg> by_kind;
            std::vector<sampling_ir::ProgramHandle> row_handles(
                num_sampling, sampling_ir::kInvalidProgram);
            std::vector<int> row_kind(num_sampling, 0);
            bool resolve_ok = true;
            for (int r = 0; r < num_sampling && resolve_ok; ++r) {
                const sampling_ir::StandardSamplerKind kind =
                    sampling_ir::infer_sampler_kind(sampling_ir::params_from_slot(
                        per_slot_temp[r], per_slot_top_k[r], per_slot_top_p[r],
                        per_slot_min_p[r], static_cast<std::int32_t>(vocab)));
                const int ki = static_cast<int>(kind);
                auto it = by_kind.find(ki);
                if (it == by_kind.end()) {
                    sampling_ir::StandardSamplerProgram prog =
                        sampling_ir::standard_sampler_program(kind, vocab);
                    if (!prog.valid) { resolve_ok = false; break; }
                    const sampling_ir::ProgramHandle h = ir_backend->get_or_compile(
                        std::span<const std::uint8_t>(prog.bytecode, prog.len),
                        prog.manifest);
                    if (h == sampling_ir::kInvalidProgram) { resolve_ok = false; break; }
                    it = by_kind.emplace(ki, KindProg{h, prog, kind}).first;
                }
                row_handles[r] = it->second.handle;
                row_kind[r] = ki;
            }
            std::vector<sampling_ir::ProgramGroup> groups;
            bool all_contiguous = resolve_ok;
            if (resolve_ok) {
                groups = sampling_ir::partition_by_program(row_handles);
                for (const auto& g : groups) {
                    if (!sampling_ir::is_contiguous(g)) { all_contiguous = false; break; }
                }
            }
            // MVP (#10 contiguous): launch only when every group is contiguous —
            // in place at its base row, no gather copy. A scattered group defers to
            // the gather→launch→scatter path (the multi-group honesty bench rides #7
            // temp+min_p) → fall back to legacy now (token-exact).
            if (resolve_ok && all_contiguous && !groups.empty()) {
                bool all_handled = true;
                for (const auto& g : groups) {
                    const KindProg& kp = by_kind.at(row_kind[g.rows[0]]);
                    const int first = static_cast<int>(g.rows[0]);
                    const int ng    = static_cast<int>(g.rows.size());
                    // Per-row params at offset `first`, keyed by the manifest host
                    // keys (T = 0; min_p = 1 for MinP). The runtime stages [ng] host
                    // bytes → device, binds the batched base; the kernel strides/row.
                    std::vector<sampling_ir::SubmitInput> gin;
                    sampling_ir::SubmitInput ti;
                    ti.key = 0;
                    ti.data = reinterpret_cast<const std::uint8_t*>(
                        per_slot_temp.data() + first);
                    ti.len_bytes = static_cast<std::uint32_t>(ng * sizeof(float));
                    gin.push_back(ti);
                    if (kp.kind == sampling_ir::StandardSamplerKind::MinP) {
                        sampling_ir::SubmitInput mi;
                        mi.key = 1;
                        mi.data = reinterpret_cast<const std::uint8_t*>(
                            per_slot_min_p.data() + first);
                        mi.len_bytes = static_cast<std::uint32_t>(ng * sizeof(float));
                        gin.push_back(mi);
                    }
                    sampling_ir::FireContext ctx;
                    ctx.program_bytecode = std::span<const std::uint8_t>(
                        kp.prog.bytecode, kp.prog.len);
                    ctx.manifest      = kp.prog.manifest;
                    ctx.submit_inputs = gin;
                    ctx.logits        = ws.logits.data();
                    ctx.pi            = &pi;
                    ctx.vocab_size    = static_cast<int>(vocab);
                    ctx.sample_row    = first;  // contiguous group base (dense ⇒ workspace row)
                    ctx.num_rows      = ng;
                    // Ambient per-row RNG seed S (Model B): the group's [ng] u32 slice
                    // of the [N] block uploaded pre-sampling. Dense ⇒ sample_seed[r] is
                    // sampling row r's seed (= what legacy sample_temp uses).
                    ctx.row_seeds     = pi.sample_seed.data() + first;
                    ctx.prng_offset   = static_cast<std::uint64_t>(handled);
                    ctx.stream        = cublas.stream();
                    if (executor.sampling_ir_runtime.try_run(ctx) !=
                        sampling_ir::RunStatus::Handled) {
                        all_handled = false;
                        break;
                    }
                }
                if (all_handled) {
                    ir_status = sampling_ir::RunStatus::Handled;
                    if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
                        std::cerr << "[ir-trace] de-hardwire baked-IR HANDLED groups="
                                  << groups.size() << " rows=" << num_sampling << "\n";
                    }
                } else {
                    std::cerr << "[pie-driver-cuda] de-hardwire baked-IR partial launch"
                                 " — falling back to legacy\n";
                    // ir_status stays NoProgram → legacy sampler runs (token-exact).
                }
            } else if (std::getenv("PIE_SAMPLING_IR_TRACE")) {
                std::cerr << "[ir-trace] de-hardwire baked-IR deferred to legacy"
                             " (resolve_ok=" << resolve_ok
                          << " all_contiguous=" << all_contiguous << ")\n";
            }
        }
        if (ir_status == sampling_ir::RunStatus::SkippedLateBindMiss) {
            // spec §7.4: a late-bound input was missing → discard + retry,
            // fail loud (no block, no default).
            std::cerr << "[pie-driver-cuda] sampling-ir late-bind miss, req_id="
                      << req_id << " — discarding fire\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }
        if (ir_status == sampling_ir::RunStatus::Failed) {
            std::cerr << "[pie-driver-cuda] sampling-ir program failed, req_id="
                      << req_id << "\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }

        if (ir_status == sampling_ir::RunStatus::Handled) {
            // IR program wrote pi.sampled[sample_row]; skip the legacy sampler.
        } else if (num_sampling == 0) {
            // nothing to sample
        } else if (tp_greedy_argmax) {
            kernels::launch_select_global_argmax_pairs(
                reinterpret_cast<const std::uint64_t*>(ws.greedy_pairs_all.data()),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, executor.tp_comm->world_size(), cublas.stream());
        } else if (graph_captures_single_gpu_argmax) {
            // The captured forward graph already wrote pi.sampled.
        } else if (forward_handles_argmax) {
            // The forward's fused lm_head-argmax wrote pi.sampled.
        } else if (single_gpu_greedy_argmax) {
            const int argmax_parts =
                greedy_argmax_parts(engine.hf_config().vocab_size);
            if (argmax_parts > 1 && !ws.greedy_pairs_all.empty()) {
                kernels::launch_argmax_bf16_partitioned_pairs(
                    ws.logits.data(),
                    reinterpret_cast<std::uint64_t*>(ws.greedy_pairs_all.data()),
                    N, engine.hf_config().vocab_size, argmax_parts,
                    cublas.stream());
                kernels::launch_select_global_argmax_pairs(
                    reinterpret_cast<const std::uint64_t*>(ws.greedy_pairs_all.data()),
                    reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                    N, argmax_parts, cublas.stream());
            } else {
                kernels::launch_argmax_bf16(
                    ws.logits.data(),
                    reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                    N, engine.hf_config().vocab_size, cublas.stream());
            }
        } else {
            if (compact_logit_rows) {
                kernels::launch_sample_temp_bf16_compact_scatter(
                    ws.logits.data(),
                    pi.sample_idx.data(),
                    pi.sample_temp.data(),
                    pi.sample_min_p.data(),
                    pi.sample_seed.data(),
                    reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                    num_sampling, engine.hf_config().vocab_size,
                    cublas.stream());
            } else {
                launch_sampling_kernel(
                    ws, pi.sampled.data(), pi, sample_plan,
                    N, num_sampling, engine.hf_config().vocab_size,
                    /*prng_offset=*/static_cast<std::uint64_t>(handled),
                    /*stream=*/cublas.stream());
            }
        }

        // ── #6 WS8 P2: device-resident next-input retain ────────────────
        // If this forward is a next-input source (`pipeline_source_link != 0`),
        // preserve its `pi.sampled[N]` device-side under that global link id: a
        // later consumer pass reads it via inject, but `pi.sampled` is alloc-once /
        // reused, so t+1's forward would overwrite the producer. Copy (D2D) +
        // record a sample-done event on the sampler stream; the retained entry is
        // freed when the host signals the link via `next_input_free_links`.
        if (view.pipeline_source_link != 0 && N > 0) {
            Executor::RetainedSampled& r =
                executor.retained_next_input[view.pipeline_source_link];
            if (r.copy.size() < static_cast<std::size_t>(N))
                r.copy = DeviceBuffer<std::int32_t>::alloc(static_cast<std::size_t>(N));
            if (r.done == nullptr) CUDA_CHECK(cudaEventCreate(&r.done));
            CUDA_CHECK(cudaMemcpyAsync(
                r.copy.data(), pi.sampled.data(),
                sizeof(std::int32_t) * static_cast<std::size_t>(N),
                cudaMemcpyDeviceToDevice, cublas.stream()));
            CUDA_CHECK(cudaEventRecord(r.done, cublas.stream()));
            if (ir_trace)
                std::cerr << "[ir-trace] next-input RETAIN link="
                          << view.pipeline_source_link << " rows=" << N
                          << " (rows>1 ⇒ shared batch link → refcount-N first-row-free"
                             " probe holds; rows=1 ⇒ per-seq link)\n";
        }

        // Sample plan was built above the prepare hook (hoisted so the
        // sampling uploads are ready before the forward graph launch). The host
        // variables (`need_msgpack`, `per_slot_*`, `any_topk_topp`,
        // `h_per_*`, `h_sample_idx`) are still in scope here for the
        // response builder.

        // Only copy the first N entries — `pi.sampled` is sized for
        // max_workspace_tokens, but only [0, N) are valid this fire.
        // Async on the same stream the sampler ran on so it slots into
        // the stream's FIFO; we sync immediately after because the
        // response payload depends on these tokens. (Future work moves
        // the sync past the host-side response-prep so the host
        // and GPU can overlap.)
        std::int32_t* sampled_host =
            sampled_pinned_buf(static_cast<std::size_t>(N));
        CUDA_CHECK(cudaMemcpyAsync(sampled_host, pi.sampled.data(),
                                   sizeof(std::int32_t) * N,
                                   cudaMemcpyDeviceToHost,
                                   cublas.stream()));
        const auto t_kernel_launch_end = clock::now();
        verify_timer.finish(cublas.stream());
        CUDA_CHECK(cudaStreamSynchronize(cublas.stream()));
        const auto t_sync_end = clock::now();

        // ── #6 WS8 P2: free retained next-input sources signaled this pass ──
        // The host appends a producer link id to `next_input_free_links` once its
        // LAST consumer drained (whose inject read the retained copy above). The
        // stream sync just completed → that inject finished → freeing the retained
        // buffer + its event is hazard-free, and the global id is reclaimable.
        // Driver is count-agnostic: it frees strictly on the host signal.
        if (!view.next_input_free_links.empty()) {
            const std::uint32_t* freed = view.next_input_free_links.data();
            const std::size_t nfree = view.next_input_free_links.size();
            for (std::size_t k = 0; k < nfree; ++k) {
                auto it = executor.retained_next_input.find(freed[k]);
                if (it != executor.retained_next_input.end()) {
                    if (it->second.done != nullptr)
                        CUDA_CHECK(cudaEventDestroy(it->second.done));
                    executor.retained_next_input.erase(it);  // DeviceBuffer dtor frees the copy
                    if (ir_trace)
                        std::cerr << "[ir-trace] next-input FREE link=" << freed[k]
                                  << " (host all-consumers-drained signal)\n";
                }
            }
        }

        // ── #27 cut #1 (a2): output→tensor fast-path ─────────────────────
        // If the host bound per-output pinned dsts (`sampling_output_dst_ptrs`),
        // eager-D2H each program's output VALUE into its dst on the tensor-I/O
        // copy stream and DEFER the forward-done to a copy-stream host-func — it
        // fires once the pinned buffer is filled (the (a2) seam; the host's
        // output() reads the pinned bytes, not the ForwardResponse channels).
        // MVP slice: one Token per program (the greedy `cuda_runahead` path) →
        // device_src = pi.sampled[p]. Multi-output / rich programs fall through to
        // the legacy marshal below (a follow-on wires last_output_ptrs()).
        if (!view.sampling_output_dst_ptrs.empty() &&
            view.sampling_output_indptr.size() == 2 &&
            view.sampling_output_dst_ptrs.size() == 1 && N >= 1) {
            const std::uint64_t dst = view.sampling_output_dst_ptrs.data()[0];
            const std::uint32_t cap = view.sampling_output_dst_lens.data()[0];
            if (dst != 0 && cap >= sizeof(std::int32_t)) {
                // Single-sequence MVP: the sampled token is the LAST sampled row
                // pi.sampled[N-1] (carrier src_row = producer n_rows-1 — the prime
                // samples its last position; decode N=1 → row 0), NOT row p.
                const void* src =
                    static_cast<const void*>(pi.sampled.data() + (N - 1));
                const std::size_t nbytes = sizeof(std::int32_t);
                cudaEvent_t sample_done = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(&sample_done,
                                                    cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(sample_done, cublas.stream()));
                cudaEvent_t t_d2h_done =
                    sampling_ir::TensorIoEngine::instance().eager_d2h_outputs(
                        &dst, &cap, &src, &nbytes, 1, sample_done);
                CUDA_CHECK(cudaEventDestroy(sample_done));
                // Persist for delta's WAR guard at the NEXT forward's sampling
                // tail (it waits this before t+1 overwrites single-buffer
                // pi.sampled). The previous fire's event was already consumed by
                // this fire's guard wait, so it's safe to destroy.
                if (executor.last_eager_d2h_done)
                    CUDA_CHECK(cudaEventDestroy(executor.last_eager_d2h_done));
                executor.last_eager_d2h_done = t_d2h_done;
                executor.last_fire_deferred = true;
                // Cut #1 (a2) deferred forward-done: count = batch_size with an
                // EMPTY tokens[] — the sampled token rides the pinned buffer (the
                // eager-D2H above), NOT the response. `num_requests = R` is what
                // the scheduler counts (an empty/num_requests=0 response was the
                // `got 0` bug); leaving tokens[] empty closes the false-pass
                // escape hatch by construction (the gate can only pass via the
                // pinned read). `last_fire_deferred` makes serve_forever send
                // this response EXACTLY ONCE from the copy-stream host-func
                // post-D2H (no early/pre-D2H send into the single-shot slot), so
                // the host's `rx` fires only once the pinned dst is filled. The
                // scheduler's a2-mode counts + resolves Ok(Some(success)) and
                // skips tokens[] extraction (token is in the pinned Tensor).
                out_resp = pie_driver::PieForwardResponseView{};
                out_resp.num_requests = static_cast<std::uint32_t>(R);
                write_probes(out_resp, t_entry, t_wire_parse_end, t_plan_end,
                             t_h2d_end, t_kernel_launch_end, t_sync_end);
                return;
            }
        }

        // ── Sampling-IR rich / multi-output marshaling ──────────────────
        // A program that emits anything beyond a single scalar Token —
        // mirostat's (token, surprise S); a spec-verify Vector<k> accept-
        // prefix; entropy/logprob probes — is marshaled here into the
        // ForwardResponse channels in the program's declared output order,
        // mirroring the legacy rich path (response_builder.build). Token →
        // tokens, Scalar/Entropy → entropies, a multi-element (Vector<k>)
        // Token accept-prefix → spec_tokens (side-channel). Single-token
        // programs fall through to the dense token path below, unchanged.
        // MVP: single IR request (outputs land on request 0).
        if (ir_status == sampling_ir::RunStatus::Handled && R >= 1) {
            const sampling_ir::ProgramInterface* pif =
                executor.sampling_ir_runtime.last_interface();
            const bool ir_rich =
                pif != nullptr &&
                !(pif->outputs.size() == 1 &&
                  pif->outputs[0].cls == sampling_ir::OutputClass::Token &&
                  pif->outputs[0].elem_count <= 1);
            if (ir_rich) {
                std::span<void* const> outs =
                    executor.sampling_ir_runtime.last_output_ptrs();
                std::vector<pie_driver::PerRequestOutput> per_req(
                    static_cast<std::size_t>(R));
                auto& pr = per_req[0];
                for (std::size_t i = 0;
                     i < pif->outputs.size() && i < outs.size(); ++i) {
                    const sampling_ir::DeclaredOutput& o = pif->outputs[i];
                    if (outs[i] == nullptr) continue;
                    switch (o.cls) {
                        case sampling_ir::OutputClass::Token: {
                            if (o.elem_count <= 1) {
                                std::int32_t t = 0;
                                CUDA_CHECK(cudaMemcpy(&t, outs[i], sizeof(t),
                                                      cudaMemcpyDeviceToHost));
                                pr.tokens.push_back(
                                    static_cast<std::uint32_t>(t));
                            } else {
                                // Vector<k> accept-prefix → spec_tokens; emit
                                // the non-(-1) prefix (sentinel = first reject).
                                std::vector<std::int32_t> v(o.elem_count);
                                CUDA_CHECK(cudaMemcpy(
                                    v.data(), outs[i],
                                    sizeof(std::int32_t) * o.elem_count,
                                    cudaMemcpyDeviceToHost));
                                for (std::int32_t x : v) {
                                    if (x < 0) break;
                                    pr.spec_tokens.push_back(
                                        static_cast<std::uint32_t>(x));
                                }
                            }
                            break;
                        }
                        case sampling_ir::OutputClass::Scalar:
                        case sampling_ir::OutputClass::Entropy: {
                            float s = 0.f;
                            CUDA_CHECK(cudaMemcpy(&s, outs[i], sizeof(s),
                                                  cudaMemcpyDeviceToHost));
                            pr.entropies.push_back(s);
                            break;
                        }
                        default:
                            // Logits / Logprobs / Dist rich outputs: later wave.
                            break;
                    }
                }
                executor.response_builder.build(per_req, out_resp);
                write_probes(out_resp, t_entry, t_wire_parse_end, t_plan_end,
                             t_h2d_end, t_kernel_launch_end, t_sync_end);
                return;
            }
        }

        // CUDA's fast token samplers do not yet consume BRLE logit masks.
        // When constrained decoding supplies one, keep correctness by
        // overriding token-sampler rows with a host-side masked argmax. This
        // path is cold and only runs for constrained requests; normal decode
        // and benchmark traffic keep the GPU sampler result.
        const std::int32_t* all_sampled = sampled_host;
        std::vector<std::int32_t> sampled_override;
        if (!logit_masks_view.empty()) {
            sampled_override.assign(sampled_host, sampled_host + N);
            apply_logit_mask_overrides(
                ws, sampled_override, logit_masks_view, logit_mask_indptr_view,
                qo_view, sptr_view, sidx_view, std::span<const std::uint32_t>(per_slot_type),
                R, N, engine.hf_config().vocab_size);
            all_sampled = sampled_override.data();
        }

        bool single_token_per_request =
            !need_msgpack &&
            sample_rows_are_dense &&
            num_sampling == R &&
            N == R;
        if (single_token_per_request) {
            for (int r = 0; r < R; ++r) {
                if (h_sptr[r + 1] - h_sptr[r] != 1u) {
                    single_token_per_request = false;
                    break;
                }
            }
        }
        if (single_token_per_request) {
            executor.response_builder.build_token_only_dense(
                std::span<const std::int32_t>(all_sampled,
                                              static_cast<std::size_t>(R)),
                out_resp);
            write_probes(out_resp, t_entry, t_wire_parse_end,
                         t_plan_end, t_h2d_end,
                         t_kernel_launch_end, t_sync_end);
            if (executor.verbose && (handled <= 4 || handled % 100 == 0)) {
                std::cerr << "[pie-driver-cuda] req_id=" << req_id
                          << " R=" << R << " N=" << N
                          << " sampled=" << num_sampling
                          << " max_kv=" << max_kv_len << "\n";
            }
            return;
        }

        // Flat-path arrays: token sampler is the only slot type allowed
        // here (need_msgpack would have flipped otherwise), so counts
        // align 1:1 with sampling slots.
        std::vector<std::uint32_t> per_request_counts(R);
        std::vector<std::uint32_t> sampled_tokens;
        sampled_tokens.reserve(num_sampling);
        for (int r = 0; r < R; ++r) {
            const std::uint32_t lo = h_sptr[r];
            const std::uint32_t hi = h_sptr[r + 1];
            const std::uint32_t qo_lo = h_qo[r];
            per_request_counts[r] = hi - lo;
            for (std::uint32_t k = lo; k < hi; ++k) {
                const std::uint32_t row = qo_lo + h_sidx[k];
                sampled_tokens.push_back(
                    static_cast<std::uint32_t>(all_sampled[row]));
            }
        }
        // Single structured response. The fast path (need_msgpack ==
        // false) populates only `tokens`; rich paths additionally fill
        // dists/logits/logprobs/entropies via the per-sampler sub-passes.

        if (need_msgpack) {
            std::vector<pie_driver::PerRequestOutput> per_req(R);
            std::vector<int> spec_accepted_drafts(
                static_cast<std::size_t>(R), -1);
            std::vector<std::int32_t> mtp_base_rows(
                static_cast<std::size_t>(R), -1);
            std::vector<std::uint32_t> mtp_draft_positions(
                static_cast<std::size_t>(R), 0);
            auto remember_mtp_source = [&](int r, std::uint32_t row) {
                if (r < 0 || r >= R) return;
                if (row >= static_cast<std::uint32_t>(N)) return;
                const std::uint64_t source_pos = pos_view[row];
                const std::uint64_t draft_pos = source_pos + 2ull;
                if (draft_pos > std::numeric_limits<std::uint32_t>::max()) {
                    return;
                }
                mtp_base_rows[static_cast<std::size_t>(r)] =
                    static_cast<std::int32_t>(row);
                mtp_draft_positions[static_cast<std::size_t>(r)] =
                    static_cast<std::uint32_t>(draft_pos);
            };
            if (has_rich_sampler_slots) {
                const ResponseSubpassContext sub_ctx{
                    ws,
                    R, num_sampling, engine.hf_config().vocab_size,
                    std::span<const std::uint32_t>(per_slot_type),
                    std::span<const float>(per_slot_temp),
                    std::span<const std::int32_t>(per_slot_top_k),
                    qo_view, sptr_view, sidx_view, rns_view,
                };
                gather_raw_logits(sub_ctx, per_req);
                compute_entropy_slots(sub_ctx, per_req);
                compute_logprob_slots(sub_ctx, view, per_req);
                compute_dist_slots(sub_ctx, per_req);
            }

            // Per-request token list. For non-spec requests this is the
            // token-typed slots' samples. For spec requests we walk the
            // verification block (cloned token samplers at the bonus +
            // each draft position) and produce the accepted prefix; the
            // inferlet's own samples for that request are discarded.
            for (int r = 0; r < R; ++r) {
                const std::uint32_t qo_lo = h_qo[r];
                auto& bucket = per_req[r].tokens;

                if (has_spec_drafts && verify_slot_start[r] >= 0) {
                    const int vs = verify_slot_start[r];
                    const int n_d = verify_n_drafts[r];
                    const int spec_lo = (r < static_cast<int>(spec_iptr_view.size()))
                        ? static_cast<int>(spec_iptr_view[r]) : 0;
                    std::vector<std::uint32_t> block(n_d + 1);
                    for (int j = 0; j <= n_d; ++j) {
                        const std::uint32_t row = qo_lo + h_sidx[vs + j];
                        block[j] = static_cast<std::uint32_t>(all_sampled[row]);
                    }
                    int match = 0;
                    for (int k = 0; k < n_d; ++k) {
                        if (block[k] == spec_tok_view[spec_lo + k]) match++;
                        else break;
                    }
                    if (mtp_trace_take()) {
                        std::cerr << "[pie-mtp-trace] verify r=" << r
                                  << " n_drafts=" << n_d
                                  << " accepted=" << match
                                  << " pos=[";
                        for (int j = 0; j <= n_d; ++j) {
                            if (j) std::cerr << ",";
                            const std::uint32_t row = qo_lo + h_sidx[vs + j];
                            std::cerr << pos_view[row];
                        }
                        std::cerr << "] draft=[";
                        for (int j = 0; j < n_d; ++j) {
                            if (j) std::cerr << ",";
                            std::cerr << spec_tok_view[spec_lo + j];
                        }
                        std::cerr << "] verify=[";
                        for (int j = 0; j <= n_d; ++j) {
                            if (j) std::cerr << ",";
                            std::cerr << block[j];
                        }
                        std::cerr << "]\n";
                    }
                    spec_accepted_drafts[static_cast<std::size_t>(r)] = match;
                    bucket.assign(block.begin(), block.begin() + match + 1);
                    if (!bucket.empty()) {
                        const int j = static_cast<int>(bucket.size()) - 1;
                        const std::uint32_t row = qo_lo + h_sidx[vs + j];
                        remember_mtp_source(r, row);
                    }
                } else {
                    const std::uint32_t lo = h_sptr[r];
                    const std::uint32_t hi = h_sptr[r + 1];
                    bucket.reserve(hi - lo);
                    std::uint32_t last_token_row =
                        std::numeric_limits<std::uint32_t>::max();
                    for (std::uint32_t k = lo; k < hi; ++k) {
                        const std::uint32_t type = per_slot_type[k];
                        if (!pie_cuda_driver::is_token_sampler(type)) continue;
                        const std::uint32_t row = qo_lo + h_sidx[k];
                        bucket.push_back(static_cast<std::uint32_t>(all_sampled[row]));
                        last_token_row = row;
                    }
                    if (!bucket.empty()) {
                        remember_mtp_source(r, last_token_row);
                    }
                }
            }
            std::vector<SystemSpecDraftRequest> system_draft_requests;
            if (executor.system_drafter) {
                system_draft_requests.reserve(static_cast<std::size_t>(R));
                for (int r = 0; r < R; ++r) {
                    const bool wants_spec =
                        r < static_cast<int>(outspec_view.size()) &&
                        outspec_view[r] != 0;
                    if (!wants_spec) continue;
                    const auto& bucket =
                        per_req[static_cast<std::size_t>(r)].tokens;
                    if (bucket.empty()) continue;
                    const std::int32_t row =
                        mtp_base_rows[static_cast<std::size_t>(r)];
                    if (row < 0) continue;
                    const std::uint32_t draft_pos =
                        mtp_draft_positions[static_cast<std::size_t>(r)];
                    if (draft_pos < 2u) continue;
                    const int last_match =
                        spec_accepted_drafts[static_cast<std::size_t>(r)];
                    const bool has_prior_drafts =
                        has_spec_drafts &&
                        r < static_cast<int>(verify_slot_start.size()) &&
                        verify_slot_start[r] >= 0;
                    const int last_num_drafts =
                        has_prior_drafts ? verify_n_drafts[r] : 0;
                    system_draft_requests.push_back(SystemSpecDraftRequest{
                        .request_index = r,
                        .source_row = row,
                        .accepted_token = bucket.back(),
                        .source_position = draft_pos - 2u,
                        .first_draft_position = draft_pos,
                        .last_match = last_match,
                        .last_num_drafts = last_num_drafts,
                    });
                }
            }

            bool native_commit_cache = false;
            if (!system_draft_requests.empty() &&
                executor.system_drafter.commit_verified_prefix) {
                std::vector<std::uint32_t> mtp_commit_tokens;
                std::vector<std::uint32_t> mtp_commit_positions;
                std::vector<std::uint32_t> mtp_commit_qo(
                    static_cast<std::size_t>(R) + 1, 0);
                std::vector<std::uint32_t> mtp_commit_lpl(
                    static_cast<std::size_t>(R), 0);
                std::vector<std::int32_t> mtp_commit_source_rows;
                mtp_commit_tokens.reserve(static_cast<std::size_t>(N));
                mtp_commit_positions.reserve(static_cast<std::size_t>(N));
                mtp_commit_source_rows.reserve(static_cast<std::size_t>(N));

                auto bump_last_page_len = [&](std::uint32_t lpl,
                                              int extra_tokens) {
                    if (extra_tokens <= 0) return lpl;
                    const int bumped = static_cast<int>(lpl) + extra_tokens;
                    return static_cast<std::uint32_t>(
                        ((bumped - 1) % page_size) + 1);
                };
                for (int r = 0; r < R; ++r) {
                    mtp_commit_qo[static_cast<std::size_t>(r)] =
                        static_cast<std::uint32_t>(mtp_commit_tokens.size());
                    mtp_commit_lpl[static_cast<std::size_t>(r)] =
                        r < static_cast<int>(kvlpl_view_orig.size())
                            ? kvlpl_view_orig[r]
                            : (r < static_cast<int>(kvlpl_view.size())
                                   ? kvlpl_view[r]
                                   : 0u);
                    const bool wants_spec =
                        r < static_cast<int>(outspec_view.size()) &&
                        outspec_view[r] != 0;
                    if (!wants_spec) continue;

                    const int orig_qo_lo =
                        static_cast<int>(qo_view_orig[r]);
                    const int orig_n_in =
                        static_cast<int>(qo_view_orig[r + 1]) - orig_qo_lo;
                    int accepted = 0;
                    if (has_spec_drafts &&
                        r < static_cast<int>(spec_accepted_drafts.size()) &&
                        spec_accepted_drafts[static_cast<std::size_t>(r)] > 0) {
                        accepted =
                            spec_accepted_drafts[static_cast<std::size_t>(r)];
                    }
                    const int commit_len = orig_n_in + accepted;
                    if (commit_len <= 0) continue;

                    const int active_qo_lo = static_cast<int>(qo_view[r]);
                    const int active_qo_hi = static_cast<int>(qo_view[r + 1]);
                    const int bounded_commit_len =
                        std::min(commit_len, active_qo_hi - active_qo_lo);
                    for (int j = 0; j < bounded_commit_len; ++j) {
                        const int row = active_qo_lo + j;
                        mtp_commit_tokens.push_back(tok_view[row]);
                        mtp_commit_positions.push_back(pos_view[row]);
                        mtp_commit_source_rows.push_back(
                            static_cast<std::int32_t>(row));
                    }
                    mtp_commit_lpl[static_cast<std::size_t>(r)] =
                        bump_last_page_len(
                            mtp_commit_lpl[static_cast<std::size_t>(r)],
                            std::max(0, bounded_commit_len - orig_n_in));
                }
                mtp_commit_qo[static_cast<std::size_t>(R)] =
                    static_cast<std::uint32_t>(mtp_commit_tokens.size());

                native_commit_cache =
                    executor.tp_comm == nullptr &&
                    !mtp_commit_tokens.empty();
                if (native_commit_cache) {
                    pi.tokens.copy_from_host(
                        std::span<const std::uint32_t>(mtp_commit_tokens));
                    pi.positions.copy_from_host(
                        std::span<const std::uint32_t>(mtp_commit_positions));
                    pi.qo_indptr.copy_from_host(
                        std::span<const std::uint32_t>(mtp_commit_qo));
                    pi.kv_last_page_lens.copy_from_host(
                        std::span<const std::uint32_t>(mtp_commit_lpl));
                    pi.sample_idx.copy_from_host(
                        std::span<const std::int32_t>(mtp_commit_source_rows));
                    profile_mtp_process_call(
                        "commit", cublas.stream(),
                        static_cast<int>(mtp_commit_tokens.size()), R,
                        [&] {
                            executor.system_drafter.commit_verified_prefix(
                                NativeSystemCommitInputs{
                                    .target_ws = ws,
                                    .kv_cache = kv_cache,
                                    .cublas = cublas,
                                    .token_ids =
                                        reinterpret_cast<const std::int32_t*>(
                                            pi.tokens.data()),
                                    .positions =
                                        reinterpret_cast<const std::int32_t*>(
                                            pi.positions.data()),
                                    .qo_indptr = pi.qo_indptr.data(),
                                    .kv_page_indices =
                                        pi.kv_page_indices.data(),
                                    .kv_page_indptr =
                                        pi.kv_page_indptr.data(),
                                    .kv_last_page_lens =
                                        pi.kv_last_page_lens.data(),
                                    .slot_ids =
                                        use_slots ? pi.slot_ids.data()
                                                  : nullptr,
                                    .source_row_indices = pi.sample_idx.data(),
                                    .total_tokens =
                                        static_cast<int>(
                                            mtp_commit_tokens.size()),
                                    .num_requests = R,
                                });
                        });
                }
            }

            // The driver is pure mechanism: it auto-drafts exactly when the
            // runtime requested it (system_draft_requests, derived from the
            // runtime's output_spec_flags). The decision of WHETHER to speculate
            // this step is the runtime's — see the runtime spec policy.
            if (!system_draft_requests.empty() &&
                executor.system_drafter.draft_next) {
                StepProfileTimer system_spec_timer(
                    "system_drafter", cublas.stream(),
                    static_cast<int>(system_draft_requests.size()), R);
                executor.system_drafter.draft_next(
                    SystemSpecDraftInputs{
                        .target_ws = ws,
                        .kv_cache = kv_cache,
                        .attn_ws = attn_ws,
                        .cublas = cublas,
                        .requests =
                            std::span<const SystemSpecDraftRequest>(
                                system_draft_requests.data(),
                                system_draft_requests.size()),
                        .kv_page_indices = kvpi_view,
                        .kv_page_indptr = kvpp_view,
                        .page_size = kv_cache.page_size(),
                        .max_drafts = executor.system_drafter.max_drafts,
                    },
                    std::span<pie_driver::PerRequestOutput>(
                        per_req.data(), per_req.size()));
                system_spec_timer.finish(cublas.stream());
            } else if (!system_draft_requests.empty() &&
                       executor.system_drafter.draft_step) {
                run_step_chained_system_drafter(
                    executor,
                    std::span<const SystemSpecDraftRequest>(
                        system_draft_requests.data(),
                        system_draft_requests.size()),
                    std::span<pie_driver::PerRequestOutput>(
                        per_req.data(), per_req.size()),
                    native_commit_cache);
            }
            // Advance each committed rs_cache slot from its pre-verify value
            // (the frozen verify left it untouched) to its confirmed prefix
            // [input | accepted] via ACTIVATION REPLAY: replay only the linear-
            // attn recurrence over the accepted tokens, fed from the verify's
            // stashed in-proj activations (mixed_qkv + a,b) — no in_proj GEMM, no
            // attention/MLP, no lm_head. Lossless by construction (the rejected
            // drafts' state is never committed). The MTP-head commit above must
            // precede this (invoke_body overwrites ws). The stash is always
            // configured (entry.cpp), so this is the sole hybrid-SSM spec path.
            if (rs_frozen_verify) {
                executor.rs_cache->set_verify_frozen(false);

                // Replay over the verify window with a per-request commit_len
                // (n_in + accepted): linear_attn_layer_body loads the stashed
                // in-proj activations and SKIPS in_proj, so conv+prep+fla fold
                // only the confirmed prefix and write the committed state at that
                // boundary. No in_proj GEMM means no GEMM-tiling sensitivity, so
                // the replay is bit-consistent with the verify.
                std::vector<std::int32_t> commit_len(static_cast<std::size_t>(R));
                std::vector<std::int32_t> commit_slots(static_cast<std::size_t>(R));
                for (int r = 0; r < R; ++r) {
                    const int n_in = static_cast<int>(
                        qo_view_orig[r + 1] - qo_view_orig[r]);
                    const int accepted =
                        (r < static_cast<int>(spec_accepted_drafts.size()) &&
                         spec_accepted_drafts[static_cast<std::size_t>(r)] > 0)
                            ? spec_accepted_drafts[static_cast<std::size_t>(r)]
                            : 0;
                    commit_len[static_cast<std::size_t>(r)] = n_in + accepted;
                    commit_slots[static_cast<std::size_t>(r)] = slot_ids_h[r];
                }

                if (N > 0) {
                    // Reuse pi.tokens (uint32, unused — no embed) for the int32
                    // commit_len array; restore the full verify qo + slots.
                    pi.tokens.copy_from_host(std::span<const std::uint32_t>(
                        reinterpret_cast<const std::uint32_t*>(commit_len.data()),
                        commit_len.size()));
                    pi.qo_indptr.copy_from_host(std::span<const std::uint32_t>(
                        qo_view.data(), qo_view.size()));
                    pi.slot_ids.copy_from_host(
                        std::span<const std::int32_t>(commit_slots));

                    pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
                    fwd_in.qo_indptr_d   = pi.qo_indptr.data();
                    fwd_in.qo_indptr_h   = qo_view.data();
                    fwd_in.total_tokens  = N;       // full verify window
                    fwd_in.num_requests  = R;
                    fwd_in.is_pure_decode = false;  // N > R (has drafts)
                    fwd_in.slot_ids_h    = commit_slots.data();
                    fwd_in.slot_ids_d    = pi.slot_ids.data();
                    fwd_in.num_logit_rows = -1;
                    fwd_in.commit_advance_gather_d =
                        reinterpret_cast<const std::int32_t*>(pi.tokens.data());
                    forward_fn.invoke_body(ws, kv_cache, attn_ws, cublas,
                                           fwd_in);
                }
            }
            // Ph7 RS working-set fold-buffered (W9 piggyback): for each request
            // flagged RS_FLAG_FOLD, replay the GDN recurrence over its first
            // rs_fold_lens[r] BUFFERED tokens — gathered page-major from the
            // buffered-activation pool (rs_buffer_slot_ids) rather than the
            // verify stash — and write the advanced state into
            // recurrent_state[rs_slot_ids[r]]. Reuses the commit-advance replay
            // (commit_len-clamped conv+prep+fla; no in_proj/attention/MLP/lm_head),
            // just sourced from the pool. RS_FLAG_RESET zeroes the slot first
            // (a first fold replays from zero).
            if (rs_is_fold) {
                executor.rs_cache->set_verify_frozen(false);
                std::vector<std::int32_t> fold_commit_len(
                    static_cast<std::size_t>(R));
                std::vector<std::int32_t> fold_slots(static_cast<std::size_t>(R));
                std::vector<std::uint32_t> fold_qo(
                    static_cast<std::size_t>(R) + 1);
                fold_qo[0] = 0;
                for (int r = 0; r < R; ++r) {
                    const std::uint32_t n =
                        (r < static_cast<int>(rs_fold_view.size()))
                            ? rs_fold_view[r] : 0u;
                    fold_commit_len[static_cast<std::size_t>(r)] =
                        static_cast<std::int32_t>(n);
                    fold_slots[static_cast<std::size_t>(r)] =
                        static_cast<std::int32_t>(rs_slot_view[r]);
                    fold_qo[static_cast<std::size_t>(r) + 1] = fold_qo[r] + n;
                    if ((rs_flag_view[r] & 1u) != 0) {  // RS_FLAG_RESET
                        executor.rs_cache->reset_slot(
                            static_cast<int>(rs_slot_view[r]), cublas.stream());
                    }
                }
                const int fold_N =
                    static_cast<int>(fold_qo[static_cast<std::size_t>(R)]);
                if (fold_N > 0) {
                    pi.tokens.copy_from_host(std::span<const std::uint32_t>(
                        reinterpret_cast<const std::uint32_t*>(
                            fold_commit_len.data()),
                        fold_commit_len.size()));
                    pi.qo_indptr.copy_from_host(std::span<const std::uint32_t>(
                        fold_qo.data(), fold_qo.size()));
                    pi.slot_ids.copy_from_host(
                        std::span<const std::int32_t>(fold_slots));

                    pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
                    fwd_in.qo_indptr_d   = pi.qo_indptr.data();
                    fwd_in.qo_indptr_h   = fold_qo.data();
                    fwd_in.total_tokens  = fold_N;
                    fwd_in.num_requests  = R;
                    fwd_in.is_pure_decode = false;
                    fwd_in.slot_ids_h    = fold_slots.data();
                    fwd_in.slot_ids_d    = pi.slot_ids.data();
                    fwd_in.num_logit_rows = -1;
                    fwd_in.commit_advance_gather_d =
                        reinterpret_cast<const std::int32_t*>(pi.tokens.data());
                    fwd_in.rs_buffer_fold = true;
                    fwd_in.rs_buffer_slot_ids_h = rs_buf_id_view.data();
                    fwd_in.rs_buffer_slot_indptr_h = rs_buf_indptr_view.data();
                    forward_fn.invoke_body(ws, kv_cache, attn_ws, cublas, fwd_in);
                }
            }
            executor.response_builder.build(per_req, out_resp);
            write_probes(out_resp, t_entry, t_wire_parse_end,
                         t_plan_end, t_h2d_end,
                         t_kernel_launch_end, t_sync_end);

            if (executor.verbose && (handled <= 4 || handled % 100 == 0)) {
                std::cerr << "[pie-driver-cuda] req_id=" << req_id
                          << " R=" << R << " N=" << N
                          << " sampled=" << num_sampling
                          << " max_kv=" << max_kv_len << "\n";
            }
            return;
        }

        executor.response_builder.build_token_only(
            std::span<const std::uint32_t>(per_request_counts),
            std::span<const std::uint32_t>(sampled_tokens),
            out_resp);
        write_probes(out_resp, t_entry, t_wire_parse_end,
                     t_plan_end, t_h2d_end,
                     t_kernel_launch_end, t_sync_end);
        if (executor.verbose && (handled <= 4 || handled % 100 == 0)) {
            std::cerr << "[pie-driver-cuda] req_id=" << req_id
                      << " R=" << R << " N=" << N
                      << " sampled=" << num_sampling
                      << " max_kv=" << max_kv_len << "\n";
        }
        return;

    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req_id << ": " << e.what() << "\n";
        out_resp = pie_driver::PieForwardResponseView{};
        return;
    }
}

// ============================================================================
// TP follower service loop
// ============================================================================
//
// Symmetric counterpart of `handle_fire_batch` for ranks > 0:
//
//   * No shmem decode — the inputs arrive via NCCL broadcast from rank 0.
//   * No sampling — only rank 0 owns the response buffer + sampler RNG.
//   * Graph capture/replay mirrors rank 0 for graph-safe pure decode so
//     NCCL collectives inside the body enter capture or replay on every
//     rank in the same order.
//
// The loop blocks on `ncclBroadcast` for the header. NCCL serialises ops
// per-comm, so a follower naturally idles until rank 0 issues the
// matching broadcast in `tp_broadcast_inputs`.
void tp_follower_serve(Executor& executor, std::atomic<bool>& stop) {
    if (executor.tp_comm == nullptr) {
        std::cerr << "[pie-driver-cuda] tp_follower_serve: no tp_comm\n";
        return;
    }
    auto& pi      = executor.inputs;
    auto& comm    = *executor.tp_comm;
    auto* d_hdr   = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    std::uint64_t cpu_gate_seq = 0;

    // Sized lazily; R is at most max_workspace_tokens (one request per token).
    std::vector<std::uint32_t> h_qo, h_kvpp;

    while (!stop.load()) {
        tp_cpu_gate_wait(executor.tp_cpu_gate_key, cpu_gate_seq, stop);
        // 1. Receive header.
        NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader),
                                       ncclChar, 0, comm.comm(), stream),
                         comm.comm());
        TpFireHeader hdr{};
        CUDA_CHECK(cudaMemcpyAsync(&hdr, d_hdr, sizeof(hdr),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (hdr.magic == TP_STOP_MAGIC) break;
        if (hdr.magic == TP_MTP_MAGIC) {
            const int S = hdr.total_tokens;
            const int draft_step = hdr.num_requests;
            NCCL_CHECK(ncclGroupStart());
            if (S > 0) {
                NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                         static_cast<std::size_t>(S) * 4,
                                         ncclChar, 0, comm.comm(), stream));
                NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                         static_cast<std::size_t>(S) * 4,
                                         ncclChar, 0, comm.comm(), stream));
                NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(),
                                         pi.sample_idx.data(),
                                         static_cast<std::size_t>(S) * 4,
                                         ncclChar, 0, comm.comm(), stream));
                NCCL_CHECK(ncclBroadcast(pi.mtp_request_ids.data(),
                                         pi.mtp_request_ids.data(),
                                         static_cast<std::size_t>(S) * 4,
                                         ncclChar, 0, comm.comm(), stream));
            }
            NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
            if (executor.system_drafter.draft_step && S > 0) {
                executor.system_drafter.draft_step(
                    executor.ws, executor.kv_cache, executor.cublas,
                    reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                    reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                    pi.sample_idx.data(),
                    pi.mtp_request_ids.data(),
                    pi.kv_page_indices.data(),
                    pi.kv_page_indptr.data(),
                    pi.kv_last_page_lens.data(),
                    nullptr,
                    S, draft_step, /*max_global_tokens=*/0);
            }
            continue;
        }
        if (hdr.magic != TP_FIRE_MAGIC) {
            std::cerr << "[pie-driver-cuda] tp follower: unexpected header "
                      << "magic 0x" << std::hex << hdr.magic << std::dec
                      << "; aborting\n";
            break;
        }

        const int N = hdr.total_tokens;
        const int R = hdr.num_requests;
        const bool is_pure_decode = (hdr.is_pure_decode != 0);
        const bool tp_greedy_argmax = (hdr.tp_greedy_argmax != 0);
        const int logit_rows = hdr.logit_rows;

        // 2. Receive payloads. Mirror order in `tp_broadcast_inputs`,
        //    grouped so NCCL submits the batch as a single op.
        const bool have_custom_mask = (hdr.mask_bytes > 0);
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                 static_cast<std::size_t>(N) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                 static_cast<std::size_t>(N) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indptr.data(),
                                 pi.kv_page_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        if (R > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                     pi.kv_last_page_lens.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (hdr.kv_indices_count > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                     pi.kv_page_indices.data(),
                                     static_cast<std::size_t>(hdr.kv_indices_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (have_custom_mask) {
            NCCL_CHECK(ncclBroadcast(pi.custom_mask.data(),
                                     pi.custom_mask.data(),
                                     static_cast<std::size_t>(hdr.mask_bytes),
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                     pi.custom_mask_indptr.data(),
                                     static_cast<std::size_t>(hdr.mask_indptr_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        const bool have_slot_ids = (hdr.has_slot_ids != 0) && R > 0;
        if (have_slot_ids) {
            NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                     static_cast<std::size_t>(R),
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (logit_rows > 0) {
            NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                     static_cast<std::size_t>(logit_rows) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());

        // 3. Pull the host views of qo/KV layout for the per-arch
        // attention planner (lives outside the captured kernel sequence).
        h_qo.resize(R + 1);
        h_kvpp.resize(R + 1);
        std::vector<std::uint32_t> h_kvpi(
            static_cast<std::size_t>(std::max(0, hdr.kv_indices_count)));
        std::vector<std::uint32_t> h_kvlpl(R);
        CUDA_CHECK(cudaMemcpyAsync(h_qo.data(), pi.qo_indptr.data(),
                                   static_cast<std::size_t>(R + 1) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_kvpp.data(), pi.kv_page_indptr.data(),
                                   static_cast<std::size_t>(R + 1) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        if (R > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_kvlpl.data(), pi.kv_last_page_lens.data(),
                                       static_cast<std::size_t>(R) * 4,
                                       cudaMemcpyDeviceToHost, stream));
        }
        if (!h_kvpi.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(
                h_kvpi.data(), pi.kv_page_indices.data(),
                h_kvpi.size() * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost, stream));
        }
        std::vector<std::int32_t> h_slot_ids;
        std::vector<std::uint8_t> h_is_fresh;
        if (have_slot_ids) {
            h_slot_ids.resize(R);
            h_is_fresh.resize(R);
            CUDA_CHECK(cudaMemcpyAsync(h_slot_ids.data(), pi.slot_ids.data(),
                                       static_cast<std::size_t>(R) * 4,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_is_fresh.data(), pi.is_fresh.data(),
                                       static_cast<std::size_t>(R),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 4. Run the same forward function as rank 0. The all-reduces
        // inside synchronise both ranks; we don't sample or write a
        // response — that's rank-0-only.
        executor.forward_fn.invoke_prepare(
            executor.attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = h_qo.data(),
                .kv_page_indices_h = h_kvpi.data(),
                .kv_page_indices_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
                .kv_page_indptr_h = h_kvpp.data(),
                .kv_page_indptr_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
                .kv_last_page_lens_h = h_kvlpl.data(),
                .kv_last_page_lens_d =
                    reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
                .total_tokens = N,
                .num_requests = R,
                .is_pure_decode = is_pure_decode,
            });
        // Mirror rank 0's graph capture/replay decision so NCCL ops
        // inside the body record on both ranks simultaneously (otherwise
        // rank 0 would record while rank 1 executes, deadlocking the
        // first capture). The same `(R)` shape key keeps the per-rank
        // graph caches in lockstep; the captured graph on rank 1 has no
        // sampling / response work, just the forward kernels + NCCL.
        const bool try_graphs =
            executor.graph_cache != nullptr && is_pure_decode && !have_custom_mask
            && executor.forward_fn.graph_safe;
        const std::uint32_t graph_layout =
            executor.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            (tp_greedy_argmax ? 1u : 0u) |
            (graph_layout << 3);
        if (try_graphs) {
            const ForwardGraphKey key{R, N, graph_variant};
            cudaGraphExec_t exec = executor.graph_cache->get(key);
            if (exec == nullptr) {
                exec = capture_forward_graph_exec(
                    executor, h_qo.data(), h_kvpi.data(), h_kvpp.data(),
                    h_kvlpl.data(),
                    N, R, is_pure_decode,
                    have_slot_ids ? h_slot_ids.data() : nullptr,
                    have_slot_ids ? h_is_fresh.data() : nullptr,
                    have_slot_ids ? pi.slot_ids.data() : nullptr,
                    logit_rows > 0 ? pi.sample_idx.data() : nullptr,
                    logit_rows,
                    /*single_gpu_greedy_argmax=*/false,
                    tp_greedy_argmax);
                executor.graph_cache->put(key, exec);
            }
            CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
        } else {
            pie_cuda_driver::ForwardFn::ForwardInputs fwd_in;
            fwd_in.token_ids = reinterpret_cast<const std::int32_t*>(pi.tokens.data());
            fwd_in.positions = reinterpret_cast<const std::int32_t*>(pi.positions.data());
            fwd_in.qo_indptr_d         = pi.qo_indptr.data();
            fwd_in.kv_page_indices_d   = pi.kv_page_indices.data();
            fwd_in.kv_page_indptr_d    = pi.kv_page_indptr.data();
            fwd_in.kv_last_page_lens_d = pi.kv_last_page_lens.data();
            fwd_in.qo_indptr_h         = h_qo.data();
            fwd_in.kv_page_indices_h   = h_kvpi.data();
            fwd_in.kv_page_indptr_h    = h_kvpp.data();
            fwd_in.kv_last_page_lens_h = h_kvlpl.data();
            fwd_in.total_tokens        = N;
            fwd_in.num_requests        = R;
            fwd_in.is_pure_decode      = is_pure_decode;
            fwd_in.custom_mask_d        = have_custom_mask ? pi.custom_mask.data()        : nullptr;
            fwd_in.custom_mask_indptr_d = have_custom_mask ? pi.custom_mask_indptr.data() : nullptr;
            fwd_in.slot_ids_h          = have_slot_ids ? h_slot_ids.data() : nullptr;
            fwd_in.is_fresh_h          = have_slot_ids ? h_is_fresh.data() : nullptr;
            fwd_in.slot_ids_d          = have_slot_ids ? pi.slot_ids.data() : nullptr;
            fwd_in.logit_row_indices_d = logit_rows > 0 ? pi.sample_idx.data() : nullptr;
            fwd_in.num_logit_rows      = logit_rows;
            fwd_in.tp_greedy_argmax    = tp_greedy_argmax;
            executor.forward_fn.invoke_body(
                executor.ws, executor.kv_cache, executor.attn_ws, executor.cublas,
                fwd_in);
        }
    }
}

void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key) {
    tp_cpu_gate_notify(cpu_gate_key);
    auto* d_hdr = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    TpFireHeader hdr{TP_STOP_MAGIC, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                                   comm.comm(), stream),
                     comm.comm());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace pie_cuda_driver
