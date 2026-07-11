#include "executor/executor.hpp"
#include "executor/graph_variant.hpp"

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
#include "kernels/argmax.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "kv_cache.hpp"
#include "recurrent_state_cache.hpp"
#include "model/imodel.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "ptir/batch_compose.hpp"

#include "kernels/pack_dense_mask.hpp"
#include "kernels/sample_temp.hpp"

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

// #24 graph-variant bitfield helper: `make_graph_variant()` + the named flag
// constants + the boundary static_asserts now live in
// "executor/graph_variant.hpp" (so they're unit-testable host-side).

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

int tensor_rows(const DeviceTensor& t) {
    if (t.shape().empty()) return 0;
    return static_cast<int>(t.shape()[0]);
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
            make_graph_variant(tp_greedy_argmax, single_gpu_graph_argmax,
                               fwd_handles_argmax_precapture,
                               /*small_spec=*/false, /*rs_verify=*/false,
                               graph_layout);
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
    // Explicit KV-write descriptor present (device-geometry WSlot/WOff, B2).
    // When set, the forward routes the per-layer KV append through the explicit
    // (physical page, offset) kernel from pi.w_page/pi.w_off.
    bool has_write_desc = false;
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
    const std::uint32_t graph_variant = make_graph_variant(
        in.tp_greedy_argmax, out.graph_captures_single_gpu_argmax,
        in.forward_handles_argmax, in.small_spec_graph_shape,
        // Frozen verify skips the GDN state writeback, baked into the captured
        // kernel — must not share a graph with a normal-writeback forward of
        // the same shape.
        rs_verify_frozen, out.graph_layout);

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
        fwd_in.w_page_d             = in.has_write_desc ? pi.w_page.data() : nullptr;
        fwd_in.w_off_d              = in.has_write_desc ? pi.w_off.data()  : nullptr;
        fwd_in.has_write_desc       = in.has_write_desc;
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

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_native::LaunchView& view,
    Executor& executor,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion)
{
    using clock = std::chrono::steady_clock;
    const auto t_entry = clock::now();

    // Diagnostic trace for the direct launch path.
    const bool ir_trace = std::getenv("PIE_SAMPLING_IR_TRACE") != nullptr;
    if (ir_trace) {
        std::cerr << "[ir-trace] fire entry req_id=" << req_id << "\n";
        std::cerr.flush();
    }
    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr, "[ptir-serve] entry req_id=%u: ptir_hashes=%zu tokens=%zu\n",
                     req_id, view.ptir_program_hashes.size(), view.token_ids.size());
    }

    // Local references for the most-touched Executor members.
    auto& ws                   = executor.ws;
    auto& kv_cache             = executor.kv_cache;
    auto& attn_ws              = executor.attn_ws;
    auto& cublas               = executor.cublas;
    auto& pi                   = executor.inputs;  // persistent input slabs
    auto& forward_fn           = executor.forward_fn;
    const int max_workspace_tokens = executor.max_workspace_tokens;

    // Track whether the custom-mask path was populated this fire so the
    // forward kernel knows whether to consume `pi.custom_mask`. Sizes are
    // stashed alongside so the TP broadcast knows how many bytes to fan
    // out to followers.
    bool have_custom_mask = false;
    bool has_write_desc = false;
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
        const auto kvpi_view_wire = view.kv_page_indices.as<std::uint32_t>();
        const auto kvpp_view_wire = view.kv_page_indptr.as<std::uint32_t>();
        const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();

        const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
        const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();

        // ── W1.1: pre-forward device-geometry descriptor resolution ──────
        // EVERY device-geometry PTIR program in the batch (WSlot/WOff write
        // descriptors + a channel-bound [B, P>1] Pages port — the runtime's
        // `detect_device_geometry` mirror) ships EMPTY wire geometry; the
        // driver reads its port channels at fire time and COMPOSES the
        // resolved geometries with the wire programs' launch slices into one
        // flat forward batch (batch_compose.hpp) — no program-specific
        // assembly (owner constraint §3.1). A not-ready descriptor channel
        // fails the fire (W1.6). Pure-wire batches resolve nothing
        // (dg_resolved = false, empty *err) and use the wire geometry
        // unchanged.
        ptir::ResolvedPrograms rpg;
        ptir::ComposedBatch composed;
        // Per-PROGRAM offsets of each program's sampled rows within the
        // gathered logits buffer (`n_prog + 1` entries) — what
        // `PtirDispatch::run` slices each program's logits base from.
        std::vector<std::uint32_t> prog_sample_offsets;
        bool dg_resolved = false;
        if (!view.ptir_program_hashes.empty()) {
            if (!executor.ptir_dispatch)
                executor.ptir_dispatch = std::make_unique<ptir::PtirDispatch>();
            std::string dg_err;
            dg_resolved = executor.ptir_dispatch->resolve_descriptors(
                view,
                static_cast<std::uint32_t>(kv_cache.page_size()),
                static_cast<std::uint32_t>(kv_cache.num_pages()),
                rpg,
                &dg_err);
            if (!dg_resolved && !dg_err.empty()) {
                throw std::runtime_error(dg_err);
            }
            if (dg_resolved) {
                // v1 mask scope: a dense device mask (AttnMask channel)
                // composes only SOLO — the runtime scheduler batches such
                // fires alone; fail loud if the contract is violated.
                if (rpg.per_program.size() > 1) {
                    for (std::size_t p = 0; p < rpg.per_program.size(); ++p) {
                        if (rpg.is_device_geometry[p] &&
                            rpg.per_program[p].has_mask) {
                            throw std::runtime_error(
                                "ptir: dense device mask in a multi-program "
                                "batch (scheduler contract violated)");
                        }
                    }
                }
                std::string compose_err;
                if (!ptir::compose_forward_batch(
                        view, rpg,
                        static_cast<std::uint32_t>(kv_cache.page_size()),
                        composed, &compose_err)) {
                    throw std::runtime_error(compose_err);
                }
                prog_sample_offsets = composed.prog_sample_offsets;
            } else {
                std::string offs_err;
                if (!ptir::wire_program_sample_offsets(
                        view, prog_sample_offsets, &offs_err)) {
                    throw std::runtime_error(offs_err);
                }
            }
        }
        // Only the SOLO device-geometry fire may carry a dense device mask;
        // its resolved geometry equals the composed batch.
        const ptir::FireGeometry* solo_fg =
            (dg_resolved && rpg.per_program.size() == 1 &&
             rpg.is_device_geometry[0])
                ? &rpg.per_program[0]
                : nullptr;

        // Composed geometry takes precedence over the borrowed launch
        // slices. No request/speculation carrier is expanded in the driver.
        const int R = static_cast<int>(
            dg_resolved ? composed.qo_indptr.size() : qo_view_orig.size()) - 1;

        const std::span<const std::uint32_t> tok_view   = dg_resolved ? std::span<const std::uint32_t>(composed.token_ids)         : tok_view_orig;
        const std::span<const std::uint32_t> pos_view   = dg_resolved ? std::span<const std::uint32_t>(composed.position_ids)      : pos_view_orig;
        const std::span<const std::uint32_t> qo_view    = dg_resolved ? std::span<const std::uint32_t>(composed.qo_indptr)         : qo_view_orig;
        const std::span<const std::uint32_t> kvpi_view  = dg_resolved ? std::span<const std::uint32_t>(composed.kv_page_indices)   : kvpi_view_wire;
        const std::span<const std::uint32_t> kvpp_view  = dg_resolved ? std::span<const std::uint32_t>(composed.kv_page_indptr)    : kvpp_view_wire;
        const std::span<const std::uint32_t> kvlpl_view = dg_resolved ? std::span<const std::uint32_t>(composed.kv_last_page_lens) : kvlpl_view_orig;
        const std::span<const std::uint32_t> sidx_view  = dg_resolved ? std::span<const std::uint32_t>(composed.sampling_indices)  : sidx_view_orig;
        const std::span<const std::uint32_t> sptr_view  = dg_resolved ? std::span<const std::uint32_t>(composed.sampling_indptr)   : sptr_view_orig;

        const auto t_wire_parse_end = clock::now();

        const int N = static_cast<int>(tok_view.size());
        const int num_sampling = static_cast<int>(sidx_view.size());
        dbg_R = R; dbg_N = N;
        if (ir_trace) {
            std::cerr << "[ir-trace] fire shape req_id=" << req_id
                      << " N=" << N << " R=" << R
                      << " num_sampling=" << num_sampling << "\n";
            std::cerr.flush();
        }

        // Qwen3-VL: assemble the per-token [N,3] M-RoPE positions. Text rows
        // carry (p,p,p) from the 1-D `pos_view`; image-token rows are
        // overwritten with the staged 3-axis (t,h,w) positions for each image
        // (image i's rows start at batch row `img_anchor[i]`). Built only when
        // image mrope data is present (image prefills never carry spec drafts).
        if (img_num_images > 0 && !img_mrope_pos.empty()) {
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
            if (!executor.ptir_dispatch)
                executor.ptir_dispatch = std::make_unique<ptir::PtirDispatch>();
            executor.ptir_dispatch->run(
                view, nullptr, 0, executor.cublas.stream(), &runtime, completion);
            return;
        }
        if (N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            throw std::runtime_error("forward batch exceeds workspace capacity");
        }

        // Detect pure decode so the model can choose its decode kernel.
        const std::uint32_t* h_kvpp  = kvpp_view.data();
        const std::uint32_t* h_kvlpl = kvlpl_view.data();
        const std::uint32_t* h_qo    = qo_view.data();
        bool is_pure_decode = (R > 0);
        for (int r = 0; r < R; ++r) {
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
        // Ph7 RS working-set buffered-activation channel. Single-role per pass
        // (v1): a FOLD pass (FOLD-bit=2 set) gathers+replays from the buffered
        // pool into recurrent_state (separate fold-replay dispatch below); an
        // rs-output write pass (FOLD-bit clear + buffered slabs present)
        // scatters in-proj [mixed_qkv|a|b] to the pool during the main forward.
        const auto rs_buf_id_view = view.rs_buffer_slot_ids.as<std::uint32_t>();
        const auto rs_buf_indptr_view = view.rs_buffer_slot_indptr.as<std::uint32_t>();
        const bool rs_is_fold = use_slots && std::any_of(
            rs_flag_view.begin(), rs_flag_view.end(),
            [](std::uint8_t v) { return (v & 2u) != 0; });
        const bool rs_is_write =
            use_slots && !rs_is_fold && rs_buf_id_view.size() > 0;

        // Direct PTIR launches keep the forward geometry exact. The program runs
        // after the model and therefore cannot use the legacy graph variants that
        // fused sampling into the captured forward.
        ForwardInputViews forward_inputs = make_forward_input_views(
            tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view,
            R, R, -1);
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

        // BRLE attention masks. For any batch that isn't pure causal, decode +
        // upload a packed bitmap and route through the flashinfer kCustom path.
        // NOTE: `is_pure_decode` is intentionally NOT gated here. A decode-shaped
        // batch (qo_len==1/req) that carries a per-cell custom mask — e.g. the
        // §6.2 beam fire, whose kvm expresses fork-freeze mid-page holes the
        // decode/xqa kernels can't — must ALSO build the mask; the forward then
        // routes it through the custom-mask prefill kernel (llama_like's
        // `has_custom_mask` gate). This is the previously-noted "route decode
        // through the prefill kernel for custom-mask inferlets" fix; a normal
        // decode batch carries no mask (`fmask_view` empty) so it is unaffected.
        // Wire BRLE masks are indexed by the WIRE request layout, so the
        // causality check and decode run against the WIRE spans (identical to
        // the selected spans on a pure-wire batch). A composed batch never
        // carries a custom wire mask (the runtime scheduler keeps custom-mask
        // wire fires and device-geometry fires apart — fail loud otherwise);
        // pure-causal wire masks are simply dropped, as before.
        const auto fmask_view  = view.flattened_masks.as<std::uint32_t>();
        const auto mskptr_view = view.mask_indptr.as<std::uint32_t>();
        if (!fmask_view.empty()) {
            const auto qo_span = std::span<const std::uint32_t>(
                qo_view_orig.data(), qo_view_orig.size());
            const auto kvpp_span = std::span<const std::uint32_t>(
                kvpp_view_wire.data(), kvpp_view_wire.size());
            const auto kvlpl_span = std::span<const std::uint32_t>(
                kvlpl_view_orig.data(), kvlpl_view_orig.size());
            if (!pie_cuda_driver::brle::is_pure_causal(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size())) {
                if (dg_resolved) {
                    // A MULTI-program batch cannot honor wire BRLE masks
                    // (they index the wire request layout; the scheduler
                    // batches mask-carrying fires solo — fail loud if not).
                    // On a SOLO device-geometry fire the wire rows are
                    // engine-SYNTHESIZED causal (a guest's mask is the DENSE
                    // channel mask, packed below) and simply drop — the
                    // resolved geometry runs the standard causal path.
                    if (view.has_user_mask &&
                        view.ptir_program_hashes.size() > 1) {
                        throw std::runtime_error(
                            "ptir: custom wire masks cannot co-batch with "
                            "device-geometry programs (scheduler contract "
                            "violated)");
                    }
                } else {
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
        }

        // ── W1.3: device-geometry AttnMask → FlashInfer packed custom mask ──
        // A device-geometry fire may carry a DENSE [lanes, stride] per-cell mask
        // on its AttnMask descriptor port (resolved into fg.mask). Pack it to
        // FlashInfer's bit-packed custom mask (launch_pack_dense_mask) INTO
        // pi.custom_mask, so the standard custom-mask forward path consumes it
        // exactly like a BRLE-decoded wire mask. DORMANT unless a device-geometry
        // program binds an AttnMask channel (fg.has_mask); the guest producer is
        // W2.1. Correctness is validated once a real device-geometry fire exists.
        if (solo_fg != nullptr && solo_fg->has_mask && !solo_fg->mask.empty()) {
            const ptir::FireGeometry& fg = *solo_fg;
            const int lanes = static_cast<int>(qo_view.size()) - 1;
            // Total query rows = qo_indptr.back(). For a 1-query/lane decode this
            // equals `lanes`; for a variable-length prefill a single lane carries
            // N query rows, so the dense mask is [TOTAL_Q, STRIDE] (one row per
            // QUERY token), STRIDE = mask.size()/TOTAL_Q.
            const int total_q =
                lanes > 0 ? static_cast<int>(qo_view[lanes]) : 0;
            if (lanes > 0 && total_q > 0 &&
                fg.mask.size() % static_cast<std::size_t>(total_q) == 0) {
                const int stride =
                    static_cast<int>(fg.mask.size() / static_cast<std::size_t>(total_q));
                const std::uint32_t page =
                    static_cast<std::uint32_t>(kv_cache.page_size());
                // Per-lane physical KV span klen[l] from the resolved page geometry,
                // and the packed byte-offset CSR (ceil(qo_len[l]·klen[l]/8) per lane).
                std::vector<std::uint32_t> klen(static_cast<std::size_t>(lanes), 0);
                std::vector<std::int32_t> mindptr(static_cast<std::size_t>(lanes) + 1, 0);
                for (int l = 0; l < lanes; ++l) {
                    const std::uint32_t np =
                        (l + 1 < static_cast<int>(fg.kv_page_indptr.size()))
                            ? fg.kv_page_indptr[l + 1] - fg.kv_page_indptr[l] : 0u;
                    const std::uint32_t lpl =
                        (l < static_cast<int>(fg.kv_last_page_lens.size()))
                            ? fg.kv_last_page_lens[l] : 0u;
                    klen[l] = np == 0 ? 0u : (np - 1) * page + lpl;
                    const std::uint32_t qo_len =
                        qo_view[l + 1] - qo_view[l];
                    const std::uint64_t bits =
                        static_cast<std::uint64_t>(qo_len) * klen[l];
                    mindptr[l + 1] = mindptr[l] +
                        static_cast<std::int32_t>((bits + 7u) / 8u);
                }
                const std::size_t packed_bytes =
                    static_cast<std::size_t>(mindptr[lanes]);
                if (packed_bytes > 0 &&
                    packed_bytes <= pi.custom_mask.size() &&
                    static_cast<std::size_t>(lanes) + 1 <= pi.custom_mask_indptr.size()) {
                    auto kvm_dev = DeviceBuffer<std::uint8_t>::from_bytes(
                        std::span<const std::uint8_t>(fg.mask));
                    auto klen_dev = DeviceBuffer<std::uint32_t>::from_host(
                        std::span<const std::uint32_t>(klen));
                    auto qo_dev = DeviceBuffer<std::uint32_t>::from_host(
                        std::span<const std::uint32_t>(qo_view.data(), qo_view.size()));
                    pi.custom_mask_indptr.copy_from_host(
                        std::span<const std::int32_t>(mindptr));
                    CUDA_CHECK(cudaMemsetAsync(pi.custom_mask.data(), 0,
                                               packed_bytes, cublas.stream()));
                    kernels::launch_pack_dense_mask(
                        kvm_dev.data(), klen_dev.data(), qo_dev.data(),
                        pi.custom_mask_indptr.data(), pi.custom_mask.data(),
                        lanes, stride, cublas.stream());
                    have_custom_mask = true;
                    mask_bytes = static_cast<int>(packed_bytes);
                    mask_indptr_count = lanes + 1;
                }
            }
        }

        // Explicit KV-write descriptor upload (device-geometry WSlot/WOff, B2).
        // Parallels the mask pack above: when any composed program bound
        // WSlot/WOff ports, the composition carries per-TOKEN physical page
        // ids + offsets for EVERY batch row (device-geometry rows from their
        // translated descriptors; wire rows synthesized to their standard
        // append target — `has_write_desc` routes the whole forward's
        // per-layer KV append through launch_write_kv_explicit_bf16, so every
        // row needs a target). Beam fork/freeze correctness: a frozen fork's
        // cell is not overwritten (a sibling's mask hides it).
        if (dg_resolved && composed.has_write_desc && !composed.w_page.empty()) {
            if (composed.w_page.size() != composed.w_off.size() ||
                composed.w_page.size() > pi.w_page.size()) {
                throw std::runtime_error(
                    "ptir: composed write descriptor exceeds persistent "
                    "input capacity");
            }
            pi.w_page.copy_from_host(
                std::span<const std::uint32_t>(composed.w_page));
            pi.w_off.copy_from_host(
                std::span<const std::uint32_t>(composed.w_off));
            has_write_desc = true;
        }

        // Linear-attention rs_cache slots. Runtime owns slot assignment;
        // RS-capable models must receive one slot id per request.
        std::vector<std::int32_t> slot_ids_h;
        std::vector<std::uint8_t> is_fresh_h;
        if (use_slots) {
            const int slot_count = R;
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

        if (executor.rs_cache != nullptr) {
            executor.rs_cache->set_verify_frozen(false);
        }

        if (sptr_view.size() != static_cast<std::size_t>(R + 1) ||
            sptr_view.back() != sidx_view.size()) {
            throw std::runtime_error("sampling CSR does not match launched instances");
        }
        std::vector<std::int32_t> sample_rows;
        sample_rows.reserve(sidx_view.size());
        const std::uint32_t* h_sptr = sptr_view.data();
        const std::uint32_t* h_sidx = sidx_view.data();
        for (int r = 0; r < R; ++r) {
            const std::uint32_t qo_begin = h_qo[r];
            const std::uint32_t qo_len = h_qo[r + 1] - qo_begin;
            for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k) {
                if (h_sidx[k] >= qo_len) {
                    throw std::runtime_error("sampling row exceeds request query span");
                }
                sample_rows.push_back(
                    static_cast<std::int32_t>(qo_begin + h_sidx[k]));
            }
        }
        if (sample_rows.size() > pi.sample_idx.size()) {
            throw std::runtime_error("sampling rows exceed persistent input capacity");
        }
        if (N > tensor_rows(ws.logits)) {
            throw std::runtime_error("forward batch exceeds logits workspace");
        }
        if (!sample_rows.empty()) {
            pi.sample_idx.copy_from_host(
                std::span<const std::int32_t>(sample_rows));
        }

        const auto t_plan_end = clock::now();

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
                                /*tp_greedy_argmax=*/false,
                                /*logit_rows=*/0,
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

        forward_fn.invoke_set_logits_argmax_only(false);
        forward_fn.invoke_set_fused_argmax_output(nullptr);
        const auto t_h2d_end = clock::now();

        // ── Forward pass ────────────────────────────────────────
        StepProfileTimer verify_timer(
            "verify", cublas.stream(), forward_N, forward_R);
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-begin req_id=" << req_id
                      << " forward_N=" << forward_N
                      << " forward_R=" << forward_R << "\n";
            std::cerr.flush();
        }
        auto dump_rs = [&](const char* tag) {
            if (!std::getenv("PIE_RS_TRACE") || executor.rs_cache == nullptr ||
                !use_slots || R < 1) return;
            const int slot = static_cast<int>(rs_slot_view[0]);
            std::uint32_t rw[4] = {0, 0, 0, 0}, cw[4] = {0, 0, 0, 0};
            cudaMemcpy(rw, executor.rs_cache->recurrent_state_raw(0, slot),
                       sizeof(rw), cudaMemcpyDeviceToHost);
            cudaMemcpy(cw, executor.rs_cache->conv_state(0, slot),
                       sizeof(cw), cudaMemcpyDeviceToHost);
            std::cerr << "[rs-trace] " << tag << " req_id=" << req_id
                      << " slot=" << slot
                      << " bf16=" << executor.rs_cache->recurrent_state_bf16()
                      << " N=" << N << " rs_is_fold=" << rs_is_fold
                      << " rs_is_write=" << rs_is_write << std::hex
                      << " recur16B=" << rw[0] << "," << rw[1] << "," << rw[2] << "," << rw[3]
                      << " conv16B=" << cw[0] << "," << cw[1] << "," << cw[2] << "," << cw[3]
                      << std::dec << "\n";
        };
        dump_rs("PRE ");

        const ForwardDispatchResult fd = run_forward_dispatch(
            executor, ForwardDispatchInputs{
                .R = R,
                .forward_R = forward_R,
                .forward_N = forward_N,
                .N = N,
                .num_sampling = num_sampling,
                .is_pure_decode = is_pure_decode,
                .have_custom_mask = have_custom_mask,
                .has_write_desc = has_write_desc,
                .small_spec_graph_shape = false,
                .graph_shape_ok = false,
                .tp_greedy_argmax = false,
                .forward_handles_argmax = false,
                .single_gpu_greedy_argmax = false,
                .compact_logit_rows = false,
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
        static_cast<void>(fd);
        dump_rs("POST");
        if (ir_trace) {
            std::cerr << "[ir-trace] forward-returned req_id=" << req_id << "\n";
            std::cerr.flush();
        }
        verify_timer.finish(cublas.stream());
        const auto t_kernel_launch_end = clock::now();
        static_cast<void>(t_entry);
        static_cast<void>(t_wire_parse_end);
        static_cast<void>(t_plan_end);
        static_cast<void>(t_h2d_end);
        static_cast<void>(t_kernel_launch_end);

        if (view.ptir_program_hashes.empty()) {
            throw std::runtime_error(
                "legacy sampler launches are removed; direct PTIR launch required");
        }
        if (!executor.ptir_dispatch)
            executor.ptir_dispatch = std::make_unique<ptir::PtirDispatch>();
        const std::uint32_t vocab = static_cast<std::uint32_t>(
            executor.loaded_model.hf_config().vocab_size);
        const std::size_t n_conv =
            static_cast<std::size_t>(num_sampling) * vocab;
        const void* ptir_logits = nullptr;
        if (n_conv > 0) {
            if (executor.ptir_logits_bf16.size() < n_conv) {
                executor.ptir_logits_bf16 =
                    DeviceBuffer<std::uint16_t>::alloc(n_conv);
            }
            if (executor.ptir_logits_f32.size() < n_conv) {
                executor.ptir_logits_f32 = DeviceBuffer<float>::alloc(n_conv);
            }
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(executor.ws.logits.data()),
                executor.inputs.sample_idx.data(),
                executor.ptir_logits_bf16.data(),
                num_sampling,
                static_cast<int>(vocab),
                executor.cublas.stream());
            kernels::launch_cast_bf16_to_fp32(
                executor.ptir_logits_bf16.data(),
                executor.ptir_logits_f32.data(),
                n_conv,
                executor.cublas.stream());
            ptir_logits = executor.ptir_logits_f32.data();
        }
        // The post-forward dispatch slices each program's logits base from
        // `sampling_indptr[p]` — hand it the PER-PROGRAM gathered-row offsets
        // (`n_prog + 1` entries), not the per-request sampling CSR (the two
        // coincided only while every batched program was exactly one wire
        // request).
        pie_native::LaunchView dispatch_view = view;
        dispatch_view.sampling_indices = pie_native::slice_from_u32(
            sidx_view.data(), sidx_view.size());
        dispatch_view.sampling_indptr = pie_native::slice_from_u32(
            prog_sample_offsets.data(), prog_sample_offsets.size());
        executor.ptir_dispatch->run(
            dispatch_view, ptir_logits, vocab, executor.cublas.stream(),
            &runtime, completion);
        return;

    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req_id << ": " << e.what() << "\n";
        throw;
    }
}

// ============================================================================
// TP follower service loop
// ============================================================================
//
// Symmetric counterpart of `handle_fire_batch` for ranks > 0:
//
//   * Inputs arrive via NCCL broadcast from rank 0.
//   * No sampling — only rank 0 owns the direct PTIR publish path.
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

        // 4. Run the same forward function as rank 0. Channel publication is
        // rank-0-only after the collectives complete.
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
        // PTIR publication, just the forward kernels + NCCL.
        const bool try_graphs =
            executor.graph_cache != nullptr && is_pure_decode && !have_custom_mask
            && executor.forward_fn.graph_safe;
        const std::uint32_t graph_layout =
            executor.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(tp_greedy_argmax, /*single_gpu=*/false,
                               /*fwd_handles=*/false, /*small_spec=*/false,
                               /*rs_verify=*/false, graph_layout);
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
