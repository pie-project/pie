#include "batch/forward.hpp"
#include "batch/graph_variant.hpp"

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

#include "ops/attention_workspace.hpp"
#include "kernels/custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/argmax.hpp"
#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "store/kv_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "model/imodel.hpp"
#include "model/stage_hooks.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"

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
    supports_runtime_window       = caps.supports_runtime_window;
}

void ForwardFn::invoke_prepare(AttentionWorkspace& aws,
                               const PrepareInputs& in) {
    if (model) model->prepare(aws, in);
}

void ForwardFn::invoke_body(model::Workspace& ws,
                            KvCache& kv,
                            AttentionWorkspace& aws,
                            ops::CublasHandle& cublas,
                            const ForwardInputs& in) {
    if (model) {
        model::ScopedStageHooks hooks(in.stage_hooks);
        model->body(ws, kv, aws, cublas, in);
    }
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
// "batch/graph_variant.hpp" (so they're unit-testable host-side).

bool graph_single_gpu_argmax_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_GRAPH_ARGMAX");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

class CudaStreamOwner {
      public:
        CudaStreamOwner() {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &stream_, cudaStreamNonBlocking));
        }
        ~CudaStreamOwner() noexcept {
            if (stream_ != nullptr) cudaStreamDestroy(stream_);
        }
        CudaStreamOwner(const CudaStreamOwner&) = delete;
        CudaStreamOwner& operator=(const CudaStreamOwner&) = delete;
        cudaStream_t get() const noexcept { return stream_; }

      private:
        cudaStream_t stream_ = nullptr;
    };

    class CudaGraphOwner {
      public:
        explicit CudaGraphOwner(cudaGraph_t graph = nullptr) : graph_(graph) {}
        ~CudaGraphOwner() noexcept {
            if (graph_ != nullptr) cudaGraphDestroy(graph_);
        }
        CudaGraphOwner(const CudaGraphOwner&) = delete;
        CudaGraphOwner& operator=(const CudaGraphOwner&) = delete;
        cudaGraph_t get() const noexcept { return graph_; }

      private:
        cudaGraph_t graph_ = nullptr;
    };

    class CudaGraphExecOwner {
      public:
        CudaGraphExecOwner() = default;
        ~CudaGraphExecOwner() noexcept {
            if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
        }
        CudaGraphExecOwner(const CudaGraphExecOwner&) = delete;
        CudaGraphExecOwner& operator=(const CudaGraphExecOwner&) = delete;
        cudaGraphExec_t* out() noexcept { return &exec_; }
        cudaGraphExec_t get() const noexcept { return exec_; }
        cudaGraphExec_t release() noexcept {
            const cudaGraphExec_t result = exec_;
            exec_ = nullptr;
            return result;
        }

      private:
        cudaGraphExec_t exec_ = nullptr;
    };

    class CublasStreamScope {
      public:
        explicit CublasStreamScope(ops::CublasHandle& handle)
            : handle_(handle), previous_(handle.stream()) {}
        void bind(cudaStream_t stream) {
            handle_.set_stream(stream);
        }
        ~CublasStreamScope() noexcept {
            if (!active_) return;
            try {
                handle_.set_stream(previous_);
            } catch (...) {
            }
        }
        CublasStreamScope(const CublasStreamScope&) = delete;
        CublasStreamScope& operator=(const CublasStreamScope&) = delete;
        void restore() {
            handle_.set_stream(previous_);
            active_ = false;
        }

      private:
        ops::CublasHandle& handle_;
        cudaStream_t previous_ = nullptr;
        bool active_ = true;
};

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

// CPU-side barrier that keeps every TP rank's graph-capture calls in
// lockstep: NCCL collectives inside the captured region require every rank
// to enter `cudaStreamBeginCapture`/`EndCapture` at the same logical point,
// or the collective ops deadlock waiting on a peer that hasn't started
// capturing yet.
void tp_graph_capture_barrier(const BatchEngine& engine) {
    if (engine.tp_comm == nullptr) return;
    if (engine.tp_cpu_gate_key.empty()) return;
    const int world = engine.tp_comm->world_size();
    if (world <= 1) return;

    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<std::barrier<>>>
        registry;

    std::shared_ptr<std::barrier<>> b;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[engine.tp_cpu_gate_key + ":graph_capture"];
        if (!entry) entry = std::make_shared<std::barrier<>>(world);
        b = entry;
    }
    b->arrive_and_wait();
}

}  // namespace

bool step_profile_take() {
    if (!step_profile_enabled()) return false;
    static std::atomic<std::uint64_t> seq{0};
    return seq.fetch_add(1, std::memory_order_relaxed) < step_profile_limit();
}

cudaGraphExec_t capture_forward_graph_exec(
    BatchEngine& engine,
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
    bool tp_greedy_argmax,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    bool has_write_desc,
    int runtime_window_left)
{
    auto& pi = engine.inputs;

    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    CudaStreamOwner capture_stream;
    const cudaStream_t cstream = capture_stream.get();
    CublasStreamScope cublas_stream(engine.cublas);
    cublas_stream.bind(cstream);
    StreamCaptureGuard capture_guard(cstream);
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
        fwd_in.w_page_d = w_page_d;
        fwd_in.w_off_d = w_off_d;
        fwd_in.has_write_desc = has_write_desc;
        fwd_in.runtime_window_left = runtime_window_left;
        engine.forward_fn.invoke_body(
            engine.ws, engine.kv_cache, engine.attn_ws, engine.cublas,
            fwd_in);
    }
    if (single_gpu_greedy_argmax) {
        const int argmax_parts =
            greedy_argmax_parts(engine.loaded_model.hf_config().vocab_size);
        if (argmax_parts > 1 && !engine.ws.greedy_pairs_all.empty()) {
            kernels::launch_argmax_bf16_partitioned_pairs(
                engine.ws.logits.data(),
                reinterpret_cast<std::uint64_t*>(
                    engine.ws.greedy_pairs_all.data()),
                N, engine.loaded_model.hf_config().vocab_size, argmax_parts,
                cstream);
            kernels::launch_select_global_argmax_pairs(
                reinterpret_cast<const std::uint64_t*>(
                    engine.ws.greedy_pairs_all.data()),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, argmax_parts, cstream);
        } else {
            kernels::launch_argmax_bf16(
                engine.ws.logits.data(),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, engine.loaded_model.hf_config().vocab_size,
                cstream);
        }
    }
    CudaGraphOwner graph(capture_guard.end());
    if (engine.tp_comm != nullptr &&
        engine.tp_comm->custom_all_reduce() != nullptr) {
        engine.tp_comm->custom_all_reduce()
            ->register_graph_buffers(*engine.tp_comm);
    }
    cublas_stream.restore();

    CudaGraphExecOwner exec;
    CUDA_CHECK(cudaGraphInstantiate(
        exec.out(), graph.get(), nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(exec.get(), nullptr));
    return exec.release();
}

std::size_t capture_forward_graph_lattice(BatchEngine& engine) {
    if (engine.graph_cache == nullptr) return 0;
    if (!engine.forward_fn.graph_safe) return 0;
    if (engine.forward_fn.model == nullptr) return 0;
    if (engine.loaded_model.hf_config().model_type == "nemotron_h") {
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
        std::min(engine.max_forward_requests, engine.max_workspace_tokens);
    if (max_requests <= 0) return 0;

    auto buckets = forward_graph_request_lattice(max_requests);
    if (buckets.empty()) return 0;

    auto& pi = engine.inputs;
    const int num_pages = std::max(1, engine.kv_cache.num_pages());
    std::size_t captured = 0;
    const bool tp_greedy_argmax =
        engine.tp_comm != nullptr &&
        engine.tp_comm->world_size() <= 8 &&
        engine.forward_fn.supports_tp_greedy_argmax;
    const bool fwd_handles_argmax_precapture =
        engine.forward_fn.supports_fused_lmhead_argmax &&
        engine.tp_comm == nullptr;
    const bool single_gpu_graph_argmax =
        graph_single_gpu_argmax_enabled() && engine.tp_comm == nullptr &&
        !fwd_handles_argmax_precapture;
    const bool log_rank =
        engine.verbose &&
        (engine.tp_comm == nullptr || engine.tp_comm->rank() == 0);
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
        if (engine.rs_cache != nullptr) {
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

        engine.forward_fn.invoke_prepare(
            engine.attn_ws,
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
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(tp_greedy_argmax, single_gpu_graph_argmax,
                               fwd_handles_argmax_precapture,
                               /*small_spec=*/false, /*rs_verify=*/false,
                               graph_layout);
        const ForwardGraphKey key{R, N, graph_variant};
        if (engine.graph_cache->get(key) != nullptr) continue;

        if (fwd_handles_argmax_precapture) {
            engine.forward_fn.invoke_set_logits_argmax_only(true);
            engine.forward_fn.invoke_set_fused_argmax_output(
                reinterpret_cast<std::int32_t*>(pi.sampled.data()));
        }
        tp_graph_capture_barrier(engine);
        cudaGraphExec_t exec = capture_forward_graph_exec(
            engine, qo.data(), kvpi.data(), kvpp.data(), kvlpl.data(),
            N, R, /*is_pure_decode=*/true,
            /*slot_ids_h=*/nullptr, /*is_fresh_h=*/nullptr,
            engine.rs_cache != nullptr ? pi.slot_ids.data() : nullptr,
            /*logit_row_indices_d=*/nullptr,
            /*num_logit_rows=*/0, single_gpu_graph_argmax,
            tp_greedy_argmax);
        if (fwd_handles_argmax_precapture) {
            engine.forward_fn.invoke_set_fused_argmax_output(nullptr);
        }
        engine.graph_cache->put(key, exec);
        ++captured;
        tp_graph_capture_barrier(engine);
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
                  << " (cache size=" << engine.graph_cache->size()
                  << ", graph_mem~" << (graph_bytes / (1024 * 1024))
                  << " MiB"
                  << ", " << dt << " ms)\n";
    }
    return captured;
}

ForwardInputViews make_forward_input_views(
    std::span<const std::uint32_t> tokens,
    std::span<const std::uint32_t> positions,
    std::span<const std::uint32_t> qo_indptr,
    std::span<const std::uint32_t> kv_page_indices,
    std::span<const std::uint32_t> kv_page_indptr,
    std::span<const std::uint32_t> kv_last_page_lens,
    int num_requests)
{
    return ForwardInputViews{
        tokens,
        positions,
        qo_indptr,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        static_cast<int>(tokens.size()),
        num_requests,
    };
}

void run_forward_dispatch(BatchEngine& engine, const ForwardDispatchInputs& in) {
    auto& ws = engine.ws;
    auto& kv_cache = engine.kv_cache;
    auto& attn_ws = engine.attn_ws;
    auto& cublas = engine.cublas;
    auto& pi = engine.inputs;
    auto& forward_fn = engine.forward_fn;

    const bool graph_eligible =
        engine.graph_cache != nullptr &&
        engine.forward_fn.graph_safe &&
        in.is_pure_decode &&
        !in.have_custom_mask &&
        !in.rs_buffer_write &&
        !in.rs_buffer_fold &&
        graph_replay_has_no_host_resets(
            in.use_slots,
            in.is_fresh_h_data,
            static_cast<std::size_t>(std::max(in.forward_R, 0))) &&
        in.num_images == 0 &&
        in.num_clips == 0 &&
        in.stage_hooks == nullptr;
    if (graph_eligible) {
        const std::uint32_t graph_layout =
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(
                /*tp_greedy_argmax=*/false,
                /*single_gpu_graph_argmax=*/false,
                /*fwd_handles=*/false,
                /*small_spec=*/false,
                /*rs_verify=*/false,
                graph_layout);
        const ForwardGraphKey key{
            in.forward_R,
            in.forward_N,
            graph_variant,
            in.program_set_hash,
            in.has_write_desc,
            in.structured_window_left,
        };
        cudaGraphExec_t exec = engine.graph_cache->get(key);
        if (exec == nullptr) {
            exec = capture_forward_graph_exec(
                engine,
                in.h_qo_forward,
                in.h_kvpi_forward,
                in.h_kvpp_forward,
                in.h_kvlpl_forward,
                in.forward_N,
                in.forward_R,
                true,
                in.use_slots ? in.slot_ids_h_data : nullptr,
                in.use_slots ? in.is_fresh_h_data : nullptr,
                in.use_slots ? pi.slot_ids.data() : nullptr,
                nullptr,
                0,
                false,
                false,
                in.has_write_desc ? pi.w_page.data() : nullptr,
                in.has_write_desc ? pi.w_off.data() : nullptr,
                in.has_write_desc,
                in.structured_window_left);
            engine.graph_cache->put(key, exec);
        }
        CUDA_CHECK(cudaGraphLaunch(exec, cublas.stream()));
        return;
    }

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
    fwd_in.runtime_window_left = in.structured_window_left;
    fwd_in.w_page_d             = in.has_write_desc ? pi.w_page.data() : nullptr;
    fwd_in.w_off_d              = in.has_write_desc ? pi.w_off.data()  : nullptr;
    fwd_in.has_write_desc       = in.has_write_desc;
    fwd_in.slot_ids_h          = in.use_slots ? in.slot_ids_h_data : nullptr;
    fwd_in.is_fresh_h          = in.use_slots ? in.is_fresh_h_data : nullptr;
    fwd_in.slot_ids_d          = in.use_slots ? pi.slot_ids.data() : nullptr;
    fwd_in.rs_slot_flags_h     = in.use_slots
        ? pi.rs_slot_flags_host.data()
        : nullptr;
    fwd_in.rs_buffer_slot_ids_h    = in.rs_buffer_slot_ids_h;
    fwd_in.rs_buffer_slot_indptr_h = in.rs_buffer_slot_indptr_h;
    fwd_in.rs_fold_lens_h           = in.rs_fold_lens_h;
    fwd_in.rs_fold_lens_d           = in.rs_fold_lens_d;
    fwd_in.rs_buffer_write         = in.rs_buffer_write;
    fwd_in.rs_buffer_fold          = in.rs_buffer_fold;
    // Compact/gathered logit rows are a legacy graph-variant fast path that
    // no direct-PTIR fire uses (the sampler reads straight out of ws.logits
    // via pi.sample_idx after this call — see batch/logits.hpp).
    fwd_in.logit_row_indices_d = nullptr;
    fwd_in.num_logit_rows      = 0;
    fwd_in.emit_logits         = in.num_sampling > 0;
    // TP greedy-argmax fusion is a graph-capture-only fast path; direct PTIR
    // launches leave the selected logits available to the stage program.
    fwd_in.tp_greedy_argmax    = false;
    // Multimodal: image data for the encode+scatter (no-op if none).
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
    fwd_in.precomputed_embeddings       = in.precomputed_embeddings;
    fwd_in.stage_hooks                  = in.stage_hooks;
    forward_fn.invoke_body(ws, kv_cache, attn_ws, cublas, fwd_in);
}

}  // namespace pie_cuda_driver
