#include "batch/forward.hpp"
#include "batch/graph_variant.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <map>
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
#include "kernels/argmax.hpp"
#include "kernels/custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "distributed.hpp"
#include "model/loaded_model.hpp"
#include "store/kv_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "model/imodel.hpp"
#include "model/stage_hooks.hpp"
#include "model/attn_observation.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver {

void ForwardFn::attach_model(model::IModel* m) {
    model = m;
    if (m == nullptr) return;
    const auto caps = m->capabilities();
    if (caps.graph_safe && !caps.graph_padding_kv_write_safe) {
        throw std::runtime_error(
            "graph-safe model must gate every KV write on row_valid");
    }
    graph_safe                   = caps.graph_safe;
    // Diagnostic escape hatch. A capturing stream forbids the event syncs the
    // in-engine stage profilers need, so the decode path -- the one that
    // decides throughput -- is the one path that cannot be measured while it
    // is captured. Dropping capability, rather than replay, keeps the plan
    // and the executor consistent with each other. Costs throughput; it is
    // for measurement, not for serving.
    if (graph_safe && std::getenv("PIE_CUDA_DISABLE_GRAPH_CAPTURE") != nullptr) {
        graph_safe = false;
    }
    graph_padding_kv_write_safe  = caps.graph_padding_kv_write_safe;
    supports_compact_logits      = caps.supports_compact_logits;
    supports_small_prefill_graph = caps.supports_small_prefill_graph;
    supports_runtime_window       = caps.supports_runtime_window;
    supports_fused_lm_head_argmax = caps.supports_fused_lm_head_argmax;
}

void ForwardFn::invoke_prepare(AttentionWorkspace& aws,
                               const PrepareInputs& in) {
    if (model) model->prepare(aws, in);
}

namespace {

thread_local const model::AttentionObservation* g_attention_observation = nullptr;

}  // namespace

namespace model {

const AttentionObservation* active_attention_observation() noexcept {
    return g_attention_observation;
}

ScopedAttentionObservation::ScopedAttentionObservation(
    const AttentionObservation* observation) noexcept
    : previous_(g_attention_observation) {
    g_attention_observation = observation;
}

ScopedAttentionObservation::~ScopedAttentionObservation() noexcept {
    g_attention_observation = previous_;
}

}  // namespace model

void ForwardFn::invoke_body(model::Workspace& ws,
                            KvCache& kv,
                            AttentionWorkspace& aws,
                            ops::CublasHandle& cublas,
                            const ForwardInputs& in) {
    if (model) {
        model::ScopedStageHooks hooks(in.stage_hooks);
        // Publish the fire's KV geometry for the duration of the body so an
        // attention-stage PTIR program can score the cache it is about to
        // attend over. Scoped to the body, so a stage hook can never observe a
        // page table from a different fire.
        const model::AttentionObservation observation{
            .kv = &kv,
            .kv_page_indices_d = in.kv_page_indices_d,
            .kv_page_indptr_d = in.kv_page_indptr_d,
            .kv_last_page_lens_d = in.kv_last_page_lens_d,
            .qo_indptr_h = in.qo_indptr_h,
            .kv_page_indptr_h = in.kv_page_indptr_h,
            .kv_last_page_lens_h = in.kv_last_page_lens_h,
            .num_requests = in.num_requests,
            .total_tokens = in.total_tokens,
        };
        model::ScopedAttentionObservation observed(
            in.stage_hooks != nullptr ? &observation : nullptr);
        model->body(ws, kv, aws, cublas, in);
    }
}

std::uint32_t ForwardFn::invoke_graph_layout() {
    return model ? model->graph_layout() : 0u;
}

bool ForwardFn::invoke_prefill_graph_capturable() const {
    return model != nullptr && model->prefill_graph_capturable();
}

// Lets a wave carrying a prefill reach the graph cache instead of falling to
// the eager path along with all of its decode lanes. Without it one arriving
// request costs every decode lane in its wave the replay: 7,290 us of host
// enqueue against 10 us at the same width.
//
// **Default ON**; `PIE_PREFILL_GRAPH=0` restores the decode-only gate.
// Measured on 4096x64 @c256: +10.6% (3 of 3 rounds), and mixed-sign neutral on
// 768x512, 256x256 and the shared-prefix shape — it pays where request
// turnover is high and costs nothing where it is not.
//
// The price is memory, not speed: sizing the float workspace for graph-mode
// planning takes it 80 -> 672 MiB, i.e. 592 MiB out of the KV pool (338 pages
// of 22,039, ~1.5%). A deployment tighter on KV than on throughput should turn
// this off.
//
// It does NOT cost the decode path, despite an earlier reading to that effect:
// normalised per LANE across runs of matched width the decode host cost rises
// 6.1%, uniformly across phases that share no machinery, which tracks the 6.9%
// shorter wall rather than any decode-side regression. Net on the cell is
// -880 ms of prefill enqueue against +85 ms of decode.
bool graph_key_check_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_GRAPH_KEY_CHECK");
        return env != nullptr && *env != '\0' && env[0] != '0';
    }();
    return value;
}

// `PIE_PREFILL_GRAPH_PLAN` -- graph-mode planning for the real prefill/mixed
// batch, which is what makes a prefill-carrying wave capturable at all.
//
// **Default OFF, and the default is the measurement.** `PIE_PREFILL_GRAPH`
// gates three separate things: this plan mode, the request/token padding in
// `frame.cpp`, and the enlarged float workspace in `batch/workspace.cu`. Until
// this knob existed they could only be moved together, which is how
// `8775dda9` came to be credited with "prefill graph replay" -- a mechanism
// that, measured by `PIE_GRAPH_STATS`, had never run: `prefill: hits=0
// misses=0 ineligible=175`.
//
// Armed, the mechanism works (`prefill: hits=115 misses=62`, 4096/4096, zero
// key-check violations) and it LOSES. Holding the workspace and the padding
// fixed and toggling only this, on the S cell, 4 rounds ABBA:
//
//     plan on   31317  30720  31396  31695   mean 31282
//     plan off  32140  32585  32246  32753   mean 32431   -3.54%, 4/4
//
// Prefill capture buys a ~1.3 ms eager enqueue back on a wave that replays,
// but graph-mode planning always splits KV and carves float partials sized by
// the PADDED work-item count, and on this cell that costs more than the
// enqueue it saves. Off by default until a cell is found where it wins;
// everything downstream of it stays built, tested and reachable behind
// `PIE_PREFILL_GRAPH_PLAN=1`.
bool prefill_graph_plan_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_PREFILL_GRAPH_PLAN");
        return env != nullptr && *env != '\0' && env[0] != '0';
    }();
    return value;
}

// `PIE_PREFILL_GRAPH_PAD` -- the request/token padding half of the lever, so
// it can be priced against the workspace half. `PIE_PREFILL_GRAPH` moves both
// plus the plan mode; `PIE_PREFILL_GRAPH_PLAN` already split the plan off, and
// this splits the remaining two. Follows `prefill_graph_enabled()` unless
// overridden, so default behaviour is unchanged.
bool prefill_graph_pad_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_PREFILL_GRAPH_PAD");
        if (env == nullptr || *env == '\0') return prefill_graph_enabled();
        return env[0] != '0';
    }();
    return value;
}

bool graph_stats_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_GRAPH_STATS");
        return env != nullptr && *env != '\0' && env[0] != '0';
    }();
    return value;
}

bool prefill_graph_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_PREFILL_GRAPH");
        return env == nullptr || *env == '\0' || env[0] != '0';
    }();
    return value;
}

namespace {

// #24 graph-variant bitfield helper: `make_graph_variant()` + the named flag
// constants + the boundary static_asserts now live in
// "batch/graph_variant.hpp" (so they're unit-testable host-side).

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

constexpr bool step_profile_enabled() { return false; }

constexpr std::uint64_t step_profile_limit() {
    return std::uint64_t{32};
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

int logits_argmax_chunk_tokens() {
    static const int tokens = [] {
        const char* v = std::getenv("PIE_LOGITS_CHUNK_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        const long parsed = std::strtol(v, nullptr, 10);
        if (parsed <= 0) return 0;
        // Validated, not clamped. The width itself is a tuning question this
        // layer refuses to answer, but a value the mechanism cannot express is
        // a config error, and failing at startup beats capturing a graph with
        // tens of thousands of slab nodes and appearing to hang.
        if (parsed > std::numeric_limits<int>::max() ||
            parsed < static_cast<long>(kernels::kArgmaxAccumSlots)) {
            throw std::runtime_error(
                "PIE_LOGITS_CHUNK_TOKENS must be at least " +
                std::to_string(kernels::kArgmaxAccumSlots) +
                " (the per-row accumulator width) and fit in an int; a slab "
                "narrower than that carries more running state than it "
                "summarises");
        }
        return static_cast<int>(parsed);
    }();
    return tokens;
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
    bool have_custom_mask,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    bool has_write_desc,
    int runtime_window_left,
    int logits_argmax_chunk)
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
        fwd_in.custom_mask_d = have_custom_mask
            ? pi.custom_mask.data()
            : nullptr;
        fwd_in.custom_mask_indptr_d = have_custom_mask
            ? pi.custom_mask_indptr.data()
            : nullptr;
        fwd_in.slot_ids_h          = slot_ids_h;
        fwd_in.is_fresh_h          = is_fresh_h;
        fwd_in.slot_ids_d          = slot_ids_d;
        fwd_in.is_fresh_d          = pi.is_fresh.data();
        fwd_in.logit_row_indices_d = logit_row_indices_d;
        fwd_in.num_logit_rows      = num_logit_rows;
        fwd_in.logits_argmax_chunk_tokens = logits_argmax_chunk;
        fwd_in.w_page_d = w_page_d;
        fwd_in.w_off_d = w_off_d;
        fwd_in.row_valid_d = pi.row_valid.data();
        fwd_in.has_write_desc = has_write_desc;
        fwd_in.runtime_window_left = runtime_window_left;
        engine.forward_fn.invoke_body(
            engine.ws, engine.kv_cache, engine.attn_ws, engine.cublas,
            fwd_in);
    }
    CudaGraphOwner graph(capture_guard.end());
    if (engine.tp_comm != nullptr &&
        engine.tp_comm->custom_all_reduce() != nullptr) {
        engine.tp_comm->custom_all_reduce()
            ->register_graph_buffers(*engine.tp_comm);
    }
    cublas_stream.restore();

    CudaGraphExecOwner exec;
    if (graph.get() == nullptr) {
        throw std::runtime_error(
            "forward graph capture produced a null graph (N=" +
            std::to_string(N) + ", R=" + std::to_string(R) + ")");
    }
    // A sticky error left by an earlier async launch surfaces on the next
    // runtime call and would be misreported as an instantiate failure.
    const cudaError_t pending = cudaGetLastError();
    const cudaError_t inst = cudaGraphInstantiate(
        exec.out(), graph.get(), nullptr, nullptr, 0);
    if (inst != cudaSuccess) {
        std::size_t nodes = 0;
        cudaGraphGetNodes(graph.get(), nullptr, &nodes);
        std::string histogram;
        if (nodes > 0) {
            std::vector<cudaGraphNode_t> node_list(nodes);
            if (cudaGraphGetNodes(graph.get(), node_list.data(), &nodes) ==
                cudaSuccess) {
                std::map<int, int> by_type;
                for (cudaGraphNode_t node : node_list) {
                    cudaGraphNodeType type{};
                    if (cudaGraphNodeGetType(node, &type) == cudaSuccess) {
                        ++by_type[static_cast<int>(type)];
                    }
                }
                for (const auto& [type, count] : by_type) {
                    histogram += " t" + std::to_string(type) + "=" +
                                 std::to_string(count);
                }
            }
        }
        int device = -1;
        cudaGetDevice(&device);
        throw std::runtime_error(
            std::string("cudaGraphInstantiate failed: ") +
            cudaGetErrorString(inst) + " (N=" + std::to_string(N) +
            ", R=" + std::to_string(R) + ", nodes=" + std::to_string(nodes) +
            ", device=" + std::to_string(device) + ", tp_rank=" +
            std::to_string(engine.tp_comm != nullptr
                               ? engine.tp_comm->rank()
                               : -1) +
            ", node_types:" + histogram + ", pending_before=" +
            cudaGetErrorName(pending) + ")");
    }
    CUDA_CHECK(cudaGraphUpload(exec.get(), nullptr));
    return exec.release();
}

// W8: an upfront-capture skip is a real performance decision, never a
// silent one — without the lattice, every request-count bucket a ramp
// touches pays a lazy capture+instantiate (~10 ms each, stream-synced)
// INSIDE the ramp; ~32 buckets on the 7->256 hard-default shape
// (measured, V6 iteration 53: submit spikes to 6.7 ms and a per-run
// variance source).
static std::size_t skip_upfront_capture(const char* why) {
    std::fprintf(
        stderr,
        "[pie-driver-cuda] UPFRONT GRAPH CAPTURE SKIPPED (%s): decode "
        "buckets will capture lazily on first use (~10 ms stream-synced "
        "stall per bucket, inside the ramp)\n",
        why);
    return 0;
}

std::size_t capture_forward_graph_lattice(BatchEngine& engine) {
    if (engine.graph_cache == nullptr) return 0;
    if (!engine.forward_fn.graph_safe) {
        return skip_upfront_capture("model is not graph-safe");
    }
    if (engine.forward_fn.model == nullptr) return 0;
    if (engine.loaded_model.hf_config().model_type == "nemotron_h") {
        // Nemotron-H has recurrent Mamba state in addition to attention state.
        // Synthetic upfront capture replays incorrectly; first-use capture with
        // real slot/page metadata is correct, so leave the cache cold here.
        return skip_upfront_capture(
            "nemotron_h recurrent state requires first-use capture");
    }
    const int max_requests =
        std::min(engine.max_forward_requests, engine.max_workspace_tokens);
    if (max_requests <= 0) return 0;

    auto buckets = forward_graph_request_lattice(max_requests);
    if (buckets.empty()) return 0;

    auto& pi = engine.inputs;
    engine.kv_cache.ensure_pages(1);
    std::size_t captured = 0;
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
        std::vector<std::uint32_t> kvpi(static_cast<std::size_t>(R), 0u);
        std::vector<std::uint32_t> write_page(static_cast<std::size_t>(N), 0u);
        std::vector<std::uint32_t> write_offset(static_cast<std::size_t>(N), 0u);
        std::vector<std::int32_t> slot_ids;

        for (int r = 0; r <= R; ++r) {
            qo[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
            kvpp[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
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
        pi.w_page.copy_from_host(std::span<const std::uint32_t>(write_page));
        pi.w_off.copy_from_host(std::span<const std::uint32_t>(write_offset));
        CUDA_CHECK(cudaMemsetAsync(
            pi.row_valid.data(), 1,
            static_cast<std::size_t>(N), nullptr));
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
        // The lattice pre-captures the shape the hot path will ask for. Setting
        // the width is an explicit opt-in, so betting on the fused shape is the
        // right bet -- but it IS a bet: a deployment that sets the width and
        // then runs guests whose epilogues are not bare argmaxes gets no useful
        // lattice at all and pays lazy capture per bucket on the ramp.
        const int lattice_chunk =
            engine.tp_comm == nullptr && engine.forward_fn.supports_fused_lm_head_argmax
                ? logits_argmax_chunk_tokens()
                : 0;
        const std::uint32_t graph_variant =
            make_graph_variant(/*small_spec=*/false, /*rs_verify=*/false,
                               /*custom_mask=*/false,
                               /*fused_argmax=*/lattice_chunk > 0,
                               graph_layout);
        const ForwardGraphKey key{R, N, graph_variant};
        if (engine.graph_cache->get(key) != nullptr) continue;

        tp_graph_capture_barrier(engine);
        cudaGraphExec_t exec = capture_forward_graph_exec(
            engine, qo.data(), kvpi.data(), kvpp.data(), kvlpl.data(),
            N, R, /*is_pure_decode=*/true,
            /*have_custom_mask=*/false,
            /*slot_ids_h=*/nullptr, /*is_fresh_h=*/nullptr,
            engine.rs_cache != nullptr ? pi.slot_ids.data() : nullptr,
            /*logit_row_indices_d=*/nullptr,
            /*num_logit_rows=*/0,
            pi.w_page.data(), pi.w_off.data(),
            /*has_write_desc=*/true,
            /*runtime_window_left=*/-2,
            lattice_chunk);
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

bool forward_graph_replay_eligible(
    const BatchEngine& engine,
    bool is_pure_decode,
    bool have_custom_mask,
    bool rs_buffer_write,
    bool rs_buffer_fold,
    bool has_write_desc,
    int structured_window_left,
    bool use_slots,
    const std::uint8_t* is_fresh_h_data,
    int forward_R,
    int num_images,
    int num_clips,
    bool has_stage_hooks,
    int num_logit_rows) {
    const bool mask_pointers_stable =
        !have_custom_mask ||
        (engine.inputs.custom_mask.data() != nullptr &&
         engine.inputs.custom_mask_indptr.data() != nullptr);
    // A wave is replayable if its geometry is content-independent. Pure decode
    // always is. A wave carrying a prefill is when the PLANNER says so --
    // `PrefillPlanCache::graph_capturable`, which is exactly the "FA2 causal
    // path planned in graph mode" condition and already computed per fire.
    //
    // Without the second clause one arriving request costs every decode lane in
    // the wave its replay: measured 7,290 us of host enqueue on a prefill-
    // carrying wave against 10 us on a pure-decode wave of the SAME width.
    // `num_logit_rows` is BAKED INTO the captured body -- it sizes the logits
    // gather and the lm_head rows -- but `ForwardGraphKey` carries only the
    // (R, N) buckets. Replay is therefore safe only when the baked count is
    // itself derivable from the key: either 0 (full-N emit, which is what
    // every pure-decode capture gets, since `compact_logits` requires
    // `!is_pure_decode`) or exactly `forward_R`, which `frame.cpp` arranges by
    // padding the compact row list up to the request bucket.
    //
    // This is the invariant rather than a proxy for it on purpose: a wave whose
    // row list could not be padded -- device-composed, over capacity, unpadded
    // -- fails this test and runs eager, instead of replaying a body that
    // gathers the wrong number of rows and leaves the surplus requests sampling
    // the previous fire's residue.
    const bool logit_rows_keyed =
        num_logit_rows == 0 || num_logit_rows == forward_R;
    const bool planner_capturable =
        engine.forward_fn.invoke_prefill_graph_capturable();
    const bool geometry_replayable =
        is_pure_decode ||
        (prefill_graph_enabled() &&
         logit_rows_keyed &&
         planner_capturable);

    // Named terms, then a first-failure scan, so `PIE_GRAPH_STATS` can say
    // WHICH clause refused a wave. A bare bool told us only that 175 of 175
    // prefill-carrying waves were ineligible on the S cell, which is the same
    // report whether the planner refused, the row list could not be padded, or
    // the flag was simply off -- three findings with nothing in common.
    // Order is the reporting order; a wave is attributed to its first
    // failing clause.
    const struct { const char* name; bool ok; } clauses[] = {
        {"cache_absent",      engine.graph_cache != nullptr},
        {"forward_not_safe",  engine.forward_fn.graph_safe},
        {"flag_off",          is_pure_decode || prefill_graph_enabled()},
        {"logit_rows",        is_pure_decode || logit_rows_keyed},
        {"planner_refused",   is_pure_decode || planner_capturable},
        {"mask_pointers",     mask_pointers_stable},
        {"rs_buffer_write",   !rs_buffer_write},
        {"rs_buffer_fold",    !rs_buffer_fold},
        {"structured_window", structured_window_left == -2},
        // Pure-decode captures record the explicit w_page/w_off KV-write
        // path, so a decode fire without write descriptors must stay eager.
        {"no_write_desc",     has_write_desc},
        {"host_resets",       graph_replay_has_no_host_resets(
                                  use_slots, is_fresh_h_data,
                                  static_cast<std::size_t>(
                                      std::max(forward_R, 0)))},
        {"images",            num_images == 0},
        {"clips",             num_clips == 0},
        {"stage_hooks",       !has_stage_hooks},
    };
    for (const auto& clause : clauses) {
        if (!clause.ok) {
            if (engine.graph_cache != nullptr) {
                engine.graph_cache->note_refusal(is_pure_decode, clause.name);
            }
            // The `logit_rows` clause is the one `frame.cpp` is supposed to
            // satisfy by padding, so a refusal here means the padding did not
            // fire. Print the pair it disagreed on, bounded, rather than
            // leaving the count to be explained by reading the padding
            // conditions and guessing which one was false.
            if (!is_pure_decode && clause.name[0] == 'l' &&
                graph_stats_enabled()) {
                static std::atomic<int> shown{0};
                if (shown.fetch_add(1) < 10) {
                    std::fprintf(stderr,
                                 "[graph-stats]   logit_rows refusal: "
                                 "num_logit_rows=%d forward_R=%d\n",
                                 num_logit_rows, forward_R);
                }
            }
            return false;
        }
    }
    static_cast<void>(geometry_replayable);
    return true;
}

void run_forward_dispatch(BatchEngine& engine, const ForwardDispatchInputs& in) {
    auto& ws = engine.ws;
    auto& kv_cache = engine.kv_cache;
    auto& attn_ws = engine.attn_ws;
    auto& cublas = engine.cublas;
    auto& pi = engine.inputs;
    auto& forward_fn = engine.forward_fn;

    const bool graph_eligible = forward_graph_replay_eligible(
        engine,
        in.is_pure_decode,
        in.have_custom_mask,
        in.rs_buffer_write,
        in.rs_buffer_fold,
        in.has_write_desc,
        in.structured_window_left,
        in.use_slots,
        in.is_fresh_h_data,
        in.forward_R,
        in.num_images,
        in.num_clips,
        in.stage_hooks != nullptr,
        in.num_logit_rows);
    if (graph_eligible) {
        const std::uint32_t graph_layout =
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(
                /*small_spec=*/false,
                /*rs_verify=*/false,
                in.have_custom_mask,
                /*fused_argmax=*/in.logits_argmax_chunk_tokens > 0,
                graph_layout,
                /*compact_logits=*/in.num_logit_rows > 0);
        const ForwardGraphKey key{
            in.forward_R,
            in.forward_N,
            graph_variant,
        };
        // `PIE_GRAPH_KEY_CHECK=1`: the invariant this whole path rests on is
        // that a key determines the baked `num_logit_rows`. Assert it directly
        // rather than inferring it from output agreement -- a mismatch here
        // produces occasional wrong tokens, which is indistinguishable from
        // this engine's own run-to-run nondeterminism and is exactly how the
        // first version of this shipped.
        if (graph_key_check_enabled()) {
            static std::mutex mu;
            static std::unordered_map<ForwardGraphKey, int,
                                      ForwardGraphKeyHash> seen;
            std::lock_guard<std::mutex> lock(mu);
            auto [it, fresh] = seen.emplace(key, in.num_logit_rows);
            if (!fresh && it->second != in.num_logit_rows) {
                std::fprintf(stderr,
                             "[graph-key-check] VIOLATION R=%d N=%d var=%u "
                             "baked=%d now=%d\n",
                             key.num_requests, key.num_tokens, key.variant,
                             it->second, in.num_logit_rows);
            }
        }
        cudaGraphExec_t exec =
            engine.graph_cache->get(key, in.is_pure_decode);
        if (exec == nullptr) {
            if (step_profile_enabled()) {
                std::fprintf(stderr,
                             "[step-profile] graph capture R=%d N=%d variant=%u"
                             " (cache size %zu)\n",
                             key.num_requests, key.num_tokens, key.variant,
                             engine.graph_cache->size());
            }
            exec = capture_forward_graph_exec(
                engine,
                in.h_qo_forward,
                in.h_kvpi_forward,
                in.h_kvpp_forward,
                in.h_kvlpl_forward,
                in.forward_N,
                in.forward_R,
                in.is_pure_decode,
                in.have_custom_mask,
                in.use_slots ? in.slot_ids_h_data : nullptr,
                in.use_slots ? in.is_fresh_h_data : nullptr,
                in.use_slots ? pi.slot_ids.data() : nullptr,
                // Pure decode captures full-N logits (N == R); a prefill
                // capture records the compact row list instead — its count
                // is pinned to R by the eligibility gate above.
                in.compact_logits ? pi.sample_idx.data() : nullptr,
                in.num_logit_rows,
                pi.w_page.data(),
                pi.w_off.data(),
                in.has_write_desc,
                in.structured_window_left,
                in.logits_argmax_chunk_tokens);
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
    fwd_in.row_valid_d          = pi.row_valid.data();
    fwd_in.has_write_desc       = in.has_write_desc;
    fwd_in.slot_ids_h          = in.use_slots ? in.slot_ids_h_data : nullptr;
    fwd_in.is_fresh_h          = in.use_slots ? in.is_fresh_h_data : nullptr;
    fwd_in.slot_ids_d          = in.use_slots ? pi.slot_ids.data() : nullptr;
    fwd_in.is_fresh_d          = in.use_slots ? pi.is_fresh.data() : nullptr;
    fwd_in.rs_slot_flags_h     = in.use_slots
        ? in.rs_slot_flags_h
        : nullptr;
    fwd_in.rs_buffer_slot_ids_h    = in.rs_buffer_slot_ids_h;
    fwd_in.rs_buffer_slot_indptr_h = in.rs_buffer_slot_indptr_h;
    fwd_in.rs_buffer_read_slot_ids_h = in.rs_buffer_read_slot_ids_h;
    fwd_in.rs_buffer_read_indptr_h   = in.rs_buffer_read_indptr_h;
    fwd_in.rs_buffer_read_lens_h     = in.rs_buffer_read_lens_h;
    fwd_in.rs_buffer_heads_h         = in.rs_buffer_heads_h;
    fwd_in.rs_fold_lens_h           = in.rs_fold_lens_h;
    fwd_in.rs_fold_lens_d           = in.rs_fold_lens_d;
    fwd_in.rs_buffer_write         = in.rs_buffer_write;
    fwd_in.rs_buffer_fold          = in.rs_buffer_fold;
    fwd_in.logit_row_indices_d =
        in.compact_logits ? pi.sample_idx.data() : nullptr;
    fwd_in.num_logit_rows =
        in.num_logit_rows;
    fwd_in.emit_logits         = in.num_sampling > 0;
    fwd_in.logits_argmax_chunk_tokens = in.logits_argmax_chunk_tokens;
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
