#include "batch/forward.hpp"

#include "batch/supergraph.hpp"
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
    graph_padding_kv_write_safe  = caps.graph_padding_kv_write_safe;
    supports_compact_logits      = caps.supports_compact_logits;
    supports_small_prefill_graph = caps.supports_small_prefill_graph;
    supports_runtime_window       = caps.supports_runtime_window;
    supports_hook_graph_capture   = caps.supports_hook_graph_capture;
    supports_supergraph           = caps.supports_supergraph;
    supports_fused_lm_head_argmax = caps.supports_fused_lm_head_argmax;
    upfront_capture_safe          = caps.upfront_capture_safe;
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
        if (in.stage_hooks == nullptr) {
            model->body(ws, kv, aws, cublas, in);
            return;
        }
        // Attach the fire's KV geometry to the hooks for the duration of the
        // body so an attention-stage PTIR program can score the cache it is
        // about to attend over. The observation rides on a body-local copy of
        // the hooks — never ambient state — so a stage hook can only ever see
        // the page table of the fire that invoked it.
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
        model::StageHooks hooks = *in.stage_hooks;
        hooks.observation = &observation;
        ForwardInputs body_in = in;
        body_in.stage_hooks = &hooks;
        model->body(ws, kv, aws, cublas, body_in);
    }
}

std::uint64_t ForwardFn::invoke_lora_stage(model::Workspace& ws,
                                           const model::LoraTable* lora,
                                           int total_tokens,
                                           cudaStream_t stream) {
    return model ? model->lora_stage(ws, lora, total_tokens, stream) : 0;
}

bool ForwardFn::invoke_supergraph_body(model::Workspace& ws,
                                       KvCache& kv,
                                       AttentionWorkspace& aws,
                                       ops::CublasHandle& cublas,
                                       const ForwardInputs& in,
                                       batch::SupergraphBuilder& sg) {
    return model != nullptr &&
           model->supergraph_body(ws, kv, aws, cublas, in, sg);
}

std::uint32_t ForwardFn::invoke_graph_layout() {
    return model ? model->graph_layout() : 0u;
}

std::uint32_t ForwardFn::invoke_supergraph_graph_layout() {
    return model ? model->supergraph_graph_layout() : 0u;
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
    const model::StageHooks* stage_hooks,
    bool use_supergraph,
    const model::LoraTable* lora,
    // NS-3: the spatial split (UINT32_MAX = not a spatial fire). The
    // captured body splits its attention and reads the identity qo from
    // pi.mask_suffix_qo_indptr.
    std::uint32_t unmasked_prefix_rows,
    int logits_argmax_chunk)
{
    auto& pi = engine.inputs;

    // Supergraph capture (S3): the mask arm's captured dispatch needs the
    // custom prefill plan MATERIALIZED at capture time (the emission's
    // null-plan check runs on the host, during capture) — and the else
    // arm needs the decode plan. Prepare builds exactly one per call, so
    // the supergraph capture runs prepare twice, mask-first; per-fire
    // prepare at replay then updates whichever plan that fire's live arm
    // consumes.
    if (use_supergraph) {
        ForwardFn::PrepareInputs prep{};
        prep.qo_indptr_h = qo_indptr_h;
        prep.kv_page_indices_h = kv_page_indices_h;
        prep.kv_page_indices_d =
            reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data());
        prep.kv_page_indptr_h = kv_page_indptr_h;
        prep.kv_page_indptr_d =
            reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data());
        prep.kv_last_page_lens_h = kv_last_page_lens_h;
        prep.kv_last_page_lens_d =
            reinterpret_cast<const std::uint32_t*>(
                pi.kv_last_page_lens.data());
        prep.total_tokens = N;
        prep.num_requests = R;
        prep.is_pure_decode = is_pure_decode;
        prep.runtime_window_left = runtime_window_left;
        prep.have_custom_mask = true;
        engine.forward_fn.invoke_prepare(engine.attn_ws, prep);
        prep.have_custom_mask = false;
        engine.forward_fn.invoke_prepare(engine.attn_ws, prep);
    }

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
        // Hook capture (stage 6 increment 4): the body's per-layer hook
        // invocations run inside the recorded region. `invoke_body` attaches
        // the observation to a body-local hooks copy exactly as on the eager
        // path — the observation's HOST pointers are consumed by host code
        // at capture time only; its DEVICE pointers (the persistent-input
        // KV CSRs) are replay-stable per key by the same argument the plain
        // captured body makes for every other `pi.*` buffer.
        fwd_in.stage_hooks = stage_hooks;
        fwd_in.lora = lora;
        // Peel device-window campaign: a hook capture reads the row split
        // from the device word, so the exec replays across compositions
        // and the fingerprint no longer keys on it. Non-hook captures keep
        // host windows (their split is degenerate and capture-stable).
        fwd_in.peel_window_d =
            stage_hooks != nullptr ? pi.peel_window.data() : nullptr;
        if (unmasked_prefix_rows != 0xffffffffu) {
            fwd_in.unmasked_prefix_rows = unmasked_prefix_rows;
            fwd_in.mask_suffix_qo_indptr_d =
                pi.mask_suffix_qo_indptr.data();
            fwd_in.mask_suffix_kv_page_indptr_d =
                pi.mask_suffix_kv_page_indptr.data();
        }
        if (use_supergraph) {
            // The union body: the mask/write-desc data pointers must be
            // the persistent buffers UNCONDITIONALLY — a masked replay
            // reads them even though this capture-time fire may carry no
            // mask.
            fwd_in.custom_mask_d = pi.custom_mask.data();
            fwd_in.custom_mask_indptr_d = pi.custom_mask_indptr.data();
            batch::SupergraphBuilder sg(cstream, pi.supergraph_preds.data());
            if (!engine.forward_fn.invoke_supergraph_body(
                    engine.ws, engine.kv_cache, engine.attn_ws,
                    engine.cublas, fwd_in, sg)) {
                throw std::runtime_error(
                    "supergraph capture requested but the model has no "
                    "emitted build for this deployment");
            }
        } else {
            engine.forward_fn.invoke_body(
                engine.ws, engine.kv_cache, engine.attn_ws, engine.cublas,
                fwd_in);
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
    // PIE_GRAPH_NODE_TRACE: per-capture node census — the discriminating
    // probe for "does a slow bucket's graph CONTAIN more work or just
    // slower kernels".
    static const bool node_trace = [] {
        const char* v = std::getenv("PIE_GRAPH_NODE_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    // PIE_GRAPH_DOT_DIR: dump every captured graph's full topology (node
    // kinds, event edges, kernel names) as DOT — the tool that names which
    // subsystem's event-record/wait pairs ended up inside a bucket graph.
    static const char* dot_dir = std::getenv("PIE_GRAPH_DOT_DIR");
    if (dot_dir != nullptr && dot_dir[0] != '\0') {
        static std::atomic<int> dot_seq{0};
        const int seq = dot_seq.fetch_add(1);
        const std::string path = std::string(dot_dir) + "/graph_R" +
            std::to_string(R) + "_N" + std::to_string(N) + "_" +
            std::to_string(seq) + ".dot";
        cudaGraphDebugDotPrint(graph.get(), path.c_str(),
                               cudaGraphDebugDotFlagsVerbose);
        cudaGetLastError();
    }
    if (node_trace) {
        std::size_t nodes = 0;
        cudaGraphGetNodes(graph.get(), nullptr, &nodes);
        std::string histogram;
        std::vector<cudaGraphNode_t> node_list(nodes);
        if (nodes > 0 &&
            cudaGraphGetNodes(graph.get(), node_list.data(), &nodes) ==
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
        std::fprintf(stderr,
                     "[graph-nodes] N=%d R=%d nodes=%zu%s\n",
                     N, R, nodes, histogram.c_str());
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
    if (!engine.forward_fn.upfront_capture_safe) {
        return skip_upfront_capture(
            "model declares synthetic upfront capture unsafe (plan-free "
            "force_prefill attention bakes capture-shape launch config); "
            "buckets capture on first use with real geometry");
    }
    const char* disable_upfront = std::getenv("PIE_CUDA_DISABLE_UPFRONT_GRAPHS");
    if (disable_upfront != nullptr && disable_upfront[0] != '\0' &&
        disable_upfront[0] != '0') {
        return skip_upfront_capture("PIE_CUDA_DISABLE_UPFRONT_GRAPHS is set");
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
            /*stage_hooks=*/nullptr,
            /*use_supergraph=*/false,
            /*lora=*/nullptr,
            /*unmasked_prefix_rows=*/0xffffffffu,
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

namespace {

bool hook_graph_trace_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_HOOK_GRAPH_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool hook_graphs_disabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("PIE_DISABLE_HOOK_GRAPHS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return disabled;
}

// PIE_HOOK_GRAPH_TRACE=1 evidence counters ([hook-graph] log lines).
struct HookGraphCounters {
    std::uint64_t captures = 0;
    std::uint64_t replays = 0;
    std::uint64_t prepare_vetoes = 0;
    std::uint64_t recaptures = 0;
    std::uint64_t bans = 0;
};
HookGraphCounters g_hook_graph_counters;

}  // namespace

bool hook_fire_blocks_graph(
    const BatchEngine& engine,
    bool has_stage_hooks) {
    if (!has_stage_hooks) return false;
    if (hook_graphs_disabled()) return true;
    // Increment-4 scope: hook fires on a capture-safe (llama_like) body may
    // replay — including page-mask fires (snapkv/h2o decode): the mask
    // consumer branch in the model body is host-STRUCTURAL (tagged whenever
    // the program's stage carries the sink — see resolve_lane_page_mask),
    // the mask carve is arena-stable, and the compaction is device-resolved
    // against the live CSR. Quest stays on the legacy interleaved body via
    // the dispatch-side prepare veto on `envelope_dot` (its query envelope
    // needs the body-time Query cast). TP stays off GRAPH replay: rank 0
    // replaying a hook graph while followers replay plain ones has no
    // replay-time branch agreement (`tp.cpp` hardcodes followers hook-free),
    // and a divergent NCCL op order deadlocks — but rank 0's hook fires
    // still run PREPARED, eagerly, through the unified seam (host-side
    // hoisting only; the device stream sees the same launches).
    if (!engine.forward_fn.supports_hook_graph_capture) return true;
    if (engine.tp_comm != nullptr && engine.tp_comm->world_size() > 1) {
        return true;
    }
    return false;
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
    bool has_lora) {
    (void)has_lora;
    const bool mask_pointers_stable =
        !have_custom_mask ||
        (engine.inputs.custom_mask.data() != nullptr &&
         engine.inputs.custom_mask_indptr.data() != nullptr);
    // A kill switch for the decode graphs alone. `PIE_CUDA_PREFILL_DECODE_
    // NOGRAPHS` reshapes the whole plan, so it cannot answer "is this bug in
    // the graph or in the kernels the graph records?" -- the question every
    // hang inside a replay asks first.
    static const bool graphs_disabled = [] {
        const char* v = std::getenv("PIE_CUDA_DISABLE_DECODE_GRAPHS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return !graphs_disabled &&
        engine.graph_cache != nullptr &&
        engine.forward_fn.graph_safe &&
        is_pure_decode &&
        mask_pointers_stable &&
        !rs_buffer_write &&
        !rs_buffer_fold &&
        structured_window_left == -2 &&
        // Pure-decode captures record the explicit w_page/w_off KV-write
        // path, so a decode fire without write descriptors must stay eager.
        has_write_desc &&
        graph_replay_has_no_host_resets(
            use_slots,
            is_fresh_h_data,
            static_cast<std::size_t>(std::max(forward_R, 0))) &&
        num_images == 0 &&
        num_clips == 0 &&
        !has_stage_hooks;
    // (campaign step 3b: lora fires are graph-eligible now — their execs
    // live in the fingerprint-partitioned lora store; `has_lora` remains
    // a parameter so TP callers can keep refusing, where lora cannot
    // occur anyway.)
}

// NS-2 (the spatial mask fire), DEFAULT ON since the ladder's sweep: a
// hook-free lora-free masked pure-decode fire with a planned unmasked
// prefix splits its attention — decode kernel over the prefix, custom
// kernel over the rebased suffix — and captures/replays in split-keyed
// execs (NS-3). PIE_SPATIAL_MASK=0 disarms.
static bool spatial_mask_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_SPATIAL_MASK");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

// The unionized supergraph gate: RETIRED BY PROMOTION (NS-5). The
// union's one live axis was the mask, and the spatial mask fire now
// serves every plannable masked pure-decode composition by row window
// inside ONE fire — no fire can arm the union's mask conditional
// anymore, so the two-path exec reduces to the plain graph plus dead
// capture weight (dual-prepare, re-key, pred upload). Default OFF;
// PIE_SUPERGRAPH=1 re-arms it for study. The MACHINERY stays: the
// SupergraphBuilder (capture-time conditional insertion, device
// predicate words, handle scope rules) is exactly what the IR's
// STRUCTURAL class needs when a genuinely different-operator axis
// (spec verify, early exit) lands — at region granularity, per the
// measured 250us floor.
static bool supergraph_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_SUPERGRAPH");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

void run_forward_dispatch(BatchEngine& engine, const ForwardDispatchInputs& in) {
    auto& ws = engine.ws;
    auto& kv_cache = engine.kv_cache;
    auto& attn_ws = engine.attn_ws;
    auto& cublas = engine.cublas;
    auto& pi = engine.inputs;
    auto& forward_fn = engine.forward_fn;

    const bool has_hooks = in.stage_hooks != nullptr;
    // Lora campaign step 3a: stage the fire's lora state HERE — outside
    // any capture region, before the graph decision — so both the eager
    // body and (step 3b) a captured one consume the same pre-staged
    // state. A null table clears; a lora fire re-stages fresh, so the
    // body's identity check always sees THIS fire's staging.
    const std::uint64_t lora_fingerprint =
        engine.forward_fn.invoke_lora_stage(
            engine.ws, in.lora, in.forward_N, cublas.stream());
    const bool hook_blocks = hook_fire_blocks_graph(engine, has_hooks);
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
        hook_blocks,
        in.lora != nullptr) &&
        // STRUCTURAL v0 (S-1): truncated fires are eager — the graph
        // exec family is full-depth (the depth peel is the recorded
        // union rung).
        in.planned_max_layers == 0xffffffffu &&
        // ④ Act 1: banded fires are eager (per-band plans + boundary
        // walk are not in any captured layout).
        in.depth_band_count == 0;
    const bool use_spatial_mask = spatial_mask_enabled() &&
        in.is_pure_decode && in.have_custom_mask &&
        // AC-2/AC-4: neither lora nor hooks disarm the split — the
        // correction lands in the shared QKV, and hooked lanes sit in
        // the unmasked prefix (order [plain|trunc|hooked|masked]); the
        // prefix decode consults the hook-narrowed page views.
        in.unmasked_prefix_rows != 0xffffffffu &&
        in.unmasked_prefix_rows < static_cast<std::uint32_t>(in.forward_R);
    // THE MIXED FIRE (M-2): a prefill-shaped fire with a planned
    // unmasked prefix — prefill + plain-decode rows serve the causal
    // dispatch, the masked 1-token suffix the custom dispatch. The
    // planned word counts TOKEN ROWS here; the request split derives
    // from the host qo indptr (the suffix starts on a request
    // boundary). Prepare re-validates the shape and keeps the
    // fire-level arm when it does not hold.
    // DEFAULT ON (PIE_SPATIAL_MIXED=0 disarms): the mixed fire — a
    // prefill-shaped masked fire splits into the prefix causal dispatch
    // and the masked suffix custom dispatch (its own plan workspace, its
    // own stream). All three legs serve it; numerics sit in the generic
    // co-batch equality class (control-proven).
    static const bool spatial_mixed_armed = [] {
        const char* v = std::getenv("PIE_SPATIAL_MIXED");
        return v == nullptr || v[0] != '0';
    }();
    const bool use_spatial_mask_mixed = spatial_mixed_armed &&
        spatial_mask_enabled() &&
        !in.is_pure_decode && in.have_custom_mask && !has_hooks &&
        in.lora == nullptr &&
        in.unmasked_prefix_rows != 0xffffffffu &&
        in.unmasked_prefix_rows > 0 &&
        in.unmasked_prefix_rows < static_cast<std::uint32_t>(in.forward_R);
    bool run_graph = graph_eligible;
    if (!use_spatial_mask && in.is_pure_decode && in.have_custom_mask &&
        std::getenv("PIE_SPATIAL_MASK_TRACE")) {
        std::fprintf(stderr,
                     "[spatial-mask] REJECT R=%d planned=%u hooks=%d "
                     "lora=%d enabled=%d\n",
                     in.forward_R, in.unmasked_prefix_rows,
                     has_hooks ? 1 : 0, in.lora != nullptr ? 1 : 0,
                     spatial_mask_enabled() ? 1 : 0);
    }
    // AC-0 observability: one line per fire with every axis's state —
    // the composition truth table's raw data (PIE_FIRE_TRACE=1).
    if (std::getenv("PIE_FIRE_TRACE") != nullptr) {
        std::fprintf(
            stderr,
            "[fire] R=%d N=%d mask=%d hooks=%d lora=%d k=%d dsplit=%d "
            "msplit=%d\n",
            in.forward_R, in.forward_N,
            in.have_custom_mask ? 1 : 0,
            has_hooks ? 1 : 0,
            in.lora != nullptr ? 1 : 0,
            in.planned_max_layers == 0xffffffffu
                ? -1
                : static_cast<int>(in.planned_max_layers),
            in.planned_full_depth_rows == 0xffffffffu
                ? -1
                : static_cast<int>(in.planned_full_depth_rows),
            in.unmasked_prefix_rows == 0xffffffffu
                ? -1
                : static_cast<int>(in.unmasked_prefix_rows));
    }
    if (in.have_custom_mask && std::getenv("PIE_SPATIAL_MASK_TRACE")) {
        std::fprintf(stderr,
                     "[spatial-mask] SHAPE R=%d N=%u pure=%d planned=%u "
                     "mixed_gate=%d\n",
                     in.forward_R,
                     in.h_qo_forward != nullptr
                         ? in.h_qo_forward[in.forward_R]
                         : 0u,
                     in.is_pure_decode ? 1 : 0,
                     in.unmasked_prefix_rows,
                     use_spatial_mask_mixed ? 1 : 0);
    }
    if (use_spatial_mask_mixed && std::getenv("PIE_SPATIAL_MASK_TRACE")) {
        std::fprintf(stderr,
                     "[spatial-mask] MIXED R=%d N=%u rows_split=%u qo=[",
                     in.forward_R,
                     in.h_qo_forward != nullptr
                         ? in.h_qo_forward[in.forward_R]
                         : 0u,
                     in.unmasked_prefix_rows);
        for (int r = 0; r <= in.forward_R && r <= 8; ++r) {
            std::fprintf(stderr, "%u%s",
                         in.h_qo_forward != nullptr ? in.h_qo_forward[r]
                                                    : 0u,
                         r < in.forward_R ? "," : "");
        }
        std::fprintf(stderr, "] (prefix causal + suffix custom)\n");
    }
    if (use_spatial_mask && std::getenv("PIE_SPATIAL_MASK_TRACE")) {
        std::fprintf(stderr,
                     "[spatial-mask] R=%d split=%u (prefix decode + suffix "
                     "custom, one fire)\n",
                     in.forward_R, in.unmasked_prefix_rows);
    }
    std::uint64_t hook_fingerprint = 0;
    // Eager-path unification (the increment-4 "future point"): the
    // fire-level `prepare_replay` pass runs for EVERY pure-decode hook fire
    // — eager and graph alike — so both modes drive the attention phases
    // through the same prepare-then-launch seam and the in-body
    // `execute_attention_phase` is a cursor-checked launch replay either
    // way; graph mode merely adds capture on top. Restricted to pure decode
    // because that is the fire class the prepare pass's sideband planners
    // model (`prepare_decode_score_capture` is decode-shaped; a prefill
    // fire's body publishes the PREFILL score capture, which carves a
    // different arena block than the plan expects) — exactly the class
    // graph mode has always prepared. A 0 return (veto: Query readers,
    // lora/envelope_dot, scalable nucleus, off-boundary lanes, …) has no
    // side effects and the fire runs the legacy interleaved eager body,
    // which reproduces any refusal loudly.
    bool hook_prepared = false;
    HookGraphKeyState* hook_state = nullptr;
    HookGraphKeyState::Entry* hook_entry = nullptr;
    // The unionized supergraph (S3): a pure-decode fire whose
    // attachments live inside the union (mask x write-desc; no hooks, no
    // lora — graph_eligible already excludes lora and hook-blocked
    // fires) replays the conditional graph, and its cache key FOLDS the
    // mask bit: masked and unmasked fires share the exec, the branch is
    // the device predicate word's job.
    const bool use_supergraph = supergraph_enabled() &&
        engine.forward_fn.supports_supergraph && graph_eligible &&
        in.is_pure_decode && !has_hooks && in.lora == nullptr &&
        // NS-3: a spatial-split fire must NOT take the union's fire-level
        // mask arm; it graphs in its own split-keyed partition.
        !use_spatial_mask;
    ForwardGraphKey key{};
    if (graph_eligible) {
        const std::uint32_t graph_layout = use_supergraph
            ? engine.forward_fn.invoke_supergraph_graph_layout()
            : engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(
                /*small_spec=*/false,
                /*rs_verify=*/false,
                use_supergraph ? false : in.have_custom_mask,
                /*fused_argmax=*/in.logits_argmax_chunk_tokens > 0,
                graph_layout,
                /*has_hooks=*/has_hooks) |
            (use_supergraph ? kGvSupergraph : 0u) |
            (in.lora != nullptr ? kGvLora : 0u) |
            (use_spatial_mask ? kGvSpatial : 0u);
        key = ForwardGraphKey{
            in.forward_R,
            in.forward_N,
            graph_variant,
        };
    }
    if (has_hooks) {
        if (in.stage_hooks->prepare_replay != nullptr && in.is_pure_decode) {
            hook_fingerprint = in.stage_hooks->prepare_replay(
                in.stage_hooks->context, cublas.stream());
            hook_prepared = hook_fingerprint != 0;
            if (!hook_prepared) {
                ++g_hook_graph_counters.prepare_vetoes;
                if (hook_graph_trace_enabled()) {
                    std::fprintf(
                        stderr,
                        "[hook-graph] prepare veto R=%d N=%d "
                        "-> legacy eager (vetoes=%llu)\n",
                        in.forward_R, in.forward_N,
                        static_cast<unsigned long long>(
                            g_hook_graph_counters.prepare_vetoes));
                }
            }
        }
        if (!hook_prepared) {
            run_graph = false;
        } else if (run_graph) {
            hook_state = &engine.hook_graph_states[key];
            // Exec storage is partitioned by the fire's program-set hash:
            // snapkv and h2o at one (R, N, variant) prepare different
            // baked state, and a single slot would ping-pong recaptures.
            hook_entry = hook_state->find(in.hook_program_set_hash);
            if (hook_state->banned ||
                (hook_entry != nullptr && hook_entry->banned)) {
                // A ban is a capture-cost economy, not a correctness veto:
                // the fire still runs PREPARED, just eagerly — same
                // launches, no capture churn.
                run_graph = false;
            }
        }
    }
    if (has_hooks && !run_graph && hook_graph_trace_enabled()) {
        static std::atomic<std::uint64_t> eager_logs{0};
        if (eager_logs.fetch_add(1, std::memory_order_relaxed) < 32) {
            std::fprintf(
                stderr,
                "[hook-graph] eager hook fire: prepared=%d blocks=%d "
                "eligible=%d pure_decode=%d custom_mask=%d write_desc=%d "
                "window=%d slots=%d R=%d N=%d wants_mask=%d cap=%d lora=%d\n",
                hook_prepared ? 1 : 0,
                hook_blocks ? 1 : 0, graph_eligible ? 1 : 0,
                in.is_pure_decode ? 1 : 0, in.have_custom_mask ? 1 : 0,
                in.has_write_desc ? 1 : 0, in.structured_window_left,
                in.use_slots ? 1 : 0, in.forward_R, in.forward_N,
                in.stage_hooks->wants_page_mask ? 1 : 0,
                engine.forward_fn.supports_hook_graph_capture ? 1 : 0,
                in.lora != nullptr ? 1 : 0);
        }
    }
    // Lora campaign step 3b: lora execs live in per-fingerprint entries
    // (the hook store's discipline; the fingerprint IS the entry hash, so
    // a changed lane structure selects a different entry rather than a
    // stale one). Ban semantics carry over unchanged.
    const bool has_lora_fire = in.lora != nullptr;
    HookGraphKeyState* lora_store = nullptr;
    HookGraphKeyState::Entry* lora_entry = nullptr;
    // PIE_LORA_GRAPH=0 keeps lora fires eager (the measurement/rollback
    // lever; default on since step 3b's batteries).
    static const bool lora_graphs_enabled = [] {
        const char* v = std::getenv("PIE_LORA_GRAPH");
        return v == nullptr || v[0] != '0';
    }();
    if (run_graph && has_lora_fire && !lora_graphs_enabled) {
        run_graph = false;
    }
    if (run_graph && has_lora_fire) {
        if (lora_fingerprint == 0) {
            // The model has no stage support (or nothing usable) — a
            // lora fire without staged state must not capture.
            run_graph = false;
        } else {
            lora_store = &engine.lora_graph_states[key];
            lora_entry = lora_store->find(lora_fingerprint);
            if (lora_store->banned ||
                (lora_entry != nullptr && lora_entry->banned)) {
                run_graph = false;
            }
        }
    }
    if (run_graph) {
        // Hook execs live in the per-program-set entries of
        // `hook_graph_states`, NOT in the shared shape-keyed cache — two hook
        // programs at one (R, N, variant) capture different graphs.
        cudaGraphExec_t exec = has_hooks
            ? (hook_entry != nullptr ? hook_entry->exec : nullptr)
            : has_lora_fire
                ? (lora_entry != nullptr ? lora_entry->exec : nullptr)
                : engine.graph_cache->get(key);
        const bool stale =
            has_hooks && exec != nullptr &&
            hook_entry->fingerprint != hook_fingerprint;
        if (exec == nullptr || stale) {
            if (step_profile_enabled()) {
                std::fprintf(stderr,
                             "[step-profile] graph capture R=%d N=%d variant=%u"
                             " (cache size %zu)\n",
                             key.num_requests, key.num_tokens, key.variant,
                             engine.graph_cache->size());
            }
            if (stale) {
                ++g_hook_graph_counters.recaptures;
                ++hook_entry->mismatches;
                if (hook_entry->mismatches >
                    HookGraphKeyState::kMaxMismatches) {
                    // This fire still captures+launches correctly; future
                    // fires of this program set stop paying ~10 ms per
                    // capture. Other program sets on the key keep replaying.
                    hook_entry->banned = true;
                    ++g_hook_graph_counters.bans;
                    if (hook_graph_trace_enabled()) {
                        std::fprintf(
                            stderr,
                            "[hook-graph] BAN R=%d N=%d variant=%u "
                            "ps=%016llx after %u fingerprint churns\n",
                            key.num_requests, key.num_tokens, key.variant,
                            static_cast<unsigned long long>(
                                in.hook_program_set_hash),
                            hook_entry->mismatches);
                    }
                }
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
                in.compact_logits ? in.num_sampling : 0,
                pi.w_page.data(),
                pi.w_off.data(),
                in.has_write_desc,
                in.structured_window_left,
                has_hooks ? in.stage_hooks : nullptr,
                use_supergraph,
                in.lora,
                (use_spatial_mask || use_spatial_mask_mixed)
                    ? in.unmasked_prefix_rows
                    : 0xffffffffu,
                in.logits_argmax_chunk_tokens);
            if (has_hooks) {
                // The capture is the one moment the model's per-layer hook
                // coverage is observable; a body that skipped hooks would
                // otherwise replay an incomplete graph forever.
                if (in.stage_hooks->verify_replay_capture != nullptr) {
                    in.stage_hooks->verify_replay_capture(
                        in.stage_hooks->context);
                }
                if (hook_entry == nullptr) {
                    hook_entry =
                        hook_state->insert(in.hook_program_set_hash);
                    if (hook_state->banned) {
                        // Program-set churn ban (LRU thrash — more live
                        // program sets than entry slots would recapture per
                        // fire). This exec still launches below; future
                        // fires of the key go eager.
                        ++g_hook_graph_counters.bans;
                        if (hook_graph_trace_enabled()) {
                            std::fprintf(
                                stderr,
                                "[hook-graph] BAN R=%d N=%d variant=%u "
                                "after %u program-set evictions\n",
                                key.num_requests, key.num_tokens,
                                key.variant, hook_state->evictions);
                        }
                    }
                } else if (hook_entry->exec != nullptr) {
                    // Stale recapture: replace this program set's exec.
                    cudaGraphExecDestroy(hook_entry->exec);
                }
                hook_entry->exec = exec;
                hook_entry->fingerprint = hook_fingerprint;
                ++g_hook_graph_counters.captures;
                if (hook_graph_trace_enabled()) {
                    std::fprintf(
                        stderr,
                        "[hook-graph] capture R=%d N=%d variant=%u "
                        "ps=%016llx fp=%016llx%s (captures=%llu)\n",
                        key.num_requests, key.num_tokens, key.variant,
                        static_cast<unsigned long long>(
                            in.hook_program_set_hash),
                        static_cast<unsigned long long>(hook_fingerprint),
                        stale ? " (fingerprint recapture)" : "",
                        static_cast<unsigned long long>(
                            g_hook_graph_counters.captures));
                }
            } else if (has_lora_fire) {
                if (lora_entry == nullptr) {
                    lora_entry = lora_store->insert(lora_fingerprint);
                    if (lora_store->banned) {
                        ++g_hook_graph_counters.bans;
                    }
                } else if (lora_entry->exec != nullptr) {
                    cudaGraphExecDestroy(lora_entry->exec);
                }
                lora_entry->exec = exec;
                lora_entry->fingerprint = lora_fingerprint;
                if (hook_graph_trace_enabled()) {
                    std::fprintf(
                        stderr,
                        "[lora-graph] capture R=%d N=%d variant=%u "
                        "fp=%016llx\n",
                        key.num_requests, key.num_tokens, key.variant,
                        static_cast<unsigned long long>(lora_fingerprint));
                }
            } else {
                if (use_supergraph) {
                    if (step_profile_enabled()) {
                        std::fprintf(
                            stderr,
                            "[supergraph] capture at lookup variant=%u "
                            "(mask=%d wdesc=%d)\n",
                            key.variant, in.have_custom_mask ? 1 : 0,
                            in.has_write_desc ? 1 : 0);
                    }
                    // The union key includes the CUSTOM prefill plan's
                    // layout, and the capture's dual-prepare is what
                    // materializes that plan on a bucket's first fire —
                    // so the pre-capture key was computed against a null
                    // plan. Re-key from the post-capture state and store
                    // under THAT: the next fire (masked or not) recomputes
                    // the same pair and hits.
                    const std::uint32_t graph_layout =
                        engine.forward_fn.invoke_supergraph_graph_layout();
                    key.variant = make_graph_variant(
                        /*small_spec=*/false,
                        /*rs_verify=*/false,
                        /*custom_mask=*/false,
                        graph_layout,
                        /*has_hooks=*/false) | kGvSupergraph;
                    if (step_profile_enabled()) {
                        std::fprintf(stderr,
                                     "[supergraph] stored variant=%u\n",
                                     key.variant);
                    }
                }
                engine.graph_cache->put(key, exec);
            }
        } else if (has_hooks) {
            // A clean replay proves the entry's baked state is stable again,
            // so the churn ban only counts CONSECUTIVE stale fires. Without
            // the reset, the legitimate one-recapture-per-new-instance
            // cadence (instance churn re-bakes sideband addresses) would
            // accumulate to a ban after ~kMaxMismatches requests, forcing
            // eager forever on a healthy key.
            hook_entry->mismatches = 0;
            ++g_hook_graph_counters.replays;
            if (hook_graph_trace_enabled()) {
                std::fprintf(
                    stderr,
                    "[hook-graph] replay R=%d N=%d variant=%u ps=%016llx "
                    "(replays=%llu)\n",
                    key.num_requests, key.num_tokens, key.variant,
                    static_cast<unsigned long long>(
                        in.hook_program_set_hash),
                    static_cast<unsigned long long>(
                        g_hook_graph_counters.replays));
            }
        }
        if (use_supergraph) {
            // Arm the fire's branches: the graph's set_cond kernels read
            // these slots (batch/supergraph.hpp). Peel's all-fast bit is
            // constitutively 1 here — hook fires are outside the union,
            // so every row is hook-free.
            std::uint8_t preds[batch::kSupergraphPredSlots] = {};
            preds[0] = in.has_write_desc ? 1 : 0;
            preds[4] = in.have_custom_mask ? 1 : 0;
            preds[batch::kPredSlotPeelAllFast] = 1;
            preds[batch::kPredSlotPeelAllHooked] = 0;
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.supergraph_preds.data(), preds,
                sizeof(preds), cudaMemcpyHostToDevice, cublas.stream()));
        }
        if (use_spatial_mask) {
            // The captured suffix dispatch reads the identity qo from
            // pi.mask_suffix_qo_indptr; identity content is
            // split-invariant, but re-upload per fire keeps the buffer
            // owned by no particular capture (R+1 u32s, trivial).
            std::vector<std::uint32_t> qo(
                static_cast<std::size_t>(in.forward_R) + 1);
            for (int i = 0; i <= in.forward_R; ++i) {
                qo[static_cast<std::size_t>(i)] =
                    static_cast<std::uint32_t>(i);
            }
            CUDA_CHECK(cudaMemcpy(
                engine.inputs.mask_suffix_qo_indptr.data(), qo.data(),
                qo.size() * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice));
        }
        if (has_hooks) {
            // Arm the fire's Peel window: the captured devwin kernels read
            // {tail_start, tail_len} from this word, so ONE exec serves
            // every row split (the device-window campaign's endpoint).
            const std::uint32_t fast = std::min(
                in.stage_hooks->hook_free_prefix_rows,
                static_cast<std::uint32_t>(std::max(in.forward_R, 0)));
            const std::uint32_t win[2] = {
                fast, static_cast<std::uint32_t>(in.forward_N) - fast};
            CUDA_CHECK(cudaMemcpyAsync(
                engine.inputs.peel_window.data(), win, sizeof(win),
                cudaMemcpyHostToDevice, cublas.stream()));
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
        in.compact_logits ? in.num_sampling : 0;
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
    fwd_in.lora                         = in.lora;
    fwd_in.max_layers                   = in.planned_max_layers;
    fwd_in.full_depth_rows              = in.planned_full_depth_rows;
    fwd_in.depth_band_k                 = in.depth_band_k;
    fwd_in.depth_band_rows              = in.depth_band_rows;
    fwd_in.depth_band_count             = in.depth_band_count;
    if (use_spatial_mask || use_spatial_mask_mixed) {
        // The masked suffix's rebased device CSRs (pure decode: qo is the
        // identity, kv_page_indptr rebases by its page base; every other
        // suffix array is a pointer offset in the body). Synchronous
        // copies: tiny, eager-only, and the host staging dies here.
        // The planned word is the REQUEST/lane index in BOTH shapes
        // (measured live on the mixed fire: R=4 qo=[0,221,...] plans 3,
        // the masked member's lane start) — pure-decode fires never
        // showed the distinction (row == lane).
        const int split = static_cast<int>(in.unmasked_prefix_rows);
        const int rs = in.forward_R - split;
        std::vector<std::uint32_t> qo(static_cast<std::size_t>(rs) + 1);
        std::vector<std::uint32_t> kvpp(static_cast<std::size_t>(rs) + 1);
        const std::uint32_t page_base = in.h_kvpp_forward[split];
        for (int i = 0; i <= rs; ++i) {
            qo[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
            kvpp[static_cast<std::size_t>(i)] =
                in.h_kvpp_forward[split + i] - page_base;
        }
        CUDA_CHECK(cudaMemcpy(
            pi.mask_suffix_qo_indptr.data(), qo.data(),
            qo.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            pi.mask_suffix_kv_page_indptr.data(), kvpp.data(),
            kvpp.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        fwd_in.unmasked_prefix_rows = in.unmasked_prefix_rows;
        fwd_in.mask_suffix_qo_indptr_d = pi.mask_suffix_qo_indptr.data();
        fwd_in.mask_suffix_kv_page_indptr_d =
            pi.mask_suffix_kv_page_indptr.data();
    }
    forward_fn.invoke_body(ws, kv_cache, attn_ws, cublas, fwd_in);
    if (hook_prepared && in.stage_hooks->verify_replay_capture != nullptr) {
        // Prepared-EAGER hook fire (the unified seam, no capture): the same
        // coverage proof as capture time. The prepare pass pre-credited the
        // per-layer invocation counters, so a body that stopped invoking its
        // hooks would otherwise go unnoticed — `finish`'s partial-consumption
        // check reads 0 consumed as "graph replay" and cannot distinguish a
        // body that skipped every hook.
        in.stage_hooks->verify_replay_capture(in.stage_hooks->context);
    }
}

}  // namespace pie_cuda_driver
