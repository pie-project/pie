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
#include "response_subpass.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_temp.hpp"
#include "sampler_type.hpp"
#include "sampling_dispatch.hpp"
#include "spec_expansion.hpp"

namespace pie_cuda_driver {

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

bool graph_single_gpu_argmax_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_GRAPH_ARGMAX");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
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
    const std::uint32_t* kv_page_indptr_h,
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
    executor.forward_fn.body(
        executor.ws, executor.kv_cache, executor.attn_ws, executor.cublas,
        reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
        reinterpret_cast<const std::int32_t*>(pi.positions.data()),
        pi.qo_indptr.data(), pi.kv_page_indices.data(),
        pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
        qo_indptr_h, kv_page_indptr_h,
        N, R, is_pure_decode, nullptr, nullptr,
        slot_ids_h, is_fresh_h, slot_ids_d,
        logit_row_indices_d, num_logit_rows, tp_greedy_argmax);
    if (single_gpu_greedy_argmax) {
        const int argmax_parts = partitioned_argmax_parts();
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
    NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                             comm.comm(), stream));
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
    NCCL_CHECK(ncclGroupEnd());
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
    if (!executor.forward_fn.graph_safe || !executor.forward_fn.body) return 0;
    if (!executor.forward_fn.prepare) return 0;
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
    const bool single_gpu_graph_argmax =
        graph_single_gpu_argmax_enabled() && executor.tp_comm == nullptr;
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

        for (int r = 0; r <= R; ++r) {
            qo[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
            kvpp[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(r);
        }
        for (int r = 0; r < R; ++r) {
            kvpi[static_cast<std::size_t>(r)] =
                static_cast<std::uint32_t>(r % num_pages);
        }

        pi.tokens.copy_from_host(std::span<const std::uint32_t>(tokens));
        pi.positions.copy_from_host(std::span<const std::uint32_t>(positions));
        pi.qo_indptr.copy_from_host(std::span<const std::uint32_t>(qo));
        pi.kv_page_indices.copy_from_host(std::span<const std::uint32_t>(kvpi));
        pi.kv_page_indptr.copy_from_host(std::span<const std::uint32_t>(kvpp));
        pi.kv_last_page_lens.copy_from_host(std::span<const std::uint32_t>(kvlpl));

        executor.forward_fn.prepare(
            executor.attn_ws,
            ForwardFn::PrepareInputs{
                .qo_indptr_h = qo.data(),
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
            executor.forward_fn.graph_layout ? executor.forward_fn.graph_layout() : 0u;
        const std::uint32_t graph_variant =
            (tp_greedy_argmax ? 1u : 0u) |
            (single_gpu_graph_argmax ? 2u : 0u) |
            (graph_layout << 2);
        const ForwardGraphKey key{R, graph_variant};
        if (executor.graph_cache->get(key) != nullptr) continue;

        tp_graph_capture_barrier(executor);
        cudaGraphExec_t exec = capture_forward_graph_exec(
            executor, qo.data(), kvpp.data(), N, R, /*is_pure_decode=*/true,
            /*slot_ids_h=*/nullptr, /*is_fresh_h=*/nullptr,
            /*slot_ids_d=*/nullptr, /*logit_row_indices_d=*/nullptr,
            /*num_logit_rows=*/0, single_gpu_graph_argmax,
            tp_greedy_argmax);
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

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_driver::PieForwardRequestView& view,
    pie_driver::PieForwardResponseView& out_resp,
    Executor& executor,
    std::uint64_t handled)
{
    // Local references so the (lifted) body uses the same names it had
    // when it lived as a `[&]`-capturing lambda in main.cpp. Avoids a
    // mechanical rename across ~900 lines.
    auto& engine               = executor.loaded_model;
    auto& ws                   = executor.ws;
    auto& kv_cache             = executor.kv_cache;
    auto& attn_ws              = executor.attn_ws;
    auto& cublas               = executor.cublas;
    auto& pi                   = executor.inputs;  // persistent input slabs
    auto& forward_fn           = executor.forward_fn;
    const int max_workspace_tokens = executor.max_workspace_tokens;
    const int max_forward_requests = executor.max_forward_requests;
    const int graph_pad_page = executor.graph_pad_page;

    // Track whether the custom-mask path was populated this fire so the
    // forward kernel knows whether to consume `pi.custom_mask`. Sizes are
    // stashed alongside so the TP broadcast knows how many bytes to fan
    // out to followers.
    bool have_custom_mask = false;
    int mask_bytes = 0;
    int mask_indptr_count = 0;

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

        const int N = static_cast<int>(tok_view.size());
        const int num_sampling = static_cast<int>(sidx_view.size());

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

        bool graph_shape_ok =
            executor.graph_cache != nullptr && is_pure_decode &&
            forward_fn.graph_safe;
        int graph_requests = R;
        if (graph_shape_ok) {
            const int bucket =
                forward_graph_request_bucket(R, max_forward_requests);
            const int pad = bucket - R;
            const bool exact_bucket = (bucket == R);
            const bool can_pad =
                bucket > R &&
                graph_pad_page >= 0 &&
                bucket <= max_workspace_tokens &&
                pad <= page_size &&
                kvpi_view.size() + static_cast<std::size_t>(pad) <=
                    pi.kv_page_indices.size();
            graph_shape_ok = bucket > 0 && (exact_bucket || can_pad);
            if (graph_shape_ok) graph_requests = bucket;
        }

        ForwardInputViews forward_inputs = make_forward_input_views(
            tok_view, pos_view, qo_view, kvpi_view, kvpp_view, kvlpl_view,
            R, graph_requests, graph_pad_page);
        const int forward_N = forward_inputs.total_tokens;
        const int forward_R = forward_inputs.num_requests;
        const std::uint32_t* h_qo_forward = forward_inputs.qo_indptr.data();
        const std::uint32_t* h_kvpp_forward =
            forward_inputs.kv_page_indptr.data();

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
        const auto rs_slot_view = view.rs_slot_ids.as<std::uint32_t>();
        const auto rs_flag_view = view.rs_slot_flags.as<std::uint8_t>();
        bool use_slots =
            R > 0 && rs_slot_view.size() == static_cast<std::size_t>(R);
        if (executor.rs_cache != nullptr && R > 0 && !use_slots) {
            throw std::runtime_error(
                "rs_cache forward missing runtime-assigned slot ids");
        }
        if (use_slots) {
            slot_ids_h.resize(R);
            is_fresh_h.resize(R);
            for (int r = 0; r < R; ++r) {
                slot_ids_h[r] = static_cast<std::int32_t>(rs_slot_view[r]);
                is_fresh_h[r] = (r < static_cast<int>(rs_flag_view.size()) &&
                                 (rs_flag_view[r] & 1u))
                                    ? 1u
                                    : 0u;
            }
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids_h));
            pi.is_fresh.copy_from_host(std::span<const std::uint8_t>(is_fresh_h));
        }

        // ── Sample-plan construction (hoisted) ──────────────────
        // Sampling stays outside the CUDA graph because sampler/probe
        // layouts can vary even when the decode shape is identical. Build
        // the per-row plan before forward so the common response path can
        // upload stable device inputs once and launch sampling after the
        // forward body has produced logits.
        const auto outspec_view = view.output_spec_flags.as<std::uint8_t>();
        bool need_msgpack = false;
        for (auto t : types_view) {
            if (pie_cuda_driver::is_msgpack_only(t)) { need_msgpack = true; break; }
        }
        if (!need_msgpack) {
            for (auto f : outspec_view) {
                if (f) { need_msgpack = true; break; }
            }
        }
        if (has_spec_drafts) need_msgpack = true;

        const std::uint32_t* h_sptr  = sptr_view.data();
        const std::uint32_t* h_sidx  = sidx_view.data();
        const std::uint32_t* h_rns   = rns_view.data();
        const float*         h_temp  = temp_view.data();
        const std::uint32_t* h_top_k = top_k_view.data();
        const float*         h_top_p = top_p_view.data();
        const float*         h_min_p = min_p_view.data();
        const std::uint32_t* h_seed  = seed_view.data();

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

        bool fast_dense_greedy_argmax =
            executor.tp_comm == nullptr &&
            !have_custom_mask &&
            !need_msgpack &&
            !has_spec_drafts &&
            logit_masks_view.empty() &&
            is_pure_decode &&
            N == R &&
            num_sampling == R &&
            sptr_view.size() == static_cast<std::size_t>(R + 1) &&
            sidx_view.size() == static_cast<std::size_t>(R) &&
            rns_view.size() >= static_cast<std::size_t>(R) &&
            types_view.size() >= static_cast<std::size_t>(R) &&
            temp_view.size() >= static_cast<std::size_t>(R);
        if (fast_dense_greedy_argmax) {
            for (int r = 0; r < R; ++r) {
                if (h_sptr[r] != static_cast<std::uint32_t>(r) ||
                    h_sptr[r + 1] != static_cast<std::uint32_t>(r + 1) ||
                    h_sidx[r] != 0u ||
                    h_rns[r] != 1u ||
                    !pie_cuda_driver::is_token_sampler(types_view[r]) ||
                    h_temp[r] > 0.f) {
                    fast_dense_greedy_argmax = false;
                    break;
                }
            }
        }

        bool any_topk_topp = false;
        bool sample_rows_are_dense = fast_dense_greedy_argmax;
        bool all_rows_greedy = fast_dense_greedy_argmax;
        bool all_slots_token = fast_dense_greedy_argmax;
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
                    (rns_view.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
                const std::uint32_t lo = h_sptr[r];
                const std::uint32_t hi = h_sptr[r + 1];
                const std::uint32_t qo_lo = h_qo[r];
                for (std::uint32_t k = lo; k < hi; ++k) {
                    const std::uint32_t s_idx = sampler_off + (k - lo);
                    const std::uint32_t type =
                        (s_idx < types_view.size()) ? types_view[s_idx] : 1u;
                    per_slot_type[k] = type;
                    const float T = (s_idx < temp_view.size()) ? h_temp[s_idx] : 1.f;
                    const float Tp = (s_idx < top_p_view.size()) ? h_top_p[s_idx] : 1.f;
                    const float Mp = (s_idx < min_p_view.size()) ? h_min_p[s_idx] : 0.f;
                    const std::int32_t Tk_raw = (s_idx < top_k_view.size())
                        ? static_cast<std::int32_t>(h_top_k[s_idx]) : 0;
                    const std::int32_t Tk =
                        (Tk_raw == 0) ? engine.hf_config().vocab_size : Tk_raw;
                    std::uint32_t s = (s_idx < seed_view.size()) ? h_seed[s_idx] : 0u;
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
                        if ((Tk_raw > 0 || Tp < 1.f) && T > 0.f) any_topk_topp = true;
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

            // Per-slot → row mapping for the topk+top-p scatter.
            h_sample_idx.assign(static_cast<std::size_t>(num_sampling), 0);
            int k_g = 0;
            for (int r = 0; r < R; ++r) {
                const std::uint32_t qo_lo = h_qo[r];
                for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k, ++k_g) {
                    h_sample_idx[k_g] =
                        static_cast<std::int32_t>(qo_lo + h_sidx[k]);
                }
            }

            sample_rows_are_dense = (num_sampling == N);
            if (sample_rows_are_dense) {
                for (int i = 0; i < N; ++i) {
                    if (h_sample_idx[i] != i) {
                        sample_rows_are_dense = false;
                        break;
                    }
                }
            }
            all_rows_greedy = true;
            for (int i = 0; i < N; ++i) {
                if (h_per_temp[i] > 0.f) {
                    all_rows_greedy = false;
                    break;
                }
            }
            all_slots_token = true;
            for (auto type : per_slot_type) {
                if (!pie_cuda_driver::is_token_sampler(type)) {
                    all_slots_token = false;
                    break;
                }
            }
        }
        const bool compact_logit_rows =
            executor.forward_fn.supports_compact_logits &&
            !is_pure_decode &&
            !have_custom_mask &&
            !need_msgpack &&
            !has_spec_drafts &&
            logit_masks_view.empty() &&
            !any_topk_topp &&
            all_slots_token &&
            num_sampling > 0 &&
            num_sampling < N;
        const bool tp_greedy_argmax =
            executor.tp_comm != nullptr &&
            executor.tp_comm->world_size() <= 8 &&
            forward_fn.supports_tp_greedy_argmax &&
            !have_custom_mask &&
            !need_msgpack &&
            !has_spec_drafts &&
            logit_masks_view.empty() &&
            !any_topk_topp &&
            all_slots_token &&
            all_rows_greedy &&
            is_pure_decode &&
            sample_rows_are_dense;
        // Dense greedy decode can bypass the general sampler machinery: a
        // single argmax over logits produces the requested token output.
        const bool single_gpu_greedy_argmax =
            executor.tp_comm == nullptr &&
            !have_custom_mask &&
            !need_msgpack &&
            !has_spec_drafts &&
            logit_masks_view.empty() &&
            !any_topk_topp &&
            all_slots_token &&
            all_rows_greedy &&
            is_pure_decode &&
            sample_rows_are_dense;
        const int logit_rows_required =
            compact_logit_rows ? num_sampling : N;
        const int prob_rows_required = any_topk_topp ? N : 0;
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
        if (forward_fn.prepare) {
            forward_fn.prepare(
                attn_ws,
                ForwardFn::PrepareInputs{
                    .qo_indptr_h = h_qo_forward,
                    .kv_page_indices_d =
                        reinterpret_cast<const std::uint32_t*>(pi.kv_page_indices.data()),
                    .kv_page_indptr_h = h_kvpp_forward,
                    .kv_page_indptr_d =
                        reinterpret_cast<const std::uint32_t*>(pi.kv_page_indptr.data()),
                    .kv_last_page_lens_h =
                        forward_inputs.kv_last_page_lens.data(),
                    .kv_last_page_lens_d =
                        reinterpret_cast<const std::uint32_t*>(pi.kv_last_page_lens.data()),
                    .total_tokens = forward_N,
                    .num_requests = forward_R,
                    .is_pure_decode = is_pure_decode,
                });
        }

        // ── Upload sampling inputs (must precede sampling launch) ──
        // Per-row sampler params land in `pi.sample_*`. Sampling runs after
        // forward, but the upload is kept here so the response path can use
        // the same prepared device buffers for captured and uncaptured
        // forward bodies.
        if (!tp_greedy_argmax && !single_gpu_greedy_argmax) {
            upload_sampling_inputs(pi, sample_plan, N, /*stream=*/nullptr);
        }

        // ── Forward pass ────────────────────────────────────────
        // Graph-capture path activates only when the arch declares itself
        // graph-safe. Today that means llama-like pure decode, where the
        // per-fire attention plan is prepared before capture/replay and the
        // captured body observes stable device pointers.
        const bool try_graphs =
            graph_shape_ok && !have_custom_mask;
        const std::uint32_t graph_layout =
            forward_fn.graph_layout ? forward_fn.graph_layout() : 0u;
        const bool graph_captures_single_gpu_argmax =
            try_graphs && single_gpu_greedy_argmax &&
            graph_single_gpu_argmax_enabled();
        const std::uint32_t graph_variant =
            (tp_greedy_argmax ? 1u : 0u) |
            (graph_captures_single_gpu_argmax ? 2u : 0u) |
            (graph_layout << 2);
        if (try_graphs) {
            const ForwardGraphKey key{forward_R, graph_variant};
            cudaGraphExec_t exec = executor.graph_cache->get(key);
            if (exec == nullptr) {
                // First fire of this shape: capture the forward body.
                // Body writes its output to `ws` workspace buffers +
                // `cache.k/v` pages. Persistent inputs (pi.*) provide
                // stable kernel-arg pointers; the next replay reads new
                // contents from the same addresses, refreshed by `prepare`
                // above.
                //
                // Bind cuBLAS to `cstream` for the duration of capture
                // so its kernel launches are recorded onto the captured
                // graph rather than slipping onto the default stream.
                // Relaxed mode allows cross-stream operations during
                // capture but only operations on the captured stream
                // (or streams joined to it) make it into the graph.
                //
                // The per-fire persistent inputs were uploaded on the
                // default stream above. Capture uses a nonblocking stream,
                // so start recording with no in-flight default-stream writes
                // to the stable device buffers. The capture only records the
                // graph; the exec is launched below for this same fire.
                exec = capture_forward_graph_exec(
                    executor, h_qo_forward, h_kvpp_forward,
                    forward_N, forward_R, is_pure_decode,
                    use_slots ? slot_ids_h.data() : nullptr,
                    use_slots ? is_fresh_h.data() : nullptr,
                    use_slots ? pi.slot_ids.data() : nullptr,
                    compact_logit_rows ? pi.sample_idx.data() : nullptr,
                    compact_logit_rows ? num_sampling : 0,
                    graph_captures_single_gpu_argmax,
                    tp_greedy_argmax);
                executor.graph_cache->put(key, exec);
                const auto sz = executor.graph_cache->size();
                if (executor.verbose && (sz <= 4 || sz % 16 == 0)) {
                    std::cerr << "[pie-driver-cuda] graph captured: R="
                              << forward_R
                              << (forward_inputs.padded ? " padded" : "")
                              << " real_R=" << R
                              << " variant=" << graph_variant
                              << " layout=" << graph_layout
                              << " (cache size=" << sz << ")\n";
                }
            }
            CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
        } else {
            forward_fn.body(
                ws, kv_cache, attn_ws, cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.qo_indptr.data(), pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                /*qo_indptr_h=*/h_qo_forward,
                /*kv_page_indptr_h=*/h_kvpp_forward,
                forward_N, forward_R, is_pure_decode,
                have_custom_mask ? pi.custom_mask.data()        : nullptr,
                have_custom_mask ? pi.custom_mask_indptr.data() : nullptr,
                use_slots ? slot_ids_h.data() : nullptr,
                use_slots ? is_fresh_h.data() : nullptr,
                use_slots ? pi.slot_ids.data() : nullptr,
                compact_logit_rows ? pi.sample_idx.data() : nullptr,
                compact_logit_rows ? num_sampling : 0,
                tp_greedy_argmax);
        }
        // Sampling is deliberately outside the CUDA graph. The forward
        // graph key is only `R`; sampler/probe layouts vary independently
        // (for example top-p token-only decode vs. argmax + rich probes).
        // Running the current fire's sampling kernel after the graph launch
        // keeps that R-only key valid.
        if (tp_greedy_argmax) {
            kernels::launch_select_global_argmax_pairs(
                reinterpret_cast<const std::uint64_t*>(ws.greedy_pairs_all.data()),
                reinterpret_cast<std::int32_t*>(pi.sampled.data()),
                N, executor.tp_comm->world_size(), cublas.stream());
        } else if (graph_captures_single_gpu_argmax) {
            // The captured forward graph already wrote pi.sampled.
        } else if (single_gpu_greedy_argmax && !graph_captures_single_gpu_argmax) {
            const int argmax_parts = partitioned_argmax_parts();
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
        CUDA_CHECK(cudaStreamSynchronize(cublas.stream()));

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
                    bucket.assign(block.begin(), block.begin() + match + 1);
                } else {
                    const std::uint32_t lo = h_sptr[r];
                    const std::uint32_t hi = h_sptr[r + 1];
                    bucket.reserve(hi - lo);
                    for (std::uint32_t k = lo; k < hi; ++k) {
                        const std::uint32_t type = per_slot_type[k];
                        if (!pie_cuda_driver::is_token_sampler(type)) continue;
                        const std::uint32_t row = qo_lo + h_sidx[k];
                        bucket.push_back(static_cast<std::uint32_t>(all_sampled[row]));
                    }
                }
            }
            executor.response_builder.build(per_req, out_resp);

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
        NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader),
                                 ncclChar, 0, comm.comm(), stream));
        TpFireHeader hdr{};
        CUDA_CHECK(cudaMemcpyAsync(&hdr, d_hdr, sizeof(hdr),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (hdr.magic == TP_STOP_MAGIC) break;
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
        NCCL_CHECK(ncclGroupEnd());

        // 3. Pull the host views of qo/kv_page indptrs for the per-arch
        // attention planner (lives outside the captured kernel sequence).
        h_qo.resize(R + 1);
        h_kvpp.resize(R + 1);
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
        if (executor.forward_fn.prepare) {
            executor.forward_fn.prepare(
                executor.attn_ws,
                ForwardFn::PrepareInputs{
                    .qo_indptr_h = h_qo.data(),
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
        }
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
            executor.forward_fn.graph_layout ? executor.forward_fn.graph_layout() : 0u;
        const std::uint32_t graph_variant =
            (tp_greedy_argmax ? 1u : 0u) |
            (graph_layout << 2);
        if (try_graphs) {
            const ForwardGraphKey key{R, graph_variant};
            cudaGraphExec_t exec = executor.graph_cache->get(key);
            if (exec == nullptr) {
                exec = capture_forward_graph_exec(
                    executor, h_qo.data(), h_kvpp.data(),
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
            executor.forward_fn.body(
                executor.ws, executor.kv_cache, executor.attn_ws, executor.cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.qo_indptr.data(), pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                h_qo.data(), h_kvpp.data(),
                N, R, is_pure_decode,
                have_custom_mask ? pi.custom_mask.data()        : nullptr,
                have_custom_mask ? pi.custom_mask_indptr.data() : nullptr,
                have_slot_ids ? h_slot_ids.data() : nullptr,
                have_slot_ids ? h_is_fresh.data() : nullptr,
                have_slot_ids ? pi.slot_ids.data() : nullptr,
                logit_rows > 0 ? pi.sample_idx.data() : nullptr,
                logit_rows,
                tp_greedy_argmax);
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
    NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                             comm.comm(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace pie_cuda_driver
