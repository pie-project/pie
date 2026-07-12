#include "batch/tp.hpp"

#include "batch/forward.hpp"
#include "batch/graph_variant.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"

namespace pie_cuda_driver {

namespace {

struct TpCpuGate {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<std::uint64_t> seq{0};
};

std::mutex g_tp_cpu_gates_mu;
std::unordered_map<std::string, std::shared_ptr<TpCpuGate>> g_tp_cpu_gates;

std::shared_ptr<TpCpuGate> tp_cpu_gate_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_cpu_gates_mu);
    auto& gate = g_tp_cpu_gates[key];
    if (!gate) gate = std::make_shared<TpCpuGate>();
    return gate;
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
// through BatchEngine.
std::int32_t* tp_hdr_dev_buf() {
    thread_local std::int32_t* buf = nullptr;
    if (buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf, sizeof(TpFireHeader)));
    }
    return buf;
}

}  // namespace

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

void tp_cpu_gate_notify(const std::string& key) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    gate->seq.fetch_add(1, std::memory_order_release);
    gate->cv.notify_all();
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
void tp_follower_serve(BatchEngine& engine, std::atomic<bool>& stop) {
    if (engine.tp_comm == nullptr) {
        std::cerr << "[pie-driver-cuda] tp_follower_serve: no tp_comm\n";
        return;
    }
    auto& pi      = engine.inputs;
    auto& comm    = *engine.tp_comm;
    auto* d_hdr   = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    std::uint64_t cpu_gate_seq = 0;

    // Sized lazily; R is at most max_workspace_tokens (one request per token).
    std::vector<std::uint32_t> h_qo, h_kvpp;

    while (!stop.load()) {
        tp_cpu_gate_wait(engine.tp_cpu_gate_key, cpu_gate_seq, stop);
        // 1. Receive header.
        NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader),
                                       ncclChar, 0, comm.comm(), stream),
                         comm.comm());
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
        engine.forward_fn.invoke_prepare(
            engine.attn_ws,
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
            engine.graph_cache != nullptr && is_pure_decode && !have_custom_mask
            && engine.forward_fn.graph_safe;
        const std::uint32_t graph_layout =
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(tp_greedy_argmax, /*single_gpu=*/false,
                               /*fwd_handles=*/false, /*small_spec=*/false,
                               /*rs_verify=*/false, graph_layout);
        if (try_graphs) {
            const ForwardGraphKey key{R, N, graph_variant};
            cudaGraphExec_t exec = engine.graph_cache->get(key);
            if (exec == nullptr) {
                exec = capture_forward_graph_exec(
                    engine, h_qo.data(), h_kvpi.data(), h_kvpp.data(),
                    h_kvlpl.data(),
                    N, R, is_pure_decode,
                    have_slot_ids ? h_slot_ids.data() : nullptr,
                    have_slot_ids ? h_is_fresh.data() : nullptr,
                    have_slot_ids ? pi.slot_ids.data() : nullptr,
                    logit_rows > 0 ? pi.sample_idx.data() : nullptr,
                    logit_rows,
                    /*single_gpu_greedy_argmax=*/false,
                    tp_greedy_argmax);
                engine.graph_cache->put(key, exec);
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
            engine.forward_fn.invoke_body(
                engine.ws, engine.kv_cache, engine.attn_ws, engine.cublas,
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
