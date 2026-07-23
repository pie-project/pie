#include "batch/tp.hpp"

#include "batch/forward.hpp"
#include "batch/graph_variant.hpp"
#include "batch/tp_gate.hpp"
#include "pipeline/batch_compose.hpp"
#include "store/recurrent_state_cache.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"

namespace pie_cuda_driver {

namespace {

std::mutex g_tp_cpu_gates_mu;
std::unordered_map<std::string, std::shared_ptr<TpSequenceGate>>
    g_tp_cpu_gates;

std::shared_ptr<TpSequenceGate> tp_cpu_gate_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_cpu_gates_mu);
    auto& gate = g_tp_cpu_gates[key];
    if (!gate) gate = std::make_shared<TpSequenceGate>();
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
        const std::uint64_t seq = gate->published();
        if (tp_cpu_gate_consume_one(seq, seen)) return;
        if (std::chrono::steady_clock::now() - start >= spin_budget) break;
        cpu_relax();
    }

    static_cast<void>(gate->wait_one(seen, stop));
}

// Broadcast header sent from rank 0 → followers before each fire's
// per-fire payload. Followers parse it to size the subsequent broadcasts
// + the forward call. Two magic values:
//
//   * TP_FIRE_MAGIC: a regular fire is incoming; payload broadcasts follow.
//   * TP_STOP_MAGIC: shutdown sentinel; follower exits its serve loop.
//
// Trivially copyable so the exact header bytes can be broadcast to followers.
struct TpFireHeader {
    std::int32_t magic;
    std::int32_t total_tokens;
    std::int32_t num_requests;
    std::int32_t is_pure_decode;
    std::int32_t kv_indices_count;
    std::int32_t required_kv_pages;
    std::int32_t mask_bytes;
    std::int32_t mask_indptr_count;
    // 1 = slot_ids[R] (int32) and is_fresh[R] (uint8) follow the
    // existing payload broadcasts. Inert (0) for archs that don't use
    // rs_cache — followers skip those broadcasts.
    std::int32_t has_slot_ids;
    // 1 = w_page[N] and w_off[N] explicit write descriptors follow.
    std::int32_t has_write_desc;
    // Number of compact logit rows in pi.sample_idx.
    std::int32_t logit_rows;
    std::int32_t structured_window_left;
    std::int32_t rs_mode;
    std::int32_t rs_fold_lens_count;
    std::int32_t rs_buffer_ids_count;
};
static_assert(std::is_trivially_copyable_v<TpFireHeader>);
constexpr std::int32_t TP_FIRE_MAGIC = 0x55504954;  // 'TPIU' tag
constexpr std::int32_t TP_MTP_MAGIC  = 0x50544D54;  // 'TMTP' tag
constexpr std::int32_t TP_STOP_MAGIC = 0x504F5453;  // 'STOP' tag

// Lazily allocated device buffer holding the broadcast header.
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
                         int required_kv_pages,
                         int mask_bytes, int mask_indptr_count,
                         bool has_slot_ids,
                         bool has_write_desc,
                         int logit_rows,
                         int structured_window_left,
                         RsExecutionMode rs_mode,
                         int rs_fold_lens_count,
                         int rs_buffer_ids_count,
                         cudaStream_t stream)
{
    if (mask_bytes < 0 || mask_indptr_count < 0 ||
        (mask_indptr_count != 0 && mask_indptr_count != R + 1) ||
        static_cast<std::size_t>(mask_bytes) > pi.custom_mask.size() ||
        static_cast<std::size_t>(mask_indptr_count) >
            pi.custom_mask_indptr.size()) {
        throw std::runtime_error(
            "TP root custom mask metadata exceeds persistent capacity");
    }
    auto* d_hdr = tp_hdr_dev_buf();
    TpFireHeader hdr{
        TP_FIRE_MAGIC, N, R, is_pure_decode ? 1 : 0,
        kv_indices_count, required_kv_pages, mask_bytes, mask_indptr_count,
        has_slot_ids ? 1 : 0,
        has_write_desc ? 1 : 0,
        logit_rows,
        structured_window_left,
        static_cast<std::int32_t>(rs_mode),
        rs_fold_lens_count,
        rs_buffer_ids_count,
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
    const bool state_only_fold =
        rs_mode == RsExecutionMode::BufferFold;
    NCCL_CHECK(ncclGroupStart());
    if (!state_only_fold) {
        NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                 static_cast<std::size_t>(N) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                 static_cast<std::size_t>(N) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(
            pi.row_valid.data(), pi.row_valid.data(),
            static_cast<std::size_t>(N), ncclChar, 0,
            comm.comm(), stream));
    }
    if (!state_only_fold && has_write_desc && N > 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.w_page.data(), pi.w_page.data(),
            static_cast<std::size_t>(N) * 4, ncclChar, 0,
            comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(
            pi.w_off.data(), pi.w_off.data(),
            static_cast<std::size_t>(N) * 4, ncclChar, 0,
            comm.comm(), stream));
    }
    NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    if (!state_only_fold) {
        NCCL_CHECK(ncclBroadcast(
            pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
            static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
            comm.comm(), stream));
    }
    if (!state_only_fold && R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                 pi.kv_last_page_lens.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
    }
    if (!state_only_fold && kv_indices_count > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                 pi.kv_page_indices.data(),
                                 static_cast<std::size_t>(kv_indices_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (!state_only_fold && mask_indptr_count > 0) {
        if (mask_bytes > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.custom_mask.data(), pi.custom_mask.data(),
                static_cast<std::size_t>(mask_bytes), ncclChar, 0,
                comm.comm(), stream));
        }
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
        NCCL_CHECK(ncclBroadcast(
            pi.rs_slot_flags.data(), pi.rs_slot_flags.data(),
            static_cast<std::size_t>(R), ncclChar, 0, comm.comm(), stream));
    }
    if (rs_fold_lens_count > 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.rs_fold_lens.data(), pi.rs_fold_lens.data(),
            static_cast<std::size_t>(rs_fold_lens_count) *
                sizeof(std::uint32_t),
            ncclChar, 0, comm.comm(), stream));
    }
    if ((rs_mode == RsExecutionMode::BufferWrite ||
         rs_mode == RsExecutionMode::BufferFold) &&
        R >= 0) {
        NCCL_CHECK(ncclBroadcast(
            pi.rs_buffer_slot_indptr.data(),
            pi.rs_buffer_slot_indptr.data(),
            static_cast<std::size_t>(R + 1) * sizeof(std::uint32_t),
            ncclChar, 0, comm.comm(), stream));
        if (rs_buffer_ids_count > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_slot_ids.data(),
                pi.rs_buffer_slot_ids.data(),
                static_cast<std::size_t>(rs_buffer_ids_count) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
    }
    if (!state_only_fold && logit_rows > 0) {
        NCCL_CHECK(ncclBroadcast(pi.sample_idx.data(), pi.sample_idx.data(),
                                 static_cast<std::size_t>(logit_rows) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

void tp_broadcast_mtp_step(
    NcclComm& comm,
    PersistentInputs& pi,
    int rows,
    int draft_step,
    int max_global_tokens,
    cudaStream_t stream) {
    auto* device_header = tp_hdr_dev_buf();
    TpFireHeader header{
        .magic = TP_MTP_MAGIC,
        .total_tokens = rows,
        .num_requests = draft_step,
        .is_pure_decode = max_global_tokens,
        .structured_window_left = -2,
    };
    CUDA_CHECK(cudaMemcpyAsync(
        device_header, &header, sizeof(header),
        cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(
        ncclBroadcast(
            device_header, device_header, sizeof(header), ncclChar, 0,
            comm.comm(), stream),
        comm.comm());
    NCCL_CHECK(ncclGroupStart());
    for (void* buffer : {
             static_cast<void*>(pi.tokens.data()),
             static_cast<void*>(pi.positions.data()),
             static_cast<void*>(pi.sample_idx.data()),
             static_cast<void*>(pi.mtp_request_ids.data())}) {
        NCCL_CHECK(ncclBroadcast(
            buffer, buffer, static_cast<std::size_t>(rows) * 4,
            ncclChar, 0, comm.comm(), stream));
    }
    NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
}

void tp_cpu_gate_notify(const std::string& key) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    gate->publish();
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
    if (engine.runtime_quant_context == nullptr) {
        throw std::runtime_error(
            "TP follower has no runtime-quant context");
    }
    ops::ScopedRuntimeQuantContext quant_scope(
        *engine.runtime_quant_context);

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
        if (hdr.magic == TP_MTP_MAGIC) {
            const int rows = hdr.total_tokens;
            NCCL_CHECK(ncclGroupStart());
            for (void* buffer : {
                     static_cast<void*>(pi.tokens.data()),
                     static_cast<void*>(pi.positions.data()),
                     static_cast<void*>(pi.sample_idx.data()),
                     static_cast<void*>(pi.mtp_request_ids.data())}) {
                NCCL_CHECK(ncclBroadcast(
                    buffer, buffer, static_cast<std::size_t>(rows) * 4,
                    ncclChar, 0, comm.comm(), stream));
            }
            NCCL_CHECK_ASYNC(ncclGroupEnd(), comm.comm());
            if (!engine.system_drafter.draft_step) {
                throw std::runtime_error(
                    "TP follower received MTP work without a native draft head");
            }
            engine.system_drafter.draft_step(
                engine.ws,
                engine.kv_cache,
                engine.cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.sample_idx.data(),
                pi.mtp_request_ids.data(),
                pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(),
                pi.kv_last_page_lens.data(),
                nullptr,
                rows,
                hdr.num_requests,
                hdr.is_pure_decode);
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
        if (hdr.required_kv_pages < 0 ||
            hdr.required_kv_pages > engine.kv_cache.num_pages()) {
            throw std::runtime_error(
                "TP follower received invalid required KV page high-water");
        }
        engine.kv_cache.ensure_pages(hdr.required_kv_pages);
        const bool is_pure_decode = (hdr.is_pure_decode != 0);
        const bool has_write_desc = hdr.has_write_desc != 0;
        const int logit_rows = hdr.logit_rows;
        const int structured_window_left =
            hdr.structured_window_left;
        if (!valid_rs_execution_mode(hdr.rs_mode) ||
            hdr.rs_fold_lens_count < 0 ||
            hdr.rs_buffer_ids_count < 0) {
            throw std::runtime_error(
                "TP follower received invalid RS metadata header");
        }
        const RsExecutionMode rs_mode =
            static_cast<RsExecutionMode>(hdr.rs_mode);
        const bool state_only_fold =
            rs_mode == RsExecutionMode::BufferFold;
        const int rs_fold_lens_count = hdr.rs_fold_lens_count;
        const int rs_buffer_ids_count = hdr.rs_buffer_ids_count;
        const std::size_t rs_rows =
            static_cast<std::size_t>(std::max(R, 0));
        const bool header_has_slots = hdr.has_slot_ids != 0;
        const bool header_buffered =
            rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold;
        if (!tp_rs_metadata_shape_valid(
                rs_mode,
                rs_rows,
                header_has_slots ? rs_rows : 0,
                header_has_slots ? rs_rows : 0,
                static_cast<std::size_t>(rs_fold_lens_count),
                static_cast<std::size_t>(rs_buffer_ids_count),
                header_buffered ? rs_rows + 1 : 0)) {
            throw std::runtime_error(
                "TP follower received inconsistent RS metadata header");
        }
        if (static_cast<std::size_t>(std::max(R, 0)) >
                pi.rs_fold_lens.size() ||
            static_cast<std::size_t>(rs_fold_lens_count) >
                pi.rs_fold_lens.size() ||
            static_cast<std::size_t>(rs_buffer_ids_count) >
                pi.rs_buffer_slot_ids.size()) {
            throw std::runtime_error(
                "TP follower RS metadata exceeds persistent capacity");
        }

        // 2. Receive payloads. Mirror order in `tp_broadcast_inputs`,
        //    grouped so NCCL submits the batch as a single op.
        if (hdr.mask_bytes < 0 || hdr.mask_indptr_count < 0 ||
            (hdr.mask_indptr_count != 0 &&
             hdr.mask_indptr_count != R + 1) ||
            static_cast<std::size_t>(hdr.mask_bytes) >
                pi.custom_mask.size() ||
            static_cast<std::size_t>(hdr.mask_indptr_count) >
                pi.custom_mask_indptr.size()) {
            throw std::runtime_error(
                "TP follower custom mask exceeds persistent capacity");
        }
        const bool have_custom_mask =
            !state_only_fold && hdr.mask_indptr_count > 0;
        NCCL_CHECK(ncclGroupStart());
        if (!state_only_fold) {
            NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                     static_cast<std::size_t>(N) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                     static_cast<std::size_t>(N) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.row_valid.data(), pi.row_valid.data(),
                static_cast<std::size_t>(N), ncclChar, 0,
                comm.comm(), stream));
        }
        if (!state_only_fold && has_write_desc && N > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.w_page.data(), pi.w_page.data(),
                static_cast<std::size_t>(N) * 4, ncclChar, 0,
                comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(
                pi.w_off.data(), pi.w_off.data(),
                static_cast<std::size_t>(N) * 4, ncclChar, 0,
                comm.comm(), stream));
        }
        NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        if (!state_only_fold) {
            NCCL_CHECK(ncclBroadcast(
                pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
                static_cast<std::size_t>(R + 1) * 4,
                ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && R > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                     pi.kv_last_page_lens.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && hdr.kv_indices_count > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                     pi.kv_page_indices.data(),
                                     static_cast<std::size_t>(hdr.kv_indices_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (!state_only_fold && have_custom_mask) {
            if (hdr.mask_bytes > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.custom_mask.data(), pi.custom_mask.data(),
                    static_cast<std::size_t>(hdr.mask_bytes),
                    ncclChar, 0, comm.comm(), stream));
            }
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
            NCCL_CHECK(ncclBroadcast(
                pi.rs_slot_flags.data(), pi.rs_slot_flags.data(),
                static_cast<std::size_t>(R), ncclChar, 0,
                comm.comm(), stream));
        }
        if (rs_fold_lens_count > 0) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_fold_lens.data(), pi.rs_fold_lens.data(),
                static_cast<std::size_t>(rs_fold_lens_count) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
        }
        if (rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold) {
            NCCL_CHECK(ncclBroadcast(
                pi.rs_buffer_slot_indptr.data(),
                pi.rs_buffer_slot_indptr.data(),
                static_cast<std::size_t>(R + 1) *
                    sizeof(std::uint32_t),
                ncclChar, 0, comm.comm(), stream));
            if (rs_buffer_ids_count > 0) {
                NCCL_CHECK(ncclBroadcast(
                    pi.rs_buffer_slot_ids.data(),
                    pi.rs_buffer_slot_ids.data(),
                    static_cast<std::size_t>(rs_buffer_ids_count) *
                        sizeof(std::uint32_t),
                    ncclChar, 0, comm.comm(), stream));
            }
        }
        if (!state_only_fold && logit_rows > 0) {
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
            state_only_fold
                ? 0
                : static_cast<std::size_t>(
                      std::max(0, hdr.kv_indices_count)));
        std::vector<std::uint32_t> h_kvlpl(
            state_only_fold ? 0 : static_cast<std::size_t>(R));
        CUDA_CHECK(cudaMemcpyAsync(h_qo.data(), pi.qo_indptr.data(),
                                   static_cast<std::size_t>(R + 1) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        if (!state_only_fold) {
            CUDA_CHECK(cudaMemcpyAsync(
                h_kvpp.data(), pi.kv_page_indptr.data(),
                static_cast<std::size_t>(R + 1) * 4,
                cudaMemcpyDeviceToHost, stream));
        }
        if (!state_only_fold && R > 0) {
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
            CUDA_CHECK(cudaMemcpyAsync(
                pi.rs_slot_flags_host.data(), pi.rs_slot_flags.data(),
                static_cast<std::size_t>(R), cudaMemcpyDeviceToHost, stream));
        }
        if (rs_fold_lens_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                pi.rs_fold_lens_host.data(), pi.rs_fold_lens.data(),
                static_cast<std::size_t>(rs_fold_lens_count) *
                    sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost, stream));
        }
        if (rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold) {
            CUDA_CHECK(cudaMemcpyAsync(
                pi.rs_buffer_slot_indptr_host.data(),
                pi.rs_buffer_slot_indptr.data(),
                static_cast<std::size_t>(R + 1) *
                    sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost, stream));
            if (rs_buffer_ids_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    pi.rs_buffer_slot_ids_host.data(),
                    pi.rs_buffer_slot_ids.data(),
                    static_cast<std::size_t>(rs_buffer_ids_count) *
                        sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int request = 0; request < R && have_slot_ids; ++request) {
            const std::uint8_t flags =
                pi.rs_slot_flags_host[static_cast<std::size_t>(request)];
            const std::uint8_t expected_fresh =
                (flags & PIE_RS_FLAG_RESET) != 0 ? 1u : 0u;
            if (h_is_fresh[static_cast<std::size_t>(request)] !=
                expected_fresh) {
                throw std::runtime_error(
                    "TP follower RS flags/reset metadata mismatch");
            }
        }

        std::vector<std::uint32_t> h_rs_slots(
            h_slot_ids.begin(), h_slot_ids.end());
        const std::span<const std::uint32_t> rs_slots(h_rs_slots);
        const std::span<const std::uint8_t> rs_flags(
            have_slot_ids ? pi.rs_slot_flags_host.data() : nullptr,
            have_slot_ids ? static_cast<std::size_t>(R) : 0);
        const std::span<const std::uint32_t> rs_fold_lens(
            rs_fold_lens_count > 0
                ? pi.rs_fold_lens_host.data()
                : nullptr,
            static_cast<std::size_t>(rs_fold_lens_count));
        const bool buffered =
            rs_mode == RsExecutionMode::BufferWrite ||
            rs_mode == RsExecutionMode::BufferFold;
        const std::span<const std::uint32_t> rs_buffer_ids(
            buffered ? pi.rs_buffer_slot_ids_host.data() : nullptr,
            buffered ? static_cast<std::size_t>(rs_buffer_ids_count) : 0);
        const std::span<const std::uint32_t> rs_buffer_indptr(
            buffered ? pi.rs_buffer_slot_indptr_host.data() : nullptr,
            buffered ? static_cast<std::size_t>(R + 1) : 0);
        pipeline::RsExecutionPlan follower_rs_plan;
        std::string rs_error;
        if (!pipeline::plan_rs_execution(
                rs_slots, rs_flags, rs_fold_lens,
                rs_buffer_ids, rs_buffer_indptr, h_qo,
                engine.rs_cache != nullptr,
                engine.rs_cache != nullptr &&
                    engine.rs_cache->rs_buffer_pool_enabled(),
                engine.rs_cache != nullptr
                    ? static_cast<std::uint32_t>(
                          engine.rs_cache->rs_buffer_page_tokens())
                    : 0,
                follower_rs_plan, &rs_error) ||
            follower_rs_plan.mode != rs_mode) {
            throw std::runtime_error(
                "TP follower RS metadata mismatch: " + rs_error);
        }

        // 4. Run the same forward function as rank 0. Channel publication is
        // rank-0-only after the collectives complete.
        if (rs_mode != RsExecutionMode::BufferFold) {
            engine.forward_fn.invoke_prepare(
                engine.attn_ws,
                ForwardFn::PrepareInputs{
                    .qo_indptr_h = h_qo.data(),
                    .kv_page_indices_h = h_kvpi.data(),
                    .kv_page_indices_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indices.data()),
                    .kv_page_indptr_h = h_kvpp.data(),
                    .kv_page_indptr_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_page_indptr.data()),
                    .kv_last_page_lens_h = h_kvlpl.data(),
                    .kv_last_page_lens_d =
                        reinterpret_cast<const std::uint32_t*>(
                            pi.kv_last_page_lens.data()),
                    .total_tokens = N,
                    .num_requests = R,
                    .is_pure_decode = is_pure_decode,
                    .have_custom_mask = have_custom_mask,
                    .runtime_window_left = structured_window_left,
                    .graphs_enabled = engine.graph_cache != nullptr &&
                        engine.forward_fn.graph_safe &&
                        prefill_graph_capture_enabled(),
                });
        }
        // Mirror rank 0's graph capture/replay decision so NCCL ops
        // inside the body record on both ranks simultaneously (otherwise
        // rank 0 would record while rank 1 executes, deadlocking the
        // first capture). The same `(R)` shape key keeps the per-rank
        // graph caches in lockstep; the captured graph on rank 1 has no
        // PTIR publication, just the forward kernels + NCCL.
        const bool try_graphs = forward_graph_replay_eligible(
            engine,
            is_pure_decode,
            // TP captures record the logits rows into the graph (unlike the
            // single-GPU path, whose logits epilogue runs outside), so a
            // prefill fire is only admitted when its sampled-row count is
            // pinned by the key's R.
            !is_pure_decode &&
                engine.forward_fn.invoke_prefill_graph_ready() &&
                logit_rows == R,
            have_custom_mask,
            rs_mode == RsExecutionMode::BufferWrite,
            rs_mode == RsExecutionMode::BufferFold,
            has_write_desc,
            structured_window_left,
            have_slot_ids,
            h_is_fresh.data(),
            R,
            /*num_images=*/0,
            /*num_clips=*/0,
            /*has_stage_hooks=*/false);
        const std::uint32_t graph_layout =
            engine.forward_fn.invoke_graph_layout();
        const std::uint32_t graph_variant =
            make_graph_variant(/*small_spec=*/false,
                               /*rs_verify=*/false,
                               have_custom_mask,
                               graph_layout);
        if (try_graphs) {
            const ForwardGraphKey key{R, N, graph_variant};
            cudaGraphExec_t exec = engine.graph_cache->get(key);
            if (exec == nullptr) {
                exec = capture_forward_graph_exec(
                    engine, h_qo.data(), h_kvpi.data(), h_kvpp.data(),
                    h_kvlpl.data(),
                    N, R, is_pure_decode,
                    have_custom_mask,
                    have_slot_ids ? h_slot_ids.data() : nullptr,
                    have_slot_ids ? h_is_fresh.data() : nullptr,
                    have_slot_ids ? pi.slot_ids.data() : nullptr,
                    logit_rows > 0 ? pi.sample_idx.data() : nullptr,
                    logit_rows,
                    pi.w_page.data(),
                    pi.w_off.data(),
                    has_write_desc,
                    structured_window_left);
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
            fwd_in.is_fresh_d          = have_slot_ids ? pi.is_fresh.data() : nullptr;
            fwd_in.rs_slot_flags_h     =
                have_slot_ids ? pi.rs_slot_flags_host.data() : nullptr;
            fwd_in.rs_buffer_slot_ids_h =
                buffered ? pi.rs_buffer_slot_ids_host.data() : nullptr;
            fwd_in.rs_buffer_slot_indptr_h =
                buffered ? pi.rs_buffer_slot_indptr_host.data() : nullptr;
            fwd_in.rs_fold_lens_h =
                rs_fold_lens_count > 0
                    ? pi.rs_fold_lens_host.data()
                    : nullptr;
            fwd_in.rs_fold_lens_d =
                rs_fold_lens_count > 0
                    ? reinterpret_cast<const std::int32_t*>(
                          pi.rs_fold_lens.data())
                    : nullptr;
            fwd_in.rs_buffer_write =
                rs_mode == RsExecutionMode::BufferWrite;
            fwd_in.rs_buffer_fold =
                rs_mode == RsExecutionMode::BufferFold;
            fwd_in.logit_row_indices_d = logit_rows > 0 ? pi.sample_idx.data() : nullptr;
            fwd_in.num_logit_rows      = logit_rows;
            fwd_in.runtime_window_left = structured_window_left;
            fwd_in.w_page_d = has_write_desc ? pi.w_page.data() : nullptr;
            fwd_in.w_off_d = has_write_desc ? pi.w_off.data() : nullptr;
            fwd_in.row_valid_d = pi.row_valid.data();
            fwd_in.has_write_desc = has_write_desc;
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
    TpFireHeader hdr{
        .magic = TP_STOP_MAGIC,
        .structured_window_left = -2,
    };
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK_ASYNC(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                                   comm.comm(), stream),
                     comm.comm());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace pie_cuda_driver
