#include "distributed.hpp"

#include "batch/tp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cuda_check.hpp"
#include "kernels/custom_all_reduce.hpp"

namespace pie_cuda_driver {

namespace {

/// Copy a known pattern from `src` to `dst` over the peer path and check it
/// arrives. Returns false on any CUDA error or on a mismatch.
///
/// `cudaDeviceCanAccessPeer` answers "is there a path the driver is willing to
/// map", not "does traffic over that path arrive". On a PCIe box with ACS
/// enabled upstream of the GPUs -- the common case inside a VM or a container
/// on a virtualised host -- the answer is yes for every pair and peer copies
/// then land as zeros, silently. So ask the question that matters by moving
/// bytes.
bool peer_copy_round_trips(int src, int dst) {
    constexpr int kCount = 64;
    constexpr std::size_t kBytes = kCount * sizeof(std::uint32_t);

    std::vector<std::uint32_t> pattern(kCount);
    for (int i = 0; i < kCount; ++i) {
        pattern[static_cast<std::size_t>(i)] = 0xA5A50000u ^ static_cast<std::uint32_t>(i * 2654435761u);
    }

    void* src_buf = nullptr;
    void* dst_buf = nullptr;
    bool ok = false;

    if (cudaSetDevice(src) != cudaSuccess) return false;
    // Enable the direct mapping the real collectives use. Already-enabled is
    // success for our purposes; anything else means there is no peer path.
    const cudaError_t peer = cudaDeviceEnablePeerAccess(dst, 0);
    if (peer != cudaSuccess && peer != cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return false;
    }
    if (cudaMalloc(&src_buf, kBytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaMemcpy(src_buf, pattern.data(), kBytes, cudaMemcpyHostToDevice) ==
        cudaSuccess) {
        if (cudaSetDevice(dst) == cudaSuccess &&
            cudaMalloc(&dst_buf, kBytes) == cudaSuccess) {
            std::vector<std::uint32_t> got(kCount, 0u);
            // Poison the destination so "nothing arrived" cannot be mistaken
            // for "the pattern arrived".
            if (cudaMemset(dst_buf, 0, kBytes) == cudaSuccess &&
                cudaMemcpyPeer(dst_buf, dst, src_buf, src, kBytes) ==
                    cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess &&
                cudaMemcpy(got.data(), dst_buf, kBytes,
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                ok = got == pattern;
            }
        }
    }

    if (dst_buf != nullptr) {
        cudaSetDevice(dst);
        cudaFree(dst_buf);
    }
    if (src_buf != nullptr) {
        cudaSetDevice(src);
        cudaFree(src_buf);
    }
    cudaGetLastError();
    return ok;
}

}  // namespace

bool p2p_transport_usable() {
    static const bool usable = [] {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            return false;
        }
        int original = 0;
        if (cudaGetDevice(&original) != cudaSuccess) return false;

        bool ok = true;
        for (int src = 0; ok && src < devices; ++src) {
            for (int dst = 0; dst < devices; ++dst) {
                if (src == dst) continue;
                int can_access = 0;
                if (cudaDeviceCanAccessPeer(&can_access, src, dst) !=
                        cudaSuccess ||
                    can_access == 0 || !peer_copy_round_trips(src, dst)) {
                    ok = false;
                    break;
                }
            }
        }
        cudaSetDevice(original);
        cudaGetLastError();
        return ok;
    }();
    return usable;
}

std::string nccl_unique_id_to_hex(const ncclUniqueId& id) {
    static const char* k = "0123456789abcdef";
    const auto* p = reinterpret_cast<const std::uint8_t*>(&id);
    std::string out(NCCL_UNIQUE_ID_BYTES * 2, '0');
    for (std::size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        out[2 * i]     = k[(p[i] >> 4) & 0xF];
        out[2 * i + 1] = k[p[i] & 0xF];
    }
    return out;
}

ncclUniqueId nccl_unique_id_from_hex(const std::string& hex) {
    if (hex.size() != NCCL_UNIQUE_ID_BYTES * 2) {
        throw std::runtime_error(
            "nccl_unique_id_from_hex: bad length " +
            std::to_string(hex.size()) + " (expected " +
            std::to_string(NCCL_UNIQUE_ID_BYTES * 2) + ")");
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        throw std::runtime_error("non-hex character in unique-id");
    };
    ncclUniqueId id;
    auto* p = reinterpret_cast<std::uint8_t*>(&id);
    for (std::size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        p[i] = static_cast<std::uint8_t>(
            (nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return id;
}

void nccl_check_async(ncclResult_t result,
                      ncclComm_t comm,
                      const char* expr,
                      const char* file,
                      int line) {
    if (result != ncclSuccess && result != ncclInProgress) {
        throw std::runtime_error(
            std::string("NCCL error at ") + file + ":" +
            std::to_string(line) + " (" + expr + "): " +
            ncclGetErrorString(result));
    }
    if (result == ncclSuccess) {
        return;
    }
    // Non-blocking NCCL reports `ncclInProgress` for essentially every call,
    // so this poll loop sits on the TP critical path once per collective (or
    // once per group). It must NOT start by sleeping: a fixed 1 ms delay
    // before the first poll quantised every fire's fan-out to milliseconds
    // regardless of message size — measured as a flat 2.24 ms on rank 0 and
    // 1.13 ms on the follower, identical at 16 and 256 concurrent requests.
    // Poll first, then back off progressively so a genuinely slow collective
    // still yields the core.
    ncclResult_t state = ncclInProgress;
    for (std::uint64_t spins = 0;; ++spins) {
        const ncclResult_t poll = ncclCommGetAsyncError(comm, &state);
        if (poll != ncclSuccess) {
            throw std::runtime_error(
                std::string("NCCL async poll error at ") + file + ":" +
                std::to_string(line) + " (" + expr + "): " +
                ncclGetErrorString(poll));
        }
        if (state != ncclInProgress) break;
        if (spins < 512) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } else if (spins < 4096) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    if (state != ncclSuccess) {
        throw std::runtime_error(
            std::string("NCCL async error at ") + file + ":" +
            std::to_string(line) + " (" + expr + "): " +
            ncclGetErrorString(state));
    }
}

namespace {

// Every communicator alive in this process. TP ranks are threads, so one
// rank's failure handler needs a way to reach its peers' communicators; this
// is that list. Touched only at communicator construction/destruction and on
// the failure path, never per collective.
std::mutex g_live_comms_mu;
std::vector<NcclComm*> g_live_comms;

void register_live_comm(NcclComm* comm) {
    std::lock_guard<std::mutex> lk(g_live_comms_mu);
    g_live_comms.push_back(comm);
}

void unregister_live_comm(NcclComm* comm) {
    std::lock_guard<std::mutex> lk(g_live_comms_mu);
    g_live_comms.erase(
        std::remove(g_live_comms.begin(), g_live_comms.end(), comm),
        g_live_comms.end());
}

void replace_live_comm(NcclComm* from, NcclComm* to) {
    std::lock_guard<std::mutex> lk(g_live_comms_mu);
    for (auto*& entry : g_live_comms) {
        if (entry == from) entry = to;
    }
}

}  // namespace

void nccl_abort_all_comms() noexcept {
    std::lock_guard<std::mutex> lk(g_live_comms_mu);
    for (auto* comm : g_live_comms) {
        if (comm != nullptr) comm->abort();
    }
}

NcclComm::NcclComm(int world_size, int rank, const ncclUniqueId& uid)
    : world_size_(world_size), rank_(rank) {
    static std::once_flag env_once;
    std::call_once(env_once, [] {
        // The embedded TP launcher runs ranks as threads in one process. On
        // cross-socket PCIe topologies NCCL's P2P transport can hang before it
        // falls back, which is why this used to be disabled unconditionally.
        // But that also gives up NVLink on machines that have it. Probe the
        // topology instead: keep P2P when every visible device can reach every
        // other directly, disable it only where the original hazard exists.
        //
        // "Can reach" has to be measured, not asked. `cudaDeviceCanAccessPeer`
        // returns 1 for all 64 pairs on a PCIe box whose peer copies land as
        // zeros, so trusting it left P2P on exactly where it is broken -- and
        // a broken peer path does not announce itself: NCCL hangs on the first
        // collective, or worse, reduces zeros and the model emits fluent
        // nonsense. `p2p_transport_usable` moves bytes and checks them.
        if (std::getenv("NCCL_P2P_DISABLE") != nullptr) return;
        if (!p2p_transport_usable()) {
            setenv("NCCL_P2P_DISABLE", "1", /*overwrite=*/0);
            std::fprintf(stderr,
                         "[pie-driver-cuda] peer-to-peer transport failed a "
                         "round-trip check; disabling NCCL P2P and the custom "
                         "all-reduce (collectives stage through the host)\n");
        }
    });
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    // TP ranks are threads inside one process. Blocking NCCL calls can wedge
    // when one rank waits inside NCCL while another rank still needs to enter
    // the matching call. Non-blocking mode returns ncclInProgress instead; the
    // wrappers below poll the communicator without holding NCCL's launch path.
    config.blocking = 0;
    NCCL_CHECK_ASYNC(
        ncclCommInitRankConfig(&comm_, world_size, uid, rank, &config),
        comm_);
    register_live_comm(this);
}

NcclComm::~NcclComm() {
    // Unregister BEFORE aborting, and under the same lock `nccl_abort_all_comms`
    // takes: that is what makes the two paths exclusive, so neither can abort a
    // handle the other has already released.
    unregister_live_comm(this);
    if (comm_ != nullptr) {
        // NOT `ncclCommDestroy`: that finalizes collectively, so every rank
        // has to be inside it at the same time. Both TP ranks live in this
        // process and their Contexts are destroyed one after another, which
        // means the first rank's destroy would wait forever for a peer that
        // only starts destroying after it returns. `ncclCommAbort` releases
        // the communicator locally and is the documented teardown path when
        // the peers cannot rendezvous.
        ncclCommAbort(comm_);
        comm_ = nullptr;
    }
}

NcclComm& NcclComm::operator=(NcclComm&& o) noexcept {
    if (this != &o) {
        if (comm_) {
            unregister_live_comm(this);
            ncclCommAbort(comm_);
        }
        comm_ = o.comm_;             o.comm_ = nullptr;
        world_size_ = o.world_size_; o.world_size_ = 1;
        rank_ = o.rank_;             o.rank_ = 0;
        // The handle moved, so the registry entry has to follow it or the
        // failure path would abort through a dangling object.
        replace_live_comm(&o, this);
    }
    return *this;
}

void NcclComm::all_reduce_bf16(void* sendrecv, std::size_t count,
                               ncclRedOp_t op, cudaStream_t stream) {
    if (custom_ar_ != nullptr && op == ncclSum) {
        const std::size_t bytes = count * sizeof(std::uint16_t);
        if (custom_ar_->can_handle(sendrecv, bytes, stream)) {
            pie_cuda_driver::tp_watchdog_count_collective(rank_);
            custom_ar_->all_reduce_bf16(sendrecv, sendrecv, count, stream);
            return;
        }
    }
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclAllReduce(sendrecv, sendrecv, count, ncclBfloat16,
                                   op, comm_, stream),
                     comm_);
}

void NcclComm::all_reduce_bf16_out(const void* send, void* recv,
                                   std::size_t count, ncclRedOp_t op,
                                   cudaStream_t stream) {
    if (custom_ar_ != nullptr && op == ncclSum) {
        const std::size_t bytes = count * sizeof(std::uint16_t);
        if (custom_ar_->can_handle(send, bytes, stream)) {
            custom_ar_->all_reduce_bf16(send, recv, count, stream);
            return;
        }
    }
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclAllReduce(send, recv, count, ncclBfloat16, op,
                                   comm_, stream),
                     comm_);
}

void NcclComm::all_reduce_fp32(void* sendrecv, std::size_t count,
                               ncclRedOp_t op, cudaStream_t stream) {
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclAllReduce(sendrecv, sendrecv, count, ncclFloat32,
                                   op, comm_, stream),
                     comm_);
}

void NcclComm::all_gather_bf16(const void* send, void* recv,
                               std::size_t count_per_rank,
                               cudaStream_t stream) {
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclAllGather(send, recv, count_per_rank, ncclBfloat16,
                                   comm_, stream),
                     comm_);
}

void NcclComm::broadcast_bytes(void* sendrecv, std::size_t bytes, int root,
                               cudaStream_t stream) {
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclBroadcast(sendrecv, sendrecv, bytes, ncclChar, root,
                                   comm_, stream),
                     comm_);
}

void NcclComm::all_gather_bytes(const void* send, void* recv,
                                std::size_t count_per_rank,
                                cudaStream_t stream) {
    pie_cuda_driver::tp_watchdog_count_collective(rank_);
    NCCL_CHECK_ASYNC(ncclAllGather(send, recv, count_per_rank, ncclChar,
                                   comm_, stream),
                     comm_);
}

void NcclComm::all_gather_host_bytes(const void* send, void* recv,
                                     std::size_t bytes_per_rank) {
    if (bytes_per_rank == 0) return;
    void* d_send = nullptr;
    void* d_recv = nullptr;
    const std::size_t total = bytes_per_rank * world_size_;
    CUDA_CHECK(cudaMalloc(&d_send, bytes_per_rank));
    try {
        CUDA_CHECK(cudaMalloc(&d_recv, total));
        CUDA_CHECK(cudaMemcpy(d_send, send, bytes_per_rank,
                              cudaMemcpyHostToDevice));
        all_gather_bytes(d_send, d_recv, bytes_per_rank, /*stream=*/nullptr);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        CUDA_CHECK(cudaMemcpy(recv, d_recv, total, cudaMemcpyDeviceToHost));
    } catch (...) {
        if (d_recv != nullptr) cudaFree(d_recv);
        cudaFree(d_send);
        throw;
    }
    CUDA_CHECK(cudaFree(d_recv));
    CUDA_CHECK(cudaFree(d_send));
}

void NcclComm::barrier(cudaStream_t stream) {
    // Allocate-on-the-fly is fine here — barrier is cold-path only
    // (startup smoke test, shutdown). Hot-path code uses `broadcast_bytes`.
    static thread_local int* d_one = nullptr;
    if (d_one == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_one, sizeof(int)));
        const int one = 1;
        CUDA_CHECK(cudaMemcpy(d_one, &one, sizeof(int), cudaMemcpyHostToDevice));
    }
    NCCL_CHECK_ASYNC(
        ncclAllReduce(d_one, d_one, 1, ncclInt32, ncclSum, comm_, stream),
        comm_);
}

void NcclComm::abort() noexcept {
    if (comm_ != nullptr) {
        // Aborting frees the communicator, so drop the handle: a later
        // destructor must not touch it a second time.
        ncclCommAbort(comm_);
        comm_ = nullptr;
    }
}

}  // namespace pie_cuda_driver
