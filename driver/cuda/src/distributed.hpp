#pragma once

// Tensor-parallel plumbing on top of NCCL.
//
// Each TP rank runs as its own `pie_driver_cuda` process and joins a single
// `ncclComm_t` keyed by a shared `ncclUniqueId`. The wrapper
// (`pie_driver_cuda_native/worker.py`) generates the unique-id, passes it
// to all ranks via the startup TOML, and only rank 0 of each group exposes
// a shmem server to the runtime. Followers consume their inputs by NCCL
// broadcast from rank 0 — see request_handler for the broadcast plumbing.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <nccl.h>

namespace pie_cuda_driver {

class CustomAllReduce;  // see custom_all_reduce.hpp

#define NCCL_CHECK(expr)                                                       \
    do {                                                                       \
        ncclResult_t _r = (expr);                                              \
        if (_r != ncclSuccess) {                                               \
            throw std::runtime_error(std::string("NCCL error: ") +             \
                                     ncclGetErrorString(_r));                  \
        }                                                                      \
    } while (0)

// Hex-encode / decode an `ncclUniqueId`. The wrapper passes the id to the
// driver as 256 hex chars (NCCL_UNIQUE_ID_BYTES = 128 bytes).
std::string nccl_unique_id_to_hex(const ncclUniqueId& id);
ncclUniqueId nccl_unique_id_from_hex(const std::string& hex);

// RAII wrapper around `ncclCommInitRank` / `ncclCommDestroy` plus a few
// convenience collective shims keyed to the dtypes we actually use.
class NcclComm {
public:
    NcclComm() = default;
    NcclComm(int world_size, int rank, const ncclUniqueId& uid);
    ~NcclComm();

    NcclComm(const NcclComm&) = delete;
    NcclComm& operator=(const NcclComm&) = delete;
    NcclComm(NcclComm&& o) noexcept { *this = std::move(o); }
    NcclComm& operator=(NcclComm&& o) noexcept;

    explicit operator bool() const noexcept { return comm_ != nullptr; }

    int world_size() const noexcept { return world_size_; }
    int rank() const noexcept { return rank_; }
    ncclComm_t comm() const noexcept { return comm_; }

    // bf16 in-place all-reduce. `count` is element count.
    void all_reduce_bf16(void* sendrecv, std::size_t count, ncclRedOp_t op,
                         cudaStream_t stream);

    // fp32 in-place all-reduce. Used by the runtime-quant absmax MAX
    // reduction (row-parallel weights need cross-rank absmax to compute
    // the global per-row scale).
    void all_reduce_fp32(void* sendrecv, std::size_t count, ncclRedOp_t op,
                         cudaStream_t stream);

    // bf16 all-gather. `count_per_rank` is per-rank element count; `recv`
    // must be sized for `world_size * count_per_rank`.
    void all_gather_bf16(const void* send, void* recv,
                         std::size_t count_per_rank, cudaStream_t stream);

    // Byte-wise broadcast (rank `root` is the source). Used by the
    // dispatch path to fan out small fire_batch headers + payload bytes.
    void broadcast_bytes(void* sendrecv, std::size_t bytes, int root,
                         cudaStream_t stream);

    // Byte-wise all-gather: each rank contributes `count_per_rank` bytes
    // from `send`; recv gets `world_size * count_per_rank` bytes laid
    // out as [rank0_bytes | rank1_bytes | ...]. Used for the one-shot
    // IPC handle exchange in the custom all-reduce setup.
    void all_gather_bytes(const void* send, void* recv,
                          std::size_t count_per_rank, cudaStream_t stream);

    // Stream-synchronous device barrier. Implemented as a 1-byte all-reduce
    // so we don't depend on `ncclAllReduce(0, …)` semantics across versions.
    void barrier(cudaStream_t stream);

    // Optional fast-path: when set, `all_reduce_bf16(... ncclSum ...)`
    // routes small messages through the NVLink P2P custom kernel
    // (flashinfer's vllm_custom_all_reduce). Caller still owns the
    // CustomAllReduce instance — NcclComm just borrows the pointer.
    void set_custom_all_reduce(CustomAllReduce* car) noexcept { custom_ar_ = car; }
    CustomAllReduce* custom_all_reduce() const noexcept { return custom_ar_; }

private:
    ncclComm_t comm_ = nullptr;
    int world_size_ = 1;
    int rank_ = 0;
    CustomAllReduce* custom_ar_ = nullptr;
};

}  // namespace pie_cuda_driver
