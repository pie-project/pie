#include "distributed.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "cuda_check.hpp"
#include "custom_all_reduce.hpp"

namespace pie_cuda_driver {

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

NcclComm::NcclComm(int world_size, int rank, const ncclUniqueId& uid)
    : world_size_(world_size), rank_(rank) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    // Followers often wait for rank 0's next fire-batch broadcast. The
    // default non-blocking communicator can surface that idle wait as a
    // persistent 100% GPU-utilization spin. Use NCCL's blocking mode so idle
    // collectives sleep instead of burning a device while rank 0 is between
    // requests or still publishing readiness.
    const char* blocking_env = std::getenv("PIE_NCCL_BLOCKING");
    config.blocking = (blocking_env && std::string(blocking_env) == "0") ? 0 : 1;
    NCCL_CHECK(ncclCommInitRankConfig(&comm_, world_size, uid, rank, &config));
}

NcclComm::~NcclComm() {
    if (comm_ != nullptr) {
        // Best-effort: aborting comms during teardown is non-fatal.
        ncclCommDestroy(comm_);
    }
}

NcclComm& NcclComm::operator=(NcclComm&& o) noexcept {
    if (this != &o) {
        if (comm_) ncclCommDestroy(comm_);
        comm_ = o.comm_;             o.comm_ = nullptr;
        world_size_ = o.world_size_; o.world_size_ = 1;
        rank_ = o.rank_;             o.rank_ = 0;
    }
    return *this;
}

void NcclComm::all_reduce_bf16(void* sendrecv, std::size_t count,
                               ncclRedOp_t op, cudaStream_t stream) {
    NCCL_CHECK(ncclAllReduce(sendrecv, sendrecv, count, ncclBfloat16, op,
                             comm_, stream));
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
    NCCL_CHECK(ncclAllReduce(send, recv, count, ncclBfloat16, op,
                             comm_, stream));
}

void NcclComm::all_reduce_fp32(void* sendrecv, std::size_t count,
                               ncclRedOp_t op, cudaStream_t stream) {
    NCCL_CHECK(ncclAllReduce(sendrecv, sendrecv, count, ncclFloat32, op,
                             comm_, stream));
}

void NcclComm::all_gather_bf16(const void* send, void* recv,
                               std::size_t count_per_rank,
                               cudaStream_t stream) {
    NCCL_CHECK(ncclAllGather(send, recv, count_per_rank, ncclBfloat16,
                             comm_, stream));
}

void NcclComm::broadcast_bytes(void* sendrecv, std::size_t bytes, int root,
                               cudaStream_t stream) {
    NCCL_CHECK(ncclBroadcast(sendrecv, sendrecv, bytes, ncclChar, root,
                             comm_, stream));
}

void NcclComm::all_gather_bytes(const void* send, void* recv,
                                std::size_t count_per_rank,
                                cudaStream_t stream) {
    NCCL_CHECK(ncclAllGather(send, recv, count_per_rank, ncclChar,
                             comm_, stream));
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
    NCCL_CHECK(ncclAllReduce(d_one, d_one, 1, ncclInt32, ncclSum, comm_, stream));
}

}  // namespace pie_cuda_driver
