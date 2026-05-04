#include "custom_all_reduce.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"

#include <flashinfer/comm/vllm_custom_all_reduce.cuh>

namespace pie_cuda_driver {

namespace {

// Helper to enable peer access from `device` to every other GPU in the
// group. Idempotent — subsequent calls just return cudaErrorPeerAccessAlreadyEnabled
// which we swallow.
void enable_peer_access(int self_device, int world_size) {
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == self_device) continue;
        const auto err = cudaDeviceEnablePeerAccess(peer, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            // Fail open — the constructor will throw at the IPC step if
            // peer access truly isn't available.
            std::cerr << "[custom_all_reduce] cudaDeviceEnablePeerAccess "
                      << self_device << "→" << peer
                      << " failed: " << cudaGetErrorString(err) << "\n";
        }
        // Reset the sticky error.
        (void)cudaGetLastError();
    }
}

// Look up the base allocation address for `ptr`. The vllm kernel needs
// the *base* pointer for the IPC handle exchange — sub-allocation
// pointers won't round-trip across processes correctly.
void* get_base_ptr(void* ptr) {
    void* base = nullptr;
    const auto rc = cuPointerGetAttribute(&base,
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        reinterpret_cast<CUdeviceptr>(ptr));
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "custom_all_reduce: cuPointerGetAttribute(RANGE_START_ADDR) failed");
    }
    return base;
}

}  // namespace

// Default ctor defined here (not = default in header) so the compiler
// doesn't have to see the full vllm::CustomAllreduce type to handle the
// implicit member initialisation of std::unique_ptr<vllm::CustomAllreduce>
// at every use site of the wrapper.
CustomAllReduce::CustomAllReduce() = default;

CustomAllReduce::CustomAllReduce(NcclComm& comm, int max_registrations) {
    rank_       = comm.rank();
    world_size_ = comm.world_size();
    if (world_size_ < 2 || world_size_ > 8 || (world_size_ % 2) != 0) {
        throw std::runtime_error(
            "custom_all_reduce: vllm kernel supports world_size ∈ {2,4,6,8}; got " +
            std::to_string(world_size_));
    }

    // Best-effort peer access; cudaIpcOpenMemHandle below will surface a
    // hard error if NVLink P2P isn't actually available between this
    // GPU and any peer.
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    enable_peer_access(dev, world_size_);

    // Allocate the local Signal struct. vllm::Signal is
    // ~1 KB; we zero-init it (Lamport-style sync expects zero counters).
    const std::size_t signal_bytes = sizeof(vllm::Signal);
    CUDA_CHECK(cudaMalloc(&signal_self_, signal_bytes));
    CUDA_CHECK(cudaMemset(signal_self_, 0, signal_bytes));

    // Exchange IPC handles for the Signal across ranks. ncclAllGather
    // does the byte transfer; cudaIpcOpenMemHandle on each peer's
    // handle gives us a pointer mapped into our address space.
    cudaIpcMemHandle_t self_signal_handle{};
    CUDA_CHECK(cudaIpcGetMemHandle(&self_signal_handle, signal_self_));

    std::vector<cudaIpcMemHandle_t> all_signal_handles(world_size_);
    {
        // Stage on device: ncclAllGather is a device-side op.
        cudaIpcMemHandle_t* d_send = nullptr;
        cudaIpcMemHandle_t* d_recv = nullptr;
        const std::size_t hsz = sizeof(cudaIpcMemHandle_t);
        CUDA_CHECK(cudaMalloc(&d_send, hsz));
        CUDA_CHECK(cudaMalloc(&d_recv, hsz * world_size_));
        CUDA_CHECK(cudaMemcpy(d_send, &self_signal_handle, hsz,
                              cudaMemcpyHostToDevice));
        comm.all_gather_bytes(d_send, d_recv, hsz, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(all_signal_handles.data(), d_recv,
                              hsz * world_size_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_send));
        CUDA_CHECK(cudaFree(d_recv));
    }

    signal_peers_.resize(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        if (r == rank_) {
            signal_peers_[r] = static_cast<vllm::Signal*>(signal_self_);
            continue;
        }
        void* peer_signal = nullptr;
        const auto rc = cudaIpcOpenMemHandle(&peer_signal, all_signal_handles[r],
                                             cudaIpcMemLazyEnablePeerAccess);
        if (rc != cudaSuccess) {
            throw std::runtime_error(
                std::string("custom_all_reduce: cudaIpcOpenMemHandle(rank=") +
                std::to_string(r) + ") failed: " + cudaGetErrorString(rc));
        }
        signal_peers_[r] = static_cast<vllm::Signal*>(peer_signal);
    }

    // RankData scratch — one 64-byte slot per registered buffer. Sized
    // for `max_registrations` so subsequent `register_buffer` calls
    // don't have to grow.
    rank_data_bytes_ = static_cast<std::size_t>(max_registrations) *
                       sizeof(vllm::RankData);
    CUDA_CHECK(cudaMalloc(&rank_data_, rank_data_bytes_));

    impl_ = std::make_unique<vllm::CustomAllreduce>(
        signal_peers_.data(), rank_data_, rank_data_bytes_,
        rank_, world_size_, full_nvlink_);

    std::cerr << "[custom_all_reduce] initialised (world=" << world_size_
              << ", rank=" << rank_ << ", NVLink=" << (full_nvlink_ ? "yes" : "no")
              << ")\n";
}

CustomAllReduce::~CustomAllReduce() {
    // Close peer signal handles before freeing local state. The vllm
    // CustomAllreduce destructor takes care of buffers it opened via
    // `register_buffer`'s IPC path. Iterate by signal_peers_ size, not
    // world_size_, so the moved-from-empty-but-world_size_=2 case stays
    // correct (move ctor empties the vector but doesn't reset world_size_).
    impl_.reset();
    for (std::size_t r = 0; r < signal_peers_.size(); ++r) {
        if (static_cast<int>(r) == rank_) continue;
        if (signal_peers_[r] != nullptr) {
            cudaIpcCloseMemHandle(signal_peers_[r]);
        }
    }
    if (signal_self_) cudaFree(signal_self_);
    if (rank_data_)   cudaFree(rank_data_);
}

CustomAllReduce::CustomAllReduce(CustomAllReduce&& o) noexcept
    : rank_(o.rank_), world_size_(o.world_size_), full_nvlink_(o.full_nvlink_),
      signal_self_(o.signal_self_), signal_peers_(std::move(o.signal_peers_)),
      rank_data_(o.rank_data_), rank_data_bytes_(o.rank_data_bytes_),
      impl_(std::move(o.impl_)),
      registered_bases_(std::move(o.registered_bases_))
{
    o.signal_self_ = nullptr;
    o.rank_data_ = nullptr;
}

CustomAllReduce& CustomAllReduce::operator=(CustomAllReduce&& o) noexcept {
    if (this != &o) {
        this->~CustomAllReduce();
        new (this) CustomAllReduce(std::move(o));
    }
    return *this;
}

bool CustomAllReduce::can_handle(std::size_t bytes) const noexcept {
    if (!impl_) return false;
    // The vllm one-shot kernel handles arbitrary sizes correctly; the
    // question is just where NCCL takes over on raw bandwidth. On
    // NVLink/NVSwitch the crossover is around a few MB for 2 ranks,
    // less for higher TP. Be generous on small TP (2 ranks); tighter
    // thresholds for larger TP where the kernel becomes the bottleneck.
    if (world_size_ <= 2)         return bytes < 8 * 1024 * 1024;
    if (world_size_ <= 4)         return bytes < 1 * 1024 * 1024;
    return bytes < 256 * 1024;
}

void CustomAllReduce::register_buffer(NcclComm& comm, void* buf,
                                      std::size_t /*buf_bytes*/)
{
    if (!impl_) return;
    void* self_base = get_base_ptr(buf);
    if (registered_bases_.find(self_base) != registered_bases_.end()) return;

    cudaIpcMemHandle_t self_handle{};
    CUDA_CHECK(cudaIpcGetMemHandle(&self_handle, self_base));

    // Gather every rank's IPC handle for its own buffer.
    std::vector<cudaIpcMemHandle_t> all_handles(world_size_);
    {
        const std::size_t hsz = sizeof(cudaIpcMemHandle_t);
        cudaIpcMemHandle_t* d_send = nullptr;
        cudaIpcMemHandle_t* d_recv = nullptr;
        CUDA_CHECK(cudaMalloc(&d_send, hsz));
        CUDA_CHECK(cudaMalloc(&d_recv, hsz * world_size_));
        CUDA_CHECK(cudaMemcpy(d_send, &self_handle, hsz, cudaMemcpyHostToDevice));
        comm.all_gather_bytes(d_send, d_recv, hsz, /*stream=*/nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(all_handles.data(), d_recv,
                              hsz * world_size_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_send));
        CUDA_CHECK(cudaFree(d_recv));
    }

    // Open peer handles, build the per-rank pointer array, and register.
    std::vector<void*> peer_bases(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        if (r == rank_) {
            peer_bases[r] = self_base;
        } else {
            void* peer = nullptr;
            const auto rc = cudaIpcOpenMemHandle(&peer, all_handles[r],
                                                 cudaIpcMemLazyEnablePeerAccess);
            if (rc != cudaSuccess) {
                throw std::runtime_error(
                    std::string("custom_all_reduce: register_buffer "
                                "cudaIpcOpenMemHandle(rank=") +
                    std::to_string(r) + ") failed: " +
                    cudaGetErrorString(rc));
            }
            peer_bases[r] = peer;
        }
    }
    impl_->register_buffer(peer_bases.data());
    registered_bases_[self_base] = self_base;
}

void CustomAllReduce::all_reduce_bf16(void* sendrecv, std::size_t count,
                                      cudaStream_t stream)
{
    if (!impl_) {
        throw std::runtime_error("custom_all_reduce: not initialised");
    }
    // The vllm kernel registers buffers by base address; passing a
    // sub-pointer works as long as the *base* was registered. We pass
    // the input pointer directly (kernel does its own offset math on
    // the registered RankData).
    impl_->allreduce<__nv_bfloat16>(
        stream,
        static_cast<__nv_bfloat16*>(sendrecv),
        static_cast<__nv_bfloat16*>(sendrecv),
        static_cast<int>(count),
        /*block_limit=*/16, /*threads=*/512);
}

}  // namespace pie_cuda_driver
