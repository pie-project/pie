#pragma once

// Custom NVLink P2P all-reduce, wrapping flashinfer's vllm-style kernel
// (`flashinfer/comm/vllm_custom_all_reduce.cuh`). Roughly 2-3× lower
// latency than NCCL for sub-MB BF16 reductions on NVSwitch/NVLink, which
// is the regime our per-layer attn-O / MLP-down all-reduces hit at TP=2
// for small-to-medium models.
//
// Lifecycle:
//   1. Construct once at startup (after NCCL is up).
//   2. `register_buffer` each persistent device buffer that will be the
//      input/output of an all-reduce. The base address is IPC-shared
//      across ranks so every rank holds peer pointers; subsequent
//      all-reduces on any prefix of the buffer reuse the registration.
//   3. `all_reduce_bf16` dispatches the kernel; falls back to NCCL when
//      the message exceeds the kernel's NVLink-friendly threshold.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace vllm {
class CustomAllreduce;
struct Signal;
}

namespace pie_cuda_driver {

class NcclComm;

class CustomAllReduce {
public:
    CustomAllReduce();
    CustomAllReduce(NcclComm& comm, int max_registrations = 8);
    ~CustomAllReduce();

    CustomAllReduce(const CustomAllReduce&) = delete;
    CustomAllReduce& operator=(const CustomAllReduce&) = delete;
    CustomAllReduce(CustomAllReduce&&) noexcept;
    CustomAllReduce& operator=(CustomAllReduce&&) noexcept;

    explicit operator bool() const noexcept { return impl_ != nullptr; }

    // IPC-exchange `buf`'s base address with peers and register it with
    // the underlying kernel. Must be called by every rank with its own
    // buffer's base pointer (the buffers don't have to be the same size
    // across ranks, but typically they are). Subsequent `all_reduce_bf16`
    // calls passing a pointer >= `buf` and < `buf + buf_bytes` use this
    // registration; we resolve buf-base via cuPointerGetAttribute.
    void register_buffer(NcclComm& comm, void* buf, std::size_t buf_bytes);

    // Returns true when the kernel will handle `bytes` directly. Above
    // the threshold (~512 KB for 2-rank, less for higher TP) the kernel
    // falls off NCCL on bandwidth, so we short-circuit and return false
    // — caller should fall back to ncclAllReduce.
    bool can_handle(std::size_t bytes) const noexcept;

    // bf16 in-place all-reduce. `count` is element count (NOT bytes).
    // The buffer must have been registered via `register_buffer`.
    void all_reduce_bf16(void* sendrecv, std::size_t count, cudaStream_t stream);

private:
    int rank_ = 0;
    int world_size_ = 1;
    bool full_nvlink_ = true;
    vllm::Signal* signal_self_ = nullptr;
    std::vector<vllm::Signal*> signal_peers_;  // size = world_size
    void* rank_data_ = nullptr;
    std::size_t rank_data_bytes_ = 0;
    std::unique_ptr<vllm::CustomAllreduce> impl_;
    // Track which base pointers have already been IPC-registered so
    // subsequent all-reduces don't reopen handles.
    std::unordered_map<void*, void*> registered_bases_;  // self_base -> self_base
};

}  // namespace pie_cuda_driver
