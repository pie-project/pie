#pragma once

// Custom P2P all-reduce, wrapping flashinfer's vllm-style kernel
// (`flashinfer/comm/vllm_custom_all_reduce.cuh`). Roughly 2-3× lower
// latency than NCCL for sub-MB BF16 reductions on TP=2, which is the
// regime our per-layer attn-O / MLP-down all-reduces hit for
// small-to-medium models. For TP>2 the vLLM kernel needs a fully-connected
// fast interconnect topology.
//
// Lifecycle:
//   1. Construct once at startup (after NCCL is up).
//   2. `register_buffer` each persistent device buffer that will be the
//      input/output of an all-reduce. The base address is IPC-shared
//      across ranks so every rank holds peer pointers; subsequent
//      all-reduces on any prefix of the buffer reuse the registration.
//   3. During CUDA graph capture, `register_graph_buffers` fills the
//      deferred rank-data slots recorded by flashinfer's custom AR body.
//   4. `all_reduce_bf16` dispatches the kernel; falls back to NCCL when
//      the message exceeds the custom kernel's useful threshold.

#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace vllm {
class CustomAllreduce;
struct Signal;
}

namespace pie_cuda_driver::kernels::comm {


// What this needs from the collective, and nothing more.
//
// The wrapper used to take an `NcclComm&`. It reads exactly two things off it
// -- the world size, and one bootstrap-time all-gather of IPC handles -- and
// taking the class instead of those two made a compute kernel depend on the
// driver's comm plane. It is a compute kernel: `all_reduce_residual_rmsnorm_bf16`
// fuses a reduction with a residual add and an RMSNorm, and the unfused halves
// of that live in `kernels-cuda`.
//
// So the seam is a callback. `gather` takes HOST buffers; whatever H2D dance a
// given collective needs is the caller's business, which is where NCCL knowledge
// belongs. Who decides custom-vs-NCCL by message size stays the caller's too --
// `can_handle()` only reports.
struct HostAllgather {
    int rank = 0;
    int world_size = 1;
    // send[bytes] from this rank -> recv[bytes * world_size], rank-major.
    std::function<void(const void* send, void* recv, std::size_t bytes)> gather;
};

class CustomAllReduce {
public:
    CustomAllReduce();
    // `group_devices` holds the CUDA device ordinal of every rank in the
    // group, indexed by rank. A TP group is not necessarily devices
    // 0..world_size-1, so the ordinals have to be supplied rather than
    // inferred from rank indices.
    CustomAllReduce(HostAllgather ag,
                    bool same_process,
                    std::vector<int> group_devices,
                    std::size_t max_bytes = 8 * 1024 * 1024,
                    std::size_t rank_data_bytes = 8 * 1024 * 1024,
                    int fusion_max_tokens = 0,
                    int fusion_hidden = 0);
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
    void register_buffer(void* buf, std::size_t buf_bytes);
    void register_graph_buffers();

    // Returns true when the kernel will handle `bytes` directly. Above
    // the threshold the kernel falls off NCCL on bandwidth, so we
    // short-circuit and return false — caller should fall back to
    // ncclAllReduce.
    bool can_handle(const void* input, std::size_t bytes,
                    cudaStream_t stream) const noexcept;
    bool can_fuse_residual_rmsnorm(int tokens, int hidden,
                                   cudaStream_t stream) const noexcept;

    // bf16 in-place all-reduce. `count` is element count (NOT bytes).
    // The buffer must have been registered via `register_buffer`.
    void all_reduce_bf16(const void* input, void* output, std::size_t count,
                         cudaStream_t stream);
    void all_reduce_residual_rmsnorm_bf16(
        const void* input,
        void* residual_inout,
        const void* rms_gamma,
        void* norm_out,
        int tokens,
        int hidden,
        float eps,
        cudaStream_t stream);
    void all_reduce_residual_rmsnorm_bf16_exact(
        const void* input,
        void* residual_inout,
        const void* rms_gamma,
        void* norm_out,
        int tokens,
        int hidden,
        float eps,
        cudaStream_t stream);

private:
    int rank_ = 0;
    int world_size_ = 1;
    bool fully_connected_ = false;
    bool same_process_ = false;
    std::size_t max_bytes_ = 0;
    vllm::Signal* signal_self_ = nullptr;
    std::vector<vllm::Signal*> signal_peers_;  // size = world_size
    void* rank_data_ = nullptr;
    std::size_t rank_data_bytes_ = 0;
    std::unique_ptr<vllm::CustomAllreduce> impl_;
    // Track which base pointers have already been IPC-registered so
    // subsequent all-reduces don't reopen handles.
    std::unordered_map<void*, void*> registered_bases_;  // self_base -> self_base
    HostAllgather ag_;
    std::vector<void*> fusion_buffers_;
    void* fusion_workspace_dev_ = nullptr;
    void* fusion_flag_dev_ = nullptr;
    int fusion_max_tokens_ = 0;
    int fusion_hidden_ = 0;
    std::size_t fusion_lamport_comm_bytes_ = 0;
};

// ── the FREE form, for the launch ABI ──────────────────────────────
//
// `all_reduce_residual_rmsnorm_bf16` is a METHOD, and a method has no
// address the generated launch ABI can forward to: `KernelSig` describes
// a free function, and the shim it emits takes a function POINTER. So
// the row that names this kernel could not describe it, and the one
// family that states the symbol had to bind it by hand.
//
// This is the same shape as `gemm`'s scaled entry points before 1b: the
// call was reachable only through an object the declaration never
// mentioned. The fix is the same too -- take the object as the first
// argument and let the statement name the symbol.
//
// The instance stays the CALLER's. `car` is borrowed, exactly as
// `NcclComm` borrows it, and a null one is a deployment that configured
// no custom all-reduce, which is a refusal rather than a fallback: the
// fused landing IS this kernel, and there is no other way to spell it.
// The plain P2P all-reduce, same free form and same reason. It is the
// arm a declaration takes when the message fits the NVLink kernel;
// `dist::all_reduce_bf16` -- NCCL -- is the other, and WHICH is a guard
// in the text rather than an `if` inside a driver method.
inline void all_reduce_bf16(
    CustomAllReduce* car,
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream) {
    if (car == nullptr) {
        throw std::runtime_error(
            "comm: the P2P all-reduce is stated but this deployment "
            "configured no custom all-reduce");
    }
    car->all_reduce_bf16(input, output, count, stream);
}

//
// `inline` in the header rather than a TU of its own: the class has two
// implementations (the real `.cu` and a stub for builds without the
// P2P kernel) and this forwards to whichever is linked, so putting it
// in either would have made it available only half the time.
inline void all_reduce_residual_rmsnorm_bf16(
    CustomAllReduce* car,
    const void* input,
    void* residual_inout,
    const void* rms_gamma,
    void* norm_out,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream) {
    if (car == nullptr) {
        throw std::runtime_error(
            "comm: the fused all-reduce landing is stated but this "
            "deployment configured no custom all-reduce");
    }
    car->all_reduce_residual_rmsnorm_bf16(input, residual_inout, rms_gamma,
                                          norm_out, tokens, hidden, eps,
                                          stream);
}

}  // namespace pie_cuda_driver::kernels::comm
