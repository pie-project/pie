#include "kernels/slot_ops.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

__global__ void zero_slots_if_fresh_kernel(
    std::uint8_t* base,
    std::size_t slot_bytes,
    std::size_t layer_stride_bytes,
    const std::int32_t* slot_ids,
    const std::uint8_t* is_fresh,
    std::size_t request_count)
{
    const std::size_t request = blockIdx.x;
    const std::size_t layer = blockIdx.y;
    if (request >= request_count || is_fresh[request] == 0) return;
    const std::int32_t slot = slot_ids[request];
    if (slot < 0) return;
    std::uint8_t* out =
        base + layer * layer_stride_bytes +
        static_cast<std::size_t>(slot) * slot_bytes;
    for (std::size_t i = threadIdx.x; i < slot_bytes; i += blockDim.x) {
        out[i] = 0;
    }
}

__global__ void copy_if_valid_slot_kernel(
    const std::uint8_t* src,
    std::uint8_t* dst,
    std::size_t bytes,
    const std::int32_t* slot_ids,
    std::size_t request)
{
    if (slot_ids[request] < 0) return;
    for (std::size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

void launch_zero_slots_if_fresh(
    std::uint8_t* base,
    std::size_t slot_bytes,
    std::size_t layer_stride_bytes,
    std::size_t layer_count,
    const std::int32_t* slot_ids,
    const std::uint8_t* is_fresh,
    std::size_t request_count,
    cudaStream_t stream)
{
    if (base == nullptr || slot_bytes == 0 || layer_count == 0 ||
        request_count == 0) {
        return;
    }
    constexpr int kThreads = 256;
    zero_slots_if_fresh_kernel<<<
        dim3(
            static_cast<unsigned int>(request_count),
            static_cast<unsigned int>(layer_count)),
        kThreads, 0, stream>>>(
        base,
        slot_bytes,
        layer_stride_bytes,
        slot_ids,
        is_fresh,
        request_count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_copy_if_valid_slot(
    const std::uint8_t* src,
    std::uint8_t* dst,
    std::size_t bytes,
    const std::int32_t* slot_ids,
    std::size_t request,
    cudaStream_t stream)
{
    if (bytes == 0) return;
    constexpr int kThreads = 256;
    copy_if_valid_slot_kernel<<<1, kThreads, 0, stream>>>(
        src, dst, bytes, slot_ids, request);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver
