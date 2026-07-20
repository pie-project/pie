#include "kernels/slab_scatter.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

__global__ void slab_scatter_kernel(
    const std::uint8_t* __restrict__ src,
    std::uint8_t* __restrict__ dst,
    const SlabScatterPlacement* __restrict__ placements,
    std::size_t placement_count)
{
    const std::size_t placement_id = blockIdx.x;
    if (placement_id >= placement_count) {
        return;
    }
    const auto placement = placements[placement_id];
    const std::uint8_t* in = src + placement.src_offset;
    std::uint8_t* out = dst + placement.dest_offset;
    const std::uint64_t n = placement.bytes;

    std::uint64_t i = static_cast<std::uint64_t>(threadIdx.x) * 16;
    const std::uint64_t vector_stride = static_cast<std::uint64_t>(blockDim.x) * 16;
    std::uint64_t scalar_start = 0;
    if ((((reinterpret_cast<std::uintptr_t>(in) |
            reinterpret_cast<std::uintptr_t>(out)) & 0xF) == 0)) {
        for (; i + 16 <= n; i += vector_stride) {
            *reinterpret_cast<uint4*>(out + i) =
                *reinterpret_cast<const uint4*>(in + i);
        }
        scalar_start = (n / 16) * 16;
    }

    for (std::uint64_t j = scalar_start + threadIdx.x; j < n; j += blockDim.x) {
        out[j] = in[j];
    }
}

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

void launch_slab_scatter(
    const std::uint8_t* src,
    std::uint8_t* dst,
    const SlabScatterPlacement* placements,
    std::size_t placement_count,
    cudaStream_t stream)
{
    if (placement_count == 0) {
        return;
    }
    constexpr int kThreads = 256;
    slab_scatter_kernel<<<static_cast<unsigned int>(placement_count), kThreads, 0, stream>>>(
        src,
        dst,
        placements,
        placement_count);
    CUDA_CHECK(cudaGetLastError());
}

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
