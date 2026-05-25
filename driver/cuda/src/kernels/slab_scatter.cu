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

}  // namespace pie_cuda_driver
