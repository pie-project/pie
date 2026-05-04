#include "kernels/gather_rows.hpp"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void gather_bf16_rows_kernel(
    const std::uint16_t* __restrict__ src,
    const std::int32_t* __restrict__ row_indices,
    std::uint16_t* __restrict__ dst,
    int vocab)
{
    const int slot = blockIdx.x;
    const int row = row_indices[slot];
    const std::uint16_t* src_row = src + static_cast<long long>(row) * vocab;
    std::uint16_t* dst_row = dst + static_cast<long long>(slot) * vocab;

    // Vectorize via uint4 (8 × u16) when the vocab is 8-aligned and
    // both rows are 16-byte aligned (which they are here: contiguous
    // `cudaMalloc`'d allocations satisfy 256-byte alignment).
    if ((vocab & 7) == 0) {
        const auto* src4 = reinterpret_cast<const uint4*>(src_row);
        auto*       dst4 = reinterpret_cast<uint4*>(dst_row);
        const int n4 = vocab >> 3;
        for (int j = threadIdx.x; j < n4; j += BLOCK) {
            dst4[j] = src4[j];
        }
    } else {
        for (int j = threadIdx.x; j < vocab; j += BLOCK) {
            dst_row[j] = src_row[j];
        }
    }
}

}  // namespace

void launch_gather_bf16_rows(
    const std::uint16_t* src,
    const std::int32_t*  row_indices,
    std::uint16_t*       dst,
    int                  num_dst_rows,
    int                  vocab,
    cudaStream_t         stream)
{
    if (num_dst_rows <= 0) return;
    gather_bf16_rows_kernel<<<num_dst_rows, BLOCK, 0, stream>>>(
        src, row_indices, dst, vocab);
}

}  // namespace pie_cuda_driver::kernels
