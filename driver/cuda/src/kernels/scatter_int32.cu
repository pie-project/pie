#include "kernels/scatter_int32.hpp"

namespace pie_cuda_driver::kernels {

namespace {

__global__ void zero_int32_kernel(std::int32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0;
}

__global__ void scatter_int32_kernel(
    std::int32_t* __restrict__ out,
    const std::int32_t* __restrict__ indices,
    const std::int32_t* __restrict__ values,
    int num_samples, int n_out)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_samples) return;
    const std::int32_t idx = indices[k];
    if (idx < 0 || idx >= n_out) return;  // defensive; planner builds in-range
    out[idx] = values[k];
}

}  // namespace

void launch_scatter_int32(
    std::int32_t* d_out,
    const std::int32_t* d_indices,
    const std::int32_t* d_values,
    int num_samples,
    int n_out,
    cudaStream_t stream)
{
    if (n_out <= 0) return;
    constexpr int BLOCK = 256;
    // Two-pass: zero, then scatter. Single-kernel "memset + scatter"
    // would require a barrier across blocks; two launches is simpler
    // and the zero is trivial.
    const int zblocks = (n_out + BLOCK - 1) / BLOCK;
    zero_int32_kernel<<<zblocks, BLOCK, 0, stream>>>(d_out, n_out);
    if (num_samples > 0) {
        const int sblocks = (num_samples + BLOCK - 1) / BLOCK;
        scatter_int32_kernel<<<sblocks, BLOCK, 0, stream>>>(
            d_out, d_indices, d_values, num_samples, n_out);
    }
}

}  // namespace pie_cuda_driver::kernels
