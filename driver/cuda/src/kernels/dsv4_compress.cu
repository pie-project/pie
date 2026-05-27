#include "kernels/dsv4_compress.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void average_pool_kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, dim]
    __nv_bfloat16* __restrict__ output,       // [N/ratio, dim]
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int out_tok = idx / dim;
    const int in_start = out_tok * ratio;

    float sum = 0.f;
    const int end = min(in_start + ratio, N);
    for (int t = in_start; t < end; ++t) {
        sum += __bfloat162float(input[static_cast<long long>(t) * dim + d]);
    }
    output[static_cast<long long>(out_tok) * dim + d] =
        __float2bfloat16(sum / static_cast<float>(end - in_start));
}

__global__ void add_ape_kernel(
    __nv_bfloat16* __restrict__ data,    // [N_compressed, dim]
    const float* __restrict__ ape,       // [ratio, dim]
    int N_compressed,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_compressed * dim) return;

    const int d = idx % dim;
    const int tok = idx / dim;
    const int pos_in_window = tok % ratio;

    const float val = __bfloat162float(data[idx]) +
                      ape[pos_in_window * dim + d];
    data[idx] = __float2bfloat16(val);
}

}  // namespace

void launch_average_pool_bf16(
    const void* input,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    const int out_tokens = N / ratio;
    if (out_tokens <= 0 || dim <= 0) return;
    const int total = out_tokens * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    average_pool_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        N, dim, ratio);
}

void launch_add_ape_f32(
    void* data,
    const float* ape,
    int N_compressed,
    int dim,
    int ratio,
    cudaStream_t stream)
{
    if (N_compressed <= 0 || dim <= 0) return;
    const int total = N_compressed * dim;
    const int grid = (total + BLOCK - 1) / BLOCK;
    add_ape_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(data),
        ape,
        N_compressed, dim, ratio);
}

}  // namespace pie_cuda_driver::kernels
