#include "kernels/head_dim_pad.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 128;

__global__ void pad_head_dim_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16*       __restrict__ padded,
    int num_heads, int head_dim, int head_dim_padded)
{
    // Each block handles one (token, head). Threads stride over the
    // padded extent so every thread executes a single bf16 store
    // (either a copy from `packed` or a zero) — no divergence.
    const int n = blockIdx.y;
    const int h = blockIdx.x;
    const __nv_bfloat16* in =
        packed + (static_cast<long long>(n) * num_heads + h) * head_dim;
    __nv_bfloat16* out =
        padded + (static_cast<long long>(n) * num_heads + h) * head_dim_padded;
    for (int d = threadIdx.x; d < head_dim_padded; d += BLOCK) {
        out[d] = (d < head_dim) ? in[d] : __float2bfloat16(0.f);
    }
}

__global__ void strip_head_dim_bf16_kernel(
    const __nv_bfloat16* __restrict__ padded,
    __nv_bfloat16*       __restrict__ packed,
    int num_heads, int head_dim, int head_dim_padded)
{
    const int n = blockIdx.y;
    const int h = blockIdx.x;
    const __nv_bfloat16* in =
        padded + (static_cast<long long>(n) * num_heads + h) * head_dim_padded;
    __nv_bfloat16* out =
        packed + (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
        out[d] = in[d];
    }
}

}  // namespace

void launch_pad_head_dim_bf16(
    const void* packed, void* padded,
    int num_tokens, int num_heads, int head_dim, int head_dim_padded,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_heads <= 0) return;
    dim3 grid(num_heads, num_tokens);
    dim3 block(BLOCK);
    pad_head_dim_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(padded),
        num_heads, head_dim, head_dim_padded);
}

void launch_strip_head_dim_bf16(
    const void* padded, void* packed,
    int num_tokens, int num_heads, int head_dim, int head_dim_padded,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_heads <= 0) return;
    dim3 grid(num_heads, num_tokens);
    dim3 block(BLOCK);
    strip_head_dim_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(padded),
        static_cast<__nv_bfloat16*>(packed),
        num_heads, head_dim, head_dim_padded);
}

}  // namespace pie_cuda_driver::kernels
