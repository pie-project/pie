// The host launchers, and nothing else. Both `__global__`s live in
// `attn/head_dim_pad.cuh` -- ONE definition, read by nvcc here and by NVRTC
// from the same text at run time.
//
// Neither has a row: the grid is per-head and `LaunchRule::PerHead` is not
// evaluated by this backend yet. `attn/head_dim_pad.cuh` says why, and why
// approximating it with a ported rule was refused.
#include "attn/head_dim_pad.cuh"
#include "attn/head_dim_pad.hpp"


namespace pie_cuda_driver::kernels::attn {

namespace {

using bf16 = ::pie_cuda_driver::kernels::device::bf16;

constexpr int BLOCK = device::kPadBlock;

}  // namespace

void pad_head_dim_bf16(
    const void* packed, void* padded,
    int num_tokens, int num_heads, int head_dim, int head_dim_padded,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_heads <= 0) return;
    dim3 grid(num_heads, num_tokens);
    dim3 block(BLOCK);
    device::pad_head_dim<bf16><<<grid, block, 0, stream>>>(
        static_cast<const bf16*>(packed),
        static_cast<bf16*>(padded),
        num_heads, head_dim, head_dim_padded);
}

void strip_head_dim_bf16(
    const void* padded, void* packed,
    int num_tokens, int num_heads, int head_dim, int head_dim_padded,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_heads <= 0) return;
    dim3 grid(num_heads, num_tokens);
    dim3 block(BLOCK);
    device::strip_head_dim<bf16><<<grid, block, 0, stream>>>(
        static_cast<const bf16*>(padded),
        static_cast<bf16*>(packed),
        num_heads, head_dim, head_dim_padded);
}

}  // namespace pie_cuda_driver::kernels::attn
