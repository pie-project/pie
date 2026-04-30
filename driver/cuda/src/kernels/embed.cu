#include "kernels/embed.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// One block per token. Threads stride across `hidden`. Bounds-clamp the
// token id so a runaway BPIQ payload can't OOB-read. (Out-of-vocab → 0 row.)
__global__ void embed_bf16_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden, int vocab)
{
    const int n = blockIdx.x;
    const std::int32_t tid_raw = token_ids[n];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const __nv_bfloat16* row = weight + static_cast<long long>(tid) * hidden;
    __nv_bfloat16* out = y + static_cast<long long>(n) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = row[h];
    }
}

}  // namespace

void launch_embed_bf16(
    const std::int32_t* token_ids,
    const void* weight,
    void* y,
    int num_tokens, int hidden, int vocab,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    embed_bf16_kernel<<<grid, block, 0, stream>>>(
        token_ids,
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, vocab);
}

}  // namespace pie_cuda_driver::kernels
