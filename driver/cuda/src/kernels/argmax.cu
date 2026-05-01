#include "kernels/argmax.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

// One block per row. Threads stride across `vocab`. Tie-break: lowest index
// wins — matches torch.argmax / numpy.argmax.
__global__ void argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;

    float best_val = -INFINITY;
    int   best_idx = 0;

    for (int i = tid; i < vocab; i += BLOCK) {
        const float v = __bfloat162float(row_ptr[i]);
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    __shared__ float vals[BLOCK];
    __shared__ int   idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = vals[tid + off];
            const int   oi = idxs[tid + off];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = idxs[0];
}

}  // namespace

void launch_argmax_bf16(
    const void* logits, std::int32_t* token_ids,
    int num_rows, int vocab, cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    argmax_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits), token_ids, vocab);
}

}  // namespace pie_cuda_driver::kernels
