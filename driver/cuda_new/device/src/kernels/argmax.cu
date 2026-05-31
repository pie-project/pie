#include "argmax.cuh"

#include <cstddef>
#include <cstdlib>

#include <cuda_bf16.h>

// Lifted verbatim from driver/cuda/src/kernels/argmax.cu — the
// `launch_argmax_bf16` entry point and its dependency closure only. Lifted
// pieces: the BLOCK constant, the PIE_ARGMAX_VEC2 toggle (argmax_vec2_enabled),
// the update_argmax tie-break helper, and the two argmax kernels
// (argmax_bf16_kernel scalar + argmax_bf16_vec2_kernel). Every other variant in
// the source file is dropped. The only changes are the namespace
// (pie_cuda_driver -> pie_cuda_device) and the launcher name
// (launch_argmax_bf16 -> argmax_bf16).

namespace pie_cuda_device::kernels {

namespace {

constexpr int BLOCK = 256;

bool argmax_vec2_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_ARGMAX_VEC2");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

__device__ __forceinline__ void update_argmax(
    float v, int idx, float& best_val, int& best_idx)
{
    if (v > best_val || (v == best_val && idx < best_idx)) {
        best_val = v;
        best_idx = idx;
    }
}

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
        update_argmax(v, i, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ int   idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            update_argmax(vals[tid + off], idxs[tid + off],
                          vals[tid], idxs[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = idxs[0];
}

__global__ void argmax_bf16_vec2_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;
    const auto* row2 = reinterpret_cast<const __nv_bfloat162*>(row_ptr);

    float best_val = -INFINITY;
    int   best_idx = 0;

    const int even_end = vocab & ~1;
    for (int j = tid; j < even_end / 2; j += BLOCK) {
        const float2 vals = __bfloat1622float2(row2[j]);
        const int i = j * 2;
        update_argmax(vals.x, i, best_val, best_idx);
        update_argmax(vals.y, i + 1, best_val, best_idx);
    }
    if ((vocab & 1) && tid == 0) {
        update_argmax(__bfloat162float(row_ptr[vocab - 1]),
                      vocab - 1, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ int   idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            update_argmax(vals[tid + off], idxs[tid + off],
                          vals[tid], idxs[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) out[row] = idxs[0];
}

}  // namespace

void argmax_bf16(
    const void* logits, std::int32_t* token_ids,
    int num_rows, int vocab, cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    if (argmax_vec2_enabled()) {
        argmax_bf16_vec2_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), token_ids, vocab);
    } else {
        argmax_bf16_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), token_ids, vocab);
    }
}

}  // namespace pie_cuda_device::kernels
