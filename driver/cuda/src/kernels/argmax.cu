#include "kernels/argmax.hpp"

#include <cstddef>
#include <cstdlib>

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

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

__device__ __forceinline__ std::uint64_t pack_argmax_pair(float value, int token) {
    const std::uint32_t value_bits = __float_as_uint(value);
    return (static_cast<std::uint64_t>(value_bits) << 32) |
           static_cast<std::uint32_t>(token);
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

__global__ void argmax_bf16_partitioned_pairs_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::uint64_t* __restrict__ partial_pairs,
    int vocab,
    int parts)
{
    const int row = blockIdx.x;
    const int part = blockIdx.y;
    const int tid = threadIdx.x;
    const int chunk = (vocab + parts - 1) / parts;
    const int begin = part * chunk;
    const int end = min(vocab, begin + chunk);
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;

    float best_val = -INFINITY;
    int best_idx = begin;
    for (int i = begin + tid; i < end; i += BLOCK) {
        const float v = __bfloat162float(row_ptr[i]);
        update_argmax(v, i, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];
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

    if (tid == 0) {
        partial_pairs[static_cast<std::size_t>(part) * gridDim.x + row] =
            pack_argmax_pair(vals[0], idxs[0]);
    }
}

__global__ void argmax_bf16_partitioned_pairs_vec2_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::uint64_t* __restrict__ partial_pairs,
    int vocab,
    int parts)
{
    const int row = blockIdx.x;
    const int part = blockIdx.y;
    const int tid = threadIdx.x;
    const int chunk = (vocab + parts - 1) / parts;
    const int begin = part * chunk;
    const int end = min(vocab, begin + chunk);
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;
    const auto* row2 = reinterpret_cast<const __nv_bfloat162*>(row_ptr);

    float best_val = -INFINITY;
    int best_idx = begin;

    int even_begin = begin;
    if ((even_begin & 1) != 0) {
        if (tid == 0 && even_begin < end) {
            update_argmax(__bfloat162float(row_ptr[even_begin]),
                          even_begin, best_val, best_idx);
        }
        ++even_begin;
    }
    const int even_end = end & ~1;
    for (int j = even_begin / 2 + tid; j < even_end / 2; j += BLOCK) {
        const float2 pair = __bfloat1622float2(row2[j]);
        const int i = j * 2;
        update_argmax(pair.x, i, best_val, best_idx);
        update_argmax(pair.y, i + 1, best_val, best_idx);
    }
    if (tid == 0 && even_end < end) {
        update_argmax(__bfloat162float(row_ptr[even_end]),
                      even_end, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];
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

    if (tid == 0) {
        partial_pairs[static_cast<std::size_t>(part) * gridDim.x + row] =
            pack_argmax_pair(vals[0], idxs[0]);
    }
}

}  // namespace

void launch_argmax_bf16(
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

void launch_argmax_bf16_partitioned_pairs(
    const void* logits,
    std::uint64_t* partial_pairs,
    int num_rows,
    int vocab,
    int parts,
    cudaStream_t stream)
{
    if (num_rows <= 0 || vocab <= 0 || parts <= 1) return;
    dim3 grid(num_rows, parts);
    dim3 block(BLOCK);
    if (argmax_vec2_enabled()) {
        argmax_bf16_partitioned_pairs_vec2_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), partial_pairs, vocab, parts);
    } else {
        argmax_bf16_partitioned_pairs_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), partial_pairs, vocab, parts);
    }
}

}  // namespace pie_cuda_driver::kernels
