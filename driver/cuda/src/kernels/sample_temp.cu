#include "kernels/sample_temp.hpp"

#include <cuda_bf16.h>
#include <cstdint>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

// SplitMix64 → [0, 1) float. Mixes a row seed with the column index so
// concurrent threads don't share PRNG state but each (seed, j) pair is
// stable across runs.
__device__ __forceinline__ float hash_uniform(std::uint64_t seed, int j) {
    std::uint64_t x = seed + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(j + 1);
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    // High 24 bits → [0, 1). Avoids exact 0 by masking and offsetting.
    const std::uint32_t bits = static_cast<std::uint32_t>(x >> 40);
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float gumbel_noise(std::uint64_t seed, int j) {
    const float u = hash_uniform(seed, j);
    return -logf(-logf(u));
}

__global__ void sample_temp_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ temperatures,
    const float* __restrict__ min_ps,
    const std::uint32_t* __restrict__ seeds,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * vocab;

    const float T = temperatures[row];
    const bool greedy = !(T > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / T);
    const float min_p = min_ps[row];
    const bool apply_min_p = min_p > 0.f && !greedy;
    const std::uint64_t seed = static_cast<std::uint64_t>(seeds[row]) ^ 0xa5a5a5a5ull;

    __shared__ float reduce_buf[BLOCK];

    // ── Pass 1: max logit (only needed for min-p masking).
    float min_threshold = -INFINITY;
    if (apply_min_p) {
        float local_max = -INFINITY;
        for (int j = tid; j < vocab; j += BLOCK) {
            local_max = fmaxf(local_max, __bfloat162float(row_ptr[j]));
        }
        reduce_buf[tid] = local_max;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
            __syncthreads();
        }
        const float max_logit = reduce_buf[0];
        // prob[j]/max_prob = exp(logit[j] - max_logit). Keep iff >= min_p.
        // logit[j] >= max_logit + log(min_p).
        min_threshold = max_logit + logf(min_p);
        __syncthreads();
    }

    // ── Pass 2: Gumbel-max over the (possibly filtered) logits.
    float best_val = -INFINITY;
    int   best_idx = 0;

    for (int j = tid; j < vocab; j += BLOCK) {
        const float logit = __bfloat162float(row_ptr[j]);
        if (apply_min_p && logit < min_threshold) continue;
        const float score = greedy
            ? logit
            : (logit * inv_T + gumbel_noise(seed, j));
        if (score > best_val || (score == best_val && j < best_idx)) {
            best_val = score;
            best_idx = j;
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

__global__ void sample_temp_compact_scatter_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ row_indices,
    const float* __restrict__ temperatures,
    const float* __restrict__ min_ps,
    const std::uint32_t* __restrict__ seeds,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int compact_row = blockIdx.x;
    const int original_row = row_indices[compact_row];
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr =
        logits + static_cast<long long>(compact_row) * vocab;

    const float T = temperatures[original_row];
    const bool greedy = !(T > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / T);
    const float min_p = min_ps[original_row];
    const bool apply_min_p = min_p > 0.f && !greedy;
    const std::uint64_t seed =
        static_cast<std::uint64_t>(seeds[original_row]) ^ 0xa5a5a5a5ull;

    __shared__ float reduce_buf[BLOCK];

    float min_threshold = -INFINITY;
    if (apply_min_p) {
        float local_max = -INFINITY;
        for (int j = tid; j < vocab; j += BLOCK) {
            local_max = fmaxf(local_max, __bfloat162float(row_ptr[j]));
        }
        reduce_buf[tid] = local_max;
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
            __syncthreads();
        }
        const float max_logit = reduce_buf[0];
        min_threshold = max_logit + logf(min_p);
        __syncthreads();
    }

    float best_val = -INFINITY;
    int best_idx = 0;
    for (int j = tid; j < vocab; j += BLOCK) {
        const float logit = __bfloat162float(row_ptr[j]);
        if (apply_min_p && logit < min_threshold) continue;
        const float score = greedy
            ? logit
            : (logit * inv_T + gumbel_noise(seed, j));
        if (score > best_val || (score == best_val && j < best_idx)) {
            best_val = score;
            best_idx = j;
        }
    }

    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = vals[tid + off];
            const int oi = idxs[tid + off];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out[original_row] = idxs[0];
}

__global__ void argmax_bf16_with_offset_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ out_tokens,
    float* __restrict__ out_values,
    int vocab,
    int vocab_offset)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr =
        logits + static_cast<long long>(row) * vocab;

    float best_val = -INFINITY;
    int best_idx = vocab_offset;
    for (int j = tid; j < vocab; j += BLOCK) {
        const float val = __bfloat162float(row_ptr[j]);
        const int tok = vocab_offset + j;
        if (val > best_val || (val == best_val && tok < best_idx)) {
            best_val = val;
            best_idx = tok;
        }
    }

    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = vals[tid + off];
            const int oi = idxs[tid + off];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_values[row] = vals[0];
        out_tokens[row] = idxs[0];
    }
}

__device__ __forceinline__ std::uint64_t pack_argmax_pair(float value, int token) {
    const std::uint32_t value_bits = __float_as_uint(value);
    return (static_cast<std::uint64_t>(value_bits) << 32) |
           static_cast<std::uint32_t>(token);
}

__device__ __forceinline__ float unpack_argmax_value(std::uint64_t pair) {
    return __uint_as_float(static_cast<std::uint32_t>(pair >> 32));
}

__device__ __forceinline__ int unpack_argmax_token(std::uint64_t pair) {
    return static_cast<int>(static_cast<std::uint32_t>(pair));
}

__global__ void argmax_bf16_pair_with_offset_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::uint64_t* __restrict__ out_pairs,
    int vocab,
    int vocab_offset)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr =
        logits + static_cast<long long>(row) * vocab;

    float best_val = -INFINITY;
    int best_idx = vocab_offset;
    for (int j = tid; j < vocab; j += BLOCK) {
        const float val = __bfloat162float(row_ptr[j]);
        const int tok = vocab_offset + j;
        if (val > best_val || (val == best_val && tok < best_idx)) {
            best_val = val;
            best_idx = tok;
        }
    }

    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const float ov = vals[tid + off];
            const int oi = idxs[tid + off];
            if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_pairs[row] = pack_argmax_pair(vals[0], idxs[0]);
    }
}

__global__ void select_global_argmax_kernel(
    const float* __restrict__ values_by_rank,
    const std::int32_t* __restrict__ tokens_by_rank,
    std::int32_t* __restrict__ out_tokens,
    int num_rows,
    int world_size)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float best_val = values_by_rank[row];
    int best_tok = tokens_by_rank[row];
    for (int rank = 1; rank < world_size; ++rank) {
        const int idx = rank * num_rows + row;
        const float val = values_by_rank[idx];
        const int tok = tokens_by_rank[idx];
        if (val > best_val || (val == best_val && tok < best_tok)) {
            best_val = val;
            best_tok = tok;
        }
    }
    out_tokens[row] = best_tok;
}

__global__ void select_global_argmax_pairs_kernel(
    const std::uint64_t* __restrict__ pairs_by_rank,
    std::int32_t* __restrict__ out_tokens,
    int num_rows,
    int world_size)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    std::uint64_t best_pair = pairs_by_rank[row];
    float best_val = unpack_argmax_value(best_pair);
    int best_tok = unpack_argmax_token(best_pair);
    for (int rank = 1; rank < world_size; ++rank) {
        const std::uint64_t pair = pairs_by_rank[rank * num_rows + row];
        const float val = unpack_argmax_value(pair);
        const int tok = unpack_argmax_token(pair);
        if (val > best_val || (val == best_val && tok < best_tok)) {
            best_val = val;
            best_tok = tok;
            best_pair = pair;
        }
    }
    (void)best_pair;
    out_tokens[row] = best_tok;
}

}  // namespace

void launch_sample_temp_bf16(
    const void* logits,
    const float* temperatures,
    const float* min_ps,
    const std::uint32_t* seeds,
    std::int32_t* out,
    int num_rows, int vocab,
    cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    sample_temp_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        temperatures, min_ps, seeds, out, vocab);
}

void launch_sample_temp_bf16_compact_scatter(
    const void* logits,
    const std::int32_t* row_indices,
    const float* temperatures,
    const float* min_ps,
    const std::uint32_t* seeds,
    std::int32_t* out,
    int num_samples, int vocab,
    cudaStream_t stream)
{
    if (num_samples <= 0) return;
    dim3 grid(num_samples);
    dim3 block(BLOCK);
    sample_temp_compact_scatter_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        row_indices, temperatures, min_ps, seeds, out, vocab);
}

void launch_argmax_bf16_with_offset(
    const void* logits,
    std::int32_t* out_tokens,
    float* out_values,
    int num_rows,
    int vocab_shard,
    int vocab_offset,
    cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    argmax_bf16_with_offset_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        out_tokens, out_values, vocab_shard, vocab_offset);
}

void launch_argmax_bf16_pair_with_offset(
    const void* logits,
    std::uint64_t* out_pairs,
    int num_rows,
    int vocab_shard,
    int vocab_offset,
    cudaStream_t stream)
{
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    argmax_bf16_pair_with_offset_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        out_pairs, vocab_shard, vocab_offset);
}

void launch_select_global_argmax(
    const float* values_by_rank,
    const std::int32_t* tokens_by_rank,
    std::int32_t* out_tokens,
    int num_rows,
    int world_size,
    cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((num_rows + block.x - 1) / block.x);
    select_global_argmax_kernel<<<grid, block, 0, stream>>>(
        values_by_rank, tokens_by_rank, out_tokens, num_rows, world_size);
}

void launch_select_global_argmax_pairs(
    const std::uint64_t* pairs_by_rank,
    std::int32_t* out_tokens,
    int num_rows,
    int world_size,
    cudaStream_t stream)
{
    dim3 block(128);
    dim3 grid((num_rows + block.x - 1) / block.x);
    select_global_argmax_pairs_kernel<<<grid, block, 0, stream>>>(
        pairs_by_rank, out_tokens, num_rows, world_size);
}

}  // namespace pie_cuda_driver::kernels
