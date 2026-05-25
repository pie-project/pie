#include "kernels/argmax.hpp"

#include <cstddef>
#include <cstdlib>

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_MASKED_TOP_K = 64;
constexpr int MASKED_TILE_TOKENS = 8;

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

__global__ void argmax_bf16_compact_scatter_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ row_indices,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int compact_row = blockIdx.x;
    const int original_row = row_indices[compact_row];
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr =
        logits + static_cast<long long>(compact_row) * vocab;

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

    if (tid == 0) out[original_row] = idxs[0];
}

__global__ void argmax_bf16_compact_scatter_vec2_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ row_indices,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int compact_row = blockIdx.x;
    const int original_row = row_indices[compact_row];
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr =
        logits + static_cast<long long>(compact_row) * vocab;
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

    if (tid == 0) out[original_row] = idxs[0];
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

__global__ void masked_embedding_argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ centroid_logits,
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    const std::int64_t* __restrict__ token_ordering,
    std::int32_t* __restrict__ out,
    int hidden,
    int num_centroids,
    int centroid_top_k,
    int vocab_per_centroid)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const auto* centroid_row =
        centroid_logits + static_cast<long long>(row) * num_centroids;
    const auto* hidden_row =
        hidden_states + static_cast<long long>(row) * hidden;

    __shared__ int top_centroids[MAX_MASKED_TOP_K];
    __shared__ float top_values[MAX_MASKED_TOP_K];
    __shared__ float vals[BLOCK];
    __shared__ int idxs[BLOCK];

    if (tid == 0) {
        for (int k = 0; k < centroid_top_k; ++k) {
            top_centroids[k] = 0;
            top_values[k] = -INFINITY;
        }
        for (int c = 0; c < num_centroids; ++c) {
            const float v = __bfloat162float(centroid_row[c]);
            int insert = centroid_top_k;
            for (int k = 0; k < centroid_top_k; ++k) {
                if (v > top_values[k] ||
                    (v == top_values[k] && c < top_centroids[k])) {
                    insert = k;
                    break;
                }
            }
            if (insert < centroid_top_k) {
                for (int k = centroid_top_k - 1; k > insert; --k) {
                    top_values[k] = top_values[k - 1];
                    top_centroids[k] = top_centroids[k - 1];
                }
                top_values[insert] = v;
                top_centroids[insert] = c;
            }
        }
    }
    __syncthreads();

    float best_val = -INFINITY;
    int best_tok = 0;
    const int selected = centroid_top_k * vocab_per_centroid;
    for (int s = tid; s < selected; s += BLOCK) {
        const int centroid_rank = s / vocab_per_centroid;
        const int centroid_off = s - centroid_rank * vocab_per_centroid;
        const int centroid = top_centroids[centroid_rank];
        const std::int64_t tok64 =
            token_ordering[static_cast<long long>(centroid) *
                               vocab_per_centroid +
                           centroid_off];
        const int tok = static_cast<int>(tok64);
        const auto* wrow =
            lm_head_weight + static_cast<long long>(tok) * hidden;
        float dot = 0.f;
        for (int h = 0; h < hidden; ++h) {
            dot += __bfloat162float(hidden_row[h]) *
                   __bfloat162float(wrow[h]);
        }
        update_argmax(dot, tok, best_val, best_tok);
    }

    vals[tid] = best_val;
    idxs[tid] = best_tok;
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

__global__ void topk_centroids_bf16_kernel(
    const __nv_bfloat16* __restrict__ centroid_logits,
    std::int32_t* __restrict__ top_centroids,
    int num_centroids,
    int centroid_top_k)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const auto* centroid_row =
        centroid_logits + static_cast<long long>(row) * num_centroids;

    __shared__ int top_idx[MAX_MASKED_TOP_K];
    __shared__ float top_val[MAX_MASKED_TOP_K];

    if (tid == 0) {
        for (int k = 0; k < centroid_top_k; ++k) {
            top_idx[k] = 0;
            top_val[k] = -INFINITY;
        }
        for (int c = 0; c < num_centroids; ++c) {
            const float v = __bfloat162float(centroid_row[c]);
            int insert = centroid_top_k;
            for (int k = 0; k < centroid_top_k; ++k) {
                if (v > top_val[k] || (v == top_val[k] && c < top_idx[k])) {
                    insert = k;
                    break;
                }
            }
            if (insert < centroid_top_k) {
                for (int k = centroid_top_k - 1; k > insert; --k) {
                    top_val[k] = top_val[k - 1];
                    top_idx[k] = top_idx[k - 1];
                }
                top_val[insert] = v;
                top_idx[insert] = c;
            }
        }
        for (int k = 0; k < centroid_top_k; ++k) {
            top_centroids[static_cast<long long>(row) * centroid_top_k + k] =
                top_idx[k];
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ void masked_embedding_tile_argmax_pairs_bf16_kernel(
    const std::int32_t* __restrict__ top_centroids,
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    const std::int64_t* __restrict__ token_ordering,
    std::uint64_t* __restrict__ partial_pairs,
    int hidden,
    int centroid_top_k,
    int vocab_per_centroid,
    int selected,
    int num_tiles)
{
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int selected_idx = tile * MASKED_TILE_TOKENS + warp;
    const auto* hidden_row =
        hidden_states + static_cast<long long>(row) * hidden;

    __shared__ float vals[MASKED_TILE_TOKENS];
    __shared__ int toks[MASKED_TILE_TOKENS];

    float dot = -INFINITY;
    int tok = 0;
    if (warp < MASKED_TILE_TOKENS && selected_idx < selected) {
        const int centroid_rank = selected_idx / vocab_per_centroid;
        const int centroid_off = selected_idx - centroid_rank * vocab_per_centroid;
        if (centroid_rank < centroid_top_k) {
            const int centroid =
                top_centroids[static_cast<long long>(row) * centroid_top_k +
                              centroid_rank];
            const std::int64_t tok64 =
                token_ordering[static_cast<long long>(centroid) *
                                   vocab_per_centroid +
                               centroid_off];
            tok = static_cast<int>(tok64);
            const auto* wrow =
                lm_head_weight + static_cast<long long>(tok) * hidden;
            float sum = 0.f;
            for (int h = lane; h < hidden; h += 32) {
                sum += __bfloat162float(hidden_row[h]) *
                       __bfloat162float(wrow[h]);
            }
            dot = warp_sum(sum);
        }
    }

    if (lane == 0) {
        vals[warp] = dot;
        toks[warp] = tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_val = vals[0];
        int best_tok = toks[0];
        for (int i = 1; i < MASKED_TILE_TOKENS; ++i) {
            update_argmax(vals[i], toks[i], best_val, best_tok);
        }
        partial_pairs[static_cast<std::size_t>(tile) * gridDim.x + row] =
            pack_argmax_pair(best_val, best_tok);
    }
    (void)num_tiles;
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

void launch_argmax_bf16_compact_scatter(
    const void* logits,
    const std::int32_t* row_indices,
    std::int32_t* token_ids,
    int num_rows,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || vocab <= 0) return;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    if (argmax_vec2_enabled()) {
        argmax_bf16_compact_scatter_vec2_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), row_indices,
            token_ids, vocab);
    } else {
        argmax_bf16_compact_scatter_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), row_indices,
            token_ids, vocab);
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

void launch_masked_embedding_argmax_bf16(
    const void* centroid_logits,
    const void* hidden_states,
    const void* lm_head_weight,
    const std::int64_t* token_ordering,
    std::int32_t* token_ids,
    int num_rows,
    int hidden,
    int num_centroids,
    int centroid_top_k,
    int vocab_per_centroid,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || num_centroids <= 0 ||
        centroid_top_k <= 0 || vocab_per_centroid <= 0) {
        return;
    }
    if (centroid_top_k > MAX_MASKED_TOP_K) {
        centroid_top_k = MAX_MASKED_TOP_K;
    }
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    masked_embedding_argmax_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(centroid_logits),
        static_cast<const __nv_bfloat16*>(hidden_states),
        static_cast<const __nv_bfloat16*>(lm_head_weight),
        token_ordering,
        token_ids,
        hidden,
        num_centroids,
        centroid_top_k,
        vocab_per_centroid);
}

void launch_topk_centroids_bf16(
    const void* centroid_logits,
    std::int32_t* top_centroids,
    int num_rows,
    int num_centroids,
    int centroid_top_k,
    cudaStream_t stream)
{
    if (num_rows <= 0 || num_centroids <= 0 || centroid_top_k <= 0) return;
    if (centroid_top_k > MAX_MASKED_TOP_K) centroid_top_k = MAX_MASKED_TOP_K;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    topk_centroids_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(centroid_logits),
        top_centroids,
        num_centroids,
        centroid_top_k);
}

void launch_masked_embedding_tile_argmax_pairs_bf16(
    const std::int32_t* top_centroids,
    const void* hidden_states,
    const void* lm_head_weight,
    const std::int64_t* token_ordering,
    std::uint64_t* partial_pairs,
    int num_rows,
    int hidden,
    int centroid_top_k,
    int vocab_per_centroid,
    int num_tiles,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || centroid_top_k <= 0 ||
        vocab_per_centroid <= 0 || num_tiles <= 0) {
        return;
    }
    if (centroid_top_k > MAX_MASKED_TOP_K) centroid_top_k = MAX_MASKED_TOP_K;
    const int selected = centroid_top_k * vocab_per_centroid;
    dim3 grid(num_rows, num_tiles);
    dim3 block(BLOCK);
    masked_embedding_tile_argmax_pairs_bf16_kernel<<<grid, block, 0, stream>>>(
        top_centroids,
        static_cast<const __nv_bfloat16*>(hidden_states),
        static_cast<const __nv_bfloat16*>(lm_head_weight),
        token_ordering,
        partial_pairs,
        hidden,
        centroid_top_k,
        vocab_per_centroid,
        selected,
        num_tiles);
}

}  // namespace pie_cuda_driver::kernels
