#include "kernels/argmax.hpp"

#include <cstddef>
#include <cstdlib>

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_MASKED_TOP_K = 64;
constexpr int MASKED_TILE_TOKENS = 8;
constexpr int LM_HEAD_TILE_TOKENS = 8;

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

__global__ void argmax_fp32_kernel(
    const float* __restrict__ logits,
    std::int32_t* __restrict__ out,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_ptr = logits + static_cast<long long>(row) * vocab;

    float best_val = -INFINITY;
    int   best_idx = 0;

    for (int i = tid; i < vocab; i += BLOCK) {
        update_argmax(row_ptr[i], i, best_val, best_idx);
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

__global__ void argmax_bf16_tile_pair_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::uint64_t* __restrict__ out_pair,
    int tile_vocab,
    int token_offset)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * tile_vocab;

    float best_val = -INFINITY;
    int best_idx = token_offset;

    for (int i = tid; i < tile_vocab; i += BLOCK) {
        const float v = __bfloat162float(row_ptr[i]);
        update_argmax(v, token_offset + i, best_val, best_idx);
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
        out_pair[row] = pack_argmax_pair(vals[0], idxs[0]);
    }
}

__global__ void argmax_bf16_tile_pair_vec2_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::uint64_t* __restrict__ out_pair,
    int tile_vocab,
    int token_offset)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_ptr = logits + static_cast<long long>(row) * tile_vocab;
    const auto* row2 = reinterpret_cast<const __nv_bfloat162*>(row_ptr);

    float best_val = -INFINITY;
    int best_idx = token_offset;

    const int even_end = tile_vocab & ~1;
    for (int j = tid; j < even_end / 2; j += BLOCK) {
        const float2 vals = __bfloat1622float2(row2[j]);
        const int i = j * 2;
        update_argmax(vals.x, token_offset + i, best_val, best_idx);
        update_argmax(vals.y, token_offset + i + 1, best_val, best_idx);
    }
    if ((tile_vocab & 1) && tid == 0) {
        update_argmax(__bfloat162float(row_ptr[tile_vocab - 1]),
                      token_offset + tile_vocab - 1, best_val, best_idx);
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
        out_pair[row] = pack_argmax_pair(vals[0], idxs[0]);
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

__global__ void lm_head_argmax_pairs_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    std::uint64_t* __restrict__ partial_pairs,
    int hidden,
    int vocab)
{
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tok = tile * LM_HEAD_TILE_TOKENS + warp;
    const auto* hidden_row =
        hidden_states + static_cast<long long>(row) * hidden;

    __shared__ float vals[LM_HEAD_TILE_TOKENS];
    __shared__ int toks[LM_HEAD_TILE_TOKENS];

    float dot = -INFINITY;
    if (warp < LM_HEAD_TILE_TOKENS && tok < vocab) {
        const auto* wrow =
            lm_head_weight + static_cast<long long>(tok) * hidden;
        float sum = 0.f;
        for (int h = lane; h < hidden; h += 32) {
            sum += __bfloat162float(hidden_row[h]) *
                   __bfloat162float(wrow[h]);
        }
        dot = warp_sum(sum);
    }

    if (lane == 0) {
        vals[warp] = dot;
        toks[warp] = tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_val = vals[0];
        int best_tok = toks[0];
        for (int i = 1; i < LM_HEAD_TILE_TOKENS; ++i) {
            update_argmax(vals[i], toks[i], best_val, best_tok);
        }
        partial_pairs[static_cast<std::size_t>(tile) * gridDim.x + row] =
            pack_argmax_pair(best_val, best_tok);
    }
}

__device__ __forceinline__ float unpack_argmax_value(std::uint64_t pair) {
    const std::uint32_t bits = static_cast<std::uint32_t>(pair >> 32);
    return __uint_as_float(bits);
}

__device__ __forceinline__ int unpack_argmax_token(std::uint64_t pair) {
    return static_cast<int>(static_cast<std::uint32_t>(pair));
}

__global__ void select_lm_head_argmax_pairs_kernel(
    const std::uint64_t* __restrict__ partial_pairs,
    std::int32_t* __restrict__ out_tokens,
    int num_rows,
    int num_tiles)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    std::uint64_t best_pair = partial_pairs[row];
    float best_val = unpack_argmax_value(best_pair);
    int best_tok = unpack_argmax_token(best_pair);
    for (int tile = 1; tile < num_tiles; ++tile) {
        const std::uint64_t pair =
            partial_pairs[static_cast<std::size_t>(tile) * num_rows + row];
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

void launch_lm_head_argmax_bf16(
    const void* hidden_states,
    const void* lm_head_weight,
    std::uint64_t* partial_pairs,
    std::int32_t* token_ids,
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || vocab <= 0) return;
    const int tiles = (vocab + LM_HEAD_TILE_TOKENS - 1) / LM_HEAD_TILE_TOKENS;
    dim3 score_grid(num_rows, tiles);
    dim3 score_block(BLOCK);
    lm_head_argmax_pairs_bf16_kernel<<<score_grid, score_block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_states),
        static_cast<const __nv_bfloat16*>(lm_head_weight),
        partial_pairs,
        hidden,
        vocab);
    dim3 select_block(128);
    dim3 select_grid((num_rows + select_block.x - 1) / select_block.x);
    select_lm_head_argmax_pairs_kernel<<<select_grid, select_block, 0, stream>>>(
        partial_pairs, token_ids, num_rows, tiles);
}

// ── Fused INT8 GEMV + argmax ──────────────────────────────────────
// Persistent-block design: each block has 8 warps, each warp computes
// one dot product (one vocab row). Blocks loop over vocab rows with a
// grid-stride pattern. Hidden vector is loaded once to shared memory.
// After all assigned rows are scored, block-level argmax is written to
// partial_pairs for a small final reduction.

constexpr int GEMV_WARPS = 8;
constexpr int GEMV_BLOCK_DIM = GEMV_WARPS * 32; // 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void lm_head_gemv_argmax_int8_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const std::int8_t* __restrict__ lm_head_weight,
    const float* __restrict__ scale_inv,
    std::uint64_t* __restrict__ partial_pairs,
    int num_rows,
    int hidden,
    int vocab,
    int num_blocks_x)
{
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ char shmem_raw[];
    float* sh_hidden = reinterpret_cast<float*>(shmem_raw);

    const auto* h_row = hidden_states +
        static_cast<long long>(row) * hidden;
    for (int i = threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM) {
        sh_hidden[i] = __bfloat162float(h_row[i]);
    }
    __syncthreads();

    float best_val = -INFINITY;
    int best_tok = -1;

    for (int v = blockIdx.x * GEMV_WARPS + warp;
         v < vocab;
         v += num_blocks_x * GEMV_WARPS) {
        const auto* w_row = lm_head_weight +
            static_cast<long long>(v) * hidden;
        float sum = 0.f;
        const int h_vec4 = hidden / 4;
        const auto* w_row_v4 = reinterpret_cast<const char4*>(w_row);
        for (int i = lane; i < h_vec4; i += 32) {
            const int h = i * 4;
            const char4 w4 = __ldg(&w_row_v4[i]);
            sum += sh_hidden[h]     * static_cast<float>(w4.x)
                 + sh_hidden[h + 1] * static_cast<float>(w4.y)
                 + sh_hidden[h + 2] * static_cast<float>(w4.z)
                 + sh_hidden[h + 3] * static_cast<float>(w4.w);
        }
        float dot = warp_reduce_sum(sum) * scale_inv[v];

        if (lane == 0) {
            if (dot > best_val || (dot == best_val && v < best_tok)) {
                best_val = dot;
                best_tok = v;
            }
        }
    }

    // Warp-level best is in lane 0. Collect across warps.
    __shared__ float sh_vals[GEMV_WARPS];
    __shared__ int sh_toks[GEMV_WARPS];
    if (lane == 0) {
        sh_vals[warp] = best_val;
        sh_toks[warp] = best_tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bv = sh_vals[0];
        int bt = sh_toks[0];
        for (int w = 1; w < GEMV_WARPS; ++w) {
            update_argmax(sh_vals[w], sh_toks[w], bv, bt);
        }
        partial_pairs[static_cast<std::size_t>(blockIdx.x) * num_rows + row] =
            pack_argmax_pair(bv, bt);
    }
}

void launch_lm_head_gemv_argmax_int8(
    const void* hidden_states,
    const std::int8_t* lm_head_weight,
    const float* scale_inv,
    std::int32_t* token_ids,
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || vocab <= 0) return;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    static const int blocks_per_sm = [] {
        const char* v = std::getenv("PIE_GEMV_BLOCKS_PER_SM");
        if (v == nullptr || v[0] == '\0') return 2;
        return std::max(1, std::min(8, std::atoi(v)));
    }();
    const int max_blocks_x = num_sms * blocks_per_sm;
    const int min_blocks_x =
        (vocab + GEMV_WARPS - 1) / GEMV_WARPS;
    const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
    const std::size_t shmem_bytes =
        static_cast<std::size_t>(hidden) * sizeof(float);

    const std::size_t pairs_elems =
        static_cast<std::size_t>(num_blocks_x) * num_rows;
    static std::uint64_t* s_partial_pairs = nullptr;
    static std::size_t s_pairs_cap = 0;
    if (pairs_elems > s_pairs_cap) {
        if (s_partial_pairs) cudaFree(s_partial_pairs);
        cudaMalloc(&s_partial_pairs, pairs_elems * sizeof(std::uint64_t));
        s_pairs_cap = pairs_elems;
    }

    dim3 grid(num_blocks_x, num_rows);
    dim3 block(GEMV_BLOCK_DIM);
    lm_head_gemv_argmax_int8_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_states),
        lm_head_weight,
        scale_inv,
        s_partial_pairs,
        num_rows,
        hidden,
        vocab,
        num_blocks_x);

    dim3 sel_block(128);
    dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
    select_lm_head_argmax_pairs_kernel<<<sel_grid, sel_block, 0, stream>>>(
        s_partial_pairs, token_ids, num_rows, num_blocks_x);
}

// ── BF16 variant of fused GEMV + argmax ──────────────────────────
__global__ void lm_head_gemv_argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ lm_head_weight,
    std::uint64_t* __restrict__ partial_pairs,
    int num_rows,
    int hidden,
    int vocab,
    int num_blocks_x)
{
    const int row = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ char shmem_raw2[];
    float* sh_hidden = reinterpret_cast<float*>(shmem_raw2);

    const auto* h_row = hidden_states +
        static_cast<long long>(row) * hidden;
    for (int i = threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM) {
        sh_hidden[i] = __bfloat162float(h_row[i]);
    }
    __syncthreads();

    float best_val = -INFINITY;
    int best_tok = -1;

    for (int v = blockIdx.x * GEMV_WARPS + warp;
         v < vocab;
         v += num_blocks_x * GEMV_WARPS) {
        const auto* w_row = lm_head_weight +
            static_cast<long long>(v) * hidden;
        float sum = 0.f;
        const int h_vec2 = hidden / 2;
        const auto* w_row_v2 = reinterpret_cast<const __nv_bfloat162*>(w_row);
        for (int i = lane; i < h_vec2; i += 32) {
            const int h = i * 2;
            const __nv_bfloat162 w2 = __ldg(&w_row_v2[i]);
            sum += sh_hidden[h]     * __bfloat162float(w2.x)
                 + sh_hidden[h + 1] * __bfloat162float(w2.y);
        }
        float dot = warp_reduce_sum(sum);

        if (lane == 0) {
            if (dot > best_val || (dot == best_val && v < best_tok)) {
                best_val = dot;
                best_tok = v;
            }
        }
    }

    __shared__ float sh_vals2[GEMV_WARPS];
    __shared__ int sh_toks2[GEMV_WARPS];
    if (lane == 0) {
        sh_vals2[warp] = best_val;
        sh_toks2[warp] = best_tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bv = sh_vals2[0];
        int bt = sh_toks2[0];
        for (int w = 1; w < GEMV_WARPS; ++w) {
            update_argmax(sh_vals2[w], sh_toks2[w], bv, bt);
        }
        partial_pairs[static_cast<std::size_t>(blockIdx.x) * num_rows + row] =
            pack_argmax_pair(bv, bt);
    }
}

void launch_lm_head_gemv_argmax_bf16(
    const void* hidden_states,
    const void* lm_head_weight,
    std::int32_t* token_ids,
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || hidden <= 0 || vocab <= 0) return;

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    const int blocks_per_sm = 4;
    const int max_blocks_x = num_sms * blocks_per_sm;
    const int min_blocks_x =
        (vocab + GEMV_WARPS - 1) / GEMV_WARPS;
    const int num_blocks_x = std::min(max_blocks_x, min_blocks_x);
    const std::size_t shmem_bytes =
        static_cast<std::size_t>(hidden) * sizeof(float);

    const std::size_t pairs_elems =
        static_cast<std::size_t>(num_blocks_x) * num_rows;
    static std::uint64_t* s_partial_pairs_bf16 = nullptr;
    static std::size_t s_pairs_cap_bf16 = 0;
    if (pairs_elems > s_pairs_cap_bf16) {
        if (s_partial_pairs_bf16) cudaFree(s_partial_pairs_bf16);
        cudaMalloc(&s_partial_pairs_bf16, pairs_elems * sizeof(std::uint64_t));
        s_pairs_cap_bf16 = pairs_elems;
    }

    dim3 grid(num_blocks_x, num_rows);
    dim3 block(GEMV_BLOCK_DIM);
    lm_head_gemv_argmax_bf16_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_states),
        static_cast<const __nv_bfloat16*>(lm_head_weight),
        s_partial_pairs_bf16,
        num_rows,
        hidden,
        vocab,
        num_blocks_x);

    dim3 sel_block(128);
    dim3 sel_grid((num_rows + sel_block.x - 1) / sel_block.x);
    select_lm_head_argmax_pairs_kernel<<<sel_grid, sel_block, 0, stream>>>(
        s_partial_pairs_bf16, token_ids, num_rows, num_blocks_x);
}

void launch_argmax_fp32(
    const void* logits,
    std::int32_t* token_ids,
    int num_rows,
    int vocab,
    cudaStream_t stream)
{
    if (num_rows <= 0 || vocab <= 0) return;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    argmax_fp32_kernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(logits), token_ids, vocab);
}

void launch_argmax_bf16_tile_pair(
    const void* logits,
    std::uint64_t* out_pair,
    int num_rows,
    int tile_vocab,
    int token_offset,
    cudaStream_t stream)
{
    if (num_rows <= 0 || tile_vocab <= 0) return;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    if (argmax_vec2_enabled()) {
        argmax_bf16_tile_pair_vec2_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), out_pair,
            tile_vocab, token_offset);
    } else {
        argmax_bf16_tile_pair_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), out_pair,
            tile_vocab, token_offset);
    }
}

void launch_select_global_argmax_pairs(
    const std::uint64_t* pairs,
    std::int32_t* token_ids,
    int num_rows,
    int num_parts,
    cudaStream_t stream)
{
    if (num_rows <= 0 || num_parts <= 0) return;
    dim3 block(128);
    dim3 grid((num_rows + block.x - 1) / block.x);
    select_lm_head_argmax_pairs_kernel<<<grid, block, 0, stream>>>(
        pairs, token_ids, num_rows, num_parts);
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
