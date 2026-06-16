#include "kernels/split_packed.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// Vectorise copies as ushort4 = 8 bf16 values. The matmul output dims
// (Hq, Hk, intermediate) are all multiples of head_dim or fc width and
// in practice multiples of 8 for every model we ship. Fall back to a
// scalar tail just in case.
__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int q_dim, int kv_dim)
{
    const int n = blockIdx.y;
    const int stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row = src + static_cast<long long>(n) * stride;

    // Q block: cols [0, q_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < q_dim;
         j += blockDim.x * gridDim.x) {
        q_out[static_cast<long long>(n) * q_dim + j] = src_row[j];
    }
    // K block: cols [q_dim, q_dim + kv_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kv_dim;
         j += blockDim.x * gridDim.x) {
        k_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + j];
    }
    // V block: cols [q_dim + kv_dim, q_dim + 2*kv_dim)
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < kv_dim;
         j += blockDim.x * gridDim.x) {
        v_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + kv_dim + j];
    }
}

__global__ void split_gate_up_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int inter)
{
    const int n = blockIdx.y;
    const int stride = 2 * inter;
    const __nv_bfloat16* src_row = src + static_cast<long long>(n) * stride;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        gate_out[static_cast<long long>(n) * inter + j] = src_row[j];
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        up_out[static_cast<long long>(n) * inter + j] = src_row[inter + j];
    }
}

template <int BLOCK, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_write_kv_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    const float* __restrict__ rope_table,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int r = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row =
        packed + static_cast<long long>(r) * packed_stride;
    const __nv_bfloat16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = __bfloat162float(src[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    __nv_bfloat16* dst = nullptr;
    __nv_bfloat16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(r) * num_q_heads + local_head) *
                      head_dim;
    } else {
        const int pages_first = kv_page_indptr[r];
        const int pages_last = kv_page_indptr[r + 1];
        const int num_pages_r = pages_last - pages_first;
        const int abs_kv_pos =
            (num_pages_r - 1) * page_size +
            static_cast<int>(kv_last_page_lens[r]) - 1;
        const int page_in_req = abs_kv_pos / page_size;
        const int offset_in_page = abs_kv_pos % page_size;
        const int actual_page =
            static_cast<int>(kv_page_indices[pages_first + page_in_req]);
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    if (!is_q) {
        const __nv_bfloat16* v_src =
            src_row + q_dim + kv_dim + local_head * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = v_src[i];
        }
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(r) * head_dim;
    } else {
        pos = positions[r];
    }
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = __bfloat162float(src[dim_pair]) *
            inv_rms * __bfloat162float(weight[dim_pair]);
        const float b = __bfloat162float(src[dim_pair + half]) *
            inv_rms * __bfloat162float(weight[dim_pair + half]);
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + half];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        dst[dim_pair] = __float2bfloat16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

template <int HEAD_DIM, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_write_kv_warp_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    const float* __restrict__ rope_table,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    static_assert(HEAD_DIM % 64 == 0);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_qk_heads = num_q_heads + num_kv_heads;
    const int unit = blockIdx.x * warps_per_block + warp_id;
    if (unit >= num_requests * total_qk_heads) return;

    const int r = unit / total_qk_heads;
    const int head_idx = unit - r * total_qk_heads;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * HEAD_DIM;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int packed_stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row =
        packed + static_cast<long long>(r) * packed_stride;
    const __nv_bfloat16* src = is_q
        ? src_row + local_head * HEAD_DIM
        : src_row + q_dim + local_head * HEAD_DIM;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float vals[ELEMS_PER_THREAD];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float v = __bfloat162float(src[dim]);
        vals[i] = v;
        sum += v * v;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset, 32);
    }

    const float inv_rms =
        rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        vals[i] *= inv_rms * __bfloat162float(weight[dim]);
    }

    const int pair_offset = (HEAD_DIM / 2) / ELEMS_PER_THREAD;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(r) * HEAD_DIM;
    } else {
        pos = positions[r];
    }
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float pair = __shfl_xor_sync(FULL_MASK, vals[i], pair_offset, 32);
        const float signed_pair = (lane < pair_offset) ? -pair : pair;
        const int dim_pair = (dim * 2) % HEAD_DIM / 2;
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + HEAD_DIM / 2];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(HEAD_DIM));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        vals[i] = vals[i] * cos_v + signed_pair * sin_v;
    }

    __nv_bfloat16* dst = nullptr;
    __nv_bfloat16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(r) * num_q_heads + local_head) *
                      HEAD_DIM;
    } else {
        const int pages_first = kv_page_indptr[r];
        const int pages_last = kv_page_indptr[r + 1];
        const int num_pages_r = pages_last - pages_first;
        const int abs_kv_pos =
            (num_pages_r - 1) * page_size +
            static_cast<int>(kv_last_page_lens[r]) - 1;
        const int page_in_req = abs_kv_pos / page_size;
        const int offset_in_page = abs_kv_pos % page_size;
        const int actual_page =
            static_cast<int>(kv_page_indices[pages_first + page_in_req]);
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * HEAD_DIM;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * HEAD_DIM;
            v_dst = v_pages + page_row + local_head * HEAD_DIM;
        }
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        dst[dim] = __float2bfloat16(vals[i]);
    }
    if (!is_q) {
        const __nv_bfloat16* v_src =
            src_row + q_dim + kv_dim + local_head * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int dim = lane * ELEMS_PER_THREAD + i;
            v_dst[dim] = v_src[dim];
        }
    }
}

template <int BLOCK>
__global__ void qkv_packed_qk_norm_rope_vnorm_write_kv_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int row = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    const __nv_bfloat16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    float local_v = 0.f;
    const __nv_bfloat16* v_src = nullptr;
    if (!is_q) {
        v_src = src_row + q_dim + kv_dim + local_head * head_dim;
    }
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = __bfloat162float(src[i]);
        local += v * v;
        if (!is_q) {
            const float vv = __bfloat162float(v_src[i]);
            local_v += vv * vv;
        }
    }

    __shared__ float buf[BLOCK];
    __shared__ float buf_v[BLOCK];
    buf[threadIdx.x] = local;
    buf_v[threadIdx.x] = local_v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            buf[threadIdx.x] += buf[threadIdx.x + off];
            buf_v[threadIdx.x] += buf_v[threadIdx.x + off];
        }
        __syncthreads();
    }

    __nv_bfloat16* dst = nullptr;
    __nv_bfloat16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      head_dim;
    } else {
        const int pages_first = kv_page_indptr[row];
        const int pages_last = kv_page_indptr[row + 1];
        const int num_pages_r = pages_last - pages_first;
        const int abs_kv_pos =
            (num_pages_r - 1) * page_size +
            static_cast<int>(kv_last_page_lens[row]) - 1;
        const int page_in_req = abs_kv_pos / page_size;
        const int offset_in_page = abs_kv_pos % page_size;
        const int actual_page =
            static_cast<int>(kv_page_indices[pages_first + page_in_req]);
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    const float inv_rms =
        rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[row];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const __nv_bfloat16 norm_a = __float2bfloat16(
            __bfloat162float(src[dim_pair]) *
            inv_rms * __bfloat162float(weight[dim_pair]));
        const __nv_bfloat16 norm_b = __float2bfloat16(
            __bfloat162float(src[dim_pair + half]) *
            inv_rms * __bfloat162float(weight[dim_pair + half]));
        const float a = __bfloat162float(norm_a);
        const float b = __bfloat162float(norm_b);
        const float freq = powf(
            theta,
            -2.f * static_cast<float>(dim_pair) /
                static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        dst[dim_pair] = __float2bfloat16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }

    if (!is_q) {
        const float inv_v =
            rsqrtf(buf_v[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = __float2bfloat16(__bfloat162float(v_src[i]) * inv_v);
        }
    }
}

}  // namespace

void launch_split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
    const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    split_qkv_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_out),
        static_cast<__nv_bfloat16*>(v_out),
        q_dim, kv_dim);
}

void launch_split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const int xblocks = (inter + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    split_gate_up_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        inter);
}

void launch_qkv_decode_qk_norm_rope_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const float* rope_table,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_requests == 0) return;
    constexpr int WARP_BLOCK = 256;
    const int total_units = num_requests * (num_q_heads + num_kv_heads);
    dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
#define LAUNCH_QKV_DECODE_POST_WARP(HEAD_DIM_VALUE)                         \
    do {                                                                     \
        if (rope_table != nullptr) {                                         \
            qkv_decode_qk_norm_rope_write_kv_warp_kernel<                   \
                (HEAD_DIM_VALUE), true><<<warp_grid, WARP_BLOCK, 0, stream>>>( \
                    static_cast<const __nv_bfloat16*>(packed),               \
                    static_cast<__nv_bfloat16*>(q_out),                      \
                    static_cast<__nv_bfloat16*>(k_pages),                    \
                    static_cast<__nv_bfloat16*>(v_pages),                    \
                    static_cast<const __nv_bfloat16*>(q_weight),             \
                    static_cast<const __nv_bfloat16*>(k_weight),             \
                    positions, rope_table, kv_page_indices, kv_page_indptr,  \
                    kv_last_page_lens, num_requests, num_q_heads,            \
                    num_kv_heads, page_size, hnd_layout, theta, eps);        \
        } else {                                                             \
            qkv_decode_qk_norm_rope_write_kv_warp_kernel<                   \
                (HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>( \
                    static_cast<const __nv_bfloat16*>(packed),               \
                    static_cast<__nv_bfloat16*>(q_out),                      \
                    static_cast<__nv_bfloat16*>(k_pages),                    \
                    static_cast<__nv_bfloat16*>(v_pages),                    \
                    static_cast<const __nv_bfloat16*>(q_weight),             \
                    static_cast<const __nv_bfloat16*>(k_weight),             \
                    positions, rope_table, kv_page_indices, kv_page_indptr,  \
                    kv_last_page_lens, num_requests, num_q_heads,            \
                    num_kv_heads, page_size, hnd_layout, theta, eps);        \
        }                                                                    \
    } while (0)
    if (head_dim == 64) {
        LAUNCH_QKV_DECODE_POST_WARP(64);
        return;
    }
    if (head_dim == 128) {
        LAUNCH_QKV_DECODE_POST_WARP(128);
        return;
    }
    if (head_dim == 256) {
        LAUNCH_QKV_DECODE_POST_WARP(256);
        return;
    }
#undef LAUNCH_QKV_DECODE_POST_WARP

    constexpr int BLOCK = 128;
    dim3 grid(num_requests, num_q_heads + num_kv_heads);
    if (rope_table != nullptr) {
        qkv_decode_qk_norm_rope_write_kv_kernel<BLOCK, true>
            <<<grid, BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(packed),
                static_cast<__nv_bfloat16*>(q_out),
                static_cast<__nv_bfloat16*>(k_pages),
                static_cast<__nv_bfloat16*>(v_pages),
                static_cast<const __nv_bfloat16*>(q_weight),
                static_cast<const __nv_bfloat16*>(k_weight),
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps);
    } else {
        qkv_decode_qk_norm_rope_write_kv_kernel<BLOCK, false>
            <<<grid, BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(packed),
                static_cast<__nv_bfloat16*>(q_out),
                static_cast<__nv_bfloat16*>(k_pages),
                static_cast<__nv_bfloat16*>(v_pages),
                static_cast<const __nv_bfloat16*>(q_weight),
                static_cast<const __nv_bfloat16*>(k_weight),
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps);
    }
}

void launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream)
{
    if (num_rows == 0) return;
    constexpr int BLOCK = 256;
    dim3 grid(num_rows, num_q_heads + num_kv_heads);
    qkv_packed_qk_norm_rope_vnorm_write_kv_kernel<BLOCK>
        <<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(q_out),
            static_cast<__nv_bfloat16*>(k_pages),
            static_cast<__nv_bfloat16*>(v_pages),
            static_cast<const __nv_bfloat16*>(q_weight),
            static_cast<const __nv_bfloat16*>(k_weight),
            positions, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            num_q_heads, num_kv_heads, head_dim, page_size, hnd_layout,
            theta, eps);
}

}  // namespace pie_cuda_driver::kernels
