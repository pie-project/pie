#include "kernels/split_packed.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// Which request owns query row `row`? Binary search over the CSR row pointer
// `qo_indptr` (R+1 ascending entries). `kv_paged.cu` scans linearly because it
// resolves once per token-block; here the resolve runs once per (row, kv-head)
// warp, so the O(log R) form earns its keep at R in the hundreds.
__device__ __forceinline__ int find_request_bsearch(
    const std::uint32_t* __restrict__ qo_indptr, int num_requests, int row)
{
    int lo = 0;
    int hi = num_requests - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (static_cast<int>(qo_indptr[mid]) <= row) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Resolve the paged-KV cell that query row `row` must append into.
//
// Decode (`qo_indptr == nullptr`): one new token per request, so the row IS
// the request and the target is that request's last occupied slot --
// `total_kv_after - 1`.
//
// Prefill / mixed (`qo_indptr != nullptr`): request `r` owns the row span
// [qo_indptr[r], qo_indptr[r+1]). `kv_last_page_lens` already counts the rows
// this fire appends, so the span's first row lands at
// `total_kv_after - new_tokens_r`. This is the same arithmetic
// `write_kv_kernel` performs, deliberately: the fused and unfused paths must
// pick the identical cell or a fire that switches between them would attend
// over stale keys.
__device__ __forceinline__ void resolve_kv_cell(
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int row, int num_requests, int page_size,
    int& actual_page, int& offset_in_page)
{
    int r = row;
    int new_tokens_r = 1;
    int offset_in_new = 0;
    if (qo_indptr != nullptr) {
        r = find_request_bsearch(qo_indptr, num_requests, row);
        const int qo_lo = static_cast<int>(qo_indptr[r]);
        new_tokens_r = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
        offset_in_new = row - qo_lo;
    }
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int pages_last = static_cast<int>(kv_page_indptr[r + 1]);
    const int total_kv_after =
        (pages_last - pages_first - 1) * page_size +
        static_cast<int>(kv_last_page_lens[r]);
    const int abs_kv_pos = total_kv_after - new_tokens_r + offset_in_new;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(
        kv_page_indices[pages_first + abs_kv_pos / page_size]);
}

// Copy one contiguous `dim`-element segment, 128 bits at a time.
//
// The comment this file shipped with promised a vectorised copy, but the body
// issued scalar 2-byte transactions: a 4096-wide QKV row moved at ~8% of HBM
// roof. `uint4` carries 8 bf16 per lane, which is what the hardware wants for
// a pure streaming copy. Callers that cannot guarantee 8-element alignment get
// the scalar instantiation instead.
__device__ __forceinline__ void copy_segment_vec8(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int dim, int tid, int nthreads)
{
    const int vectors = dim >> 3;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (int j = tid; j < vectors; j += nthreads) d4[j] = s4[j];
    for (int j = (vectors << 3) + tid; j < dim; j += nthreads) dst[j] = src[j];
}

__device__ __forceinline__ void copy_segment_scalar(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int dim, int tid, int nthreads)
{
    for (int j = tid; j < dim; j += nthreads) dst[j] = src[j];
}

// `VEC8` is decided on the host: it holds when every segment length is a
// multiple of 8, which makes each segment's base a multiple of 16 bytes given
// a cudaMalloc'd allocation. Every model we ship has head_dim and fc widths
// that are multiples of 8; the scalar instantiation is the honest fallback
// rather than a silently misaligned load.
template <bool VEC8>
__global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int q_dim, int kv_dim)
{
    const int n = blockIdx.y;
    const long long stride = q_dim + 2 * kv_dim;
    const __nv_bfloat16* src_row = src + static_cast<long long>(n) * stride;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    __nv_bfloat16* q_dst = q_out + static_cast<long long>(n) * q_dim;
    __nv_bfloat16* k_dst = k_out + static_cast<long long>(n) * kv_dim;
    __nv_bfloat16* v_dst = v_out + static_cast<long long>(n) * kv_dim;

    if constexpr (VEC8) {
        copy_segment_vec8(src_row, q_dst, q_dim, tid, nthreads);
        copy_segment_vec8(src_row + q_dim, k_dst, kv_dim, tid, nthreads);
        copy_segment_vec8(src_row + q_dim + kv_dim, v_dst, kv_dim,
                          tid, nthreads);
    } else {
        copy_segment_scalar(src_row, q_dst, q_dim, tid, nthreads);
        copy_segment_scalar(src_row + q_dim, k_dst, kv_dim, tid, nthreads);
        copy_segment_scalar(src_row + q_dim + kv_dim, v_dst, kv_dim,
                            tid, nthreads);
    }
}

template <bool VEC8>
__global__ void split_gate_up_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ up_out,
    int inter)
{
    const int n = blockIdx.y;
    const __nv_bfloat16* src_row =
        src + static_cast<long long>(n) * 2 * inter;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    __nv_bfloat16* gate_dst = gate_out + static_cast<long long>(n) * inter;
    __nv_bfloat16* up_dst = up_out + static_cast<long long>(n) * inter;

    if constexpr (VEC8) {
        copy_segment_vec8(src_row, gate_dst, inter, tid, nthreads);
        copy_segment_vec8(src_row + inter, up_dst, inter, tid, nthreads);
    } else {
        copy_segment_scalar(src_row, gate_dst, inter, tid, nthreads);
        copy_segment_scalar(src_row + inter, up_dst, inter, tid, nthreads);
    }
}

// Fused QKV postprocess: per-head Q/K RMSNorm + standard RoPE + direct paged
// KV append, in one pass over the packed projection output.
//
// `qo_indptr == nullptr` is the decode shape (row == request, one appended
// token each). Non-null generalises the kernel to prefill and mixed fires:
// rows are query TOKENS, `qo_indptr` maps them to requests, and `num_requests`
// bounds the search. Everything except the KV destination is indexed by ROW in
// both shapes -- `positions`, `rope_table`, `row_valid` and the WSlot/WOff
// descriptor are all per-token already.
template <int BLOCK, bool USE_ROPE_TABLE>
__global__ void qkv_qk_norm_rope_write_kv_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    const float* __restrict__ rope_table,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    const std::uint32_t* __restrict__ w_page,
    const std::uint32_t* __restrict__ w_off,
    const std::uint8_t* __restrict__ row_valid,
    int num_requests,
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
    if (!is_q && row_valid != nullptr && row_valid[r] == 0) return;
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
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[r]);
            offset_in_page = static_cast<int>(w_off[r]);
        } else {
            resolve_kv_cell(qo_indptr, kv_page_indices, kv_page_indptr,
                            kv_last_page_lens, r, num_requests, page_size,
                            actual_page, offset_in_page);
        }
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
__global__ void qkv_qk_norm_rope_write_kv_warp_kernel(
    const __nv_bfloat16* __restrict__ packed,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const std::int32_t* __restrict__ positions,
    const float* __restrict__ rope_table,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    const std::uint32_t* __restrict__ w_page,
    const std::uint32_t* __restrict__ w_off,
    const std::uint8_t* __restrict__ row_valid,
    int num_rows,
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
    if (unit >= num_rows * total_qk_heads) return;

    const int r = unit / total_qk_heads;
    const int head_idx = unit - r * total_qk_heads;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[r] == 0) return;
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
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[r]);
            offset_in_page = static_cast<int>(w_off[r]);
        } else {
            resolve_kv_cell(qo_indptr, kv_page_indices, kv_page_indptr,
                            kv_last_page_lens, r, num_requests, page_size,
                            actual_page, offset_in_page);
        }
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
    const std::uint8_t* __restrict__ row_valid,
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
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
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
    const bool vec8 = (q_dim % 8 == 0) && (kv_dim % 8 == 0);
    const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
    const int work = vec8 ? (max_dim >> 3) : max_dim;
    const int xblocks = (work + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks > 0 ? xblocks : 1, n_tokens);
    if (vec8) {
        split_qkv_kernel<true><<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(q_out),
            static_cast<__nv_bfloat16*>(k_out),
            static_cast<__nv_bfloat16*>(v_out),
            q_dim, kv_dim);
    } else {
        split_qkv_kernel<false><<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(q_out),
            static_cast<__nv_bfloat16*>(k_out),
            static_cast<__nv_bfloat16*>(v_out),
            q_dim, kv_dim);
    }
}

void launch_split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const bool vec8 = (inter % 8 == 0);
    const int work = vec8 ? (inter >> 3) : inter;
    const int xblocks = (work + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks > 0 ? xblocks : 1, n_tokens);
    if (vec8) {
        split_gate_up_kernel<true><<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(gate_out),
            static_cast<__nv_bfloat16*>(up_out),
            inter);
    } else {
        split_gate_up_kernel<false><<<grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(packed),
            static_cast<__nv_bfloat16*>(gate_out),
            static_cast<__nv_bfloat16*>(up_out),
            inter);
    }
}

void launch_qkv_qk_norm_rope_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const float* rope_table,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint8_t* row_valid,
    int num_rows,
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
    if (num_rows == 0) return;
    constexpr int WARP_BLOCK = 256;
    const int total_units = num_rows * (num_q_heads + num_kv_heads);
    dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
#define LAUNCH_QKV_POST_WARP(HEAD_DIM_VALUE, USE_TABLE)                      \
    qkv_qk_norm_rope_write_kv_warp_kernel<                                   \
        (HEAD_DIM_VALUE), (USE_TABLE)><<<warp_grid, WARP_BLOCK, 0, stream>>>( \
            static_cast<const __nv_bfloat16*>(packed),                       \
            static_cast<__nv_bfloat16*>(q_out),                              \
            static_cast<__nv_bfloat16*>(k_pages),                            \
            static_cast<__nv_bfloat16*>(v_pages),                            \
            static_cast<const __nv_bfloat16*>(q_weight),                     \
            static_cast<const __nv_bfloat16*>(k_weight),                     \
            positions, rope_table, qo_indptr, kv_page_indices,               \
            kv_page_indptr, kv_last_page_lens, w_page, w_off, row_valid,     \
            num_rows, num_requests, num_q_heads,                             \
            num_kv_heads, page_size, hnd_layout, theta, eps)
#define LAUNCH_QKV_POST_WARP_DIM(HEAD_DIM_VALUE)                             \
    do {                                                                     \
        if (rope_table != nullptr) {                                         \
            LAUNCH_QKV_POST_WARP(HEAD_DIM_VALUE, true);                      \
        } else {                                                             \
            LAUNCH_QKV_POST_WARP(HEAD_DIM_VALUE, false);                     \
        }                                                                    \
    } while (0)
    if (head_dim == 64) {
        LAUNCH_QKV_POST_WARP_DIM(64);
        return;
    }
    if (head_dim == 128) {
        LAUNCH_QKV_POST_WARP_DIM(128);
        return;
    }
    if (head_dim == 256) {
        LAUNCH_QKV_POST_WARP_DIM(256);
        return;
    }
#undef LAUNCH_QKV_POST_WARP_DIM
#undef LAUNCH_QKV_POST_WARP

    constexpr int BLOCK = 128;
    dim3 grid(num_rows, num_q_heads + num_kv_heads);
#define LAUNCH_QKV_POST_BLOCK(USE_TABLE)                                     \
    qkv_qk_norm_rope_write_kv_kernel<BLOCK, (USE_TABLE)>                     \
        <<<grid, BLOCK, 0, stream>>>(                                        \
            static_cast<const __nv_bfloat16*>(packed),                       \
            static_cast<__nv_bfloat16*>(q_out),                              \
            static_cast<__nv_bfloat16*>(k_pages),                            \
            static_cast<__nv_bfloat16*>(v_pages),                            \
            static_cast<const __nv_bfloat16*>(q_weight),                     \
            static_cast<const __nv_bfloat16*>(k_weight),                     \
            positions, rope_table, qo_indptr, kv_page_indices,               \
            kv_page_indptr, kv_last_page_lens, w_page, w_off, row_valid,     \
            num_requests, num_q_heads, num_kv_heads, head_dim, page_size,    \
            hnd_layout, theta, eps)
    if (rope_table != nullptr) {
        LAUNCH_QKV_POST_BLOCK(true);
    } else {
        LAUNCH_QKV_POST_BLOCK(false);
    }
#undef LAUNCH_QKV_POST_BLOCK
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
    const std::uint8_t* row_valid,
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
            row_valid,
            num_q_heads, num_kv_heads, head_dim, page_size, hnd_layout,
            theta, eps);
}

}  // namespace pie_cuda_driver::kernels
