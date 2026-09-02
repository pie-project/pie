#pragma once

#include "prelude/device.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace pie::attn {

enum class KvScheme : u8 {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
};

enum class KvDType : u8 {
    BF16 = 0,
    FP16 = 1,
    FP32 = 2,
    INT8 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
    FP8_E4M3 = 7,
    FP8_E5M2 = 8,
    INT4_PACKED = 9,

    MXFP4_PACKED = 10,
    E8M0 = 11,
};

__device__ __forceinline__ int find_request(const u32* qo_indptr,
                                            int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

template <bool HND_LAYOUT>
__global__ void kv_append(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ devwin,
    int R,
    int page_size,
    int h_kv,
    int d,
    int first_token)
{

    const int t = blockIdx.x + first_token;

    // `devwin` is the pre-staged device window pair, `(start, count)`:
    // word 0 is a START. The staged-geometry seat's `win` is `(count,
    // start)` — same pointer shape, opposite word order, and the rename
    // is what keeps one from ever arming the other.
    if (devwin != nullptr) {
        const int w0 = static_cast<int>(devwin[0]);
        const int w1 = static_cast<int>(devwin[1]);
        if (t < w0 || t >= w0 + w1) return;
    }
    if (row_valid != nullptr && row_valid[t] == 0) return;

    const int r = find_request(qo_indptr, R, t);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = t - qo_lo;

    const int pages_first = kv_page_indptr[r];
    const int pages_last  = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after = (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;

    const int page_in_req     = abs_kv_pos / page_size;
    const int offset_in_page  = abs_kv_pos % page_size;
    const int actual_page     = static_cast<int>(kv_page_indices[pages_first + page_in_req]);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(t) * row;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        long long dst;
        if constexpr (HND_LAYOUT) {
            const int h = i / d;
            const int j = i - h * d;
            dst = ((static_cast<long long>(actual_page) * h_kv + h) *
                   page_size + offset_in_page) * d + j;
        } else {
            dst = ((static_cast<long long>(actual_page) * page_size) +
                   offset_in_page) * row + i;
        }
        k_pages[dst] = k_curr[src + i];
        v_pages[dst] = v_curr[src + i];
    }
}

template <bool HND_LAYOUT>
__global__ void kv_append_at_positions(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const i32* __restrict__ positions,
    int position_delta,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int R,
    int page_size,
    int h_kv,
    int d)
{
    const int t = blockIdx.x;
    const int abs_kv_pos = positions[t] + position_delta;
    if (abs_kv_pos < 0) return;

    const int r = find_request(qo_indptr, R, t);
    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int page_in_req = abs_kv_pos / page_size;
    if (page_in_req < 0 || pages_first + page_in_req >= pages_last) return;
    const int offset_in_page = abs_kv_pos % page_size;
    const int actual_page =
        static_cast<int>(kv_page_indices[pages_first + page_in_req]);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(t) * row;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        long long dst;
        if constexpr (HND_LAYOUT) {
            const int h = i / d;
            const int j = i - h * d;
            dst = ((static_cast<long long>(actual_page) * h_kv + h) *
                   page_size + offset_in_page) * d + j;
        } else {
            dst = ((static_cast<long long>(actual_page) * page_size) +
                   offset_in_page) * row + i;
        }
        k_pages[dst] = k_curr[src + i];
        v_pages[dst] = v_curr[src + i];
    }
}

template <bool HND_LAYOUT>
__global__ void kv_append_explicit(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int B,
    int page_size,
    int h_kv,
    int d,
    const u32* __restrict__ win)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && b >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The SOURCE
    // rows move by it, and with them `row_valid` and the staged `w_page` /
    // `w_off` write tables, which are row planes handed at their base too. The
    // cell they name does not move: it is a page ordinal and an offset inside
    // that page, the lanes' own, and no window's zero rebases it.
    const int b_row = win != nullptr ? b + static_cast<int>(win[1]) : b;
    if (row_valid != nullptr && row_valid[b_row] == 0) return;
    const int actual_page = static_cast<int>(w_page[b_row]);
    const int offset_in_page = static_cast<int>(w_off[b_row]);
    if (offset_in_page < 0 || offset_in_page >= page_size) return;

    const long long row = static_cast<long long>(h_kv) * d;
    const long long src = static_cast<long long>(b_row) * row;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        long long dst;
        if constexpr (HND_LAYOUT) {
            const int h = i / d;
            const int j = i - h * d;
            dst = ((static_cast<long long>(actual_page) * h_kv + h) *
                   page_size + offset_in_page) * d + j;
        } else {
            dst = ((static_cast<long long>(actual_page) * page_size) +
                   offset_in_page) * row + i;
        }
        k_pages[dst] = k_curr[src + i];
        v_pages[dst] = v_curr[src + i];
    }
}

template <bool HND_LAYOUT>
__global__ void copy_kv_cells(
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ dst_page,
    const u32* __restrict__ dst_off,
    const u32* __restrict__ src_page,
    const u32* __restrict__ src_off,
    int N,
    int page_size,
    int h_kv,
    int d)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const int dpage = static_cast<int>(dst_page[n]);
    const int doff  = static_cast<int>(dst_off[n]);
    const int spage = static_cast<int>(src_page[n]);
    const int soff  = static_cast<int>(src_off[n]);
    if (doff < 0 || doff >= page_size || soff < 0 || soff >= page_size) return;

    const long long row = static_cast<long long>(h_kv) * d;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        long long dst, src;
        if constexpr (HND_LAYOUT) {
            const int h = i / d;
            const int j = i - h * d;
            dst = ((static_cast<long long>(dpage) * h_kv + h) * page_size + doff) * d + j;
            src = ((static_cast<long long>(spage) * h_kv + h) * page_size + soff) * d + j;
        } else {
            dst = ((static_cast<long long>(dpage) * page_size) + doff) * row + i;
            src = ((static_cast<long long>(spage) * page_size) + soff) * row + i;
        }
        k_pages[dst] = k_pages[src];
        v_pages[dst] = v_pages[src];
    }
}

__device__ __forceinline__ void resolve_dst(
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int token_idx,
    int& actual_page,
    int& offset_in_page)
{
    const int r = find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;
    const int pages_first = kv_page_indptr[r];
    const int pages_last  = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after = (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(kv_page_indices[pages_first + page_in_req]);
}

__global__ void kv_append_fp8_per_tensor(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    __nv_fp8_storage_t*  __restrict__ k_pages,
    __nv_fp8_storage_t*  __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d,
    __nv_fp8_interpretation_t fp8_kind,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The SOURCE
    // rows move by it; the cell they land in does not, because the qo
    // boundaries this resolves against are the window's own, rebased to its
    // zero, and the page tables are the lanes'.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(t_row) * row;
    const long long dst =
        ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * row;

    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        const float kf = bf16_to_f32(k_curr[src + i]);
        const float vf = bf16_to_f32(v_curr[src + i]);
        k_pages[dst + i] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, fp8_kind);
        v_pages[dst + i] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, fp8_kind);
    }
}

template <bool UseFp8>
__global__ void kv_append_per_token_head(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    void*                __restrict__ k_pages_raw,
    void*                __restrict__ v_pages_raw,
    float*               __restrict__ k_scales,
    float*               __restrict__ v_scales,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The SOURCE
    // rows move by it; the cell they land in does not, because the qo
    // boundaries this resolves against are the window's own, rebased to its
    // zero, and the page tables are the lanes'.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float shmem[];
    float* k_warp = shmem;
    float* v_warp = shmem + blockDim.x / 32;

    const long long src_base =
        (static_cast<long long>(t_row) * h_kv + h) * d;
    float k_abs = 0.f;
    float v_abs = 0.f;
    for (int j = tid; j < d; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(bf16_to_f32(k_curr[src_base + j])));
        v_abs = fmaxf(v_abs, fabsf(bf16_to_f32(v_curr[src_base + j])));
    }
    for (int off = 16; off > 0; off >>= 1) {
        k_abs = fmaxf(k_abs, __shfl_down_sync(0xffffffff, k_abs, off));
        v_abs = fmaxf(v_abs, __shfl_down_sync(0xffffffff, v_abs, off));
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) {
        k_warp[warp] = k_abs;
        v_warp[warp] = v_abs;
    }
    __syncthreads();
    if (warp == 0) {
        k_abs = (tid < blockDim.x / 32) ? k_warp[lane] : 0.f;
        v_abs = (tid < blockDim.x / 32) ? v_warp[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            k_abs = fmaxf(k_abs, __shfl_down_sync(0xffffffff, k_abs, off));
            v_abs = fmaxf(v_abs, __shfl_down_sync(0xffffffff, v_abs, off));
        }
        if (lane == 0) {
            k_warp[0] = k_abs;
            v_warp[0] = v_abs;
        }
    }
    __syncthreads();
    k_abs = k_warp[0];
    v_abs = v_warp[0];

    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);
    const long long dst_base =
        ((static_cast<long long>(actual_page) * page_size + offset_in_page) *
         h_kv + h) * d;
    const long long scale_idx =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        h_kv + h;

    const float qmax = UseFp8 ? 448.f : 127.f;
    const float k_scale = (k_abs > 0.f) ? (k_abs / qmax) : 1.f;
    const float v_scale = (v_abs > 0.f) ? (v_abs / qmax) : 1.f;
    if (tid == 0) {
        k_scales[scale_idx] = k_scale;
        v_scales[scale_idx] = v_scale;
    }
    const float k_inv = (k_scale > 0.f) ? (1.f / k_scale) : 0.f;
    const float v_inv = (v_scale > 0.f) ? (1.f / v_scale) : 0.f;

    if constexpr (UseFp8) {
        auto* k_pages = static_cast<__nv_fp8_storage_t*>(k_pages_raw);
        auto* v_pages = static_cast<__nv_fp8_storage_t*>(v_pages_raw);
        for (int j = tid; j < d; j += blockDim.x) {
            k_pages[dst_base + j] = __nv_cvt_float_to_fp8(
                bf16_to_f32(k_curr[src_base + j]) * k_inv,
                __NV_SATFINITE, __NV_E4M3);
            v_pages[dst_base + j] = __nv_cvt_float_to_fp8(
                bf16_to_f32(v_curr[src_base + j]) * v_inv,
                __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        auto* k_pages = static_cast<i8*>(k_pages_raw);
        auto* v_pages = static_cast<i8*>(v_pages_raw);
        for (int j = tid; j < d; j += blockDim.x) {
            int kq = static_cast<int>(rintf(bf16_to_f32(k_curr[src_base + j]) * k_inv));
            int vq = static_cast<int>(rintf(bf16_to_f32(v_curr[src_base + j]) * v_inv));
            kq = kq > 127 ? 127 : (kq < -128 ? -128 : kq);
            vq = vq > 127 ? 127 : (vq < -128 ? -128 : vq);
            k_pages[dst_base + j] = static_cast<i8>(kq);
            v_pages[dst_base + j] = static_cast<i8>(vq);
        }
    }
}

__device__ __forceinline__ float fp4_e2m1_value(u8 code) {
    const bool neg = (code & 0x8) != 0;
    const int mag = code & 0x7;
    float v = 0.f;
    switch (mag) {
        case 0: v = 0.f; break;
        case 1: v = 0.5f; break;
        case 2: v = 1.f; break;
        case 3: v = 1.5f; break;
        case 4: v = 2.f; break;
        case 5: v = 3.f; break;
        case 6: v = 4.f; break;
        default: v = 6.f; break;
    }
    return neg ? -v : v;
}

__device__ __forceinline__ u8 quant_fp4_e2m1(float x) {
    const bool neg = x < 0.f;
    float ax = fabsf(x);
    constexpr float levels[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    int best = 0;
    float best_err = fabsf(ax - levels[0]);
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - levels[i]);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    return static_cast<u8>((neg ? 0x8 : 0) | best);
}

__global__ void kv_append_fp4_block(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    u8*        __restrict__ k_pages,
    u8*        __restrict__ v_pages,
    float*               __restrict__ k_scales,
    float*               __restrict__ v_scales,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d,
    int block_size,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The SOURCE
    // rows move by it; the cell they land in does not, because the qo
    // boundaries this resolves against are the window's own, rebased to its
    // zero, and the page tables are the lanes'.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int start = b * block_size;
    const int end = (start + block_size < d) ? start + block_size : d;
    const int tid = threadIdx.x;
    __shared__ float scales[2];
    if (tid == 0) {
        scales[0] = 0.f;
        scales[1] = 0.f;
    }
    __syncthreads();

    const long long src_base =
        (static_cast<long long>(t_row) * h_kv + h) * d;
    float k_abs = 0.f;
    float v_abs = 0.f;
    for (int j = start + tid; j < end; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(bf16_to_f32(k_curr[src_base + j])));
        v_abs = fmaxf(v_abs, fabsf(bf16_to_f32(v_curr[src_base + j])));
    }
    for (int off = 16; off > 0; off >>= 1) {
        k_abs = fmaxf(k_abs, __shfl_down_sync(0xffffffff, k_abs, off));
        v_abs = fmaxf(v_abs, __shfl_down_sync(0xffffffff, v_abs, off));
    }
    if ((tid & 31) == 0) {
        scales[0] = fmaxf(scales[0], k_abs);
        scales[1] = fmaxf(scales[1], v_abs);
    }
    __syncthreads();
    if (tid == 0) {
        scales[0] = (scales[0] > 0.f) ? scales[0] / 6.f : 1.f;
        scales[1] = (scales[1] > 0.f) ? scales[1] / 6.f : 1.f;
    }
    __syncthreads();

    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);

    const int packed_d = (d + 1) / 2;
    const int blocks_per_head = (d + block_size - 1) / block_size;
    const long long packed_base =
        ((static_cast<long long>(actual_page) * page_size + offset_in_page) *
         h_kv + h) * packed_d;
    const long long scale_idx =
        ((static_cast<long long>(actual_page) * page_size + offset_in_page) *
         h_kv + h) * blocks_per_head + b;
    if (tid == 0) {
        k_scales[scale_idx] = scales[0];
        v_scales[scale_idx] = scales[1];
    }

    const float k_inv = (scales[0] > 0.f) ? 1.f / scales[0] : 0.f;
    const float v_inv = (scales[1] > 0.f) ? 1.f / scales[1] : 0.f;
    for (int byte_j = start / 2 + tid; byte_j <= (end - 1) / 2; byte_j += blockDim.x) {
        const int j0 = byte_j * 2;
        const int j1 = j0 + 1;
        u8 k0 = 0;
        u8 k1 = 0;
        u8 v0 = 0;
        u8 v1 = 0;
        if (j0 < d) {
            k0 = quant_fp4_e2m1(bf16_to_f32(k_curr[src_base + j0]) * k_inv);
            v0 = quant_fp4_e2m1(bf16_to_f32(v_curr[src_base + j0]) * v_inv);
        }
        if (j1 < d) {
            k1 = quant_fp4_e2m1(bf16_to_f32(k_curr[src_base + j1]) * k_inv);
            v1 = quant_fp4_e2m1(bf16_to_f32(v_curr[src_base + j1]) * v_inv);
        }
        k_pages[packed_base + byte_j] = static_cast<u8>(k0 | (k1 << 4));
        v_pages[packed_base + byte_j] = static_cast<u8>(v0 | (v1 << 4));
    }
}

__global__ void dequant_fp8_pages_active(
    const __nv_fp8_storage_t* __restrict__ k_pages,
    const __nv_fp8_storage_t* __restrict__ v_pages,
    bf16*           __restrict__ k_out,
    bf16*           __restrict__ v_out,
    const u32*     __restrict__ page_indices,
    long long n,
    int page_elems,
    __nv_fp8_interpretation_t fp8_kind)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int active_page = static_cast<int>(i / page_elems);
    const int local = static_cast<int>(i % page_elems);
    const long long page = static_cast<long long>(page_indices[active_page]);
    const long long elem = page * page_elems + local;
    const __half kh = __nv_cvt_fp8_to_halfraw(k_pages[elem], fp8_kind);
    const __half vh = __nv_cvt_fp8_to_halfraw(v_pages[elem], fp8_kind);
    k_out[elem] = f32_to_bf16(__half2float(kh));
    v_out[elem] = f32_to_bf16(__half2float(vh));
}

template <class T>
__global__ void dequant_fp8_per_token_head_pages_active(
    const __nv_fp8_storage_t* __restrict__ k_pages,
    const __nv_fp8_storage_t* __restrict__ v_pages,
    const float*              __restrict__ k_scales,
    const float*              __restrict__ v_scales,
    T*            __restrict__ k_out,
    T*            __restrict__ v_out,
    const u32*      __restrict__ page_indices,
    long long n,
    int page_size,
    int h_kv,
    int d)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int page_elems = page_size * h_kv * d;
    const int active_page = static_cast<int>(i / page_elems);
    const int local = static_cast<int>(i % page_elems);
    const long long page = static_cast<long long>(page_indices[active_page]);
    const long long elem = page * page_elems + local;
    const int token_head = local / d;
    const long long scale_idx =
        (page * page_size * h_kv) + token_head;
    const __half kh = __nv_cvt_fp8_to_halfraw(k_pages[elem], __NV_E4M3);
    const __half vh = __nv_cvt_fp8_to_halfraw(v_pages[elem], __NV_E4M3);
    k_out[elem] = Elem<T>::from_f32(__half2float(kh) * k_scales[scale_idx]);
    v_out[elem] = Elem<T>::from_f32(__half2float(vh) * v_scales[scale_idx]);
}

template <class T>
__global__ void dequant_int8_per_token_head_pages_active(
    const i8* __restrict__ k_pages,
    const i8* __restrict__ v_pages,
    const float*       __restrict__ k_scales,
    const float*       __restrict__ v_scales,
    T*     __restrict__ k_out,
    T*     __restrict__ v_out,
    const u32* __restrict__ page_indices,
    long long n,
    int page_size,
    int h_kv,
    int d)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int page_elems = page_size * h_kv * d;
    const int active_page = static_cast<int>(i / page_elems);
    const int local = static_cast<int>(i % page_elems);
    const long long page = static_cast<long long>(page_indices[active_page]);
    const long long elem = page * page_elems + local;
    const int token_head = local / d;
    const long long scale_idx =
        (page * page_size * h_kv) + token_head;
    k_out[elem] = Elem<T>::from_f32(static_cast<float>(k_pages[elem]) * k_scales[scale_idx]);
    v_out[elem] = Elem<T>::from_f32(static_cast<float>(v_pages[elem]) * v_scales[scale_idx]);
}

template <class T>
__global__ void dequant_fp4_pages_active(
    const u8* __restrict__ k_pages,
    const u8* __restrict__ v_pages,
    const float*        __restrict__ k_scales,
    const float*        __restrict__ v_scales,
    T*      __restrict__ k_out,
    T*      __restrict__ v_out,
    const u32* __restrict__ page_indices,
    long long logical_n,
    int page_size,
    int h_kv,
    int d,
    int block_size)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= logical_n) return;
    const int logical_page_elems = page_size * h_kv * d;
    const int packed_d = (d + 1) / 2;
    const int packed_page_elems = page_size * h_kv * packed_d;
    const int blocks_per_head = (d + block_size - 1) / block_size;
    const int active_page = static_cast<int>(i / logical_page_elems);
    const int local = static_cast<int>(i % logical_page_elems);
    const long long page = static_cast<long long>(page_indices[active_page]);
    const int row = local / d;
    const int j = local % d;
    const long long packed_i =
        page * packed_page_elems + static_cast<long long>(row) * packed_d + j / 2;
    const int nibble_shift = (j & 1) ? 4 : 0;
    const u8 kc = (k_pages[packed_i] >> nibble_shift) & 0xf;
    const u8 vc = (v_pages[packed_i] >> nibble_shift) & 0xf;
    const long long scale_idx =
        (page * page_size * h_kv + row) * blocks_per_head + j / block_size;
    const long long out_i = page * logical_page_elems + local;
    k_out[out_i] = Elem<T>::from_f32(fp4_e2m1_value(kc) * k_scales[scale_idx]);
    v_out[out_i] = Elem<T>::from_f32(fp4_e2m1_value(vc) * v_scales[scale_idx]);
}

template <bool HND_LAYOUT>
__global__ void kv_append_explicit_devwin(
    const bf16* __restrict__ k_curr,
    const bf16* __restrict__ v_curr,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ devwin,
    int n_max,
    int page_size,
    int h_kv,
    int d)
{
    const int b = blockIdx.x;
    if (b >= n_max) return;
    // `devwin` is the pre-staged device window pair, `(start, count)`:
    // word 0 is a START. The staged-geometry seat's `win` is `(count,
    // start)` — same pointer shape, opposite word order, and the rename
    // is what keeps one from ever arming the other.
    const int w0 = static_cast<int>(devwin[0]);
    const int w1 = static_cast<int>(devwin[1]);
    if (b < w0 || b >= w0 + w1) return;
    if (row_valid != nullptr && row_valid[b] == 0) return;
    const int actual_page = static_cast<int>(w_page[b]);
    const int offset_in_page = static_cast<int>(w_off[b]);
    if (offset_in_page < 0 || offset_in_page >= page_size) return;

    const long long row = static_cast<long long>(h_kv) * d;
    const long long src = static_cast<long long>(b) * row;
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        long long dst;
        if constexpr (HND_LAYOUT) {
            const int h = i / d;
            const int j = i - h * d;
            dst = ((static_cast<long long>(actual_page) * h_kv + h) *
                   page_size + offset_in_page) * d + j;
        } else {
            dst = ((static_cast<long long>(actual_page) * page_size) +
                   offset_in_page) * row + i;
        }
        k_pages[dst] = k_curr[src + i];
        v_pages[dst] = v_curr[src + i];
    }
}

__global__ void build_window_page_view(
    const u32* __restrict__ src_indices,
    const u32* __restrict__ src_indptr,
    int keep_pages,
    u32* __restrict__ dst_indptr,
    u32* __restrict__ dst_indices,
    int R)
{
    if (threadIdx.x == 0) {
        u32 acc = 0;
        dst_indptr[0] = 0;
        for (int r = 0; r < R; ++r) {
            const u32 have = src_indptr[r + 1] - src_indptr[r];
            const u32 keep =
                have < static_cast<u32>(keep_pages)
                    ? have : static_cast<u32>(keep_pages);
            acc += keep;
            dst_indptr[r + 1] = acc;
        }
    }
    __syncthreads();
    for (int r = 0; r < R; ++r) {
        const u32 src_end = src_indptr[r + 1];
        const u32 dst_beg = dst_indptr[r];
        const u32 keep    = dst_indptr[r + 1] - dst_beg;

        for (u32 i = threadIdx.x; i < keep; i += blockDim.x) {
            dst_indices[dst_beg + i] = src_indices[src_end - keep + i];
        }
    }
}

__global__ void build_full_split_view(
    const u32* __restrict__ src_indptr,
    const u32* __restrict__ src_last_page_len,
    int splits,
    int page_size,
    u32* __restrict__ dst_indptr,
    u32* __restrict__ dst_indices,
    u32* __restrict__ dst_last,
    const u32* __restrict__ src_indices)
{
    if (threadIdx.x != 0) return;
    const u32 base = src_indptr[0];
    const int pages = static_cast<int>(src_indptr[1] - base);
    const u32 tail = src_last_page_len[0];
    u32 acc = 0;
    dst_indptr[0] = 0;
    for (int i = 0; i < splits; ++i) {

        const int lo = static_cast<int>(
            (static_cast<long long>(i) * pages) / splits);
        const int hi = static_cast<int>(
            (static_cast<long long>(i + 1) * pages) / splits);
        if (hi > lo) {
            for (int p = lo; p < hi; ++p) {
                dst_indices[acc + (p - lo)] = src_indices[base + p];
            }
            acc += static_cast<u32>(hi - lo);
            dst_last[i] = (hi == pages)
                ? tail : static_cast<u32>(page_size);
        } else {

            dst_indices[acc] = src_indices[base];
            acc += 1;
            dst_last[i] = 0;
        }
        dst_indptr[i + 1] = acc;
    }
}

}
