#include "kernels/kv_paged.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {

namespace {

// One block per current-step token. Threads stride over the (h_kv * head_dim)
// destination row.
//
// Linear scan to find the request index — `R` is small (≤ batch_size, which
// is bounded by max_forward_requests, ≤ a few hundred). A binary search would be
// nice, but the scan is fine in the M1.4 reference path.
__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

template <bool HND_LAYOUT>
__global__ void write_kv_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d)
{
    const int t = blockIdx.x;

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

__device__ __forceinline__ void resolve_dst(
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
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

__global__ void write_kv_fp8_per_tensor_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    __nv_fp8_storage_t*  __restrict__ k_pages,
    __nv_fp8_storage_t*  __restrict__ v_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d,
    __nv_fp8_interpretation_t fp8_kind)
{
    const int t = blockIdx.x;
    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(t) * row;
    const long long dst =
        ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * row;

    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        const float kf = __bfloat162float(k_curr[src + i]);
        const float vf = __bfloat162float(v_curr[src + i]);
        k_pages[dst + i] = __nv_cvt_float_to_fp8(kf, __NV_SATFINITE, fp8_kind);
        v_pages[dst + i] = __nv_cvt_float_to_fp8(vf, __NV_SATFINITE, fp8_kind);
    }
}

template <bool UseFp8>
__global__ void write_kv_per_token_head_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    void*                __restrict__ k_pages_raw,
    void*                __restrict__ v_pages_raw,
    float*               __restrict__ k_scales,
    float*               __restrict__ v_scales,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float shmem[];
    float* k_warp = shmem;
    float* v_warp = shmem + blockDim.x / 32;

    const long long src_base =
        (static_cast<long long>(t) * h_kv + h) * d;
    float k_abs = 0.f;
    float v_abs = 0.f;
    for (int j = tid; j < d; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(__bfloat162float(k_curr[src_base + j])));
        v_abs = fmaxf(v_abs, fabsf(__bfloat162float(v_curr[src_base + j])));
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
                __bfloat162float(k_curr[src_base + j]) * k_inv,
                __NV_SATFINITE, __NV_E4M3);
            v_pages[dst_base + j] = __nv_cvt_float_to_fp8(
                __bfloat162float(v_curr[src_base + j]) * v_inv,
                __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        auto* k_pages = static_cast<std::int8_t*>(k_pages_raw);
        auto* v_pages = static_cast<std::int8_t*>(v_pages_raw);
        for (int j = tid; j < d; j += blockDim.x) {
            int kq = static_cast<int>(rintf(__bfloat162float(k_curr[src_base + j]) * k_inv));
            int vq = static_cast<int>(rintf(__bfloat162float(v_curr[src_base + j]) * v_inv));
            kq = kq > 127 ? 127 : (kq < -128 ? -128 : kq);
            vq = vq > 127 ? 127 : (vq < -128 ? -128 : vq);
            k_pages[dst_base + j] = static_cast<std::int8_t>(kq);
            v_pages[dst_base + j] = static_cast<std::int8_t>(vq);
        }
    }
}

__device__ __forceinline__ float fp4_e2m1_value(std::uint8_t code) {
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

__device__ __forceinline__ std::uint8_t quant_fp4_e2m1(float x) {
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
    return static_cast<std::uint8_t>((neg ? 0x8 : 0) | best);
}

__global__ void write_kv_fp4_block_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    std::uint8_t*        __restrict__ k_pages,
    std::uint8_t*        __restrict__ v_pages,
    float*               __restrict__ k_scales,
    float*               __restrict__ v_scales,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d,
    int block_size)
{
    const int t = blockIdx.x;
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
        (static_cast<long long>(t) * h_kv + h) * d;
    float k_abs = 0.f;
    float v_abs = 0.f;
    for (int j = start + tid; j < end; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(__bfloat162float(k_curr[src_base + j])));
        v_abs = fmaxf(v_abs, fabsf(__bfloat162float(v_curr[src_base + j])));
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
        std::uint8_t k0 = 0;
        std::uint8_t k1 = 0;
        std::uint8_t v0 = 0;
        std::uint8_t v1 = 0;
        if (j0 < d) {
            k0 = quant_fp4_e2m1(__bfloat162float(k_curr[src_base + j0]) * k_inv);
            v0 = quant_fp4_e2m1(__bfloat162float(v_curr[src_base + j0]) * v_inv);
        }
        if (j1 < d) {
            k1 = quant_fp4_e2m1(__bfloat162float(k_curr[src_base + j1]) * k_inv);
            v1 = quant_fp4_e2m1(__bfloat162float(v_curr[src_base + j1]) * v_inv);
        }
        k_pages[packed_base + byte_j] = static_cast<std::uint8_t>(k0 | (k1 << 4));
        v_pages[packed_base + byte_j] = static_cast<std::uint8_t>(v0 | (v1 << 4));
    }
}

__global__ void dequant_fp8_pages_active_kernel(
    const __nv_fp8_storage_t* __restrict__ k_pages,
    const __nv_fp8_storage_t* __restrict__ v_pages,
    __nv_bfloat16*           __restrict__ k_out,
    __nv_bfloat16*           __restrict__ v_out,
    const std::uint32_t*     __restrict__ page_indices,
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
    k_out[elem] = __float2bfloat16(__half2float(kh));
    v_out[elem] = __float2bfloat16(__half2float(vh));
}

__global__ void dequant_fp8_per_token_head_pages_active_kernel(
    const __nv_fp8_storage_t* __restrict__ k_pages,
    const __nv_fp8_storage_t* __restrict__ v_pages,
    const float*              __restrict__ k_scales,
    const float*              __restrict__ v_scales,
    __nv_bfloat16*            __restrict__ k_out,
    __nv_bfloat16*            __restrict__ v_out,
    const std::uint32_t*      __restrict__ page_indices,
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
    k_out[elem] = __float2bfloat16(__half2float(kh) * k_scales[scale_idx]);
    v_out[elem] = __float2bfloat16(__half2float(vh) * v_scales[scale_idx]);
}

__global__ void dequant_int8_per_token_head_pages_active_kernel(
    const std::int8_t* __restrict__ k_pages,
    const std::int8_t* __restrict__ v_pages,
    const float*       __restrict__ k_scales,
    const float*       __restrict__ v_scales,
    __nv_bfloat16*     __restrict__ k_out,
    __nv_bfloat16*     __restrict__ v_out,
    const std::uint32_t* __restrict__ page_indices,
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
    k_out[elem] = __float2bfloat16(static_cast<float>(k_pages[elem]) * k_scales[scale_idx]);
    v_out[elem] = __float2bfloat16(static_cast<float>(v_pages[elem]) * v_scales[scale_idx]);
}

__global__ void dequant_fp4_pages_active_kernel(
    const std::uint8_t* __restrict__ k_pages,
    const std::uint8_t* __restrict__ v_pages,
    const float*        __restrict__ k_scales,
    const float*        __restrict__ v_scales,
    __nv_bfloat16*      __restrict__ k_out,
    __nv_bfloat16*      __restrict__ v_out,
    const std::uint32_t* __restrict__ page_indices,
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
    const std::uint8_t kc = (k_pages[packed_i] >> nibble_shift) & 0xf;
    const std::uint8_t vc = (v_pages[packed_i] >> nibble_shift) & 0xf;
    const long long scale_idx =
        (page * page_size * h_kv + row) * blocks_per_head + j / block_size;
    const long long out_i = page * logical_page_elems + local;
    k_out[out_i] = __float2bfloat16(fp4_e2m1_value(kc) * k_scales[scale_idx]);
    v_out[out_i] = __float2bfloat16(fp4_e2m1_value(vc) * v_scales[scale_idx]);
}

}  // namespace

void launch_write_kv_to_pages_bf16(
    void* k_pages, void* v_pages,
    const void* k_curr, const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    if (hnd_layout) {
        write_kv_kernel<true><<<total_tokens, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k_curr),
            static_cast<const __nv_bfloat16*>(v_curr),
            static_cast<__nv_bfloat16*>(k_pages),
            static_cast<__nv_bfloat16*>(v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            num_requests, page_size, num_kv_heads, head_dim);
    } else {
        write_kv_kernel<false><<<total_tokens, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k_curr),
            static_cast<const __nv_bfloat16*>(v_curr),
            static_cast<__nv_bfloat16*>(k_pages),
            static_cast<__nv_bfloat16*>(v_pages),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            num_requests, page_size, num_kv_heads, head_dim);
    }
}

void launch_write_kv_to_pages(
    KvCacheLayerView layer,
    const void* k_curr,
    const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream)
{
    const int page_size = layer.page_size;
    const int num_kv_heads = layer.num_kv_heads;
    const int head_dim = layer.head_dim;
    if (layer.is_native_bf16()) {
        launch_write_kv_to_pages_bf16(
            layer.k_pages, layer.v_pages, k_curr, v_curr,
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            total_tokens, num_requests, page_size, num_kv_heads, head_dim,
            layer.hnd_layout, stream);
        return;
    }

    constexpr int BLOCK = 256;
    switch (layer.format->scheme) {
        case KvCacheScheme::Fp8PerTensor: {
            const auto fp8_kind = layer.format->storage_dtype == DType::FP8_E5M2
                ? __NV_E5M2
                : __NV_E4M3;
            write_kv_fp8_per_tensor_kernel<<<total_tokens, BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                static_cast<__nv_fp8_storage_t*>(layer.k_pages),
                static_cast<__nv_fp8_storage_t*>(layer.v_pages),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim, fp8_kind);
            break;
        }
        case KvCacheScheme::Int8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            write_kv_per_token_head_kernel<false><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                layer.k_pages, layer.v_pages,
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvCacheScheme::Fp8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            write_kv_per_token_head_kernel<true><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                layer.k_pages, layer.v_pages,
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvCacheScheme::Fp4Block: {
            const int block_size = layer.format->block_size > 0
                ? layer.format->block_size
                : 16;
            const int blocks = (head_dim + block_size - 1) / block_size;
            const dim3 grid(total_tokens, num_kv_heads, blocks);
            write_kv_fp4_block_kernel<<<grid, 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                static_cast<std::uint8_t*>(layer.k_pages),
                static_cast<std::uint8_t*>(layer.v_pages),
                static_cast<float*>(layer.k_scales),
                static_cast<float*>(layer.v_scales),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim, block_size);
            break;
        }
        case KvCacheScheme::Native:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream)
{
    if (layer.is_native_bf16() || num_pages_in_batch <= 0) return;
    constexpr int BLOCK = 256;
    const int page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
    const long long logical_n =
        static_cast<long long>(num_pages_in_batch) * page_elems;
    const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);

    switch (layer.format->scheme) {
        case KvCacheScheme::Fp8PerTensor: {
            const auto fp8_kind = layer.format->storage_dtype == DType::FP8_E5M2
                ? __NV_E5M2
                : __NV_E4M3;
            dequant_fp8_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<__nv_bfloat16*>(layer.k_bf16_pages),
                static_cast<__nv_bfloat16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, page_elems, fp8_kind);
            break;
        }
        case KvCacheScheme::Fp8PerTokenHead:
            dequant_fp8_per_token_head_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<__nv_bfloat16*>(layer.k_bf16_pages),
                static_cast<__nv_bfloat16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Int8PerTokenHead:
            dequant_int8_per_token_head_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::int8_t*>(layer.k_pages),
                static_cast<const std::int8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<__nv_bfloat16*>(layer.k_bf16_pages),
                static_cast<__nv_bfloat16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Fp4Block: {
            const int block_size = layer.format->block_size > 0
                ? layer.format->block_size
                : 16;
            dequant_fp4_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::uint8_t*>(layer.k_pages),
                static_cast<const std::uint8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<__nv_bfloat16*>(layer.k_bf16_pages),
                static_cast<__nv_bfloat16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim, block_size);
            break;
        }
        case KvCacheScheme::Native:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels
