#include "kv_append_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// De-branded from driver/cuda/src/kernels/kv_paged.cu. The kernel bodies
// (`find_request`, `resolve_dst`, `write_kv_fp8_per_tensor_kernel`,
// `write_kv_per_token_head_kernel<UseFp8>`) are lifted verbatim apart from the
// namespace (`pie_cuda_driver::kernels` -> `pie_cuda_device::kernels`). The
// launcher replaces the `KvCacheLayerView`-based `launch_write_kv_to_pages`
// dispatch with a raw-pointer entry (`write_kv_to_pages_quant`) that takes the
// scheme via the `KvQuantScheme` tag. Fp4Block is intentionally NOT lifted here
// (the active selftest exercises int8 + fp8 per-token-head); the per-tensor and
// per-token-head write paths are the ones the new attention decode needs.

namespace pie_cuda_device::kernels {

namespace {

// Linear scan to find the request index — `R` is small (≤ batch_size).
__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

// Resolve (actual_page, offset_in_page) for a current-step token — verbatim
// from kv_paged.cu's `resolve_dst`.
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

// FP8 per-tensor: no side scales. Quantize each element directly with
// SATFINITE saturation. One block per token; threads stride the h_kv*d row.
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

    const long long row = static_cast<long long>(h_kv) * d;
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

// Int8 / FP8 per-(token,head). One block per (token, head). Block computes the
// per-row amax via warp + cross-warp reduction, derives scale = amax/qmax, and
// writes the quantized row plus the fp32 scale.
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

}  // namespace

void write_kv_to_pages_quant(
    void* k_pages, void* v_pages,
    float* k_scales, float* v_scales,
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
    KvQuantScheme scheme,
    KvQuantFp8Kind fp8_kind,
    cudaStream_t stream)
{
    if (num_requests <= 0 || total_tokens <= 0) return;
    constexpr int BLOCK = 256;
    switch (scheme) {
        case KvQuantScheme::Fp8PerTensor: {
            const auto kind = (fp8_kind == KvQuantFp8Kind::E5M2)
                ? __NV_E5M2 : __NV_E4M3;
            write_kv_fp8_per_tensor_kernel<<<total_tokens, BLOCK, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                static_cast<__nv_fp8_storage_t*>(k_pages),
                static_cast<__nv_fp8_storage_t*>(v_pages),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim, kind);
            break;
        }
        case KvQuantScheme::Int8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            write_kv_per_token_head_kernel<false><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                k_pages, v_pages, k_scales, v_scales,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvQuantScheme::Fp8PerTokenHead: {
            const dim3 grid(total_tokens, num_kv_heads);
            const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
            write_kv_per_token_head_kernel<true><<<grid, BLOCK, shmem, stream>>>(
                static_cast<const __nv_bfloat16*>(k_curr),
                static_cast<const __nv_bfloat16*>(v_curr),
                k_pages, v_pages, k_scales, v_scales,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_requests, page_size, num_kv_heads, head_dim);
            break;
        }
        case KvQuantScheme::Native:
        case KvQuantScheme::Fp4Block:
            // Native: caller should use write_kv_to_pages_bf16.
            // Fp4Block: not lifted in this TU.
            break;
    }
}

}  // namespace pie_cuda_device::kernels
