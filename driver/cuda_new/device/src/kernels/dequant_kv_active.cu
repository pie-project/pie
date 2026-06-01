#include "dequant_kv_active.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// De-branded from driver/cuda/src/kernels/kv_paged.cu. Kernel bodies lifted
// verbatim apart from the namespace; the launcher replaces the
// `KvCacheLayerView` dispatch with a raw-pointer entry. Fp4Block is not lifted.

namespace pie_cuda_device::kernels {

namespace {

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

}  // namespace

void dequant_kv_layer_to_bf16_active(
    const void* k_pages,
    const void* v_pages,
    const float* k_scales,
    const float* v_scales,
    void* k_bf16_pages,
    void* v_bf16_pages,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    int page_size,
    int num_kv_heads,
    int head_dim,
    KvQuantScheme scheme,
    KvQuantFp8Kind fp8_kind,
    cudaStream_t stream)
{
    if (num_pages_in_batch <= 0 || scheme == KvQuantScheme::Native) return;
    constexpr int BLOCK = 256;
    const int page_elems = page_size * num_kv_heads * head_dim;
    const long long logical_n =
        static_cast<long long>(num_pages_in_batch) * page_elems;
    const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);

    switch (scheme) {
        case KvQuantScheme::Fp8PerTensor: {
            const auto kind = (fp8_kind == KvQuantFp8Kind::E5M2)
                ? __NV_E5M2 : __NV_E4M3;
            dequant_fp8_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(k_pages),
                static_cast<const __nv_fp8_storage_t*>(v_pages),
                static_cast<__nv_bfloat16*>(k_bf16_pages),
                static_cast<__nv_bfloat16*>(v_bf16_pages),
                kv_page_indices, logical_n, page_elems, kind);
            break;
        }
        case KvQuantScheme::Fp8PerTokenHead:
            dequant_fp8_per_token_head_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(k_pages),
                static_cast<const __nv_fp8_storage_t*>(v_pages),
                k_scales, v_scales,
                static_cast<__nv_bfloat16*>(k_bf16_pages),
                static_cast<__nv_bfloat16*>(v_bf16_pages),
                kv_page_indices, logical_n, page_size, num_kv_heads, head_dim);
            break;
        case KvQuantScheme::Int8PerTokenHead:
            dequant_int8_per_token_head_pages_active_kernel<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::int8_t*>(k_pages),
                static_cast<const std::int8_t*>(v_pages),
                k_scales, v_scales,
                static_cast<__nv_bfloat16*>(k_bf16_pages),
                static_cast<__nv_bfloat16*>(v_bf16_pages),
                kv_page_indices, logical_n, page_size, num_kv_heads, head_dim);
            break;
        case KvQuantScheme::Native:
        case KvQuantScheme::Fp4Block:
            break;
    }
}

}  // namespace pie_cuda_device::kernels
