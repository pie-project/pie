#include "prim/device.cuh"
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace pie::attn {

// The pages a fire reads of an e4m3 pool, widened to bf16 in its page
// list's order (the list's i-th page at the shadow's page i), and the
// shadow's own list: 16 codes a thread.
__global__ void widen_e4m3_pages(
    const uint4* __restrict__ k_pages,
    const uint4* __restrict__ v_pages,
    uint4* __restrict__ k_out,
    uint4* __restrict__ v_out,
    const u32* __restrict__ page_indices,
    i32* __restrict__ shadow_indices,
    long long n,
    int page_chunks)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long active = i / page_chunks;
    const int local = static_cast<int>(i % page_chunks);
    if (local == 0) shadow_indices[active] = static_cast<i32>(active);
    const long long src = static_cast<long long>(page_indices[active]) * page_chunks + local;
    const uint4 kv[2] = {k_pages[src], v_pages[src]};
    uint4* out[2] = {k_out, v_out};
#pragma unroll
    for (int x = 0; x < 2; ++x) {
        const unsigned w[4] = {kv[x].x, kv[x].y, kv[x].z, kv[x].w};
        unsigned b[8];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            b[2 * u] = pack_bf16x2(e4m3_to_f32(w[u] & 0xff), e4m3_to_f32((w[u] >> 8) & 0xff));
            b[2 * u + 1] = pack_bf16x2(e4m3_to_f32((w[u] >> 16) & 0xff), e4m3_to_f32(w[u] >> 24));
        }
        out[x][2 * i] = make_uint4(b[0], b[1], b[2], b[3]);
        out[x][2 * i + 1] = make_uint4(b[4], b[5], b[6], b[7]);
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

}
