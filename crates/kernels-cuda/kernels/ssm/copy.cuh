#pragma once

#include "ssm/state.cuh"

namespace pie::attn {

__device__ __forceinline__ void gdn_load8(
    const __nv_bfloat16* __restrict__ p, float (&s)[8]) {
    if ((reinterpret_cast<usize>(p) & 15) == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = __bfloat1622float2(pairs[u]);
            s[2 * u] = f.x;
            s[2 * u + 1] = f.y;
        }
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) s[u] = __bfloat162float(p[u]);
    }
}

__device__ __forceinline__ void gdn_store8(
    __nv_bfloat16* __restrict__ p, const float (&s)[8]) {
    if ((reinterpret_cast<usize>(p) & 15) == 0) {
        uint4 raw;
        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            pairs[u] = __floats2bfloat162_rn(s[2 * u], s[2 * u + 1]);
        }
        *reinterpret_cast<uint4*>(p) = raw;
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) p[u] = __float2bfloat16(s[u]);
    }
}

// A bf16 pair packed from two floats (round to nearest even) and unpacked.
__device__ __forceinline__ unsigned gdn_pack(float lo, float hi) {
    unsigned bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return bits;
}

__device__ __forceinline__ float2 gdn_unpack(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

// 16- and 4-byte `cp.async` copies that zero-fill what they do not read, and
// their group fences.
__device__ __forceinline__ void gdn_cp16(void* dst, const void* src, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(live ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void gdn_cp4(void* dst, const void* src, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(live ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ void gdn_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void gdn_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// A warp's column sum: the lanes that share a column half hold disjoint key
// rows of it, so a butterfly over the other four lane bits folds the sum
// into every one of them.
__device__ __forceinline__ void gdn_rows_sum(float (&x)[8]) {
    #pragma unroll
    for (int off = 2; off < 32; off <<= 1) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) x[u] += __shfl_xor_sync(0xffffffffu, x[u], off);
    }
}

// `N` consecutive floats a lane loads and stores as one access.
template <int N>
struct Packed {
    __device__ __forceinline__ static void load(float (&x)[N], const float* p) {
        if constexpr (N == 4) {
            const float4 v = __ldcs(reinterpret_cast<const float4*>(p));
            x[0] = v.x; x[1] = v.y; x[2] = v.z; x[3] = v.w;
        } else if constexpr (N == 2) {
            const float2 v = *reinterpret_cast<const float2*>(p);
            x[0] = v.x; x[1] = v.y;
        } else {
            x[0] = *p;
        }
    }
    __device__ __forceinline__ static void store(float* p, const float (&x)[N]) {
        if constexpr (N == 4) {
            __stcs(reinterpret_cast<float4*>(p), make_float4(x[0], x[1], x[2], x[3]));
        } else if constexpr (N == 2) {
            *reinterpret_cast<float2*>(p) = make_float2(x[0], x[1]);
        } else {
            *p = x[0];
        }
    }
};

}
