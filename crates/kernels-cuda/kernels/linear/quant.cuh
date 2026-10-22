#pragma once

#include "prim/device.cuh"

#ifdef __CUDACC_RTC__
#include "prim/half2.cuh"
#include "prim/fp8.cuh"
#include "prim/mma_sm80.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie::linear {

__device__ __constant__ float kFp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

__device__ __forceinline__ void mxfp4_unpack8(unsigned word, __half2 out[4]) {
    constexpr unsigned kMagHi01234567 = 0x3E3C3800u;
    constexpr unsigned kMagHi4567     = 0x46444240u;
    constexpr unsigned kSignBytes     = 0x80808080u;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned sel = (word >> (half * 16)) & 0xFFFFu;
        const unsigned mag =
            __byte_perm(kMagHi01234567, kMagHi4567, sel & 0x7777u);
        const unsigned sgn =
            __byte_perm(0u, kSignBytes, (sel & 0x8888u) >> 1);
        const unsigned hi = mag | sgn;
        const unsigned a = __byte_perm(hi, 0u, 0x1404u);
        const unsigned b = __byte_perm(hi, 0u, 0x3424u);
        out[half * 2 + 0] = *reinterpret_cast<const __half2*>(&a);
        out[half * 2 + 1] = *reinterpret_cast<const __half2*>(&b);
    }
}

struct alignas(16) MoeGroupBases {
    const u8* codes;
    const u8* scales;

    const u8* biases;
    const u8* pad;
};

__device__ __forceinline__ void moe_note_group(
    unsigned int* __restrict__ group_hits)
{
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) {
        atomicAdd(group_hits, 1u);
    }
}

// Eight e2m1 codes of a word (code `i` its nibble `i`) dotted with `xv`
// onto `acc`, the codes laid into bf16 by integer ops: each 2^-126 times its
// value, exact, so `xv` comes 2^100 up and the caller's scale takes back the
// 2^26 left. No f16 converts.
__device__ __forceinline__ float e2m1x8_dot_lean(unsigned wd, const float (&xv)[8], float acc) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        // Codes 3 - p and 7 - p at bits 12..15 and 28..31.
        const unsigned t = wd << (4 * p);
        const unsigned b2 = (t & 0x80008000u) | ((t >> 6) & 0x01C001C0u);
        acc = fmaf(__uint_as_float(b2 << 16), xv[3 - p], acc);
        acc = fmaf(__uint_as_float(b2 & 0xFFFF0000u), xv[7 - p], acc);
    }
    return acc;
}

__device__ __forceinline__ void pie_ldmatrix_x4(unsigned (&r)[4], const void* row) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at));
}

__device__ __forceinline__ uint4 pie_ld_line(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void pie_mma_bf16_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

using f32 = float;

}
