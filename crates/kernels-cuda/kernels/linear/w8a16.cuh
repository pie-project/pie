#pragma once

#include "prim/device.cuh"
#include "prim/mma_sm80.cuh"
#include "prim/smem.cuh"

#include <cuda_bf16.h>

namespace pie::linear {

namespace w8a16 {

// The byte of scale `(row, col)` in an atom-layout tensor of `cols` columns
// (`linear/act_quant.cuh`'s, which this file cannot include: its converts
// are sm_89's).
__device__ __forceinline__ long long atom_offset(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return (static_cast<long long>(row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16
        + ((row / 32) % 4) * 4 + col % 4;
}

__device__ __forceinline__ void cp_async16(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(bytes));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int kPending>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void group_sync(int id, int threads) {
    if (threads == 32) {
        __syncwarp();
    } else {
        asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads));
    }
}

__device__ __forceinline__ uint4 ld_nc(const u8* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// A block exponent `e` (ue8m0) as the bf16x2 factor a widened code takes:
// 2^(e - 127) for the block, 2^120 for the e4m3-to-bf16 exponent bias.
__device__ __forceinline__ unsigned factor2(unsigned e) {
    const unsigned f = min(e + 120u, 254u) << 7;
    return f | (f << 16);
}

// Two of a word's codes as a bf16x2, scaled by `f2`: `sel` lays each code
// in a half's low byte with its sign replicated through the high one
// (0x9180 codes 0 and 1, 0xB3A2 codes 2 and 3), so a shift of four puts the
// sign at bit 15 (the copies above the code fall to the mask), the exponent
// and mantissa at bf16's places.
__device__ __forceinline__ unsigned widen2(unsigned w, unsigned sel, unsigned f2) {
    unsigned p;
    // PTX's prmt replicates a byte's sign where its selector nibble's msb is
    // set (`__byte_perm` reads three bits of it).
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(p) : "r"(w), "r"(sel));
    const unsigned r = (p << 4) & 0x87F087F0u;
    __nv_bfloat162 a, b;
    a.x.raw = static_cast<unsigned short>(r & 0xffffu);
    a.y.raw = static_cast<unsigned short>(r >> 16);
    b.x.raw = static_cast<unsigned short>(f2 & 0xffffu);
    b.y.raw = static_cast<unsigned short>(f2 >> 16);
    const __nv_bfloat162 c = __hmul2(a, b);
    return (static_cast<unsigned>(c.y.raw) << 16) | static_cast<unsigned>(c.x.raw);
}

__device__ __forceinline__ unsigned word_of(const uint4& v, int c) {
    return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// Two of a 16-bit run's four codes (`half` 0, codes 0 and 1; 1, codes 2 and
// 3) as a bf16x2 of `s * q + b`: a code laid in a half's mantissa under
// 0x4300 reads 128 + q exactly, 128 comes off exactly, and one fused
// multiply-add applies the group's scale and bias.
__device__ __forceinline__ unsigned affine2(unsigned run, int half, unsigned s2, unsigned b2) {
    const unsigned p = run >> (8 * half);
    const unsigned h = ((p & 0xFu) | ((p & 0xF0u) << 12)) | 0x43004300u;
    unsigned q, w;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(q) : "r"(h), "r"(0x3F803F80u), "r"(0xC300C300u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(w) : "r"(q), "r"(s2), "r"(b2));
    return w;
}

}

}
