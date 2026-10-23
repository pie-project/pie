#pragma once

#include "prim/smem.cuh"

namespace pie {

// `d += a b` over one 16x8x16 tile on the tensor cores (sm_80 up): bf16
// operands in the mma.sync fragment layout, f32 accumulators.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const unsigned (&a)[4],
                                             const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The same over f16 operands.
__device__ __forceinline__ void mma_m16n8k16_f16(float (&d)[4], const unsigned (&a)[4],
                                                 const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The fragments `mma_m16n8k16` reads, loaded by `ldmatrix` from a row-major
// 16-bit tile in shared memory whose rows are `ld` elements apart.

// A: 16 rows by 16 columns from `tile`.
template <typename T>
__device__ __forceinline__ void ld_a(unsigned (&a)[4], const T* tile, int ld, int lane) {
    ldmatrix_x4(a, tile + static_cast<long long>(lane & 15) * ld + ((lane & 16) ? 8 : 0));
}

// B of `x tileᵀ`: 8 rows (the n) by 16 columns (the k) from `tile`.
template <typename T>
__device__ __forceinline__ void ld_b(unsigned (&b)[2], const T* tile, int ld, int lane) {
    ldmatrix_x2(b, tile + static_cast<long long>(lane & 7) * ld + ((lane & 8) ? 8 : 0));
}

// B of `x tile`: 16 rows (the k) by 8 columns (the n) from `tile`, transposed.
template <typename T>
__device__ __forceinline__ void ld_b_trans(unsigned (&b)[2], const T* tile, int ld, int lane) {
    ldmatrix_x2_trans(b, tile + static_cast<long long>(lane & 15) * ld);
}

}  // namespace pie
