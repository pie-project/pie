#pragma once

namespace pie {

// The shared-window address of a generic pointer into shared memory: what
// `ldmatrix`, `cp.async` and the mbarrier instructions name.
__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// `ldmatrix` of 16-bit elements (sm_75 up): each lane names one eight-element
// row of shared memory at `row` (lanes 0-7 the first matrix, 8-15 the second,
// and so on), and register `i` of each lane lands its share of matrix `i`.
// The `_trans` forms land the matrices transposed.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(row)));
}

__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(row)));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_addr(row)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(unsigned (&r)[2], const void* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_addr(row)));
}

}  // namespace pie
