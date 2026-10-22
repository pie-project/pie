#pragma once

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

}  // namespace pie
