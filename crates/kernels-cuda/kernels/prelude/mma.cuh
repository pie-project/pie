#pragma once

#include "prelude/device.cuh"

using __nv_bfloat16 = ::pie::bf16;

namespace nvcuda {
namespace wmma {

struct matrix_a;
struct matrix_b;
struct accumulator;

struct row_major;
struct col_major;

enum layout_t { mem_row_major, mem_col_major, mem_undefined };

namespace detail {

template <typename...>
struct dependent_false {
    static constexpr bool value = false;
};

__device__ __forceinline__ int lane_group() { return static_cast<int>(threadIdx.x % 32u) >> 2; }

__device__ __forceinline__ int lane_in_group() { return static_cast<int>(threadIdx.x % 32u) & 3; }

__device__ __forceinline__ unsigned pack(pie::bf16 low, pie::bf16 high) {
    return (static_cast<unsigned>(high.raw) << 16) | static_cast<unsigned>(low.raw);
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4],
    const unsigned (&a)[4],
    const unsigned (&b)[2],
    const float (&c)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else

    (void)a;
    (void)b;
    d[0] = c[0];
    d[1] = c[1];
    d[2] = c[2];
    d[3] = c[3];
    __trap();
#endif
}

}

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    static_assert(
        detail::dependent_false<Use, T>::value,
        "pie_mma.cuh implements exactly one WMMA instantiation set, because that is "
        "what this tree instantiates: fragment<matrix_a, 16,16,16, __nv_bfloat16, "
        "row_major>, fragment<matrix_b, 16,16,16, __nv_bfloat16, col_major> and "
        "fragment<accumulator, 16,16,16, float>. See moe/moe_dispatch.cu:547 and "
        "moe/moe_grouped_gemm.cu:69 for the two callers, and this file's header for "
        "why a shape is hand-written rather than resolved out of <mma.h>. A new "
        "shape needs a new lane map from PTX ISA 9.7.15.5 and a parity run against "
        "nvcuda::wmma on a device -- an untested lane map compiles and answers wrong.");
};

template <>
struct fragment<matrix_a, 16, 16, 16, ::pie::bf16, row_major> {

    unsigned reg[4];
};

template <>
struct fragment<matrix_b, 16, 16, 16, ::pie::bf16, col_major> {
    unsigned reg[4];
};

template <>
struct fragment<accumulator, 16, 16, 16, float, void> {
    float reg[8];
};

__device__ __forceinline__ void fill_fragment(
    fragment<accumulator, 16, 16, 16, float>& frag,
    float value)
{
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        frag.reg[i] = value;
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, 16, 16, 16, ::pie::bf16, row_major>& frag,
    const ::pie::bf16* ptr,
    unsigned ldm)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {

        const int col = 2 * t + 8 * half;
        const int top = g * stride + col;
        const int bottom = (g + 8) * stride + col;
        frag.reg[2 * half + 0] = detail::pack(ptr[top], ptr[top + 1]);
        frag.reg[2 * half + 1] = detail::pack(ptr[bottom], ptr[bottom + 1]);
    }
}

__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, 16, 16, 16, ::pie::bf16, col_major>& frag,
    const ::pie::bf16* ptr,
    unsigned ldm)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {

        const int base = (8 * half + g) * stride + 2 * t;
        frag.reg[2 * half + 0] = detail::pack(ptr[base], ptr[base + 1]);
        frag.reg[2 * half + 1] = detail::pack(ptr[base + 8], ptr[base + 9]);
    }
}

__device__ __forceinline__ void mma_sync(
    fragment<accumulator, 16, 16, 16, float>& d,
    const fragment<matrix_a, 16, 16, 16, ::pie::bf16, row_major>& a,
    const fragment<matrix_b, 16, 16, 16, ::pie::bf16, col_major>& b,
    const fragment<accumulator, 16, 16, 16, float>& c)
{
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const unsigned b_half[2] = {b.reg[2 * half + 0], b.reg[2 * half + 1]};
        const float c_half[4] = {
            c.reg[4 * half + 0],
            c.reg[4 * half + 1],
            c.reg[4 * half + 2],
            c.reg[4 * half + 3],
        };
        float d_half[4];
        detail::mma_m16n8k16(d_half, a.reg, b_half, c_half);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            d.reg[4 * half + i] = d_half[i];
        }
    }
}

__device__ __forceinline__ void store_matrix_sync(
    float* ptr,
    const fragment<accumulator, 16, 16, 16, float>& frag,
    unsigned ldm,
    layout_t layout)
{
    const int g = detail::lane_group();
    const int t = detail::lane_in_group();
    const int stride = static_cast<int>(ldm);

#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = g + 8 * (i >> 1);
            const int col = 8 * half + 2 * t + (i & 1);

            const int at = (layout == mem_col_major) ? (col * stride + row) : (row * stride + col);
            ptr[at] = frag.reg[4 * half + i];
        }
    }
}

}
}
