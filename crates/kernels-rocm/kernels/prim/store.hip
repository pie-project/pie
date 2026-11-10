#pragma once

#include "prim/device.hip"

namespace pie {

// Where a result's row lands, the row skeleton's (and, later, the
// contraction skeletons') alike (`kernels_rocm::elemwise::skeleton::Store`
// mirrors it): the row's columns `from` up to `to` (every one where `to` is
// 0), at its plane's row from the row's first element, or `head` of every
// `stride` of them landed one head after another; or, where `page` is set,
// at the paged cache row `slot[r]` of page `page[r]` (none where the slot
// lies past the page or `valid[r]` is 0): a page's rows start `page_stride`
// of the plane's elements after the page before's, or, where that is 0,
// right after its `page_size` rows. Its elements are bf16, or OCP e4m3fn
// codes where `e4m3`, each `e4m3(x / scale)`.
struct Store {
    const int* page;
    const int* slot;
    const unsigned char* valid;
    int from;
    int to;
    int page_size;
    int e4m3;
    float scale;
    int head;
    int stride;
    int page_stride;
};

// The landed column of the row's column `c`, or -1 past the window (`to`
// the window's end).
__device__ __forceinline__ long long store_at(const Store& s, long long c, long long to) {
    if (c < s.from || c >= to) return -1;
    const long long at = c - s.from;
    if (s.head == 0) return at;
    const long long j = at % s.stride;
    return j < s.head ? at / s.stride * s.head + j : -1;
}

// The plane element the cache row row `r` lands at starts on, its plane's
// rows `row_stride` elements apart, or -1 where it lands none.
__device__ __forceinline__ long long store_row(const Store& s, long long r, long long row_stride) {
    const int page = s.page[r];
    const int slot = s.slot[r];
    if (slot < 0 || slot >= s.page_size || page < 0 || (s.valid != nullptr && s.valid[r] == 0)) return -1;
    const long long pages = s.page_stride != 0 ? static_cast<long long>(s.page_stride)
                                               : static_cast<long long>(s.page_size) * row_stride;
    return static_cast<long long>(page) * pages + slot * row_stride;
}

// The OCP e4m3fn code nearest `q`, ties to even, saturated to ±448 (as
// CUDA's `__nv_fp8_e4m3` rounds a value, `satfinite`); zero is 0x00 whatever
// its sign. gfx942's own conversion is fnuz, so the code is made here.
__device__ __forceinline__ u8 e4m3fn_code(float q) {
    const float a = fminf(fabsf(q), 448.f);
    u32 bits;
    if (a < 0.015625f) {
        bits = static_cast<u32>(rintf(a * 512.f));
    } else {
        const int e = static_cast<int>((__float_as_uint(a) >> 23) & 0xFFu) - 127;
        // a / 2^(e-3), an exact product: 8 .. 16
        const u32 mant = static_cast<u32>(rintf(a * __uint_as_float(static_cast<u32>(127 + 3 - e) << 23)));
        bits = mant == 16u ? static_cast<u32>(e + 8) << 3 : (static_cast<u32>(e + 7) << 3) | (mant - 8u);
    }
    return static_cast<u8>(bits != 0u && q < 0.f ? bits | 0x80u : bits);
}

// Eight bf16 (one vector) to `plane`'s element `at` and on, as the store
// holds its elements.
__device__ __forceinline__ void store8(void* plane, long long at, const uint4& v, const Store& s) {
    if (s.e4m3 != 0) {
        const bf16* held = reinterpret_cast<const bf16*>(&v);
        u64 q = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) q |= static_cast<u64>(e4m3fn_code(to_f32(held[e]) / s.scale)) << (8 * e);
        *reinterpret_cast<u64*>(static_cast<unsigned char*>(plane) + at) = q;
    } else {
        *reinterpret_cast<uint4*>(static_cast<bf16*>(plane) + at) = v;
    }
}

// One bf16 to `plane`'s element `at`, as the store holds its elements.
__device__ __forceinline__ void store1(void* plane, long long at, bf16 v, const Store& s) {
    if (s.e4m3 != 0) {
        static_cast<u8*>(plane)[at] = e4m3fn_code(to_f32(v) / s.scale);
    } else {
        static_cast<bf16*>(plane)[at] = v;
    }
}

}  // namespace pie
