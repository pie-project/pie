#pragma once

#include "prim/device.cuh"
#include <cuda_fp8.h>

namespace pie {

// Where a result's row lands, the row skeleton's and the contraction
// skeletons' alike (`kernels_cuda::elemwise::skeleton::Store` mirrors it):
// the row's columns `from` up to `to` (every one where `to` is 0), at its
// plane's row from the row's first element, or `head` of every `stride` of
// them landed one head after another; or, where `page` is set, at the paged
// cache row `page[r] * page_size + slot[r]` (none where the slot lies past
// the page or `valid[r]` is 0). Its elements are bf16, or e4m3 where `e4m3`,
// each `e4m3(x / scale)`.
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
    int unused;
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

// The cache row row `r` lands at, or -1 where it lands none.
__device__ __forceinline__ long long store_cell(const Store& s, long long r) {
    const int page = s.page[r];
    const int slot = s.slot[r];
    if (slot < 0 || slot >= s.page_size || page < 0 || (s.valid != nullptr && s.valid[r] == 0)) return -1;
    return static_cast<long long>(page) * s.page_size + slot;
}

// Eight bf16 (one vector) to `plane`'s element `at` and on, as the store
// holds its elements.
__device__ __forceinline__ void store8(void* plane, long long at, const uint4& v, const Store& s) {
    if (s.e4m3 != 0) {
        const bf16* held = reinterpret_cast<const bf16*>(&v);
        __align__(8) __nv_fp8_e4m3 q[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) q[e] = __nv_fp8_e4m3(bf16_to_f32(held[e]) / s.scale);
        *reinterpret_cast<uint2*>(static_cast<unsigned char*>(plane) + at) = *reinterpret_cast<const uint2*>(q);
    } else {
        *reinterpret_cast<uint4*>(static_cast<bf16*>(plane) + at) = v;
    }
}

// One bf16 to `plane`'s element `at`, as the store holds its elements.
__device__ __forceinline__ void store1(void* plane, long long at, bf16 v, const Store& s) {
    if (s.e4m3 != 0) {
        static_cast<__nv_fp8_e4m3*>(plane)[at] = __nv_fp8_e4m3(bf16_to_f32(v) / s.scale);
    } else {
        static_cast<bf16*>(plane)[at] = v;
    }
}

}
