#pragma once

// CUDA's async global→shared copy has no AMD counterpart a unit can name
// (gfx94x's buffer→LDS loads are a different shape), so a copy here is an
// ordinary load and store, done when the call returns. Every unit commits
// and waits before it reads, then barriers, so a copy that has already
// landed satisfies the same order; it only gives up the overlap.

__device__ __forceinline__ void __pipeline_memcpy_async(void* dst, const void* src,
                                                        unsigned long n) {
    if (n == 16) {
        *static_cast<uint4*>(dst) = *static_cast<const uint4*>(src);
    } else if (n == 8) {
        *static_cast<uint2*>(dst) = *static_cast<const uint2*>(src);
    } else {
        *static_cast<unsigned*>(dst) = *static_cast<const unsigned*>(src);
    }
}

__device__ __forceinline__ void __pipeline_commit() {}

template <int N>
__device__ __forceinline__ void pie_pipeline_wait_prior() {}
#define __pipeline_wait_prior(N) pie_pipeline_wait_prior<(N)>()
