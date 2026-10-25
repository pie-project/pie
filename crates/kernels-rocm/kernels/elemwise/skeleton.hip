#pragma once

// The row skeleton, HIP's: one row of bf16 planes a block, held whole in
// registers, around a body kernels-ir's schedule prints in the skeleton's
// hooks (`kernels_rocm::synth::print`). The same hooks as kernels-cuda's
// `elemwise/skeleton.cuh`, over a wavefront's reductions.

#include "prim/device.hip"

namespace pie::elemwise {

// A bf16 plane a row skeleton reads or writes: its first element and how far
// apart its rows sit, in elements (0 for one row every row shares, as a
// norm's weight).
struct Plane {
    bf16* base;
    long long row_stride;
};

// What a row skeleton fires over: its body's IN parameters and OUT results,
// every row `width` wide.
template <int IN, int OUT>
struct Planes {
    Plane in[IN];
    Plane out[OUT];
    int width;
};

// One row of bf16 planes held whole in registers: a thread holds PER
// vectors of 8 elements of each parameter and each result. A body names an
// element by its index `e` in [0, N), column (tid + (e / 8) * BLOCK) * 8 +
// e % 8, and asks the skeleton for everything else. An element past the row
// (`!valid(e)`) reads garbage and is never stored; a body keeps it out of
// its reductions. A parameter whose bit `ONES` sets is one element every
// row reads (a layer's scale), read once and named `one(k)`.
template <int BLOCK, int PER, int IN, int OUT, unsigned ONES = 0u>
struct RowSkeleton {
    static constexpr int N = PER * 8;

    int width;
    int tid;
    int vectors;
    float* buf;
    uint4* dst[OUT];
    uint4 in[IN][PER];
    uint4 out[OUT][PER];
    float ones[IN];

    // Every parameter's vectors are issued up front, before any body math.
    __device__ __forceinline__ RowSkeleton(const Planes<IN, OUT>& planes, long long row, float* buf)
        : width(planes.width),
          tid(static_cast<int>(threadIdx.x)),
          vectors(planes.width / 8),
          buf(buf) {
#pragma unroll
        for (int k = 0; k < OUT; ++k) {
            dst[k] = reinterpret_cast<uint4*>(planes.out[k].base + row * planes.out[k].row_stride);
        }
#pragma unroll
        for (int k = 0; k < IN; ++k) {
            if ((ONES >> k) & 1u) {
                ones[k] = to_f32(planes.in[k].base[0]);
                continue;
            }
            const uint4* src =
                reinterpret_cast<const uint4*>(planes.in[k].base + row * planes.in[k].row_stride);
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                const int i = tid + p * BLOCK;
                if (i < vectors) in[k][p] = src[i];
            }
        }
    }

    __device__ __forceinline__ bool valid(int e) const { return tid + (e / 8) * BLOCK < vectors; }

    __device__ __forceinline__ float one(int k) const { return ones[k]; }

    __device__ __forceinline__ unsigned col(int e) const {
        return static_cast<unsigned>((tid + (e / 8) * BLOCK) * 8 + e % 8);
    }

    // An angle as the hand kernels turn one.
    __device__ __forceinline__ float sin(float v) const { return __sinf(v); }
    __device__ __forceinline__ float cos(float v) const { return __cosf(v); }

    __device__ __forceinline__ float ld(int k, int e) const {
        return to_f32(reinterpret_cast<const bf16*>(&in[k][e / 8])[e % 8]);
    }

    // `v` as result `k` holds it: rounded to the plane's bf16, what a later
    // read of the result sees.
    __device__ __forceinline__ float st(int k, int e, float v) {
        const bf16 stored = from_f32<bf16>(v);
        reinterpret_cast<bf16*>(&out[k][e / 8])[e % 8] = stored;
        return to_f32(stored);
    }

    // `v` rounded to bf16 and back: a value the program a body came from
    // stores between two ops, read on as it stored it.
    __device__ __forceinline__ float round_bf16(float v) const { return to_f32(from_f32<bf16>(v)); }

    __device__ __forceinline__ float reduce_sum(float local) { return block_sum(local, buf); }

    __device__ __forceinline__ float reduce_max(float local) { return block_max(local, buf); }

    // Result `K`'s row to memory, once its last element is written.
    template <int K>
    __device__ __forceinline__ void flush() {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = tid + p * BLOCK;
            if (i < vectors) dst[K][i] = out[K][p];
        }
    }
};

// A row skeleton a block a row: `body(f)` computes the row in hooks over `f`.
// The rows past a staged window's live count are left as they are.
template <class Body, int BLOCK, int PER, int IN, int OUT, unsigned ONES = 0u>
__global__ void __launch_bounds__(BLOCK) row_skeleton(
    const Planes<IN, OUT> planes,
    const Body body,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;

    __shared__ float buf[BLOCK / 32 + 1];
    RowSkeleton<BLOCK, PER, IN, OUT, ONES> f(planes, plane_row, buf);
    body(f);
}

}
