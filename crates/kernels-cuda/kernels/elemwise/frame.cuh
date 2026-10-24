#pragma once

#include "elemwise/row.cuh"

namespace pie::elemwise {

// A bf16 plane a row frame reads or writes: its first element and how far
// apart its rows sit, in elements (0 for one row every row shares, as a
// norm's weight).
struct Plane {
    bf16* base;
    long long row_stride;
};

// What a row frame fires over: its body's IN parameters and OUT results,
// every row `width` wide. `kernels_cuda::elemwise::frame` mirrors it.
template <int IN, int OUT>
struct Planes {
    Plane in[IN];
    Plane out[OUT];
    int width;
};

// One row of bf16 planes held whole in registers: a thread holds PER
// vectors of 8 elements of each parameter and each result. A body names an
// element by its index `e` in [0, N), column (tid + (e / 8) * BLOCK) * 8 +
// e % 8, and asks the frame for everything else: its parameters' values,
// where its results go, and the row's sums. Every index a body passes is a
// constant once its loops unroll, so the arrays live in registers. An
// element past the row (`!valid(e)`) reads garbage and is never stored; a
// body keeps it out of its reductions.
template <int BLOCK, int PER, int IN, int OUT>
struct RowFrame {
    static constexpr int N = PER * 8;

    int width;
    int tid;
    int vectors;
    float* buf;
    uint4* dst[OUT];
    uint4 in[IN][PER];
    uint4 out[OUT][PER];

    // Every parameter's vectors are issued up front, before any body math.
    __device__ __forceinline__ RowFrame(const Planes<IN, OUT>& planes, long long row, float* buf)
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

    __device__ __forceinline__ float ld(int k, int e) const {
        return bf16_to_f32(reinterpret_cast<const bf16*>(&in[k][e / 8])[e % 8]);
    }

    // `v` as result `k` holds it: rounded to the plane's bf16, which is
    // what a later read of the result sees. It comes back as a permute of
    // the stored pair rather than the rounded f32, so a body that keeps it
    // across a reduction holds the pair packed and unpacks it where it is
    // read: a row of 4 vectors a thread then fits 64 registers, as the hand
    // register kernel does.
    __device__ __forceinline__ float st(int k, int e, float v) {
        reinterpret_cast<bf16*>(&out[k][e / 8])[e % 8] = f32_to_bf16(v);
        const unsigned pair = reinterpret_cast<const unsigned*>(&out[k][e / 8])[e % 8 / 2];
        return __uint_as_float(__byte_perm(pair, 0, e % 2 == 0 ? 0x1044 : 0x3244));
    }

    // `v` rounded to bf16 and back: a value the program a body came from
    // stores between two ops, read on as it stored it.
    __device__ __forceinline__ float round_bf16(float v) const { return bf16_to_f32(f32_to_bf16(v)); }

    __device__ __forceinline__ float reduce_sum(float local) {
        return block_reduce_sum_fast<BLOCK>(local, buf);
    }

    __device__ __forceinline__ float reduce_max(float local) {
        return block_reduce_max_fast<BLOCK>(local, buf);
    }

    // Result `K`'s row to memory: a body flushes a result once its last
    // element is written, so the store overlaps whatever follows.
    template <int K>
    __device__ __forceinline__ void flush() {
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = tid + p * BLOCK;
            if (i < vectors) dst[K][i] = out[K][p];
        }
    }
};

// A row frame a block a row: `body(f)` computes the row in hooks over `f`,
// its own fields (a norm's eps) passed by value. The rows past a staged
// window's live count are left as they are.
template <class Body, int BLOCK, int PER, int IN, int OUT>
__global__ void __launch_bounds__(BLOCK) row_frame(
    const Planes<IN, OUT> planes,
    const Body body,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;

    __shared__ float buf[BLOCK / 32 + 1];
    RowFrame<BLOCK, PER, IN, OUT> f(planes, plane_row, buf);
    body(f);
}

}
