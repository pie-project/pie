#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

// A blockwise butterfly FWHT over the last dim. The row is cut into contiguous
// N-vectors (N a power of two) and each is turned by the normalized Sylvester
// Hadamard matrix H, whose entries are (-1)^popcount(i & j) / sqrt(N). The
// standard radix-2 fast Walsh-Hadamard transform pairs each index g with
// g ^ 2^b at stage b, and its natural (Sylvester) ordering is exactly H, so the
// pure transform is symmetric and orthonormal: H . H = I.
//
// When a sign buffer is bound (signs != nullptr) each element is multiplied by
// its +-1 sign on load, before the butterfly -- this realizes the Randomized
// Hadamard H.S. Because S and H do not commute, H.S is not self-inverse. The
// sign vector repeats block-wise: the element at flat offset `p` reads
// signs[p % signs_width].
//
// The elements of one N-block are spread across the group's threads in an
// interleaved layout: register i of thread t holds the block's element
// (i * THREADS + t). Stages whose stride is below the warp width are exchanged
// with __shfl_xor_sync; the wide variant exchanges the mid stages (one warp
// width up to the block size) through shared memory; the top stages live
// entirely within a thread's own registers.

// N <= 256: one warp (32 lanes) owns one block, no shared memory.
template <class T, int N>
__global__ void fwht_warp(
    T* __restrict__ x,
    const T* __restrict__ signs,
    u32 signs_width)
{
    constexpr int NW = 32;
    constexpr int NE = N / NW;
    const float scale = 1.0f / sqrtf(static_cast<float>(N));
    const usize base = static_cast<usize>(blockIdx.x) * static_cast<usize>(N);
    const u32 lane = threadIdx.x;

    float reg[NE];
    for (int i = 0; i < NE; ++i) {
        const u32 within = static_cast<u32>(i) * static_cast<u32>(NW) + lane;
        const usize p = base + static_cast<usize>(within);
        float s = 1.0f;
        if (signs != nullptr) {
            s = Elem<T>::to_f32(signs[static_cast<u32>(p % static_cast<usize>(signs_width))]);
        }
        reg[i] = Elem<T>::to_f32(x[p]) * s * scale;
    }

    for (u32 b = 1u; b < static_cast<u32>(NW); b <<= 1) {
        for (int i = 0; i < NE; ++i) {
            const float v = reg[i];
            const float v2 = __shfl_xor_sync(0xffffffffu, v, b);
            reg[i] = ((lane & b) == 0u) ? (v2 + v) : (v2 - v);
        }
    }

    for (int step = 1; step < NE; step <<= 1) {
        for (int j = 0; j < NE; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k];
                const float bb = reg[j + k + step];
                reg[j + k] = a + bb;
                reg[j + k + step] = a - bb;
            }
        }
    }

    for (int i = 0; i < NE; ++i) {
        x[base + static_cast<usize>(static_cast<u32>(i) * static_cast<u32>(NW) + lane)] =
            Elem<T>::from_f32(reg[i]);
    }
}

// N >= 512: one block (THREADS threads) owns one block; the mid stages go
// through shared memory.
template <class T, int N, int THREADS>
__global__ void fwht_block(
    T* __restrict__ x,
    const T* __restrict__ signs,
    u32 signs_width)
{
    constexpr int NW = 32;
    constexpr int NE = N / THREADS;
    const float scale = 1.0f / sqrtf(static_cast<float>(N));
    const usize base = static_cast<usize>(blockIdx.x) * static_cast<usize>(N);
    const u32 tid = threadIdx.x;
    __shared__ float shmem[N];

    float reg[NE];
    for (int i = 0; i < NE; ++i) {
        const u32 within = static_cast<u32>(i) * static_cast<u32>(THREADS) + tid;
        const usize p = base + static_cast<usize>(within);
        float s = 1.0f;
        if (signs != nullptr) {
            s = Elem<T>::to_f32(signs[static_cast<u32>(p % static_cast<usize>(signs_width))]);
        }
        reg[i] = Elem<T>::to_f32(x[p]) * s * scale;
    }

    for (u32 b = 1u; b < static_cast<u32>(NW); b <<= 1) {
        for (int i = 0; i < NE; ++i) {
            const float v = reg[i];
            const float v2 = __shfl_xor_sync(0xffffffffu, v, b);
            reg[i] = ((tid & b) == 0u) ? (v2 + v) : (v2 - v);
        }
    }

    for (u32 b = static_cast<u32>(NW); b < static_cast<u32>(THREADS); b <<= 1) {
        __syncthreads();
        for (int i = 0; i < NE; ++i) {
            shmem[static_cast<u32>(i) * static_cast<u32>(THREADS) + tid] = reg[i];
        }
        __syncthreads();
        for (int i = 0; i < NE; ++i) {
            const u32 within = static_cast<u32>(i) * static_cast<u32>(THREADS) + tid;
            const float partner = shmem[within ^ b];
            reg[i] = ((tid & b) == 0u) ? (reg[i] + partner) : (partner - reg[i]);
        }
    }

    for (int step = 1; step < NE; step <<= 1) {
        for (int j = 0; j < NE; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k];
                const float bb = reg[j + k + step];
                reg[j + k] = a + bb;
                reg[j + k + step] = a - bb;
            }
        }
    }

    for (int i = 0; i < NE; ++i) {
        x[base + static_cast<usize>(static_cast<u32>(i) * static_cast<u32>(THREADS) + tid)] =
            Elem<T>::from_f32(reg[i]);
    }
}

}
