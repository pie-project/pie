#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <type_traits>

#include "prelude/device.cuh"

// DSpark's Markov head over a drafted block's readouts (DeepSeek-V4.1): the
// readouts fall in runs, a lane's each; row `i` of a run takes the bias
// `head . embed[t]` of the token `t` the run's row `i - 1` picked (the run's
// first row, its own token), and picks its biased logits' argmax. The rows
// at one depth are walked together: `markov_plan` finds each readout's lane
// and depth, then `markov_step` (once a depth) multiplies a tile of the
// head's rows by the depth's embeddings on mma.sync, adds the logits, lands
// the biased rows and folds each row's argmax into its key.
namespace pie::attn::markov {

constexpr int kRank = 256;
constexpr int kTileV = 128;   // vocab rows a CTA
constexpr int kChunk = 32;    // readouts a pass over the tile
constexpr int kThreads = 256;

// A readout's biased-logit argmax as one orderable word: the value's
// ordered bits high, the index complemented low, so the max is the first
// index of the largest value.
__device__ __forceinline__ unsigned long long key_of(float v, unsigned idx) {
    const unsigned u = __float_as_uint(v);
    const unsigned ordered = (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(ordered) << 32) | (0xFFFFFFFFu - idx);
}

__device__ __forceinline__ int pick_of(unsigned long long key) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<unsigned>(key & 0xFFFFFFFFull));
}

// The readouts a depth holds (`lists`, `steps` lists of `readouts` each,
// `counts` a depth) and the keys zeroed. A readout's depth is its position
// past its block's first (`ends - steps + 1`); one whose readout before it
// is not its block's previous row is walked as a block's first (its own
// token), and one outside its block not at all. One thread walks them in
// order.
__global__ void markov_plan(const int* __restrict__ positions, const int* __restrict__ ends,
                            int readouts, int steps, int* __restrict__ lists,
                            int* __restrict__ counts, int* __restrict__ depth,
                            unsigned long long* __restrict__ keys) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int s = 0; s < steps; ++s) counts[s] = 0;
    for (int r = 0; r < readouts; ++r) {
        keys[r] = 0ull;
        int d = positions[r] - (ends[r] - steps + 1);
        if (d < 0 || d >= steps) {
            depth[r] = -1;
            continue;
        }
        if (d > 0 && (r == 0 || depth[r - 1] != d - 1 || ends[r - 1] != ends[r])) d = 0;
        depth[r] = d;
        lists[d * readouts + counts[d]++] = r;
    }
}

__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <typename T>
__device__ __forceinline__ float load_logit(const T* p) {
    if constexpr (std::is_same_v<T, float>) {
        return *p;
    } else {
        return __bfloat162float(*p);
    }
}

template <typename T>
__device__ __forceinline__ void store_logit(T* p, float v) {
    if constexpr (std::is_same_v<T, float>) {
        *p = v;
    } else {
        *p = __float2bfloat16_rn(v);
    }
}

// Depth `s`: every readout at it, against a tile of 128 vocab rows a CTA.
// The argmax folds the biased sum in f32, before it lands at `T`.
template <typename T>
__global__ void __launch_bounds__(kThreads) markov_step(
    const T* __restrict__ logits, const int* __restrict__ tokens,
    const __nv_bfloat16* __restrict__ embed, const __nv_bfloat16* __restrict__ head,
    T* __restrict__ biased, const int* __restrict__ lists, const int* __restrict__ counts,
    const int* __restrict__ depth,
    unsigned long long* __restrict__ keys, int readouts, int vocab, int s) {
    const int n = counts[s];
    if (n == 0) return;
    const int* list = lists + s * readouts;
    __shared__ __align__(16) __nv_bfloat16 emb[kChunk][kRank + 8];
    __shared__ unsigned long long best[kChunk];
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane >> 2, t = lane & 3;
    const int v0 = static_cast<int>(blockIdx.x) * kTileV + warp * 16;
    for (int c0 = 0; c0 < n; c0 += kChunk) {
        const int m = min(kChunk, n - c0);
        __syncthreads();
        // The chunk's embeddings: each row's previous pick (its own token
        // at depth 0).
        for (int i = static_cast<int>(threadIdx.x); i < kChunk * (kRank / 8); i += kThreads) {
            const int j = i / (kRank / 8);
            const int col = (i % (kRank / 8)) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (j < m) {
                const int r = list[c0 + j];
                const int tok = depth[r] == 0 ? tokens[r] : pick_of(keys[r - 1]);
                v = *reinterpret_cast<const uint4*>(embed + static_cast<long long>(tok) * kRank + col);
            }
            *reinterpret_cast<uint4*>(&emb[j][col]) = v;
        }
        if (threadIdx.x < kChunk) best[threadIdx.x] = 0ull;
        __syncthreads();
        float acc[kChunk / 8][4];
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
        if (v0 < vocab) {
            const int r0 = min(v0 + g, vocab - 1), r1 = min(v0 + g + 8, vocab - 1);
            const __nv_bfloat16* h0 = head + static_cast<long long>(r0) * kRank;
            const __nv_bfloat16* h1 = head + static_cast<long long>(r1) * kRank;
#pragma unroll 4
            for (int k0 = 0; k0 < kRank; k0 += 16) {
                unsigned a[4];
                a[0] = *reinterpret_cast<const unsigned*>(h0 + k0 + 2 * t);
                a[1] = *reinterpret_cast<const unsigned*>(h1 + k0 + 2 * t);
                a[2] = *reinterpret_cast<const unsigned*>(h0 + k0 + 8 + 2 * t);
                a[3] = *reinterpret_cast<const unsigned*>(h1 + k0 + 8 + 2 * t);
#pragma unroll
                for (int nt = 0; nt < kChunk / 8; ++nt) {
                    const __nv_bfloat16* e = &emb[nt * 8 + g][k0];
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(e + 2 * t);
                    const unsigned b1 = *reinterpret_cast<const unsigned*>(e + 8 + 2 * t);
                    mma16816(acc[nt], a, b0, b1);
                }
            }
        }
        // acc[nt]: vocab rows (g, g + 8) of the warp's 16 by readouts
        // nt * 8 + 2t (+1): add the logits, land, fold the argmax.
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int j = nt * 8 + 2 * t + c;
                unsigned long long mine = 0ull;
                if (j < m) {
                    const int r = list[c0 + j];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int v = v0 + g + 8 * h;
                        if (v < vocab) {
                            const long long at = static_cast<long long>(r) * vocab + v;
                            const float y = load_logit(logits + at) + acc[nt][2 * h + c];
                            store_logit(biased + at, y);
                            const unsigned long long k = key_of(y, static_cast<unsigned>(v));
                            mine = k > mine ? k : mine;
                        }
                    }
                }
                // The 8 lanes sharing this readout (same t, every g).
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    const unsigned long long other = __shfl_xor_sync(0xffffffffu, mine, o);
                    mine = other > mine ? other : mine;
                }
                if (g == 0 && j < m) atomicMax(&best[j], mine);
            }
        }
        __syncthreads();
        if (threadIdx.x < m) atomicMax(&keys[list[c0 + threadIdx.x]], best[threadIdx.x]);
    }
}

// Each readout's pick, out of its key.
__global__ void markov_picks(const unsigned long long* __restrict__ keys, int* __restrict__ picks,
                             int readouts) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r < readouts) picks[r] = pick_of(keys[r]);
}

}  // namespace pie::attn::markov
