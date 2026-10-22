#include <cstdint>

#include <cuda_bf16.h>
#include <type_traits>

#include "prim/device.cuh"

// DSpark's Markov head over a drafted block's readouts (DeepSeek-V4.1): a
// readout at depth `i` of its block takes the bias `head . embed[t]` of the
// token `t` the readout at depth `i - 1` picked (depth 0, its own token),
// and picks its biased logits' argmax. The readouts at one depth are walked
// together: `plan` finds each one's depth, then `step` (once
// a depth) holds a tile of the head's rows in registers, multiplies them by
// the depth's embeddings on mma.sync a chunk at a time, and lands each
// chunk's biased rows a readout row at a time, coalesced, folding its
// argmax into the readout's key.
namespace pie::sampling::markov {

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
// token), and one outside its block not at all. One block: depths first,
// then the lists (a list's order is free: each readout reads the key of the
// readout before it, settled a step earlier).
__global__ void plan(const int* __restrict__ positions, const int* __restrict__ ends,
                            int readouts, int steps, int* __restrict__ lists,
                            int* __restrict__ counts, int* __restrict__ depth,
                            unsigned long long* __restrict__ keys) {
    for (int s = static_cast<int>(threadIdx.x); s < steps; s += static_cast<int>(blockDim.x)) {
        counts[s] = 0;
    }
    for (int r = static_cast<int>(threadIdx.x); r < readouts; r += static_cast<int>(blockDim.x)) {
        keys[r] = 0ull;
        const int d = positions[r] - (ends[r] - steps + 1);
        depth[r] = d >= 0 && d < steps ? d : -1;
    }
    __syncthreads();
    for (int r = static_cast<int>(threadIdx.x); r < readouts; r += static_cast<int>(blockDim.x)) {
        int d = depth[r];
        if (d < 0) continue;
        if (d > 0 && (r == 0 || depth[r - 1] != d - 1 || ends[r - 1] != ends[r])) d = 0;
        lists[d * readouts + atomicAdd(&counts[d], 1)] = r;
    }
    __syncthreads();
    // A readout walked as a block's first reads its own token: its depth,
    // as the step reads it, is 0.
    for (int r = static_cast<int>(threadIdx.x); r < readouts; r += static_cast<int>(blockDim.x)) {
        const int d = depth[r];
        if (d > 0 && (r == 0 || depth[r - 1] != d - 1 || ends[r - 1] != ends[r])) depth[r] = 0;
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

// Four contiguous logits, 16-byte (f32) or 8-byte (bf16) aligned.
template <typename T>
__device__ __forceinline__ void load_quad(const T* p, float (&out)[4]) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
        out[0] = __bfloat162float(lo.x);
        out[1] = __bfloat162float(lo.y);
        out[2] = __bfloat162float(hi.x);
        out[3] = __bfloat162float(hi.y);
    }
}

template <typename T>
__device__ __forceinline__ void store_quad(T* p, const float (&in)[4]) {
    if constexpr (std::is_same_v<T, float>) {
        *reinterpret_cast<float4*>(p) = make_float4(in[0], in[1], in[2], in[3]);
    } else {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(in[0], in[1]);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(in[2], in[3]);
        uint2 v;
        v.x = *reinterpret_cast<const unsigned*>(&lo);
        v.y = *reinterpret_cast<const unsigned*>(&hi);
        *reinterpret_cast<uint2*>(p) = v;
    }
}

// Depth `s`'s embeddings, its list's order: each readout's previous pick
// (its own token at depth 0), gathered once for every tile's CTA to read
// contiguously. A warp a readout, 16 bytes a lane.
__global__ void gather(const int* __restrict__ tokens, const __nv_bfloat16* __restrict__ embed,
                              const int* __restrict__ lists, const int* __restrict__ counts,
                              const int* __restrict__ depth,
                              const unsigned long long* __restrict__ keys,
                              __nv_bfloat16* __restrict__ gathered, int readouts, int s) {
    const int i = static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int lane = static_cast<int>(threadIdx.x % 32);
    if (i >= counts[s]) return;
    const int r = lists[s * readouts + i];
    const int tok = depth[r] == 0 ? tokens[r] : pick_of(keys[r - 1]);
    reinterpret_cast<uint4*>(gathered + static_cast<long long>(i) * kRank)[lane] =
        __ldg(reinterpret_cast<const uint4*>(embed + static_cast<long long>(tok) * kRank) + lane);
}

// A shared row of the head's tile or a chunk's embeddings, padded 16 bytes
// so `ldmatrix` reads its eight rows a phase on distinct banks. A step's
// shared memory is the tile, the chunk's embeddings and its f32 products
// (`markov.rs`'s `STEP_SMEM`).
constexpr int kLd = kRank + 8;

// Depth `s`: every readout at it, against a tile of 128 vocab rows a CTA.
// The tile's head rows stream into shared memory 16 bytes a copy and feed
// the mma by `ldmatrix`; a chunk's products land in shared memory and each
// warp then walks whole readout rows of the tile (128 contiguous logits),
// adding, landing and folding the argmax in f32 before it lands at `T`.
template <typename T>
__global__ void __launch_bounds__(kThreads, 2) step(
    const T* __restrict__ logits,
    const __nv_bfloat16* __restrict__ gathered, const __nv_bfloat16* __restrict__ head,
    T* __restrict__ biased, const int* __restrict__ lists, const int* __restrict__ counts,
    unsigned long long* __restrict__ keys, int readouts, int vocab, int s) {
    const int n = counts[s];
    if (n == 0) return;
    const int* list = lists + s * readouts;
    extern __shared__ __align__(128) unsigned char smem[];
    auto* hs = reinterpret_cast<__nv_bfloat16 (*)[kLd]>(smem);
    auto* emb = reinterpret_cast<__nv_bfloat16 (*)[kLd]>(smem + kTileV * kLd * 2);
    auto* prod = reinterpret_cast<float (*)[kTileV + 4]>(smem + (kTileV + kChunk) * kLd * 2);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane >> 2, t = lane & 3;
    const int tile = static_cast<int>(blockIdx.x) * kTileV;

    // The tile's head rows, 16 bytes a copy (a row past the vocabulary
    // reads the last, its products never landing).
    for (int i = static_cast<int>(threadIdx.x); i < kTileV * (kRank / 8); i += kThreads) {
        const int row = i / (kRank / 8);
        const int col = (i % (kRank / 8)) * 8;
        const __nv_bfloat16* src = head + static_cast<long long>(min(tile + row, vocab - 1)) * kRank + col;
        const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(&hs[row][col]));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    for (int c0 = 0; c0 < n; c0 += kChunk) {
        const int m = min(kChunk, n - c0);
        __syncthreads();
        // The chunk's embeddings, gathered in the list's order.
        for (int i = static_cast<int>(threadIdx.x); i < kChunk * (kRank / 8); i += kThreads) {
            const int j = i / (kRank / 8);
            const int col = (i % (kRank / 8)) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (j < m) {
                v = *reinterpret_cast<const uint4*>(gathered + static_cast<long long>(c0 + j) * kRank + col);
            }
            *reinterpret_cast<uint4*>(&emb[j][col]) = v;
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        float acc[kChunk / 8][4];
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.f;
        const unsigned a_row = static_cast<unsigned>(
            __cvta_generic_to_shared(&hs[warp * 16 + (lane % 16)][(lane / 16) * 8]));
#pragma unroll
        for (int ks = 0; ks < kRank / 16; ++ks) {
            unsigned a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(a_row + ks * 32));
#pragma unroll
            for (int nt = 0; nt < kChunk / 8; ++nt) {
                const __nv_bfloat16* e = &emb[nt * 8 + g][ks * 16];
                const unsigned b0 = *reinterpret_cast<const unsigned*>(e + 2 * t);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(e + 8 + 2 * t);
                mma16816(acc[nt], a, b0, b1);
            }
        }
        // acc[nt]: vocab rows (g, g + 8) of the warp's 16 by readouts
        // nt * 8 + 2t (+1), into the chunk's product tile.
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int j = nt * 8 + 2 * t + c;
                prod[j][warp * 16 + g] = acc[nt][c];
                prod[j][warp * 16 + g + 8] = acc[nt][2 + c];
            }
        }
        __syncthreads();
        // A warp a readout row at a time, four contiguous vocab columns a
        // lane (one 8- or 16-byte access each way): every row's logits read
        // first, so the warp holds them all in flight.
        constexpr int kPer = kChunk / (kThreads / 32);
        const int col = 4 * lane;
        const int v = tile + col;
        // A whole quad of the vocabulary, aligned: else column by column.
        const bool quad = v + 3 < vocab && (vocab & 3) == 0;
        float y[kPer][4];
        int rows[kPer];
#pragma unroll
        for (int jj = 0; jj < kPer; ++jj) {
            const int j = warp + jj * (kThreads / 32);
            rows[jj] = j < m ? list[c0 + j] : -1;
            const T* src = logits + static_cast<long long>(max(rows[jj], 0)) * vocab + v;
            if (rows[jj] >= 0 && quad) {
                load_quad(src, y[jj]);
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    y[jj][e] = rows[jj] >= 0 && v + e < vocab ? load_logit(src + e) : 0.f;
                }
            }
        }
#pragma unroll
        for (int jj = 0; jj < kPer; ++jj) {
            if (rows[jj] < 0) continue;
            const int j = warp + jj * (kThreads / 32);
            T* dst = biased + static_cast<long long>(rows[jj]) * vocab + v;
            const float4 p4 = *reinterpret_cast<const float4*>(&prod[j][col]);
            const float b[4] = {y[jj][0] + p4.x, y[jj][1] + p4.y, y[jj][2] + p4.z, y[jj][3] + p4.w};
            unsigned long long mine = 0ull;
            if (quad) {
                store_quad(dst, b);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const unsigned long long k = key_of(b[e], static_cast<unsigned>(v + e));
                    mine = k > mine ? k : mine;
                }
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (v + e < vocab) {
                        store_logit(dst + e, b[e]);
                        const unsigned long long k = key_of(b[e], static_cast<unsigned>(v + e));
                        mine = k > mine ? k : mine;
                    }
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const unsigned long long other = __shfl_xor_sync(0xffffffffu, mine, o);
                mine = other > mine ? other : mine;
            }
            if (lane == 0) atomicMax(&keys[rows[jj]], mine);
        }
    }
}

// Each readout's pick, out of its key.
__global__ void picks(const unsigned long long* __restrict__ keys, int* __restrict__ picks,
                             int readouts) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r < readouts) picks[r] = pick_of(keys[r]);
}

}  // namespace pie::sampling::markov
