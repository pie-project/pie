#pragma once

#include <cuda_pipeline.h>

#include "prim/device.cuh"
#include "prim/act.cuh"

namespace pie::linear {

namespace bf16mma {
__device__ __forceinline__ void ld_a(unsigned (&a)[4], const bf16* base, int ld, int lane) {
    const bf16* r = base + static_cast<long long>(lane & 15) * ld + ((lane & 16) ? 8 : 0);
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(r));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ void ld_b(unsigned (&b)[2], const bf16* base, int ld, int lane) {
    const bf16* q = base + static_cast<long long>(lane & 7) * ld + ((lane & 8) ? 8 : 0);
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(q));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}
}

namespace w4mma {

// A staged activation chunk's 16-byte run `q` (0..15, codes 8q..8q+7 of
// the chunk's 128) of route row `row`, where the stage keeps it: a lane's
// four runs (4c..4c+3), and its quarter-warp's two rows, on eight bank
// quads.
__device__ __forceinline__ int x_slot(int row, int q) {
    return row * 16 + (q ^ (((q >> 3) & 1) << 1) ^ (row & 1));
}

// A staged run of eight activations relaid in place as the B words the
// register-dequant mma reads them in (the byte perms `compute` does a warp
// a run, done once a CTA): the first k-step's (b0, b1), then the second's.
__device__ __forceinline__ void lay_staged(uint4* at) {
    const uint4 v = *at;
    uint4 o;
    o.x = __byte_perm(v.y, v.w, 0x7632);
    o.y = __byte_perm(v.y, v.w, 0x5410);
    o.z = __byte_perm(v.x, v.z, 0x7632);
    o.w = __byte_perm(v.x, v.z, 0x5410);
    *at = o;
}

// `compute` with the routes' activations read from the stage (`xs`, the
// chunk's runs) rather than held a chunk in registers.
template <bool wide, bool kNv, int kTiles, int kNb>
__device__ __forceinline__ void compute_staged(const uint4 (&cw)[kTiles][2], const unsigned (&cs)[kTiles][2],
                                               const uint4* __restrict__ xs, int g, int c,
                                               float (&acc)[kTiles][kNb][4]) {
    unsigned s1[kTiles][2], s2[kTiles][2], nv[kTiles][2][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned s = cs[t][h];
            if constexpr (kNv) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const unsigned v = nvfp4_scale_bf16((s >> (8 * q)) & 0xFFu);
                    nv[t][h][q] = v | (v << 16);
                }
                s1[t][h] = s2[t][h] = 0u;
            } else {
                const unsigned a = !wide || s <= 128u ? (s + 126u) << 7 : 253u << 7;
                const unsigned b = s <= 128u ? 0x3F80u : s << 7;
                s1[t][h] = a | (a << 16);
                s2[t][h] = b | (b << 16);
            }
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if constexpr (kNv) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) s1[t][h] = nv[t][h][w >> 1];
        }
        // The run already laid as the mma's B words (`lay_staged`): its two
        // k-steps' (b0, b1) pairs.
        uint4 xv[kNb];
#pragma unroll
        for (int b = 0; b < kNb; ++b) xv[b] = xs[x_slot(8 * b + g, 4 * c + w)];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned b0[kNb], b1[kNb];
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                b0[b] = jj == 0 ? xv[b].x : xv[b].z;
                b1[b] = jj == 0 ? xv[b].y : xv[b].w;
            }
            const int p0 = 2 * jj;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const unsigned w0 = (&cw[t][0].x)[w];
                const unsigned w1 = (&cw[t][1].x)[w];
                unsigned a0 = mul2(e2m1_pair(w0 << (4 * p0)), s1[t][0]);
                unsigned a1 = mul2(e2m1_pair(w1 << (4 * p0)), s1[t][1]);
                unsigned a2 = mul2(e2m1_pair(w0 << (4 * p0 + 4)), s1[t][0]);
                unsigned a3 = mul2(e2m1_pair(w1 << (4 * p0 + 4)), s1[t][1]);
                if constexpr (wide) {
                    a0 = mul2(a0, s2[t][0]);
                    a1 = mul2(a1, s2[t][1]);
                    a2 = mul2(a2, s2[t][0]);
                    a3 = mul2(a3, s2[t][1]);
                }
#pragma unroll
                for (int b = 0; b < kNb; ++b) mma(acc[t][b], a0, a1, a2, a3, b0[b], b1[b]);
            }
        }
    }
}

}

}
