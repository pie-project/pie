#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/shuffle.slang"

// y = x * w^T on bf16 for WGSL: every store lands a pair of columns, one
// word. `PIE_GEMV`: a few rows (`PIE_VROWS` a workgroup over one read of
// its columns' weights), a 32-lane strip a column walking the contraction (four words a step where it is even), the strip folded
// through workgroup memory or, at the subgroup tier, by shuffles; column
// pairs land from the even strip. Otherwise a `PIE_BM` x 64 tile a
// workgroup of 16 x 16, k-major float4 tiles in workgroup memory so an
// invocation writes whole vectors (a store to one component of a workgroup
// vector is a read-modify-write of the whole vector on Metal). The host
// refuses an odd column count.

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> w;
[[vk::binding(2)]] RWStructuredBuffer<uint> out_;

struct Params {
    int m;
    int n;
    int k;
};
PIE_PARAMS(3, Params, params);

float load_x(uint row, uint kk) {
    return bf16_at(x, row * uint(params.k) + kk);
}

float load_w(uint col, uint kk) {
    return bf16_at(w, col * uint(params.k) + kk);
}

void write_pair(uint row, uint col, float lo, float hi) {
    if (row >= uint(params.m) || col >= uint(params.n)) {
        return;
    }
    out_[(row * uint(params.n) + col) >> 1] = pack_bf16(lo, hi);
}

#if defined(PIE_GEMV)

#define PIE_KLANES 32u
#define PIE_NLANES 8u
#define PIE_WORDS 4u
#define PIE_WIDE (PIE_KLANES * PIE_WORDS)
// The rows a workgroup carries over one read of its columns' weights: a
// row of workgroups (group.z) a block of them.
#ifndef PIE_VROWS
#define PIE_VROWS 4u
#endif

#if defined(PIE_SUBGROUP)
groupshared float partial[PIE_NLANES * PIE_VROWS];
#else
groupshared float partial[PIE_VROWS * PIE_NLANES * PIE_KLANES];
#endif

float pair_dot(uint xw, uint ww) {
    return bf16_to_f32(xw & 0xffffu) * bf16_to_f32(ww & 0xffffu)
         + bf16_to_f32(xw >> 16) * bf16_to_f32(ww >> 16);
}

[numthreads(32, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint klane = local.x;
    uint clane = local.y;
    uint col = gid.y;
    uint row0 = group.z * PIE_VROWS;
    uint rows = min(uint(params.m) - min(row0, uint(params.m)), PIE_VROWS);
    float acc[PIE_VROWS];
    [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) acc[r] = 0.0;
    if (col < uint(params.n)) {
        uint k = uint(params.k);
        uint words = k >> 1;
        uint wide = 0u;
        if ((k & 1u) == 0u) {
            wide = (words / PIE_WIDE) * PIE_WIDE;
        }
        uint wbase = col * words;
        for (uint wi = klane * PIE_WORDS; wi < wide; wi += PIE_WIDE) {
            uint wa = wbase + wi;
            uint w0 = w[wa];
            uint w1 = w[wa + 1u];
            uint w2 = w[wa + 2u];
            uint w3 = w[wa + 3u];
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                if (r < rows) {
                    uint xa = (row0 + r) * words + wi;
                    acc[r] += pair_dot(x[xa], w0);
                    acc[r] += pair_dot(x[xa + 1u], w1);
                    acc[r] += pair_dot(x[xa + 2u], w2);
                    acc[r] += pair_dot(x[xa + 3u], w3);
                }
            }
        }
        for (uint kk = wide * 2u + klane; kk < k; kk += PIE_KLANES) {
            float wv = load_w(col, kk);
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                if (r < rows) acc[r] += load_x(row0 + r, kk) * wv;
            }
        }
    }
#if defined(PIE_SUBGROUP)
    // A subgroup of at least 32 lanes holds a strip: fold it by shuffles.
    [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
        [ForceUnroll] for (uint off = 16u; off > 0u; off >>= 1u) {
            acc[r] += pie_shuffle_xor(acc[r], off);
        }
        if (klane == 0u) {
            partial[r * PIE_NLANES + clane] = acc[r];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (klane == 0u && (clane & 1u) == 0u) {
        [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
            if (r < rows) {
                write_pair(row0 + r, col, partial[r * PIE_NLANES + clane], partial[r * PIE_NLANES + clane + 1u]);
            }
        }
    }
#else
    [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
        partial[(r * PIE_NLANES + clane) * PIE_KLANES + klane] = acc[r];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint step = PIE_KLANES / 2u; step > 0u; step >>= 1u) {
        if (klane < step) {
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                uint slot = (r * PIE_NLANES + clane) * PIE_KLANES + klane;
                partial[slot] += partial[slot + step];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (klane == 0u && (clane & 1u) == 0u) {
        [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
            if (r < rows) {
                write_pair(row0 + r, col, partial[(r * PIE_NLANES + clane) * PIE_KLANES],
                    partial[(r * PIE_NLANES + clane + 1u) * PIE_KLANES]);
            }
        }
    }
#endif
}

#else

#define PIE_BN 64u
#define PIE_BK 16u
#define PIE_BN4 16u
#define PIE_LANES 256u
#define PIE_RPL (uint(PIE_BM) / 16u)
#define PIE_ROWS4 (uint(PIE_BM) / 4u)

groupshared float4 xs[PIE_BK * PIE_ROWS4];
groupshared float4 ws[PIE_BK * PIE_BN4];

// Elements `kk` and `kk + 1` of a row `k` long, zero past its end.
float2 pair_of(StructuredBuffer<uint> buf, uint row, uint kk, uint k, bool k_even) {
    if (kk >= k) {
        return float2(0.0, 0.0);
    }
    uint i0 = row * k + kk;
    if (k_even) {
        uint word = buf[i0 >> 1];
        return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
    }
    float hi = kk + 1u < k ? bf16_at(buf, i0 + 1u) : 0.0;
    return float2(bf16_at(buf, i0), hi);
}

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint tx = local.x;
    uint ty = local.y;
    uint flat = tx + 16u * ty;
    uint tile_row = group.y * uint(PIE_BM);
    uint tile_col = group.x * PIE_BN;
    uint m = uint(params.m);
    uint n = uint(params.n);
    uint k = uint(params.k);
    bool k_even = (k & 1u) == 0u;
    float4 acc[PIE_RPL];
    [ForceUnroll] for (uint i = 0u; i < PIE_RPL; i++) acc[i] = float4(0.0, 0.0, 0.0, 0.0);
    for (uint kb = 0u; kb < k; kb += PIE_BK) {
        GroupMemoryBarrierWithGroupSync();
        for (uint e = flat; e < PIE_ROWS4 * 8u; e += PIE_LANES) {
            uint rq = e / 8u;
            uint kk = (e - rq * 8u) * 2u;
            float4 lo = float4(0.0, 0.0, 0.0, 0.0);
            float4 hi = float4(0.0, 0.0, 0.0, 0.0);
            [ForceUnroll] for (uint j = 0u; j < 4u; j++) {
                uint row = tile_row + rq * 4u + j;
                if (row < m) {
                    float2 pair = pair_of(x, row, kb + kk, k, k_even);
                    lo[j] = pair.x;
                    hi[j] = pair.y;
                }
            }
            xs[kk * PIE_ROWS4 + rq] = lo;
            xs[(kk + 1u) * PIE_ROWS4 + rq] = hi;
        }
        for (uint e = flat; e < PIE_BN4 * 8u; e += PIE_LANES) {
            uint cq = e / 8u;
            uint kk = (e - cq * 8u) * 2u;
            float4 lo = float4(0.0, 0.0, 0.0, 0.0);
            float4 hi = float4(0.0, 0.0, 0.0, 0.0);
            [ForceUnroll] for (uint j = 0u; j < 4u; j++) {
                uint col = tile_col + cq * 4u + j;
                if (col < n) {
                    float2 pair = pair_of(w, col, kb + kk, k, k_even);
                    lo[j] = pair.x;
                    hi[j] = pair.y;
                }
            }
            ws[kk * PIE_BN4 + cq] = lo;
            ws[(kk + 1u) * PIE_BN4 + cq] = hi;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint kk = 0u; kk < PIE_BK; kk++) {
            float4 wv = ws[kk * PIE_BN4 + tx];
#if PIE_BM == 64
            float4 xv = xs[kk * PIE_ROWS4 + ty];
            acc[0] += xv.x * wv;
            acc[1] += xv.y * wv;
            acc[2] += xv.z * wv;
            acc[3] += xv.w * wv;
#else
            float xv = xs[kk * PIE_ROWS4 + ty / 4u][ty & 3u];
            acc[0] += xv * wv;
#endif
        }
    }
    uint col = tile_col + tx * 4u;
    [ForceUnroll] for (uint i = 0u; i < PIE_RPL; i++) {
        uint row = tile_row + ty * PIE_RPL + i;
        write_pair(row, col, acc[i].x, acc[i].y);
        write_pair(row, col + 2u, acc[i].z, acc[i].w);
    }
}

#endif

// pie:instantiate linear_gemm_pairs_gemv_bf16 PIE_GEMV=1 PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_bf16 PIE_GEMV=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v1_bf16 PIE_GEMV=1 PIE_VROWS=1u PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v1_bf16 PIE_GEMV=1 PIE_VROWS=1u PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v8_bf16 PIE_GEMV=1 PIE_VROWS=8u PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v8_bf16 PIE_GEMV=1 PIE_VROWS=8u PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v16_bf16 PIE_GEMV=1 PIE_VROWS=16u PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_gemv_v16_bf16 PIE_GEMV=1 PIE_VROWS=16u PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_tiled_bf16 PIE_BM=64 PIE_TARGET=wgsl
// pie:instantiate linear_gemm_pairs_tiled_bm_16_bf16 PIE_BM=16 PIE_TARGET=wgsl
