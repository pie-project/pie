#include "prim/affine.slang"
#include "prim/bf16.slang"
#include "prim/params.slang"

// y = x * dequant(w)^T, tiled for WGSL: every store lands whole bf16 words.
// A `PIE_BM` x `PIE_BN` output tile a workgroup of 64 * `PIE_WM`
// invocations, the contraction in 32-code blocks through workgroup memory.
// The 16-column tiles at 8 and 16 rows hold two neighbouring columns an
// invocation; the rest hold four columns (a quad) of `PIE_RPL` rows, the
// tiles stored k-major as float4s so an invocation writes whole vectors (a
// store to one component of a workgroup vector is a read-modify-write of
// the whole vector on Metal, so neighbours filling one would race). The
// host refuses an odd column count: a word holds two columns.

#define PIE_BK 32u
#if PIE_BM < 16
#define PIE_WM 1u
#else
#define PIE_WM 2u
#endif
#define PIE_THREADS (64u * PIE_WM)
#if PIE_GROUP % 32 != 0
#error "a K block must not straddle a scale group"
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
[[vk::binding(1)]] StructuredBuffer<uint> scales;
[[vk::binding(2)]] StructuredBuffer<uint> biases;
[[vk::binding(3)]] StructuredBuffer<uint> x;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int k;
    int n;
    int m;
};
PIE_PARAMS(5, Params, params);

#if (PIE_BM == 16 || PIE_BM == 8) && PIE_BN == 16

#define PIE_PER ((uint(PIE_BM) * uint(PIE_BN)) / (2u * PIE_THREADS))

groupshared float xs[PIE_BM * PIE_BK];
groupshared float ws[PIE_BN * PIE_BK];

[numthreads(32, 2, PIE_WM)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint row0 = group.y * uint(PIE_BM);
    uint col0 = group.x * uint(PIE_BN);
    uint k = uint(max(params.k, 0));
    uint n = uint(max(params.n, 0));
    uint m = uint(max(params.m, 0));
    uint cpw = PIE_CODES_PER_WORD;
    uint wpr = k / cpw;
    uint gpr = k / uint(PIE_GROUP);
    uint words_per_block = PIE_BK / cpw;
    uint half_bn = uint(PIE_BN) / 2u;

    float2 acc[PIE_PER];
    [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) acc[p] = float2(0.0, 0.0);
    for (uint k0 = 0u; k0 < k; k0 += PIE_BK) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = flat; i < uint(PIE_BM) * PIE_BK; i += PIE_THREADS) {
            uint r = i / PIE_BK;
            uint kk = i - r * PIE_BK;
            uint row = row0 + r;
            xs[i] = row < m ? bf16_at(x, row * k + k0 + kk) : 0.0;
        }
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < uint(PIE_BN) * words_per_block; i += PIE_THREADS) {
            uint c = i / words_per_block;
            uint wi = i - c * words_per_block;
            uint col = col0 + c;
            if (col < n) {
                uint word = w[col * wpr + k0 / cpw + wi];
                float s = bf16_at(scales, col * gpr + g);
                float b = bf16_at(biases, col * gpr + g);
                for (uint q = 0u; q < cpw; q++) {
                    ws[c * PIE_BK + wi * cpw + q] = affine_code(word, q) * s + b;
                }
            } else {
                for (uint q = 0u; q < cpw; q++) {
                    ws[c * PIE_BK + wi * cpw + q] = 0.0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) {
            uint idx = flat + p * PIE_THREADS;
            uint r = idx / half_bn;
            uint c = 2u * (idx - r * half_bn);
            float2 sum = float2(0.0, 0.0);
            for (uint kk = 0u; kk < PIE_BK; kk++) {
                float xv = xs[r * PIE_BK + kk];
                sum += xv * float2(ws[c * PIE_BK + kk], ws[(c + 1u) * PIE_BK + kk]);
            }
            acc[p] += sum;
        }
    }
    [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) {
        uint idx = flat + p * PIE_THREADS;
        uint r = idx / half_bn;
        uint c = 2u * (idx - r * half_bn);
        uint row = row0 + r;
        uint col = col0 + c;
        if (row < m && col < n) {
            y[(row * n + col) >> 1] = pack_bf16(acc[p].x, acc[p].y);
        }
    }
}

#else

#if PIE_BM == 64 && PIE_BN == 32
#define PIE_RPL 4u
#elif (PIE_BM == 64 && PIE_BN == 16) || (PIE_BM == 32 && PIE_BN == 32)
#define PIE_RPL 2u
#else
#define PIE_RPL 1u
#endif
#define PIE_QUADS (uint(PIE_BN) / 4u)
#define PIE_ROWS4 (uint(PIE_BM) / 4u)

groupshared float4 xs[PIE_BK * PIE_ROWS4];
groupshared float4 ws[PIE_BK * PIE_QUADS];

[numthreads(32, 2, PIE_WM)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint row0 = group.y * uint(PIE_BM);
    uint col0 = group.x * uint(PIE_BN);
    uint k = uint(max(params.k, 0));
    uint n = uint(max(params.n, 0));
    uint m = uint(max(params.m, 0));
    uint cpw = PIE_CODES_PER_WORD;
    uint wpr = k / cpw;
    uint gpr = k / uint(PIE_GROUP);
    uint words_per_block = PIE_BK / cpw;
    uint cq = flat % PIE_QUADS;
    uint rg = flat / PIE_QUADS;

    float4 acc[PIE_RPL];
    [ForceUnroll] for (uint i = 0u; i < PIE_RPL; i++) acc[i] = float4(0.0, 0.0, 0.0, 0.0);
    for (uint k0 = 0u; k0 < k; k0 += PIE_BK) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = flat; i < PIE_ROWS4 * (PIE_BK / 2u); i += PIE_THREADS) {
            uint rq = i / (PIE_BK / 2u);
            uint kk = (i - rq * (PIE_BK / 2u)) * 2u;
            float4 lo = float4(0.0, 0.0, 0.0, 0.0);
            float4 hi = float4(0.0, 0.0, 0.0, 0.0);
            [ForceUnroll] for (uint j = 0u; j < 4u; j++) {
                uint row = row0 + rq * 4u + j;
                if (row < m) {
                    uint e = row * k + k0 + kk;
                    lo[j] = bf16_at(x, e);
                    hi[j] = bf16_at(x, e + 1u);
                }
            }
            xs[kk * PIE_ROWS4 + rq] = lo;
            xs[(kk + 1u) * PIE_ROWS4 + rq] = hi;
        }
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < PIE_QUADS * words_per_block; i += PIE_THREADS) {
            uint cq4 = i / words_per_block;
            uint wi = i - cq4 * words_per_block;
            uint col = col0 + cq4 * 4u;
            uint4 word = uint4(0u, 0u, 0u, 0u);
            float4 sc = float4(0.0, 0.0, 0.0, 0.0);
            float4 bi = float4(0.0, 0.0, 0.0, 0.0);
            [ForceUnroll] for (uint j = 0u; j < 4u; j++) {
                uint c = col + j;
                if (c < n) {
                    word[j] = w[c * wpr + k0 / cpw + wi];
                    sc[j] = bf16_at(scales, c * gpr + g);
                    bi[j] = bf16_at(biases, c * gpr + g);
                }
            }
            for (uint q = 0u; q < cpw; q++) {
                float4 code = float4(affine_code(word.x, q), affine_code(word.y, q),
                                     affine_code(word.z, q), affine_code(word.w, q));
                ws[(wi * cpw + q) * PIE_QUADS + cq4] = code * sc + bi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint kk = 0u; kk < PIE_BK; kk++) {
            float4 wv = ws[kk * PIE_QUADS + cq];
#if PIE_BM == 64 && PIE_BN == 32
            float4 xv = xs[kk * PIE_ROWS4 + rg];
            acc[0] += xv.x * wv;
            acc[1] += xv.y * wv;
            acc[2] += xv.z * wv;
            acc[3] += xv.w * wv;
#else
            [ForceUnroll] for (uint i = 0u; i < PIE_RPL; i++) {
                uint r = rg * PIE_RPL + i;
                acc[i] += xs[kk * PIE_ROWS4 + r / 4u][r & 3u] * wv;
            }
#endif
        }
    }
    uint col = col0 + cq * 4u;
    [ForceUnroll] for (uint i = 0u; i < PIE_RPL; i++) {
        uint row = row0 + rg * PIE_RPL + i;
        if (row < m) {
            if (col < n) {
                y[(row * n + col) >> 1] = pack_bf16(acc[i].x, acc[i].y);
            }
            if (col + 2u < n) {
                y[(row * n + col + 2u) >> 1] = pack_bf16(acc[i].z, acc[i].w);
            }
        }
    }
}

#endif

// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_2_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_2_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_2_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16 PIE_TARGET=wgsl
// pie:instantiate linear_qmm_quad_bf16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32 PIE_TARGET=wgsl
