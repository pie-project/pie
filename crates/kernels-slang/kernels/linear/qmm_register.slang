#include "prim/affine.slang"
#include "prim/params.slang"
#include "prim/slot16.slang"
// y = x * dequant(w)^T, register-tiled: each invocation holds an 8x4 block of
// the output, and each k step reads two packed quads of activations and one
// of weight codes from workgroup memory for 32 multiply-adds. A GPU whose
// workgroup memory is slow next to its ALUs (Adreno) runs the plain tiled
// point, which reads two scalars per multiply-add, several times slower.

#ifndef PIE_BM
#error "a register-tiled qmm is compiled at a row tile: -DPIE_BM=64"
#endif

#define PIE_BN 64
#ifndef PIE_BK
#define PIE_BK 32
#endif
#define PIE_TX (PIE_BN / 4)
#define PIE_TY (PIE_BM / 8)
#define PIE_QUADS (PIE_BM / 4)
#define PIE_THREADS (PIE_TX * PIE_TY)
#if PIE_GROUP % PIE_BK != 0
#error "a K block must not straddle a scale group"
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_SLOTS_RO(1, scales);
PIE_SLOTS_RO(2, biases);
PIE_SLOTS_RO(3, x);
PIE_SLOTS_RW(4, y);

struct Params {
    int k;
    int n;
    int m;
};
PIE_PARAMS(5, Params, params);

// Workgroup memory is what Adreno pays for most, so both tiles stay packed
// there: activations as their bf16 bits (four rows of one k in a uint2),
// weights as their code words (four neighbouring columns' word in a uint4),
// widened and dequantized in registers. A K block never straddles a scale
// group, so each column's scale and bias hold for the whole block.
groupshared uint2 xs[PIE_BK * PIE_QUADS];
groupshared uint4 ws[(PIE_BK / PIE_CODES_PER_WORD) * PIE_TX];

[numthreads(PIE_TX, PIE_TY, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + PIE_TX * local.y;
    uint row0 = group.y * uint(PIE_BM);
    uint col0 = group.x * uint(PIE_BN);
    uint k = uint(params.k);
    uint n = uint(params.n);
    uint m = uint(params.m);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint wpr = k * uint(PIE_BITS) / 32u;
    uint gpr = k / uint(PIE_GROUP);
    uint words_per_block = uint(PIE_BK) / cpw;
    uint mycol = col0 + local.x * 4u;

    float4 acc[8];
    [ForceUnroll] for (uint i = 0u; i < 8u; i++) acc[i] = float4(0.0);

    for (uint k0 = 0u; k0 < k; k0 += uint(PIE_BK)) {
        GroupMemoryBarrierWithGroupSync();
        // Activations: four rows of one k per invocation and pass, raw bits.
        for (uint e = flat; e < uint(PIE_BK * PIE_QUADS); e += uint(PIE_THREADS)) {
            uint kk = e / uint(PIE_QUADS);
            uint quad = e - kk * uint(PIE_QUADS);
            uint h[4];
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint row = row0 + quad * 4u + r;
                h[r] = row < m ? uint(x[row * k + k0 + kk]) : 0u;
            }
            xs[e] = uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
        }
        // Weights: one word of four neighbouring columns per invocation.
        for (uint e = flat; e < uint(PIE_TX) * words_per_block; e += uint(PIE_THREADS)) {
            uint wi = e / uint(PIE_TX);
            uint quad = e - wi * uint(PIE_TX);
            uint4 words = uint4(0u);
            [ForceUnroll] for (uint c = 0u; c < 4u; c++) {
                uint col = col0 + quad * 4u + c;
                if (col < n) words[c] = w[col * wpr + k0 / cpw + wi];
            }
            ws[e] = words;
        }
        uint g = k0 / uint(PIE_GROUP);
        float4 s;
        float4 b;
        [ForceUnroll] for (uint c = 0u; c < 4u; c++) {
            uint col = min(mycol + c, n - 1u);
            s[c] = slot_at(scales, col * gpr + g);
            b[c] = slot_at(biases, col * gpr + g);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint wi = 0u; wi < words_per_block; wi++) {
            uint4 cw = ws[wi * uint(PIE_TX) + local.x];
            [ForceUnroll] for (uint q = 0u; q < uint(PIE_CODES_PER_WORD); q++) {
                uint kk = wi * cpw + q;
                float4 bw = float4(affine_code(cw.x, q), affine_code(cw.y, q),
                                   affine_code(cw.z, q), affine_code(cw.w, q)) * s + b;
                uint2 p0 = xs[kk * uint(PIE_QUADS) + local.y * 2u];
                uint2 p1 = xs[kk * uint(PIE_QUADS) + local.y * 2u + 1u];
                acc[0] += asfloat(p0.x << 16) * bw;
                acc[1] += asfloat(p0.x & 0xffff0000u) * bw;
                acc[2] += asfloat(p0.y << 16) * bw;
                acc[3] += asfloat(p0.y & 0xffff0000u) * bw;
                acc[4] += asfloat(p1.x << 16) * bw;
                acc[5] += asfloat(p1.x & 0xffff0000u) * bw;
                acc[6] += asfloat(p1.y << 16) * bw;
                acc[7] += asfloat(p1.y & 0xffff0000u) * bw;
            }
        }
    }

    [ForceUnroll] for (uint r = 0u; r < 8u; r++) {
        uint row = row0 + local.y * 8u + r;
        if (row >= m) continue;
        [ForceUnroll] for (uint c = 0u; c < 4u; c++) {
            uint col = mycol + c;
            if (col < n) slot_put(y, row * n + col, acc[r][c]);
        }
    }
}

// pie:instantiate linear_qmm_register_bf16_gs_32_b_2_bm_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_32_b_2_bm_64 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_32_b_4_bm_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_32_b_4_bm_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_32_b_8_bm_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_32_b_8_bm_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_2_bm_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_2_bm_64 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_4_bm_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_4_bm_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_8_bm_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_64_b_8_bm_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_2_bm_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_2_bm_64 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_4_bm_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_4_bm_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_8_bm_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_bf16_gs_128_b_8_bm_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_2_bm_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_2_bm_64 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_4_bm_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_4_bm_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_8_bm_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_32_b_8_bm_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_2_bm_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_2_bm_64 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_4_bm_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_4_bm_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_8_bm_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_64_b_8_bm_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_2_bm_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_2_bm_64 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_4_bm_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_4_bm_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_8_bm_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BK=16 PIE_TARGET=spirv
// pie:instantiate linear_qmm_register_k16_bf16_gs_128_b_8_bm_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BK=16 PIE_TARGET=spirv
