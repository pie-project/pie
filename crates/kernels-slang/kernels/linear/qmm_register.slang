#include "prim/affine.slang"
#include "prim/params.slang"
#include "prim/slot16.slang"
// y = x * dequant(w)^T, register-tiled: each invocation holds a 4x4 block of
// the output, and each k step reads one float4 of activations and one of
// weights from workgroup memory for sixteen multiply-adds. A GPU whose
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
#define PIE_TY (PIE_BM / 4)
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

// Both tiles k-major, so a row of four outputs reads one float4.
groupshared float4 xs[PIE_BK * PIE_TY];
groupshared float4 ws[PIE_BK * PIE_TX];

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

    float4 acc[4];
    [ForceUnroll] for (uint i = 0u; i < 4u; i++) acc[i] = float4(0.0);

    for (uint k0 = 0u; k0 < k; k0 += uint(PIE_BK)) {
        GroupMemoryBarrierWithGroupSync();
        // Activations: four rows of one k per invocation and pass.
        for (uint e = flat; e < uint(PIE_BK * PIE_TY); e += uint(PIE_THREADS)) {
            uint kk = e / uint(PIE_TY);
            uint quad = e - kk * uint(PIE_TY);
            float4 v;
            [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
                uint row = row0 + quad * 4u + r;
                v[r] = row < m ? slot_at(x, row * k + k0 + kk) : 0.0;
            }
            xs[e] = v;
        }
        // Weights: one packed word of one column per invocation and pass.
        uint g = k0 / uint(PIE_GROUP);
        for (uint e = flat; e < uint(PIE_BN) * words_per_block; e += uint(PIE_THREADS)) {
            uint c = e / words_per_block;
            uint wi = e - c * words_per_block;
            uint col = col0 + c;
            uint word = 0u;
            float s = 0.0;
            float b = 0.0;
            if (col < n) {
                word = w[col * wpr + k0 / cpw + wi];
                s = slot_at(scales, col * gpr + g);
                b = slot_at(biases, col * gpr + g);
            }
            [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                uint kk = wi * cpw + q;
                uint at = kk * uint(PIE_TX) + c / 4u;
                float value = col < n ? affine_code(word, q) * s + b : 0.0;
                ws[at][c % 4u] = value;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < uint(PIE_BK); kk++) {
            float4 a = xs[kk * uint(PIE_TY) + local.y];
            float4 bw = ws[kk * uint(PIE_TX) + local.x];
            acc[0] += a.x * bw;
            acc[1] += a.y * bw;
            acc[2] += a.z * bw;
            acc[3] += a.w * bw;
        }
    }

    [ForceUnroll] for (uint r = 0u; r < 4u; r++) {
        uint row = row0 + local.y * 4u + r;
        if (row >= m) continue;
        [ForceUnroll] for (uint c = 0u; c < 4u; c++) {
            uint col = col0 + local.x * 4u + c;
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
