#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/slot16.slang"

// `x . w^T` register-tiled, as `conv/conv3d_tiled.slang` tiles a
// convolution: a lane eight rows by four or eight columns, the contraction in
// 16-wide steps staged in workgroup memory. `k` is a multiple of 16. SPIR-V
// only: its workgroup memory is past WGSL's, and the result lands a value a
// 16-bit slot.

#ifndef PIE_BN
#define PIE_BN 64
#endif
#ifndef PIE_CN
#define PIE_CN 4
#endif
#define PIE_BK 16
#define PIE_TX (PIE_BN / PIE_CN)
#define PIE_TY (256 / PIE_TX)
#define PIE_BM (PIE_TY * 8)
#define PIE_THREADS 256

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> w;
PIE_SLOTS_RW(2, out_);

struct Params {
    int m;
    int n;
    int k;
};
PIE_PARAMS(3, Params, pc);

groupshared float4 xs[PIE_BK * (PIE_BM / 4)];
groupshared float4 ws[PIE_BK * (PIE_BN / 4)];

[shader("compute")]
[numthreads(PIE_TX, PIE_TY, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + PIE_TX * local.y;
    uint row0 = group.y * uint(PIE_BM);
    uint col0 = group.x * uint(PIE_BN);
    uint m = uint(pc.m);
    uint n = uint(pc.n);
    uint words = uint(pc.k) / 2u;

    float4 acc[8][PIE_CN / 4];
    [ForceUnroll] for (uint i = 0u; i < 8u; i++) {
        [ForceUnroll] for (uint j = 0u; j < uint(PIE_CN / 4); j++) acc[i][j] = float4(0.0);
    }

    for (uint k0 = 0u; k0 < uint(pc.k); k0 += uint(PIE_BK)) {
        GroupMemoryBarrierWithGroupSync();
        // The activations: a row's sixteen values as two runs of four words.
        for (uint e = flat; e < uint(PIE_BM * 2); e += uint(PIE_THREADS)) {
            uint r = e / 2u;
            uint half_ = e % 2u;
            uint row = row0 + r;
            uint base = row * words + k0 / 2u + half_ * 4u;
            uint at = r / 4u;
            uint lane = r % 4u;
            [ForceUnroll] for (uint q = 0u; q < 4u; q++) {
                uint word = 0u;
                if (row < m) {
                    word = x[base + q];
                }
                uint kk = half_ * 8u + q * 2u;
                xs[kk * uint(PIE_BM / 4) + at][lane] = bf16_to_f32(word & 0xffffu);
                xs[(kk + 1u) * uint(PIE_BM / 4) + at][lane] = bf16_to_f32(word >> 16);
            }
        }
        // The weights: a column's sixteen values as four runs of two words.
        for (uint e = flat; e < uint(PIE_BN * 4); e += uint(PIE_THREADS)) {
            uint c = e / 4u;
            uint q = e % 4u;
            uint col = col0 + c;
            uint base = col * words + k0 / 2u + q * 2u;
            uint at = c / 4u;
            uint lane = c % 4u;
            [ForceUnroll] for (uint p = 0u; p < 2u; p++) {
                uint word = 0u;
                if (col < n) {
                    word = w[base + p];
                }
                uint kk = q * 4u + p * 2u;
                ws[kk * uint(PIE_BN / 4) + at][lane] = bf16_to_f32(word & 0xffffu);
                ws[(kk + 1u) * uint(PIE_BN / 4) + at][lane] = bf16_to_f32(word >> 16);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < uint(PIE_BK); kk++) {
            float4 a0 = xs[kk * uint(PIE_BM / 4) + local.y * 2u];
            float4 a1 = xs[kk * uint(PIE_BM / 4) + local.y * 2u + 1u];
            [ForceUnroll] for (uint j = 0u; j < uint(PIE_CN / 4); j++) {
                float4 b = ws[kk * uint(PIE_BN / 4) + local.x * uint(PIE_CN / 4) + j];
                acc[0][j] += a0.x * b;
                acc[1][j] += a0.y * b;
                acc[2][j] += a0.z * b;
                acc[3][j] += a0.w * b;
                acc[4][j] += a1.x * b;
                acc[5][j] += a1.y * b;
                acc[6][j] += a1.z * b;
                acc[7][j] += a1.w * b;
            }
        }
    }

    [ForceUnroll] for (uint r = 0u; r < 8u; r++) {
        uint row = row0 + local.y * 8u + r;
        if (row >= m) continue;
        [ForceUnroll] for (uint c = 0u; c < uint(PIE_CN); c++) {
            uint col = col0 + local.x * uint(PIE_CN) + c;
            if (col >= n) continue;
            slot_put(out_, row * n + col, acc[r][c / 4u][c % 4u]);
        }
    }
}

// pie:instantiate linear_gemm_register_bf16 PIE_CN=8 PIE_TARGET=spirv
// pie:instantiate linear_gemm_register_bn_32_bf16 PIE_BN=32 PIE_TARGET=spirv
