#include "common/bf16.slang"
#include "common/affine.slang"

#if !defined(PIE_COOPMAT)
#ifndef PIE_BM
#error "a tiled qmm is compiled at a row tile: -DPIE_BM=32"
#endif
#ifndef PIE_BN
#error "a tiled qmm is compiled at a column tile: -DPIE_BN=32"
#endif
#else
#define PIE_BM 64
#define PIE_BN 64
#endif

#define PIE_BK 32
#if PIE_BM < 16
#define PIE_WM 1
#else
#define PIE_WM 2
#endif
#define PIE_THREADS (32 * 2 * PIE_WM)
#define PIE_PER ((PIE_BM * PIE_BN) / PIE_THREADS)
#if (PIE_BM * PIE_BN) % PIE_THREADS != 0
#error "the tile is not a whole number of outputs per thread"
#endif
#if PIE_GROUP % PIE_BK != 0
#error "a K block must not straddle a scale group"
#endif

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);

struct Push {
    int k;
    int n;
    int m;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if defined(PIE_COOPMAT)

using namespace linalg;
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

#define CM_BM 64u
#define CM_BN 64u
#define CM_BK 32u
#define CM_THREADS 128u

groupshared half sa[CM_BM * CM_BK];
groupshared half sb[CM_BN * CM_BK];
groupshared float sc[4u * 256u];
groupshared float row_scale[CM_BM];
groupshared float row_max[CM_BM * 2u];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lane = local.x;
    uint sg = local.y;
    uint flat = lane + 32u * sg;
    uint row0 = group.y * CM_BM;
    uint col0 = group.x * CM_BN;
    uint k = uint(pc.k);
    uint n = uint(pc.n);
    uint m = uint(pc.m);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint wpr = k * uint(PIE_BITS) / 32u;
    uint gpr = k / uint(PIE_GROUP);
    uint words_per_block = CM_BK / cpw;

    {
        uint r = flat >> 1u;
        uint half_ = flat & 1u;
        float amax = 0.0;
        uint row = row0 + r;
        if (row < m) {
            for (uint kk = half_; kk < k; kk += 2u) {
                amax = max(amax, abs(PIE_LOAD(x[row * k + kk])));
            }
        }
        row_max[flat] = amax;
    }
    GroupMemoryBarrierWithGroupSync();
    if (flat < CM_BM) {
        float amax = max(row_max[flat * 2u], row_max[flat * 2u + 1u]);
        float scale = 1.0;
        if (amax > 32768.0 && !isinf(amax) && !isnan(amax)) {
            scale = exp2(-ceil(log2(amax / 32768.0)));
        }
        row_scale[flat] = scale;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAcc acc[4];
    [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) acc[ni] = MatAcc(0.0);

    for (uint k0 = 0u; k0 < k; k0 += CM_BK) {
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < (CM_BM * CM_BK) / CM_THREADS; i++) {
            uint e = flat + i * CM_THREADS;
            uint r = e / CM_BK;
            uint kk = e - r * CM_BK;
            uint row = row0 + r;
            sa[e] = half(row < m ? PIE_LOAD(x[row * k + k0 + kk]) * row_scale[r] : 0.0);
        }
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < CM_BN * words_per_block; i += CM_THREADS) {
            uint c = i / words_per_block;
            uint wi = i - c * words_per_block;
            uint col = col0 + c;
            if (col < n) {
                uint word = w[col * wpr + k0 / cpw + wi];
                float s = PIE_LOAD(scales[col * gpr + g]);
                float b = PIE_LOAD(biases[col * gpr + g]);
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    sb[c * CM_BK + wi * cpw + q] = half(pie_affine_code(word, q) * s + b);
                }
            } else {
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    sb[c * CM_BK + wi * cpw + q] = half(0.0);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < CM_BK; kk += 16u) {
            MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sa, sg * 16u * CM_BK + kk, CM_BK);
            [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) {
                MatB bm = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sb, ni * 16u * CM_BK + kk, CM_BK);
                acc[ni] = coopMatMulAdd<float, false>(a, bm, acc[ni]);
            }
        }
    }

    [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) {
        GroupMemoryBarrierWithGroupSync();
        acc[ni].Store<CoopMatMatrixLayout.RowMajor>(sc, sg * 256u, 16u);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < 8u; i++) {
            uint e = lane + i * 32u;
            uint r = e >> 4u;
            uint c = e & 15u;
            uint row = row0 + sg * 16u + r;
            uint col = col0 + ni * 16u + c;
            if (row < m && col < n) {
                y[row * n + col] = PIE_STORE(sc[sg * 256u + e] / row_scale[sg * 16u + r]);
            }
        }
    }
}

#else

groupshared float xs[PIE_BM * PIE_BK];
groupshared float ws[PIE_BN * PIE_BK];

[numthreads(32, 2, PIE_WM)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint row0 = group.y * uint(PIE_BM);
    uint col0 = group.x * uint(PIE_BN);
    uint k = uint(pc.k);
    uint n = uint(pc.n);
    uint m = uint(pc.m);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint wpr = k * uint(PIE_BITS) / 32u;
    uint gpr = k / uint(PIE_GROUP);
    uint words_per_block = uint(PIE_BK) / cpw;

    float acc[PIE_PER];
    [ForceUnroll] for (uint p = 0u; p < uint(PIE_PER); p++) acc[p] = 0.0;

    for (uint k0 = 0u; k0 < k; k0 += uint(PIE_BK)) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = flat; i < uint(PIE_BM * PIE_BK); i += uint(PIE_THREADS)) {
            uint r = i / uint(PIE_BK);
            uint kk = i - r * uint(PIE_BK);
            uint row = row0 + r;
            xs[i] = row < m ? PIE_LOAD(x[row * k + k0 + kk]) : 0.0;
        }
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < uint(PIE_BN) * words_per_block; i += uint(PIE_THREADS)) {
            uint c = i / words_per_block;
            uint wi = i - c * words_per_block;
            uint col = col0 + c;
            if (col < n) {
                uint word = w[col * wpr + k0 / cpw + wi];
                float s = PIE_LOAD(scales[col * gpr + g]);
                float b = PIE_LOAD(biases[col * gpr + g]);
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    ws[c * uint(PIE_BK) + wi * cpw + q] = pie_affine_code(word, q) * s + b;
                }
            } else {
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    ws[c * uint(PIE_BK) + wi * cpw + q] = 0.0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint p = 0u; p < uint(PIE_PER); p++) {
            uint idx = flat + p * uint(PIE_THREADS);
            uint r = idx / uint(PIE_BN);
            uint c = idx - r * uint(PIE_BN);
            float sum = 0.0;
            [ForceUnroll] for (uint kk = 0u; kk < uint(PIE_BK); kk++) {
                sum += xs[r * uint(PIE_BK) + kk] * ws[c * uint(PIE_BK) + kk];
            }
            acc[p] += sum;
        }
    }

    [ForceUnroll] for (uint p = 0u; p < uint(PIE_PER); p++) {
        uint idx = flat + p * uint(PIE_THREADS);
        uint r = idx / uint(PIE_BN);
        uint c = idx - r * uint(PIE_BN);
        uint row = row0 + r;
        uint col = col0 + c;
        if (row < m && col < n) {
            y[row * n + col] = PIE_STORE(acc[p]);
        }
    }
}

#endif

// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_2_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=2 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_8_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_8_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_2_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=2 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_8_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_8_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_2_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=2 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_8_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_8_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=8 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_bf16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_cm_bf16_gs_32_b_2 @coopmat PIE_COOPMAT=1 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmm_t_cm_bf16_gs_32_b_4 @coopmat PIE_COOPMAT=1 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmm_t_cm_bf16_gs_32_b_8 @coopmat PIE_COOPMAT=1 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmm_t_cm_bf16_gs_64_b_2 @coopmat PIE_COOPMAT=1 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_cm_bf16_gs_64_b_4 @coopmat PIE_COOPMAT=1 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_cm_bf16_gs_64_b_8 @coopmat PIE_COOPMAT=1 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmm_t_cm_bf16_gs_128_b_2 @coopmat PIE_COOPMAT=1 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmm_t_cm_bf16_gs_128_b_4 @coopmat PIE_COOPMAT=1 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmm_t_cm_bf16_gs_128_b_8 @coopmat PIE_COOPMAT=1 PIE_GROUP=128 PIE_BITS=8
