#include "common/bf16.slang"
#if defined(PIE_MXFP4)
#include "common/mxfp4.slang"
#else
#include "common/affine.slang"
#endif

#define CM_BK 32u

[[vk::binding(0)]] StructuredBuffer<uint> w;
#if defined(PIE_MXFP4)
[[vk::binding(1)]] StructuredBuffer<uint8_t> scales;
#else
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
#endif
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
#if defined(PIE_BIASED)
PIE_BUFFER_RO(5, bias);
#endif
[[vk::binding(6)]] StructuredBuffer<int> tile_expert;

struct Push {
    int k;
    int n;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if defined(PIE_COOPMAT)
using namespace linalg;
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

#define PIE_BN 64u
#define CM_THREADS 128u
#define CM_SG_ROWS (uint(PIE_BM) / 16u)
#define CM_SG_COLS (4u / CM_SG_ROWS)
#define CM_STRIP (PIE_BN / CM_SG_COLS)
#define CM_NACC (CM_STRIP / 16u)

groupshared half sa[uint(PIE_BM) * CM_BK];
groupshared half sb[PIE_BN * CM_BK];
groupshared float sc[4u * 256u];
groupshared float row_scale[uint(PIE_BM)];
groupshared float row_max[uint(PIE_BM) * 2u];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lane = local.x;
    uint sg = local.y;
    uint flat = lane + 32u * sg;
    uint tile = group.y;
    int expert = tile_expert[tile];
    if (expert < 0) return;
    uint e = uint(expert);
    uint row0 = tile * uint(PIE_BM);
    uint col0 = group.x * PIE_BN;
    uint k = uint(pc.k);
    uint n = uint(pc.n);
    uint sg_r = sg / CM_SG_COLS;
    uint sg_c = sg - sg_r * CM_SG_COLS;

    if (flat < uint(PIE_BM) * 2u) {
        uint r = flat >> 1u;
        uint half_ = flat & 1u;
        float amax = 0.0;
        for (uint kk = half_; kk < k; kk += 2u) {
            amax = max(amax, abs(PIE_LOAD(x[(row0 + r) * k + kk])));
        }
        row_max[flat] = amax;
    }
    GroupMemoryBarrierWithGroupSync();
    if (flat < uint(PIE_BM)) {
        float amax = max(row_max[flat * 2u], row_max[flat * 2u + 1u]);
        float scale = 1.0;
        if (amax > 32768.0 && !isinf(amax) && !isnan(amax)) {
            scale = exp2(-ceil(log2(amax / 32768.0)));
        }
        row_scale[flat] = scale;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAcc acc[CM_NACC];
    [ForceUnroll] for (uint ni = 0u; ni < CM_NACC; ni++) acc[ni] = MatAcc(0.0);

    for (uint k0 = 0u; k0 < k; k0 += CM_BK) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = flat; i < uint(PIE_BM) * CM_BK; i += CM_THREADS) {
            uint r = i / CM_BK;
            uint kk = i - r * CM_BK;
            sa[i] = half(PIE_LOAD(x[(row0 + r) * k + k0 + kk]) * row_scale[r]);
        }
#if defined(PIE_MXFP4)

        for (uint i = flat; i < PIE_BN * 4u; i += CM_THREADS) {
            uint c = i >> 2u;
            uint wi = i & 3u;
            uint col = col0 + c;
            if (col < n) {
                uint row = e * n + col;
                uint word = w[row * (k / 8u) + k0 / 8u + wi];
                float scale = mxfp4_block_scale(uint(scales[row * (k / 32u) + k0 / 32u]));
                [ForceUnroll] for (uint j = 0u; j < 8u; j++) {
                    sb[c * CM_BK + wi * 8u + j] = half(mxfp4_code((word >> (4u * j)) & 0xfu) * scale);
                }
            } else {
                [ForceUnroll] for (uint j = 0u; j < 8u; j++) sb[c * CM_BK + wi * 8u + j] = half(0.0);
            }
        }
#else
        uint cpw = uint(PIE_CODES_PER_WORD);
        uint words_per_block = CM_BK / cpw;
        uint wpr = k * uint(PIE_BITS) / 32u;
        uint gpr = k / uint(PIE_GROUP);
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < PIE_BN * words_per_block; i += CM_THREADS) {
            uint c = i / words_per_block;
            uint wi = i - c * words_per_block;
            uint col = col0 + c;
            if (col < n) {
                uint row = e * n + col;
                uint word = w[row * wpr + k0 / cpw + wi];
                float s = PIE_LOAD(scales[row * gpr + g]);
                float b = PIE_LOAD(biases[row * gpr + g]);
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    sb[c * CM_BK + wi * cpw + q] = half(pie_affine_code(word, q) * s + b);
                }
            } else {
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) sb[c * CM_BK + wi * cpw + q] = half(0.0);
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < CM_BK; kk += 16u) {
            MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sa, sg_r * 16u * CM_BK + kk, CM_BK);
            [ForceUnroll] for (uint ni = 0u; ni < CM_NACC; ni++) {
                uint cb = sg_c * CM_STRIP + ni * 16u;
                MatB bm = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sb, cb * CM_BK + kk, CM_BK);
                acc[ni] = coopMatMulAdd<float, false>(a, bm, acc[ni]);
            }
        }
    }

    [ForceUnroll] for (uint ni = 0u; ni < CM_NACC; ni++) {
        GroupMemoryBarrierWithGroupSync();
        acc[ni].Store<CoopMatMatrixLayout.RowMajor>(sc, sg * 256u, 16u);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < 8u; i++) {
            uint el = lane + i * 32u;
            uint r = el >> 4u;
            uint c = el & 15u;
            uint row = row0 + sg_r * 16u + r;
            uint col = col0 + sg_c * CM_STRIP + ni * 16u + c;
            if (col < n) {
                float v = sc[sg * 256u + el] / row_scale[sg_r * 16u + r];
#if defined(PIE_BIASED)
                v += PIE_LOAD(bias[e * n + col]);
#endif
                y[row * n + col] = PIE_STORE(v);
            }
        }
    }
}

#else

#define PIE_BN 32u
#define PIE_THREADS 128u
#define PIE_PER ((uint(PIE_BM) * PIE_BN) / PIE_THREADS)

groupshared float xs[uint(PIE_BM) * CM_BK];
groupshared float ws[PIE_BN * CM_BK];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y;
    uint tile = group.y;
    int expert = tile_expert[tile];
    if (expert < 0) return;
    uint e = uint(expert);
    uint row0 = tile * uint(PIE_BM);
    uint col0 = group.x * PIE_BN;
    uint k = uint(pc.k);
    uint n = uint(pc.n);

    float acc[PIE_PER];
    [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) acc[p] = 0.0;

    for (uint k0 = 0u; k0 < k; k0 += CM_BK) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = flat; i < uint(PIE_BM) * CM_BK; i += PIE_THREADS) {
            uint r = i / CM_BK;
            uint kk = i - r * CM_BK;
            xs[i] = PIE_LOAD(x[(row0 + r) * k + k0 + kk]);
        }
#if defined(PIE_MXFP4)
        for (uint i = flat; i < PIE_BN * 4u; i += PIE_THREADS) {
            uint c = i >> 2u;
            uint wi = i & 3u;
            uint col = col0 + c;
            if (col < n) {
                uint row = e * n + col;
                uint word = w[row * (k / 8u) + k0 / 8u + wi];
                float scale = mxfp4_block_scale(uint(scales[row * (k / 32u) + k0 / 32u]));
                [ForceUnroll] for (uint j = 0u; j < 8u; j++) {
                    ws[c * CM_BK + wi * 8u + j] = mxfp4_code((word >> (4u * j)) & 0xfu) * scale;
                }
            } else {
                [ForceUnroll] for (uint j = 0u; j < 8u; j++) ws[c * CM_BK + wi * 8u + j] = 0.0;
            }
        }
#else
        uint cpw = uint(PIE_CODES_PER_WORD);
        uint words_per_block = CM_BK / cpw;
        uint wpr = k * uint(PIE_BITS) / 32u;
        uint gpr = k / uint(PIE_GROUP);
        uint g = k0 / uint(PIE_GROUP);
        for (uint i = flat; i < PIE_BN * words_per_block; i += PIE_THREADS) {
            uint c = i / words_per_block;
            uint wi = i - c * words_per_block;
            uint col = col0 + c;
            if (col < n) {
                uint row = e * n + col;
                uint word = w[row * wpr + k0 / cpw + wi];
                float s = PIE_LOAD(scales[row * gpr + g]);
                float b = PIE_LOAD(biases[row * gpr + g]);
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    ws[c * CM_BK + wi * cpw + q] = pie_affine_code(word, q) * s + b;
                }
            } else {
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) ws[c * CM_BK + wi * cpw + q] = 0.0;
            }
        }
#endif
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) {
            uint idx = flat + p * PIE_THREADS;
            uint r = idx / PIE_BN;
            uint c = idx - r * PIE_BN;
            float sum = 0.0;
            [ForceUnroll] for (uint kk = 0u; kk < CM_BK; kk++) {
                sum += xs[r * CM_BK + kk] * ws[c * CM_BK + kk];
            }
            acc[p] += sum;
        }
    }

    [ForceUnroll] for (uint p = 0u; p < PIE_PER; p++) {
        uint idx = flat + p * PIE_THREADS;
        uint r = idx / PIE_BN;
        uint c = idx - r * PIE_BN;
        uint row = row0 + r;
        uint col = col0 + c;
        if (col < n) {
            float v = acc[p];
#if defined(PIE_BIASED)
            v += PIE_LOAD(bias[e * n + col]);
#endif
            y[row * n + col] = PIE_STORE(v);
        }
    }
}
#endif

// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_16 PIE_BM=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_16_cm @coopmat PIE_BM=16 PIE_MXFP4=1 PIE_COOPMAT=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_32 PIE_BM=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_32_cm @coopmat PIE_BM=32 PIE_MXFP4=1 PIE_COOPMAT=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_64 PIE_BM=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bf16_gs_32_b_4_bm_64_cm @coopmat PIE_BM=64 PIE_MXFP4=1 PIE_COOPMAT=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_16 PIE_BM=16 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_16_cm @coopmat PIE_BM=16 PIE_MXFP4=1 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_32 PIE_BM=32 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_32_cm @coopmat PIE_BM=32 PIE_MXFP4=1 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_64 PIE_BM=64 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bf16_gs_32_b_4_bm_64_cm @coopmat PIE_BM=64 PIE_MXFP4=1 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=64 PIE_BITS=4 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=64 PIE_BITS=4 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_4_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=64 PIE_BITS=4 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_4_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=64 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=64 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_64_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=64 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_16 PIE_BM=16 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_32 PIE_BM=32 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_64 PIE_BM=64 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_64_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_16 PIE_BM=16 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=32 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_32 PIE_BM=32 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=32 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_64 PIE_BM=64 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_32_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=32 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_16 PIE_BM=16 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_32 PIE_BM=32 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_64 PIE_BM=64 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_32_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_16 PIE_BM=16 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=128 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_32 PIE_BM=32 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=128 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_64 PIE_BM=64 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmm_t_routed_bf16_gs_128_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=128 PIE_BITS=2 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_16 PIE_BM=16 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_16_cm @coopmat PIE_BM=16 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_32 PIE_BM=32 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_32_cm @coopmat PIE_BM=32 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_64 PIE_BM=64 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmm_t_routed_bias_bf16_gs_128_b_2_bm_64_cm @coopmat PIE_BM=64 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1 PIE_COOPMAT=1
