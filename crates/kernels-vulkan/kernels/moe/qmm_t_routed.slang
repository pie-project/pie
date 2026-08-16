// Routed quantized GEMM for sorted MoE rows.
//
// Ported from the routed Metal qmm_t instantiations. This body keeps the expert
// choice per row tile and the expert-major weight layout; it uses a scalar
// tiled implementation rather than sharing dense `quant/qmm_t`, so the routed
// family can stand alone.
//
// The Slang port changed spellings only. Slang has no workgroup-size builtin,
// so the two walk strides are `PIE_TILE_X`/`PIE_TILE_Y`, stated once beside
// `[numthreads]` so the declaration and the stride cannot drift apart. A stride
// smaller than the workgroup width would compute some output twice and, worse,
// a stride LARGER would leave holes reading as whatever the arena held.

#include "common/bf16.slang"
#include "common/affine.slang"
#include "common/mxfp4.slang"

#ifndef PIE_BM
#error "routed qmm is compiled with PIE_BM"
#endif
#ifndef PIE_BN
#error "routed qmm is compiled with PIE_BN"
#endif

/// The workgroup, and therefore the two walk strides.
#define PIE_TILE_X 16u
#define PIE_TILE_Y 16u

[[vk::binding(0)]] StructuredBuffer<uint> w;
#if defined(PIE_MXFP4)
[[vk::binding(1)]] StructuredBuffer<uint8_t> exponents;
PIE_BUFFER_RO(2, x);
PIE_BUFFER_RW(3, y);
PIE_BUFFER_RO(4, bias);
[[vk::binding(5)]] StructuredBuffer<int> tile_expert;
#else
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
[[vk::binding(5)]] StructuredBuffer<int> tile_expert;
#endif

struct Push { int K; int N; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

uint byte_at(uint byte_index) {
    uint word = w[byte_index >> 2u];
    return (word >> ((byte_index & 3u) * 8u)) & 0xffu;
}

float weight_at(uint e, uint n, uint k) {
#if defined(PIE_MXFP4)
    uint b = byte_at(e * uint(pc.N) * uint(pc.K / 2) + n * uint(pc.K / 2) + (k >> 1u));
    float code = (k & 1u) == 0u ? mxfp4_lo(b) : mxfp4_hi(b);
    uint s = e * uint(pc.N) * uint(pc.K / PIE_MXFP4_BLOCK)
           + n * uint(pc.K / PIE_MXFP4_BLOCK) + k / uint(PIE_MXFP4_BLOCK);
    return code * mxfp4_block_scale(uint(exponents[s]));
#else
    uint wi = e * uint(pc.N) * uint(pc.K * PIE_BITS / 32)
            + n * uint(pc.K * PIE_BITS / 32) + k / uint(PIE_CODES_PER_WORD);
    uint g = k / uint(PIE_GROUP);
    uint sg = e * uint(pc.N) * uint(pc.K / PIE_GROUP) + n * uint(pc.K / PIE_GROUP) + g;
    return pie_affine_code(w[wi], k & uint(PIE_CODES_PER_WORD - 1)) * PIE_LOAD(scales[sg]) + PIE_LOAD(biases[sg]);
#endif
}

[numthreads(16, 16, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint tile_m = group.y;
    int expert = tile_expert[tile_m];
    if (expert < 0) return;
    uint e = uint(expert);
    uint col0 = group.x * uint(PIE_BN);
    uint row0 = tile_m * uint(PIE_BM);

    for (uint rr = local.y; rr < uint(PIE_BM); rr += PIE_TILE_Y) {
        for (uint cc = local.x; cc < uint(PIE_BN); cc += PIE_TILE_X) {
            uint n = col0 + cc;
            if (n >= uint(pc.N)) continue;
            float acc = 0.0;
            for (uint k = 0u; k < uint(pc.K); k++) acc += PIE_LOAD(x[(row0 + rr) * uint(pc.K) + k]) * weight_at(e, n, k);
#if defined(PIE_MXFP4)
            acc += PIE_LOAD(bias[e * uint(pc.N) + n]);
#endif
            y[(row0 + rr) * uint(pc.N) + n] = PIE_STORE(acc);
        }
    }
}

// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_16 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_64 PIE_GROUP=128 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=16 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=32 PIE_BN=64
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=16
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=32
// pie:instantiate affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=8 PIE_BM=64 PIE_BN=64
// `PIE_FP16` below is INERT: no body in this file reads it, so these nine
// modules are byte-identical to their non-`fp16` siblings and the name is a
// promise the code does not keep. It is not a live defect, because
// `affine_qmm_t_routed_fp16`'s row states no operands and so cannot be bound
// by any driver -- but it is a trap laid for whoever states it. In the dense
// sibling `quant/qmm_t.slang` the `_fp16_precast` variants are NOT cosmetic:
// they bind a separate pre-cast fp16 activation buffer. A driver that selects
// this name because it has pre-cast its activations would hand fp16 bytes to a
// shader reading bf16 and get silent garbage.
//
// So: implement the pre-cast path here BEFORE the row is stated, or drop the
// nine names -- which cannot be done unilaterally, since `kernels-metal` has
// the same nine and the two tables are checked against each other.
// `scripts/vulkan-kernel-audit.py --defines` holds the line meanwhile: this
// one is allow-listed with its reason and any NEW inert define fails.
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_FP16=1
// pie:instantiate affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64 PIE_GROUP=64 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_FP16=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=16 PIE_BN=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=32 PIE_BN=64 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=16 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=32 PIE_MXFP4=1
// pie:instantiate mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64 PIE_GROUP=32 PIE_BITS=4 PIE_BM=64 PIE_BN=64 PIE_MXFP4=1
