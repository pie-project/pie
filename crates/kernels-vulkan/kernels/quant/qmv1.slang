#include "common/bf16.slang"
#include "common/affine.slang"

// y = dequant(w) x for one activation row, the decode shape. Half a 64-lane
// subgroup walks one output row in 128-bit loads of codes and activations,
// and the halves fold their sums with shuffles: no workgroup memory, no
// barrier, and no multiply-adds for vectors the fire does not carry. The
// shuffles fold 32 lanes, so the point needs subgroups of 32 or more.

#define PIE_HALF 32u
#define PIE_ROWS 8u
// The codes one 128-bit load carries; a load stays inside one scale group.
#define PIE_QUAD_CODES (128u / uint(PIE_BITS))
#if (128 / PIE_BITS) > PIE_GROUP
#error "a 128-bit load of codes must not straddle a scale group"
#endif

[[vk::binding(0)]] StructuredBuffer<uint4> w4;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
[[vk::binding(3)]] StructuredBuffer<uint4> x4;
PIE_BUFFER_RW(4, y);

struct Push {
    int in_vec_size;
    int out_vec_size;
    int vecs;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

// Eight bf16 activations from one uint4, in order.
void unpack8(uint4 pairs, out float xs[8]) {
    [ForceUnroll] for (uint p = 0u; p < 4u; p++) {
        xs[2u * p] = asfloat(pairs[p] << 16u);
        xs[2u * p + 1u] = asfloat(pairs[p] & 0xffff0000u);
    }
}

[numthreads(64, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lane = local.x % PIE_HALF;
    uint row = group.x * PIE_ROWS + local.y * 2u + local.x / PIE_HALF;
    uint k = uint(pc.in_vec_size);
    uint n = uint(pc.out_vec_size);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint quads = k * uint(PIE_BITS) / 128u;
    uint gpr = k / uint(PIE_GROUP);

    float acc = 0.0;
    if (row < n) {
        for (uint q = lane; q < quads; q += PIE_HALF) {
            uint4 words = w4[row * quads + q];
            uint kb = q * PIE_QUAD_CODES;
            float dot = 0.0;
            float xsum = 0.0;
            [ForceUnroll] for (uint at = 0u; at < PIE_QUAD_CODES; at += 8u) {
                float xs[8];
                unpack8(x4[(kb + at) / 8u], xs);
                [ForceUnroll] for (uint i = 0u; i < 8u; i++) {
                    uint code = at + i;
                    dot += pie_affine_code(words[code / cpw], code % cpw) * xs[i];
                    xsum += xs[i];
                }
            }
            uint g = kb / uint(PIE_GROUP);
            acc += PIE_LOAD(scales[row * gpr + g]) * dot + PIE_LOAD(biases[row * gpr + g]) * xsum;
        }
    }
    [ForceUnroll] for (uint step = PIE_HALF / 2u; step > 0u; step >>= 1u) {
        acc += subgroupShuffleXor(acc, step);
    }
    if (lane == 0u && row < n) y[row] = PIE_STORE(acc);
}

// pie:instantiate affine_qmv1_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv1_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv1_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv1_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv1_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv1_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
