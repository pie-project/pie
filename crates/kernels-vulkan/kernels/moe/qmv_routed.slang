#include "common/bf16.slang"
#if defined(PIE_MXFP4)
#include "common/mxfp4.slang"
#define PIE_CHUNK 32u
#elif defined(PIE_DENSE)
#define PIE_CHUNK 4u
#else
#include "common/affine.slang"
#define PIE_CHUNK uint(PIE_CODES_PER_WORD)
#endif

#define PIE_LANES 32u
#define PIE_ROWS 8u

#if defined(PIE_DENSE)
PIE_BUFFER_RO(0, bank);
PIE_BUFFER_RO(1, x);
PIE_BUFFER_RW(2, y);
[[vk::binding(3)]] StructuredBuffer<int> expert_ids;
#else
#if defined(PIE_MXFP4)
[[vk::binding(0)]] StructuredBuffer<uint4> w4;
[[vk::binding(1)]] StructuredBuffer<uint8_t> scales;
[[vk::binding(3)]] StructuredBuffer<uint> x2;
#else
[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
PIE_BUFFER_RO(3, x);
#endif
PIE_BUFFER_RW(4, y);
PIE_BUFFER_RO(5, bias);
[[vk::binding(6)]] StructuredBuffer<int> expert_ids;
#endif

struct Push {
    int in_vec_size;
    int out_vec_size;
    int x_slot_stride;
    int x_row_stride;
    int slots_per_row;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partial[PIE_ROWS][PIE_LANES];

float chunk_dot(uint e, uint out_row, uint k, uint x_base) {
    uint in_size = uint(pc.in_vec_size);
    uint out_size = uint(pc.out_vec_size);
    float acc = 0.0;
#if defined(PIE_DENSE)
    uint at = (e * out_size + out_row) * in_size + k;
    [ForceUnroll] for (uint j = 0u; j < PIE_CHUNK; j++) {
        acc += PIE_LOAD(x[x_base + k + j]) * PIE_LOAD(bank[at + j]);
    }
#elif defined(PIE_MXFP4)

    uint blocks_per_row = in_size / 32u;
    uint at = (e * out_size + out_row) * blocks_per_row + k / 32u;
    uint4 words = w4[at];
    float scale = mxfp4_block_scale(uint(scales[at]));
    uint xb = (x_base + k) / 2u;
    uint ws[4] = { words.x, words.y, words.z, words.w };
    [ForceUnroll] for (uint wi = 0u; wi < 4u; wi++) {
        [ForceUnroll] for (uint p = 0u; p < 4u; p++) {
            uint pair = x2[xb + wi * 4u + p];
            acc += asfloat(pair << 16u) * mxfp4_code((ws[wi] >> (8u * p)) & 0xfu);
            acc += asfloat(pair & 0xffff0000u) * mxfp4_code((ws[wi] >> (8u * p + 4u)) & 0xfu);
        }
    }
    acc *= scale;
#else
    uint words_per_row = in_size / PIE_CHUNK;
    uint word = w[(e * out_size + out_row) * words_per_row + k / PIE_CHUNK];
    uint groups_per_row = in_size / uint(PIE_GROUP);
    uint sg = (e * out_size + out_row) * groups_per_row + k / uint(PIE_GROUP);
    float scale = PIE_LOAD(scales[sg]);
    float zero = PIE_LOAD(biases[sg]);
    float xs[PIE_CODES_PER_WORD];
    [ForceUnroll] for (uint j = 0u; j < PIE_CHUNK; j++) {
        xs[j] = PIE_LOAD(x[x_base + k + j]);
    }
    acc = pie_affine_word_dot(word, xs, scale, zero);
#endif
    return acc;
}

[numthreads(32, 8, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lane = local.x;
    uint local_out = local.y;
    uint row = group.x;
    uint block = group.y;
    uint slot = group.z;
    uint out_row = block * PIE_ROWS + local_out;
    uint out_size = uint(pc.out_vec_size);
    uint in_size = uint(pc.in_vec_size);
    bool active_out = out_row < out_size;

    uint sel = row * uint(pc.slots_per_row) + slot;
    int expert = expert_ids[sel];

    if (expert < 0) return;
    uint e = uint(expert);
    uint x_base = row * uint(pc.x_row_stride) + slot * uint(pc.x_slot_stride);

    float acc = 0.0;
    if (active_out) {
        for (uint k = lane * PIE_CHUNK; k < in_size; k += PIE_LANES * PIE_CHUNK) {
            acc += chunk_dot(e, out_row, k, x_base);
        }
    }

    partial[local_out][lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 16u; step > 0u; step >>= 1u) {
        if (lane < step) partial[local_out][lane] += partial[local_out][lane + step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && active_out) {
        acc = partial[local_out][0];
#if defined(PIE_BIASED)
        acc += PIE_LOAD(bias[e * out_size + out_row]);
#endif
        y[sel * out_size + out_row] = PIE_STORE(acc);
    }
}

// pie:instantiate mxfp4_qmv_routed_bfloat16_gs_32_b_4 PIE_MXFP4=1
// pie:instantiate mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate affine_qmv_routed_bfloat16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate select_gemv_bfloat16 PIE_DENSE=1
