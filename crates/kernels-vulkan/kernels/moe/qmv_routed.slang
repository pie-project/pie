// Routed quantized GEMV for MoE expert slots.
//
// Ported from Metal's routed qmv instantiations. The Vulkan ABI compacts the
// routed matvec row to weight buffers, x/y, optional bias, expert ids, then
// push constants for the five scalar extents; the format is intentionally only
// affine g64/b4 plus MXFP4 g32/b4.
//
// The Slang port changed spellings, not arithmetic. `gl_WorkGroupSize.x` is
// gone -- Slang has no such builtin, and the lane count is `[numthreads]`'s own
// 32, stated once as `PIE_LANES` so the stride and the declaration cannot drift
// apart. That is the one place a port of this file could go quietly wrong: a
// stride smaller than the workgroup width sums some products twice.

#include "common/bf16.slang"
#include "common/affine.slang"
#include "common/mxfp4.slang"

/// Lanes in a row of the workgroup, and therefore the `k` stride.
#define PIE_LANES 32u
/// Output rows one workgroup covers.
#define PIE_ROWS 8u

[[vk::binding(0)]] StructuredBuffer<uint> w;
#if defined(PIE_MXFP4)
[[vk::binding(1)]] StructuredBuffer<uint8_t> scales;
#else
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
#endif
PIE_BUFFER_RO(3, x);
PIE_BUFFER_RW(4, y);
PIE_BUFFER_RO(5, bias);
[[vk::binding(6)]] StructuredBuffer<int> expert_ids;

struct Push {
    int in_vec_size;
    int out_vec_size;
    int x_slot_stride;
    int x_row_stride;
    int slots_per_row;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partial[8][32];

uint byte_at(uint byte_index) {
    uint word = w[byte_index >> 2u];
    return (word >> ((byte_index & 3u) * 8u)) & 0xffu;
}

float dequant_at(uint e, uint out_row, uint k) {
#if defined(PIE_MXFP4)
    uint byte_index = e * uint(pc.out_vec_size) * uint(pc.in_vec_size / 2)
                    + out_row * uint(pc.in_vec_size / 2) + (k >> 1u);
    uint b = byte_at(byte_index);
    float code = (k & 1u) == 0u ? mxfp4_lo(b) : mxfp4_hi(b);
    uint sg = e * uint(pc.out_vec_size) * uint(pc.in_vec_size / PIE_MXFP4_BLOCK)
            + out_row * uint(pc.in_vec_size / PIE_MXFP4_BLOCK) + k / uint(PIE_MXFP4_BLOCK);
    return code * mxfp4_block_scale(uint(scales[sg]));
#else
    uint word_index = e * uint(pc.out_vec_size) * uint(pc.in_vec_size * PIE_BITS / 32)
                    + out_row * uint(pc.in_vec_size * PIE_BITS / 32)
                    + k / uint(PIE_CODES_PER_WORD);
    uint code_i = k & uint(PIE_CODES_PER_WORD - 1);
    uint g = k / uint(PIE_GROUP);
    uint sg = e * uint(pc.out_vec_size) * uint(pc.in_vec_size / PIE_GROUP)
            + out_row * uint(pc.in_vec_size / PIE_GROUP) + g;
    return pie_affine_code(w[word_index], code_i) * PIE_LOAD(scales[sg]) + PIE_LOAD(biases[sg]);
#endif
}

[numthreads(32, 8, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lane = local.x;
    uint local_out = local.y;
    uint row = group.x;
    uint block = group.y;
    uint slot = group.z;
    uint out_row = block * PIE_ROWS + local_out;
    bool active_out = out_row < uint(pc.out_vec_size);

    uint sel = row * uint(pc.slots_per_row) + slot;
    int expert = expert_ids[sel];
    if (expert < 0) return;
    uint e = uint(expert);
    uint x_base = row * uint(pc.x_row_stride) + slot * uint(pc.x_slot_stride);

    float acc = 0.0;
    if (active_out) {
        for (uint k = lane; k < uint(pc.in_vec_size); k += PIE_LANES) {
            acc += PIE_LOAD(x[x_base + k]) * dequant_at(e, out_row, k);
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
        acc += PIE_LOAD(bias[e * uint(pc.out_vec_size) + out_row]);
#endif
        y[sel * uint(pc.out_vec_size) + out_row] = PIE_STORE(acc);
    }
}
// pie:instantiate affine_qmv_routed_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_routed_bias_bfloat16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4 PIE_MXFP4=1 PIE_BIASED=1
