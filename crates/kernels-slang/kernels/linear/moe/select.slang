#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/shuffle.slang"
#if defined(PIE_MXFP4)
#include "prim/mxfp4.slang"
#elif !defined(PIE_DENSE)
#include "prim/affine.slang"
#endif
#if !defined(PIE_WGSL)
#include "prim/slot16.slang"
#endif

// Routed matmuls one route at a time (`linear.moe_select`, its bias
// and quantized forms): a workgroup lands eight output rows of one route,
// a 32-lane strip a row walking the contraction in chunks, the strip folded
// through workgroup memory or, at WGSL's subgroup tier, by shuffles (that
// tier also stages the route's activations in workgroup memory). The
// expert's plane is dense bf16 (`PIE_DENSE`), MXFP4 (`PIE_MXFP4`) or
// affine. A negative route lands nothing. SPIR-V reads bf16 and the MXFP4
// scales a slot at a time and lands a value a 16-bit slot; WGSL reads words
// and lands a value into its word with atomics, as neighbouring rows of a
// word belong to different strips.

#define PIE_LANES 32u
#define PIE_ROWS 8u

struct Params {
    int in_vec_size;
    int out_vec_size;
    int x_slot_stride;
    int x_row_stride;
    int slots_per_row;
};

#if !defined(PIE_WGSL)

#if defined(PIE_MXFP4)
#define PIE_CHUNK 32u
#elif defined(PIE_DENSE)
#define PIE_CHUNK 4u
#else
#define PIE_CHUNK (32u / uint(PIE_BITS))
#endif

#if defined(PIE_DENSE)
PIE_SLOTS_RO(0, bank);
PIE_SLOTS_RO(1, x);
PIE_SLOTS_RW(2, y);
[[vk::binding(3)]] StructuredBuffer<int> expert_ids;
PIE_PARAMS(4, Params, params);
#else
#if defined(PIE_MXFP4)
[[vk::binding(0)]] StructuredBuffer<uint4> w4;
[[vk::binding(1)]] StructuredBuffer<uint8_t> scales;
[[vk::binding(3)]] StructuredBuffer<uint> x2;
#else
[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_SLOTS_RO(1, scales);
PIE_SLOTS_RO(2, biases);
PIE_SLOTS_RO(3, x);
#endif
PIE_SLOTS_RW(4, y);
PIE_SLOTS_RO(5, bias);
[[vk::binding(6)]] StructuredBuffer<int> expert_ids;
PIE_PARAMS(7, Params, params);
#endif

groupshared float partial[PIE_ROWS][PIE_LANES];

float chunk_dot(uint e, uint out_row, uint k, uint x_base) {
    uint in_size = uint(params.in_vec_size);
    uint out_size = uint(params.out_vec_size);
    float acc = 0.0;
#if defined(PIE_DENSE)
    uint at = (e * out_size + out_row) * in_size + k;
    [ForceUnroll] for (uint j = 0u; j < PIE_CHUNK; j++) {
        acc += slot_at(x, x_base + k + j) * slot_at(bank, at + j);
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
    float scale = slot_at(scales, sg);
    float zero = slot_at(biases, sg);
    float xs[32 / PIE_BITS];
    [ForceUnroll] for (uint j = 0u; j < PIE_CHUNK; j++) {
        xs[j] = slot_at(x, x_base + k + j);
    }
    acc = affine_word_dot(word, xs, scale, zero);
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
    uint out_size = uint(params.out_vec_size);
    uint in_size = uint(params.in_vec_size);
    bool active_out = out_row < out_size;

    uint sel = row * uint(params.slots_per_row) + slot;
    int expert = expert_ids[sel];

    if (expert < 0) return;
    uint e = uint(expert);
    uint x_base = row * uint(params.x_row_stride) + slot * uint(params.x_slot_stride);

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
        acc += slot_at(bias, e * out_size + out_row);
#endif
        slot_put(y, sel * out_size + out_row, acc);
    }
}

#else

#if defined(PIE_MXFP4)
#define PIE_CHUNK 8u
#elif defined(PIE_DENSE)
#define PIE_CHUNK 4u
#else
#define PIE_CHUNK (32u / uint(PIE_BITS))
#endif

#if defined(PIE_DENSE)
[[vk::binding(0)]] StructuredBuffer<uint> bank;
[[vk::binding(1)]] StructuredBuffer<uint> x;
[[vk::binding(2)]] RWStructuredBuffer<Atomic<uint>> y;
[[vk::binding(3)]] StructuredBuffer<int> expert_ids;
PIE_PARAMS(4, Params, params);
#else
[[vk::binding(0)]] StructuredBuffer<uint> w;
[[vk::binding(1)]] StructuredBuffer<uint> scales;
[[vk::binding(2)]] StructuredBuffer<uint> biases;
[[vk::binding(3)]] StructuredBuffer<uint> x;
[[vk::binding(4)]] RWStructuredBuffer<Atomic<uint>> y;
[[vk::binding(5)]] StructuredBuffer<uint> bias;
[[vk::binding(6)]] StructuredBuffer<int> expert_ids;
PIE_PARAMS(7, Params, params);
#endif

#if defined(PIE_SUBGROUP)
#define PIE_XS_WORDS 4096u
groupshared uint xs_shared[PIE_XS_WORDS];
groupshared uint xs_staged;

float load_x(uint base, uint rel) {
    if (xs_staged == 1u) {
        uint word = xs_shared[rel >> 1];
        return bf16_to_f32((rel & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
    }
    return bf16_at(x, base + rel);
}
#else
groupshared float partial[PIE_ROWS][PIE_LANES];

float load_x(uint base, uint rel) {
    return bf16_at(x, base + rel);
}
#endif

// Value `i` into its word, the neighbour's half kept.
void store_y(uint i, float v) {
    uint at = i >> 1;
    uint b = f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        y[at].and(0x0000ffffu);
        y[at].or(b << 16);
    } else {
        y[at].and(0xffff0000u);
        y[at].or(b);
    }
}

float chunk_dot(uint e, uint out_row, uint k, uint x_base) {
    uint in_size = uint(params.in_vec_size);
    uint out_size = uint(params.out_vec_size);
    float acc = 0.0;
#if defined(PIE_DENSE)
    uint at = (e * out_size + out_row) * in_size + k;
    for (uint j = 0u; j < PIE_CHUNK; j++) {
        acc += load_x(x_base, k + j) * bf16_at(bank, at + j);
    }
#elif defined(PIE_MXFP4)
    uint words_per_row = in_size / 8u;
    uint word = w[(e * out_size + out_row) * words_per_row + k / 8u];
    uint groups_per_row = in_size / uint(PIE_MXFP4_BLOCK);
    uint sg = (e * out_size + out_row) * groups_per_row + k / uint(PIE_MXFP4_BLOCK);
    float scale = mxfp4_block_scale((scales[sg >> 2] >> ((sg & 3u) * 8u)) & 0xffu);
    for (uint j = 0u; j < 8u; j++) {
        acc += load_x(x_base, k + j) * mxfp4_code_of(word, j);
    }
    acc *= scale;
#else
    uint words_per_row = in_size / PIE_CHUNK;
    uint word = w[(e * out_size + out_row) * words_per_row + k / PIE_CHUNK];
    uint groups_per_row = in_size / uint(PIE_GROUP);
    uint sg = (e * out_size + out_row) * groups_per_row + k / uint(PIE_GROUP);
    float scale = bf16_at(scales, sg);
    float zero = bf16_at(biases, sg);
    float xs[32 / PIE_BITS];
    for (uint j = 0u; j < PIE_CHUNK; j++) {
        xs[j] = load_x(x_base, k + j);
    }
    acc = affine_word_dot(word, xs, scale, zero);
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
    uint out_size = uint(params.out_vec_size);
    uint in_size = uint(params.in_vec_size);
    bool active_out = out_row < out_size;

    uint sel = row * uint(params.slots_per_row) + slot;
    int expert = expert_ids[sel];

    bool routed = expert >= 0;
    uint e = uint(max(expert, 0));
    uint x_base = row * uint(params.x_row_stride) + slot * uint(params.x_slot_stride);
#if defined(PIE_SUBGROUP)
    uint x_words = (in_size + 1u) >> 1;
    bool staged = x_words <= PIE_XS_WORDS && (x_base & 1u) == 0u;
    if (staged) {
        for (uint wi = lane + PIE_LANES * local_out; wi < x_words; wi += PIE_LANES * PIE_ROWS) {
            xs_shared[wi] = x[(x_base >> 1) + wi];
        }
    }
    if (lane == 0u && local_out == 0u) {
        xs_staged = staged ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
#endif

    float acc = 0.0;
    if (active_out && routed) {
        for (uint k = lane * PIE_CHUNK; k < in_size; k += PIE_LANES * PIE_CHUNK) {
            acc += chunk_dot(e, out_row, k, x_base);
        }
    }

#if defined(PIE_SUBGROUP)
    [ForceUnroll] for (uint off = 16u; off > 0u; off >>= 1u) {
        acc += pie_shuffle_xor(acc, off);
    }
    float out = acc;
#else
    partial[local_out][lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 16u; step > 0u; step >>= 1u) {
        if (lane < step) {
            partial[local_out][lane] += partial[local_out][lane + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float out = partial[local_out][0];
#endif
    if (lane == 0u && active_out && routed) {
#if defined(PIE_BIASED)
        out += bf16_at(bias, e * out_size + out_row);
#endif
        store_y(sel * out_size + out_row, out);
    }
}

#endif

// pie:instantiate linear_moe_select_mxfp4_bf16_gs_32_b_4 PIE_MXFP4=1
// pie:instantiate linear_moe_select_mxfp4_bias_bf16_gs_32_b_4 PIE_MXFP4=1 PIE_BIASED=1
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIASED=1
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate linear_moe_select_affine_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate linear_moe_select_affine_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1
// pie:instantiate linear_moe_select_dense_bf16 PIE_DENSE=1
// pie:instantiate linear_moe_select_mxfp4_bf16_gs_32_b_4 PIE_MXFP4=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_mxfp4_bias_bf16_gs_32_b_4 PIE_MXFP4=1 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_affine_bias_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2 PIE_BIASED=1 PIE_TIER=subgroup PIE_TARGET=wgsl
// pie:instantiate linear_moe_select_dense_bf16 PIE_DENSE=1 PIE_TIER=subgroup PIE_TARGET=wgsl
