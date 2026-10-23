#include "prim/bf16.slang"
#include "prim/params.slang"
#if defined(PIE_SUBGROUP)
#include "prim/slot16.slang"
#include "prim/subgroup.slang"
#else
#include "prim/reduce.slang"
#endif

// The LoRA correction `y += up(down(x))` a row a workgroup, the row's
// adapter from `routes` (negative: none, and the row is left alone). The
// down projection lands `rank` values in workgroup memory, the up
// projection adds onto the row. At the subgroup tier (SPIR-V) each
// subgroup folds a rank's dot by itself and the row lands a value a 16-bit
// slot; the baseline folds through workgroup memory and lands column pairs
// as words, so its host refuses an odd output width.

#define PIE_MAX_RANK 128

struct Params {
    int in_width;
    int out_width;
    int rank;
};
PIE_PARAMS(5, Params, params);

groupshared float waist[PIE_MAX_RANK];

#if defined(PIE_SUBGROUP)

PIE_SLOTS_RO(0, x);
PIE_SLOTS_RO(1, bank_a);
PIE_SLOTS_RO(2, bank_b);
[[vk::binding(3)]] StructuredBuffer<int> routes;
PIE_SLOTS_RW(4, y);

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    int adapter = routes[row];
    if (adapter < 0) {
        return;
    }
    uint in_width = uint(params.in_width);
    uint out_width = uint(params.out_width);
    uint r = min(uint(params.rank), uint(PIE_MAX_RANK));
    uint down = uint(adapter) * r * in_width;
    uint up = uint(adapter) * out_width * r;
    uint a = row * in_width;
    uint out = row * out_width;

    uint lane = WaveGetLaneIndex();
    uint width = WaveGetLaneCount();
    for (uint i = pie_subgroup_id(); i < r; i += pie_num_subgroups()) {
        uint wrow = down + i * in_width;
        float acc = 0.0;
        for (uint c = lane; c < in_width; c += width) {
            acc += slot_at(bank_a, wrow + c) * slot_at(x, a + c);
        }
        acc = WaveActiveSum(acc);
        if (lane == 0u) {
            waist[i] = acc;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint n = local.x; n < out_width; n += uint(PIE_GROUP_X)) {
        uint brow = up + n * r;
        float acc = 0.0;
        for (uint i = 0u; i < r; i++) {
            acc += slot_at(bank_b, brow + i) * waist[i];
        }
        slot_put(y, out + n, bf16_to_f32(uint(y[out + n])) + acc);
    }
}

#else

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> bank_a;
[[vk::binding(2)]] StructuredBuffer<uint> bank_b;
[[vk::binding(3)]] StructuredBuffer<int> routes;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

// The row's adapter, made workgroup-uniform: a barrier follows the branch on
// it, and WGSL proves uniformity only through a workgroup-uniform load.
groupshared int route;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    uint lid = local.x;
    if (lid == 0u) {
        route = routes[row];
    }
    GroupMemoryBarrierWithGroupSync();
    int adapter = workgroupUniformLoad(route);
    if (adapter < 0) {
        return;
    }
    uint in_width = uint(max(params.in_width, 0));
    uint out_width = uint(max(params.out_width, 0));
    uint r = min(uint(max(params.rank, 0)), uint(PIE_MAX_RANK));
    uint down = uint(adapter) * r * in_width;
    uint up = uint(adapter) * out_width * r;
    uint a = row * in_width;
    uint out = row * out_width;

    for (uint i = 0u; i < r; i++) {
        uint wrow = down + i * in_width;
        float acc = 0.0;
        for (uint c = lid; c < in_width; c += uint(PIE_GROUP_X)) {
            acc += bf16_at(bank_a, wrow + c) * bf16_at(x, a + c);
        }
        float total = pie_workgroup_sum(acc, lid);
        if (lid == 0u) {
            waist[i] = total;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint n = 2u * lid; n < out_width; n += 2u * uint(PIE_GROUP_X)) {
        uint brow0 = up + n * r;
        uint brow1 = brow0 + r;
        float acc0 = 0.0;
        float acc1 = 0.0;
        for (uint i = 0u; i < r; i++) {
            acc0 += bf16_at(bank_b, brow0 + i) * waist[i];
            acc1 += bf16_at(bank_b, brow1 + i) * waist[i];
        }
        uint at = (out + n) >> 1;
        uint old = y[at];
        y[at] = pack_bf16(bf16_to_f32(old & 0xffffu) + acc0, bf16_to_f32(old >> 16) + acc1);
    }
}

#endif

// pie:instantiate linear_lora_correct_bf16 PIE_GROUP_X=256
// pie:instantiate linear_lora_correct_bf16 PIE_GROUP_X=256 PIE_TIER=subgroup PIE_TARGET=spirv
