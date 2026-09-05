#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif
#define PIE_MAX_RANK 128

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, bank_a);
PIE_BUFFER_RO(2, bank_b);
[[vk::binding(3)]] StructuredBuffer<int> routes;
PIE_BUFFER_RW(4, y);

struct Push { int in_width; int out_width; int rank; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float waist[PIE_MAX_RANK];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    int adapter = routes[row];
    if (adapter < 0) {
        return;
    }
    uint in_width = uint(pc.in_width);
    uint out_width = uint(pc.out_width);
    uint r = min(uint(pc.rank), uint(PIE_MAX_RANK));
    uint down = uint(adapter) * r * in_width;
    uint up = uint(adapter) * out_width * r;
    uint a = row * in_width;
    uint out = row * out_width;

    uint lane = WaveGetLaneIndex();
    uint width = WaveGetLaneCount();
    for (uint i = gl_SubgroupID; i < r; i += gl_NumSubgroups) {
        uint wrow = down + i * in_width;
        float acc = 0.0;
        for (uint c = lane; c < in_width; c += width) {
            acc += PIE_LOAD(bank_a[wrow + c]) * PIE_LOAD(x[a + c]);
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
            acc += PIE_LOAD(bank_b[brow + i]) * waist[i];
        }
        y[out + n] = PIE_STORE(PIE_LOAD(y[out + n]) + acc);
    }
}

// pie:instantiate lora_correct_bf16 PIE_GROUP_X=256
