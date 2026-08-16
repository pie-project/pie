// Device argmax plus EOS compare.
//
// The Metal kernel's contract is preserved: bf16 widens exactly to fp32, the
// scan keeps the lowest index on ties, and one workgroup owns one request row.
// Vulkan uses shared memory for the pair reduction so it does not assume a
// 32-lane subgroup.

#include "common/bf16.slang"

#define PIE_GROUP_X 1024

PIE_BUFFER_RO(0, logits);
[[vk::binding(1)]] RWStructuredBuffer<uint> next_token;
struct ArgmaxParams { uint vocab; uint n_eos; uint eos_ids[8]; };
PIE_PARAMS(2, ArgmaxParams, p);
[[vk::binding(3)]] RWStructuredBuffer<uint> eos_flag;

groupshared float sh_v[PIE_GROUP_X];
groupshared uint sh_i[PIE_GROUP_X];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    uint lid = local.x;
    float best_v = -1.0 / 0.0;
    uint best_i = 0u;
    for (uint i = lid; i < p.vocab; i += uint(PIE_GROUP_X)) {
        float v = PIE_LOAD(logits[row * p.vocab + i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    sh_v[lid] = best_v;
    sh_i[lid] = best_i;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = uint(PIE_GROUP_X) >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            float ov = sh_v[lid + stride];
            uint oi = sh_i[lid + stride];
            if (ov > sh_v[lid] || (ov == sh_v[lid] && oi < sh_i[lid])) {
                sh_v[lid] = ov;
                sh_i[lid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        uint tok = sh_i[0];
        next_token[row] = tok;
        uint flag = 0u;
        for (uint e = 0u; e < p.n_eos; e++) {
            if (tok == p.eos_ids[e]) { flag = 1u; break; }
        }
        eos_flag[row] = flag;
    }
}

// pie:instantiate argmax_logits_bfloat16
