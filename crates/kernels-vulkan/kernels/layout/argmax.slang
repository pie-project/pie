#include "common/bf16.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

#if defined(PIE_F32)
[[vk::binding(0)]] StructuredBuffer<float> x;
#define LOAD_X(i) (x[i])
#else
PIE_BUFFER_RO(0, x);
#define LOAD_X(i) PIE_LOAD(x[i])
#endif
[[vk::binding(1)]] RWStructuredBuffer<int> y;

struct Push { int width; int depth; int column; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float sh_v[PIE_GROUP_X];
groupshared uint sh_i[PIE_GROUP_X];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.y;
    uint lid = local.x;
    uint width = uint(pc.width);
    float best = -1.0 / 0.0;
    uint best_i = 0xFFFFFFFFu;
    for (uint c = lid; c < width; c += uint(PIE_GROUP_X)) {
        float v = LOAD_X(row * width + c);
        if (!isnan(v) && (v > best || (v == best && c < best_i))) {
            best = v;
            best_i = c;
        }
    }
    sh_v[lid] = best;
    sh_i[lid] = best_i;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = uint(PIE_GROUP_X) >> 1; stride > 0u; stride >>= 1) {
        if (lid < stride) {
            float ov = sh_v[lid + stride];
            uint oi = sh_i[lid + stride];
            if (oi != 0xFFFFFFFFu && (ov > sh_v[lid] || (ov == sh_v[lid] && oi < sh_i[lid]))) {
                sh_v[lid] = ov;
                sh_i[lid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        uint top = sh_i[0];
        y[row * uint(pc.depth) + uint(pc.column)] = int(top == 0xFFFFFFFFu ? 0u : top);
    }
}

// pie:instantiate argmax_rows_bf16 PIE_GROUP_X=256
// pie:instantiate argmax_rows_f32 PIE_GROUP_X=256 PIE_F32=1
