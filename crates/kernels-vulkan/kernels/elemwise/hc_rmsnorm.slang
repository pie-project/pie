#include "common/bf16.slang"
#include "common/reduce.slang"

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] RWStructuredBuffer<float> out_;

struct Push { int dim; float eps; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint dim = uint(pc.dim);
    uint base = group.x * dim;
    float acc = 0.0;
    for (uint d = local.x; d < dim; d += uint(PIE_GROUP_X)) {
        float v = PIE_LOAD(x[base + d]);
        acc += v * v;
    }
    float inv = pie_inv_rms(acc, dim, pc.eps);
    for (uint d = local.x; d < dim; d += uint(PIE_GROUP_X)) {
        out_[base + d] = PIE_LOAD(x[base + d]) * inv;
    }
}

// pie:instantiate hc_rmsnorm_f32_bf16 PIE_GROUP_X=256
